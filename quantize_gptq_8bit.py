#!/usr/bin/env python3
"""
将已 merge 的 FP16/BF16 HuggingFace 目录量化为 **GPTQ INT8**（W8A16），供 **vLLM** ``--quantization gptq`` 加载。

依赖（二选一，优先 GPTQModel；云上 Python 3.12 + 新 torch 时 auto-gptq 常编译失败）：
  pip install -U gptqmodel
  # 或：pip install -U auto-gptq  （需与本机 torch/CUDA 匹配的 wheel 或能源码编译）

在仓库根执行示例：
  python3 quantize_gptq_8bit.py \\
    --model_path training_runs/lora_qwen25_3b_merged_fp16 \\
    --output_dir training_runs/lora_qwen25_3b_merged_gptq8 \\
    --calib_jsonl data/val.jsonl \\
    --calib_max_samples 128

默认 ``--backend auto``：若已安装 ``gptqmodel`` 则用其量化，否则尝试 ``auto_gptq``。

量化完成后 vLLM 示例（在终端直接执行，勿用 .sh）：
  python3 -m vllm.entrypoints.openai.api_server \\
    --model training_runs/lora_qwen25_3b_merged_gptq8 \\
    --quantization gptq \\
    --served-model-name beauty_merged_gptq8 \\
    --dtype float16 \\
    --max-model-len 4096 \\
    --trust-remote-code \\
    --host 0.0.0.0 --port 8001 \\
    --gpu-memory-utilization 0.32

若 vLLM 报 GPTQ 相关错误，请对照本机 vLLM 文档调整 ``--quantization`` 或升级版本。
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List

import torch


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc


def build_calib_texts(
    calib_jsonl: Path,
    max_samples: int,
    max_chars: int,
) -> List[str]:
    texts: List[str] = []
    for row in iter_jsonl(calib_jsonl):
        inst = str(row.get("instruction", "")).strip()
        content = str(row.get("content", "")).strip()
        if not inst or not content:
            continue
        u = f"Instruction:\n{inst}\n\nContent:\n{content}"
        if max_chars > 0 and len(u) > max_chars:
            u = u[:max_chars]
        texts.append(u)
        if len(texts) >= max_samples:
            break
    if not texts:
        raise SystemExit(f"error: 从 {calib_jsonl} 未得到任何校准样本（需要 instruction+content）")
    return texts


def build_gptq_examples(
    tokenizer,
    texts: List[str],
    max_seq_length: int,
) -> List[Dict[str, torch.Tensor]]:
    """auto-gptq ``model.quantize`` 用的样本列表。"""
    out: List[Dict[str, torch.Tensor]] = []
    for t in texts:
        enc = tokenizer(
            t,
            truncation=True,
            max_length=max_seq_length,
            padding=False,
            return_tensors="pt",
        )
        out.append(
            {
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
            }
        )
    return out


def _run_gptqmodel(
    model_path: Path,
    out_dir: Path,
    calib_texts: List[str],
    args: argparse.Namespace,
) -> None:
    from gptqmodel import GPTQModel, QuantizeConfig
    from transformers import AutoTokenizer

    qc = QuantizeConfig(
        bits=args.bits,
        group_size=args.group_size,
        desc_act=args.desc_act or None,
    )
    print(f"使用 GPTQModel 加载: {model_path}")
    model = GPTQModel.load(
        str(model_path),
        quantize_config=qc,
        trust_remote_code=True,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        use_fast=False,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"开始 GPTQModel 量化（bits={args.bits}, group_size={args.group_size}）…")
    model.quantize(
        calib_texts,
        batch_size=args.quantize_batch_size,
        tokenizer=tokenizer,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"保存到: {out_dir}")
    model.save(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print("完成。")


def _run_auto_gptq(
    model_path: Path,
    out_dir: Path,
    calib_texts: List[str],
    args: argparse.Namespace,
) -> None:
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    from transformers import AutoTokenizer

    quantize_config = BaseQuantizeConfig(
        bits=args.bits,
        group_size=args.group_size,
        damp_percent=0.01,
        desc_act=args.desc_act,
    )
    print(f"使用 AutoGPTQ 加载: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        use_fast=False,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoGPTQForCausalLM.from_pretrained(
        str(model_path),
        quantize_config=quantize_config,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto",
    )
    examples = build_gptq_examples(tokenizer, calib_texts, args.max_seq_length)
    print(f"开始 AutoGPTQ 量化（bits={args.bits}, group_size={args.group_size}）…")
    model.quantize(
        examples,
        cache_examples_on_gpu=args.cache_examples_on_gpu,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"保存到: {out_dir}")
    try:
        model.save_quantized(str(out_dir), use_safetensors=True)
    except TypeError:
        model.save_quantized(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print("完成。")


def _resolve_backend(name: str) -> str:
    if name == "gptqmodel":
        try:
            import gptqmodel  # noqa: F401
        except ImportError:
            print("error: 未安装 gptqmodel： pip install -U gptqmodel", file=sys.stderr)
            sys.exit(1)
        return "gptqmodel"
    if name == "auto_gptq":
        try:
            import auto_gptq  # noqa: F401
        except ImportError:
            print(
                "error: 未安装 auto_gptq： pip install -U auto-gptq（或改用 --backend gptqmodel）",
                file=sys.stderr,
            )
            sys.exit(1)
        return "auto_gptq"
    try:
        import gptqmodel  # noqa: F401

        return "gptqmodel"
    except ImportError:
        pass
    try:
        import auto_gptq  # noqa: F401

        return "auto_gptq"
    except ImportError:
        print(
            "error: 未安装 gptqmodel 或 auto_gptq。\n"
            "  推荐： pip install -U gptqmodel\n"
            "  或： pip install -U auto-gptq --no-build-isolation",
            file=sys.stderr,
        )
        sys.exit(1)


def main() -> None:
    p = argparse.ArgumentParser(description="FP16 merged HF 目录 -> GPTQ（默认 8bit，供 vLLM）")
    p.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="merge 后的 HuggingFace 模型目录（含 config.json、权重）",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="量化结果输出目录（将创建）",
    )
    p.add_argument(
        "--calib_jsonl",
        type=str,
        default="data/val.jsonl",
        help="校准 JSONL（instruction+content，与 bench / AWQ 拼法一致）",
    )
    p.add_argument(
        "--calib_max_samples",
        type=int,
        default=128,
        help="校准最多条数（常见 64–256）",
    )
    p.add_argument(
        "--calib_max_chars",
        type=int,
        default=8000,
        help="单条校准文本最大字符数（0 不截断）",
    )
    p.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="仅 AutoGPTQ 路径：tokenizer 截断长度",
    )
    p.add_argument(
        "--bits",
        type=int,
        default=8,
        choices=(4, 8),
        help="GPTQ 位宽（默认 8）",
    )
    p.add_argument(
        "--group_size",
        type=int,
        default=128,
        help="GPTQ group_size",
    )
    p.add_argument(
        "--desc_act",
        action="store_true",
        help="激活顺序量化（默认关）",
    )
    p.add_argument(
        "--cache_examples_on_gpu",
        action="store_true",
        help="仅 AutoGPTQ：校准样本常驻 GPU",
    )
    p.add_argument(
        "--quantize_batch_size",
        type=int,
        default=1,
        help="仅 GPTQModel：quantize batch_size",
    )
    p.add_argument(
        "--backend",
        type=str,
        choices=("auto", "gptqmodel", "auto_gptq"),
        default="auto",
        help="量化后端：auto 优先 gptqmodel",
    )
    args = p.parse_args()

    model_path = Path(args.model_path)
    out_dir = Path(args.output_dir)
    calib_path = Path(args.calib_jsonl)

    if not model_path.is_dir():
        print(f"error: model_path 不是目录: {model_path}", file=sys.stderr)
        sys.exit(1)
    if not (model_path / "config.json").is_file():
        print(f"error: 缺少 config.json: {model_path}", file=sys.stderr)
        sys.exit(1)
    if not calib_path.is_file():
        print(f"error: 找不到校准文件: {calib_path}", file=sys.stderr)
        sys.exit(1)

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    calib_texts = build_calib_texts(
        calib_path,
        max_samples=args.calib_max_samples,
        max_chars=args.calib_max_chars,
    )
    print(f"校准文本条数: {len(calib_texts)}（来自 {calib_path}）")

    backend = _resolve_backend(args.backend)
    print(f"选用后端: {backend}")
    if backend == "gptqmodel":
        _run_gptqmodel(model_path, out_dir, calib_texts, args)
    else:
        try:
            import auto_gptq  # noqa: F401
        except ImportError:
            print(
                "error: 当前后端为 auto_gptq 但未安装 auto-gptq。\n"
                "可改用： --backend gptqmodel 并 pip install -U gptqmodel",
                file=sys.stderr,
            )
            sys.exit(1)
        _run_auto_gptq(model_path, out_dir, calib_texts, args)


if __name__ == "__main__":
    main()
