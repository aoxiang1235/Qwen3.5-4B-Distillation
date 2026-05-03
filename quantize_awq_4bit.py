#!/usr/bin/env python3
"""
将已 merge 的 FP16/BF16 全量目录量化为 AWQ 4-bit（权重 4bit，推理多为 W4A16）。

依赖（在云 GPU 上安装）：
  pip install autoawq

示例（在仓库根、有 GPU）：
  python3 quantize_awq_4bit.py \\
    --model_path training_runs/lora_qwen25_3b_merged_fp16 \\
    --output_dir training_runs/lora_qwen25_3b_merged_awq4 \\
    --calib_jsonl data/val.jsonl \\
    --calib_max_samples 128

量化完成后 vLLM 启动示例：
  python3 -m vllm.entrypoints.openai.api_server \\
    --model training_runs/lora_qwen25_3b_merged_awq4 \\
    --quantization awq \\
    --dtype float16 \\
    --max-model-len 4096 \\
    --trust-remote-code \\
    --host 0.0.0.0 --port 8001

（具体参数以你本机 vLLM 版本文档为准；若不支持 --quantization awq，需升级 vLLM。）
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List


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


def main() -> None:
    try:
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer
    except ImportError:
        print(
            "error: 需要安装 autoawq： pip install autoawq\n"
            "若编译失败可尝试： pip install autoawq --no-build-isolation",
            file=sys.stderr,
        )
        sys.exit(1)

    p = argparse.ArgumentParser(description="FP16 merged HF 目录 -> AWQ 4-bit")
    p.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="merge 后的 HuggingFace 模型目录（含 config.json、model*.safetensors）",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="量化结果输出目录（将创建/覆盖）",
    )
    p.add_argument(
        "--calib_jsonl",
        type=str,
        default="data/val.jsonl",
        help="用于校准的 JSONL（每行含 instruction、content；与 bench 拼法一致）",
    )
    p.add_argument(
        "--calib_max_samples",
        type=int,
        default=128,
        help="最多使用多少条样本做校准（越大越慢，一般 64–256）",
    )
    p.add_argument(
        "--calib_max_chars",
        type=int,
        default=8000,
        help="单条校准文本最大字符数，超长截断（0 表示不截断）",
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

    calib_data = build_calib_texts(
        calib_path,
        max_samples=args.calib_max_samples,
        max_chars=args.calib_max_chars,
    )
    print(f"校准样本数: {len(calib_data)}（来自 {calib_path}）")

    quant_config = {
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
        "version": "GEMM",
    }

    print(f"加载模型: {model_path}")
    model = AutoAWQForCausalLM.from_pretrained(
        str(model_path),
        safetensors=True,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True,
    )

    print("开始 AWQ 量化（4-bit）…")
    model.quantize(tokenizer, quant_config=quant_config, calib_data=calib_data)

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"保存到: {out_dir}")
    model.save_quantized(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print("完成。")


if __name__ == "__main__":
    main()
