#!/usr/bin/env python3
"""
Qwen3.5-4B 合并（FP16）目录 → **GPTQ INT8**，专为本项目基座与默认路径而设（不修改 ``quantize_gptq_8bit.py``）。

基模标识（与 ``new_mian.py`` 默认魔搭一致）见下方常量，并会写入输出目录 ``quantize_meta.json``，
便于区分产物对应的底座。

依赖：``pip install -U gptqmodel``（或 ``auto-gptq``，见通用脚本）。

量化前建议停 vLLM、释放显存；若遇 ``libgomp: Thread creation failed``：
  OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python3 quantize_gptq_8bit_qwen35_merged.py ...

示例：
  python3 quantize_gptq_8bit_qwen35_merged.py \\
    --model_path training_runs/lora_qwen35_4b_merged_fp16_ckpt500 \\
    --output_dir training_runs/lora_qwen35_4b_merged_gptq8_ckpt500

vLLM 推理示例：
  python3 -m vllm.entrypoints.openai.api_server \\
    --model training_runs/lora_qwen35_4b_merged_gptq8_ckpt500 \\
    --quantization gptq \\
    --dtype float16 \\
    --trust-remote-code \\
    --default-chat-template-kwargs '{"enable_thinking": false}' \\
    --host 0.0.0.0 --port 8001
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

# ---------------------------------------------------------------------------
# 大模型基座标识（训练 / merge 对齐；写入 quantize_meta.json）
# ---------------------------------------------------------------------------
BASE_MODEL_MODELSCOPE_ID = "Qwen/Qwen3.5-4B"
BASE_MODEL_HF_ID = "Qwen/Qwen3.5-4B"
# 魔搭本机缓存目录名常见写法（仅作文档性提示，不强制存在）
BASE_MODEL_LOCAL_MODELSCOPE_SNAPSHOT_HINT = ".modelscope/Qwen/Qwen3___5-4B"

DEFAULT_MERGED_FP16_DIR = "training_runs/lora_qwen35_4b_merged_fp16_ckpt500"
DEFAULT_OUT_GPTQ8_DIR = "training_runs/lora_qwen35_4b_merged_gptq8_ckpt500"

from quantize_gptq_8bit import (  # noqa: E402
    build_calib_texts,
    _resolve_backend,
    _run_auto_gptq,
    _run_gptqmodel,
)


def _write_quantize_meta(
    out_dir: Path,
    *,
    base_model_modelscope_id: str,
    base_model_hf_id: str,
    base_snapshot_hint: str,
    source_merged_dir: Path,
    bits: int,
    backend: str,
    calib_jsonl: Path,
    calib_samples: int,
) -> None:
    meta: Dict[str, Any] = {
        "schema": "quantize_meta.v1",
        "base_model": {
            "modelscope_id": base_model_modelscope_id,
            "huggingface_id": base_model_hf_id,
            "local_modelscope_snapshot_hint": base_snapshot_hint,
            "note": "合并权重由上述基座 + LoRA merge 得到；GPTQ 仅压缩权重表示。",
        },
        "quantization": {
            "format": "gptq",
            "bits": bits,
            "backend": backend,
        },
        "inputs": {
            "merged_fp16_dir": str(source_merged_dir.resolve()),
            "calib_jsonl": str(calib_jsonl.resolve()),
            "calib_max_samples": calib_samples,
        },
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    path = out_dir / "quantize_meta.json"
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已写入基模标识与元数据: {path}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Qwen3.5-4B 合并 FP16 → GPTQ INT8（带基模标识元数据）",
    )
    p.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_MERGED_FP16_DIR,
        help=f"merge 后 HF 目录（默认 {DEFAULT_MERGED_FP16_DIR}）",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUT_GPTQ8_DIR,
        help=f"GPTQ 输出目录（默认 {DEFAULT_OUT_GPTQ8_DIR}）",
    )
    p.add_argument(
        "--base-model-modelscope-id",
        type=str,
        default=BASE_MODEL_MODELSCOPE_ID,
        help="写入 quantize_meta.json 的魔搭基座 ID",
    )
    p.add_argument(
        "--base-model-hf-id",
        type=str,
        default=BASE_MODEL_HF_ID,
        help="写入 quantize_meta.json 的 Hugging Face 基座 ID（与官方同名时常一致）",
    )
    p.add_argument(
        "--base-model-local-hint",
        type=str,
        default=BASE_MODEL_LOCAL_MODELSCOPE_SNAPSHOT_HINT,
        help="写入 meta 的本机魔搭 snapshot 路径提示（不强校验存在）",
    )
    p.add_argument(
        "--calib_jsonl",
        type=str,
        default="data/val.jsonl",
        help="校准 JSONL（instruction+content）",
    )
    p.add_argument(
        "--calib_max_samples",
        type=int,
        default=128,
        help="校准最多条数",
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
        help="仅 AutoGPTQ：tokenizer 截断长度",
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

    print(
        f"基模标识: modelscope={args.base_model_modelscope_id!r}, "
        f"huggingface={args.base_model_hf_id!r}"
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

    _write_quantize_meta(
        out_dir,
        base_model_modelscope_id=args.base_model_modelscope_id,
        base_model_hf_id=args.base_model_hf_id,
        base_snapshot_hint=args.base_model_local_hint,
        source_merged_dir=model_path,
        bits=args.bits,
        backend=backend,
        calib_jsonl=calib_path,
        calib_samples=len(calib_texts),
    )


if __name__ == "__main__":
    main()
