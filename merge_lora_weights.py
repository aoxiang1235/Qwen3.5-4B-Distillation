#!/usr/bin/env python3
"""
将 PEFT LoRA 适配器合并进基座权重，得到可单独 from_pretrained 的完整模型目录（方案 C）。

依赖：pip install transformers peft accelerate safetensors torch

典型用法（云上 / 本机）：
  python3 merge_lora_weights.py \\
    --base_model Qwen/Qwen2.5-3B-Instruct \\
    --adapter ./training_runs/lora_qwen25_3b_train/checkpoint-500 \\
    --out ./merged_models/beauty_brand_merged \\
    --dtype float16

合并后用 vLLM 直接加载 --model 指向 --out 目录即可，无需再挂 LoRA。
"""
from __future__ import annotations

import argparse
import sys


def main() -> None:
    p = argparse.ArgumentParser(description="LoRA adapter + base -> merged full weights (HF folder)")
    p.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="基座：HF Hub id 或本地目录（需与训练时一致）",
    )
    p.add_argument(
        "--adapter",
        type=str,
        required=True,
        help="LoRA 输出目录（含 adapter_config.json；多为 checkpoint-*/ 子目录）",
    )
    p.add_argument(
        "--out",
        type=str,
        required=True,
        help="合并后保存目录（将新建）",
    )
    p.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=("float16", "bfloat16", "float32"),
        help="合并后权重 dtype；无 bf16 的 GPU 请用 float16",
    )
    p.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help="传给 from_pretrained 的 device_map，默认 auto",
    )
    args = p.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    dt = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]

    print(f"[merge] load base: {args.base_model}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dt,
        device_map=args.device_map,
        trust_remote_code=True,
    )
    print(f"[merge] load adapter: {args.adapter}", flush=True)
    model = PeftModel.from_pretrained(model, args.adapter, torch_dtype=dt)
    print("[merge] merge_and_unload() ...", flush=True)
    model = model.merge_and_unload()
    print(f"[merge] save -> {args.out}", flush=True)
    model.save_pretrained(args.out, safe_serialization=True)

    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tok.save_pretrained(args.out)
    print("[merge] done. 可用 vLLM: vllm serve <out> --trust-remote-code", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
