#!/usr/bin/env python3
"""
将 LoRA 适配器合并进基座模型（PEFT merge_and_unload），写出完整权重目录。

典型用途：
- `new_mian.py` / 其他脚本训练后目录里只有 adapter + 配置，需要单目录全量权重给
  `transformers` 直接 `from_pretrained`，或给部分推理栈做「全量 FP16 部署」。

注意：
- 合并应在「非 4bit 量化」基座上完成；若训练时用了 4bit，请用同一基座的 FP16/BF16
  重新加载再挂 adapter 后合并（本脚本默认 float16）。
- 合并后体积约等于「基座 + 少量元数据」，请预留磁盘空间。
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    p = argparse.ArgumentParser(description="LoRA adapter -> merged full weights")
    p.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="基座模型：HF id 或本机目录（须与训练 adapter 时一致）",
    )
    p.add_argument(
        "--adapter_path",
        type=str,
        required=True,
        help="含 adapter 权重的目录（一般为训练 output_dir / checkpoint-*）",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="合并后保存目录（将新建/覆盖同名文件）",
    )
    p.add_argument(
        "--torch_dtype",
        type=str,
        choices=["float16", "bfloat16", "float32"],
        default="float16",
        help="加载与写出时的 torch dtype（合并勿用 4bit）",
    )
    p.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help="传给 from_pretrained，默认 auto（多卡可 auto）",
    )
    args = p.parse_args()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.torch_dtype]

    if not os.path.isdir(args.adapter_path):
        print(f"error: adapter_path 不是目录: {args.adapter_path}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"加载基座: {args.base_model} ({args.torch_dtype})")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch_dtype,
        device_map=args.device_map,
        trust_remote_code=True,
    )
    print(f"加载适配器: {args.adapter_path}")
    model = PeftModel.from_pretrained(base, args.adapter_path, is_trainable=False)
    print("merge_and_unload() …")
    merged = model.merge_and_unload()
    print(f"保存到: {args.output_dir}")
    merged.save_pretrained(args.output_dir, safe_serialization=True)

    tok_src = args.adapter_path
    if not os.path.isfile(os.path.join(tok_src, "tokenizer_config.json")):
        tok_src = args.base_model
    print(f"保存 tokenizer（来源: {tok_src}）")
    tokenizer = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)
    tokenizer.save_pretrained(args.output_dir)
    print("完成。")


if __name__ == "__main__":
    main()
