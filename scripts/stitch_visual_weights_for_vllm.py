#!/usr/bin/env python3
"""
合并目录里通常只有 LoRA merge 后的 model.language_model.*；vLLM 按 Qwen3.5-VL 配置加载时
还需要基座里的 model.visual.*。本脚本把魔搭本机基座中的视觉塔权重拼进合并后的 safetensors。

用法:
  python3 scripts/stitch_visual_weights_for_vllm.py \\
    --merged_safetensors training_runs/.../model.safetensors \\
    --base_model_dir .modelscope/Qwen/Qwen3___5-4B \\
    --output training_runs/.../model.safetensors

可先备份:
  cp model.safetensors model.language_only.safetensors.bak
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

from safetensors import safe_open
from safetensors.torch import load_file, save_file


def main() -> None:
    p = argparse.ArgumentParser(description="拼入基座 visual 权重供 vLLM 加载 Qwen3.5-VL")
    p.add_argument("--merged_safetensors", required=True, help="合并产物里的 model.safetensors（仅含 LM）")
    p.add_argument("--base_model_dir", required=True, help="魔搭 Qwen3.5-4B 本机目录（含分片 safetensors）")
    p.add_argument("--output", required=True, help="写出路径（可与 merged 相同以原地替换）")
    args = p.parse_args()

    if not os.path.isfile(args.merged_safetensors):
        print(f"error: 不存在 {args.merged_safetensors}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(args.base_model_dir):
        print(f"error: 不存在 {args.base_model_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"加载合并权重: {args.merged_safetensors}")
    tensors = load_file(args.merged_safetensors)
    n0 = len(tensors)

    n_vis = 0
    for fp in sorted(glob.glob(os.path.join(args.base_model_dir, "*.safetensors"))):
        with safe_open(fp, framework="pt", device="cpu") as f:
            for k in f.keys():
                if k.startswith("model.visual"):
                    tensors[k] = f.get_tensor(k).clone()
                    n_vis += 1
    print(f"LM 张量数 {n0}，自基座追加 model.visual.* → {n_vis}，合计 {len(tensors)}")

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    save_file(tensors, args.output)
    print(f"已写出: {args.output}")


if __name__ == "__main__":
    main()
