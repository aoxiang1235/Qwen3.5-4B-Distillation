#!/usr/bin/env python3
"""
已合并的 FP16 目录若缺 vLLM 所需侧文件，可从魔搭本机 snapshot 补拷（不必重新 merge）。

例（与训练 default 魔搭 id 一致）:
  python3 scripts/prepare_merged_for_vllm.py \\
    --base_dir .modelscope/Qwen/Qwen3___5-4B \\
    --merged_dir training_runs/lora_qwen35_4b_merged_fp16_ckpt500
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# 与仓库根目录的 merge_lora_weights 同逻辑
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from merge_lora_weights import sync_modelscope_assets_for_vllm  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(
        description="从魔搭本机基座目录向合并目录同步 config / preprocessor 等（供 vLLM 启动 Qwen3.5）",
    )
    p.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="魔搭 snapshot 路径，如 <cwd>/.modelscope/Qwen/Qwen3___5-4B",
    )
    p.add_argument(
        "--merged_dir",
        type=str,
        required=True,
        help="合并产物目录（含 model.safetensors）",
    )
    args = p.parse_args()
    base = os.path.abspath(args.base_dir)
    merged = os.path.abspath(args.merged_dir)
    if not os.path.isdir(base):
        print(f"error: base_dir 不存在: {base}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(merged):
        print(f"error: merged_dir 不存在: {merged}", file=sys.stderr)
        sys.exit(1)
    n = sync_modelscope_assets_for_vllm(base, merged)
    print(f"完成：写入 {n} 个侧文件到 {merged}")
    sys.exit(0 if n else 2)


if __name__ == "__main__":
    main()
