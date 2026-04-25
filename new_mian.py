#!/usr/bin/env python3
"""
Wrapper launcher for running training with Qwen3.5-4B defaults.

Usage example:
  python new_mian.py --train_jsonl data/train.jsonl --max_steps 30
"""

import subprocess
import sys


def _has_flag(argv: list[str], flag: str) -> bool:
    return any(arg == flag or arg.startswith(flag + "=") for arg in argv)


def main() -> int:
    argv = sys.argv[1:]

    # Apply Qwen3.5-4B defaults unless user explicitly overrides.
    if not _has_flag(argv, "--model_name"):
        argv += ["--model_name", "Qwen/Qwen3.5-4B-Instruct"]
    if not _has_flag(argv, "--output_dir"):
        argv += ["--output_dir", "distilled-qwen-3_5-4b"]
    if not _has_flag(argv, "--model_dtype"):
        argv += ["--model_dtype", "float32"]

    cmd = [sys.executable, "main.py", *argv]
    print("Running:", " ".join(cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
