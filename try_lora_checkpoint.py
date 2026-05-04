#!/usr/bin/env python3
"""
用指定 LoRA checkpoint（如 training_runs/.../checkpoint-400）在单条或少量 val 样本上试生成，
与 ``new_mian.py`` 使用相同的 system / user 拼法（Instruction + Content chat 模板）。

示例（云机、仓库根）：

  python3 try_lora_checkpoint.py \\
    --base_model /workspace/Qwen3.5-4B-Distillation/.modelscope/Qwen/Qwen3___5-4B \\
    --adapter_path training_runs/lora_qwen35_4b_full_B_20260504_050130/checkpoint-400 \\
    --val_jsonl data/val.jsonl \\
    --max_new_tokens 256
"""
from __future__ import annotations

import argparse
import json
import gc
import sys
from typing import Any, Dict, List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# 与 new_mian.py 一致
DEFAULT_SYSTEM_PROMPT = "You are a strict structured information extraction assistant."


def _user_block(instruction: str, content: str) -> str:
    return f"Instruction:\n{instruction}\n\nContent:\n{content}"


def _load_val_rows(path: str, limit: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            rows.append(json.loads(s))
            if len(rows) >= limit:
                break
    if not rows:
        raise SystemExit(f"error: {path} 无有效行")
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="LoRA checkpoint 试生成（对齐训练模板）")
    p.add_argument("--base_model", type=str, required=True, help="基座 HF 目录或 id（与训练一致）")
    p.add_argument("--adapter_path", type=str, required=True, help="checkpoint-* 目录")
    p.add_argument("--val_jsonl", type=str, default="data/val.jsonl")
    p.add_argument("--limit", type=int, default=1, help="从 val 前几行试跑")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--torch_dtype", type=str, default="float16", choices=("float16", "bfloat16", "float32"))
    args = p.parse_args()

    dt = getattr(torch, args.torch_dtype)

    tok = AutoTokenizer.from_pretrained(args.adapter_path, trust_remote_code=True, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.truncation_side = "left"

    print(f"加载基座: {args.base_model} ({args.torch_dtype})")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dt,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"加载 LoRA: {args.adapter_path}")
    model = PeftModel.from_pretrained(base, args.adapter_path, is_trainable=False)
    model.eval()

    rows = _load_val_rows(args.val_jsonl, args.limit)
    for i, row in enumerate(rows):
        inst = str(row.get("instruction", "")).strip()
        content = str(row.get("content", "")).strip()
        post_id = str(row.get("post_id", ""))
        if not inst or not content:
            print(f"[{i}] skip: 缺 instruction/content post_id={post_id}", file=sys.stderr)
            continue
        messages = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": _user_block(inst, content)},
        ]
        prompt = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tok(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
        gen = tok.decode(
            out[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        ).strip()
        print(f"\n=== sample {i} post_id={post_id} ===\n{gen}\n")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("完成。")


if __name__ == "__main__":
    main()
