#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path

def load_item_by_post_id(jsonl_path: Path, post_id: str):
    if not jsonl_path.exists():
        raise FileNotFoundError(f"找不到文件: {jsonl_path}")
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if str(obj.get("post_id", "")).strip() == post_id:
                return obj
    raise ValueError(f"未找到 post_id={post_id} 的记录（文件: {jsonl_path}）")

def call_model(url: str, instruction: str, content: str, max_new_tokens: int):
    payload = {
        "instruction": instruction,
        "content": content,
        "max_new_tokens": max_new_tokens,
    }
    cmd = [
        "curl", "-sS", "-X", "POST", url,
        "-H", "Content-Type: application/json",
        "-d", json.dumps(payload, ensure_ascii=False),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if res.returncode != 0:
        raise RuntimeError(res.stderr.strip() or res.stdout.strip() or "curl 调用失败")
    txt = res.stdout.strip()
    if not txt:
        raise RuntimeError("模型返回为空")
    try:
        return json.loads(txt)
    except Exception:
        return {"raw_response": txt}

def pretty(title, obj):
    print(f"\n=== {title} ===")
    print(json.dumps(obj, ensure_ascii=False, indent=2))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("post_id", type=str)
    parser.add_argument("--data", type=str, default="data/val_v2.jsonl")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    item = load_item_by_post_id(Path(args.data), args.post_id.strip())
    instruction = str(item.get("instruction", "")).strip()
    content = str(item.get("content", "")).strip()
    output = item.get("output", {})

    if not instruction or not content:
        raise ValueError("样本缺少 instruction 或 content")

    url = f"http://127.0.0.1:{args.port}/generate"

    print(f"post_id: {args.post_id}")
    print(f"data: {args.data}")
    print(f"url: {url}")

    pretty("文件中的 output", output)
    resp = call_model(url, instruction, content, args.max_new_tokens)
    pretty("模型返回结果", resp)

if __name__ == "__main__":
    main()


