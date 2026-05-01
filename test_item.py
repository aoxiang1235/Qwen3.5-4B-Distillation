#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict


def resolve_data_path(raw_path: str) -> Path:
    """
    解析数据文件路径：
    1) 若传入路径存在，直接用；
    2) 否则尝试常见候选，避免因 cwd 不同导致 FileNotFoundError。
    """
    p = Path(raw_path)
    if p.exists():
        return p

    candidates = [
        Path("data/val_v2.jsonl"),
        Path("val_v2.jsonl"),
        Path("data/train_v2.jsonl"),
        Path("train_v2.jsonl"),
        Path("data/train.jsonl"),
        Path("train.jsonl"),
    ]
    for c in candidates:
        if c.exists():
            return c

    return p


def load_item_by_post_id(jsonl_path: Path, post_id: str) -> Dict[str, Any]:
    if not jsonl_path.exists():
        raise FileNotFoundError(f"找不到文件: {jsonl_path}")

    with jsonl_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON 解析失败: {jsonl_path}:{idx} ({exc})") from exc

            if str(obj.get("post_id", "")).strip() == post_id:
                return obj

    raise ValueError(f"未找到 post_id={post_id} 的记录（文件: {jsonl_path}）")


def call_model_with_curl(url: str, instruction: str, content: str, max_new_tokens: int) -> Dict[str, Any]:
    payload = {
        "instruction": instruction,
        "content": content,
        "max_new_tokens": max_new_tokens,
    }
    payload_str = json.dumps(payload, ensure_ascii=False)

    cmd = [
        "curl",
        "-sS",
        "-X",
        "POST",
        url,
        "-H",
        "Content-Type: application/json",
        "-d",
        payload_str,
    ]

    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if res.returncode != 0:
        raise RuntimeError(f"curl 调用失败: {res.stderr.strip() or res.stdout.strip()}")

    text = res.stdout.strip()
    if not text:
        raise RuntimeError("模型接口返回空内容")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"raw_response": text}


def print_json_block(title: str, obj: Any) -> None:
    print(f"\n=== {title} ===")
    try:
        print(json.dumps(obj, ensure_ascii=False, indent=2))
    except TypeError:
        print(str(obj))


def main() -> None:
    parser = argparse.ArgumentParser(description="按 post_id 测试单条样本并对比标注 output 与模型输出")
    parser.add_argument("post_id", type=str, help="要测试的 post_id")
    parser.add_argument(
        "--data",
        type=str,
        default="data/val_v2.jsonl",
        help="测试数据 JSONL 文件路径（默认: data/val_v2.jsonl）",
    )
    parser.add_argument(
        "--url",
        type=str,
        default="",
        help="模型服务接口地址（如 http://127.0.0.1:8000/generate）。若留空则按 --port 组装",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="模型服务端口（默认: 8000，仅在 --url 留空时生效）",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="生成 token 上限（默认: 128）",
    )
    args = parser.parse_args()

    data_path = resolve_data_path(args.data)
    try:
        item = load_item_by_post_id(data_path, args.post_id.strip())
    except Exception as exc:
        print(f"[错误] 读取样本失败: {exc}")
        sys.exit(1)

    instruction = str(item.get("instruction", "")).strip()
    content = str(item.get("content", "")).strip()
    expected_output = item.get("output", {})

    if not instruction or not content:
        print("[错误] 样本缺少 instruction 或 content")
        sys.exit(1)

    print(f"post_id: {args.post_id}")
    print(f"data: {data_path}")
    url = args.url.strip() or f"http://127.0.0.1:{args.port}/generate"
    print(f"url: {url}")

    print_json_block("文件中的 instruction", instruction)
    print_json_block("文件中的 content", content)
    print_json_block("文件中的标注 output", expected_output)

    try:
        model_resp = call_model_with_curl(
            url, instruction, content, args.max_new_tokens
        )
    except Exception as exc:
        print(f"\n[错误] 调用模型失败: {exc}")
        sys.exit(2)

    print_json_block("模型返回结果", model_resp)

    # 生成可复现的 curl 命令（便于手工复制）
    req = {
        "instruction": instruction,
        "content": content,
        "max_new_tokens": args.max_new_tokens,
    }
    req_str = json.dumps(req, ensure_ascii=False)
    print("\n=== 可复现 curl 命令 ===")
    print(
        f"curl -X POST \"{url}\" -H \"Content-Type: application/json\" -d '{req_str}'"
    )


def test_by_post_id(post_id: str, file_path: str = "data/val_v2.jsonl", port: int = 8000) -> None:
    """
    兼容旧版本调用方式：
    python test_item.py <post_id>
    旧代码若 import 并调用 test_by_post_id 也可继续使用。
    """
    data_path = resolve_data_path(file_path)
    item = load_item_by_post_id(data_path, post_id.strip())
    instruction = str(item.get("instruction", "")).strip()
    content = str(item.get("content", "")).strip()
    expected_output = item.get("output", {})
    url = f"http://127.0.0.1:{port}/generate"

    print(f"post_id: {post_id}")
    print(f"data: {data_path}")
    print(f"url: {url}")
    print_json_block("文件中的标注 output", expected_output)
    model_resp = call_model_with_curl(url, instruction, content, 128)
    print_json_block("模型返回结果", model_resp)


if __name__ == "__main__":
    main()
