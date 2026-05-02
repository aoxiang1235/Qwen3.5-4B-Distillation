#!/usr/bin/env python3
"""
逐步遍历 data/train_v2.jsonl，按 vLLM /v1/chat/completions 格式请求（Instruction + Content 与 val 一致），
将耗时、post_id、数据集中的标注 output、以及接口返回的 message.content 写入日志。

默认日志每行格式（TAB 分隔）：
  time_sec<TAB>post_id<TAB>output_json<TAB>content_json

- time_sec：单次 HTTP 请求耗时（秒）；跳过请求时为 0。
- output_json：来自 jsonl 行里的「output」字段（dict/list）经紧凑 json.dumps；缺失或类型不对时
  为 {"missing_or_invalid_output_field": ...}。
- content_json：vLLM 响应 choices[0].message.content（请求后的模型原文）；默认 json.dumps 包一层；
  加 --content-plain 时去掉 JSON 转义并把 TAB/换行压成空格。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Tuple


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc


def post_chat(
    url: str,
    model: str,
    user_content: str,
    max_tokens: int,
    temperature: float,
    timeout_sec: float,
) -> Tuple[float, Dict[str, Any]]:
    """返回 (elapsed_seconds, response_json_dict)。HTTP 非 2xx 或非法 JSON 时 dict 内含 ok: false。"""
    body: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": user_content}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    raw_b = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=raw_b,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            text = resp.read().decode("utf-8")
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return elapsed, {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    elapsed = time.perf_counter() - t0
    try:
        return elapsed, json.loads(text)
    except json.JSONDecodeError:
        return elapsed, {"ok": False, "error": "invalid_json_response", "raw": text[:4000]}


def extract_assistant_content(resp: Dict[str, Any]) -> str:
    choices = resp.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    msg = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(msg, dict):
        return ""
    c = msg.get("content")
    return c if isinstance(c, str) else ""


def format_content_log_field(text: str, plain: bool) -> str:
    """第四列：plain 时去掉 JSON 层面的转义写法，便于肉眼读；否则为合法 JSON 字符串字面量。"""
    if plain:
        return text.replace("\t", " ").replace("\r", " ").replace("\n", " ")
    return json.dumps(text, ensure_ascii=False)


def dataset_output_compact(row: Dict[str, Any], compact_fn: Callable[[Any], str]) -> str:
    """第三列：train_v2.jsonl 的 output 字段（标注 JSON），紧凑序列化。"""
    v = row.get("output")
    if isinstance(v, (dict, list)):
        return compact_fn(v)
    return compact_fn({"missing_or_invalid_output_field": v})


def main() -> None:
    p = argparse.ArgumentParser(description="train_v2.jsonl -> vLLM chat bench + log")
    p.add_argument("--data", type=str, default="data/train_v2.jsonl")
    p.add_argument(
        "--url",
        type=str,
        default="http://127.0.0.1:8000/v1/chat/completions",
        help="vLLM OpenAI 兼容 chat 接口完整 URL",
    )
    p.add_argument("--model", type=str, default="qwen-3.5-4b")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--log", type=str, default="train_v2_vllm_bench.log")
    p.add_argument("--limit", type=int, default=0, help="只跑前 N 条，0 表示全部")
    p.add_argument("--sleep", type=float, default=0.0, help="每条间隔秒数")
    p.add_argument(
        "--skip-health",
        action="store_true",
        help="跳过启动前对 base URL 的 GET /health（仅去掉 /v1/chat/completions 后缀测端口）",
    )
    p.add_argument(
        "--content-plain",
        action="store_true",
        help="第四列不用 json.dumps，直接写 message.content 原文；TAB/换行改为空格（无 \\\\n \\\" 等 JSON 转义）",
    )
    args = p.parse_args()

    data_path = Path(args.data)
    if not data_path.is_file():
        print(f"error: 找不到 {data_path}", file=sys.stderr)
        sys.exit(1)

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    base = args.url
    if not args.skip_health:
        if base.endswith("/v1/chat/completions"):
            health_url = base[: -len("/v1/chat/completions")] + "/health"
        else:
            health_url = base.rstrip("/") + "/health"
        try:
            req = urllib.request.Request(health_url, method="GET")
            with urllib.request.urlopen(req, timeout=10.0) as r:
                r.read()
        except Exception as exc:
            print(f"error: health check failed {health_url}: {exc}", file=sys.stderr)
            sys.exit(1)

    sep = "\t"
    compact = lambda o: json.dumps(o, ensure_ascii=False, separators=(",", ":"))

    n = 0
    with log_path.open("w", encoding="utf-8") as logf:
        for row in iter_jsonl(data_path):
            if args.limit and n >= args.limit:
                break
            post_id = str(row.get("post_id", ""))
            instruction = str(row.get("instruction", "")).strip()
            content = str(row.get("content", "")).strip()
            if not instruction or not content:
                line = sep.join(
                    [
                        "0.000",
                        post_id,
                        dataset_output_compact(row, compact),
                        format_content_log_field("", args.content_plain),
                    ]
                )
                logf.write(line + "\n")
                logf.flush()
                print(line, flush=True)
                n += 1
                continue

            user_content = f"Instruction:\n{instruction}\n\nContent:\n{content}"
            elapsed, resp = post_chat(
                args.url,
                args.model,
                user_content,
                args.max_tokens,
                args.temperature,
                args.timeout,
            )

            assistant_text = ""
            if resp.get("ok") is not False and "error" not in resp:
                assistant_text = extract_assistant_content(resp)
            else:
                assistant_text = ""

            # 第三列：数据集中的 output（标注）；第四列：接口返回的 message.content
            out_json = dataset_output_compact(row, compact)
            content_json = format_content_log_field(assistant_text, args.content_plain)
            line = sep.join(
                [
                    f"{elapsed:.6f}",
                    post_id,
                    out_json,
                    content_json,
                ]
            )
            logf.write(line + "\n")
            logf.flush()
            print(line, flush=True)
            n += 1
            if args.sleep > 0:
                time.sleep(args.sleep)

    print(f"done: {n} lines -> {log_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
