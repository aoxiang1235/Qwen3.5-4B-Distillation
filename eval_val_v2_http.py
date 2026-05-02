#!/usr/bin/env python3
"""
遍历 data/val_v2.jsonl，对每条调用 HTTP /generate，将模型返回与金标一并写入 JSONL。

默认针对本仓库 serve_qwen35_full_http.py（或兼容同接口）的返回字段，例如：
ok, elapsed_ms, output_json, max_new_tokens_used, note 等。

用法示例：
  python3 eval_val_v2_http.py --port 8012 --output data/val_v2_eval_8012.jsonl
  python3 eval_val_v2_http.py --base-url http://127.0.0.1:8012 --limit 3 --output /tmp/smoke.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterator, Optional


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


def post_generate(
    base_url: str,
    instruction: str,
    content: str,
    max_new_tokens: int,
    timeout_sec: float,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    url = base_url.rstrip("/") + "/generate"
    body: Dict[str, Any] = {
        "instruction": instruction,
        "content": content,
        "max_new_tokens": max_new_tokens,
    }
    if extra:
        body.update(extra)
    raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=raw,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            text = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode("utf-8", errors="replace")
        return {
            "ok": False,
            "http_status": exc.code,
            "error": err_body or str(exc),
        }
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"ok": False, "error": "invalid_json_response", "raw_response": text[:8000]}


def main() -> None:
    parser = argparse.ArgumentParser(description="用 val_v2.jsonl 批量测 HTTP /generate 并落盘")
    parser.add_argument("--data", type=str, default="data/val_v2.jsonl", help="输入 JSONL")
    parser.add_argument(
        "--output",
        type=str,
        default="data/val_v2_eval_8012.jsonl",
        help="输出 JSONL（每行一条评测记录）",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="",
        help="如 http://127.0.0.1:8012；留空则用 --host + --port 拼",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8012)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="单条请求超时（秒）",
    )
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条，0 表示全部")
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="每条请求间隔（秒），避免压垮服务",
    )
    parser.add_argument(
        "--append-json-hint",
        dest="append_json_hint",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="若服务端支持，透传 append_json_hint（默认不传，由服务端默认）",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.is_file():
        print(f"error: 找不到数据文件: {data_path}", file=sys.stderr)
        sys.exit(1)

    base = (args.base_url or "").strip() or f"http://{args.host}:{args.port}"
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    extra: Dict[str, Any] = {}
    if args.append_json_hint is not None:
        extra["append_json_hint"] = args.append_json_hint

    n_done = 0
    t0 = time.perf_counter()
    with out_path.open("w", encoding="utf-8") as out_f:
        for row in iter_jsonl(data_path):
            if args.limit and n_done >= args.limit:
                break
            post_id = row.get("post_id", "")
            instruction = str(row.get("instruction", "")).strip()
            content = str(row.get("content", "")).strip()
            gold = row.get("output")

            if not instruction or not content:
                rec = {
                    "post_id": post_id,
                    "skipped": True,
                    "reason": "missing instruction or content",
                    "gold_output": gold,
                }
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out_f.flush()
                n_done += 1
                continue

            resp = post_generate(
                base,
                instruction,
                content,
                args.max_new_tokens,
                args.timeout,
                extra if extra else None,
            )
            rec = {
                "post_id": post_id,
                "gold_output": gold,
                "model_response": resp,
            }
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()
            n_done += 1
            ok = resp.get("ok") is True
            ms = resp.get("elapsed_ms", "")
            print(f"[{n_done}] post_id={post_id} ok={ok} elapsed_ms={ms}", flush=True)
            if args.sleep > 0:
                time.sleep(args.sleep)

    elapsed = time.perf_counter() - t0
    print(f"done: wrote {n_done} rows -> {out_path} (total {elapsed:.1f}s)")


if __name__ == "__main__":
    main()
