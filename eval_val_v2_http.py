#!/usr/bin/env python3
"""
遍历 data/val_v2.jsonl，对每条调用 HTTP /generate，将模型返回与金标一并写入 JSONL。

每行除 gold_output（来自 val 金标）外，另有：
- model_output：仅来自服务端解析结果（如 output_json），与金标无关；
- model_reasoning：从 model_output.reasoning 抽出，便于直接对比；
- model_response：完整 HTTP JSON（含 ok、elapsed_ms、错误信息等）。

默认针对 serve_qwen35_full_http.py 的 output_json；若接 vanilla 仅有 text，则 model_output 为 {"text": "..."}，model_reasoning 为 null。

启动前默认会 GET /health；连不上则直接退出、不会清空输出文件。

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


def check_health(base_url: str, timeout_sec: float = 5.0) -> tuple[bool, str]:
    """请求 GET /health；失败时返回 (False, 可读错误说明)。"""
    url = base_url.rstrip("/") + "/health"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            _ = resp.read()
        return True, ""
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def extract_model_output(resp: Dict[str, Any]) -> Any:
    """
    从大模型 HTTP 返回里取出「模型侧」结构化结果，不写金标。
    - serve_qwen35_full_http：使用 output_json（dict 或 null）。
    - serve_qwen35_vanilla_http：无 output_json 时退回 {"text": ...}。
    """
    if not isinstance(resp, dict) or resp.get("ok") is not True:
        return None
    oj = resp.get("output_json")
    if isinstance(oj, dict):
        return oj
    if oj is None and isinstance(resp.get("text"), str):
        return {"text": resp["text"]}
    return None


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
        help="输出 JSONL（每行一条评测记录；含 model_output：仅来自接口解析结果，与 gold_output 分离）",
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
    parser.add_argument(
        "--skip-health-check",
        action="store_true",
        help="跳过启动前 GET /health（不推荐；无服务时会写满 Connection refused）",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.is_file():
        print(f"error: 找不到数据文件: {data_path}", file=sys.stderr)
        sys.exit(1)

    base = (args.base_url or "").strip() or f"http://{args.host}:{args.port}"
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.skip_health_check:
        ok_h, err_h = check_health(base, timeout_sec=min(10.0, args.timeout))
        if not ok_h:
            print(
                f"error: 无法连接推理服务 {base}（GET /health 失败: {err_h}）。\n"
                "请先在本机启动 serve_qwen35_full_http.py（或 SSH 转发后再跑），\n"
                "或传正确 --base-url；确需无服务跑批请加 --skip-health-check。",
                file=sys.stderr,
            )
            sys.exit(1)

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
            model_out = extract_model_output(resp)
            model_reasoning: Any = None
            if isinstance(model_out, dict):
                model_reasoning = model_out.get("reasoning")

            rec = {
                "post_id": post_id,
                "gold_output": gold,
                "model_output": model_out,
                "model_reasoning": model_reasoning,
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
