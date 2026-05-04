#!/usr/bin/env python3
"""
逐步遍历 data/train_v2.jsonl，按 vLLM /v1/chat/completions 格式请求（Instruction + Content 与 val 一致），
将耗时、post_id、数据集中的标注 output、以及接口返回的 message.content 写入日志。
可选 --instruction-file：用固定 UTF-8 提示词覆盖每条样本的 instruction（仅换 content）。

本仓库约定：vLLM 默认监听 127.0.0.1:8000（勿与仅作 SSH 转发示例的 8080 混淆）。自检示例：
  curl -sS http://127.0.0.1:8000/v1/models | head
  对应 --url 默认为 http://127.0.0.1:8000/v1/chat/completions；若你把远端映射到本机其它端口，再改 --url。
  默认会 GET 同源的 /v1/models，读取 max_model_len（含 LoRA parent）并按 prompt 长度自动收紧 max_tokens，减少 400；可用 --no-auto-cap-max-tokens 关闭。

默认日志每行格式（TAB 分隔）：
  time_sec<TAB>post_id<TAB>output_json<TAB>content_json

- time_sec：单次 HTTP 请求耗时（秒）；跳过请求时为 0。
- output_json：来自 jsonl 行里的「output」字段（dict/list）经紧凑 json.dumps；缺失或类型不对时
  为 {"missing_or_invalid_output_field": ...}。
- content_json：优先为模型正文 choices[0].message.content（json.dumps 包一层）；--content-plain 时压成单行。
  若正文为空：失败请求写入 {"http_layer_error", "raw_head"}；成功但无正文则写入 {"assistant_text_empty", "message"|"choice0"} 便于对照原始返回。

输出习惯：默认不把每行完整 TSV 打到 stdout（终端/管道易截断，看起来像「不完整」）；完整内容只写入 --log。
  需要旧行为时加 --stdout-lines；进度见 stderr 每行简短摘要。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

_BENCH_FAILED = "_bench_http_failed"


def chat_completions_url_to_models_url(chat_url: str) -> str:
    suf = "/v1/chat/completions"
    if chat_url.rstrip("/").endswith(suf):
        base = chat_url[: -len(suf)].rstrip("/")
        return base + "/v1/models"
    return chat_url.rstrip("/") + "/v1/models"


def fetch_max_model_len_for_model(
    models_url: str, model_id: str, timeout_sec: float
) -> Optional[int]:
    """从 OpenAI 兼容 /v1/models 取 max_model_len；LoRA 常为 null 则继承 parent。"""
    try:
        req = urllib.request.Request(models_url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout_sec) as r:
            payload = json.loads(r.read().decode("utf-8"))
    except Exception:
        return None
    data = payload.get("data")
    if not isinstance(data, list):
        return None
    by_id: Dict[str, Dict[str, Any]] = {
        str(m["id"]): m for m in data if isinstance(m, dict) and m.get("id") is not None
    }
    m = by_id.get(model_id)
    if not isinstance(m, dict):
        return None
    lim = m.get("max_model_len")
    if isinstance(lim, int) and lim > 0:
        return lim
    parent_id = m.get("parent")
    if isinstance(parent_id, str) and parent_id in by_id:
        plim = by_id[parent_id].get("max_model_len")
        if isinstance(plim, int) and plim > 0:
            return plim
    return None


def estimate_prompt_tokens_upper_bound(user_content: str) -> int:
    """
    无 tokenizer 时对「输入 token 上界」的保守估计，避免 max_tokens 过大触发 400。
    偏保守：短文本约 2 字符/token + 模板/role 开销。
    """
    n = len(user_content)
    # 略抬高上界，减少 tokenizer 与 chat 模板导致的 400
    return max(64, n // 2 + 180)


def clamp_max_tokens_for_context(
    user_content: str,
    requested_max_tokens: int,
    max_model_len: Optional[int],
    reserved_completion_overhead: int = 24,
) -> int:
    """在已知 max_model_len 时收紧 max_tokens；未知则返回 requested。"""
    if max_model_len is None or max_model_len <= 0:
        return requested_max_tokens
    est_in = estimate_prompt_tokens_upper_bound(user_content)
    room = max_model_len - est_in - reserved_completion_overhead
    return max(1, min(requested_max_tokens, room))


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
    """失败时 dict 带 _bench_http_failed（避免与 API 自带的 ok/error 键冲突）。"""
    body: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": user_content}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
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
    except urllib.error.HTTPError as exc:
        elapsed = time.perf_counter() - t0
        try:
            err_body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            err_body = ""
        return elapsed, {
            _BENCH_FAILED: True,
            "error": f"HTTPError {exc.code}: {exc.reason}",
            "raw": err_body[:4000],
        }
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return elapsed, {
            _BENCH_FAILED: True,
            "error": f"{type(exc).__name__}: {exc}",
        }
    elapsed = time.perf_counter() - t0
    parsed, err = parse_openai_chat_response_text(text)
    if parsed is not None:
        if isinstance(parsed, dict):
            # 第四列回退时对照原始 HTTP 正文（截断）
            parsed = {**parsed, "_bench_http_raw": text[:12000]}
        return elapsed, parsed
    return elapsed, {
        _BENCH_FAILED: True,
        "error": err or "invalid_json_response",
        "raw": text[:4000],
    }


def parse_openai_chat_response_text(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """整段 JSON，或 SSE `data: {...}` 流（拼接 delta.content）。"""
    text = text.strip()
    if not text:
        return None, "empty_response_body"
    try:
        return json.loads(text), None
    except json.JSONDecodeError:
        pass

    merged_delta: List[str] = []
    last_obj: Optional[Dict[str, Any]] = None
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            continue
        last_obj = obj
        choices = obj.get("choices")
        if not isinstance(choices, list) or not choices:
            continue
        ch0 = choices[0]
        if not isinstance(ch0, dict):
            continue
        delta = ch0.get("delta")
        if isinstance(delta, dict):
            frag = _delta_content_to_str(delta.get("content"))
            if frag:
                merged_delta.append(frag)

    if merged_delta:
        return (
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "".join(merged_delta),
                        }
                    }
                ]
            },
            None,
        )
    if last_obj is not None:
        return last_obj, None
    return None, "invalid_json_response"


def _delta_content_to_str(raw: Any) -> str:
    """流式 delta.content：str 或 OpenAI 多段 list。"""
    if isinstance(raw, str):
        return raw
    if raw is None:
        return ""
    if isinstance(raw, list):
        return _message_content_to_str(raw)
    return ""


def _message_content_to_str(raw: Any) -> str:
    """OpenAI / vLLM 里 message.content 可能是 str、null、或多段 list。"""
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        parts: list[str] = []
        for item in raw:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                t = item.get("type")
                if t == "text" and isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif isinstance(item.get("text"), str):
                    parts.append(item["text"])
        return "".join(parts)
    if isinstance(raw, dict) and isinstance(raw.get("text"), str):
        return raw["text"]
    return ""


def extract_assistant_content(resp: Dict[str, Any]) -> str:
    choices = resp.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    ch0 = choices[0]
    if not isinstance(ch0, dict):
        return ""
    tx = ch0.get("text")
    if isinstance(tx, str) and tx.strip():
        return tx
    msg = ch0.get("message")
    if not isinstance(msg, dict):
        return ""
    c = msg.get("content")
    s = _message_content_to_str(c)
    if not s:
        rc = msg.get("reasoning_content")
        if isinstance(rc, str):
            s = rc
        elif isinstance(rc, list):
            s = _message_content_to_str(rc)
    if not s:
        delta = ch0.get("delta")
        if isinstance(delta, dict):
            s = _delta_content_to_str(delta.get("content"))
    return s


def content_json_column(
    resp: Dict[str, Any],
    assistant_text: str,
    plain: bool,
    compact_fn: Callable[[Any], str],
) -> str:
    """第四列：有模型正文则按原规则；否则写入紧凑 JSON（含 choice0 / 原始 HTTP 前缀）。"""
    if assistant_text:
        return format_content_log_field(assistant_text, plain)
    raw = ""
    if isinstance(resp, dict):
        raw = str(resp.get("_bench_http_raw") or "")
    if resp.get(_BENCH_FAILED):
        return compact_fn(
            {
                "http_layer_error": resp.get("error"),
                "raw_head": (resp.get("raw") or raw or "")[:2000],
            }
        )
    ch = resp.get("choices")
    if isinstance(ch, list) and ch and isinstance(ch[0], dict):
        return compact_fn(
            {
                "assistant_text_empty": True,
                "choice0": ch[0],
                "http_body_prefix": raw[:2000],
            }
        )
    return compact_fn(
        {
            "assistant_text_empty": True,
            "no_choices": True,
            "keys": sorted(resp.keys()) if isinstance(resp, dict) else [],
            "http_body_prefix": raw[:2000],
        }
    )


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
    p = argparse.ArgumentParser(description="val.jsonl -> vLLM chat bench + log")
    p.add_argument("--data", type=str, default="data/val.jsonl")
    p.add_argument(
        "--url",
        type=str,
        default="http://127.0.0.1:8000/v1/chat/completions",
        help="vLLM OpenAI 兼容 chat 接口完整 URL",
    )
    p.add_argument("--model", type=str, default="qwen-3.5-4b")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument(
        "--no-auto-cap-max-tokens",
        action="store_true",
        help="关闭按 max_model_len 与 prompt 长度自动收紧 max_tokens（调试用，易触发 400）",
    )
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
    p.add_argument(
        "--instruction-file",
        type=str,
        default="",
        help="若非空：从该 UTF-8 文件读取整段作为 instruction，忽略 JSONL 里的 instruction 字段",
    )
    p.add_argument(
        "--stdout-lines",
        action="store_true",
        help="把每条完整 TSV 打到 stdout（行很长时易被终端截断；默认不写 stdout，只写 --log）",
    )
    args = p.parse_args()

    fixed_instruction = ""
    if str(args.instruction_file).strip():
        inst_path = Path(args.instruction_file)
        if not inst_path.is_file():
            print(f"error: --instruction-file 不是有效文件: {inst_path}", file=sys.stderr)
            sys.exit(1)
        fixed_instruction = inst_path.read_text(encoding="utf-8").strip()
        if not fixed_instruction:
            print(f"error: 提示词文件为空: {inst_path}", file=sys.stderr)
            sys.exit(1)

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

    models_url = chat_completions_url_to_models_url(args.url)
    ctx_limit: Optional[int] = None
    if not args.no_auto_cap_max_tokens:
        ctx_limit = fetch_max_model_len_for_model(
            models_url, args.model, min(15.0, args.timeout)
        )
        if ctx_limit is not None:
            print(
                f"[bench] auto-cap max_tokens: max_model_len={ctx_limit} "
                f"(from {models_url}; disable: --no-auto-cap-max-tokens)",
                file=sys.stderr,
            )
        else:
            print(
                f"[bench] auto-cap skipped: could not read max_model_len from {models_url}",
                file=sys.stderr,
            )

    n = 0
    with log_path.open("w", encoding="utf-8", newline="\n") as logf:
        for row in iter_jsonl(data_path):
            if args.limit and n >= args.limit:
                break
            post_id = str(row.get("post_id", ""))
            instruction = str(row.get("instruction", "")).strip()
            instruction = (
                fixed_instruction
                if fixed_instruction
                else str(row.get("instruction", "")).strip()
            )
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
                if args.stdout_lines:
                    print(line, flush=True)
                else:
                    print(
                        f"[bench] line {n + 1}\t{post_id}\tskipped_missing_prompt_or_content",
                        file=sys.stderr,
                        flush=True,
                    )
                n += 1
                continue

            user_content = f"Instruction:\n{instruction}\n\nContent:\n{content}"
            eff_max_tokens = (
                args.max_tokens
                if args.no_auto_cap_max_tokens
                else clamp_max_tokens_for_context(
                    user_content, args.max_tokens, ctx_limit
                )
            )
            elapsed, resp = post_chat(
                args.url,
                args.model,
                user_content,
                eff_max_tokens,
                args.temperature,
                args.timeout,
            )

            assistant_text = (
                "" if resp.get(_BENCH_FAILED) else extract_assistant_content(resp)
            )

            # 第三列：数据集中的 output（标注）；第四列：模型正文，或空时的回退 JSON
            out_json = dataset_output_compact(row, compact)
            content_json = content_json_column(
                resp, assistant_text, args.content_plain, compact
            )
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
            if args.stdout_lines:
                print(line, flush=True)
            else:
                st = (
                    "http_err"
                    if "http_layer_error" in content_json
                    else (
                        "empty_out"
                        if "assistant_text_empty" in content_json
                        else "ok"
                    )
                )
                print(
                    f"[bench] line {n + 1}\t{post_id}\t{elapsed:.6f}s\t{st}",
                    file=sys.stderr,
                    flush=True,
                )
            n += 1
            if args.sleep > 0:
                time.sleep(args.sleep)
        try:
            os.fsync(logf.fileno())
        except OSError:
            pass

    ap = log_path.resolve()
    print(
        f"done: {n} lines -> {ap}"
        + (" (--stdout-lines: 每行已同步打印到 stdout)" if args.stdout_lines else "（完整日志仅在文件；请加 --stdout-lines 才打印整行到 stdout）"),
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()

