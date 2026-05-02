#!/usr/bin/env python3
import argparse
import inspect
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 必须与 new_mian.py 一致，否则微调权重对不上分布，易出现长废话、不出 JSON
DEFAULT_SYSTEM_PROMPT = "You are a strict structured information extraction assistant."


def _build_user_content(instruction: str, content: str) -> str:
    return f"Instruction:\n{instruction}\n\nContent:\n{content}"


_THINK_END_MARKERS = ("</" + "think" + ">", "</" + "thinking" + ">")


def _is_instruction_schema_echo(obj: Dict[str, Any]) -> bool:
    r = str(obj.get("reasoning", "")).strip().lower()
    if r in ("briefly state why.", "briefly state why"):
        return True
    rels = obj.get("relationships")
    if not isinstance(rels, list) or len(rels) != 1:
        return False
    rel0 = rels[0] if rels else None
    if not isinstance(rel0, dict):
        return False
    if str(rel0.get("brand_text", "")).strip() == "Exact Brand Name":
        return True
    return False


def _extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    """扫描所有 `{...}`；优先返回含 is_beauty 且非指令示例回显的最后一个 dict。"""
    decoder = json.JSONDecoder()
    candidates: list[Dict[str, Any]] = []
    idx = 0
    while True:
        j = text.find("{", idx)
        if j < 0:
            break
        try:
            obj, end = decoder.raw_decode(text[j:])
            if isinstance(obj, dict) and (
                "is_beauty" in obj or isinstance(obj.get("relationships"), list)
            ):
                candidates.append(obj)
            idx = j + max(1, end)
        except Exception:
            idx = j + 1
    for obj in reversed(candidates):
        if not _is_instruction_schema_echo(obj):
            return obj
    return None


def _strip_thinking_text(text: str) -> str:
    """Qwen3.x：JSON 在 </think> 之后；Thinking Process 段内常有 Schema 的 `{`，需从首个 `{` 起截。"""
    t = text.strip()
    for end in _THINK_END_MARKERS:
        if end in t:
            t = t[t.rfind(end) + len(end) :].strip()
    tp = "Thinking Process:"
    if tp in t:
        i = t.find(tp)
        brace = t.find("{", i)
        if brace >= 0:
            return t[brace:].strip()
        return t[:i].strip()
    return t


def _normalize_output(obj: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(obj, dict):
        return None
    rels_pre = obj.get("relationships")
    if "is_beauty" in obj:
        inferred_beauty = bool(obj.get("is_beauty"))
    else:
        inferred_beauty = isinstance(rels_pre, list) and len(rels_pre) > 0
    out: Dict[str, Any] = {
        "is_beauty": inferred_beauty,
        "reasoning": str(obj.get("reasoning", "")),
        "relationships": [],
    }
    rels = obj.get("relationships", [])
    if isinstance(rels, list):
        for r in rels:
            if not isinstance(r, dict):
                continue
            brand = str(r.get("brand_text", "")).strip()
            if not brand:
                continue
            out["relationships"].append({"brand_text": brand, "start": "", "end": ""})
    return out


def _postprocess_relationships(obj: Optional[Dict[str, Any]], content: str) -> Optional[Dict[str, Any]]:
    if not isinstance(obj, dict):
        return obj
    rels = obj.get("relationships", [])
    if not isinstance(rels, list):
        obj["relationships"] = []
        return obj

    deny_tokens = ("official", "shop", "store", "creator", "avenue")
    out = []
    seen = set()
    text_l = content.lower()

    for r in rels:
        brand = str(r.get("brand_text", "")).strip()
        if not brand:
            continue

        brand_clean = brand[1:].strip() if brand.startswith("@") else brand
        b_l = brand_clean.lower()

        if any(tok in b_l for tok in deny_tokens):
            continue

        pos = text_l.find(b_l)
        if pos >= 0:
            end = pos + len(brand_clean)
            key = (b_l, pos)
            if key not in seen:
                seen.add(key)
                out.append({
                    "brand_text": brand_clean,
                    "start": str(pos),
                    "end": str(end),
                })

    obj["relationships"] = out
    return obj


NO_THINK_PREFIX = "/no_think\n\n"
JSON_HINT_SUFFIX = (
    "\n\nAfter brief analysis, output exactly one JSON object matching the schema "
    "(do not copy the Example Output)."
)

# 低于此值的 max_new_tokens 极易在「思考/前缀」后截断，导致 output_json 为 null
MIN_EFFECTIVE_MAX_NEW_TOKENS = 512


def make_handler(state):
    tokenizer, model, args = state["tokenizer"], state["model"], state["args"]

    class Handler(BaseHTTPRequestHandler):
        def _resp(self, status, payload):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            if self.path == "/health":
                self._resp(200, {"ok": True, "mode": "fp16"})
                return
            self._resp(404, {"error": "not_found"})

        def do_POST(self):
            if self.path != "/generate":
                self._resp(404, {"error": "not_found"})
                return
            t0 = time.perf_counter()
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                instruction = str(payload.get("instruction", "")).strip()
                content = str(payload.get("content", "")).strip()
                req_max = int(payload.get("max_new_tokens", args.max_new_tokens))
                req_max = max(64, min(req_max, 8192))
                allow_low = bool(payload.get("allow_under_min_tokens"))
                if req_max < MIN_EFFECTIVE_MAX_NEW_TOKENS and not allow_low:
                    max_tokens = MIN_EFFECTIVE_MAX_NEW_TOKENS
                    token_bump_note = (
                        f"已将 max_new_tokens 从请求的 {req_max} 提升为 {max_tokens}，"
                        "避免生成在 JSON 完成前被截断。若坚持用小值请传 allow_under_min_tokens: true。"
                    )
                else:
                    max_tokens = req_max
                    token_bump_note = ""
                debug_parse = bool(payload.get("debug_parse"))
                use_hint = bool(payload.get("append_json_hint", args.append_json_hint))

                user_text = _build_user_content(instruction, content)
                if use_hint:
                    user_text = NO_THINK_PREFIX + user_text + JSON_HINT_SUFFIX

                messages = [
                    {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_text},
                ]
                tpl_kw: Dict[str, Any] = {}
                try:
                    sig = inspect.signature(tokenizer.apply_chat_template)
                    if "chat_template_kwargs" in sig.parameters:
                        tpl_kw["chat_template_kwargs"] = {"enable_thinking": False}
                except (TypeError, ValueError, AttributeError):
                    pass
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, **tpl_kw
                )

                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    gen = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)

                output_text = tokenizer.decode(
                    gen[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
                )

                stripped = _strip_thinking_text(output_text)
                parsed = _extract_first_json_object(stripped)
                norm = _normalize_output(parsed)
                if args.postprocess_relationships:
                    final_json = _postprocess_relationships(norm, content)
                else:
                    final_json = norm

                elapsed_ms = int((time.perf_counter() - t0) * 1000)
                out_body: Dict[str, Any] = {
                    "ok": True,
                    "elapsed_ms": elapsed_ms,
                    "output_json": final_json,
                    "max_new_tokens_used": max_tokens,
                }
                if token_bump_note:
                    out_body["note"] = token_bump_note
                if debug_parse:
                    lim = 12000
                    out_body["raw_output"] = output_text[:lim]
                    out_body["stripped_for_json"] = stripped[:lim]
                self._resp(200, out_body)
            except Exception as e:
                self._resp(500, {"error": str(e)})

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument(
        "--postprocess_relationships",
        action="store_true",
        help="按原文匹配并补 start/end、过滤部分词（默认关闭，与训练标签一致）",
    )
    parser.add_argument(
        "--append-json-hint",
        dest="append_json_hint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否在用户侧加 /no_think 与 JSON 提示（默认开；可用 --no-append-json-hint 关闭）",
    )
    args = parser.parse_args()

    print(f"Loading model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map="auto"
    )

    server = ThreadingHTTPServer(
        ("0.0.0.0", args.port),
        make_handler({"tokenizer": tokenizer, "model": model, "args": args}),
    )
    print(f"Server started at port {args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
