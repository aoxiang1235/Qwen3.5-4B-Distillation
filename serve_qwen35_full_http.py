#!/usr/bin/env python3
import argparse
import json
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    idx = text.find("{")
    if idx < 0:
        return None
    decoder = json.JSONDecoder()
    while idx >= 0 and idx < len(text):
        try:
            obj, _ = decoder.raw_decode(text[idx:])
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        idx = text.find("{", idx + 1)
    return None


def _strip_thinking_text(text: str) -> str:
    # Remove common chain-of-thought markers before JSON extraction.
    for marker in ("Thinking Process:", "</think>"):
        idx = text.find(marker)
        if idx >= 0:
            text = text[:idx]
    return text.strip()


def _normalize_output(obj: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(obj, dict):
        return None
    out: Dict[str, Any] = {
        "is_beauty": bool(obj.get("is_beauty", False)),
        "reasoning": str(obj.get("reasoning", "")),
        "relationships": [],
    }
    rels = obj.get("relationships", [])
    if isinstance(rels, list):
        for r in rels:
            if not isinstance(r, dict):
                continue
            brand = str(r.get("brand_text", "")).strip()
            start = str(r.get("start", "")).strip()
            end = str(r.get("end", "")).strip()
            if not brand:
                continue
            out["relationships"].append(
                {"brand_text": brand, "start": start, "end": end}
            )
    return out


def _quality_score(obj: Optional[Dict[str, Any]]) -> int:
    if not isinstance(obj, dict):
        return -100
    rels = obj.get("relationships", [])
    if not isinstance(rels, list):
        return -50
    score = 10
    for r in rels:
        if not isinstance(r, dict):
            score -= 2
            continue
        brand = str(r.get("brand_text", "")).strip()
        if not brand:
            score -= 2
            continue
        score += 2
        if brand.startswith("@"):
            score -= 2
        # Heuristic: likely person-like @name surname
        body = brand[1:] if brand.startswith("@") else brand
        if " " in body and len(body.split()) == 2 and all(len(x) > 1 for x in body.split()):
            score -= 4
    return score


def _postprocess_relationships(
    obj: Optional[Dict[str, Any]], content: str
) -> Optional[Dict[str, Any]]:
    if not isinstance(obj, dict):
        return obj
    rels = obj.get("relationships", [])
    if not isinstance(rels, list):
        obj["relationships"] = []
        return obj

    deny_tokens = ("avenue", "official", "shop", "store", "creator")
    out = []
    seen = set()
    text_l = content.lower()

    for r in rels:
        if not isinstance(r, dict):
            continue
        brand = str(r.get("brand_text", "")).strip()
        if not brand:
            continue

        # Normalize "from the house of X" to X.
        m = re.search(r"from the house of\\s+([a-z0-9_\\-]+)", brand, flags=re.I)
        if m:
            brand = m.group(1)

        # Strip @ prefix for brand matching.
        brand_no_at = brand[1:].strip() if brand.startswith("@") else brand
        b_l = brand_no_at.lower()

        # Remove likely person-like account names.
        parts = [p for p in brand_no_at.split() if p]
        if len(parts) == 2 and all(p.isalpha() for p in parts):
            continue

        # Remove likely channel/store handles.
        if any(tok in b_l for tok in deny_tokens):
            continue

        # Keep only if appears in content.
        pos = text_l.find(b_l)
        if pos < 0:
            continue
        end = pos + len(brand_no_at)
        key = (b_l, pos, end)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "brand_text": brand_no_at,
                "start": str(pos),
                "end": str(end),
            }
        )

    obj["relationships"] = out[:3]
    return obj


def build_parser() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Qwen3.5-4B full-model HTTP server")
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument(
        "--model_path",
        type=str,
        default="training_runs/best_B_full_20260425_184108_merged",
    )
    p.add_argument("--max_new_tokens", type=int, default=256)
    return p.parse_args()


def create_state(args: argparse.Namespace) -> Dict[str, Any]:
    print(f"[load] model_path={args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    print("[load] model ready")
    return {"tokenizer": tokenizer, "model": model, "args": args}


def make_handler(state: Dict[str, Any]):
    tokenizer = state["tokenizer"]
    model = state["model"]
    args = state["args"]

    class Handler(BaseHTTPRequestHandler):
        def _resp(self, status: int, payload: Dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            if self.path == "/health":
                self._resp(200, {"ok": True})
                return
            self._resp(404, {"error": "not_found"})

        def do_POST(self):  # noqa: N802
            if self.path != "/generate":
                self._resp(404, {"error": "not_found"})
                return

            t0 = time.perf_counter()
            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length)
                payload = json.loads(raw.decode("utf-8"))
            except Exception as exc:
                self._resp(400, {"error": f"invalid_json: {exc}"})
                return

            instruction = str(payload.get("instruction", "")).strip()
            content = str(payload.get("content", "")).strip()
            if not instruction or not content:
                self._resp(400, {"error": "instruction and content are required"})
                return

            max_new_tokens = int(payload.get("max_new_tokens", args.max_new_tokens))
            base_system = (
                "你是一个严谨的结构化信息抽取助手。"
                "禁止输出思考过程、解释或额外文本；只输出一个合法 JSON 对象。"
            )
            base_user = (
                f"指令：{instruction}\n\n文本：{content}\n\n"
                "只返回一个 JSON 对象，且必须以 { 开头、以 } 结尾，不得输出任何额外文本。\n"
                'JSON schema: {"is_beauty": true/false, "reasoning": "short reason", '
                '"relationships": [{"brand_text": "...", "start": "...", "end": "..."}]}'
            )
            messages = [
                {
                    "role": "system",
                    "content": base_system,
                },
                {
                    "role": "user",
                    "content": base_user,
                },
            ]
            prefix = '{"is_beauty": '

            def _run_once(msgs, tokens):
                prompt = tokenizer.apply_chat_template(
                    msgs,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                prompt = prompt + prefix
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=tokens,
                        do_sample=False,
                    )
                text_raw = (
                    prefix
                    + tokenizer.decode(
                        out[0][inputs["input_ids"].shape[1] :],
                        skip_special_tokens=True,
                    ).strip()
                )
                cleaned = _strip_thinking_text(text_raw)
                parsed_raw = None
                try:
                    parsed_raw = json.loads(cleaned)
                except Exception:
                    parsed_raw = _extract_first_json_object(cleaned)
                norm = _normalize_output(parsed_raw)
                norm = _postprocess_relationships(norm, content)
                return cleaned, norm

            text, parsed = _run_once(messages, max_new_tokens)
            score = _quality_score(parsed)

            # Retry with anti-noise instruction when likely mis-extracting @accounts.
            if score < 8:
                retry_messages = [
                    {"role": "system", "content": base_system},
                    {
                        "role": "user",
                        "content": (
                            base_user
                            + "\n\n纠偏规则：忽略人名账号和店铺/渠道账号；"
                            "优先抽取品牌主体（例如 'from the house of armaf' 应抽取 armaf）。"
                        ),
                    },
                ]
                text2, parsed2 = _run_once(retry_messages, max(max_new_tokens, 1024))
                if _quality_score(parsed2) >= score:
                    text, parsed = text2, parsed2

            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            if parsed is None:
                self._resp(
                    422,
                    {
                        "ok": False,
                        "error": "non_json_model_output",
                        "elapsed_ms": elapsed_ms,
                        "output_text": "",
                        "output_json": None,
                    },
                )
                return
            self._resp(
                200,
                {
                    "ok": True,
                    "elapsed_ms": elapsed_ms,
                    "output_text": text,
                    "output_json": parsed,
                },
            )

        def log_message(self, format: str, *args_) -> None:
            return

    return Handler


def main() -> None:
    args = build_parser()
    st = create_state(args)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(st))
    print(f"[serve] http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
