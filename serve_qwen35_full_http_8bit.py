#!/usr/bin/env python3
"""
Qwen 全量权重 HTTP 服务：优先 BitsAndBytes **INT8**（load_in_8bit），非 P100。

- 依赖：transformers + bitsandbytes，CUDA GPU。
- P100：部分环境 int8 cublasLt 不稳定，自动改 **4bit (NF4)**，与 serve_qwen35_full_http_4bit 行为一致。
- 接口：GET /health ，POST /generate JSON {instruction, content, max_new_tokens?}。

默认端口 8014（与 4bit 脚本默认 8013 错开）。模型路径请用已 merge 的 FP16/BF16 目录。
"""
from __future__ import annotations

import argparse
import json
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


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
        for rel in rels:
            if not isinstance(rel, dict):
                continue
            brand = str(rel.get("brand_text", "")).strip()
            if not brand:
                continue
            out["relationships"].append(
                {
                    "brand_text": brand,
                    "start": str(rel.get("start", "")).strip(),
                    "end": str(rel.get("end", "")).strip(),
                }
            )
    return out


def _strip_thinking_text(text: str) -> str:
    t = text.strip()
    markers = ["Thinking Process:", "思考过程", "<think>", "</think>"]
    for m in markers:
        idx = t.find(m)
        if idx >= 0:
            t = t[idx + len(m) :].strip()
    return t


def _heuristic_fallback(content: str) -> Dict[str, Any]:
    rels = []
    seen = set()
    for m in re.finditer(r"@[^@#\n\r]+", content):
        raw = m.group(0).strip()
        brand = raw.lstrip("@").strip(" ,.;:!?")
        if not brand:
            continue
        key = (brand.lower(), m.start())
        if key in seen:
            continue
        seen.add(key)
        rels.append(
            {
                "brand_text": brand,
                "start": str(m.start()),
                "end": str(m.end()),
            }
        )

    lc = content.lower()
    beauty_keys = [
        "beauty",
        "cosmetic",
        "cosmetics",
        "skincare",
        "makeup",
        "lipstick",
        "fragrance",
        "perfume",
    ]
    is_beauty = any(k in lc for k in beauty_keys) or bool(rels)
    reason = (
        "keyword/mention based fallback parser used"
        if is_beauty
        else "no beauty keyword found in fallback parser"
    )
    return {"is_beauty": is_beauty, "reasoning": reason, "relationships": rels}


def build_parser() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Qwen 全量 HTTP 服务（BitsAndBytes INT8，P100 时回退 4bit）"
    )
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=8014)
    p.add_argument(
        "--model_path",
        type=str,
        default="training_runs/best_B_full_20260425_184108_merged",
    )
    p.add_argument("--max_new_tokens", type=int, default=128)
    return p.parse_args()


def create_state(args: argparse.Namespace) -> Dict[str, Any]:
    print(f"[load] model_path={args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    mode = "8bit"
    gpu_name = ""
    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            gpu_name = ""

    if "P100" in gpu_name.upper():
        print(f"[warn] detected gpu={gpu_name}, int8 unstable on some drivers; use 4bit fallback")
        mode = "4bit_fallback"
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            quantization_config=bnb,
            device_map="auto",
            torch_dtype=torch.float16,
        )
    else:
        bnb = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            quantization_config=bnb,
            device_map="auto",
            torch_dtype=torch.float16,
        )
    model.eval()
    print(f"[load] model ready ({mode})")
    return {"tokenizer": tokenizer, "model": model, "args": args, "mode": mode}


def make_handler(state: Dict[str, Any]):
    tokenizer = state["tokenizer"]
    model = state["model"]
    args = state["args"]
    mode = state.get("mode", "unknown")

    class Handler(BaseHTTPRequestHandler):
        def _generate_once(self, instruction: str, content: str, max_new_tokens: int) -> str:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "你是一个严谨的结构化信息抽取助手。"
                        "只输出一个合法JSON对象，禁止输出思考过程、解释、markdown。"
                    ),
                },
                {
                    "role": "user",
                    "content": f"指令：{instruction}\n\n文本：{content}\n\n仅输出JSON。",
                },
            ]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            return tokenizer.decode(
                out[0][inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            ).strip()

        def _resp(self, status: int, payload: Dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            if self.path == "/health":
                self._resp(200, {"ok": True, "mode": mode, "script": "serve_qwen35_full_http_8bit"})
                return
            self._resp(404, {"error": "not_found"})

        def do_POST(self):  # noqa: N802
            if self.path != "/generate":
                self._resp(404, {"error": "not_found"})
                return

            t0 = time.perf_counter()
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
            except Exception as exc:
                self._resp(400, {"error": f"invalid_json: {exc}"})
                return

            instruction = str(payload.get("instruction", "")).strip()
            content = str(payload.get("content", "")).strip()
            if not instruction or not content:
                self._resp(400, {"error": "instruction and content are required"})
                return

            max_new_tokens = min(int(payload.get("max_new_tokens", args.max_new_tokens)), 256)
            try:
                text_raw = self._generate_once(instruction, content, max_new_tokens)
            except Exception as exc:
                elapsed_ms = int((time.perf_counter() - t0) * 1000)
                self._resp(
                    500,
                    {
                        "ok": False,
                        "elapsed_ms": elapsed_ms,
                        "error": f"generate_failed: {type(exc).__name__}: {exc}",
                    },
                )
                return

            text = _strip_thinking_text(text_raw)
            parsed = None
            try:
                parsed = json.loads(text)
            except Exception:
                parsed = _extract_first_json_object(text)

            if parsed is None:
                repair_instruction = (
                    "你上一次没有按要求输出。"
                    "请仅返回一个JSON对象，键必须是 is_beauty, reasoning, relationships。"
                )
                try:
                    text_retry = self._generate_once(
                        repair_instruction, content, max_new_tokens
                    )
                    text_retry = _strip_thinking_text(text_retry)
                    try:
                        parsed = json.loads(text_retry)
                    except Exception:
                        parsed = _extract_first_json_object(text_retry)
                    if parsed is not None:
                        text = text_retry
                except Exception:
                    pass

            parsed = _normalize_output(parsed)
            fallback_used = False
            if parsed is None:
                parsed = _heuristic_fallback(content)
                fallback_used = True

            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            self._resp(
                200,
                {
                    "ok": True,
                    "elapsed_ms": elapsed_ms,
                    "output_text": text,
                    "output_json": parsed,
                    "fallback_used": fallback_used,
                },
            )

        def log_message(self, format: str, *args_) -> None:
            return

    return Handler


def main() -> None:
    args = build_parser()
    state = create_state(args)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    print(f"[serve] serve_qwen35_full_http_8bit http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
