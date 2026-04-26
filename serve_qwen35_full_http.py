#!/usr/bin/env python3
import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


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
            messages = [
                {"role": "system", "content": "你是一个严谨的结构化信息抽取助手。"},
                {"role": "user", "content": f"指令：{instruction}\n\n文本：{content}"},
            ]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
            text = tokenizer.decode(
                out[0][inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            ).strip()

            parsed = None
            try:
                parsed = json.loads(text)
            except Exception:
                pass

            elapsed_ms = int((time.perf_counter() - t0) * 1000)
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
