#!/usr/bin/env python3
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA HTTP inference server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument(
        "--adapter_path", type=str, default="distilled-qwen-f32_full_20260424"
    )
    parser.add_argument("--max_new_tokens", type=int, default=256)
    return parser.parse_args()


def create_app_state(args: argparse.Namespace) -> Dict[str, Any]:
    print(f"[load] base_model={args.base_model}")
    print(f"[load] adapter={args.adapter_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()
    print("[load] model ready")
    return {"tokenizer": tokenizer, "model": model, "args": args}


def make_handler(state: Dict[str, Any]):
    tokenizer = state["tokenizer"]
    model = state["model"]
    args = state["args"]

    class Handler(BaseHTTPRequestHandler):
        def _json_response(self, status: int, payload: Dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            if self.path == "/health":
                self._json_response(200, {"ok": True})
                return
            self._json_response(404, {"error": "not_found"})

        def do_POST(self):  # noqa: N802
            if self.path != "/generate":
                self._json_response(404, {"error": "not_found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length)
                payload = json.loads(raw.decode("utf-8"))
            except Exception as exc:
                self._json_response(400, {"error": f"invalid_json: {exc}"})
                return

            instruction = str(payload.get("instruction", "")).strip()
            content = str(payload.get("content", "")).strip()
            if not instruction or not content:
                self._json_response(
                    400, {"error": "instruction and content are required"}
                )
                return

            max_new_tokens = int(payload.get("max_new_tokens", args.max_new_tokens))
            messages = [
                {"role": "system", "content": "你是一个严谨的结构化信息抽取助手。"},
                {
                    "role": "user",
                    "content": f"指令：{instruction}\n\n文本：{content}",
                },
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
                out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            ).strip()

            parsed = None
            try:
                parsed = json.loads(text)
            except Exception:
                pass

            self._json_response(
                200,
                {
                    "ok": True,
                    "output_text": text,
                    "output_json": parsed,
                },
            )

        def log_message(self, format: str, *args_) -> None:
            # Keep server output concise in nohup logs.
            return

    return Handler


def main() -> None:
    args = build_parser()
    state = create_app_state(args)
    handler = make_handler(state)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"[serve] http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
