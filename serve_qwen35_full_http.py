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
    if idx < 0: return None
    decoder = json.JSONDecoder()
    while idx >= 0 and idx < len(text):
        try:
            obj, _ = decoder.raw_decode(text[idx:])
            if isinstance(obj, dict): return obj
        except Exception:
            pass
        idx = text.find("{", idx + 1)
    return None


def _strip_thinking_text(text: str) -> str:
    for marker in ("Thinking Process:", "</think>"):
        idx = text.find(marker)
        if idx >= 0: text = text[:idx]
    return text.strip()


def _normalize_output(obj: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(obj, dict): return None
    out: Dict[str, Any] = {
        "is_beauty": bool(obj.get("is_beauty", False)),
        "reasoning": str(obj.get("reasoning", "")),
        "relationships": [],
    }
    rels = obj.get("relationships", [])
    if isinstance(rels, list):
        for r in rels:
            if not isinstance(r, dict): continue
            brand = str(r.get("brand_text", "")).strip()
            if not brand: continue
            out["relationships"].append({"brand_text": brand, "start": "", "end": ""})
    return out


def _postprocess_relationships(obj: Optional[Dict[str, Any]], content: str) -> Optional[Dict[str, Any]]:
    if not isinstance(obj, dict): return obj
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
        if not brand: continue

        # 预处理：去掉 @ 符号
        brand_clean = brand[1:].strip() if brand.startswith("@") else brand
        b_l = brand_clean.lower()

        # 过滤掉明显的店铺/官方账号关键词
        if any(tok in b_l for tok in deny_tokens): continue

        # 检查是否在原文中出现
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

    # 【关键修改】：不再截断为 [:3]，保留所有识别结果
    obj["relationships"] = out
    return obj


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
                max_tokens = int(payload.get("max_new_tokens", args.max_new_tokens))

                # 与 new_mian.py 训练时 chat 模板一致：system 固定，用户消息里放请求体中的 instruction + content
                messages = [
                    {"role": "system", "content": "你是一个严谨的结构化信息抽取助手。"},
                    {
                        "role": "user",
                        "content": f"指令：{instruction}\n\n文本：{content}",
                    },
                ]
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    gen = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)

                output_text = tokenizer.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

                parsed = _extract_first_json_object(_strip_thinking_text(output_text))
                norm = _normalize_output(parsed)
                if args.postprocess_relationships:
                    final_json = _postprocess_relationships(norm, content)
                else:
                    final_json = norm

                elapsed_ms = int((time.perf_counter() - t0) * 1000)
                self._resp(200, {
                    "ok": True,
                    "elapsed_ms": elapsed_ms,
                    "output_json": final_json
                })
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
    args = parser.parse_args()

    print(f"Loading model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16, device_map="auto")

    server = ThreadingHTTPServer(("0.0.0.0", args.port),
                                 make_handler({"tokenizer": tokenizer, "model": model, "args": args}))
    print(f"Server started at port {args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()