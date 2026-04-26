#!/usr/bin/env python3
"""
LoRA HTTP inference server (strict JSON mode).
Fixed: JSON parsing, thinking process removal, output normalization, retry logic.
"""
import argparse
import json
import time
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

def _extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    first = text.find("{")
    last = text.rfind("}")
    if first < 0 or last < first:
        return None
    cand = text[first:last+1]
    cand = re.sub(r",\s*}", "}", cand)
    cand = re.sub(r",\s*]", "]", cand)
    try:
        return json.loads(cand)
    except Exception:
        pass
    decoder = json.JSONDecoder()
    idx = first
    while idx <= last:
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
    out = {
        "is_beauty": False,
        "reasoning": "",
        "relationships": [],
    }
    out["is_beauty"] = bool(obj.get("is_beauty"))
    reasoning = str(obj.get("reasoning", "")).strip()
    out["reasoning"] = reasoning if reasoning else "Extracted beauty brands"
    rels = obj.get("relationships", [])
    if isinstance(rels, list):
        valid = []
        for r in rels:
            if not isinstance(r, dict):
                continue
            b = str(r.get("brand_text", "")).strip()
            s = str(r.get("start", "")).strip()
            e = str(r.get("end", "")).strip()
            if b:
                valid.append({"brand_text": b, "start": s, "end": e})
        out["relationships"] = valid
    return out

def build_parser():
    parser = argparse.ArgumentParser(description="LoRA HTTP strict JSON server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3.5-4B")
    parser.add_argument("--adapter_path", type=str, default="distilled-qwen-f32_full_20260424")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--retry_max_new_tokens", type=int, default=384)
    return parser.parse_args()

def create_app_state(args):
    print(f"[load] base={args.base_model}")
    print(f"[load] lora={args.adapter_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()
    print("[load] model ready ✅")
    return {"tokenizer": tokenizer, "model": model, "args": args}

def make_handler(state):
    tok = state["tokenizer"]
    model = state["model"]
    args = state["args"]

    class Handler(BaseHTTPRequestHandler):
        def _json(self, code, data):
            b = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def _gen(self, instruction, content, max_new, force_clean):
            sys_prompt = (
                "你是专业结构化抽取助手。只输出合法JSON，禁止任何思考过程、分析、列表、markdown。"
                "输出必须是单JSON对象，包含is_beauty, reasoning, relationships。"
            )
            if force_clean:
                sys_prompt += " 绝对禁止输出Thinking Process、解释、换行外多余符号。"
            user = (
                f"指令：{instruction}\n文本：{content}\n"
                "只输出JSON，以{开头，以}结尾。"
            )
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user},
            ]
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tok(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_new,
                    do_sample=False,
                    temperature=0.0,
                    top_p=1.0,
                    num_beams=1,
                    pad_token_id=tok.pad_token_id,
                    eos_token_id=tok.eos_token_id,
                )
            text = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            try:
                parsed = json.loads(text)
            except:
                parsed = _extract_first_json_object(text)
            return {"text": text, "parsed": _normalize_output(parsed)}

        def do_GET(self):
            if self.path == "/health":
                self._json(200, {"ok": True})
                return
            self._json(404, {"error": "not_found"})

        def do_POST(self):
            if self.path != "/generate":
                self._json(404, {"error": "not_found"})
                return
            st = time.perf_counter()
            try:
                l = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(l)
                payload = json.loads(raw.decode("utf-8"))
            except:
                self._json(400, {"error": "invalid_json"})
                return
            inst = str(payload.get("instruction", "")).strip()
            cont = str(payload.get("content", "")).strip()
            if not inst or not cont:
                self._json(400, {"error": "need instruction and content"})
                return
            req_max_new = int(payload.get("max_new_tokens", args.max_new_tokens))
            # Hard cap to avoid runaway generation latency.
            max_new = min(req_max_new, args.max_new_tokens)
            max_retries = 2
            res = None
            retried = False
            for i in range(max_retries):
                cur_max = max_new if i == 0 else min(args.retry_max_new_tokens, max_new + 128)
                temp = self._gen(inst, cont, cur_max, force_clean=(i>0))
                if temp["parsed"] is not None:
                    res = temp
                    retried = (i>0)
                    break
            if not res:
                res = {"text": "", "parsed": None}
            elapsed = int((time.perf_counter()-st)*1000)
            self._json(200, {
                "ok": True,
                "elapsed_ms": elapsed,
                "retried": retried,
                "output_text": res["text"],
                "output_json": res["parsed"],
            })

        def log_message(self, *args):
            pass

    return Handler

def main():
    args = build_parser()
    state = create_app_state(args)
    handler = make_handler(state)
    s = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"[serve] http://{args.host}:{args.port} ✅")
    s.serve_forever()

if __name__ == "__main__":
    main()