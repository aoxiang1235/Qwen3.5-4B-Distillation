#!/usr/bin/env python3
"""
Qwen3.5 基座 HTTP 推理：权重从魔搭 ModelScope 拉取（或本机已下载目录），尽量保持 from_pretrained 默认参数。

远程模型 ID 使用 modelscope.snapshot_download，不走 Hugging Face Hub。
加载仍使用 transformers 的 Auto*（对已下载到本地的目录做 from_pretrained，与 new_mian.py 一致）。

与 serve_qwen35_full_http.py 的区别：
- AutoModelForCausalLM.from_pretrained / AutoTokenizer.from_pretrained 仅传路径，
  不指定 torch_dtype、device_map、量化等；
- 不在代码里改写 model.config / generation_config；
- 不使用固定 system prompt、/no_think 后缀、JSON 抽取等任务相关逻辑；
- 不读取项目内任何数据文件（无 jsonl / 数据集路径）。

请求体（POST /generate）：
- 推荐：{"messages": [{"role":"user","content":"..."}], "max_new_tokens": 512}
- 或兼容：{"instruction": "...", "content": "...", "max_new_tokens": 512}
  会拼成单条 user 消息，不自动加 system。

返回：{"ok": true, "elapsed_ms": ..., "text": "<模型续写原文>"}
"""
from __future__ import annotations

import argparse
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def resolve_pretrained_dir(model_id: str, modelscope_cache: str) -> str:
    """
    与 new_mian.resolve_pretrained_path 的 ModelScope 分支一致：
    - 若 model_id 为已含 config.json 的本地目录，直接返回绝对路径；
    - 否则用 ModelScope snapshot_download 下载/复用缓存并返回本地目录。
    """
    name = (model_id or "").strip()
    if not name:
        raise ValueError("model_id 不能为空")
    if os.path.isfile(os.path.join(name, "config.json")):
        return os.path.abspath(name)
    try:
        from modelscope import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "使用魔搭下载需安装: pip install modelscope"
        ) from exc
    cache = (modelscope_cache or "").strip() or os.environ.get(
        "MODELSCOPE_CACHE", os.path.join(os.getcwd(), ".modelscope")
    )
    os.makedirs(cache, exist_ok=True)
    print(f"  - ModelScope: {name}（cache_dir={cache}）")
    return snapshot_download(name, cache_dir=cache)


def make_handler(tokenizer: Any, model: Any) -> type:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args_: object) -> None:  # noqa: A003
            return

        def _resp(self, status: int, payload: Dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self._resp(
                    200,
                    {"ok": True, "serve": "qwen35_vanilla_http", "weights": "modelscope"},
                )
                return
            self._resp(404, {"error": "not_found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/generate":
                self._resp(404, {"error": "not_found"})
                return
            t0 = time.perf_counter()
            try:
                length = int(self.headers.get("Content-Length", "0"))
                req = json.loads(self.rfile.read(length).decode("utf-8"))
            except Exception as exc:
                self._resp(400, {"error": f"invalid_json: {exc}"})
                return

            max_new_tokens = int(req.get("max_new_tokens", 256))
            max_new_tokens = max(1, min(max_new_tokens, 8192))

            messages: List[Dict[str, Any]]
            if "messages" in req and isinstance(req["messages"], list):
                messages = req["messages"]
            else:
                inst = str(req.get("instruction", "")).strip()
                cont = str(req.get("content", "")).strip()
                if not inst and not cont:
                    self._resp(
                        400,
                        {"error": "provide messages[] or non-empty instruction/content"},
                    )
                    return
                parts = [p for p in (inst, cont) if p]
                messages = [{"role": "user", "content": "\n\n".join(parts)}]

            try:
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception as exc:
                self._resp(400, {"error": f"chat_template_failed: {exc}"})
                return

            dev = _model_device(model)
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(dev) for k, v in inputs.items()}

            try:
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=max_new_tokens)
            except Exception as exc:
                self._resp(500, {"ok": False, "error": f"generate_failed: {exc}"})
                return

            new_tokens = out[0, inputs["input_ids"].shape[1] :]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            self._resp(
                200,
                {"ok": True, "elapsed_ms": elapsed_ms, "text": text},
            )

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Qwen3.5 vanilla HTTP：魔搭 ModelScope 下载/缓存 + from_pretrained 默认参数。",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="Qwen/Qwen3.5-4B",
        help="魔搭模型 ID（如 Qwen/Qwen3.5-4B）或本机已含 config.json 的目录",
    )
    parser.add_argument(
        "--modelscope_cache",
        type=str,
        default="",
        help="ModelScope 缓存根目录；默认环境变量 MODELSCOPE_CACHE 或 <cwd>/.modelscope",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    pretrained = resolve_pretrained_dir(args.model_path, args.modelscope_cache)
    print(f"[vanilla] Loading model (default kwargs): {pretrained}")
    tokenizer = AutoTokenizer.from_pretrained(pretrained)
    model = AutoModelForCausalLM.from_pretrained(pretrained)
    model.eval()

    handler = make_handler(tokenizer, model)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"[vanilla] http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
