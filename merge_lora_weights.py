#!/usr/bin/env python3
"""
将 LoRA 适配器合并进基座模型（PEFT merge_and_unload），写出完整权重目录。

典型用途：
- `new_mian.py` / 其他脚本训练后目录里只有 adapter + 配置，需要单目录全量权重给
  `transformers` 直接 `from_pretrained`，或给部分推理栈做「全量 FP16 部署」。

注意：
- 合并应在「非 4bit 量化」基座上完成；若训练时用了 4bit，请用同一基座的 FP16/BF16
  重新加载再挂 adapter 后合并（本脚本默认 float16）。
- 合并后体积约等于「基座 + 少量元数据」，请预留磁盘空间。
- 国内常用魔搭 ModelScope：`Qwen/Qwen3.5-4B`（本机缓存多为 `.modelscope/Qwen/Qwen3___5-4B`）。
  合并写出的是扁平 `qwen3_5_text` 的 `config.json`，vLLM 对 Qwen3.5 需完整多模态配置与
  `preprocessor_config.json`。若 `--base_model` 为本机目录，默认会从该目录同步这些侧文件。
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# 从魔搭 snapshot / 本机基座拷入，供 vLLM（Qwen3.5 VL 注册路径）加载；权重仍以合并结果为准。
_VLLM_SIDE_CAR_FILES = (
    "config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "video_preprocessor_config.json",
    "generation_config.json",
    "chat_template.jinja",
)


def sync_modelscope_assets_for_vllm(base_dir: str, output_dir: str) -> int:
    """
    从本机基座目录（与训练一致的 ModelScope 解压路径）复制 vLLM 需要的配置与图像预处理文件，
    覆盖 merge 后扁平 text config，并补齐 preprocessor 等。
    返回成功复制的文件数。
    """
    if not base_dir or not os.path.isdir(base_dir):
        return 0
    n = 0
    for name in _VLLM_SIDE_CAR_FILES:
        src = os.path.join(base_dir, name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(output_dir, name))
            print(f"  已同步 (vLLM/多模态侧文件): {name}")
            n += 1
    if n:
        print(f"  侧文件来源: {os.path.abspath(base_dir)}（共 {n} 个）")
    return n


def main() -> None:
    p = argparse.ArgumentParser(description="LoRA adapter -> merged full weights")
    p.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="基座：魔搭 ModelScope id（如 Qwen/Qwen3.5-4B）、HF id，或与训练一致的本机 snapshot 目录",
    )
    p.add_argument(
        "--sync_assets_from",
        type=str,
        default="",
        help="从该本机目录复制 config/preprocessor 等供 vLLM；留空且 --base_model 为目录时自动使用 base_model",
    )
    p.add_argument(
        "--no_sync_base_assets",
        action="store_true",
        help="不复制基座侧文件（默认：本机基座目录时会同步以兼容 vLLM）",
    )
    p.add_argument(
        "--adapter_path",
        type=str,
        required=True,
        help="含 adapter 权重的目录（一般为训练 output_dir / checkpoint-*）",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="合并后保存目录（将新建/覆盖同名文件）",
    )
    p.add_argument(
        "--torch_dtype",
        type=str,
        choices=["float16", "bfloat16", "float32"],
        default="float16",
        help="加载与写出时的 torch dtype（合并勿用 4bit）",
    )
    p.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help="传给 from_pretrained，默认 auto（多卡可 auto）",
    )
    args = p.parse_args()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.torch_dtype]

    if not os.path.isdir(args.adapter_path):
        print(f"error: adapter_path 不是目录: {args.adapter_path}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"加载基座: {args.base_model} ({args.torch_dtype})")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch_dtype,
        device_map=args.device_map,
        trust_remote_code=True,
    )
    print(f"加载适配器: {args.adapter_path}")
    model = PeftModel.from_pretrained(base, args.adapter_path, is_trainable=False)
    print("merge_and_unload() …")
    merged = model.merge_and_unload()
    print(f"保存到: {args.output_dir}")
    merged.save_pretrained(args.output_dir, safe_serialization=True)

    tok_src = args.adapter_path
    if not os.path.isfile(os.path.join(tok_src, "tokenizer_config.json")):
        tok_src = args.base_model
    print(f"保存 tokenizer（来源: {tok_src}）")
    tokenizer = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)
    tokenizer.save_pretrained(args.output_dir)

    sync_from = (args.sync_assets_from or "").strip()
    if not sync_from and os.path.isdir(args.base_model):
        sync_from = os.path.abspath(args.base_model)
    if sync_from and not args.no_sync_base_assets:
        k = sync_modelscope_assets_for_vllm(sync_from, args.output_dir)
        if not k:
            print(
                "  提示: 未从基座复制任何侧文件（路径无效或缺少 preprocessor 等）。"
                "vLLM 需魔搭本机目录中的 config.json、preprocessor_config.json；"
                "可安装 modelscope 后下载 Qwen/Qwen3.5-4B 再指定 --sync_assets_from。",
                file=sys.stderr,
            )
    elif not args.no_sync_base_assets and not os.path.isdir(args.base_model):
        print(
            "  提示: --base_model 为远端 id 时未自动同步侧文件；部署 vLLM 前请将魔搭 snapshot 目录"
            " 传给 --sync_assets_from 再运行一次本脚本，或用手工拷贝 config/preprocessor 至合并目录。",
            file=sys.stderr,
        )

    print("完成。")


if __name__ == "__main__":
    main()
