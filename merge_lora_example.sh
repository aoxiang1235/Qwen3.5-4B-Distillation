#!/usr/bin/env bash
# 合并 LoRA：改下面三行后执行  bash merge_lora_example.sh
set -euo pipefail

# 与训练 new_mian.py 时 --model_name 一致
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-3B-Instruct}"

# 含 adapter_config.json 的目录（常见为 checkpoint-*；或 Trainer 直接 save 的 output_dir）
ADAPTER_DIR="${ADAPTER_DIR:-./training_runs/lora_qwen25_3b_train/checkpoint-500}"

# 合并产物目录（勿与 ADAPTER 相同）
OUT_DIR="${OUT_DIR:-./merged_models/beauty_brand_merged_fp16}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ ! -f "$ADAPTER_DIR/adapter_config.json" ]]; then
  echo "[error] 找不到 adapter: $ADAPTER_DIR/adapter_config.json"
  echo "请把 ADAPTER_DIR 改成实际 checkpoint 路径，例如:"
  echo "  ls training_runs/lora_qwen25_3b_train/"
  exit 1
fi

mkdir -p "$(dirname "$OUT_DIR")"
python3 merge_lora_weights.py \
  --base_model "$BASE_MODEL" \
  --adapter "$ADAPTER_DIR" \
  --out "$OUT_DIR" \
  --dtype float16

echo "[ok] 合并完成: $OUT_DIR"
echo "vLLM 示例: vllm serve $OUT_DIR --host 0.0.0.0 --port 8000 --trust-remote-code --served-model-name beauty_merged"
