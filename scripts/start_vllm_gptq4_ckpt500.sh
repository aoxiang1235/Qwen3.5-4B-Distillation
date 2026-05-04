#!/usr/bin/env bash
# 云机/本机：GPTQ 4bit 合并目录 → vLLM OpenAI 服务（默认端口 8002，避免与 FP16 的 8001 冲突）
#
# 产物目录由 quantize_gptq_4bit_qwen35_merged.py 生成；需与合并 FP16 同套路径约定。
# 若缺 preprocessor 等：先保证魔搭基座存在，本脚本会尝试 sync（与 start_vllm_merged_ckpt500.sh 一致）。
#
# 单卡上通常只跑一个 vLLM；若需独占 GPU，可先停 FP16： pkill -f '[v]llm.entrypoints.openai.api_server'
#
set -euo pipefail
REPO_ROOT="${VLLM_REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_ROOT"
MS_BASE="${MODELSCOPE_QWEN35_BASE:-$REPO_ROOT/.modelscope/Qwen/Qwen3___5-4B}"
GPTQ4_DIR="${GPTQ4_CKPT500_DIR:-training_runs/lora_qwen35_4b_merged_gptq4_ckpt500}"
VLLM_PORT="${VLLM_GPTQ4_PORT:-8002}"
if [[ "$GPTQ4_DIR" = /* ]]; then
  GPTQ4_ABS="$GPTQ4_DIR"
else
  GPTQ4_ABS="${REPO_ROOT}/${GPTQ4_DIR}"
fi

if [[ -d "$MS_BASE" ]]; then
  python3 scripts/prepare_merged_for_vllm.py --base_dir "$MS_BASE" --merged_dir "$GPTQ4_ABS" || true
else
  echo "提示: 未找到魔搭基座目录 $MS_BASE ，跳过侧文件同步；若 vLLM 报错缺 preprocessor，请先下载 Qwen/Qwen3.5-4B 到 .modelscope 并设置 MODELSCOPE_QWEN35_BASE" >&2
fi

pkill -f "[v]llm.entrypoints.openai.api_server" 2>/dev/null || true
sleep 2
: > /tmp/vllm_gptq4_ckpt500.log
# 与 quantize_gptq_4bit_qwen35_merged.py 文档一致；enable_thinking 关闭以对齐训练 JSON 行为
nohup python3 -m vllm.entrypoints.openai.api_server \
  --model "$GPTQ4_ABS" \
  --quantization gptq \
  --served-model-name beauty_lora_merged_gptq4_ckpt500 \
  --host 0.0.0.0 \
  --port "$VLLM_PORT" \
  --max-model-len 4096 \
  --dtype float16 \
  --trust-remote-code \
  --gpu-memory-utilization 0.88 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  </dev/null >> /tmp/vllm_gptq4_ckpt500.log 2>&1 &
echo $! > /tmp/vllm_gptq4_ckpt500.pid
echo "vLLM GPTQ4 PID=$(cat /tmp/vllm_gptq4_ckpt500.pid) port=$VLLM_PORT log=/tmp/vllm_gptq4_ckpt500.log"
