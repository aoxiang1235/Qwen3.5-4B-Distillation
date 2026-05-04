#!/usr/bin/env bash
# 云机：合并后的 FP16 目录 → vLLM OpenAI 服务（端口 8001）
#
# --model 必须是「本机合并目录」，不要用 Hugging Face 上的 Qwen3.5-4B id（国内不一定齐、且与训练魔搭基座可能不一致）。
# 基座请用魔搭 ModelScope：Qwen/Qwen3.5-4B，缓存目录常为：
#   <仓库>/.modelscope/Qwen/Qwen3___5-4B
# 首次无缓存：pip install modelscope && python3 -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen3.5-4B')"
#
# 合并后若只含权重：须从魔搭 snapshot 同步 config / preprocessor（否则 vLLM 报缺 preprocessor_config）。
# merge_lora_weights 在 --base_model 为本机目录时会自动同步；已合并过则可：
#   python3 scripts/prepare_merged_for_vllm.py --base_dir .modelscope/Qwen/Qwen3___5-4B --merged_dir <合并目录>
#
# 合并产物通常只有 model.language_model.*，vLLM 按 VL 配置加载还需要基座里的 model.visual.*（否则会报
# visual 权重未初始化）。首次部署或重新 merge 后执行一次：
#   cp training_runs/.../model.safetensors training_runs/.../model.safetensors.lm_only_bak
#   python3 scripts/stitch_visual_weights_for_vllm.py \\
#     --merged_safetensors training_runs/.../model.safetensors.lm_only_bak \\
#     --base_model_dir .modelscope/Qwen/Qwen3___5-4B \\
#     --output training_runs/.../model.safetensors
#
set -euo pipefail
REPO_ROOT="${VLLM_REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_ROOT"
# 魔搭本机基座（可用环境变量覆盖，便于本机与云机路径不同）
MS_BASE="${MODELSCOPE_QWEN35_BASE:-$REPO_ROOT/.modelscope/Qwen/Qwen3___5-4B}"
MERGED_DIR="${MERGED_FP16_DIR:-training_runs/lora_qwen35_4b_merged_fp16_ckpt500}"
# vLLM 子进程工作目录未必是仓库根，相对路径会找不到 preprocessor；统一用绝对路径
if [[ "$MERGED_DIR" = /* ]]; then
  MERGED_ABS="$MERGED_DIR"
else
  MERGED_ABS="${REPO_ROOT}/${MERGED_DIR}"
fi

if [[ -d "$MS_BASE" ]]; then
  python3 scripts/prepare_merged_for_vllm.py --base_dir "$MS_BASE" --merged_dir "$MERGED_ABS" || true
else
  echo "提示: 未找到魔搭基座目录 $MS_BASE ，跳过侧文件同步；若 vLLM 报错缺 preprocessor，请先下载 Qwen/Qwen3.5-4B 到 .modelscope 并设置 MODELSCOPE_QWEN35_BASE" >&2
fi

# 注意：pkill -f 会匹配「整条命令行」；须用 [v] 避免误杀含同串的 ssh 子 shell
pkill -f "[v]llm.entrypoints.openai.api_server" 2>/dev/null || true
sleep 2
: > /tmp/vllm_merged_ckpt500.log
# Qwen3：默认关闭「思考链」，否则易占满 max_tokens、输出冗长且不像训练时的 JSON（见 serve_qwen35_full_http.py）
nohup python3 -m vllm.entrypoints.openai.api_server \
  --model "$MERGED_ABS" \
  --served-model-name beauty_lora_merged_ckpt500 \
  --host 0.0.0.0 \
  --port 8001 \
  --max-model-len 4096 \
  --dtype float16 \
  --trust-remote-code \
  --gpu-memory-utilization 0.88 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  </dev/null >> /tmp/vllm_merged_ckpt500.log 2>&1 &
echo $! > /tmp/vllm_merged_ckpt500.pid
echo "vLLM PID=$(cat /tmp/vllm_merged_ckpt500.pid) log=/tmp/vllm_merged_ckpt500.log"
