#!/usr/bin/env bash
set -euo pipefail

PROJECT=~/Qwen3.5-4B-Distillation-github-smoke
VENV=~/Qwen3.5-4B-Distillation/.venv
PORT=8000
MODEL_DIR="$PROJECT/training_runs/best_B_full_20260425_184108_merged"
LOG="$PROJECT/qwen35_full_http_8000.log"
PID_FILE="$PROJECT/qwen35_full_http_8000.pid"

cd "$PROJECT"
source "$VENV/bin/activate"

if [ ! -f "$MODEL_DIR/config.json" ]; then
  echo "[error] full model missing: $MODEL_DIR/config.json"
  exit 1
fi

if lsof -iTCP:"$PORT" -sTCP:LISTEN -t >/dev/null 2>&1; then
  echo "[error] port $PORT already in use"
  exit 1
fi

# stop previous same service if pid file exists
if [ -f "$PID_FILE" ]; then
  OLD_PID=$(cat "$PID_FILE" || true)
  if [ -n "${OLD_PID:-}" ] && ps -p "$OLD_PID" >/dev/null 2>&1; then
    kill "$OLD_PID" || true
    sleep 1
  fi
fi

nohup python3 -u serve_qwen35_full_http.py \
  --host 0.0.0.0 \
  --port "$PORT" \
  --model_path "$MODEL_DIR" \
  --max_new_tokens 256 \
  > "$LOG" 2>&1 &

NEW_PID=$!
echo "$NEW_PID" > "$PID_FILE"
echo "PID=$NEW_PID"
echo "LOG=$LOG"
echo "MODEL=$MODEL_DIR"
