#!/usr/bin/env bash
set -euo pipefail

PROJECT=~/Qwen3.5-4B-Distillation-github-smoke
VENV=~/Qwen3.5-4B-Distillation/.venv
PORT=8000
BASE_MODEL="Qwen/Qwen3.5-4B"
ADAPTER_DIR="$PROJECT/training_runs/best_B_full_20260425_184108_out"
LOG="$PROJECT/strict_json_8000.log"
PID_FILE="$PROJECT/strict_json_8000.pid"

cd "$PROJECT"
source "$VENV/bin/activate"

if [ ! -f "$ADAPTER_DIR/adapter_config.json" ]; then
  echo "[error] lora adapter missing: $ADAPTER_DIR/adapter_config.json"
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

lsof -iTCP:"$PORT" -sTCP:LISTEN -t | xargs -r kill || true
sleep 1

nohup python3 -u serve_qwen35_full_http_4bit.py \
  --host 0.0.0.0 \
  --port "$PORT" \
  --base_model "$BASE_MODEL" \
  --adapter_path "$ADAPTER_DIR" \
  --max_new_tokens 256 \
  --retry_max_new_tokens 384 \
  > "$LOG" 2>&1 &

NEW_PID=$!
echo "$NEW_PID" > "$PID_FILE"
sleep 2
if ! kill -0 "$NEW_PID" >/dev/null 2>&1; then
  echo "[error] service exited early, check log: $LOG"
  exit 1
fi

for i in 1 2 3 4 5 6; do
  if curl -sS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

echo "PID=$NEW_PID"
echo "LOG=$LOG"
echo "BASE_MODEL=$BASE_MODEL"
echo "ADAPTER=$ADAPTER_DIR"
