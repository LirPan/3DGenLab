#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR" || exit 2

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/full_eval_demo_${TS}.log"

echo "[full-eval] start: $(date -Is)" | tee -a "$LOG_FILE"
echo "[full-eval] root: $ROOT_DIR" | tee -a "$LOG_FILE"

run_cmd() {
  local name="$1"
  shift
  echo "[full-eval] >>> $name" | tee -a "$LOG_FILE"
  "$@" 2>&1 | tee -a "$LOG_FILE"
  local rc=${PIPESTATUS[0]}
  echo "[full-eval] <<< $name exit_code=$rc" | tee -a "$LOG_FILE"
  return "$rc"
}

RC_MAIN=0
RC_TEXT_HY=0
RC_TEXT_TR=0

run_cmd "main_image_set+challenge_set all image-capable models" \
  python3 scripts/run_dataset_eval.py \
    --config configs/triposr_gpu.yaml \
    --group main_image_set \
    --models triposr,instantmesh,hunyuan3d,trellis \
    --no-dry-run \
    --benchmark || RC_MAIN=$?

run_cmd "challenge_set all image-capable models" \
  python3 scripts/run_dataset_eval.py \
    --config configs/triposr_gpu.yaml \
    --group challenge_set \
    --models triposr,instantmesh,hunyuan3d,trellis \
    --no-dry-run \
    --benchmark || RC_MAIN=$?

run_cmd "text_extension_set hunyuan3d" \
  python3 scripts/run_dataset_eval.py \
    --config configs/hunyuan3d_text_gpu.yaml \
    --group text_extension_set \
    --models hunyuan3d \
    --no-dry-run \
    --benchmark || RC_TEXT_HY=$?

run_cmd "text_extension_set trellis" \
  python3 scripts/run_dataset_eval.py \
    --config configs/trellis_text_gpu.yaml \
    --group text_extension_set \
    --models trellis \
    --no-dry-run \
    --benchmark || RC_TEXT_TR=$?

echo "[full-eval] done: $(date -Is)" | tee -a "$LOG_FILE"
echo "[full-eval] summary rc_main=$RC_MAIN rc_text_hunyuan=$RC_TEXT_HY rc_text_trellis=$RC_TEXT_TR" | tee -a "$LOG_FILE"

if [[ "$RC_MAIN" -ne 0 || "$RC_TEXT_HY" -ne 0 || "$RC_TEXT_TR" -ne 0 ]]; then
  exit 1
fi

exit 0
