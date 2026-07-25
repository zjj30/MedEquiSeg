#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
PYTHON=${MEDEQUISEG_PYTHON:-python3}
SESSION=medequiseg_nr_matched_busi_20260725
LOG_ROOT="$ROOT/logs/protocol_v3_matched_no_rewrite_20260725"
LOG_FILE="$LOG_ROOT/busi_3seed.log"

cd "$ROOT"
mkdir -p "$LOG_ROOT"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "SKIP existing session=$SESSION"
  exit 0
fi

if pgrep -af '[r]un_medequiseg_no_rewrite_matched.py' >/dev/null; then
  echo "A matched MedEquiSeg rewrite control is already active." >&2
  exit 1
fi

tmux new-session -d -s "$SESSION" \
  "cd $ROOT && \
   $PYTHON smoke_tests/run_medequiseg_no_rewrite_matched.py \
     --protocol-lock smoke_tests/protocol_v3/protocol_lock.yaml \
     --stage confirmatory \
     --run-ids V3_CTRL_REWRITE_MATCHED_20260725 V3_CTRL_NO_REWRITE_MATCHED_20260725 \
     --datasets medclipseg_busi \
     --seeds 123 456 789 \
     --gpus 1 2 \
     --jobs-per-gpu 1 \
     --batch-size 8 \
     --workers 2 \
     > $LOG_FILE 2>&1"

echo "START session=$SESSION datasets=medclipseg_busi seeds=123,456,789 gpus=1,2"
echo "matched_arms=rewrite,no_rewrite"
echo "log=$LOG_FILE"
