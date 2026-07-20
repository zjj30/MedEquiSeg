#!/usr/bin/env bash
set -euo pipefail

DATASET=$1
SEED=$2
GPU=$3

ROOT=<PROJECT_ROOT>
PYTHON=<ENV_ROOT>/bin/python
SOURCE=$ROOT/logs/protocol_v3_pure_image/rolling_${DATASET}_common_light_seed${SEED}
CHECKPOINT=$SOURCE/RollingUNet_${DATASET}_aug_common_light_seed${SEED}_best.pt
OUT=$ROOT/logs/protocol_v3_rolling_canonical/$DATASET/seed$SEED

case "$DATASET" in
  busi_hf)
    MANIFEST=<PRIVATE_MANIFEST_NOT_RELEASED>
    LOCK=<PRIVATE_LOCK_NOT_RELEASED>
    ;;
  medclipseg_busi|medclipseg_clinicdb|medclipseg_busbra|medclipseg_brisc|medclipseg_covid19)
    MANIFEST=smoke_tests/protocol_v3/manifests/${DATASET}_full.csv
    LOCK=smoke_tests/protocol_v3/protocol_lock.yaml
    ;;
  *)
    echo "Unknown dataset: $DATASET" >&2
    exit 2
    ;;
esac

cd "$ROOT"
mkdir -p "$OUT/controls/true"
exec 9>"$OUT/task.lock"
if ! flock -n 9; then
  echo "SKIP locked dataset=$DATASET seed=$SEED"
  exit 0
fi
if [[ -s "$OUT/controls/true/summary.csv" && -f "$OUT/complete.status" ]]; then
  echo "SKIP complete dataset=$DATASET seed=$SEED"
  exit 0
fi

test -s "$CHECKPOINT"
rm -f "$OUT/failed.status"
{
  echo "status=running"
  echo "model=rollingunet"
  echo "dataset=$DATASET"
  echo "seed=$SEED"
  echo "gpu=$GPU"
  echo "checkpoint=$CHECKPOINT"
  echo "manifest=$MANIFEST"
  echo "protocol_lock=$LOCK"
  echo "started_at=$(date -Is)"
} > "$OUT/run_meta.txt"

fail_task() {
  code=$?
  echo "failed_at=$(date -Is) exit_code=$code" > "$OUT/failed.status"
  exit "$code"
}
trap fail_task ERR

CUDA_VISIBLE_DEVICES=$GPU "$PYTHON" smoke_tests/predict_rolling_protocol_v3.py \
  --checkpoint "$CHECKPOINT" \
  --manifest "$MANIFEST" \
  --dataset "$DATASET" \
  --protocol-lock "$LOCK" \
  --output-dir "$OUT/controls/true" \
  --batch-size 8 \
  > "$OUT/predict.log" 2>&1

"$PYTHON" smoke_tests/evaluate_predictions_v3.py \
  --prediction-index "$OUT/controls/true/prediction_index.csv" \
  --protocol-lock "$LOCK" \
  --per-case-csv "$OUT/controls/true/per_case.csv" \
  --summary-csv "$OUT/controls/true/summary.csv" \
  > "$OUT/evaluate.log" 2>&1

{
  echo "status=complete"
  echo "completed_at=$(date -Is)"
  echo "checkpoint_sha256=$(sha256sum "$CHECKPOINT" | awk '{print $1}')"
} >> "$OUT/run_meta.txt"
echo "complete_at=$(date -Is)" > "$OUT/complete.status"
rm -f "$OUT/failed.status"
trap - ERR
echo "PASS dataset=$DATASET seed=$SEED gpu=$GPU"
