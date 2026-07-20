#!/usr/bin/env bash
set -euo pipefail

ROOT=<PROJECT_ROOT>
PYTHON=<ENV_ROOT>/bin/python
MANIFEST=smoke_tests/protocol_v3/manifests/medclipseg_covid19_groupval_sensitivity.csv
LOCK=smoke_tests/protocol_v3/protocol_lock_covid_groupval_sensitivity.yaml
AUDIT=paper/results/protocol_v3_covid_groupval_sensitivity/manifest_audit.json
LOG_ROOT=logs/protocol_v3_covid_groupval_sensitivity

cd "$ROOT"
mkdir -p "$LOG_ROOT" "$(dirname "$AUDIT")"
exec 9>"$LOG_ROOT/launcher.lock"
if ! flock -n 9; then
  echo "Another COVID grouped-validation sensitivity launcher is active" >&2
  exit 2
fi

"$PYTHON" smoke_tests/build_protocol_v3_covid_groupval_sensitivity.py \
  --input smoke_tests/protocol_v3/manifests/medclipseg_covid19_full.csv \
  --output "$MANIFEST" \
  --audit-json "$AUDIT" \
  --split-seed 123 \
  --val-fraction 0.2 \
  > "$LOG_ROOT/manifest_build.log" 2>&1

"$PYTHON" smoke_tests/run_protocol_v3_covid_groupval_sensitivity.py \
  --protocol-lock "$LOCK" \
  --stage confirmatory \
  --datasets medclipseg_covid19 \
  --seeds 123 456 789 \
  --gpus 2 3 \
  --jobs-per-gpu 2 \
  --epochs 100 \
  --batch-size 8 \
  --workers 2 \
  > "$LOG_ROOT/r11_r11nr.log" 2>&1

run_baseline() {
  local seed=$1
  local gpu=$2
  MANIFEST_OVERRIDE="$MANIFEST" \
  LOCK_OVERRIDE="$LOCK" \
  RESULT_ROOT_OVERRIDE="$ROOT/$LOG_ROOT/image_baselines" \
  bash smoke_tests/run_protocol_v3_image_baseline_task.sh \
    unetplusplus medclipseg_covid19 "$seed" "$gpu" \
    > "$LOG_ROOT/unetplusplus_seed${seed}.log" 2>&1
}

run_baseline 123 2 &
pid_a=$!
run_baseline 456 3 &
pid_b=$!
wait "$pid_a" "$pid_b"
run_baseline 789 2

date -Is > "$LOG_ROOT/complete.status"
echo "PASS COVID grouped-validation sensitivity"
