#!/usr/bin/env bash
set -euo pipefail

ROOT=<PROJECT_ROOT>
PYTHON=<ENV_ROOT>/bin/python
LOG_ROOT=$ROOT/logs/protocol_v3_covid_groupval_sensitivity
RUN_ROOT=$ROOT/logs/protocol_v3/c0e828a220a5ce116fbe40fac2f113df4b198e1c3c7ca57171bee4b7702adfe3
STATUS=$LOG_ROOT/watcher_status.txt

cd "$ROOT"
mkdir -p "$LOG_ROOT"
exec 9>"$LOG_ROOT/watcher.lock"
if ! flock -n 9; then
  echo "Watcher already active" >&2
  exit 2
fi

write_status() {
  local completed multimodal_running baseline_running launcher_running
  completed=$(find "$RUN_ROOT" -name complete.status -type f 2>/dev/null | wc -l)
  multimodal_running=$(pgrep -fc "run_protocol_v3_covid_groupval_sensitivity.py|V3_R11.*COVID_GROUPVAL" || true)
  baseline_running=$(pgrep -fc "protocol_v3_covid_groupval_sensitivity/image_baselines" || true)
  launcher_running=$(pgrep -fc "bash smoke_tests/launch_protocol_v3_covid_groupval_sensitivity.sh" || true)
  {
    echo "checked_at=$(date -Is)"
    echo "multimodal_complete=$completed/6"
    echo "multimodal_process_matches=$multimodal_running"
    echo "baseline_process_matches=$baseline_running"
    echo "launcher_process_matches=$launcher_running"
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
  } > "$STATUS.tmp"
  mv "$STATUS.tmp" "$STATUS"
}

while true; do
  write_status
  if [[ -s "$LOG_ROOT/complete.status" ]]; then
    "$PYTHON" paper/analysis/summarize_covid_groupval_sensitivity.py \
      --project-root "$ROOT" \
      > "$LOG_ROOT/summarize.log" 2>&1
    date -Is > "$LOG_ROOT/summary_complete.status"
    write_status
    exit 0
  fi
  if ! pgrep -f "bash smoke_tests/launch_protocol_v3_covid_groupval_sensitivity.sh" >/dev/null; then
    echo "failed_at=$(date -Is) reason=launcher_exited_without_complete_status" \
      > "$LOG_ROOT/watcher_failed.status"
    exit 1
  fi
  sleep 900
done
