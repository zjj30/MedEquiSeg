#!/usr/bin/env bash
set -euo pipefail

MODEL=$1
DATASET=$2
SEED=$3
GPU=$4

ROOT=<PROJECT_ROOT>
RMTFD=<ENV_ROOT>/bin/python
SOTA_BIN=<ENV_ROOT>/sota_baselines/bin
UMAMBA_BIN=<ENV_ROOT>/umamba_py310/bin
NNROOT=$ROOT/outputs/protocol_v3_nnunet
UMAMBA_ROOT=$ROOT/outputs/protocol_v3_umamba
NNUNET_TRAINER_SRC=$ROOT/smoke_tests/nnunet_trainers/nnUNetTrainerProtocolV3Seeds.py
NNUNET_TRAINER_DST=$SOTA_BIN/../lib/python3.11/site-packages/nnunetv2/training/nnUNetTrainer/nnUNetTrainerProtocolV3Seeds.py
UMAMBA_TRAINER_SRC=$ROOT/smoke_tests/nnunet_trainers/nnUNetTrainerUMambaBotProtocolV3Seeds.py
UMAMBA_TRAINER_DST=$ROOT/repos/U-Mamba/umamba/nnunetv2/training/nnUNetTrainer/nnUNetTrainerUMambaBotProtocolV3Seeds.py
UMAMBA_PATHS_SRC=$ROOT/smoke_tests/nnunet_trainers/umamba_paths_protocol_v3.py
UMAMBA_PATHS_DST=$ROOT/repos/U-Mamba/umamba/nnunetv2/paths.py
UMAMBA_PLANS_SRC=$ROOT/smoke_tests/nnunet_trainers/umamba_plans_handler_protocol_v3.py
UMAMBA_PLANS_DST=$ROOT/repos/U-Mamba/umamba/nnunetv2/utilities/plans_handling/plans_handler.py

case "$DATASET" in
  medclipseg_busi)
    DATASET_ID=951
    DATASET_NAME=Dataset951_MedclipsegBusi
    MANIFEST=smoke_tests/protocol_v3/manifests/medclipseg_busi_full.csv
    LOCK=smoke_tests/protocol_v3/protocol_lock.yaml
    ;;
  medclipseg_clinicdb)
    DATASET_ID=952
    DATASET_NAME=Dataset952_MedclipsegClinicdb
    MANIFEST=smoke_tests/protocol_v3/manifests/medclipseg_clinicdb_full.csv
    LOCK=smoke_tests/protocol_v3/protocol_lock.yaml
    ;;
  medclipseg_busbra)
    DATASET_ID=953
    DATASET_NAME=Dataset953_MedclipsegBusbra
    MANIFEST=smoke_tests/protocol_v3/manifests/medclipseg_busbra_full.csv
    LOCK=smoke_tests/protocol_v3/protocol_lock.yaml
    ;;
  medclipseg_brisc)
    DATASET_ID=954
    DATASET_NAME=Dataset954_MedclipsegBrisc
    MANIFEST=smoke_tests/protocol_v3/manifests/medclipseg_brisc_full.csv
    LOCK=smoke_tests/protocol_v3/protocol_lock.yaml
    ;;
  medclipseg_covid19)
    DATASET_ID=955
    DATASET_NAME=Dataset955_MedclipsegCovid19
    MANIFEST=smoke_tests/protocol_v3/manifests/medclipseg_covid19_full.csv
    LOCK=smoke_tests/protocol_v3/protocol_lock.yaml
    ;;
  busi_hf)
    DATASET_ID=956
    DATASET_NAME=Dataset956_BusiHf
    MANIFEST=<PRIVATE_MANIFEST_NOT_RELEASED>
    LOCK=<PRIVATE_LOCK_NOT_RELEASED>
    ;;
  *)
    echo "Unknown dataset: $DATASET" >&2
    exit 2
    ;;
esac

# Optional overrides support isolated sensitivity analyses without changing the
# canonical Protocol V3 task paths or behavior.
MANIFEST=${MANIFEST_OVERRIDE:-$MANIFEST}
LOCK=${LOCK_OVERRIDE:-$LOCK}
RESULT_ROOT=${RESULT_ROOT_OVERRIDE:-$ROOT/logs/protocol_v3_image_baselines}
OUT=$RESULT_ROOT/$MODEL/$DATASET/seed$SEED

cd "$ROOT"
mkdir -p "$OUT/controls/true"

# Multiple GPU queues may discover the same missing cell. Hold a non-blocking
# per-cell lock so only one process can train or materialize that result.
exec 9>"$OUT/task.lock"
if ! flock -n 9; then
  echo "SKIP locked model=$MODEL dataset=$DATASET seed=$SEED"
  exit 0
fi

if [[ -s "$OUT/controls/true/summary.csv" && -f "$OUT/complete.status" ]]; then
  echo "SKIP complete model=$MODEL dataset=$DATASET seed=$SEED"
  exit 0
fi

rm -f "$OUT/failed.status"
{
  echo "status=running"
  echo "model=$MODEL"
  echo "dataset=$DATASET"
  echo "seed=$SEED"
  echo "split_seed=123"
  echo "gpu=$GPU"
  echo "epochs=100"
  echo "manifest=$MANIFEST"
  echo "manifest_sha256=$(sha256sum "$MANIFEST" | awk '{print $1}')"
  echo "protocol_lock=$LOCK"
  echo "protocol_lock_sha256=$(sha256sum "$LOCK" | awk '{print $1}')"
  echo "started_at=$(date -Is)"
} > "$OUT/run_meta.txt"

fail_task() {
  code=$?
  echo "failed_at=$(date -Is) exit_code=$code" > "$OUT/failed.status"
  exit "$code"
}
trap fail_task ERR

if [[ "$MODEL" == "unet" || "$MODEL" == "unetplusplus" || "$MODEL" == "ukan" ]]; then
  mkdir -p "$OUT/model"
  LEGACY_DIR=$ROOT/logs/protocol_v3_pure_image/ukan_${DATASET}_common_light_seed123
  if [[ "$MODEL" == "ukan" && "$SEED" == "123" && -s "$LEGACY_DIR/run_meta.txt" ]]; then
    CURRENT_MANIFEST_SHA=$(sha256sum "$MANIFEST" | awk '{print $1}')
    LEGACY_MANIFEST_SHA=$(awk -F= '$1 == "manifest_sha256" {print $2}' "$LEGACY_DIR/run_meta.txt")
    if [[ "$CURRENT_MANIFEST_SHA" != "$LEGACY_MANIFEST_SHA" ]]; then
      echo "Verified U-KAN seed123 manifest mismatch: $DATASET" >&2
      exit 3
    fi
    CHECKPOINT=$(find "$LEGACY_DIR" -maxdepth 1 -name '*_best.pt' -type f | head -1)
    echo "reused_verified_v3_seed123_checkpoint=$CHECKPOINT" >> "$OUT/run_meta.txt"
    echo "source_run_meta=$LEGACY_DIR/run_meta.txt" >> "$OUT/run_meta.txt"
    echo "Reusing verified Protocol V3 seed123 checkpoint: $CHECKPOINT" > "$OUT/train.log"
  else
    CUDA_VISIBLE_DEVICES=$GPU "$RMTFD" smoke_tests/run_ukan_image_aug_dataset.py \
      --architecture "$MODEL" \
      --dataset "$DATASET" \
      --manifest "$MANIFEST" \
      --epochs 100 \
      --batch-size 8 \
      --workers 2 \
      --image-size 224 \
      --resize-mode stretch \
      --seed "$SEED" \
      --split-seed 123 \
      --profile common_light \
      --augment \
      --out-dir "$OUT/model" \
      > "$OUT/train.log" 2>&1
    CHECKPOINT=$(find "$OUT/model" -maxdepth 1 -name '*_best.pt' -type f | head -1)
  fi
  test -n "$CHECKPOINT"
  CUDA_VISIBLE_DEVICES=$GPU "$RMTFD" smoke_tests/predict_monai_protocol_v3.py \
    --checkpoint "$CHECKPOINT" \
    --architecture "$MODEL" \
    --manifest "$MANIFEST" \
    --dataset "$DATASET" \
    --protocol-lock "$LOCK" \
    --output-dir "$OUT/controls/true" \
    --batch-size 8 \
    > "$OUT/predict.log" 2>&1
else
  # Trainer shims live in shared environments. Serialize their installation
  # across task slots, then release the lock before GPU work begins.
  exec 8>"$ROOT/logs/protocol_v3_image_baselines/trainer_install.lock"
  flock 8
  if [[ "$MODEL" == "nnunet" ]]; then
    install -m 0644 "$NNUNET_TRAINER_SRC" "$NNUNET_TRAINER_DST"
  elif [[ "$MODEL" == "umamba" ]]; then
    install -m 0644 "$UMAMBA_TRAINER_SRC" "$UMAMBA_TRAINER_DST"
    install -m 0644 "$UMAMBA_PATHS_SRC" "$UMAMBA_PATHS_DST"
    install -m 0644 "$UMAMBA_PLANS_SRC" "$UMAMBA_PLANS_DST"
  fi
  flock -u 8
  "$RMTFD" smoke_tests/prepare_protocol_v3_nnunet.py \
    --dataset "$DATASET" \
    --out-root "$NNROOT" \
    > "$OUT/prepare.log" 2>&1

  export nnUNet_raw=$NNROOT/nnUNet_raw
  if [[ "$MODEL" == "nnunet" ]]; then
    BIN=$SOTA_BIN
    TRAINER=nnUNetTrainerProtocolV3Seed$SEED
    export nnUNet_preprocessed=$NNROOT/nnUNet_preprocessed
    export nnUNet_results=$NNROOT/nnUNet_results
  elif [[ "$MODEL" == "umamba" ]]; then
    BIN=$UMAMBA_BIN
    TRAINER=nnUNetTrainerUMambaBotProtocolV3Seed$SEED
    export nnUNet_preprocessed=$UMAMBA_ROOT/nnUNet_preprocessed
    export nnUNet_results=$UMAMBA_ROOT/nnUNet_results
    mkdir -p "$nnUNet_preprocessed" "$nnUNet_results"
  else
    echo "Unknown model: $MODEL" >&2
    exit 2
  fi

  PLANS=$nnUNet_preprocessed/$DATASET_NAME/nnUNetPlans.json
  if [[ ! -s "$PLANS" ]]; then
    PATH=$BIN:$PATH nnUNetv2_plan_and_preprocess \
      -d "$DATASET_ID" \
      --verify_dataset_integrity \
      -c 2d \
      > "$OUT/plan_preprocess.log" 2>&1
    "$RMTFD" smoke_tests/set_nnunet_plan_batch_size.py \
      --plans "$PLANS" \
      --config 2d \
      --batch-size 8 \
      --report-md "$OUT/plan_batch_size.md" \
      > "$OUT/plan_batch_size.log" 2>&1
  fi
  if [[ "$MODEL" == "umamba" ]]; then
    cp "$NNROOT/nnUNet_preprocessed/$DATASET_NAME/splits_final.json" \
      "$nnUNet_preprocessed/$DATASET_NAME/splits_final.json"
  fi

  RESULT_DIR=$nnUNet_results/$DATASET_NAME/$TRAINER"__nnUNetPlans__2d"/fold_0
  CONTINUE_ARGS=
  if [[ -s "$RESULT_DIR/checkpoint_latest.pth" ]]; then
    CONTINUE_ARGS=--c
  fi
  CUDA_VISIBLE_DEVICES=$GPU PATH=$BIN:$PATH nnUNetv2_train \
    "$DATASET_ID" 2d 0 \
    -tr "$TRAINER" \
    $CONTINUE_ARGS \
    > "$OUT/train.log" 2>&1
  CHECKPOINT=$RESULT_DIR/checkpoint_best.pth
  test -s "$CHECKPOINT"
  PRED_DIR=$OUT/controls/true/nnunet_predictions
  CUDA_VISIBLE_DEVICES=$GPU PATH=$BIN:$PATH nnUNetv2_predict \
    -i "$nnUNet_raw/$DATASET_NAME/imagesTs" \
    -o "$PRED_DIR" \
    -d "$DATASET_ID" \
    -c 2d \
    -f 0 \
    -tr "$TRAINER" \
    -chk checkpoint_best.pth \
    > "$OUT/predict.log" 2>&1
  "$RMTFD" smoke_tests/build_protocol_v3_nnunet_prediction_index.py \
    --mapping "$nnUNet_raw/$DATASET_NAME/protocol_v3_mapping.csv" \
    --prediction-dir "$PRED_DIR" \
    --normalized-dir "$OUT/controls/true/predictions" \
    --checkpoint "$CHECKPOINT" \
    --protocol-lock "$LOCK" \
    --output-index "$OUT/controls/true/prediction_index.csv" \
    > "$OUT/index.log" 2>&1
fi

"$RMTFD" smoke_tests/evaluate_predictions_v3.py \
  --prediction-index "$OUT/controls/true/prediction_index.csv" \
  --protocol-lock "$LOCK" \
  --per-case-csv "$OUT/controls/true/per_case.csv" \
  --summary-csv "$OUT/controls/true/summary.csv" \
  > "$OUT/evaluate.log" 2>&1

{
  echo "status=complete"
  echo "completed_at=$(date -Is)"
  echo "checkpoint=$CHECKPOINT"
  echo "checkpoint_sha256=$(sha256sum "$CHECKPOINT" | awk '{print $1}')"
} >> "$OUT/run_meta.txt"
echo "complete_at=$(date -Is)" > "$OUT/complete.status"
rm -f "$OUT/failed.status"
trap - ERR
echo "PASS model=$MODEL dataset=$DATASET seed=$SEED gpu=$GPU"
