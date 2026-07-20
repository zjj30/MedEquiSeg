#!/usr/bin/env python3
"""Run the Public-5 stratified prompt control on the frozen R11 checkpoints."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


PROTOCOL_HASH = "abb8ccb2d46cf039a4a0c573d733e3bc510636e6dfa7e59032508ab96fc0f718"
RUN_ID = "V3_ABL_EQUIPROMPT"
MODEL = "CausalCLIPSegRN50DiceBCE"
DATASETS = (
    "medclipseg_busi",
    "medclipseg_clinicdb",
    "medclipseg_busbra",
    "medclipseg_brisc",
    "medclipseg_covid19",
)
SEEDS = (123, 456, 789)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("<PROJECT_ROOT>"))
    parser.add_argument("--gpu", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_logged(command: list[str], log_path: Path, env: dict[str, str], root: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("command=" + " ".join(command) + "\n")
        handle.flush()
        result = subprocess.run(command, cwd=root, env=env, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed rc={result.returncode}; see {log_path}")


def write_status(path: Path, rows: list[dict[str, object]]) -> None:
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    python = root / ".." / "envs" / "rmtfd" / "bin" / "python"
    if not python.is_file():
        python = Path("<ENV_ROOT>/bin/python")
    protocol_lock = root / "smoke_tests" / "protocol_v3" / "protocol_lock.yaml"
    predictor = root / "smoke_tests" / "predict_protocol_v3.py"
    evaluator = root / "smoke_tests" / "evaluate_predictions_v3.py"
    cache = root / "outputs" / "text_embeddings" / f"biomedclip_protocol_v3_{PROTOCOL_HASH[:16]}.npz"
    log_root = root / "logs" / "protocol_v3" / PROTOCOL_HASH
    result_root = root / "paper" / "results" / "protocol_v3_stratified_derangement_20260719"
    run_root = result_root / "runs"
    runner_sha = sha256(Path(__file__))
    predictor_sha = sha256(predictor)
    status_rows: list[dict[str, object]] = []
    failures = 0

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    for dataset in DATASETS:
        manifest = root / "smoke_tests" / "protocol_v3" / "manifests" / f"{dataset}_full.csv"
        control_map = result_root / "control_maps" / f"{dataset}_stratified_derangement_v1.csv"
        for seed in SEEDS:
            started = time.time()
            cell = f"{dataset}/seed{seed}"
            output = run_root / dataset / f"seed{seed}"
            output.mkdir(parents=True, exist_ok=True)
            source_meta_path = log_root / RUN_ID / dataset / f"seed{seed}" / "run_meta.json"
            source_meta = load_json(source_meta_path)
            if source_meta.get("protocol_hash") != PROTOCOL_HASH or source_meta.get("status") != "complete":
                raise RuntimeError(f"Invalid source run metadata: {source_meta_path}")
            checkpoint = Path(source_meta["checkpoint"])
            expected = {
                "protocol_hash": PROTOCOL_HASH,
                "dataset": dataset,
                "seed": seed,
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": source_meta["checkpoint_sha256"],
                "manifest_sha256": sha256(manifest),
                "control_map_sha256": sha256(control_map),
                "predictor_code_sha256": predictor_sha,
                "runner_code_sha256": runner_sha,
                "batch_size": args.batch_size,
                "workers": args.workers,
            }
            expected_hash = canonical_hash(expected)
            complete_path = output / "complete.json"
            if complete_path.is_file() and not args.force:
                actual = load_json(complete_path)
                required = ("prediction_index.csv", "per_case.csv", "summary.csv", "run_meta.json")
                if actual.get("expected_hash") == expected_hash and all((output / name).is_file() for name in required):
                    status_rows.append(
                        {
                            "dataset": dataset,
                            "seed": seed,
                            "status": "SKIP_COMPLETE",
                            "seconds": f"{time.time() - started:.3f}",
                            "message": "hash-matched",
                        }
                    )
                    write_status(result_root / "task_status.csv", status_rows)
                    print(f"SKIP {cell}", flush=True)
                    continue

            predict_command = [
                str(python),
                str(predictor),
                "--model",
                MODEL,
                "--checkpoint",
                str(checkpoint),
                "--manifest",
                str(manifest),
                "--protocol-lock",
                str(protocol_lock),
                "--dataset",
                dataset,
                "--output-dir",
                str(output),
                "--prompt-control",
                "stratified_derangement",
                "--stratified-control-map",
                str(control_map),
                "--control-seed",
                "123",
                "--batch-size",
                str(args.batch_size),
                "--workers",
                str(args.workers),
                "--text-encoder-cache",
                str(cache),
            ]
            evaluate_command = [
                str(python),
                str(evaluator),
                "--prediction-index",
                str(output / "prediction_index.csv"),
                "--protocol-lock",
                str(protocol_lock),
                "--per-case-csv",
                str(output / "per_case.csv"),
                "--summary-csv",
                str(output / "summary.csv"),
            ]
            try:
                run_logged(predict_command, output / "predict.log", env, root)
                prediction_meta = load_json(output / "run_meta.json")
                checks = {
                    "protocol_hash": PROTOCOL_HASH,
                    "dataset": dataset,
                    "prompt_control": "stratified_derangement",
                    "checkpoint_sha256": source_meta["checkpoint_sha256"],
                    "control_map_sha256": expected["control_map_sha256"],
                    "predictor_code_sha256": predictor_sha,
                }
                for key, value in checks.items():
                    if prediction_meta.get(key) != value:
                        raise RuntimeError(f"Prediction metadata mismatch {cell} {key}: {prediction_meta.get(key)} != {value}")
                run_logged(evaluate_command, output / "evaluate.log", env, root)
                evidence = {
                    "status": "complete",
                    "expected": expected,
                    "expected_hash": expected_hash,
                    "prediction_index_sha256": sha256(output / "prediction_index.csv"),
                    "per_case_sha256": sha256(output / "per_case.csv"),
                    "summary_sha256": sha256(output / "summary.csv"),
                }
                complete_path.write_text(json.dumps(evidence, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
                message = "complete"
                status = "PASS"
            except Exception as exc:
                failures += 1
                message = str(exc)
                status = "FAIL"
            status_rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "status": status,
                    "seconds": f"{time.time() - started:.3f}",
                    "message": message,
                }
            )
            write_status(result_root / "task_status.csv", status_rows)
            print(f"{status} {cell} seconds={time.time() - started:.1f} {message}", flush=True)

    payload = {
        "status": "PASS" if failures == 0 and len(status_rows) == 15 else "FAIL",
        "protocol_hash": PROTOCOL_HASH,
        "tasks": len(status_rows),
        "failures": failures,
        "runner_code_sha256": runner_sha,
        "predictor_code_sha256": predictor_sha,
        "gpu": args.gpu,
        "rows": status_rows,
    }
    (result_root / "matrix_audit.json").write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "tasks": len(status_rows), "failures": failures}), flush=True)
    return 1 if payload["status"] != "PASS" else 0


if __name__ == "__main__":
    raise SystemExit(main())
