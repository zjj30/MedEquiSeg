#!/usr/bin/env python3
"""Run the five retained Public-5 configurations and prompt controls."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import queue
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from causal_clip_recipe import get_recipe
from protocol_v3.core import canonical_hash, file_sha256, load_protocol_lock, protocol_sha256


ROOT = Path(__file__).resolve().parents[1]
PYTHON = os.environ.get("MEDEQUISEG_PYTHON", sys.executable)
MODEL = "CausalCLIPSegRN50DiceBCE"
PILOT_RUNS = (
    "V3_ABL_BASE",
    "V3_ABL_BIOMED",
    "V3_ABL_ATCONV",
    "V3_ABL_BIOMED_ATCONV",
    "V3_ABL_EQUIPROMPT",
)
CONFIRMATORY_RUNS = PILOT_RUNS
V3_RUNS = {
    "V3_ABL_BASE": {"recipe": "default", "augmentation": "lcaug_v2_hflip_dataset"},
    "V3_ABL_BIOMED": {"recipe": "biomed_lcaug", "augmentation": "lcaug_v2_hflip_dataset"},
    "V3_ABL_ATCONV": {"recipe": "default_atconv4", "augmentation": "lcaug_v2_hflip_dataset"},
    "V3_ABL_BIOMED_ATCONV": {"recipe": "biomed_lcaug_atconv4", "augmentation": "lcaug_v2_hflip_dataset"},
    "V3_ABL_EQUIPROMPT": {
        "recipe": "biomed_lcaug_v2_atconv4",
        "augmentation": "lcaug_v2_dynamic_shared_plan_dataset",
    },
}
PROMPT_CONTROLS = ("true", "shuffled", "fixed", "empty")
CODE_FILES = (
    "smoke_tests/run_protocol_v3.py",
    "smoke_tests/build_group_disjoint_public5_manifests.py",
    "smoke_tests/train_baselines.py",
    "smoke_tests/augmentation_plugins.py",
    "smoke_tests/lcaug_v2_direction.py",
    "smoke_tests/causal_clip_recipe.py",
    "smoke_tests/causal_atconv_plugin.py",
    "smoke_tests/causal_text_encoder_plugins.py",
    "smoke_tests/biomedclip_offline.py",
    "smoke_tests/image_resize.py",
    "smoke_tests/text_encoders.py",
    "smoke_tests/build_causal_biomedclip_cache.py",
    "smoke_tests/build_text_encoder_ablation_cache.py",
    "smoke_tests/predict_protocol_v3.py",
    "smoke_tests/evaluate_predictions_v3.py",
    "smoke_tests/paper_metrics.py",
    "smoke_tests/protocol_v3/core.py",
)


@dataclass(frozen=True)
class Task:
    run_id: str
    dataset: str
    seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-lock", type=Path, default=ROOT / "smoke_tests/protocol_v3/protocol_lock.yaml")
    parser.add_argument("--stage", choices=["pilot", "confirmatory"], default="pilot")
    parser.add_argument("--run-ids", nargs="+", default=None)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--jobs-per-gpu", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=0, help="Defaults to the locked training_epochs value.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def code_sha256() -> str:
    return canonical_hash({path: file_sha256(ROOT / path) for path in CODE_FILES})


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def parse_checkpoint(log_path: Path) -> Path:
    matches = re.findall(r"best_checkpoint:\s*(\S+)", log_path.read_text(encoding="utf-8", errors="replace"))
    if not matches:
        raise RuntimeError(f"No best checkpoint in {log_path}")
    return Path(matches[-1])


def run_logged(cmd: list[str], log_path: Path, env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("command=" + " ".join(cmd) + "\n")
        handle.flush()
        result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed rc={result.returncode}; see {log_path}")


def cache_path(protocol_hash: str) -> Path:
    return ROOT / "outputs/text_embeddings" / f"biomedclip_protocol_v3_{protocol_hash[:16]}.npz"


def cache_for_run(run_id: str, cache: Path | None) -> Path | None:
    recipe = get_recipe(V3_RUNS[run_id]["recipe"])
    return cache if recipe.text_encoder != "clip_rn50" else None


def ensure_cache(lock: dict, protocol_hash: str, gpu: int, dry_run: bool) -> Path:
    manifests = [str(ROOT / cfg["manifest"]) for cfg in lock["datasets"].values()]
    output = cache_path(protocol_hash)
    cmd = [
        PYTHON,
        "smoke_tests/build_causal_biomedclip_cache.py",
        "--manifest",
        *manifests,
        "--include-lcaug-variants",
        "--include-empty-control",
        "--output",
        str(output),
        "--device",
        "cuda:0",
    ]
    if dry_run:
        print("DRY cache:", " ".join(cmd), flush=True)
        return output
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    result = subprocess.run(cmd, cwd=ROOT, env=env, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Cache preflight/build failed rc={result.returncode}")
    return output


def task_dir(protocol_hash: str, task: Task) -> Path:
    return ROOT / "logs/protocol_v3" / protocol_hash / task.run_id / task.dataset / f"seed{task.seed}"


def expected_meta(task: Task, lock_path: Path, protocol_hash: str, cache: Path | None, epochs: int) -> dict:
    lock = load_protocol_lock(lock_path)
    manifest = ROOT / lock["datasets"][task.dataset]["manifest"]
    run_meta = V3_RUNS[task.run_id]
    payload = {
        "protocol_id": lock["protocol_id"],
        "protocol_hash": protocol_hash,
        "code_sha256": code_sha256(),
        "git_commit": git_head(),
        "manifest": str(manifest),
        "manifest_sha256": file_sha256(manifest),
        "cache": str(cache) if cache else "",
        "cache_sha256": file_sha256(cache) if cache and cache.is_file() else "",
        "task": asdict(task),
        "recipe": run_meta["recipe"],
        "augmentation": run_meta["augmentation"],
        "epochs": epochs,
    }
    payload["run_config_sha256"] = canonical_hash(payload)
    return payload


def is_complete(directory: Path, expected: dict) -> bool:
    meta_path = directory / "run_meta.json"
    if not meta_path.is_file():
        return False
    actual = json.loads(meta_path.read_text(encoding="utf-8"))
    if actual.get("status") != "complete" or actual.get("run_config_sha256") != expected["run_config_sha256"]:
        return False
    return all((directory / "controls" / control / "summary.csv").is_file() for control in PROMPT_CONTROLS)


def execute_task(task: Task, gpu: int, args: argparse.Namespace, lock: dict, protocol_hash: str, cache: Path | None) -> str:
    directory = task_dir(protocol_hash, task)
    run_meta = V3_RUNS[task.run_id]
    recipe = get_recipe(run_meta["recipe"])
    task_cache = cache_for_run(task.run_id, cache)
    expected = expected_meta(task, args.protocol_lock, protocol_hash, task_cache, args.epochs)
    if is_complete(directory, expected):
        return f"SKIP complete {task.run_id} {task.dataset} seed={task.seed}"
    manifest = ROOT / lock["datasets"][task.dataset]["manifest"]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONUNBUFFERED"] = "1"
    directory.mkdir(parents=True, exist_ok=True)
    starting = {**expected, "status": "running", "gpu": gpu}
    (directory / "run_meta.json").write_text(json.dumps(starting, indent=2), encoding="utf-8")

    train_cmd = [
        PYTHON,
        "smoke_tests/train_baselines.py",
        "--model",
        MODEL,
        "--manifest",
        str(manifest),
        "--protocol-lock",
        str(args.protocol_lock),
        "--datasets",
        task.dataset,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--workers",
        str(args.workers),
        "--max-train-samples",
        "0",
        "--max-val-samples",
        "0",
        "--seed",
        str(task.seed),
        "--split-seed",
        "123",
        "--scheduler",
        "cosine",
        "--min-lr",
        "1e-6",
        "--causal-recipe",
        recipe.name,
    ]
    if run_meta["augmentation"] != "none":
        train_cmd += ["--augment", "--aug-strength", run_meta["augmentation"]]
    if task_cache:
        train_cmd += ["--text-encoder-cache", str(task_cache)]
    run_logged(train_cmd, directory / "train.log", env)
    checkpoint = parse_checkpoint(directory / "train.log")

    for control in PROMPT_CONTROLS:
        control_dir = directory / "controls" / control
        prediction_cmd = [
            PYTHON,
            "smoke_tests/predict_protocol_v3.py",
            "--model",
            MODEL,
            "--checkpoint",
            str(checkpoint),
            "--manifest",
            str(manifest),
            "--protocol-lock",
            str(args.protocol_lock),
            "--dataset",
            task.dataset,
            "--output-dir",
            str(control_dir),
            "--prompt-control",
            control,
            "--control-seed",
            "123",
            "--batch-size",
            str(args.batch_size),
        ]
        if task_cache:
            prediction_cmd += ["--text-encoder-cache", str(task_cache)]
        run_logged(prediction_cmd, control_dir / "predict.log", env)
        evaluation_cmd = [
            PYTHON,
            "smoke_tests/evaluate_predictions_v3.py",
            "--prediction-index",
            str(control_dir / "prediction_index.csv"),
            "--protocol-lock",
            str(args.protocol_lock),
            "--per-case-csv",
            str(control_dir / "per_case.csv"),
            "--summary-csv",
            str(control_dir / "summary.csv"),
        ]
        run_logged(evaluation_cmd, control_dir / "evaluate.log", env)

    completed = {**expected, "status": "complete", "gpu": gpu, "checkpoint": str(checkpoint), "checkpoint_sha256": file_sha256(checkpoint)}
    (directory / "run_meta.json").write_text(json.dumps(completed, indent=2), encoding="utf-8")
    return f"PASS {task.run_id} {task.dataset} seed={task.seed} gpu={gpu}"


def main() -> None:
    args = parse_args()
    lock = load_protocol_lock(args.protocol_lock)
    locked_epochs = int(lock["training_epochs"])
    if args.epochs == 0:
        args.epochs = locked_epochs
    if args.epochs != locked_epochs:
        raise ValueError(f"Protocol V3 requires exactly {locked_epochs} epochs; received {args.epochs}")
    protocol_hash = protocol_sha256(args.protocol_lock)
    datasets = args.datasets or list(lock["datasets"])
    run_ids = args.run_ids or list(PILOT_RUNS if args.stage == "pilot" else CONFIRMATORY_RUNS)
    seeds = args.seeds or ([123] if args.stage == "pilot" else list(lock["seeds"]))
    for run_id in run_ids:
        if run_id not in V3_RUNS:
            raise ValueError(f"Unknown V3 run id: {run_id}")
    tasks = [Task(run_id, dataset, seed) for run_id in run_ids for dataset in datasets for seed in seeds]
    print(f"protocol_hash={protocol_hash} tasks={len(tasks)} stage={args.stage}")
    if args.dry_run:
        for task in tasks:
            print("DRY task", task)
        return

    needs_biomed = any(get_recipe(V3_RUNS[task.run_id]["recipe"]).text_encoder != "clip_rn50" for task in tasks)
    cache = ensure_cache(lock, protocol_hash, args.gpus[0], False) if needs_biomed else None
    gpu_pool: queue.Queue[int] = queue.Queue()
    for gpu in args.gpus:
        for _ in range(args.jobs_per_gpu):
            gpu_pool.put(gpu)

    def worker(task: Task) -> str:
        gpu = gpu_pool.get()
        try:
            return execute_task(task, gpu, args, lock, protocol_hash, cache)
        finally:
            gpu_pool.put(gpu)

    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.gpus) * args.jobs_per_gpu) as executor:
        futures = {executor.submit(worker, task): task for task in tasks}
        for future in concurrent.futures.as_completed(futures):
            task = futures[future]
            try:
                print(future.result(), flush=True)
            except Exception as exc:
                failures.append((task, str(exc)))
                directory = task_dir(protocol_hash, task)
                meta_path = directory / "run_meta.json"
                meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
                meta.update({"status": "failed", "error": str(exc)})
                directory.mkdir(parents=True, exist_ok=True)
                meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
                print(f"FAIL {task}: {exc}", flush=True)
    if failures:
        raise SystemExit(f"Protocol V3 failed tasks: {failures}")
    if args.stage == "pilot":
        subprocess.run(
            [PYTHON, "smoke_tests/summarize_protocol_v3_gate.py", "--protocol-lock", str(args.protocol_lock)],
            cwd=ROOT,
            check=True,
        )
    print("final_status: PASS")


if __name__ == "__main__":
    main()
