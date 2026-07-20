#!/usr/bin/env python3
"""Run official MedCLIPSeg on all Protocol V3 datasets and text controls."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import queue
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from protocol_v3.core import canonical_hash, file_sha256, load_protocol_lock, protocol_sha256


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "repos/MedCLIPSeg"
PYTHON = "<ENV_ROOT>/sota_baselines/bin/python"
PROJECT_PYTHON = "<ENV_ROOT>/bin/python"
RUN_ID = "V3_MEDCLIPSEG_OFFICIAL"
MODEL_SLUG = "MedCLIPSeg_unimedclip_ViT-B-16"
CONTROLS = {"true": "original", "shuffled": "shuffled", "fixed": "fixed", "empty": "empty"}
CODE_FILES = (
    "smoke_tests/run_medclipseg_protocol_v3.py",
    "smoke_tests/prepare_medclipseg_protocol_v3.py",
    "smoke_tests/evaluate_medclipseg_protocol_v3.py",
    "smoke_tests/evaluate_predictions_v3.py",
    "smoke_tests/paper_metrics.py",
    "smoke_tests/protocol_v3/core.py",
    "repos/MedCLIPSeg/train.py",
    "repos/MedCLIPSeg/test.py",
    "repos/MedCLIPSeg/datasets/dataloader.py",
    "repos/MedCLIPSeg/trainers/medclipseg_unimedclip.py",
    "repos/MedCLIPSeg/utils/main_utils.py",
)


@dataclass(frozen=True)
class Task:
    dataset: str
    seed: int
    lock_path: str
    protocol_hash: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol-locks",
        nargs="+",
        type=Path,
        default=[
            ROOT / "smoke_tests/protocol_v3/protocol_lock.yaml",
            ROOT / "<PRIVATE_LOCK_NOT_RELEASED>",
        ],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[123, 456, 789])
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--jobs-per-gpu", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--prepare", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite-staging", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def git_head(path: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()
    except Exception:
        return "unknown"


def code_sha256() -> str:
    return canonical_hash({path: file_sha256(ROOT / path) for path in CODE_FILES if (ROOT / path).is_file()})


def run_logged(cmd: list[str], log_path: Path, env: dict[str, str], cwd: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("command=" + " ".join(cmd) + "\n")
        handle.flush()
        result = subprocess.run(cmd, cwd=cwd, env=env, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed rc={result.returncode}; see {log_path}")


def dataset_root(dataset: str) -> Path:
    return ROOT / "outputs/medclipseg_protocol_v3" / dataset


def runs_root() -> Path:
    return ROOT / "outputs/medclipseg_protocol_v3/runs"


def prepare_dataset(lock_path: Path, dataset: str, args: argparse.Namespace) -> None:
    lock = load_protocol_lock(lock_path)
    manifest = ROOT / lock["datasets"][dataset]["manifest"]
    command = [
        PROJECT_PYTHON,
        "smoke_tests/prepare_medclipseg_protocol_v3.py",
        "--manifest",
        str(manifest),
        "--protocol-lock",
        str(lock_path),
        "--dataset",
        dataset,
        "--batch-size",
        str(args.batch_size),
        "--epochs",
        str(args.epochs),
    ]
    if args.overwrite_staging:
        command.append("--overwrite")
    log = dataset_root(dataset) / "prepare.log"
    run_logged(command, log, os.environ.copy(), ROOT)


def prepare_meta(dataset: str) -> dict:
    path = dataset_root(dataset) / "prepare_meta.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing staged MedCLIPSeg metadata: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def task_dir(task: Task) -> Path:
    return ROOT / "logs/protocol_v3" / task.protocol_hash / RUN_ID / task.dataset / f"seed{task.seed}"


def expected_meta(task: Task, args: argparse.Namespace) -> dict:
    staged = prepare_meta(task.dataset)
    lock_path = Path(task.lock_path)
    lock = load_protocol_lock(lock_path)
    manifest = ROOT / lock["datasets"][task.dataset]["manifest"]
    payload = {
        "protocol_id": lock["protocol_id"],
        "protocol_hash": task.protocol_hash,
        "code_sha256": code_sha256(),
        "project_git_commit": git_head(ROOT),
        "medclipseg_git_commit": git_head(REPO),
        "task": asdict(task),
        "model": "official MedCLIPSeg unimedclip ViT-B/16",
        "manifest": str(manifest.resolve()),
        "manifest_sha256": file_sha256(manifest),
        "staging_meta": staged,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "checkpoint_selection": "best_validation_dice",
        "prompt_controls": list(CONTROLS),
        "from_scratch": True,
    }
    payload["run_config_sha256"] = canonical_hash(payload)
    return payload


def is_complete(directory: Path, expected: dict) -> bool:
    path = directory / "run_meta.json"
    if not path.is_file():
        return False
    actual = json.loads(path.read_text(encoding="utf-8"))
    if actual.get("status") != "complete" or actual.get("run_config_sha256") != expected["run_config_sha256"]:
        return False
    return all((directory / "controls" / control / "summary.csv").is_file() for control in CONTROLS)


def checkpoint_paths(dataset_name: str, seed: int) -> tuple[Path, Path]:
    root = runs_root() / dataset_name / "trained_models" / f"seed{seed}"
    return root / f"{MODEL_SLUG}_best_dice.pth", root / f"{MODEL_SLUG}_latest.pth"


def execute(task: Task, gpu: int, args: argparse.Namespace) -> str:
    directory = task_dir(task)
    expected = expected_meta(task, args)
    if is_complete(directory, expected):
        return f"SKIP complete {task.dataset} seed={task.seed}"
    staged = prepare_meta(task.dataset)
    dataset_name = staged["dataset_name"]
    config = Path(staged["config"])
    adapter_manifest = dataset_root(task.dataset) / "adapter_manifest.csv"
    best_checkpoint, latest_checkpoint = checkpoint_paths(dataset_name, task.seed)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "run_meta.json").write_text(
        json.dumps({**expected, "status": "running", "gpu": gpu}, indent=2), encoding="utf-8"
    )
    for checkpoint in (best_checkpoint, latest_checkpoint):
        checkpoint.unlink(missing_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONUNBUFFERED"] = "1"
    env["OMP_NUM_THREADS"] = "2"
    train_command = [
        PYTHON,
        "train.py",
        "--config-file",
        str(config),
        "--output-dir",
        str(runs_root()),
        "--seed",
        str(task.seed),
        "TRAIN.BATCH_SIZE",
        str(args.batch_size),
        "TRAIN.NUM_EPOCHS",
        str(args.epochs),
        "MODEL.DEVICE",
        "cuda",
    ]
    run_logged(train_command, directory / "train.log", env, REPO)
    if not best_checkpoint.is_file():
        raise FileNotFoundError(f"MedCLIPSeg best checkpoint missing: {best_checkpoint}")

    for control, prompt_design in CONTROLS.items():
        control_dir = directory / "controls" / control
        prediction_dir = (
            runs_root()
            / dataset_name
            / "seg_results"
            / f"seed{task.seed}"
            / f"{MODEL_SLUG}_Prompt-{prompt_design}"
        )
        if prediction_dir.exists():
            shutil.rmtree(prediction_dir)
        test_command = [
            PYTHON,
            "test.py",
            "--config-file",
            str(config),
            "--output-dir",
            str(runs_root()),
            "--source_dataset",
            dataset_name,
            "--seed",
            str(task.seed),
            "--prompt_design",
            prompt_design,
            "MODEL.DEVICE",
            "cuda",
        ]
        run_logged(test_command, control_dir / "test.log", env, REPO)
        evaluate_command = [
            PROJECT_PYTHON,
            "smoke_tests/evaluate_medclipseg_protocol_v3.py",
            "--adapter-manifest",
            str(adapter_manifest),
            "--prediction-dir",
            str(prediction_dir),
            "--checkpoint",
            str(best_checkpoint),
            "--protocol-lock",
            task.lock_path,
            "--prompt-control",
            control,
            "--output-dir",
            str(control_dir),
        ]
        run_logged(evaluate_command, control_dir / "evaluate.log", env, ROOT)

    completed = {
        **expected,
        "status": "complete",
        "gpu": gpu,
        "checkpoint": str(best_checkpoint),
        "checkpoint_sha256": file_sha256(best_checkpoint),
    }
    (directory / "run_meta.json").write_text(json.dumps(completed, indent=2), encoding="utf-8")
    return f"PASS {task.dataset} seed={task.seed} gpu={gpu}"


def main() -> None:
    args = parse_args()
    tasks: list[Task] = []
    for lock_path in args.protocol_locks:
        lock_path = lock_path.resolve()
        lock = load_protocol_lock(lock_path)
        locked_epochs = int(lock["training_epochs"])
        if args.epochs != locked_epochs:
            raise ValueError(f"Protocol lock requires {locked_epochs} epochs, received {args.epochs}")
        current_hash = protocol_sha256(lock_path)
        for dataset in lock["datasets"]:
            if args.prepare and not args.dry_run:
                prepare_dataset(lock_path, dataset, args)
            for seed in args.seeds:
                tasks.append(Task(dataset, seed, str(lock_path), current_hash))

    print(f"run_id={RUN_ID} tasks={len(tasks)} gpus={args.gpus} jobs_per_gpu={args.jobs_per_gpu}")
    if args.dry_run:
        for task in tasks:
            print("DRY task", task)
        return

    gpu_pool: queue.Queue[int] = queue.Queue()
    for gpu in args.gpus:
        for _ in range(args.jobs_per_gpu):
            gpu_pool.put(gpu)

    def worker(task: Task) -> tuple[Task, str, str]:
        gpu = gpu_pool.get()
        try:
            try:
                return task, execute(task, gpu, args), ""
            except Exception as exc:
                directory = task_dir(task)
                meta_path = directory / "run_meta.json"
                meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
                meta.update({"status": "failed", "gpu": gpu, "error": str(exc)})
                directory.mkdir(parents=True, exist_ok=True)
                meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
                return task, "", str(exc)
        finally:
            gpu_pool.put(gpu)

    remaining = tasks
    failures: list[tuple[Task, str]] = []
    for attempt in range(args.max_retries + 1):
        failures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.gpus) * args.jobs_per_gpu) as executor:
            futures = [executor.submit(worker, task) for task in remaining]
            for future in concurrent.futures.as_completed(futures):
                task, message, error = future.result()
                if error:
                    failures.append((task, error))
                    print(f"FAIL attempt={attempt} task={task}: {error}", flush=True)
                else:
                    print(message, flush=True)
        if not failures:
            break
        remaining = [task for task, _ in failures]
        print(f"retrying={len(remaining)} next_attempt={attempt + 1}", flush=True)
    if failures:
        raise SystemExit(f"MedCLIPSeg Protocol V3 failures: {failures}")
    print("final_status: PASS")


if __name__ == "__main__":
    main()
