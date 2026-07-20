#!/usr/bin/env python3
"""Compute paired strict MedEquiSeg-versus-Base accuracy contrasts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROTOCOL_HASH = "abb8ccb2d46cf039a4a0c573d733e3bc510636e6dfa7e59032508ab96fc0f718"
SEEDS = (123, 456, 789)
DATASETS = (
    ("BUSI", "medclipseg_busi"),
    ("ClinicDB", "medclipseg_clinicdb"),
    ("BUS-BRA", "medclipseg_busbra"),
    ("BRISC", "medclipseg_brisc"),
    ("COVID-19", "medclipseg_covid19"),
)
LEFT_RUN = "V3_ABL_EQUIPROMPT"
RIGHT_RUN = "V3_ABL_BASE"
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20260718


def per_case_path(root: Path, run_id: str, dataset: str, seed: int) -> Path:
    return (
        root
        / "logs"
        / "protocol_v3"
        / PROTOCOL_HASH
        / run_id
        / dataset
        / f"seed{seed}"
        / "controls"
        / "true"
        / "per_case.csv"
    )


def delta_matrix(root: Path, dataset: str, metric: str) -> np.ndarray:
    columns: list[np.ndarray] = []
    expected_ids: list[str] | None = None
    for seed in SEEDS:
        frames = []
        for run_id in (LEFT_RUN, RIGHT_RUN):
            path = per_case_path(root, run_id, dataset, seed)
            frame = pd.read_csv(path, usecols=["case_id", metric]).sort_values("case_id")
            if frame["case_id"].duplicated().any():
                raise ValueError(f"Duplicate case_id in {path}")
            frames.append(frame)
        if frames[0]["case_id"].tolist() != frames[1]["case_id"].tolist():
            raise ValueError(f"Case mismatch for {dataset}, seed {seed}")
        case_ids = frames[0]["case_id"].tolist()
        if expected_ids is None:
            expected_ids = case_ids
        elif case_ids != expected_ids:
            raise ValueError(f"Seed-level case mismatch for {dataset}")
        columns.append(frames[0][metric].to_numpy() - frames[1][metric].to_numpy())
    return np.column_stack(columns)


def bootstrap_dataset(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    n_cases, n_seeds = values.shape
    estimates = np.empty(BOOTSTRAP_SAMPLES)
    for index in range(BOOTSTRAP_SAMPLES):
        cases = rng.integers(0, n_cases, n_cases)
        seeds = rng.integers(0, n_seeds, n_seeds)
        estimates[index] = values[np.ix_(cases, seeds)].mean()
    return tuple(float(value) for value in np.quantile(estimates, (0.025, 0.975)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper/results/medequiseg_base_contrast_20260718"),
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows: list[dict[str, float | int | str]] = []
    matrices: dict[str, dict[str, np.ndarray]] = {}

    for display, dataset in DATASETS:
        matrices[display] = {}
        for metric in ("dice", "iou"):
            values = delta_matrix(root, dataset, metric)
            matrices[display][metric] = values
            low, high = bootstrap_dataset(values, rng)
            rows.append(
                {
                    "dataset": display,
                    "metric": metric,
                    "n_cases": values.shape[0],
                    "delta": float(values.mean()),
                    "ci_low": low,
                    "ci_high": high,
                }
            )

    for metric in ("dice", "iou"):
        estimates = np.empty(BOOTSTRAP_SAMPLES)
        for index in range(BOOTSTRAP_SAMPLES):
            effects = []
            for display, _ in DATASETS:
                values = matrices[display][metric]
                n_cases, n_seeds = values.shape
                cases = rng.integers(0, n_cases, n_cases)
                seeds = rng.integers(0, n_seeds, n_seeds)
                effects.append(float(values[np.ix_(cases, seeds)].mean()))
            estimates[index] = np.mean(effects)
        low, high = np.quantile(estimates, (0.025, 0.975))
        rows.append(
            {
                "dataset": "Public-5 macro",
                "metric": metric,
                "n_cases": sum(matrix[metric].shape[0] for matrix in matrices.values()),
                "delta": float(np.mean([matrix[metric].mean() for matrix in matrices.values()])),
                "ci_low": float(low),
                "ci_high": float(high),
            }
        )

    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "base_contrast_statistics.csv", index=False, float_format="%.8g")
    payload = {
        "left_run": LEFT_RUN,
        "right_run": RIGHT_RUN,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "resampling": "paired test cases and three training seeds; equal-dataset macro",
        "rows": frame.to_dict(orient="records"),
    }
    (output_dir / "base_contrast_statistics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
