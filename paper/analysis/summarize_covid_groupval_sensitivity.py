#!/usr/bin/env python3
"""Summarize the COVID-19 grouped-validation sensitivity experiment."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


SEEDS = (123, 456, 789)
RUNS = {
    "R11": "V3_R11_COVID_GROUPVAL",
    "R11NR": "V3_R11NR_COVID_GROUPVAL",
}
BOOTSTRAP_SEED = 20260715
BOOTSTRAP_SAMPLES = 10_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument(
        "--protocol-lock",
        type=Path,
        default=Path("smoke_tests/protocol_v3/protocol_lock_covid_groupval_sensitivity.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper/results/protocol_v3_covid_groupval_sensitivity"),
    )
    return parser.parse_args()


def dataset_summary(path: Path) -> pd.Series:
    frame = pd.read_csv(path)
    row = frame[frame["scope"] == "dataset"]
    if len(row) != 1:
        raise ValueError(f"expected one dataset row in {path}")
    return row.iloc[0]


def hierarchical_ci(delta: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n_cases, n_seeds = delta.shape
    estimates = np.empty(BOOTSTRAP_SAMPLES, dtype=np.float64)
    for index in range(BOOTSTRAP_SAMPLES):
        case_index = rng.integers(0, n_cases, size=n_cases)
        seed_index = rng.integers(0, n_seeds, size=n_seeds)
        estimates[index] = delta[np.ix_(case_index, seed_index)].mean()
    return tuple(float(value) for value in np.quantile(estimates, (0.025, 0.975)))


def recover_subjects(case_ids: pd.Index) -> np.ndarray:
    subjects = []
    for case_id in case_ids.astype(str):
        match = re.search(r"sub-([^_]+)", case_id)
        if not match:
            raise ValueError(f"cannot recover COVID-19 subject from case_id: {case_id}")
        subjects.append(match.group(1))
    return np.asarray(subjects, dtype=object)


def subject_cluster_inference(
    delta: np.ndarray, case_ids: pd.Index
) -> dict[str, float | int]:
    """Sensitivity inference that treats repeated images as subject clusters."""
    subjects = recover_subjects(case_ids)
    unique_subjects = np.unique(subjects)
    subject_indices = {
        subject: np.flatnonzero(subjects == subject) for subject in unique_subjects
    }
    subject_delta = np.asarray(
        [delta[indices].mean(axis=0).mean() for indices in subject_indices.values()]
    )
    wilcoxon = stats.wilcoxon(subject_delta, zero_method="wilcox", method="auto")

    rng = np.random.default_rng(BOOTSTRAP_SEED + 1)
    estimates = np.empty(BOOTSTRAP_SAMPLES, dtype=np.float64)
    n_seeds = delta.shape[1]
    for index in range(BOOTSTRAP_SAMPLES):
        sampled_subjects = rng.choice(unique_subjects, size=len(unique_subjects), replace=True)
        sampled_cases = np.concatenate(
            [subject_indices[subject] for subject in sampled_subjects]
        )
        sampled_seeds = rng.integers(0, n_seeds, size=n_seeds)
        estimates[index] = delta[np.ix_(sampled_cases, sampled_seeds)].mean()
    low, high = np.quantile(estimates, (0.025, 0.975))
    return {
        "n_subjects": int(len(unique_subjects)),
        "subject_mean_delta": float(subject_delta.mean()),
        "subject_wilcoxon_p": float(wilcoxon.pvalue),
        "cluster_ci_low": float(low),
        "cluster_ci_high": float(high),
    }


def load_per_case(root: Path, protocol_hash: str, run_id: str, seed: int) -> pd.DataFrame:
    path = (
        root
        / "logs/protocol_v3"
        / protocol_hash
        / run_id
        / "medclipseg_covid19"
        / f"seed{seed}"
        / "controls/true/per_case.csv"
    )
    frame = pd.read_csv(path, usecols=["case_id", "dice", "iou"]).set_index("case_id")
    if frame.index.duplicated().any():
        raise ValueError(f"duplicate case_id in {path}")
    return frame.sort_index()


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    lock = args.protocol_lock if args.protocol_lock.is_absolute() else root / args.protocol_lock
    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(root / "smoke_tests"))
    from protocol_v3.core import protocol_sha256

    protocol_hash = protocol_sha256(lock)
    seed_rows: list[dict[str, float | int | str]] = []
    case_frames: dict[str, dict[int, pd.DataFrame]] = {name: {} for name in RUNS}

    for name, run_id in RUNS.items():
        for seed in SEEDS:
            run_root = (
                root
                / "logs/protocol_v3"
                / protocol_hash
                / run_id
                / "medclipseg_covid19"
                / f"seed{seed}"
                / "controls/true"
            )
            row = dataset_summary(run_root / "summary.csv")
            seed_rows.append(
                {
                    "method": name,
                    "seed": seed,
                    "dice": float(row["dice_mean"]),
                    "iou": float(row["iou_mean"]),
                    "samples": int(row["samples"]),
                    "source": str(run_root / "summary.csv"),
                }
            )
            case_frames[name][seed] = load_per_case(root, protocol_hash, run_id, seed)

    baseline_root = root / "logs/protocol_v3_covid_groupval_sensitivity/image_baselines"
    for seed in SEEDS:
        path = (
            baseline_root
            / "unetplusplus/medclipseg_covid19"
            / f"seed{seed}/controls/true/summary.csv"
        )
        row = dataset_summary(path)
        seed_rows.append(
            {
                "method": "U-Net++",
                "seed": seed,
                "dice": float(row["dice_mean"]),
                "iou": float(row["iou_mean"]),
                "samples": int(row["samples"]),
                "source": str(path),
            }
        )

    seeds = pd.DataFrame(seed_rows)
    aggregate = (
        seeds.groupby("method", sort=False)
        .agg(
            seeds=("seed", "nunique"),
            dice_mean=("dice", "mean"),
            dice_std=("dice", lambda x: x.std(ddof=1)),
            iou_mean=("iou", "mean"),
            iou_std=("iou", lambda x: x.std(ddof=1)),
        )
        .reset_index()
    )

    inference: list[dict[str, float | str]] = []
    for metric in ("dice", "iou"):
        r11_columns = []
        nr_columns = []
        expected_index = None
        for seed in SEEDS:
            r11 = case_frames["R11"][seed][metric]
            nr = case_frames["R11NR"][seed][metric]
            if not r11.index.equals(nr.index):
                raise ValueError(f"R11/R11NR case mismatch for seed {seed}")
            if expected_index is not None and not expected_index.equals(r11.index):
                raise ValueError(f"case mismatch across seeds for {metric}")
            expected_index = r11.index
            r11_columns.append(r11.to_numpy())
            nr_columns.append(nr.to_numpy())
        r11_matrix = np.column_stack(r11_columns)
        nr_matrix = np.column_stack(nr_columns)
        delta = r11_matrix - nr_matrix
        case_mean_delta = delta.mean(axis=1)
        low, high = hierarchical_ci(delta)
        wilcoxon = stats.wilcoxon(case_mean_delta, zero_method="wilcox", method="auto")
        inference.append(
            {
                "metric": metric,
                "delta": float(delta.mean()),
                "ci_low": low,
                "ci_high": high,
                "wilcoxon_p": float(wilcoxon.pvalue),
                "n_cases": int(delta.shape[0]),
                **subject_cluster_inference(delta, expected_index),
            }
        )

    original_stats = pd.read_csv(root / "paper/results/r11nr_final_statistics.csv")
    original = {"R11": {}, "R11NR": {}, "U-Net++": {}}
    for metric in ("dice", "iou"):
        row = original_stats[
            (original_stats["dataset"] == "COVID-19")
            & (original_stats["metric"] == metric)
        ]
        if len(row) != 1:
            raise ValueError(f"missing original COVID-19 {metric} R11/R11NR result")
        original["R11"][metric] = float(row.iloc[0]["r11_mean"])
        original["R11NR"][metric] = float(row.iloc[0]["r11nr_mean"])

    original_baseline = pd.read_csv(root / "paper/results/protocol_v3_image_baseline_aggregate.csv")
    for metric in ("dice", "iou"):
        row = original_baseline[
            (original_baseline["model"] == "unetplusplus")
            & (original_baseline["dataset"] == "medclipseg_covid19")
            & (original_baseline["metric"] == metric)
        ]
        if len(row) != 1:
            raise ValueError(f"missing original COVID-19 {metric} U-Net++ result")
        original["U-Net++"][metric] = float(row.iloc[0]["mean"])
    comparisons = []
    for row in aggregate.to_dict(orient="records"):
        reference = original[row["method"]]
        comparisons.append(
            {
                "method": row["method"],
                "groupval_dice": row["dice_mean"],
                "original_dice": reference["dice"],
                "dice_change": row["dice_mean"] - reference["dice"],
                "groupval_iou": row["iou_mean"],
                "original_iou": reference["iou"],
                "iou_change": row["iou_mean"] - reference["iou"],
            }
        )

    seeds.to_csv(output / "seed_metrics.csv", index=False, float_format="%.8g")
    aggregate.to_csv(output / "aggregate.csv", index=False, float_format="%.8g")
    pd.DataFrame(inference).to_csv(
        output / "r11_vs_r11nr_statistics.csv", index=False, float_format="%.8g"
    )
    pd.DataFrame(comparisons).to_csv(
        output / "comparison_with_original_protocol.csv", index=False, float_format="%.8g"
    )
    payload = {
        "protocol_hash": protocol_hash,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "aggregate": aggregate.to_dict(orient="records"),
        "r11_vs_r11nr": inference,
        "comparison_with_original_protocol": comparisons,
    }
    (output / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
