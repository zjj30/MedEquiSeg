#!/usr/bin/env python3
"""Generate subject-clustered COVID-19 sensitivity statistics for Protocol V3."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


PUBLIC_HASH = "abb8ccb2d46cf039a4a0c573d733e3bc510636e6dfa7e59032508ab96fc0f718"
SEEDS = (123, 456, 789)
BOOTSTRAP_SEED = 20260715
BOOTSTRAP_SAMPLES = 10_000
FULL_RUN_ID = "V3_ABL_EQUIPROMPT"
NO_REWRITE_RUN_ID = "V3_ABL_SHARED_NR"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper/results/protocol_v3_covid_subject_cluster_20260718"),
    )
    return parser.parse_args()


def multimodal_path(root: Path, run_id: str, seed: int, control: str) -> Path:
    return (
        root
        / "logs/protocol_v3"
        / PUBLIC_HASH
        / run_id
        / "medclipseg_covid19"
        / f"seed{seed}"
        / "controls"
        / control
        / "per_case.csv"
    )


def unetpp_path(root: Path, seed: int) -> Path:
    return (
        root
        / "logs/protocol_v3_image_baselines/unetplusplus/medclipseg_covid19"
        / f"seed{seed}/controls/true/per_case.csv"
    )


def load_metric(path: Path, metric: str) -> pd.Series:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, usecols=["case_id", metric])
    if frame["case_id"].duplicated().any():
        raise ValueError(f"duplicate case_id in {path}")
    return frame.set_index("case_id")[metric].sort_index()


def recover_subjects(case_ids: pd.Index) -> np.ndarray:
    subjects = []
    for case_id in case_ids.astype(str):
        match = re.search(r"sub-([^_]+)", case_id)
        if not match:
            raise ValueError(f"cannot recover subject from case_id: {case_id}")
        subjects.append(match.group(1))
    return np.asarray(subjects, dtype=object)


def comparison_matrix(
    root: Path, comparison: str, metric: str
) -> tuple[pd.Index, np.ndarray, list[str]]:
    left: list[np.ndarray] = []
    right: list[np.ndarray] = []
    sources: list[str] = []
    expected_index: pd.Index | None = None
    for seed in SEEDS:
        if comparison == "R11_minus_R11NR":
            left_path = multimodal_path(root, FULL_RUN_ID, seed, "true")
            right_path = multimodal_path(root, NO_REWRITE_RUN_ID, seed, "true")
        elif comparison == "R11_minus_UNetPP":
            left_path = multimodal_path(root, FULL_RUN_ID, seed, "true")
            right_path = unetpp_path(root, seed)
        elif comparison == "R11_true_minus_shuffled":
            left_path = multimodal_path(root, FULL_RUN_ID, seed, "true")
            right_path = multimodal_path(root, FULL_RUN_ID, seed, "shuffled")
        else:
            raise ValueError(f"unknown comparison: {comparison}")
        left_series = load_metric(left_path, metric)
        right_series = load_metric(right_path, metric)
        if not left_series.index.equals(right_series.index):
            raise ValueError(f"case mismatch for {comparison}/{metric}/seed{seed}")
        if expected_index is not None and not expected_index.equals(left_series.index):
            raise ValueError(f"case mismatch across seeds for {comparison}/{metric}")
        expected_index = left_series.index
        left.append(left_series.to_numpy(dtype=np.float64))
        right.append(right_series.to_numpy(dtype=np.float64))
        sources.extend(
            [
                left_path.relative_to(root).as_posix(),
                right_path.relative_to(root).as_posix(),
            ]
        )
    assert expected_index is not None
    return expected_index, np.column_stack(left) - np.column_stack(right), sources


def clustered_summary(
    comparison: str,
    metric: str,
    case_ids: pd.Index,
    delta: np.ndarray,
    sources: list[str],
) -> dict[str, object]:
    subjects = recover_subjects(case_ids)
    unique_subjects = np.unique(subjects)
    subject_indices = {
        subject: np.flatnonzero(subjects == subject) for subject in unique_subjects
    }
    case_delta = delta.mean(axis=1)
    subject_delta = np.asarray(
        [case_delta[indices].mean() for indices in subject_indices.values()],
        dtype=np.float64,
    )
    wilcoxon = stats.wilcoxon(subject_delta, zero_method="wilcox", method="auto")
    t_test = stats.ttest_1samp(subject_delta, popmean=0.0)

    rng_offset = sum(ord(character) for character in comparison + metric)
    rng = np.random.default_rng(BOOTSTRAP_SEED + rng_offset)
    estimates = np.empty(BOOTSTRAP_SAMPLES, dtype=np.float64)
    for index in range(BOOTSTRAP_SAMPLES):
        sampled_subjects = rng.choice(unique_subjects, size=len(unique_subjects), replace=True)
        sampled_cases = np.concatenate(
            [subject_indices[subject] for subject in sampled_subjects]
        )
        sampled_seeds = rng.integers(0, delta.shape[1], size=delta.shape[1])
        estimates[index] = delta[np.ix_(sampled_cases, sampled_seeds)].mean()
    low, high = np.quantile(estimates, (0.025, 0.975))
    return {
        "comparison": comparison,
        "metric": metric,
        "n_cases": int(len(case_ids)),
        "n_subjects": int(len(unique_subjects)),
        "image_weighted_delta": float(delta.mean()),
        "subject_weighted_delta": float(subject_delta.mean()),
        "cluster_bootstrap_ci_low": float(low),
        "cluster_bootstrap_ci_high": float(high),
        "subject_wilcoxon_p": float(wilcoxon.pvalue),
        "subject_paired_t_p": float(t_test.pvalue),
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED + rng_offset,
        "sources": sorted(set(sources)),
    }


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for comparison in (
        "R11_minus_R11NR",
        "R11_minus_UNetPP",
        "R11_true_minus_shuffled",
    ):
        for metric in ("dice", "iou"):
            case_ids, delta, sources = comparison_matrix(root, comparison, metric)
            rows.append(clustered_summary(comparison, metric, case_ids, delta, sources))

    frame = pd.DataFrame(rows)
    csv_frame = frame.drop(columns="sources")
    csv_frame.to_csv(
        output / "clustered_statistics.csv", index=False, float_format="%.8g", lineterminator="\n"
    )
    payload = {
        "protocol_hash": PUBLIC_HASH,
        "estimand_note": (
            "Image-weighted point differences retain the main evaluator estimand. "
            "Sensitivity intervals resample recovered subjects and training seeds; "
            "tests use equal-subject mean differences."
        ),
        "rows": rows,
    }
    (output / "clustered_statistics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    readme = [
        "# COVID-19 Subject-Cluster Sensitivity",
        "",
        "The Protocol V3 COVID-19 test set contains repeated images from recovered",
        "subject tokens. Main-table means remain image weighted. This sensitivity",
        "analysis resamples subjects and training seeds for confidence intervals and",
        "performs Wilcoxon/paired t-tests on equal-subject mean differences.",
        "",
        f"- Cases: {int(frame['n_cases'].iloc[0])}",
        f"- Recovered test subjects: {int(frame['n_subjects'].iloc[0])}",
        f"- Bootstrap replicates per comparison: {BOOTSTRAP_SAMPLES}",
        "",
        "See `clustered_statistics.csv` for compact results and",
        "`clustered_statistics.json` for source paths and bootstrap seeds.",
    ]
    (output / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(csv_frame.to_string(index=False))


if __name__ == "__main__":
    main()
