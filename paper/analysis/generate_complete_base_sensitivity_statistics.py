#!/usr/bin/env python3
"""Generate no-retraining sensitivity statistics for Complete minus Base.

The primary manuscript interval averages the three seed-paired case values before
resampling cases.  This script adds two explicitly secondary checks:

1. a hierarchical bootstrap that resamples paired test cases and the three
   training-seed pairs; and
2. the same analyses after excluding cases whose dataset-provided reference mask
   is empty.

All comparisons use existing per-case exports.  No checkpoint is retrained.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


PROTOCOL_HASH = "abb8ccb2d46cf039a4a0c573d733e3bc510636e6dfa7e59032508ab96fc0f718"
SEEDS = (123, 456, 789)
DATASETS = (
    ("BUSI", "medclipseg_busi"),
    ("ClinicDB", "medclipseg_clinicdb"),
    ("BUS-BRA", "medclipseg_busbra"),
    ("BRISC", "medclipseg_brisc"),
    ("COVID-19", "medclipseg_covid19"),
)
COMPLETE_RUN = "V3_ABL_EQUIPROMPT"
BASE_RUN = "V3_ABL_BASE"
METRICS = ("dice", "iou")
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20260725
ESTABLISHED_ALL_CASE_SEED = 20260718


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


def reference_mask_is_nonempty(path: str) -> bool:
    mask_path = Path(path)
    if not mask_path.is_file():
        raise FileNotFoundError(f"Reference mask not found: {mask_path}")
    if mask_path.suffix.lower() == ".npy":
        values = np.load(mask_path, mmap_mode="r")
    else:
        with Image.open(mask_path) as image:
            values = np.asarray(image)
    return bool(np.any(values > 0))


def load_dataset(root: Path, dataset: str) -> tuple[list[str], np.ndarray, dict[str, np.ndarray]]:
    expected_ids: list[str] | None = None
    expected_masks: list[str] | None = None
    columns: dict[str, list[np.ndarray]] = {metric: [] for metric in METRICS}

    for seed in SEEDS:
        frames: dict[str, pd.DataFrame] = {}
        for run_id in (COMPLETE_RUN, BASE_RUN):
            path = per_case_path(root, run_id, dataset, seed)
            frame = pd.read_csv(
                path,
                usecols=[
                    "case_id",
                    "mask_path",
                    "prompt_control",
                    "protocol_hash",
                    *METRICS,
                ],
            ).sort_values("case_id", kind="stable")
            if frame["case_id"].duplicated().any():
                raise ValueError(f"Duplicate case_id in {path}")
            if set(frame["protocol_hash"].astype(str)) != {PROTOCOL_HASH}:
                raise ValueError(f"Protocol hash mismatch in {path}")
            if set(frame["prompt_control"].astype(str).str.lower()) != {"true"}:
                raise ValueError(f"Expected true-prompt export in {path}")
            frames[run_id] = frame.reset_index(drop=True)

        complete = frames[COMPLETE_RUN]
        base = frames[BASE_RUN]
        case_ids = complete["case_id"].astype(str).tolist()
        masks = complete["mask_path"].astype(str).tolist()
        if case_ids != base["case_id"].astype(str).tolist():
            raise ValueError(f"Complete/Base case mismatch for {dataset}, seed {seed}")
        if masks != base["mask_path"].astype(str).tolist():
            raise ValueError(f"Complete/Base mask-path mismatch for {dataset}, seed {seed}")
        if expected_ids is None:
            expected_ids = case_ids
            expected_masks = masks
        elif case_ids != expected_ids or masks != expected_masks:
            raise ValueError(f"Seed-level case or mask mismatch for {dataset}")

        for metric in METRICS:
            columns[metric].append(
                complete[metric].to_numpy(dtype=float) - base[metric].to_numpy(dtype=float)
            )

    assert expected_ids is not None and expected_masks is not None
    nonempty = np.asarray(
        [reference_mask_is_nonempty(path) for path in expected_masks], dtype=bool
    )
    matrices = {metric: np.column_stack(values) for metric, values in columns.items()}
    return expected_ids, nonempty, matrices


def bootstrap_dataset(
    values: np.ndarray,
    rng: np.random.Generator,
    *,
    hierarchical: bool,
) -> np.ndarray:
    """Bootstrap one dataset while preserving Complete/Base pairing."""

    n_cases, n_seeds = values.shape
    estimates = np.empty(BOOTSTRAP_SAMPLES, dtype=float)
    chunk_size = 256
    for start in range(0, BOOTSTRAP_SAMPLES, chunk_size):
        stop = min(start + chunk_size, BOOTSTRAP_SAMPLES)
        size = stop - start
        cases = rng.integers(0, n_cases, size=(size, n_cases))
        sampled = values[cases]
        if hierarchical:
            seeds = rng.integers(0, n_seeds, size=(size, n_seeds))
            seeds = np.broadcast_to(seeds[:, None, :], sampled.shape)
            sampled = np.take_along_axis(sampled, seeds, axis=2)
            estimates[start:stop] = sampled.mean(axis=(1, 2))
        else:
            estimates[start:stop] = sampled.mean(axis=2).mean(axis=1)
    return estimates


def established_all_case_hierarchical_bootstraps(
    loaded: dict[str, dict[str, object]],
) -> dict[tuple[str, str], np.ndarray]:
    """Reproduce the previously archived all-case hierarchical intervals exactly.

    Draws intentionally follow the scalar loop and RNG order used by
    generate_medequiseg_base_contrast_statistics.py.  This prevents a harmless
    Monte Carlo seed/order difference from creating two reported all-case CIs.
    """

    rng = np.random.default_rng(ESTABLISHED_ALL_CASE_SEED)
    estimates: dict[tuple[str, str], np.ndarray] = {}
    for display, _ in DATASETS:
        matrices = loaded[display]["matrices"]
        assert isinstance(matrices, dict)
        for metric in METRICS:
            values = matrices[metric]
            n_cases, n_seeds = values.shape
            draws = np.empty(BOOTSTRAP_SAMPLES, dtype=float)
            for index in range(BOOTSTRAP_SAMPLES):
                cases = rng.integers(0, n_cases, n_cases)
                seeds = rng.integers(0, n_seeds, n_seeds)
                draws[index] = values[np.ix_(cases, seeds)].mean()
            estimates[(display, metric)] = draws

    for metric in METRICS:
        draws = np.empty(BOOTSTRAP_SAMPLES, dtype=float)
        for index in range(BOOTSTRAP_SAMPLES):
            effects = []
            for display, _ in DATASETS:
                matrices = loaded[display]["matrices"]
                assert isinstance(matrices, dict)
                values = matrices[metric]
                n_cases, n_seeds = values.shape
                cases = rng.integers(0, n_cases, n_cases)
                seeds = rng.integers(0, n_seeds, n_seeds)
                effects.append(float(values[np.ix_(cases, seeds)].mean()))
            draws[index] = np.mean(effects)
        estimates[("Public-5 macro", metric)] = draws
    return estimates


def ci(values: np.ndarray) -> tuple[float, float]:
    low, high = np.quantile(values, (0.025, 0.975))
    return float(low), float(high)


def latex_interval(low: float, high: float) -> str:
    return f"[{100 * low:+.2f}, {100 * high:+.2f}]"


def write_latex_table(frame: pd.DataFrame, path: Path) -> None:
    macro = frame[frame["dataset"] == "Public-5 macro"].copy()
    lookup = {(row.scope, row.metric): row for row in macro.itertuples(index=False)}
    lines = [
        "% Auto-generated by generate_complete_base_sensitivity_statistics.py.",
        "\\begin{table}[!htbp]",
        "\\centering",
        "\\caption{Secondary complete-minus-Base sensitivity analyses}",
        "\\label{tab:complete_base_sensitivity}",
        "\\small",
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "Analysis set & Dice, pp (95\\% CI) & IoU, pp (95\\% CI) \\\\",
        "\\midrule",
    ]
    for scope, label in (
        ("all_cases", "All test cases"),
        ("nonempty_reference", "Non-empty reference masks"),
    ):
        dice = lookup[(scope, "dice")]
        iou = lookup[(scope, "iou")]
        lines.append(
            f"{label} & {100 * dice.delta:+.2f} {latex_interval(dice.seed_case_ci_low, dice.seed_case_ci_high)} "
            f"& {100 * iou.delta:+.2f} {latex_interval(iou.seed_case_ci_low, iou.seed_case_ci_high)} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\begin{minipage}{0.98\\linewidth}",
            "\\footnotesize Differences are complete configuration minus Base and use an equal-dataset macro-average. Intervals are secondary hierarchical paired bootstraps (10,000 replicates) that resample test cases within dataset and the three training-seed pairs. Empty status is determined from the dataset-provided reference mask. These analyses reuse existing predictions and do not add independent training seeds or retrain any model.",
            "\\end{minipage}",
            "\\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper/results/medequiseg_complete_base_sensitivity_20260725"),
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    loaded: dict[str, dict[str, object]] = {}
    for display, dataset in DATASETS:
        case_ids, nonempty, matrices = load_dataset(root, dataset)
        loaded[display] = {
            "case_ids": case_ids,
            "nonempty": nonempty,
            "matrices": matrices,
        }
    established_all_case = established_all_case_hierarchical_bootstraps(loaded)

    rows: list[dict[str, float | int | str]] = []
    bootstrap_cache: dict[tuple[str, str, str, str], np.ndarray] = {}
    scopes = {
        "all_cases": lambda mask: np.ones(mask.shape, dtype=bool),
        "nonempty_reference": lambda mask: mask,
    }

    for scope, selector in scopes.items():
        for metric in METRICS:
            for dataset_index, (display, _) in enumerate(DATASETS):
                mask = loaded[display]["nonempty"]
                assert isinstance(mask, np.ndarray)
                selected = selector(mask)
                matrices = loaded[display]["matrices"]
                assert isinstance(matrices, dict)
                values = matrices[metric][selected]
                if not len(values):
                    raise ValueError(f"No cases remain for {display}, {scope}")
                case_rng = np.random.default_rng(
                    BOOTSTRAP_SEED + 10000 * list(scopes).index(scope) + 1000 * METRICS.index(metric) + dataset_index
                )
                hierarchical_rng = np.random.default_rng(
                    BOOTSTRAP_SEED + 50000 + 10000 * list(scopes).index(scope) + 1000 * METRICS.index(metric) + dataset_index
                )
                case_estimates = bootstrap_dataset(values, case_rng, hierarchical=False)
                if scope == "all_cases":
                    hierarchical_estimates = established_all_case[(display, metric)]
                else:
                    hierarchical_estimates = bootstrap_dataset(
                        values, hierarchical_rng, hierarchical=True
                    )
                bootstrap_cache[(scope, metric, display, "case_first")] = case_estimates
                bootstrap_cache[(scope, metric, display, "seed_case")] = hierarchical_estimates
                case_low, case_high = ci(case_estimates)
                hierarchical_low, hierarchical_high = ci(hierarchical_estimates)
                rows.append(
                    {
                        "dataset": display,
                        "scope": scope,
                        "metric": metric,
                        "n_cases": int(values.shape[0]),
                        "delta": float(values.mean()),
                        "case_first_ci_low": case_low,
                        "case_first_ci_high": case_high,
                        "seed_case_ci_low": hierarchical_low,
                        "seed_case_ci_high": hierarchical_high,
                    }
                )

            case_macro = np.mean(
                [
                    bootstrap_cache[(scope, metric, display, "case_first")]
                    for display, _ in DATASETS
                ],
                axis=0,
            )
            if scope == "all_cases":
                hierarchical_macro = established_all_case[("Public-5 macro", metric)]
            else:
                hierarchical_macro = np.mean(
                    [
                        bootstrap_cache[(scope, metric, display, "seed_case")]
                        for display, _ in DATASETS
                    ],
                    axis=0,
                )
            point_values = []
            total_cases = 0
            for display, _ in DATASETS:
                mask = loaded[display]["nonempty"]
                matrices = loaded[display]["matrices"]
                assert isinstance(mask, np.ndarray) and isinstance(matrices, dict)
                values = matrices[metric][selector(mask)]
                point_values.append(float(values.mean()))
                total_cases += int(values.shape[0])
            case_low, case_high = ci(case_macro)
            hierarchical_low, hierarchical_high = ci(hierarchical_macro)
            rows.append(
                {
                    "dataset": "Public-5 macro",
                    "scope": scope,
                    "metric": metric,
                    "n_cases": total_cases,
                    "delta": float(np.mean(point_values)),
                    "case_first_ci_low": case_low,
                    "case_first_ci_high": case_high,
                    "seed_case_ci_low": hierarchical_low,
                    "seed_case_ci_high": hierarchical_high,
                }
            )

    frame = pd.DataFrame(rows)
    csv_path = output_dir / "complete_base_sensitivity.csv"
    frame.to_csv(csv_path, index=False, float_format="%.10g")
    payload = {
        "complete_run": COMPLETE_RUN,
        "base_run": BASE_RUN,
        "protocol_hash": PROTOCOL_HASH,
        "training_seeds": list(SEEDS),
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "case_first_and_nonempty_seed_base": BOOTSTRAP_SEED,
        "established_all_case_hierarchical_seed": ESTABLISHED_ALL_CASE_SEED,
        "analysis_status": "secondary post hoc sensitivity analysis; existing predictions only; no retraining",
        "case_first_resampling": "average the three paired seed values per case, then resample paired cases within dataset",
        "seed_case_resampling": "resample paired test cases within dataset and resample the three paired training-seed indices",
        "macro_weighting": "equal weight for each of the five datasets",
        "empty_case_definition": "reference mask contains no pixel value greater than zero",
        "rows": frame.to_dict(orient="records"),
    }
    (output_dir / "complete_base_sensitivity.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    write_latex_table(frame, output_dir / "complete_base_sensitivity_table.tex")
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
