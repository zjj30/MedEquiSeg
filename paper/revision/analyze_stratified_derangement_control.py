#!/usr/bin/env python3
"""Case-first analysis of true versus presence/class-stratified prompt derangement."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


PROTOCOL_HASH = "abb8ccb2d46cf039a4a0c573d733e3bc510636e6dfa7e59032508ab96fc0f718"
RUN_ID = "V3_ABL_EQUIPROMPT"
DATASETS = (
    "medclipseg_busi",
    "medclipseg_clinicdb",
    "medclipseg_busbra",
    "medclipseg_brisc",
    "medclipseg_covid19",
)
SEEDS = (123, 456, 789)
METRICS = ("dice", "iou")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--bootstrap-reps", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=12345)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def holm_adjust(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues), dtype=np.float64)
    running = 0.0
    count = len(pvalues)
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * pvalues[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def bootstrap_mean(delta: np.ndarray, reps: int, rng: np.random.Generator) -> np.ndarray:
    output = np.empty(reps, dtype=np.float64)
    chunk_size = 256
    for start in range(0, reps, chunk_size):
        stop = min(reps, start + chunk_size)
        indices = rng.integers(0, len(delta), size=(stop - start, len(delta)))
        output[start:stop] = delta[indices].mean(axis=1)
    return output


def paired_pvalues(delta: np.ndarray) -> tuple[float, float]:
    try:
        wilcoxon = float(stats.wilcoxon(delta, zero_method="wilcox", alternative="two-sided").pvalue)
    except ValueError:
        wilcoxon = 1.0
    return wilcoxon, float(stats.ttest_1samp(delta, 0.0).pvalue)


def load_values(root: Path):
    log_root = root / "logs" / "protocol_v3" / PROTOCOL_HASH
    result_root = root / "paper" / "results" / "protocol_v3_stratified_derangement_20260719"
    values = {
        "true": defaultdict(lambda: defaultdict(lambda: defaultdict(list))),
        "stratified_derangement": defaultdict(lambda: defaultdict(lambda: defaultdict(list))),
    }
    maps: dict[str, dict[str, dict[str, str]]] = {}
    inputs: list[dict[str, str]] = []
    seed_summary: list[dict[str, object]] = []

    for dataset in DATASETS:
        map_path = result_root / "control_maps" / f"{dataset}_stratified_derangement_v1.csv"
        map_rows = read_csv(map_path)
        maps[dataset] = {row["case_id"]: row for row in map_rows}
        inputs.append({"kind": "control_map", "path": str(map_path), "sha256": sha256(map_path)})
        expected_cases: set[str] | None = None
        for seed in SEEDS:
            paths = {
                "true": log_root / RUN_ID / dataset / f"seed{seed}" / "controls" / "true" / "per_case.csv",
                "stratified_derangement": result_root / "runs" / dataset / f"seed{seed}" / "per_case.csv",
            }
            seed_rows: dict[str, dict[str, dict[str, str]]] = {}
            for control, path in paths.items():
                rows = read_csv(path)
                by_case = {row["case_id"]: row for row in rows}
                if len(by_case) != len(rows):
                    raise ValueError(f"Duplicate case_id in {path}")
                if any(row["protocol_hash"] != PROTOCOL_HASH for row in rows):
                    raise ValueError(f"Protocol mismatch in {path}")
                seed_rows[control] = by_case
                inputs.append({"kind": control, "path": str(path), "sha256": sha256(path)})
            case_ids = set(seed_rows["true"])
            if case_ids != set(seed_rows["stratified_derangement"]):
                raise ValueError(f"Unpaired true/control cases for {dataset}/seed{seed}")
            if case_ids != set(maps[dataset]):
                raise ValueError(f"Control-map closure mismatch for {dataset}/seed{seed}")
            if expected_cases is None:
                expected_cases = case_ids
            elif case_ids != expected_cases:
                raise ValueError(f"Seed case-set mismatch for {dataset}/seed{seed}")
            true_checkpoints = {row["checkpoint_sha256"] for row in seed_rows["true"].values()}
            control_checkpoints = {
                row["checkpoint_sha256"] for row in seed_rows["stratified_derangement"].values()
            }
            if true_checkpoints != control_checkpoints or len(true_checkpoints) != 1:
                raise ValueError(f"Checkpoint mismatch for {dataset}/seed{seed}")

            for control in values:
                for case_id, row in seed_rows[control].items():
                    for metric in METRICS:
                        values[control][dataset][case_id][metric].append(float(row[metric]))
                seed_summary.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "control": control,
                        "n_cases": len(case_ids),
                        "dice_mean": float(np.mean([float(row["dice"]) for row in seed_rows[control].values()])),
                        "iou_mean": float(np.mean([float(row["iou"]) for row in seed_rows[control].values()])),
                        "checkpoint_sha256": next(iter(true_checkpoints)),
                    }
                )
    return values, maps, inputs, seed_summary


def analyze_scope(
    values,
    maps,
    scope: str,
    reps: int,
    bootstrap_seed: int,
) -> tuple[list[dict[str, object]], dict[str, dict[str, np.ndarray]]]:
    output: list[dict[str, object]] = []
    stored: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    for metric_offset, metric in enumerate(METRICS):
        rng = np.random.default_rng(bootstrap_seed + metric_offset + (100 if scope == "text_changed" else 0))
        dataset_rows: list[dict[str, object]] = []
        wilcoxon_pvalues: list[float] = []
        paired_t_pvalues: list[float] = []
        for dataset in DATASETS:
            case_ids = sorted(values["true"][dataset])
            if scope == "text_changed":
                case_ids = [case_id for case_id in case_ids if maps[dataset][case_id]["text_changed"] == "1"]
            delta = np.asarray(
                [
                    np.mean(values["true"][dataset][case_id][metric])
                    - np.mean(values["stratified_derangement"][dataset][case_id][metric])
                    for case_id in case_ids
                ],
                dtype=np.float64,
            )
            if not len(delta):
                raise ValueError(f"Empty analysis scope {scope}/{dataset}")
            bootstrap = bootstrap_mean(delta, reps, rng)
            wilcoxon_p, paired_t_p = paired_pvalues(delta)
            wilcoxon_pvalues.append(wilcoxon_p)
            paired_t_pvalues.append(paired_t_p)
            stored[metric][dataset] = delta
            dataset_rows.append(
                {
                    "analysis_scope": scope,
                    "scope": "dataset",
                    "dataset": dataset,
                    "metric": metric,
                    "n_cases": len(delta),
                    "n_seeds": len(SEEDS),
                    "delta_true_minus_control": float(delta.mean()),
                    "ci_low": float(np.percentile(bootstrap, 2.5)),
                    "ci_high": float(np.percentile(bootstrap, 97.5)),
                    "wilcoxon_p": wilcoxon_p,
                    "paired_t_p": paired_t_p,
                }
            )
        for row, p_w, p_t in zip(dataset_rows, holm_adjust(wilcoxon_pvalues), holm_adjust(paired_t_pvalues)):
            row["wilcoxon_p_holm"] = p_w
            row["paired_t_p_holm"] = p_t
        output.extend(dataset_rows)

        macro_bootstrap = np.empty(reps, dtype=np.float64)
        chunk_size = 256
        for start in range(0, reps, chunk_size):
            stop = min(reps, start + chunk_size)
            sampled_means = np.empty((stop - start, len(DATASETS)), dtype=np.float64)
            for column, dataset in enumerate(DATASETS):
                delta = stored[metric][dataset]
                indices = rng.integers(0, len(delta), size=(stop - start, len(delta)))
                sampled_means[:, column] = delta[indices].mean(axis=1)
            macro_bootstrap[start:stop] = sampled_means.mean(axis=1)
        macro_delta = float(np.mean([stored[metric][dataset].mean() for dataset in DATASETS]))
        output.append(
            {
                "analysis_scope": scope,
                "scope": "dataset_macro",
                "dataset": "Public-5",
                "metric": metric,
                "n_cases": int(sum(len(stored[metric][dataset]) for dataset in DATASETS)),
                "n_seeds": len(SEEDS),
                "delta_true_minus_control": macro_delta,
                "ci_low": float(np.percentile(macro_bootstrap, 2.5)),
                "ci_high": float(np.percentile(macro_bootstrap, 97.5)),
                "wilcoxon_p": "",
                "paired_t_p": "",
                "wilcoxon_p_holm": "",
                "paired_t_p_holm": "",
            }
        )
    return output, stored


def stratum_analysis(values, maps, reps: int, bootstrap_seed: int) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    rng = np.random.default_rng(bootstrap_seed + 999)
    for dataset in DATASETS:
        grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
        for case_id, row in maps[dataset].items():
            grouped[(row["presence_stratum"], row["class_stratum"])].append(case_id)
        for (presence, class_label), case_ids in sorted(grouped.items()):
            delta = np.asarray(
                [
                    np.mean(values["true"][dataset][case_id]["dice"])
                    - np.mean(values["stratified_derangement"][dataset][case_id]["dice"])
                    for case_id in sorted(case_ids)
                ],
                dtype=np.float64,
            )
            bootstrap = bootstrap_mean(delta, reps, rng)
            output.append(
                {
                    "dataset": dataset,
                    "presence_stratum": presence,
                    "class_stratum": class_label,
                    "n_cases": len(delta),
                    "n_text_changed": sum(maps[dataset][case_id]["text_changed"] == "1" for case_id in case_ids),
                    "dice_delta_true_minus_control": float(delta.mean()),
                    "ci_low": float(np.percentile(bootstrap, 2.5)),
                    "ci_high": float(np.percentile(bootstrap, 97.5)),
                }
            )
    return output


def covid_subject_analysis(values, reps: int, bootstrap_seed: int) -> list[dict[str, object]]:
    dataset = "medclipseg_covid19"
    output: list[dict[str, object]] = []
    for metric_offset, metric in enumerate(METRICS):
        by_subject: dict[str, list[float]] = defaultdict(list)
        for case_id in sorted(values["true"][dataset]):
            match = re.search(r"sub-([^_]+)", case_id)
            if match is None:
                raise ValueError(f"Cannot recover COVID subject id: {case_id}")
            delta = float(
                np.mean(values["true"][dataset][case_id][metric])
                - np.mean(values["stratified_derangement"][dataset][case_id][metric])
            )
            by_subject[match.group(1)].append(delta)
        subject_delta = np.asarray(
            [np.mean(by_subject[subject]) for subject in sorted(by_subject)], dtype=np.float64
        )
        rng = np.random.default_rng(bootstrap_seed + 2000 + metric_offset)
        bootstrap = bootstrap_mean(subject_delta, reps, rng)
        wilcoxon_p, paired_t_p = paired_pvalues(subject_delta)
        output.append(
            {
                "dataset": dataset,
                "metric": metric,
                "n_cases": sum(len(rows) for rows in by_subject.values()),
                "n_subjects": len(by_subject),
                "equal_subject_delta_true_minus_control": float(subject_delta.mean()),
                "cluster_ci_low": float(np.percentile(bootstrap, 2.5)),
                "cluster_ci_high": float(np.percentile(bootstrap, 97.5)),
                "subject_wilcoxon_p": wilcoxon_p,
                "subject_paired_t_p": paired_t_p,
            }
        )
    return output


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    result_root = root / "paper" / "results" / "protocol_v3_stratified_derangement_20260719"
    values, maps, inputs, seed_summary = load_values(root)
    all_rows, _ = analyze_scope(values, maps, "all_cases", args.bootstrap_reps, args.bootstrap_seed)
    changed_rows, _ = analyze_scope(values, maps, "text_changed", args.bootstrap_reps, args.bootstrap_seed)
    stratum_rows = stratum_analysis(values, maps, args.bootstrap_reps, args.bootstrap_seed)
    covid_subject_rows = covid_subject_analysis(values, args.bootstrap_reps, args.bootstrap_seed)

    all_path = result_root / "casefirst_true_vs_stratified_derangement.csv"
    changed_path = result_root / "casefirst_true_vs_stratified_derangement_changed_only.csv"
    stratum_path = result_root / "stratum_dice_statistics.csv"
    seed_path = result_root / "seed_summary.csv"
    covid_subject_path = result_root / "covid_subject_cluster_statistics.csv"
    write_csv(all_path, all_rows)
    write_csv(changed_path, changed_rows)
    write_csv(stratum_path, stratum_rows)
    write_csv(seed_path, seed_summary)
    write_csv(covid_subject_path, covid_subject_rows)
    metadata = {
        "status": "PASS",
        "analysis": "case-first three-seed mean; paired true minus presence/class-stratified prompt derangement",
        "protocol_hash": PROTOCOL_HASH,
        "bootstrap_repetitions": args.bootstrap_reps,
        "bootstrap_seed": args.bootstrap_seed,
        "analysis_code_sha256": sha256(Path(__file__)),
        "matrix_audit_sha256": sha256(result_root / "matrix_audit.json"),
        "inputs": inputs,
        "outputs": {
            str(path.name): sha256(path)
            for path in (all_path, changed_path, stratum_path, seed_path, covid_subject_path)
        },
    }
    meta_path = result_root / "analysis_meta.json"
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    for rows in (all_rows, changed_rows):
        for row in rows:
            if row["scope"] == "dataset_macro":
                print(
                    f"{row['analysis_scope']} {row['metric']} "
                    f"delta={row['delta_true_minus_control']:.9f} "
                    f"ci=[{row['ci_low']:.9f},{row['ci_high']:.9f}]"
                )
    print(json.dumps({"status": "PASS", "inputs": len(inputs), "output": str(result_root)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PROJECT_ROOT = Path(__file__).resolve().parents[2]
