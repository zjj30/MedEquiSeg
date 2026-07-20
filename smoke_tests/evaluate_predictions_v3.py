#!/usr/bin/env python3
"""Canonical Protocol V3 evaluation on original-resolution prediction masks."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from paper_metrics import protocol_v3_metrics
from protocol_v3.core import binarize_mask, protocol_sha256


METRICS = ("dice", "iou", "nsd", "hd95", "assd")
OVERLAP_METRICS = ("dice", "iou", "nsd")
DISTANCE_METRICS = ("hd95", "assd")


def fmt(value: float) -> str:
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    if math.isnan(value):
        return "nan"
    return f"{value:.6f}"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def summarize_items(items: list[dict[str, float | int | str]], scope: str, dataset: str) -> dict[str, str]:
    summary: dict[str, str] = {
        "scope": scope,
        "dataset": dataset,
        "samples": str(len(items)),
        "both_empty_count": str(sum(int(item["both_empty"]) for item in items)),
        "one_empty_failure_count": str(sum(int(item["one_empty_failure"]) for item in items)),
    }
    for metric in OVERLAP_METRICS:
        values = np.asarray([float(item[metric]) for item in items], dtype=np.float64)
        summary[f"{metric}_mean"] = fmt(float(values.mean()))
        summary[f"{metric}_std"] = fmt(float(values.std(ddof=1)) if len(values) > 1 else 0.0)
    for metric in DISTANCE_METRICS:
        values = np.asarray([float(item[metric]) for item in items], dtype=np.float64)
        finite = values[np.isfinite(values)]
        inf_count = int(np.isinf(values).sum())
        summary[f"{metric}_mean"] = "inf" if inf_count else fmt(float(values.mean()))
        summary[f"{metric}_std"] = "inf" if inf_count else fmt(float(values.std(ddof=1)) if len(values) > 1 else 0.0)
        summary[f"{metric}_finite_mean"] = fmt(float(finite.mean())) if finite.size else "nan"
        summary[f"{metric}_finite_std"] = fmt(float(finite.std(ddof=1)) if finite.size > 1 else 0.0) if finite.size else "nan"
        summary[f"{metric}_inf_count"] = str(inf_count)
    summary["one_empty_failure_rate"] = fmt(
        float(int(summary["one_empty_failure_count"]) / len(items)) if items else float("nan")
    )
    return summary


def dataset_macro(dataset_summaries: list[dict[str, str]]) -> dict[str, str]:
    row: dict[str, str] = {
        "scope": "dataset_macro",
        "dataset": "MACRO",
        "samples": str(len(dataset_summaries)),
        "both_empty_count": str(sum(int(item["both_empty_count"]) for item in dataset_summaries)),
        "one_empty_failure_count": str(sum(int(item["one_empty_failure_count"]) for item in dataset_summaries)),
    }
    for metric in OVERLAP_METRICS:
        values = np.asarray([float(item[f"{metric}_mean"]) for item in dataset_summaries])
        row[f"{metric}_mean"] = fmt(float(values.mean()))
        row[f"{metric}_std"] = fmt(float(values.std(ddof=1)) if len(values) > 1 else 0.0)
    for metric in DISTANCE_METRICS:
        means = np.asarray([float(item[f"{metric}_mean"]) for item in dataset_summaries])
        finite_means = np.asarray([float(item[f"{metric}_finite_mean"]) for item in dataset_summaries])
        inf_count = int(np.isinf(means).sum())
        row[f"{metric}_mean"] = "inf" if inf_count else fmt(float(means.mean()))
        row[f"{metric}_std"] = "inf" if inf_count else fmt(float(means.std(ddof=1)) if len(means) > 1 else 0.0)
        row[f"{metric}_finite_mean"] = fmt(float(np.nanmean(finite_means)))
        row[f"{metric}_finite_std"] = fmt(float(np.nanstd(finite_means, ddof=1)) if len(finite_means) > 1 else 0.0)
        row[f"{metric}_inf_count"] = str(sum(int(item[f"{metric}_inf_count"]) for item in dataset_summaries))
    total = sum(int(item["samples"]) for item in dataset_summaries)
    row["one_empty_failure_rate"] = fmt(int(row["one_empty_failure_count"]) / total if total else float("nan"))
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-index", type=Path, required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--per-case-csv", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--nsd-tolerance-px", type=float, default=2.0)
    args = parser.parse_args()

    expected_protocol = protocol_sha256(args.protocol_lock)
    index_rows = read_csv(args.prediction_index)
    if not index_rows:
        raise ValueError("Prediction index is empty")
    per_case: list[dict[str, float | int | str]] = []
    for row in index_rows:
        if row["protocol_hash"] != expected_protocol:
            raise ValueError(f"Prediction protocol mismatch for {row['case_id']}")
        with Image.open(row["prediction_path"]) as image:
            pred = np.asarray(image.convert("L")) >= 127
        with Image.open(row["mask_path"]) as image:
            target = binarize_mask(np.asarray(image.convert("L")), row["mask_mode"]).astype(bool)
        if pred.shape != target.shape:
            raise ValueError(f"Prediction/GT shape mismatch for {row['case_id']}: {pred.shape} != {target.shape}")
        values = protocol_v3_metrics(pred, target, nsd_tolerance=args.nsd_tolerance_px).as_dict()
        per_case.append(
            {
                "dataset": row["dataset"],
                "case_id": row["case_id"],
                **values,
                "prediction_path": row["prediction_path"],
                "mask_path": row["mask_path"],
                "prompt_control": row.get("prompt_control", "true"),
                "checkpoint_sha256": row["checkpoint_sha256"],
                "protocol_hash": row["protocol_hash"],
            }
        )

    args.per_case_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.per_case_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_case[0]))
        writer.writeheader()
        for row in per_case:
            writer.writerow({key: fmt(value) if isinstance(value, float) else value for key, value in row.items()})

    grouped: dict[str, list[dict[str, float | int | str]]] = defaultdict(list)
    for row in per_case:
        grouped[str(row["dataset"])].append(row)
    dataset_rows = [summarize_items(items, "dataset", dataset) for dataset, items in sorted(grouped.items())]
    case_weighted = summarize_items(per_case, "case_weighted_overall", "ALL_CASES")
    summary_rows = dataset_rows + [case_weighted, dataset_macro(dataset_rows)]
    fields = list(summary_rows[0])
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"per_case_csv={args.per_case_csv}")
    print(f"summary_csv={args.summary_csv}")
    print(f"cases={len(per_case)} protocol_hash={expected_protocol}")
    print("final_status: PASS")


if __name__ == "__main__":
    main()
