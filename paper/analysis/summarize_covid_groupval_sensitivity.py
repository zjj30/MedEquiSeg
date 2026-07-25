#!/usr/bin/env python3
"""Summarize the retained COVID-19 grouped train--validation sensitivity runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


SEEDS = (123, 456, 789)
MEDEQUISEG_RUN_ID = "V3_R11_COVID_GROUPVAL"


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
        default=Path("paper/results/protocol_v3_covid_groupval_public5_20260725"),
    )
    parser.add_argument(
        "--original-reference",
        type=Path,
        default=Path(
            "paper/results/protocol_v3_covid_groupval_public5_20260725/"
            "original_split_reference.csv"
        ),
    )
    return parser.parse_args()


def dataset_summary(path: Path) -> pd.Series:
    frame = pd.read_csv(path)
    row = frame[frame["scope"] == "dataset"]
    if len(row) != 1:
        raise ValueError(f"Expected one dataset-scope row in {path}, found {len(row)}")
    return row.iloc[0]


def resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    lock = resolve(root, args.protocol_lock)
    output = resolve(root, args.output_dir)
    original_reference = resolve(root, args.original_reference)
    output.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(root / "smoke_tests"))
    from protocol_v3.core import protocol_sha256

    protocol_hash = protocol_sha256(lock)
    seed_rows: list[dict[str, float | int | str]] = []
    for seed in SEEDS:
        path = (
            root
            / "logs/protocol_v3"
            / protocol_hash
            / MEDEQUISEG_RUN_ID
            / "medclipseg_covid19"
            / f"seed{seed}/controls/true/summary.csv"
        )
        row = dataset_summary(path)
        seed_rows.append(
            {
                "method": "MedEquiSeg",
                "seed": seed,
                "dice": float(row["dice_mean"]),
                "iou": float(row["iou_mean"]),
                "samples": int(row["samples"]),
                "source": str(path),
            }
        )

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
            dice_std=("dice", lambda values: values.std(ddof=1)),
            iou_mean=("iou", "mean"),
            iou_std=("iou", lambda values: values.std(ddof=1)),
        )
        .reset_index()
    )
    if set(aggregate["method"]) != {"MedEquiSeg", "U-Net++"}:
        raise ValueError("The retained grouped-validation summary must contain two methods")

    references = pd.read_csv(original_reference).set_index("method")
    comparisons: list[dict[str, float | str]] = []
    for row in aggregate.to_dict(orient="records"):
        reference = references.loc[row["method"]]
        comparisons.append(
            {
                "method": row["method"],
                "groupval_dice": row["dice_mean"],
                "original_dice": float(reference["dice"]),
                "dice_change": row["dice_mean"] - float(reference["dice"]),
                "groupval_iou": row["iou_mean"],
                "original_iou": float(reference["iou"]),
                "iou_change": row["iou_mean"] - float(reference["iou"]),
            }
        )

    seeds.to_csv(output / "seed_metrics.csv", index=False, float_format="%.8g")
    aggregate.to_csv(output / "aggregate.csv", index=False, float_format="%.8g")
    pd.DataFrame(comparisons).to_csv(
        output / "comparison_with_original_protocol.csv", index=False, float_format="%.8g"
    )
    payload = {
        "protocol_hash": protocol_hash,
        "scope": "Retained MedEquiSeg and U-Net++ grouped train--validation sensitivity",
        "aggregate": aggregate.to_dict(orient="records"),
        "comparison_with_original_protocol": comparisons,
    }
    (output / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
