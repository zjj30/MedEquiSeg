#!/usr/bin/env python3
"""Generate frozen-rule Protocol V3 success and failure overlays."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from PIL import Image


PUBLIC_HASH = "abb8ccb2d46cf039a4a0c573d733e3bc510636e6dfa7e59032508ab96fc0f718"
DEFAULT_METHOD_RUN_ID = "V3_ABL_EQUIPROMPT"
SEEDS = (123, 456, 789)
PANELS = (
    {
        "dataset": "medclipseg_busi",
        "label": "BUSI",
        "baseline": "ukan",
        "baseline_label": "U-KAN",
        "role": "typical gain",
        "quantile": 0.90,
    },
    {
        "dataset": "medclipseg_covid19",
        "label": "COVID-19",
        "baseline": "unetplusplus",
        "baseline_label": "U-Net++",
        "role": "typical gain",
        "quantile": 0.90,
    },
    {
        "dataset": "medclipseg_clinicdb",
        "label": "ClinicDB",
        "baseline": "nnunet",
        "baseline_label": "nnU-Net",
        "role": "typical harm",
        "quantile": 0.10,
    },
    {
        "dataset": "medclipseg_busbra",
        "label": "BUS-BRA",
        "baseline": "umamba",
        "baseline_label": "U-Mamba",
        "role": "typical harm",
        "quantile": 0.10,
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("<PROJECT_ROOT>"))
    parser.add_argument("--method-run-id", default=DEFAULT_METHOD_RUN_ID)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--selection-csv", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def metric_path(
    root: Path,
    dataset: str,
    seed: int,
    baseline: str | None = None,
    method_run_id: str = DEFAULT_METHOD_RUN_ID,
) -> Path:
    if baseline is not None:
        return root / "logs/protocol_v3_image_baselines" / baseline / dataset / f"seed{seed}/controls/true/per_case.csv"
    return (
        root
        / "logs/protocol_v3"
        / PUBLIC_HASH
        / method_run_id
        / dataset
        / f"seed{seed}/controls/true/per_case.csv"
    )


def manifest_path(root: Path, dataset: str) -> Path:
    return root / "smoke_tests/protocol_v3/manifests" / f"{dataset}_full.csv"


def rows_by_case(path: Path) -> dict[str, dict[str, str]]:
    return {row["case_id"]: row for row in read_csv(path)}


def load_binary(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L")) >= 128


def load_rgb(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    lo, hi = np.percentile(array, (1, 99))
    if hi > lo:
        array = np.clip((array - lo) / (hi - lo), 0.0, 1.0)
    return array


def majority_vote(paths: list[str]) -> np.ndarray:
    masks = [load_binary(path) for path in paths]
    shape = masks[0].shape
    if any(mask.shape != shape for mask in masks):
        raise ValueError(f"Prediction shape mismatch: {[mask.shape for mask in masks]}")
    return np.stack(masks).sum(axis=0) >= 2


def dice(prediction: np.ndarray, target: np.ndarray) -> float:
    denominator = int(prediction.sum()) + int(target.sum())
    if denominator == 0:
        return 1.0
    return 2.0 * float(np.logical_and(prediction, target).sum()) / denominator


def error_overlay(image: np.ndarray, target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    overlay = image.copy()
    true_positive = np.logical_and(target, prediction)
    false_positive = np.logical_and(~target, prediction)
    false_negative = np.logical_and(target, ~prediction)
    colors = (
        (true_positive, np.array([0.15, 0.80, 0.36], dtype=np.float32)),
        (false_positive, np.array([0.95, 0.25, 0.20], dtype=np.float32)),
        (false_negative, np.array([0.18, 0.48, 0.95], dtype=np.float32)),
    )
    for mask, color in colors:
        overlay[mask] = 0.35 * overlay[mask] + 0.65 * color
    return overlay


def target_overlay(image: np.ndarray, target: np.ndarray) -> np.ndarray:
    overlay = image.copy()
    color = np.array([0.15, 0.80, 0.36], dtype=np.float32)
    overlay[target] = 0.35 * overlay[target] + 0.65 * color
    return overlay


def crop_bounds(mask: np.ndarray, target_aspect: float = 1.28, padding: float = 0.22) -> tuple[slice, slice]:
    """Return one shared context crop for image, target, and both predictions."""
    rows, cols = np.where(mask)
    height, width = mask.shape
    if rows.size == 0:
        return slice(0, height), slice(0, width)
    y0, y1 = int(rows.min()), int(rows.max()) + 1
    x0, x1 = int(cols.min()), int(cols.max()) + 1
    box_h = max(y1 - y0, 24)
    box_w = max(x1 - x0, 24)
    box_h = int(np.ceil(box_h * (1.0 + 2.0 * padding)))
    box_w = int(np.ceil(box_w * (1.0 + 2.0 * padding)))
    if box_w / box_h < target_aspect:
        box_w = int(np.ceil(box_h * target_aspect))
    else:
        box_h = int(np.ceil(box_w / target_aspect))
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    y0 = max(0, min(height - box_h, cy - box_h // 2))
    x0 = max(0, min(width - box_w, cx - box_w // 2))
    y1, x1 = min(height, y0 + box_h), min(width, x0 + box_w)
    return slice(y0, y1), slice(x0, x1)


def select_case(root: Path, panel: dict, method_run_id: str) -> dict[str, object]:
    dataset = panel["dataset"]
    r11_by_seed = {
        seed: rows_by_case(metric_path(root, dataset, seed, method_run_id=method_run_id))
        for seed in SEEDS
    }
    baseline_by_seed = {
        seed: rows_by_case(metric_path(root, dataset, seed, panel["baseline"])) for seed in SEEDS
    }
    common = set.intersection(
        *(set(rows) for rows in (*r11_by_seed.values(), *baseline_by_seed.values()))
    )
    candidates = []
    for case_id in sorted(common):
        r11_rows = [r11_by_seed[seed][case_id] for seed in SEEDS]
        baseline_rows = [baseline_by_seed[seed][case_id] for seed in SEEDS]
        target = load_binary(r11_rows[0]["mask_path"])
        if not target.any():
            continue
        r11_mean = statistics.mean(float(row["dice"]) for row in r11_rows)
        baseline_mean = statistics.mean(float(row["dice"]) for row in baseline_rows)
        candidates.append(
            {
                "case_id": case_id,
                "r11_mean_dice": r11_mean,
                "baseline_mean_dice": baseline_mean,
                "delta_mean_dice": r11_mean - baseline_mean,
                "mask_path": r11_rows[0]["mask_path"],
                "r11_prediction_paths": [row["prediction_path"] for row in r11_rows],
                "baseline_prediction_paths": [row["prediction_path"] for row in baseline_rows],
            }
        )
    if not candidates:
        raise ValueError(f"No non-empty common cases for {dataset}")
    deltas = np.asarray([row["delta_mean_dice"] for row in candidates])
    target_delta = float(np.quantile(deltas, panel["quantile"], method="linear"))
    selected = min(candidates, key=lambda row: (abs(row["delta_mean_dice"] - target_delta), row["case_id"]))
    manifest = rows_by_case(manifest_path(root, dataset))
    selected["image_path"] = manifest[selected["case_id"]]["image_path"]
    selected["prompt"] = manifest[selected["case_id"]].get("text", "").replace("\n", " ").strip()
    selected["quantile_target"] = target_delta
    selected["eligible_cases"] = len(candidates)
    return selected


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    selections = []
    for panel in PANELS:
        selected = select_case(root, panel, args.method_run_id)
        target = load_binary(selected["mask_path"])
        r11_vote = majority_vote(selected["r11_prediction_paths"])
        baseline_vote = majority_vote(selected["baseline_prediction_paths"])
        selections.append(
            {
                "method_run_id": args.method_run_id,
                **panel,
                **selected,
                "r11_consensus_dice": dice(r11_vote, target),
                "baseline_consensus_dice": dice(baseline_vote, target),
                "r11_vote": r11_vote,
                "baseline_vote": baseline_vote,
                "target": target,
            }
        )

    args.selection_csv.parent.mkdir(parents=True, exist_ok=True)
    csv_fields = (
        "method_run_id",
        "dataset",
        "label",
        "role",
        "quantile",
        "baseline_label",
        "case_id",
        "eligible_cases",
        "quantile_target",
        "r11_mean_dice",
        "baseline_mean_dice",
        "delta_mean_dice",
        "r11_consensus_dice",
        "baseline_consensus_dice",
        "image_path",
        "mask_path",
        "prompt",
    )
    with args.selection_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in csv_fields} for row in selections)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })
    figure, axes = plt.subplots(len(selections), 4, figsize=(7.2, 6.25))
    figure.subplots_adjust(left=0.125, right=0.99, top=0.945, bottom=0.075, wspace=0.035, hspace=0.22)
    column_titles = ("Image", "Ground truth", "Best image-only", "MedEquiSeg")
    for column, title in enumerate(column_titles):
        axes[0, column].set_title(title, fontsize=8.5, fontweight="bold", pad=6)
    for row_index, selected in enumerate(selections):
        image = load_rgb(selected["image_path"])
        target = selected["target"]
        crop = crop_bounds(np.logical_or.reduce((target, selected["baseline_vote"], selected["r11_vote"])))
        image = image[crop]
        target = target[crop]
        baseline_vote = selected["baseline_vote"][crop]
        r11_vote = selected["r11_vote"][crop]
        displays = (
            image,
            target_overlay(image, target),
            error_overlay(image, target, baseline_vote),
            error_overlay(image, target, r11_vote),
        )
        for column, display in enumerate(displays):
            axes[row_index, column].imshow(display, aspect="auto")
            axes[row_index, column].set_xticks([])
            axes[row_index, column].set_yticks([])
            for spine in axes[row_index, column].spines.values():
                spine.set_edgecolor("#cad2d9")
                spine.set_linewidth(0.7)
        quantile_percent = int(round(100 * selected["quantile"]))
        axes[row_index, 0].text(
            -0.08,
            0.5,
            f"{selected['label']}\n{quantile_percent}th-percentile\ncontrast\n$\\Delta$={100 * selected['delta_mean_dice']:+.1f} pp",
            transform=axes[row_index, 0].transAxes,
            fontsize=6.6,
            fontweight="bold",
            ha="right",
            va="center",
        )
        axes[row_index, 2].set_xlabel(
            f"{selected['baseline_label']}  |  Dice {100 * selected['baseline_consensus_dice']:.1f}%",
            fontsize=6.8,
            labelpad=3,
        )
        axes[row_index, 3].set_xlabel(
            f"Dice {100 * selected['r11_consensus_dice']:.1f}%",
            fontsize=6.8,
            labelpad=3,
        )
    figure.legend(
        handles=(
            Patch(facecolor="#29ad51", label="TP"),
            Patch(facecolor="#e6332e", label="FP"),
            Patch(facecolor="#1f6fe5", label="FN"),
        ),
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=7.0,
        handlelength=1.1,
        columnspacing=1.2,
    )
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(args.output_prefix.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(f"selection={args.selection_csv} rows={len(selections)}")
    print(f"figure={args.output_prefix.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
