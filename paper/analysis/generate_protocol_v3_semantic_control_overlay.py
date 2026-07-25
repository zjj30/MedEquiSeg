#!/usr/bin/env python3
"""Generate true-versus-shuffled prompt controls for MedEquiSeg."""

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
DEFAULT_RUN_ID = "V3_ABL_EQUIPROMPT"
SEEDS = (123, 456, 789)
CONTROL_SEED = 123
DATASETS = (
    ("medclipseg_busi", "BUSI"),
    ("medclipseg_clinicdb", "ClinicDB"),
    ("medclipseg_busbra", "BUS-BRA"),
    ("medclipseg_brisc", "BRISC"),
    ("medclipseg_covid19", "COVID-19"),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def rows_by_case(path: Path) -> dict[str, dict[str, str]]:
    return {row["case_id"]: row for row in read_csv(path)}


def per_case_path(root: Path, run_id: str, dataset: str, seed: int, control: str) -> Path:
    return (
        root
        / "logs/protocol_v3"
        / PUBLIC_HASH
        / run_id
        / dataset
        / f"seed{seed}/controls/{control}/per_case.csv"
    )


def manifest_rows(root: Path, dataset: str) -> list[dict[str, str]]:
    path = root / "smoke_tests/protocol_v3/manifests" / f"{dataset}_full.csv"
    return [row for row in read_csv(path) if row["split"].strip().lower() == "test"]


def shuffled_prompts(rows: list[dict[str, str]]) -> dict[str, str]:
    texts = [row["text"] for row in rows]
    shift = 1 + CONTROL_SEED % (len(rows) - 1)
    shifted = texts[shift:] + texts[:shift]
    return {row["case_id"]: text for row, text in zip(rows, shifted)}


def load_binary(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L")) >= 128


def load_rgb(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    lo, hi = np.percentile(array, (1, 99))
    return np.clip((array - lo) / (hi - lo), 0, 1) if hi > lo else array


def majority_vote(paths: list[str]) -> np.ndarray:
    return np.stack([load_binary(path) for path in paths]).sum(axis=0) >= 2


def dice(prediction: np.ndarray, target: np.ndarray) -> float:
    denominator = int(prediction.sum()) + int(target.sum())
    return 1.0 if denominator == 0 else 2.0 * np.logical_and(prediction, target).sum() / denominator


def overlay(image: np.ndarray, target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    result = image.copy()
    regions = (
        (np.logical_and(target, prediction), np.array([0.10, 0.68, 0.32])),
        (np.logical_and(~target, prediction), np.array([0.90, 0.20, 0.18])),
        (np.logical_and(target, ~prediction), np.array([0.12, 0.42, 0.90])),
    )
    for region, color in regions:
        result[region] = 0.35 * result[region] + 0.65 * color
    return result


def target_overlay(image: np.ndarray, target: np.ndarray) -> np.ndarray:
    result = image.copy()
    result[target] = 0.35 * result[target] + 0.65 * np.array([0.10, 0.68, 0.32])
    return result


def crop_bounds(mask: np.ndarray, target_aspect: float = 1.28, padding: float = 0.22) -> tuple[slice, slice]:
    """Return a shared context crop around all relevant foreground pixels."""
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


def select_case(root: Path, run_id: str, dataset: str) -> dict[str, object]:
    true = {
        seed: rows_by_case(per_case_path(root, run_id, dataset, seed, "true"))
        for seed in SEEDS
    }
    shuffled = {
        seed: rows_by_case(per_case_path(root, run_id, dataset, seed, "shuffled"))
        for seed in SEEDS
    }
    common = set.intersection(*(set(rows) for rows in (*true.values(), *shuffled.values())))
    candidates = []
    for case_id in sorted(common):
        true_rows = [true[seed][case_id] for seed in SEEDS]
        shuffled_rows = [shuffled[seed][case_id] for seed in SEEDS]
        target = load_binary(true_rows[0]["mask_path"])
        if not target.any():
            continue
        delta = statistics.mean(float(row["dice"]) for row in true_rows) - statistics.mean(
            float(row["dice"]) for row in shuffled_rows
        )
        candidates.append((case_id, delta, true_rows, shuffled_rows, target))
    values = np.asarray([row[1] for row in candidates])
    target_delta = float(np.quantile(values, 0.75, method="linear"))
    case_id, delta, true_rows, shuffled_rows, target = min(
        candidates, key=lambda row: (abs(row[1] - target_delta), row[0])
    )
    manifest = manifest_rows(root, dataset)
    manifest_by_case = {row["case_id"]: row for row in manifest}
    shuffled_text = shuffled_prompts(manifest)[case_id]
    item = manifest_by_case[case_id]
    true_vote = majority_vote([row["prediction_path"] for row in true_rows])
    shuffled_vote = majority_vote([row["prediction_path"] for row in shuffled_rows])
    return {
        "case_id": case_id,
        "delta_mean_dice": delta,
        "image_path": item["image_path"],
        "mask_path": item["mask_path"],
        "true_prompt": item["text"],
        "shuffled_prompt": shuffled_text,
        "target": target,
        "true_vote": true_vote,
        "shuffled_vote": shuffled_vote,
        "true_consensus_dice": dice(true_vote, target),
        "shuffled_consensus_dice": dice(shuffled_vote, target),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--selection-csv", type=Path, required=True)
    args = parser.parse_args()

    selected = []
    for dataset, label in DATASETS:
        selected.append(
            {
                "run_id": args.run_id,
                "dataset": dataset,
                "label": label,
                **select_case(args.project_root, args.run_id, dataset),
            }
        )

    args.selection_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "run_id",
        "dataset",
        "label",
        "case_id",
        "delta_mean_dice",
        "true_consensus_dice",
        "shuffled_consensus_dice",
        "image_path",
        "mask_path",
        "true_prompt",
        "shuffled_prompt",
    )
    with args.selection_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in selected)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })
    n_rows = len(selected)
    figure, axes = plt.subplots(n_rows, 4, figsize=(7.2, 1.42 * n_rows + 0.55))
    figure.subplots_adjust(left=0.125, right=0.99, top=0.945, bottom=0.075, wspace=0.035, hspace=0.22)
    for column, title in enumerate(("Image", "Ground truth", "True prompt", "Shuffled prompt")):
        axes[0, column].set_title(title, fontsize=8.5, fontweight="bold", pad=6)
    for index, row in enumerate(selected):
        image = load_rgb(row["image_path"])
        crop = crop_bounds(np.logical_or.reduce((row["target"], row["true_vote"], row["shuffled_vote"])))
        image = image[crop]
        target = row["target"][crop]
        true_vote = row["true_vote"][crop]
        shuffled_vote = row["shuffled_vote"][crop]
        displays = (
            image,
            target_overlay(image, target),
            overlay(image, target, true_vote),
            overlay(image, target, shuffled_vote),
        )
        for column, display in enumerate(displays):
            axes[index, column].imshow(display, aspect="auto")
            axes[index, column].set_xticks([])
            axes[index, column].set_yticks([])
            for spine in axes[index, column].spines.values():
                spine.set_edgecolor("#c6cdd3")
                spine.set_linewidth(0.7)
        axes[index, 0].text(
            -0.08,
            0.5,
            f"{row['label']}\n75th-percentile\nprompt contrast\n$\\Delta$={100 * row['delta_mean_dice']:+.1f} pp",
            transform=axes[index, 0].transAxes,
            ha="right",
            va="center",
            fontsize=6.8,
            fontweight="bold",
        )
        axes[index, 2].set_xlabel(
            f"Dice {100 * row['true_consensus_dice']:.1f}%",
            fontsize=7.0,
            labelpad=3,
        )
        axes[index, 3].set_xlabel(
            f"Dice {100 * row['shuffled_consensus_dice']:.1f}%",
            fontsize=7.0,
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
    figure.savefig(args.output_prefix.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    figure.savefig(args.output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(f"figure={args.output_prefix.with_suffix('.pdf')}")
    print(f"selection={args.selection_csv}")


if __name__ == "__main__":
    main()
PROJECT_ROOT = Path(__file__).resolve().parents[2]
