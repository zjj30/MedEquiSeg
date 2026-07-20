#!/usr/bin/env python3
"""Plot Protocol V3 accuracy gains, ordered ablations, and rewrite controls."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RUN_ORDER = (
    "V3_ABL_BASE",
    "V3_ABL_BIOMED",
    "V3_ABL_ATCONV",
    "V3_ABL_BIOMED_ATCONV",
    "V3_ABL_SHARED_NR",
    "V3_ABL_EQUIPROMPT",
)
LABELS = (
    "Base",
    "+ BioMedCLIP",
    "+ ATConv",
    "+ BioMedCLIP + ATConv",
    "+ shared plan, no rewrite",
    "MedEquiSeg",
)
DATASET_ORDER = ("BUSI", "ClinicDB", "BUS-BRA", "BRISC", "COVID-19")
DATASET_SLUGS = {
    "medclipseg_busi": "BUSI",
    "medclipseg_clinicdb": "ClinicDB",
    "medclipseg_busbra": "BUS-BRA",
    "medclipseg_brisc": "BRISC",
    "medclipseg_covid19": "COVID-19",
}

NAVY = "#163A63"
BLUE = "#2C6EA3"
TEAL = "#157A6E"
ORANGE = "#C06A16"
RED = "#B23A48"
GRAY = "#607080"
LIGHT = "#D8E0E8"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.2,
            "axes.titlesize": 9.2,
            "axes.labelsize": 8.4,
            "xtick.labelsize": 7.6,
            "ytick.labelsize": 7.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    result_dir = ROOT / "paper/results/medequiseg_factorial_public5_20260715"
    parser.add_argument("--aggregate", type=Path, default=result_dir / "aggregate.csv")
    parser.add_argument("--seed-metrics", type=Path, default=result_dir / "seed_metrics.csv")
    parser.add_argument(
        "--rewrite-statistics",
        type=Path,
        default=ROOT
        / "paper/results/protocol_v3_final_controls_20260718/casefirst_full_vs_no_rewrite.csv",
    )
    parser.add_argument(
        "--output-prefix", type=Path, default=ROOT / "paper/figures/strict_factorial_evidence"
    )
    args = parser.parse_args()

    configure_style()
    aggregate = pd.read_csv(args.aggregate)
    seed_metrics = pd.read_csv(args.seed_metrics)
    macro = aggregate[aggregate["dataset"] == "Public-5 macro"].set_index("run_id")
    missing = [run_id for run_id in RUN_ORDER if run_id not in macro.index]
    if missing:
        raise ValueError(f"Missing complete ordered-ablation rows: {missing}")

    rewrite = pd.read_csv(args.rewrite_statistics)
    rewrite["display_dataset"] = rewrite["dataset"].map(DATASET_SLUGS).fillna(rewrite["dataset"])
    rewrite_dataset = rewrite[rewrite["scope"] == "dataset"].set_index(
        ["display_dataset", "metric"]
    )
    macro_rewrite = rewrite[
        (rewrite["scope"] == "dataset_macro") & (rewrite["metric"] == "dice")
    ].iloc[0]

    fig = plt.figure(figsize=(7.35, 4.65))
    grid = fig.add_gridspec(2, 2, height_ratios=(1.05, 1.0), hspace=0.58, wspace=0.58)
    ax_a = fig.add_subplot(grid[0, :])
    ax_b = fig.add_subplot(grid[1, 0])
    ax_c = fig.add_subplot(grid[1, 1])

    y = np.arange(len(RUN_ORDER))[::-1]
    means = np.array([100 * macro.loc[run_id, "dice_mean"] for run_id in RUN_ORDER])
    stds = np.array([100 * macro.loc[run_id, "dice_std"] for run_id in RUN_ORDER])
    colors = [GRAY, BLUE, ORANGE, "#6A5A9E", RED, TEAL]
    for index, (value, std, color) in enumerate(zip(means, stds, colors)):
        ax_a.errorbar(
            value,
            y[index],
            xerr=std,
            fmt="o",
            ms=5.5 if index >= 4 else 4.8,
            color=color,
            ecolor=color,
            elinewidth=1.2,
            capsize=2.4,
            zorder=3,
        )
        ax_a.text(value + std + 0.06, y[index], f"{value:.2f}", va="center", fontsize=7.2, color=color)
    ax_a.set_yticks(y, LABELS)
    ax_a.set_xlabel("Public-5 macro Dice (%)")
    ax_a.set_xlim(min(means - stds) - 0.35, max(means + stds) + 0.55)
    ax_a.grid(axis="x", color=LIGHT, linewidth=0.7, alpha=0.8)
    ax_a.set_title("A  Frozen three-seed ordered ablation", loc="left", fontweight="bold", color=NAVY)
    ax_a.text(1.0, 1.02, "mean $\\pm$ sample SD across seeds", transform=ax_a.transAxes, ha="right", va="bottom", fontsize=6.8, color=GRAY)

    paired = seed_metrics[seed_metrics["run_id"].isin(("V3_ABL_BASE", "V3_ABL_EQUIPROMPT"))]
    paired = paired.pivot(index=["dataset", "seed"], columns="run_id", values="dice").reset_index()
    paired["delta"] = 100 * (paired["V3_ABL_EQUIPROMPT"] - paired["V3_ABL_BASE"])
    gain_rows = []
    for dataset in DATASET_ORDER:
        values = paired.loc[paired["dataset"] == dataset, "delta"].to_numpy(dtype=float)
        gain_rows.append((dataset, float(values.mean()), float(values.std(ddof=1))))
    macro_by_seed = paired.groupby("seed", sort=True)["delta"].mean().to_numpy(dtype=float)
    gain_rows.append(("Public-5 macro", float(macro_by_seed.mean()), float(macro_by_seed.std(ddof=1))))

    y_b = np.arange(len(gain_rows))[::-1]
    for index, (label, delta, std) in enumerate(gain_rows):
        is_macro = label == "Public-5 macro"
        ax_b.errorbar(
            delta,
            y_b[index],
            xerr=std,
            fmt="D" if is_macro else "o",
            ms=5.2 if is_macro else 4.6,
            color=TEAL if delta >= 0 else RED,
            ecolor=TEAL if delta >= 0 else RED,
            elinewidth=1.2,
            capsize=2.3,
            zorder=3,
        )
    ax_b.axvline(0, color="#303840", linewidth=0.85, linestyle="--", zorder=1)
    ax_b.set_yticks(y_b, [row[0] for row in gain_rows])
    ax_b.set_xlabel("MedEquiSeg minus Base, Dice (pp)")
    ax_b.grid(axis="x", color=LIGHT, linewidth=0.7, alpha=0.8)
    ax_b.set_title("B  Accuracy gain over Base", loc="left", fontweight="bold", color=NAVY)

    rows = []
    for dataset in DATASET_ORDER:
        row = rewrite_dataset.loc[(dataset, "dice")]
        rows.append((dataset, 100 * row["delta"], 100 * row["ci_low"], 100 * row["ci_high"]))
    rows.append(
        (
            "Public-5 macro",
            100 * macro_rewrite["delta"],
            100 * macro_rewrite["ci_low"],
            100 * macro_rewrite["ci_high"],
        )
    )
    y_c = np.arange(len(rows))[::-1]
    for index, (label, delta, low, high) in enumerate(rows):
        is_macro = label == "Public-5 macro"
        color = TEAL if delta >= 0 else RED
        marker = "D" if is_macro else "o"
        ax_c.errorbar(
            delta,
            y_c[index],
            xerr=[[delta - low], [high - delta]],
            fmt=marker,
            ms=5.2 if is_macro else 4.6,
            color=color,
            ecolor=color,
            elinewidth=1.2,
            capsize=2.3,
            zorder=3,
        )
    ax_c.axvline(0, color="#303840", linewidth=0.85, linestyle="--", zorder=1)
    ax_c.set_yticks(y_c, [row[0] for row in rows])
    ax_c.set_xlabel("Rewrite minus no rewrite, Dice (pp)")
    ax_c.grid(axis="x", color=LIGHT, linewidth=0.7, alpha=0.8)
    ax_c.set_title("C  Exact-plan rewrite contrast", loc="left", fontweight="bold", color=NAVY)
    fig.subplots_adjust(left=0.18, right=0.98, top=0.94, bottom=0.10)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix, options in (("pdf", {}), ("png", {"dpi": 400})):
        fig.savefig(
            args.output_prefix.with_suffix(f".{suffix}"),
            bbox_inches="tight",
            pad_inches=0.04,
            **options,
        )
    plt.close(fig)
    print(f"figure={args.output_prefix.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
