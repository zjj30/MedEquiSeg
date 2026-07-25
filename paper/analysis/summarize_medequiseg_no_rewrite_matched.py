#!/usr/bin/env python3
"""Audit and summarize the targeted BUSI rewrite/no-rewrite matched control."""

from __future__ import annotations

import argparse
import json
import shlex
import statistics
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


PROTOCOL_HASH = "abb8ccb2d46cf039a4a0c573d733e3bc510636e6dfa7e59032508ab96fc0f718"
DATASET = "medclipseg_busi"
SEEDS = (123, 456, 789)
REWRITE_RUN_ID = "V3_CTRL_REWRITE_MATCHED_20260725"
NO_REWRITE_RUN_ID = "V3_CTRL_NO_REWRITE_MATCHED_20260725"
EXPECTED_AUGMENTATIONS = {
    REWRITE_RUN_ID: "lcaug_v2_dynamic_shared_plan_dataset",
    NO_REWRITE_RUN_ID: "lcaug_v2_dynamic_shared_plan_no_text_rewrite_dataset",
}
PAIR_INVARIANTS = (
    "protocol_id",
    "protocol_hash",
    "code_sha256",
    "git_commit",
    "manifest",
    "manifest_sha256",
    "cache",
    "cache_sha256",
    "recipe",
    "epochs",
)
OVERLAP_METRICS = ("dice", "iou")
REPORT_METRICS = ("dice", "iou", "nsd", "hd95", "assd")
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20260725


def task_dir(root: Path, run_id: str, seed: int) -> Path:
    return root / "logs/protocol_v3" / PROTOCOL_HASH / run_id / DATASET / f"seed{seed}"


def read_meta(root: Path, run_id: str, seed: int) -> dict:
    path = task_dir(root, run_id, seed) / "run_meta.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        raise RuntimeError(f"Incomplete run metadata: {path} status={payload.get('status')!r}")
    if payload.get("augmentation") != EXPECTED_AUGMENTATIONS[run_id]:
        raise ValueError(f"Unexpected augmentation in {path}: {payload.get('augmentation')!r}")
    if payload.get("task", {}).get("run_id") != run_id:
        raise ValueError(f"Run-id mismatch in {path}")
    if payload.get("task", {}).get("dataset") != DATASET:
        raise ValueError(f"Dataset mismatch in {path}")
    if int(payload.get("task", {}).get("seed")) != seed:
        raise ValueError(f"Seed mismatch in {path}")
    return payload


def read_train_command(root: Path, run_id: str, seed: int) -> list[str]:
    path = task_dir(root, run_id, seed) / "train.log"
    if not path.is_file():
        raise FileNotFoundError(path)
    first_line = path.open(encoding="utf-8", errors="replace").readline().strip()
    if not first_line.startswith("command="):
        raise ValueError(f"Missing command record in {path}")
    return shlex.split(first_line.removeprefix("command="))


def flag_value(command: list[str], flag: str) -> str:
    try:
        return command[command.index(flag) + 1]
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Missing value for {flag} in training command") from exc


def normalized_training_command(command: list[str]) -> list[str]:
    normalized = list(command)
    index = normalized.index("--aug-strength") + 1
    normalized[index] = "<MATCHED_AUGMENTATION_ARM>"
    return normalized


def audit_pair(root: Path) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    reference: dict | None = None
    for seed in SEEDS:
        rewrite = read_meta(root, REWRITE_RUN_ID, seed)
        no_rewrite = read_meta(root, NO_REWRITE_RUN_ID, seed)
        rewrite_command = read_train_command(root, REWRITE_RUN_ID, seed)
        no_rewrite_command = read_train_command(root, NO_REWRITE_RUN_ID, seed)
        if normalized_training_command(rewrite_command) != normalized_training_command(no_rewrite_command):
            raise ValueError(f"Training-command mismatch beyond augmentation arm for seed {seed}")
        locked_flags = {
            "--epochs": "100",
            "--batch-size": "8",
            "--workers": "2",
            "--max-train-samples": "0",
            "--max-val-samples": "0",
            "--seed": str(seed),
            "--split-seed": "123",
            "--scheduler": "cosine",
            "--min-lr": "1e-6",
            "--causal-recipe": "biomed_lcaug_v2_atconv4",
        }
        for flag, expected_value in locked_flags.items():
            actual_value = flag_value(rewrite_command, flag)
            if actual_value != expected_value:
                raise ValueError(
                    f"Unexpected {flag} for seed {seed}: {actual_value!r} != {expected_value!r}"
                )
        mismatches = {
            key: [rewrite.get(key), no_rewrite.get(key)]
            for key in PAIR_INVARIANTS
            if rewrite.get(key) != no_rewrite.get(key)
        }
        if mismatches:
            raise ValueError(f"Matched-pair invariant failure for seed {seed}: {mismatches}")
        invariant_payload = {key: rewrite.get(key) for key in PAIR_INVARIANTS}
        if reference is None:
            reference = invariant_payload
        elif any(reference[key] != invariant_payload[key] for key in PAIR_INVARIANTS):
            raise ValueError(f"Cross-seed invariant failure for seed {seed}")
        rows.append(
            {
                "seed": seed,
                "rewrite_run_id": REWRITE_RUN_ID,
                "no_rewrite_run_id": NO_REWRITE_RUN_ID,
                "code_sha256": rewrite["code_sha256"],
                "manifest_sha256": rewrite["manifest_sha256"],
                "cache_sha256": rewrite["cache_sha256"],
                "recipe": rewrite["recipe"],
                "epochs": rewrite["epochs"],
                "batch_size": flag_value(rewrite_command, "--batch-size"),
                "workers": flag_value(rewrite_command, "--workers"),
                "split_seed": flag_value(rewrite_command, "--split-seed"),
                "scheduler": flag_value(rewrite_command, "--scheduler"),
                "min_lr": flag_value(rewrite_command, "--min-lr"),
                "rewrite_augmentation": rewrite["augmentation"],
                "no_rewrite_augmentation": no_rewrite["augmentation"],
                "training_command_audit": "PASS",
                "pair_audit": "PASS",
            }
        )
    return rows, reference or {}


def read_cases(root: Path, run_id: str, seed: int) -> pd.DataFrame:
    path = task_dir(root, run_id, seed) / "controls/true/per_case.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    required = {"case_id", *REPORT_METRICS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    if frame["case_id"].duplicated().any():
        raise ValueError(f"Duplicate case_id values in {path}")
    return frame.set_index("case_id").sort_index()


def hierarchical_ci(delta: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    n_cases, n_seeds = delta.shape
    estimates = np.empty(BOOTSTRAP_SAMPLES, dtype=np.float64)
    for index in range(BOOTSTRAP_SAMPLES):
        cases = rng.integers(0, n_cases, n_cases)
        seeds = rng.integers(0, n_seeds, n_seeds)
        estimates[index] = delta[np.ix_(cases, seeds)].mean()
    low, high = np.quantile(estimates, (0.025, 0.975))
    return float(low), float(high)


def summarize(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed_rows: list[dict] = []
    case_frames: dict[str, dict[int, pd.DataFrame]] = {
        "rewrite": {},
        "no_rewrite": {},
    }
    for seed in SEEDS:
        case_frames["rewrite"][seed] = read_cases(root, REWRITE_RUN_ID, seed)
        case_frames["no_rewrite"][seed] = read_cases(root, NO_REWRITE_RUN_ID, seed)
        if not case_frames["rewrite"][seed].index.equals(case_frames["no_rewrite"][seed].index):
            raise ValueError(f"Case-id mismatch between arms for seed {seed}")
        for arm in ("rewrite", "no_rewrite"):
            frame = case_frames[arm][seed]
            row = {"dataset": "BUSI", "arm": arm, "seed": seed, "n_cases": len(frame)}
            row.update({metric: float(frame[metric].mean()) for metric in REPORT_METRICS})
            seed_rows.append(row)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    stats_rows: list[dict] = []
    for metric in OVERLAP_METRICS:
        rewrite_matrix = np.column_stack(
            [case_frames["rewrite"][seed][metric].to_numpy(dtype=float) for seed in SEEDS]
        )
        no_rewrite_matrix = np.column_stack(
            [case_frames["no_rewrite"][seed][metric].to_numpy(dtype=float) for seed in SEEDS]
        )
        delta = rewrite_matrix - no_rewrite_matrix
        rewrite_case_mean = rewrite_matrix.mean(axis=1)
        no_rewrite_case_mean = no_rewrite_matrix.mean(axis=1)
        low, high = hierarchical_ci(delta, rng)
        wilcoxon = stats.wilcoxon(
            rewrite_case_mean,
            no_rewrite_case_mean,
            zero_method="wilcox",
            alternative="two-sided",
            method="auto",
        )
        rewrite_seed_means = rewrite_matrix.mean(axis=0)
        no_rewrite_seed_means = no_rewrite_matrix.mean(axis=0)
        stats_rows.append(
            {
                "dataset": "BUSI",
                "metric": metric,
                "n_cases": delta.shape[0],
                "n_seeds": delta.shape[1],
                "rewrite_mean": float(rewrite_seed_means.mean()),
                "rewrite_std_across_seeds": float(statistics.stdev(rewrite_seed_means)),
                "no_rewrite_mean": float(no_rewrite_seed_means.mean()),
                "no_rewrite_std_across_seeds": float(statistics.stdev(no_rewrite_seed_means)),
                "delta_rewrite_minus_no_rewrite": float(delta.mean()),
                "hierarchical_ci_low": low,
                "hierarchical_ci_high": high,
                "wilcoxon_case_mean_p": float(wilcoxon.pvalue),
            }
        )
    return pd.DataFrame(seed_rows), pd.DataFrame(stats_rows)


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without pandas' optional tabulate dependency."""
    headers = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        cells = []
        for value in row:
            if isinstance(value, float):
                cell = f"{value:.6f}"
            else:
                cell = str(value)
            cells.append(cell.replace("|", "\\|"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_markdown(path: Path, seed_metrics: pd.DataFrame, paired_stats: pd.DataFrame, audit: dict) -> None:
    lines = [
        "# Targeted matched prompt-rewrite control (BUSI)",
        "",
        "The two arms passed the metadata audit. They share the same current code snapshot, protocol, manifest, BioMedCLIP cache, recipe, epoch count, seeds, and deterministic image/mask augmentation plans. The only intended intervention is prompt rewriting after discrete flips or right-angle rotations.",
        "",
        f"- code SHA-256: `{audit['code_sha256']}`",
        f"- manifest SHA-256: `{audit['manifest_sha256']}`",
        f"- cache SHA-256: `{audit['cache_sha256']}`",
        f"- recipe: `{audit['recipe']}`",
        f"- seeds: `{', '.join(map(str, SEEDS))}`",
        "",
        "## Seed-level metrics",
        "",
        dataframe_to_markdown(seed_metrics),
        "",
        "## Paired statistics",
        "",
        dataframe_to_markdown(paired_stats),
        "",
        "The hierarchical confidence interval resamples both test cases and training seeds. Statistical results describe association under this controlled intervention and do not by themselves establish a broader causal mechanism.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_latex_table(path: Path, seed_metrics: pd.DataFrame, paired_stats: pd.DataFrame) -> None:
    arm_labels = {
        "no_rewrite": "Prompt rewriting disabled",
        "rewrite": "Prompt rewriting enabled",
    }
    lines = [
        "% Auto-generated by summarize_medequiseg_no_rewrite_matched.py.",
        "% Do not edit numerical cells manually.",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Targeted matched prompt-rewrite comparison on BUSI.}",
        r"\label{tab:matched_no_rewrite_busi}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{@{}lcc@{}}",
        r"\toprule",
        r"Training arm & Dice (\%) & IoU (\%) \\",
        r"\midrule",
    ]
    for arm in ("no_rewrite", "rewrite"):
        rows = seed_metrics[seed_metrics["arm"] == arm]
        dice = 100.0 * rows["dice"].to_numpy(dtype=float)
        iou = 100.0 * rows["iou"].to_numpy(dtype=float)
        lines.append(
            f"{arm_labels[arm]} & "
            f"${dice.mean():.2f}\\pm{dice.std(ddof=1):.2f}$ & "
            f"${iou.mean():.2f}\\pm{iou.std(ddof=1):.2f}$ \\\\"
        )
    delta_cells = []
    for metric in OVERLAP_METRICS:
        row = paired_stats[paired_stats["metric"] == metric].iloc[0]
        delta = 100.0 * float(row["delta_rewrite_minus_no_rewrite"])
        low = 100.0 * float(row["hierarchical_ci_low"])
        high = 100.0 * float(row["hierarchical_ci_high"])
        delta_cells.append(f"${delta:+.2f}$ [{low:+.2f}, {high:+.2f}]")
    lines.extend(
        [
            r"\midrule",
            f"Enabled $-$ disabled & {delta_cells[0]} & {delta_cells[1]} \\\\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{2pt}",
            r"\begin{minipage}{0.96\linewidth}",
            r"\footnotesize\raggedright",
            r"\textit{Notes:} Both arms were newly trained for 100 epochs from the same pretrained initialization with seeds 123, 456, and 789, the same fixed BUSI partition, pooled BioMedCLIP cache, two forward-active projector ATConv operators, and identical deterministic image--mask augmentation plans. Only prompt rewriting after flips and right-angle rotations differed. Arm values are mean$\pm$sample standard deviation across seeds. Brackets give the 95\% hierarchical bootstrap interval for the percentage-point difference after resampling both test cases and training seeds. This targeted BUSI comparison does not estimate an across-dataset or clinical effect.\par",
            r"\end{minipage}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    root = args.project_root.resolve()
    output = (args.output_dir or root / "paper/results/medequiseg_no_rewrite_matched_20260725").resolve()
    output.mkdir(parents=True, exist_ok=True)

    audit_rows, invariants = audit_pair(root)
    seed_metrics, paired_stats = summarize(root)
    pd.DataFrame(audit_rows).to_csv(output / "pair_audit.csv", index=False)
    seed_metrics.to_csv(output / "seed_metrics.csv", index=False, float_format="%.8g")
    paired_stats.to_csv(output / "paired_statistics.csv", index=False, float_format="%.8g")
    payload = {
        "status": "PASS",
        "protocol_hash": PROTOCOL_HASH,
        "dataset": DATASET,
        "seeds": list(SEEDS),
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "rewrite_run_id": REWRITE_RUN_ID,
        "no_rewrite_run_id": NO_REWRITE_RUN_ID,
        "pair_invariants": invariants,
        "paired_statistics": paired_stats.to_dict(orient="records"),
    }
    (output / "audit_and_statistics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    write_markdown(output / "summary.md", seed_metrics, paired_stats, invariants)
    write_latex_table(output / "matched_no_rewrite_busi_table.tex", seed_metrics, paired_stats)
    print(seed_metrics.to_string(index=False))
    print(paired_stats.to_string(index=False))
    print(f"output_dir={output}")


if __name__ == "__main__":
    main()
