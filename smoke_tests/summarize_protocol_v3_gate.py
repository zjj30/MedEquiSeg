#!/usr/bin/env python3
"""Apply the preregistered Protocol V3 method-versus-audit decision gate."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from protocol_v3.core import load_protocol_lock, protocol_sha256


ROOT = Path(__file__).resolve().parents[1]
CONTROLS = ("shuffled", "fixed", "empty")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def case_values(path: Path) -> dict[str, float]:
    return {row["case_id"]: float(row["dice"]) for row in read_csv(path)}


def paired_delta(left: Path, right: Path) -> np.ndarray:
    baseline, method = case_values(left), case_values(right)
    if set(baseline) != set(method):
        raise ValueError(f"Unpaired case files: {left} {right}")
    return np.asarray([method[key] - baseline[key] for key in sorted(baseline)], dtype=np.float64)


def bootstrap_ci(values: np.ndarray, reps: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), size=(reps, len(values)))].mean(axis=1)
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def macro_bootstrap(deltas: dict[str, np.ndarray], reps: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = np.empty(reps, dtype=np.float64)
    for rep in range(reps):
        means = []
        for values in deltas.values():
            sample = values[rng.integers(0, len(values), size=len(values))]
            means.append(float(sample.mean()))
        draws[rep] = float(np.mean(means))
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-lock", type=Path, default=ROOT / "smoke_tests/protocol_v3/protocol_lock.yaml")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--bootstrap-reps", type=int, default=10000)
    parser.add_argument("--output-csv", type=Path, default=ROOT / "paper/tables/protocol_v3_gate.csv")
    parser.add_argument("--decision-md", type=Path, default=ROOT / "paper/PAPER_DECISION.md")
    args = parser.parse_args()

    lock = load_protocol_lock(args.protocol_lock)
    protocol_hash = protocol_sha256(args.protocol_lock)
    root = ROOT / "logs/protocol_v3" / protocol_hash
    datasets = list(lock["datasets"])
    required = []
    invalid_meta = []
    for run_id in ("V3_R7", "V3_R8", "V3_R8NR"):
        for dataset in datasets:
            task_root = root / run_id / dataset / f"seed{args.seed}"
            meta_path = task_root / "run_meta.json"
            if not meta_path.is_file():
                invalid_meta.append(f"missing:{meta_path}")
            else:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if meta.get("status") != "complete" or int(meta.get("epochs", 0)) != int(lock["training_epochs"]):
                    invalid_meta.append(f"invalid:{meta_path}")
            controls = ("true", *CONTROLS) if run_id == "V3_R8" else ("true",)
            for control in controls:
                required.append(root / run_id / dataset / f"seed{args.seed}" / "controls" / control / "per_case.csv")
    missing = [path for path in required if not path.is_file()]
    rows: list[dict[str, str]] = []
    if missing or invalid_meta:
        route = "pending"
        first = missing[0] if missing else invalid_meta[0]
        reason = f"Missing artifacts={len(missing)} invalid run metadata={len(invalid_meta)}; first={first}"
    else:
        gate = lock["promotion_gate"]
        r8_vs_r7: dict[str, np.ndarray] = {}
        r8_vs_nr: dict[str, np.ndarray] = {}
        semantic_pass = 0
        wins = 0
        worst_drop = 0.0
        for dataset_index, dataset in enumerate(datasets):
            base = root / "V3_R7" / dataset / f"seed{args.seed}" / "controls/true/per_case.csv"
            r8 = root / "V3_R8" / dataset / f"seed{args.seed}" / "controls/true/per_case.csv"
            nr = root / "V3_R8NR" / dataset / f"seed{args.seed}" / "controls/true/per_case.csv"
            delta = paired_delta(base, r8)
            delta_nr = paired_delta(nr, r8)
            r8_vs_r7[dataset] = delta
            r8_vs_nr[dataset] = delta_nr
            mean_delta = float(delta.mean())
            wins += int(mean_delta >= float(gate["minimum_dice_gain"]))
            worst_drop = min(worst_drop, mean_delta)
            semantic_controls = []
            for control_index, control in enumerate(CONTROLS):
                control_path = root / "V3_R8" / dataset / f"seed{args.seed}" / "controls" / control / "per_case.csv"
                control_delta = paired_delta(control_path, r8)
                low, high = bootstrap_ci(control_delta, args.bootstrap_reps, 1000 + dataset_index * 10 + control_index)
                semantic_controls.append(low > 0.0)
                rows.append(
                    {
                        "gate": "semantic_control",
                        "dataset": dataset,
                        "comparison": f"true_minus_{control}",
                        "delta_mean": f"{control_delta.mean():.6f}",
                        "ci95_low": f"{low:.6f}",
                        "ci95_high": f"{high:.6f}",
                        "pass": str(low > 0.0),
                    }
                )
            semantic_pass += int(all(semantic_controls))
            low, high = bootstrap_ci(delta, args.bootstrap_reps, 2000 + dataset_index)
            rows.append(
                {
                    "gate": "r8_vs_r7",
                    "dataset": dataset,
                    "comparison": "R8_minus_R7",
                    "delta_mean": f"{mean_delta:.6f}",
                    "ci95_low": f"{low:.6f}",
                    "ci95_high": f"{high:.6f}",
                    "pass": str(mean_delta >= float(gate["minimum_dice_gain"])),
                }
            )
        nr_low, nr_high = macro_bootstrap(r8_vs_nr, args.bootstrap_reps, 3000)
        pass_wins = wins >= int(gate["minimum_dataset_wins"])
        pass_drop = worst_drop >= -float(gate["maximum_dataset_drop"])
        pass_nr = nr_low > 0.0
        pass_semantic = semantic_pass >= int(gate["semantic_control_minimum_datasets"])
        route = "method_paper" if all((pass_wins, pass_drop, pass_nr, pass_semantic)) else "semantic_reliability_audit"
        reason = (
            f"wins={wins}/{len(datasets)}; worst_R8_minus_R7={worst_drop:.6f}; "
            f"R8_minus_no_rewrite_macro_CI=[{nr_low:.6f},{nr_high:.6f}]; "
            f"semantic_datasets={semantic_pass}/{len(datasets)}"
        )
        rows.extend(
            [
                {"gate": "minimum_dataset_wins", "dataset": "ALL", "comparison": ">=3", "delta_mean": str(wins), "ci95_low": "", "ci95_high": "", "pass": str(pass_wins)},
                {"gate": "maximum_dataset_drop", "dataset": "ALL", "comparison": ">=-0.003", "delta_mean": f"{worst_drop:.6f}", "ci95_low": "", "ci95_high": "", "pass": str(pass_drop)},
                {"gate": "r8_over_no_rewrite", "dataset": "MACRO", "comparison": "CI_low>0", "delta_mean": "", "ci95_low": f"{nr_low:.6f}", "ci95_high": f"{nr_high:.6f}", "pass": str(pass_nr)},
                {"gate": "semantic_dataset_count", "dataset": "ALL", "comparison": ">=3", "delta_mean": str(semantic_pass), "ci95_low": "", "ci95_high": "", "pass": str(pass_semantic)},
            ]
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    decision = [
        "# Protocol V3 Paper Decision",
        "",
        f"- protocol_id: `{lock['protocol_id']}`",
        f"- protocol_hash: `{protocol_hash}`",
        f"- route: `{route}`",
        f"- evidence: {reason}",
        "",
        "## Claim Boundary",
        "",
    ]
    if route == "method_paper":
        decision.append("Proceed with the preregistered three-seed Public-5 method confirmation.")
    elif route == "semantic_reliability_audit":
        decision.append("Do not claim LCAug rewrite as a new causal mechanism. Complete the three-seed true/shuffled/fixed/empty reliability study.")
    else:
        decision.append("Pilot evidence is incomplete. No method or external-validation claim is permitted.")
    args.decision_md.write_text("\n".join(decision) + "\n", encoding="utf-8")
    print(f"route={route}")
    print(f"decision={args.decision_md}")
    print("final_status: PASS")


if __name__ == "__main__":
    main()
