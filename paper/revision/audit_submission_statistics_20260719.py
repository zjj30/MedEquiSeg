#!/usr/bin/env python3
"""Trace the submission's headline numbers back to frozen CSV/JSON evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--project-root", type=Path, default=None)
args = parser.parse_args()

if args.project_root is None:
    ROOT = Path(__file__).resolve().parent
    TEX_DIR = ROOT / "bmc_work"
    RESULT_DIR = ROOT / "protocol_v3_final_controls_20260718"
    FACTORIAL_PATH = ROOT / "factorial_aggregate.csv"
    COVID_GROUPED_PATH = ROOT / "covid_groupval_statistics.csv"
    OUT_DIR = ROOT / "submission_audit_20260719"
else:
    ROOT = args.project_root.resolve()
    TEX_DIR = ROOT / "paper" / "latex" / "bmc_work"
    RESULT_DIR = ROOT / "paper" / "results" / "protocol_v3_final_controls_20260718"
    FACTORIAL_PATH = ROOT / "paper" / "results" / "medequiseg_factorial_public5_20260715" / "aggregate.csv"
    COVID_GROUPED_PATH = ROOT / "paper" / "results" / "protocol_v3_covid_groupval_sensitivity" / "r11_vs_r11nr_statistics.csv"
    OUT_DIR = ROOT / "paper" / "results" / "bmc_submission_audit_20260719"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def signed_pp(value: float) -> str:
    return f"{value * 100:+.2f}"


def unsigned_pct(value: float) -> str:
    return f"{value * 100:.2f}"


checks: list[dict[str, object]] = []


def record(name: str, passed: bool, evidence: object) -> None:
    checks.append({"name": name, "status": "PASS" if passed else "FAIL", "evidence": evidence})


def macro_row(path: Path, metric: str) -> dict[str, str]:
    matches = [
        row
        for row in read_rows(path)
        if row["scope"] == "dataset_macro" and row["dataset"] == "Public-5" and row["metric"] == metric
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one Public-5 {metric} macro row in {path}, found {len(matches)}")
    return matches[0]


tex_files = sorted(TEX_DIR.glob("*.tex"))
tex = "\n".join(path.read_text(encoding="utf-8") for path in tex_files)
flat_tex = " ".join(tex.split())

# The six-configuration, five-dataset, three-seed matrix contributes 90 cells.
factorial_path = FACTORIAL_PATH
factorial = read_rows(factorial_path)
run_ids = (
    "V3_ABL_BASE",
    "V3_ABL_BIOMED",
    "V3_ABL_ATCONV",
    "V3_ABL_BIOMED_ATCONV",
    "V3_ABL_SHARED_NR",
    "V3_ABL_EQUIPROMPT",
)
datasets = {row["dataset"] for row in factorial if row["dataset"] != "Public-5 macro"}
matrix_keys = {
    (row["run_id"], row["dataset"])
    for row in factorial
    if row["run_id"] in run_ids and row["dataset"] != "Public-5 macro"
}
record(
    "factorial_matrix_is_6x5_with_three_seeds",
    len(datasets) == 5
    and len(matrix_keys) == 30
    and all(int(row["n_seeds"]) == 3 for row in factorial if row["run_id"] in run_ids),
    {"datasets": sorted(datasets), "dataset_configuration_cells": len(matrix_keys), "training_cells": len(matrix_keys) * 3},
)

macro_factorial = {
    row["run_id"]: row
    for row in factorial
    if row["dataset"] == "Public-5 macro" and row["run_id"] in run_ids
}
factorial_fragments = []
for run_id in run_ids:
    row = macro_factorial[run_id]
    dice = f"{unsigned_pct(float(row['dice_mean']))}\\pm{unsigned_pct(float(row['dice_std']))}"
    iou = f"{unsigned_pct(float(row['iou_mean']))}\\pm{unsigned_pct(float(row['iou_std']))}"
    factorial_fragments.extend((dice, iou))
record(
    "all_public5_factorial_numbers_are_in_tex",
    all(fragment in tex for fragment in factorial_fragments),
    factorial_fragments,
)

contrast_files = {
    "full_vs_base": RESULT_DIR / "casefirst_full_vs_base.csv",
    "full_vs_no_rewrite": RESULT_DIR / "casefirst_full_vs_no_rewrite.csv",
    "base_plan_vs_base": RESULT_DIR / "casefirst_base_plan_vs_base.csv",
    "full_vs_base_plan": RESULT_DIR / "full_vs_base_plan_statistics.csv",
    "full_vs_constant": RESULT_DIR / "casefirst_full_vs_constant.csv",
}
contrast_labels = {
    "full_vs_base": "\\methodname{} $-$ Base reference",
    "full_vs_no_rewrite": "\\methodname{} $-$ w/o Rewrite",
    "base_plan_vs_base": "Base $+$ shared plan $-$ Base reference",
    "full_vs_base_plan": "\\methodname{} $-$ Base $+$ shared plan",
    "full_vs_constant": "\\methodname{} $-$ constant-text full model",
}

attribution_lines = []
for key, path in contrast_files.items():
    cells = []
    for metric in ("dice", "iou"):
        row = macro_row(path, metric)
        cells.append(
            f"${signed_pp(float(row['delta']))}$ "
            f"[${signed_pp(float(row['ci_low']))},{signed_pp(float(row['ci_high']))}$]"
        )
    attribution_lines.append(f"{contrast_labels[key]} & {cells[0]} & {cells[1]} \\\\")
record(
    "case_first_attribution_table_matches_csv",
    all(line in tex for line in attribution_lines),
    attribution_lines,
)

# The semantic-control table should carry the case-first true-minus-shuffled Dice intervals.
semantic_rows = [
    row
    for row in read_rows(RESULT_DIR / "casefirst_true_vs_shuffled.csv")
    if row["scope"] == "dataset" and row["metric"] == "dice"
]
semantic_fragments = [
    f"${signed_pp(float(row['delta']))}$ "
    f"[${unsigned_pct(float(row['ci_low']))},{unsigned_pct(float(row['ci_high']))}$]"
    for row in semantic_rows
]
record(
    "true_vs_shuffled_case_first_intervals_are_in_tex",
    len(semantic_fragments) == 5 and all(fragment in tex for fragment in semantic_fragments),
    semantic_fragments,
)

covid_rows = read_rows(COVID_GROUPED_PATH)
covid_dice = next(row for row in covid_rows if row["metric"] == "dice")
covid_expected = {
    "case_delta_pp": signed_pp(float(covid_dice["delta"])),
    "case_ci_pp": [signed_pp(float(covid_dice["ci_low"])), signed_pp(float(covid_dice["ci_high"]))],
    "subject_delta_pp": signed_pp(float(covid_dice["subject_mean_delta"])),
    "cluster_ci_pp": [
        signed_pp(float(covid_dice["cluster_ci_low"])),
        signed_pp(float(covid_dice["cluster_ci_high"])),
    ],
}
covid_fragments = ["$-0.29$", "$-0.55$", "$-0.03$", "$-0.40$", "$-0.58$", "$-0.02$"]
record(
    "covid_grouped_rewrite_sensitivity_matches_tex",
    covid_expected
    == {
        "case_delta_pp": "-0.29",
        "case_ci_pp": ["-0.55", "-0.03"],
        "subject_delta_pp": "-0.40",
        "cluster_ci_pp": ["-0.58", "-0.02"],
    }
    and all(fragment in tex for fragment in covid_fragments),
    covid_expected,
)

atconv_path = RESULT_DIR / "atconv_forward_activation_audit.json"
atconv = json.loads(atconv_path.read_text(encoding="utf-8"))
active_calls = [atconv["calls"][name]["calls"] for name in atconv["active_targets"]]
inactive_calls = [atconv["calls"][name]["calls"] for name in atconv["inactive_targets"]]
record(
    "atconv_runtime_forward_graph",
    atconv["status"] == "PASS"
    and atconv["registered_atconv_count"] == 4
    and atconv["forward_active_count"] == 2
    and atconv["forward_inactive_count"] == 2
    and active_calls == [1, 1]
    and inactive_calls == [0, 0],
    {
        "checkpoint_sha256": atconv["checkpoint_sha256"],
        "active_targets": atconv["active_targets"],
        "inactive_targets": atconv["inactive_targets"],
        "active_calls": active_calls,
        "inactive_calls": inactive_calls,
    },
)
record(
    "manuscript_does_not_claim_atconv4",
    "four active ATConv" not in tex
    and "four ATConv operators" not in tex
    and "no ATConv4 claim" in tex
    and "two forward-active" in tex,
    {
        "ATConv4_occurrences_in_negative_disclosures": tex.count("ATConv4"),
        "no_ATConv4_claim_occurrences": tex.count("no ATConv4 claim"),
        "two_forward_active_occurrences": tex.count("two forward-active"),
    },
)
record(
    "shared_plan_is_disclosed_as_protocol_amendment",
    "separately dated R11" in flat_tex
    and "amendment on the fixed V3 data and evaluation protocol" in flat_tex
    and "not encode the full training policy" in flat_tex,
    {
        "separately_dated_r11": tex.count("separately dated R11"),
        "amendment_phrase": tex.count("amendment on the fixed V3 data and evaluation protocol"),
    },
)

source_hashes = {
    str(path.relative_to(ROOT)): sha256(path)
    for path in [factorial_path, *contrast_files.values(), RESULT_DIR / "casefirst_true_vs_shuffled.csv", atconv_path, COVID_GROUPED_PATH]
}
failed = [check for check in checks if check["status"] != "PASS"]
payload = {
    "status": "PASS" if not failed else "FAIL",
    "audit_date": "2026-07-19",
    "scope": "submission headline statistics, 90-cell matrix, protocol disclosure, and ATConv runtime activation",
    "checks": checks,
    "source_sha256": source_hashes,
}
OUT_DIR.mkdir(parents=True, exist_ok=True)
output = OUT_DIR / "submission_statistics_audit.json"
output.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
print(json.dumps({"status": payload["status"], "checks": len(checks), "failed": len(failed), "output": str(output)}))
raise SystemExit(1 if failed else 0)
