#!/usr/bin/env python3
"""Audit the stratified prompt control against manuscript and table claims."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path("<PROJECT_ROOT>")
RESULTS = ROOT / "paper/results/protocol_v3_stratified_derangement_20260719"
OUTPUT = ROOT / "paper/results/bmc_submission_audit_20260720/stratified_derangement_submission_audit.json"
PROTOCOL_HASH = "abb8ccb2d46cf039a4a0c573d733e3bc510636e6dfa7e59032508ab96fc0f718"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def find_row(rows: list[dict[str, str]], **expected: str) -> dict[str, str]:
    matches = [row for row in rows if all(row.get(key) == value for key, value in expected.items())]
    if len(matches) != 1:
        raise AssertionError(f"Expected one row {expected}, found {len(matches)}")
    return matches[0]


def close(actual: str, expected: float, tolerance: float = 5e-10) -> None:
    if abs(float(actual) - expected) > tolerance:
        raise AssertionError(f"Numeric mismatch: {actual} != {expected}")


def main() -> int:
    checks: list[dict[str, object]] = []

    def check(name: str, condition: bool, detail: object) -> None:
        checks.append({"name": name, "status": "PASS" if condition else "FAIL", "detail": detail})
        if not condition:
            raise AssertionError(f"{name}: {detail}")

    matrix = json.loads((RESULTS / "matrix_audit.json").read_text(encoding="utf-8"))
    check(
        "matrix_complete",
        matrix.get("status") == "PASS" and matrix.get("tasks") == 15 and matrix.get("failures") == 0,
        {"status": matrix.get("status"), "tasks": matrix.get("tasks"), "failures": matrix.get("failures")},
    )
    check("matrix_protocol_hash", matrix.get("protocol_hash") == PROTOCOL_HASH, matrix.get("protocol_hash"))
    check(
        "matrix_rows",
        len(matrix.get("rows", [])) == 15 and all(row.get("status") == "PASS" for row in matrix.get("rows", [])),
        matrix.get("rows", []),
    )

    expected_artifacts = []
    for dataset in ("medclipseg_busi", "medclipseg_clinicdb", "medclipseg_busbra", "medclipseg_brisc", "medclipseg_covid19"):
        for seed in (123, 456, 789):
            run_dir = RESULTS / "runs" / dataset / f"seed{seed}"
            for name in ("per_case.csv", "summary.csv", "run_meta.json", "complete.json"):
                expected_artifacts.append(run_dir / name)
    missing = [str(path) for path in expected_artifacts if not path.is_file()]
    check("task_artifacts", not missing, {"expected": len(expected_artifacts), "missing": missing})

    all_rows = read_csv(RESULTS / "casefirst_true_vs_stratified_derangement.csv")
    macro = find_row(all_rows, analysis_scope="all_cases", scope="dataset_macro", dataset="Public-5", metric="dice")
    close(macro["delta_true_minus_control"], 0.05633994608619428)
    close(macro["ci_low"], 0.04707845181396785)
    close(macro["ci_high"], 0.06672940691249273)
    check("public5_casefirst", True, macro)

    expected_dataset = {
        "medclipseg_busi": (78, 0.05908005128205128, 0.02618690833333334, 0.09862688888888888, 0.03799230504478439),
        "medclipseg_clinicdb": (61, 0.03365259562841532, 0.005421618579234995, 0.06959632500000004, 0.14169051118793144),
        "medclipseg_busbra": (282, 0.0007638900709219863, -0.0009666548463357042, 0.0025778378250591047, 0.311234344392899),
        "medclipseg_brisc": (892, 0.003706202914798203, 0.0012609028120328803, 0.006518170861360236, 0.011492524752219884),
        "medclipseg_covid19": (2113, 0.18449699053478466, 0.1768139856996372, 0.1921565303991166, 2.4572574710622047e-299),
    }
    for dataset, values in expected_dataset.items():
        row = find_row(all_rows, analysis_scope="all_cases", scope="dataset", dataset=dataset, metric="dice")
        check(f"{dataset}_cases", int(row["n_cases"]) == values[0], row["n_cases"])
        for key, expected in zip(("delta_true_minus_control", "ci_low", "ci_high", "wilcoxon_p_holm"), values[1:]):
            close(row[key], expected, tolerance=1e-12)

    changed_rows = read_csv(RESULTS / "casefirst_true_vs_stratified_derangement_changed_only.csv")
    changed = find_row(changed_rows, analysis_scope="text_changed", scope="dataset_macro", dataset="Public-5", metric="dice")
    check("changed_cases", int(changed["n_cases"]) == 3413, changed["n_cases"])
    close(changed["delta_true_minus_control"], 0.058703148137476346)
    close(changed["ci_low"], 0.048650936479613)
    close(changed["ci_high"], 0.07025945768722446)

    covid_rows = read_csv(RESULTS / "covid_subject_cluster_statistics.csv")
    covid = find_row(covid_rows, dataset="medclipseg_covid19", metric="dice")
    check("covid_subjects", int(covid["n_subjects"]) == 474, covid["n_subjects"])
    close(covid["equal_subject_delta_true_minus_control"], 0.22422871517301932)
    close(covid["cluster_ci_low"], 0.2111840260529929)
    close(covid["cluster_ci_high"], 0.2379576056736635)

    main_tex = (ROOT / "paper/latex/bmc_work/main_bmc.tex").read_text(encoding="utf-8")
    body_tex = (ROOT / "paper/latex/bmc_work/bmc_body.tex").read_text(encoding="utf-8")
    supplement_tex = (ROOT / "paper/latex/bmc_work/main_bmc_supplement.tex").read_text(encoding="utf-8")
    table_tex = (ROOT / "paper/latex/bmc_work/bmc_stratified_derangement_table.tex").read_text(encoding="utf-8")
    body_flat = " ".join(body_tex.split())
    required_claims = {
        "abstract_effect": "$+5.63$ Dice points" in main_tex and "$+4.71$ to $+6.67$" in main_tex,
        "methods_control": "post hoc frozen-checkpoint" in body_tex and "3,413 of 3,426" in body_tex,
        "results_effect": "$+5.63$ Dice pp" in body_tex and "$+5.87$ pp" in body_tex,
        "covid_cluster": "$+22.42$ pp" in body_tex and "$+21.12$ to $+23.80$" in body_tex,
        "supplement_input": "bmc_stratified_derangement_table.tex" in supplement_tex,
        "table_public5": "3413/3426" in table_tex and "$+5.63$ [$+4.71,+6.67$]" in table_tex,
        "claim_boundary": "does not establish clinical-report understanding" in body_flat,
    }
    for name, condition in required_claims.items():
        check(name, condition, condition)

    payload = {
        "status": "PASS",
        "protocol_hash": PROTOCOL_HASH,
        "checks": checks,
        "matrix_audit_sha256": "6d473eee3081bc498c8068b599d4bdca3ba763e38a0ad767603513059b94ac42",
        "analysis_code_sha256": "5c18f19644e6ce29982cde54f1cf576d62414c2474f3b39d939f45d5f9ca8738",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "checks": len(checks), "output": str(OUTPUT)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
