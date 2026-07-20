#!/usr/bin/env python3
"""Audit the 90 factorial training cells and their 360 prompt-control exports."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path


PROTOCOL_ID = "MEDSEG_TEXT_V3_20260710"
PROTOCOL_HASH = "abb8ccb2d46cf039a4a0c573d733e3bc510636e6dfa7e59032508ab96fc0f718"
DATASETS = (
    "medclipseg_busi",
    "medclipseg_clinicdb",
    "medclipseg_busbra",
    "medclipseg_brisc",
    "medclipseg_covid19",
)
SEEDS = (123, 456, 789)
CONTROLS = ("true", "shuffled", "fixed", "empty")
RUNS = {
    "V3_ABL_BASE": ("default", "lcaug_v2_hflip_dataset"),
    "V3_ABL_BIOMED": ("biomed_lcaug", "lcaug_v2_hflip_dataset"),
    "V3_ABL_ATCONV": ("default_atconv4", "lcaug_v2_hflip_dataset"),
    "V3_ABL_BIOMED_ATCONV": ("biomed_lcaug_atconv4", "lcaug_v2_hflip_dataset"),
    "V3_ABL_SHARED_NR": (
        "biomed_lcaug_v2_atconv4",
        "lcaug_v2_dynamic_shared_plan_no_text_rewrite_dataset",
    ),
    "V3_ABL_EQUIPROMPT": (
        "biomed_lcaug_v2_atconv4",
        "lcaug_v2_dynamic_shared_plan_dataset",
    ),
}
ATCONV_RUNS = {
    "V3_ABL_ATCONV",
    "V3_ABL_BIOMED_ATCONV",
    "V3_ABL_SHARED_NR",
    "V3_ABL_EQUIPROMPT",
}
EXPECTED_ATCONV_TARGETS = {
    "base_model.proj.vis.1.0",
    "base_model.proj_ad.vis.1.0",
    "base_model.neck_ad.coordconv.1.0",
    "base_model.neck_ad.f4_proj3.0",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("<PROJECT_ROOT>"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("<PROJECT_ROOT>/paper/results/bmc_submission_audit_20260719/factorial_run_lineage.json"),
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def last_match(pattern: str, text: str) -> str:
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    return matches[-1] if matches else ""


def main() -> int:
    args = parse_args()
    log_root = args.root / "logs" / "protocol_v3" / PROTOCOL_HASH
    failures: list[dict[str, object]] = []
    warnings: list[dict[str, object]] = []
    code_hashes: set[str] = set()
    commits: set[str] = set()
    manifest_hashes: dict[str, set[str]] = defaultdict(set)
    cache_hashes: set[str] = set()
    checkpoint_hashes: set[str] = set()
    recomputed_checkpoint_hashes: set[str] = set()
    run_config_hashes: set[str] = set()
    training_count = 0
    control_count = 0
    complete_training_logs = 0
    atconv_registered_cells = 0
    standard_conv_cells = 0
    training_log_exceptions: list[dict[str, object]] = []

    def fail(cell: str, issue: str, evidence: object = None) -> None:
        failures.append({"cell": cell, "issue": issue, "evidence": evidence})

    for run_id, (recipe, augmentation) in RUNS.items():
        for dataset in DATASETS:
            for seed in SEEDS:
                training_count += 1
                cell = f"{run_id}/{dataset}/seed{seed}"
                directory = log_root / run_id / dataset / f"seed{seed}"
                meta_path = directory / "run_meta.json"
                train_log = directory / "train.log"
                if not meta_path.is_file():
                    fail(cell, "missing run_meta.json")
                    continue
                if not train_log.is_file():
                    fail(cell, "missing train.log")
                    continue

                meta = load_json(meta_path)
                expected = {
                    "protocol_id": PROTOCOL_ID,
                    "protocol_hash": PROTOCOL_HASH,
                    "status": "complete",
                    "recipe": recipe,
                    "augmentation": augmentation,
                    "epochs": 100,
                }
                for key, value in expected.items():
                    if meta.get(key) != value:
                        fail(cell, f"run_meta mismatch: {key}", {"expected": value, "actual": meta.get(key)})
                if meta.get("task") != {"run_id": run_id, "dataset": dataset, "seed": seed}:
                    fail(cell, "task identity mismatch", meta.get("task"))

                for key in ("code_sha256", "git_commit", "manifest_sha256", "run_config_sha256", "checkpoint_sha256"):
                    if not meta.get(key):
                        fail(cell, f"missing {key}")
                code_hashes.add(meta.get("code_sha256", ""))
                commits.add(meta.get("git_commit", ""))
                manifest_hashes[dataset].add(meta.get("manifest_sha256", ""))
                run_config_hashes.add(meta.get("run_config_sha256", ""))
                checkpoint_hashes.add(meta.get("checkpoint_sha256", ""))
                if meta.get("cache_sha256"):
                    cache_hashes.add(meta["cache_sha256"])
                checkpoint = Path(meta.get("checkpoint", ""))
                if not checkpoint.is_file():
                    fail(cell, "checkpoint path missing", str(checkpoint))

                if checkpoint.is_file():
                    actual_checkpoint_hash = file_sha256(checkpoint)
                    recomputed_checkpoint_hashes.add(actual_checkpoint_hash)
                    if actual_checkpoint_hash != meta.get("checkpoint_sha256"):
                        fail(
                            cell,
                            "checkpoint SHA256 mismatch",
                            {"recorded": meta.get("checkpoint_sha256"), "actual": actual_checkpoint_hash},
                        )

                raw_log = train_log.read_bytes()
                train_text = raw_log.decode("utf-8", errors="replace")
                first_line = train_text.splitlines()[0].strip() if train_text.splitlines() else ""
                required_tokens = (
                    "--epochs 100",
                    "--batch-size 8",
                    "--workers 2",
                    f"--seed {seed}",
                    "--split-seed 123",
                    "--scheduler cosine",
                    "--min-lr 1e-6",
                    f"--causal-recipe {recipe}",
                    f"--aug-strength {augmentation}",
                )
                missing_tokens = [token for token in required_tokens if token not in first_line]
                if missing_tokens:
                    fail(cell, "training command mismatch", missing_tokens)

                conv_plugin = last_match(r"^conv_plugin:\s*(\S+)\s*$", train_text)
                atconv_layers = last_match(r"^atconv_layers:\s*(\d+)\s*$", train_text)
                atconv_targets = last_match(r"^atconv_targets:\s*(.*)$", train_text)
                if run_id in ATCONV_RUNS:
                    actual_targets = {target for target in atconv_targets.split("|") if target}
                    if conv_plugin != "atconv":
                        fail(cell, "ATConv-designated run did not log conv_plugin=atconv", conv_plugin)
                    if atconv_layers != "4":
                        fail(cell, "ATConv-designated run did not register four replacements", atconv_layers)
                    if actual_targets != EXPECTED_ATCONV_TARGETS:
                        fail(
                            cell,
                            "ATConv registered targets mismatch",
                            {"expected": sorted(EXPECTED_ATCONV_TARGETS), "actual": sorted(actual_targets)},
                        )
                    if (
                        conv_plugin == "atconv"
                        and atconv_layers == "4"
                        and actual_targets == EXPECTED_ATCONV_TARGETS
                    ):
                        atconv_registered_cells += 1
                else:
                    if conv_plugin != "standard":
                        fail(cell, "standard-conv control did not log conv_plugin=standard", conv_plugin)
                    else:
                        standard_conv_cells += 1

                epochs = [int(value) for value in re.findall(r"^epoch=(\d+)\b", train_text, flags=re.MULTILINE)]
                declared_checkpoints = re.findall(r"^checkpoint:\s*(\S+)", train_text, flags=re.MULTILINE)
                best_checkpoints = re.findall(r"^best_checkpoint:\s*(\S+)", train_text, flags=re.MULTILINE)
                final_statuses = re.findall(r"^final_status:\s*(\S+)", train_text, flags=re.MULTILINE)
                log_issues: list[str] = []
                nul_bytes = raw_log.count(b"\x00")
                if nul_bytes:
                    log_issues.append(f"contains {nul_bytes} NUL bytes")
                if epochs != list(range(1, 101)):
                    log_issues.append(
                        f"epoch sequence is not exactly 1..100 (count={len(epochs)}, first={epochs[:3]}, last={epochs[-3:]})"
                    )
                if len(declared_checkpoints) != 1 or declared_checkpoints[0] != str(checkpoint):
                    log_issues.append(
                        f"declared checkpoint mismatch (logged={declared_checkpoints}, run_meta={checkpoint})"
                    )
                if len(best_checkpoints) != 1 or best_checkpoints[0] != str(checkpoint):
                    log_issues.append(
                        f"best checkpoint mismatch (logged={best_checkpoints}, run_meta={checkpoint})"
                    )
                if final_statuses != ["PASS"]:
                    log_issues.append(f"final_status entries are {final_statuses}")
                if log_issues:
                    training_log_exceptions.append(
                        {"cell": cell, "issues": log_issues, "checkpoint_sha256": meta.get("checkpoint_sha256")}
                    )
                else:
                    complete_training_logs += 1

                for control in CONTROLS:
                    control_count += 1
                    control_dir = directory / "controls" / control
                    required_files = (
                        "run_meta.json",
                        "per_case.csv",
                        "prediction_index.csv",
                        "summary.csv",
                        "predict.log",
                        "evaluate.log",
                    )
                    missing = [name for name in required_files if not (control_dir / name).is_file()]
                    if missing:
                        fail(f"{cell}/controls/{control}", "missing control artifacts", missing)
                        continue
                    control_meta = load_json(control_dir / "run_meta.json")
                    control_expected = {
                        "protocol_id": PROTOCOL_ID,
                        "protocol_hash": PROTOCOL_HASH,
                        "dataset": dataset,
                        "prompt_control": control,
                        "prediction_threshold": 0.5,
                        "manifest_sha256": meta.get("manifest_sha256"),
                        "checkpoint_sha256": meta.get("checkpoint_sha256"),
                    }
                    for key, value in control_expected.items():
                        if control_meta.get(key) != value:
                            fail(
                                f"{cell}/controls/{control}",
                                f"control run_meta mismatch: {key}",
                                {"expected": value, "actual": control_meta.get(key)},
                            )

    if len(code_hashes - {""}) != 1:
        fail("matrix", "multiple training code hashes", sorted(code_hashes))
    if len(commits - {""}) != 1:
        fail("matrix", "multiple git commits", sorted(commits))
    for dataset, hashes in manifest_hashes.items():
        if len(hashes - {""}) != 1:
            fail(dataset, "multiple manifest hashes", sorted(hashes))
    if len(checkpoint_hashes - {""}) != training_count:
        fail("matrix", "checkpoint hashes are missing or duplicated", len(checkpoint_hashes - {""}))
    if recomputed_checkpoint_hashes != checkpoint_hashes - {""}:
        fail(
            "matrix",
            "recomputed checkpoint hash set differs from recorded hash set",
            {
                "recorded": len(checkpoint_hashes - {""}),
                "recomputed": len(recomputed_checkpoint_hashes),
            },
        )
    if len(run_config_hashes - {""}) != training_count:
        fail("matrix", "run-config hashes are missing or duplicated", len(run_config_hashes - {""}))
    if atconv_registered_cells != 60:
        fail("matrix", "unexpected ATConv-registered cell count", atconv_registered_cells)
    if standard_conv_cells != 30:
        fail("matrix", "unexpected standard-conv cell count", standard_conv_cells)

    # This is an intentional disclosure, not a failed execution-lineage check.
    warnings.append(
        {
            "issue": "protocol hash does not encode the complete training policy",
            "detail": "batch size, workers, optimizer/scheduler details, model id, and the R11 policy amendment are preserved in command logs but are not all members of run_config_sha256",
        }
    )
    warnings.append(
        {
            "issue": "internal ATConv recipe names retain the atconv4 lineage label",
            "detail": "runtime hooks show two of four registered replacements are forward-active; manuscripts must describe the effective two-projector graph",
        }
    )

    if failures:
        status = "FAIL"
    elif training_log_exceptions:
        status = "PASS_WITH_DISCLOSED_DEVIATIONS_AND_LOG_EXCEPTIONS"
    else:
        status = "PASS_WITH_DISCLOSED_DEVIATIONS"
    payload = {
        "status": status,
        "protocol_id": PROTOCOL_ID,
        "protocol_hash": PROTOCOL_HASH,
        "expected_training_cells": 90,
        "audited_training_cells": training_count,
        "expected_prompt_control_exports": 360,
        "audited_prompt_control_exports": control_count,
        "code_sha256": sorted(code_hashes - {""}),
        "git_commits": sorted(commits - {""}),
        "manifest_sha256_by_dataset": {key: sorted(value - {""}) for key, value in manifest_hashes.items()},
        "cache_sha256": sorted(cache_hashes),
        "unique_checkpoint_hashes": len(checkpoint_hashes - {""}),
        "recomputed_checkpoint_hashes": len(recomputed_checkpoint_hashes),
        "unique_run_config_hashes": len(run_config_hashes - {""}),
        "complete_training_logs": complete_training_logs,
        "training_log_exceptions": training_log_exceptions,
        "atconv_registered_cells": atconv_registered_cells,
        "standard_conv_cells": standard_conv_cells,
        "failures": failures,
        "warnings": warnings,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "failures": len(failures), "output": str(args.output)}))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
