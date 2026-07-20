#!/usr/bin/env python3
"""Preflight tests for the Public-5 stratified prompt control maps."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


DATASETS = (
    "medclipseg_busi",
    "medclipseg_clinicdb",
    "medclipseg_busbra",
    "medclipseg_brisc",
    "medclipseg_covid19",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("<PROJECT_ROOT>"))
    args = parser.parse_args()
    sys.path.insert(0, str(args.root / "smoke_tests"))

    from predict_protocol_v3 import apply_stratified_control_map
    from protocol_v3.core import load_manifest_splits

    total_cases = 0
    changed_cases = 0
    unchanged_cases = 0
    self_cases = 0
    for dataset in DATASETS:
        manifest = args.root / "smoke_tests" / "protocol_v3" / "manifests" / f"{dataset}_full.csv"
        control_map = (
            args.root
            / "paper"
            / "results"
            / "protocol_v3_stratified_derangement_20260719"
            / "control_maps"
            / f"{dataset}_stratified_derangement_v1.csv"
        )
        splits = load_manifest_splits(manifest, [dataset], require_train_val_test=True, check_files=True)
        rows, meta = apply_stratified_control_map(splits, dataset, control_map)
        assert len(rows) == len(splits["test"])
        assert meta["source_self_cases"] == 0
        total_cases += len(rows)
        changed_cases += int(meta["text_changed_cases"])
        unchanged_cases += int(meta["text_unchanged_cases"])
        self_cases += int(meta["source_self_cases"])

    assert total_cases == 3426, total_cases
    assert changed_cases == 3413, changed_cases
    assert unchanged_cases == 13, unchanged_cases
    assert self_cases == 0, self_cases
    print(
        f"PASS datasets={len(DATASETS)} cases={total_cases} "
        f"text_changed={changed_cases} unavoidable_unchanged={unchanged_cases} source_self={self_cases}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
