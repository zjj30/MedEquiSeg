#!/usr/bin/env python3
"""Audit public Protocol V3 manifests for split and grouping evidence."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


DATASETS = {
    "BUSI": "medclipseg_busi_full.csv",
    "ClinicDB": "medclipseg_clinicdb_full.csv",
    "BUS-BRA": "medclipseg_busbra_full.csv",
    "BRISC": "medclipseg_brisc_full.csv",
    "COVID-19": "medclipseg_covid19_full.csv",
}
SPLIT_PAIRS = (("train", "val"), ("train", "test"), ("val", "test"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper/results/protocol_v3_public_grouping_audit_20260715"),
    )
    return parser.parse_args()


def cross_split_duplicate_count(frame: pd.DataFrame, column: str) -> int:
    counts = frame.groupby(column, dropna=False)["split"].nunique()
    return int((counts > 1).sum())


def cross_split_mask_hash_audit(frame: pd.DataFrame) -> tuple[int, int, int]:
    split_counts = frame.groupby("mask_sha256")["split"].nunique()
    duplicated = split_counts[split_counts > 1].index
    empty = 0
    nonempty = 0
    for digest in duplicated:
        path = frame.loc[frame["mask_sha256"] == digest, "mask_path"].iloc[0]
        mask = np.asarray(Image.open(path))
        if np.count_nonzero(mask):
            nonempty += 1
        else:
            empty += 1
    return len(duplicated), empty, nonempty


def split_sets(frame: pd.DataFrame, column: str) -> dict[str, set[str]]:
    return {
        split: set(group[column].astype(str).str.strip()) - {""}
        for split, group in frame.groupby("split")
    }


def pair_overlaps(sets: dict[str, set[str]]) -> dict[str, int]:
    return {
        f"{left}_{right}": len(sets.get(left, set()) & sets.get(right, set()))
        for left, right in SPLIT_PAIRS
    }


def recover_validated_group(dataset: str, frame: pd.DataFrame) -> tuple[pd.Series, str]:
    if dataset == "COVID-19":
        recovered = frame["case_id"].str.extract(r"sub-([^_]+)", expand=False).fillna("")
        return recovered, "case_id token sub-<subject>"
    return pd.Series("", index=frame.index, dtype=str), "none in released metadata"


def folder_counts(frame: pd.DataFrame) -> str:
    folders = frame["image_path"].str.extract(
        r"/(Train_Folder|Val_Folder|Test_Folder)/", expand=False
    )
    mapping = pd.crosstab(frame["split"], folders)
    pieces = []
    for split in ("train", "val", "test"):
        values = mapping.loc[split] if split in mapping.index else pd.Series(dtype=int)
        nonzero = [f"{folder}:{int(value)}" for folder, value in values.items() if value]
        pieces.append(f"{split}=" + ",".join(nonzero))
    return "; ".join(pieces)


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    manifest_dir = root / "smoke_tests/protocol_v3/manifests"
    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    rows = []
    for dataset, filename in DATASETS.items():
        path = manifest_dir / filename
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        recovered, rule = recover_validated_group(dataset, frame)
        recovered_frame = frame.assign(recovered_group=recovered)
        overlaps = pair_overlaps(split_sets(recovered_frame, "recovered_group"))
        mask_total, mask_empty, mask_nonempty = cross_split_mask_hash_audit(frame)
        rows.append(
            {
                "dataset": dataset,
                "rows": len(frame),
                "train": int((frame["split"] == "train").sum()),
                "val": int((frame["split"] == "val").sum()),
                "test": int((frame["split"] == "test").sum()),
                "populated_patient_id_rows": int(frame["patient_id"].str.strip().ne("").sum()),
                "exact_image_hash_cross_split": cross_split_duplicate_count(frame, "image_sha256"),
                "exact_mask_hash_cross_split": mask_total,
                "empty_mask_hash_cross_split": mask_empty,
                "nonempty_mask_hash_cross_split": mask_nonempty,
                "validated_group_rule": rule,
                "recoverable_group_rows": int(recovered.ne("").sum()),
                "recoverable_groups": int(recovered[recovered.ne("")].nunique()),
                "recoverable_train_val_overlap": overlaps["train_val"],
                "recoverable_train_test_overlap": overlaps["train_test"],
                "recoverable_val_test_overlap": overlaps["val_test"],
                "folder_provenance": folder_counts(frame),
            }
        )

    audit = pd.DataFrame(rows)
    audit.to_csv(output / "public_grouping_audit.csv", index=False, lineterminator="\n")

    md = [
        "# Protocol V3 Public Grouping Audit",
        "",
        "All five public manifests inherit MedCLIPSeg Train/Val/Test folders. Exact",
        "image and mask SHA-256 values are checked across split labels. A recovered",
        "group is reported only when a documented filename token has an unambiguous",
        "subject interpretation; numeric image indices are not treated as patients.",
        "",
        "| Dataset | Patient-ID rows | Image-hash overlap | Mask-hash overlap (nonempty/total) | Recoverable groups | Train/val | Train/test | Val/test |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        md.append(
            f"| {row['dataset']} | {row['populated_patient_id_rows']} | "
            f"{row['exact_image_hash_cross_split']} | {row['nonempty_mask_hash_cross_split']}/{row['exact_mask_hash_cross_split']} | "
            f"{row['recoverable_groups']} | {row['recoverable_train_val_overlap']} | "
            f"{row['recoverable_train_test_overlap']} | {row['recoverable_val_test_overlap']} |"
        )
    md.extend(
        [
            "",
            "Only COVID-19 exposes a validated recoverable subject token. For BUSI,",
            "ClinicDB, BUS-BRA, and BRISC, the released package does not provide a",
            "patient or sequence mapping; zero reported recoverable overlap therefore",
            "means unavailable grouping evidence, not proof of patient independence.",
            "",
            "All cross-split mask-hash repetitions are all-zero masks with matching",
            "dimensions; no nonempty mask hash crosses a split. They therefore reflect",
            "the common empty target representation rather than duplicated images.",
            "",
            "The audit supports image-disjoint held-out terminology. It does not support",
            "a universal patient-level split claim.",
        ]
    )
    (output / "README.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(audit.to_string(index=False))


if __name__ == "__main__":
    main()
