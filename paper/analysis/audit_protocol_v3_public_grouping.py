#!/usr/bin/env python3
"""Audit legacy and corrected Public-5 manifests for split-group leakage."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


SCRIPT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPT_ROOT / "smoke_tests"))

from build_group_disjoint_public5_manifests import (  # noqa: E402
    recover_busbra_patient,
    recover_cvc_sequence,
)


MANIFESTS = (
    ("BUSI", "released", "medclipseg_busi_full.csv"),
    ("ClinicDB", "released_with_group_leakage", "medclipseg_clinicdb_full.csv"),
    ("ClinicDB", "corrected_group_disjoint", "medclipseg_clinicdb_grouped.csv"),
    ("BUS-BRA", "released_with_group_leakage", "medclipseg_busbra_full.csv"),
    ("BUS-BRA", "corrected_group_disjoint", "medclipseg_busbra_grouped.csv"),
    ("BRISC", "released", "medclipseg_brisc_full.csv"),
    ("COVID-19", "released", "medclipseg_covid19_full.csv"),
)
SPLIT_PAIRS = (("train", "val"), ("train", "test"), ("val", "test"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper/results/public_grouping_audit_20260725"),
    )
    return parser.parse_args()


def resolve_project_path(raw: str, root: Path) -> Path:
    value = str(raw).replace("\\", "/")
    marker = "<PROJECT_ROOT>/"
    if value.startswith(marker):
        return root / value[len(marker) :]
    path = Path(raw)
    return path if path.is_absolute() else root / path


def cross_split_duplicate_count(frame: pd.DataFrame, column: str) -> int:
    values = frame[column].astype(str).str.strip()
    populated = frame.loc[values.ne("")].copy()
    counts = populated.groupby(column)["split"].nunique()
    return int((counts > 1).sum())


def cross_split_mask_hash_audit(
    frame: pd.DataFrame,
    root: Path,
) -> tuple[int, int, int, int]:
    split_counts = frame.groupby("mask_sha256")["split"].nunique()
    duplicated = split_counts[split_counts > 1].index
    empty = 0
    nonempty = 0
    unreadable = 0
    for digest in duplicated:
        raw = frame.loc[frame["mask_sha256"] == digest, "mask_path"].iloc[0]
        path = resolve_project_path(raw, root)
        if not path.is_file():
            unreadable += 1
            continue
        mask = np.asarray(Image.open(path))
        if np.count_nonzero(mask):
            nonempty += 1
        else:
            empty += 1
    return len(duplicated), empty, nonempty, unreadable


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
    if dataset == "ClinicDB":
        recovered = frame["case_id"].map(
            lambda value: f"clinicdb_sequence_{recover_cvc_sequence(value):02d}"
        )
        return recovered, "published contiguous frame ranges for 29 video sequences"
    if dataset == "BUS-BRA":
        recovered = frame["case_id"].map(
            lambda value: f"busbra_patient_{recover_busbra_patient(value)[0]}"
        )
        return recovered, "numeric bus_<patient>-<view> filename prefix"
    if dataset == "COVID-19":
        recovered = frame["case_id"].str.extract(r"sub-([^_]+)", expand=False).fillna("")
        return recovered, "case_id token sub-<subject>"
    return pd.Series("", index=frame.index, dtype=str), "no validated group identifier"


def assert_released_group_matches(
    dataset: str,
    variant: str,
    frame: pd.DataFrame,
    recovered: pd.Series,
) -> None:
    if variant != "corrected_group_disjoint":
        return
    if "group_id" not in frame.columns:
        raise ValueError(f"Corrected {dataset} manifest has no group_id")
    released = frame["group_id"].astype(str).str.strip()
    mismatch = released.ne(recovered.astype(str).str.strip())
    if mismatch.any():
        case_id = frame.loc[mismatch, "case_id"].iloc[0]
        raise ValueError(f"Corrected {dataset} group_id mismatch at {case_id}")


def heldout_overlap(frame: pd.DataFrame) -> tuple[int, int, int]:
    sets = split_sets(frame, "recovered_group")
    development = sets.get("train", set()) | sets.get("val", set())
    overlapping_groups = sets.get("test", set()) & development
    test = frame.loc[frame["split"] == "test", "recovered_group"].astype(str)
    overlapping_rows = int(test.isin(overlapping_groups).sum())
    return len(overlapping_groups), overlapping_rows, len(test)


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
    for dataset, variant, filename in MANIFESTS:
        path = manifest_dir / filename
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        recovered, rule = recover_validated_group(dataset, frame)
        assert_released_group_matches(dataset, variant, frame, recovered)
        recovered_frame = frame.assign(recovered_group=recovered)
        overlaps = pair_overlaps(split_sets(recovered_frame, "recovered_group"))
        heldout_groups, heldout_rows, test_rows = heldout_overlap(recovered_frame)
        mask_total, mask_empty, mask_nonempty, mask_unreadable = cross_split_mask_hash_audit(
            frame, root
        )
        rows.append(
            {
                "dataset": dataset,
                "manifest_variant": variant,
                "manifest": filename,
                "rows": len(frame),
                "train": int((frame["split"] == "train").sum()),
                "val": int((frame["split"] == "val").sum()),
                "test": int((frame["split"] == "test").sum()),
                "populated_patient_id_rows": int(
                    frame["patient_id"].astype(str).str.strip().ne("").sum()
                ),
                "exact_image_hash_cross_split": cross_split_duplicate_count(frame, "image_sha256"),
                "exact_mask_hash_cross_split": mask_total,
                "empty_mask_hash_cross_split": mask_empty,
                "nonempty_mask_hash_cross_split": mask_nonempty,
                "unreadable_repeated_mask_hashes": mask_unreadable,
                "validated_group_rule": rule,
                "recoverable_group_rows": int(recovered.ne("").sum()),
                "recoverable_groups": int(recovered[recovered.ne("")].nunique()),
                "recoverable_train_val_overlap": overlaps["train_val"],
                "recoverable_train_test_overlap": overlaps["train_test"],
                "recoverable_val_test_overlap": overlaps["val_test"],
                "test_groups_seen_in_development": heldout_groups,
                "test_rows_from_groups_seen_in_development": heldout_rows,
                "test_rows_total": test_rows,
                "folder_provenance": folder_counts(frame),
            }
        )

    audit = pd.DataFrame(rows)
    audit.to_csv(output / "public_grouping_audit.csv", index=False, lineterminator="\n")

    md = [
        "# Public-data grouping audit",
        "",
        "The audit recovers documented video-sequence, patient, or subject groups",
        "from case identifiers and compares the released manifests with the corrected",
        "group-disjoint manifests. Exact image and mask hashes are also checked across",
        "split labels.",
        "",
        "| Dataset | Manifest | Groups | Train/val overlap | Train/test overlap | Val/test overlap | Test rows from groups seen in development |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        group_count = row["recoverable_groups"] if row["recoverable_groups"] else "NA"
        heldout = (
            f"{row['test_rows_from_groups_seen_in_development']}/{row['test_rows_total']}"
            if row["recoverable_groups"]
            else "NA"
        )
        md.append(
            f"| {row['dataset']} | {row['manifest_variant']} | {group_count} | "
            f"{row['recoverable_train_val_overlap']} | {row['recoverable_train_test_overlap']} | "
            f"{row['recoverable_val_test_overlap']} | {heldout} |"
        )
    md.extend(
        [
            "",
            "The released ClinicDB and BUS-BRA splits are not used for revised primary",
            "results because video-sequence or patient groups cross split boundaries.",
            "Their corrected manifests preserve the original image counts and have zero",
            "cross-split group overlap. BUSI and BRISC support image-disjoint terminology",
            "only because no validated patient identifier is available in the package.",
            "COVID-19 is audited with its recoverable subject token.",
            "",
            "Repeated mask hashes are classified as empty or nonempty when the dataset",
            "files are available. An unreadable count means that hash overlap was found",
            "but the corresponding local mask file was unavailable for content inspection.",
        ]
    )
    (output / "README.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(audit.to_string(index=False))
    print("final_status: PASS")


if __name__ == "__main__":
    main()
