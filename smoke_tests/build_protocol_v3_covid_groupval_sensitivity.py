#!/usr/bin/env python3
"""Build a COVID-19 train/val subject-grouped sensitivity manifest.

The original Protocol V3 test rows are preserved byte-for-byte at the field
level. Only rows currently assigned to train or validation are regrouped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd


SUBJECT_PATTERN = re.compile(r"sub-([^_]+)")
FINGERPRINT_COLUMNS = (
    "case_id",
    "image_path",
    "mask_path",
    "text",
    "image_sha256",
    "mask_sha256",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=123)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    return parser.parse_args()


def recover_subject(case_id: str) -> str:
    match = SUBJECT_PATTERN.search(str(case_id))
    return match.group(1) if match else ""


def group_key(row: pd.Series) -> str:
    subject = str(row["patient_id"] or "").strip()
    return f"patient:{subject}" if subject else f"case:{row['case_id']}"


def hash_fraction(seed: int, value: str) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def frame_fingerprint(frame: pd.DataFrame) -> str:
    ordered = frame.loc[:, FINGERPRINT_COLUMNS].sort_values("case_id")
    payload = ordered.to_json(orient="records", force_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def split_patient_sets(frame: pd.DataFrame) -> dict[str, set[str]]:
    return {
        split: set(group["patient_id"].astype(str).str.strip()) - {""}
        for split, group in frame.groupby("split")
    }


def overlap_counts(patient_sets: dict[str, set[str]]) -> dict[str, int]:
    pairs = (("train", "val"), ("train", "test"), ("val", "test"))
    return {
        f"{left}_{right}": len(patient_sets.get(left, set()) & patient_sets.get(right, set()))
        for left, right in pairs
    }


def main() -> None:
    args = parse_args()
    if not 0.0 < args.val_fraction < 1.0:
        raise ValueError("--val-fraction must be between zero and one")

    original = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    if set(original["split"]) != {"train", "val", "test"}:
        raise ValueError("input manifest must contain train, val, and test")
    if original["case_id"].duplicated().any():
        raise ValueError("duplicate case_id in input manifest")

    updated = original.copy()
    recovered = updated["case_id"].map(recover_subject)
    updated.loc[recovered.ne(""), "patient_id"] = recovered[recovered.ne("")]

    original_with_ids = original.copy()
    original_with_ids.loc[recovered.ne(""), "patient_id"] = recovered[recovered.ne("")]
    original_sets = split_patient_sets(original_with_ids)

    trainval_mask = updated["split"].isin(("train", "val"))
    groups = updated.loc[trainval_mask].apply(group_key, axis=1)
    updated.loc[trainval_mask, "split"] = [
        "val" if hash_fraction(args.split_seed, key) < args.val_fraction else "train"
        for key in groups
    ]

    original_test = original[original["split"] == "test"]
    updated_test = updated[updated["split"] == "test"]
    if frame_fingerprint(original_test) != frame_fingerprint(updated_test):
        raise ValueError("test rows changed while building sensitivity manifest")

    updated_sets = split_patient_sets(updated)
    updated_overlap = overlap_counts(updated_sets)
    if any(updated_overlap.values()):
        raise ValueError(f"recoverable patient leakage remains: {updated_overlap}")

    for field in ("case_id", "image_path", "mask_path", "image_sha256", "mask_sha256"):
        if updated[field].duplicated().any():
            raise ValueError(f"duplicate {field} in output manifest")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    updated.to_csv(args.output, index=False, lineterminator="\n")

    counts = updated["split"].value_counts().to_dict()
    trainval_total = counts["train"] + counts["val"]
    audit = {
        "input": str(args.input),
        "output": str(args.output),
        "split_seed": args.split_seed,
        "target_val_fraction": args.val_fraction,
        "rows": len(updated),
        "split_counts": counts,
        "realized_trainval_val_fraction": counts["val"] / trainval_total,
        "recoverable_patient_rows": int(recovered.ne("").sum()),
        "recoverable_patients": int(recovered[recovered.ne("")].nunique()),
        "unknown_patient_rows": int(recovered.eq("").sum()),
        "original_patient_overlap": overlap_counts(original_sets),
        "sensitivity_patient_overlap": updated_overlap,
        "changed_trainval_rows": int(
            (original.loc[trainval_mask, "split"] != updated.loc[trainval_mask, "split"]).sum()
        ),
        "original_test_rows": len(original_test),
        "test_fingerprint": frame_fingerprint(updated_test),
        "test_unchanged": True,
    }
    args.audit_json.parent.mkdir(parents=True, exist_ok=True)
    args.audit_json.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
