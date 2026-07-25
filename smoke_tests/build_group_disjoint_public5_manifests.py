#!/usr/bin/env python3
"""Build group-disjoint ClinicDB and BUS-BRA manifests.

ClinicDB frames are grouped by the documented 29-sequence frame ranges.
BUS-BRA images are grouped by the numeric patient prefix in each case id.  The
builder preserves the original image-count targets for train/validation/test,
changes no image, mask, prompt, or hash field, and emits a machine-readable
audit of every assigned group.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import itertools
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = ROOT / "smoke_tests/protocol_v3/manifests"
DEFAULT_AUDIT_DIR = ROOT / "paper/results/protocol_grouped_split_20260725"

CVC_SEQUENCE_ENDS = (
    25,
    50,
    67,
    78,
    103,
    126,
    151,
    177,
    199,
    205,
    227,
    252,
    277,
    297,
    317,
    342,
    363,
    383,
    408,
    428,
    447,
    466,
    478,
    503,
    528,
    546,
    571,
    591,
    612,
)
CVC_CASE_PATTERN = re.compile(r":(\d+)$")
BUSBRA_CASE_PATTERN = re.compile(r":bus_(\d+)-([a-z]+)$", re.IGNORECASE)
UNCHANGED_COLUMNS = (
    "dataset",
    "case_id",
    "image_path",
    "mask_path",
    "text",
    "mask_mode",
    "prompt_source",
    "image_sha256",
    "mask_sha256",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clinicdb-input",
        type=Path,
        default=MANIFEST_DIR / "medclipseg_clinicdb_full.csv",
    )
    parser.add_argument(
        "--busbra-input",
        type=Path,
        default=MANIFEST_DIR / "medclipseg_busbra_full.csv",
    )
    parser.add_argument(
        "--clinicdb-output",
        type=Path,
        default=MANIFEST_DIR / "medclipseg_clinicdb_grouped.csv",
    )
    parser.add_argument(
        "--busbra-output",
        type=Path,
        default=MANIFEST_DIR / "medclipseg_busbra_grouped.csv",
    )
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--split-seed", type=int, default=123)
    return parser.parse_args()


def read_manifest(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    if len({row["case_id"] for row in rows}) != len(rows):
        raise ValueError(f"Duplicate case_id in {path}")
    return fields, rows


def write_manifest(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output_fields = list(fields)
    for field in ("group_id", "group_type"):
        if field not in output_fields:
            output_fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_path(path: Path) -> str:
    """Return a release-safe path without embedding a developer workstation."""
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(ROOT.resolve())
    except ValueError:
        return str(resolved)
    return f"<PROJECT_ROOT>/{relative.as_posix()}"


def rows_fingerprint(rows: Iterable[dict[str, str]]) -> str:
    material = [
        {field: row.get(field, "") for field in UNCHANGED_COLUMNS}
        for row in sorted(rows, key=lambda item: item["case_id"])
    ]
    payload = json.dumps(material, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def stable_rank(seed: int, *parts: object) -> int:
    payload = ":".join([str(seed), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:16], "big")


def split_overlap(rows: list[dict[str, str]], field: str) -> dict[str, int]:
    sets = {
        split: {row[field] for row in rows if row["split"] == split and row.get(field, "")}
        for split in ("train", "val", "test")
    }
    return {
        "train_val": len(sets["train"] & sets["val"]),
        "train_test": len(sets["train"] & sets["test"]),
        "val_test": len(sets["val"] & sets["test"]),
    }


def split_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counts = Counter(row["split"] for row in rows)
    return {split: counts[split] for split in ("train", "val", "test")}


def split_group_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    return {
        split: len({row["group_id"] for row in rows if row["split"] == split})
        for split in ("train", "val", "test")
    }


def recover_cvc_sequence(case_id: str) -> int:
    match = CVC_CASE_PATTERN.search(str(case_id))
    if not match:
        raise ValueError(f"Unrecognized ClinicDB case id: {case_id}")
    frame = int(match.group(1))
    if not 1 <= frame <= CVC_SEQUENCE_ENDS[-1]:
        raise ValueError(f"ClinicDB frame outside documented range: {frame}")
    return bisect.bisect_left(CVC_SEQUENCE_ENDS, frame) + 1


def choose_sequence_holdouts(
    group_sizes: dict[str, int],
    *,
    target_images: int,
    target_groups: int,
    seed: int,
) -> tuple[set[str], set[str]]:
    combinations = list(itertools.combinations(sorted(group_sizes), target_groups))
    if not combinations:
        raise ValueError("No ClinicDB sequence combinations available")

    def ranked(purpose: str, excluded: set[str]) -> list[tuple[str, ...]]:
        candidates = [combo for combo in combinations if not (set(combo) & excluded)]
        return sorted(
            candidates,
            key=lambda combo: (
                abs(sum(group_sizes[group] for group in combo) - target_images),
                stable_rank(seed, "clinicdb", purpose, *combo),
            ),
        )

    test = ranked("test", set())[0]
    val = ranked("val", set(test))[0]
    for name, groups in (("test", test), ("val", val)):
        realized = sum(group_sizes[group] for group in groups)
        if realized != target_images:
            raise ValueError(
                f"ClinicDB {name} cannot preserve target images: {realized} != {target_images}"
            )
    return set(val), set(test)


def build_clinicdb(
    source: Path,
    output: Path,
    *,
    seed: int,
) -> dict[str, object]:
    fields, original = read_manifest(source)
    target_counts = split_counts(original)
    if target_counts["val"] != target_counts["test"]:
        raise ValueError("ClinicDB builder expects equal validation and test image targets")

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in original:
        sequence = recover_cvc_sequence(row["case_id"])
        grouped[f"clinicdb_sequence_{sequence:02d}"].append(row)
    if len(grouped) != 29:
        raise ValueError(f"Expected 29 ClinicDB sequences, found {len(grouped)}")

    target_holdout_groups = round(len(grouped) * target_counts["test"] / len(original))
    val_groups, test_groups = choose_sequence_holdouts(
        {group: len(rows) for group, rows in grouped.items()},
        target_images=target_counts["test"],
        target_groups=target_holdout_groups,
        seed=seed,
    )

    updated: list[dict[str, str]] = []
    for group_id, rows in grouped.items():
        split = "test" if group_id in test_groups else "val" if group_id in val_groups else "train"
        for source_row in rows:
            row = dict(source_row)
            row["split"] = split
            row["group_id"] = group_id
            row["group_type"] = "video_sequence"
            updated.append(row)
    updated.sort(key=lambda row: row["case_id"])

    if rows_fingerprint(original) != rows_fingerprint(updated):
        raise ValueError("ClinicDB non-grouping fields changed")
    if split_counts(updated) != target_counts:
        raise ValueError(f"ClinicDB split counts changed: {split_counts(updated)} != {target_counts}")
    overlap = split_overlap(updated, "group_id")
    if any(overlap.values()):
        raise ValueError(f"ClinicDB sequence overlap remains: {overlap}")

    write_manifest(output, fields, updated)
    return {
        "dataset": "ClinicDB",
        "group_type": "video_sequence",
        "mapping": "documented contiguous frame ranges for 29 CVC-ClinicDB sequences",
        "mapping_source": "https://polyp.grand-challenge.org/CVCClinicDB/",
        "source": portable_path(source),
        "source_sha256": file_sha256(source),
        "output": portable_path(output),
        "output_sha256": file_sha256(output),
        "rows_fingerprint": rows_fingerprint(updated),
        "rows": len(updated),
        "groups": len(grouped),
        "split_counts": split_counts(updated),
        "split_group_counts": split_group_counts(updated),
        "group_overlap": overlap,
        "val_groups": sorted(val_groups),
        "test_groups": sorted(test_groups),
    }


def recover_busbra_patient(case_id: str) -> tuple[str, str]:
    match = BUSBRA_CASE_PATTERN.search(str(case_id))
    if not match:
        raise ValueError(f"Unrecognized BUS-BRA case id: {case_id}")
    return match.group(1), match.group(2).lower()


def recover_busbra_label(text: str) -> str:
    value = str(text).lower()
    has_benign = bool(re.search(r"\bbenign\b", value))
    has_malignant = bool(re.search(r"\bmalignant\b", value))
    if has_benign == has_malignant:
        raise ValueError(f"BUS-BRA prompt has ambiguous class: {text!r}")
    return "benign" if has_benign else "malignant"


def largest_remainder(
    totals: dict[str, int],
    target_total: int,
) -> dict[str, int]:
    grand_total = sum(totals.values())
    quotas = {key: totals[key] * target_total / grand_total for key in totals}
    allocated = {key: math.floor(value) for key, value in quotas.items()}
    remainder = target_total - sum(allocated.values())
    order = sorted(
        totals,
        key=lambda key: (-(quotas[key] - allocated[key]), key),
    )
    for key in order[:remainder]:
        allocated[key] += 1
    return allocated


def solve_size_allocation(
    *,
    target_images: int,
    target_patients: int,
    available_size1: int,
    available_size2: int,
) -> dict[int, int]:
    size2 = target_images - target_patients
    size1 = 2 * target_patients - target_images
    if size1 < 0 or size2 < 0 or size1 > available_size1 or size2 > available_size2:
        raise ValueError(
            "No one/two-image patient allocation for "
            f"images={target_images}, patients={target_patients}, "
            f"available=({available_size1}, {available_size2})"
        )
    return {1: size1, 2: size2}


def build_busbra(
    source: Path,
    output: Path,
    *,
    seed: int,
) -> dict[str, object]:
    fields, original = read_manifest(source)
    target_counts = split_counts(original)
    if target_counts["val"] != target_counts["test"]:
        raise ValueError("BUS-BRA builder expects equal validation and test image targets")

    patients: dict[str, list[dict[str, str]]] = defaultdict(list)
    suffixes: dict[str, set[str]] = defaultdict(set)
    labels: dict[str, str] = {}
    for row in original:
        prefix, suffix = recover_busbra_patient(row["case_id"])
        patient_id = f"busbra_patient_{prefix}"
        label = recover_busbra_label(row["text"])
        if patient_id in labels and labels[patient_id] != label:
            raise ValueError(f"Conflicting BUS-BRA class within {patient_id}")
        labels[patient_id] = label
        suffixes[patient_id].add(suffix)
        patients[patient_id].append(row)

    if len(patients) != 1064:
        raise ValueError(f"Expected 1064 BUS-BRA patients, found {len(patients)}")
    if any(len(rows) not in (1, 2) for rows in patients.values()):
        raise ValueError("BUS-BRA grouping expects one or two images per patient")
    if any(suffixes[patient] not in ({"s"}, {"l", "r"}) for patient in patients):
        raise ValueError("BUS-BRA patient suffix pattern is not {s} or {l,r}")

    patient_label_totals = Counter(labels.values())
    image_label_totals = Counter(
        label for patient, label in labels.items() for _ in patients[patient]
    )
    expected_patient_totals = {"benign": 722, "malignant": 342}
    expected_image_totals = {"benign": 1268, "malignant": 607}
    if dict(patient_label_totals) != expected_patient_totals:
        raise ValueError(
            f"BUS-BRA patient class totals differ from the dataset article: "
            f"{dict(patient_label_totals)} != {expected_patient_totals}"
        )
    if dict(image_label_totals) != expected_image_totals:
        raise ValueError(
            f"BUS-BRA image class totals differ from the dataset article: "
            f"{dict(image_label_totals)} != {expected_image_totals}"
        )
    target_holdout_images = target_counts["test"]
    target_holdout_patients = round(len(patients) * target_holdout_images / len(original))
    label_image_targets = largest_remainder(dict(image_label_totals), target_holdout_images)
    label_patient_targets = largest_remainder(dict(patient_label_totals), target_holdout_patients)

    strata: dict[tuple[str, int], list[str]] = defaultdict(list)
    for patient, rows in patients.items():
        strata[(labels[patient], len(rows))].append(patient)
    for key in strata:
        strata[key].sort(key=lambda patient: stable_rank(seed, "busbra", *key, patient))

    holdout_allocations: dict[str, dict[int, int]] = {}
    for label in sorted(patient_label_totals):
        holdout_allocations[label] = solve_size_allocation(
            target_images=label_image_targets[label],
            target_patients=label_patient_targets[label],
            available_size1=len(strata[(label, 1)]) // 2,
            available_size2=len(strata[(label, 2)]) // 2,
        )

    assignment: dict[str, str] = {}
    for label in sorted(patient_label_totals):
        for size in (1, 2):
            members = strata[(label, size)]
            count = holdout_allocations[label][size]
            for patient in members[:count]:
                assignment[patient] = "test"
            for patient in members[count : 2 * count]:
                assignment[patient] = "val"
            for patient in members[2 * count :]:
                assignment[patient] = "train"

    updated: list[dict[str, str]] = []
    for patient, rows in patients.items():
        for source_row in rows:
            row = dict(source_row)
            row["split"] = assignment[patient]
            row["patient_id"] = patient
            row["group_id"] = patient
            row["group_type"] = "patient"
            updated.append(row)
    updated.sort(key=lambda row: row["case_id"])

    if rows_fingerprint(original) != rows_fingerprint(updated):
        raise ValueError("BUS-BRA non-grouping fields changed")
    if split_counts(updated) != target_counts:
        raise ValueError(f"BUS-BRA split counts changed: {split_counts(updated)} != {target_counts}")
    overlap = split_overlap(updated, "group_id")
    if any(overlap.values()):
        raise ValueError(f"BUS-BRA patient overlap remains: {overlap}")

    realized_labels = {
        split: dict(
            Counter(
                recover_busbra_label(row["text"])
                for row in updated
                if row["split"] == split
            )
        )
        for split in ("train", "val", "test")
    }
    write_manifest(output, fields, updated)
    return {
        "dataset": "BUS-BRA",
        "group_type": "patient",
        "mapping": "numeric bus_<patient>-<l|r|s> filename prefix",
        "mapping_evidence": {
            "unique_prefixes": len(patients),
            "published_patients": 1064,
            "patient_class_counts": dict(patient_label_totals),
            "published_patient_class_counts": expected_patient_totals,
            "image_class_counts": dict(image_label_totals),
            "published_image_class_counts": expected_image_totals,
        },
        "mapping_source": "https://doi.org/10.1002/mp.16812",
        "source": portable_path(source),
        "source_sha256": file_sha256(source),
        "output": portable_path(output),
        "output_sha256": file_sha256(output),
        "rows_fingerprint": rows_fingerprint(updated),
        "rows": len(updated),
        "groups": len(patients),
        "split_counts": split_counts(updated),
        "split_group_counts": split_group_counts(updated),
        "split_class_image_counts": realized_labels,
        "group_overlap": overlap,
        "val_groups": sorted(patient for patient, split in assignment.items() if split == "val"),
        "test_groups": sorted(patient for patient, split in assignment.items() if split == "test"),
        "holdout_image_targets_by_class": label_image_targets,
        "holdout_patient_targets_by_class": label_patient_targets,
        "holdout_size_allocations_by_class": holdout_allocations,
    }


def write_audit(audit_dir: Path, payload: dict[str, object]) -> None:
    audit_dir.mkdir(parents=True, exist_ok=True)
    json_path = audit_dir / "grouped_split_audit.json"
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    csv_path = audit_dir / "grouped_split_audit.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fields = (
            "dataset",
            "group_type",
            "rows",
            "groups",
            "train_images",
            "val_images",
            "test_images",
            "train_groups",
            "val_groups",
            "test_groups",
            "train_val_overlap",
            "train_test_overlap",
            "val_test_overlap",
            "output_sha256",
        )
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for item in payload["datasets"]:
            writer.writerow(
                {
                    "dataset": item["dataset"],
                    "group_type": item["group_type"],
                    "rows": item["rows"],
                    "groups": item["groups"],
                    "train_images": item["split_counts"]["train"],
                    "val_images": item["split_counts"]["val"],
                    "test_images": item["split_counts"]["test"],
                    "train_groups": item["split_group_counts"]["train"],
                    "val_groups": item["split_group_counts"]["val"],
                    "test_groups": item["split_group_counts"]["test"],
                    "train_val_overlap": item["group_overlap"]["train_val"],
                    "train_test_overlap": item["group_overlap"]["train_test"],
                    "val_test_overlap": item["group_overlap"]["val_test"],
                    "output_sha256": item["output_sha256"],
                }
            )

    clinic, busbra = payload["datasets"]
    readme = [
        "# Group-disjoint split audit",
        "",
        "The corrected manifests preserve the original image counts while assigning",
        "ClinicDB by video sequence and BUS-BRA by patient. No image, mask, prompt,",
        "or content hash field changes.",
        "",
        "| Dataset | Grouping unit | Train images/groups | Val images/groups | Test images/groups | Cross-split group overlap |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for item in (clinic, busbra):
        readme.append(
            f"| {item['dataset']} | {item['group_type']} | "
            f"{item['split_counts']['train']}/{item['split_group_counts']['train']} | "
            f"{item['split_counts']['val']}/{item['split_group_counts']['val']} | "
            f"{item['split_counts']['test']}/{item['split_group_counts']['test']} | 0 |"
        )
    readme.extend(
        [
            "",
            f"Split seed: {payload['split_seed']}.",
            "",
            "ClinicDB uses the published contiguous frame-range mapping for 29 video",
            "sequences. BUS-BRA uses the numeric filename prefix; its 1,064 unique",
            "prefixes and benign/malignant patient counts exactly match the dataset",
            "article's 1,064 patients (722 benign and 342 malignant).",
            "",
            "The JSON audit records every holdout group, source/output hash, class",
            "allocation, and overlap check.",
        ]
    )
    (audit_dir / "README.md").write_text(
        "\n".join(readme) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    args = parse_args()
    clinic = build_clinicdb(args.clinicdb_input, args.clinicdb_output, seed=args.split_seed)
    busbra = build_busbra(args.busbra_input, args.busbra_output, seed=args.split_seed)
    payload: dict[str, object] = {
        "protocol": "MedEquiSeg corrected group-disjoint public-data revision",
        "split_seed": args.split_seed,
        "unchanged_columns": list(UNCHANGED_COLUMNS),
        "datasets": [clinic, busbra],
    }
    write_audit(args.audit_dir, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    print("final_status: PASS")


if __name__ == "__main__":
    main()
