#!/usr/bin/env python3
"""Build a deterministic, label/presence-stratified prompt permutation.

The control never reads image pixels or masks. Within each stratum it preserves
the complete prompt multiset and maximizes the number of rows receiving a
different text string. Identical-only strata are reported as unavoidable
unchanged text rather than silently borrowing from another class.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


DATASETS = (
    "medclipseg_busi",
    "medclipseg_clinicdb",
    "medclipseg_busbra",
    "medclipseg_brisc",
    "medclipseg_covid19",
)
CONTROL_ID = "STRATIFIED_DERANGEMENT_V1_20260719"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def canonical_text(value: str) -> str:
    return " ".join(value.split())


def text_sha256(value: str) -> str:
    return hashlib.sha256(canonical_text(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_test_rows(path: Path, dataset: str) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == "test" and row["dataset"] == dataset]
    if not rows:
        raise RuntimeError(f"No test rows for {dataset}: {path}")
    if len({row["case_id"] for row in rows}) != len(rows):
        raise RuntimeError(f"Duplicate case_id in {path}")
    return rows


def contains_any(text: str, values: tuple[str, ...]) -> bool:
    return any(value in text for value in values)


def stratum_for(row: dict[str, str]) -> tuple[str, str]:
    dataset = row["dataset"]
    case_id = row["case_id"].lower()
    text = canonical_text(row["text"]).lower()
    negative = bool(re.search(r"\b(no|without|absent|negative for)\b", text))

    if dataset == "medclipseg_busi":
        if "_normal" in case_id or negative:
            return "absent", "normal"
        if "_benign" in case_id:
            return "present", "benign"
        if "_malignant" in case_id:
            return "present", "malignant"
        raise ValueError(f"Unrecognized BUSI class: {row['case_id']}")

    if dataset == "medclipseg_busbra":
        if negative:
            return "absent", "no_mass"
        if "malignant" in text:
            return "present", "malignant"
        if "benign" in text:
            return "present", "benign"
        raise ValueError(f"Unrecognized BUS-BRA class text: {row['case_id']} {text}")

    if dataset == "medclipseg_brisc":
        if negative or "_no_" in case_id or contains_any(text, ("normal morphology", "normal brain", "no abnormal")):
            return "absent", "no_tumor"
        if contains_any(text, ("glioma", "glial")):
            return "present", "glioma"
        if "meningioma" in text:
            return "present", "meningioma"
        if contains_any(text, ("pituitary", "hypophys")):
            return "present", "pituitary"
        raise ValueError(f"Unrecognized BRISC class text: {row['case_id']} {text}")

    if dataset == "medclipseg_clinicdb":
        return ("absent", "no_polyp") if negative else ("present", "polyp")

    if dataset == "medclipseg_covid19":
        return ("absent", "no_infection") if negative else ("present", "infection")

    raise ValueError(dataset)


def maximal_text_derangement(rows: list[dict[str, str]]) -> list[tuple[dict[str, str], dict[str, str]]]:
    """Return recipient/donor pairs while minimizing identical assigned text."""
    ordered = sorted(rows, key=lambda row: (canonical_text(row["text"]), row["case_id"]))
    if len(ordered) == 1:
        return [(ordered[0], ordered[0])]
    max_frequency = max(Counter(canonical_text(row["text"]) for row in ordered).values())
    shift = max_frequency % len(ordered)
    if shift == 0:
        shift = 1
    return [(row, ordered[(index + shift) % len(ordered)]) for index, row in enumerate(ordered)]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_summary: list[dict[str, object]] = []
    output_hashes: dict[str, str] = {}

    for dataset in DATASETS:
        manifest = args.manifest_dir / f"{dataset}_full.csv"
        rows = read_test_rows(manifest, dataset)
        groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            groups[stratum_for(row)].append(row)

        mappings: list[dict[str, object]] = []
        for (presence, class_label), group in sorted(groups.items()):
            pairs = maximal_text_derangement(group)
            original_multiset = Counter(canonical_text(row["text"]) for row in group)
            assigned_multiset = Counter(canonical_text(donor["text"]) for _, donor in pairs)
            if original_multiset != assigned_multiset:
                raise RuntimeError(f"Prompt multiset changed in {dataset}/{presence}/{class_label}")
            for recipient, donor in pairs:
                original_raw = recipient["text"]
                assigned_raw = donor["text"]
                original = canonical_text(original_raw)
                assigned = canonical_text(assigned_raw)
                mappings.append(
                    {
                        "control_id": CONTROL_ID,
                        "dataset": dataset,
                        "case_id": recipient["case_id"],
                        "source_case_id": donor["case_id"],
                        "presence_stratum": presence,
                        "class_stratum": class_label,
                        "original_text": original_raw,
                        "assigned_text": assigned_raw,
                        "original_text_sha256": text_sha256(original),
                        "assigned_text_sha256": text_sha256(assigned),
                        "source_is_self": int(recipient["case_id"] == donor["case_id"]),
                        "text_changed": int(original != assigned),
                    }
                )

            unique_texts = len(original_multiset)
            max_frequency = max(original_multiset.values())
            changed = sum(canonical_text(recipient["text"]) != canonical_text(donor["text"]) for recipient, donor in pairs)
            all_summary.append(
                {
                    "dataset": dataset,
                    "presence_stratum": presence,
                    "class_stratum": class_label,
                    "n_cases": len(group),
                    "n_unique_texts": unique_texts,
                    "max_text_frequency": max_frequency,
                    "n_text_changed": changed,
                    "text_changed_rate": changed / len(group),
                    "strict_full_derangement_feasible": int(changed == len(group)),
                }
            )

        mappings.sort(key=lambda row: str(row["case_id"]))
        if len(mappings) != len(rows):
            raise RuntimeError(f"Mapping closure mismatch for {dataset}")
        map_path = args.output_dir / f"{dataset}_stratified_derangement_v1.csv"
        write_csv(map_path, mappings)
        output_hashes[dataset] = file_sha256(map_path)

    summary_path = args.output_dir / "stratified_derangement_v1_summary.csv"
    write_csv(summary_path, all_summary)
    payload = {
        "control_id": CONTROL_ID,
        "construction": "within-presence-and-class prompt-multiset-preserving maximal text derangement",
        "mask_or_image_pixels_read": False,
        "manifest_dir": str(args.manifest_dir.resolve()),
        "mapping_sha256": output_hashes,
        "summary_sha256": file_sha256(summary_path),
        "strata": all_summary,
    }
    json_path = args.output_dir / "stratified_derangement_v1_summary.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output_dir": str(args.output_dir), "datasets": len(DATASETS), "strata": len(all_summary)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
