#!/usr/bin/env python3
"""Create public-only CSV derivatives used by the Public-5 release archive."""

from __future__ import annotations

import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PRIVATE_PATTERN = re.compile(
    r"BUSI[-_]HF|private\s+(?:cohort|development|analysis)|external\s+cohort",
    flags=re.IGNORECASE,
)

TABLES = {
    "paper/tables/protocol_v3_claim_evidence.csv":
        "paper/tables/protocol_v3_public5_claim_evidence.csv",
    "paper/results/protocol_v3_image_baseline_aggregate.csv":
        "paper/results/protocol_v3_image_baseline_aggregate_public5.csv",
    "paper/results/protocol_v3_image_baseline_seed_metrics.csv":
        "paper/results/protocol_v3_image_baseline_seed_metrics_public5.csv",
    "paper/results/protocol_v3_boundary_20260715/boundary_aggregate.csv":
        "paper/results/protocol_v3_boundary_20260715/boundary_aggregate_public5.csv",
    "paper/results/protocol_v3_boundary_20260715/boundary_seed_metrics.csv":
        "paper/results/protocol_v3_boundary_20260715/boundary_seed_metrics_public5.csv",
}


def contains_nonpublic_value(row: dict[str, str]) -> bool:
    return any(PRIVATE_PATTERN.search(str(value or "")) for value in row.values())


def filter_csv(source: Path, output: Path) -> tuple[int, int]:
    with source.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if not fieldnames:
        raise ValueError(f"CSV has no header: {source}")

    kept = [row for row in rows if not contains_nonpublic_value(row)]
    removed = len(rows) - len(kept)
    if removed < 1:
        raise ValueError(f"Expected at least one non-public row in source: {source}")
    if not kept:
        raise ValueError(f"Filtering removed every row: {source}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(kept)
    temporary.replace(output)

    released_text = output.read_text(encoding="utf-8")
    if PRIVATE_PATTERN.search(released_text):
        raise AssertionError(f"Non-public token remains in {output}")
    return len(kept), removed


def main() -> None:
    total_kept = 0
    total_removed = 0
    for source_name, output_name in TABLES.items():
        source = ROOT / source_name
        output = ROOT / output_name
        kept, removed = filter_csv(source, output)
        total_kept += kept
        total_removed += removed
        print(f"PASS {output_name} kept={kept} removed={removed}")
    print(f"final_status=PASS kept={total_kept} removed={total_removed}")


if __name__ == "__main__":
    main()
