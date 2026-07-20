#!/usr/bin/env python3
"""Convert official MedCLIPSeg masks into a Protocol V3 prediction index."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
from pathlib import Path

from PIL import Image

from protocol_v3.core import file_sha256, protocol_sha256


ROOT = Path(__file__).resolve().parents[1]
PYTHON = "<ENV_ROOT>/bin/python"


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", value) + ".png"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-manifest", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--prompt-control", choices=["true", "shuffled", "fixed", "empty"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = [row for row in read_rows(args.adapter_manifest) if row["split"] == "test"]
    if not rows:
        raise ValueError(f"No test rows in {args.adapter_manifest}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    normalized_dir = args.output_dir / "predictions_original_size"
    normalized_dir.mkdir(parents=True, exist_ok=True)
    protocol_hash = protocol_sha256(args.protocol_lock)
    checkpoint_hash = file_sha256(args.checkpoint)
    index_rows: list[dict[str, str]] = []

    for row in rows:
        source_prediction = args.prediction_dir / row["mask"]
        if not source_prediction.is_file():
            raise FileNotFoundError(f"Missing MedCLIPSeg prediction: {source_prediction}")
        with Image.open(row["source_mask"]) as mask:
            original_size = mask.size
        with Image.open(source_prediction) as prediction:
            binary = prediction.convert("L").point(lambda value: 255 if value >= 127 else 0)
            binary = binary.resize(original_size, Image.Resampling.NEAREST)
        output_path = normalized_dir / safe_name(row["case_id"])
        binary.save(output_path)
        index_rows.append(
            {
                "dataset": row["dataset"],
                "case_id": row["case_id"],
                "prediction_path": str(output_path.resolve()),
                "mask_path": row["source_mask"],
                "mask_mode": row["mask_mode"],
                "prompt_control": args.prompt_control,
                "checkpoint": str(args.checkpoint.resolve()),
                "checkpoint_sha256": checkpoint_hash,
                "protocol_hash": protocol_hash,
            }
        )

    index_path = args.output_dir / "prediction_index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(index_rows[0]))
        writer.writeheader()
        writer.writerows(index_rows)
    command = [
        PYTHON,
        "smoke_tests/evaluate_predictions_v3.py",
        "--prediction-index",
        str(index_path),
        "--protocol-lock",
        str(args.protocol_lock),
        "--per-case-csv",
        str(args.output_dir / "per_case.csv"),
        "--summary-csv",
        str(args.output_dir / "summary.csv"),
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    print(f"prediction_index={index_path}")
    print(f"cases={len(index_rows)} protocol_hash={protocol_hash}")
    print("final_status: PASS")


if __name__ == "__main__":
    main()
