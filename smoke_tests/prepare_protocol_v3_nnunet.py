#!/usr/bin/env python3
"""Prepare one Protocol V3 manifest as an nnU-Net v2 dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path("<PROJECT_ROOT>")
OUT_ROOT = ROOT / "outputs" / "protocol_v3_nnunet"
DATASETS = {
    "medclipseg_busi": (951, "MedclipsegBusi", "medclipseg_busi_full.csv"),
    "medclipseg_clinicdb": (952, "MedclipsegClinicdb", "medclipseg_clinicdb_full.csv"),
    "medclipseg_busbra": (953, "MedclipsegBusbra", "medclipseg_busbra_full.csv"),
    "medclipseg_brisc": (954, "MedclipsegBrisc", "medclipseg_brisc_full.csv"),
    "medclipseg_covid19": (955, "MedclipsegCovid19", "medclipseg_covid19_full.csv"),
    "busi_hf": (956, "BusiHf", "busi_hf_r8_811_predicted_medclipseg_v3.csv"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def resolve_path(value: str, root: Path) -> Path:
    path = Path(value)
    if path.is_file():
        return path.resolve()
    if not path.is_absolute() and (root / path).is_file():
        return (root / path).resolve()
    old_root = Path("<PROJECT_ROOT>")
    try:
        mapped = root / path.relative_to(old_root)
    except ValueError:
        mapped = path
    if mapped.is_file():
        return mapped.resolve()
    raise FileNotFoundError(value)


def safe_id(row: dict[str, str], index: int) -> str:
    raw = row.get("case_id") or Path(row["image_path"]).stem
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", raw).strip("_")[:80]
    return f"p3_{index:06d}_{cleaned or 'case'}"


def write_image(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        return
    with Image.open(source) as image:
        image.convert("RGB").save(target)


def write_mask(source: Path, target: Path, mode: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        return
    with Image.open(source) as image:
        values = np.asarray(image.convert("L"))
    binary = values > 127 if mode == "threshold_127" else values != 0
    Image.fromarray(binary.astype(np.uint8)).save(target)


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dataset_id, short_name, manifest_name = DATASETS[args.dataset]
    dataset_name = f"Dataset{dataset_id:03d}_{short_name}"
    manifest = args.project_root / "smoke_tests" / "protocol_v3" / "manifests" / manifest_name
    manifest_hash = sha256(manifest)
    raw_dir = args.out_root / "nnUNet_raw" / dataset_name
    prep_dir = args.out_root / "nnUNet_preprocessed" / dataset_name
    marker = raw_dir / "protocol_v3_prepare.json"

    if marker.is_file() and not args.overwrite:
        existing = json.loads(marker.read_text(encoding="utf-8"))
        if existing.get("manifest_sha256") == manifest_hash:
            print(f"SKIP prepared dataset={args.dataset} path={raw_dir}")
            return
        raise RuntimeError(f"Stale prepared dataset: {raw_dir}; use --overwrite")

    if args.overwrite:
        shutil.rmtree(raw_dir, ignore_errors=True)
        shutil.rmtree(prep_dir, ignore_errors=True)
    for name in ("imagesTr", "labelsTr", "imagesTs", "labelsTs"):
        (raw_dir / name).mkdir(parents=True, exist_ok=True)
    prep_dir.mkdir(parents=True, exist_ok=True)

    rows = [row for row in read_csv(manifest) if row.get("dataset") == args.dataset]
    rows.sort(key=lambda row: (row["split"], row.get("case_id", ""), row["image_path"]))
    counts = {split: sum(row["split"] == split for row in rows) for split in ("train", "val", "test")}
    if min(counts.values()) <= 0:
        raise ValueError(f"Incomplete Protocol V3 split for {args.dataset}: {counts}")

    train_ids: list[str] = []
    val_ids: list[str] = []
    mapping: list[dict[str, str]] = []
    for index, row in enumerate(rows):
        identifier = safe_id(row, index)
        split = row["split"].strip().lower()
        image_source = resolve_path(row["image_path"], args.project_root)
        mask_source = resolve_path(row["mask_path"], args.project_root)
        mask_mode = row.get("mask_mode") or "threshold_127"
        suffix = "Tr" if split in {"train", "val"} else "Ts"
        image_target = raw_dir / f"images{suffix}" / f"{identifier}_0000.png"
        label_target = raw_dir / f"labels{suffix}" / f"{identifier}.png"
        write_image(image_source, image_target)
        write_mask(mask_source, label_target, mask_mode)
        if split == "train":
            train_ids.append(identifier)
        elif split == "val":
            val_ids.append(identifier)
        mapping.append(
            {
                "identifier": identifier,
                "dataset": args.dataset,
                "split": split,
                "case_id": row.get("case_id") or identifier,
                "image_path": str(image_source),
                "mask_path": str(mask_source),
                "mask_mode": mask_mode,
                "nnunet_image": str(image_target),
                "nnunet_label": str(label_target),
            }
        )

    dataset_json = {
        "channel_names": {"0": "R", "1": "G", "2": "B"},
        "labels": {"background": 0, "foreground": 1},
        "numTraining": len(train_ids) + len(val_ids),
        "file_ending": ".png",
        "overwrite_image_reader_writer": "NaturalImage2DIO",
    }
    (raw_dir / "dataset.json").write_text(json.dumps(dataset_json, indent=2) + "\n", encoding="utf-8")
    (prep_dir / "splits_final.json").write_text(
        json.dumps([{"train": train_ids, "val": val_ids}], indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv(raw_dir / "protocol_v3_mapping.csv", mapping)
    marker.write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "dataset_id": dataset_id,
                "dataset_name": dataset_name,
                "manifest": str(manifest),
                "manifest_sha256": manifest_hash,
                "counts": counts,
                "split_seed": 123,
                "test_is_training_input": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"prepared dataset={args.dataset} id={dataset_id} counts={counts} path={raw_dir}")


if __name__ == "__main__":
    main()
