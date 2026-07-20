#!/usr/bin/env python3
"""Stage one Protocol V3 manifest for the official MedCLIPSeg data loader."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

from PIL import Image

from prepare_medclipseg_fixedsplit import ensure_clean_dir, place_file, write_csv, write_xlsx
from protocol_v3.core import file_sha256, load_manifest_splits, protocol_sha256


ROOT = Path(__file__).resolve().parents[1]
SPLIT_FOLDERS = {"train": "Train_Folder", "val": "Val_Folder", "test": "Test_Folder"}
PROMPT_FILES = {
    "train": "Train_text.xlsx",
    "val": "Val_text.xlsx",
    "test": "Test_text_original.xlsx",
}


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._") or "case"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def prompt_rows(rows: list[dict[str, str]], names: dict[str, tuple[str, str]]) -> list[dict[str, str]]:
    return [
        {
            "Image": names[row["case_id"]][0],
            "Ground Truth": names[row["case_id"]][1],
            "Description": row["text"],
        }
        for row in rows
    ]


def write_config(path: Path, dataset_name: str, data_root: Path, batch_size: int, epochs: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'''DATASET:
  NAME: "{dataset_name}"
  TRAIN_PATH: "{(data_root / 'Train_Folder').as_posix()}/"
  VAL_PATH: "{(data_root / 'Val_Folder').as_posix()}/"
  TEST_PATH: "{(data_root / 'Test_Folder').as_posix()}/"
  TEXT_PROMPT_PATH: "{(data_root / 'Prompts_Folder').as_posix()}/"
  SIZE: 224

TRAIN:
  BATCH_SIZE: {batch_size}
  NUM_EPOCHS: {epochs}
  LEARNING_RATE: 0.0003
  DICE_WEIGHT: 0.5
  CE_WEIGHT: 0.5
  CLIP_WEIGHT: 0.1
  AUG_POLICY: "baseline"
  REWRITE_TEXT_HFLIP: False

TEST:
  NUM_SAMPLES: 30
  USE_LATEST: False

MODEL:
  CLIP_MODEL: unimedclip
  BACKBONE: "ViT-B/16"
  ADAPTER_DIM: 256
  NUM_UPSCALE: 2
  BETA: 2.35
  GATE_INIT: 0.0
  LAYERS: [1,2,3,4,5,6,7,8,9,10]
  TEMPERATURE: 0.2
  DEVICE: "cuda"
''',
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out-root", type=Path, default=ROOT / "outputs/medclipseg_protocol_v3")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--link-mode", choices=["symlink", "hardlink", "copy"], default="symlink")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    splits = load_manifest_splits(args.manifest, [args.dataset], require_train_val_test=True, check_files=True)
    all_rows = splits["train"] + splits["val"] + splits["test"]
    if not all_rows:
        raise ValueError(f"No rows for dataset={args.dataset}")
    if any(row["dataset"] != args.dataset for row in all_rows):
        raise ValueError("Prepared rows contain more than one dataset")
    if len({row["case_id"] for row in all_rows}) != len(all_rows):
        raise ValueError("case_id values must be unique across train/val/test")

    dataset_name = "V3_" + safe_name(args.dataset)
    dataset_root = args.out_root / args.dataset
    data_root = dataset_root / "data" / dataset_name
    config_path = dataset_root / "config.yaml"
    adapter_manifest = dataset_root / "adapter_manifest.csv"
    ensure_clean_dir(data_root, args.overwrite)
    for folder in SPLIT_FOLDERS.values():
        (data_root / folder / "img").mkdir(parents=True, exist_ok=True)
        (data_root / folder / "label").mkdir(parents=True, exist_ok=True)
    prompt_root = data_root / "Prompts_Folder"
    prompt_root.mkdir(parents=True, exist_ok=True)

    names: dict[str, tuple[str, str]] = {}
    staged_rows: list[dict[str, str]] = []
    placements: Counter[str] = Counter()
    for split, rows in splits.items():
        folder = SPLIT_FOLDERS[split]
        for index, row in enumerate(rows):
            image_src = Path(row["image_path"])
            mask_src = Path(row["mask_path"])
            stem = f"{index:06d}_{safe_name(row['case_id'])}"
            image_name = stem + (image_src.suffix.lower() or ".png")
            mask_name = stem + ".png"
            names[row["case_id"]] = (image_name, mask_name)
            placements[place_file(image_src, data_root / folder / "img" / image_name, args.link_mode)] += 1

            mask_dst = data_root / folder / "label" / mask_name
            if row["mask_mode"] == "threshold_127" and mask_src.suffix.lower() == ".png":
                placements[place_file(mask_src, mask_dst, args.link_mode)] += 1
            else:
                with Image.open(mask_src) as mask:
                    array = mask.convert("L")
                    if row["mask_mode"] == "binary_01":
                        array = array.point(lambda value: 255 if value else 0)
                    elif row["mask_mode"] == "nonzero_label":
                        array = array.point(lambda value: 255 if value > 0 else 0)
                    else:
                        array = array.point(lambda value: 255 if value >= 127 else 0)
                    array.save(mask_dst)
                placements["normalized_copy"] += 1

            staged_rows.append(
                {
                    "split": split,
                    "dataset": row["dataset"],
                    "case_id": row["case_id"],
                    "image": image_name,
                    "mask": mask_name,
                    "mask_mode": row["mask_mode"],
                    "text": row["text"],
                    "source_image": str(image_src),
                    "source_mask": str(mask_src),
                }
            )

    fields = ["Image", "Ground Truth", "Description"]
    for split, filename in PROMPT_FILES.items():
        write_xlsx(prompt_root / filename, prompt_rows(splits[split], names), fields)
    write_xlsx(prompt_root / "Test_text.xlsx", prompt_rows(splits["test"], names), fields)
    test_rows = [dict(row) for row in splits["test"]]
    if len(test_rows) > 1:
        shifted = [row["text"] for row in test_rows[1:]] + [test_rows[0]["text"]]
        for row, text in zip(test_rows, shifted):
            row["text"] = text
    write_xlsx(prompt_root / "Test_text_shuffled.xlsx", prompt_rows(test_rows, names), fields)
    train_text_counts = Counter(row["text"] for row in splits["train"])
    fixed_text = sorted(train_text_counts, key=lambda value: (-train_text_counts[value], value))[0]
    fixed_rows = [{**row, "text": fixed_text} for row in splits["test"]]
    empty_rows = [{**row, "text": ""} for row in splits["test"]]
    write_xlsx(prompt_root / "Test_text_fixed.xlsx", prompt_rows(fixed_rows, names), fields)
    write_xlsx(prompt_root / "Test_text_empty.xlsx", prompt_rows(empty_rows, names), fields)
    write_config(config_path, dataset_name, data_root, args.batch_size, args.epochs)
    write_csv(
        adapter_manifest,
        staged_rows,
        ["split", "dataset", "case_id", "image", "mask", "mask_mode", "text", "source_image", "source_mask"],
    )

    metadata = {
        "dataset": args.dataset,
        "dataset_name": dataset_name,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": file_sha256(args.manifest),
        "protocol_lock": str(args.protocol_lock.resolve()),
        "protocol_hash": protocol_sha256(args.protocol_lock),
        "split_counts": {split: len(rows) for split, rows in splits.items()},
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "mask_policy": "Protocol V3 mask_mode normalized for MedCLIPSeg threshold semantics",
        "placements": dict(sorted(placements.items())),
        "config": str(config_path.resolve()),
    }
    (dataset_root / "prepare_meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    print("final_status: PASS")


if __name__ == "__main__":
    main()
