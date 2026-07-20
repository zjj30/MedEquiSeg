#!/usr/bin/env python3
"""Run a MONAI U-Net-family checkpoint on the locked Protocol V3 test split."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from protocol_v3.core import file_sha256, protocol_sha256
from run_ukan_image_aug_dataset import build_model


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")[:100] or "case"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--architecture",
        default="",
        help="Architecture override for legacy checkpoints that predate this metadata field.",
    )
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    train_args = checkpoint["args"]
    architecture = args.architecture or train_args.get("architecture")
    if not architecture:
        raise ValueError("Checkpoint does not identify its architecture; pass --architecture explicitly.")
    image_size = int(train_args.get("image_size", 224))
    model = build_model(image_size, no_kan=bool(train_args.get("no_kan", False)), architecture=architecture)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()

    rows = [
        row
        for row in read_csv(args.manifest)
        if row.get("dataset") == args.dataset and row.get("split", "").strip().lower() == "test"
    ]
    pred_dir = args.output_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    protocol_hash = protocol_sha256(args.protocol_lock)
    checkpoint_hash = file_sha256(args.checkpoint)
    output: list[dict[str, str]] = []
    for start in range(0, len(rows), args.batch_size):
        batch_rows = rows[start : start + args.batch_size]
        arrays = []
        sizes = []
        for row in batch_rows:
            with Image.open(row["image_path"]) as image:
                image = image.convert("RGB")
                sizes.append(image.size)
                image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
                arrays.append(np.asarray(image).transpose(2, 0, 1).copy())
        tensor = torch.from_numpy(np.stack(arrays)).float().div_(255.0).to(device)
        with torch.no_grad():
            logits = model(tensor)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            preds = (torch.sigmoid(logits.float()) >= 0.5).cpu().numpy()[:, 0]
        for row, pred, original_size in zip(batch_rows, preds, sizes):
            case_id = row.get("case_id") or Path(row["image_path"]).stem
            target = Image.fromarray(pred.astype(np.uint8) * 255)
            target = target.resize(original_size, Image.Resampling.NEAREST)
            target_path = pred_dir / f"{safe_name(case_id)}.png"
            target.save(target_path)
            output.append(
                {
                    "protocol_hash": protocol_hash,
                    "dataset": row["dataset"],
                    "case_id": case_id,
                    "prediction_path": str(target_path.resolve()),
                    "mask_path": str(Path(row["mask_path"]).resolve()),
                    "mask_mode": row.get("mask_mode") or "threshold_127",
                    "prompt_control": "true",
                    "checkpoint_sha256": checkpoint_hash,
                }
            )
    index_path = args.output_dir / "prediction_index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    print(f"prediction_index={index_path} cases={len(output)}")


if __name__ == "__main__":
    main()
