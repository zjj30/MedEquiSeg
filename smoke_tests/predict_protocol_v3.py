#!/usr/bin/env python3
"""Export Protocol V3 binary predictions at each ground-truth image resolution."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from protocol_v3.core import file_sha256, load_manifest_splits, protocol_sha256
from text_encoders import load_text_embedding_cache
from train_baselines import ManifestDataset, build_model, forward_model, seed_all


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--prompt-control",
        choices=["true", "shuffled", "fixed", "empty", "stratified_derangement"],
        default="true",
    )
    parser.add_argument("--control-seed", type=int, default=123)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--text-encoder-cache", default="")
    parser.add_argument(
        "--fixed-prompt-manifest",
        type=Path,
        default=None,
        help="Optional source-domain manifest used to select the fixed-prompt control.",
    )
    parser.add_argument(
        "--fixed-prompt-dataset",
        default="",
        help="Dataset id in --fixed-prompt-manifest; defaults to --dataset.",
    )
    parser.add_argument(
        "--stratified-control-map",
        type=Path,
        default=None,
        help="Case-level prompt permutation map for --prompt-control stratified_derangement.",
    )
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def apply_prompt_control(
    splits: dict[str, list[dict[str, str]]],
    dataset: str,
    control: str,
    seed: int,
    *,
    fixed_prompt_rows: list[dict[str, str]] | None = None,
):
    test_rows = [dict(row) for row in splits["test"] if row["dataset"] == dataset]
    train_rows = [row for row in splits["train"] if row["dataset"] == dataset]
    if not test_rows:
        raise ValueError(f"No Protocol V3 test rows for {dataset}")
    if control == "true":
        return test_rows
    if control == "empty":
        for row in test_rows:
            row["text"] = ""
        return test_rows
    if control == "fixed":
        counts = Counter(row["text"] for row in (fixed_prompt_rows or train_rows))
        if not counts:
            raise ValueError(f"No training prompt is available for fixed control: {dataset}")
        fixed = sorted(counts, key=lambda value: (-counts[value], value))[0]
        for row in test_rows:
            row["text"] = fixed
        return test_rows
    if len(test_rows) < 2:
        raise ValueError("A shuffled control requires at least two test cases")
    original = [row["text"] for row in test_rows]
    shift = 1 + (int(seed) % (len(test_rows) - 1))
    shuffled = original[shift:] + original[:shift]
    for row, text in zip(test_rows, shuffled):
        row["text"] = text
    return test_rows


def canonical_text(value: str) -> str:
    return " ".join(value.split())


def text_sha256(value: str) -> str:
    return hashlib.sha256(canonical_text(value).encode("utf-8")).hexdigest()


def apply_stratified_control_map(
    splits: dict[str, list[dict[str, str]]],
    dataset: str,
    map_path: Path,
) -> tuple[list[dict[str, str]], dict[str, object]]:
    test_rows = [dict(row) for row in splits["test"] if row["dataset"] == dataset]
    by_case = {row["case_id"]: row for row in test_rows}
    original_raw_text_by_case = {case_id: row["text"] for case_id, row in by_case.items()}
    original_text_by_case = {
        case_id: canonical_text(value) for case_id, value in original_raw_text_by_case.items()
    }
    with map_path.open(newline="", encoding="utf-8") as handle:
        map_rows = [row for row in csv.DictReader(handle) if row["dataset"] == dataset]
    map_by_case = {row["case_id"]: row for row in map_rows}
    if len(map_by_case) != len(map_rows):
        raise ValueError(f"Duplicate case_id in stratified control map: {map_path}")
    if set(map_by_case) != set(by_case):
        missing = sorted(set(by_case) - set(map_by_case))[:5]
        extra = sorted(set(map_by_case) - set(by_case))[:5]
        raise ValueError(f"Control-map closure mismatch missing={missing} extra={extra}")
    if Counter(row["source_case_id"] for row in map_rows) != Counter(by_case.keys()):
        raise ValueError("Stratified control map is not a one-to-one case permutation")

    control_ids = {row["control_id"] for row in map_rows}
    if len(control_ids) != 1:
        raise ValueError(f"Expected one control_id in {map_path}: {sorted(control_ids)}")
    changed = 0
    source_self = 0
    for case_id, row in by_case.items():
        mapping = map_by_case[case_id]
        source_case_id = mapping["source_case_id"]
        original = original_text_by_case[case_id]
        assigned_raw = mapping["assigned_text"]
        assigned = canonical_text(assigned_raw)
        if mapping["original_text_sha256"] != text_sha256(original):
            raise ValueError(f"Original prompt hash mismatch: {case_id}")
        if mapping["assigned_text_sha256"] != text_sha256(assigned):
            raise ValueError(f"Assigned prompt hash mismatch: {case_id}")
        if assigned != original_text_by_case[source_case_id]:
            raise ValueError(f"Assigned prompt does not match source row: {case_id} <- {source_case_id}")
        recipient_stratum = (mapping["presence_stratum"], mapping["class_stratum"])
        source_mapping = map_by_case[source_case_id]
        source_stratum = (source_mapping["presence_stratum"], source_mapping["class_stratum"])
        if recipient_stratum != source_stratum:
            raise ValueError(f"Cross-stratum assignment: {case_id} <- {source_case_id}")
        row["text"] = assigned_raw
        changed += int(original != assigned)
        source_self += int(case_id == source_case_id)

    return test_rows, {
        "control_id": next(iter(control_ids)),
        "control_map": str(map_path.resolve()),
        "control_map_sha256": file_sha256(map_path),
        "text_changed_cases": changed,
        "text_unchanged_cases": len(test_rows) - changed,
        "source_self_cases": source_self,
    }


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def safe_case_name(case_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", case_id) + ".png"


def main() -> None:
    args = parse_args()
    seed_all(args.control_seed)
    protocol_hash = protocol_sha256(args.protocol_lock)
    splits = load_manifest_splits(args.manifest, [args.dataset], require_train_val_test=True, check_files=True)
    fixed_prompt_rows = None
    fixed_prompt_dataset = args.fixed_prompt_dataset or args.dataset
    if args.prompt_control == "fixed" and args.fixed_prompt_manifest is not None:
        fixed_splits = load_manifest_splits(
            args.fixed_prompt_manifest,
            [fixed_prompt_dataset],
            require_train_val_test=True,
            check_files=True,
        )
        fixed_prompt_rows = fixed_splits["train"]
    stratified_meta: dict[str, object] = {}
    if args.prompt_control == "stratified_derangement":
        if args.stratified_control_map is None:
            raise ValueError("--stratified-control-map is required for stratified_derangement")
        test_rows, stratified_meta = apply_stratified_control_map(
            splits,
            args.dataset,
            args.stratified_control_map,
        )
    else:
        test_rows = apply_prompt_control(
            splits,
            args.dataset,
            args.prompt_control,
            args.control_seed,
            fixed_prompt_rows=fixed_prompt_rows,
        )

    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_args = checkpoint.get("args", {})
    checkpoint_protocol = ckpt_args.get("protocol_hash", "legacy")
    if checkpoint_protocol != protocol_hash:
        raise ValueError(f"Checkpoint protocol mismatch: {checkpoint_protocol} != {protocol_hash}")
    recipe = ckpt_args.get("causal_recipe", "default")
    model, meta = build_model(
        args.model,
        device,
        causal_recipe_name=recipe,
        conv_plugin=ckpt_args.get("conv_plugin", "standard"),
        atconv_layers=int(ckpt_args.get("atconv_layers", 0) or 0),
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()

    cache_path = args.text_encoder_cache or ckpt_args.get("text_encoder_cache", "")
    text_cache = load_text_embedding_cache(cache_path) if cache_path else None
    if text_cache is not None:
        text_cache.validate_texts(sorted({row["text"] for row in test_rows}))

    resize_mode = ckpt_args.get("resize_mode", "stretch")
    dataset = ManifestDataset(test_rows, meta["image_size"], resize_mode=resize_mode, seed=args.control_seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    prediction_dir = args.output_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_hash = file_sha256(args.checkpoint)
    rows: list[dict[str, str]] = []
    offset = 0
    with torch.no_grad():
        for images, _, texts, _, _ in loader:
            images = images.to(device, non_blocking=True)
            pred = forward_model(model, meta, images, texts, device, text_cache=text_cache)
            if isinstance(pred, dict):
                pred = pred["logits"]
            prob = torch.sigmoid(pred.float()) if meta["uses_logits"] else pred.float().clamp(0.0, 1.0)
            binary = (prob.detach().cpu().numpy() >= args.threshold).astype("uint8") * 255
            batch_rows = test_rows[offset : offset + len(images)]
            for index, row in enumerate(batch_rows):
                with Image.open(row["mask_path"]) as ground_truth:
                    original_size = ground_truth.size
                prediction = Image.fromarray(binary[index, 0]).resize(original_size, Image.Resampling.NEAREST)
                prediction_path = prediction_dir / safe_case_name(row["case_id"])
                prediction.save(prediction_path)
                rows.append(
                    {
                        "dataset": row["dataset"],
                        "case_id": row["case_id"],
                        "prediction_path": str(prediction_path.resolve()),
                        "mask_path": row["mask_path"],
                        "mask_mode": row["mask_mode"],
                        "prompt_control": args.prompt_control,
                        "checkpoint": str(args.checkpoint.resolve()),
                        "checkpoint_sha256": checkpoint_hash,
                        "protocol_hash": protocol_hash,
                    }
                )
            offset += len(images)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    index_path = args.output_dir / "prediction_index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    run_meta = {
        "protocol_id": "MEDSEG_TEXT_V3_20260710",
        "protocol_hash": protocol_hash,
        "git_commit": git_head(),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": file_sha256(args.manifest),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "cache": cache_path,
        "cache_sha256": file_sha256(cache_path) if cache_path else "",
        "dataset": args.dataset,
        "prompt_control": args.prompt_control,
        "predictor_code_sha256": file_sha256(Path(__file__)),
        "fixed_prompt_manifest": str(args.fixed_prompt_manifest.resolve()) if args.fixed_prompt_manifest else "",
        "fixed_prompt_dataset": fixed_prompt_dataset if args.fixed_prompt_manifest else "",
        "prediction_threshold": args.threshold,
        "cases": len(rows),
        **stratified_meta,
    }
    (args.output_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
    print(f"prediction_index={index_path}")
    print(f"cases={len(rows)} protocol_hash={protocol_hash}")
    print("final_status: PASS")


if __name__ == "__main__":
    main()
