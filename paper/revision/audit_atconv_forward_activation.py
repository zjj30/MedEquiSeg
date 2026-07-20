#!/usr/bin/env python3
"""Verify which checkpoint ATConv replacements execute in one real forward pass."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]
SMOKE_TESTS = ROOT / "smoke_tests"
if str(SMOKE_TESTS) not in sys.path:
    sys.path.insert(0, str(SMOKE_TESTS))

from causal_atconv_plugin import ATConvDropIn  # noqa: E402
from predict_protocol_v3 import apply_prompt_control  # noqa: E402
from protocol_v3.core import file_sha256, load_manifest_splits, protocol_sha256  # noqa: E402
from text_encoders import load_text_embedding_cache  # noqa: E402
from train_baselines import ManifestDataset, build_model, forward_model, seed_all  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--text-encoder-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    seed_all(123)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda:0")
    protocol_hash = protocol_sha256(args.protocol_lock)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_args = checkpoint.get("args", {})
    if ckpt_args.get("protocol_hash") != protocol_hash:
        raise ValueError("Checkpoint and protocol-lock hashes do not match")

    model, meta = build_model(
        ckpt_args.get("model", "causal_clip"),
        device,
        causal_recipe_name=ckpt_args.get("causal_recipe", "default"),
        conv_plugin=ckpt_args.get("conv_plugin", "standard"),
        atconv_layers=int(ckpt_args.get("atconv_layers", 0) or 0),
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()

    calls: dict[str, dict[str, float | int]] = {}
    hooks = []
    for name, module in model.named_modules():
        if not isinstance(module, ATConvDropIn):
            continue
        calls[name] = {"calls": 0, "output_abs_mean": 0.0, "output_abs_max": 0.0}

        def record(_module, _inputs, output, *, module_name=name):
            tensor = output.detach().float()
            calls[module_name] = {
                "calls": int(calls[module_name]["calls"]) + 1,
                "output_abs_mean": float(tensor.abs().mean().cpu()),
                "output_abs_max": float(tensor.abs().max().cpu()),
            }

        hooks.append(module.register_forward_hook(record))

    splits = load_manifest_splits(
        args.manifest,
        [args.dataset],
        require_train_val_test=True,
        check_files=True,
    )
    test_rows = apply_prompt_control(splits, args.dataset, "true", 123)[:1]
    dataset = ManifestDataset(
        test_rows,
        meta["image_size"],
        resize_mode=ckpt_args.get("resize_mode", "stretch"),
        seed=123,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    cache = load_text_embedding_cache(args.text_encoder_cache)
    cache.validate_texts([test_rows[0]["text"]])
    with torch.no_grad():
        images, _, texts, _, _ = next(iter(loader))
        prediction = forward_model(
            model,
            meta,
            images.to(device),
            texts,
            device,
            text_cache=cache,
        )
        if isinstance(prediction, dict):
            prediction = prediction["logits"]

    for hook in hooks:
        hook.remove()

    active = sorted(name for name, values in calls.items() if values["calls"] > 0)
    inactive = sorted(name for name, values in calls.items() if values["calls"] == 0)
    expected_active = sorted(("base_model.proj.vis.1.0", "base_model.proj_ad.vis.1.0"))
    expected_inactive = sorted(
        ("base_model.neck_ad.coordconv.1.0", "base_model.neck_ad.f4_proj3.0")
    )
    if active != expected_active or inactive != expected_inactive:
        raise AssertionError(f"Unexpected ATConv execution graph: active={active}, inactive={inactive}")

    result = {
        "status": "PASS",
        "audit_type": "single_real_test_case_forward_hooks",
        "device": str(device),
        "dataset": args.dataset,
        "case_id": test_rows[0]["case_id"],
        "protocol_hash": protocol_hash,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "registered_atconv_count": len(calls),
        "forward_active_count": len(active),
        "forward_inactive_count": len(inactive),
        "active_targets": active,
        "inactive_targets": inactive,
        "calls": calls,
        "prediction_shape": list(prediction.shape),
        "prediction_abs_mean": float(prediction.detach().float().abs().mean().cpu()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
