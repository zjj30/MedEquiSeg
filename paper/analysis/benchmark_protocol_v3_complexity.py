#!/usr/bin/env python3
"""Benchmark instantiated models and their executed Protocol V3 forward paths."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("<PROJECT_ROOT>"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Optional method slugs: r11, r0, lvit, rolling, or unetplusplus.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def first_test_prompt(manifest: Path) -> str:
    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row.get("split", "").strip().lower() == "test":
                return row.get("text", "")
    raise ValueError(f"No test row in {manifest}")


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def count_parameters(model: torch.nn.Module) -> tuple[int, int, float]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    model_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
    model_bytes += sum(buffer.numel() * buffer.element_size() for buffer in model.buffers())
    return total, trainable, model_bytes / (1024**2)


def benchmark(
    model: torch.nn.Module,
    forward,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> dict[str, float | str]:
    model.eval()
    torch.cuda.empty_cache()
    with torch.inference_mode():
        for _ in range(warmup):
            output = forward()
        torch.cuda.synchronize(device)

        torch.cuda.reset_peak_memory_stats(device)
        elapsed_ms: list[float] = []
        output = None
        for _ in range(repeats):
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            output = forward()
            torch.cuda.synchronize(device)
            elapsed_ms.append((time.perf_counter() - start) * 1000.0)

    if isinstance(output, dict):
        output = output.get("logits", next(iter(output.values())))
    if isinstance(output, (tuple, list)):
        output = output[0]
    shape = "x".join(str(value) for value in output.shape) if hasattr(output, "shape") else "unknown"
    return {
        "latency_mean_ms": statistics.mean(elapsed_ms),
        "latency_std_ms": statistics.stdev(elapsed_ms) if len(elapsed_ms) > 1 else 0.0,
        "latency_median_ms": statistics.median(elapsed_ms),
        "latency_p95_ms": percentile(elapsed_ms, 0.95),
        "throughput_images_s": 1000.0 / statistics.mean(elapsed_ms),
        "peak_gpu_memory_mb": torch.cuda.max_memory_allocated(device) / (1024**2),
        "output_shape": shape,
    }


def multimodal_spec(root: Path, run_id: str, label: str, slug: str) -> dict:
    protocol_hash = "abb8ccb2d46cf039a4a0c573d733e3bc510636e6dfa7e59032508ab96fc0f718"
    meta_path = root / "logs/protocol_v3" / protocol_hash / run_id / "medclipseg_busi/seed123/run_meta.json"
    meta = read_json(meta_path)
    return {
        "method": label,
        "slug": slug,
        "kind": "train_baselines",
        "model_name": "LViT" if run_id == "V3_LVIT_PUBMEDBERT" else "CausalCLIPSegRN50DiceBCE",
        "checkpoint": Path(meta["checkpoint"]),
        "manifest": Path(meta["manifest"]),
        "cache": Path(meta["cache"]) if meta.get("cache") else None,
        "recipe": meta.get("recipe", "default"),
        "online_text_encoder": (
            "cached PubMedBERT token embeddings"
            if run_id == "V3_LVIT_PUBMEDBERT"
            else "cached BioMedCLIP embedding adapter"
            if run_id in {"V3_R11", "V3_ABL_EQUIPROMPT"}
            else "CLIP RN50 BPE text tower"
        ),
        "notes": (
            "MedEquiSeg training policy; augmentation adds zero inference parameters"
            if run_id in {"V3_R11", "V3_ABL_EQUIPROMPT"}
            else "unaugmented LViT comparator"
            if run_id == "V3_LVIT_PUBMEDBERT"
            else "CausalCLIPSeg Protocol V3 comparator"
        ),
    }


def build_specs(root: Path) -> list[dict]:
    specs = [
        multimodal_spec(root, "V3_ABL_EQUIPROMPT", "MedEquiSeg", "r11"),
        multimodal_spec(root, "V3_R0", "CausalCLIPSeg (R0)", "r0"),
        multimodal_spec(root, "V3_LVIT_PUBMEDBERT", "LViT", "lvit"),
    ]
    specs.extend(
        [
            {
                "method": "RollingUNet",
                "slug": "rolling",
                "kind": "rolling",
                "checkpoint": root
                / "logs/protocol_v3_pure_image/rolling_medclipseg_busi_common_light_seed123"
                / "RollingUNet_medclipseg_busi_aug_common_light_seed123_best.pt",
                "online_text_encoder": "none",
                "notes": "image-only comparator",
            },
            {
                "method": "U-Net++",
                "slug": "unetplusplus",
                "kind": "monai",
                "architecture": "unetplusplus",
                "checkpoint": root
                / "logs/protocol_v3_image_baselines/unetplusplus/medclipseg_busi/seed123/model"
                / "UNetPlusPlus_medclipseg_busi_aug_common_light_seed123_best.pt",
                "online_text_encoder": "none",
                "notes": "representative image-only comparator",
            },
        ]
    )
    return specs


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    sys.path.insert(0, str(root / "smoke_tests"))
    from run_rolling_image_aug_dataset import build_model as build_rolling
    from run_ukan_image_aug_dataset import build_model as build_monai
    from text_encoders import load_text_embedding_cache
    from train_baselines import build_model, forward_model

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the paper timing protocol")
    device = torch.device("cuda:0")
    torch.manual_seed(123)
    torch.backends.cudnn.benchmark = True
    gpu_name = torch.cuda.get_device_name(device)
    rows: list[dict[str, object]] = []

    specs = build_specs(root)
    if args.methods:
        requested = set(args.methods)
        known = {spec["slug"] for spec in specs}
        unknown = requested - known
        if unknown:
            raise ValueError(f"Unknown method slugs: {sorted(unknown)}; expected {sorted(known)}")
        specs = [spec for spec in specs if spec["slug"] in requested]

    for spec in specs:
        checkpoint_path = Path(spec["checkpoint"])
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        cache = None
        if spec["kind"] == "train_baselines":
            checkpoint_args = checkpoint.get("args", {})
            model, meta = build_model(
                spec["model_name"],
                device,
                causal_recipe_name=checkpoint_args.get("causal_recipe", spec.get("recipe", "default")),
                conv_plugin=checkpoint_args.get("conv_plugin", "standard"),
                atconv_layers=int(checkpoint_args.get("atconv_layers", 0) or 0),
            )
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            image_size = int(meta["image_size"])
            prompt = first_test_prompt(Path(spec["manifest"]))
            if spec.get("cache"):
                cache = load_text_embedding_cache(spec["cache"])
                cache.validate_texts([prompt])
            image = torch.rand(1, 3, image_size, image_size, device=device)
            forward = lambda: forward_model(model, meta, image, [prompt], device, text_cache=cache)
        elif spec["kind"] == "rolling":
            checkpoint_args = checkpoint.get("args", {})
            image_size = int(checkpoint_args.get("image_size", 224))
            model = build_rolling(image_size).to(device)
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            image = torch.rand(1, 3, image_size, image_size, device=device)
            forward = lambda: model(image)
        else:
            checkpoint_args = checkpoint.get("args", {})
            image_size = int(checkpoint_args.get("image_size", 224))
            model = build_monai(image_size, architecture=spec["architecture"]).to(device)
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            image = torch.rand(1, 3, image_size, image_size, device=device)
            forward = lambda: model(image)

        total, trainable, model_memory_mb = count_parameters(model)
        timing = benchmark(model, forward, device, args.warmup, args.repeats)
        rows.append(
            {
                "method": spec["method"],
                "modality": "image-only" if spec["kind"] in {"rolling", "monai"} else "image+text",
                "input_size": f"{image_size}x{image_size}",
                "batch_size": 1,
                "precision": "FP32",
                "parameters": total,
                "parameters_million": total / 1e6,
                "trainable_parameters": trainable,
                "trainable_parameters_million": trainable / 1e6,
                "model_parameter_buffer_mb": model_memory_mb,
                "checkpoint_mb": checkpoint_path.stat().st_size / (1024**2),
                "online_text_encoder": spec["online_text_encoder"],
                "gpu": gpu_name,
                "warmup": args.warmup,
                "repeats": args.repeats,
                **timing,
                "notes": spec["notes"],
                "checkpoint": str(checkpoint_path.resolve()),
            }
        )
        print(
            f"PASS {spec['method']}: params={total / 1e6:.2f}M "
            f"latency={timing['latency_mean_ms']:.2f}ms",
            flush=True,
        )
        del forward, image, model, checkpoint, cache
        torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "protocol": "single-image synchronized wall-clock forward latency",
        "gpu": gpu_name,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "data_loading_included": False,
        "postprocessing_included": False,
        "cached_text_lookup_included": True,
        "output": str(args.output.resolve()),
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"output={args.output} rows={len(rows)}")


if __name__ == "__main__":
    main()
