#!/usr/bin/env python3
"""Benchmark official MedCLIPSeg under its Protocol V3 evaluation settings."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("<PROJECT_ROOT>"))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round(fraction * (len(ordered) - 1))]


def benchmark(model, image, prompt: str, samples: int, warmup: int, repeats: int, device) -> dict:
    with torch.inference_mode():
        for _ in range(warmup):
            output = model(image=image, text=[prompt], num_samples=samples)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        elapsed = []
        for _ in range(repeats):
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            output = model(image=image, text=[prompt], num_samples=samples)
            torch.cuda.synchronize(device)
            elapsed.append((time.perf_counter() - start) * 1000.0)
    return {
        "latency_mean_ms": statistics.mean(elapsed),
        "latency_std_ms": statistics.stdev(elapsed) if len(elapsed) > 1 else 0.0,
        "latency_median_ms": statistics.median(elapsed),
        "latency_p95_ms": percentile(elapsed, 0.95),
        "throughput_images_s": 1000.0 / statistics.mean(elapsed),
        "peak_gpu_memory_mb": torch.cuda.max_memory_allocated(device) / (1024**2),
        "output_shape": "x".join(str(value) for value in output.shape),
    }


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    output = args.output if args.output.is_absolute() else (root / args.output).resolve()
    repo = root / "repos/MedCLIPSeg"
    os.chdir(repo)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    sys.path.insert(0, str(repo))
    from trainers.medclipseg_unimedclip import build_medclipseg_unimedclip
    from utils.main_utils import load_cfg_from_cfg_file

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the paper timing protocol")
    device = torch.device("cuda:0")
    config = root / "outputs/medclipseg_protocol_v3/medclipseg_busi/config.yaml"
    checkpoint_path = (
        root
        / "outputs/medclipseg_protocol_v3/runs/V3_medclipseg_busi/trained_models/seed123"
        / "MedCLIPSeg_unimedclip_ViT-B-16_best_dice.pth"
    )
    cfg = load_cfg_from_cfg_file(str(config))
    cfg.MODEL.DEVICE = "cuda"
    model = build_medclipseg_unimedclip(cfg)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval().to(device)
    torch.backends.cudnn.benchmark = True
    image_size = int(cfg.DATASET.SIZE)
    image = torch.rand(1, 3, image_size, image_size, device=device)
    prompt = "The lesion is located in the center with an irregular shape."

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    model_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
    model_bytes += sum(buffer.numel() * buffer.element_size() for buffer in model.buffers())
    gpu_name = torch.cuda.get_device_name(device)
    rows = []
    for samples, warmup, repeats, label in (
        (1, 10, 50, "MedCLIPSeg (1 sample)"),
        (30, 2, 10, "MedCLIPSeg (official 30 samples)"),
    ):
        timing = benchmark(model, image, prompt, samples, warmup, repeats, device)
        rows.append(
            {
                "method": label,
                "modality": "image+text",
                "input_size": f"{image_size}x{image_size}",
                "batch_size": 1,
                "precision": "FP32",
                "parameters": total,
                "parameters_million": total / 1e6,
                "trainable_parameters": trainable,
                "trainable_parameters_million": trainable / 1e6,
                "model_parameter_buffer_mb": model_bytes / (1024**2),
                "checkpoint_mb": checkpoint_path.stat().st_size / (1024**2),
                "online_text_encoder": "UniMedCLIP image tower + BiomedBERT text tower",
                "gpu": gpu_name,
                "warmup": warmup,
                "repeats": repeats,
                **timing,
                "notes": f"num_samples={samples}; official test.py uses 30",
                "checkpoint": str(checkpoint_path.resolve()),
            }
        )
        print(f"PASS {label}: params={total / 1e6:.2f}M latency={timing['latency_mean_ms']:.2f}ms")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "protocol": "single-image synchronized wall-clock forward latency",
                "gpu": gpu_name,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "data_loading_included": False,
                "postprocessing_included": False,
                "official_num_samples": 30,
                "output": str(output),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"output={output} rows={len(rows)}")


if __name__ == "__main__":
    main()
