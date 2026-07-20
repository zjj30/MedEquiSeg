#!/usr/bin/env python3
"""Run the R11 continuous-location recomputation ablation under Protocol V3."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import run_protocol_v3 as runner


RUN_ID = "V3_R11LR"
RUN_CONFIG = {
    "recipe": "biomed_lcaug_v2_atconv4",
    "augmentation": "lcaug_v2_dynamic_shared_plan_recompute_location_dataset",
}

R11LR_DATASETS = ("medclipseg_busi", "medclipseg_clinicdb")
R11LR_SEEDS = (123,)
R11LR_EPOCHS = 100


def requested_cache_scope() -> tuple[tuple[str, ...], tuple[int, ...], int]:
    """Read only the runner options that determine the mask-derived prompt cache."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--epochs", type=int, default=0)
    args, _ = parser.parse_known_args()
    datasets = tuple(args.datasets or R11LR_DATASETS)
    seeds = tuple(args.seeds or R11LR_SEEDS)
    epochs = args.epochs or R11LR_EPOCHS
    return datasets, seeds, epochs


def ensure_r11lr_cache(lock: dict, protocol_hash: str, gpu: int, dry_run: bool) -> Path:
    datasets, seeds, epochs = requested_cache_scope()
    unsupported = sorted(set(datasets) - set(R11LR_DATASETS))
    if unsupported:
        raise ValueError(f"R11-LR cache only supports {R11LR_DATASETS}; received {unsupported}")
    base_manifests = [str(runner.ROOT / cfg["manifest"]) for cfg in lock["datasets"].values()]
    selected_manifests = [
        str(runner.ROOT / lock["datasets"][dataset]["manifest"])
        for dataset in datasets
    ]
    seed_token = "seed" + "-".join(str(seed) for seed in seeds)
    prompt_manifest = (
        runner.ROOT
        / "outputs"
        / "text_embeddings"
        / f"r11lr_exact_prompts_{protocol_hash[:16]}_{seed_token}_{epochs}e.csv"
    )
    output = (
        runner.ROOT
        / "outputs"
        / "text_embeddings"
        / f"biomedclip_protocol_v3_r11lr_{protocol_hash[:16]}_{seed_token}_{epochs}e.npz"
    )
    collect_cmd = [
        runner.PYTHON,
        "smoke_tests/build_r11lr_prompt_manifest.py",
        "--manifest",
        *selected_manifests,
        "--seeds",
        *(str(seed) for seed in seeds),
        "--epochs",
        str(epochs),
        "--output",
        str(prompt_manifest),
    ]
    cache_cmd = [
        runner.PYTHON,
        "smoke_tests/build_causal_biomedclip_cache.py",
        "--manifest",
        *base_manifests,
        str(prompt_manifest),
        "--include-lcaug-variants",
        "--include-empty-control",
        "--output",
        str(output),
        "--device",
        "cuda:0",
    ]
    if dry_run:
        print("DRY R11-LR prompt manifest:", " ".join(collect_cmd), flush=True)
        print("DRY R11-LR cache:", " ".join(cache_cmd), flush=True)
        return output

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    subprocess.run(collect_cmd, cwd=runner.ROOT, env=env, check=True)
    subprocess.run(cache_cmd, cwd=runner.ROOT, env=env, check=True)
    return output


def main() -> None:
    runner.V3_RUNS[RUN_ID] = RUN_CONFIG
    runner.CODE_FILES = (
        *runner.CODE_FILES,
        "smoke_tests/run_protocol_v3_r11lr.py",
        "smoke_tests/build_r11lr_prompt_manifest.py",
    )
    runner.ensure_cache = ensure_r11lr_cache
    runner.main()


if __name__ == "__main__":
    main()
