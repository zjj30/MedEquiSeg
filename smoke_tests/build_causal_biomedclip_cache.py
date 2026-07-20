#!/usr/bin/env python3
"""Build BiomedCLIP text cache for CausalCLIPSeg incremental recipes."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from text_encoders import (
    TextEmbeddingCache,
    cache_expectations,
    unique_texts_from_manifest,
)

ROOT = Path("<PROJECT_ROOT>")
PYTHON = "<ENV_ROOT>/bin/python"
DEFAULT_MANIFEST = ROOT / "smoke_tests/dataset_manifest.csv"
DEFAULT_OUTPUT = ROOT / "outputs/text_embeddings/biomedclip_dataset_manifest_seed123.npz"
FALLBACK_CACHE = ROOT / "outputs/text_embeddings/biomedclip_public_private_lcaug_pooled_norm.npz"
DEFAULT_LOCAL_DIR = ROOT / "models/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", nargs="+", default=[str(DEFAULT_MANIFEST)])
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--include-lcaug-variants", action="store_true")
    parser.add_argument("--include-empty-control", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--biomedclip-local-dir", default=str(DEFAULT_LOCAL_DIR))
    parser.add_argument("--rebuild", action="store_true", help="Force an atomic cache rebuild.")
    parser.add_argument("--reuse-only", action="store_true", help="Fail instead of rebuilding a stale cache.")
    args = parser.parse_args()

    output = Path(args.output)
    texts = unique_texts_from_manifest(args.manifest, include_lcaug_variants=args.include_lcaug_variants)
    if args.include_empty_control:
        texts = sorted(set(texts) | {""})
    expected = cache_expectations(
        args.manifest,
        texts,
        args.include_lcaug_variants,
        args.include_empty_control,
    )
    if output.exists() and not args.rebuild:
        try:
            cache = TextEmbeddingCache(output)
            cache.validate_metadata(expected)
            cache.validate_texts(texts)
            print(f"Cache validated: {output} texts={len(texts)}")
            return
        except (KeyError, ValueError) as exc:
            if args.reuse_only:
                raise SystemExit(f"Cache validation failed under --reuse-only: {exc}")
            print(f"Cache stale; rebuilding atomically: {exc}")

    cmd = [
        PYTHON,
        str(ROOT / "smoke_tests/build_text_encoder_ablation_cache.py"),
        "--encoder",
        "biomedclip",
        "--manifest",
        *args.manifest,
        "--output",
        args.output,
        "--device",
        args.device,
        "--biomedclip-local-dir",
        args.biomedclip_local_dir,
    ]
    if args.include_lcaug_variants:
        cmd.append("--include-lcaug-variants")
    if args.include_empty_control:
        cmd.append("--include-empty-control")
    print("Running:", " ".join(cmd))
    rc = subprocess.call(cmd, cwd=ROOT)
    if rc != 0:
        raise SystemExit(rc)
    cache = TextEmbeddingCache(output)
    cache.validate_metadata(expected)
    cache.validate_texts(texts)
    print(f"Cache rebuilt and validated: {output} texts={len(texts)}")


if __name__ == "__main__":
    main()
