#!/usr/bin/env python3
"""Offline BiomedCLIP helpers (HF hub cache symlinks + open_clip load)."""

from __future__ import annotations

import os
from pathlib import Path

BIOMEDCLIP_HF_ID = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
BIOMEDCLIP_OPENCLIP_ID = f"hf-hub:{BIOMEDCLIP_HF_ID}"
DEFAULT_LOCAL_DIR = Path("<PROJECT_ROOT>/models/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
REQUIRED_FILES = (
    "open_clip_config.json",
    "open_clip_pytorch_model.bin",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "vocab.txt",
)


def resolve_local_dir(local_dir: str | Path | None = None) -> Path:
    env = os.environ.get("BIOMEDCLIP_LOCAL_DIR", "").strip()
    path = Path(local_dir or env or DEFAULT_LOCAL_DIR)
    if not path.is_dir():
        raise FileNotFoundError(f"BiomedCLIP local dir not found: {path}")
    missing = [name for name in REQUIRED_FILES if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"BiomedCLIP local dir missing files in {path}: {missing}")
    return path


def ensure_hf_hub_cache(local_dir: str | Path | None = None, cache_root: str | Path | None = None) -> Path:
    """Symlink local BiomedCLIP files into an open_clip HF hub cache layout."""
    model_dir = resolve_local_dir(local_dir)
    root = Path(cache_root or os.environ.get("OPENCLIP_HF_CACHE", "<PROJECT_ROOT>/repos/NextGen-UIA/ckpt"))
    cache = root / f"models--{BIOMEDCLIP_HF_ID.replace('/', '--')}"
    snapshot = cache / "snapshots" / "local"
    (cache / "refs").mkdir(parents=True, exist_ok=True)
    snapshot.mkdir(parents=True, exist_ok=True)
    (cache / "refs" / "main").write_text("local\n", encoding="utf-8")
    for name in REQUIRED_FILES:
        target = snapshot / name
        if target.is_symlink() or target.exists():
            continue
        target.symlink_to(model_dir / name)
    return snapshot


def open_clip_model_id(local_dir: str | Path | None = None) -> str:
    """Prefer offline local-dir schema when weights are available on disk."""
    try:
        return f"local-dir:{resolve_local_dir(local_dir)}"
    except FileNotFoundError:
        return BIOMEDCLIP_OPENCLIP_ID


def load_open_clip_biomedclip(device, local_dir: str | Path | None = None):
    import open_clip

    model_id = open_clip_model_id(local_dir)
    if model_id.startswith("local-dir:"):
        model, _, _ = open_clip.create_model_and_transforms(model_id)
    else:
        ensure_hf_hub_cache(local_dir=local_dir)
        model, _, _ = open_clip.create_model_and_transforms(BIOMEDCLIP_OPENCLIP_ID)
    tokenizer = open_clip.get_tokenizer(model_id)
    return model.eval().to(device), tokenizer
