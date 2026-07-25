#!/usr/bin/env python3
import argparse
import csv
import datetime as _dt
import hashlib
import importlib
import os
import random
import socket
import subprocess
import sys
import time
import traceback
import types
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import DataLoader, Dataset

from augmentation_plugins import get_augmentation_plugin
from causal_atconv_plugin import apply_atconv_plugin, atconv_meta, list_atconv_candidates
from causal_clip_recipe import apply_recipe_to_args, get_recipe, recipe_meta, recipe_names, resolve_conv_plugin
from causal_text_encoder_plugins import build_text_encoder_plugin
from image_resize import resize_image_mask_pair
from text_encoders import lcaug_text_variants, load_text_embedding_cache
from protocol_v3.core import (
    binarize_mask,
    load_manifest_splits as load_protocol_manifest_splits,
    protocol_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "smoke_tests/protocol_v3/manifests/medclipseg_busi_full.csv"
OUT_DIR = ROOT / "outputs" / "training"
ARTIFACT_DIR = ROOT / "smoke_tests" / "_artifacts"
OPENAI_CLIP_RN50_URL = (
    "https://openaipublic.azureedge.net/clip/models/"
    "afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt"
)
OPENAI_CLIP_RN101_URL = (
    "https://openaipublic.azureedge.net/clip/models/"
    "8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt"
)
OPENAI_CLIP_BPE_URL = "https://openaipublic.azureedge.net/clip/bpe_simple_vocab_16e6.txt.gz"


def repo_path(name):
    return ROOT / "repos" / name


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stable_seed(*parts):
    h = hashlib.sha256("||".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little") % (2**32)


def augmentation_plan_seed_key(aug_strength):
    """Pair LCAug-v2 rewrite/no-rewrite runs with identical image transforms."""
    key = str(aug_strength or "")
    if key.startswith("lcaug_v2"):
        return key.replace("_no_text_rewrite", "")
    return key


def stable_text_embedding(texts, tokens, dim, device):
    arrays = []
    for text in texts:
        rng = np.random.default_rng(stable_seed(text, tokens, dim))
        arrays.append(rng.normal(0.0, 0.02, size=(tokens, dim)).astype("float32"))
    return torch.from_numpy(np.stack(arrays)).to(device)


def stable_text_tokens(texts, length, vocab_size, device):
    rows = []
    for text in texts:
        rng = np.random.default_rng(stable_seed(text, "tokens"))
        row = rng.integers(1, vocab_size - 1, size=(length,), dtype=np.int64)
        row[-1] = vocab_size - 1
        rows.append(row)
    return torch.from_numpy(np.stack(rows)).long().to(device)


def ensure_artifact(path, url):
    path = Path(path)
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(path)
    return path


_CLIP_TOKENIZER_CACHE = {}


def clip_bpe_tokens(texts, length, device, bpe_path):
    if "ftfy" not in sys.modules:
        ftfy_stub = types.ModuleType("ftfy")
        ftfy_stub.fix_text = lambda text: text
        sys.modules["ftfy"] = ftfy_stub
    from simple_tokenizer import SimpleTokenizer

    bpe_path = str(bpe_path)
    tokenizer = _CLIP_TOKENIZER_CACHE.get(bpe_path)
    if tokenizer is None:
        tokenizer = SimpleTokenizer(bpe_path=bpe_path)
        _CLIP_TOKENIZER_CACHE[bpe_path] = tokenizer
    sot = tokenizer.encoder["<|startoftext|>"]
    eot = tokenizer.encoder["<|endoftext|>"]
    result = torch.zeros(len(texts), length, dtype=torch.long)
    for i, text in enumerate(texts):
        tokens = [sot] + tokenizer.encode(str(text)) + [eot]
        if len(tokens) > length:
            tokens = tokens[:length]
            tokens[-1] = eot
        result[i, : len(tokens)] = torch.tensor(tokens, dtype=torch.long)
    return result.to(device)


def text_bank(name, classes=4, groups=4, tokens=30, dim=768, device=None):
    rng = np.random.default_rng(stable_seed(name, classes, groups, tokens, dim))
    bank = torch.from_numpy(rng.normal(0.0, 0.02, size=(classes, groups, tokens, dim)).astype("float32"))
    return bank.to(device) if device else bank


def tiny_config():
    transformer = SimpleNamespace(
        num_heads=4,
        num_layers=1,
        embeddings_dropout_rate=0.0,
        attention_dropout_rate=0.0,
        dropout_rate=0.0,
    )
    return SimpleNamespace(
        transformer=transformer,
        KV_size=960,
        expand_ratio=4,
        patch_sizes=[16, 8, 4, 2],
        base_channel=64,
        n_classes=1,
    )


class ManifestDataset(Dataset):
    AUGMENT_PROFILES = {
        "light": {
            "rotate_deg": 10.0,
            "shift_frac": 0.04,
            "zoom_max": 1.12,
            "color_jitter": 0.10,
            "noise_std": 0.01,
            "hflip_p": 0.5,
            "vflip_p": 0.2,
            "rot90_p": 0.35,
            "affine_p": 0.7,
            "zoom_p": 0.45,
            "color_p": 0.8,
        },
        "medium": {
            "rotate_deg": 18.0,
            "shift_frac": 0.07,
            "zoom_max": 1.22,
            "color_jitter": 0.16,
            "noise_std": 0.015,
            "hflip_p": 0.5,
            "vflip_p": 0.2,
            "rot90_p": 0.35,
            "affine_p": 0.7,
            "zoom_p": 0.45,
            "color_p": 0.8,
        },
        "strong": {
            "rotate_deg": 28.0,
            "shift_frac": 0.10,
            "zoom_max": 1.35,
            "color_jitter": 0.24,
            "noise_std": 0.025,
            "hflip_p": 0.5,
            "vflip_p": 0.2,
            "rot90_p": 0.35,
            "affine_p": 0.7,
            "zoom_p": 0.45,
            "color_p": 0.8,
        },
        "safe_light": {
            "rotate_deg": 8.0,
            "shift_frac": 0.035,
            "zoom_max": 1.10,
            "color_jitter": 0.08,
            "noise_std": 0.005,
            "gamma_range": (0.95, 1.05),
            "gamma_p": 0.0,
            "blur_p": 0.0,
            "blur_radius": 0.0,
            "elastic_p": 0.0,
            "elastic_alpha_frac": 0.0,
            "elastic_sigma_frac": 0.0,
            "hflip_p": 0.0,
            "vflip_p": 0.0,
            "rot90_p": 0.0,
            "affine_p": 0.75,
            "zoom_p": 0.35,
            "color_p": 0.65,
        },
        "appearance": {
            "rotate_deg": 0.0,
            "shift_frac": 0.0,
            "zoom_max": 1.0,
            "color_jitter": 0.12,
            "noise_std": 0.008,
            "gamma_range": (0.85, 1.18),
            "gamma_p": 0.50,
            "blur_p": 0.25,
            "blur_radius": 0.7,
            "elastic_p": 0.0,
            "elastic_alpha_frac": 0.0,
            "elastic_sigma_frac": 0.0,
            "hflip_p": 0.0,
            "vflip_p": 0.0,
            "rot90_p": 0.0,
            "affine_p": 0.0,
            "zoom_p": 0.0,
            "color_p": 0.85,
        },
        "elastic_light": {
            "rotate_deg": 6.0,
            "shift_frac": 0.025,
            "zoom_max": 1.08,
            "color_jitter": 0.06,
            "noise_std": 0.004,
            "gamma_range": (0.92, 1.10),
            "gamma_p": 0.35,
            "blur_p": 0.10,
            "blur_radius": 0.5,
            "elastic_p": 0.20,
            "elastic_alpha_frac": 0.018,
            "elastic_sigma_frac": 0.055,
            "hflip_p": 0.0,
            "vflip_p": 0.0,
            "rot90_p": 0.0,
            "affine_p": 0.60,
            "zoom_p": 0.25,
            "color_p": 0.55,
        },
    }
    DATASET_SAFE_LIGHT = {
        "cvc": {
            "rotate_deg": 8.0,
            "shift_frac": 0.035,
            "zoom_max": 1.10,
            "color_jitter": 0.08,
            "noise_std": 0.005,
            "gamma_range": (0.92, 1.08),
            "gamma_p": 0.25,
            "blur_p": 0.0,
            "blur_radius": 0.0,
            "elastic_p": 0.0,
            "elastic_alpha_frac": 0.0,
            "elastic_sigma_frac": 0.0,
            "hflip_p": 0.0,
            "vflip_p": 0.0,
            "rot90_p": 0.0,
            "affine_p": 0.75,
            "zoom_p": 0.35,
            "color_p": 0.65,
        },
        "glas": {
            "rotate_deg": 10.0,
            "shift_frac": 0.04,
            "zoom_max": 1.12,
            "color_jitter": 0.05,
            "noise_std": 0.003,
            "gamma_range": (0.95, 1.05),
            "gamma_p": 0.20,
            "blur_p": 0.0,
            "blur_radius": 0.0,
            "elastic_p": 0.0,
            "elastic_alpha_frac": 0.0,
            "elastic_sigma_frac": 0.0,
            "hflip_p": 0.0,
            "vflip_p": 0.0,
            "rot90_p": 0.0,
            "affine_p": 0.75,
            "zoom_p": 0.40,
            "color_p": 0.55,
        },
        "busi": {
            "rotate_deg": 6.0,
            "shift_frac": 0.03,
            "zoom_max": 1.08,
            "color_jitter": 0.06,
            "noise_std": 0.004,
            "gamma_range": (0.90, 1.10),
            "gamma_p": 0.30,
            "blur_p": 0.0,
            "blur_radius": 0.0,
            "elastic_p": 0.0,
            "elastic_alpha_frac": 0.0,
            "elastic_sigma_frac": 0.0,
            "hflip_p": 0.0,
            "vflip_p": 0.0,
            "rot90_p": 0.0,
            "affine_p": 0.65,
            "zoom_p": 0.30,
            "color_p": 0.55,
        },
    }
    DATASET_APPEARANCE = {
        "cvc": {
            "rotate_deg": 0.0,
            "shift_frac": 0.0,
            "zoom_max": 1.0,
            "color_jitter": 0.14,
            "noise_std": 0.006,
            "gamma_range": (0.84, 1.20),
            "gamma_p": 0.55,
            "blur_p": 0.18,
            "blur_radius": 0.6,
            "elastic_p": 0.0,
            "elastic_alpha_frac": 0.0,
            "elastic_sigma_frac": 0.0,
            "hflip_p": 0.0,
            "vflip_p": 0.0,
            "rot90_p": 0.0,
            "affine_p": 0.0,
            "zoom_p": 0.0,
            "color_p": 0.85,
        },
        "glas": {
            "rotate_deg": 0.0,
            "shift_frac": 0.0,
            "zoom_max": 1.0,
            "color_jitter": 0.10,
            "noise_std": 0.003,
            "gamma_range": (0.90, 1.12),
            "gamma_p": 0.45,
            "blur_p": 0.08,
            "blur_radius": 0.45,
            "elastic_p": 0.0,
            "elastic_alpha_frac": 0.0,
            "elastic_sigma_frac": 0.0,
            "hflip_p": 0.0,
            "vflip_p": 0.0,
            "rot90_p": 0.0,
            "affine_p": 0.0,
            "zoom_p": 0.0,
            "color_p": 0.75,
        },
        "busi": {
            "rotate_deg": 0.0,
            "shift_frac": 0.0,
            "zoom_max": 1.0,
            "color_jitter": 0.08,
            "noise_std": 0.010,
            "gamma_range": (0.82, 1.22),
            "gamma_p": 0.60,
            "blur_p": 0.22,
            "blur_radius": 0.65,
            "elastic_p": 0.0,
            "elastic_alpha_frac": 0.0,
            "elastic_sigma_frac": 0.0,
            "hflip_p": 0.0,
            "vflip_p": 0.0,
            "rot90_p": 0.0,
            "affine_p": 0.0,
            "zoom_p": 0.0,
            "color_p": 0.65,
        },
    }
    DATASET_ELASTIC_LIGHT = {
        "cvc": {
            "rotate_deg": 6.0,
            "shift_frac": 0.025,
            "zoom_max": 1.08,
            "color_jitter": 0.06,
            "noise_std": 0.004,
            "gamma_range": (0.92, 1.10),
            "gamma_p": 0.30,
            "blur_p": 0.08,
            "blur_radius": 0.45,
            "elastic_p": 0.18,
            "elastic_alpha_frac": 0.016,
            "elastic_sigma_frac": 0.055,
            "hflip_p": 0.0,
            "vflip_p": 0.0,
            "rot90_p": 0.0,
            "affine_p": 0.60,
            "zoom_p": 0.25,
            "color_p": 0.55,
        },
        "glas": {
            "rotate_deg": 8.0,
            "shift_frac": 0.030,
            "zoom_max": 1.10,
            "color_jitter": 0.04,
            "noise_std": 0.002,
            "gamma_range": (0.96, 1.06),
            "gamma_p": 0.20,
            "blur_p": 0.05,
            "blur_radius": 0.35,
            "elastic_p": 0.25,
            "elastic_alpha_frac": 0.020,
            "elastic_sigma_frac": 0.060,
            "hflip_p": 0.0,
            "vflip_p": 0.0,
            "rot90_p": 0.0,
            "affine_p": 0.60,
            "zoom_p": 0.30,
            "color_p": 0.45,
        },
        "busi": {
            "rotate_deg": 4.0,
            "shift_frac": 0.020,
            "zoom_max": 1.06,
            "color_jitter": 0.05,
            "noise_std": 0.008,
            "gamma_range": (0.90, 1.12),
            "gamma_p": 0.35,
            "blur_p": 0.12,
            "blur_radius": 0.45,
            "elastic_p": 0.12,
            "elastic_alpha_frac": 0.012,
            "elastic_sigma_frac": 0.050,
            "hflip_p": 0.0,
            "vflip_p": 0.0,
            "rot90_p": 0.0,
            "affine_p": 0.45,
            "zoom_p": 0.20,
            "color_p": 0.45,
        },
    }

    def __init__(
        self,
        rows,
        image_size,
        augment=False,
        aug_strength="medium",
        resize_mode="stretch",
        seed=123,
    ):
        self.rows = rows
        self.image_size = image_size
        self.resize_mode = resize_mode
        self.augment = augment
        self.aug_strength = aug_strength
        self.seed = int(seed)
        self.epoch = 0
        self.augmentation_plugin = get_augmentation_plugin(aug_strength)
        self.profile = self.AUGMENT_PROFILES.get(aug_strength, self.AUGMENT_PROFILES["safe_light"])

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.rows)

    def _profile_for_row(self, row):
        if self.aug_strength == "safe_light_dataset":
            return self.DATASET_SAFE_LIGHT.get(row["dataset"], self.AUGMENT_PROFILES["safe_light"])
        if self.aug_strength == "appearance_dataset":
            return self.DATASET_APPEARANCE.get(row["dataset"], self.AUGMENT_PROFILES["appearance"])
        if self.aug_strength == "elastic_light_dataset":
            return self.DATASET_ELASTIC_LIGHT.get(row["dataset"], self.AUGMENT_PROFILES["elastic_light"])
        return self.profile

    def _elastic_deform_pair(self, img, mask, profile):
        from scipy.ndimage import gaussian_filter, map_coordinates

        h, w = self.image_size, self.image_size
        alpha = max(0.0, float(profile.get("elastic_alpha_frac", 0.0)) * self.image_size)
        sigma = max(1.0, float(profile.get("elastic_sigma_frac", 0.0)) * self.image_size)
        if alpha <= 0:
            return img, mask

        dx = gaussian_filter((np.random.rand(h, w) * 2.0 - 1.0), sigma=sigma, mode="reflect") * alpha
        dy = gaussian_filter((np.random.rand(h, w) * 2.0 - 1.0), sigma=sigma, mode="reflect") * alpha
        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        coords = (yy + dy, xx + dx)

        img_arr = np.asarray(img).astype("float32")
        warped = np.stack(
            [map_coordinates(img_arr[..., c], coords, order=1, mode="reflect") for c in range(3)],
            axis=-1,
        )
        mask_arr = np.asarray(mask).astype("float32")
        warped_mask = map_coordinates(mask_arr, coords, order=0, mode="nearest")
        return Image.fromarray(warped.clip(0, 255).astype("uint8")), Image.fromarray((warped_mask > 127).astype("uint8") * 255)

    def _apply_gamma(self, img, profile):
        lo, hi = profile.get("gamma_range", (1.0, 1.0))
        gamma = random.uniform(float(lo), float(hi))
        arr = np.asarray(img).astype("float32") / 255.0
        arr = np.power(np.clip(arr, 0.0, 1.0), gamma)
        return Image.fromarray((arr * 255.0).clip(0, 255).astype("uint8"))

    def _augment_pair(self, img, mask, profile):
        if random.random() < profile.get("hflip_p", 0.5):
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if random.random() < profile.get("vflip_p", 0.2):
            img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        if random.random() < profile.get("rot90_p", 0.35):
            k = random.randint(1, 3)
            img = img.rotate(90 * k, resample=Image.Resampling.BILINEAR)
            mask = mask.rotate(90 * k, resample=Image.Resampling.NEAREST)
        if random.random() < profile.get("affine_p", 0.7):
            angle = random.uniform(-profile["rotate_deg"], profile["rotate_deg"])
            max_shift = int(round(self.image_size * profile["shift_frac"]))
            shift = (random.randint(-max_shift, max_shift), random.randint(-max_shift, max_shift))
            img = img.rotate(
                angle,
                resample=Image.Resampling.BILINEAR,
                translate=shift,
                fillcolor=(0, 0, 0),
            )
            mask = mask.rotate(
                angle,
                resample=Image.Resampling.NEAREST,
                translate=shift,
                fillcolor=0,
            )
        if random.random() < profile.get("zoom_p", 0.45):
            zoom = random.uniform(1.0, profile["zoom_max"])
            crop_size = max(16, int(round(self.image_size / zoom)))
            left = random.randint(0, self.image_size - crop_size)
            top = random.randint(0, self.image_size - crop_size)
            box = (left, top, left + crop_size, top + crop_size)
            img = img.crop(box).resize((self.image_size, self.image_size), Image.BILINEAR)
            mask = mask.crop(box).resize((self.image_size, self.image_size), Image.NEAREST)
        jitter = profile["color_jitter"]
        if random.random() < profile.get("color_p", 0.8):
            img = ImageEnhance.Brightness(img).enhance(random.uniform(1.0 - jitter, 1.0 + jitter))
            img = ImageEnhance.Contrast(img).enhance(random.uniform(1.0 - jitter, 1.0 + jitter))
            img = ImageEnhance.Color(img).enhance(random.uniform(1.0 - jitter, 1.0 + jitter))
        if random.random() < profile.get("gamma_p", 0.0):
            img = self._apply_gamma(img, profile)
        if random.random() < profile.get("blur_p", 0.0):
            img = img.filter(ImageFilter.GaussianBlur(radius=profile.get("blur_radius", 0.5)))
        if random.random() < profile.get("elastic_p", 0.0):
            img, mask = self._elastic_deform_pair(img, mask, profile)
        return img, mask

    def __getitem__(self, idx):
        row = self.rows[idx]
        with Image.open(row["image_path"]) as img:
            img = img.convert("RGB")
        with Image.open(row["mask_path"]) as mask:
            mask = mask.convert("L")
        img, mask = resize_image_mask_pair(img, mask, self.image_size, self.resize_mode)
        text = row["text"]
        profile = self._profile_for_row(row)
        if self.augment:
            if self.augmentation_plugin is not None:
                if self.aug_strength.startswith("lcaug_v2"):
                    case_id = row.get("case_id") or row.get("filename") or row["image_path"]
                    plan_key = augmentation_plan_seed_key(self.aug_strength)
                    rng = random.Random(stable_seed(self.seed, self.epoch, case_id, plan_key))
                    img, mask, text = self.augmentation_plugin(
                        img,
                        mask,
                        text,
                        row["dataset"],
                        self.image_size,
                        rng=rng,
                    )
                else:
                    img, mask, text = self.augmentation_plugin(img, mask, text, row["dataset"], self.image_size)
            else:
                img, mask = self._augment_pair(img, mask, profile)
        image = torch.from_numpy(np.asarray(img).transpose(2, 0, 1).copy()).float() / 255.0
        if self.augment and self.augmentation_plugin is None and profile["noise_std"] > 0 and random.random() < 0.25:
            image = (image + torch.randn_like(image) * profile["noise_std"]).clamp(0.0, 1.0)
        mask_mode = row.get("mask_mode", "").strip()
        if mask_mode:
            mask_arr = binarize_mask(np.asarray(mask).copy(), mask_mode)
        else:
            mask_arr = (np.asarray(mask).copy() > 0).astype("float32")
        mask_tensor = torch.from_numpy(mask_arr)[None, :, :]
        return image, mask_tensor, text, row["dataset"], Path(row["image_path"]).name


def load_rows(seed, max_train, max_val, dataset_filter=None, split_seed=None, manifest_path=None):
    manifest_path = Path(manifest_path) if manifest_path else MANIFEST
    with manifest_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if dataset_filter:
        allowed = set(dataset_filter)
        rows = [r for r in rows if r["dataset"] in allowed]
    rows = sorted(rows, key=lambda r: (r["dataset"], r["image_path"]))
    split_values = {r.get("split", "").strip().lower() for r in rows}
    has_explicit_split = bool(split_values & {"train", "val", "valid", "validation", "test"})
    if has_explicit_split:
        train_rows = [r for r in rows if r.get("split", "").strip().lower() == "train"]
        val_rows = [
            r
            for r in rows
            if r.get("split", "").strip().lower() in {"val", "valid", "validation"}
        ]
        if not val_rows and "test" in split_values:
            # Legacy manifests may use test as the only held-out split.
            val_rows = [r for r in rows if r.get("split", "").strip().lower() == "test"]
    else:
        rng = random.Random(seed if split_seed is None else split_seed)
        rng.shuffle(rows)
        split = max(1, int(0.8 * len(rows)))
        train_rows = rows[:split]
        val_rows = rows[split:]
    if max_train:
        train_rows = train_rows[:max_train]
    if max_val:
        val_rows = val_rows[:max_val]
    return train_rows, val_rows


def load_manifest_splits(manifest_path, dataset_filter=None, check_files=True):
    """Load Protocol V3 splits without ever aliasing test rows to validation."""
    return load_protocol_manifest_splits(
        manifest_path,
        dataset_filter,
        require_train_val_test=True,
        check_files=check_files,
    )


def prepend_sys_path(path):
    path = str(path)
    sys.path.insert(0, path)
    return path


def cleanup_sys_path(path):
    try:
        sys.path.remove(str(path))
    except ValueError:
        pass


def cleanup_modules(prefixes):
    for name in list(sys.modules):
        if any(name == p or name.startswith(p + ".") for p in prefixes):
            sys.modules.pop(name, None)


def install_stpnet_imageaggr_shim():
    module = types.ModuleType("nets.ImageAggr")

    class EncoderImageAggr(nn.Module):
        def __init__(self, img_dim=768, embed_size=768):
            super().__init__()
            self.proj = nn.Identity() if img_dim == embed_size else nn.Linear(img_dim, embed_size)

        def forward(self, images, image_lengths):
            return self.proj(images.mean(dim=1))

    module.EncoderImageAggr = EncoderImageAggr
    sys.modules["nets.ImageAggr"] = module


def patch_torchvision_resnet101(module):
    import torchvision.models as tv_models

    def no_download_resnet101(pretrained=True, *args, **kwargs):
        return tv_models.resnet101(weights=None)

    module.resnet101 = no_download_resnet101


def patch_stage1_resnet_no_download(module):
    original = module.resnet50

    def no_download_resnet50(*args, **kwargs):
        return original(pretrained=False)

    module.resnet50 = no_download_resnet50


def make_causal_clip_checkpoint(path):
    if path.exists():
        return
    from nets.clip import CLIP

    class TraceableCLIP(CLIP):
        def forward(self, image, text):
            return self.encode_text(text)[1]

    clip_stub = TraceableCLIP(
        embed_dim=512,
        image_resolution=224,
        vision_layers=(1, 1, 1, 1),
        vision_width=64,
        vision_patch_size=None,
        context_length=16,
        txt_length=16,
        vocab_size=1000,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=1,
    ).eval()
    example_image = torch.randn(1, 3, 224, 224)
    example_text = torch.randint(1, 999, (1, 16), dtype=torch.long)
    example_text[:, -1] = 999
    traced = torch.jit.trace(clip_stub, (example_image, example_text), strict=False)
    torch.jit.save(traced, str(path))


class TextFiLMLViT(nn.Module):
    def __init__(self, base_model, text_dim=768, hidden_dim=128, scale=0.10):
        super().__init__()
        self.base_model = base_model
        self.scale = scale
        self.film = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 6),
        )
        nn.init.zeros_(self.film[-1].weight)
        nn.init.zeros_(self.film[-1].bias)

    def forward(self, images, text_embeddings):
        pooled = text_embeddings.mean(dim=1)
        gamma_beta = self.film(pooled).view(images.shape[0], 6, 1, 1)
        gamma, beta = gamma_beta[:, :3], gamma_beta[:, 3:]
        conditioned = images * (1.0 + self.scale * torch.tanh(gamma)) + self.scale * torch.tanh(beta)
        return self.base_model(conditioned, text_embeddings)


class TextAdapterLViT(nn.Module):
    def __init__(self, base_model, text_dim=768, hidden_dim=1536, scale=0.50):
        super().__init__()
        self.base_model = base_model
        self.scale = scale
        self.adapter = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_dim, text_dim),
        )
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

    def forward(self, images, text_embeddings):
        adapted_text = text_embeddings + self.scale * self.adapter(text_embeddings)
        return self.base_model(images, adapted_text)


class TextImageAdapterLViT(nn.Module):
    def __init__(self, base_model, text_dim=768, text_hidden_dim=1536, image_channels=24, text_scale=0.50, image_scale=0.25):
        super().__init__()
        self.base_model = base_model
        self.text_scale = text_scale
        self.image_scale = image_scale
        self.text_adapter = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, text_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(text_hidden_dim, text_dim),
        )
        self.image_in = nn.Conv2d(3, image_channels, kernel_size=3, padding=1, bias=False)
        self.image_norm = nn.GroupNorm(num_groups=4, num_channels=image_channels)
        self.image_gate = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, image_channels * 2),
        )
        self.image_out = nn.Conv2d(image_channels, 3, kernel_size=3, padding=1)
        nn.init.zeros_(self.text_adapter[-1].weight)
        nn.init.zeros_(self.text_adapter[-1].bias)
        nn.init.zeros_(self.image_gate[-1].weight)
        nn.init.zeros_(self.image_gate[-1].bias)
        nn.init.zeros_(self.image_out.weight)
        nn.init.zeros_(self.image_out.bias)

    def forward(self, images, text_embeddings):
        adapted_text = text_embeddings + self.text_scale * self.text_adapter(text_embeddings)
        pooled = adapted_text.mean(dim=1)
        gamma, beta = self.image_gate(pooled).chunk(2, dim=1)
        gamma = gamma.view(images.shape[0], -1, 1, 1)
        beta = beta.view(images.shape[0], -1, 1, 1)
        features = self.image_norm(self.image_in(images))
        features = features * (1.0 + self.image_scale * torch.tanh(gamma)) + self.image_scale * torch.tanh(beta)
        delta = self.image_out(F.gelu(features))
        conditioned = images + self.image_scale * delta
        return self.base_model(conditioned, adapted_text)


class ConvGNAct(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        groups = min(8, out_channels)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class TextGate2d(nn.Module):
    def __init__(self, text_dim, channels, scale=0.25):
        super().__init__()
        self.scale = scale
        self.to_gate = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, channels * 2),
        )
        nn.init.zeros_(self.to_gate[-1].weight)
        nn.init.zeros_(self.to_gate[-1].bias)

    def forward(self, x, text_embeddings):
        pooled = text_embeddings.mean(dim=1)
        gamma, beta = self.to_gate(pooled).chunk(2, dim=1)
        gamma = gamma.view(x.shape[0], -1, 1, 1)
        beta = beta.view(x.shape[0], -1, 1, 1)
        return x * (1.0 + self.scale * torch.tanh(gamma)) + self.scale * torch.tanh(beta)


class TextCrossAttention2d(nn.Module):
    def __init__(self, channels, text_dim=768, heads=4, scale=0.20):
        super().__init__()
        self.scale = scale
        self.norm_x = nn.GroupNorm(min(8, channels), channels)
        self.norm_text = nn.LayerNorm(text_dim)
        self.text_proj = nn.Linear(text_dim, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads=heads, batch_first=True)
        self.out = nn.Linear(channels, channels)

    def forward(self, x, text_embeddings):
        b, c, h, w = x.shape
        q = self.norm_x(x).flatten(2).transpose(1, 2)
        kv = self.text_proj(self.norm_text(text_embeddings))
        attended, _ = self.attn(q, kv, kv, need_weights=False)
        attended = self.out(attended).transpose(1, 2).view(b, c, h, w)
        return x + self.scale * attended


class RN50TokenCrossAttention2d(nn.Module):
    def __init__(self, channels, text_dim=512, attn_dim=256, heads=4, scale=0.15):
        super().__init__()
        self.scale = scale
        self.norm_x = nn.GroupNorm(min(8, channels), channels)
        self.q_proj = nn.Conv2d(channels, attn_dim, kernel_size=1, bias=False)
        self.text_norm = nn.LayerNorm(text_dim)
        self.text_proj = nn.Linear(text_dim, attn_dim)
        self.attn = nn.MultiheadAttention(attn_dim, num_heads=heads, batch_first=True)
        self.out_proj = nn.Conv2d(attn_dim, channels, kernel_size=1)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x, text_tokens):
        b, _, h, w = x.shape
        q = self.q_proj(self.norm_x(x)).flatten(2).transpose(1, 2)
        kv = self.text_proj(self.text_norm(text_tokens.float()))
        attended, _ = self.attn(q, kv, kv, need_weights=False)
        attended = attended.transpose(1, 2).view(b, -1, h, w)
        return x + self.scale * self.out_proj(attended).to(dtype=x.dtype)


class TextCrossAttentionUNet(nn.Module):
    def __init__(self, text_dim=768, base_channels=32):
        super().__init__()
        c1, c2, c3, c4, cb = base_channels, base_channels * 2, base_channels * 4, base_channels * 8, base_channels * 8
        self.enc1 = ConvGNAct(3, c1)
        self.enc2 = ConvGNAct(c1, c2)
        self.enc3 = ConvGNAct(c2, c3)
        self.enc4 = ConvGNAct(c3, c4)
        self.bottleneck = ConvGNAct(c4, cb)
        self.pool = nn.MaxPool2d(2)
        self.gate1 = TextGate2d(text_dim, c1)
        self.gate2 = TextGate2d(text_dim, c2)
        self.gate3 = TextGate2d(text_dim, c3)
        self.gate4 = TextGate2d(text_dim, c4)
        self.cross = TextCrossAttention2d(cb, text_dim=text_dim, heads=4)
        self.up4 = nn.ConvTranspose2d(cb, c4, kernel_size=2, stride=2)
        self.dec4 = ConvGNAct(c4 + c4, c4)
        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = ConvGNAct(c3 + c3, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = ConvGNAct(c2 + c2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = ConvGNAct(c1 + c1, c1)
        self.dec_gate4 = TextGate2d(text_dim, c4)
        self.dec_gate3 = TextGate2d(text_dim, c3)
        self.dec_gate2 = TextGate2d(text_dim, c2)
        self.dec_gate1 = TextGate2d(text_dim, c1)
        self.out = nn.Conv2d(c1, 1, kernel_size=1)

    def forward(self, images, text_embeddings):
        e1 = self.gate1(self.enc1(images), text_embeddings)
        e2 = self.gate2(self.enc2(self.pool(e1)), text_embeddings)
        e3 = self.gate3(self.enc3(self.pool(e2)), text_embeddings)
        e4 = self.gate4(self.enc4(self.pool(e3)), text_embeddings)
        b = self.cross(self.bottleneck(self.pool(e4)), text_embeddings)
        d4 = self.dec_gate4(self.dec4(torch.cat([self.up4(b), e4], dim=1)), text_embeddings)
        d3 = self.dec_gate3(self.dec3(torch.cat([self.up3(d4), e3], dim=1)), text_embeddings)
        d2 = self.dec_gate2(self.dec2(torch.cat([self.up2(d3), e2], dim=1)), text_embeddings)
        d1 = self.dec_gate1(self.dec1(torch.cat([self.up1(d2), e1], dim=1)), text_embeddings)
        return self.out(d1)


class CausalCLIPSegRN50Adapter(nn.Module):
    def __init__(
        self,
        base_model,
        freeze_backbone=False,
        text_gate=False,
        refine_decoder=False,
        token_cross_attn=False,
        deep_fuse=False,
        albef_fuse=False,
        mome_fuse=False,
        state_dim=1024,
        token_dim=512,
        v5_dim=1024,
        text_encoder_module=None,
    ):
        super().__init__()
        self.base_model = base_model
        self.text_encoder_module = text_encoder_module
        self.freeze_backbone = freeze_backbone
        self.text_gate = text_gate
        self.refine_decoder = refine_decoder
        self.token_cross_attn = token_cross_attn
        self.deep_fuse = deep_fuse
        self.albef_fuse = albef_fuse
        self.mome_fuse = mome_fuse
        self.v5_proj = nn.Sequential(
            nn.Conv2d(v5_dim, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        if self.text_gate:
            self.v3_gate = self._make_gate(state_dim, 512)
            self.v4_gate = self._make_gate(state_dim, 1024)
            self.v5_gate = self._make_gate(state_dim, 512)
        if self.refine_decoder:
            self.refine_head = RN50TextRefinementHead(text_dim=state_dim)
        if self.token_cross_attn:
            self.v3_cross = RN50TokenCrossAttention2d(512, text_dim=token_dim, attn_dim=256, heads=4, scale=0.15)
            self.v4_cross = RN50TokenCrossAttention2d(1024, text_dim=token_dim, attn_dim=256, heads=4, scale=0.15)
            self.v5_cross = RN50TokenCrossAttention2d(512, text_dim=token_dim, attn_dim=256, heads=4, scale=0.15)
        if self.deep_fuse:
            self.deep_fuse_head = RN50TextDeepFuseHead(text_dim=state_dim, token_dim=token_dim)
        if self.albef_fuse:
            self.albef_head = RN50AlignBeforeFuseHead(text_dim=state_dim, token_dim=token_dim)
        if self.mome_fuse:
            self.mome_head = RN50MoMEFuseHead(text_dim=state_dim, token_dim=token_dim)
        if self.freeze_backbone:
            self.base_model.backbone.requires_grad_(False)
            self.base_model.backbone.eval()

    @staticmethod
    def _make_gate(text_dim, channels):
        gate = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, max(128, channels // 2)),
            nn.GELU(),
            nn.Linear(max(128, channels // 2), channels * 2),
        )
        nn.init.zeros_(gate[-1].weight)
        nn.init.zeros_(gate[-1].bias)
        return gate

    @staticmethod
    def _apply_gate(feature, state, gate):
        gamma_beta = gate(state).view(state.shape[0], -1, 1, 1)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        return feature * (1.0 + 0.25 * torch.tanh(gamma)) + 0.25 * torch.tanh(beta)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.base_model.backbone.eval()
        return self

    def forward(self, img, word=None, texts=None, text_features=None, mask=None):
        v3, v4, v5 = self.base_model.backbone.encode_image(img)
        v5 = self.v5_proj(v5)
        if self.text_encoder_module is not None:
            if text_features is not None:
                word_tokens, state = self.text_encoder_module.forward_from_cache(text_features)
            elif texts is not None:
                word_tokens, state = self.text_encoder_module(texts, device=img.device)
            else:
                word_tokens, state = self.text_encoder_module(word, device=img.device)
        else:
            word_tokens, state = self.base_model.backbone.encode_text(word)
        if self.token_cross_attn:
            v3 = self.v3_cross(v3, word_tokens)
            v4 = self.v4_cross(v4, word_tokens)
            v5 = self.v5_cross(v5, word_tokens)
        if self.text_gate:
            v3 = self._apply_gate(v3, state, self.v3_gate)
            v4 = self._apply_gate(v4, state, self.v4_gate)
            v5 = self._apply_gate(v5, state, self.v5_gate)
        fq_sup, fq_inf = self.base_model.neck_ad((v3, v4, v5), state)
        pred = self.base_model.proj(fq_sup, state)
        pred_ad = self.base_model.proj_ad(fq_inf, state)
        if self.albef_fuse:
            fused, aux_loss = self.albef_head(img, pred, pred_ad, v3, v4, v5, fq_sup, fq_inf, word_tokens, state)
            if aux_loss is not None:
                return {"logits": fused, "aux_loss": aux_loss}
            return fused, fused
        if self.mome_fuse:
            fused = self.mome_head(img, pred, pred_ad, v3, v4, v5, fq_sup, fq_inf, word_tokens, state)
            return fused, fused
        if self.deep_fuse:
            fused = self.deep_fuse_head(img, pred, pred_ad, v3, v4, v5, fq_sup, fq_inf, word_tokens, state)
            return fused, fused
        if self.refine_decoder:
            refined = self.refine_head(img, pred, pred_ad, v3, state)
            return refined, refined
        return pred, pred_ad


class RN50TextDeepFuseHead(nn.Module):
    def __init__(
        self,
        text_dim=1024,
        token_dim=512,
        hidden=192,
        mid=96,
        residual_scale=0.35,
    ):
        super().__init__()
        self.residual_scale = residual_scale
        self.v3_proj = nn.Sequential(
            nn.Conv2d(512, 96, kernel_size=1, bias=False),
            nn.GroupNorm(8, 96),
            nn.GELU(),
        )
        self.v4_proj = nn.Sequential(
            nn.Conv2d(1024, 96, kernel_size=1, bias=False),
            nn.GroupNorm(8, 96),
            nn.GELU(),
        )
        self.v5_proj = nn.Sequential(
            nn.Conv2d(512, 96, kernel_size=1, bias=False),
            nn.GroupNorm(8, 96),
            nn.GELU(),
        )
        self.sup_proj = nn.Sequential(
            nn.Conv2d(512, 96, kernel_size=1, bias=False),
            nn.GroupNorm(8, 96),
            nn.GELU(),
        )
        self.inf_proj = nn.Sequential(
            nn.Conv2d(512, 96, kernel_size=1, bias=False),
            nn.GroupNorm(8, 96),
            nn.GELU(),
        )
        self.fuse14 = ConvGNAct(96 * 5, hidden)
        self.token_cross = RN50TokenCrossAttention2d(hidden, text_dim=token_dim, attn_dim=128, heads=4, scale=0.20)
        self.state_gate = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, hidden * 2),
        )
        self.up3 = nn.ConvTranspose2d(hidden, mid, kernel_size=2, stride=2)
        self.v3_skip = nn.Sequential(
            nn.Conv2d(512, mid // 2, kernel_size=1, bias=False),
            nn.GroupNorm(8, mid // 2),
            nn.GELU(),
        )
        self.dec3 = ConvGNAct(mid + mid // 2, mid)
        self.up2 = nn.ConvTranspose2d(mid, mid // 2, kernel_size=2, stride=2)
        self.dec2 = ConvGNAct(mid // 2, mid // 2)
        self.up1 = nn.ConvTranspose2d(mid // 2, mid // 2, kernel_size=2, stride=2)
        self.dec1 = ConvGNAct(mid // 2, mid // 2)
        self.high_fuse = ConvGNAct(mid // 2 + 3 + 2, 64)
        self.out = nn.Conv2d(64, 1, kernel_size=1)
        nn.init.zeros_(self.state_gate[-1].weight)
        nn.init.zeros_(self.state_gate[-1].bias)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    @staticmethod
    def _apply_state_gate(feature, state, gate, scale=0.25):
        gamma, beta = gate(state).chunk(2, dim=1)
        gamma = gamma.view(feature.shape[0], -1, 1, 1)
        beta = beta.view(feature.shape[0], -1, 1, 1)
        return feature * (1.0 + scale * torch.tanh(gamma)) + scale * torch.tanh(beta)

    def forward(self, image, pred, pred_ad, v3, v4, v5, fq_sup, fq_inf, word_tokens, state):
        base_logits = 0.5 * (pred + pred_ad)
        target14 = fq_sup.shape[-2:]
        v3_14 = F.adaptive_avg_pool2d(v3, target14)
        v4_14 = F.interpolate(v4, size=target14, mode="bilinear", align_corners=False)
        v5_14 = F.interpolate(v5, size=target14, mode="bilinear", align_corners=False)
        fused14 = torch.cat(
            [
                self.v3_proj(v3_14),
                self.v4_proj(v4_14),
                self.v5_proj(v5_14),
                self.sup_proj(fq_sup),
                self.inf_proj(fq_inf),
            ],
            dim=1,
        )
        x = self.fuse14(fused14)
        x = self.token_cross(x, word_tokens)
        x = self._apply_state_gate(x, state, self.state_gate)
        x = self.up3(x)
        v3_skip = self.v3_skip(v3)
        if v3_skip.shape[-2:] != x.shape[-2:]:
            v3_skip = F.interpolate(v3_skip, size=x.shape[-2:], mode="bilinear", align_corners=False)
        x = self.dec3(torch.cat([x, v3_skip], dim=1))
        x = self.dec2(self.up2(x))
        x = self.dec1(self.up1(x))
        x = F.interpolate(x, size=base_logits.shape[-2:], mode="bilinear", align_corners=False)
        image_resized = image
        if image_resized.shape[-2:] != base_logits.shape[-2:]:
            image_resized = F.interpolate(image_resized, size=base_logits.shape[-2:], mode="bilinear", align_corners=False)
        high = torch.cat([x, image_resized, pred, pred_ad], dim=1)
        delta = self.out(self.high_fuse(high))
        return base_logits + self.residual_scale * delta


class RN50AlignBeforeFuseHead(nn.Module):
    def __init__(self, text_dim=1024, token_dim=512, align_dim=256, align_loss_weight=0.03):
        super().__init__()
        self.align_loss_weight = align_loss_weight
        self.deep_fuse = RN50TextDeepFuseHead(text_dim=text_dim, token_dim=token_dim)
        self.image_proj = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, align_dim),
        )
        self.text_proj = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, align_dim),
        )
        self.image_to_state = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, text_dim),
        )
        self.state_gate = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, 1),
        )
        self.temperature = nn.Parameter(torch.tensor(0.07).log())
        nn.init.zeros_(self.image_to_state[-1].weight)
        nn.init.zeros_(self.image_to_state[-1].bias)
        nn.init.zeros_(self.state_gate[-1].weight)
        nn.init.zeros_(self.state_gate[-1].bias)

    @staticmethod
    def _image_vector(fq_sup, fq_inf):
        return 0.5 * (fq_sup.mean(dim=(2, 3)) + fq_inf.mean(dim=(2, 3)))

    def _alignment_loss(self, image_vec, state):
        image_z = F.normalize(self.image_proj(image_vec.float()), dim=-1)
        text_z = F.normalize(self.text_proj(state.float()), dim=-1)
        logits = image_z @ text_z.t()
        scale = torch.exp(self.temperature).clamp(0.01, 0.50)
        logits = logits / scale
        labels = torch.arange(logits.shape[0], device=logits.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))

    def forward(self, image, pred, pred_ad, v3, v4, v5, fq_sup, fq_inf, word_tokens, state):
        image_vec = self._image_vector(fq_sup, fq_inf)
        state_delta = torch.tanh(self.image_to_state(image_vec.float())).to(dtype=state.dtype)
        aligned_state = state + 0.25 * state_delta
        gate = torch.sigmoid(self.state_gate(aligned_state.float())).view(state.shape[0], 1, 1).to(dtype=word_tokens.dtype)
        aligned_tokens = word_tokens * (0.5 + gate)
        logits = self.deep_fuse(image, pred, pred_ad, v3, v4, v5, fq_sup, fq_inf, aligned_tokens, aligned_state)
        aux_loss = None
        if self.training and self.align_loss_weight > 0:
            aux_loss = self.align_loss_weight * self._alignment_loss(image_vec, state)
        return logits, aux_loss


class RN50ImageRefinementHead(nn.Module):
    def __init__(self, v3_channels=512, hidden=96, v3_out=48, residual_scale=0.35):
        super().__init__()
        self.residual_scale = residual_scale
        self.v3_proj = nn.Sequential(
            nn.Conv2d(v3_channels, v3_out, kernel_size=1, bias=False),
            nn.GroupNorm(8, v3_out),
            nn.GELU(),
        )
        self.conv1 = ConvGNAct(3 + 2 + v3_out, hidden)
        self.conv2 = ConvGNAct(hidden, hidden // 2)
        self.out = nn.Conv2d(hidden // 2, 1, kernel_size=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, image, pred, pred_ad, v3):
        base_logits = 0.5 * (pred + pred_ad)
        v3_up = F.interpolate(self.v3_proj(v3), size=base_logits.shape[-2:], mode="bilinear", align_corners=False)
        image_resized = image
        if image_resized.shape[-2:] != base_logits.shape[-2:]:
            image_resized = F.interpolate(image_resized, size=base_logits.shape[-2:], mode="bilinear", align_corners=False)
        delta = self.out(self.conv2(self.conv1(torch.cat([image_resized, pred, pred_ad, v3_up], dim=1))))
        return base_logits + self.residual_scale * delta


class RN50MoMEFuseHead(nn.Module):
    def __init__(self, text_dim=1024, token_dim=512):
        super().__init__()
        self.text_expert = RN50TextDeepFuseHead(text_dim=text_dim, token_dim=token_dim)
        self.image_expert = RN50ImageRefinementHead()
        self.gate = nn.Sequential(
            nn.LayerNorm(512 + text_dim),
            nn.Linear(512 + text_dim, 256),
            nn.GELU(),
            nn.Linear(256, 3),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    @staticmethod
    def _image_vector(fq_sup, fq_inf):
        return 0.5 * (fq_sup.mean(dim=(2, 3)) + fq_inf.mean(dim=(2, 3)))

    def forward(self, image, pred, pred_ad, v3, v4, v5, fq_sup, fq_inf, word_tokens, state):
        shared_logits = 0.5 * (pred + pred_ad)
        image_logits = self.image_expert(image, pred, pred_ad, v3)
        text_logits = self.text_expert(image, pred, pred_ad, v3, v4, v5, fq_sup, fq_inf, word_tokens, state)
        gate_input = torch.cat([self._image_vector(fq_sup, fq_inf).float(), state.float()], dim=1)
        weights = F.softmax(self.gate(gate_input), dim=1).to(dtype=shared_logits.dtype)
        weights = weights.view(weights.shape[0], 3, 1, 1, 1)
        experts = torch.stack([image_logits, text_logits, shared_logits], dim=1)
        return (weights * experts).sum(dim=1)


class RN50TextRefinementHead(nn.Module):
    def __init__(self, text_dim=1024, v3_channels=512, hidden=64, v3_out=32, residual_scale=0.35):
        super().__init__()
        self.residual_scale = residual_scale
        self.v3_proj = nn.Sequential(
            nn.Conv2d(v3_channels, v3_out, kernel_size=1, bias=False),
            nn.GroupNorm(8, v3_out),
            nn.GELU(),
        )
        self.conv1 = ConvGNAct(3 + 2 + v3_out, hidden)
        self.text_gate = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, hidden * 2),
        )
        self.conv2 = ConvGNAct(hidden, hidden // 2)
        self.out = nn.Conv2d(hidden // 2, 1, kernel_size=1)
        nn.init.zeros_(self.text_gate[-1].weight)
        nn.init.zeros_(self.text_gate[-1].bias)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, image, pred, pred_ad, v3, state):
        v3_low = self.v3_proj(v3)
        v3_up = F.interpolate(v3_low, size=pred.shape[-2:], mode="bilinear", align_corners=False)
        features = torch.cat([image, pred, pred_ad, v3_up], dim=1)
        features = self.conv1(features)
        gamma, beta = self.text_gate(state).chunk(2, dim=1)
        gamma = gamma.view(state.shape[0], -1, 1, 1)
        beta = beta.view(state.shape[0], -1, 1, 1)
        features = features * (1.0 + 0.25 * torch.tanh(gamma)) + 0.25 * torch.tanh(beta)
        delta = self.out(self.conv2(features))
        return 0.5 * (pred + pred_ad) + self.residual_scale * delta


def build_model(model_name, device, causal_recipe_name="default", conv_plugin="standard", atconv_layers=0):
    cwd = None
    if model_name == "TextCrossAttentionUNet":
        model = TextCrossAttentionUNet().to(device)
        return model, {"mode": "text_cross_unet", "image_size": 224, "uses_logits": True}

    if model_name in {"LViT", "LViTTextFiLM", "LViTTextAdapter", "LViTTextImageAdapter"}:
        path = repo_path("LViT")
        cleanup_modules(["nets", "utils"])
        prepend_sys_path(path)
        from nets.LViT import LViT

        model = LViT(tiny_config(), n_channels=3, n_classes=1, img_size=224, vis=False).to(device)
        if model_name == "LViTTextFiLM":
            model = TextFiLMLViT(model).to(device)
        if model_name == "LViTTextAdapter":
            model = TextAdapterLViT(model).to(device)
        if model_name == "LViTTextImageAdapter":
            model = TextImageAdapterLViT(model).to(device)
        return model, {"mode": "lvit", "image_size": 224, "uses_logits": False}

    if model_name == "STPNet":
        path = repo_path("STPNet") / "STPNet-main"
        cleanup_modules(["nets", "utils"])
        install_stpnet_imageaggr_shim()
        prepend_sys_path(path)
        stp_module = importlib.import_module("nets.STPNet")
        patch_torchvision_resnet101(stp_module)
        model = stp_module.STPNet(tiny_config(), n_channels=3, n_classes=1, img_size=224, vis=False).to(device)
        bank = text_bank("STPNet", device=device)
        model.Unilateral_emb = bank
        model.num_emb = bank.roll(1, dims=0)
        model.left_loc_emb = bank.roll(2, dims=0)
        model.right_loc_emb = bank.roll(3, dims=0)
        return model, {"mode": "stpnet", "image_size": 224, "uses_logits": False}

    if model_name == "TAMISegStage1":
        path = repo_path("TAMISeg") / "Stage1_CAE" / "network"
        cleanup_modules(["model_stage1", "resnet"])
        prepend_sys_path(path)
        stage1_module = importlib.import_module("model_stage1")
        patch_stage1_resnet_no_download(stage1_module)
        model = stage1_module.ConDSegStage1(H=256, W=256).to(device)
        return model, {"mode": "plain", "image_size": 256, "uses_logits": False}

    if model_name == "TAMISegMRA":
        if not torch.cuda.is_available():
            raise RuntimeError("TAMISegMRA hard-codes .cuda() for text banks and requires CUDA.")
        path = repo_path("TAMISeg") / "MRA_Block"
        cleanup_modules(["nets", "utils"])
        old = Path.cwd()
        os.chdir(path)
        cwd = old
        prepend_sys_path(path)
        try:
            module = importlib.import_module("nets.STPNet")
            patch_torchvision_resnet101(module)
            model = module.STPNet(tiny_config(), n_channels=3, n_classes=1, img_size=224, vis=False).to(device)
        finally:
            if cwd is not None:
                os.chdir(cwd)
        return model, {"mode": "tamiseg_mra", "image_size": 224, "uses_logits": False}

    if model_name == "CausalCLIPSeg":
        path = repo_path("CausalCLIPSeg")
        cleanup_modules(["nets"])
        prepend_sys_path(path)
        ckpt_dir = ARTIFACT_DIR
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / "causalclipseg_random_clip_jit_v2.pt"
        make_causal_clip_checkpoint(ckpt_path)
        from nets.segmenter import CausalCLIPSeg

        cfg = SimpleNamespace(
            clip_pretrain=str(ckpt_path),
            word_len=16,
            fpn_in=[512, 1024, 512],
            fpn_out=[256, 512, 512],
            word_dim=512,
            vis_dim=512,
        )
        model = CausalCLIPSeg(cfg).to(device)
        return model, {"mode": "causal_random", "image_size": 224, "uses_logits": True}

    if model_name in {
        "CausalCLIPSegRN50",
        "CausalCLIPSegRN50Frozen",
        "CausalCLIPSegRN50TextGate",
        "CausalCLIPSegRN50WD1e3",
        "CausalCLIPSegRN50DiceBCE",
        "CausalCLIPSegRN50BoundaryDice",
        "CausalCLIPSegRN50Refine",
        "CausalCLIPSegRN50CrossAttn",
        "CausalCLIPSegRN50DeepFuse",
        "CausalCLIPSegRN50DeepFuseDiceBCE",
        "CausalCLIPSegRN50ALBEFSeg",
        "CausalCLIPSegRN50MoMESeg",
        "CausalCLIPSegRN50LCFusion",
        "CausalCLIPSegRN50Distill",
        "CausalCLIPSegRN101",
        "CausalCLIPSegRN101DeepFuseDiceBCE",
    }:
        path = repo_path("CausalCLIPSeg")
        cleanup_modules(["nets"])
        prepend_sys_path(path)
        if model_name in {"CausalCLIPSegRN101", "CausalCLIPSegRN101DeepFuseDiceBCE"}:
            ckpt_path = ensure_artifact(ARTIFACT_DIR / "openai_clip_RN101.pt", OPENAI_CLIP_RN101_URL)
        else:
            ckpt_path = ensure_artifact(ARTIFACT_DIR / "openai_clip_RN50.pt", OPENAI_CLIP_RN50_URL)
        bpe_path = ensure_artifact(ARTIFACT_DIR / "bpe_simple_vocab_16e6.txt.gz", OPENAI_CLIP_BPE_URL)
        from nets.segmenter import CausalCLIPSeg

        clip_state = torch.jit.load(str(ckpt_path), map_location="cpu").state_dict()
        state_dim = int(clip_state["text_projection"].shape[1])
        token_dim = int(clip_state["ln_final.weight"].shape[0])
        v5_dim = state_dim
        cfg = SimpleNamespace(
            clip_pretrain=str(ckpt_path),
            word_len=16,
            fpn_in=[512, 1024, 512],
            fpn_out=[256, 512, 512],
            word_dim=state_dim,
            vis_dim=512,
        )
        recipe = get_recipe(causal_recipe_name)
        text_encoder_module = None
        if recipe.text_encoder != "clip_rn50":
            text_encoder_module = build_text_encoder_plugin(
                recipe.text_encoder,
                None,
                token_dim=token_dim,
                state_dim=state_dim,
                word_len=16,
            )
        model = CausalCLIPSegRN50Adapter(
            CausalCLIPSeg(cfg),
            freeze_backbone=(model_name == "CausalCLIPSegRN50Frozen"),
            text_gate=(model_name == "CausalCLIPSegRN50TextGate"),
            refine_decoder=(model_name == "CausalCLIPSegRN50Refine"),
            token_cross_attn=model_name in {"CausalCLIPSegRN50CrossAttn", "CausalCLIPSegRN50LCFusion"},
            deep_fuse=model_name
            in {
                "CausalCLIPSegRN50DeepFuse",
                "CausalCLIPSegRN50DeepFuseDiceBCE",
                "CausalCLIPSegRN50LCFusion",
                "CausalCLIPSegRN101DeepFuseDiceBCE",
            },
            albef_fuse=model_name == "CausalCLIPSegRN50ALBEFSeg",
            mome_fuse=model_name == "CausalCLIPSegRN50MoMESeg",
            state_dim=state_dim,
            token_dim=token_dim,
            v5_dim=v5_dim,
            text_encoder_module=text_encoder_module,
        )
        atconv_targets: list[str] = []
        if conv_plugin == "atconv" and atconv_layers > 0:
            atconv_targets = apply_atconv_plugin(model, num_layers=atconv_layers)
        model = model.to(device)
        meta = {
            "mode": "causal_clip_bpe",
            "image_size": 224,
            "uses_logits": True,
            "word_len": 16,
            "bpe_path": str(bpe_path),
            "freeze_backbone": model_name == "CausalCLIPSegRN50Frozen",
            "text_gate": model_name == "CausalCLIPSegRN50TextGate",
            "refine_decoder": model_name == "CausalCLIPSegRN50Refine",
            "token_cross_attn": model_name in {"CausalCLIPSegRN50CrossAttn", "CausalCLIPSegRN50LCFusion"},
            "deep_fuse": model_name
            in {
                "CausalCLIPSegRN50DeepFuse",
                "CausalCLIPSegRN50DeepFuseDiceBCE",
                "CausalCLIPSegRN50LCFusion",
                "CausalCLIPSegRN101DeepFuseDiceBCE",
            },
            "albef_fuse": model_name == "CausalCLIPSegRN50ALBEFSeg",
            "mome_fuse": model_name == "CausalCLIPSegRN50MoMESeg",
            "lc_fusion": model_name == "CausalCLIPSegRN50LCFusion",
            "clip_backbone": "RN101" if model_name in {"CausalCLIPSegRN101", "CausalCLIPSegRN101DeepFuseDiceBCE"} else "RN50",
            "state_dim": state_dim,
            "token_dim": token_dim,
            "v5_dim": v5_dim,
            "regularization_recipe": "weight_decay_1e-3" if model_name == "CausalCLIPSegRN50WD1e3" else "default",
            "loss_recipe": (
                "bce_dice_boundary_0.5_0.05"
                if model_name == "CausalCLIPSegRN50BoundaryDice"
                else "bce_dice_0.5"
                if model_name
                in {
                    "CausalCLIPSegRN50DiceBCE",
                    "CausalCLIPSegRN50DeepFuseDiceBCE",
                    "CausalCLIPSegRN50ALBEFSeg",
                    "CausalCLIPSegRN50MoMESeg",
                    "CausalCLIPSegRN50LCFusion",
                    "CausalCLIPSegRN101DeepFuseDiceBCE",
                }
                else "bce"
            ),
        }
        meta.update(recipe_meta(recipe, atconv_targets="|".join(atconv_targets)))
        if atconv_targets:
            meta.update(atconv_meta(atconv_targets))
        return model, meta

    raise ValueError(model_name)


def forward_model(model, meta, images, texts, device, text_cache=None):
    mode = meta["mode"]
    if mode in {"lvit", "text_cross_unet"}:
        if text_cache is not None:
            txt = text_cache.batch(texts, tokens=10, dim=768, device=device)
        else:
            txt = stable_text_embedding(texts, tokens=10, dim=768, device=device)
        return model(images, txt)
    if mode == "stpnet":
        return model(images, model.Unilateral_emb, model.num_emb, model.left_loc_emb, model.right_loc_emb)[0]
    if mode == "tamiseg_mra":
        return model(images)[0]
    if mode == "causal_random":
        tokens = stable_text_tokens(texts, length=16, vocab_size=1000, device=device)
        pred, pred_ad = model(images, tokens)
        return 0.5 * (pred + pred_ad)
    if mode == "causal_clip_bpe":
        text_encoder = meta.get("text_encoder", "clip_rn50")
        if text_encoder != "clip_rn50":
            text_features = None
            if text_cache is not None and hasattr(text_cache, "batch_features"):
                text_features = text_cache.batch_features(texts, device=device)
            out = model(images, texts=texts, text_features=text_features)
        else:
            tokens = clip_bpe_tokens(texts, length=meta.get("word_len", 16), device=device, bpe_path=meta["bpe_path"])
            out = model(images, word=tokens)
        if isinstance(out, dict):
            return out
        pred, pred_ad = out
        return 0.5 * (pred + pred_ad)
    return model(images)


def soft_dice_loss(prob, target):
    dims = tuple(range(1, prob.ndim))
    inter = (prob * target).sum(dim=dims)
    prob_sum = prob.sum(dim=dims)
    target_sum = target.sum(dim=dims)
    dice = (2 * inter + 1e-6) / (prob_sum + target_sum + 1e-6)
    return 1.0 - dice.mean()


def soft_boundary_loss(prob, target):
    prob_dx = torch.abs(prob[..., :, 1:] - prob[..., :, :-1])
    target_dx = torch.abs(target[..., :, 1:] - target[..., :, :-1])
    prob_dy = torch.abs(prob[..., 1:, :] - prob[..., :-1, :])
    target_dy = torch.abs(target[..., 1:, :] - target[..., :-1, :])
    return 0.5 * (F.l1_loss(prob_dx, target_dx) + F.l1_loss(prob_dy, target_dy))


def loss_and_prob(pred, mask, uses_logits, loss_mode="bce", dice_loss_weight=0.5, boundary_loss_weight=0.05):
    aux_loss = None
    if isinstance(pred, dict):
        aux_loss = pred.get("aux_loss")
        pred = pred["logits"]
    if pred.shape[-2:] != mask.shape[-2:]:
        mask = F.interpolate(mask.float(), size=pred.shape[-2:], mode="nearest")
    target = mask.float()
    if uses_logits:
        bce_loss = F.binary_cross_entropy_with_logits(pred.float(), target)
        prob = torch.sigmoid(pred.float())
    else:
        prob = pred.float().clamp(1e-5, 1 - 1e-5)
        bce_loss = F.binary_cross_entropy(prob, target)
    if loss_mode == "bce":
        loss = bce_loss
    elif loss_mode == "bce_dice":
        dice_loss = soft_dice_loss(prob, target)
        loss = (1.0 - dice_loss_weight) * bce_loss + dice_loss_weight * dice_loss
    elif loss_mode == "bce_dice_boundary":
        dice_loss = soft_dice_loss(prob, target)
        boundary_loss = soft_boundary_loss(prob, target)
        bce_weight = max(0.0, 1.0 - dice_loss_weight - boundary_loss_weight)
        loss = bce_weight * bce_loss + dice_loss_weight * dice_loss + boundary_loss_weight * boundary_loss
    else:
        raise ValueError(f"Unknown loss_mode: {loss_mode}")
    if aux_loss is not None:
        loss = loss + aux_loss.to(device=loss.device, dtype=loss.dtype)
    return loss, prob, mask


def batch_metrics(prob, mask):
    pred = (prob >= 0.5).float()
    mask = (mask >= 0.5).float()
    dims = tuple(range(1, pred.ndim))
    inter = (pred * mask).sum(dim=dims)
    pred_sum = pred.sum(dim=dims)
    mask_sum = mask.sum(dim=dims)
    dice = ((2 * inter + 1e-6) / (pred_sum + mask_sum + 1e-6)).mean().item()
    iou = ((inter + 1e-6) / (pred_sum + mask_sum - inter + 1e-6)).mean().item()
    return dice, iou


def evaluate(model, meta, loader, device, text_cache=None, loss_mode="bce", dice_loss_weight=0.5, boundary_loss_weight=0.05):
    model.eval()
    losses, dices, ious = [], [], []
    with torch.no_grad():
        for images, masks, texts, _, _ in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            pred = forward_model(model, meta, images, texts, device, text_cache=text_cache)
            loss, prob, target = loss_and_prob(
                pred,
                masks,
                meta["uses_logits"],
                loss_mode=loss_mode,
                dice_loss_weight=dice_loss_weight,
                boundary_loss_weight=boundary_loss_weight,
            )
            dice, iou = batch_metrics(prob, target)
            losses.append(loss.item())
            dices.append(dice)
            ious.append(iou)
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "dice": float(np.mean(dices)) if dices else float("nan"),
        "iou": float(np.mean(ious)) if ious else float("nan"),
    }


def train(args):
    seed_all(args.seed)
    split_seed = args.seed if args.split_seed is None else args.split_seed
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")
    recipe = apply_recipe_to_args(args)
    conv_plugin, atconv_layers = resolve_conv_plugin(recipe, args)
    args.conv_plugin = conv_plugin
    args.atconv_layers = atconv_layers
    model, meta = build_model(
        args.model,
        device,
        causal_recipe_name=recipe.name,
        conv_plugin=conv_plugin,
        atconv_layers=atconv_layers,
    )
    if args.text_encoder_cache:
        meta["text_encoder_cache"] = args.text_encoder_cache
    text_cache = load_text_embedding_cache(args.text_encoder_cache) if args.text_encoder_cache else None
    if args.protocol_lock:
        if not args.manifest:
            raise ValueError("--protocol-lock requires an explicit Protocol V3 --manifest")
        splits = load_manifest_splits(args.manifest, args.datasets, check_files=True)
        train_rows = splits["train"]
        val_rows = splits["val"]
        if args.max_train_samples:
            train_rows = train_rows[: args.max_train_samples]
        if args.max_val_samples:
            val_rows = val_rows[: args.max_val_samples]
        args.protocol_hash = protocol_sha256(args.protocol_lock)
    else:
        train_rows, val_rows = load_rows(
            args.seed,
            args.max_train_samples,
            args.max_val_samples,
            args.datasets,
            split_seed=split_seed,
            manifest_path=args.manifest,
        )
        args.protocol_hash = "legacy"
    if text_cache is not None:
        required_texts = {row["text"] for row in train_rows + val_rows}
        if args.augment and "lcaug" in args.aug_strength:
            expanded = set()
            for text in required_texts:
                expanded.update(lcaug_text_variants(text, use_v2=args.aug_strength.startswith("lcaug_v2")))
            required_texts = expanded
        text_cache.validate_texts(sorted(required_texts))
    train_ds = ManifestDataset(
        train_rows,
        meta["image_size"],
        augment=args.augment,
        aug_strength=args.aug_strength,
        resize_mode=args.resize_mode,
        seed=args.seed,
    )
    val_ds = ManifestDataset(val_rows, meta["image_size"], resize_mode=args.resize_mode, seed=args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=torch.cuda.is_available())
    if args.model == "CausalCLIPSegRN50WD1e3" and args.weight_decay == 1e-4:
        args.weight_decay = 1e-3
    if args.model in {
        "CausalCLIPSegRN50DiceBCE",
        "CausalCLIPSegRN50DeepFuseDiceBCE",
        "CausalCLIPSegRN50ALBEFSeg",
        "CausalCLIPSegRN50MoMESeg",
        "CausalCLIPSegRN50LCFusion",
        "CausalCLIPSegRN101DeepFuseDiceBCE",
    } and args.loss_mode == "bce":
        args.loss_mode = "bce_dice"
    if args.model == "CausalCLIPSegRN50BoundaryDice" and args.loss_mode == "bce":
        args.loss_mode = "bce_dice_boundary"
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    else:
        scheduler = None

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    aug_tag = args.aug_strength if args.augment else "noaug"
    sched_tag = args.scheduler
    run_stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_name = f"{args.model}_{aug_tag}_{sched_tag}_seed{args.seed}_split{split_seed}_{run_stamp}"
    ckpt_path = OUT_DIR / f"{run_name}_best.pt"

    print(f"# {args.model} training")
    print(f"date: {_dt.datetime.now().isoformat(timespec='seconds')}")
    print(f"hostname: {socket.gethostname()}")
    print(f"python: {sys.executable}")
    print(f"torch: {torch.__version__}")
    print(f"device: {device}")
    print(f"seed: {args.seed}")
    print(f"split_seed: {split_seed}")
    print(f"manifest: {args.manifest or MANIFEST}")
    print(f"protocol_lock: {args.protocol_lock or 'legacy'}")
    print(f"protocol_hash: {args.protocol_hash}")
    print(f"train_samples: {len(train_ds)}")
    print(f"val_samples: {len(val_ds)}")
    print(f"image_size: {meta['image_size']}")
    print(f"resize_mode: {args.resize_mode}")
    print(f"epochs: {args.epochs}")
    print(f"batch_size: {args.batch_size}")
    print(f"lr: {args.lr}")
    print(f"weight_decay: {args.weight_decay}")
    print(f"loss_mode: {args.loss_mode}")
    print(f"dice_loss_weight: {args.dice_loss_weight}")
    print(f"boundary_loss_weight: {args.boundary_loss_weight}")
    print(f"scheduler: {args.scheduler}")
    print(f"min_lr: {args.min_lr}")
    print(f"augment: {args.augment}")
    print(f"aug_strength: {args.aug_strength}")
    print(f"causal_recipe: {recipe.name}")
    print(f"text_encoder: {meta.get('text_encoder', 'clip_rn50')}")
    print(f"conv_plugin: {meta.get('conv_plugin', 'standard')}")
    if meta.get("atconv_layers", 0):
        print(f"atconv_layers: {meta.get('atconv_layers', 0)}")
        print(f"atconv_targets: {meta.get('atconv_targets', '')}")
    print(f"checkpoint: {ckpt_path}")

    best_dice = -1.0
    best_metrics = None
    for epoch in range(1, args.epochs + 1):
        train_ds.set_epoch(epoch)
        start = time.time()
        model.train()
        losses, dices, ious = [], [], []
        for step, (images, masks, texts, _, _) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = forward_model(model, meta, images, texts, device, text_cache=text_cache)
            loss, prob, target = loss_and_prob(
                pred,
                masks,
                meta["uses_logits"],
                loss_mode=args.loss_mode,
                dice_loss_weight=args.dice_loss_weight,
                boundary_loss_weight=args.boundary_loss_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            dice, iou = batch_metrics(prob.detach(), target.detach())
            losses.append(loss.item())
            dices.append(dice)
            ious.append(iou)
            if args.max_steps_per_epoch and step >= args.max_steps_per_epoch:
                break
        train_metrics = {"loss": float(np.mean(losses)), "dice": float(np.mean(dices)), "iou": float(np.mean(ious))}
        val_metrics = evaluate(
            model,
            meta,
            val_loader,
            device,
            text_cache=text_cache,
            loss_mode=args.loss_mode,
            dice_loss_weight=args.dice_loss_weight,
            boundary_loss_weight=args.boundary_loss_weight,
        )
        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step()
        elapsed = time.time() - start
        print(
            f"epoch={epoch} elapsed={elapsed:.1f}s "
            f"lr={current_lr:.8f} "
            f"train_loss={train_metrics['loss']:.5f} train_dice={train_metrics['dice']:.5f} train_iou={train_metrics['iou']:.5f} "
            f"val_loss={val_metrics['loss']:.5f} val_dice={val_metrics['dice']:.5f} val_iou={val_metrics['iou']:.5f}",
            flush=True,
        )
        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            best_metrics = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
            torch.save({"model": args.model, "state_dict": model.state_dict(), "metrics": best_metrics, "args": vars(args)}, ckpt_path)

    print(f"best_epoch: {best_metrics['epoch'] if best_metrics else 'none'}")
    print(f"best_val_dice: {best_metrics['val']['dice'] if best_metrics else float('nan'):.6f}")
    print(f"best_val_iou: {best_metrics['val']['iou'] if best_metrics else float('nan'):.6f}")
    print(f"best_checkpoint: {ckpt_path}")
    print("final_status: PASS")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["LViT", "LViTTextFiLM", "LViTTextAdapter", "LViTTextImageAdapter", "TextCrossAttentionUNet", "STPNet", "TAMISegStage1", "TAMISegMRA", "CausalCLIPSeg", "CausalCLIPSegRN50", "CausalCLIPSegRN50Frozen", "CausalCLIPSegRN50TextGate", "CausalCLIPSegRN50WD1e3", "CausalCLIPSegRN50DiceBCE", "CausalCLIPSegRN50BoundaryDice", "CausalCLIPSegRN50Refine", "CausalCLIPSegRN50CrossAttn", "CausalCLIPSegRN50DeepFuse", "CausalCLIPSegRN50DeepFuseDiceBCE", "CausalCLIPSegRN50ALBEFSeg", "CausalCLIPSegRN50MoMESeg", "CausalCLIPSegRN50LCFusion", "CausalCLIPSegRN50Distill", "CausalCLIPSegRN101", "CausalCLIPSegRN101DeepFuseDiceBCE"])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--loss-mode", choices=["bce", "bce_dice", "bce_dice_boundary"], default="bce")
    parser.add_argument("--dice-loss-weight", type=float, default=0.5)
    parser.add_argument("--boundary-loss-weight", type=float, default=0.05)
    parser.add_argument("--scheduler", choices=["none", "cosine"], default="none")
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--max-train-samples", type=int, default=64)
    parser.add_argument("--max-val-samples", type=int, default=32)
    parser.add_argument("--max-steps-per-epoch", type=int, default=0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--split-seed", type=int, default=None, help="Data split seed. Defaults to --seed for legacy behavior; set a fixed value for paper-grade multi-seed runs.")
    parser.add_argument("--manifest", default="", help="Optional train/validation manifest CSV. Defaults to the public dataset manifest.")
    parser.add_argument(
        "--protocol-lock",
        default="",
        help="Protocol V3 YAML lock enabling strict split validation and protocol hashing.",
    )
    parser.add_argument("--datasets", nargs="*", default=None, help="Optional dataset names, e.g. cvc busi glas")
    parser.add_argument(
        "--resize-mode",
        choices=["stretch", "letterbox"],
        default="stretch",
        help="Resize policy before square input: stretch (legacy) or letterbox (aspect-preserving pad).",
    )
    parser.add_argument("--augment", action="store_true", help="Enable synchronized train-time image/mask augmentation.")
    parser.add_argument(
        "--aug-strength",
        choices=[
            "light",
            "medium",
            "strong",
            "safe_light",
            "safe_light_dataset",
            "appearance",
            "appearance_dataset",
            "elastic_light",
            "elastic_light_dataset",
            "dataset_policy_v1",
            "dataset_policy_v2",
            "glas_appearance_dataset",
            "lcaug_hflip_dataset",
            "lcaug_hflip_no_text_rewrite_dataset",
            "lcaug_v2_hflip_dataset",
            "lcaug_v2_hflip_no_text_rewrite_dataset",
            "lcaug_v2_busi_dataset",
            "lcaug_v2_busi_no_text_rewrite_dataset",
            "lcaug_v2_dynamic_shared_plan_dataset",
            "lcaug_v2_dynamic_shared_plan_recompute_location_dataset",
            "lcaug_v2_dynamic_shared_plan_no_text_rewrite_dataset",
            "lcaug_v2_medclipseg_dataset",
            "lcaug_v2_medclipseg_no_text_rewrite_dataset",
            "text_geo_dataset",
        ],
        default="medium",
    )
    parser.add_argument("--text-encoder-cache", default="", help="Optional .npz cache from text_encoders.py for LViT/BiomedCLIP text embeddings.")
    parser.add_argument(
        "--causal-recipe",
        choices=recipe_names(),
        default="default",
        help="Composable CausalCLIPSeg recipe for text/loss/conv plugins.",
    )
    parser.add_argument(
        "--enable-boundary-loss",
        action="store_true",
        help="Enable boundary-aware DiceBCE (overrides loss when recipe loss_mode is unset).",
    )
    parser.add_argument(
        "--enable-atconv",
        action="store_true",
        help="Replace decoder same-channel 3x3 convs with ATConv (orthogonal to recipe).",
    )
    parser.add_argument(
        "--atconv-layers",
        type=int,
        default=4,
        help="How many decoder 3x3 conv layers to replace when ATConv is enabled.",
    )
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("final_status: FAIL")
        traceback.print_exc()
        raise
