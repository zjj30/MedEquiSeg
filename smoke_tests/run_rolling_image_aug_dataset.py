#!/usr/bin/env python3
import argparse
import csv
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter
from scipy.ndimage import binary_erosion, distance_transform_edt
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_MANIFEST = ROOT / "smoke_tests" / "dataset_manifest.csv"
PRIVATE_MANIFEST = ROOT / "smoke_tests" / "private_manifest.csv"
ROLLING_REPO = ROOT / "repos" / "Rolling-Unet"
sys.path.insert(0, str(ROOT / "smoke_tests"))
from image_resize import resize_image_mask_pair


PROFILES = {
    "common_light": {
        "hflip_p": 0.50,
        "vflip_p": 0.00,
        "affine_p": 0.80,
        "rotate_deg": 15.0,
        "shift_frac": 0.06,
        "zoom_p": 0.45,
        "zoom_min": 0.90,
        "zoom_max": 1.15,
        "color_p": 0.80,
        "color_jitter": 0.18,
        "gamma_p": 0.45,
        "gamma_range": (0.86, 1.18),
        "blur_p": 0.12,
        "blur_radius": 0.55,
        "noise_p": 0.25,
        "noise_std": 0.010,
    },
    "common_medium": {
        "hflip_p": 0.50,
        "vflip_p": 0.15,
        "affine_p": 0.85,
        "rotate_deg": 22.0,
        "shift_frac": 0.08,
        "zoom_p": 0.55,
        "zoom_min": 0.85,
        "zoom_max": 1.25,
        "color_p": 0.85,
        "color_jitter": 0.24,
        "gamma_p": 0.55,
        "gamma_range": (0.78, 1.28),
        "blur_p": 0.18,
        "blur_radius": 0.75,
        "noise_p": 0.30,
        "noise_std": 0.015,
    },
}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def read_manifest(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


PUBLIC_DATASETS = {"busi", "cvc", "glas"}
BRAIN_MRI_MANIFEST = ROOT / "smoke_tests" / "brain_mri_plane_aware_manifest.csv"
BRAIN_MRI_TUMOR_MANIFEST = ROOT / "smoke_tests" / "brain_mri_tumor_only_manifest.csv"
MEDCLIPSEG_MANIFEST_DIR = ROOT / "smoke_tests" / "medclipseg_manifests"
MEDCLIPSEG_DATASETS = {
    "medclipseg_busi",
    "medclipseg_clinicdb",
    "medclipseg_busbra",
    "medclipseg_brisc",
    "medclipseg_covid19",
}


def normalize_dataset_name(value):
    raw = str(value).strip()
    low = raw.lower().replace("-", "_").replace(" ", "_")
    if low in {"busi_hf", "busihf", "private_bh_external", "private_bh"}:
        return "busi_hf"
    if low in PUBLIC_DATASETS:
        return low
    return raw


def read_val_filenames(path, dataset):
    if not path:
        return None
    rows = read_manifest(path)
    target = normalize_dataset_name(dataset)
    filenames = set()
    for row in rows:
        row_dataset = normalize_dataset_name(row.get("dataset", target))
        if row_dataset != target:
            continue
        filename = row.get("filename") or Path(row.get("image_path", "")).name
        if filename:
            filenames.add(filename)
    if not filenames:
        raise ValueError(f"No validation filenames found for dataset={dataset} in {path}")
    return filenames


def explicit_split(rows, dataset, val_filenames):
    by_name = {Path(row["image_path"]).name: row for row in rows}
    missing = sorted(name for name in val_filenames if name not in by_name)
    if missing:
        preview = ",".join(missing[:10])
        raise ValueError(f"Missing {len(missing)} validation filenames for dataset={dataset}: {preview}")
    val_rows = [by_name[name] for name in sorted(val_filenames)]
    train_rows = [row for row in rows if Path(row["image_path"]).name not in val_filenames]
    return train_rows, val_rows


def _rows_from_manifest(manifest_path, dataset=None, max_train=0, max_val=0):
    rows = [r for r in read_manifest(manifest_path) if not dataset or r.get("dataset") == dataset]
    rows = sorted(rows, key=lambda r: r["image_path"])
    split_values = {r.get("split", "").strip().lower() for r in rows}
    has_explicit_split = bool(split_values & {"train", "val", "valid", "validation", "test"})
    if not has_explicit_split:
        raise ValueError(f"Manifest {manifest_path} must include explicit train/val/test split column")
    train_rows = [r for r in rows if r.get("split", "").strip().lower() == "train"]
    val_rows = [
        r
        for r in rows
        if r.get("split", "").strip().lower() in {"val", "valid", "validation"}
    ]
    if max_train:
        train_rows = train_rows[:max_train]
    if max_val:
        val_rows = val_rows[:max_val]
    return train_rows, val_rows


def load_rows(dataset, split_seed, max_train=0, max_val=0, split_mode="dataset", val_filenames=None, manifest_path=None):
    if dataset in MEDCLIPSEG_DATASETS:
        manifest = Path(manifest_path) if manifest_path else MEDCLIPSEG_MANIFEST_DIR / f"{dataset}_full.csv"
        return _rows_from_manifest(manifest, dataset=dataset, max_train=max_train, max_val=max_val)
    if dataset in {"brain_mri", "brain_mri_tumor"}:
        default_manifest = BRAIN_MRI_MANIFEST if dataset == "brain_mri" else BRAIN_MRI_TUMOR_MANIFEST
        manifest = Path(manifest_path) if manifest_path else default_manifest
        dataset_name = "brain_mri" if dataset == "brain_mri" else "brain_mri_tumor"
        return _rows_from_manifest(manifest, dataset=dataset_name, max_train=max_train, max_val=max_val)
    if dataset in PUBLIC_DATASETS:
        public_rows = read_manifest(PUBLIC_MANIFEST)
        if split_mode == "global_public":
            if dataset != "glas":
                raise ValueError("split_mode=global_public is only defined for the GLaS global-heldout reference")
            rows = sorted(public_rows, key=lambda r: (r["dataset"], r["image_path"]))
            rng = random.Random(split_seed)
            rng.shuffle(rows)
            split = max(1, int(round(0.8 * len(rows))))
            train_rows = [r for r in rows[:split] if r["dataset"] == dataset]
            val_rows = [r for r in rows[split:] if r["dataset"] == dataset]
            if max_train:
                train_rows = train_rows[:max_train]
            if max_val:
                val_rows = val_rows[:max_val]
            return train_rows, val_rows
        rows = [r for r in public_rows if r["dataset"] == dataset]
    elif dataset == "busi_hf":
        if manifest_path:
            return _rows_from_manifest(Path(manifest_path), dataset="busi_hf", max_train=max_train, max_val=max_val)
        if split_mode != "dataset":
            raise ValueError(f"split_mode={split_mode} is only supported for public GlaS rows")
        rows = read_manifest(PRIVATE_MANIFEST)
        for row in rows:
            row["dataset"] = "busi_hf"
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    rows = sorted(rows, key=lambda r: r["image_path"])
    if val_filenames is not None:
        train_rows, val_rows = explicit_split(rows, dataset, val_filenames)
        if max_train:
            train_rows = train_rows[:max_train]
        if max_val:
            val_rows = val_rows[:max_val]
        return train_rows, val_rows

    rng = random.Random(split_seed)
    rng.shuffle(rows)
    split = max(1, int(round(0.8 * len(rows))))
    train_rows = rows[:split]
    val_rows = rows[split:]
    if max_train:
        train_rows = train_rows[:max_train]
    if max_val:
        val_rows = val_rows[:max_val]
    return train_rows, val_rows


def apply_gamma(img, gamma_range):
    gamma = random.uniform(float(gamma_range[0]), float(gamma_range[1]))
    arr = np.asarray(img).astype("float32") / 255.0
    arr = np.power(np.clip(arr, 0.0, 1.0), gamma)
    return Image.fromarray((arr * 255.0).clip(0, 255).astype("uint8"))


def zoom_pair(img, mask, profile, image_size):
    scale = random.uniform(profile["zoom_min"], profile["zoom_max"])
    if abs(scale - 1.0) < 1e-4:
        return img, mask
    if scale > 1.0:
        crop_size = max(16, int(round(image_size / scale)))
        left = random.randint(0, image_size - crop_size)
        top = random.randint(0, image_size - crop_size)
        box = (left, top, left + crop_size, top + crop_size)
        img = img.crop(box).resize((image_size, image_size), Image.Resampling.BILINEAR)
        mask = mask.crop(box).resize((image_size, image_size), Image.Resampling.NEAREST)
        return img, mask

    new_size = max(16, int(round(image_size * scale)))
    pad_left = random.randint(0, image_size - new_size)
    pad_top = random.randint(0, image_size - new_size)
    img_small = img.resize((new_size, new_size), Image.Resampling.BILINEAR)
    mask_small = mask.resize((new_size, new_size), Image.Resampling.NEAREST)
    img_canvas = Image.new("RGB", (image_size, image_size), (0, 0, 0))
    mask_canvas = Image.new("L", (image_size, image_size), 0)
    img_canvas.paste(img_small, (pad_left, pad_top))
    mask_canvas.paste(mask_small, (pad_left, pad_top))
    return img_canvas, mask_canvas


def augment_pair(img, mask, profile, image_size):
    if random.random() < profile["hflip_p"]:
        img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if random.random() < profile["vflip_p"]:
        img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        mask = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    if random.random() < profile["affine_p"]:
        max_shift = int(round(image_size * profile["shift_frac"]))
        translate = (
            random.randint(-max_shift, max_shift),
            random.randint(-max_shift, max_shift),
        )
        angle = random.uniform(-profile["rotate_deg"], profile["rotate_deg"])
        img = img.rotate(
            angle,
            resample=Image.Resampling.BILINEAR,
            translate=translate,
            fillcolor=(0, 0, 0),
        )
        mask = mask.rotate(
            angle,
            resample=Image.Resampling.NEAREST,
            translate=translate,
            fillcolor=0,
        )
    if random.random() < profile["zoom_p"]:
        img, mask = zoom_pair(img, mask, profile, image_size)

    if random.random() < profile["color_p"]:
        jitter = profile["color_jitter"]
        img = ImageEnhance.Brightness(img).enhance(random.uniform(1.0 - jitter, 1.0 + jitter))
        img = ImageEnhance.Contrast(img).enhance(random.uniform(1.0 - jitter, 1.0 + jitter))
    if random.random() < profile["gamma_p"]:
        img = apply_gamma(img, profile["gamma_range"])
    if random.random() < profile["blur_p"]:
        img = img.filter(ImageFilter.GaussianBlur(radius=profile["blur_radius"]))
    return img, mask


class SegDataset(Dataset):
    def __init__(self, rows, image_size, augment=False, profile_name="common_light", resize_mode="stretch"):
        self.rows = rows
        self.image_size = image_size
        self.augment = augment
        self.profile = PROFILES[profile_name]
        self.resize_mode = resize_mode

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        with Image.open(row["image_path"]) as img:
            img = img.convert("RGB")
        with Image.open(row["mask_path"]) as mask:
            mask = mask.convert("L")
        img, mask = resize_image_mask_pair(
            img, mask, self.image_size, self.resize_mode, image_resample=Image.Resampling.BILINEAR
        )

        if self.augment:
            img, mask = augment_pair(img, mask, self.profile, self.image_size)

        image = torch.from_numpy(np.asarray(img).transpose(2, 0, 1).copy()).float() / 255.0
        if self.augment and random.random() < self.profile["noise_p"]:
            image = (image + torch.randn_like(image) * self.profile["noise_std"]).clamp(0.0, 1.0)
        mask_arr = (np.asarray(mask).copy() > 0).astype("float32")
        mask_tensor = torch.from_numpy(mask_arr)[None, :, :]
        return image, mask_tensor, row["dataset"], Path(row["image_path"]).name


def import_rolling_unet():
    sys.path.insert(0, str(ROLLING_REPO))
    from archs import Rolling_Unet_S

    return Rolling_Unet_S


def build_model(image_size):
    RollingUnet = import_rolling_unet()
    return RollingUnet(num_classes=1, input_channels=3, img_size=image_size)


def load_init_checkpoint(model, checkpoint_path, device):
    if not checkpoint_path:
        return
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"loaded_init_checkpoint={checkpoint_path}")
    print(f"init_missing_keys={len(missing)} init_unexpected_keys={len(unexpected)}")
    if missing:
        print("init_missing_preview=" + ",".join(missing[:8]))
    if unexpected:
        print("init_unexpected_preview=" + ",".join(unexpected[:8]))


def soft_dice_loss(prob, target):
    dims = tuple(range(1, prob.ndim))
    inter = (prob * target).sum(dim=dims)
    denom = prob.sum(dim=dims) + target.sum(dim=dims)
    return 1.0 - ((2.0 * inter + 1e-6) / (denom + 1e-6)).mean()


def loss_and_prob(logits, masks):
    if logits.shape[-2:] != masks.shape[-2:]:
        masks = F.interpolate(masks.float(), size=logits.shape[-2:], mode="nearest")
    bce = F.binary_cross_entropy_with_logits(logits.float(), masks.float())
    prob = torch.sigmoid(logits.float())
    dice = soft_dice_loss(prob, masks.float())
    return 0.5 * bce + 0.5 * dice, prob, masks


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


def surface_distances(pred, mask):
    pred = pred.astype(bool)
    mask = mask.astype(bool)
    if not pred.any() and not mask.any():
        return np.array([0.0], dtype=np.float32), np.array([0.0], dtype=np.float32)
    if not pred.any() or not mask.any():
        diag = math.sqrt(pred.shape[-2] ** 2 + pred.shape[-1] ** 2)
        return np.array([diag], dtype=np.float32), np.array([diag], dtype=np.float32)

    pred_surface = pred ^ binary_erosion(pred)
    mask_surface = mask ^ binary_erosion(mask)
    if not pred_surface.any():
        pred_surface = pred
    if not mask_surface.any():
        mask_surface = mask
    dt_pred = distance_transform_edt(~pred_surface)
    dt_mask = distance_transform_edt(~mask_surface)
    return dt_mask[pred_surface], dt_pred[mask_surface]


def sample_metrics(prob, mask):
    pred = (prob >= 0.5).astype(np.uint8)
    target = (mask >= 0.5).astype(np.uint8)
    inter = float((pred * target).sum())
    pred_sum = float(pred.sum())
    target_sum = float(target.sum())
    dice = (2.0 * inter + 1e-6) / (pred_sum + target_sum + 1e-6)
    iou = (inter + 1e-6) / (pred_sum + target_sum - inter + 1e-6)
    precision = (inter + 1e-6) / (pred_sum + 1e-6)
    recall = (inter + 1e-6) / (target_sum + 1e-6)
    d_pm, d_mp = surface_distances(pred.squeeze(), target.squeeze())
    all_d = np.concatenate([d_pm, d_mp])
    hd95 = float(np.percentile(all_d, 95))
    assd = float(all_d.mean())
    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "hd95": hd95,
        "assd": assd,
    }


@torch.no_grad()
def evaluate(model, loader, device, collect_samples=False):
    model.eval()
    losses, dices, ious = [], [], []
    samples = []
    for images, masks, datasets, names in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = model(images)
        loss, prob, target = loss_and_prob(logits, masks)
        dice, iou = batch_metrics(prob, target)
        losses.append(loss.item())
        dices.append(dice)
        ious.append(iou)
        if collect_samples:
            probs = prob.detach().cpu().numpy()
            targets = target.detach().cpu().numpy()
            for i, name in enumerate(names):
                row = sample_metrics(probs[i, 0], targets[i, 0])
                row.update({"dataset": datasets[i], "filename": name})
                samples.append(row)
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "dice": float(np.mean(dices)) if dices else float("nan"),
        "iou": float(np.mean(ious)) if ious else float("nan"),
        "samples": samples,
    }


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean_std(values):
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0


def write_results(args, best_metrics, final_eval, checkpoint_path, out_dir):
    tag = "aug" if args.augment else "noaug"
    stem = f"RollingUNet_{args.dataset}_{tag}_{args.profile}_seed{args.seed}"
    sample_rows = []
    for row in final_eval["samples"]:
        sample_rows.append(
            {
                "model": "RollingUNet",
                "dataset": row["dataset"],
                "filename": row["filename"],
                "dice": row["dice"],
                "iou": row["iou"],
                "precision": row["precision"],
                "recall": row["recall"],
                "hd95": row["hd95"],
                "assd": row["assd"],
                "threshold": 0.5,
                "checkpoint": str(checkpoint_path),
                "augmentation": args.profile if args.augment else "none",
                "seed": args.seed,
                "split_seed": args.split_seed,
                "split_mode": args.split_mode,
                "init_checkpoint": args.init_checkpoint,
            }
        )
    per_sample_path = out_dir / f"eval_{stem}_per_sample.csv"
    sample_fields = [
        "model",
        "dataset",
        "filename",
        "dice",
        "iou",
        "precision",
        "recall",
        "hd95",
        "assd",
        "threshold",
        "checkpoint",
        "augmentation",
        "seed",
        "split_seed",
        "split_mode",
        "init_checkpoint",
    ]
    write_csv(per_sample_path, sample_rows, sample_fields)

    summary = {
        "model": "RollingUNet",
        "dataset": args.dataset,
        "augmentation": args.profile if args.augment else "none",
        "seed": args.seed,
        "split_seed": args.split_seed,
        "split_mode": args.split_mode,
        "init_checkpoint": args.init_checkpoint,
        "checkpoint": str(checkpoint_path),
        "best_epoch": best_metrics["epoch"],
        "best_val_loss": best_metrics["loss"],
        "best_val_dice": best_metrics["dice"],
        "best_val_iou": best_metrics["iou"],
        "n_val": len(sample_rows),
    }
    for metric in ["dice", "iou", "precision", "recall", "hd95", "assd"]:
        mean, std = mean_std([r[metric] for r in final_eval["samples"]])
        summary[f"{metric}_mean"] = mean
        summary[f"{metric}_std"] = std

    summary_path = out_dir / f"eval_{stem}_summary.csv"
    write_csv(summary_path, [summary], list(summary.keys()))
    status_path = out_dir / f"{stem}.status"
    status_path.write_text(
        "pass "
        f"dataset={args.dataset} seed={args.seed} dice={summary['dice_mean']:.6f} "
        f"iou={summary['iou_mean']:.6f} best_epoch={summary['best_epoch']} "
        f"checkpoint={checkpoint_path}\n",
        encoding="utf-8",
    )
    return per_sample_path, summary_path, status_path


def train(args):
    seed_all(args.seed)
    args.split_seed = args.seed if args.split_seed is None else args.split_seed
    val_filenames = read_val_filenames(args.val_filenames_csv, args.dataset)
    if args.split_mode == "explicit_val" and val_filenames is None:
        raise ValueError("--split-mode explicit_val requires --val-filenames-csv")
    if val_filenames is not None:
        args.split_mode = "explicit_val"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")

    train_rows, val_rows = load_rows(
        args.dataset,
        split_seed=args.split_seed,
        max_train=args.max_train_samples,
        max_val=args.max_val_samples,
        split_mode=args.split_mode,
        val_filenames=val_filenames,
        manifest_path=args.manifest or None,
    )
    if not train_rows or not val_rows:
        raise RuntimeError(f"Empty split for dataset={args.dataset}: train={len(train_rows)} val={len(val_rows)}")

    model = build_model(args.image_size).to(device)
    load_init_checkpoint(model, args.init_checkpoint, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs),
        eta_min=args.min_lr,
    )

    train_ds = SegDataset(
        train_rows, args.image_size, augment=args.augment, profile_name=args.profile, resize_mode=args.resize_mode
    )
    val_ds = SegDataset(val_rows, args.image_size, augment=False, profile_name=args.profile, resize_mode=args.resize_mode)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available() and not args.cpu,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available() and not args.cpu,
    )

    tag = "aug" if args.augment else "noaug"
    stem = f"RollingUNet_{args.dataset}_{tag}_{args.profile}_seed{args.seed}"
    checkpoint_path = out_dir / f"{stem}_best.pt"
    best = {"epoch": 0, "loss": float("inf"), "dice": -1.0, "iou": -1.0}

    print(f"model=RollingUNet")
    print(f"dataset={args.dataset}")
    print(f"train_samples={len(train_rows)} val_samples={len(val_rows)}")
    print(f"augment={args.augment} profile={args.profile}")
    print(f"epochs={args.epochs} batch_size={args.batch_size} image_size={args.image_size} resize_mode={args.resize_mode}")
    print(f"device={device} lr={args.lr} min_lr={args.min_lr} weight_decay={args.weight_decay}")
    print(f"init_checkpoint={args.init_checkpoint}")
    print(f"split_seed={args.split_seed} seed={args.seed}")
    print(f"split_mode={args.split_mode}")
    print(f"val_filenames_csv={args.val_filenames_csv}")
    print(f"val_filename_count={len(val_filenames) if val_filenames is not None else 0}")
    print(f"out_dir={out_dir}")

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        model.train()
        losses, dices, ious = [], [], []
        for images, masks, _, _ in train_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss, prob, target = loss_and_prob(logits, masks)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            dice, iou = batch_metrics(prob.detach(), target.detach())
            losses.append(loss.item())
            dices.append(dice)
            ious.append(iou)
        scheduler.step()

        val_metrics = evaluate(model, val_loader, device, collect_samples=False)
        train_loss = float(np.mean(losses)) if losses else float("nan")
        train_dice = float(np.mean(dices)) if dices else float("nan")
        train_iou = float(np.mean(ious)) if ious else float("nan")
        elapsed = time.time() - start
        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.5f} train_dice={train_dice:.5f} train_iou={train_iou:.5f} "
            f"val_loss={val_metrics['loss']:.5f} val_dice={val_metrics['dice']:.5f} "
            f"val_iou={val_metrics['iou']:.5f} lr={scheduler.get_last_lr()[0]:.6g} "
            f"time={elapsed:.1f}s",
            flush=True,
        )
        if val_metrics["dice"] > best["dice"]:
            best = {
                "epoch": epoch,
                "loss": val_metrics["loss"],
                "dice": val_metrics["dice"],
                "iou": val_metrics["iou"],
            }
            torch.save(
                {
                    "model": "RollingUNet",
                    "state_dict": model.state_dict(),
                    "args": vars(args),
                    "best": best,
                },
                checkpoint_path,
            )
            print(f"best_checkpoint: {checkpoint_path}", flush=True)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    final_eval = evaluate(model, val_loader, device, collect_samples=True)
    per_sample_path, summary_path, status_path = write_results(args, best, final_eval, checkpoint_path, out_dir)
    print(f"best_epoch: {best['epoch']}")
    print(f"best_val_dice: {best['dice']:.6f}")
    print(f"best_val_iou: {best['iou']:.6f}")
    print(f"per_sample_csv: {per_sample_path}")
    print(f"summary_csv: {summary_path}")
    print(f"status: {status_path.read_text(encoding='utf-8').strip()}")

    if args.test_manifest:
        test_rows = read_manifest(args.test_manifest)
        if args.dataset:
            test_rows = [r for r in test_rows if r.get("dataset") == args.dataset]
        split_values = {r.get("split", "").strip().lower() for r in test_rows}
        if split_values & {"train", "val", "valid", "validation", "test"}:
            test_rows = [r for r in test_rows if r.get("split", "").strip().lower() == "test"]
        if not test_rows:
            raise ValueError(f"No test rows found in {args.test_manifest} for dataset={args.dataset}")
        test_ds = SegDataset(
            test_rows, args.image_size, augment=False, profile_name=args.profile, resize_mode=args.resize_mode
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=torch.cuda.is_available() and not args.cpu,
        )
        test_eval = evaluate(model, test_loader, device, collect_samples=True)
        test_per = out_dir / "test_eval_per_sample.csv"
        test_summary = out_dir / "test_eval_summary.csv"
        sample_rows = []
        for row in test_eval["samples"]:
            sample_rows.append(
                {
                    "model": "RollingUNet",
                    "dataset": row["dataset"],
                    "filename": row["filename"],
                    "dice": row["dice"],
                    "iou": row["iou"],
                    "precision": row["precision"],
                    "recall": row["recall"],
                    "hd95": row["hd95"],
                    "assd": row["assd"],
                    "threshold": 0.5,
                    "checkpoint": str(checkpoint_path),
                }
            )
        fields = ["model", "dataset", "filename", "dice", "iou", "precision", "recall", "hd95", "assd", "threshold", "checkpoint"]
        write_csv(test_per, sample_rows, fields)
        summary_rows = []
        grouped = defaultdict(list)
        for row in sample_rows:
            grouped[(row["model"], row["dataset"])].append(row)
            grouped[(row["model"], "ALL")].append(row)
        for (model_name, dataset_name), items in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1] != "ALL", x[0][1])):
            summary = {"model": model_name, "dataset": dataset_name, "samples": len(items)}
            for metric in ["dice", "iou", "precision", "recall", "hd95", "assd"]:
                values = np.asarray([float(item[metric]) for item in items], dtype=np.float64)
                finite = values[np.isfinite(values)]
                summary[f"{metric}_mean"] = f"{finite.mean():.6f}" if finite.size else "inf"
                summary[f"{metric}_std"] = f"{finite.std(ddof=1):.6f}" if finite.size > 1 else "0.000000"
            summary_rows.append(summary)
        with test_summary.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["model", "dataset", "samples", "dice_mean", "dice_std", "iou_mean", "iou_std", "precision_mean", "precision_std", "recall_mean", "recall_std", "hd95_mean", "hd95_std", "assd_mean", "assd_std"],
            )
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"test_per_sample_csv: {test_per}")
        print(f"test_summary_csv: {test_summary}")
        for row in summary_rows:
            if row["dataset"] == "ALL":
                print(f"test_ALL dice={row['dice_mean']} iou={row['iou_mean']}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        required=True,
        choices=[
            "busi",
            "cvc",
            "glas",
            "busi_hf",
            "brain_mri",
            "brain_mri_tumor",
            "medclipseg_busi",
            "medclipseg_clinicdb",
            "medclipseg_busbra",
            "medclipseg_brisc",
            "medclipseg_covid19",
        ],
    )
    parser.add_argument("--manifest", default="", help="Optional manifest CSV for brain_mri or medclipseg datasets.")
    parser.add_argument("--test-manifest", default="", help="Optional test-only manifest CSV for post-train evaluation.")
    parser.add_argument("--out-dir", default=str(ROOT / "logs" / "rolling_image_aug_pilot"))
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--resize-mode",
        choices=["stretch", "letterbox"],
        default="stretch",
        help="Resize policy: stretch (legacy) or letterbox (aspect-preserving pad).",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--split-mode", choices=["dataset", "global_public", "explicit_val"], default="dataset")
    parser.add_argument("--val-filenames-csv", default="")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="common_light")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
