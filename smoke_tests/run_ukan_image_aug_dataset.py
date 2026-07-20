#!/usr/bin/env python3
import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from run_rolling_image_aug_dataset import (
    PROFILES,
    ROOT,
    SegDataset,
    batch_metrics,
    evaluate,
    load_rows,
    loss_and_prob,
    mean_std,
    read_manifest,
    seed_all,
    write_csv,
)


UKAN_REPO = ROOT / "repos" / "U-KAN" / "Seg_UKAN"
MODEL_NAME = "UKAN"
PROTOCOL_DATASETS = (
    "medclipseg_busi",
    "medclipseg_clinicdb",
    "medclipseg_busbra",
    "medclipseg_brisc",
    "medclipseg_covid19",
)


class SingleOutputModel(nn.Module):
    """Normalize MONAI deep-supervision-style outputs to one logits tensor."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs):
        outputs = self.model(inputs)
        return outputs[0] if isinstance(outputs, (list, tuple)) else outputs


def architecture_name(architecture: str) -> str:
    return {
        "ukan": "UKAN",
        "unet": "UNet",
        "unetplusplus": "UNetPlusPlus",
    }[architecture]


def architecture_source(architecture: str) -> str:
    if architecture == "ukan":
        return str(UKAN_REPO)
    import monai

    return f"MONAI-{monai.__version__}:monai.networks.nets"


def build_model(image_size, no_kan=False, architecture="ukan"):
    if architecture == "ukan":
        sys.path.insert(0, str(UKAN_REPO))
        from archs import UKAN

        return UKAN(num_classes=1, input_channels=3, img_size=image_size, no_kan=no_kan)
    from monai.networks.nets import BasicUNetPlusPlus, UNet

    if architecture == "unet":
        return UNet(
            spatial_dims=2,
            in_channels=3,
            out_channels=1,
            channels=(32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )
    if architecture == "unetplusplus":
        return SingleOutputModel(
            BasicUNetPlusPlus(
                spatial_dims=2,
                in_channels=3,
                out_channels=1,
                features=(32, 32, 64, 128, 256, 32),
                deep_supervision=False,
            )
        )
    raise ValueError(f"Unsupported architecture: {architecture}")


def write_results(args, best_metrics, final_eval, checkpoint_path, out_dir):
    model_name = architecture_name(args.architecture)
    source = architecture_source(args.architecture)
    tag = "aug" if args.augment else "noaug"
    stem = f"{model_name}_{args.dataset}_{tag}_{args.profile}_seed{args.seed}"
    sample_rows = []
    for row in final_eval["samples"]:
        sample_rows.append(
            {
                "model": model_name,
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
                "no_kan": args.no_kan,
                "source": source,
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
        "no_kan",
        "source",
    ]
    write_csv(per_sample_path, sample_rows, sample_fields)

    summary = {
        "model": model_name,
        "dataset": args.dataset,
        "augmentation": args.profile if args.augment else "none",
        "seed": args.seed,
        "split_seed": args.split_seed,
        "split_mode": args.split_mode,
        "no_kan": args.no_kan,
        "checkpoint": str(checkpoint_path),
        "source": source,
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


def write_test_results(args, test_eval, checkpoint_path, out_dir):
    model_name = architecture_name(args.architecture)
    source = architecture_source(args.architecture)
    sample_rows = []
    for row in test_eval["samples"]:
        sample_rows.append(
            {
                "model": model_name,
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
                "source": source,
            }
        )
    fields = [
        "model", "dataset", "filename", "dice", "iou", "precision", "recall",
        "hd95", "assd", "threshold", "checkpoint", "source",
    ]
    per_sample_path = out_dir / "test_eval_per_sample.csv"
    write_csv(per_sample_path, sample_rows, fields)

    grouped = defaultdict(list)
    for row in sample_rows:
        grouped[(row["model"], row["dataset"])].append(row)
        grouped[(row["model"], "ALL")].append(row)
    summary_rows = []
    for (model_name, dataset_name), items in sorted(grouped.items()):
        summary = {"model": model_name, "dataset": dataset_name, "samples": len(items)}
        for metric in ("dice", "iou", "precision", "recall", "hd95", "assd"):
            values = np.asarray([float(item[metric]) for item in items], dtype=np.float64)
            finite = values[np.isfinite(values)]
            summary[f"{metric}_mean"] = f"{finite.mean():.6f}" if finite.size else "inf"
            summary[f"{metric}_std"] = f"{finite.std(ddof=1):.6f}" if finite.size > 1 else "0.000000"
        summary_rows.append(summary)
    summary_path = out_dir / "test_eval_summary.csv"
    summary_fields = ["model", "dataset", "samples"] + [
        f"{metric}_{stat}"
        for metric in ("dice", "iou", "precision", "recall", "hd95", "assd")
        for stat in ("mean", "std")
    ]
    write_csv(summary_path, summary_rows, summary_fields)
    return per_sample_path, summary_path


def train(args):
    seed_all(args.seed)
    args.split_seed = args.seed if args.split_seed is None else args.split_seed
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")

    train_rows, val_rows = load_rows(
        args.dataset,
        split_seed=args.split_seed,
        max_train=args.max_train_samples,
        max_val=args.max_val_samples,
        split_mode=args.split_mode,
        manifest_path=args.manifest or None,
    )
    if not train_rows or not val_rows:
        raise RuntimeError(f"Empty split for dataset={args.dataset}: train={len(train_rows)} val={len(val_rows)}")

    model_name = architecture_name(args.architecture)
    model = build_model(args.image_size, no_kan=args.no_kan, architecture=args.architecture).to(device)
    if args.forward_smoke:
        with torch.no_grad():
            smoke = model(torch.zeros(1, 3, args.image_size, args.image_size, device=device))
        print(f"forward_smoke_output={tuple(smoke.shape)}")
        return

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs),
        eta_min=args.min_lr,
    )

    train_ds = SegDataset(
        train_rows, args.image_size, augment=args.augment, profile_name=args.profile, resize_mode=args.resize_mode
    )
    val_ds = SegDataset(
        val_rows, args.image_size, augment=False, profile_name=args.profile, resize_mode=args.resize_mode
    )
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
    stem = f"{model_name}_{args.dataset}_{tag}_{args.profile}_seed{args.seed}"
    checkpoint_path = out_dir / f"{stem}_best.pt"
    best = {"epoch": 0, "loss": float("inf"), "dice": -1.0, "iou": -1.0}

    print(f"model={model_name}")
    print(f"architecture={args.architecture}")
    print(f"dataset={args.dataset}")
    print(f"train_samples={len(train_rows)} val_samples={len(val_rows)}")
    print(f"augment={args.augment} profile={args.profile}")
    print(f"epochs={args.epochs} batch_size={args.batch_size} image_size={args.image_size}")
    print(f"device={device} lr={args.lr} min_lr={args.min_lr} weight_decay={args.weight_decay}")
    print(f"split_seed={args.split_seed} seed={args.seed}")
    print(f"split_mode={args.split_mode}")
    print(f"no_kan={args.no_kan}")
    print(f"source={architecture_source(args.architecture)}")
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
                    "model": model_name,
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
        test_rows = [row for row in test_rows if row.get("dataset") == args.dataset]
        if any(row.get("split", "").strip().lower() for row in test_rows):
            test_rows = [row for row in test_rows if row.get("split", "").strip().lower() == "test"]
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
        test_per_sample, test_summary = write_test_results(args, test_eval, checkpoint_path, out_dir)
        print(f"test_per_sample_csv: {test_per_sample}")
        print(f"test_summary_csv: {test_summary}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", required=True, choices=["busi", "cvc", "glas", "busi_hf", *PROTOCOL_DATASETS]
    )
    parser.add_argument("--manifest", default="")
    parser.add_argument("--test-manifest", default="")
    parser.add_argument("--out-dir", default=str(ROOT / "logs" / "ukan_image_aug_pilot"))
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--resize-mode", choices=["stretch", "letterbox"], default="stretch")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--split-mode", choices=["dataset", "global_public"], default="dataset")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="common_light")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--no-kan", action="store_true")
    parser.add_argument("--architecture", choices=["ukan", "unet", "unetplusplus"], default="ukan")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--forward-smoke", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
