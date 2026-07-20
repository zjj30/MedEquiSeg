#!/usr/bin/env python3
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage


EPS = 1e-6


@dataclass(frozen=True)
class SegmentationMetrics:
    dice: float
    iou: float
    precision: float
    recall: float
    hd95: float
    assd: float

    def as_dict(self) -> dict[str, float]:
        return {
            "dice": self.dice,
            "iou": self.iou,
            "precision": self.precision,
            "recall": self.recall,
            "hd95": self.hd95,
            "assd": self.assd,
        }


def _surface(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    if not mask.any():
        return mask
    eroded = ndimage.binary_erosion(mask)
    return mask ^ eroded


def _surface_distances(pred: np.ndarray, target: np.ndarray, spacing: tuple[float, float] = (1.0, 1.0)) -> tuple[np.ndarray, np.ndarray]:
    pred_surface = _surface(pred)
    target_surface = _surface(target)
    if not pred_surface.any() or not target_surface.any():
        return np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32)
    target_distance = ndimage.distance_transform_edt(~target_surface, sampling=spacing)
    pred_distance = ndimage.distance_transform_edt(~pred_surface, sampling=spacing)
    return target_distance[pred_surface].astype(np.float32), pred_distance[target_surface].astype(np.float32)


def binary_segmentation_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    spacing: tuple[float, float] = (1.0, 1.0),
    empty_value: float = 1.0,
) -> SegmentationMetrics:
    pred = pred.astype(bool)
    target = target.astype(bool)
    tp = float(np.logical_and(pred, target).sum())
    fp = float(np.logical_and(pred, ~target).sum())
    fn = float(np.logical_and(~pred, target).sum())

    pred_sum = tp + fp
    target_sum = tp + fn
    union = tp + fp + fn

    if pred_sum == 0 and target_sum == 0:
        dice = iou = precision = recall = empty_value
        hd95 = assd = 0.0
    else:
        dice = (2.0 * tp + EPS) / (pred_sum + target_sum + EPS)
        iou = (tp + EPS) / (union + EPS)
        precision = (tp + EPS) / (pred_sum + EPS)
        recall = (tp + EPS) / (target_sum + EPS)
        pred_to_target, target_to_pred = _surface_distances(pred, target, spacing=spacing)
        if pred_to_target.size == 0 or target_to_pred.size == 0:
            hd95 = assd = float("inf")
        else:
            all_distances = np.concatenate([pred_to_target, target_to_pred])
            hd95 = float(np.percentile(all_distances, 95))
            assd = float((pred_to_target.mean() + target_to_pred.mean()) / 2.0)

    return SegmentationMetrics(
        dice=float(dice),
        iou=float(iou),
        precision=float(precision),
        recall=float(recall),
        hd95=float(hd95) if math.isfinite(hd95) else float("inf"),
        assd=float(assd) if math.isfinite(assd) else float("inf"),
    )


@dataclass(frozen=True)
class ProtocolV3Metrics:
    dice: float
    iou: float
    nsd: float
    hd95: float
    assd: float
    both_empty: int
    one_empty_failure: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "dice": self.dice,
            "iou": self.iou,
            "nsd": self.nsd,
            "hd95": self.hd95,
            "assd": self.assd,
            "both_empty": self.both_empty,
            "one_empty_failure": self.one_empty_failure,
        }


def protocol_v3_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    spacing: tuple[float, float] = (1.0, 1.0),
    nsd_tolerance: float = 2.0,
) -> ProtocolV3Metrics:
    """Canonical overlap/boundary metrics with explicit empty-case semantics."""
    pred = np.asarray(pred).astype(bool)
    target = np.asarray(target).astype(bool)
    pred_any, target_any = bool(pred.any()), bool(target.any())
    if not pred_any and not target_any:
        return ProtocolV3Metrics(1.0, 1.0, 1.0, 0.0, 0.0, 1, 0)
    if pred_any != target_any:
        return ProtocolV3Metrics(0.0, 0.0, 0.0, float("inf"), float("inf"), 0, 1)

    intersection = float(np.logical_and(pred, target).sum())
    pred_sum = float(pred.sum())
    target_sum = float(target.sum())
    union = float(np.logical_or(pred, target).sum())
    dice = (2.0 * intersection) / (pred_sum + target_sum)
    iou = intersection / union
    pred_to_target, target_to_pred = _surface_distances(pred, target, spacing=spacing)
    if pred_to_target.size == 0 or target_to_pred.size == 0:
        return ProtocolV3Metrics(dice, iou, 0.0, float("inf"), float("inf"), 0, 1)
    distances = np.concatenate([pred_to_target, target_to_pred])
    nsd = float(
        ((pred_to_target <= nsd_tolerance).sum() + (target_to_pred <= nsd_tolerance).sum())
        / (pred_to_target.size + target_to_pred.size)
    )
    hd95 = float(np.percentile(distances, 95))
    assd = float((pred_to_target.mean() + target_to_pred.mean()) / 2.0)
    return ProtocolV3Metrics(dice, iou, nsd, hd95, assd, 0, 0)
