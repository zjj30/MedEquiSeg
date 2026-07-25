#!/usr/bin/env python3
"""Shared Protocol V3 data, hashing, split, and mask semantics."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_DIR = Path(__file__).resolve().parent
PROTOCOL_ID = "MEDSEG_TEXT_V3_20260710"
DEFAULT_PROTOCOL_LOCK = PROTOCOL_DIR / "protocol_lock.yaml"
DEFAULT_DATASET_REGISTRY = PROTOCOL_DIR / "dataset_registry.yaml"

REQUIRED_MANIFEST_FIELDS = (
    "dataset",
    "split",
    "case_id",
    "patient_id",
    "image_path",
    "mask_path",
    "text",
    "mask_mode",
    "prompt_source",
    "image_sha256",
    "mask_sha256",
)

SPLIT_ALIASES = {
    "train": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "test": "test",
}


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def manifest_sha256(path: str | Path) -> str:
    return file_sha256(path)


def _resolve_project_path(raw: str | Path, *, base: Path = ROOT) -> Path:
    value = str(raw).strip().replace("\\", "/")
    marker = "<PROJECT_ROOT>"
    if value == marker:
        return base
    if value.startswith(marker + "/"):
        return base / value[len(marker) + 1 :]
    path = Path(value)
    return path if path.is_absolute() else base / path


def load_protocol_lock(path: str | Path = DEFAULT_PROTOCOL_LOCK) -> dict[str, Any]:
    lock_path = Path(path)
    payload = yaml.safe_load(lock_path.read_text(encoding="utf-8")) or {}
    protocol_id = str(payload.get("protocol_id") or "").strip()
    if not protocol_id:
        raise ValueError(f"Protocol lock has no protocol_id: {lock_path}")
    return payload


def load_dataset_registry(path: str | Path = DEFAULT_DATASET_REGISTRY) -> dict[str, dict[str, Any]]:
    registry_path = Path(path)
    payload = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    datasets = payload.get("datasets") or {}
    if not isinstance(datasets, dict) or not datasets:
        raise ValueError(f"No datasets in registry: {registry_path}")
    return datasets


def protocol_sha256(path: str | Path = DEFAULT_PROTOCOL_LOCK) -> str:
    lock_path = Path(path)
    payload = load_protocol_lock(lock_path)
    registry_raw = payload.get("dataset_registry", str(DEFAULT_DATASET_REGISTRY))
    registry_path = _resolve_project_path(registry_raw)
    material: dict[str, Any] = {
        "lock": payload,
        "lock_sha256": file_sha256(lock_path),
        "registry_sha256": file_sha256(registry_path),
        "manifests": {},
    }
    for dataset, cfg in sorted((payload.get("datasets") or {}).items()):
        manifest_raw = cfg.get("manifest")
        if not manifest_raw:
            raise ValueError(f"Dataset {dataset} has no manifest in {lock_path}")
        manifest_path = _resolve_project_path(manifest_raw)
        material["manifests"][dataset] = {
            "path": str(manifest_path),
            "sha256": file_sha256(manifest_path) if manifest_path.is_file() else "missing",
        }
    return canonical_hash(material)


def read_manifest(path: str | Path) -> tuple[list[str], list[dict[str, str]]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
        fields = list(reader.fieldnames or [])
    for row in rows:
        for field in ("image_path", "mask_path"):
            value = str(row.get(field, "")).strip()
            if value:
                row[field] = str(_resolve_project_path(value))
    return fields, rows


def normalize_split(value: str) -> str:
    key = str(value or "").strip().lower()
    if key not in SPLIT_ALIASES:
        raise ValueError(f"Unsupported split: {value!r}")
    return SPLIT_ALIASES[key]


def validate_manifest_rows(
    fields: Iterable[str],
    rows: list[dict[str, str]],
    *,
    require_train_val_test: bool = True,
    check_files: bool = True,
) -> dict[str, Any]:
    fields_set = set(fields)
    missing_fields = [field for field in REQUIRED_MANIFEST_FIELDS if field not in fields_set]
    if missing_fields:
        raise ValueError(f"Manifest missing fields: {', '.join(missing_fields)}")
    if not rows:
        raise ValueError("Manifest has no rows")

    split_counts: dict[str, int] = defaultdict(int)
    seen_by_kind: dict[str, dict[str, str]] = {
        "case_id": {},
        "image_path": {},
        "mask_path": {},
        "image_sha256": {},
        "patient_id": {},
    }
    if "group_id" in fields_set:
        seen_by_kind["group_id"] = {}
    duplicates: list[str] = []
    for index, row in enumerate(rows, start=2):
        split = normalize_split(row.get("split", ""))
        row["split"] = split
        split_counts[split] += 1
        for field in REQUIRED_MANIFEST_FIELDS:
            if field == "patient_id":
                continue
            if not str(row.get(field, "")).strip():
                raise ValueError(f"Row {index} has empty required field {field}")
        if check_files:
            for field in ("image_path", "mask_path"):
                path = Path(row[field])
                if not path.is_file():
                    raise FileNotFoundError(f"Row {index} missing {field}: {path}")
        for kind in seen_by_kind:
            value = str(row.get(kind, "")).strip()
            if not value:
                continue
            previous = seen_by_kind[kind].get(value)
            if previous is not None and previous != split:
                duplicates.append(f"{kind}={value} crosses {previous}/{split}")
            else:
                seen_by_kind[kind][value] = split
    if duplicates:
        preview = "; ".join(duplicates[:20])
        raise ValueError(f"Cross-split leakage detected ({len(duplicates)}): {preview}")
    required = {"train", "val", "test"} if require_train_val_test else {"train", "val"}
    missing_splits = sorted(required - set(split_counts))
    if missing_splits:
        raise ValueError(f"Manifest missing required splits: {', '.join(missing_splits)}")
    return {"rows": len(rows), "split_counts": dict(sorted(split_counts.items()))}


def load_manifest_splits(
    manifest: str | Path,
    dataset_filter: Iterable[str] | None = None,
    *,
    require_train_val_test: bool = True,
    check_files: bool = True,
) -> dict[str, list[dict[str, str]]]:
    fields, rows = read_manifest(manifest)
    if dataset_filter:
        allowed = set(dataset_filter)
        rows = [row for row in rows if row.get("dataset") in allowed]
    validate_manifest_rows(
        fields,
        rows,
        require_train_val_test=require_train_val_test,
        check_files=check_files,
    )
    splits = {"train": [], "val": [], "test": []}
    for row in rows:
        splits[normalize_split(row["split"])].append(row)
    for split in splits:
        splits[split].sort(key=lambda row: (row["dataset"], row["case_id"], row["image_path"]))
    return splits


def binarize_mask(mask: np.ndarray, mode: str) -> np.ndarray:
    array = np.asarray(mask)
    key = str(mode).strip().lower()
    if key == "nonzero_label":
        out = array > 0
    elif key == "threshold_127":
        out = array >= 127
    elif key == "binary_01":
        values = set(np.unique(array).tolist())
        if not values.issubset({0, 1}):
            raise ValueError(f"binary_01 mask contains values outside 0/1: {sorted(values)[:20]}")
        out = array == 1
    else:
        raise ValueError(f"Unsupported mask_mode: {mode!r}")
    return out.astype("float32")
