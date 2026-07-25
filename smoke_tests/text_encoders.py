#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from protocol_v3.core import canonical_hash, file_sha256


CACHE_SCHEMA_VERSION = "text_embedding_cache_v3"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFESTS = tuple(
    ROOT / "smoke_tests/protocol_v3/manifests" / name
    for name in (
        "medclipseg_busi_full.csv",
        "medclipseg_clinicdb_full.csv",
        "medclipseg_busbra_full.csv",
        "medclipseg_brisc_full.csv",
        "medclipseg_covid19_full.csv",
    )
)


def lcaug_text_variants(text: str, *, use_v2: bool = True) -> set[str]:
    if use_v2:
        from lcaug_v2_direction import lcaug_v2_text_variants

        return lcaug_v2_text_variants(text)

    from augmentation_plugins import transform_direction_text

    variants = {text}
    for hflip in [False, True]:
        for vflip in [False, True]:
            for rot90 in [0, 1, 2, 3]:
                current = text
                if hflip:
                    current = transform_direction_text(current, "hflip")
                if vflip:
                    current = transform_direction_text(current, "vflip")
                if rot90:
                    current = transform_direction_text(current, "rot90_ccw", times=rot90)
                variants.add(current)
    return variants


def text_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def manifest_fingerprints(manifests) -> list[dict[str, str]]:
    items = manifests if isinstance(manifests, (list, tuple)) else [manifests]
    return [
        {"path": str(Path(item).resolve()), "sha256": file_sha256(item)}
        for item in items
    ]


def cache_expectations(
    manifests,
    texts: list[str],
    include_lcaug_variants: bool,
    include_empty_control: bool = False,
) -> dict:
    from lcaug_v2_direction import DIRECTION_VERSION

    return {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "manifest_fingerprints": manifest_fingerprints(manifests),
        "prompt_set_sha256": canonical_hash(sorted(texts)),
        "direction_version": DIRECTION_VERSION if include_lcaug_variants else "none",
        "include_lcaug_variants": bool(include_lcaug_variants),
        "include_empty_control": bool(include_empty_control),
        "expected_text_count": len(texts),
    }


class TextEmbeddingCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        data = np.load(self.path, allow_pickle=False)
        self.keys = [str(k) for k in data["keys"]]
        self.texts = [str(t) for t in data["texts"]]
        self.embeddings = data["embeddings"].astype("float32")
        self.dim = int(self.embeddings.shape[-1])
        self.tokens = int(self.embeddings.shape[1])
        self.index = {key: i for i, key in enumerate(self.keys)}
        meta_path = self.path.with_suffix(self.path.suffix + ".json")
        self.meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}

    def missing_texts(self, texts) -> list[str]:
        return [str(text) for text in texts if text_key(str(text)) not in self.index]

    def validate_texts(self, texts) -> None:
        missing = self.missing_texts(texts)
        if missing:
            preview = "; ".join(repr(text) for text in missing[:10])
            raise KeyError(f"Embedding cache misses {len(missing)} required texts: {preview}")

    def validate_metadata(self, expected: dict) -> None:
        mismatches = []
        for key, value in expected.items():
            if self.meta.get(key) != value:
                mismatches.append(f"{key}: cache={self.meta.get(key)!r} expected={value!r}")
        if mismatches:
            raise ValueError("Embedding cache metadata mismatch: " + "; ".join(mismatches))

    def batch(self, texts, tokens: int, dim: int, device):
        rows = []
        for text in texts:
            key = text_key(str(text))
            if key not in self.index:
                raise KeyError(f"Text not present in embedding cache: {text!r}")
            emb = self.embeddings[self.index[key]]
            emb = _resize_embedding(emb, tokens=tokens, dim=dim)
            rows.append(emb)
        return torch.from_numpy(np.stack(rows).astype("float32")).to(device)

    def batch_features(self, texts, device):
        rows = []
        for text in texts:
            key = text_key(str(text))
            if key not in self.index:
                raise KeyError(f"Text not present in embedding cache: {text!r}")
            emb = self.embeddings[self.index[key]].astype("float32")
            if emb.ndim == 2:
                emb = emb.mean(axis=0)
            rows.append(emb)
        return torch.from_numpy(np.stack(rows).astype("float32")).to(device)


def _resize_embedding(embedding: np.ndarray, tokens: int, dim: int) -> np.ndarray:
    out = embedding
    if out.shape[0] < tokens:
        pad = np.zeros((tokens - out.shape[0], out.shape[1]), dtype=out.dtype)
        out = np.concatenate([out, pad], axis=0)
    else:
        out = out[:tokens]
    if out.shape[1] < dim:
        pad = np.zeros((out.shape[0], dim - out.shape[1]), dtype=out.dtype)
        out = np.concatenate([out, pad], axis=1)
    else:
        out = out[:, :dim]
    return out


def load_text_embedding_cache(path):
    if not path:
        return None
    return TextEmbeddingCache(path)


def unique_texts_from_manifest(manifest, include_lcaug_variants=False) -> list[str]:
    manifests = manifest if isinstance(manifest, (list, tuple)) else [manifest]
    rows = []
    for item in manifests:
        with Path(item).open(newline="", encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
    texts = {row["text"] for row in rows}
    if include_lcaug_variants:
        expanded = set()
        for text in texts:
            expanded.update(lcaug_text_variants(text))
        texts = expanded
    return sorted(texts)


def build_hf_cache(args):
    from transformers import AutoModel, AutoTokenizer

    texts = unique_texts_from_manifest(args.manifest, include_lcaug_variants=args.include_lcaug_variants)
    if args.include_empty_control:
        texts = sorted(set(texts) | {""})
    tokenizer = AutoTokenizer.from_pretrained(args.encoder, local_files_only=args.local_files_only)
    model = AutoModel.from_pretrained(args.encoder, local_files_only=args.local_files_only).eval()
    device = torch.device(args.device)
    model.to(device)

    embeddings = []
    with torch.no_grad():
        for start in range(0, len(texts), args.batch_size):
            batch = texts[start : start + args.batch_size]
            encoded = tokenizer(
                batch,
                padding="max_length",
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            ).to(device)
            output = model(**encoded)
            hidden = output.last_hidden_state.detach().cpu().numpy().astype("float32")
            embeddings.append(hidden)
    embeddings = np.concatenate(embeddings, axis=0) if embeddings else np.zeros((0, args.max_length, 768), dtype="float32")
    keys = np.asarray([text_key(text) for text in texts])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, keys=keys, texts=np.asarray(texts), embeddings=embeddings)

    meta = {
        "encoder": args.encoder,
        "manifest": [str(item) for item in args.manifest],
        "texts": len(texts),
        "include_lcaug_variants": bool(args.include_lcaug_variants),
        "include_empty_control": bool(args.include_empty_control),
        "max_length": args.max_length,
        "embedding_shape": list(embeddings.shape),
    }
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print(f"output: {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", nargs="+", default=[str(path) for path in DEFAULT_MANIFESTS])
    parser.add_argument("--encoder", default="emilyalsentzer/Bio_ClinicalBERT")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-length", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--include-lcaug-variants", action="store_true", help="Also encode all text variants reachable by LCAug direction rewrites.")
    parser.add_argument("--include-empty-control", action="store_true", help="Include the empty prompt used by Protocol V3 semantic controls.")
    args = parser.parse_args()
    build_hf_cache(args)


if __name__ == "__main__":
    main()
