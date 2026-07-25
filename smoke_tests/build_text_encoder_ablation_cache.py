#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from biomedclip_offline import BIOMEDCLIP_OPENCLIP_ID, load_open_clip_biomedclip
from protocol_v3.core import canonical_hash, file_sha256
from text_encoders import cache_expectations, text_key, unique_texts_from_manifest


PUBMEDBERT_ID = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
BIOMEDCLIP_ID = BIOMEDCLIP_OPENCLIP_ID
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


def _device(name: str) -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def _l2_normalize(x: torch.Tensor, enabled: bool) -> torch.Tensor:
    return F.normalize(x, dim=-1) if enabled else x


def encoder_weights_fingerprint(args, encoder_meta: dict) -> str:
    candidates: list[Path] = []
    if args.encoder == "biomedclip" and args.biomedclip_local_dir:
        root = Path(args.biomedclip_local_dir)
        for name in (
            "open_clip_pytorch_model.bin",
            "pytorch_model.bin",
            "model.safetensors",
            "open_clip_config.json",
            "config.json",
        ):
            path = root / name
            if path.is_file():
                candidates.append(path)
    material = {
        "encoder": encoder_meta,
        "files": [
            {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": file_sha256(path)}
            for path in candidates
        ],
    }
    return canonical_hash(material)


def build_pubmedbert_cache(args, texts: list[str]) -> tuple[np.ndarray, dict]:
    from transformers import AutoModel, AutoTokenizer

    device = _device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.pubmedbert_model, local_files_only=args.local_files_only)
    model = AutoModel.from_pretrained(args.pubmedbert_model, local_files_only=args.local_files_only).eval().to(device)
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
            hidden = output.last_hidden_state.float()
            if args.pubmedbert_pool == "cls":
                pooled = hidden[:, 0, :]
            else:
                mask = encoded["attention_mask"].float().unsqueeze(-1)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            pooled = _l2_normalize(pooled, args.normalize)
            embeddings.append(pooled[:, None, :].detach().cpu().numpy().astype("float32"))
    arr = np.concatenate(embeddings, axis=0) if embeddings else np.zeros((0, 1, 768), dtype="float32")
    meta = {
        "encoder": "pubmedbert",
        "model": args.pubmedbert_model,
        "pool": args.pubmedbert_pool,
        "normalize": bool(args.normalize),
        "max_length": args.max_length,
    }
    return arr, meta


def build_biomedclip_cache(args, texts: list[str]) -> tuple[np.ndarray, dict]:
    try:
        import open_clip
    except Exception as exc:  # pragma: no cover - depends on optional remote package.
        raise RuntimeError(
            "BioMedCLIP cache generation requires open_clip. "
            "Install open_clip_torch in the active environment first."
        ) from exc

    device = _device(args.device)
    if args.biomedclip_local_dir:
        model, tokenizer = load_open_clip_biomedclip(device, local_dir=args.biomedclip_local_dir)
        model_id = f"local-dir:{args.biomedclip_local_dir}"
    else:
        model, _, _ = open_clip.create_model_and_transforms(args.biomedclip_model)
        tokenizer = open_clip.get_tokenizer(args.biomedclip_model)
        model = model.eval().to(device)
        model_id = args.biomedclip_model
    embeddings = []
    with torch.no_grad():
        for start in range(0, len(texts), args.batch_size):
            batch = texts[start : start + args.batch_size]
            tokenized = tokenizer(batch)
            if isinstance(tokenized, dict):
                tokenized = {key: value.to(device) for key, value in tokenized.items()}
                try:
                    features = model.encode_text(tokenized, normalize=args.normalize)
                except TypeError:
                    features = model.encode_text(tokenized)
                    features = _l2_normalize(features.float(), args.normalize)
            else:
                tokenized = tokenized.to(device)
                try:
                    features = model.encode_text(tokenized, normalize=args.normalize)
                except TypeError:
                    features = model.encode_text(tokenized)
                    features = _l2_normalize(features.float(), args.normalize)
            features = features.float()
            features = _l2_normalize(features, args.normalize)
            embeddings.append(features[:, None, :].detach().cpu().numpy().astype("float32"))
    arr = np.concatenate(embeddings, axis=0) if embeddings else np.zeros((0, 1, 512), dtype="float32")
    meta = {
        "encoder": "biomedclip",
        "model": model_id,
        "pool": "open_clip_encode_text",
        "normalize": bool(args.normalize),
    }
    return arr, meta


def write_cache(args, texts: list[str], embeddings: np.ndarray, encoder_meta: dict) -> None:
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    keys = np.asarray([text_key(text) for text in texts])
    temp_output = output.with_suffix(output.suffix + ".tmp")
    with temp_output.open("wb") as handle:
        np.savez_compressed(handle, keys=keys, texts=np.asarray(texts), embeddings=embeddings.astype("float32"))
    temp_output.replace(output)
    meta = {
        **encoder_meta,
        **cache_expectations(
            args.manifest,
            texts,
            args.include_lcaug_variants,
            args.include_empty_control,
        ),
        "encoder_weights_sha256": encoder_weights_fingerprint(args, encoder_meta),
        "manifest": [str(Path(item).resolve()) for item in args.manifest],
        "texts": len(texts),
        "include_lcaug_variants": bool(args.include_lcaug_variants),
        "embedding_shape": list(embeddings.shape),
        "output": str(output),
    }
    meta_path = output.with_suffix(output.suffix + ".json")
    temp_meta = meta_path.with_suffix(meta_path.suffix + ".tmp")
    temp_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    temp_meta.replace(meta_path)
    print(json.dumps(meta, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", choices=["pubmedbert", "biomedclip"], required=True)
    parser.add_argument("--manifest", nargs="+", default=[str(path) for path in DEFAULT_MANIFESTS])
    parser.add_argument("--output", required=True)
    parser.add_argument("--include-lcaug-variants", action="store_true")
    parser.add_argument("--include-empty-control", action="store_true")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--pubmedbert-model", default=PUBMEDBERT_ID)
    parser.add_argument("--pubmedbert-pool", choices=["mean", "cls"], default="mean")
    parser.add_argument("--biomedclip-model", default=BIOMEDCLIP_ID)
    parser.add_argument(
        "--biomedclip-local-dir",
        default="",
        help="Offline BiomedCLIP weights dir (symlinked into open_clip HF cache).",
    )
    args = parser.parse_args()

    texts = unique_texts_from_manifest(args.manifest, include_lcaug_variants=args.include_lcaug_variants)
    if args.include_empty_control:
        texts = sorted(set(texts) | {""})
    if args.encoder == "pubmedbert":
        embeddings, meta = build_pubmedbert_cache(args, texts)
    else:
        embeddings, meta = build_biomedclip_cache(args, texts)
    write_cache(args, texts, embeddings, meta)


if __name__ == "__main__":
    main()
