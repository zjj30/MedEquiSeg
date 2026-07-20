#!/usr/bin/env python3
"""Plug-in text encoders for CausalCLIPSeg RN50 adapter."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from biomedclip_offline import load_open_clip_biomedclip

BIOMEDCLIP_ID = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


class ClipRN50TextEncoder(nn.Module):
    """Default CLIP RN50 BPE text tower (token ids in, embeddings out)."""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, texts_or_tokens, device=None):
        if not torch.is_tensor(texts_or_tokens):
            raise TypeError("ClipRN50TextEncoder expects BPE token tensor.")
        tokens = texts_or_tokens
        if device is not None and tokens.device != device:
            tokens = tokens.to(device)
        return self.backbone.encode_text(tokens)


class BiomedCLIPTextEncoder(nn.Module):
    """Frozen pooled BiomedCLIP embedding adapter projected to RN50 dimensions."""

    def __init__(
        self,
        token_dim: int = 512,
        state_dim: int = 1024,
        word_len: int = 16,
        mode: str = "biomedclip_frozen",
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
    ):
        super().__init__()
        self.token_dim = int(token_dim)
        self.state_dim = int(state_dim)
        self.word_len = int(word_len)
        self.mode = str(mode)
        self.lora_rank = int(lora_rank)
        self.lora_alpha = float(lora_alpha)
        self.source_dim = 512
        self._open_clip_model = None
        self._tokenizer = None

        self.token_proj = nn.Linear(self.source_dim, self.token_dim, bias=False)
        self.state_proj = nn.Linear(self.source_dim, self.state_dim, bias=False)
        self.token_expand = nn.Parameter(torch.zeros(1, self.word_len, self.token_dim))
        nn.init.normal_(self.token_expand, std=0.02)

        if self.mode == "biomedclip_lora":
            self.lora_a = nn.Linear(self.source_dim, self.lora_rank, bias=False)
            self.lora_b = nn.Linear(self.lora_rank, self.source_dim, bias=False)
            nn.init.kaiming_uniform_(self.lora_a.weight, a=5**0.5)
            nn.init.zeros_(self.lora_b.weight)
        else:
            self.lora_a = None
            self.lora_b = None

    def _load_open_clip(self, device):
        if self._open_clip_model is not None:
            return
        try:
            self._open_clip_model, self._tokenizer = load_open_clip_biomedclip(device)
        except Exception as exc:
            raise RuntimeError(
                "BiomedCLIP text encoder requires open_clip_torch and local weights. "
                "Set BIOMEDCLIP_LOCAL_DIR or pass --text-encoder-cache."
            ) from exc

    def _encode_text_features(self, texts: Sequence[str], device) -> torch.Tensor:
        self._load_open_clip(device)
        assert self._tokenizer is not None and self._open_clip_model is not None
        tokenized = self._tokenizer(list(texts))
        if isinstance(tokenized, dict):
            tokenized = {key: value.to(device) for key, value in tokenized.items()}
            try:
                features = self._open_clip_model.encode_text(tokenized, normalize=True)
            except TypeError:
                features = self._open_clip_model.encode_text(tokenized)
                features = F.normalize(features.float(), dim=-1)
        else:
            tokenized = tokenized.to(device)
            try:
                features = self._open_clip_model.encode_text(tokenized, normalize=True)
            except TypeError:
                features = self._open_clip_model.encode_text(tokenized)
                features = F.normalize(features.float(), dim=-1)
        return features.float()

    def _project_features(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.lora_a is not None and self.lora_b is not None:
            delta = self.lora_b(self.lora_a(features)) * (self.lora_alpha / max(self.lora_rank, 1))
            features = features + delta
        state = self.state_proj(features)
        token_base = self.token_proj(features)
        word_tokens = token_base.unsqueeze(1) + self.token_expand
        return word_tokens, state

    def forward_from_cache(self, cache_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # cache_batch: [B, T, D] or [B, D]
        if cache_batch.ndim == 2:
            features = cache_batch
        else:
            features = cache_batch.mean(dim=1)
        return self._project_features(features)

    def forward(self, texts_or_tokens, device=None):
        if torch.is_tensor(texts_or_tokens):
            return self.forward_from_cache(texts_or_tokens)
        if not isinstance(texts_or_tokens, (list, tuple)):
            texts_or_tokens = [texts_or_tokens]
        dev = device if device is not None else next(self.parameters()).device
        features = self._encode_text_features(texts_or_tokens, dev)
        return self._project_features(features)


TEXT_ENCODER_PLUGINS = {
    "clip_rn50": "clip_rn50",
    "biomedclip_frozen": "biomedclip_frozen",
    "biomedclip_lora": "biomedclip_lora",
}


def build_text_encoder_plugin(name: str, backbone, token_dim: int, state_dim: int, word_len: int = 16):
    key = (name or "clip_rn50").strip()
    if key == "clip_rn50":
        return ClipRN50TextEncoder(backbone)
    if key == "biomedclip_frozen":
        return BiomedCLIPTextEncoder(
            token_dim=token_dim,
            state_dim=state_dim,
            word_len=word_len,
            mode="biomedclip_frozen",
        )
    if key == "biomedclip_lora":
        return BiomedCLIPTextEncoder(
            token_dim=token_dim,
            state_dim=state_dim,
            word_len=word_len,
            mode="biomedclip_lora",
        )
    raise ValueError(f"Unknown text encoder plugin: {name!r}")
