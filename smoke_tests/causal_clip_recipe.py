#!/usr/bin/env python3
"""Composable training recipes for CausalCLIPSeg RN50 incremental ablations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CausalClipRecipe:
    name: str
    text_encoder: str
    loss_mode: str | None
    dice_loss_weight: float
    boundary_loss_weight: float
    conv_plugin: str
    atconv_layers: int
    description: str


RECIPES: dict[str, CausalClipRecipe] = {
    "default": CausalClipRecipe(
        name="default",
        text_encoder="clip_rn50",
        loss_mode=None,
        dice_loss_weight=0.5,
        boundary_loss_weight=0.05,
        conv_plugin="standard",
        atconv_layers=0,
        description="Frozen candidate: CLIP RN50 text + DiceBCE (loss from model name).",
    ),
    "boundary_only": CausalClipRecipe(
        name="boundary_only",
        text_encoder="clip_rn50",
        loss_mode="bce_dice_boundary",
        dice_loss_weight=0.5,
        boundary_loss_weight=0.05,
        conv_plugin="standard",
        atconv_layers=0,
        description="CLIP RN50 text + boundary-aware DiceBCE.",
    ),
    "biomed_noaug": CausalClipRecipe(
        name="biomed_noaug",
        text_encoder="biomedclip_frozen",
        loss_mode="bce_dice",
        dice_loss_weight=0.5,
        boundary_loss_weight=0.05,
        conv_plugin="standard",
        atconv_layers=0,
        description="Frozen pooled BiomedCLIP embedding adapter + DiceBCE (isolate text adapter).",
    ),
    "biomed_lcaug": CausalClipRecipe(
        name="biomed_lcaug",
        text_encoder="biomedclip_frozen",
        loss_mode="bce_dice",
        dice_loss_weight=0.5,
        boundary_loss_weight=0.05,
        conv_plugin="standard",
        atconv_layers=0,
        description="Frozen pooled BiomedCLIP embedding adapter + DiceBCE + external LCAug.",
    ),
    "biomed_boundary": CausalClipRecipe(
        name="biomed_boundary",
        text_encoder="biomedclip_frozen",
        loss_mode="bce_dice_boundary",
        dice_loss_weight=0.5,
        boundary_loss_weight=0.05,
        conv_plugin="standard",
        atconv_layers=0,
        description="BiomedCLIP frozen + boundary-aware DiceBCE + external LCAug.",
    ),
    "biomed_lora": CausalClipRecipe(
        name="biomed_lora",
        text_encoder="biomedclip_lora",
        loss_mode="bce_dice_boundary",
        dice_loss_weight=0.5,
        boundary_loss_weight=0.05,
        conv_plugin="standard",
        atconv_layers=0,
        description="BiomedCLIP with trainable projection LoRA + boundary DiceBCE.",
    ),
    "atconv4": CausalClipRecipe(
        name="atconv4",
        text_encoder="clip_rn50",
        loss_mode=None,
        dice_loss_weight=0.5,
        boundary_loss_weight=0.05,
        conv_plugin="atconv",
        atconv_layers=4,
        description="Replace 4 decoder same-channel 3x3 convs with ATConv (neck+projector).",
    ),
    "default_atconv4": CausalClipRecipe(
        name="default_atconv4",
        text_encoder="clip_rn50",
        loss_mode=None,
        dice_loss_weight=0.5,
        boundary_loss_weight=0.05,
        conv_plugin="atconv",
        atconv_layers=4,
        description="Frozen candidate stack + ATConv plug-in on 4 decoder conv layers.",
    ),
    "biomed_lcaug_atconv4": CausalClipRecipe(
        name="biomed_lcaug_atconv4",
        text_encoder="biomedclip_frozen",
        loss_mode="bce_dice",
        dice_loss_weight=0.5,
        boundary_loss_weight=0.05,
        conv_plugin="atconv",
        atconv_layers=4,
        description="BiomedCLIP frozen + LCAug + ATConv on 4 decoder conv layers.",
    ),
    "biomed_lcaug_v2_atconv4": CausalClipRecipe(
        name="biomed_lcaug_v2_atconv4",
        text_encoder="biomedclip_frozen",
        loss_mode="bce_dice",
        dice_loss_weight=0.5,
        boundary_loss_weight=0.05,
        conv_plugin="atconv",
        atconv_layers=4,
        description="BiomedCLIP frozen + LCAugv2 tiered MedCLIPSeg aug + ATConv on 4 decoder layers.",
    ),
}

PHASE1_RUNS = {
    "R0": {"recipe": "default", "augmentation": "lcaug_hflip_dataset"},
    "R1": {"recipe": "boundary_only", "augmentation": "lcaug_hflip_dataset"},
    "R2": {"recipe": "biomed_noaug", "augmentation": "none"},
    "R3": {"recipe": "biomed_lcaug", "augmentation": "lcaug_hflip_dataset"},
    "R4": {"recipe": "biomed_boundary", "augmentation": "lcaug_hflip_dataset"},
    "R5": {"recipe": "biomed_lora", "augmentation": "lcaug_hflip_dataset"},
    "R6": {"recipe": "default_atconv4", "augmentation": "lcaug_hflip_dataset"},
    "R7": {"recipe": "biomed_lcaug_atconv4", "augmentation": "lcaug_hflip_dataset"},
    "R8": {"recipe": "biomed_lcaug_v2_atconv4", "augmentation": "lcaug_v2_medclipseg_dataset"},
    "R8NR": {
        "recipe": "biomed_lcaug_v2_atconv4",
        "augmentation": "lcaug_v2_medclipseg_no_text_rewrite_dataset",
    },
}


def recipe_names() -> list[str]:
    return sorted(RECIPES)


def get_recipe(name: str) -> CausalClipRecipe:
    key = (name or "default").strip()
    if key not in RECIPES:
        raise ValueError(f"Unknown causal recipe: {name!r}. Choices: {', '.join(recipe_names())}")
    return RECIPES[key]


def resolve_conv_plugin(recipe: CausalClipRecipe, args: Any) -> tuple[str, int]:
    if getattr(args, "enable_atconv", False):
        layers = int(getattr(args, "atconv_layers", 4) or 4)
        return "atconv", max(1, layers)
    if recipe.conv_plugin == "atconv" and recipe.atconv_layers > 0:
        return "atconv", int(recipe.atconv_layers)
    return "standard", 0


def apply_recipe_to_args(args: Any) -> CausalClipRecipe:
    recipe = get_recipe(getattr(args, "causal_recipe", "default") or "default")
    if getattr(args, "enable_boundary_loss", False) and recipe.loss_mode is None:
        args.loss_mode = "bce_dice_boundary"
    elif recipe.loss_mode is not None:
        args.loss_mode = recipe.loss_mode
    if recipe.dice_loss_weight is not None:
        args.dice_loss_weight = recipe.dice_loss_weight
    if recipe.boundary_loss_weight is not None:
        args.boundary_loss_weight = recipe.boundary_loss_weight
    return recipe


def recipe_meta(recipe: CausalClipRecipe, text_encoder_cache: str = "", atconv_targets: str = "") -> dict:
    return {
        "causal_recipe": recipe.name,
        "text_encoder": recipe.text_encoder,
        "text_encoder_cache": text_encoder_cache or "",
        "loss_recipe": recipe.loss_mode or "from_model_name",
        "conv_plugin": recipe.conv_plugin,
        "atconv_layers": recipe.atconv_layers,
        "atconv_targets": atconv_targets,
    }
