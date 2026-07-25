#!/usr/bin/env python3
import random
import re
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageEnhance
from scipy import ndimage

from lcaug_v2_direction import transform_direction_text_v2


DIRECTION_TO_VECTOR = {
    "upper middle": (0, -1),
    "upper left": (-1, -1),
    "upper right": (1, -1),
    "lower middle": (0, 1),
    "lower left": (-1, 1),
    "lower right": (1, 1),
    "middle left": (-1, 0),
    "middle right": (1, 0),
    "top left": (-1, -1),
    "top right": (1, -1),
    "bottom left": (-1, 1),
    "bottom right": (1, 1),
    "upper": (0, -1),
    "lower": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
    "top": (0, -1),
    "bottom": (0, 1),
    "central": (0, 0),
    "center": (0, 0),
    "centre": (0, 0),
    "middle": (0, 0),
}
PHRASES = sorted(DIRECTION_TO_VECTOR, key=len, reverse=True)
DIRECTION_PATTERN = re.compile(r"\b(" + "|".join(re.escape(p) for p in PHRASES) + r")\b", re.IGNORECASE)


def _vector_to_phrase(vector, original):
    x, y = vector
    original_lower = original.lower()
    if x == 0 and y == 0:
        if original_lower == "centre":
            return "centre"
        if original_lower == "central":
            return "central"
        if original_lower == "middle":
            return "middle"
        return "center"
    vertical_negative = "upper" if ("upper" in original_lower or "lower" in original_lower) else "top"
    vertical_positive = "lower" if ("upper" in original_lower or "lower" in original_lower) else "bottom"
    parts = []
    if y < 0:
        parts.append(vertical_negative)
    elif y > 0:
        parts.append(vertical_positive)
    elif "middle" in original_lower and x != 0:
        parts.append("middle")
    if x < 0:
        parts.append("left")
    elif x > 0:
        parts.append("right")
    return " ".join(parts)


def _apply_vector_op(vector, op):
    x, y = vector
    if op == "hflip":
        return -x, y
    if op == "vflip":
        return x, -y
    if op == "rot90_ccw":
        return y, -x
    raise ValueError(op)


def transform_direction_text(text, op, times=1):
    def repl(match):
        original = match.group(0)
        vector = DIRECTION_TO_VECTOR[original.lower()]
        for _ in range(times):
            vector = _apply_vector_op(vector, op)
        return _vector_to_phrase(vector, original)

    return DIRECTION_PATTERN.sub(repl, text)


TEXT_GEO_PROFILES = {
    "cvc": {
        "hflip_p": 0.45,
        "vflip_p": 0.15,
        "rot90_p": 0.25,
        "affine_p": 0.55,
        "rotate_deg": 8.0,
        "shift_frac": 0.035,
        "zoom_p": 0.30,
        "zoom_max": 1.10,
        "color_p": 0.60,
        "color_jitter": 0.08,
        "gamma_p": 0.25,
        "gamma_range": (0.90, 1.10),
        "noise_p": 0.20,
        "noise_std": 0.004,
    },
    "glas": {
        "hflip_p": 0.45,
        "vflip_p": 0.35,
        "rot90_p": 0.35,
        "affine_p": 0.55,
        "rotate_deg": 10.0,
        "shift_frac": 0.040,
        "zoom_p": 0.35,
        "zoom_max": 1.12,
        "color_p": 0.45,
        "color_jitter": 0.04,
        "gamma_p": 0.15,
        "gamma_range": (0.96, 1.06),
        "noise_p": 0.12,
        "noise_std": 0.002,
    },
    "busi": {
        "hflip_p": 0.35,
        "vflip_p": 0.10,
        "rot90_p": 0.12,
        "affine_p": 0.45,
        "rotate_deg": 5.0,
        "shift_frac": 0.025,
        "zoom_p": 0.25,
        "zoom_max": 1.07,
        "color_p": 0.45,
        "color_jitter": 0.05,
        "gamma_p": 0.25,
        "gamma_range": (0.90, 1.12),
        "noise_p": 0.25,
        "noise_std": 0.006,
    },
}


LCAUG_V2_BUSI_PROFILE = dict(TEXT_GEO_PROFILES["busi"])
LCAUG_V2_BUSI_PROFILE["affine_p"] = 0.15


HFLIP_LIGHT_PROFILES = {
    "cvc": {
        "hflip_p": 0.35,
        "affine_p": 0.55,
        "rotate_deg": 6.0,
        "shift_frac": 0.030,
        "zoom_p": 0.25,
        "zoom_max": 1.08,
        "color_p": 0.55,
        "color_jitter": 0.06,
        "gamma_p": 0.20,
        "gamma_range": (0.92, 1.08),
        "noise_p": 0.15,
        "noise_std": 0.003,
    },
    "glas": {
        "hflip_p": 0.35,
        "affine_p": 0.50,
        "rotate_deg": 8.0,
        "shift_frac": 0.035,
        "zoom_p": 0.28,
        "zoom_max": 1.10,
        "color_p": 0.35,
        "color_jitter": 0.035,
        "gamma_p": 0.10,
        "gamma_range": (0.97, 1.05),
        "noise_p": 0.08,
        "noise_std": 0.002,
    },
    "busi": {
        "hflip_p": 0.25,
        "affine_p": 0.40,
        "rotate_deg": 4.0,
        "shift_frac": 0.020,
        "zoom_p": 0.20,
        "zoom_max": 1.06,
        "color_p": 0.35,
        "color_jitter": 0.04,
        "gamma_p": 0.18,
        "gamma_range": (0.92, 1.10),
        "noise_p": 0.18,
        "noise_std": 0.004,
    },
}


def _profile(dataset):
    return TEXT_GEO_PROFILES.get(dataset, TEXT_GEO_PROFILES["cvc"])


def _hflip_light_profile(dataset):
    return HFLIP_LIGHT_PROFILES.get(dataset, HFLIP_LIGHT_PROFILES["cvc"])


def _apply_gamma(img, gamma_range, rng=None):
    source = rng or random
    gamma = source.uniform(float(gamma_range[0]), float(gamma_range[1]))
    arr = np.asarray(img).astype("float32") / 255.0
    arr = np.power(np.clip(arr, 0.0, 1.0), gamma)
    return Image.fromarray((arr * 255.0).clip(0, 255).astype("uint8"))


def _add_noise(img, noise_std, seed=None):
    arr = np.asarray(img).astype("float32") / 255.0
    if seed is None:
        noise = np.random.normal(0.0, float(noise_std), size=arr.shape)
    else:
        noise = np.random.default_rng(seed).normal(0.0, float(noise_std), size=arr.shape)
    arr = arr + noise.astype("float32")
    return Image.fromarray((arr.clip(0.0, 1.0) * 255.0).astype("uint8"))


def apply_text_geo_dataset(img, mask, text, dataset, image_size):
    profile = _profile(dataset)

    if random.random() < profile["hflip_p"]:
        img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        text = transform_direction_text(text, "hflip")

    if random.random() < profile["vflip_p"]:
        img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        mask = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        text = transform_direction_text(text, "vflip")

    if random.random() < profile["rot90_p"]:
        k = random.randint(1, 3)
        img = img.rotate(90 * k, resample=Image.Resampling.BILINEAR)
        mask = mask.rotate(90 * k, resample=Image.Resampling.NEAREST)
        text = transform_direction_text(text, "rot90_ccw", times=k)

    if random.random() < profile["affine_p"]:
        angle = random.uniform(-profile["rotate_deg"], profile["rotate_deg"])
        max_shift = int(round(image_size * profile["shift_frac"]))
        shift = (random.randint(-max_shift, max_shift), random.randint(-max_shift, max_shift))
        img = img.rotate(angle, resample=Image.Resampling.BILINEAR, translate=shift, fillcolor=(0, 0, 0))
        mask = mask.rotate(angle, resample=Image.Resampling.NEAREST, translate=shift, fillcolor=0)

    if random.random() < profile["zoom_p"]:
        zoom = random.uniform(1.0, profile["zoom_max"])
        crop_size = max(16, int(round(image_size / zoom)))
        left = random.randint(0, image_size - crop_size)
        top = random.randint(0, image_size - crop_size)
        box = (left, top, left + crop_size, top + crop_size)
        img = img.crop(box).resize((image_size, image_size), Image.BILINEAR)
        mask = mask.crop(box).resize((image_size, image_size), Image.NEAREST)

    if random.random() < profile["color_p"]:
        jitter = profile["color_jitter"]
        img = ImageEnhance.Brightness(img).enhance(random.uniform(1.0 - jitter, 1.0 + jitter))
        img = ImageEnhance.Contrast(img).enhance(random.uniform(1.0 - jitter, 1.0 + jitter))
        img = ImageEnhance.Color(img).enhance(random.uniform(1.0 - jitter, 1.0 + jitter))

    if random.random() < profile["gamma_p"]:
        img = _apply_gamma(img, profile["gamma_range"])

    if random.random() < profile["noise_p"]:
        img = _add_noise(img, profile["noise_std"])

    return img, mask, text


def _apply_lcaug_hflip_dataset_impl(img, mask, text, dataset, image_size, rewrite_text):
    profile = _hflip_light_profile(dataset)

    if random.random() < profile["hflip_p"]:
        img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if rewrite_text:
            text = transform_direction_text(text, "hflip")

    if random.random() < profile["affine_p"]:
        angle = random.uniform(-profile["rotate_deg"], profile["rotate_deg"])
        max_shift = int(round(image_size * profile["shift_frac"]))
        shift = (random.randint(-max_shift, max_shift), random.randint(-max_shift, max_shift))
        img = img.rotate(angle, resample=Image.Resampling.BILINEAR, translate=shift, fillcolor=(0, 0, 0))
        mask = mask.rotate(angle, resample=Image.Resampling.NEAREST, translate=shift, fillcolor=0)

    if random.random() < profile["zoom_p"]:
        zoom = random.uniform(1.0, profile["zoom_max"])
        crop_size = max(16, int(round(image_size / zoom)))
        left = random.randint(0, image_size - crop_size)
        top = random.randint(0, image_size - crop_size)
        box = (left, top, left + crop_size, top + crop_size)
        img = img.crop(box).resize((image_size, image_size), Image.BILINEAR)
        mask = mask.crop(box).resize((image_size, image_size), Image.NEAREST)

    if random.random() < profile["color_p"]:
        jitter = profile["color_jitter"]
        img = ImageEnhance.Brightness(img).enhance(random.uniform(1.0 - jitter, 1.0 + jitter))
        img = ImageEnhance.Contrast(img).enhance(random.uniform(1.0 - jitter, 1.0 + jitter))
        img = ImageEnhance.Color(img).enhance(random.uniform(1.0 - jitter, 1.0 + jitter))

    if random.random() < profile["gamma_p"]:
        img = _apply_gamma(img, profile["gamma_range"])

    if random.random() < profile["noise_p"]:
        img = _add_noise(img, profile["noise_std"])

    return img, mask, text


def apply_lcaug_hflip_dataset(img, mask, text, dataset, image_size):
    return _apply_lcaug_hflip_dataset_impl(img, mask, text, dataset, image_size, rewrite_text=True)


def apply_lcaug_hflip_no_text_rewrite_dataset(img, mask, text, dataset, image_size):
    return _apply_lcaug_hflip_dataset_impl(img, mask, text, dataset, image_size, rewrite_text=False)


@dataclass(frozen=True)
class LCAugPlan:
    """A deterministic transform plan shared by rewrite/no-rewrite controls."""

    hflip: bool
    vflip: bool
    rot90_k: int
    brightness: float | None
    contrast: float | None
    color: float | None
    gamma: float | None
    noise_seed: int | None
    affine_angle: float | None
    shift_x_frac: float | None
    shift_y_frac: float | None
    zoom: float | None
    zoom_anchor_x: float | None
    zoom_anchor_y: float | None


def sample_lcaug_plan(profile, rng=None, *, multi_geometry=False, include_affine_zoom=False):
    source = rng or random
    hflip = source.random() < profile["hflip_p"]
    vflip = bool(multi_geometry and source.random() < profile.get("vflip_p", 0.0))
    rot90_k = source.randint(1, 3) if multi_geometry and source.random() < profile.get("rot90_p", 0.0) else 0
    if source.random() < profile["color_p"]:
        jitter = profile["color_jitter"]
        brightness = source.uniform(1.0 - jitter, 1.0 + jitter)
        contrast = source.uniform(1.0 - jitter, 1.0 + jitter)
        color = source.uniform(1.0 - jitter, 1.0 + jitter)
    else:
        brightness = contrast = color = None
    gamma = source.uniform(*profile["gamma_range"]) if source.random() < profile["gamma_p"] else None
    noise_seed = source.randint(0, 2**32 - 1) if source.random() < profile["noise_p"] else None
    if include_affine_zoom and source.random() < profile.get("affine_p", 0.0):
        affine_angle = source.uniform(-profile["rotate_deg"], profile["rotate_deg"])
        shift_x_frac = source.uniform(-profile["shift_frac"], profile["shift_frac"])
        shift_y_frac = source.uniform(-profile["shift_frac"], profile["shift_frac"])
    else:
        affine_angle = shift_x_frac = shift_y_frac = None
    if include_affine_zoom and source.random() < profile.get("zoom_p", 0.0):
        zoom = source.uniform(1.0, profile["zoom_max"])
        zoom_anchor_x = source.random()
        zoom_anchor_y = source.random()
    else:
        zoom = zoom_anchor_x = zoom_anchor_y = None
    return LCAugPlan(
        hflip,
        vflip,
        rot90_k,
        brightness,
        contrast,
        color,
        gamma,
        noise_seed,
        affine_angle,
        shift_x_frac,
        shift_y_frac,
        zoom,
        zoom_anchor_x,
        zoom_anchor_y,
    )


def apply_lcaug_plan(img, mask, text, profile, plan, *, rewrite_text):
    if plan.hflip:
        img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if rewrite_text:
            text = transform_direction_text_v2(text, "hflip")
    if plan.vflip:
        img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        mask = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        if rewrite_text:
            text = transform_direction_text_v2(text, "vflip")
    if plan.rot90_k:
        img = img.rotate(90 * plan.rot90_k, resample=Image.Resampling.BILINEAR)
        mask = mask.rotate(90 * plan.rot90_k, resample=Image.Resampling.NEAREST)
        if rewrite_text:
            text = transform_direction_text_v2(text, "rot90_ccw", times=plan.rot90_k)
    if plan.affine_angle is not None:
        width, height = img.size
        shift = (int(round(width * plan.shift_x_frac)), int(round(height * plan.shift_y_frac)))
        img = img.rotate(
            plan.affine_angle,
            resample=Image.Resampling.BILINEAR,
            translate=shift,
            fillcolor=(0, 0, 0),
        )
        mask = mask.rotate(
            plan.affine_angle,
            resample=Image.Resampling.NEAREST,
            translate=shift,
            fillcolor=0,
        )
    if plan.zoom is not None:
        width, height = img.size
        crop_width = max(16, int(round(width / plan.zoom)))
        crop_height = max(16, int(round(height / plan.zoom)))
        left = int(round((width - crop_width) * plan.zoom_anchor_x))
        top = int(round((height - crop_height) * plan.zoom_anchor_y))
        box = (left, top, left + crop_width, top + crop_height)
        img = img.crop(box).resize((width, height), Image.Resampling.BILINEAR)
        mask = mask.crop(box).resize((width, height), Image.Resampling.NEAREST)
    if plan.brightness is not None:
        img = ImageEnhance.Brightness(img).enhance(plan.brightness)
        img = ImageEnhance.Contrast(img).enhance(plan.contrast)
        img = ImageEnhance.Color(img).enhance(plan.color)
    if plan.gamma is not None:
        arr = np.asarray(img).astype("float32") / 255.0
        img = Image.fromarray((np.power(np.clip(arr, 0.0, 1.0), plan.gamma) * 255.0).clip(0, 255).astype("uint8"))
    if plan.noise_seed is not None:
        img = _add_noise(img, profile["noise_std"], seed=plan.noise_seed)
    return img, mask, text


def _apply_lcaug_v2_impl(
    img,
    mask,
    text,
    dataset,
    rewrite_text,
    rng,
    *,
    multi_geometry,
    include_affine_zoom=False,
):
    profile = LCAUG_V2_BUSI_PROFILE if multi_geometry else _hflip_light_profile(dataset)
    plan = sample_lcaug_plan(
        profile,
        rng=rng,
        multi_geometry=multi_geometry,
        include_affine_zoom=include_affine_zoom,
    )
    return apply_lcaug_plan(img, mask, text, profile, plan, rewrite_text=rewrite_text)


def apply_lcaug_v2_hflip_dataset(img, mask, text, dataset, image_size, rng=None):
    return _apply_lcaug_v2_impl(img, mask, text, dataset, True, rng, multi_geometry=False)


def apply_lcaug_v2_hflip_no_text_rewrite_dataset(img, mask, text, dataset, image_size, rng=None):
    return _apply_lcaug_v2_impl(img, mask, text, dataset, False, rng, multi_geometry=False)


def apply_lcaug_v2_busi_dataset(img, mask, text, dataset, image_size, rng=None):
    return _apply_lcaug_v2_impl(img, mask, text, dataset, True, rng, multi_geometry=True)


def apply_lcaug_v2_busi_no_text_rewrite_dataset(img, mask, text, dataset, image_size, rng=None):
    return _apply_lcaug_v2_impl(img, mask, text, dataset, False, rng, multi_geometry=True)


def apply_lcaug_v2_dynamic_shared_plan_dataset(img, mask, text, dataset, image_size, rng=None):
    """R11 DSP-LCAug: online multi-geometry plus mild affine/translation/zoom."""
    return _apply_lcaug_v2_impl(
        img,
        mask,
        text,
        dataset,
        True,
        rng,
        multi_geometry=True,
        include_affine_zoom=True,
    )


def _coarse_location_from_centroid(cx, cy, width, height):
    """Map an image-frame centroid to the 3x3 location vocabulary used by prompts."""
    x_ratio = float(cx) / max(1, width - 1)
    y_ratio = float(cy) / max(1, height - 1)
    horizontal = "left" if x_ratio < 1.0 / 3.0 else "right" if x_ratio > 2.0 / 3.0 else ""
    vertical = "top" if y_ratio < 1.0 / 3.0 else "bottom" if y_ratio > 2.0 / 3.0 else ""
    if vertical and horizontal:
        return f"{vertical} {horizontal}"
    return vertical or horizontal or "center"


def recompute_direction_text_from_mask(text, mask):
    """Recompute spatial phrases from transformed-mask component centroids.

    This is a benchmark-only diagnostic for prompts whose positions were already
    derived from masks. If the number of spatial phrases and connected components
    cannot be matched unambiguously, the input text is returned unchanged.
    """
    mask_array = np.asarray(mask)
    if mask_array.ndim == 3:
        mask_array = mask_array.max(axis=2)
    foreground = mask_array > 0
    if not foreground.any():
        return text

    matches = list(DIRECTION_PATTERN.finditer(text))
    if not matches:
        return text

    labels, component_count = ndimage.label(foreground)
    if component_count != len(matches):
        return text

    height, width = foreground.shape
    locations = []
    for component_id in range(1, component_count + 1):
        cy, cx = ndimage.center_of_mass(foreground, labels, component_id)
        locations.append((_coarse_location_from_centroid(cx, cy, width, height), cy, cx))
    locations.sort(key=lambda item: (item[1], item[2]))

    parts = []
    cursor = 0
    for match, (location, _, _) in zip(matches, locations):
        parts.append(text[cursor : match.start()])
        parts.append(location)
        cursor = match.end()
    parts.append(text[cursor:])
    return "".join(parts)


def apply_lcaug_v2_dynamic_shared_plan_recompute_location_dataset(
    img,
    mask,
    text,
    dataset,
    image_size,
    rng=None,
):
    """R11-LR: R11 plus post-transform location recomputation.

    Recalculation is activated only when affine/translation or zoom is sampled;
    flip and right-angle rotation behavior therefore remains identical to R11.
    """
    profile = LCAUG_V2_BUSI_PROFILE
    plan = sample_lcaug_plan(profile, rng=rng, multi_geometry=True, include_affine_zoom=True)
    img, mask, text = apply_lcaug_plan(img, mask, text, profile, plan, rewrite_text=True)
    if plan.affine_angle is not None or plan.zoom is not None:
        text = recompute_direction_text_from_mask(text, mask)
    return img, mask, text


def apply_lcaug_v2_dynamic_shared_plan_no_text_rewrite_dataset(
    img,
    mask,
    text,
    dataset,
    image_size,
    rng=None,
):
    return _apply_lcaug_v2_impl(
        img,
        mask,
        text,
        dataset,
        False,
        rng,
        multi_geometry=True,
        include_affine_zoom=True,
    )


LCAUG_V2_MULTI_GEOMETRY_DATASETS = {
    "busi",
    "medclipseg_busi",
    "medclipseg_busbra",
    "medclipseg_brisc",
}


def apply_lcaug_v2_medclipseg_dataset(img, mask, text, dataset, image_size, rng=None):
    """Protocol V3 routing: ClinicDB/COVID19 are hflip-only."""
    slug = str(dataset).lower()
    if slug in LCAUG_V2_MULTI_GEOMETRY_DATASETS:
        return apply_lcaug_v2_busi_dataset(img, mask, text, dataset, image_size, rng=rng)
    return apply_lcaug_v2_hflip_dataset(img, mask, text, dataset, image_size, rng=rng)


def apply_lcaug_v2_medclipseg_no_text_rewrite_dataset(img, mask, text, dataset, image_size, rng=None):
    slug = str(dataset).lower()
    if slug in LCAUG_V2_MULTI_GEOMETRY_DATASETS:
        return apply_lcaug_v2_busi_no_text_rewrite_dataset(img, mask, text, dataset, image_size, rng=rng)
    return apply_lcaug_v2_hflip_no_text_rewrite_dataset(img, mask, text, dataset, image_size, rng=rng)


def apply_dataset_policy_v1(img, mask, text, dataset, image_size):
    """Dataset-aware policy derived from fixed-split public gates.

    BUSI benefits most from full text-synchronized geometry, CVC from the
    constrained hflip LCAug profile, while GlaS is the current caution case and
    is left unaugmented.
    """
    dataset = str(dataset).lower()
    if dataset == "busi":
        return apply_text_geo_dataset(img, mask, text, dataset, image_size)
    if dataset == "cvc":
        return apply_lcaug_hflip_dataset(img, mask, text, dataset, image_size)
    if dataset == "glas":
        return img, mask, text
    return apply_lcaug_hflip_dataset(img, mask, text, dataset, image_size)


GLAS_APPEARANCE_PROFILE = {
    "color_jitter": 0.10,
    "gamma_range": (0.90, 1.12),
    "gamma_p": 0.45,
    "noise_std": 0.003,
    "noise_p": 0.25,
    "color_p": 0.75,
}


def apply_glas_appearance_dataset(img, mask, text, dataset, image_size):
    profile = GLAS_APPEARANCE_PROFILE
    if random.random() < profile["color_p"]:
        jitter = profile["color_jitter"]
        img = ImageEnhance.Brightness(img).enhance(random.uniform(1.0 - jitter, 1.0 + jitter))
        img = ImageEnhance.Contrast(img).enhance(random.uniform(1.0 - jitter, 1.0 + jitter))
        img = ImageEnhance.Color(img).enhance(random.uniform(1.0 - jitter, 1.0 + jitter))
    if random.random() < profile["gamma_p"]:
        img = _apply_gamma(img, profile["gamma_range"])
    if random.random() < profile["noise_p"]:
        img = _add_noise(img, profile["noise_std"])
    return img, mask, text


def apply_dataset_policy_v2(img, mask, text, dataset, image_size):
    """Histology-aware policy: BUSI text-geo, CVC hflip-LCAug, GlaS appearance-only."""
    dataset = str(dataset).lower()
    if dataset == "busi":
        return apply_text_geo_dataset(img, mask, text, dataset, image_size)
    if dataset == "cvc":
        return apply_lcaug_hflip_dataset(img, mask, text, dataset, image_size)
    if dataset == "glas":
        return apply_glas_appearance_dataset(img, mask, text, dataset, image_size)
    return apply_lcaug_hflip_dataset(img, mask, text, dataset, image_size)


AUGMENTATION_PLUGINS = {
    "dataset_policy_v1": apply_dataset_policy_v1,
    "dataset_policy_v2": apply_dataset_policy_v2,
    "glas_appearance_dataset": apply_glas_appearance_dataset,
    "lcaug_hflip_dataset": apply_lcaug_hflip_dataset,
    "lcaug_hflip_no_text_rewrite_dataset": apply_lcaug_hflip_no_text_rewrite_dataset,
    "lcaug_v2_hflip_dataset": apply_lcaug_v2_hflip_dataset,
    "lcaug_v2_hflip_no_text_rewrite_dataset": apply_lcaug_v2_hflip_no_text_rewrite_dataset,
    "lcaug_v2_busi_dataset": apply_lcaug_v2_busi_dataset,
    "lcaug_v2_busi_no_text_rewrite_dataset": apply_lcaug_v2_busi_no_text_rewrite_dataset,
    "lcaug_v2_dynamic_shared_plan_dataset": apply_lcaug_v2_dynamic_shared_plan_dataset,
    "lcaug_v2_dynamic_shared_plan_recompute_location_dataset": apply_lcaug_v2_dynamic_shared_plan_recompute_location_dataset,
    "lcaug_v2_dynamic_shared_plan_no_text_rewrite_dataset": apply_lcaug_v2_dynamic_shared_plan_no_text_rewrite_dataset,
    "lcaug_v2_medclipseg_dataset": apply_lcaug_v2_medclipseg_dataset,
    "lcaug_v2_medclipseg_no_text_rewrite_dataset": apply_lcaug_v2_medclipseg_no_text_rewrite_dataset,
    "text_geo_dataset": apply_text_geo_dataset,
}


def get_augmentation_plugin(name):
    return AUGMENTATION_PLUGINS.get(name)


def plugin_names():
    return sorted(AUGMENTATION_PLUGINS)
