"""Shared image/mask resize helpers for segmentation pipelines."""

from __future__ import annotations

from PIL import Image


def resize_image_mask_pair(
    img: Image.Image,
    mask: Image.Image,
    size: int,
    mode: str = "stretch",
    *,
    image_resample=Image.BILINEAR,
    mask_resample=Image.NEAREST,
) -> tuple[Image.Image, Image.Image]:
    size = int(size)
    if mode == "stretch":
        return (
            img.resize((size, size), image_resample),
            mask.resize((size, size), mask_resample),
        )
    if mode == "letterbox":
        w, h = img.size
        scale = min(size / w, size / h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        img = img.resize((new_w, new_h), image_resample)
        mask = mask.resize((new_w, new_h), mask_resample)
        canvas_img = Image.new("RGB", (size, size), (0, 0, 0))
        canvas_mask = Image.new("L", (size, size), 0)
        pad_left = (size - new_w) // 2
        pad_top = (size - new_h) // 2
        canvas_img.paste(img, (pad_left, pad_top))
        canvas_mask.paste(mask, (pad_left, pad_top))
        return canvas_img, canvas_mask
    raise ValueError(f"Unknown resize mode: {mode!r}; expected 'stretch' or 'letterbox'")
