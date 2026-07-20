#!/usr/bin/env python3
"""Exact, compositional direction rewriting for Protocol V3 LCAug."""

from __future__ import annotations

import csv
import re
from pathlib import Path


DIRECTION_VERSION = "lcaug_direction_v4_finite_closure_20260712"

ATOMIC_DIRECTIONS = (
    "upper middle",
    "upper left",
    "upper right",
    "lower middle",
    "lower left",
    "lower right",
    "middle left",
    "middle right",
    "top left",
    "top right",
    "bottom left",
    "bottom right",
    "upper",
    "lower",
    "left",
    "right",
    "top",
    "bottom",
    "central",
    "center",
    "centre",
    "middle",
)

# Exported examples only. They are intentionally parsed as comma-separated atoms.
COMPOUND_DIRECTIONS = (
    "left, center",
    "right, center",
    "top, top left",
    "top, top right",
    "bottom, bottom left",
    "bottom, bottom right",
)

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

ALL_PHRASES = sorted(ATOMIC_DIRECTIONS, key=len, reverse=True)
DIRECTION_PATTERN_V2 = re.compile(
    r"\b(" + "|".join(re.escape(phrase) for phrase in ALL_PHRASES) + r")\b",
    re.IGNORECASE,
)


def _canonical_op(op: str) -> str:
    normalized = str(op).strip().lower()
    aliases = {
        "rot90": "rot90_ccw",
        "rot90_pil_ccw": "rot90_ccw",
        "rot90_ccw": "rot90_ccw",
        "rot90_cw": "rot90_cw",
        "hflip": "hflip",
        "vflip": "vflip",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported op: {op}")
    return aliases[normalized]


def _apply_vector_op(vector: tuple[int, int], op: str) -> tuple[int, int]:
    x, y = vector
    if op == "hflip":
        return -x, y
    if op == "vflip":
        return x, -y
    if op == "rot90_ccw":
        return y, -x
    if op == "rot90_cw":
        return -y, x
    raise ValueError(f"Unsupported op: {op}")


def _vector_to_phrase(vector: tuple[int, int], original: str) -> str:
    x, y = vector
    original_lower = original.lower()
    if x == 0 and y == 0:
        if original_lower == "centre":
            mapped = "centre"
        elif original_lower == "central":
            mapped = "central"
        elif original_lower == "middle":
            mapped = "middle"
        else:
            mapped = "center"
    else:
        vertical_negative = "upper" if {"upper", "lower"} & set(original_lower.split()) else "top"
        vertical_positive = "lower" if {"upper", "lower"} & set(original_lower.split()) else "bottom"
        parts: list[str] = []
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
        mapped = " ".join(parts)
    if original[:1].isupper():
        return mapped[:1].upper() + mapped[1:]
    return mapped


def _map_atomic_phrase(phrase: str, op: str, times: int = 1) -> str:
    key = phrase.lower()
    vector = DIRECTION_TO_VECTOR[key]
    canonical = _canonical_op(op)
    count = max(0, int(times))
    if canonical.startswith("rot90"):
        count %= 4
    else:
        count %= 2
    for _ in range(count):
        vector = _apply_vector_op(vector, canonical)
    return _vector_to_phrase(vector, phrase)


def transform_direction_text_v2(text: str, op: str, times: int = 1) -> str:
    """Rewrite every atomic direction in one regex pass.

    Comma-separated targets remain separate, so ``top, top left`` transforms to
    ``left, bottom left`` for a 90-degree counter-clockwise rotation.
    """
    canonical = _canonical_op(op)
    return DIRECTION_PATTERN_V2.sub(
        lambda match: _map_atomic_phrase(match.group(0), canonical, times=times),
        str(text),
    )


def rewrite_directions(text: str, ops: list[str]) -> str:
    current = str(text)
    for op in ops:
        current = transform_direction_text_v2(current, op, times=1)
    return current


def lcaug_v2_text_variants(text: str) -> set[str]:
    """Return the finite lexical closure reachable by online geometry ops.

    Rewriting can change a phrase's lexical form (for example, ``middle
    right`` may become ``upper middle``).  Enumerating transforms only from
    the original sentence therefore misses variants produced by sequential
    online augmentation.  Expanding until stable mirrors the actual training
    path while remaining small because direction vectors form a finite set.
    """
    variants = {str(text)}
    frontier = set(variants)
    operations = ("hflip", "vflip", "rot90_ccw", "rot90_cw")
    while frontier:
        expanded = {
            transform_direction_text_v2(candidate, operation)
            for candidate in frontier
            for operation in operations
        }
        frontier = expanded - variants
        variants.update(frontier)
    return variants


def build_direction_maps() -> dict[str, dict[str, str]]:
    phrases = list(ATOMIC_DIRECTIONS) + list(COMPOUND_DIRECTIONS)
    return {
        op: {phrase: transform_direction_text_v2(phrase, op) for phrase in phrases}
        for op in ("hflip", "vflip", "rot90_ccw", "rot90_cw")
    }


DIRECTION_MAP = build_direction_maps()


def export_direction_map_markdown(path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# LCAugv2 Direction Mapping Table",
        "",
        f"Mapping version: `{DIRECTION_VERSION}`.",
        "",
        "| phrase | hflip | vflip | rot90_ccw | rot90_cw |",
        "| --- | --- | --- | --- | --- |",
    ]
    for phrase in list(ATOMIC_DIRECTIONS) + list(COMPOUND_DIRECTIONS):
        lines.append(
            "| "
            + " | ".join([phrase] + [DIRECTION_MAP[op][phrase] for op in ("hflip", "vflip", "rot90_ccw", "rot90_cw")])
            + " |"
        )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def export_direction_map_csv(path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["phrase", "hflip", "vflip", "rot90_ccw", "rot90_cw"]
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for phrase in list(ATOMIC_DIRECTIONS) + list(COMPOUND_DIRECTIONS):
            writer.writerow({"phrase": phrase, **{op: DIRECTION_MAP[op][phrase] for op in fields[1:]}})
    return out


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    md = export_direction_map_markdown(root / "paper/tables/lcaug_v2_direction_map.md")
    csv_path = export_direction_map_csv(root / "paper/tables/lcaug_v2_direction_map.csv")
    print(f"markdown={md}")
    print(f"csv={csv_path}")
    print("final_status: PASS")
