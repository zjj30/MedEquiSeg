#!/usr/bin/env python3
"""Generate the MedEquiSeg architecture and EquiPrompt mechanism figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


OUT = Path(__file__).resolve().parent
ASSETS = OUT / "assets"

INK = "#16212B"
MUTED = "#52616F"
LINE = "#CBD5DE"
PALE = "#F6F8FA"
WHITE = "#FFFFFF"
NAVY = "#163A5F"
BLUE = "#2C6E9F"
BLUE_FILL = "#EAF2F8"
TEAL = "#16796B"
TEAL_FILL = "#E7F4F0"
AMBER = "#B86210"
AMBER_FILL = "#FFF2E2"
RED = "#A83B43"
RED_FILL = "#FCECEE"
PURPLE = "#62548C"
PURPLE_FILL = "#F0EDF7"


mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def panel(ax, xy, wh, letter, title, note=""):
    x, y = xy
    w, h = wh
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.004,rounding_size=0.008",
            linewidth=0.8,
            edgecolor=LINE,
            facecolor=WHITE,
            zorder=0,
        )
    )
    ax.text(x + 0.012, y + h - 0.018, letter, ha="left", va="top", fontsize=11, fontweight="bold", color=NAVY)
    ax.text(x + 0.044, y + h - 0.018, title, ha="left", va="top", fontsize=8.6, fontweight="bold", color=INK)
    if note:
        ax.text(x + w - 0.012, y + h - 0.018, note, ha="right", va="top", fontsize=7.0, color=MUTED)


def box(
    ax,
    xy,
    wh,
    title,
    subtitle="",
    edge=BLUE,
    fill=WHITE,
    dashed=False,
    title_size=7.6,
    subtitle_size=7.0,
    zorder=2,
):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.004,rounding_size=0.006",
        linewidth=1.05,
        edgecolor=edge,
        facecolor=fill,
        linestyle=(0, (3, 2)) if dashed else "solid",
        zorder=zorder,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h * (0.61 if subtitle else 0.50),
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        color=edge,
        fontweight="bold",
        zorder=zorder + 1,
    )
    if subtitle:
        ax.text(
            x + w / 2,
            y + h * 0.28,
            subtitle,
            ha="center",
            va="center",
            fontsize=subtitle_size,
            color=INK,
            linespacing=1.12,
            zorder=zorder + 1,
        )
    return patch


def arrow(ax, start, end, color=MUTED, lw=1.15, dashed=False, connection="arc3", zorder=1):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=8.5,
        linewidth=lw,
        color=color,
        linestyle=(0, (3, 2)) if dashed else "solid",
        connectionstyle=connection,
        shrinkA=1.5,
        shrinkB=1.5,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def _load_example(operation="identity"):
    image = plt.imread(ASSETS / "busi_104_benign.png")
    mask = plt.imread(ASSETS / "busi_104_benign_mask.png")
    if mask.ndim == 3:
        mask = mask[..., 0]
    if operation == "hflip":
        image, mask = np.fliplr(image), np.fliplr(mask)
    elif operation == "vflip":
        image, mask = np.flipud(image), np.flipud(mask)
    elif operation == "rot90_ccw":
        image, mask = np.rot90(image, 1), np.rot90(mask, 1)
    return image, mask


def thumbnail(ax, xy, wh, operation="identity", label="", edge=BLUE, view="overlay"):
    x, y = xy
    w, h = wh
    image, mask = _load_example(operation)
    if view == "mask":
        ax.imshow(mask > 0.5, extent=(x, x + w, y, y + h), origin="upper", aspect="auto", cmap="gray", vmin=0, vmax=1, zorder=2)
    else:
        ax.imshow(image, extent=(x, x + w, y, y + h), origin="upper", aspect="auto", zorder=2)
    if view == "overlay":
        rgba = np.zeros((*mask.shape, 4), dtype=float)
        rgba[..., :3] = mpl.colors.to_rgb(TEAL)
        rgba[..., 3] = (mask > 0.5) * 0.52
        ax.imshow(rgba, extent=(x, x + w, y, y + h), origin="upper", aspect="auto", zorder=3)
    ax.add_patch(Rectangle((x, y), w, h, linewidth=0.95, edgecolor=edge, facecolor="none", zorder=4))
    if label:
        ax.text(x + w / 2, y - 0.008, label, ha="center", va="top", fontsize=7.0, color=INK)


def feature_pyramid(ax, xy, wh):
    x, y = xy
    w, h = wh
    for idx, (scale, color) in enumerate(((0.72, "#C8DDEC"), (0.84, "#9FC3DA"), (1.0, "#6FA5C6"))):
        ww = w * scale
        hh = h * scale
        dx = (w - ww) * 0.5 + idx * 0.004
        dy = (h - hh) * 0.5 + idx * 0.004
        ax.add_patch(Rectangle((x + dx, y + dy), ww, hh, linewidth=0.75, edgecolor=NAVY, facecolor=color, zorder=2 + idx))
    ax.text(x + w / 2 + 0.008, y + h / 2, r"$v_3,v_4,v_5$", ha="center", va="center", fontsize=7.1, color=NAVY, fontweight="bold", zorder=6)
    ax.text(x + w / 2, y - 0.012, "multi-scale features", ha="center", va="top", fontsize=7.0, color=MUTED)


def generate_architecture():
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor(WHITE)

    panel(ax, (0.018, 0.765), (0.964, 0.215), "A", "Paired training input", "TRAINING ONLY")
    thumbnail(ax, (0.050, 0.805), (0.085, 0.105), label=r"triplet $(x,y,p)$")
    arrow(ax, (0.140, 0.858), (0.188, 0.858), color=AMBER)
    box(
        ax,
        (0.190, 0.805),
        (0.155, 0.105),
        "Shared keyed plan",
        r"$A_{c,e,s}$",
        edge=AMBER,
        fill=AMBER_FILL,
        title_size=7.6,
        subtitle_size=6.7,
    )
    arrow(ax, (0.347, 0.858), (0.392, 0.858), color=AMBER)
    thumbnail(ax, (0.398, 0.805), (0.085, 0.105), operation="hflip", label=r"aligned $x',y'$")
    arrow(ax, (0.485, 0.858), (0.525, 0.858), color=AMBER)
    box(
        ax,
        (0.530, 0.805),
        (0.155, 0.105),
        "EquiPrompt",
        r"$p'=T_A(p)$",
        edge=AMBER,
        fill=AMBER_FILL,
    )
    box(
        ax,
        (0.735, 0.805),
        (0.205, 0.105),
        "Validity contract",
        "image-mask alignment\nprompt validity",
        edge=TEAL,
        fill=TEAL_FILL,
        title_size=7.6,
        subtitle_size=6.2,
    )
    arrow(ax, (0.685, 0.858), (0.732, 0.858), color=TEAL)

    panel(ax, (0.018, 0.235), (0.964, 0.500), "B", "Multimodal segmentation pathway", "solid = optimized   dashed = frozen   gray = cached")

    # Visual path.
    ax.text(0.045, 0.676, "VISUAL PATH", ha="left", va="center", fontsize=7.2, color=BLUE, fontweight="bold")
    thumbnail(ax, (0.045, 0.475), (0.085, 0.125), operation="hflip", label=r"transformed $x'$", view="image")
    arrow(ax, (0.132, 0.537), (0.172, 0.537), color=BLUE)
    box(ax, (0.176, 0.478), (0.115, 0.118), "CLIP RN50", "fine-tuned", edge=BLUE, fill=BLUE_FILL, dashed=False, subtitle_size=7.0)
    arrow(ax, (0.293, 0.537), (0.329, 0.537), color=BLUE)
    feature_pyramid(ax, (0.334, 0.486), (0.095, 0.098))
    arrow(ax, (0.433, 0.537), (0.468, 0.537), color=BLUE)
    box(
        ax,
        (0.472, 0.447),
        (0.135, 0.180),
        "Causal split",
        r"$F_{sup}$ / $F_{inf}$",
        edge=PURPLE,
        fill=PURPLE_FILL,
        title_size=7.0,
        subtitle_size=6.5,
    )

    # Dual projectors are the actual active text-conditioning sites.
    arrow(ax, (0.609, 0.572), (0.653, 0.584), color=PURPLE)
    arrow(ax, (0.609, 0.500), (0.653, 0.486), color=PURPLE)
    box(ax, (0.657, 0.545), (0.145, 0.092), r"$P_{sup}(F_{sup},s_p)$", "ATConv projector", edge=TEAL, fill=TEAL_FILL, title_size=7.0, subtitle_size=6.2)
    box(ax, (0.657, 0.437), (0.145, 0.092), r"$P_{ad}(F_{inf},s_p)$", "ATConv projector", edge=TEAL, fill=TEAL_FILL, title_size=7.0, subtitle_size=6.2)
    arrow(ax, (0.804, 0.591), (0.842, 0.548), color=TEAL)
    arrow(ax, (0.804, 0.483), (0.842, 0.530), color=TEAL)
    box(ax, (0.844, 0.493), (0.070, 0.090), "Mean", r"$\frac{z_{sup}+z_{ad}}{2}$", edge=NAVY, fill=WHITE, title_size=7.0)
    arrow(ax, (0.916, 0.538), (0.944, 0.538), color=NAVY)
    ax.add_patch(Rectangle((0.947, 0.487), 0.024, 0.102, linewidth=0.9, edgecolor=TEAL, facecolor=TEAL_FILL, zorder=2))
    ax.text(0.959, 0.538, r"$\hat y$", ha="center", va="center", fontsize=7.8, color=TEAL, fontweight="bold")

    # Text path.
    ax.text(0.045, 0.396, "TEXT PATH", ha="left", va="center", fontsize=7.2, color=AMBER, fontweight="bold")
    box(ax, (0.045, 0.285), (0.105, 0.080), r"prompt $p'$", '"top left"', edge=AMBER, fill=AMBER_FILL, title_size=7.1)
    arrow(ax, (0.152, 0.325), (0.185, 0.325), color=AMBER)
    box(ax, (0.189, 0.278), (0.135, 0.094), "BioMedCLIP", "frozen text encoder", edge=AMBER, fill=AMBER_FILL, dashed=True, title_size=7.2, subtitle_size=6.2)
    arrow(ax, (0.326, 0.325), (0.359, 0.325), color=AMBER)
    box(ax, (0.363, 0.285), (0.125, 0.080), "Cached feature", "512-D pooled", edge=MUTED, fill=PALE, dashed=False, title_size=7.0, subtitle_size=6.2)
    arrow(ax, (0.490, 0.325), (0.523, 0.325), color=AMBER)
    box(ax, (0.527, 0.278), (0.130, 0.094), "Prompt state", r"trainable $s_p\in\mathbb{R}^{1024}$", edge=TEAL, fill=TEAL_FILL, title_size=7.0, subtitle_size=6.1)
    arrow(ax, (0.657, 0.325), (0.716, 0.437), color=TEAL, connection="arc3,rad=-0.18")
    arrow(ax, (0.657, 0.325), (0.716, 0.545), color=TEAL, connection="arc3,rad=-0.27")

    # Training objective: prediction and transformed target meet at the loss.
    box(ax, (0.850, 0.345), (0.045, 0.055), r"$y'$", "", edge=RED, fill=RED_FILL, title_size=8.0)
    box(ax, (0.920, 0.338), (0.050, 0.070), r"$\mathcal{L}_{seg}$", "", edge=RED, fill=WHITE, title_size=7.2)
    arrow(ax, (0.959, 0.487), (0.950, 0.412), color=RED, dashed=True)
    arrow(ax, (0.897, 0.370), (0.917, 0.370), color=RED, dashed=True)

    panel(ax, (0.018, 0.025), (0.964, 0.180), "C", "Prompt-reliability audit", "EVALUATION ONLY; NO LEARNED GATE")
    prompts = (("true", TEAL, TEAL_FILL), ("shuffled", RED, RED_FILL), ("fixed", PURPLE, PURPLE_FILL), ("empty", MUTED, PALE))
    x0 = 0.050
    for idx, (name, color, fill) in enumerate(prompts):
        box(ax, (x0 + idx * 0.103, 0.075), (0.088, 0.065), name, "prompt", edge=color, fill=fill, title_size=7.2, subtitle_size=6.5)
    arrow(ax, (0.447, 0.108), (0.505, 0.108), color=MUTED)
    box(ax, (0.510, 0.067), (0.165, 0.082), "Frozen checkpoint", "fixed image and weights", edge=NAVY, fill=BLUE_FILL, title_size=7.2, subtitle_size=6.5)
    arrow(ax, (0.677, 0.108), (0.727, 0.108), color=MUTED)
    box(ax, (0.732, 0.067), (0.215, 0.082), "Semantic sensitivity", r"$\Delta$Dice / $\Delta$prediction", edge=TEAL, fill=TEAL_FILL, title_size=7.2, subtitle_size=6.5)

    for suffix in ("pdf", "svg"):
        fig.savefig(OUT / f"medequiseg_architecture.{suffix}", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(OUT / "medequiseg_architecture.png", dpi=320, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def compass(ax, center, vector, label, color):
    cx, cy = center
    size = 0.042
    ax.add_patch(Rectangle((cx - size, cy - size), 2 * size, 2 * size, linewidth=0.8, edgecolor=LINE, facecolor=WHITE, zorder=1))
    ax.axhline(cy, xmin=cx - size, xmax=cx + size, color=LINE, lw=0.45)
    ax.plot([cx, cx], [cy - size, cy + size], color=LINE, lw=0.45)
    dx, dy = vector
    arrow(ax, (cx, cy), (cx + dx * size * 0.72, cy - dy * size * 0.72), color=color, lw=1.3, zorder=3)
    ax.text(cx, cy - size - 0.012, label, ha="center", va="top", fontsize=7.0, color=color, fontweight="bold")


def generate_equiprompt():
    fig, ax = plt.subplots(figsize=(7.2, 4.25))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor(WHITE)

    panel(ax, (0.018, 0.055), (0.310, 0.915), "A", "Aligned views")
    thumbnail(ax, (0.050, 0.695), (0.105, 0.150), label=r"image $x$ + mask $y$")
    ax.text(0.103, 0.880, r'$p$: "top right"', ha="center", va="center", fontsize=7.0, color=AMBER, fontweight="bold")
    arrow(ax, (0.160, 0.770), (0.205, 0.770), color=AMBER)
    box(ax, (0.208, 0.715), (0.080, 0.105), "HFlip", r"$T_H$", edge=AMBER, fill=AMBER_FILL, title_size=7.1)
    arrow(ax, (0.248, 0.712), (0.248, 0.650), color=AMBER)
    thumbnail(ax, (0.050, 0.445), (0.105, 0.150), operation="hflip", label=r"$x'=A^x(x)$", view="image")
    thumbnail(ax, (0.180, 0.445), (0.105, 0.150), operation="hflip", label=r"$y'=A^y(y)$", view="mask", edge=TEAL)
    box(ax, (0.050, 0.250), (0.235, 0.105), "Prompt view", r'$p\prime=T_H(p)$: "top left"', edge=TEAL, fill=TEAL_FILL, title_size=7.2)
    ax.text(0.168, 0.190, "Aligned triplet", ha="center", va="center", fontsize=7.0, color=TEAL, fontweight="bold")
    ax.plot([0.050, 0.285], [0.160, 0.160], color=LINE, lw=0.7)
    ax.text(0.168, 0.115, "Appearance: image only", ha="center", va="center", fontsize=6.8, color=MUTED)

    panel(ax, (0.343, 0.055), (0.305, 0.915), "B", "Direction action")
    compass(ax, (0.400, 0.770), (1, -1), "top right", AMBER)
    ax.text(0.488, 0.770, r"$\longrightarrow$", ha="center", va="center", fontsize=10, color=MUTED)
    ax.text(0.488, 0.800, r"$H$", ha="center", va="center", fontsize=6.7, color=MUTED)
    compass(ax, (0.575, 0.770), (-1, -1), "top left", TEAL)
    ax.text(0.488, 0.635, r"$T_H(d_x,d_y)=(-d_x,d_y)$", ha="center", va="center", fontsize=6.5, color=INK)

    rows = [
        ("horizontal flip", "top left"),
        ("vertical flip", "bottom right"),
        (r"$90^\circ$ CCW", "top left"),
    ]
    xcols = (0.375, 0.570)
    ytop = 0.545
    ax.add_patch(Rectangle((0.360, ytop), 0.270, 0.055, facecolor=NAVY, edgecolor=NAVY, zorder=1))
    ax.text(0.495, ytop + 0.027, 'Mappings for "top right"', ha="center", va="center", fontsize=7.0, color=WHITE, fontweight="bold")
    for idx, (op, phrase) in enumerate(rows):
        yy = ytop - (idx + 1) * 0.072
        ax.add_patch(Rectangle((0.360, yy), 0.270, 0.072, facecolor=WHITE if idx % 2 == 0 else PALE, edgecolor=LINE, linewidth=0.5, zorder=1))
        ax.text(xcols[0], yy + 0.036, op, ha="left", va="center", fontsize=6.5, color=INK)
        ax.text(xcols[1], yy + 0.036, phrase, ha="center", va="center", fontsize=6.5, color=TEAL, fontweight="bold")
    ax.text(0.495, 0.255, "Compose, then lexicalize", ha="center", va="center", fontsize=6.7, color=INK)
    box(ax, (0.370, 0.115), (0.250, 0.095), "Atomic phrase handling", '"upper right" maps once', edge=PURPLE, fill=PURPLE_FILL, title_size=7.0, subtitle_size=6.6)

    panel(ax, (0.663, 0.055), (0.319, 0.915), "C", "Matched no-rewrite control")
    ax.text(0.823, 0.875, "Matched across both runs", ha="center", va="center", fontsize=7.0, color=NAVY, fontweight="bold")
    ax.text(0.823, 0.836, "same model, initialization, optimizer", ha="center", va="center", fontsize=6.5, color=MUTED)
    ax.text(0.823, 0.805, r"same order and $A_{c,e,s}$ plans", ha="center", va="center", fontsize=6.5, color=MUTED)
    ax.plot([0.672, 0.952], [0.770, 0.770], color=LINE, lw=0.8, ls=(0, (3, 2)))

    thumbnail(ax, (0.682, 0.555), (0.076, 0.115), operation="hflip", label=r"same $x',y'$")
    arrow(ax, (0.758, 0.615), (0.797, 0.680), color=TEAL)
    arrow(ax, (0.758, 0.600), (0.797, 0.470), color=RED)

    box(ax, (0.803, 0.625), (0.145, 0.100), "MedEquiSeg", r"$p'=T_A(p)$", edge=TEAL, fill=TEAL_FILL, title_size=7.2, subtitle_size=7.0)
    ax.text(0.873, 0.585, "ALIGNED", ha="center", va="center", fontsize=7.0, color=TEAL, fontweight="bold")
    ax.plot([0.800, 0.945], [0.550, 0.550], color=LINE, lw=0.7)
    box(ax, (0.803, 0.405), (0.145, 0.100), "No rewrite", r"$p'=p$", edge=RED, fill=RED_FILL, title_size=7.0, subtitle_size=7.0)
    ax.text(0.873, 0.365, "VIEW-INCONSISTENT", ha="center", va="center", fontsize=6.6, color=RED, fontweight="bold")

    ax.add_patch(Rectangle((0.680, 0.190), 0.270, 0.105, facecolor=PALE, edgecolor=LINE, linewidth=0.7, zorder=1))
    ax.text(0.815, 0.260, "Isolated factor", ha="center", va="center", fontsize=6.8, color=NAVY, fontweight="bold")
    ax.text(0.815, 0.220, "only rewrite differs", ha="center", va="center", fontsize=7.0, color=INK)
    ax.text(0.815, 0.125, "Tests prompt validity without\nassuming an accuracy gain", ha="center", va="center", fontsize=6.7, color=MUTED, linespacing=1.2)

    for suffix in ("pdf", "svg"):
        fig.savefig(OUT / f"equiprompt_mechanism.{suffix}", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(OUT / "equiprompt_mechanism.png", dpi=320, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


if __name__ == "__main__":
    generate_architecture()
    generate_equiprompt()
    print("generated medequiseg_architecture and equiprompt_mechanism")
