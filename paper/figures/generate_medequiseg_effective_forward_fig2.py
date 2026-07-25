#!/usr/bin/env python3
"""Generate the source-verified effective MedEquiSeg forward graph (Fig. 2)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


INK = "#15232E"
MUTED = "#52616F"
LINE = "#C8D3DC"
WHITE = "#FFFFFF"
PALE = "#F5F7F9"
BLUE = "#155A9C"
BLUE_FILL = "#EAF3FB"
ORANGE = "#C66712"
ORANGE_FILL = "#FFF2E4"
GREEN = "#237A45"
GREEN_FILL = "#EAF6EE"
PURPLE = "#6945A5"
PURPLE_FILL = "#F2ECFA"
RED = "#B43B45"
RED_FILL = "#FCEDEF"
GRAY = "#66737E"
GRAY_FILL = "#ECEFF2"


mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 18,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.unicode_minus": False,
    }
)


def rounded(ax, x, y, w, h, edge=LINE, fill=WHITE, lw=1.4, dashed=False, radius=0.006, z=1):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.004,rounding_size={radius}",
        linewidth=lw,
        edgecolor=edge,
        facecolor=fill,
        linestyle=(0, (4, 2.5)) if dashed else "solid",
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def panel(ax, x, y, w, h, letter, title, note=""):
    rounded(ax, x, y, w, h, edge=LINE, fill=WHITE, lw=1.1, radius=0.008, z=0)
    ax.text(x + 0.009, y + h - 0.017, letter, ha="left", va="top", fontsize=24, fontweight="bold", color=BLUE)
    ax.text(x + 0.037, y + h - 0.017, title, ha="left", va="top", fontsize=21, fontweight="bold", color=INK)
    if note:
        ax.text(x + w - 0.009, y + h - 0.017, note, ha="right", va="top", fontsize=15.5, color=MUTED)


def module(ax, x, y, w, h, title, body="", edge=BLUE, fill=WHITE, dashed=False,
           title_size=18.5, body_size=16.0, align="left", z=2):
    rounded(ax, x, y, w, h, edge=edge, fill=fill, lw=1.45, dashed=dashed, radius=0.005, z=z)
    tx = x + (0.010 if align == "left" else w / 2)
    ha = "left" if align == "left" else "center"
    ax.text(tx, y + h - 0.018, title, ha=ha, va="top", fontsize=title_size, fontweight="bold", color=edge, zorder=z + 1)
    if body:
        ax.text(tx, y + h - 0.052, body, ha=ha, va="top", fontsize=body_size, color=INK,
                linespacing=1.18, zorder=z + 1)


def small_box(ax, x, y, w, h, text, edge=BLUE, fill=WHITE, size=16.5, bold=False, dashed=False, z=3):
    rounded(ax, x, y, w, h, edge=edge, fill=fill, lw=1.35, dashed=dashed, radius=0.004, z=z)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size,
            color=edge if bold else INK, fontweight="bold" if bold else "normal", linespacing=1.12, zorder=z + 1)


def arrow(ax, start, end, color=MUTED, lw=1.7, dashed=False, rad=0.0, z=2):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=15,
        linewidth=lw,
        color=color,
        linestyle=(0, (4, 2.5)) if dashed else "solid",
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=2.0,
        shrinkB=2.0,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def poly_arrow(ax, points, color=MUTED, lw=1.7, dashed=False, z=2):
    style = (0, (4, 2.5)) if dashed else "solid"
    if len(points) < 2:
        return
    if len(points) > 2:
        xs, ys = zip(*points[:-1])
        ax.plot(xs, ys, color=color, lw=lw, linestyle=style, zorder=z)
    arrow(ax, points[-2], points[-1], color=color, lw=lw, dashed=dashed, z=z)


def draw_figure(out_dir: Path):
    fig, ax = plt.subplots(figsize=(18.0, 10.7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor(WHITE)

    ax.text(0.5, 0.982, "Detailed Effective MedEquiSeg Forward Graph", ha="center", va="top",
            fontsize=29, fontweight="bold", color=INK)

    # Legend.
    ax.plot([0.710, 0.745], [0.946, 0.946], color=BLUE, lw=2.2)
    ax.text(0.750, 0.946, "trainable / fine-tuned", ha="left", va="center", fontsize=15.5, color=MUTED)
    ax.plot([0.835, 0.870], [0.946, 0.946], color=ORANGE, lw=2.2, ls=(0, (4, 2.5)))
    ax.text(0.875, 0.946, "frozen", ha="left", va="center", fontsize=15.5, color=MUTED)
    rounded(ax, 0.935, 0.938, 0.020, 0.016, edge=GRAY, fill=GRAY_FILL, lw=1.0)
    ax.text(0.960, 0.946, "cached", ha="left", va="center", fontsize=15.5, color=MUTED)

    # ------------------------------------------------------------------
    # Panel A: effective visual graph.
    # ------------------------------------------------------------------
    panel(ax, 0.012, 0.590, 0.976, 0.340, "A", "Fine-tuned visual path and effective two-projector decoder",
          "solid modules are optimized")

    small_box(ax, 0.022, 0.744, 0.055, 0.084, "$x'$\n$[B\\times3\\times224\\times224]$",
              edge=BLUE, fill=BLUE_FILL, size=17.0, bold=True)

    module(
        ax, 0.090, 0.624, 0.205, 0.267,
        "Fine-tuned OpenAI CLIP RN50 Visual Encoder",
        "Stem: 3x3/s2 3->32; 3x3 32->32; 3x3 32->64; AvgPool/s2\n"
        "Layer 1: Bottleneck x3 -> [B x 256 x 56 x 56]\n"
        "Layer 2: Bottleneck x4 -> V3 [B x 512 x 28 x 28]\n"
        "Layer 3: Bottleneck x6 -> V4 [B x 1024 x 14 x 14]\n"
        "Layer 4: Bottleneck x3 -> [B x 2048 x 7 x 7]\n"
        "Spatial AttentionPool2d: 49 spatial tokens; 32 heads;\n"
        "Q/K/V 2048->2048; projection 2048->1024;\n"
        "residual 1x1 2048->1024; Add + ReLU -> V5raw\n"
        "1x1 Conv 1024->512 + BN + ReLU -> V5 [B x 512 x 7 x 7]",
        edge=BLUE, fill=BLUE_FILL, title_size=18.0, body_size=15.3,
    )
    arrow(ax, (0.077, 0.786), (0.089, 0.786), color=BLUE)

    module(
        ax, 0.310, 0.655, 0.142, 0.220,
        "Multi-scale alignment",
        "V3 -> AvgPool 2x2/s2\n   -> F3 [B x 512 x 14 x 14]\n\n"
        "V4 -> Conv3x3 1024->512\n   -> Conv3x3 512->512\n   -> F4 [B x 512 x 14 x 14]\n\n"
        "V5 -> one CARAFE (upsampling x2)\n   -> F5 [B x 512 x 14 x 14]",
        edge=BLUE, fill=WHITE, title_size=18.5, body_size=15.5,
    )
    arrow(ax, (0.295, 0.786), (0.309, 0.786), color=BLUE)

    module(
        ax, 0.468, 0.655, 0.145, 0.220,
        "Causal Feature Partition",
        "Three Maskers: same structure,\nseparate parameters\n\n"
        "$M_k = \\mathrm{Masker}_k(F_k)$\n"
        "$F_{k,sup}=F_k\\odot M_k$\n"
        "$F_{k,inf}=F_k\\odot(1-M_k)$\n\n"
        "Concat three supportive scales\nConcat three inferior scales",
        edge=PURPLE, fill=PURPLE_FILL, title_size=18.0, body_size=16.0,
    )
    arrow(ax, (0.452, 0.786), (0.467, 0.786), color=BLUE)

    module(
        ax, 0.629, 0.692, 0.090, 0.145,
        "Shared aggregation",
        "same weights for both calls\n\n1x1 Conv 1536->512\n+ BN + ReLU",
        edge=BLUE, fill=PALE, title_size=17.5, body_size=15.3, align="center",
    )
    arrow(ax, (0.613, 0.812), (0.628, 0.812), color=GREEN)
    arrow(ax, (0.613, 0.714), (0.628, 0.714), color=PURPLE)
    small_box(ax, 0.724, 0.803, 0.083, 0.045, "$F_{sup}$  [B x 512 x 14 x 14]", edge=GREEN,
              fill=GREEN_FILL, size=15.5, bold=True)
    small_box(ax, 0.724, 0.676, 0.083, 0.050, "$F_{inf}$ (inferior)\n[B x 512 x 14 x 14]", edge=PURPLE,
              fill=PURPLE_FILL, size=14.8, bold=True)
    arrow(ax, (0.719, 0.812), (0.723, 0.825), color=GREEN)
    arrow(ax, (0.719, 0.714), (0.723, 0.701), color=PURPLE)

    # Independent projector heads; each has exactly one active ATConv.
    module(
        ax, 0.820, 0.760, 0.130, 0.130,
        r"Supportive Projector $P_{sup}$",
        "independent parameters\n"
        "Fsup -> Bilinear x4 -> ATConv3x3 512->512 -> BN+ReLU\n"
        "-> Bilinear x4 -> Conv3x3 512->256 + BN+ReLU\n"
        "-> Conv1x1 256->256 -> U [B x 256 x 224 x 224]\n"
        "sp -> Linear 1024->2305 -> W [B x 256 x 3 x 3], b [B]\n"
        "per-sample dynamic Conv3x3(U;W,b)",
        edge=GREEN, fill=GREEN_FILL, title_size=17.0, body_size=13.3,
    )
    module(
        ax, 0.820, 0.605, 0.130, 0.130,
        r"Inferior Projector $P_{ad}$",
        "independent parameters\n"
        "Finf -> Bilinear x4 -> ATConv3x3 512->512 -> BN+ReLU\n"
        "-> Bilinear x4 -> Conv3x3 512->256 + BN+ReLU\n"
        "-> Conv1x1 256->256 -> U [B x 256 x 224 x 224]\n"
        "sp -> Linear 1024->2305 -> W [B x 256 x 3 x 3], b [B]\n"
        "per-sample dynamic Conv3x3(U;W,b)",
        edge=PURPLE, fill=PURPLE_FILL, title_size=17.0, body_size=13.3,
    )
    arrow(ax, (0.807, 0.825), (0.819, 0.825), color=GREEN)
    poly_arrow(ax, [(0.807, 0.701), (0.813, 0.701), (0.813, 0.670), (0.819, 0.670)], color=PURPLE)
    small_box(ax, 0.958, 0.802, 0.026, 0.052, "$z_{sup}$\n[B x 1 x 224 x 224]", edge=GREEN,
              fill=WHITE, size=13.2, bold=True)
    small_box(ax, 0.958, 0.646, 0.026, 0.052, "$z_{ad}$\n[B x 1 x 224 x 224]", edge=PURPLE,
              fill=WHITE, size=13.2, bold=True)
    arrow(ax, (0.950, 0.825), (0.957, 0.828), color=GREEN)
    arrow(ax, (0.950, 0.670), (0.957, 0.672), color=PURPLE)
    ax.text(0.886, 0.597, "Effective forward: 2 active ATConv operators", ha="center", va="top",
            fontsize=16.0, color=INK, fontweight="bold")

    # ------------------------------------------------------------------
    # Panel B: frozen pooled text path.
    # ------------------------------------------------------------------
    panel(ax, 0.012, 0.447, 0.705, 0.126, "B", "Frozen pooled BioMedCLIP path",
          "sp conditions only the two projectors")
    small_box(ax, 0.025, 0.478, 0.065, 0.055, "rewritten\nprompt $p'$", edge=ORANGE,
              fill=ORANGE_FILL, size=16.0, bold=True)
    small_box(ax, 0.101, 0.478, 0.080, 0.055, "WordPiece\ncontext length 256", edge=ORANGE,
              fill=ORANGE_FILL, size=15.0)
    module(
        ax, 0.192, 0.466, 0.225, 0.080,
        "Frozen BioMedCLIP PubMedBERT text encoder",
        "token + position + token-type embeddings -> PubMedBERT x12\n"
        "12-head self-attention, width 768; FFN 768->3072->768\n"
        "CLS last-hidden-state pooling -> MLP 768->640->512 -> L2 normalization",
        edge=ORANGE, fill=ORANGE_FILL, dashed=True, title_size=17.0, body_size=13.8,
    )
    small_box(ax, 0.429, 0.478, 0.104, 0.055, "offline cached pooled\n$e_{bio}$ [B x 512]", edge=GRAY,
              fill=GRAY_FILL, size=15.0, bold=True)
    small_box(ax, 0.545, 0.478, 0.105, 0.055, "Trainable Prompt Adapter\nLinear 512->1024, no bias", edge=ORANGE,
              fill=WHITE, size=14.7, bold=True)
    small_box(ax, 0.662, 0.478, 0.043, 0.055, "$s_p$\n[B x 1024]", edge=ORANGE,
              fill=ORANGE_FILL, size=15.0, bold=True)
    for x0, x1 in ((0.090, 0.100), (0.181, 0.191), (0.417, 0.428), (0.533, 0.544), (0.650, 0.661)):
        arrow(ax, (x0, 0.505), (x1, 0.505), color=ORANGE, lw=1.6)

    # Exactly two conditioning arrows, both terminate inside projector boxes.
    poly_arrow(ax, [(0.683, 0.533), (0.683, 0.561), (0.800, 0.561), (0.800, 0.789), (0.819, 0.789)], color=ORANGE, lw=1.8)
    poly_arrow(ax, [(0.683, 0.478), (0.683, 0.437), (0.800, 0.437), (0.800, 0.635), (0.819, 0.635)], color=ORANGE, lw=1.8)

    # ------------------------------------------------------------------
    # Panel D: fusion, loss, and inference. Only logits enter the mean.
    # ------------------------------------------------------------------
    panel(ax, 0.731, 0.447, 0.257, 0.126, "D", "Logit fusion and objective")
    small_box(ax, 0.752, 0.487, 0.078, 0.052, "Mean logits\n$z=(z_{sup}+z_{ad})/2$", edge=INK,
              fill=WHITE, size=15.8, bold=True)
    # Route zsup and zad down the outside edge and into Mean logits; no other inputs.
    poly_arrow(ax, [(0.971, 0.802), (0.971, 0.565), (0.840, 0.565), (0.840, 0.522), (0.831, 0.522)], color=GREEN, lw=1.7)
    poly_arrow(ax, [(0.971, 0.646), (0.971, 0.555), (0.846, 0.555), (0.846, 0.505), (0.831, 0.505)], color=PURPLE, lw=1.7)
    small_box(ax, 0.848, 0.524, 0.058, 0.033, "aligned $y'$\n[B x 1 x 224 x 224]", edge=RED,
              fill=RED_FILL, size=12.9, bold=True)
    small_box(ax, 0.916, 0.505, 0.063, 0.052,
              "TRAINING\n0.5 BCEWithLogits(z,y')\n+ 0.5 Soft-Dice(sigmoid(z),y')", edge=RED,
              fill=RED_FILL, size=12.7, bold=True)
    small_box(ax, 0.848, 0.462, 0.131, 0.035,
              "$L_{seg}=0.5L_{BCE}+0.5L_{Soft-Dice}$", edge=RED, fill=WHITE, size=14.0, bold=True)
    arrow(ax, (0.830, 0.522), (0.915, 0.530), color=RED, lw=1.6)
    arrow(ax, (0.906, 0.540), (0.915, 0.540), color=RED, lw=1.5)
    arrow(ax, (0.947, 0.505), (0.925, 0.498), color=RED, lw=1.5)
    ax.text(0.791, 0.476, "INFERENCE", ha="center", va="top", fontsize=13.5, color=BLUE, fontweight="bold")
    ax.text(0.791, 0.459, "sigmoid -> P -> threshold 0.5 -> $\\hat{y}$", ha="center", va="top",
            fontsize=13.0, color=INK)
    arrow(ax, (0.790, 0.487), (0.790, 0.466), color=BLUE, lw=1.5)

    # ------------------------------------------------------------------
    # Panel C: source-verified implementation insets.
    # ------------------------------------------------------------------
    panel(ax, 0.012, 0.025, 0.976, 0.405, "C", "Layerwise implementation insets",
          "standard components are expanded for reproducibility")

    module(
        ax, 0.025, 0.238, 0.285, 0.150,
        "Anti-aliased RN50 Bottleneck",
        "Main: Conv1x1 -> BN -> ReLU -> Conv3x3 -> BN -> ReLU\n"
        "       -> optional AvgPool(stride>1) -> Conv1x1 -> BN\n"
        "Skip: identity, or AvgPool(stride>1) -> Conv1x1 -> BN\n"
        "Residual Add -> ReLU",
        edge=BLUE, fill=BLUE_FILL, title_size=18.0, body_size=16.2,
    )
    module(
        ax, 0.324, 0.238, 0.248, 0.150,
        "CARAFE (one module, upsampling factor x2)",
        "Conv1x1 512->128 -> Conv3x3 128->36 -> PixelShuffle x2\n"
        "-> nine 3x3 reassembly weights -> Softmax\n"
        "-> content-aware reassembly -> Conv1x1 512->512",
        edge=BLUE, fill=WHITE, title_size=17.2, body_size=16.0,
    )
    module(
        ax, 0.586, 0.238, 0.389, 0.150,
        "Masker (one of three; same structure, separate parameters)",
        "coordinate concatenation 512+2->514 -> Conv3x3 514->512 + BN + ReLU\n"
        "-> Conv3x3 512->512 + BN + ReLU -> coordinate concatenation\n"
        "-> Conv3x3 514->512 + BN + ReLU -> Conv3x3 512->512 + BN -> Sigmoid\n"
        "$M_k$ [B x 512 x 14 x 14]",
        edge=PURPLE, fill=PURPLE_FILL, title_size=17.2, body_size=15.5,
    )

    module(
        ax, 0.025, 0.055, 0.420, 0.155,
        "Frozen pooled BioMedCLIP representation (effective path)",
        "WordPiece -> token/position/token-type embeddings -> PubMedBERT Transformer x12\n"
        "Each layer: 12-head Q/K/V 768->768 -> Add & LayerNorm -> FFN 768->3072->768\n"
        "CLS last hidden state [B x 768] -> no-bias MLP 768->640->512 -> L2 normalization\n"
        "offline cache $e_{bio}$ [B x 512] -> trainable no-bias Linear 512->1024 -> pooled $s_p$\n"
        "No token-level cross-attention, LoRA, or text gate enters the reported forward graph.",
        edge=ORANGE, fill=ORANGE_FILL, dashed=True, title_size=17.5, body_size=15.2,
    )
    module(
        ax, 0.460, 0.055, 0.515, 0.155,
        "ATConv (one active operator inside each projector)",
        "Kernel branch: Conv1x1 -> reshape -> AdaptiveAvgPool1d(9) -> GELU -> Linear 9->9\n"
        "$K$ [B x 512 x 3 x 3]; lateral inhibition: $K'=K-\\mathrm{sigmoid}(d)\\cdot\\mathrm{mean}(K)$\n"
        "Feature branch: Conv1x1 -> reshape [1, Bx512, H, W]\n"
        "-> dynamic depthwise grouped Conv using $K'$, groups=Bx512 -> reshape -> Conv1x1\n"
        "ATConv is visual-only; $s_p$ parameterizes the later per-sample segmentation classifier.",
        edge=BLUE, fill=BLUE_FILL, title_size=17.5, body_size=15.2,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / "medequiseg_effective_forward_fig2_chatgpt_v1"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.06)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.06)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    draw_figure(args.out_dir)


if __name__ == "__main__":
    main()
