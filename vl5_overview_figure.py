"""
vl5_overview_figure.py -- Study-design overview (Figure 1 of the manuscript).
Pure matplotlib, no data or GPU needed. Output: figures/Fig1_overview.png and .pdf
Run: python vl5_overview_figure.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

BASE = Path(__file__).resolve().parent
FIG = BASE / "figures"
FIG.mkdir(exist_ok=True)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})

# palette shared with the result figures
C_PRE, C_STD, C_CAP, C_LAB = "#4C72B0", "#DD8452", "#55A868", "#C44E52"
C_VIS = "#8172B3"
INK, MUTED = "#222222", "#555555"
BG_DATA, BG_ALIGN, BG_EVAL, BG_RIG = "#EEF3FA", "#F3F8F1", "#FBF6EE", "#F2F2F2"


def box(ax, x, y, w, h, fc, ec="#9A9A9A", lw=0.8, r=0.012):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw))


def arrow(ax, x0, y0, x1, y1, color="#777777", lw=1.2, style="-|>"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=9,
                                 color=color, lw=lw, shrinkA=0, shrinkB=0))


def text(ax, x, y, s, size=8, weight="normal", color=INK, ha="left", va="center", style="normal"):
    ax.text(x, y, s, fontsize=size, fontweight=weight, color=color, ha=ha, va=va,
            fontstyle=style, linespacing=1.25)


def matrix(ax, x, y, s, pattern, color, asp=1.0):
    """Small 4x4 target-matrix icon (square on the page); pattern = set of (row, col) positives."""
    n, cw = 4, s / 4
    ch = cw * asp
    for i in range(n):
        for j in range(n):
            fc = color if (i, j) in pattern else "white"
            ax.add_patch(Rectangle((x + j * cw, y + (n - 1 - i) * ch), cw, ch, fc=fc,
                                   ec="#888888", lw=0.4))


def image_icon(ax, x, y, s):
    """Schematic bronchoscopic frame: dark octagon with a lumen."""
    ax.add_patch(FancyBboxPatch((x, y), s, s, boxstyle="round,pad=0,rounding_size=0.006",
                                fc="#2B2B2B", ec="none"))
    ax.add_patch(matplotlib.patches.Circle((x + s / 2, y + s / 2), s * 0.38, fc="#D9907A", ec="none"))
    ax.add_patch(matplotlib.patches.Ellipse((x + s * 0.42, y + s * 0.5), s * 0.22, s * 0.3,
                                            fc="#5A2E22", ec="none"))
    ax.add_patch(matplotlib.patches.Ellipse((x + s * 0.62, y + s * 0.47), s * 0.14, s * 0.2,
                                            fc="#5A2E22", ec="none"))


def main():
    fig = plt.figure(figsize=(7.2, 3.9))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    asp = 7.2 / 3.9  # width/height, to draw square icons

    # ───────────────── column A: data
    xa, wa = 0.012, 0.215
    box(ax, xa, 0.205, wa, 0.775, BG_DATA)
    text(ax, xa + 0.012, 0.945, "A  BERD dataset", 9, "bold")
    text(ax, xa + 0.012, 0.893, "6330 images from 3692\nbronchoscopy reports", 7.3, color=MUTED, va="top")
    # image + caption pair
    s = 0.075
    image_icon(ax, xa + 0.014, 0.700, s)
    box(ax, xa + 0.014 + s + 0.008, 0.700, wa - s - 0.044, s * asp * 0.55 + 0.005, "white", r=0.006)
    text(ax, xa + 0.014 + s + 0.014, 0.700 + (s * asp * 0.55 + 0.005) / 2,
         "\u201cmild mucosa\ncongestion\u201d", 6.6, style="italic", color=INK)
    text(ax, xa + 0.014, 0.672, "image\u2013caption pair (report text)", 6.4, color=MUTED)
    # split table
    y0 = 0.615
    text(ax, xa + 0.012, y0, "Patient-level split", 7.5, "bold")
    rows = [("Train", "4826", "2809"), ("Validation", "1186", "702"), ("Test", "316", "180")]
    text(ax, xa + 0.105, y0 - 0.045, "images", 6.4, color=MUTED, ha="center")
    text(ax, xa + 0.172, y0 - 0.045, "patients", 6.4, color=MUTED, ha="center")
    for k, (n, i, p) in enumerate(rows):
        yy = y0 - 0.09 - k * 0.048
        text(ax, xa + 0.012, yy, n, 7.2)
        text(ax, xa + 0.105, yy, i, 7.2, ha="center")
        text(ax, xa + 0.172, yy, p, 7.2, ha="center")
    text(ax, xa + 0.012, 0.335, "23 finding classes\n2083 unique training captions\n33% frequent captions",
         6.8, color=MUTED, va="top")

    # ───────────────── column B: alignment
    xb, wb = 0.262, 0.335
    box(ax, xb, 0.205, wb, 0.775, BG_ALIGN)
    text(ax, xb + 0.012, 0.945, "B  Report-supervised alignment", 9, "bold")
    text(ax, xb + 0.012, 0.893, "BiomedCLIP, both encoders fine-tuned", 7.3, color=MUTED)
    # encoders
    ey = 0.735
    box(ax, xb + 0.018, ey, 0.125, 0.085, "white", ec=C_PRE, lw=1.0)
    text(ax, xb + 0.0805, ey + 0.055, "Image encoder", 7.3, "bold", ha="center")
    text(ax, xb + 0.0805, ey + 0.022, "ViT-B/16", 6.8, color=MUTED, ha="center")
    box(ax, xb + 0.192, ey, 0.125, 0.085, "white", ec=C_PRE, lw=1.0)
    text(ax, xb + 0.2545, ey + 0.055, "Text encoder", 7.3, "bold", ha="center")
    text(ax, xb + 0.2545, ey + 0.022, "PubMedBERT", 6.8, color=MUTED, ha="center")
    # shared space
    sy = 0.615
    box(ax, xb + 0.06, sy, 0.215, 0.068, "white", r=0.02)
    text(ax, xb + 0.1675, sy + 0.034, "shared embedding space", 7.2, ha="center")
    arrow(ax, xb + 0.0805, ey, xb + 0.115, sy + 0.068)
    arrow(ax, xb + 0.2545, ey, xb + 0.22, sy + 0.068)
    text(ax, xb + 0.1675, sy - 0.035, "symmetric soft-target InfoNCE loss", 6.8, color=MUTED, ha="center")
    # three objectives with target matrices
    obj = [("Standard", "own caption", {(i, i) for i in range(4)}, C_STD),
           ("Caption-aware", "same caption", {(i, i) for i in range(4)} | {(0, 2), (2, 0)}, C_CAP),
           ("Label-aware*", "same caption\nor same labels",
            {(i, i) for i in range(4)} | {(0, 2), (2, 0), (1, 3), (3, 1)}, C_LAB)]
    ms = 0.05
    for k, (name, sub, pat, col) in enumerate(obj):
        cxm = xb + 0.0575 + k * 0.11          # column centre
        matrix(ax, cxm - ms / 2, 0.395, ms, pat, col, asp=asp)
        text(ax, cxm, 0.37, name, 6.8, "bold", color=col, ha="center")
        text(ax, cxm, 0.357, sub, 5.9, color=MUTED, ha="center", va="top")
    text(ax, xb + 0.012, 0.525, "Positives in the target matrix T", 7.2, "bold")
    text(ax, xb + 0.012, 0.292, "3 objectives \u00d7 3 seeds (42, 7, 123)\nselection by validation R-sum (captions only)",
         6.6, color=MUTED, va="top")
    text(ax, xb + 0.012, 0.224, "* uses finding labels (supervised reference)", 6.0, color=MUTED, style="italic")

    # ───────────────── column C: evaluations
    xc, wc = 0.632, 0.356
    box(ax, xc, 0.205, wc, 0.775, BG_EVAL)
    text(ax, xc + 0.012, 0.945, "C  Evaluation on the test set", 9, "bold")
    text(ax, xc + 0.012, 0.893, "316 images, 180 patients; final evaluation only", 7.3, color=MUTED)
    evals = [
        ("Zero-shot recognition", "text prompts, no labelled training\ngeneric vs. BERD-vocabulary prompts",
         "AUPRC, AUC-ROC"),
        ("Cross-modal retrieval", "image\u2194caption, unique-caption gallery\nincl. captions unseen in training",
         "Recall@K, R-sum"),
        ("Linear probing (frozen)", "1, 5, 10, 25, 100% labelled patients\nvs. DINOv2, Swin-T; location control",
         "AUPRC, F1"),
        ("Fine-tuning", "pretrained vs. aligned initialisation\n100% and 10\u201325% labelled patients",
         "F1, AUPRC, Brier"),
    ]
    for k, (t, d, m) in enumerate(evals):
        yy = 0.69 - k * 0.158
        box(ax, xc + 0.012, yy, wc - 0.024, 0.146, "white", r=0.008)
        text(ax, xc + 0.024, yy + 0.118, f"{k + 1}  {t}", 7.6, "bold")
        text(ax, xc + 0.024, yy + 0.068, d, 6.3, color=MUTED)
        text(ax, xc + 0.024, yy + 0.022, m, 6.3, color=C_PRE, weight="bold")

    # arrows between columns
    arrow(ax, xa + wa + 0.004, 0.59, xb - 0.004, 0.59, lw=1.6)
    arrow(ax, xb + wb + 0.004, 0.59, xc - 0.004, 0.59, lw=1.6)
    text(ax, (xb + wb + xc) / 2, 0.625, "frozen /\ninitialise", 6.0, color=MUTED, ha="center")

    # ───────────────── rigor bar
    box(ax, 0.012, 0.03, 0.976, 0.145, BG_RIG)
    text(ax, 0.026, 0.135, "Statistical and data-integrity safeguards", 7.6, "bold")
    items = ["leakage audit (patients, exact\nand near-duplicate images)",
             "patient-level bootstrap\n95% CIs (2000 resamples)",
             "3 training seeds; Holm and\nWilcoxon paired tests",
             "sensitivity: duplicates removed;\nprobes without validation labels"]
    for k, it in enumerate(items):
        text(ax, 0.026 + k * 0.243, 0.068, it, 6.6, color=MUTED)

    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"Fig1_overview.{ext}", dpi=400 if ext == "png" else None)
    plt.close(fig)
    print("\u2713 figures/Fig1_overview.png and .pdf")


if __name__ == "__main__":
    main()
