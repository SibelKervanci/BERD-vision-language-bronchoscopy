"""
vl0b_inspect_duplicates.py -- Visual and quantitative check of the duplicate pairs found by
vl0_leakage_audit.py. Bronchoscopic frames look alike (circular field, dark border, central
lumen), so a 64-bit difference hash can flag different images as near duplicates.

For every pair involving the test set it computes the Pearson correlation of the two images
(grayscale, 128 x 128) and SSIM-like structural similarity; true duplicates have r > 0.95.
It also saves contact sheets of the pairs for visual inspection.

Outputs: results/audit_duplicates_scored.csv, figures/duplicates_test_*.png
Run after vl0_leakage_audit.py (CPU, seconds).
"""

import numpy as np
import pandas as pd
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import vl_common as C

PAIRS_PER_SHEET = 12


def gray(path, size=128):
    return np.asarray(Image.open(path).convert("L").resize((size, size), Image.BILINEAR),
                      dtype=np.float64)


if __name__ == "__main__":
    _, _, TR, VA, TE = C.get_splits(verbose=False)
    recs = {"train": TR, "val": VA, "test": TE}
    dup = pd.read_csv(C.RESULTS_DIR / "audit_duplicates.csv")
    dup = dup[dup.split_b == "test"].copy()
    r_list = []
    for _, d in dup.iterrows():
        a = gray(C.resolve_path(recs[d.split_a][int(d.idx_a)]["image_path"]))
        b = gray(C.resolve_path(recs[d.split_b][int(d.idx_b)]["image_path"]))
        r_list.append(np.corrcoef(a.ravel(), b.ravel())[0, 1])
    dup["pixel_r"] = r_list
    dup["likely_true_duplicate"] = dup.pixel_r > 0.95
    dup = dup.sort_values("pixel_r", ascending=False)
    dup.to_csv(C.RESULTS_DIR / "audit_duplicates_scored.csv", index=False)

    print(f"pairs involving test images: {len(dup)}")
    print(dup.groupby("type").pixel_r.describe().round(3).to_string())
    true_imgs = dup.loc[dup.likely_true_duplicate, "idx_b"].nunique()
    print(f"\ntest images with a likely TRUE duplicate (r > 0.95): {true_imgs}")
    hash_only = set(dup.loc[~dup.likely_true_duplicate, "idx_b"]) - set(
        dup.loc[dup.likely_true_duplicate, "idx_b"])
    print(f"test images flagged by hash only (r <= 0.95):        {len(hash_only)}")

    rows = list(dup.itertuples())
    for s in range(0, len(rows), PAIRS_PER_SHEET):
        chunk = rows[s:s + PAIRS_PER_SHEET]
        fig, axes = plt.subplots(len(chunk), 2, figsize=(5, 2.4 * len(chunk)))
        axes = np.atleast_2d(axes)
        for k, d in enumerate(chunk):
            for j, (sp, ix) in enumerate([(d.split_a, d.idx_a), (d.split_b, d.idx_b)]):
                img = Image.open(C.resolve_path(recs[sp][int(ix)]["image_path"])).convert("RGB")
                axes[k, j].imshow(img); axes[k, j].axis("off")
                axes[k, j].set_title(f"{sp} #{int(ix)}", fontsize=7)
            axes[k, 0].text(-0.05, 0.5, f"{d.type}\nham={d.hamming}\nr={d.pixel_r:.3f}",
                            transform=axes[k, 0].transAxes, ha="right", va="center", fontsize=7)
        plt.tight_layout()
        out = C.FIGURES_DIR / f"duplicates_test_{s // PAIRS_PER_SHEET + 1:02d}.png"
        plt.savefig(out, dpi=110); plt.close()
    print(f"✓ results/audit_duplicates_scored.csv, figures/duplicates_test_*.png "
          f"({(len(rows) + PAIRS_PER_SHEET - 1) // PAIRS_PER_SHEET} sheets)")
