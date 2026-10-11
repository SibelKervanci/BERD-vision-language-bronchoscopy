"""
vl3_figures_tables.py -- Publication figures and tables, generated ONLY from the CSV files
written by vl2_evaluate.py (no model, no GPU; seconds). Every number in the manuscript should
come from the tables written here, so text, tables and figures stay consistent.

Aligned models are summarised as mean ± SD over the three alignment seeds (each seed first
averaged over its labelled subsets); single-run references show mean ± SD over subsets.

Outputs
  figures/Fig2_zeroshot_per_class.png   figures/Fig3_retrieval.png
  figures/Fig4_label_efficiency.png     figures/Fig5_supervised.png
  results/tables/Table*.csv  (formatted, ready for the manuscript)
  results/tables/TableS*.csv (supplementary: extended metrics)
Run: python vl3_figures_tables.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
R = BASE / "results"
FIG = BASE / "figures"
TAB = R / "tables"
FIG.mkdir(exist_ok=True); TAB.mkdir(exist_ok=True)

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                     "legend.fontsize": 8, "font.family": "DejaVu Sans"})

ORDER = ["pretrained", "clip", "caption_aware", "label_aware",
         "dinov2", "dinov2_clspatch", "swin_t_imagenet"]
NAME = {"pretrained": "BiomedCLIP (pretrained)", "clip": "BERD-CLIP, standard",
        "caption_aware": "BERD-CLIP, caption-aware", "label_aware": "BERD-CLIP, label-aware*",
        "dinov2": "DINOv2 (CLS)", "dinov2_clspatch": "DINOv2 (CLS+patch)",
        "swin_t_imagenet": "Swin-T (ImageNet)"}
COL = {"pretrained": "#4C72B0", "clip": "#DD8452", "caption_aware": "#55A868",
       "label_aware": "#C44E52", "dinov2": "#8172B3", "dinov2_clspatch": "#937860",
       "swin_t_imagenet": "#DA8BC3"}
LS = {v: ("-" if v in ("pretrained", "clip", "caption_aware") else "--") for v in ORDER}
VL = ["clip", "caption_aware", "label_aware"]


def msd(x, d=3):
    x = pd.Series(x).dropna()
    if len(x) == 0:
        return ""
    return f"{x.mean():.{d}f} ± {x.std(ddof=1):.{d}f}" if len(x) > 1 else f"{x.mean():.{d}f}"


def seed_stats(df, keys, cols):
    """mean and SD over seeds (rows with seed NaN are single runs)."""
    g = df.groupby(keys)[cols]
    return g.mean(), g.std(ddof=1)


def enc2variant(n):
    """encoder name -> variant (pretrained / clip / caption_aware / label_aware / references)."""
    if n == "biomedclip_pt":
        return "pretrained"
    if n.startswith("vl_"):
        return n[3:].split("_s")[0] if "_s" in n[3:] and n.rsplit("_s", 1)[-1].isdigit() else n[3:]
    return {"dinov2_frozen": "dinov2", "dinov2_clspatch": "dinov2_clspatch",
            "swin_t_imagenet": "swin_t_imagenet"}.get(n, n)


def savefig(fig, name):
    fig.savefig(FIG / name, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"  ✓ figures/{name}")


def savetab(df, name):
    df.to_csv(TAB / name, index=False); print(f"  ✓ results/tables/{name}")


# ════════════════════════════════════════════════════════════════
#  TABLE 2 + FIGURE 2: ZERO-SHOT
# ════════════════════════════════════════════════════════════════

def zeroshot():
    z = pd.read_csv(R / "zeroshot_summary.csv")
    rows = []
    for v in ["pretrained"] + VL:
        for ps, psn in [("generic", "Generic"), ("berd_vocab", "BERD vocabulary")]:
            g = z[(z.variant == v) & (z.prompt_set == ps)]
            rows.append({"Encoder": NAME[v], "Prompts": psn, "Seeds": len(g),
                         "AUPRC": msd(g.auprc_macro), "AUC-ROC": msd(g.auc_roc_macro),
                         "F1 (val. thr.)": msd(g.f1_macro), "MCC": msd(g.mcc_macro),
                         "LRAP": msd(g.lrap)})
    savetab(pd.DataFrame(rows), "Table2_zeroshot.csv")

    p = pd.read_csv(R / "zeroshot_paired_auprc.csv")
    rows = []
    for t, tn in [("alignment_generic", "Aligned − pretrained (generic prompts)"),
                  ("alignment_berd_vocab", "Aligned − pretrained (BERD prompts)"),
                  ("vocabulary", "BERD − generic prompts"),
                  ("interaction", "Interaction (aligned vs. pretrained)")]:
        for v in (["pretrained"] if t == "vocabulary" else []) + VL:
            g = p[(p.type == t) & (p.variant == v)]
            if g.empty:
                continue
            rows.append({"Comparison": tn, "Encoder": NAME[v], "Seeds": len(g),
                         "ΔAUPRC": msd(g.delta), "Seeds with 95% CI excluding 0":
                         f"{int(((g.lo > 0) | (g.hi < 0)).sum())}/{len(g)}",
                         "p range": f"{g.p.min():.3f}–{g.p.max():.3f}" if len(g) > 1 else f"{g.p.iloc[0]:.3f}"})
    savetab(pd.DataFrame(rows), "Table2b_zeroshot_paired.csv")

    pc = pd.read_csv(R / "zeroshot_per_class.csv").dropna(subset=["auprc"])
    pc["variant"] = pc.encoder.map(enc2variant)
    pc = pc[pc.prompt_set == "berd_vocab"]
    m = pc.groupby(["variant", "class"]).auprc.mean().unstack(0)
    s = pc.groupby(["variant", "class"]).auprc.std(ddof=1).unstack(0)
    info = pc.drop_duplicates("class").set_index("class")[["n_pos_test", "prevalence_test",
                                                           "prevalence_train"]]
    info = info.sort_values("prevalence_train", ascending=False)
    m, s = m.reindex(info.index), s.reindex(info.index)
    fig, ax = plt.subplots(figsize=(11, 3.8))
    x = np.arange(len(info)); w = 0.2
    for k, v in enumerate(["pretrained"] + VL):
        ax.bar(x + (k - 1.5) * w, m[v] / info.prevalence_test, w,
               yerr=(s[v] / info.prevalence_test) if v in VL else None,
               color=COL[v], label=NAME[v], error_kw={"lw": 0.6, "capsize": 1.5})
    ax.axhline(1, color="k", lw=0.8, ls="--")
    for i, n in enumerate(info.n_pos_test):
        if n < 5:
            ax.axvspan(i - 0.5, i + 0.5, color="0.92", zorder=0)
    ax.set_yscale("log"); ax.set_xlim(-0.6, len(info) - 0.4)
    ax.set_xticks(x); ax.set_xticklabels([f"{c} ({n})" for c, n in
                                          zip(info.index, info.n_pos_test)], rotation=55, ha="right")
    ax.set_ylabel("AUPRC / test prevalence")
    ax.legend(ncol=4, loc="upper left", frameon=False)
    savefig(fig, "Fig2_zeroshot_per_class.png")


# ════════════════════════════════════════════════════════════════
#  TABLE 3 + FIGURE 3: RETRIEVAL
# ════════════════════════════════════════════════════════════════

def retrieval():
    r = pd.read_csv(R / "retrieval.csv")
    met = ["i2t_R@1", "i2t_R@5", "i2t_R@10", "t2i_R@1", "t2i_R@5", "t2i_R@10", "rsum",
           "i2t_medR", "t2i_medR"]
    rows = []
    for sub, sn in [("all", "All queries"), ("excl_frequent", "Excluding frequent captions"),
                    ("novel_captions", "Captions unseen in training")]:
        for v in ["pretrained"] + VL:
            g = r[(r.variant == v) & (r.subset == sub)]
            rows.append({"Query set": sn, "Encoder": NAME[v], "Seeds": len(g),
                         **{c: msd(g[c], 1 if "medR" in c else 3) for c in met},
                         "Image queries": int(g.n_img_queries.iloc[0]),
                         "Text queries": int(g.n_txt_queries.iloc[0])})
    savetab(pd.DataFrame(rows), "Table3_retrieval.csv")

    ri = pd.read_csv(R / "retrieval_image.csv")
    rows = [{"Encoder": NAME[v], "Seeds": len(g), **{c: msd(g[c]) for c in
             ["i2i_P@1", "i2i_P@5", "i2i_P@10", "i2i_mAP"]}}
            for v in ORDER if not (g := ri[ri.variant == v]).empty]
    savetab(pd.DataFrame(rows), "TableS_image_retrieval.csv")

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
    labels = ["I→T R@1", "I→T R@5", "I→T R@10", "T→I R@1", "T→I R@5", "T→I R@10"]
    for ax, (sub, sn) in zip(axes, [("all", "All queries"),
                                    ("excl_frequent", "Excluding frequent captions"),
                                    ("novel_captions", "Captions unseen in training")]):
        x = np.arange(6); w = 0.2
        for k, v in enumerate(["pretrained"] + VL):
            g = r[(r.variant == v) & (r.subset == sub)][met[:6]]
            ax.bar(x + (k - 1.5) * w, g.mean(), w, yerr=g.std(ddof=1) if len(g) > 1 else None,
                   color=COL[v], label=NAME[v], error_kw={"lw": 0.6, "capsize": 1.5})
        ax.set_title(sn); ax.set_xticks(x); ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Recall")
    h, l = axes[0].get_legend_handles_labels()
    for ax, letter in zip(axes, "ABC"):
        ax.text(-0.08, 1.06, letter, transform=ax.transAxes, fontweight="bold", fontsize=11)
    fig.legend(h, l, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.2), frameon=False)
    savefig(fig, "Fig3_retrieval.png")


# ════════════════════════════════════════════════════════════════
#  TABLE 4 + FIGURE 4: LABEL EFFICIENCY (LINEAR PROBES)
# ════════════════════════════════════════════════════════════════

def fewshot():
    fs = pd.read_csv(R / "fewshot_all_runs.csv")
    fs["seed"] = fs.seed.fillna(-1)
    per = fs.groupby(["variant", "seed", "fraction"])[["auprc_macro", "f1_macro"]].mean().reset_index()
    rows = []
    for v in ORDER:
        row = {"Encoder": NAME[v]}
        for fr in sorted(fs.fraction.unique()):
            if v in VL:
                g = per[(per.variant == v) & (per.fraction == fr)].auprc_macro
            else:
                g = fs[(fs.variant == v) & (fs.fraction == fr)].auprc_macro
            row[f"{fr * 100:g}%"] = msd(g)
        rows.append(row)
    savetab(pd.DataFrame(rows), "Table4_fewshot_auprc.csv")
    rows = []
    for v in ORDER:
        row = {"Encoder": NAME[v]}
        for fr in sorted(fs.fraction.unique()):
            g = (per[(per.variant == v) & (per.fraction == fr)].f1_macro if v in VL
                 else fs[(fs.variant == v) & (fs.fraction == fr)].f1_macro)
            row[f"{fr * 100:g}%"] = msd(g)
        rows.append(row)
    savetab(pd.DataFrame(rows), "Table4b_fewshot_f1.csv")
    w = pd.read_csv(R / "seeds_fewshot_wilcoxon.csv")
    w["p_wilcoxon"] = w.p_wilcoxon.map(lambda p: "<0.001" if p < 0.001 else f"{p:.3f}")
    w["mean_diff"] = w.mean_diff.round(4)
    savetab(w, "Table4c_fewshot_wilcoxon.csv")
    fc = pd.read_csv(R / "sensitivity_fewshot_fixedC.csv")
    t = fc.groupby(["encoder", "fraction"]).auprc_macro.agg(["mean", "std"])
    t = t.apply(lambda r: f"{r['mean']:.3f}" + ("" if np.isnan(r["std"]) else f" ± {r['std']:.3f}"),
                axis=1).unstack(1)
    t.columns = [f"{c * 100:g}%" for c in t.columns]
    savetab(t.reset_index(), "TableS_fewshot_fixedC.csv")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, (col, lab) in zip(axes, [("auprc_macro", "Macro AUPRC"),
                                     ("f1_macro", "Macro F1 (validation thresholds)")]):
        for v in ORDER:
            if v in VL:
                d = per[per.variant == v].groupby("fraction")[col].agg(["mean", "std"])
            else:
                d = fs[fs.variant == v].groupby("fraction")[col].agg(["mean", "std"])
            ax.errorbar(d.index * 100, d["mean"], yerr=d["std"].fillna(0), marker="o", ms=3.5,
                        capsize=2, lw=1.3, ls=LS[v], color=COL[v], label=NAME[v])
        ax.set_xscale("log"); ax.set_xticks([1, 5, 10, 25, 100])
        ax.set_xticklabels(["1", "5", "10", "25", "100"])
        ax.set_xlabel("Labelled training patients (%)"); ax.set_ylabel(lab); ax.grid(alpha=0.3)
    for ax, letter in zip(axes, "AB"):
        ax.text(-0.12, 1.04, letter, transform=ax.transAxes, fontweight="bold", fontsize=11)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.13), frameon=False)
    savefig(fig, "Fig4_label_efficiency.png")


# ════════════════════════════════════════════════════════════════
#  TABLE 5 + FIGURE 5: SUPERVISED AND LABEL-LIMITED FINE-TUNING
# ════════════════════════════════════════════════════════════════

def supervised():
    su = pd.read_csv(R / "supervised_summary.csv")
    su = su[su.init.isin(["pretrained"] + VL)]
    rows = []
    for fr in [1.0, 0.25, 0.10]:
        for v in ["pretrained"] + VL:
            g = su[(su.init == v) & (su.fraction == fr)]
            if g.empty:
                continue
            rows.append({"Labelled patients": f"{fr * 100:g}%", "Initialisation": NAME[v],
                         "Seeds": len(g), "F1": msd(g.f1_macro), "MCC": msd(g.mcc_macro),
                         "AUPRC": msd(g.auprc_macro), "AUC-ROC": msd(g.auc_roc_macro),
                         "Exact match": msd(g.exact_match), "Brier": msd(g.brier_macro),
                         "ECE": msd(g.ece_macro)})
    savetab(pd.DataFrame(rows), "Table5_supervised.csv")
    d = pd.read_csv(R / "supervised_paired_differences.csv")
    keep = ["family", "init_1", "init_2", "seed", "fraction", "dF1", "dF1_lo", "dF1_hi", "dF1_p",
            "dF1_p_holm", "dAUPRC", "dAUPRC_lo", "dAUPRC_hi", "dAUPRC_p"]
    savetab(d[keep].round(3), "Table5b_supervised_paired.csv")

    # dot-and-whisker (no truncated bars): mean ± SD over three seeds
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.1), gridspec_kw={"wspace": 0.35})
    mets = [("f1_macro", "Macro F1"), ("auprc_macro", "Macro AUPRC"),
            ("exact_match", "Exact match"), ("brier_macro", "Brier score (lower is better)")]
    inits = ["pretrained", "clip", "caption_aware", "label_aware"]
    for ax, (c, lab), letter in zip(axes, mets, "ABCD"):
        for k, v in enumerate(inits):
            xs, mu, sd = [], [], []
            for xi, fr in enumerate([0.10, 0.25, 1.0]):
                g = su[(su.init == v) & (su.fraction == fr)][c]
                if len(g):
                    xs.append(xi + (k - 1.5) * 0.14); mu.append(g.mean())
                    sd.append(g.std(ddof=1) if len(g) > 1 else 0)
            ax.errorbar(xs, mu, yerr=sd, fmt="o", ms=4.5, capsize=2.5, lw=1.1, color=COL[v],
                        label=NAME[v])
        ax.set_xticks(range(3)); ax.set_xticklabels(["10%", "25%", "100%"])
        ax.set_xlim(-0.5, 2.5); ax.set_xlabel("Labelled training patients")
        ax.set_title(lab); ax.grid(alpha=0.3)
        ax.text(-0.2, 1.06, letter, transform=ax.transAxes, fontweight="bold", fontsize=11)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.16), frameon=False)
    savefig(fig, "Fig5_supervised.png")


# ════════════════════════════════════════════════════════════════
#  TABLE 6 (LOCATION), AUDIT, SUPPLEMENTARY EXTENDED METRICS
# ════════════════════════════════════════════════════════════════

def other():
    loc = pd.read_csv(R / "location.csv")
    loc["Encoder"] = loc.encoder.map(lambda n: {"biomedclip_pt": NAME["pretrained"],
                                                "vl_clip": NAME["clip"],
                                                "vl_caption_aware": NAME["caption_aware"],
                                                "vl_label_aware": NAME["label_aware"],
                                                "dinov2_frozen": NAME["dinov2"],
                                                "dinov2_clspatch": NAME["dinov2_clspatch"],
                                                "swin_t_imagenet": NAME["swin_t_imagenet"]}.get(n, n))
    t = loc[["Encoder", "method", "accuracy", "top3_accuracy", "balanced_accuracy", "f1_macro",
             "cohen_kappa"]].round(3)
    savetab(t, "Table6_location.csv")

    if (R / "audit_duplicates_scored.csv").exists():
        a = pd.read_csv(R / "audit_duplicates_scored.csv")
        t = a.groupby(["split_a", "type"]).agg(pairs=("pixel_r", "size"),
                                               test_images=("idx_b", "nunique"),
                                               r_min=("pixel_r", "min"),
                                               r_median=("pixel_r", "median"),
                                               r_max=("pixel_r", "max")).round(3).reset_index()
        savetab(t, "TableS_duplicates.csv")
    sd = pd.read_csv(R / "sensitivity_duplicates.csv")
    savetab(sd.round(4), "TableS_sensitivity_duplicates.csv")

    ext = ["f1_macro", "f1_micro", "f1_weighted", "f1_samples", "precision_macro", "recall_macro",
           "specificity_macro", "balanced_acc_macro", "mcc_macro", "jaccard_samples", "exact_match",
           "hamming_loss", "auc_roc_macro", "auc_roc_micro", "auprc_macro", "auprc_micro", "lrap",
           "ranking_loss", "coverage_error", "brier_macro", "ece_macro"]
    su = pd.read_csv(R / "supervised_summary.csv")
    rows = []
    for (v, fr), g in su[su.init.isin(["pretrained"] + VL)].groupby(["init", "fraction"]):
        rows.append({"Initialisation": NAME[v], "Labelled": f"{fr * 100:g}%",
                     **{c: msd(g[c]) for c in ext if c in g}})
    savetab(pd.DataFrame(rows), "TableS_supervised_extended.csv")
    fs = pd.read_csv(R / "fewshot_all_runs.csv")
    full = fs[fs.fraction == 1.0]
    rows = [{"Encoder": NAME[v], **{c: msd(g[c]) for c in ext if c in g}}
            for v in ORDER if not (g := full[full.variant == v]).empty]
    savetab(pd.DataFrame(rows), "TableS_probe100_extended.csv")
    z = pd.read_csv(R / "zeroshot_summary.csv")
    rows = [{"Encoder": NAME[v], "Prompts": ps, **{c: msd(g[c]) for c in ext if c in g}}
            for (v, ps), g in z.groupby(["variant", "prompt_set"])]
    savetab(pd.DataFrame(rows), "TableS_zeroshot_extended.csv")


if __name__ == "__main__":
    zeroshot(); retrieval(); fewshot(); supervised(); other()
    print("\n✓ figures/Fig2-5_*.png and results/tables/*.csv")
