"""
vl4_supplementary_tables.py -- Writes the LaTeX tables of the Supplementary Materials directly
from the CSV files of vl2_evaluate.py / vl3_figures_tables.py (no GPU, no PyTorch; seconds).

Output: supplementary/supp_tables.tex  (included by supplementary/supplementary.tex)
Run:    python vl4_supplementary_tables.py
"""

import ast
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
R = BASE / "results"
OUT = BASE / "supplementary"
OUT.mkdir(exist_ok=True)

NAME = {"pretrained": "BiomedCLIP (pretrained)", "clip": "BERD-CLIP, standard",
        "caption_aware": "BERD-CLIP, caption-aware", "label_aware": "BERD-CLIP, label-aware *",
        "dinov2": "DINOv2 (CLS)", "dinov2_clspatch": "DINOv2 (CLS + patch)",
        "swin_t_imagenet": "Swin-T (ImageNet)"}
SHORT = {"pretrained": "BiomedCLIP", "clip": "Standard", "caption_aware": "Caption-aware",
         "label_aware": "Label-aware *", "dinov2": "DINOv2", "dinov2_clspatch": "DINOv2 C+P",
         "swin_t_imagenet": "Swin-T"}
VL = ["clip", "caption_aware", "label_aware"]
ORDER = ["pretrained"] + VL + ["dinov2", "dinov2_clspatch", "swin_t_imagenet"]
FOOT = r"\noindent{\footnotesize{* Uses finding labels during alignment (supervised reference).}}"

METRICS = [("f1_macro", "F1, macro"), ("f1_micro", "F1, micro"), ("f1_weighted", "F1, weighted"),
           ("f1_samples", "F1, example-based"), ("precision_macro", "Precision, macro"),
           ("precision_micro", "Precision, micro"), ("recall_macro", "Recall (sensitivity), macro"),
           ("recall_micro", "Recall (sensitivity), micro"), ("specificity_macro", "Specificity, macro"),
           ("balanced_acc_macro", "Balanced accuracy, macro"), ("mcc_macro", "MCC, macro"),
           ("jaccard_samples", "Jaccard, example-based"), ("exact_match", "Exact match"),
           ("hamming_loss", r"Hamming loss $\downarrow$"), ("auc_roc_macro", "AUC-ROC, macro"),
           ("auc_roc_micro", "AUC-ROC, micro"), ("auprc_macro", "AUPRC, macro"),
           ("auprc_micro", "AUPRC, micro"), ("lrap", "LRAP"), ("ranking_loss", r"Ranking loss $\downarrow$"),
           ("coverage_error", r"Coverage error $\downarrow$"), ("brier_macro", r"Brier score $\downarrow$"),
           ("ece_macro", r"ECE $\downarrow$")]


def enc2variant(n):
    if n == "biomedclip_pt":
        return "pretrained"
    if n.startswith("vl_"):
        r = n[3:]
        return r.rsplit("_s", 1)[0] if r.rsplit("_s", 1)[-1].isdigit() else r
    return {"dinov2_frozen": "dinov2"}.get(n, n)


def msd(x, d=3):
    x = pd.Series(x).dropna()
    if not len(x):
        return "--"
    return (f"{x.mean():.{d}f} $\\pm$ {x.std(ddof=1):.{d}f}" if len(x) > 1 else f"{x.mean():.{d}f}")


def ci(lo, hi, d=3):
    return f"{lo:.{d}f}, {hi:.{d}f}".replace("-", "$-$")


def pfmt(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def sgn(x, d=3):
    return (f"+{x:.{d}f}" if x >= 0 else f"$-${abs(x):.{d}f}")


def table(label, caption, colspec, header, rows, foot="", wide=True, size=r"\footnotesize"):
    body = "\n".join(" & ".join(r) + r" \\" if not r == "MID" else r"\midrule" for r in rows)
    w0 = r"\begin{adjustwidth}{-\extralength}{0cm}" + "\n\\centering\n" if wide else "\\centering\n"
    w1 = "\\end{adjustwidth}\n" if wide else ""
    width = r"\fulllength" if wide else r"\textwidth"
    return (f"\\begin{{table}}[H]\n\\caption{{{caption}\\label{{{label}}}}}\n{w0}{size}\n"
            f"\\begin{{tabularx}}{{{width}}}{{{colspec}}}\n\\toprule\n{header} \\\\\n\\midrule\n"
            f"{body}\n\\bottomrule\n\\end{{tabularx}}\n{w1}{foot}\n\\end{{table}}\n")


def lettered(tex, num, letter):
    """Give a table a fixed lettered number (e.g. S6a) without disturbing the counter."""
    return (f"\\renewcommand{{\\thetable}}{{S{num}{letter}}}\n" + tex +
            f"\\setcounter{{table}}{{{num}}}\\renewcommand{{\\thetable}}{{S\\arabic{{table}}}}\n")


def tex_escape(s):
    return s.replace("{}", r"\{\}").replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


# ─────────────────────────────────────────────────────────────── S1 prompts
def prompts_from_vl_common():
    src = (BASE / "vl_common.py").read_text(encoding="utf-8")
    out = {}
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ("PROMPT_TEMPLATES", "CLASS_PHRASES", "BERD_PROMPTS",
                                      "LOCATION_TEMPLATES"):
                out[node.targets[0].id] = ast.literal_eval(node.value)
    return out


def s1():
    P = prompts_from_vl_common()
    tmpl = "; ".join(f"``{tex_escape(t)}''" for t in P["PROMPT_TEMPLATES"])
    rows = [[c, "; ".join(P["CLASS_PHRASES"][c]), "; ".join(P["BERD_PROMPTS"][c])]
            for c in sorted(P["CLASS_PHRASES"])]
    rows = [[tex_escape(x) for x in r] for r in rows]
    cap = ("Zero-shot prompt sets, fixed before any test-set evaluation. Generic phrases were inserted "
           f"into each of three templates ({tmpl}), giving six prompts per class; BERD-vocabulary phrases "
           "were used as complete prompts. Class embeddings are the re-normalised mean of the prompt "
           "embeddings. Location prompts: " +
           "; ".join(f"``{tex_escape(t)}''" for t in P["LOCATION_TEMPLATES"]) + ".")
    return table("tab:S1", cap, ">{\\raggedright\\arraybackslash}p{3.0cm}XX",
                 r"\textbf{Class} & \textbf{Generic Phrases} & \textbf{BERD-Vocabulary Prompts}", rows)


# ─────────────────────────────────────────────────────────────── S2 zero-shot paired
def s2():
    p = pd.read_csv(R / "zeroshot_paired_auprc.csv")
    lab = {"alignment_generic": "Aligned $-$ pretrained, generic prompts",
           "alignment_berd_vocab": "Aligned $-$ pretrained, BERD prompts",
           "vocabulary": "BERD $-$ generic prompts",
           "interaction": "Interaction (aligned vs.\\ pretrained)"}
    rows = []
    for t in ["alignment_generic", "alignment_berd_vocab", "vocabulary", "interaction"]:
        g = p[p.type == t].copy()
        g["v"] = g.variant.fillna("pretrained")
        g["o"] = g.v.map({v: i for i, v in enumerate(ORDER)})
        g = g.sort_values(["o", "seed"])
        if rows:
            rows.append("MID")
        for _, r in g.iterrows():
            seed = "--" if pd.isna(r.seed) else str(int(r.seed))
            rows.append([lab[t], SHORT[r.v], seed, sgn(r.delta), ci(r.lo, r.hi), pfmt(r.p)])
    return table("tab:S2", "Paired zero-shot comparisons of macro AUPRC on the test set for each "
                 "alignment objective and seed (patient-level bootstrap, 2000 resamples; unadjusted "
                 "$p$-values). The interaction is the BERD-minus-generic difference of the aligned model "
                 "minus that of pretrained BiomedCLIP.",
                 "lllCCC", r"\textbf{Comparison} & \textbf{Encoder} & \textbf{Seed} & "
                 r"\textbf{$\Delta$AUPRC} & \textbf{95\% CI} & \textbf{$p$}", rows, FOOT,
                 size=r"\scriptsize")


# ─────────────────────────────────────────────────────────────── S3 image retrieval
def s3():
    ri = pd.read_csv(R / "retrieval_image.csv")
    rows = []
    for v in ORDER:
        g = ri[ri.variant == v]
        if len(g):
            rows.append([NAME[v], str(len(g))] + [msd(g[c]) for c in
                                                   ["i2i_P@1", "i2i_P@5", "i2i_P@10", "i2i_mAP"]])
    return table("tab:S3", "Image-to-image retrieval within the test set; an image is relevant if it "
                 "shares at least one finding with the query (aligned models: mean $\\pm$ SD over seeds).",
                 "lCCCCC", r"\textbf{Encoder} & \textbf{Runs} & \textbf{P@1} & \textbf{P@5} & "
                 r"\textbf{P@10} & \textbf{mAP}", rows, FOOT)


# ─────────────────────────────────────────────────────────────── S4 fixed C
def s4():
    fc = pd.read_csv(R / "sensitivity_fewshot_fixedC.csv")
    fc["v"] = fc.encoder.map(enc2variant)
    fr = sorted(fc.fraction.unique())
    rows = []
    for v in ORDER:
        g = fc[fc.v == v]
        if len(g):
            rows.append([NAME[v]] + [msd(g[g.fraction == f].auprc_macro) for f in fr])
    return table("tab:S4", "Linear-probe macro AUPRC without any use of validation labels "
                 "($C = 1$; seed-42 aligned encoders; mean $\\pm$ SD over five patient subsets).",
                 "l" + "C" * len(fr), r"\textbf{Encoder} & " +
                 " & ".join(f"\\textbf{{{f * 100:g}\\%}}" for f in fr), rows, FOOT)


# ─────────────────────────────────────────────────────────────── S5 duplicates
def stem_label(st):
    import re
    seed = re.search(r"(?:_seed|_s)(\d+)", st)
    seed = seed.group(1) if seed else "42"
    frac = re.search(r"_frac(\d+)", st)
    frac = f"{frac.group(1)}\\%" if frac else "100\\%"
    if st.startswith("zeroshot_"):
        rest = st[len("zeroshot_"):]
        ps = "BERD" if rest.endswith("berd_vocab") else "generic"
        enc = rest.replace("_berd_vocab", "").replace("_generic", "")
        v = enc2variant(enc)
        return "Zero-shot", f"{SHORT[v]} ({ps})", "--" if v == "pretrained" else seed
    if st.startswith("probe_full_"):
        v = enc2variant(st[len("probe_full_"):])
        return "Linear probe, 100\\%", SHORT[v], "--" if v not in VL else seed
    init = st[len("sup_"):]
    init = re.sub(r"(_seed\d+)|(_frac\d+)", "", init)
    return f"Fine-tuning, {frac}", SHORT[init], seed


def s5():
    sd = pd.read_csv(R / "sensitivity_duplicates.csv")
    lab = sd.file.map(stem_label)
    sd["fam"], sd["model"], sd["seed"] = zip(*lab)
    fam_order = ["Zero-shot", "Linear probe, 100\\%", "Fine-tuning, 100\\%", "Fine-tuning, 25\\%",
                 "Fine-tuning, 10\\%"]
    sd["fo"] = sd.fam.map({f: i for i, f in enumerate(fam_order)})
    sd["mo"] = sd.model.map(lambda m: next((i for i, v in enumerate(ORDER) if SHORT[v] in m), 9))
    sd = sd.sort_values(["fo", "mo", "model", "seed"])
    rows, last = [], None
    for _, r in sd.iterrows():
        if last is not None and r.fam != last:
            rows.append("MID")
        last = r.fam
        f1 = ("--", "--") if pd.isna(r.get("f1_all")) else (f"{r.f1_all:.3f}", f"{r.f1_kept:.3f}")
        rows.append([r.fam, r.model, r.seed, f"{r.auprc_all:.3f}", f"{r.auprc_kept:.3f}",
                     sgn(r.d_auprc), f1[0], f1[1]])
    return table("tab:S5", "Sensitivity analysis: macro AUPRC and macro F1 (threshold 0.5) on all 316 "
                 "test images and after excluding the 31 test images flagged by the duplicate checks "
                 "(285 kept). Zero-shot scores are similarities, so F1 at 0.5 is not reported.",
                 "lllCCCCC", r"\textbf{Analysis} & \textbf{Model} & \textbf{Seed} & "
                 r"\textbf{AUPRC, All} & \textbf{AUPRC, Kept} & \textbf{$\Delta$} & "
                 r"\textbf{F1, All} & \textbf{F1, Kept}", rows, FOOT, size=r"\scriptsize")


# ─────────────────────────────────────────────────────────────── S6-S8 extended
def ext_table(label, caption, groups, data_fn, has_prob=True, foot=FOOT):
    cols = [name for name, _ in groups]
    rows = []
    for key, lab in METRICS:
        if not has_prob and key in ("brier_macro", "ece_macro"):
            continue
        rows.append([lab] + [data_fn(g, key) for _, g in groups])
    return table(label, caption, "l" + "C" * len(cols),
                 r"\textbf{Metric} & " + " & ".join(f"\\textbf{{{c}}}" for c in cols), rows, foot,
                 size=r"\scriptsize")


def s6():
    z = pd.read_csv(R / "zeroshot_summary.csv")
    out = []
    for ps, psn, lab in [("generic", "generic", "tab:S6a"), ("berd_vocab", "BERD-vocabulary", "tab:S6b")]:
        groups = [(SHORT[v], z[(z.variant == v) & (z.prompt_set == ps)]) for v in ["pretrained"] + VL]
        out.append(lettered(ext_table(lab, f"Extended zero-shot metrics with {psn} prompts (thresholds "
                             "selected on the validation subset; aligned models: mean $\\pm$ SD over three "
                             "seeds; $\\downarrow$ lower is better).", groups,
                             lambda g, k: msd(g[k]), has_prob=False), 6, lab[-1]))
    return "\n".join(out)


def s7():
    fs = pd.read_csv(R / "fewshot_all_runs.csv")
    full = fs[fs.fraction == 1.0]
    groups = [(SHORT[v], full[full.variant == v]) for v in ORDER if (full.variant == v).any()]
    return ext_table("tab:S7", "Extended metrics of linear probes trained with all labelled training "
                     "patients (aligned models: mean $\\pm$ SD over three seeds; C+P, class token plus "
                     "mean patch token; $\\downarrow$ lower is better).", groups,
                     lambda g, k: msd(g[k]))


def s8():
    su = pd.read_csv(R / "supervised_summary.csv")
    out = []
    for fr_list, lab, desc in [([1.0], "tab:S8a", "all labelled training patients"),
                               ([0.25, 0.10], "tab:S8b", "25\\% and 10\\% of the training patients")]:
        groups = []
        for fr in fr_list:
            for v in ["pretrained"] + VL:
                g = su[(su.init == v) & (su.fraction == fr)]
                if len(g):
                    nm = SHORT[v] + ("" if len(fr_list) == 1 else f", {fr * 100:g}\\%")
                    groups.append((nm, g))
        out.append(lettered(ext_table(lab, f"Extended metrics of fine-tuned models trained with {desc} "
                             "(threshold 0.5; mean $\\pm$ SD over three seeds; $\\downarrow$ lower is "
                             "better).", groups, lambda g, k: msd(g[k]),
                             foot=FOOT if lab.endswith("a") else ""), 8, lab[-1]))
    return "\n".join(out)


# ─────────────────────────────────────────────────────────────── S9 supervised paired
def s9():
    d = pd.read_csv(R / "supervised_paired_differences.csv")
    d["fo"] = d.fraction.map({1.0: 0, 0.25: 1, 0.1: 2})
    d["so"] = d.seed.map({42: 0, 7: 1, 123: 2})
    d = d.sort_values(["fo", "so", "init_1", "init_2"])
    rows, last = [], None
    for _, r in d.iterrows():
        if last is not None and r.fo != last:
            rows.append("MID")
        last = r.fo
        holm = "--" if pd.isna(r.dF1_p_holm) else pfmt(r.dF1_p_holm)
        rows.append([f"{r.fraction * 100:g}\\%", f"{SHORT[r.init_1]} $-$ {SHORT[r.init_2]}",
                     str(int(r.seed)), f"{sgn(r.dF1)} ({ci(r.dF1_lo, r.dF1_hi)})", pfmt(r.dF1_p), holm,
                     f"{sgn(r.dAUPRC)} ({ci(r.dAUPRC_lo, r.dAUPRC_hi)})", pfmt(r.dAUPRC_p)])
    return table("tab:S9", "Seed-level paired comparisons of fine-tuned models on the test set "
                 "(threshold 0.5; patient-level bootstrap, 2000 resamples). Holm adjustment was applied "
                 "to the three pre-specified comparisons with all labels in seed 42; other $p$-values "
                 "are unadjusted.", "l>{\\raggedright\\arraybackslash}Xcccccc",
                 r"\textbf{Labelled} & \textbf{Comparison} & \textbf{Seed} & "
                 r"\textbf{$\Delta$F1 (95\% CI)} & \textbf{$p$} & \textbf{$p_\mathrm{Holm}$} & "
                 r"\textbf{$\Delta$AUPRC (95\% CI)} & \textbf{$p$}", rows, "",
                 size=r"\scriptsize")


if __name__ == "__main__":
    parts = [s1(), s2(), s3(), s4(), s5(), s6(), s7(), s8(), s9()]
    (OUT / "supp_tables.tex").write_text("\n\n".join(parts), encoding="utf-8")
    print("✓ supplementary/supp_tables.tex")
