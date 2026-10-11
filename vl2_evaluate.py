"""
vl2_evaluate.py -- Evaluation of the BERD vision-language study. No training of deep
networks (only logistic-regression probes on frozen features).

Sections
  A) FEATURES   : frozen image features (train/val/test) for every encoder, cached.
  B) ZERO-SHOT  : multi-label finding recognition from text prompts only (no labels for
                  training). Threshold-free macro AUPRC / AUC-ROC (primary); F1/MCC with
                  per-class thresholds selected on the VALIDATION subset (secondary).
                  Per-class AUPRC lift over prevalence, incl. the rare classes.
  C) RETRIEVAL  : image->text and text->image Recall@1/5/10 on the test set
                  (unique-caption protocol), on all queries and excluding frequent captions.
  D) FEW-SHOT   : linear probes with 1/5/10/25/100 % of the training PATIENTS
                  (5 random patient subsets per fraction), C selected on validation.
  E) LOCATION   : anatomical location (single-label): zero-shot and linear probe.
  F) SUPERVISED : fine-tuned classifiers initialised from aligned vs. pretrained BiomedCLIP
                  (same test set, patient-level bootstrap, Holm-corrected primary
                  comparisons). Optional external reference rows from EXTERNAL_PREDS_DIR.
  G) FIGURES

Outputs: results/*.csv, figures/*.png
Run: python vl2_evaluate.py
"""

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import vl_common as C
from vl_metrics import (norm_text, template_captions, retrieval_metrics, unique_caption_index,
                        thresholds_from_val_scores, sample_patient_subset, linear_probe,
                        multiclass_probe, macro_auprc, cluster_groups, boot_paired_metric,
                        auprc_per_class, full_metrics,
                        thresholds_from_val, holm, boot_single, boot_paired,
                        extended_metrics, image_retrieval_metrics)


# ╔══════════════════════════════════════════════════════════════╗
# ║  SETTINGS                                                     ║
# ╚══════════════════════════════════════════════════════════════╝

RUN_FEATURES   = True
RUN_ZERO_SHOT  = True
RUN_RETRIEVAL  = True
RUN_FEW_SHOT   = True
RUN_LOCATION   = True
RUN_SUPERVISED = True
RUN_FIGURES    = True
RUN_SEED_SUMMARY = True
RUN_SENSITIVITY  = True      # duplicates excluded, fixed-C probes (needs vl0_leakage_audit.py)
FORCE_FEATURES = False
FORCE_FEW_SHOT = False       # False: encoders already in results/fewshot_all_runs.csv are reused
PROBE_SEED     = 42          # random_state of the logistic regressions

SEEDS = [42, 7, 123]          # alignment / supervised seeds to evaluate (missing ones are skipped)
VARIANTS = ["clip", "caption_aware", "label_aware"]
VNICE = {"clip": "BERD-CLIP (standard)", "caption_aware": "BERD-CLIP (caption-aware)",
         "label_aware": "BERD-CLIP (label-aware)"}


def vl_name(v, sd):
    return f"vl_{v}" if sd == 42 else f"vl_{v}_s{sd}"


# encoder name -> (kind, checkpoint or None); META: name -> (family/variant, seed)
ENCODERS = {"biomedclip_pt": ("clip", None)}                    # no BERD training
META = {"biomedclip_pt": ("pretrained", None)}
for _sd in SEEDS:
    for _v in VARIANTS:
        ENCODERS[vl_name(_v, _sd)] = ("clip", C.CKPT_VL / f"vl_{_v}_seed{_sd}_best.pt")
        META[vl_name(_v, _sd)] = (_v, _sd)
ENCODERS.update({"dinov2_frozen": ("dinov2", None),              # vision-only references
                 "dinov2_clspatch": ("dinov2_clspatch", None),   # DINOv2 linear-probe protocol
                 "swin_t_imagenet": ("swin_t_imagenet", None)})
META.update({"dinov2_frozen": ("dinov2", None), "dinov2_clspatch": ("dinov2_clspatch", None),
             "swin_t_imagenet": ("swin_t_imagenet", None)})
LABEL_FREE = {n for n, (v, _) in META.items() if v in ("pretrained", "clip", "caption_aware")}
NICE = {"biomedclip_pt": "BiomedCLIP (pretrained)", "dinov2_frozen": "DINOv2 (frozen)",
        "dinov2_clspatch": "DINOv2 CLS+patch (frozen)",
        "swin_t_imagenet": "Swin-T ImageNet (frozen)"}
for _n, (_v, _sd) in META.items():
    if _v in VNICE:
        NICE[_n] = VNICE[_v] + ("" if _sd == 42 else f" s{_sd}")
FIG_ENCODERS = [n for n, (_, sd) in META.items() if sd in (None, 42)]   # figures: seed 42 only

FRACTIONS  = [0.01, 0.05, 0.10, 0.25, 1.0]
N_REPEATS  = 5
C_GRID     = (0.01, 0.1, 1.0, 10.0)
N_JOBS     = 8               # parallel logistic regressions (CPU cores)
LOC_MIN_TRAIN = 20           # locations with fewer training images are not evaluated

# supervised classifiers (section F): initialisation x seed x labelled fraction
SUP_INITS = ["pretrained", "caption_aware", "clip", "label_aware"]
SUP_FRACS = [1.0, 0.10, 0.25]
INIT_NICE = {"pretrained": "pretrained init", "caption_aware": "caption-aware init",
             "clip": "standard CLIP init", "label_aware": "label-aware init"}
# pre-specified primary comparisons (Holm-corrected within seed 42, all labels; primary
# endpoint macro F1); macro AUPRC differences are reported as secondary endpoints
PRIMARY = [("caption_aware", "pretrained"), ("clip", "pretrained"), ("caption_aware", "clip")]
LL_COMPARISONS = [("caption_aware", "pretrained"), ("clip", "pretrained")]
# optional external references: file stem in C.EXTERNAL_PREDS_DIR -> display name
EXTERNAL_REFS = {"hybrid_eff_swin__fixed": "Hybrid-EffSwin (external)",
                 "dinov2_vitb14__extra": "DINOv2 (external)"}


# ╔══════════════════════════════════════════════════════════════╗
# ║  SETUP                                                        ║
# ╚══════════════════════════════════════════════════════════════╝

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
label2idx, idx2label, TR, VA, TE = C.get_splits()
CAP_KEY = C.detect_key(TR, C.CAPTION_KEYS)
LOC_KEY = C.detect_key(TR, C.LOCATION_KEYS)
Ytr, Yva, Yte = (C.label_matrix(r, label2idx) for r in (TR, VA, TE))
Ptr, Pte = C.patient_ids(TR), C.patient_ids(TE)
GROUPS = cluster_groups(Pte)
MASK = Yte.sum(0) > 0                   # 21 evaluable classes (as in the first study)
print(f"[Setup] train {len(TR)} | val {len(VA)} | test {len(TE)} | "
      f"evaluable classes {MASK.sum()}/{len(MASK)} | caption key '{CAP_KEY}' | location '{LOC_KEY}'")


def available(name):
    kind, ck = ENCODERS[name]
    return ck is None or ck.exists()


def get_model(name):
    kind, ck = ENCODERS[name]
    if kind == "clip":
        return (C.load_biomedclip(device) if ck is None else C.load_aligned(ck, device))
    return C.FrozenVisionEncoder(kind).to(device), None


def metrics_block(y, s, thr, mask=MASK, is_prob=True):
    """All multi-label metrics (vl_metrics.extended_metrics); calibration only for probabilities."""
    return extended_metrics(y, s, thr, mask, is_prob=is_prob)


def save(df, name):
    df.to_csv(C.RESULTS_DIR / name, index=False); print(f"  ✓ results/{name}")


# ╔══════════════════════════════════════════════════════════════╗
# ║  A) FEATURES                                                  ║
# ╚══════════════════════════════════════════════════════════════╝

def extract_features():
    for name, (kind, _) in ENCODERS.items():
        files = [C.FEAT_CACHE / f"{name}_{s}.npy" for s in ("train", "val", "test")]
        if all(f.exists() for f in files) and not FORCE_FEATURES:
            continue
        if not available(name):
            print(f"  [features] {name}: checkpoint missing -> skipped"); continue
        try:
            model, _ = get_model(name)
        except Exception as e:
            print(f"  [features] {name} skipped: {type(e).__name__}: {e}"); continue
        tf = C.image_tf("test", "clip" if kind == "clip" else "imagenet")
        for recs, f in zip((TR, VA, TE), files):
            X = C.encode_images(model, C.make_loader(C.ImageDataset(recs, tf)), device)
            np.save(f, X)
        print(f"  [features] {name}: done")
        del model; torch.cuda.empty_cache()


def feats(name):
    try:
        return [np.load(C.FEAT_CACHE / f"{name}_{s}.npy") for s in ("train", "val", "test")]
    except FileNotFoundError:
        return None


CLIP_NAMES = [n for n, (k, _) in ENCODERS.items() if k == "clip"]


# ╔══════════════════════════════════════════════════════════════╗
# ║  B) ZERO-SHOT FINDINGS                                        ║
# ╚══════════════════════════════════════════════════════════════╝

def zero_shot():
    rows, per_class, paired = [], [], []
    prev_tr = Ytr.mean(0)
    for name in CLIP_NAMES:
        F_ = feats(name)
        if F_ is None or not available(name):
            continue
        model, tok = get_model(name)
        Ts = {ps: C.class_text_embeddings(model, tok, idx2label, device, ps) for ps in C.PROMPT_SETS}
        del model; torch.cuda.empty_cache()
        for ps, T in Ts.items():
            sv, st = F_[1] @ T.T, F_[2] @ T.T
            np.savez_compressed(C.PRED_CACHE / f"zeroshot_{name}_{ps}.npz", val_scores=sv, test_scores=st)
            thr = thresholds_from_val_scores(Yva, sv)
            m = metrics_block(Yte, st, thr, is_prob=False)   # cosine scores
            ci = boot_single(Yte, st, thr, GROUPS)            # F1, MCC, AUPRC (patient bootstrap)
            rows.append({"encoder": name, "variant": META[name][0], "seed": META[name][1],
                         "prompt_set": ps, "label_free": name in LABEL_FREE, **m,
                         "auprc_ci": f"[{ci['auprc'][0]:.3f}, {ci['auprc'][1]:.3f}]",
                         "f1_ci": f"[{ci['f1'][0]:.3f}, {ci['f1'][1]:.3f}]"})
            ap = auprc_per_class(Yte, st, MASK)
            for c, lab in enumerate(idx2label):
                prev = Yte[:, c].mean()
                per_class.append({"encoder": name, "prompt_set": ps, "class": lab,
                                  "n_pos_test": int(Yte[:, c].sum()), "prevalence_test": float(prev),
                                  "prevalence_train": float(prev_tr[c]), "auprc": ap[c],
                                  "lift": ap[c] / prev if np.isfinite(ap[c]) and prev > 0 else np.nan})
            print(f"  [zero-shot] {name:18s} {ps:10s}: AUPRC {m['auprc_macro']:.3f} | "
                  f"AUC {m['auc_roc_macro']:.3f} | F1(val-thr) {m['f1_macro']:.3f}")
    save(pd.DataFrame(rows), "zeroshot_summary.csv")
    save(pd.DataFrame(per_class), "zeroshot_per_class.csv")
    # paired AUPRC differences (patient bootstrap)
    def load(n, ps):
        f = C.PRED_CACHE / f"zeroshot_{n}_{ps}.npz"
        return np.load(f)["test_scores"] if f.exists() else None
    def paired_row(label, kind, name, S_a, S_b):
        S = {"a": S_a, "b": S_b}
        r = boot_paired_metric(lambda idx, w: macro_auprc(Yte[idx], S[w][idx]), "a", "b", GROUPS)
        v, sd = META.get(name, (None, None))
        return {"comparison": label, "type": kind, "variant": v, "seed": sd, **r}

    for name in CLIP_NAMES:
        if name == "biomedclip_pt":
            continue
        for ps in C.PROMPT_SETS:                          # alignment effect, per prompt set
            a, b = load(name, ps), load("biomedclip_pt", ps)
            if a is not None and b is not None:
                paired.append(paired_row(f"{name} - biomedclip_pt ({ps})", "alignment_" + ps,
                                         name, a, b))
    for name in CLIP_NAMES:                               # prompt-vocabulary effect, per encoder
        a, b = load(name, "berd_vocab"), load(name, "generic")
        if a is not None and b is not None:
            paired.append(paired_row(f"{name}: berd_vocab - generic", "vocabulary", name, a, b))
    # interaction: (aligned berd - aligned generic) - (pretrained berd - pretrained generic)
    pb, pg = load("biomedclip_pt", "berd_vocab"), load("biomedclip_pt", "generic")
    if pb is not None and pg is not None:
        for name in CLIP_NAMES:
            if name == "biomedclip_pt":
                continue
            ab, ag = load(name, "berd_vocab"), load(name, "generic")
            if ab is None or ag is None:
                continue
            S = {"a": (ab, ag), "b": (pb, pg)}
            fn = lambda idx, w: (macro_auprc(Yte[idx], S[w][0][idx]) -
                                 macro_auprc(Yte[idx], S[w][1][idx]))
            r = boot_paired_metric(fn, "a", "b", GROUPS)
            v, sd = META[name]
            paired.append({"comparison": f"interaction: {name} vs biomedclip_pt (berd - generic)",
                           "type": "interaction", "variant": v, "seed": sd, **r})
    if paired:
        df = pd.DataFrame(paired); save(df, "zeroshot_paired_auprc.csv")
        print(df.round(3).to_string(index=False))


# ╔══════════════════════════════════════════════════════════════╗
# ║  C) RETRIEVAL                                                 ║
# ╚══════════════════════════════════════════════════════════════╝

def retrieval():
    caps_te = C.captions_of(TE, CAP_KEY)
    uniq, ids = unique_caption_index(caps_te)
    tmpl = template_captions(C.captions_of(TR, CAP_KEY), C.TEMPLATE_MIN_COUNT)
    img_nt = np.array([norm_text(c) not in tmpl for c in caps_te])
    cap_nt = np.array([u not in tmpl for u in uniq])
    tr_caps = {norm_text(c) for c in C.captions_of(TR, CAP_KEY)}
    img_nov = np.array([norm_text(c) not in tr_caps for c in caps_te])   # caption never trained on
    cap_nov = np.array([u not in tr_caps for u in uniq])
    print(f"  [retrieval] test captions never seen in training: {cap_nov.sum()} unique, "
          f"{img_nov.sum()} images")
    print(f"  [retrieval] test: {len(TE)} images, {len(uniq)} unique captions, "
          f"{(~img_nt).sum()} images with frequent captions")
    rows = []
    for name in CLIP_NAMES:
        F_ = feats(name)
        if F_ is None or not available(name):
            continue
        model, tok = get_model(name)
        cap_f = C.encode_texts(model, tok, uniq, device)
        del model; torch.cuda.empty_cache()
        for subset, qi, qc in [("all", None, None), ("excl_frequent", img_nt, cap_nt),
                               ("novel_captions", img_nov, cap_nov)]:
            r = retrieval_metrics(F_[2], cap_f, ids, query_img_mask=qi, query_cap_mask=qc)
            rows.append({"encoder": name, "variant": META[name][0], "seed": META[name][1],
                         "subset": subset, **r})
        print(f"  [retrieval] {name}: R-sum {rows[-3]['rsum']:.3f} (all) | "
              f"{rows[-2]['rsum']:.3f} (excl. frequent) | {rows[-1]['rsum']:.3f} (novel captions)")
    save(pd.DataFrame(rows), "retrieval.csv")
    # image-to-image retrieval within the test set (relevant = shares >= 1 finding), all encoders
    rows = []
    for name in ENCODERS:
        F_ = feats(name)
        if F_ is None:
            continue
        rows.append({"encoder": name, "variant": META[name][0], "seed": META[name][1],
                     **image_retrieval_metrics(F_[2], Yte)})
    if rows:
        save(pd.DataFrame(rows), "retrieval_image.csv")


# ╔══════════════════════════════════════════════════════════════╗
# ║  D) FEW-SHOT LINEAR PROBE                                     ║
# ╚══════════════════════════════════════════════════════════════╝

def few_shot():
    rows, old = [], None
    f_old = C.RESULTS_DIR / "fewshot_all_runs.csv"
    if f_old.exists() and not FORCE_FEW_SHOT:
        old = pd.read_csv(f_old)
        if "lrap" not in old.columns:                     # file from before the extended metrics
            print("  [probe] cached runs lack the extended metrics -> recomputing all encoders")
            old = None
        else:
            old = old[old.encoder.isin(list(ENCODERS))]
    if old is not None:
        print(f"  [probe] reusing {old.encoder.nunique()} encoders from fewshot_all_runs.csv")
    for name in ENCODERS:
        if old is not None and name in set(old.encoder):
            continue
        F_ = feats(name)
        if F_ is None:
            continue
        Xtr, Xva, Xte = F_
        for frac in FRACTIONS:
            reps = 1 if frac >= 1.0 else N_REPEATS
            for rep in range(reps):
                idx = sample_patient_subset(Ptr, frac, seed=1000 + rep)
                res = linear_probe(Xtr[idx], Ytr[idx], Xva, Yva, Xte, C_GRID, N_JOBS, PROBE_SEED)
                thr = thresholds_from_val(Yva, res["val_probs"])
                m_val = metrics_block(Yte, res["test_probs"], thr)
                m_05 = metrics_block(Yte, res["test_probs"], 0.5)
                rows.append({"encoder": name, "variant": META[name][0], "seed": META[name][1],
                             "fraction": frac, "repeat": rep,
                             "n_images": len(idx), "n_patients": len(set(Ptr[idx])),
                             "classes_present": int((Ytr[idx].sum(0) > 0).sum()),
                             "C": res["C"], **m_val,
                             "f1_macro_thr05": m_05["f1_macro"]})
                if frac >= 1.0:
                    np.savez_compressed(C.PRED_CACHE / f"probe_full_{name}.npz",
                                        val_probs=res["val_probs"], test_probs=res["test_probs"])
            sub = [r for r in rows if r["encoder"] == name and r["fraction"] == frac]
            print(f"  [probe] {name:18s} {frac*100:5.1f}% | AUPRC "
                  f"{np.mean([r['auprc_macro'] for r in sub]):.3f} ± "
                  f"{np.std([r['auprc_macro'] for r in sub]):.3f}")
    df = pd.DataFrame(rows)
    if old is not None:
        df = pd.concat([old, df], ignore_index=True)
    df["variant"] = df.encoder.map(lambda n: META[n][0])
    df["seed"] = df.encoder.map(lambda n: META[n][1])
    save(df, "fewshot_all_runs.csv")
    agg = df.groupby(["encoder", "variant", "seed", "fraction"], dropna=False).agg(
        n_images=("n_images", "mean"), auprc_mean=("auprc_macro", "mean"),
        auprc_sd=("auprc_macro", "std"), auc_mean=("auc_roc_macro", "mean"),
        f1_mean=("f1_macro", "mean"), f1_sd=("f1_macro", "std"),
        mcc_mean=("mcc_macro", "mean")).reset_index()
    save(agg, "fewshot_summary.csv")
    # full-data probe: caption-aware vs. pretrained, paired patient bootstrap of AUPRC
    pa, pb = C.PRED_CACHE / "probe_full_vl_caption_aware.npz", C.PRED_CACHE / "probe_full_biomedclip_pt.npz"
    if pa.exists() and pb.exists():
        S = {"a": np.load(pa)["test_probs"], "b": np.load(pb)["test_probs"]}
        fn = lambda idx, w: macro_auprc(Yte[idx], S[w][idx])
        r = boot_paired_metric(fn, "a", "b", GROUPS)
        save(pd.DataFrame([{"comparison": "probe100: vl_caption_aware - biomedclip_pt", **r}]),
             "fewshot_paired_auprc_full.csv")


# ╔══════════════════════════════════════════════════════════════╗
# ║  E) ANATOMICAL LOCATION                                       ║
# ╚══════════════════════════════════════════════════════════════╝

def location():
    if LOC_KEY is None:
        print("  [location] no location field found -> skipped"); return
    from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, cohen_kappa_score
    lt = [norm_text(r.get(LOC_KEY, "")) for r in TR]
    keep = sorted(l for l, n in Counter(lt).items() if n >= LOC_MIN_TRAIN and l)
    k2i = {l: i for i, l in enumerate(keep)}
    def enc(recs):
        y = np.array([k2i.get(norm_text(r.get(LOC_KEY, "")), -1) for r in recs])
        return y, y >= 0
    (ytr, mtr), (yva, mva), (yte, mte) = enc(TR), enc(VA), enc(TE)
    print(f"  [location] {len(keep)} classes (>= {LOC_MIN_TRAIN} train images); "
          f"test images evaluated: {mte.sum()}/{len(TE)}")
    rows = []
    for name in FIG_ENCODERS:                       # control analysis: seed 42 only
        F_ = feats(name)
        if F_ is None:
            continue
        res = multiclass_probe(F_[0][mtr], ytr[mtr], F_[1][mva], yva[mva], F_[2][mte], C_GRID, PROBE_SEED)
        top3 = res["classes"][np.argsort(-res["test_proba"], 1)[:, :3]]
        rows.append({"encoder": name, "method": "linear_probe",
                     "accuracy": accuracy_score(yte[mte], res["test_pred"]),
                     "top3_accuracy": float((top3 == yte[mte][:, None]).any(1).mean()),
                     "balanced_accuracy": balanced_accuracy_score(yte[mte], res["test_pred"]),
                     "f1_macro": f1_score(yte[mte], res["test_pred"], average="macro", zero_division=0),
                     "cohen_kappa": cohen_kappa_score(yte[mte], res["test_pred"])})
        if name in CLIP_NAMES and available(name):
            model, tok = get_model(name)
            T = C.location_text_embeddings(model, tok, keep, device)
            del model; torch.cuda.empty_cache()
            sim = F_[2][mte] @ T.T
            pred = sim.argmax(1)
            rows.append({"encoder": name, "method": "zero_shot",
                         "accuracy": accuracy_score(yte[mte], pred),
                         "top3_accuracy": float((np.argsort(-sim, 1)[:, :3] == yte[mte][:, None]).any(1).mean()),
                         "balanced_accuracy": balanced_accuracy_score(yte[mte], pred),
                         "f1_macro": f1_score(yte[mte], pred, average="macro", zero_division=0),
                         "cohen_kappa": cohen_kappa_score(yte[mte], pred)})
    save(pd.DataFrame(rows), "location.csv")


# ╔══════════════════════════════════════════════════════════════╗
# ║  F) SUPERVISED FINE-TUNING vs. FIRST STUDY                    ║
# ╚══════════════════════════════════════════════════════════════╝

def supervised():
    import vl1_train_alignment as T1
    tl = C.make_loader(C.ImageDataset(TE, C.image_tf("test")), batch=64)
    vl = C.make_loader(C.ImageDataset(VA, C.image_tf("test")), batch=64)
    nc = len(label2idx)
    runs = [(i, sd, fr) for fr in SUP_FRACS for sd in SEEDS for i in SUP_INITS]
    P = {}
    for init, sd, fr in runs:
        name = T1.sup_name(init, sd, fr)
        f = C.PRED_CACHE / f"{name}.npz"; ck = C.CKPT_SUP / f"{name}_best.pt"
        if not f.exists() and ck.exists():
            state = torch.load(ck, map_location="cpu")
            model = C.BiomedCLIPClassifier(nc, state.get("dropout", 0.4)).to(device)
            model.load_state_dict(state["model_state"])
            pv, _ = T1.predict(model, vl, device); pt, _ = T1.predict(model, tl, device)
            np.savez_compressed(f, val_probs=pv, val_labels=Yva, test_probs=pt, test_labels=Yte)
            print(f"  [supervised] predictions cached: {name}")
            del model; torch.cuda.empty_cache()
        if f.exists():
            P[(init, sd, fr)] = dict(np.load(f))
    ext = {}
    if C.EXTERNAL_PREDS_DIR is not None:
        for stem, nice in EXTERNAL_REFS.items():
            f = Path(C.EXTERNAL_PREDS_DIR) / f"{stem}.npz"
            if not f.exists():
                print(f"  [supervised] external reference not found: {f}"); continue
            d = dict(np.load(f))
            if d["test_labels"].shape != Yte.shape or not np.array_equal(d["test_labels"], Yte):
                print(f"  [supervised] {stem}: test labels/order differ -> not used"); continue
            ext[nice] = d
    if not P:
        print("  [supervised] no predictions found"); return

    rows = []
    for key, d in list(P.items()) + [((n, None, 1.0), d) for n, d in ext.items()]:
        init, sd, fr = key
        th = thresholds_from_val(d["val_labels"], d["val_probs"])
        m05 = extended_metrics(Yte, d["test_probs"], 0.5, MASK)
        mv = full_metrics(Yte, d["test_probs"], th, MASK)
        ci = boot_single(Yte, d["test_probs"], 0.5, GROUPS)
        rows.append({"model": INIT_NICE.get(init, init), "init": init, "seed": sd, "fraction": fr,
                     **m05,
                     "f1_ci": f"[{ci['f1'][0]:.3f}, {ci['f1'][1]:.3f}]",
                     "auprc_ci": f"[{ci['auprc'][0]:.3f}, {ci['auprc'][1]:.3f}]",
                     "f1_macro_valthr": mv["f1_macro"]})
    df = pd.DataFrame(rows).sort_values(["fraction", "seed", "init"], ascending=[False, True, True])
    save(df, "supervised_summary.csv")

    def comp(a, b, sd, fr, family):
        if (a, sd, fr) not in P or (b, sd, fr) not in P:
            return None
        A, B = P[(a, sd, fr)]["test_probs"], P[(b, sd, fr)]["test_probs"]
        r = boot_paired(Yte, A, 0.5, B, 0.5, GROUPS)
        S = {"a": A, "b": B}
        ra = boot_paired_metric(lambda idx, w: macro_auprc(Yte[idx], S[w][idx]), "a", "b", GROUPS)
        return {"family": family, "init_1": a, "init_2": b, "seed": sd, "fraction": fr, **r,
                "dAUPRC": ra["delta"], "dAUPRC_lo": ra["lo"], "dAUPRC_hi": ra["hi"],
                "dAUPRC_p": ra["p"]}

    out = []
    for sd in SEEDS:
        for a, b in PRIMARY:
            r = comp(a, b, sd, 1.0, "all_labels")
            if r: out.append(r)
        for fr in [f for f in SUP_FRACS if f < 1.0]:
            for a, b in LL_COMPARISONS:
                r = comp(a, b, sd, fr, "label_limited")
                if r: out.append(r)
    if out:
        dc = pd.DataFrame(out)
        prim = (dc.family == "all_labels") & (dc.seed == 42)
        dc["dF1_p_holm"] = np.nan
        dc.loc[prim, "dF1_p_holm"] = holm(dc.loc[prim, "dF1_p"].values)
        save(dc, "supervised_paired_differences.csv")
        cols = ["family", "init_1", "init_2", "seed", "fraction", "dF1", "dF1_lo", "dF1_hi",
                "dF1_p", "dAUPRC", "dAUPRC_lo", "dAUPRC_hi", "dAUPRC_p"]
        print(dc[cols].round(3).to_string(index=False))


# ╔══════════════════════════════════════════════════════════════╗
# ║  H) SEED SUMMARY                                              ║
# ╚══════════════════════════════════════════════════════════════╝

def _msd(g, col):
    return f"{g[col].mean():.3f} ± {g[col].std(ddof=1):.3f}" if len(g) > 1 else f"{g[col].mean():.3f}"


def seed_summary():
    from scipy.stats import wilcoxon
    R = C.RESULTS_DIR
    if (R / "zeroshot_summary.csv").exists():
        z = pd.read_csv(R / "zeroshot_summary.csv")
        z = z[z.variant.isin(VARIANTS)]
        rows = [{"variant": v, "prompt_set": ps, "n_seeds": len(g),
                 "auprc": _msd(g, "auprc_macro"), "auc_roc": _msd(g, "auc_roc_macro"),
                 "f1_valthr": _msd(g, "f1_macro")}
                for (v, ps), g in z.groupby(["variant", "prompt_set"])]
        save(pd.DataFrame(rows), "seeds_zeroshot.csv")
    if (R / "zeroshot_paired_auprc.csv").exists():
        zp = pd.read_csv(R / "zeroshot_paired_auprc.csv")
        zp = zp[zp.variant.isin(VARIANTS)]
        rows = [{"type": t, "variant": v, "n_seeds": len(g), "delta": _msd(g, "delta"),
                 "seeds_ci_excludes_0": int(((g.lo > 0) | (g.hi < 0)).sum())}
                for (t, v), g in zp.groupby(["type", "variant"])]
        save(pd.DataFrame(rows), "seeds_zeroshot_paired.csv")
    if (R / "retrieval.csv").exists():
        rt = pd.read_csv(R / "retrieval.csv")
        rt = rt[rt.variant.isin(VARIANTS)]
        rows = [{"variant": v, "subset": sub, "n_seeds": len(g), "rsum": _msd(g, "rsum"),
                 "i2t_R@5": _msd(g, "i2t_R@5"), "t2i_R@5": _msd(g, "t2i_R@5")}
                for (v, sub), g in rt.groupby(["variant", "subset"])]
        save(pd.DataFrame(rows), "seeds_retrieval.csv")
    if (R / "fewshot_all_runs.csv").exists():
        fs = pd.read_csv(R / "fewshot_all_runs.csv")
        # mean over subsets per encoder, then mean ± SD over alignment seeds
        per = fs.groupby(["variant", "seed", "fraction"], dropna=False)[
            ["auprc_macro", "f1_macro"]].mean().reset_index()
        rows = [{"variant": v, "fraction": fr, "n_seeds": len(g),
                 "auprc": _msd(g, "auprc_macro"), "f1": _msd(g, "f1_macro")}
                for (v, fr), g in per.groupby(["variant", "fraction"])]
        save(pd.DataFrame(rows), "seeds_fewshot.csv")
        # paired Wilcoxon over (seed, fraction, subset) for fractions < 100 %
        low = fs[fs.fraction < 1.0]
        def key(df):
            return df.set_index(["seed", "fraction", "repeat"])
        tests = []
        for a, b in [("caption_aware", "clip"), ("label_aware", "caption_aware"),
                     ("label_aware", "clip")]:
            A, B = key(low[low.variant == a]), key(low[low.variant == b])
            for m in ["auprc_macro", "f1_macro"]:
                d = (A[m] - B[m]).dropna()
                if len(d) >= 6:
                    tests.append({"a": a, "b": b, "metric": m, "n_pairs": len(d),
                                  "mean_diff": d.mean(), "wins": int((d > 0).sum()),
                                  "p_wilcoxon": wilcoxon(d).pvalue})
        # aligned (mean over seeds) vs. vision-only / pretrained references, paired by subset
        vl_mean = low[low.variant.isin(VARIANTS)].groupby(["variant", "fraction", "repeat"])[
            ["auprc_macro", "f1_macro"]].mean()
        for v in ["clip", "caption_aware"]:
            for ref in ["pretrained", "dinov2", "dinov2_clspatch", "swin_t_imagenet"]:
                R_ = low[low.variant == ref].set_index(["fraction", "repeat"])
                if v not in vl_mean.index.get_level_values(0) or R_.empty:
                    continue
                A = vl_mean.loc[v]
                for m in ["auprc_macro", "f1_macro"]:
                    d = (A[m] - R_[m]).dropna()
                    if len(d) >= 6:
                        tests.append({"a": f"{v} (seed mean)", "b": ref, "metric": m,
                                      "n_pairs": len(d), "mean_diff": d.mean(),
                                      "wins": int((d > 0).sum()), "p_wilcoxon": wilcoxon(d).pvalue})
        if tests:
            dt = pd.DataFrame(tests); save(dt, "seeds_fewshot_wilcoxon.csv")
            print(dt.round(4).to_string(index=False))
    if (R / "supervised_summary.csv").exists():
        su = pd.read_csv(R / "supervised_summary.csv")
        su = su[su.init.isin(SUP_INITS)]
        rows = [{"init": i, "fraction": fr, "n_seeds": len(g), "f1": _msd(g, "f1_macro"),
                 "mcc": _msd(g, "mcc_macro"), "auprc": _msd(g, "auprc_macro"),
                 "exact_match": _msd(g, "exact_match"), "brier": _msd(g, "brier_macro")}
                for (i, fr), g in su.groupby(["init", "fraction"])]
        dfs = pd.DataFrame(rows).sort_values(["fraction", "init"], ascending=[False, True])
        save(dfs, "seeds_supervised.csv"); print(dfs.to_string(index=False))


# ╔══════════════════════════════════════════════════════════════╗
# ║  G) FIGURES                                                   ║
# ╚══════════════════════════════════════════════════════════════╝

def figures():
    f = C.RESULTS_DIR / "fewshot_summary.csv"
    if f.exists():
        df = pd.read_csv(f)
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for ax, (m, sd, lab) in zip(axes, [("auprc_mean", "auprc_sd", "Macro AUPRC (mAP)"),
                                            ("f1_mean", "f1_sd", "Macro F1 (val. thresholds)")]):
            for name in FIG_ENCODERS:
                d = df[df.encoder == name].sort_values("fraction")
                if d.empty:
                    continue
                ax.errorbar(d.fraction * 100, d[m], yerr=d[sd].fillna(0), marker="o",
                            capsize=3, label=NICE[name],
                            linestyle="-" if name in LABEL_FREE else "--")
            ax.set_xscale("log"); ax.set_xlabel("Labelled training patients (%)")
            ax.set_ylabel(lab); ax.grid(alpha=0.3)
            ax.set_xticks([1, 5, 10, 25, 100]); ax.set_xticklabels(["1", "5", "10", "25", "100"])
        axes[0].legend(fontsize=8)
        plt.suptitle("Linear-probe label efficiency on BERD (test set)")
        plt.tight_layout(); plt.savefig(C.FIGURES_DIR / "fewshot_learning_curves.png", dpi=300)
        plt.close(); print("  ✓ figures/fewshot_learning_curves.png")

    f = C.RESULTS_DIR / "zeroshot_per_class.csv"
    if f.exists():
        df = pd.read_csv(f).dropna(subset=["auprc"])
        df = df[df.prompt_set == "berd_vocab"]
        piv = df.pivot(index="class", columns="encoder", values="lift")
        order = df.drop_duplicates("class").sort_values("prevalence_train", ascending=False)["class"]
        piv = piv.reindex([c for c in order if c in piv.index])
        cols = [c for c in CLIP_NAMES if c in piv.columns and c in FIG_ENCODERS]
        ax = piv[cols].rename(columns=NICE).plot.bar(figsize=(14, 5), width=0.8)
        ax.axhline(1.0, color="k", lw=0.8, ls="--")
        ax.set_ylabel("AUPRC / prevalence (lift)"); ax.set_yscale("log")
        ax.set_title("Zero-shot per-class AUPRC lift, BERD-vocabulary prompts (ordered by training prevalence)")
        plt.tight_layout(); plt.savefig(C.FIGURES_DIR / "zeroshot_per_class_lift.png", dpi=300)
        plt.close(); print("  ✓ figures/zeroshot_per_class_lift.png")

    f = C.RESULTS_DIR / "retrieval.csv"
    if f.exists():
        df = pd.read_csv(f)
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
        for ax, sub in zip(axes, ["all", "excl_frequent"]):
            d = df[(df.subset == sub) & df.encoder.isin(FIG_ENCODERS)].set_index("encoder")
            cols = ["i2t_R@1", "i2t_R@5", "i2t_R@10", "t2i_R@1", "t2i_R@5", "t2i_R@10"]
            d[cols].rename(index=NICE).T.plot.bar(ax=ax, width=0.8, legend=(sub == "all"))
            ax.set_title("Test retrieval (" + ("all queries" if sub == "all"
                         else "excluding frequent captions") + ")")
            ax.set_ylabel("Recall"); ax.tick_params(axis="x", rotation=30)
        plt.tight_layout(); plt.savefig(C.FIGURES_DIR / "retrieval.png", dpi=300)
        plt.close(); print("  ✓ figures/retrieval.png")


# ╔══════════════════════════════════════════════════════════════╗
# ║  MAIN                                                         ║
# ╚══════════════════════════════════════════════════════════════╝

# ╔══════════════════════════════════════════════════════════════╗
# ║  I) SENSITIVITY ANALYSES                                      ║
# ╚══════════════════════════════════════════════════════════════╝

SENS_PROBE_ENCODERS = ["biomedclip_pt", "vl_clip", "vl_caption_aware", "dinov2_frozen",
                       "dinov2_clspatch", "swin_t_imagenet"]


def sensitivity():
    """(1) primary metrics after excluding test images with an exact or near duplicate in the
    training/validation subsets; (2) few-shot probes with a FIXED C = 1 and threshold 0.5,
    i.e. without any use of validation labels."""
    f = C.RESULTS_DIR / "audit_duplicates.csv"
    if f.exists():
        dup = pd.read_csv(f)
        bad = sorted(set(dup.loc[dup.split_b == "test", "idx_b"])) if len(dup) else []
        keep = np.ones(len(TE), bool); keep[bad] = False
        mk = Yte[keep].sum(0) > 0
        print(f"  [sensitivity] excluding {len(bad)} test images with duplicates "
              f"({keep.sum()} kept)")
        rows = []
        for npz in sorted(C.PRED_CACHE.glob("*.npz")):
            d = np.load(npz)
            if "test_scores" in d:                         # zero-shot similarity scores
                S, prob = d["test_scores"], False
            elif "test_probs" in d:
                S, prob = d["test_probs"], True
            else:
                continue
            row = {"file": npz.stem, "n_all": len(TE), "n_kept": int(keep.sum()),
                   "auprc_all": macro_auprc(Yte, S, MASK),
                   "auprc_kept": macro_auprc(Yte[keep], S[keep], mk)}
            if prob:
                fa = full_metrics(Yte, S, 0.5, MASK)["f1_macro"]
                fk = full_metrics(Yte[keep], S[keep], 0.5, mk)["f1_macro"]
                row.update({"f1_all": fa, "f1_kept": fk})
            rows.append(row)
        df = pd.DataFrame(rows)
        df["d_auprc"] = df.auprc_kept - df.auprc_all
        if "f1_all" in df:
            df["d_f1"] = df.f1_kept - df.f1_all
        save(df, "sensitivity_duplicates.csv")
        print(f"  [sensitivity] max |ΔAUPRC| = {df.d_auprc.abs().max():.4f}"
              + (f", max |ΔF1| = {df.d_f1.abs().max():.4f}" if "d_f1" in df else ""))
    else:
        print("  [sensitivity] results/audit_duplicates.csv missing -> run vl0_leakage_audit.py")

    rows = []
    for name in SENS_PROBE_ENCODERS:
        F_ = feats(name)
        if F_ is None:
            continue
        Xtr, Xva, Xte = F_
        for frac in FRACTIONS:
            for rep in range(1 if frac >= 1.0 else N_REPEATS):
                idx = sample_patient_subset(Ptr, frac, seed=1000 + rep)
                res = linear_probe(Xtr[idx], Ytr[idx], Xva, Yva, Xte, (1.0,), N_JOBS, PROBE_SEED)
                m = metrics_block(Yte, res["test_probs"], 0.5)
                rows.append({"encoder": name, "fraction": frac, "repeat": rep,
                             "auprc_macro": m["auprc_macro"], "auc_roc_macro": m["auc_roc_macro"],
                             "f1_macro_thr05": m["f1_macro"]})
    if rows:
        df = pd.DataFrame(rows); save(df, "sensitivity_fewshot_fixedC.csv")
        agg = df.groupby(["encoder", "fraction"]).auprc_macro.agg(["mean", "std"]).round(3)
        print(agg.unstack(0).to_string())


if __name__ == "__main__":
    if RUN_FEATURES:   print("\n=== A) Features ===");   extract_features()
    if RUN_ZERO_SHOT:  print("\n=== B) Zero-shot ===");  zero_shot()
    if RUN_RETRIEVAL:  print("\n=== C) Retrieval ===");  retrieval()
    if RUN_FEW_SHOT:   print("\n=== D) Few-shot ===");   few_shot()
    if RUN_LOCATION:   print("\n=== E) Location ===");   location()
    if RUN_SUPERVISED: print("\n=== F) Supervised ==="); supervised()
    if RUN_FIGURES:    print("\n=== G) Figures ===");    figures()
    if RUN_SEED_SUMMARY: print("\n=== H) Seed summary ==="); seed_summary()
    if RUN_SENSITIVITY: print("\n=== I) Sensitivity ==="); sensitivity()
    C.write_manifest("evaluate", {"seeds": SEEDS, "fractions": FRACTIONS, "repeats": N_REPEATS,
                                  "C_grid": C_GRID, "prompt_sets": C.PROMPT_SETS})
    print("\n✓ Done. Tables in results/, figures in figures/")
