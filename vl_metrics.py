"""
vl_metrics.py -- NumPy/scikit-learn utilities for the BERD vision-language study.
No PyTorch dependency, so these functions can be unit-tested on any machine.

Contents
  - text normalisation and caption / label-set ids
  - cross-modal retrieval with the unique-caption protocol (duplicate captions,
    e.g. the template text used for normal examinations, are merged into one
    gallery item so that a query is never penalised for retrieving an identical text)
  - per-class thresholds for unbounded zero-shot similarity scores (validation only)
  - patient-level sub-sampling of the training set for few-shot linear probing
  - linear probe (one logistic regression per class, C selected on validation)
  - multi-label metrics, validation thresholds and patient-level bootstrap
    (same definitions as the first BERD study)
"""

import re
from collections import Counter, defaultdict

import numpy as np


# ════════════════════════════════════════════════════════════════
#  TEXT HELPERS
# ════════════════════════════════════════════════════════════════

def norm_text(s):
    """Lower-case, replace punctuation by spaces, collapse whitespace.
    'mucosal congestion, edema' and 'mucosal congestion edema.' map to the same text."""
    s = re.sub(r"[^\w\s-]", " ", str(s or "").lower())
    return re.sub(r"\s+", " ", s).strip()


def make_ids(keys):
    """Map hashable keys to consecutive integer ids (first-seen order)."""
    table, out = {}, []
    for k in keys:
        if k not in table:
            table[k] = len(table)
        out.append(table[k])
    return np.array(out), table


def template_captions(train_captions, min_count):
    """Frequent (recurring) captions: normalised captions occurring at least `min_count`
    times in the TRAINING set, e.g. 'it is normal' or 'mild mucosa congestion'."""
    cnt = Counter(norm_text(c) for c in train_captions)
    return {c for c, n in cnt.items() if n >= min_count}


# ════════════════════════════════════════════════════════════════
#  RETRIEVAL (unique-caption protocol)
# ════════════════════════════════════════════════════════════════

def _first_hit_rank(order, correct):
    """order: (Q, G) gallery indices sorted by similarity; correct: (Q, G) bool.
    Returns the 1-based rank of the first correct item per query."""
    hit = np.take_along_axis(correct, order, axis=1)
    has = hit.any(1)
    r = np.where(has, hit.argmax(1) + 1, order.shape[1] + 1)
    return r


def retrieval_metrics(img_f, cap_f, img_cap_ids, ks=(1, 5, 10), query_img_mask=None,
                      query_cap_mask=None):
    """
    img_f        : (N, D) L2-normalised image features
    cap_f        : (U, D) L2-normalised features of the U UNIQUE captions
    img_cap_ids  : (N,)  index (0..U-1) of each image's caption in cap_f
    query_*_mask : optional boolean masks restricting the QUERIES (gallery stays full)

    image->text : query = image, gallery = unique captions, 1 correct item
    text->image : query = unique caption, gallery = images, correct = any image
                  that carries this caption
    """
    img_cap_ids = np.asarray(img_cap_ids)
    U = cap_f.shape[0]
    sim = img_f @ cap_f.T                                      # (N, U)

    # image -> text
    correct_i2t = np.zeros_like(sim, dtype=bool)
    correct_i2t[np.arange(len(img_cap_ids)), img_cap_ids] = True
    r_i2t = _first_hit_rank(np.argsort(-sim, axis=1), correct_i2t)
    if query_img_mask is not None:
        r_i2t = r_i2t[np.asarray(query_img_mask, bool)]

    # text -> image
    simT = sim.T                                               # (U, N)
    correct_t2i = img_cap_ids[None, :] == np.arange(U)[:, None]
    r_t2i = _first_hit_rank(np.argsort(-simT, axis=1), correct_t2i)
    if query_cap_mask is not None:
        r_t2i = r_t2i[np.asarray(query_cap_mask, bool)]

    out = {}
    for k in ks:
        out[f"i2t_R@{k}"] = float((r_i2t <= k).mean()) if len(r_i2t) else np.nan
        out[f"t2i_R@{k}"] = float((r_t2i <= k).mean()) if len(r_t2i) else np.nan
    out["i2t_medR"] = float(np.median(r_i2t)) if len(r_i2t) else np.nan
    out["t2i_medR"] = float(np.median(r_t2i)) if len(r_t2i) else np.nan
    out["i2t_meanR"] = float(np.mean(r_i2t)) if len(r_i2t) else np.nan
    out["t2i_meanR"] = float(np.mean(r_t2i)) if len(r_t2i) else np.nan
    out["rsum"] = float(sum(out[f"{d}_R@{k}"] for d in ("i2t", "t2i") for k in ks))
    out["n_img_queries"] = int(len(r_i2t)); out["n_txt_queries"] = int(len(r_t2i))
    return out


def unique_caption_index(captions):
    """captions (list of str) -> (unique normalised captions, per-item index)."""
    normed = [norm_text(c) for c in captions]
    ids, table = make_ids(normed)
    uniq = [None] * len(table)
    for t, i in table.items():
        uniq[i] = t
    return uniq, ids


# ════════════════════════════════════════════════════════════════
#  ZERO-SHOT THRESHOLDS (scores are cosine similarities, not probabilities)
# ════════════════════════════════════════════════════════════════

def thresholds_from_val_scores(yv, sv, q_grid=np.arange(0.01, 1.0, 0.01)):
    """Per-class threshold maximising validation F1, searched over quantiles of the
    validation scores of that class. Classes without validation positives get +inf
    (never predicted), which is the conservative choice."""
    from sklearn.metrics import f1_score
    th = np.full(yv.shape[1], np.inf)
    for c in range(yv.shape[1]):
        if yv[:, c].sum() == 0:
            continue
        cand = np.unique(np.quantile(sv[:, c], q_grid))
        best, bt = -1.0, cand[-1]
        for t in cand:
            f = f1_score(yv[:, c], sv[:, c] >= t, zero_division=0)
            if f > best + 1e-12:
                best, bt = f, t
        th[c] = bt
    return th


# ════════════════════════════════════════════════════════════════
#  FEW-SHOT SUB-SAMPLING (patient level)
# ════════════════════════════════════════════════════════════════

def sample_patient_subset(pids, fraction, seed):
    """Indices of all images belonging to a random `fraction` of the training patients."""
    pids = np.asarray(pids)
    uniq = np.array(sorted(set(pids.tolist())))
    if fraction >= 1.0:
        return np.arange(len(pids))
    rng = np.random.default_rng(seed)
    k = max(1, int(round(fraction * len(uniq))))
    keep = set(rng.choice(uniq, size=k, replace=False).tolist())
    return np.array([i for i, p in enumerate(pids) if p in keep])


# ════════════════════════════════════════════════════════════════
#  LINEAR PROBE
# ════════════════════════════════════════════════════════════════

def _fit_one(X, y, Xs, C, seed):
    from sklearn.linear_model import LogisticRegression
    if y.sum() == 0:                      # class absent from the labelled subset
        return [np.zeros(len(x)) for x in Xs]
    if y.sum() == len(y):
        return [np.ones(len(x)) for x in Xs]
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000,
                             solver="lbfgs", random_state=seed)
    clf.fit(X, y)
    return [clf.predict_proba(x)[:, 1] for x in Xs]


def macro_auprc(y, s, mask=None):
    from sklearn.metrics import average_precision_score
    cols = [c for c in range(y.shape[1])
            if y[:, c].sum() > 0 and (mask is None or mask[c])]
    if not cols:
        return np.nan
    return float(np.mean([average_precision_score(y[:, c], s[:, c]) for c in cols]))


def linear_probe(Xtr, Ytr, Xva, Yva, Xte, C_grid=(0.01, 0.1, 1.0, 10.0), n_jobs=1, seed=0):
    """One-vs-rest logistic regression on frozen features.
    A single C (shared by all classes) is chosen by validation macro AUPRC.
    Returns dict with val/test probabilities and the selected C."""
    from joblib import Parallel, delayed
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    Xtr_, Xva_, Xte_ = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)
    best = None
    for C in C_grid:
        res = Parallel(n_jobs=n_jobs)(
            delayed(_fit_one)(Xtr_, Ytr[:, c], (Xva_, Xte_), C, seed)
            for c in range(Ytr.shape[1]))
        pv = np.stack([r[0] for r in res], 1); pt = np.stack([r[1] for r in res], 1)
        score = macro_auprc(Yva, pv, mask=Ytr.sum(0) > 0)
        if best is None or (np.isfinite(score) and score > best["val_auprc"]):
            best = {"C": C, "val_auprc": score, "val_probs": pv, "test_probs": pt}
    return best


def multiclass_probe(Xtr, ytr, Xva, yva, Xte, C_grid=(0.01, 0.1, 1.0, 10.0), seed=0):
    """Multinomial logistic regression (single-label task, e.g. anatomical location);
    C chosen by validation macro F1. Returns predicted labels for val/test."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    best = None
    for C in C_grid:
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000,
                                 random_state=seed)
        clf.fit(sc.transform(Xtr), ytr)
        pv = clf.predict(sc.transform(Xva))
        f = f1_score(yva, pv, average="macro", zero_division=0)
        if best is None or f > best["val_f1"]:
            best = {"C": C, "val_f1": f, "val_pred": pv,
                    "test_pred": clf.predict(sc.transform(Xte)),
                    "test_proba": clf.predict_proba(sc.transform(Xte)), "classes": clf.classes_}
    return best


def cluster_groups(pids):
    """Patient id -> array of row indices (for the patient-level bootstrap)."""
    g = defaultdict(list)
    for i, p in enumerate(pids):
        g[p].append(i)
    return [np.array(v) for v in g.values()]


def boot_paired_metric(fn, a, b, groups, B=2000, seed=2026):
    """Patient-level paired bootstrap of fn(a_idx) - fn(b_idx).
    fn(idx, which) must return the metric of model `which` ('a' or 'b') on rows idx."""
    rng = np.random.default_rng(seed); G = len(groups)
    full = np.concatenate(groups)
    obs = fn(full, a) - fn(full, b)
    d = []
    for _ in range(B):
        idx = np.concatenate([groups[g] for g in rng.integers(0, G, G)])
        d.append(fn(idx, a) - fn(idx, b))
    d = np.array(d); d = d[np.isfinite(d)]
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
    return {"delta": obs, "lo": lo, "hi": hi, "p": p}


# ════════════════════════════════════════════════════════════════
#  MULTI-LABEL METRICS AND PATIENT-LEVEL STATISTICS
#  (identical definitions to the first BERD study, copied so that this
#   project is self-contained)
# ════════════════════════════════════════════════════════════════

def per_class_counts(y, p):
    tp = (y * p).sum(0); fp = ((1 - y) * p).sum(0)
    fn = (y * (1 - p)).sum(0); tn = ((1 - y) * (1 - p)).sum(0)
    return tp, fp, fn, tn


def f1_mcc_per_class(y, p):
    tp, fp, fn, tn = per_class_counts(y, p)
    d = 2 * tp + fp + fn
    f1 = np.where(d > 0, 2 * tp / np.maximum(d, 1e-12), 0.0)
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.where(den > 0, (tp * tn - fp * fn) / np.maximum(den, 1e-12), 0.0)
    return f1, mcc


def auprc_per_class(y, s, mask):
    from sklearn.metrics import average_precision_score
    return np.array([average_precision_score(y[:, c], s[:, c]) if mask[c] else np.nan
                     for c in range(y.shape[1])])


def auroc_per_class(y, s, mask):
    from sklearn.metrics import roc_auc_score
    out = []
    for c in range(y.shape[1]):
        ok = mask[c] and 0 < y[:, c].sum() < len(y)
        out.append(roc_auc_score(y[:, c], s[:, c]) if ok else np.nan)
    return np.array(out)


def sample_f1(y, p):
    inter = (y * p).sum(1); den = y.sum(1) + p.sum(1)
    return np.where(den > 0, 2 * inter / np.maximum(den, 1e-12), 1.0)


def full_metrics(y, s, thr, mask):
    """Macro metrics over the classes in `mask` (classes with >= 1 positive test instance)."""
    p = (s >= thr).astype(float)
    f1, mcc = f1_mcc_per_class(y, p)
    ap = auprc_per_class(y, s, mask); au = auroc_per_class(y, s, mask)
    return {
        "f1_macro": float(f1[mask].mean()), "mcc_macro": float(mcc[mask].mean()),
        "auprc_macro": float(np.nanmean(ap)), "auc_roc_macro": float(np.nanmean(au)),
        "brier_macro": float(((s - y) ** 2).mean(0)[mask].mean()),
        "exact_match": float((p == y).all(1).mean()), "hamming": float((p != y).mean()),
        "f1_samples": float(sample_f1(y, p).mean()),
    }


def thresholds_from_val(yv, sv):
    """Per-class probability thresholds maximising validation F1 (grid 0.05-0.95)."""
    from sklearn.metrics import f1_score
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    th = np.full(yv.shape[1], 0.5)
    for c in range(yv.shape[1]):
        if yv[:, c].sum() == 0:
            continue
        best, bt = f1_score(yv[:, c], sv[:, c] >= 0.5, zero_division=0), 0.5
        for t in grid:
            f = f1_score(yv[:, c], sv[:, c] >= t, zero_division=0)
            if f > best + 1e-12:
                best, bt = f, t
        th[c] = bt
    return th


def holm(pvals):
    p = np.asarray(pvals, float); m = len(p); order = np.argsort(p)
    adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i]); adj[i] = min(run, 1.0)
    return adj


def _boot_idx(groups, B, seed):
    rng = np.random.default_rng(seed); G = len(groups)
    for _ in range(B):
        yield np.concatenate([groups[g] for g in rng.integers(0, G, G)])


def boot_single(y, s, thr, groups, B=2000, seed=2026):
    """Patient-level bootstrap 95% CIs of macro F1, MCC and AUPRC."""
    p = (s >= thr).astype(float); rows = []
    for idx in _boot_idx(groups, B, seed):
        yy, pp = y[idx], p[idx]; m = yy.sum(0) > 0
        f1, mcc = f1_mcc_per_class(yy, pp)
        rows.append([f1[m].mean(), mcc[m].mean(), np.nanmean(auprc_per_class(yy, s[idx], m))])
    a = np.array(rows)
    return {k: (np.percentile(a[:, j], 2.5), np.percentile(a[:, j], 97.5))
            for j, k in enumerate(["f1", "mcc", "auprc"])}


def boot_paired(y, s1, t1, s2, t2, groups, B=2000, seed=2026):
    """Patient-level paired bootstrap of dF1 and dMCC (model 1 minus model 2)."""
    p1 = (s1 >= t1).astype(float); p2 = (s2 >= t2).astype(float)
    full = y.sum(0) > 0
    f1a, ma = f1_mcc_per_class(y, p1); f1b, mb = f1_mcc_per_class(y, p2)
    obs = (f1a[full].mean() - f1b[full].mean(), ma[full].mean() - mb[full].mean())
    d = []
    for idx in _boot_idx(groups, B, seed):
        yy = y[idx]; m = yy.sum(0) > 0
        fa, xa = f1_mcc_per_class(yy, p1[idx]); fb, xb = f1_mcc_per_class(yy, p2[idx])
        d.append((fa[m].mean() - fb[m].mean(), xa[m].mean() - xb[m].mean()))
    d = np.array(d); res = {}
    for j, k in enumerate(["dF1", "dMCC"]):
        lo, hi = np.percentile(d[:, j], [2.5, 97.5])
        pb = min(1.0, 2 * min((d[:, j] <= 0).mean(), (d[:, j] >= 0).mean()))
        res[k] = obs[j]; res[k + "_lo"] = lo; res[k + "_hi"] = hi; res[k + "_p"] = pb
    return res


# ════════════════════════════════════════════════════════════════
#  EXTENDED METRICS (for comparison with other multi-label, zero-shot
#  and retrieval studies; reported in supplementary tables)
# ════════════════════════════════════════════════════════════════

def ece_macro(y, s, mask, n_bins=10):
    """Expected calibration error per class (equal-width bins), averaged over `mask`."""
    out = []
    for c in np.where(mask)[0]:
        p, t = s[:, c], y[:, c]
        bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
        e = 0.0
        for b in range(n_bins):
            m = bins == b
            if m.any():
                e += m.mean() * abs(t[m].mean() - p[m].mean())
        out.append(e)
    return float(np.mean(out)) if out else np.nan


def extended_metrics(y, s, thr, mask, is_prob=True):
    """Threshold-dependent and -independent multi-label metrics.
    Macro = mean over classes in `mask`; micro = pooled over the same classes;
    weighted = support-weighted; samples = example-based. Calibration only if is_prob."""
    from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                                 precision_score, recall_score, jaccard_score,
                                 label_ranking_average_precision_score, label_ranking_loss,
                                 coverage_error)
    p = (s >= thr).astype(int)
    yi = y.astype(int)
    ym, pm, sm = yi[:, mask], p[:, mask], s[:, mask]
    tp, fp, fn, tn = per_class_counts(ym, pm)
    spec = np.where(tn + fp > 0, tn / np.maximum(tn + fp, 1e-12), np.nan)
    sens = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1e-12), np.nan)
    kw = dict(zero_division=0)
    out = {
        "f1_macro": f1_score(ym, pm, average="macro", **kw),
        "f1_micro": f1_score(ym, pm, average="micro", **kw),
        "f1_weighted": f1_score(ym, pm, average="weighted", **kw),
        "f1_samples": float(sample_f1(yi, p).mean()),
        "precision_macro": precision_score(ym, pm, average="macro", **kw),
        "precision_micro": precision_score(ym, pm, average="micro", **kw),
        "recall_macro": recall_score(ym, pm, average="macro", **kw),        # = sensitivity
        "recall_micro": recall_score(ym, pm, average="micro", **kw),
        "specificity_macro": float(np.nanmean(spec)),
        "balanced_acc_macro": float(np.nanmean((sens + spec) / 2)),
        "mcc_macro": float(f1_mcc_per_class(ym.astype(float), pm.astype(float))[1].mean()),
        "jaccard_samples": jaccard_score(yi, p, average="samples", **kw),
        "exact_match": float((p == yi).all(1).mean()),
        "hamming_loss": float((p != yi).mean()),
        "auc_roc_macro": float(np.nanmean(auroc_per_class(y, s, mask))),
        "auc_roc_micro": roc_auc_score(ym.ravel(), sm.ravel()),
        "auprc_macro": float(np.nanmean(auprc_per_class(y, s, mask))),
        "auprc_micro": average_precision_score(ym.ravel(), sm.ravel()),
        "lrap": label_ranking_average_precision_score(yi, s),
        "ranking_loss": label_ranking_loss(yi, s),
        "coverage_error": coverage_error(yi, s),
    }
    if is_prob:
        out["brier_macro"] = float(((s - y) ** 2).mean(0)[mask].mean())
        out["ece_macro"] = ece_macro(y, s, mask)
    return {k: float(v) for k, v in out.items()}


def image_retrieval_metrics(f, Y, ks=(1, 5, 10)):
    """Image-to-image retrieval within a set (query excluded from its own gallery).
    Relevant = shares at least one finding label with the query.
    Returns Precision@K and mean average precision (mAP)."""
    sim = f @ f.T
    np.fill_diagonal(sim, -np.inf)
    order = np.argsort(-sim, axis=1)[:, :-1]                  # drop self (ranked last)
    rel = (Y @ Y.T) > 0
    np.fill_diagonal(rel, False)
    R = np.take_along_axis(rel, order, axis=1)
    out = {f"i2i_P@{k}": float(R[:, :k].mean()) for k in ks}
    has = R.any(1)
    cum = np.cumsum(R, 1)
    prec = cum / np.arange(1, R.shape[1] + 1)
    ap = (prec * R).sum(1)[has] / R.sum(1)[has]
    out["i2i_mAP"] = float(ap.mean())
    return out
