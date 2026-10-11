"""
vl1_train_alignment.py -- Report-supervised vision-language alignment on BERD.

Steps
  0. AUDIT     : detects the caption / location keys, counts unique and frequent captions,
                 caption length, location distribution (results/audit_*).
  1. ALIGNMENT : contrastive fine-tuning of BiomedCLIP (image AND text towers) on BERD
                 image-caption pairs of the training subset. Three loss variants:
                   clip           standard symmetric InfoNCE (identity targets)
                   caption_aware  PROPOSED: images in a batch whose (normalised) captions are
                                  identical are positives of each other (soft targets). Removes
                                  the false negatives created by frequent captions (e.g. normal
                                  examinations). Uses NO finding labels.
                   label_aware    supervised reference: positives also include pairs with an
                                  identical finding-label set (uses labels; reported as an
                                  upper-bound variant, not as label-free).
                 Model selection: validation retrieval R-sum (unique-caption protocol), computed
                 from captions only -- finding labels are never used for selection.
  2. SUPERVISED: 23-class multi-label fine-tuning of the BiomedCLIP image tower, initialised
                 either from the pretrained weights ("pretrained", in-study baseline) or from an
                 aligned checkpoint OF THE SAME SEED (full-pipeline replicates). Optionally also
                 with 10 % / 25 % of the training patients (label-limited fine-tuning; patient
                 subsets identical to the few-shot probes, subset index 0). Protocol of the earlier BERD benchmark: class-weighted BCE
                 (positive weights capped at 50), weighted random sampler, MixUp (alpha 0.4,
                 p 0.3), AdamW (head 1e-4, backbone 1e-5, wd 1e-4), ReduceLROnPlateau on
                 validation macro F1, early stopping (patience 15), up to 70 epochs.

The test set is never used here. Evaluation: vl2_evaluate.py.
Run: python vl1_train_alignment.py   (or runfile(...) in Spyder)
"""

import json
import math
import random
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, WeightedRandomSampler

import vl_common as C
from vl_metrics import (norm_text, make_ids, template_captions, retrieval_metrics,
                        unique_caption_index, sample_patient_subset)


def sup_name(init, seed=42, frac=1.0):
    """Checkpoint / prediction stem. Seed 42 with all labels keeps the original name."""
    s = f"sup_{init}"
    if seed != 42:
        s += f"_seed{seed}"
    if frac < 1.0:
        s += f"_frac{int(round(frac * 100))}"
    return s


# ╔══════════════════════════════════════════════════════════════╗
# ║  SETTINGS                                                     ║
# ╚══════════════════════════════════════════════════════════════╝

RUN_AUDIT          = True
RUN_ALIGNMENT      = True
RUN_SUPERVISED_FT  = True
FORCE_RETRAIN      = False

# ── 1. alignment ──
VARIANTS        = ["clip", "caption_aware", "label_aware"]
SEEDS           = [42, 7, 123]   # full-pipeline replicates (existing checkpoints are skipped)
EPOCHS          = 30
BATCH_SIZE      = 64             # RTX 4080 16 GB with AMP; lower to 32 if out of memory
LR              = 1e-5           # both towers
WEIGHT_DECAY    = 0.05
WARMUP_EPOCHS   = 1
PATIENCE        = 5              # early stopping on validation R-sum
MAX_LOGIT_SCALE = 100.0

# ── 2. supervised fine-tuning ──
SUP_INITS     = ["pretrained", "caption_aware", "clip", "label_aware"]
SUP_SEEDS     = [42, 7, 123]
# label-limited fine-tuning: fractions of training PATIENTS, initialisations, seeds
RUN_LABEL_LIMITED = True
LL_FRACTIONS  = [0.10, 0.25]
LL_INITS      = ["pretrained", "caption_aware", "clip"]
LL_SEEDS      = [42, 7, 123]
LL_SUBSET_SEED = 1000          # = subset index 0 of the few-shot probes (vl2_evaluate.py)
SUP_EPOCHS    = 70
SUP_BATCH     = 32
SUP_LR_HEAD   = 1e-4
SUP_LR_BACK   = 1e-5
SUP_WD        = 1e-4
SUP_DROPOUT   = 0.4
SUP_MAX_POS_W = 50.0
SUP_LR_PAT    = 5
SUP_EARLY     = 15
MIXUP_P, MIXUP_ALPHA = 0.3, 0.4


def set_seed(sd):
    random.seed(sd); np.random.seed(sd); torch.manual_seed(sd); torch.cuda.manual_seed_all(sd)


# ╔══════════════════════════════════════════════════════════════╗
# ║  0. AUDIT                                                     ║
# ╚══════════════════════════════════════════════════════════════╝

def audit(train_recs, val_recs, test_recs, cap_key, loc_key):
    print(f"\n{'='*60}\nAUDIT\n{'='*60}")
    print(f"  caption key : {cap_key}\n  location key: {loc_key}")
    print(f"  example record keys: {sorted(train_recs[0].keys())}")
    info = {"caption_key": cap_key, "location_key": loc_key}
    tmpl = template_captions(C.captions_of(train_recs, cap_key), C.TEMPLATE_MIN_COUNT)
    for tag, recs in [("train", train_recs), ("val", val_recs), ("test", test_recs)]:
        caps = [norm_text(c) for c in C.captions_of(recs, cap_key)]
        words = np.array([len(c.split()) for c in caps])
        n_t = sum(c in tmpl for c in caps)
        info[tag] = {"images": len(recs), "patients": int(len(set(C.patient_ids(recs)))),
                     "unique_captions": int(len(set(caps))),
                     "frequent_caption_images": int(n_t),
                     "frequent_fraction": round(n_t / max(len(caps), 1), 4),
                     "words_mean": round(float(words.mean()), 1),
                     "words_p95": int(np.percentile(words, 95)), "words_max": int(words.max())}
        print(f"  [{tag}] {info[tag]}")
    info["n_frequent_captions"] = len(tmpl)
    top = Counter(norm_text(c) for c in C.captions_of(train_recs, cap_key)).most_common(50)
    pd.DataFrame(top, columns=["caption", "count_train"]).to_csv(
        C.RESULTS_DIR / "audit_top_captions.csv", index=False)
    if loc_key:
        for tag, recs in [("train", train_recs), ("test", test_recs)]:
            cnt = Counter(norm_text(r.get(loc_key, "")) for r in recs)
            pd.DataFrame(cnt.most_common(), columns=["location", "count"]).to_csv(
                C.RESULTS_DIR / f"audit_locations_{tag}.csv", index=False)
    (C.RESULTS_DIR / "audit_summary.json").write_text(json.dumps(info, indent=2))
    print("  ✓ results/audit_summary.json, audit_top_captions.csv")


# ╔══════════════════════════════════════════════════════════════╗
# ║  1. CONTRASTIVE ALIGNMENT                                     ║
# ╚══════════════════════════════════════════════════════════════╝

def contrastive_loss(img_f, txt_f, scale, variant, cap_ids, set_ids):
    """Symmetric soft-target InfoNCE. img_f, txt_f are L2-normalised (B, D)."""
    logits = scale * img_f @ txt_f.t()
    B = logits.size(0)
    if variant == "clip":
        tgt = torch.eye(B, device=logits.device)
    elif variant == "caption_aware":
        tgt = (cap_ids[:, None] == cap_ids[None, :]).float()
    elif variant == "label_aware":
        tgt = ((cap_ids[:, None] == cap_ids[None, :]) |
               (set_ids[:, None] == set_ids[None, :])).float()
    else:
        raise ValueError(variant)
    t_i2t = tgt / tgt.sum(1, keepdim=True)
    t_t2i = tgt.t() / tgt.t().sum(1, keepdim=True)
    l_i2t = -(t_i2t * F.log_softmax(logits, 1)).sum(1).mean()
    l_t2i = -(t_t2i * F.log_softmax(logits.t(), 1)).sum(1).mean()
    return 0.5 * (l_i2t + l_t2i)


def param_groups(model, wd):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if p.requires_grad:
            (no_decay if p.ndim < 2 or "logit_" in n else decay).append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]


def cosine_warmup(opt, warmup_steps, total_steps):
    def f(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
    return LambdaLR(opt, f)


@torch.no_grad()
def val_retrieval(model, tok, val_recs, cap_key, device):
    uniq, ids = unique_caption_index(C.captions_of(val_recs, cap_key))
    img_f = C.encode_images(model, C.make_loader(C.ImageDataset(val_recs, C.image_tf("test"))), device)
    return retrieval_metrics(img_f, C.encode_texts(model, tok, uniq, device), ids)


def train_alignment(variant, seed, label2idx, train_recs, val_recs, cap_key, device):
    ckpt = C.CKPT_VL / f"vl_{variant}_seed{seed}_best.pt"
    if ckpt.exists() and not FORCE_RETRAIN:
        print(f"  [align] {variant} seed{seed}: checkpoint exists -> skipped"); return ckpt
    set_seed(seed)
    model, tok = C.load_biomedclip(device)
    cap_ids, _ = make_ids([norm_text(c) for c in C.captions_of(train_recs, cap_key)])
    set_ids, _ = make_ids(C.label_set_keys(train_recs, label2idx))
    ds = C.ImageTextDataset(train_recs, C.image_tf("train"), tok, cap_key, cap_ids, set_ids)
    g = torch.Generator(); g.manual_seed(seed)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, generator=g,
                        num_workers=C.NUM_WORKERS, pin_memory=torch.cuda.is_available())
    opt = AdamW(param_groups(model, WEIGHT_DECAY), lr=LR)
    sched = cosine_warmup(opt, WARMUP_EPOCHS * len(loader), EPOCHS * len(loader))
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    base = val_retrieval(model, tok, val_recs, cap_key, device)
    print(f"\n[{variant} | seed {seed}] epoch 0 (pretrained) val R-sum={base['rsum']:.3f}")
    best, wait, hist = -1.0, 0, [{"epoch": 0, **base}]
    for ep in range(1, EPOCHS + 1):
        model.train(); t0 = time.time(); tot = 0.0
        for x, t, cid, sid in loader:
            x, t = x.to(device, non_blocking=True), t.to(device, non_blocking=True)
            cid, sid = cid.to(device), sid.to(device)
            opt.zero_grad(set_to_none=True)
            with C.amp_ctx(device):
                fi = F.normalize(model.encode_image(x), dim=-1)
                ft = F.normalize(model.encode_text(t), dim=-1)
            scale = model.logit_scale.exp().clamp(max=MAX_LOGIT_SCALE)
            loss = contrastive_loss(fi.float(), ft.float(), scale, variant, cid, sid)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            with torch.no_grad():
                model.logit_scale.clamp_(0, math.log(MAX_LOGIT_SCALE))
            tot += loss.item()
        vr = val_retrieval(model, tok, val_recs, cap_key, device)
        hist.append({"epoch": ep, "train_loss": tot / len(loader), **vr,
                     "time_s": round(time.time() - t0, 1)})
        print(f"  ep {ep:02d} | loss {tot/len(loader):.4f} | val R-sum {vr['rsum']:.3f} "
              f"| i2t R@5 {vr['i2t_R@5']:.3f} | t2i R@5 {vr['t2i_R@5']:.3f} | {time.time()-t0:.0f}s")
        if vr["rsum"] > best:
            best, wait = vr["rsum"], 0
            torch.save({"model_state": model.state_dict(), "variant": variant, "seed": seed,
                        "epoch": ep, "val": vr}, ckpt)
            print(f"  ✓ best (R-sum {best:.3f})")
        else:
            wait += 1
            if wait >= PATIENCE:
                print("  early stopping"); break
    (C.RESULTS_DIR / f"vl_{variant}_seed{seed}_log.json").write_text(json.dumps(hist, indent=2))
    del model, opt; torch.cuda.empty_cache()
    return ckpt


# ╔══════════════════════════════════════════════════════════════╗
# ║  2. SUPERVISED FINE-TUNING                                    ║
# ╚══════════════════════════════════════════════════════════════╝

class MixUpCollate:
    def __call__(self, batch):
        x, y = zip(*batch); x, y = torch.stack(x), torch.stack(y)
        if np.random.random() < MIXUP_P:
            lam = float(np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA)); idx = torch.randperm(x.size(0))
            x = lam * x + (1 - lam) * x[idx]; y = lam * y + (1 - lam) * y[idx]
        return x, y


def sup_loaders(label2idx, train_recs, val_recs):
    Ytr = C.label_matrix(train_recs, label2idx)
    cnt = Ytr.sum(0); tot = cnt.sum()
    lw = tot / np.maximum(cnt, 1)                                    # rare labels -> high weight
    sw = [float(lw[np.where(row > 0)[0]].max()) for row in Ytr]      # max over the image's labels
    sampler = WeightedRandomSampler(torch.DoubleTensor(sw), len(sw), replacement=True)
    tr = C.LabelledImageDataset(train_recs, label2idx, C.image_tf("train"))
    va = C.LabelledImageDataset(val_recs, label2idx, C.image_tf("test"))
    kw = dict(num_workers=C.NUM_WORKERS, pin_memory=torch.cuda.is_available())
    trl = DataLoader(tr, SUP_BATCH, sampler=sampler, collate_fn=MixUpCollate(), drop_last=True, **kw)
    val = DataLoader(va, SUP_BATCH, shuffle=False, **kw)
    n = len(Ytr); pos_w = np.minimum((n - cnt) / np.maximum(cnt, 1), SUP_MAX_POS_W)
    return trl, val, torch.tensor(pos_w, dtype=torch.float32)


@torch.no_grad()
def predict(model, loader, device):
    model.eval(); P, Y = [], []
    for batch in loader:
        x = batch[0]
        with C.amp_ctx(device):
            P.append(torch.sigmoid(model(x.to(device))).float().cpu().numpy())
        if len(batch) > 1 and batch[1].ndim == 2:
            Y.append(batch[1].numpy())
    return np.vstack(P), (np.vstack(Y) if Y else None)


def build_classifier(init, nc, seed=42):
    vis = None if init == "pretrained" else C.visual_state_from(C.CKPT_VL / f"vl_{init}_seed{seed}_best.pt")
    return C.BiomedCLIPClassifier(nc, SUP_DROPOUT, vis)


def train_supervised(init, label2idx, trl, val, pos_w, device, seed=42, frac=1.0):
    name = sup_name(init, seed, frac)
    ckpt = C.CKPT_SUP / f"{name}_best.pt"
    if ckpt.exists() and not FORCE_RETRAIN:
        print(f"  [supervised] {name}: checkpoint exists -> skipped"); return
    if init != "pretrained" and not (C.CKPT_VL / f"vl_{init}_seed{seed}_best.pt").exists():
        print(f"  [supervised] {name}: aligned checkpoint missing -> skipped"); return
    set_seed(seed)
    nc = len(label2idx)
    model = build_classifier(init, nc, seed).to(device)
    opt = AdamW([{"params": model.b.parameters(), "lr": SUP_LR_BACK},
                 {"params": model.h.parameters(), "lr": SUP_LR_HEAD}], weight_decay=SUP_WD)
    sched = ReduceLROnPlateau(opt, mode="max", patience=SUP_LR_PAT, factor=0.5)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_w.to(device))
    best, wait, hist = 0.0, 0, []
    print(f"\n[SUPERVISED] {name} (init={init}, seed={seed}, labelled patients={frac:.0%})")
    for ep in range(1, SUP_EPOCHS + 1):
        model.train(); t0 = time.time(); tot = 0.0
        for x, y in trl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with C.amp_ctx(device):
                loss = crit(model(x), y)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); tot += loss.item()
        pv, yv = predict(model, val, device)
        vf = f1_score(yv, (pv >= 0.5).astype(float), average="macro", zero_division=0)
        sched.step(vf)
        hist.append({"epoch": ep, "train_loss": tot / len(trl), "val_f1": vf,
                     "time_s": round(time.time() - t0, 1)})
        print(f"  ep {ep:02d} | loss {tot/len(trl):.4f} | val F1 {vf:.4f} | {time.time()-t0:.0f}s")
        if vf > best:
            best, wait = vf, 0
            torch.save({"model_state": model.state_dict(), "init": init, "seed": seed,
                        "fraction": frac, "epoch": ep, "val_f1": vf, "dropout": SUP_DROPOUT}, ckpt)
            print(f"  ✓ best val F1 {best:.4f}")
        else:
            wait += 1
            if wait >= SUP_EARLY:
                print("  early stopping"); break
    (C.RESULTS_DIR / f"{name}_log.json").write_text(json.dumps(hist, indent=2))
    del model, opt; torch.cuda.empty_cache()


# ╔══════════════════════════════════════════════════════════════╗
# ║  MAIN                                                         ║
# ╚══════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    label2idx, idx2label, TR, VA, TE = C.get_splits()
    cap_key = C.detect_key(TR, C.CAPTION_KEYS)
    loc_key = C.detect_key(TR, C.LOCATION_KEYS)
    if cap_key is None:
        raise KeyError(f"No caption field found. Record keys: {sorted(TR[0].keys())}. "
                       f"Add the correct key to CAPTION_KEYS in vl_common.py.")

    if RUN_AUDIT:
        audit(TR, VA, TE, cap_key, loc_key)

    if RUN_ALIGNMENT:
        print(f"\n{'='*60}\nCONTRASTIVE ALIGNMENT\n{'='*60}")
        for sd in SEEDS:
            for v in VARIANTS:
                train_alignment(v, sd, label2idx, TR, VA, cap_key, device)

    if RUN_SUPERVISED_FT:
        print(f"\n{'='*60}\nSUPERVISED FINE-TUNING\n{'='*60}")
        trl, val, pos_w = sup_loaders(label2idx, TR, VA)
        for sd in SUP_SEEDS:
            for init in SUP_INITS:
                train_supervised(init, label2idx, trl, val, pos_w, device, seed=sd)

    if RUN_LABEL_LIMITED:
        print(f"\n{'='*60}\nLABEL-LIMITED FINE-TUNING\n{'='*60}")
        pids = C.patient_ids(TR)
        for frac in LL_FRACTIONS:
            idx = sample_patient_subset(pids, frac, seed=LL_SUBSET_SEED)
            sub = [TR[i] for i in idx]
            print(f"  {frac:.0%} of training patients: {len(sub)} images, "
                  f"{len(set(pids[idx]))} patients")
            trl, val, pos_w = sup_loaders(label2idx, sub, VA)
            for sd in LL_SEEDS:
                for init in LL_INITS:
                    train_supervised(init, label2idx, trl, val, pos_w, device, seed=sd, frac=frac)

    C.write_manifest("train", {"variants": VARIANTS, "seeds": SEEDS, "epochs": EPOCHS,
                               "batch": BATCH_SIZE, "lr": LR, "wd": WEIGHT_DECAY,
                               "sup_inits": SUP_INITS, "sup_seeds": SUP_SEEDS,
                               "ll_fractions": LL_FRACTIONS, "ll_inits": LL_INITS})
    print("\n✓ Training finished. Next: python vl2_evaluate.py")
