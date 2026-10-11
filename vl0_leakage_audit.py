"""
vl0_leakage_audit.py -- Data-integrity and leakage audit for the BERD vision-language study.
CPU only, a few minutes. Run once; the outputs are used by vl2_evaluate.py (sensitivity
analyses) and belong in the Supplementary Material.

Checks
  1. Patient overlap between training, validation and test subsets (must be zero).
  2. Exact duplicates (MD5 of the image file) across subsets despite different patients.
  3. Near duplicates (64-bit difference hash, Hamming distance <= NEAR_DUP_BITS) across subsets.
  4. Caption overlap: test images whose (normalised) caption also occurs in the training
     subset. The aligned text encoder has seen these TEXTS (never the test images); retrieval
     is therefore also reported on test captions never seen during training.
  5. Label distribution of each subset (prevalence per class).
  6. Export of the exact split (image id, patient id, subset) for reproducibility.
  7. Software / hardware manifest.

Outputs (results/):
  audit_leakage_report.json, audit_duplicates.csv, audit_label_prevalence.csv,
  split_assignment.csv, manifest_audit.json
"""

import hashlib
import json

import numpy as np
import pandas as pd
from PIL import Image

import vl_common as C
from vl_metrics import norm_text

NEAR_DUP_BITS = 4          # dHash Hamming distance regarded as a near duplicate


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dhash(path, size=8):
    """64-bit difference hash (robust to resizing / recompression)."""
    g = Image.open(path).convert("L").resize((size + 1, size), Image.BILINEAR)
    a = np.asarray(g, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return np.packbits(bits)                       # 8 bytes


def hamming_matrix(A, B):
    """A: (n, 8) uint8, B: (m, 8) uint8 -> (n, m) Hamming distances."""
    x = np.bitwise_xor(A[:, None, :], B[None, :, :])
    return np.unpackbits(x, axis=2).sum(2)


if __name__ == "__main__":
    label2idx, idx2label, TR, VA, TE = C.get_splits()
    cap_key = C.detect_key(TR, C.CAPTION_KEYS)
    splits = {"train": TR, "val": VA, "test": TE}
    report = {}

    # 1. patient overlap
    P = {k: set(C.patient_ids(v)) for k, v in splits.items()}
    report["patient_overlap"] = {f"{a}-{b}": len(P[a] & P[b])
                                 for a, b in [("train", "val"), ("train", "test"), ("val", "test")]}
    print("[1] patient overlap:", report["patient_overlap"])

    # 2-3. exact and near duplicates
    info = {}
    for k, recs in splits.items():
        paths = [C.resolve_path(r["image_path"]) for r in recs]
        info[k] = {"md5": [md5(p) for p in paths],
                   "dh": np.stack([dhash(p) for p in paths]),
                   "ids": [str(r.get("image_id", r["image_path"])) for r in recs],
                   "pid": C.patient_ids(recs)}
        print(f"    hashed {k}: {len(recs)} images")
    pairs = []
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        idx_b = {}
        for j, h in enumerate(info[b]["md5"]):
            idx_b.setdefault(h, []).append(j)
        for i, h in enumerate(info[a]["md5"]):
            for j in idx_b.get(h, []):
                pairs.append({"split_a": a, "idx_a": i, "id_a": info[a]["ids"][i],
                              "patient_a": info[a]["pid"][i], "split_b": b, "idx_b": j,
                              "id_b": info[b]["ids"][j], "patient_b": info[b]["pid"][j],
                              "type": "exact", "hamming": 0})
        D = hamming_matrix(info[b]["dh"], info[a]["dh"])          # (n_b, n_a)
        for j, i in zip(*np.where(D <= NEAR_DUP_BITS)):
            if info[a]["md5"][i] == info[b]["md5"][j]:
                continue
            pairs.append({"split_a": a, "idx_a": int(i), "id_a": info[a]["ids"][i],
                          "patient_a": info[a]["pid"][i], "split_b": b, "idx_b": int(j),
                          "id_b": info[b]["ids"][j], "patient_b": info[b]["pid"][j],
                          "type": "near", "hamming": int(D[j, i])})
    dup = pd.DataFrame(pairs)
    dup.to_csv(C.RESULTS_DIR / "audit_duplicates.csv", index=False)
    summ = {}
    for (a, b), g in (dup.groupby(["split_a", "split_b"]) if len(dup) else []):
        summ[f"{a}-{b}"] = {"exact_pairs": int((g.type == "exact").sum()),
                            "near_pairs": int((g.type == "near").sum()),
                            f"{b}_images_affected": int(g.idx_b.nunique())}
    report["duplicates"] = summ
    test_dup = sorted(set(dup.loc[dup.split_b == "test", "idx_b"])) if len(dup) else []
    report["test_images_with_duplicate_in_train_or_val"] = len(test_dup)
    print("[2-3] duplicates:", summ)

    # 4. caption overlap
    tr_caps = {norm_text(c) for c in C.captions_of(TR, cap_key)}
    te_caps = [norm_text(c) for c in C.captions_of(TE, cap_key)]
    seen = np.array([c in tr_caps for c in te_caps])
    report["caption_overlap"] = {
        "test_images_caption_seen_in_train": int(seen.sum()),
        "test_images_caption_seen_in_train_%": round(100 * seen.mean(), 1),
        "unique_test_captions": len(set(te_caps)),
        "unique_test_captions_seen_in_train": len(set(te_caps) & tr_caps)}
    print("[4] caption overlap:", report["caption_overlap"])

    # 5. label prevalence
    prev = pd.DataFrame({k: C.label_matrix(v, label2idx).mean(0) for k, v in splits.items()},
                        index=idx2label)
    prev["n_test_pos"] = C.label_matrix(TE, label2idx).sum(0).astype(int)
    prev.round(4).to_csv(C.RESULTS_DIR / "audit_label_prevalence.csv")

    # 6. split export
    rows = [{"image_id": r.get("image_id", ""), "image_path": r["image_path"],
             "patient_id": r.get("patient_id", ""), "subset": k}
            for k, recs in splits.items() for r in recs]
    pd.DataFrame(rows).to_csv(C.RESULTS_DIR / "split_assignment.csv", index=False)

    # 7. manifest
    C.write_manifest("audit", {"NEAR_DUP_BITS": NEAR_DUP_BITS})
    (C.RESULTS_DIR / "audit_leakage_report.json").write_text(json.dumps(report, indent=2))
    print("\n✓ results/audit_leakage_report.json, audit_duplicates.csv, split_assignment.csv")
