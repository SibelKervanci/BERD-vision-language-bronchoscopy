"""
vl_common.py -- configuration, data handling and PyTorch utilities for the BERD
vision-language study (report-supervised alignment, zero-shot, few-shot, retrieval).

This project is SELF-CONTAINED: it does not import any code of other projects.
To keep results comparable with earlier BERD work, the data protocol is reproduced
exactly:
  - 23 finding classes in alphabetical order ("diverticulum", n = 2, excluded);
  - records kept only if the image exists and carries at least one of the 23 labels;
  - patient-level 80/20 split of the official BERD training set (sorted patient ids,
    numpy default_rng(42) shuffle, first 20 % -> validation);
  - the official BERD test set, unchanged and in its original order.
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms



# ╔══════════════════════════════════════════════════════════════╗
# ║  CONFIG                                                       ║
# ╚══════════════════════════════════════════════════════════════╝

BASE_DIR   = Path(__file__).resolve().parent
DATA_ROOT  = BASE_DIR / "dataset"          # BERD as downloaded: annotations/ and images/
TRAIN_JSON = DATA_ROOT / "annotations" / "dataset_train.json"
TEST_JSON  = DATA_ROOT / "annotations" / "dataset_test.json"
IMAGE_DIR  = DATA_ROOT / "images"

CKPT_VL     = BASE_DIR / "checkpoints_vl"        # contrastive (image-text) alignment
CKPT_SUP    = BASE_DIR / "checkpoints_sup"       # supervised fine-tuning
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"
FEAT_CACHE  = RESULTS_DIR / "features"
PRED_CACHE  = RESULTS_DIR / "preds_cache"
for _d in [CKPT_VL, CKPT_SUP, RESULTS_DIR, FIGURES_DIR, FEAT_CACHE, PRED_CACHE]:
    _d.mkdir(parents=True, exist_ok=True)

# Optional, read-only: folder with prediction files (*.npz with val_probs, val_labels,
# test_probs, test_labels) of previously published BERD models, used only as external
# reference rows in the supervised comparison. Leave as None to skip.
EXTERNAL_PREDS_DIR = None   # e.g. Path("C:/sibel/diagnostics/results/preds_cache")

SEED         = 42
VAL_FRACTION = 0.20

KNOWN_LABELS = [
    "blood", "clot", "congested", "edematous", "external pressure",
    "fistula", "granulation", "infiltration changes", "mass", "narrow",
    "necrotic", "neoplasm", "new organism", "nodules", "normal",
    "pigmentation", "postoperative change", "rough", "sputum",
    "surgical stump", "tube", "ulcer", "widened",
]

BIOMEDCLIP_ID = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)
IN_MEAN   = (0.485, 0.456, 0.406)
IN_STD    = (0.229, 0.224, 0.225)

# JSON keys are detected automatically; the first key present in most records is used.
CAPTION_KEYS  = ["caption", "description", "text", "sentence", "report", "finding", "findings"]
LOCATION_KEYS = ["location", "position", "site", "anatomical_location", "anatomy"]

TEXT_CTX           = 128     # BERD captions: mean 8 words, max 46 (tower supports 256)
TEMPLATE_MIN_COUNT = 20      # captions seen >= 20 times in TRAIN = frequent captions
NUM_WORKERS        = 0       # Windows/Spyder: keep 0 unless running from a terminal
EVAL_BATCH         = 128


# ╔══════════════════════════════════════════════════════════════╗
# ║  DATA                                                         ║
# ╚══════════════════════════════════════════════════════════════╝

def parse_labels(lf, valid=None):
    if lf is None:
        return []
    raw = [l.strip().lower() for l in
           (lf if isinstance(lf, list) else str(lf).split(",")) if str(l).strip()]
    return [l for l in raw if l in valid] if valid else raw


def resolve_path(ip, image_dir=IMAGE_DIR):
    p = Path(ip); c = Path(image_dir) / p.name
    return c if c.exists() else DATA_ROOT / ip


def load_json(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return d if isinstance(d, list) else d.get("annotations", [])


def label_encoder():
    return {l: i for i, l in enumerate(KNOWN_LABELS)}, list(KNOWN_LABELS)


def _valid_records(path, valid):
    return [r for r in load_json(path)
            if resolve_path(r.get("image_path", "")).exists()
            and parse_labels(r.get("label", ""), valid)]


def get_splits(verbose=True):
    """Returns label2idx, idx2label, train_records, val_records, test_records."""
    label2idx, idx2label = label_encoder()
    valid = set(label2idx)
    recs = _valid_records(TRAIN_JSON, valid)
    pids = sorted({r.get("patient_id", "") for r in recs if r.get("patient_id", "")})
    rng = np.random.default_rng(SEED); rng.shuffle(pids)
    n_val = int(len(pids) * VAL_FRACTION)
    val_p, tr_p = set(pids[:n_val]), set(pids[n_val:])
    train = [r for r in recs if r.get("patient_id", "") in tr_p]
    val = [r for r in recs if r.get("patient_id", "") in val_p]
    test = _valid_records(TEST_JSON, valid)
    assert not (tr_p & val_p)
    test_p = {r.get("patient_id", "") for r in test}
    assert not (test_p & (tr_p | val_p)), "patient overlap between test and training pool"
    if verbose:
        print(f"[Split] train {len(train)} img / {len(tr_p)} pat | val {len(val)} / {len(val_p)} "
              f"| test {len(test)} / {len(test_p)}")
    return label2idx, idx2label, train, val, test


def label_matrix(recs, label2idx):
    Y = np.zeros((len(recs), len(label2idx)), dtype=np.float32)
    valid = set(label2idx)
    for i, r in enumerate(recs):
        for l in parse_labels(r.get("label", ""), valid):
            Y[i, label2idx[l]] = 1.0
    return Y


def patient_ids(recs):
    return np.array([str(r.get("patient_id", f"img{i}")) for i, r in enumerate(recs)])


def detect_key(records, candidates):
    for k in candidates:
        if sum(1 for r in records if str(r.get(k, "")).strip()) > 0.5 * len(records):
            return k
    return None


def captions_of(recs, key):
    return [str(r.get(key, "") or "") for r in recs]


def label_set_keys(recs, label2idx):
    valid = set(label2idx)
    return ["|".join(sorted(parse_labels(r.get("label", ""), valid))) for r in recs]


# ╔══════════════════════════════════════════════════════════════╗
# ║  TRANSFORMS + DATASETS                                        ║
# ╚══════════════════════════════════════════════════════════════╝

def image_tf(split, norm="clip"):
    """Augmentation pipeline of the earlier BERD benchmark; normalisation statistics
    depend on the encoder (CLIP for BiomedCLIP, ImageNet for vision-only encoders)."""
    mean, std = (CLIP_MEAN, CLIP_STD) if norm == "clip" else (IN_MEAN, IN_STD)
    if split == "train":
        return transforms.Compose([
            transforms.Resize((256, 256)), transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(0.5), transforms.RandomVerticalFlip(0.2),
            transforms.RandomRotation(15), transforms.ColorJitter(0.25, 0.25, 0.1, 0.05),
            transforms.RandomAutocontrast(0.4), transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomApply([transforms.GaussianBlur(3, (0.1, 1.5))], 0.2)])
    return transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                               transforms.Normalize(mean, std)])


class ImageDataset(Dataset):
    """Images only (feature extraction / inference)."""
    def __init__(self, recs, tf):
        self.recs, self.tf = recs, tf
    def __len__(self): return len(self.recs)
    def __getitem__(self, i):
        return self.tf(Image.open(resolve_path(self.recs[i]["image_path"])).convert("RGB")), i


class LabelledImageDataset(Dataset):
    """Images + multi-hot labels (supervised fine-tuning)."""
    def __init__(self, recs, label2idx, tf):
        self.recs, self.tf = recs, tf
        self.Y = torch.as_tensor(label_matrix(recs, label2idx))
    def __len__(self): return len(self.recs)
    def __getitem__(self, i):
        img = Image.open(resolve_path(self.recs[i]["image_path"])).convert("RGB")
        return self.tf(img), self.Y[i]


class ImageTextDataset(Dataset):
    """Image + tokenised caption + caption id + label-set id (contrastive training)."""
    def __init__(self, recs, tf, tokenizer, cap_key, cap_ids, set_ids):
        self.recs, self.tf = recs, tf
        self.tokens = tokenizer(captions_of(recs, cap_key), context_length=TEXT_CTX)
        self.cap_ids = torch.as_tensor(cap_ids, dtype=torch.long)
        self.set_ids = torch.as_tensor(set_ids, dtype=torch.long)
    def __len__(self): return len(self.recs)
    def __getitem__(self, i):
        img = self.tf(Image.open(resolve_path(self.recs[i]["image_path"])).convert("RGB"))
        return img, self.tokens[i], self.cap_ids[i], self.set_ids[i]


def make_loader(ds, shuffle=False, batch=EVAL_BATCH, drop_last=False):
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=NUM_WORKERS,
                      pin_memory=torch.cuda.is_available(), drop_last=drop_last)


# ╔══════════════════════════════════════════════════════════════╗
# ║  MODELS                                                       ║
# ╚══════════════════════════════════════════════════════════════╝

def amp_ctx(device):
    return torch.autocast(device_type="cuda", dtype=torch.float16,
                          enabled=(device.type == "cuda"))


def load_biomedclip(device):
    """Pretrained BiomedCLIP (image + text towers) and its tokenizer.
    Requires: pip install open_clip_torch transformers (weights from Hugging Face)."""
    import open_clip
    model, _ = open_clip.create_model_from_pretrained(BIOMEDCLIP_ID)
    tokenizer = open_clip.get_tokenizer(BIOMEDCLIP_ID)
    return model.to(device), tokenizer


def load_aligned(ckpt_path, device):
    model, tok = load_biomedclip(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model_state"])
    return model, tok


class BiomedCLIPClassifier(nn.Module):
    """BiomedCLIP image tower + multi-label head (LayerNorm-Dropout-Linear(512)-GELU-
    Dropout-Linear), i.e. the head used for Transformer backbones in the earlier BERD
    benchmark. Inputs are CLIP-normalised. `visual_state` optionally initialises the
    image tower from an aligned checkpoint."""
    def __init__(self, nc, dropout=0.4, visual_state=None):
        super().__init__()
        import open_clip
        clip, _ = open_clip.create_model_from_pretrained(BIOMEDCLIP_ID)
        self.b = clip.visual; del clip
        if visual_state is not None:
            self.b.load_state_dict(visual_state, strict=True)
        with torch.no_grad():
            was = self.b.training; self.b.eval()
            f = self.b(torch.zeros(1, 3, 224, 224)).shape[-1]; self.b.train(was)
        self.h = nn.Sequential(nn.LayerNorm(f), nn.Dropout(dropout), nn.Linear(f, 512),
                               nn.GELU(), nn.Dropout(dropout / 2), nn.Linear(512, nc))
    def forward(self, x):
        return self.h(self.b(x))


def visual_state_from(ckpt_path):
    state = torch.load(ckpt_path, map_location="cpu")["model_state"]
    return {k[len("visual."):]: v for k, v in state.items() if k.startswith("visual.")}


@torch.no_grad()
def encode_images(model, loader, device):
    """L2-normalised image features; works for CLIP models and FrozenVisionEncoder."""
    model.eval(); out = []
    for x, _ in loader:
        with amp_ctx(device):
            f = model.encode_image(x.to(device, non_blocking=True))
        out.append(F.normalize(f.float(), dim=-1).cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def encode_texts(model, tokenizer, texts, device, batch=EVAL_BATCH):
    model.eval(); out = []
    for i in range(0, len(texts), batch):
        tok = tokenizer(texts[i:i + batch], context_length=TEXT_CTX).to(device)
        with amp_ctx(device):
            f = model.encode_text(tok)
        out.append(F.normalize(f.float(), dim=-1).cpu())
    return torch.cat(out).numpy()


class FrozenVisionEncoder(nn.Module):
    """Vision-only reference encoders for the linear probe (no text tower).
    dinov2          : CLS token (768-d)
    dinov2_clspatch : CLS token + mean of patch tokens (1536-d), the linear-probe protocol
                      of the DINOv2 paper (stronger reference)
    swin_t_imagenet : ImageNet-1k Swin-T pooled features (768-d)"""
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        if kind in ("dinov2", "dinov2_clspatch"):
            self.b = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        elif kind == "swin_t_imagenet":
            from torchvision import models as M
            from torchvision.models import Swin_T_Weights
            b = M.swin_t(weights=Swin_T_Weights.IMAGENET1K_V1); b.head = nn.Identity()
            self.b = b
        else:
            raise ValueError(kind)
    def encode_image(self, x):
        if self.kind == "dinov2_clspatch":
            o = self.b.forward_features(x)
            return torch.cat([o["x_norm_clstoken"], o["x_norm_patchtokens"].mean(1)], dim=-1)
        return self.b(x)


def write_manifest(tag, settings=None):
    """Software/hardware versions and run settings -> results/manifest_<tag>.json."""
    import platform, sys, datetime
    info = {"tag": tag, "time": datetime.datetime.now().isoformat(timespec="seconds"),
            "python": sys.version.split()[0], "platform": platform.platform(),
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "seed_split": SEED, "val_fraction": VAL_FRACTION, "text_ctx": TEXT_CTX,
            "frequent_caption_min_count": TEMPLATE_MIN_COUNT, "settings": settings or {}}
    for mod in ["torchvision", "open_clip", "transformers", "sklearn", "numpy", "pandas", "scipy"]:
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            info[mod] = None
    (RESULTS_DIR / f"manifest_{tag}.json").write_text(json.dumps(info, indent=2, default=str))
    return info


# ╔══════════════════════════════════════════════════════════════╗
# ║  ZERO-SHOT PROMPTS                                            ║
# ╚══════════════════════════════════════════════════════════════╝
# Two prompt sets, both FIXED A PRIORI (before any test-set evaluation):
#   "generic"    : textbook English phrasing inserted into bronchoscopy templates.
#   "berd_vocab" : phrasing written in the style of BERD captions, using only the
#                  vocabulary of the TRAINING captions (results/audit_top_captions.csv).
#                  No label-caption statistics were used to write them, so zero-shot
#                  remains free of finding labels.
# Comparing the two sets shows how much zero-shot performance depends on matching the
# reporting vocabulary of the target institution.

PROMPT_TEMPLATES = [
    "a bronchoscopic image showing {}.",
    "bronchoscopy shows {}.",
    "{} observed during bronchoscopy.",
]

CLASS_PHRASES = {
    "blood":                ["blood in the airway", "bleeding"],
    "clot":                 ["a blood clot", "clotted blood"],
    "congested":            ["congested mucosa", "mucosal congestion"],
    "edematous":            ["edematous mucosa", "mucosal edema"],
    "external pressure":    ["extrinsic compression of the airway", "external pressure on the bronchus"],
    "fistula":              ["a fistula", "a fistula opening"],
    "granulation":          ["granulation tissue", "granulation"],
    "infiltration changes": ["mucosal infiltration", "infiltrative changes"],
    "mass":                 ["a mass", "a mass in the airway"],
    "narrow":               ["a narrowed airway lumen", "airway stenosis"],
    "necrotic":             ["necrotic tissue", "necrosis"],
    "neoplasm":             ["a neoplasm", "a tumor"],
    "new organism":         ["a new growth", "a neoplastic growth"],
    "nodules":              ["mucosal nodules", "nodular mucosa"],
    "normal":               ["normal airway mucosa", "a normal bronchus"],
    "pigmentation":         ["mucosal pigmentation", "pigmented mucosa"],
    "postoperative change": ["postoperative changes", "changes after surgery"],
    "rough":                ["rough mucosal surface", "rough mucosa"],
    "sputum":               ["sputum", "secretions in the airway"],
    "surgical stump":       ["a surgical stump", "a bronchial stump"],
    "tube":                 ["a tube in the airway", "an airway stent"],
    "ulcer":                ["an ulcer", "mucosal ulceration"],
    "widened":              ["a widened airway lumen", "a dilated bronchus"],
}

# BERD reporting style: short phrases, "lumen"/"tube cavity", "see ...", "can be seen".
BERD_PROMPTS = {
    "blood":                ["a small amount of blood", "a small amount of fresh blood can be seen",
                             "dark red blood marks can be seen"],
    "clot":                 ["dark red blood clots can be seen in the lumen",
                             "blood clots block the tube cavity"],
    "congested":            ["mild mucosa congestion", "mucosal congestion", "mucosa slightly congested"],
    "edematous":            ["mucosal edema", "mucosa slightly congested and edema",
                             "swelling of the mucosa"],
    "external pressure":    ["external pressure stenosis", "external pressure on the lumen"],
    "fistula":              ["a fistula can be seen", "fistula in the lumen"],
    "granulation":          ["granulation can be seen", "granulation tissue in the lumen"],
    "infiltration changes": ["mucosal infiltration changes", "infiltration of the mucosa"],
    "mass":                 ["a mass can be seen in the lumen", "mass"],
    "narrow":               ["stenotic opening", "stenosis of the opening", "lumen stenosis"],
    "necrotic":             ["white necrotics attached to the surface", "necrotic tissue can be seen"],
    "neoplasm":             ["neoplasms block the tube cavity", "neoplasms can be seen in the lumen"],
    "new organism":         ["new organisms can be seen in the lumen", "new organism"],
    "nodules":              ["nodules can be seen on the mucosa", "nodular mucosa"],
    "normal":               ["it is normal"],
    "pigmentation":         ["mucosal pigmentation", "pigmentation can be seen"],
    "postoperative change": ["postoperative changes", "postoperative changes can be seen"],
    "rough":                ["the mucosa is rough", "rough mucosa"],
    "sputum":               ["medium yellow-white sticky sputum in the lumen",
                             "a little white sticky phlegm is seen in the lumen",
                             "a large amount of yellow and white sticky sputum in the cavity"],
    "surgical stump":       ["surgical stump", "see the surgical stump"],
    "tube":                 ["see y-type silicone tracheal bronchial support in the trachea",
                             "see tracheal tube", "a stent was seen in the trachea"],
    "ulcer":                ["an ulcer can be seen on the mucosa", "mucosal ulcer"],
    "widened":              ["widening", "the lumen is widened"],
}

PROMPT_SETS = ["generic", "berd_vocab"]

LOCATION_TEMPLATES = ["a bronchoscopic image of the {}.", "bronchoscopic view of the {}."]


def class_prompts(lab, prompt_set="generic"):
    if prompt_set == "berd_vocab":
        return BERD_PROMPTS.get(lab, [lab])
    return [t.format(p) for t in PROMPT_TEMPLATES for p in CLASS_PHRASES.get(lab, [lab])]


def class_text_embeddings(model, tokenizer, idx2label, device, prompt_set="generic"):
    """Prompt-ensemble embedding per class (mean of normalised prompt embeddings)."""
    embs = []
    for lab in idx2label:
        e = encode_texts(model, tokenizer, class_prompts(lab, prompt_set), device).mean(0)
        embs.append(e / np.linalg.norm(e))
    return np.stack(embs)


def location_text_embeddings(model, tokenizer, loc_names, device):
    embs = []
    for loc in loc_names:
        e = encode_texts(model, tokenizer, [t.format(loc) for t in LOCATION_TEMPLATES], device).mean(0)
        embs.append(e / np.linalg.norm(e))
    return np.stack(embs)
