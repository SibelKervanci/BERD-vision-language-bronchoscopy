# Report-Supervised Vision–Language Alignment for Label-Efficient Bronchoscopic Finding Recognition

Code for the manuscript

> Kervancı, İ.S. *Report-Supervised Vision–Language Alignment for Label-Efficient Bronchoscopic
> Finding Recognition.* Bioengineering (under review), 2026.

BiomedCLIP is aligned with the image–caption pairs of the public **Bronchoscopy Examination Report
Dataset (BERD)** and evaluated for zero-shot finding recognition, cross-modal retrieval, label-efficient
linear probing and fine-tuning, with patient-level bootstrap statistics, three training seeds and a
data-leakage audit. The data protocol (23 finding classes, patient-level 80/20 split with seed 42,
official test set) is identical to the companion benchmark
[BERD-multilabel-bronchoscopy](https://github.com/SibelKervanci/BERD-multilabel-bronchoscopy).

## Data

BERD is **not** included. Download it from the Science Data Bank:
<https://doi.org/10.57760/sciencedb.28018> and place it as

```
dataset/
├── annotations/dataset_train.json
├── annotations/dataset_test.json
└── images/
```

(or set `DATA_ROOT` in `vl_common.py`). The exact subset assignment of every image used in the paper is
provided in `results/split_assignment.csv`.

## Installation

```
pip install -r requirements.txt
```

A CUDA GPU is needed for training (developed on an RTX 4080, 16 GB). The BiomedCLIP weights are
downloaded from Hugging Face and DINOv2 from torch.hub on first use.

## Pipeline

| Step | Script | Purpose | GPU |
|---|---|---|---|
| 0 | `vl0_leakage_audit.py` | Patient overlap, exact/near duplicates, caption overlap, split export | No |
| 0b | `vl0b_inspect_duplicates.py` | Pixel correlation and contact sheets of duplicate candidates | No |
| – | `vl_check_location_mentions.py` | How often captions state the anatomical location | No |
| 1 | `vl1_train_alignment.py` | Contrastive alignment (3 objectives × 3 seeds), fine-tuning (all labels, 10 %, 25 %) | Yes |
| 2 | `vl2_evaluate.py` | Zero-shot, retrieval, linear probes, location, fine-tuning, seed summary, sensitivity | Partly |
| 3 | `vl3_figures_tables.py` | Main figures and tables from the result CSVs | No |
| 4 | `vl4_supplementary_tables.py` | Supplementary tables (LaTeX) from the result CSVs | No |

Shared code: `vl_common.py` (configuration, data split, datasets, models, prompts) and `vl_metrics.py`
(retrieval, thresholds, probes, metrics and patient-level bootstrap; NumPy/scikit-learn only).

```
python vl0_leakage_audit.py
python vl1_train_alignment.py
python vl2_evaluate.py
python vl0b_inspect_duplicates.py
python vl3_figures_tables.py
python vl4_supplementary_tables.py
```

Outputs are written to `results/`, `results/tables/`, `figures/` and `supplementary/`;
software versions of each run are recorded in `results/manifest_*.json`.

## Objectives

| Objective | Positives in the contrastive loss | Finding labels used? |
|---|---|---|
| Standard CLIP | own caption only | No |
| Caption-aware | all images with an identical normalised caption | No |
| Label-aware (reference) | identical caption or identical finding-label set | Yes |

## Leakage safeguards

- Patient-level split; zero patient overlap is asserted at load time and in the audit.
- The test set is used only in `vl2_evaluate.py`; model selection, early stopping, thresholds and probe
  regularisation use the validation subset.
- Exact (MD5) and near (difference hash) duplicates are listed and scored; all test metrics are
  recomputed without flagged test images.
- Captions are never a model input at test time; retrieval is also reported on test captions never seen
  during training.
- Linear probes are repeated with a fixed C and threshold 0.5, i.e. without validation labels.

## Citation

If you use this code, please cite the paper above (citation details will be updated on publication)
and the BERD dataset:

> Luo, X. et al. Towards Automated Reporting: A Bronchoscopy Report Dataset for Enhancing Multimodality
> Large Language Models. *Sci. Data* **13**, 339 (2026). https://doi.org/10.1038/s41597-026-06692-8

## License

MIT (code). BERD is distributed under its own terms by its authors.
