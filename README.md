# Label-Efficient Performance Monitoring of Breast Ultrasound Classifiers

When a breast ultrasound classifier is moved to a new hospital, its accuracy can drop because of
different scanners, operators and patient populations. Measuring this drop normally requires many
biopsy-confirmed labels, which are costly and slow to collect, especially in low-resource settings.

This repository contains the complete pipeline for estimating the **sensitivity, specificity and
positive predictive value (PPV)** of a breast ultrasound classifier at a new hospital using:

1. **no labels** (label-free estimation),
2. **an approximately known cancer rate** (for example, from pathology records), or
3. **a small number of labelled cases** (few-label estimation, including an empirical Bayes recalibration).

The pipeline runs end to end on a free Google Colab GPU.

---

## Overview of the pipeline

| Step | Description |
|---|---|
| Data harmonisation | Five public datasets are loaded into one manifest (benign vs. malignant lesions). |
| Leakage control | Near-duplicate images are detected with perceptual hashing; images from the same patient or near-duplicates are grouped, and all splits are made at the group level (60% train, 20% calibration, 20% test). |
| Feature extraction | Each lesion is cropped (15% margin) and encoded with frozen pre-trained models. |
| Classifier | Logistic regression on the embeddings, temperature scaling for calibration, and a decision threshold frozen at 90% source sensitivity. |
| Deployment | Single-source and leave-one-site-out deployments across sites; one site (BUSI-WHU) is held out and used only for final confirmation. |
| Estimation | Label-free, prevalence-informed and few-label estimators are compared against the true metrics at each target site. |
| Statistics | Cluster-bootstrap confidence intervals, Wilcoxon signed-rank tests with Holm correction, and sign tests. |

## Datasets

All datasets are publicly available. Please download them from their original sources and follow each
dataset's licence and citation requirements.

| Key | Dataset | Country | Source |
|---|---|---|---|
| `BUSI` | Breast Ultrasound Images (BUSI) | Egypt | [Kaggle mirror](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset) |
| `BUSBRA` | BUS-BRA | Brazil | [Zenodo, doi:10.5281/zenodo.8231412](https://doi.org/10.5281/zenodo.8231412) |
| `UCLM` | BUS-UCLM | Spain | [Mendeley Data, doi:10.17632/7fvgj4jsp7.3](https://doi.org/10.17632/7fvgj4jsp7.3) |
| `BrEaST` | BrEaST-Lesions-USG | Poland | [TCIA, doi:10.7937/9wkk-q141](https://doi.org/10.7937/9wkk-q141) |
| `WHU` | BUSI-WHU | China | [Mendeley Data, doi:10.17632/k6cpmwybk3.3](https://doi.org/10.17632/k6cpmwybk3.3) |

BUSI, BUS-BRA and BrEaST are downloaded automatically by the notebook. BUS-UCLM and BUSI-WHU are
attempted automatically; if the download fails, click **Download All** on the dataset page, rename the
file to `UCLM.zip` or `WHU.zip`, and place it in `MyDrive/bus_data/raw/`.

Images that show only normal breast tissue are excluded.

## Pre-trained models

| Key | Model | Pre-training |
|---|---|---|
| `resnet50` | ResNet-50 | ImageNet, supervised |
| `dinov2_b14` | DINOv2 ViT-B/14 | Natural images, self-supervised |
| `clip_b16` | CLIP ViT-B/16 | Image–text pairs |
| `biomedclip` | BiomedCLIP | Biomedical image–text pairs |
| `medsiglip` *(optional)* | MedSigLIP-448 | Medical images (requires accepting the licence on Hugging Face and an `HF_TOKEN`) |

All models are used frozen; no fine-tuning is performed.

## Estimators

**Label-free:** source performance (no change), ATC, DoC, CBPE, and prevalence-adjusted CBPE with the
prevalence estimated by EM or BBSE.

**Known cancer rate:** prevalence-adjusted CBPE with the target prevalence given exactly or with ±5 or
±10 percentage points of error.

**Few labels (K labelled cases):** labels only, prediction-powered inference (PPI), Platt recalibration
on K cases, and the proposed empirical Bayes recalibration, whose prior is centred on CBPE and whose
strength is chosen by evidence maximisation (it reduces to CBPE when K = 0).

## How to run

1. Open `notebooks/breast_ultrasound_performance_monitoring.ipynb` in Google Colab
   (*File → Upload notebook*, or open it directly from GitHub).
2. Select **Runtime → Change runtime type → T4 GPU**.
3. Run all cells from top to bottom. Allow access to Google Drive when asked.

The notebook in this repository is saved **with the outputs of a complete run**, so the printed tables
and figures can be viewed on GitHub without running anything.

Only cell 5 (feature extraction) needs a GPU; cells 6–9 run on CPU.

### Approximate run time (free Colab T4)

| Cell | Time |
|---|---|
| 2 – download and unpack | a few minutes on the first run, about 1 minute afterwards |
| 4 – manifest and duplicate detection | a few minutes |
| 5 – embeddings (4 models) | about 5 minutes per model |
| 6 – main experiment | about 10–20 minutes |
| 7 – few-label and sample-size experiments | about 10–20 minutes |
| 8–9 – tables, figures and supplementary analyses | about 1 minute |

### Troubleshooting

| Problem | Fix |
|---|---|
| A dataset shows `MISSING` in cell 2 | Download it manually from the link above, rename it (`UCLM.zip`, `WHU.zip`, ...) and place it in `MyDrive/bus_data/raw/`, then re-run cell 2. |
| A loader reports `no images found` | Run `inspect(DATASET_PATHS["<key>"])` in a new cell to see the folder layout of that dataset. |
| `CUDA out of memory` in cell 5 | Reduce `batch_size=64` to `16` in cell 5. |
| `NameError` after a disconnection | Run cells 1–3 again; finished steps are reloaded from the checkpoints. |
| `device: cpu` in cell 5 | Select **Runtime → Change runtime type → T4 GPU**, then run all cells again. |

### Checkpoints

All intermediate results are saved to Google Drive. If Colab disconnects, run all cells again and
finished work is skipped automatically.

```
MyDrive/bus_data/
├── raw/        dataset zip files (downloaded once)
├── work/       manifest, embeddings and classifier outputs
└── results/    per-model results, tables (CSV) and figures (PNG)
```

## Configuration

All settings are in `CONFIG` in cell 1.

| Setting | Default | Meaning |
|---|---|---|
| `image_mode` | `"roi"` | `"roi"` = square crop around the lesion, `"full"` = whole image |
| `roi_margin` | `0.15` | Margin around the lesion bounding box |
| `dedup_max_hamming` | `6` | Perceptual-hash distance below which two images are treated as the same |
| `target_sens` | `0.90` | Operating point fixed on the source calibration set |
| `prev_grid` | `[0.1, 0.2, 0.3, 0.5, 0.7]` | Target prevalences for the controlled prevalence experiment |
| `reps`, `n_sub` | `50`, `100` | Repeats and cases per resampled target |
| `models` | four models | Pre-trained models to evaluate |
| `few_k` | `[0, 5, 10, 20, 40, 80]` | Numbers of labelled target cases |
| `n_unlab` | `[25, 50, 100, 200, 400]` | Numbers of unlabelled target images |
| `known_prev_err` | `[0.05, 0.10]` | Error of the known cancer rate |
| `holdout` | `"WHU"` | Site held out for final confirmation |
| `seed` | `0` | Random seed |

## Outputs

All outputs are written to `MyDrive/bus_data/results/`.

| File | Content |
|---|---|
| `T0_datasets.csv` | Images, groups and malignancy rate per dataset |
| `T1_true_performance.csv` | Source vs. target performance for every deployment |
| `T2_estimation_error.csv`, `T2b_*`, `T2c_*` | Label-free estimation error with 95% CIs and significance tests |
| `T3_failure_detection.csv` | Detection of deployments with sensitivity below 0.85 |
| `T4_model_selection.csv` | Label-free model selection |
| `T5_prevalence_stress.csv`, `T5b_*` | Error under controlled prevalence shift; known cancer rate |
| `T6_fewlabel_error.csv`, `T6b_*`, `T7_labels_needed.csv` | Few-label error, significance tests and labels needed |
| `T8_samplesize_error.csv` | Error vs. number of unlabelled images |
| `T9_*`, `T10_*` | Held-out confirmation site results |
| `S1a_*` – `S4_*` | Supplementary analyses (pair-level tests, per model, per site, failure detection) |
| `F1`–`F5` (`.png`) | Figures |
| `raw_*.csv.gz` | Raw results for every deployment, model and repeat |

## Repository structure

```
.
├── notebooks/
│   └── breast_ultrasound_performance_monitoring.ipynb   main notebook (run this)
├── src/
│   └── pipeline.py                                     same code as the notebook, as a plain Python file
├── requirements.txt
├── LICENSE
└── README.md
```

## Reproducibility

- All random operations use fixed seeds (`CONFIG["seed"]`).
- Splits are made at the patient/near-duplicate group level to prevent data leakage.
- The held-out site is never used for training or method design.
- Results are tied to a fingerprint of the data and settings, so stale checkpoints are never reused.

## Licence

The code is released under the MIT License (see `LICENSE`). The datasets are not redistributed here
and remain subject to their own licences.
