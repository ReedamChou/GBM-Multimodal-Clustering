# Anatomically Informed MRI Radiomics for Interpretable Survival Risk Stratification in Glioblastoma Multiforme (GBM)

**Authors:** Department of Computer Engineering, [University Name]

> **Purpose.** This study evaluates whether integrating global volumetric ratios with lobe-level radiomic features from T1-weighted MRI improves survival risk stratification in Glioblastoma Multiforme. A machine learning pipeline combining tree-based feature selection with an XGBoost classifier, evaluated under a stratified 10-fold cross-validation scheme, is used to classify patients as high-risk (OS ≤ 12 months) versus low-risk (OS > 12 months) based on exactly sixteen radiomic descriptors.

> **Conference notice:** This repository is associated with a paper under review. Please do not redistribute.

---

## Setup

### 1. Create and activate virtual environment

```bash
# Create (if not already done)
python -m venv .venv

# Activate
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux/macOS:
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note on ANTs:** `antspy` is required for atlas-to-patient registration but is not available via pip on Windows. On Linux/macOS, install separately: `pip install antspy`. On Windows, set `atlas.use_ants_registration: false` in `config.json` to use the affine fallback.

### 3. Place data

#### UCSF-PDGM dataset

The UCSF-PDGM imaging data should be placed under `UCSF/`:

```
UCSF/
├── DATA-IMAGE-STRUCTURAL/
│   └── UCSF-PDGM-XXXX/
│       └── UCSF-PDGM-XXXX_T1.nii.gz
└── DATA-AUTOMATED-SEGMENT/
    └── UCSF-PDGM-XXXX_tumor_segmentation.nii.gz
```

Clinical metadata (including OS in days) is at: `data/raw/ucsf_with_clinical.csv`

#### SRI24 atlas

Download the SRI24 atlas from [NITRC](https://www.nitrc.org/projects/sri24/) and place the following files in `data/SRI24/`:

| File | Description |
|------|-------------|
| `tzo116plus.nii.gz` | Cortical parcellation (~116 regions) |
| `suptent.nii.gz` | Supratentorial mask |
| `tissues.nii.gz` | Brain tissue mask |
| `SRI24-tzo116plus.txt` | Label ID → region name mapping |
| `T1.nii.gz` | T1 template (for ANTs registration) |

> If the T1 template has a different filename in your download, update `data.sri24_t1_filename` in `config.json`.

---

## Run Order

```bash
# Step 1: Atlas registration + feature extraction
python src/atlas_registration.py            # full batch (~1-3 min/patient)
python src/atlas_registration.py --dry-run  # list patients without processing
python src/atlas_registration.py --patient UCSF-PDGM-0004  # single patient

# Step 2: Preprocessing (binary labels, imputation)
python src/preprocessing.py

# Step 3: Training (XGBoost, 10-fold CV, SHAP)
python src/train.py
```

---

## Project Structure

```
gbm_survival_stratification/
├── data/
│   ├── raw/ucsf_with_clinical.csv      # Clinical metadata + OS
│   └── SRI24/                          # SRI24 atlas files (not committed)
├── src/
│   ├── atlas_registration.py           # Step 1: T1 registration + feature extraction
│   ├── preprocessing.py                # Step 2: Binary labels, imputation, scaling
│   └── train.py                        # Step 3: XGBoost, feature selection, CV, SHAP
├── docs/
│   ├── atlas_registration.md           # Documents atlas_registration.py
│   ├── preprocessing.md                # Documents preprocessing.py
│   └── training.md                     # Documents train.py
├── outputs/                            # CSVs, model artifacts, plots (gitignored)
│   ├── sri24_4lobe_atlas.nii.gz        # Cached lobe atlas in SRI24 space
│   ├── transforms/sub-XXX/            # Per-patient ANTs transform cache
│   ├── features_raw.csv                # Raw features before preprocessing
│   ├── features_processed.csv          # Clean features + binary risk_label
│   ├── selected_features.txt           # Features selected by tree-based selector
│   ├── cv_results.json                 # Fold AUCs, mean, CI, p-value
│   ├── shap_summary.png               # SHAP beeswarm plot
│   ├── shap_importance.png             # SHAP bar chart
│   ├── final_model.json                # Saved XGBoost model
│   └── training_report.txt             # Human-readable training summary
├── config.json                         # All tunable parameters
└── README.md                           # This file
```

**Total Python files: 3.** That's it.

---

## The 16 Features

### Global features (4)

| Feature | Formula |
|---------|---------|
| `global_nc_en_ratio` | NC voxels / ET voxels |
| `global_ed_en_ratio` | ED voxels / ET voxels |
| `global_ed_total_ratio` | ED voxels / WT voxels (WT = NC+ET+ED) |
| `tumor_burden_index` | WT voxels / total brain voxels |

### Lobe-wise features (12 = 4 lobes × 3 subregions)

| Feature pattern | Formula |
|----------------|---------|
| `{lobe}_ed_ratio` | ED voxels in lobe / total lobe voxels |
| `{lobe}_en_ratio` | ET voxels in lobe / total lobe voxels |
| `{lobe}_nc_ratio` | NC voxels in lobe / total lobe voxels |

Lobes: frontal, temporal, parietal, occipital.

**BraTS label map:** Label 1 = NC, Label 2 = ED, Label 4 = ET.

---

## Output Descriptions

| File | Description |
|------|-------------|
| `features_raw.csv` | 16 features + patient_id + OS_months + QA flag |
| `features_processed.csv` | Clean features + binary `risk_label` (1=high-risk) |
| `cv_results.json` | Per-fold AUC, mean AUC, 95% CI, p-value |
| `shap_summary.png` | SHAP beeswarm plot showing feature contributions |
| `shap_importance.png` | Mean |SHAP| bar chart |
| `final_model.json` | Trained XGBoost model (full dataset) |
| `training_report.txt` | Compact text summary of the training run |

---

## Documentation

Detailed per-script documentation is in `docs/`:

- [atlas_registration.md](docs/atlas_registration.md) — Atlas building, registration, feature extraction
- [preprocessing.md](docs/preprocessing.md) — Label assignment, imputation, filtering
- [training.md](docs/training.md) — Feature selection, XGBoost, CV, SHAP
