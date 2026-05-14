
# GBM Survival Risk Stratification — Full Build Pipeline for AI

> **Conference submission project.** Code must be clean, reproducible, and minimal.  
> Do NOT add anything that is not grounded in the abstract. Strip the old repo entirely and start fresh.

---

## 0. Context & Ground Truth

**Paper title:** Anatomically Informed MRI Radiomics for Interpretable Survival Risk Stratification in Glioblastoma Multiforme (GBM)

**Core claim of the abstract:**

- Integrating global volumetric ratios with lobe-level radiomic features improves survival risk stratification.
- Binary label: high-risk (OS ≤ 12 months) vs low-risk (OS > 12 months).
- Exactly **16 features** are extracted. No more, no less.
- **Only T1-weighted MRI** is used for atlas registration. T1Gd, FLAIR, T2 are NOT used at any step.
- Machine learning: **XGBoost + tree-based feature selection + stratified 10-fold CV + SHAP**.
- Metric reported: mean AUC = 0.652 (95% CI: 0.615–0.688).

---

## 1. Delete the Old Repo

Before writing a single line of code, wipe all old scripts. The old pipeline had:

- Wrong/missing features (temporal_en_ratio, parietal_en_ratio, occipital_en_ratio missing).
- Extra features not in the abstract (4 `dominant_brain_lobe_*` one-hot columns).
- No actual T1 registration step (it assumed UCSF was already aligned — it is NOT).
- A 3-class ordinal model (the abstract is binary: high vs low risk only).
- Cluttered structure with 10+ scripts.

**Delete everything. Start with an empty folder.**

---

## 2. Final Project Structure

Create exactly this layout. Nothing extra.

```
gbm_survival_stratification/
│
├── data/                         # Raw UCSF-PDGM data lives here (NOT committed to git)
│   └── .gitkeep
│
├── src/
│   ├── atlas_registration.py     # Step 1+2: T1 registration + lobe atlas feature extraction
│   ├── preprocessing.py          # Step 3: Binary labels, imputation, scaling
│   └── train.py                  # Step 4: XGBoost, feature selection, 10-fold CV, SHAP
│
├── docs/
│   ├── atlas_registration.md     # Documents atlas_registration.py
│   ├── preprocessing.md          # Documents preprocessing.py
│   └── training.md               # Documents train.py
│
├── outputs/                      # CSVs, model artifacts, plots (gitignored except .gitkeep)
│   └── .gitkeep
│
├── config.json                   # All tunable parameters in one place
└── README.md                     # Project overview, workflow, setup instructions
```

**Total Python files: 3. That's it.**

---

## 3. The Exact 16 Features (Do Not Deviate)

These are the only features allowed in the model. Derived from the abstract's Materials & Methods.

### Global features (4)

|Feature name|Formula|
|---|---|
|`global_nc_en_ratio`|NC voxels / ET voxels|
|`global_ed_en_ratio`|ED voxels / ET voxels|
|`global_ed_total_ratio`|ED voxels / WT voxels (WT = NC+ET+ED)|
|`tumor_burden_index`|WT voxels / total brain voxels|

### Lobe-wise features (12 = 4 lobes × 3 subregions)

|Feature name|Formula|
|---|---|
|`frontal_ed_ratio`|ED voxels in frontal lobe / frontal lobe voxels|
|`frontal_en_ratio`|ET voxels in frontal lobe / frontal lobe voxels|
|`frontal_nc_ratio`|NC voxels in frontal lobe / frontal lobe voxels|
|`temporal_ed_ratio`|ED voxels in temporal lobe / temporal lobe voxels|
|`temporal_en_ratio`|ET voxels in temporal lobe / temporal lobe voxels|
|`temporal_nc_ratio`|NC voxels in temporal lobe / temporal lobe voxels|
|`parietal_ed_ratio`|ED voxels in parietal lobe / parietal lobe voxels|
|`parietal_en_ratio`|ET voxels in parietal lobe / parietal lobe voxels|
|`parietal_nc_ratio`|NC voxels in parietal lobe / parietal lobe voxels|
|`occipital_ed_ratio`|ED voxels in occipital lobe / occipital lobe voxels|
|`occipital_en_ratio`|ET voxels in occipital lobe / occipital lobe voxels|
|`occipital_nc_ratio`|NC voxels in occipital lobe / occipital lobe voxels|

**Strictly forbidden features:** `dominant_brain_lobe_*` (one-hot encoded dominant lobe) — these are NOT in the abstract and must NOT appear.

**BraTS label map for UCSF-PDGM segmentation files:**

- Label 1 = NC (necrotic core)
- Label 2 = ED (peritumoral edema)
- Label 4 = ET (enhancing tumor) — note: it's 4, not 3

---

## 4. Script 1: `src/atlas_registration.py`

This is the most critical script. It has two major responsibilities:

### 4a. Build the 4-lobe SRI24 atlas (run once, cache the result)

**Input files needed (SRI24 atlas):**

- `tzo116plus.nii.gz` — label volume in SRI24 space
- `suptent.nii.gz` — supratentorial mask
- `tissues.nii.gz` — brainmask (used for tumor burden index denominator)
- `SRI24-tzo116plus.txt` — text file mapping label IDs to names

**Logic:**

1. Parse the label name file; group label IDs into 4 lobes by string prefix:
    - prefix `frontal` → lobe 1
    - prefix `temporal` → lobe 2
    - prefix `parietal` → lobe 3
    - prefix `occipital` → lobe 4
2. Create a seed volume (same grid as `tzo116plus.nii.gz`):
    - Assign voxel value 1/2/3/4 where the atlas label belongs to a lobe.
    - Voxels not matching any lobe prefix get value 0.
3. Dilate the supratentorial mask by a small amount (3 voxels, configurable in `config.json`).
4. Run `scipy.ndimage.distance_transform_edt` on the binary seed volume to grow each lobe across the supratentorial brain using nearest-seed assignment.
5. Save the resulting 4-lobe atlas volume as `outputs/sri24_4lobe_atlas.nii.gz` — **cache it**, never rebuild per patient.

This lobe-building approach is the same efficient method as the old code. Keep it.

### 4b. Register SRI24 atlas to each patient's T1 scan (NEW — this was missing in the old code)

**Why this is needed:** UCSF-PDGM tumor segmentations are in each patient's native T1 space, NOT pre-aligned to the SRI24 template. The old code wrongly assumed alignment. This step fixes that.

**Registration approach — use ANTs (antspy):**

The registration direction is: **SRI24 T1 template → patient T1 scan**

```
Fixed image  : patient T1 (e.g., sub-XXX_T1.nii.gz)
Moving image : SRI24 T1 template (sri24_t1.nii.gz)
Transform    : Rigid + Affine (SyN deformable if time permits, but Affine is acceptable)
```

After registration, apply the **same transform** to the `sri24_4lobe_atlas.nii.gz` using nearest-neighbor interpolation (label volume — never trilinear).

**Result:** The 4-lobe atlas is now warped into patient T1 space, matching the tumor segmentation.

**Important implementation notes:**

- Use `ants.registration(fixed=patient_t1, moving=sri24_t1, type_of_transform='Affine')` for speed.
- Apply transform to atlas: `ants.apply_transforms(fixed=patient_t1, moving=atlas, transformlist=..., interpolator='nearestNeighbor')`.
- The patient T1 filename pattern for UCSF-PDGM: `sub-{ID}/anat/sub-{ID}_T1w.nii.gz` — confirm and set this in `config.json`.
- The segmentation filename pattern: `sub-{ID}/anat/sub-{ID}_tumor_segmentation.nii.gz` — confirm pattern.
- Cache registration transforms per patient in `outputs/transforms/` to avoid re-running.

**ONLY T1 is used for registration.** Do not load T1Gd, FLAIR, or T2 at any point.

### 4c. Feature extraction (per patient)

Once the 4-lobe atlas is registered into patient space:

1. Load the tumor segmentation (BraTS labels 1/2/4).
2. Load the registered 4-lobe atlas (labels 1/2/3/4 = frontal/temporal/parietal/occipital).
3. Load the registered brainmask (from `tissues.nii.gz` transformed to patient space) for TBI denominator.
4. Count voxels per label per lobe:
    
    ```
    for lobe in [1,2,3,4]:    lobe_mask = (atlas == lobe)    nc_in_lobe = np.sum((seg == 1) & lobe_mask)    et_in_lobe = np.sum((seg == 4) & lobe_mask)    ed_in_lobe = np.sum((seg == 2) & lobe_mask)    lobe_total = np.sum(lobe_mask)
    ```
    
5. Compute all 16 features using safe division (divide-by-zero → 0.0).
6. Add QA columns (not used in model):
    - `lobe_assignment_reliable`: True if ≥90% of tumor voxels have a non-zero lobe label.

### 4d. Batch runner (at the bottom of the same file)

```python
if __name__ == "__main__":
    # Iterate UCSF patient folders
    # Call registration + extraction per patient
    # Write one row per patient to outputs/features_raw.csv
```

**Output CSV columns:** `patient_id` + the 16 feature columns + `OS_months` + `lobe_assignment_reliable`.

---

## 5. Script 2: `src/preprocessing.py`

This script takes `outputs/features_raw.csv` and prepares it for training.

**Steps:**

1. Drop rows where `lobe_assignment_reliable == False`.
2. Drop rows where `OS_months` is NaN.
3. **Binary label assignment:**
    
    ```python
    df['risk_label'] = (df['OS_months'] <= 12).astype(int)# 1 = high-risk, 0 = low-risk
    ```
    
    This exactly matches the abstract: "high-risk (OS ≤ 12 months) versus low-risk (OS > 12 months)".
4. Impute missing feature values with column median.
5. No scaling needed for tree-based models (XGBoost is scale-invariant). But add StandardScaler as optional for SHAP value comparison if needed — keep it off by default.
6. Save to `outputs/features_processed.csv`.

**Output CSV columns:** 16 feature columns + `risk_label`.

---

## 6. Script 3: `src/train.py`

This is the ML pipeline. It strictly follows the abstract's Methods section.

### 6a. Feature selection (tree-based)

> "A machine learning pipeline combining tree-based feature selection with an XGBoost classifier"

1. Load `outputs/features_processed.csv`.
2. Split X (16 features) and y (`risk_label`).
3. Fit a `RandomForestClassifier` or `ExtraTreesClassifier` to get feature importances.
4. Use `SelectFromModel` (sklearn) to keep features above the mean importance threshold.
5. Record which features were selected — log this to `outputs/selected_features.txt`.

### 6b. XGBoost classifier

Parameters (start with these, adjust via `config.json`):

```json
{
  "xgb_n_estimators": 200,
  "xgb_max_depth": 4,
  "xgb_learning_rate": 0.05,
  "xgb_subsample": 0.8,
  "xgb_colsample_bytree": 0.8,
  "xgb_scale_pos_weight": "auto",
  "tree_selector_threshold": "mean"
}
```

Set `scale_pos_weight` automatically from class imbalance: `n_negatives / n_positives`.

### 6c. Stratified 10-fold cross-validation

> "evaluated under stratified 10-fold cross-validation scheme"

```python
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
aucs = []
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    # fit pipeline (selector + XGBoost) on train
    # predict_proba on val
    # compute AUC
    aucs.append(roc_auc_score(y_val, y_prob))

mean_auc = np.mean(aucs)
ci_low = np.percentile(aucs, 2.5)
ci_high = np.percentile(aucs, 97.5)
```

**Expected target:** mean AUC ≈ 0.652 (95% CI: 0.615–0.688).

**Statistical test:** one-sided t-test or permutation test vs. AUC = 0.5 (chance). Report p-value (abstract reports p = 5.58 × 10⁻⁶).

### 6d. SHAP analysis

> "Model interpretability was assessed using SHAP"

```python
import shap

# After final full-data fit:
explainer = shap.TreeExplainer(final_xgb_model)
shap_values = explainer.shap_values(X)

# Summary plot — save to outputs/shap_summary.png
shap.summary_plot(shap_values, X, feature_names=feature_cols, show=False)
plt.savefig("outputs/shap_summary.png", dpi=150, bbox_inches='tight')

# Bar plot of mean |SHAP|
shap.summary_plot(shap_values, X, plot_type='bar', show=False)
plt.savefig("outputs/shap_importance.png", dpi=150, bbox_inches='tight')
```

Expected top SHAP features (from abstract Results): `frontal_en_ratio`, `tumor_burden_index`, `frontal_ed_ratio`.

### 6e. Outputs from train.py

All saved to `outputs/`:

- `cv_results.json` — fold-wise AUC, mean, CI, p-value
- `shap_summary.png`
- `shap_importance.png`
- `final_model.json` — saved XGBoost model
- `training_report.txt` — compact text summary of the run

---

## 7. `config.json`

Single config file for all tunable parameters:

```json
{
  "data": {
    "ucsf_root": "data/UCSF-PDGM",
    "t1_suffix": "_T1w.nii.gz",
    "seg_suffix": "_tumor_segmentation.nii.gz",
    "sri24_dir": "data/SRI24"
  },
  "atlas": {
    "lobe_dilation_voxels": 3,
    "cache_atlas": true,
    "registration_type": "Affine",
    "transforms_cache_dir": "outputs/transforms"
  },
  "preprocessing": {
    "os_high_risk_threshold_months": 12,
    "min_lobe_mapping_fraction": 0.90
  },
  "training": {
    "cv_folds": 10,
    "random_seed": 42,
    "tree_selector_threshold": "mean",
    "xgb_n_estimators": 200,
    "xgb_max_depth": 4,
    "xgb_learning_rate": 0.05,
    "xgb_subsample": 0.8,
    "xgb_colsample_bytree": 0.8
  }
}
```

---

## 8. `README.md` Content Requirements

The README must contain:

1. **Title and affiliation** — match the paper header exactly.
2. **Abstract excerpt** — copy the abstract's Purpose paragraph.
3. **Setup instructions:**
    
    ```bash
    pip install antspy nibabel scipy numpy pandas scikit-learn xgboost shap matplotlib
    ```
    
4. **Data placement:** describe where to put UCSF-PDGM and SRI24 files under `data/`.
5. **Run order:**
    
    ```bash
    python src/atlas_registration.py   # generates outputs/features_raw.csvpython src/preprocessing.py        # generates outputs/features_processed.csvpython src/train.py                # generates all model outputs
    ```
    
6. **Conference notice:** "This repository is associated with a paper under review. Please do not redistribute."
7. **Output description:** what each file in `outputs/` means.

---

## 9. Docs Folder Requirements

Three markdown files, one per script:

### `docs/atlas_registration.md`

- What the script does
- Required input files (SRI24 files + UCSF structure)
- Registration direction and why
- How the 4-lobe atlas is built (seed → distance transform)
- Output format of `features_raw.csv`
- QA column meaning

### `docs/preprocessing.md`

- Binary label formula (OS ≤ 12 → high-risk)
- Imputation strategy
- Why no scaling for XGBoost
- Output format of `features_processed.csv`

### `docs/training.md`

- Tree-based selection logic
- XGBoost config rationale
- 10-fold CV procedure
- SHAP interpretation guidance
- How to read `cv_results.json`

---

## 10. Critical Rules — Do Not Break These

|Rule|Reason|
|---|---|
|Only T1 MRI used for registration|Abstract explicitly states this|
|Exactly 16 features, no more|Abstract says "sixteen radiomic descriptors"|
|No `dominant_brain_lobe_*` features|Not in abstract|
|Binary label only (OS ≤ 12 vs > 12)|Abstract is binary, old code was 3-class|
|No clinical features in model|Imaging-only, as per abstract|
|`OS` column excluded from feature matrix|Leakage prevention|
|Registration: SRI24 T1 template → patient T1|UCSF is in native space, not template space|
|Nearest-neighbor interpolation for atlas|It's a label volume — never use trilinear|
|ET label in BraTS is 4, not 3|UCSF-PDGM uses original BraTS convention|
|3 Python files only|Keep it minimal for reproducibility|

---

## 11. Dependency Summary

```txt
antspy           # ANTs registration (pip install antspy)
nibabel          # NIfTI I/O
scipy            # distance_transform_edt
numpy
pandas
scikit-learn     # StratifiedKFold, SelectFromModel, RandomForestClassifier
xgboost
shap
matplotlib
```

No deep learning. No PyTorch. No TensorFlow. The abstract does not mention any of these.

---

## 12. File Naming Convention

|Output file|Description|
|---|---|
|`outputs/sri24_4lobe_atlas.nii.gz`|Cached lobe atlas in SRI24 space|
|`outputs/transforms/sub-XXX_*.mat`|Per-patient ANTs transform files|
|`outputs/features_raw.csv`|Raw features before preprocessing|
|`outputs/features_processed.csv`|Clean features + binary risk_label|
|`outputs/selected_features.txt`|Which of the 16 features were selected|
|`outputs/cv_results.json`|Fold AUCs, mean, CI, p-value|
|`outputs/shap_summary.png`|SHAP beeswarm plot|
|`outputs/shap_importance.png`|SHAP bar chart|
|`outputs/final_model.json`|Saved XGBoost model|
|`outputs/training_report.txt`|Human-readable training summary|

---

_This document is the single source of truth for building the project. Do not add steps, features, or scripts beyond what is specified here. The goal is a clean, minimal, reproducible codebase that faithfully implements the abstract._