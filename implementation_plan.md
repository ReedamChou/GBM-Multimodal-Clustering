# GBM Survival Risk Stratification — Fresh Build Pipeline (Script 1 ✅ COMPLETE)

Rebuild the GBM project from scratch as specified in [New Project implimentaion Prompt.md](file:///c:/Users/Husain/Documents/AIml%20sideproject/GBM%20multi%20model/New%20Project%20implimentaion%20Prompt.md). The old repo had wrong features, no registration, 3-class labels, and cluttered structure. The new pipeline is **3 Python files only**, producing exactly **16 features** with binary risk labels, using **XGBoost + SHAP**.

## User Review Required

> [!IMPORTANT]
> **Lobe ratio denominator discrepancy.** The spec says `frontal_ed_ratio = ED voxels in frontal lobe / frontal lobe voxels` (denominator = entire lobe volume). But the old code and the existing `ucsf_with_clinical.csv` use `ED_in_lobe / total_tumor_in_lobe` (denominator = tumor voxels in that lobe). This produces values like 0.76, which makes sense. If the denominator were the entire lobe (~100k+ voxels), values would be ~0.001. **The existing CSV uses the old formula.** Which denominator should I use?
> - **Option A (old code / existing CSV):** `ED_in_lobe / total_tumor_in_lobe` — clinically: "of tumor in frontal lobe, what fraction is edema?"
> - **Option B (spec literal):** `ED_in_lobe / total_frontal_lobe_voxels` — clinically: "what fraction of frontal lobe is invaded by edema?"
>
> I will default to **Option A** (matching the existing CSV and old code) unless you say otherwise.

> [!IMPORTANT]
> **SRI24 atlas files are NOT in the repo.** The script needs `tzo116plus.nii.gz`, `suptent.nii.gz`, `tissues.nii.gz`, `SRI24-tzo116plus.txt`, and `sri24_t1.nii.gz` (T1 template for ANTs registration). You'll need to place these under `data/SRI24/`. I'll document the exact download instructions in the README.

> [!WARNING]
> **OS column is in days, not months.** The existing CSV has `OS` in days (e.g., `1303`). The spec's threshold is `OS ≤ 12 months`. Preprocessing will convert `OS_days / 30.4375` to months before thresholding, consistent with the old pipeline.

## Open Questions

> [!IMPORTANT]
> **ANTs registration on Windows.** `antspy` is notoriously difficult to install on Windows. Are you running this on Windows or do you have a Linux/WSL environment for the heavy NIfTI processing? If Windows-only, I can add a fallback affine-projection path (like the old code) with a config flag.

---

## Proposed Changes — Phase 1: Script 1 (`atlas_registration.py`)

This is the most complex script. I'll implement it first for your approval before moving to Scripts 2 and 3.

### Scaffolding (delete old, create new structure)

#### [DELETE] Old pipeline files

The following old files will be **deleted** to start fresh:

| File | Reason |
|------|--------|
| `scripts/step1_remove_clinical_features.py` | Replaced by new pipeline |
| `scripts/step2_assign_survival_clusters.py` | Replaced by new pipeline |
| `scripts/step3_preprocess_imaging_features.py` | Replaced by new pipeline |
| `scripts/step4_cluster_patient_correlations.py` | Replaced by new pipeline |
| `docs/step1_remove_clinical_features.md` | Replaced by new docs |
| `docs/step2_assign_survival_clusters.md` | Replaced by new docs |
| `docs/step3_preprocess_imaging_features.md` | Replaced by new docs |
| `docs/step4_cluster_patient_correlations.md` | Replaced by new docs |
| `docs/atlas_feature_nan_readme.md` | No longer relevant |
| `Readme.md` | Will be rewritten |
| `requirements.txt` | Will be rewritten |
| `classification_guide_ucsf.md` | Not in new spec |
| `training/` directory | Replaced by `src/train.py` |

> [!NOTE]
> The `old-atlasregistration-files/` directory and `data/` directory will be **preserved** (old reference code + raw data). The `UCSF/` directory with NIfTI data will also be preserved and referenced via `config.json`.

---

### New Project Structure

```
gbm_survival_stratification/          (root of the repo)
├── data/
│   ├── raw/ucsf_with_clinical.csv    (existing — has OS column)
│   ├── SRI24/                        (user must populate)
│   └── .gitkeep
├── src/
│   ├── atlas_registration.py         ← Script 1 (THIS PHASE)
│   ├── preprocessing.py              ← Script 2 (next phase)
│   └── train.py                      ← Script 3 (next phase)
├── docs/
│   ├── atlas_registration.md
│   ├── preprocessing.md
│   └── training.md
├── outputs/
│   ├── transforms/                   (ANTs transform cache)
│   └── .gitkeep
├── config.json
└── README.md
```

---

### Script 1 Architecture

#### [NEW] [config.json](file:///c:/Users/Husain/Documents/AIml%20sideproject/GBM%20multi%20model/config.json)

Central configuration with all paths and parameters. Key design decisions:
- Paths use the **actual UCSF layout** discovered on disk: `UCSF/DATA-IMAGE-STRUCTURAL/{ID}/{ID}_T1.nii.gz` and `UCSF/DATA-AUTOMATED-SEGMENT/{ID}_tumor_segmentation.nii.gz`
- T1 suffix is `_T1.nii.gz` (confirmed from `UCSF-PDGM-0004` folder)
- Segmentation suffix is `_tumor_segmentation.nii.gz` (confirmed from segment folder)
- OS column from CSV: `OS` (in days) + `patient_id` for join key

```json
{
  "data": {
    "ucsf_root": "UCSF",
    "structural_subdir": "DATA-IMAGE-STRUCTURAL",
    "segmentation_subdir": "DATA-AUTOMATED-SEGMENT",
    "t1_suffix": "_T1.nii.gz",
    "seg_suffix": "_tumor_segmentation.nii.gz",
    "clinical_csv": "data/raw/ucsf_with_clinical.csv",
    "sri24_dir": "data/SRI24"
  },
  "atlas": {
    "lobe_dilation_voxels": 3,
    "cache_atlas": true,
    "registration_type": "Affine",
    "transforms_cache_dir": "outputs/transforms",
    "use_ants_registration": true
  },
  "preprocessing": {
    "os_high_risk_threshold_months": 12,
    "os_days_per_month": 30.4375,
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

#### [NEW] [atlas_registration.py](file:///c:/Users/Husain/Documents/AIml%20sideproject/GBM%20multi%20model/src/atlas_registration.py)

**~350 lines.** Three major sections, closely following the old code's proven lobe-building logic but adding ANTs registration:

##### Section A — `build_4lobe_atlas()` (reuse old logic)

Ported directly from [extract_sri24_atlas_features.py](file:///c:/Users/Husain/Documents/AIml%20sideproject/GBM%20multi%20model/old-atlasregistration-files/atlas-based-feature-extracttion/extract_sri24_atlas_features.py):
- Parse `SRI24-tzo116plus.txt` → group label IDs by `LOBE_PREFIXES` dict
- Build seed volume (1/2/3/4 per lobe)
- Dilate supratentorial mask by `config.atlas.lobe_dilation_voxels`
- `distance_transform_edt` to fill gaps → 4-lobe atlas
- Save to `outputs/sri24_4lobe_atlas.nii.gz`, skip if cached

##### Section B — `register_atlas_to_patient()` (NEW — ANTs)

This is the critical new addition:
1. Load patient T1: `UCSF/DATA-IMAGE-STRUCTURAL/{ID}/{ID}_T1.nii.gz`
2. Load SRI24 T1 template: `data/SRI24/sri24_t1.nii.gz`
3. `ants.registration(fixed=patient_T1, moving=sri24_T1, type_of_transform='Affine')`
4. Apply transform to `sri24_4lobe_atlas.nii.gz` with `interpolator='nearestNeighbor'`
5. Apply transform to `tissues.nii.gz` (brainmask) with `interpolator='nearestNeighbor'`
6. Cache transforms in `outputs/transforms/{ID}/`

**Fallback path:** If `use_ants_registration: false` in config, fall back to the old affine-projection method (for Windows or when ANTs is unavailable).

##### Section C — `extract_features_for_patient()` (adapted from old code)

Per-patient feature extraction after atlas is in patient space:
- Load tumor segmentation (BraTS labels 1/2/4)
- Load registered 4-lobe atlas (labels 1/2/3/4)
- Load registered brainmask for TBI denominator
- Count voxels per label per lobe
- Compute all **16 features** (4 global + 12 lobe-wise)
- Compute QA column: `lobe_assignment_reliable`

**Key differences from old code:**
1. ❌ No `dominant_brain_lobe` column (forbidden by spec)
2. ❌ No extra diagnostic columns (voxel counts, path columns, etc.)
3. ✅ Only the 16 spec features + `patient_id` + `OS_months` + `lobe_assignment_reliable`
4. ✅ OS pulled from `ucsf_with_clinical.csv` and converted to months

##### Section D — `__main__` batch runner

- Discover patient folders from `UCSF/DATA-IMAGE-STRUCTURAL/`
- Match each to its segmentation in `UCSF/DATA-AUTOMATED-SEGMENT/`
- Join with clinical CSV to get `OS` (days → months)
- Run registration + extraction per patient
- Write `outputs/features_raw.csv`

**Output CSV columns (20 total):**

| Column | Type |
|--------|------|
| `patient_id` | str |
| `global_nc_en_ratio` | float |
| `global_ed_en_ratio` | float |
| `global_ed_total_ratio` | float |
| `tumor_burden_index` | float |
| `frontal_ed_ratio` | float |
| `frontal_en_ratio` | float |
| `frontal_nc_ratio` | float |
| `temporal_ed_ratio` | float |
| `temporal_en_ratio` | float |
| `temporal_nc_ratio` | float |
| `parietal_ed_ratio` | float |
| `parietal_en_ratio` | float |
| `parietal_nc_ratio` | float |
| `occipital_ed_ratio` | float |
| `occipital_en_ratio` | float |
| `occipital_nc_ratio` | float |
| `OS_months` | float |
| `lobe_assignment_reliable` | bool |

---

#### [NEW] [requirements.txt](file:///c:/Users/Husain/Documents/AIml%20sideproject/GBM%20multi%20model/requirements.txt)

```
antspy
nibabel
scipy
numpy
pandas
scikit-learn
xgboost
shap
matplotlib
```

---

#### [NEW] [docs/atlas_registration.md](file:///c:/Users/Husain/Documents/AIml%20sideproject/GBM%20multi%20model/docs/atlas_registration.md)

Documentation covering:
- What the script does (lobe atlas construction + ANTs registration + feature extraction)
- Required SRI24 input files and download instructions
- Registration direction rationale (SRI24 → patient T1)
- 4-lobe atlas building algorithm (seed → distance transform)
- Output format of `features_raw.csv`
- QA column meaning
- Fallback affine-projection mode

---

## Phases 2 & 3 (After Your Approval)

| Phase | Script | Key Deliverables |
|-------|--------|-----------------|
| 2 | `src/preprocessing.py` | Drop unreliable rows, binary labels, median imputation → `features_processed.csv` |
| 3 | `src/train.py` | Tree-based feature selection, XGBoost 10-fold CV, SHAP plots, `cv_results.json` |
| Final | `README.md` + remaining docs | Full project documentation |

---

## Verification Plan

### Script 1 Verification

1. **Syntax check:** `python -c "import src.atlas_registration"` — no import errors
2. **Config validation:** Script loads `config.json` and validates all paths exist
3. **Dry-run mode:** With `--dry-run` flag, list discovered patients and their T1/seg paths without running registration
4. **Single-patient test:** Run on one patient (e.g., `UCSF-PDGM-0004`) and verify:
   - Output CSV has exactly 19 columns (16 features + patient_id + OS_months + lobe_assignment_reliable)
   - Feature values are in reasonable ranges (0–1 for ratios, small float for TBI)
   - No `dominant_brain_lobe` column exists
5. **Compare with existing CSV:** Cross-check the 16 feature values against `ucsf_with_clinical.csv` for the same patient (they should be close but may differ due to the registration step)

> [!NOTE]
> Full batch processing requires SRI24 atlas files and takes significant time (~1-3 min per patient with ANTs). The code will be verified for correctness structurally; full batch run is at your discretion.
