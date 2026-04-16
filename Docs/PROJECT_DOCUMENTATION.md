# GBM Multi-Modal Clustering Pipeline — Project Documentation

> **Hypothesis:** Multi-modal clustering combining MRI sub-region ratios, brain lobe location, and clinical variables identifies a high-risk GBM subtype characterised by temporal lobe dominance, high necrotic core-to-enhancing tumour (NC/EN) ratio, and short overall survival.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Research Hypothesis & Goals](#2-research-hypothesis--goals)
3. [Repository Layout](#3-repository-layout)
4. [Datasets & Input Schema](#4-datasets--input-schema)
5. [Pipeline Architecture](#5-pipeline-architecture)
6. [Step-by-Step Reference](#6-step-by-step-reference)
   - [Steps 1–2: Source Data & Imaging Features (External)](#steps-12-source-data--imaging-features-external)
   - [Step 3: Build Master Feature Table](#step-3-build-master-feature-table)
   - [Step 4: Train/Test Split & Leakage-Safe Preprocessing](#step-4-traintest-split--leakage-safe-preprocessing)
   - [Step 5: Spectral Clustering & Multi-Criteria k Selection](#step-5-spectral-clustering--multi-criteria-k-selection)
   - [Step 6: Train-Set Cluster Characterisation](#step-6-train-set-cluster-characterisation)
   - [Step 7: Held-Out Test-Set Validation](#step-7-held-out-test-set-validation)
   - [Step 8: Repeated Validation Across Random Splits](#step-8-repeated-validation-across-random-splits)
   - [Step 9: Bootstrap Cluster Stability](#step-9-bootstrap-cluster-stability)
   - [Step 11: Cohort Shift Analysis (UCSF vs UPenn)](#step-11-cohort-shift-analysis-ucsf-vs-upenn)
7. [Support Modules](#7-support-modules)
8. [Key Design Decisions](#8-key-design-decisions)
9. [Output Inventory](#9-output-inventory)
10. [Environment & Setup](#10-environment--setup)
11. [Full Run — Quick Start](#11-full-run--quick-start)
12. [Troubleshooting](#12-troubleshooting)
13. [Appendix A: Feature Glossary](#appendix-a-feature-glossary)
14. [Appendix B: Statistical Methods Reference](#appendix-b-statistical-methods-reference)
15. [Appendix C: Data Preprocessing Theory](#appendix-c-data-preprocessing-theory)

---

## 1. Project Overview

This project implements an **end-to-end unsupervised clustering pipeline** for **Glioblastoma Multiforme (GBM)** patients using multi-modal data. The pipeline integrates three categories of data:

| Data Modality | Description |
|---|---|
| **MRI-derived imaging** | Per-lobe ED/EN/NC sub-region ratios, global ratios, and tumour burden index — extracted from NIfTI segmentation masks registered to the SRI24 brain atlas |
| **Brain anatomical location** | Dominant lobe of tumour involvement (frontal, temporal, parietal, or occipital) |
| **Clinical variables** | Age, sex, MGMT promoter methylation status, IDH mutation status, and overall survival (OS) |

The pipeline discovers patient subtypes via **spectral clustering** on a training cohort, characterises each subtype statistically, identifies a **high-risk cluster**, and then validates the findings on a held-out test set. Two additional robustness layers — repeated random-split validation and bootstrap cluster stability analysis — guard against overfitting and ensure reproducibility.

Two independent institutional cohorts are processed in parallel:

| Dataset | Source | Patients |
|---------|--------|----------|
| **UCSF-PDGM** | University of California, San Francisco | ~500 |
| **UPenn-GBM** | University of Pennsylvania | ~600 |

---

## 2. Research Hypothesis & Goals

### Core Hypothesis

> Specific imaging-based features — temporal lobe dominance and high necrotic core-to-enhancing tumour (NC/EN) ratio — identify a high-risk GBM subtype with significantly shorter overall survival compared to other subtypes.

### Primary Goals

1. Discover reproducible GBM subtypes through multi-modal unsupervised clustering.
2. Identify and label the high-risk subtype with a descriptive name (e.g., *"temporally-dominant high-necrosis subtype"*).
3. Validate that the subtype separation and survival differences replicate on a held-out test set.
4. Demonstrate robustness through repeated random-split validation and bootstrap stability analysis.
5. Quantify cohort shift between UCSF and UPenn to contextualise cross-cohort findings.
6. Produce publication-ready outputs: feature heatmap, Kaplan-Meier curves, lobe distribution charts, silhouette elbow plot, and cluster summary tables.

---

## 3. Repository Layout

```
GBM multi model/
│
├── Clinical+Atlas Merged Data/            # Input CSVs (atlas features + clinical data)
│   ├── UCSF_sri24_atlas_features_merged.csv
│   └── UPenn_sri24_atlas_features_merged.csv
│
├── scripts/                               # Python pipeline scripts & support modules
│   ├── step3_build_master_feature_table.py
│   ├── step4_split_train_test.py
│   ├── step5_spectral_clustering.py
│   ├── step6_characterize_clusters.py
│   ├── step7_validate_test_set.py
│   ├── step8_repeated_validation.py
│   ├── step9_cluster_stability.py
│   ├── step10_nested_validation.py        # Retired — empty file, kept for history
│   ├── step11_cohort_shift_analysis.py
│   ├── pipeline_preprocessing.py          # Shared preprocessing functions
│   ├── cluster_validation_utils.py        # Shared cluster metric & validation utilities
│   ├── repeated_evaluation_runner.py      # Shared orchestration for repeated runs
│   ├── run_pipeline_iteration.py          # Single-iteration pipeline runner
│   ├── iteration_paths.py                 # Iteration directory management
│   ├── build_project_report_data.py       # Report data computation
│   └── render_project_reports.py          # Report rendering / formatting
│
├── outputs/                               # Main pipeline outputs (auto-generated)
│   ├── step3/{ucsf,upenn}/                #   Master tables, metadata JSONs
│   ├── step4/{ucsf,upenn}/                #   Train/test splits, scalers, logs
│   ├── step5/{ucsf,upenn}/                #   Cluster labels, centroids, k-evaluation
│   ├── step6/{ucsf,upenn}/                #   Summaries, stat tests, KM plots
│   └── step7/{ucsf,upenn}/                #   Validation tables, KM plots
│
├── results/                               # Robustness & iteration outputs
│   ├── iteration-N/                       #   Per-iteration pipeline snapshots
│   └── repeated-validation/               #   Step 8 repeated split metrics
│
├── Docs/                                  # Documentation
│   ├── PROJECT_DOCUMENTATION.md           #   ← This file
│   ├── README_step3.md ... README_step9.md
│
├── Pipline.txt                            # Original pipeline specification
├── pipeline2.txt                          # Current pipeline specification
├── Readme.md                              # Project README
├── Data_Preprocessing_and_Feature_Engineering (1).md  # Theory reference
├── UCSFvUPennDIff.txt                     # Cohort difference notes
├── ppt.tex                                # Presentation TeX source
├── professor_results_summary.tex          # Results summary TeX
├── summarised-results.txt                 # Condensed results
├── presentation-results.txt               # Presentation results
└── .venv/                                 # Virtual environment (not tracked)
```

---

## 4. Datasets & Input Schema

### 4.1 Input Data

The pipeline consumes **pre-merged CSV files** located in `Clinical+Atlas Merged Data/`. These combine:
- **SRI24 atlas-mapped imaging features** extracted from NIfTI segmentation masks (Steps 1–2, performed externally).
- **Clinical metadata** from the UCSF-PDGM and UPenn-GBM cohorts.

Each row represents one patient.

### 4.2 Key Columns

| Column Group | Example Columns | Description |
|---|---|---|
| Patient ID | `ID`, `PatientID` | Unique patient identifier |
| Dominant Lobe | `dominant_lobe`, `Dominant_Lobe` | Most affected brain lobe |
| Per-Lobe Ratios | `frontal_ed_en_ratio`, `temporal_nc_en_ratio`, … | 12 ratio columns (ED/EN/NC × 4 lobes) |
| Global Ratios | `global_nc_en_ratio`, `global_ed_en_ratio`, `global_ed_total_ratio` | Whole-tumour sub-region ratios |
| Tumour Burden | `tumor_burden_index` | Normalised total tumour volume |
| Clinical | `Age at MRI`, `Sex`, `MGMT status`, `IDH` | Demographic and molecular markers |
| Survival (outcome) | `OS`, `1-dead 0-alive`, `Survival_Status` | Overall survival and censoring |

### 4.3 Column Alias Resolution

The two datasets use different naming conventions. The pipeline resolves this automatically via an alias dictionary defined in `scripts/pipeline_preprocessing.py`:

| Canonical Key | Accepted Aliases |
|---|---|
| `id` | `ID`, `PatientID`, `patient_id` |
| `sex` | `Sex`, `Gender` |
| `mgmt` | `MGMT status`, `MGMT`, `mgmt_status` |
| `idh` | `IDH`, `IDH1` |
| `os` | `OS`, `Survival_from_surgery_days_UPDATED` |
| `censor` | `1-dead 0-alive`, `Survival_Censor` |
| `survival_status` | `Survival_Status` |
| `dominant_lobe` | `Dominant_Lobe`, `dominant brain lobe` |

---

## 5. Pipeline Architecture

The pipeline is organised into three tiers:

| Tier | Steps | Role |
|---|---|---|
| **Main workflow** | 3 → 4 → 5 → 6 → 7 | Core leakage-safe clustering, characterisation, and held-out validation |
| **Robustness** | 8, 9 | Repeated-split stability and bootstrap cluster stability |
| **Diagnostics** | 11 | Cross-cohort shift analysis |

### Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                     EXTERNAL (Steps 1–2)                            │
│  NIfTI Images → Atlas Registration → Feature Extraction → CSV      │
└───────────────────────────┬─────────────────────────────────────────┘
                            │  Merged CSV per cohort
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Step 3: Build Master Feature Table                                 │
│  • Replace missing markers  • Coerce numeric columns                │
│  • Resolve column aliases   • No imputation, encoding, or scaling   │
└───────────────────────────┬─────────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Step 4: Stratified Train/Test Split + Frozen Preprocessing         │
│  • 70/30 split stratified by MGMT + OS                              │
│  • Fit imputers, encoders, scaler on TRAIN ONLY                     │
│  • Apply same transforms to test set                                │
│  • Write explicit train/test membership artifact                    │
└──────────┬──────────────────────────────────┬───────────────────────┘
           │ Train features                   │ Test features
           ▼                                  │ (sealed until Step 7)
┌───────────────────────────────┐             │
│  Step 5: Spectral Clustering  │             │
│  • Inner train/val split      │             │
│  • Evaluate k = 2–6           │             │
│  • Multi-criteria k selection │             │
│  • Refit best k on full train │             │
│  • Save labels + centroids    │             │
└──────────┬────────────────────┘             │
           │ Cluster labels + centroids       │
           ▼                                  │
┌───────────────────────────────┐             │
│  Step 6: Characterise Clusters│             │
│  • Summary table per cluster  │             │
│  • KM curves + log-rank test  │             │
│  • Kruskal-Wallis tests       │             │
│  • Chi-square tests           │             │
│  • BH correction              │             │
│  • Identify high-risk cluster │             │
└──────────┬────────────────────┘             │
           │ High-risk cluster identity       │
           ▼                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Step 7: Validate on Held-Out Test Set                              │
│  • Assign test patients to nearest train centroid                   │
│  • Compare cluster proportions (train vs test)                      │
│  • KM curves + log-rank test on test set                            │
│  • Check high-risk cluster replication                              │
└─────────────────────────────────────────────────────────────────────┘
                            │
          ┌─────────────────┼─────────────────┐
          ▼                 ▼                 ▼
    ┌───────────┐    ┌───────────┐    ┌───────────┐
    │  Step 8   │    │  Step 9   │    │  Step 11  │
    │  Repeated │    │  Bootstrap│    │  Cohort   │
    │  Split    │    │  Cluster  │    │  Shift    │
    │  Valid.   │    │  Stability│    │  Analysis │
    └───────────┘    └───────────┘    └───────────┘
```

### Anti-Leakage Architecture

The pipeline enforces strict **data leakage prevention** throughout:

| Principle | Implementation |
|---|---|
| No full-cohort fitting | Step 3 performs raw cleaning only; all learned transforms are deferred to Step 4 |
| Train-only fitting | Step 4 fits imputers, encoders, and scaler exclusively on the training set |
| Outcome exclusion | OS, censoring, and survival status are never included in clustering features |
| Test set sealing | The test set is not opened until Step 7; it receives train-derived transforms only |
| Nested k selection | Step 5 selects k inside an inner train/validation split; the outer test split remains untouched |
| Explicit membership | Step 4 writes a split membership CSV so that downstream steps never reconstruct the split |

---

## 6. Step-by-Step Reference

---

### Steps 1–2: Source Data & Imaging Features (External)

These steps are **not implemented as scripts** in this repository. They are assumed to have been completed externally.

**Step 1 — Source data:**
- Load the raw dataset CSVs in Python.
- Audit columns for missing values and data quality issues.
- Process NIfTI segmentation images as needed.

**Step 2 — Imaging feature extraction:**
- Per patient, compute:
  - Dominant brain lobe (frontal, temporal, parietal, or occipital)
  - Per-lobe ED/EN/NC ratios (12 columns across 4 lobes)
  - Global NC/EN ratio, global ED/EN ratio, global ED total ratio
  - Tumour Burden Index (TBI)
- Output one-row-per-patient CSV.

**Output:** `Clinical+Atlas Merged Data/{UCSF,UPenn}_sri24_atlas_features_merged.csv`

---

### Step 3: Build Master Feature Table

**Script:** `scripts/step3_build_master_feature_table.py`

**Purpose:** Clean raw merged CSVs without any learned transformations — a *"raw cleaning only"* step that preserves leakage safety.

#### What It Does

1. **Replace missing markers** — Converts text placeholders (`"unknown"`, `"indeterminate"`, `"not available"`, `"NA"`, `"none"`, `"null"`, etc.) to proper `NaN` values.
2. **Coerce numeric columns** — Detects string columns that are ≥80% parseable as numbers and converts them to numeric dtype.
3. **Resolve column aliases** — Maps variant column names (UCSF vs UPenn schemas) to canonical keys using the shared alias dictionary in `pipeline_preprocessing.py`.
4. **Identify outcome columns** — Marks OS, censoring, and survival status columns for exclusion from clustering features downstream.
5. **Remove obsolete artifacts** — Deletes any leftover scaler/feature files from prior versions that mistakenly fit transforms.

#### What Step 3 Does NOT Do

- No imputation
- No encoding
- No scaling
- No train/test splitting

#### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--ucsf-merged` | Auto-detected | Path to UCSF merged CSV |
| `--upenn-merged` | Auto-detected | Path to UPenn merged CSV |
| `--output-dir` | `outputs/step3` | Output directory |

#### Usage

```powershell
python scripts/step3_build_master_feature_table.py --output-dir outputs/step3
```

#### Outputs per Dataset

| File | Description |
|---|---|
| `{dataset}_master_table_step3.csv` | Raw-cleaned master table |
| `{dataset}_step3_metadata.json` | Resolved columns, predictor lists, row/col counts |

---

### Step 4: Train/Test Split & Leakage-Safe Preprocessing

**Script:** `scripts/step4_split_train_test.py`

**Purpose:** Split the master table into train (70%) and test (30%) sets, then fit **all** learned preprocessing exclusively on the training set. This is the **core anti-leakage step**.

#### What It Does

1. **Stratified splitting** — Uses a cascading strategy that tries progressively simpler stratification:
   - MGMT proportion + OS quantile bins (preferred)
   - MGMT only
   - OS only
   - Random split (fallback)

   For each strategy, it tries bin counts from 10 down to 2, selecting the finest binning where every stratum has ≥2 samples.

2. **Binary encoding** (train-derived rules):
   - `sex` → `sex_bin` (male=1, female=0)
   - `MGMT status` → `mgmt_bin` (methylated/positive=1, unmethylated/negative=0)
   - `IDH` → `idh_bin` (mutant=1, wild-type=0)
   - For each: if train missing rate > 10%, an `_unknown` indicator column is added.

3. **Dominant lobe encoding:**
   - Canonicalise to `{frontal, temporal, parietal, occipital}`
   - One-hot encode into 4 binary columns with explicit, deterministic column order
   - If train missing rate > 10%, add a `dominant_lobe_unknown` column

4. **Continuous imputation** — Fill missing values with train-set medians.

5. **Discrete numeric imputation** — Fill with train-set mode.

6. **Categorical string fill** — Fill remaining string columns with train-set mode (or `"unknown"` if > 10% missing).

7. **Feature pruning** — Drop columns that are all-missing or have zero variance in the training set.

8. **Z-score standardisation** — Fit `StandardScaler` on train continuous features, then transform both train and test.

9. **Write split membership** — Produce an explicit CSV recording which patient IDs belong to train vs test, so downstream steps never need to reconstruct the split.

#### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--datasets` | `ucsf upenn` | Which datasets to process |
| `--step3-dir` | `outputs/step3` | Step 3 output directory |
| `--output-dir` | `outputs/step4` | Output directory |
| `--test-size` | `0.30` | Test set fraction |
| `--random-state` | `42` | Random seed |
| `--log-level` | `INFO` | Logging verbosity |

#### Usage

```powershell
python scripts/step4_split_train_test.py --output-dir outputs/step4 --log-level INFO
```

#### Outputs per Dataset

| File | Description |
|---|---|
| `{dataset}_train_master_table_step4.csv` | Full train table with all columns |
| `{dataset}_test_master_table_step4.csv` | Full test table with all columns |
| `{dataset}_train_clustering_features_step4.csv` | Train features (ID + numeric predictors only) |
| `{dataset}_test_clustering_features_step4.csv` | Test features (ID + numeric predictors only) |
| `{dataset}_step4_split_membership.csv` | Explicit train/test membership per patient |
| `{dataset}_train_preprocessing_step4.joblib` | Serialised preprocessing rules |
| `{dataset}_train_scaler_step4.joblib` | Serialised scaler |
| `{dataset}_step4_split_metadata.json` | Complete preprocessing audit trail |
| `step4.log` | Per-dataset log file |

---

### Step 5: Spectral Clustering & Multi-Criteria k Selection

**Script:** `scripts/step5_spectral_clustering.py`

**Purpose:** Discover patient subtypes in the training set using spectral clustering. This step is **nested-safe**: it performs k selection entirely within an inner train/validation split of the Step 4 train set, then refits the selected k on the full train set.

#### What It Does

1. **Load** Step 4 train clustering features and the split membership artifact.
2. **Inner split** — Split the Step 4 train set into inner-train and inner-validation subsets.
3. **Clean** the feature matrix (fill any remaining NaNs with column medians).
4. **Run spectral clustering** for k ∈ {2, 3, 4, 5, 6}:
   - Uses `sklearn.cluster.SpectralClustering` with `affinity="nearest_neighbors"` and `assign_labels="kmeans"`.
5. **Multi-criteria k selection** — Instead of silhouette score alone, k is chosen via a weighted composite score combining:
   - Silhouette score
   - Gap statistic
   - Train-resample stability (ARI)
   - Inner-train survival separation
   - Inner-validation survival consistency
   - Biological interpretability of the high-risk cluster
6. **Refit** the selected k on the full Step 4 train set.
7. **Compute centroids** — Mean feature vector for each cluster (used for test-set assignment in Step 7).
8. **Generate silhouette elbow plot.**

#### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--datasets` | `ucsf upenn` | Which datasets to process |
| `--step4-dir` | `outputs/step4` | Step 4 output directory |
| `--output-dir` | `outputs/step5` | Output directory |
| `--k-values` | `2 3 4 5 6` | k values to evaluate |
| `--n-neighbors` | `10` | Nearest neighbours for affinity |
| `--random-state` | `42` | Random seed |
| `--log-level` | `INFO` | Logging verbosity |

#### Usage

```powershell
python scripts/step5_spectral_clustering.py --output-dir outputs/step5 --log-level INFO
```

#### Outputs per Dataset

| File | Description |
|---|---|
| `{dataset}_step5_k_evaluation.csv` | Per-k evaluation metrics (silhouette, gap, stability, survival, etc.) |
| `{dataset}_step5_train_cluster_labels.csv` | Patient ID + cluster label (best k, refitted on full train) |
| `{dataset}_step5_cluster_centroids.csv` | Mean feature vector per cluster + cluster size |
| `{dataset}_step5_selection.json` | Best k, feature list, imputation medians, all criteria details |
| `{dataset}_step5_silhouette_elbow.png` | Silhouette vs k plot |
| `step5.log` | Per-dataset log file |

---

### Step 6: Train-Set Cluster Characterisation

**Script:** `scripts/step6_characterize_clusters.py`

**Purpose:** Produce the statistical characterisation of each cluster — the core results for a clustering paper.

#### What It Does

1. **Cluster summary table** — For each cluster, computes:
   - n (cluster size)
   - Median OS with IQR (Q1, Q3)
   - % MGMT methylated
   - % IDH mutant
   - Mean global NC/EN ratio
   - Mean global ED/EN ratio
   - Mean tumour burden index (TBI)
   - Dominant lobe mode
   - Mean age

2. **Survival analysis:**
   - Kaplan-Meier curves for all clusters on the same axes
   - Multivariate log-rank test (p < 0.05 target)
   - Event construction from both censoring column and survival status text

3. **Continuous feature tests:**
   - Kruskal-Wallis test for every continuous feature across clusters

4. **Categorical feature tests:**
   - Chi-square test for MGMT distribution across clusters
   - Chi-square test for dominant lobe distribution across clusters

5. **Multiple comparison correction:**
   - Benjamini-Hochberg (BH) adjustment applied **globally** across all inferential tests (log-rank + all Kruskal-Wallis + all chi-square)

6. **High-risk cluster identification:**
   - Rule-based composite score: `(1 − OS_norm) + NC/EN_norm + (1 − MGMT_norm) + temporal_bonus`
   - Temporal lobe dominance awards a +0.25 bonus
   - The cluster with the highest score is labelled as the high-risk subtype

#### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--datasets` | `ucsf upenn` | Which datasets to process |
| `--step4-dir` | `outputs/step4` | Step 4 output directory |
| `--step5-dir` | `outputs/step5` | Step 5 output directory |
| `--output-dir` | `outputs/step6` | Output directory |
| `--log-level` | `INFO` | Logging verbosity |

#### Usage

```powershell
python scripts/step6_characterize_clusters.py --output-dir outputs/step6 --log-level INFO
```

#### Outputs per Dataset

| File | Description |
|---|---|
| `{dataset}_step6_cluster_summary_train.csv` | Per-cluster summary stats |
| `{dataset}_step6_train_with_clusters.csv` | Full train table with cluster labels |
| `{dataset}_step6_kruskal_tests.csv` | Kruskal-Wallis results (raw + BH-adjusted) |
| `{dataset}_step6_chi_square_tests.csv` | Chi-square results (raw + BH-adjusted) |
| `{dataset}_step6_inferential_tests_bh.csv` | All tests pooled with BH correction |
| `{dataset}_step6_kaplan_meier_train.png` | KM survival curves for train set |
| `{dataset}_step6_metadata.json` | Columns used, high-risk cluster, p-values |
| `step6.log` | Per-dataset log file |

---

### Step 7: Held-Out Test-Set Validation

**Script:** `scripts/step7_validate_test_set.py`

**Purpose:** Open the held-out test set for the first time and validate whether the cluster structure and survival differences replicate. Because Step 5 now performs k selection inside an inner split, this test set is truly untouched by any model-selection decision.

#### What It Does

1. **Load** test features, train centroids, and Step 6 metadata.
2. **Clean test features** using train-derived imputation medians from Step 5.
3. **Assign test patients** to the nearest train centroid via Euclidean distance (no re-clustering).
4. **Build test cluster summary** — same statistics as Step 6 (median OS, MGMT rate, NC/EN ratio, dominant lobe, etc.).
5. **Cluster proportion comparison** — side-by-side train vs test proportions with ±10 percentage-point threshold check.
6. **Survival validation:**
   - Test-set Kaplan-Meier curves
   - Test-set log-rank test
7. **High-risk cluster replication check:**
   - Does the high-risk cluster still show temporal dominance?
   - Does it still show high NC/EN ratio?
   - Is the proportion within ±10 pp of training?
   - Does the dominant lobe match the training signature?

#### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--datasets` | `ucsf upenn` | Which datasets to process |
| `--step4-dir` | `outputs/step4` | Step 4 output directory |
| `--step5-dir` | `outputs/step5` | Step 5 output directory |
| `--step6-dir` | `outputs/step6` | Step 6 output directory |
| `--output-dir` | `outputs/step7` | Output directory |
| `--log-level` | `INFO` | Logging verbosity |

#### Usage

```powershell
python scripts/step7_validate_test_set.py --output-dir outputs/step7 --log-level INFO
```

#### Outputs per Dataset

| File | Description |
|---|---|
| `{dataset}_step7_test_with_assigned_clusters.csv` | Test table with assigned cluster labels + distances |
| `{dataset}_step7_cluster_summary_test.csv` | Per-cluster summary stats for test set |
| `{dataset}_step7_cluster_proportions_comparison.csv` | Train vs test proportion comparison |
| `{dataset}_step7_kaplan_meier_test.png` | KM survival curves for test set |
| `{dataset}_step7_metadata.json` | Full validation audit trail |
| `step7.log` | Per-dataset log file |

---

### Step 8: Repeated Validation Across Random Splits

**Script:** `scripts/step8_repeated_validation.py`

**Purpose:** Re-run Steps 4–7 across many random seeds to check whether the findings are robust or merely the product of one fortunate split. Step 8 is a **thin wrapper** around the shared `repeated_evaluation_runner.py`.

#### What It Does

For each seed, the runner:
1. Re-performs Step 4 (split + preprocess) with a new random state.
2. Re-performs Step 5 (nested-safe clustering + k selection).
3. Re-performs Step 6 (characterisation) and Step 7 (validation).
4. Records per-run metrics.

#### Metrics Recorded per Run

- Selected k
- High-risk cluster label
- Train and test log-rank p-values
- Survival ordering consistency
- Dominant lobe matching
- High-risk cluster proportion drift
- NC/EN rank of the high-risk cluster on test

#### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--datasets` | `ucsf upenn` | Which datasets to process |
| `--step3-dir` | `outputs/step3` | Step 3 output directory |
| `--output-dir` | `results/repeated-validation` | Output directory |
| `--n-runs` | `30` | Number of random splits |
| `--seed-start` | `1` | First random seed |
| `--seeds` | *(auto)* | Explicit seed list (overrides `--n-runs`) |
| `--test-size` | `0.30` | Test set fraction |
| `--inner-test-size` | `0.25` | Inner validation fraction (for Step 5) |
| `--k-values` | `2 3 4 5 6` | k values to evaluate |
| `--n-neighbors` | `10` | Nearest neighbours for affinity |
| `--gap-refs` | `5` | Gap statistic reference datasets |
| `--stability-resamples` | `8` | Stability resamples for k selection |
| `--stability-sample-fraction` | `0.80` | Subsample fraction for stability |
| `--max-missing-feature-frac` | `1.0` | Drop features exceeding this missing rate |
| `--corr-prune-threshold` | `1.01` | Correlation threshold for feature pruning |
| `--winsorize-lower-quantile` | `0.0` | Lower winsorisation quantile |
| `--winsorize-upper-quantile` | `1.0` | Upper winsorisation quantile |
| `--scaler-type` | `standard` | Scaler type: `standard` or `robust` |
| `--power-transform` | `none` | Power transform: `none` or `yeo-johnson` |
| `--log-level` | `WARNING` | Logging verbosity |

#### Usage

```powershell
python scripts/step8_repeated_validation.py --n-runs 30 --output-dir results/repeated-validation
```

#### Key Outputs

| Path | Description |
|---|---|
| `results/repeated-validation/runs/` | Per-run pipeline output snapshots |
| `results/repeated-validation/summary/repeated_validation_run_metrics.csv` | Aggregated per-run metrics |
| `results/repeated-validation/summary/repeated_validation_summary.txt` | Human-readable stability summary |

---

### Step 9: Bootstrap Cluster Stability

**Script:** `scripts/step9_cluster_stability.py`

**Purpose:** Measure whether the train-set clusters discovered in Step 5 are reproducible under perturbation. This complements Step 7 (which tests generalisation) and Step 8 (which tests split sensitivity).

- **Step 7 asks:** Do train-derived clusters generalise to held-out patients?
- **Step 8 asks:** Are findings stable across different train/test splits?
- **Step 9 asks:** If we perturb the train set, do we recover the same clusters?

#### What It Does

For each dataset:

1. Load Step 4 train features and Step 5 cluster labels + selected k.
2. Run spectral clustering on `n_bootstraps` bootstrap resamples of the train set.
3. For each bootstrap run: collapse duplicate samples back to original patients, compare to full-train labels via ARI.
4. Compute pairwise ARI across all bootstrap runs.
5. Build a **consensus matrix**: how often two patients are assigned to the same cluster, normalised by how often they appear together.
6. Cluster the consensus matrix to obtain a consensus partition.
7. Produce a co-clustering heatmap ordered by consensus cluster.

#### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--datasets` | `ucsf upenn` | Which datasets to process |
| `--step4-dir` | `outputs/step4` | Step 4 output directory |
| `--step5-dir` | `outputs/step5` | Step 5 output directory |
| `--output-dir` | `outputs/step9` | Output directory |
| `--n-bootstraps` | `100` | Number of bootstrap runs |
| `--bootstrap-fraction` | `1.0` | Fraction of train rows sampled per run |
| `--random-state` | `42` | Random seed |
| `--n-neighbors` | *(from Step 5)* | Optional override for nearest neighbours |
| `--log-level` | `INFO` | Logging verbosity |

#### Usage

```powershell
python scripts/step9_cluster_stability.py --datasets ucsf upenn --n-bootstraps 100
```

#### Outputs per Dataset

| File | Description |
|---|---|
| `{dataset}_step9_bootstrap_runs.csv` | Per-run metrics (ARI vs full-train labels) |
| `{dataset}_step9_pairwise_ari.csv` | ARI between pairs of bootstrap runs |
| `{dataset}_step9_consensus_matrix.csv` | Patient-by-patient co-clustering frequencies |
| `{dataset}_step9_consensus_labels.csv` | Baseline Step 5 labels and consensus-cluster labels |
| `{dataset}_step9_patient_stability.csv` | Per-patient stability summaries |
| `{dataset}_step9_consensus_heatmap.png` | Consensus co-clustering heatmap |
| `{dataset}_step9_summary.json` | Summary of ARI and consensus-strength metrics |
| `step9.log` | Per-dataset log file |

#### How to Read the Results

| Metric | Interpretation |
|---|---|
| `consensus_vs_full_train_ari` | Agreement between consensus partition and original Step 5 clustering |
| `pairwise_bootstrap_ari` | Reproducibility of bootstrap clusterings against each other |
| `mean_within_cluster` / `mean_between_cluster` | Separation quality in the consensus matrix |
| Heatmap diagonal blocks | Strong block structure = well-separated, stable clusters |

Rule of thumb: ARI near 1 = strong reproducibility; 0.5–0.7 = moderate; near 0 = weak.

---

### Step 11: Cohort Shift Analysis (UCSF vs UPenn)

**Script:** `scripts/step11_cohort_shift_analysis.py`

**Purpose:** Compare the UCSF and UPenn cohorts *before* clustering to quantify whether they differ substantially. If UPenn is genuinely shifted relative to UCSF, weaker UPenn validation may reflect cohort differences rather than model weakness alone.

> **Note:** Step 10 (nested validation) has been **retired**. It is an empty file kept for history. Step 8 is now the single repeated-validation entry point.

#### What It Compares

- Age distribution (Mann-Whitney U, KS test, standardised mean difference)
- MGMT methylation rate (chi-square, Cramér's V)
- IDH mutation rate (chi-square, Cramér's V)
- Dominant lobe distribution (chi-square, Cramér's V)
- NC/EN, ED/EN, TBI distributions
- Missingness patterns across all shared columns
- Survival endpoint definitions (OS origin: diagnosis vs surgery)

#### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--step3-dir` | `outputs/step3` | Step 3 output directory |
| `--output-dir` | `outputs/step11` | Output directory |

#### Usage

```powershell
python scripts/step11_cohort_shift_analysis.py --output-dir outputs/step11
```

#### Key Outputs

| File | Description |
|---|---|
| `step11_feature_distribution_comparison.csv` | Per-feature statistical comparison |
| `step11_lobe_distribution_comparison.csv` | Lobe distribution comparison |
| `step11_missingness_comparison.csv` | Column-level missingness comparison |
| `step11_survival_endpoint_definition.csv` | Survival time-origin definitions |
| `step11_summary.json` | Machine-readable summary |
| `step11_cohort_shift_report.txt` | Human-readable text report |

---

## 7. Support Modules

Several shared Python modules in `scripts/` support the pipeline steps:

| Module | Purpose |
|---|---|
| `pipeline_preprocessing.py` | Shared functions for missing-marker replacement, numeric coercion, column alias resolution, binary encoding (sex, MGMT, IDH), and dominant lobe canonicalisation |
| `cluster_validation_utils.py` | Reusable cluster metric and validation utilities used by Steps 5–9 |
| `repeated_evaluation_runner.py` | Shared orchestration logic for running Steps 4–7 across many seeds (used by Step 8) |
| `run_pipeline_iteration.py` | Single-iteration pipeline runner that executes Steps 4–7 sequentially for one seed |
| `iteration_paths.py` | Utility for managing `results/iteration-N/` directory numbering |
| `build_project_report_data.py` | Computes report-ready data from pipeline outputs (metrics are derived once, rendered many ways) |
| `render_project_reports.py` | Renders formatted reports from computed report data |

---

## 8. Key Design Decisions

### Why Spectral Clustering?

Spectral clustering operates on a graph-based affinity matrix and can discover non-convex cluster shapes that k-means would miss. For heterogeneous cancer data with mixed feature types, this is more appropriate than centroid-based methods. The `nearest_neighbors` affinity adapts to local data density.

### Why Multi-Criteria k Selection?

Silhouette score alone can favour trivial two-cluster solutions that are statistically clean but biologically uninformative. The current Step 5 uses a weighted composite of silhouette, gap statistic, resample stability, inner-train survival separation, inner-validation survival consistency, and biological interpretability. This produces k values that are both statistically and clinically defensible.

### Why Nearest-Centroid Assignment for Test Set?

Scikit-learn's `SpectralClustering` does not provide a `predict()` method for unseen data. The standard approach in clustering studies is to:
1. Compute cluster centroids (mean feature vectors) from the training set.
2. Assign new patients to the cluster whose centroid is closest (Euclidean distance).

This is methodologically equivalent to how cluster findings would be applied in a clinical setting.

### Why Benjamini-Hochberg Correction?

When performing multiple statistical tests (log-rank + Kruskal-Wallis + chi-square), the family-wise error rate inflates. BH controls the **false discovery rate (FDR)** at 5%, which is the standard correction in biomedical research — less conservative than Bonferroni but with better statistical power.

### Why Stratified Splitting?

Simple random 70/30 splits can create imbalanced partitions where MGMT or survival distributions differ substantially between train and test. Stratification ensures both sets are representative, which is critical for survival validation in Step 7.

### Why Are Outcome Variables Excluded from Clustering?

Including OS or survival status in the clustering feature set would be **circular reasoning** — the algorithm would cluster by survival, and then the survival analysis would trivially show differences. By excluding outcomes, any survival separation between clusters is a genuine emergent property of the imaging and clinical features.

### Why a Nested Inner Split in Step 5?

Using the full train set for both k selection and final clustering means the held-out test set is indirectly seen by the model-selection process (since the same train set that determined k is fully used). The inner split ensures k is selected on data that is distinct from the final refit, making Step 7 a truly independent validation.

---

## 9. Output Inventory

All outputs are organised by step and dataset:

```
outputs/
├── step3/
│   ├── ucsf/
│   │   ├── ucsf_master_table_step3.csv
│   │   └── ucsf_step3_metadata.json
│   └── upenn/
│       ├── upenn_master_table_step3.csv
│       └── upenn_step3_metadata.json
│
├── step4/
│   ├── ucsf/
│   │   ├── ucsf_train_master_table_step4.csv
│   │   ├── ucsf_test_master_table_step4.csv
│   │   ├── ucsf_train_clustering_features_step4.csv
│   │   ├── ucsf_test_clustering_features_step4.csv
│   │   ├── ucsf_step4_split_membership.csv
│   │   ├── ucsf_train_preprocessing_step4.joblib
│   │   ├── ucsf_train_scaler_step4.joblib
│   │   ├── ucsf_step4_split_metadata.json
│   │   └── step4.log
│   └── upenn/
│       └── (same structure)
│
├── step5/
│   ├── ucsf/
│   │   ├── ucsf_step5_k_evaluation.csv
│   │   ├── ucsf_step5_train_cluster_labels.csv
│   │   ├── ucsf_step5_cluster_centroids.csv
│   │   ├── ucsf_step5_selection.json
│   │   ├── ucsf_step5_silhouette_elbow.png
│   │   └── step5.log
│   └── upenn/
│       └── (same structure)
│
├── step6/
│   ├── ucsf/
│   │   ├── ucsf_step6_cluster_summary_train.csv
│   │   ├── ucsf_step6_train_with_clusters.csv
│   │   ├── ucsf_step6_kruskal_tests.csv
│   │   ├── ucsf_step6_chi_square_tests.csv
│   │   ├── ucsf_step6_inferential_tests_bh.csv
│   │   ├── ucsf_step6_kaplan_meier_train.png
│   │   ├── ucsf_step6_metadata.json
│   │   └── step6.log
│   └── upenn/
│       └── (same structure)
│
├── step7/
│   ├── ucsf/
│   │   ├── ucsf_step7_test_with_assigned_clusters.csv
│   │   ├── ucsf_step7_cluster_summary_test.csv
│   │   ├── ucsf_step7_cluster_proportions_comparison.csv
│   │   ├── ucsf_step7_kaplan_meier_test.png
│   │   ├── ucsf_step7_metadata.json
│   │   └── step7.log
│   └── upenn/
│       └── (same structure)
│
├── step9/ (when run)
│   ├── ucsf/
│   │   ├── ucsf_step9_bootstrap_runs.csv
│   │   ├── ucsf_step9_pairwise_ari.csv
│   │   ├── ucsf_step9_consensus_matrix.csv
│   │   ├── ucsf_step9_consensus_labels.csv
│   │   ├── ucsf_step9_patient_stability.csv
│   │   ├── ucsf_step9_consensus_heatmap.png
│   │   ├── ucsf_step9_summary.json
│   │   └── step9.log
│   └── upenn/
│       └── (same structure)
│
└── step11/ (when run)
    ├── step11_feature_distribution_comparison.csv
    ├── step11_lobe_distribution_comparison.csv
    ├── step11_missingness_comparison.csv
    ├── step11_survival_endpoint_definition.csv
    ├── step11_summary.json
    └── step11_cohort_shift_report.txt

results/
├── iteration-1/                           # Pipeline snapshot (iteration runs)
├── iteration-2/
└── repeated-validation/                   # Step 8 repeated split results
    ├── runs/
    └── summary/
        ├── repeated_validation_run_metrics.csv
        └── repeated_validation_summary.txt
```

---

## 10. Environment & Setup

### Prerequisites

- Python 3.10+
- Virtual environment recommended

### Installation

```powershell
# Navigate to project root
cd "C:\Users\Husain\Documents\AIml sideproject\GBM multi model"

# Create virtual environment
python -m venv .venv

# Activate (PowerShell)
.\.venv\Scripts\Activate.ps1

# If PowerShell blocks activation:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# Install dependencies
pip install -r requirements.txt
```

### Core Dependencies

```
numpy
pandas
scikit-learn
joblib
scipy
lifelines
matplotlib
```

---

## 11. Full Run — Quick Start

### Main Workflow (Steps 3–7)

Run all steps sequentially from the project root:

```powershell
# Step 3: Build master tables
python scripts/step3_build_master_feature_table.py --output-dir outputs/step3

# Step 4: Split and preprocess
python scripts/step4_split_train_test.py --output-dir outputs/step4 --log-level INFO

# Step 5: Spectral clustering (with nested k selection)
python scripts/step5_spectral_clustering.py --output-dir outputs/step5 --log-level INFO

# Step 6: Characterise clusters
python scripts/step6_characterize_clusters.py --output-dir outputs/step6 --log-level INFO

# Step 7: Validate on test set
python scripts/step7_validate_test_set.py --output-dir outputs/step7 --log-level INFO
```

Each step reads outputs from the previous step automatically. The main workflow takes approximately 2–5 minutes depending on hardware.

### Robustness Analyses (Optional but Recommended)

```powershell
# Step 8: Repeated validation (30 random splits)
python scripts/step8_repeated_validation.py --n-runs 30

# Step 9: Bootstrap cluster stability
python scripts/step9_cluster_stability.py --n-bootstraps 100

# Step 11: Cohort shift analysis
python scripts/step11_cohort_shift_analysis.py
```

### Run a Single Dataset

To process only UCSF (or UPenn), add `--datasets ucsf` (or `--datasets upenn`) to Steps 4–9.

### Publication Workflow

For publication-ready results:

1. Run the **main workflow** (Steps 3–7) as the primary leakage-safe analysis.
2. Run **Step 8** and **Step 9** as robustness evidence.
3. Run **Step 11** to justify cohort-aware interpretation, especially for weaker UPenn findings.

---

## 12. Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError` for any package | `pip install --upgrade pandas numpy scikit-learn joblib scipy lifelines matplotlib` |
| PowerShell blocks `.ps1` activation | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` |
| "No input files found" in Step 3 | Confirm `UCSF_sri24_atlas_features_merged.csv` and `UPenn_sri24_atlas_features_merged.csv` exist in `Clinical+Atlas Merged Data/` |
| "Step N files missing" in Step N+1 | Run the preceding step first |
| matplotlib plots not generated | Ensure `matplotlib` is installed; scripts gracefully skip plots if it is missing |
| Silhouette errors for large k | Reduce `--k-values` or increase data; k must be < n_samples |
| Stratification warnings in Step 4 | Normal fallback behaviour — the script tries simpler strategies automatically |
| Step 8 runs slowly | Reduce `--n-runs` or increase `--log-level` to `WARNING` to suppress per-step logging |
| Step 9 low ARI values | May indicate genuinely unstable clusters; review k selection and feature set before concluding |
| UPenn validation weaker than UCSF | Run Step 11; cohort shift (MGMT, IDH, survival endpoint) may explain the gap |

---

## Appendix A: Feature Glossary

| Feature | Type | Description |
|---|---|---|
| `dominant_lobe` | Categorical | Brain lobe with the largest tumour volume (frontal / temporal / parietal / occipital) |
| `global_nc_en_ratio` | Continuous | Ratio of necrotic core (NC) volume to enhancing tumour (EN) volume across the whole tumour |
| `global_ed_en_ratio` | Continuous | Ratio of peritumoral oedema (ED) volume to enhancing tumour (EN) volume |
| `global_ed_total_ratio` | Continuous | Ratio of oedema volume to total tumour volume |
| `tumor_burden_index` | Continuous | Normalised total tumour volume relative to brain volume |
| `{lobe}_nc_en_ratio` | Continuous | NC/EN ratio within a specific brain lobe (frontal, temporal, parietal, occipital) |
| `{lobe}_ed_en_ratio` | Continuous | ED/EN ratio within a specific brain lobe |
| `{lobe}_ed_nc_ratio` | Continuous | ED/NC ratio within a specific brain lobe |
| `sex_bin` | Binary | Sex (male=1, female=0) |
| `mgmt_bin` | Binary | MGMT promoter methylation status (methylated=1, unmethylated=0) |
| `idh_bin` | Binary | IDH mutation status (mutant=1, wild-type=0) |
| `dominant_lobe_{lobe}` | Binary | One-hot encoded dominant lobe indicators |
| `Age at MRI` | Continuous | Patient age at time of MRI acquisition |
| `OS` | Outcome | Overall survival in days (excluded from clustering features) |
| `1-dead 0-alive` | Outcome | Censoring indicator (excluded from clustering features) |

---

## Appendix B: Statistical Methods Reference

### Silhouette Score
Measures how similar each sample is to its own cluster compared to other clusters. Ranges from −1 to +1; higher values indicate better-defined clusters.

### Gap Statistic
Compares the total within-cluster variation for different values of k against their expected values under a null reference distribution (uniform random data). The k where the gap is largest is the optimal cluster count.

### Adjusted Rand Index (ARI)
Measures agreement between two clusterings, corrected for chance. ARI = 1 means perfect agreement; ARI ≈ 0 means random-level agreement; ARI < 0 means less than chance.

### Log-Rank Test
Non-parametric hypothesis test comparing survival distributions between groups. Tests the null hypothesis that all clusters have identical survival functions. Implemented via `lifelines.statistics.multivariate_logrank_test`.

### Kruskal-Wallis Test
Non-parametric one-way ANOVA; tests whether samples from different clusters originate from the same distribution. Used for continuous features because normality is not assumed.

### Chi-Square Test of Independence
Tests whether two categorical variables (e.g., cluster label and MGMT status) are independent. Uses contingency tables.

### Mann-Whitney U Test
Non-parametric test comparing two independent samples to determine whether one distribution is stochastically greater than the other. Used in Step 11 for cohort comparisons.

### Kolmogorov-Smirnov (KS) Test
Non-parametric test comparing whether two samples are drawn from the same distribution by measuring the maximum distance between empirical CDFs. Used in Step 11 alongside Mann-Whitney.

### Cramér's V
Measures the strength of association between two categorical variables, derived from the chi-square statistic. Ranges from 0 (no association) to 1 (perfect association). Used in Step 11 for binary and lobe distribution comparisons.

### Benjamini-Hochberg (BH) Procedure
Controls the false discovery rate (FDR) when performing multiple hypothesis tests. Ranks p-values from smallest to largest and adjusts each p-value based on its rank and the total number of tests. A test is significant at FDR = 0.05 if its adjusted p-value < 0.05.

### Kaplan-Meier Estimator
Non-parametric estimator of the survival function from censored data. Produces step-function survival curves with confidence intervals. Implemented via `lifelines.KaplanMeierFitter`.

### Consensus Clustering
Obtains a robust partition by aggregating many bootstrap clusterings into a co-clustering frequency matrix and re-clustering that matrix. Strong diagonal blocks indicate stable, reproducible clusters.

---

## Appendix C: Data Preprocessing Theory

The file `Data_Preprocessing_and_Feature_Engineering (1).md` at the project root contains a comprehensive reference on data preprocessing and feature engineering theory, covering:

- **Part I** — Understanding raw data: data types, feature types, dataset characteristics, exploratory data analysis (EDA)
- **Part II** — Data cleaning: missing data mechanisms (MCAR/MAR/MNAR), imputation methods (mean/median/mode, KNN, regression, multiple imputation), noise and outlier handling (z-score, IQR, Isolation Forest), data consistency
- **Part III** — Data transformation: feature scaling (min-max, z-score, robust, log/power), categorical encoding (label, one-hot, binary, target, embeddings)
- **Part IV** — Feature engineering: ratio features, difference features, aggregate features, interaction/polynomial features, domain-driven features
- **Part V** — Feature selection: filter methods (correlation, chi-square, mutual information), wrapper methods (forward/backward/RFE), embedded methods (LASSO, tree importance)
- **Part VI** — Dimensionality reduction: PCA, LDA, t-SNE, UMAP, autoencoders
- **Part VII** — Modelling preparation: train/validation/test splitting, stratification, cross-validation, class imbalance handling (SMOTE, undersampling, class weights), pipeline construction and leakage prevention
- **Part VIII** — Case study: end-to-end preprocessing on the Titanic dataset

This document serves as the theoretical foundation for the preprocessing choices made in Steps 3–4 of the pipeline.

---

*Last updated: 2026-04-15*
