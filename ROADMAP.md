# GBM Multi-Modal Clustering: Research-Grade Roadmap

## Hardware Profile

| Component | Spec |
|-----------|------|
| CPU | AMD Ryzen 7 7000 series |
| GPU | NVIDIA RTX 4050, 6 GB VRAM |
| Datasets | UCSF-PDGM (501 patients) + UPENN-GBM (671 patients) |

> **6 GB VRAM rules out:** 3D CNNs on raw MRI, self-supervised pretraining, large batch contrastive learning.
> **6 GB VRAM is fine for:** VAE on tabular features (~200 dims), DeepSurv, SHAP, all scikit-learn methods, PyRadiomics (CPU-bound).

---

## Thesis Title (Refined)

> *"Multi-institutional radiomics-clinical consensus clustering identifies reproducible
> high-risk GBM subtypes: a study of 1,172 patients with external validation"*

---

## Day-by-Day Execution Plan (7 Days)

---

### DAY 1 — Environment Setup & MRI Preprocessing

**Goal:** Get all data loadable, skull-stripped, resampled, and sanity-checked.

- [ ] Set up conda environment
  ```
  conda create -n gbm python=3.10
  conda activate gbm
  pip install nibabel nilearn numpy pandas scikit-learn matplotlib seaborn
  pip install antspyx pyradiomics lifelines umap-learn shap
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  ```
- [ ] Write a script to load and verify all NIfTI files (T1, T1GD, T2, FLAIR + segmentation mask) for both datasets
- [ ] Verify segmentation label mapping for both datasets:
  - UCSF-PDGM: labels are typically `{1: NCR/NET, 2: ED, 4: ET}` (BraTS convention)
  - UPENN-GBM: same convention, confirm from `automated_approx_segm` files
- [ ] Resample all volumes to 1mm isotropic (if not already) using nibabel/nilearn
- [ ] Log any patients with missing/corrupt files → exclusion list

**Output:** `data/verified_patient_list.csv` with columns `[patient_id, dataset, has_t1, has_t1gd, has_t2, has_flair, has_seg]`

---

### DAY 2 — Atlas Registration & Lobe Mapping

**Goal:** Map each tumor to brain lobe(s) using a standard atlas.

- [ ] Register each patient's T1 to MNI152 space using ANTsPy (affine only — fast, ~2 min/patient)
  ```python
  import ants
  fixed = ants.image_read(ants.get_ants_data('mni'))
  moving = ants.image_read(patient_t1_path)
  tx = ants.registration(fixed, moving, type_of_transform='Affine')
  # Apply same transform to segmentation mask (use nearest-neighbor interpolation)
  seg_mni = ants.apply_transforms(fixed, seg, tx['fwdtransforms'],
                                   interpolator='nearestNeighbor')
  ```
- [ ] Download Harvard-Oxford cortical atlas (comes with FSL, also available via nilearn)
  ```python
  from nilearn import datasets
  atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-1mm')
  ```
- [ ] For each patient, compute overlap between tumor mask and atlas regions
- [ ] Aggregate into lobe-level involvement: `frontal_%, temporal_%, parietal_%, occipital_%, insular_%, other_%`
- [ ] Determine dominant lobe (highest overlap percentage)

**Output:** `features/lobe_mapping.csv` — one row per patient, columns for each lobe's involvement percentage + dominant lobe label.

> **CPU note:** ~1172 registrations at ~2 min each = ~40 hours. Parallelize using `joblib` with `n_jobs=8` on your Ryzen → ~5 hours. Start this as an overnight batch.

---

### DAY 3 — Radiomics Feature Extraction

**Goal:** Extract standardized radiomic features from each tumor sub-region.

- [ ] Define sub-regions from segmentation mask:
  - **ET** (Enhancing Tumor) — label 4
  - **NCR/NET** (Necrotic Core) — label 1
  - **ED** (Peritumoral Edema) — label 2
  - **Whole Tumor** (all labels combined)
- [ ] For each sub-region × each modality (T1, T1GD, T2, FLAIR), extract PyRadiomics features:
  ```python
  from radiomics import featureextractor
  extractor = featureextractor.RadiomicsFeatureExtractor()
  extractor.enableAllFeatures()
  # Disable features that need too much RAM for 3D volumes
  extractor.disableAllInputImages()
  extractor.enableInputImageByName('Original')
  result = extractor.execute(image_path, mask_path, label=4)  # ET
  ```
- [ ] Feature categories to extract:
  - **Shape** (14 features) — only from whole tumor mask, not per-modality
  - **First Order** (18 features) — per sub-region × per modality
  - **GLCM** (24 features) — per sub-region × per modality
  - **GLRLM** (16 features) — per sub-region × per modality
- [ ] Add hand-crafted ratio features:
  - `NC_ET_ratio = vol_NCR / vol_ET`
  - `ED_ET_ratio = vol_ED / vol_ET`
  - `tumor_burden = vol_whole_tumor / vol_brain`
  - `necrosis_fraction = vol_NCR / vol_whole_tumor`

**Output:** `features/radiomics_features.csv` — ~150-300 features per patient.

> **Compute note:** PyRadiomics is CPU-bound, ~30-60 sec per patient. With 1172 patients → ~10-20 hours. Run overnight. Parallelize extraction across patients.

---

### DAY 4 — Clinical Data Harmonization & Feature Matrix Assembly

**Goal:** Merge radiomics, lobe data, and clinical data into one clean feature matrix.

#### 4A: Harmonize Clinical Variables

- [ ] Map UCSF columns to common schema:
  | UCSF Column | Common Name |
  |---|---|
  | `Sex` | `sex` (M/F) |
  | `Age at MRI` | `age` |
  | `MGMT status` | `mgmt` (positive/negative/unknown) |
  | `IDH` | `idh` (wildtype/mutant/unknown) |
  | `1-dead 0-alive` | `event` (1=dead, 0=censored) |
  | `OS` | `survival_days` |
  | `EOR` | `extent_of_resection` |

- [ ] Map UPenn columns to same schema:
  | UPenn Column | Common Name |
  |---|---|
  | `Gender` | `sex` |
  | `Age_at_scan_years` | `age` |
  | `MGMT` | `mgmt` |
  | `IDH1` | `idh` |
  | `Survival_Status` | `event` |
  | `Survival_from_surgery_days_UPDATED` | `survival_days` |
  | `GTR_over90percent` | `extent_of_resection` |

- [ ] Handle missing values:
  - MGMT: large portion of UPenn is "Not Available" → encode as `unknown` category
  - IDH: "NOS/NEC" → `unknown`
  - Survival: drop patients with no survival data
  - For continuous features with <10% missing → median imputation
  - For continuous features with >10% missing → MICE imputation (use `sklearn.experimental.enable_iterative_imputer`)

#### 4B: Assemble Feature Matrix

- [ ] Merge: radiomics + lobe mapping + clinical → master DataFrame
- [ ] Feature selection pipeline:
  1. Remove zero-variance features
  2. Remove highly correlated features (Pearson r > 0.90, keep one from each pair)
  3. Apply mRMR (minimum Redundancy Maximum Relevance) or Boruta for final selection → target ~40-60 features
- [ ] Standardize all continuous features (z-score)
- [ ] One-hot encode categoricals (sex, dominant_lobe, mgmt, idh) — but keep `unknown` as its own category
- [ ] Save feature matrix + separate survival data (time + event)

**Output:**
- `features/merged_feature_matrix.csv`
- `features/survival_data.csv`
- `features/feature_names.json`

#### 4C: Train/Validation Split

- [ ] Discovery cohort = UCSF-PDGM (~501 patients)
- [ ] External validation cohort = UPENN-GBM (~671 patients)
- [ ] **Do NOT mix datasets for training.** This is your key selling point.

---

### DAY 5 — Clustering (Main Contribution)

**Goal:** Run multiple clustering methods, find consensus, determine optimal k.

#### 5A: Consensus Clustering on Discovery Cohort (UCSF)

- [ ] Run 4 clustering algorithms for k = 2 to 8:
  1. **Spectral Clustering** (original roadmap method — keep as baseline)
  2. **Gaussian Mixture Model** (soft assignments, captures elliptical clusters)
  3. **Agglomerative Hierarchical** (Ward linkage)
  4. **K-Medoids** (robust to outliers)
  ```python
  from sklearn.cluster import SpectralClustering, AgglomerativeClustering
  from sklearn.mixture import GaussianMixture
  from sklearn_extra.cluster import KMedoids
  ```
- [ ] Build consensus matrix for each k:
  - For each algorithm, record co-occurrence (do patients i and j land in the same cluster?)
  - Average co-occurrence across 4 algorithms = consensus matrix
  - Final cluster assignments = spectral clustering on the consensus matrix
- [ ] Select optimal k using:
  - Silhouette score
  - Gap statistic
  - Consensus CDF (cumulative distribution function) — area under CDF should plateau
  - **Log-rank test p-value** on survival curves (most clinically relevant criterion)

#### 5B: Cluster Stability Analysis

- [ ] Bootstrap 1000 iterations:
  - Resample discovery cohort with replacement
  - Recluster, compute Adjusted Rand Index (ARI) vs. original assignments
  - Report mean ARI ± std (>0.7 = stable)

#### 5C: (Optional, if time) VAE-Based Clustering

- [ ] Train a small Variational Autoencoder on the feature matrix:
  ```
  Input (50-60 features) → 128 → 64 → latent (10-dim) → 64 → 128 → Output
  ```
  - This is ~50K parameters. Trains in seconds on your GPU.
- [ ] Cluster the 10-dim latent space using the same consensus approach
- [ ] Compare with direct feature clustering → if VAE clusters give better survival separation, report as improvement

**Output:**
- `results/cluster_assignments_discovery.csv`
- `results/consensus_matrix.npy`
- `results/optimal_k_analysis.png`
- `results/cluster_stability_ari.csv`

---

### DAY 6 — Clinical Validation & Survival Analysis

**Goal:** Prove that clusters have clinical meaning and independent prognostic value.

#### 6A: Survival Analysis

- [ ] Kaplan-Meier curves per cluster (discovery cohort)
  ```python
  from lifelines import KaplanMeierFitter
  from lifelines.statistics import logrank_test, multivariate_logrank_test
  ```
- [ ] Pairwise log-rank tests between all cluster pairs
- [ ] Multivariate Cox Proportional Hazards:
  ```python
  from lifelines import CoxPHFitter
  # Model: survival ~ cluster + age + sex + mgmt + idh + eor
  # Key question: is cluster membership significant AFTER controlling for known factors?
  ```
- [ ] Report Hazard Ratios with 95% CI for each cluster (reference = best-survival cluster)
- [ ] Concordance index (C-index) comparison:
  - Clinical variables only
  - Cluster membership only
  - Combined (clinical + cluster) → must be > either alone

#### 6B: Cluster Characterization

- [ ] For each cluster, compute:
  - Mean ± std of all radiomics features
  - Distribution of MGMT, IDH, sex, dominant lobe
  - Median survival + 95% CI
- [ ] Chi-square / Fisher's exact test for categorical variables across clusters
- [ ] Kruskal-Wallis test for continuous variables across clusters
- [ ] SHAP-based explanation:
  ```python
  from sklearn.ensemble import RandomForestClassifier
  import shap
  # Train RF to predict cluster membership
  rf = RandomForestClassifier(n_estimators=500)
  rf.fit(X_discovery, cluster_labels)
  explainer = shap.TreeExplainer(rf)
  shap_values = explainer.shap_values(X_discovery)
  # Generates: which features most define each cluster
  ```

#### 6C: (Optional) DeepSurv

- [ ] If Cox PH + clusters already shows strong results, train DeepSurv as a comparison:
  ```python
  # Small MLP: features → 64 → 32 → 1 (log-hazard)
  # Loss: negative partial log-likelihood
  # Trains in <1 minute on your RTX 4050
  ```
- [ ] Compare C-index: CoxPH vs DeepSurv

**Output:**
- `results/kaplan_meier_curves.png`
- `results/cox_model_summary.txt`
- `results/cluster_characterization_table.csv`
- `results/shap_summary.png`

---

### DAY 7 — External Validation & Final Figures

**Goal:** Prove generalizability on UPenn cohort and produce publication-ready outputs.

#### 7A: External Validation on UPenn

- [ ] Apply the **same feature pipeline** (same scaling parameters, same feature set) to UPenn data
- [ ] Assign UPenn patients to clusters using:
  - **Option A (preferred):** Train a classifier (RF or KNN) on UCSF cluster assignments → predict UPenn cluster labels
  - **Option B:** Nearest centroid assignment based on UCSF cluster centroids
- [ ] Repeat all Day 6 analyses on UPenn:
  - Kaplan-Meier + log-rank
  - Cox PH
  - Cluster characterization
- [ ] Key figure: **Side-by-side KM curves** (UCSF left, UPenn right) — if the same cluster is worst in both, that's your main result

#### 7B: Visualization

- [ ] UMAP embedding of all 1172 patients:
  ```python
  import umap
  reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=42)
  embedding = reducer.fit_transform(X_all)
  # Color by: cluster, dataset source, survival (short/long), MGMT, IDH
  ```
- [ ] Radar plot: mean feature profile per cluster (normalized to [0,1])
- [ ] Heatmap: features × clusters with hierarchical clustering on features
- [ ] Representative MRI slices: pick 1 patient per cluster closest to centroid, show axial T1GD + segmentation overlay

#### 7C: Paper-Ready Outputs

- [ ] Table 1: Patient demographics by cluster
- [ ] Table 2: Cox regression results (HRs, p-values, C-index)
- [ ] Figure 1: Pipeline overview diagram
- [ ] Figure 2: Consensus matrix heatmap + optimal k selection
- [ ] Figure 3: KM survival curves (discovery + validation)
- [ ] Figure 4: UMAP + SHAP summary
- [ ] Figure 5: Cluster feature profiles (radar/heatmap)
- [ ] Supplementary: cluster stability, full feature list, sensitivity analyses

---

## Summary of Methods Used (for paper's Methods section)

| Category | Methods |
|----------|---------|
| Feature Extraction | PyRadiomics (shape, first-order, GLCM, GLRLM), hand-crafted ratios |
| Atlas Registration | ANTsPy affine to MNI152, Harvard-Oxford atlas lobe mapping |
| Missing Data | MICE imputation for continuous, separate "unknown" category for molecular |
| Feature Selection | Variance threshold + correlation filter + mRMR |
| Clustering | Consensus of Spectral + GMM + Hierarchical + K-Medoids |
| Cluster Validation | Bootstrap stability (ARI), silhouette, gap statistic |
| Survival Analysis | Kaplan-Meier, log-rank, multivariate Cox PH, C-index |
| Explainability | SHAP (TreeExplainer on Random Forest cluster predictor) |
| Visualization | UMAP, radar plots, heatmaps |
| External Validation | Train on UCSF → validate on UPenn (classifier-based assignment) |
| Optional Deep Learning | Tabular VAE for learned embeddings, DeepSurv for survival |

---

## What Makes This Stand Out for Publication

1. **Sample size:** 1,172 patients across 2 institutions — larger than most published GBM clustering studies
2. **External validation:** Genuine cross-institutional validation, not just train/test split
3. **Consensus clustering:** Not relying on a single algorithm's quirks
4. **Clinical independence:** Cox PH proves clusters add value beyond known biomarkers
5. **Reproducibility:** Both datasets are publicly available (TCIA)
6. **Explainability:** SHAP gives reviewers biologically interpretable cluster definitions

---

## Python Dependencies (Complete)

```txt
# Core
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
scipy>=1.11

# Medical imaging
nibabel>=5.0
nilearn>=0.10
antspyx>=0.4

# Radiomics
pyradiomics>=3.1

# Survival analysis
lifelines>=0.27

# Visualization
matplotlib>=3.7
seaborn>=0.12
umap-learn>=0.5

# Explainability
shap>=0.42

# Clustering extras
scikit-learn-extra>=0.3  # for KMedoids

# Deep learning (optional, for VAE/DeepSurv)
torch>=2.1

# Utilities
joblib>=1.3
tqdm>=4.65
```

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| ANTsPy registration fails for some patients | Fall back to nilearn's `resample_to_img` with MNI template; accept ~5% exclusion rate |
| PyRadiomics crashes on malformed masks | Wrap in try/except, log failures, require >80% success rate |
| Clusters don't separate survival | Try different k values, try clustering on imaging-only vs. clinical-only vs. combined; negative results are still publishable if well-analyzed |
| UPenn MGMT mostly missing | Run primary analysis without MGMT, then sensitivity analysis including MGMT on UCSF-only subset |
| 6 GB VRAM insufficient for VAE | VAE on 60 features × 500 samples fits in <100 MB; this is a non-issue. If needed, train on CPU |
| Registration takes too long | Use affine-only (skip deformable), parallelize 8 cores, run overnight |
