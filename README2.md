# GBM Multi-Modal Clustering: Integrating Radiomics and Clinical Features for Molecular Subtype Discovery

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

This project integrates **radiomic features** extracted from multi-parametric MRI (T1, T1-Gd, T2, FLAIR) with **clinical and molecular markers** to discover reproducible GBM subtypes via **consensus clustering**.  Discovered subtypes are validated for **prognostic significance** (Kaplan-Meier, Cox PH) and **external generalisability** across two independent cohorts.

### Key Contributions
1. **Multi-modal feature fusion** — 1000+ PyRadiomics features (per modality × tumour sub-region) combined with harmonised clinical variables.
2. **Consensus clustering** — Ensemble of Spectral, GMM, Agglomerative, and K-Medoids algorithms with subsampled co-association consensus.
3. **Two-cohort external validation** — Discovery on UCSF-PDGM (n ≈ 500), independent validation on UPENN-GBM (n ≈ 600) via kNN label transfer.
4. **Explainability** — SHAP-based feature importance and clinical enrichment analysis per cluster.

---

## Datasets

| Cohort | Source | Patients | Modalities |
|--------|--------|----------|------------|
| [UCSF-PDGM](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=119705830) | TCIA | ~500 | T1, T1-Gd, T2, FLAIR + segmentation |
| [UPENN-GBM](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70225642) | TCIA | ~600 | T1, T1-Gd, T2, FLAIR + segmentation |

Place the raw data under `Dataset/` as described in the **Data Layout** section below.

---

## Project Structure

```
Project Reedam/
├── config/
│   └── config.yaml                 # All hyperparameters & paths
├── src/
│   ├── preprocessing/
│   │   ├── verify_data.py          # 1. Dataset scan & manifest
│   │   ├── register_atlas.py       # 2. ANTsPy MNI registration
│   │   └── extract_radiomics.py    # 3. PyRadiomics extraction
│   ├── features/
│   │   ├── clinical_harmonization.py  # 4. Standardise clinical CSVs
│   │   └── feature_engineering.py     # 5. Merge, impute, PCA
│   ├── clustering/
│   │   └── consensus_clustering.py # 6. Consensus clustering & k selection
│   ├── validation/
│   │   ├── survival_analysis.py    # 7. KM, log-rank, Cox PH
│   │   ├── cluster_characterization.py # 8. SHAP & enrichment
│   │   └── external_validation.py  # 9. UCSF→UPenn transfer
│   ├── visualization/
│   │   └── plots.py                # 10. Publication figures
│   └── utils/
│       └── helpers.py              # Config, logging, seeds
├── scripts/
│   └── run_pipeline.py             # Master orchestrator (steps 1–10)
├── data/                           # Generated intermediate files
├── results/
│   ├── figures/                    # PDF figures
│   └── tables/                     # CSV results tables
├── Dataset/                        # Raw data (not tracked in git)
├── requirements.txt
├── ROADMAP.md
└── README.md                       # ← you are here
```

---

## Installation

```bash
# Create environment
conda create -n gbm python=3.10 -y
conda activate gbm

# Install dependencies
pip install -r requirements.txt
```

> **Note on ANTsPy**: If `pip install antspyx` fails, install from conda-forge:
> ```bash
> conda install -c aramislab antspyx
> ```

---

## Data Layout

```
Dataset/
├── UCSF-PDGM-metadata_v5.csv
├── UPENN-GBM_clinical_info_v2.1.csv
├── UCSF/
│   ├── DATA-IMAGE-STRUCTURAL/
│   │   └── UCSF-PDGM-XXXX/
│   │       ├── UCSF-PDGM-XXXX_T1.nii.gz
│   │       ├── UCSF-PDGM-XXXX_T1GD.nii.gz
│   │       ├── UCSF-PDGM-XXXX_T2.nii.gz
│   │       └── UCSF-PDGM-XXXX_FLAIR.nii.gz
│   └── DATA-AUTOMATED-SEGMENT/
│       └── UCSF-PDGM-XXXX_tumor_segmentation.nii.gz
└── UPenn/
    ├── DATA-IMAGE-STRUCTURAL/
    │   └── UPENN-GBM-XXXXX_11/
    │       ├── UPENN-GBM-XXXXX_11_T1.nii.gz
    │       ├── UPENN-GBM-XXXXX_11_T1GD.nii.gz
    │       ├── UPENN-GBM-XXXXX_11_T2.nii.gz
    │       └── UPENN-GBM-XXXXX_11_FLAIR.nii.gz
    └── DATA-AUTOMATED-SEGMENT/
        └── UPENN-GBM-XXXXX_11_automated_approx_segm.nii.gz
```

---

## Usage

### Full pipeline (all 10 steps)
```bash
python -m scripts.run_pipeline
```

### Run specific steps
```bash
# Data verification only
python -m scripts.run_pipeline --step 1

# Steps 3 through 6 (radiomics → clustering)
python -m scripts.run_pipeline --step 3-6

# Skip atlas registration (if already cached)
python -m scripts.run_pipeline --skip-registration
```

### Step Reference

| Step | Module | Description | Approx. Time* |
|------|--------|-------------|----------------|
| 1 | `verify_data` | Scan dataset, create manifest | < 1 min |
| 2 | `register_atlas` | ANTsPy T1→MNI + seg warp | ~2 h (all patients) |
| 3 | `extract_radiomics` | PyRadiomics per modality × region | ~4 h (4 threads) |
| 4 | `clinical_harmonization` | Standardise / merge CSVs | < 1 min |
| 5 | `feature_engineering` | Impute, filter, PCA | < 1 min |
| 6 | `consensus_clustering` | Sweep k, build consensus | ~10 min |
| 7 | `survival_analysis` | KM, log-rank, Cox PH | < 1 min |
| 8 | `cluster_characterization` | SHAP, enrichment tests | ~2 min |
| 9 | `external_validation` | kNN label transfer to UPenn | < 1 min |
| 10 | `plots` | Generate all publication figures | < 1 min |

\* Estimates for Ryzen 7 / RTX 4050 6 GB.

---

## Outputs

### Tables (`results/tables/`)
- `logrank_pairwise.csv` — pairwise log-rank test results
- `cox_summary.csv` — Cox PH hazard ratios and p-values
- `cluster_summary.csv` — per-cluster clinical profile
- `enrichment_tests.csv` — statistical association tests
- `shap_importance.csv` — top SHAP features
- `external_agreement.csv` — ARI / NMI between cohorts
- `logrank_validation.csv` — validation cohort survival tests

### Figures (`results/figures/`)
- Consensus heatmap, clustering metrics (silhouette / PAC)
- UMAP embedding coloured by cluster and cohort
- Kaplan-Meier curves (discovery + validation)
- PCA scree plot, Cox forest plot
- SHAP beeswarm, clinical enrichment heatmap

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 4-core | 8+ cores (registration is CPU-bound) |
| RAM | 16 GB | 32 GB |
| GPU VRAM | — | 6 GB (optional — VAE/DeepSurv) |
| Disk | 50 GB | 100 GB (raw NIfTIs) |

---

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{gbm_multimodal_clustering,
  title  = {GBM Multi-Modal Clustering Pipeline},
  year   = {2025},
  url    = {https://github.com/your-username/project-reedam}
}
```

---

## License

MIT
