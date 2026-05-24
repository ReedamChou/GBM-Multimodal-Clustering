# Multimodal Fusion (Merge + Train)

This workflow fuses the four modality-specific processed feature tables into a
single multimodal dataset and trains one SVM model on the merged 64-feature
matrix.

---

## Step 1: Merge Per-Modality Features

`merge_modalities.py` loads the four processed CSVs, prefixes feature columns by
modality, and performs an **inner join on `patient_id`** so only patients present
in all modalities are retained. It also checks that `risk_label` is consistent
across modalities and keeps a single copy in the output.

**Inputs**
- `outputs/features_processed_t1.csv`
- `outputs/features_processed_t2.csv`
- `outputs/features_processed_t1gd.csv`
- `outputs/features_processed_flair.csv`

**Output**
- `outputs/features_multimodal.csv`

---

## Step 2: Train the Multimodal SVM

`train_multimodal.py` reuses the existing SVM pipeline (scaling, Optuna tuning,
plots, metrics, SHAP, and reports) to train on the 64-feature merged dataset.

**Input**
- `outputs/features_multimodal.csv`

**Output Directory (default)**
- `outputs/multimodal/`

---

## CLI Usage

### PowerShell (run one after another)
```powershell
python src/merge_modalities.py
python src/train_multimodal.py `
  --input outputs/features_multimodal.csv `
  --output outputs/multimodal
```

### Bash (run one after another)
```bash
python src/merge_modalities.py
python src/train_multimodal.py \
  --input outputs/features_multimodal.csv \
  --output outputs/multimodal
```

---

## Optional Overrides

You can override any input/output paths if needed:

```powershell
python src/merge_modalities.py `
  --input-t1 outputs/features_processed_t1.csv `
  --input-t2 outputs/features_processed_t2.csv `
  --input-t1gd outputs/features_processed_t1gd.csv `
  --input-flair outputs/features_processed_flair.csv `
  --output outputs/features_multimodal.csv
```

```powershell
python src/train_multimodal.py `
  --input outputs/features_multimodal.csv `
  --output outputs/multimodal
```
