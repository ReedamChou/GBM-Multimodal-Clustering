# Brain Tumor Risk Classification Guide
## UCSF-PDGM Dataset — Imaging Features Only
> **Task:** Ordinal 3-class classification → `low_risk` | `medium_risk` | `high_risk`  
> **Samples:** 500 | **Features:** 17 imaging features  
> **Class Distribution:** low_risk=202, medium_risk=183, high_risk=115

---

## Table of Contents
1. [Environment Setup](#1-environment-setup)
2. [Data Loading & Sanity Checks](#2-data-loading--sanity-checks)
3. [Exploratory Data Analysis (EDA)](#3-exploratory-data-analysis-eda)
4. [Preprocessing Pipeline](#4-preprocessing-pipeline)
5. [Train / Validation / Test Split](#5-train--validation--test-split)
6. [Baseline Models](#6-baseline-models)
7. [Primary Model — XGBoost (Ordinal)](#7-primary-model--xgboost-ordinal)
8. [Hyperparameter Tuning — Optuna](#8-hyperparameter-tuning--optuna)
9. [Handling Class Imbalance](#9-handling-class-imbalance)
10. [Model Evaluation](#10-model-evaluation)
11. [Feature Importance & Explainability](#11-feature-importance--explainability)
12. [Cross-Validation — Final Estimate](#12-cross-validation--final-estimate)
13. [Saving & Loading the Best Model](#13-saving--loading-the-best-model)
14. [Pitfalls & Best Practices Checklist](#14-pitfalls--best-practices-checklist)

---

## 1. Environment Setup

Install all required packages in one shot:

```bash
pip install pandas numpy scikit-learn xgboost lightgbm imbalanced-learn \
            optuna shap matplotlib seaborn joblib
```

**Python version:** 3.9+ recommended.

---

## 2. Data Loading & Sanity Checks

```python
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv('step3_ucsf_preprocessed_features.csv')

# ── Drop the 1 row with missing label ─────────────────────────────────────────
df = df.dropna(subset=['risk_cluster_label'])
print(f"Dataset shape after cleaning: {df.shape}")   # (500, 23)

# ── Define feature columns (imaging only — exclude survival columns) ──────────
IMAGING_FEATURES = [
    'global_nc_en_ratio', 'global_ed_en_ratio', 'global_ed_total_ratio',
    'tumor_burden_index',
    'frontal_ed_ratio',   'frontal_en_ratio',   'frontal_nc_ratio',
    'temporal_ed_ratio',  'temporal_nc_ratio',
    'parietal_ed_ratio',  'parietal_nc_ratio',
    'occipital_ed_ratio', 'occipital_nc_ratio',
    'dominant_brain_lobe_frontal', 'dominant_brain_lobe_occipital',
    'dominant_brain_lobe_parietal', 'dominant_brain_lobe_temporal'
]
TARGET = 'risk_cluster_label'

X = df[IMAGING_FEATURES].copy()
y = df[TARGET].copy()

# ── Sanity checks ─────────────────────────────────────────────────────────────
print("\nClass distribution:")
print(y.value_counts())

print("\nNull values in features:")
print(X.isnull().sum().sum())   # Should be 0

print("\nFeature value range (all should be 0–1):")
print(X.describe().T[['min', 'max', 'mean', 'std']])
```

**Expected output:**
```
low_risk      202
medium_risk   183
high_risk     115
```

> ⚠️ **Never include `OS`, `survival_months`, or `risk_cluster_id` in X.**
> They are directly derived from the label — including them is data leakage.

---

## 3. Exploratory Data Analysis (EDA)

```python
import matplotlib.pyplot as plt
import seaborn as sns

# ── Ordinal label encoding (needed for all plots and models) ──────────────────
LABEL_MAP    = {'low_risk': 0, 'medium_risk': 1, 'high_risk': 2}
LABEL_NAMES  = ['low_risk', 'medium_risk', 'high_risk']
y_ord = y.map(LABEL_MAP)

# ── 3.1 Class distribution bar ────────────────────────────────────────────────
y.value_counts().reindex(LABEL_NAMES).plot(kind='bar', color=['green','orange','red'])
plt.title('Class Distribution')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('class_distribution.png', dpi=150)
plt.show()

# ── 3.2 Feature distributions per class ──────────────────────────────────────
fig, axes = plt.subplots(5, 4, figsize=(18, 16))
axes = axes.flatten()
for i, col in enumerate(IMAGING_FEATURES):
    for label in LABEL_NAMES:
        axes[i].hist(X.loc[y == label, col], bins=20, alpha=0.5, label=label)
    axes[i].set_title(col, fontsize=8)
    axes[i].legend(fontsize=6)
for j in range(len(IMAGING_FEATURES), len(axes)):
    axes[j].set_visible(False)
plt.tight_layout()
plt.savefig('feature_distributions.png', dpi=150)
plt.show()

# ── 3.3 Correlation heatmap ───────────────────────────────────────────────────
plt.figure(figsize=(12, 10))
sns.heatmap(X.corr(), annot=False, cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=150)
plt.show()

# ── 3.4 Identify highly correlated features (flag, don't drop yet) ────────────
corr_matrix = X.corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr_pairs = [(col, row, upper_tri.loc[row, col])
                   for col in upper_tri.columns
                   for row in upper_tri.index
                   if upper_tri.loc[row, col] > 0.9]
print("Highly correlated pairs (>0.9):", high_corr_pairs)
```

**What to look for:**
- Features with near-zero variance across all classes → candidates for removal
- Highly correlated pairs (>0.9) → keep only one from each pair
- Bimodal distributions aligned with class boundaries → strong features

---

## 4. Preprocessing Pipeline

All preprocessing is wrapped in a `sklearn Pipeline` to prevent data leakage
(transformations are fit **only** on training data, then applied to test data).

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

# ── Why RobustScaler? ─────────────────────────────────────────────────────────
# Several ratio features (global_nc_en_ratio, global_ed_en_ratio) have extreme
# outliers near 0 with long right tails. RobustScaler uses median/IQR and is
# more resistant to these outliers than StandardScaler.

# Note: Tree-based models (XGBoost, LightGBM, RF) do NOT need scaling.
#       Scaling is only needed for SVM, Logistic Regression, and KNN baselines.
```

```python
# ── 4.1 Remove zero-variance features ────────────────────────────────────────
from sklearn.feature_selection import VarianceThreshold

var_thresh = VarianceThreshold(threshold=0.0)
var_thresh.fit(X)
zero_var_cols = [col for col, kept in zip(IMAGING_FEATURES, var_thresh.get_support()) if not kept]
print("Zero-variance features removed:", zero_var_cols)

# ── 4.2 Pipelines for scale-sensitive vs. tree-based models ──────────────────
from sklearn.preprocessing import RobustScaler

# For Logistic Regression / SVM / KNN
scale_pipe = Pipeline([
    ('var_filter', VarianceThreshold(threshold=0.0)),
    ('scaler',     RobustScaler()),
])

# For XGBoost / LightGBM / Random Forest (no scaling needed)
tree_pipe = Pipeline([
    ('var_filter', VarianceThreshold(threshold=0.0)),
])
```

---

## 5. Train / Validation / Test Split

```python
from sklearn.model_selection import train_test_split

# ── Encode labels ordinally ───────────────────────────────────────────────────
y_ord = y.map(LABEL_MAP)   # low=0, medium=1, high=2

# ── Stratified split: 70% train | 15% val | 15% test ─────────────────────────
# Stratify ensures each split has proportional class representation.
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y_ord, test_size=0.15, random_state=42, stratify=y_ord
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.176,   # 0.176 × 0.85 ≈ 0.15 of total
    random_state=42, stratify=y_trainval
)

print(f"Train:      {X_train.shape[0]} samples")
print(f"Validation: {X_val.shape[0]} samples")
print(f"Test:       {X_test.shape[0]} samples")

# ── Verify class proportions are preserved ────────────────────────────────────
for split_name, split_y in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
    counts = split_y.value_counts(normalize=True).sort_index()
    print(f"{split_name}: {dict(counts.round(2))}")
```

> ⚠️ **Rule:** Fit all preprocessors (`RobustScaler`, `VarianceThreshold`) on
> `X_train` only. Never peek at `X_val` or `X_test` during fitting.

---

## 6. Baseline Models

Always establish baselines before tuning complex models. They anchor your
expectations and reveal whether fancy models are actually adding value.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import VarianceThreshold

# ── Fit scalers on train only ─────────────────────────────────────────────────
vt  = VarianceThreshold(threshold=0.0).fit(X_train)
scl = RobustScaler().fit(vt.transform(X_train))

X_tr_sc  = scl.transform(vt.transform(X_train))
X_val_sc = scl.transform(vt.transform(X_val))

# ── Baseline model grid ───────────────────────────────────────────────────────
baselines = {
    'LogisticRegression':   LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    'RandomForest':         RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42),
    'GradientBoosting':     GradientBoostingClassifier(n_estimators=200, random_state=42),
    'SVM (RBF)':            SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42),
    'KNN':                  KNeighborsClassifier(n_neighbors=7),
}

print(f"{'Model':<25} {'Bal. Accuracy':>14} {'Macro F1':>10}")
print('-' * 52)

results = {}
for name, model in baselines.items():
    # Scale-sensitive models use scaled data; trees use raw
    if name in ['LogisticRegression', 'SVM (RBF)', 'KNN']:
        model.fit(X_tr_sc, y_train)
        preds = model.predict(X_val_sc)
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_val)

    from sklearn.metrics import f1_score
    bal_acc = balanced_accuracy_score(y_val, preds)
    macro_f1 = f1_score(y_val, preds, average='macro')
    results[name] = {'bal_acc': bal_acc, 'macro_f1': macro_f1}
    print(f"{name:<25} {bal_acc:>14.4f} {macro_f1:>10.4f}")
```

**Use `balanced_accuracy_score` as primary metric** (not plain accuracy) because
your classes are imbalanced (202 / 183 / 115). Plain accuracy rewards predicting
the majority class.

---

## 7. Primary Model — XGBoost (Ordinal)

XGBoost with ordinal encoding is the recommended primary model because:
- Handles mixed feature types (continuous ratios + binary lobe flags) natively
- Robust to outliers (your ratio features have extreme values near 0)
- Naturally handles class imbalance via `scale_pos_weight` / `sample_weight`
- Respects ordinal label order when used with `reg:squarederror` or `multi:softmax`

```python
import xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight

# ── Compute sample weights to handle class imbalance ─────────────────────────
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

# ── XGBoost with multi:softmax (standard multiclass) ─────────────────────────
xgb_model = xgb.XGBClassifier(
    objective        = 'multi:softmax',
    num_class        = 3,
    n_estimators     = 500,
    learning_rate    = 0.05,
    max_depth        = 4,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    min_child_weight = 5,
    gamma            = 0.1,
    reg_alpha        = 0.1,      # L1 regularization
    reg_lambda       = 1.0,      # L2 regularization
    use_label_encoder= False,
    eval_metric      = 'mlogloss',
    early_stopping_rounds = 30,
    random_state     = 42,
    n_jobs           = -1,
)

xgb_model.fit(
    X_train, y_train,
    sample_weight       = sample_weights,
    eval_set            = [(X_val, y_val)],
    verbose             = 50,
)

preds_xgb = xgb_model.predict(X_val)
print("\nXGBoost Validation Results:")
print(classification_report(y_val, preds_xgb,
      target_names=LABEL_NAMES))
print("Balanced Accuracy:", balanced_accuracy_score(y_val, preds_xgb))
```

---

## 8. Hyperparameter Tuning — Optuna

Optuna uses **Bayesian optimization** (Tree-structured Parzen Estimator) which
is far more efficient than GridSearchCV for this many hyperparameters.

```python
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    params = {
        'objective':         'multi:softmax',
        'num_class':         3,
        'n_estimators':      trial.suggest_int('n_estimators', 100, 800),
        'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth':         trial.suggest_int('max_depth', 3, 8),
        'subsample':         trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight':  trial.suggest_int('min_child_weight', 1, 20),
        'gamma':             trial.suggest_float('gamma', 0, 5),
        'reg_alpha':         trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        'reg_lambda':        trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        'use_label_encoder': False,
        'eval_metric':       'mlogloss',
        'random_state':      42,
        'n_jobs':            -1,
    }

    model = xgb.XGBClassifier(**params)

    # ── Use X_trainval (train+val) for CV during tuning ───────────────────────
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    sw = compute_sample_weight('balanced', y=y_trainval)

    scores = cross_val_score(
        model, X_trainval, y_trainval,
        cv=cv,
        scoring='balanced_accuracy',
        fit_params={'sample_weight': sw},
        n_jobs=-1,
    )
    return scores.mean()


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100, show_progress_bar=True)

print(f"\nBest balanced accuracy (CV): {study.best_value:.4f}")
print("Best params:", study.best_params)
```

```python
# ── Retrain with best params on train+val ─────────────────────────────────────
best_params = study.best_params
best_params.update({
    'objective':         'multi:softmax',
    'num_class':         3,
    'use_label_encoder': False,
    'eval_metric':       'mlogloss',
    'random_state':      42,
    'n_jobs':            -1,
})

best_xgb = xgb.XGBClassifier(**best_params)
sw_trainval = compute_sample_weight('balanced', y=y_trainval)
best_xgb.fit(X_trainval, y_trainval, sample_weight=sw_trainval)
```

---

## 9. Handling Class Imbalance

Your dataset has a noticeable imbalance: **low_risk=202, medium_risk=183, high_risk=115**.
Use a combination of the techniques below.

### 9.1 Sample Weights (already applied above)
Directly penalizes misclassifying minority class samples more — easiest and most
reliable for tree-based models.

```python
from sklearn.utils.class_weight import compute_sample_weight
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
```

### 9.2 SMOTE on Training Set Only

```python
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=42, k_neighbors=5)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

print("Before SMOTE:", pd.Series(y_train).value_counts().to_dict())
print("After SMOTE: ", pd.Series(y_train_sm).value_counts().to_dict())
```

> ⚠️ **Critical:** Apply SMOTE **only to the training set**, never to
> validation or test sets. Oversampling test data inflates performance metrics.

### 9.3 Which to Use

| Situation | Recommendation |
|-----------|---------------|
| Tree-based model (XGBoost, RF) | `sample_weight='balanced'` — simplest, no synthetic data |
| Linear model (LR, SVM) | SMOTE + `class_weight='balanced'` |
| Severe imbalance (>4:1 ratio) | SMOTE + sample weights combined |
| Your case (202:115 ≈ 1.75:1) | `sample_weight='balanced'` is sufficient |

---

## 10. Model Evaluation

### 10.1 Metrics to Report

```python
from sklearn.metrics import (
    classification_report, confusion_matrix,
    balanced_accuracy_score, cohen_kappa_score,
    roc_auc_score, ConfusionMatrixDisplay
)

# ── Evaluate on held-out TEST set (only touch this once!) ─────────────────────
y_test_pred  = best_xgb.predict(X_test)
y_test_proba = best_xgb.predict_proba(X_test)

print("=" * 60)
print("FINAL TEST SET EVALUATION")
print("=" * 60)
print(classification_report(y_test, y_test_pred, target_names=LABEL_NAMES))

bal_acc   = balanced_accuracy_score(y_test, y_test_pred)
kappa     = cohen_kappa_score(y_test, y_test_pred, weights='quadratic')  # ordinal-aware
auc_ovr   = roc_auc_score(y_test, y_test_proba, multi_class='ovr', average='macro')

print(f"Balanced Accuracy : {bal_acc:.4f}")
print(f"Quadratic Kappa   : {kappa:.4f}   ← ordinal-aware metric")
print(f"AUC (macro OvR)   : {auc_ovr:.4f}")
```

### 10.2 Confusion Matrix

```python
fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_test_pred,
    display_labels=LABEL_NAMES,
    cmap='Blues',
    ax=ax
)
plt.title('Confusion Matrix — Test Set')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()
```

### 10.3 Why Quadratic Weighted Kappa?

For ordinal classification, misclassifying `low_risk` as `high_risk` is **worse**
than misclassifying it as `medium_risk`. Quadratic Kappa penalizes proportional
to the **squared distance** between predicted and true ordinal rank — exactly
what you want clinically.

### 10.4 Metrics Summary Table

| Metric | What it measures | Use for |
|--------|-----------------|---------|
| Balanced Accuracy | Mean recall per class | Primary metric — handles imbalance |
| Macro F1 | Harmonic mean of per-class F1 | Secondary — equal class importance |
| Quadratic Kappa | Agreement adjusted for ordinal distance | Ordinal correctness |
| AUC (macro OvR) | Separability between classes | Probability calibration quality |

---

## 11. Feature Importance & Explainability

### 11.1 XGBoost Built-in Importance

```python
import matplotlib.pyplot as plt

# Get feature names after VarianceThreshold (all 17 survive in this dataset)
feat_names = IMAGING_FEATURES

importances = best_xgb.feature_importances_
feat_imp_df = pd.DataFrame({'feature': feat_names, 'importance': importances})
feat_imp_df = feat_imp_df.sort_values('importance', ascending=True)

plt.figure(figsize=(8, 7))
plt.barh(feat_imp_df['feature'], feat_imp_df['importance'], color='steelblue')
plt.xlabel('Importance (gain)')
plt.title('XGBoost Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
plt.show()
```

### 11.2 SHAP Values (Global + Local)

SHAP gives **directional** importance — not just which features matter, but
whether high values push toward low_risk or high_risk.

```python
import shap

explainer   = shap.TreeExplainer(best_xgb)
shap_values = explainer.shap_values(X_test)   # shape: (n_test, n_features, n_classes)

# ── Summary plot for class 2 (high_risk) ─────────────────────────────────────
shap.summary_plot(
    shap_values[:, :, 2],   # class index 2 = high_risk
    X_test,
    feature_names=feat_names,
    plot_type='bar',
    show=False
)
plt.title('SHAP — high_risk class')
plt.tight_layout()
plt.savefig('shap_high_risk.png', dpi=150)
plt.show()

# ── Beeswarm plot (shows direction of effect) ─────────────────────────────────
shap.summary_plot(shap_values[:, :, 2], X_test, feature_names=feat_names, show=False)
plt.tight_layout()
plt.savefig('shap_beeswarm_high_risk.png', dpi=150)
plt.show()
```

---

## 12. Cross-Validation — Final Estimate

After selecting the best model and hyperparameters, estimate final performance
using **Stratified K-Fold CV on the full dataset** for the most robust estimate.

```python
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer

# ── Custom scorers ─────────────────────────────────────────────────────────────
bal_acc_scorer = make_scorer(balanced_accuracy_score)
kappa_scorer   = make_scorer(cohen_kappa_score, weights='quadratic')

final_model = xgb.XGBClassifier(**best_params)
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

sw_full = compute_sample_weight('balanced', y=y_ord)

cv_results = cross_validate(
    final_model, X, y_ord,
    cv=cv,
    scoring={'bal_acc': bal_acc_scorer, 'kappa': kappa_scorer},
    fit_params={'sample_weight': sw_full},
    return_train_score=True,
    n_jobs=-1,
)

print("10-Fold Stratified CV Results:")
print(f"  Balanced Accuracy : {cv_results['test_bal_acc'].mean():.4f} ± {cv_results['test_bal_acc'].std():.4f}")
print(f"  Quadratic Kappa   : {cv_results['test_kappa'].mean():.4f} ± {cv_results['test_kappa'].std():.4f}")
print(f"  Train Bal. Acc    : {cv_results['train_bal_acc'].mean():.4f}  ← check for overfitting")
```

**Watch for overfitting:**
- If train accuracy >> test accuracy (gap > 0.10), add more regularization
- Increase `min_child_weight`, reduce `max_depth`, or lower `n_estimators`

---

## 13. Saving & Loading the Best Model

```python
import joblib

# ── Save model ────────────────────────────────────────────────────────────────
joblib.dump(best_xgb, 'best_xgb_risk_classifier.pkl')
print("Model saved.")

# ── Save label mapping for reproducibility ────────────────────────────────────
import json
with open('label_map.json', 'w') as f:
    json.dump({'map': LABEL_MAP, 'names': LABEL_NAMES}, f, indent=2)

# ── Load and predict on new data ──────────────────────────────────────────────
loaded_model = joblib.load('best_xgb_risk_classifier.pkl')

def predict_risk(new_data_df):
    """
    new_data_df: DataFrame with the same 17 IMAGING_FEATURES columns.
    Returns: list of string labels (e.g., ['low_risk', 'high_risk', ...])
    """
    preds_ord = loaded_model.predict(new_data_df[IMAGING_FEATURES])
    inv_map   = {v: k for k, v in LABEL_MAP.items()}
    return [inv_map[p] for p in preds_ord]

# Example usage:
# sample = X_test.iloc[:5]
# print(predict_risk(sample))
```

---

## 14. Pitfalls & Best Practices Checklist

Use this before finalizing any result:

```
DATA
 ✅ Dropped the 1 null label row
 ✅ Excluded OS, survival_months, risk_cluster_id from features (leakage!)
 ✅ All 17 imaging features are in [0, 1] range — no further normalization needed for trees
 ✅ Applied SMOTE / sample_weight only on training data

SPLITTING
 ✅ Used stratified splits to preserve class proportions
 ✅ Preprocessors (scaler, variance filter) fit only on training set
 ✅ Test set touched exactly once for final evaluation

MODELING
 ✅ Established baselines before jumping to complex models
 ✅ Used ordinal label encoding (0/1/2) — not one-hot encoding
 ✅ Used sample_weight='balanced' to handle imbalance
 ✅ Applied early stopping to prevent overfitting

EVALUATION
 ✅ Used balanced_accuracy (not plain accuracy) as primary metric
 ✅ Reported Quadratic Kappa (ordinal-aware)
 ✅ Reported per-class precision, recall, F1
 ✅ Checked train vs. test gap for overfitting

EXPLAINABILITY
 ✅ Generated SHAP values per class (not just global importance)
 ✅ Inspected confusion matrix — verify errors are between adjacent classes

REPRODUCIBILITY
 ✅ Set random_state=42 everywhere
 ✅ Saved model + label mapping as artifacts
```

---

## Quick Reference — Key Design Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Label encoding | Ordinal (0/1/2) | Classes have natural order |
| Primary model | XGBoost | Handles outlier-heavy ratios, no scaling needed |
| Imbalance strategy | sample_weight='balanced' | Sufficient for 1.75:1 ratio |
| Tuning strategy | Optuna (Bayesian) | More efficient than GridSearch |
| CV strategy | Stratified 10-Fold | Small dataset, preserves class proportions |
| Primary metric | Balanced Accuracy + Quadratic Kappa | Handles imbalance + ordinal errors |
| Scaler | RobustScaler | Handles extreme outliers in ratio features |

---

*Guide authored for UCSF-PDGM dataset — 500 patients, 17 imaging features.*
*All code blocks are self-contained and run sequentially.*
