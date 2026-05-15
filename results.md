# GBM Survival Risk Stratification - Results (Run 2)

Date: 15 May 2026

## 1. Data Summary

- Input: outputs/features_processed.csv
- Samples: 367
- Positives (risk_label=1): 165
- Negatives (risk_label=0): 202
- Feature set: 16 radiomic features

## 2. Holdout Performance (Train/Val/Test)

- Validation balanced accuracy: 0.4800
- Validation macro F1: 0.4720
- Test balanced accuracy: 0.4781
- Test macro F1: 0.4780
- Test AUC: 0.5884

## 3. Final Cross-Validation Performance

- CV: Stratified 10-fold
- Mean AUC: 0.6033
- 95% CI (t-interval): [0.5369, 0.6698]
- One-sided p-value vs AUC=0.5: 0.00327774

## 4. Interpretation vs Abstract Target

The observed mean AUC (0.6033) is closer to the abstract target of
~0.652 and is statistically significant above chance at the 0.05 level.
The gap suggests remaining factors (registration quality or cohort
differences) may still limit performance.

## 5. Artifacts Produced

- outputs/cv_results.json
- outputs/training_report.txt
- outputs/final_model.json
- outputs/feature_list.json
- outputs/figures/shap_summary.png
- outputs/figures/shap_importance.png
- outputs/figures/confusion_matrix.png
- outputs/metrics/final_cv_summary.json
- outputs/metrics/final_cv_fold_metrics.csv
- outputs/metrics/test_metrics.json
- outputs/metrics/test_classification_report.csv
- outputs/metrics/validation_xgboost_metrics.json
- outputs/metrics/baseline_results.csv
- outputs/reports/training_summary.md

## 6. Recommended Next Checks

- Re-run with ANTs registration on Linux/WSL to reduce atlas misalignment
- Review lobe-assignment reliability threshold impact on sample size
- Compare SHAP top features with the abstract (frontal_en_ratio, tumor_burden_index, frontal_ed_ratio)
- Inspect fold-level AUC variance for stability
