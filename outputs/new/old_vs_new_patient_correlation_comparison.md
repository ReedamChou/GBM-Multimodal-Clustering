# Old Vs New Patient Correlation Comparison

This report compares the new patient-wise Pearson correlation matrices in `outputs/new` against the corresponding legacy matrices in `outputs/old/step5b/ucsf/patient_pearson`.

## Compared Files

- New high risk: `outputs/new/ucsf_cluster_0_high_risk_patient_correlation_matrix.csv`
- Old high risk: `outputs/old/step5b/ucsf/patient_pearson/ucsf_cluster_0_high_risk_lt_6m_patient_correlation_matrix.csv`
- New medium risk: `outputs/new/ucsf_cluster_1_medium_risk_patient_correlation_matrix.csv`
- Old medium risk: `outputs/old/step5b/ucsf/patient_pearson/ucsf_cluster_1_mid_risk_6_to_18m_patient_correlation_matrix.csv`
- New low risk: `outputs/new/ucsf_cluster_2_low_risk_patient_correlation_matrix.csv`
- Old low risk: `outputs/old/step5b/ucsf/patient_pearson/ucsf_cluster_2_low_risk_gt_18m_patient_correlation_matrix.csv`

## Comparison Method

1. Match each new matrix to the corresponding old matrix by risk cluster.
2. Confirm the patient sets are identical before comparing values.
3. Compare off-diagonal patient-patient Pearson correlations only.
4. Summarize mean, median, and proportions of stronger or weaker pairwise correlations.

## Key Finding

The new matrices are overall weaker than the old matrices across all three clusters.

- Overall new mean correlation: `0.656`
- Overall old mean correlation: `0.744`
- Overall mean change: `-0.088`
- Overall new median correlation: `0.678`
- Overall old median correlation: `0.839`
- Overall median change: `-0.055`
- Fraction of patient pairs where new > old: `34.2%`
- Fraction of patient pairs where new < old: `65.8%`

## Cluster Summary

| Cluster | Patients | New Mean | Old Mean | Mean Change | New Median | Old Median | Median Change | New > 0.8 | Old > 0.8 | New < 0.5 | Old < 0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| High Risk | 115 | 0.678 | 0.708 | -0.030 | 0.742 | 0.804 | -0.006 | 46.3% | 50.3% | 24.9% | 20.6% |
| Medium Risk | 183 | 0.630 | 0.711 | -0.081 | 0.637 | 0.798 | -0.040 | 36.7% | 49.6% | 30.1% | 18.8% |
| Low Risk | 202 | 0.670 | 0.783 | -0.112 | 0.688 | 0.864 | -0.106 | 36.4% | 65.7% | 22.4% | 12.6% |

## Cluster-Wise Notes

### High Risk

- Patient count is unchanged: `115`.
- The new matrix is slightly weaker overall than the old one.
- Mean correlation decreased by `0.030`.
- Median correlation decreased by `0.006`.
- Very strong correlations above `0.8` dropped from `50.3%` to `46.3%`.
- `52.9%` of patient-pair correlations are lower in the new matrix.

### Medium Risk

- Patient count is unchanged: `183`.
- The new matrix is clearly weaker than the old one.
- Mean correlation decreased by `0.081`.
- Median correlation decreased by `0.040`.
- Very strong correlations above `0.8` dropped from `49.6%` to `36.7%`.
- Low-to-moderate correlations below `0.5` increased from `18.8%` to `30.1%`.
- `62.3%` of patient-pair correlations are lower in the new matrix.

### Low Risk

- Patient count is unchanged: `202`.
- This cluster shows the largest drop in correlation strength.
- Mean correlation decreased by `0.112`.
- Median correlation decreased by `0.106`.
- Very strong correlations above `0.8` dropped from `65.7%` to `36.4%`.
- Correlations below `0.5` increased from `12.6%` to `22.4%`.
- `72.9%` of patient-pair correlations are lower in the new matrix.

## Interpretation

Because the patient membership is identical in old and new matrices for every cluster, the difference comes from the feature-processing pipeline rather than from a cluster assignment mismatch.

The new workflow likely reduced correlation inflation by:

- imputing missing values in a more structured way,
- one-hot encoding the categorical imaging feature `dominant_brain_lobe`,
- scaling processed imaging features to `[0, 1]`, and
- dropping highly correlated imaging features above the Pearson threshold of `0.90`.

In the current new pipeline, the following highly correlated features were removed before patient-wise correlation was computed:

- `temporal_en_ratio`
- `parietal_en_ratio`
- `occipital_en_ratio`

This makes the new patient-patient similarity matrices more conservative, with fewer extremely high pairwise correlations.
