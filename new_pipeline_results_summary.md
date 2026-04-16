# New Pipeline Results Summary (UCSF): Old vs New

## Scope
This summary compares UCSF results from:
- New pipeline run: Step4b -> Step5 -> Step6 -> Step7 -> Step5b -> Step8 outputs in `outputs/`

## Data Sources
- Old baseline reference: `ppt.tex`
- New VIF metadata: `outputs/step4b/ucsf/ucsf_step4b_metadata.json`
- New Step 5 selection: `outputs/step5/ucsf/ucsf_step5_selection.json`
- New Step 6 metadata: `outputs/step6/ucsf/ucsf_step6_metadata.json`
- New Step 6 multivariate Cox table: `outputs/step6/ucsf/ucsf_step6_cox_multivariate.csv`
- New Step 7 metadata: `outputs/step7/ucsf/ucsf_step7_metadata.json`
- New Step 5b consensus summary: `outputs/step5b/ucsf/ucsf_step5b_summary.json`

## Headline
The new pipeline introduces VIF filtering, Cox hazard-ratio high-risk assignment, multivariate Cox adjustment, and consensus clustering. The UCSF test-set survival signal remains statistically significant, while high-risk labeling is now data-driven and much more specific.

## New Terms Explained (Why They Matter)
1. VIF filtering (Variance Inflation Factor)
- VIF checks if features are repeating the same information.
- High VIF means too much overlap between features.
- We remove high-VIF features to make the model cleaner and more stable.

2. Cox hazard-ratio high-risk assignment
- Hazard ratio (HR) tells how risky a cluster is for early death.
- HR above 1 means higher risk; below 1 means lower risk.
- We now pick the high-risk cluster by highest HR, not by manual rules.

3. Multivariate Cox adjustment
- We test cluster effect together with age, MGMT, and IDH in one model.
- This checks if clusters still predict survival after accounting for these factors.
- If yes, clusters add independent clinical value.

4. Consensus clustering
- We repeat clustering many times on resampled data.
- If similar groups keep appearing, results are more reliable.
- This helps confirm whether the chosen k is stable.

5. "Survival signal remains statistically significant"
- Survival curves for clusters are still clearly different.
- The p-value is still below the significance cutoff, so this is unlikely due to chance.
- So the result still holds on test data, even after stricter pipeline updates.

## How the New Pipeline Was Implemented (Short + Simple)
1. Start with cleaner inputs
- We first reduced duplicate or overlapping features using VIF filtering.
- This keeps the model focused on the most useful signals instead of repeated information.

2. Find patient groups on training data
- We clustered only the training set and tested multiple values of k.
- We selected the k that gave better cluster quality and biological meaning.

3. Define high-risk group from survival risk (not by manual rule)
- Instead of hand-labeling, we used Cox hazard ratios.
- The cluster with the highest hazard ratio was marked as high-risk.

4. Check if cluster effect is independent
- We ran multivariate Cox with age, MGMT, IDH, and cluster.
- This tests if cluster still matters after accounting for known clinical factors.

5. Validate on unseen test patients
- Test patients were assigned to nearest train centroids.
- We then checked whether survival separation and phenotype patterns still held.

6. Add robustness checks
- Consensus clustering was used to test stability across resampling.
- This tells us whether the discovered structure is reproducible, not a one-off split artifact.

## Side-by-Side Comparison (UCSF)
| Metric | Old Baseline | New Pipeline | Change / Note |
|---|---:|---:|---|
| Feature count into clustering |  | 35 after VIF | Reduced multicollinearity (threshold = 5.0) |
| Selected k (Step 5) | 2 | 5 | More granular subtype structure |
| Train log-rank p-value | 4.79e-11 | 1.1297e-09 | Still highly significant |
| Test log-rank p-value | 1.70e-06 | 9.4013e-05 | Weaker than old split but still strongly significant |
| High-risk cluster definition | Rule-based composite + temporal bonus | Cox highest hazard ratio | Methodological upgrade |
| High-risk cluster label | | 3 | Labeling scheme changed after re-clustering |
| High-risk hazard ratio | Not reported | 9.896 | New effect-size output |
| Cluster independent predictor (multivariate Cox) | Not tested | Yes (`True`) | Cluster remains prognostic after age/MGMT/IDH |
| Min p among cluster dummies in multivariate Cox | Not available | 0.00730 | Significant adjusted cluster effect |
| High-risk train proportion | 83% | 13.43% | High-risk group is now far more specific |
| High-risk test proportion | 76% | 19.21% | Maintains minority high-risk phenotype |
| Train-test proportion drift | 7.00 pp | 5.78 pp | Better proportion stability |
| High-risk dominant lobe (train/test) | Frontal / Frontal | Frontal / Frontal | Preserved |
| High-risk NC/EN rank on test | #1 of 2 | 1 | Preserved |
| Consensus recommended k (Step 5b) | Not part of old workflow | 6 | Independent robustness evidence |
| Consensus vs Step5 ARI at k=5 | Not available | 0.6437 | Moderate agreement under resampling |

## What Improved Scientifically
1. We now remove overlapping features in a clear, trackable way.
- The VIF report shows exactly what was removed and why, so the process is transparent.

2. High-risk labeling is now based on measured survival risk, not manual rules.
- We use hazard ratio from the Cox model, so the high-risk group is chosen by actual outcome risk.

3. We check if clusters still matter after accounting for major clinical factors.
- With multivariate Cox, we adjust for age, MGMT, and IDH to see whether cluster effect is truly independent.

4. We added a stability check so cluster choice is more trustworthy.
- Consensus clustering tests whether similar groups appear again and again under resampling, which reduces chance findings.

## Interpretation from old pipeline
1. The old model produced very strong significance with a very large high-risk cluster share; the new model keeps significance while producing a much smaller, biologically sharper high-risk subgroup.
2. The increase in selected k (2 -> 5) and the reduced feature set (58 -> 35) suggest the new pipeline captures more refined structure with less redundancy.
3. Test-set p-value is higher than old but still strongly significant; this is consistent with a stricter, less over-aggregated clustering configuration.
4. New multivariate Cox evidence supports cluster label as an independent prognostic factor.

## Caveats
1. This comparison is single-cohort (UCSF) and tied to the latest random split (`random_state=42`).
2. Old values were taken from `ppt.tex` slide metrics, not recomputed in this run.
3. Step 5 and Step 5b disagree on best k (5 vs 6), which should be discussed as sensitivity rather than treated as contradiction.

