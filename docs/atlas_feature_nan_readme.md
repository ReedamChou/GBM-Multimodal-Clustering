# Atlas Feature NaN README

This note explains why NaN values appear in the atlas-derived imaging features and clarifies that they mostly come from the feature extraction logic itself, not from the batch wrapper that writes the CSV.

## Main Cause

The NaNs come from `extract_sri24_atlas_features.py`, not from the batch script.

- `batch_extract_sri24_atlas_features.py` only loads each segmentation, optionally resamples it into atlas space, calls `compute_feature_row(...)`, and writes the returned values to CSV.
- The batch wrapper does not create these NaNs on its own.

The direct source is the helper `safe_div()`:

- `extract_sri24_atlas_features.py` line `134`
- `safe_div()` explicitly returns `NaN` whenever the denominator is `0`

## Feature-By-Feature Meaning

### `dominant_brain_lobe`

This field is not directly written as `NaN`.

- It is written as an empty string `""` when no tumor voxels were assigned to any of the four requested lobes.
- When the CSV is later loaded with pandas, that blank value may appear as `NaN`.
- Logic reference: `extract_sri24_atlas_features.py` line `364`

In the UCSF atlas CSV, this happened once:

- `UCSF-PDGM-0066`
- `tumor_voxels_in_requested_lobes = 0`
- `affine_out_of_bounds_voxels = 0`

That means the tumor was not assigned to frontal, temporal, parietal, or occipital. It does not mean the tumor went out of bounds.

### `global_nc_en_ratio`

- Becomes `NaN` when `en_voxels == 0`
- Logic reference: `extract_sri24_atlas_features.py` line `379`

### `global_ed_en_ratio`

- Becomes `NaN` when `en_voxels == 0`
- Logic reference: `extract_sri24_atlas_features.py` line `380`

### `global_ed_total_ratio`

- Becomes `NaN` only when `total_tumor_voxels == 0`
- Logic reference: `extract_sri24_atlas_features.py` line `381`
- This did not occur in the UCSF CSV that was checked

### `tumor_burden_index`

- Becomes `NaN` only when `brain_voxels == 0`
- Logic reference: `extract_sri24_atlas_features.py` line `382`
- This also did not occur in the UCSF CSV that was checked

### `frontal_*_ratio`, `temporal_*_ratio`, `parietal_*_ratio`, `occipital_*_ratio`

- Each lobe-specific ratio becomes `NaN` when that lobe has `TOTAL == 0` tumor voxels assigned
- Logic reference: `extract_sri24_atlas_features.py` line `410`
- This is expected whenever a patient tumor does not occupy that lobe

## What Was Confirmed In The UCSF Atlas CSV

- `dominant_brain_lobe`: `1` blank value
- `global_nc_en_ratio`: `75` NaNs
- `global_ed_en_ratio`: `75` NaNs
- `global_ed_total_ratio`: `0` NaNs
- `tumor_burden_index`: `0` NaNs
- `frontal_*_ratio`: `7` NaNs each
- `temporal_*_ratio`: `133` NaNs each
- `parietal_*_ratio`: `128` NaNs each
- `occipital_*_ratio`: `322` NaNs each

## Plain-Language Interpretation

- The global ratio NaNs are mostly caused by tumors with no enhancing component, so dividing by `EN` is not possible.
- The per-lobe ratio NaNs are mostly caused by tumors that simply do not occupy that lobe.
- The blank `dominant_brain_lobe` value happens when none of the tumor voxels get assigned to the four requested lobes.

## Bottom Line

Most of these NaNs are expected by design and are not evidence of extraction failure.

That is why the downstream preprocessing pipeline in this repository includes explicit imputation and encoding before clustering and patient-wise correlation analysis.
