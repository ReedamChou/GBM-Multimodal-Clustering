# Atlas Registration & Feature Extraction

## What This Script Does

`src/atlas_registration.py` is the most critical script in the pipeline. It performs three tasks:

1. **Builds a 4-lobe SRI24 atlas** — Parses the TZO116 parcellation, groups ~116 cortical regions into 4 lobes (frontal, temporal, parietal, occipital), and fills gaps via distance-transform nearest-seed assignment within a dilated supratentorial mask. Result is cached as `outputs/sri24_4lobe_atlas.nii.gz`.

2. **Registers the atlas to each patient's T1** — Uses ANTs affine registration (SRI24 T1 → patient T1), then warps the atlas and brainmask into patient space using nearest-neighbor interpolation. Transforms are cached per patient in `outputs/transforms/{patient_id}/`.

3. **Extracts 16 radiomic features** — From the registered atlas + tumor segmentation, computes 4 global volumetric ratios and 12 lobe-level invasion fractions.

## Required Input Files

### SRI24 Atlas (place in `data/SRI24/`)

| File | Description |
|------|-------------|
| `tzo116plus.nii.gz` | Label volume with ~116 cortical regions |
| `suptent.nii.gz` | Supratentorial mask |
| `tissues.nii.gz` | Brain tissue mask (brainmask) |
| `SRI24-tzo116plus.txt` | Label ID → region name mapping |
| `T1.nii.gz` | SRI24 T1 template (for ANTs registration) |

Download from: https://www.nitrc.org/projects/sri24/

> **Note:** The T1 template filename varies across SRI24 distributions. Set `data.sri24_t1_filename` in `config.json` to match your download (e.g., `T1.nii.gz`, `SRI24_T1.nii`, etc.).

### UCSF-PDGM Data

- Structural: `UCSF/DATA-IMAGE-STRUCTURAL/{ID}/{ID}_T1.nii.gz`
- Segmentations: `UCSF/DATA-AUTOMATED-SEGMENT/{ID}_tumor_segmentation.nii.gz`
- Clinical CSV: `data/raw/ucsf_with_clinical.csv` (must contain `patient_id` and `OS` columns)

## Registration Direction

**SRI24 → Patient T1** (fixed = patient, moving = SRI24).

This is required because UCSF-PDGM tumor segmentations are in each patient's native T1 space, NOT aligned to SRI24. The old pipeline incorrectly assumed co-registration.

## 4-Lobe Atlas Algorithm

1. Parse TZO label file; assign each of ~116 regions to a lobe via name prefixes
2. Build seed volume: voxels get values 1 (frontal), 2 (temporal), 3 (parietal), 4 (occipital)
3. Dilate supratentorial mask by `lobe_dilation_voxels` (default: 3)
4. `scipy.ndimage.distance_transform_edt` with `return_indices=True` fills unassigned voxels with nearest seed label
5. Zero out voxels outside the dilated supratentorial mask

## Lobe Ratio Denominator

Lobe ratios use **total lobe voxels** as denominator (not tumor-in-lobe):

```
frontal_ed_ratio = ED voxels in frontal lobe / total frontal lobe voxels
```

This represents **lobe invasion fraction** — what proportion of the frontal lobe is infiltrated by edema. Values are small (~0.001) by design.

## Output: `outputs/features_raw.csv`

| Column | Type | Description |
|--------|------|-------------|
| `patient_id` | str | UCSF-PDGM identifier |
| `global_nc_en_ratio` | float | NC voxels / ET voxels |
| `global_ed_en_ratio` | float | ED voxels / ET voxels |
| `global_ed_total_ratio` | float | ED voxels / WT voxels |
| `tumor_burden_index` | float | WT voxels / brain voxels |
| `{lobe}_{sub}_ratio` | float | 12 lobe-level ratios (4 lobes × 3 subregions) |
| `OS_months` | float | Overall survival converted from days |
| `lobe_assignment_reliable` | bool | True if ≥90% of tumor voxels have a lobe label |

## QA Column

`lobe_assignment_reliable` = True when ≥90% of tumor voxels fall within a non-zero lobe region of the registered atlas. Patients with False will be dropped in preprocessing.

## Fallback Mode

Set `atlas.use_ants_registration: false` in `config.json` to use affine-based nearest-neighbor resampling via NIfTI headers. This assumes SRI24 and patient data share world coordinates (which is generally NOT true for UCSF-PDGM). Use only for testing or when ANTs is unavailable.

## CLI Flags

```bash
python src/atlas_registration.py                            # full batch
python src/atlas_registration.py --dry-run                  # list patients only
python src/atlas_registration.py --patient UCSF-PDGM-0004  # single patient
python src/atlas_registration.py --config path/to/config.json
```
