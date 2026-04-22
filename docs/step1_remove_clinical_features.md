# Step 1: Remove Clinical Features

## Script

`scripts/step1_remove_clinical_features.py`

## Input

- `data/raw/ucsf_with_clinical.csv`

## Output

- `outputs/new/step1_ucsf_imaging_os.csv`

## Procedure

1. Load the raw UCSF CSV.
2. Resolve the survival column from common aliases, with `OS` as the current UCSF field.
3. Keep identifiers (`case_id`, `patient_id`), survival, and imaging-derived columns.
4. Remove the non-survival clinical columns:
   `Sex`, `Age at MRI`, `WHO CNS Grade`, `Final pathologic diagnosis (WHO 2021)`,
   `MGMT status`, `MGMT index`, `1p/19q`, `IDH`, `1-dead 0-alive`, `EOR`,
   and `Biopsy prior to imaging`.
5. Write the reduced dataset to `outputs/new`.

## Notes

- `dominant_brain_lobe` is kept because it is treated as an imaging-derived feature for downstream encoding.
- `OS` is preserved for survival-based clustering in step 2.
