# Preprocessing

This step converts the raw atlas features into a clean training table.

## What it does

- Drops rows where `lobe_assignment_reliable` is false.
- Drops rows with missing `OS_months`.
- Assigns binary labels:
  - `risk_label = 1` if `OS_months <= 12` (high-risk)
  - `risk_label = 0` otherwise
- Imputes missing feature values with the column median.
- Leaves features unscaled by default (tree models do not require scaling).

## Usage

```bash
python src/preprocessing.py
```

Optional scaling (not in the abstract; requires confirmation):

```bash
python src/preprocessing.py --scale --confirm-scale
```

## Output

- `outputs/features_processed.csv`
- Columns: 16 features + `risk_label`

## Notes

- `OS_months` is never included as a feature.
- Scaling is optional and intended only for comparison workflows.
