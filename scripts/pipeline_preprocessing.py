from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


MISSING_MARKERS = {
    "",
    "na",
    "n/a",
    "nan",
    "none",
    "null",
    "not available",
    "not applicable",
    "unknown",
    "indeterminate",
}


COLUMN_ALIASES = {
    "id": ["ID", "id", "PatientID", "patient_id"],
    "dominant_lobe": [
        "dominant_lobe",
        "Dominant_Lobe",
        "Dominant lobe",
        "dominant brain lobe",
        "dominant_brain_lobe",
    ],
    "sex": ["Sex", "Gender", "sex", "gender"],
    "mgmt": ["MGMT status", "MGMT", "mgmt_status", "mgmt"],
    "idh": ["IDH", "IDH1", "idh", "idh1"],
    "os": ["OS", "Survival_from_surgery_days_UPDATED", "survival_days"],
    "censor": ["1-dead 0-alive", "Survival_Censor", "survival_censor"],
    "survival_status": ["Survival_Status", "survival_status"],
}


def first_present_column(columns: pd.Index, aliases: list[str]) -> str | None:
    lookup = {str(c).lower(): c for c in columns}
    for alias in aliases:
        key = alias.lower()
        if key in lookup:
            return str(lookup[key])
    return None


def normalize_string_series(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.lower()


def safe_mode(series: pd.Series, default: Any = "unknown") -> Any:
    mode_values = series.mode(dropna=True)
    if mode_values.empty:
        return default
    return mode_values.iloc[0]


def replace_missing_markers(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_object_dtype(out[col]) or pd.api.types.is_string_dtype(out[col]):
            s = out[col].astype("string").str.strip()
            lower_s = s.str.lower()
            out[col] = s.mask(lower_s.isin(MISSING_MARKERS), pd.NA)
    return out


def maybe_convert_to_numeric(df: pd.DataFrame, minimum_parse_rate: float = 0.8) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_object_dtype(out[col]) or pd.api.types.is_string_dtype(out[col]):
            non_null_count = out[col].notna().sum()
            if non_null_count == 0:
                continue
            converted = pd.to_numeric(out[col], errors="coerce")
            parse_rate = converted.notna().sum() / float(non_null_count)
            if parse_rate >= minimum_parse_rate:
                out[col] = converted
    return out


def map_sex_to_binary(series: pd.Series) -> pd.Series:
    s = normalize_string_series(series)
    out = pd.Series(np.nan, index=series.index, dtype="float")
    out[s.isin(["m", "male"])] = 1
    out[s.isin(["f", "female"])] = 0
    return out


def map_mgmt_to_binary(series: pd.Series) -> pd.Series:
    s = normalize_string_series(series)
    out = pd.Series(np.nan, index=series.index, dtype="float")
    out[s.str.contains("positive", na=False) | s.str.contains("methylated", na=False)] = 1
    out[s.str.contains("negative", na=False) | s.str.contains("unmethylated", na=False)] = 0
    return out


def map_idh_to_binary(series: pd.Series) -> pd.Series:
    s = normalize_string_series(series)
    out = pd.Series(np.nan, index=series.index, dtype="float")
    out[s.str.contains("wild", na=False)] = 0
    out[s.str.contains("mutat", na=False) | s.str.contains("idh1 p\\.", na=False) | s.str.contains("idh2 p\\.", na=False)] = 1
    return out


def canonicalize_lobe(series: pd.Series) -> pd.Series:
    s = normalize_string_series(series)
    out = pd.Series(pd.NA, index=series.index, dtype="string")
    out[s.str.contains("frontal", na=False)] = "frontal"
    out[s.str.contains("temporal", na=False)] = "temporal"
    out[s.str.contains("parietal", na=False)] = "parietal"
    out[s.str.contains("occipital", na=False)] = "occipital"
    return out
