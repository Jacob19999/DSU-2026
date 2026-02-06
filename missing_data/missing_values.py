"""
Missing values and placeholder evaluation for DSU-Dataset.csv.
Reports: NaNs, zeros as placeholders, blank/null reason strings, temporal gaps
(admissions/encounters blank at certain periods), and suggested follow-up stats.
"""
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
DATASET_PATH = REPO_ROOT / "Dataset" / "DSU-Dataset.csv"
OUTPUT_DIR = REPO_ROOT / "results"
OUTPUT_CSV = OUTPUT_DIR / "missing_values_report.csv"

# Reason-like placeholders (case-insensitive)
REASON_PLACEHOLDERS = {
    "", "nan", "none", "null", "n/a", "na", "no reason", "unknown",
    "unspecified", "other", "unknown/unspecified", "not specified",
    "missing", "blank", "no reason given", "unable to determine",
}


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATASET_PATH)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df


def eval_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Explicit NaN/null counts per column."""
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    return pd.DataFrame({
        "column": missing.index,
        "missing_count": missing.values,
        "missing_pct": missing_pct.values,
    })


def eval_zeros(df: pd.DataFrame) -> pd.DataFrame:
    """Zeros in numeric columns (potential placeholders)."""
    numeric = ["ED Enc", "ED Enc Admitted", "Hour"]
    rows = []
    for col in numeric:
        if col not in df.columns:
            continue
        zero_count = (df[col] == 0).sum()
        zero_pct = (zero_count / len(df) * 100).round(2)
        rows.append({"column": col, "zero_count": zero_count, "zero_pct": zero_pct})
    return pd.DataFrame(rows)


def eval_reason_placeholders(df: pd.DataFrame) -> pd.DataFrame:
    """Null, blank, and placeholder strings in REASON_VISIT_NAME."""
    if "REASON_VISIT_NAME" not in df.columns:
        return pd.DataFrame()
    s = df["REASON_VISIT_NAME"].astype(str).str.strip()
    is_na = df["REASON_VISIT_NAME"].isna()
    is_blank = (s == "") | (s == "nan")
    is_placeholder = s.str.lower().isin(REASON_PLACEHOLDERS)
    rows = [
        {"reason_type": "null", "count": is_na.sum()},
        {"reason_type": "blank_or_nan_str", "count": (~is_na & is_blank).sum()},
        {"reason_type": "known_placeholder_str", "count": (~is_na & ~is_blank & is_placeholder).sum()},
    ]
    total_placeholder = is_na | is_blank | is_placeholder
    rows.append({"reason_type": "total_any_placeholder", "count": total_placeholder.sum()})
    out = pd.DataFrame(rows)
    out["pct"] = (out["count"] / len(df) * 100).round(2)
    return out


def eval_reason_placeholder_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Breakdown of known_placeholder_str by actual REASON_VISIT_NAME value (what fills the 9204)."""
    if "REASON_VISIT_NAME" not in df.columns:
        return pd.DataFrame()
    s = df["REASON_VISIT_NAME"].astype(str).str.strip()
    is_na = df["REASON_VISIT_NAME"].isna()
    is_blank = (s == "") | (s == "nan")
    is_placeholder = s.str.lower().isin(REASON_PLACEHOLDERS)
    mask = ~is_na & ~is_blank & is_placeholder
    if not mask.any():
        return pd.DataFrame(columns=["reason_value", "count", "pct"])
    vc = df.loc[mask, "REASON_VISIT_NAME"].value_counts()
    out = pd.DataFrame({"reason_value": vc.index.astype(str), "count": vc.values})
    out["pct"] = (out["count"] / len(df) * 100).round(2)
    return out


def eval_temporal_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check for (Site, Date, Block) combinations that have no rows or zero encounters.
    'Blank admissions at certain time periods' = no data or all zeros for that cell.
    """
    df = df.copy()
    df["Block"] = (df["Hour"] // 6).astype(int)
    # Expected: every (Site, Date, Block) could have 0+ rows; we care about blocks with 0 encounters
    block_totals = df.groupby(["Site", "Date", "Block"]).agg(
        enc=("ED Enc", "sum"),
        admitted=("ED Enc Admitted", "sum"),
    ).reset_index()
    zero_enc = block_totals["enc"] == 0
    zero_adm = block_totals["admitted"] == 0
    n_blocks = len(block_totals)
    rows = [
        {"metric": "block_periods_with_zero_enc", "count": zero_enc.sum(), "pct": (zero_enc.sum() / n_blocks * 100).round(2)},
        {"metric": "block_periods_with_zero_admitted", "count": zero_adm.sum(), "pct": (zero_adm.sum() / n_blocks * 100).round(2)},
        {"metric": "total_site_date_block_periods", "count": n_blocks, "pct": 100.0},
    ]
    return pd.DataFrame(rows)


def eval_invalid_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """ED Enc Admitted > ED Enc, negative counts, invalid Site/Hour."""
    rows = []
    if "ED Enc" in df.columns and "ED Enc Admitted" in df.columns:
        invalid = df["ED Enc Admitted"] > df["ED Enc"]
        rows.append({"check": "ED Enc Admitted > ED Enc", "count": invalid.sum()})
    for col in ["ED Enc", "ED Enc Admitted"]:
        if col in df.columns:
            neg = (df[col] < 0).sum()
            rows.append({"check": f"{col} < 0", "count": neg})
    if "Site" in df.columns:
        valid_sites = {"A", "B", "C", "D"}
        invalid_site = ~df["Site"].astype(str).str.upper().isin(valid_sites)
        rows.append({"check": "Site not in {A,B,C,D}", "count": invalid_site.sum()})
    if "Hour" in df.columns:
        invalid_hour = (df["Hour"] < 0) | (df["Hour"] > 23)
        rows.append({"check": "Hour not in [0,23]", "count": invalid_hour.sum()})
    return pd.DataFrame(rows)


def run_descriptive_suggestions() -> list[str]:
    """Suggest descriptive stats / checks not already covered in repo."""
    return [
        "Temporal coverage: count of distinct (Site, Date) and compare to expected calendar days; flag sites with missing dates.",
        "Duplicate keys: count (Site, Date, Hour, REASON_VISIT_NAME) duplicates; if >1 row per key, ED Enc/Admitted may be redundant or need summing.",
        "Outliers: IQR or z-score on ED Enc / ED Enc Admitted per (Site, Block) or per reason; flag extreme values.",
        "Admission rate bounds: per (Site, Date, Block) admit_rate = Admitted/Enc; flag rates >1 or very high.",
        "Reason coverage over time: % of encounters with placeholder reason by year/month or site; trend of missingness.",
        "Hour distribution: check for hours with no data for some (Site, Date) — structural zeros vs missing.",
    ]


def main() -> None:
    print("Loading dataset...")
    df = load_data()
    n = len(df)
    print(f"Rows: {n:,}, Columns: {list(df.columns)}\n")

    # 1) Explicit missing
    print("=" * 60)
    print("1. EXPLICIT MISSING (NaN)")
    print("=" * 60)
    missing_df = eval_missing(df)
    print(missing_df.to_string(index=False))
    has_missing = missing_df["missing_count"].gt(0).any()
    if not has_missing:
        print("No explicit missing values.")

    # 2) Zeros as placeholders
    print("\n" + "=" * 60)
    print("2. ZEROS (potential placeholders)")
    print("=" * 60)
    zero_df = eval_zeros(df)
    print(zero_df.to_string(index=False))

    # 3) Reason null/blank/placeholder
    print("\n" + "=" * 60)
    print("3. REASON: null / blank / placeholder strings")
    print("=" * 60)
    reason_df = eval_reason_placeholders(df)
    print(reason_df.to_string(index=False))
    reason_breakdown_df = eval_reason_placeholder_breakdown(df)
    if len(reason_breakdown_df) > 0:
        print("\n  Breakdown of known_placeholder_str (by reason value):")
        print(reason_breakdown_df.to_string(index=False))

    # 4) Temporal: blank at certain periods
    print("\n" + "=" * 60)
    print("4. TEMPORAL: (Site, Date, Block) with zero encounters/admissions")
    print("=" * 60)
    temporal_df = eval_temporal_gaps(df)
    print(temporal_df.to_string(index=False))

    # 5) Consistency
    print("\n" + "=" * 60)
    print("5. CONSISTENCY (invalid values)")
    print("=" * 60)
    consistency_df = eval_invalid_consistency(df)
    print(consistency_df.to_string(index=False))

    # 6) Suggested follow-up
    print("\n" + "=" * 60)
    print("6. SUGGESTED DESCRIPTIVE STATS (not fully in repo)")
    print("=" * 60)
    for i, s in enumerate(run_descriptive_suggestions(), 1):
        print(f"  {i}. {s}")

    # Save combined report CSV (flat: section, metric/column, count, pct)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for _, row in missing_df.iterrows():
        summary_rows.append({"section": "missing", "metric": row["column"], "count": row["missing_count"], "pct": row["missing_pct"]})
    for _, row in zero_df.iterrows():
        summary_rows.append({"section": "zeros", "metric": row["column"], "count": row["zero_count"], "pct": row["zero_pct"]})
    for _, row in reason_df.iterrows():
        summary_rows.append({"section": "reason", "metric": row["reason_type"], "count": row["count"], "pct": row["pct"]})
    for _, row in reason_breakdown_df.iterrows():
        summary_rows.append({"section": "reason_placeholder_breakdown", "metric": row["reason_value"], "count": row["count"], "pct": row["pct"]})
    for _, row in temporal_df.iterrows():
        summary_rows.append({"section": "temporal", "metric": row["metric"], "count": row["count"], "pct": row["pct"]})
    for _, row in consistency_df.iterrows():
        summary_rows.append({"section": "consistency", "metric": row["check"], "count": row["count"], "pct": None})
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nReport saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
