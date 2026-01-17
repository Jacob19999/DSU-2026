from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from dsu_forecast.paths import repo_root


def build_case_mix_table(*, train_end: str) -> pd.DataFrame:
    """
    Returns (Site, Date, Block) + category share features.
    For dates > train_end, fills with climatology derived from <= train_end.
    """
    raw_path = repo_root() / "Dataset" / "DSU-Dataset.csv"
    map_path = repo_root() / "reason_categories.json"

    df = pd.read_csv(raw_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Block"] = (pd.to_numeric(df["Hour"]) // 6).astype(int)

    reason_to_cat = json.loads(Path(map_path).read_text(encoding="utf-8"))
    df["Reason_Category"] = df["REASON_VISIT_NAME"].map(reason_to_cat).fillna("Other/Unspecified")

    # Aggregate category volumes per (site,date,block)
    df = df[df["Date"] <= pd.to_datetime(train_end)].copy()
    cat = (
        df.groupby(["Site", "Date", "Block", "Reason_Category"], as_index=False)
        .agg(cat_enc=("ED Enc", "sum"))
        .sort_values(["Site", "Date", "Block"])
    )
    pivot = cat.pivot_table(index=["Site", "Date", "Block"], columns="Reason_Category", values="cat_enc", fill_value=0.0)
    pivot.columns = [f"cmix_{c}_enc" for c in pivot.columns]
    pivot = pivot.reset_index()

    # Shares
    enc_cols = [c for c in pivot.columns if c.startswith("cmix_") and c.endswith("_enc")]
    pivot["cmix_total_enc"] = pivot[enc_cols].sum(axis=1)
    for c in enc_cols:
        share_c = c.replace("_enc", "_share")
        pivot[share_c] = pivot[c] / pivot["cmix_total_enc"].clip(lower=1.0)

    # Climatology by (month, day, block) on shares only (avoid leaking total volume)
    pivot["month"] = pivot["Date"].dt.month
    pivot["day"] = pivot["Date"].dt.day
    share_cols = [c for c in pivot.columns if c.startswith("cmix_") and c.endswith("_share")]
    clim = (
        pivot.groupby(["Site", "month", "day", "Block"], as_index=False)[share_cols]
        .mean(numeric_only=True)
        .rename(columns={c: f"{c}_clim" for c in share_cols})
    )

    # Create a full grid for horizon filling later; we only return <=train_end rows here plus climatology table.
    pivot = pivot.drop(columns=["month", "day"], errors="ignore")
    clim = clim
    # Store climatology in attrs for downstream join convenience
    pivot.attrs["climatology"] = clim
    return pivot


def attach_case_mix_features(base: pd.DataFrame, *, train_end: str) -> pd.DataFrame:
    """
    Merge case-mix shares onto base and fill post-train_end rows with climatology.
    """
    base_out = base.copy()
    base_out["Date"] = pd.to_datetime(base_out["Date"])
    train_end_dt = pd.to_datetime(train_end)

    cmix = build_case_mix_table(train_end=train_end)
    clim = cmix.attrs.get("climatology")

    base_out = base_out.merge(cmix.drop(columns=[c for c in cmix.columns if c.startswith("cmix_") and c.endswith("_enc")], errors="ignore"), on=["Site", "Date", "Block"], how="left")

    if isinstance(clim, pd.DataFrame) and not clim.empty:
        base_out["month"] = base_out["Date"].dt.month
        base_out["day"] = base_out["Date"].dt.day
        base_out = base_out.merge(clim, on=["Site", "month", "day", "Block"], how="left")
        for c in [c for c in base_out.columns if c.endswith("_share")]:
            cc = f"{c}_clim"
            if cc in base_out.columns:
                # only fill future / missing
                base_out.loc[base_out["Date"] > train_end_dt, c] = base_out.loc[base_out["Date"] > train_end_dt, c].fillna(
                    base_out.loc[base_out["Date"] > train_end_dt, cc]
                )
        base_out = base_out.drop(columns=["month", "day"] + [c for c in base_out.columns if c.endswith("_clim")], errors="ignore")

    # Remaining NaNs -> 0 shares (rare)
    for c in [c for c in base_out.columns if c.startswith("cmix_") and c.endswith("_share")]:
        base_out[c] = base_out[c].fillna(0.0)

    return base_out

