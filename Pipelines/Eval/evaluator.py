"""
Eval: Core evaluation engine — implements the eval.md reference evaluator.

Pipeline-agnostic: scores any CSV that matches the submission contract
(Site, Date, Block, ED Enc, ED Enc Admitted).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from . import config as cfg
    from .config import BLOCKS, COLS, FOLDS, SITES, Fold
except ImportError:
    import config as cfg  # type: ignore[no-redef]
    from config import BLOCKS, COLS, FOLDS, SITES, Fold  # type: ignore[no-redef]


# ═══════════════════════════════════════════════════════════════════════════════
#  Date helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _to_date_str(x: pd.Series) -> pd.Series:
    """Normalize dates to YYYY-MM-DD strings for stable joins."""
    return pd.to_datetime(x).dt.strftime("%Y-%m-%d")


# ═══════════════════════════════════════════════════════════════════════════════
#  Ground truth builder
# ═══════════════════════════════════════════════════════════════════════════════

def load_raw_dataset(csv_path: Optional[str] = None) -> pd.DataFrame:
    """Load raw visits CSV. Uses config default if path not given."""
    path = csv_path or str(cfg.RAW_VISITS_CSV)
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def hourly_to_blocks_truth(df_hourly: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hourly visits to (Site, Date, Block) grain — eval.md contract."""
    df = df_hourly.copy()
    df["Block"] = (df["Hour"] // 6).astype(int)
    out = (
        df.groupby(["Site", "Date", "Block"], as_index=False)
          .agg({"ED Enc": "sum", "ED Enc Admitted": "sum"})
    )
    out["Date"] = _to_date_str(out["Date"])
    return out


def build_truth() -> pd.DataFrame:
    """Load raw visits and return ground truth at block grain."""
    print(f"  Loading ground truth from {cfg.RAW_VISITS_CSV} ...")
    raw = load_raw_dataset()
    return hourly_to_blocks_truth(raw)


# ═══════════════════════════════════════════════════════════════════════════════
#  Expected grid
# ═══════════════════════════════════════════════════════════════════════════════

def expected_grid(start_date: str, end_date: str) -> pd.DataFrame:
    """Full cartesian product of Sites × Dates × Blocks for a window."""
    dates = pd.date_range(start_date, end_date, freq="D").strftime("%Y-%m-%d")
    idx = pd.MultiIndex.from_product(
        [list(SITES), dates, list(BLOCKS)],
        names=["Site", "Date", "Block"],
    )
    return idx.to_frame(index=False)


# ═══════════════════════════════════════════════════════════════════════════════
#  Submission contract validation (fail-fast)
# ═══════════════════════════════════════════════════════════════════════════════

def validate_prediction_df(
    pred: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Enforce the eval.md submission contract.
    Raises ValueError on any violation; returns normalized df otherwise.
    """
    # Required columns
    missing = [c for c in COLS if c not in pred.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Required: {list(COLS)}")

    df = pred[list(COLS)].copy()
    df["Site"]  = df["Site"].astype(str)
    df["Date"]  = _to_date_str(df["Date"])
    df["Block"] = pd.to_numeric(df["Block"], errors="raise").astype(int)

    # Allowed values
    bad_sites = sorted(set(df["Site"]) - set(SITES))
    if bad_sites:
        raise ValueError(f"Invalid Site values: {bad_sites}. Allowed: {list(SITES)}")
    bad_blocks = sorted(set(df["Block"]) - set(BLOCKS))
    if bad_blocks:
        raise ValueError(f"Invalid Block values: {bad_blocks}. Allowed: {list(BLOCKS)}")

    # Uniqueness
    key_cols = ["Site", "Date", "Block"]
    if df.duplicated(key_cols).any():
        dups = df[df.duplicated(key_cols, keep=False)].sort_values(key_cols).head(20)
        raise ValueError(f"Duplicate (Site,Date,Block) rows found. Example:\n{dups}")

    # Coverage: every cell in the grid must be present
    grid = expected_grid(start_date, end_date)
    merged = grid.merge(df, on=key_cols, how="left", validate="one_to_one")
    if merged["ED Enc"].isna().any() or merged["ED Enc Admitted"].isna().any():
        missing_rows = merged[
            merged["ED Enc"].isna() | merged["ED Enc Admitted"].isna()
        ][key_cols].head(40)
        raise ValueError(
            f"Predictions missing required rows for the window. "
            f"Example missing keys:\n{missing_rows}"
        )

    # Numeric, finite, integer-valued
    for c in ("ED Enc", "ED Enc Admitted"):
        merged[c] = pd.to_numeric(merged[c], errors="raise")
        if not np.all(np.isfinite(merged[c].to_numpy())):
            raise ValueError(f"Non-finite values found in {c}")
        if not np.all(np.isclose(merged[c] % 1, 0)):
            bad = merged.loc[
                ~np.isclose(merged[c] % 1, 0),
                ["Site", "Date", "Block", c],
            ].head(20)
            raise ValueError(f"Non-integer values found in {c}. Example:\n{bad}")
        merged[c] = merged[c].round().astype(int)

    # Non-negativity
    if (merged["ED Enc"] < 0).any() or (merged["ED Enc Admitted"] < 0).any():
        bad = merged[(merged["ED Enc"] < 0) | (merged["ED Enc Admitted"] < 0)].head(20)
        raise ValueError(f"Negative predictions found. Example:\n{bad}")

    # Admitted ≤ Total
    if (merged["ED Enc Admitted"] > merged["ED Enc"]).any():
        bad = merged[merged["ED Enc Admitted"] > merged["ED Enc"]].head(20)
        raise ValueError(f"Admitted > Total violations found. Example:\n{bad}")

    return merged


# ═══════════════════════════════════════════════════════════════════════════════
#  Metrics (no sklearn dependency)
# ═══════════════════════════════════════════════════════════════════════════════

def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.sum(np.abs(y_true)))
    return float(np.sum(np.abs(y_true - y_pred)) / denom) if denom > 0 else float("nan")


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y = y_true.astype(float)
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")


def _metric_pack(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "wape": wape(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mae":  mae(y_true, y_pred),
        "r2":   r2(y_true, y_pred),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Score a single validation window
# ═══════════════════════════════════════════════════════════════════════════════

def score_window(
    truth_blocks: pd.DataFrame,
    pred_df: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> Dict[str, object]:
    """
    Validate + score one prediction window.
    Returns dict with overall metrics, by-site, by-block breakdowns.
    """
    pred = validate_prediction_df(pred_df, start_date, end_date)

    # Slice truth to the window
    truth = truth_blocks.copy()
    truth = truth[(truth["Date"] >= start_date) & (truth["Date"] <= end_date)].copy()
    truth = truth[["Site", "Date", "Block", "ED Enc", "ED Enc Admitted"]]

    key_cols = ["Site", "Date", "Block"]
    joined = pred.merge(
        truth, on=key_cols, how="left",
        suffixes=("_pred", "_true"), validate="one_to_one",
    )
    if joined["ED Enc_true"].isna().any():
        raise ValueError("Truth missing rows for this window (unexpected).")

    # Overall metrics
    yt_total = joined["ED Enc_true"].to_numpy()
    yp_total = joined["ED Enc_pred"].to_numpy()
    yt_adm   = joined["ED Enc Admitted_true"].to_numpy()
    yp_adm   = joined["ED Enc Admitted_pred"].to_numpy()

    overall = {
        "total":    _metric_pack(yt_total, yp_total),
        "admitted": _metric_pack(yt_adm, yp_adm),
        "primary_admitted_wape": wape(yt_adm, yp_adm),
    }

    # By-site breakdown
    by_site = []
    for s in SITES:
        sub = joined[joined["Site"] == s]
        by_site.append({
            "Site": s,
            "total_wape":    wape(sub["ED Enc_true"].to_numpy(), sub["ED Enc_pred"].to_numpy()),
            "admitted_wape": wape(sub["ED Enc Admitted_true"].to_numpy(), sub["ED Enc Admitted_pred"].to_numpy()),
            "total_rmse":    rmse(sub["ED Enc_true"].to_numpy(), sub["ED Enc_pred"].to_numpy()),
            "admitted_rmse": rmse(sub["ED Enc Admitted_true"].to_numpy(), sub["ED Enc Admitted_pred"].to_numpy()),
        })

    # By-block breakdown
    by_block = []
    for b in BLOCKS:
        sub = joined[joined["Block"] == b]
        by_block.append({
            "Block": int(b),
            "total_wape":    wape(sub["ED Enc_true"].to_numpy(), sub["ED Enc_pred"].to_numpy()),
            "admitted_wape": wape(sub["ED Enc Admitted_true"].to_numpy(), sub["ED Enc Admitted_pred"].to_numpy()),
            "total_rmse":    rmse(sub["ED Enc_true"].to_numpy(), sub["ED Enc_pred"].to_numpy()),
            "admitted_rmse": rmse(sub["ED Enc Admitted_true"].to_numpy(), sub["ED Enc Admitted_pred"].to_numpy()),
        })

    return {
        "overall":  overall,
        "by_site":  pd.DataFrame(by_site).sort_values("Site").reset_index(drop=True),
        "by_block": pd.DataFrame(by_block).sort_values("Block").reset_index(drop=True),
        "_joined":  joined,  # kept for downstream aggregation
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Score all 4 folds for a single pipeline (Mode A — CSV)
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_pipeline(
    pipeline_name: str,
    truth_blocks: pd.DataFrame,
) -> Optional[Dict]:
    """
    Score a pipeline across all 4 folds.
    Returns None if no prediction CSVs found.
    """
    fold_results = []
    fold_summaries = []

    for fold in FOLDS:
        csv_path = cfg.get_fold_csv_path(pipeline_name, fold.period_id)
        if csv_path is None:
            print(f"  WARNING: Pipeline {pipeline_name} fold {fold.period_id} -- CSV not found, skipping")
            continue

        try:
            pred_df = pd.read_csv(csv_path)
            scored = score_window(truth_blocks, pred_df, fold.test_start, fold.test_end)
        except ValueError as e:
            print(f"  ERROR: Pipeline {pipeline_name} fold {fold.period_id} failed validation:\n    {e}")
            continue

        fold_results.append(scored)
        fold_summaries.append({
            "fold_id":              fold.period_id,
            "window":               f"{fold.test_start}..{fold.test_end}",
            "primary_admitted_wape": scored["overall"]["primary_admitted_wape"],
            "total_wape":           scored["overall"]["total"]["wape"],
            "admitted_wape":        scored["overall"]["admitted"]["wape"],
            "total_rmse":           scored["overall"]["total"]["rmse"],
            "admitted_rmse":        scored["overall"]["admitted"]["rmse"],
            "total_mae":            scored["overall"]["total"]["mae"],
            "admitted_mae":         scored["overall"]["admitted"]["mae"],
            "total_r2":             scored["overall"]["total"]["r2"],
            "admitted_r2":          scored["overall"]["admitted"]["r2"],
        })

    if not fold_summaries:
        return None

    # Build summary table with mean row
    df_folds = pd.DataFrame(fold_summaries).sort_values("fold_id").reset_index(drop=True)
    metric_cols = [c for c in df_folds.columns if c not in ("fold_id", "window")]
    mean_row = {"fold_id": "mean", "window": ""}
    for c in metric_cols:
        mean_row[c] = float(df_folds[c].mean())
    df_folds.loc[len(df_folds)] = mean_row

    # Aggregate by-site and by-block across all scored folds
    all_joined = pd.concat([r["_joined"] for r in fold_results], ignore_index=True)

    by_site_agg = []
    for s in SITES:
        sub = all_joined[all_joined["Site"] == s]
        if len(sub) == 0:
            continue
        by_site_agg.append({
            "Site": s,
            "total_wape":    wape(sub["ED Enc_true"].to_numpy(), sub["ED Enc_pred"].to_numpy()),
            "admitted_wape": wape(sub["ED Enc Admitted_true"].to_numpy(), sub["ED Enc Admitted_pred"].to_numpy()),
            "total_rmse":    rmse(sub["ED Enc_true"].to_numpy(), sub["ED Enc_pred"].to_numpy()),
            "admitted_rmse": rmse(sub["ED Enc Admitted_true"].to_numpy(), sub["ED Enc Admitted_pred"].to_numpy()),
        })

    by_block_agg = []
    for b in BLOCKS:
        sub = all_joined[all_joined["Block"] == b]
        if len(sub) == 0:
            continue
        by_block_agg.append({
            "Block": int(b),
            "total_wape":    wape(sub["ED Enc_true"].to_numpy(), sub["ED Enc_pred"].to_numpy()),
            "admitted_wape": wape(sub["ED Enc Admitted_true"].to_numpy(), sub["ED Enc Admitted_pred"].to_numpy()),
            "total_rmse":    rmse(sub["ED Enc_true"].to_numpy(), sub["ED Enc_pred"].to_numpy()),
            "admitted_rmse": rmse(sub["ED Enc Admitted_true"].to_numpy(), sub["ED Enc Admitted_pred"].to_numpy()),
        })

    return {
        "pipeline":     pipeline_name,
        "n_folds":      len(fold_summaries),
        "fold_table":   df_folds,
        "by_site":      pd.DataFrame(by_site_agg),
        "by_block":     pd.DataFrame(by_block_agg),
        "mean_admitted_wape": mean_row["primary_admitted_wape"],
        "mean_total_wape":    mean_row["total_wape"],
        "_all_joined":  all_joined,
    }
