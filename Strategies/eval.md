# DSU-2026 — Pipeline Evaluation (submission-shaped, pipeline-agnostic)

This is the **one evaluation contract** every pipeline must satisfy so we can pick the best one without caring how it was built (GBDT, hierarchical, NN, etc.).

The key idea: **score pipelines purely from their output in the competition submission format**, using the repo’s official 4× 2‑month forward validation windows.

---

## Output contract (what gets evaluated)

### Required columns (exact names)

- `Site` (A/B/C/D)
- `Date` (YYYY-MM-DD)
- `Block` (0,1,2,3) where `Block = Hour // 6`
- `ED Enc` (non-negative integer)
- `ED Enc Admitted` (non-negative integer)

### Required row set for a given evaluation window

For a window `[start_date, end_date]` (inclusive), the required evaluated grid is:

```
Site ∈ {A,B,C,D}
Date ∈ each day start_date..end_date
Block ∈ {0,1,2,3}
```

Row count must be:
\[
4 \times (\text{#days}) \times 4
\]

### Hard constraints (fail fast if violated)

- **No missing combos** (must cover the full grid)
- **No duplicates** on (`Site`,`Date`,`Block`)
- **Non-negativity**: `ED Enc ≥ 0`, `ED Enc Admitted ≥ 0`
- **Admitted ≤ Total**: `ED Enc Admitted ≤ ED Enc`
- **Integers**: both targets are integers (we allow float inputs if they are integer-valued; evaluator will enforce)

Why fail fast: otherwise a pipeline can “look good” by skipping hard dates/blocks or cheating constraints.

---

## Validation protocol (matches competition lookahead)

Use the repo’s official forward splits (implemented in `baseline_model/data_ingestion/loader.py#get_validation_periods()`):

1. Train ≤ 2024-12-31 → validate 2025-01-01..2025-02-28
2. Train ≤ 2025-02-28 → validate 2025-03-01..2025-04-30
3. Train ≤ 2025-04-30 → validate 2025-05-01..2025-06-30
4. Train ≤ 2025-06-30 → validate 2025-07-01..2025-08-31

**Final pipeline score** = average across the 4 validation periods.

---

## Metrics (what we report and what we optimize)

### Primary selection metric (what we minimize)

- **Admitted WAPE** over the full validation window (submission grain):

\[
\text{WAPE}(y,\hat y) = \frac{\sum |y - \hat y|}{\sum |y|}
\]

This matches the repo’s stated intent (see `Strategies/council/master_strategy.md`) and aligns with the reality that high-volume series matter more.

### Secondary metrics (diagnostics / tie-breakers)

For both `ED Enc` and `ED Enc Admitted`, computed at submission grain:

- **RMSE**:
\[
\sqrt{\frac{1}{n}\sum (y-\hat y)^2}
\]
- **MAE**
- **R²** (coefficient of determination)
- **WAPE** (for total too)

Also report **by-site** and **by-block** breakdowns (WAPE/RMSE) because pipelines often fail in specific sites or blocks.

---

## Pipeline-agnostic interface (two supported modes)

### Mode A — score an already-produced “submission-like” CSV

Your pipeline produces a CSV for each validation window with the required columns.

Evaluator loads:

- ground-truth from `Dataset/DSU-Dataset.csv` aggregated to blocks
- your prediction CSV
- runs validation + metrics

### Mode B — score an end-to-end runner (train→predict per fold)

You provide a callable that, given a fold definition and raw history, returns a submission-like dataframe. That’s the cleanest way to avoid accidental leakage and to standardize training cutoffs.

**Compatibility with Data Source:** Mode B passes raw df_hourly directly. If your pipeline depends on the unified master_block_history.parquet from the Data Source step (see Strategies/Data/data_source.md), your predict_fold callable should either:
1. Run 
un_data_ingestion() internally on the provided df_hourly slice, or
2. Accept the pre-built master dataset and filter to 	rain_end itself.

**Recommended default:** Use **Mode A** (pre-generated CSVs) during development. Use Mode B only for final leakage-proof validation runs where you want end-to-end guarantees.

---

## Reference evaluator (copy-paste Python)

Drop this into e.g. `scripts/evaluate.py` (or just run it from a notebook). It intentionally has **no sklearn dependency**.

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


SITES = ("A", "B", "C", "D")
BLOCKS = (0, 1, 2, 3)
COLS = ("Site", "Date", "Block", "ED Enc", "ED Enc Admitted")


@dataclass(frozen=True)
class Fold:
    period_id: int
    train_end: str
    test_start: str
    test_end: str
    description: str = ""


def _to_date_str(x: pd.Series) -> pd.Series:
    # Purpose: normalize dates for stable joins and CSV parity.
    # Effect: ensures `YYYY-MM-DD` strings.
    return pd.to_datetime(x).dt.strftime("%Y-%m-%d")


def load_raw_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def hourly_to_blocks_truth(df_hourly: pd.DataFrame) -> pd.DataFrame:
    # Purpose: produce ground-truth at competition grain (Site,Date,Block).
    df = df_hourly.copy()
    df["Block"] = (df["Hour"] // 6).astype(int)
    out = (
        df.groupby(["Site", "Date", "Block"], as_index=False)
          .agg({"ED Enc": "sum", "ED Enc Admitted": "sum"})
    )
    out["Date"] = _to_date_str(out["Date"])
    return out


def expected_grid(start_date: str, end_date: str) -> pd.DataFrame:
    dates = pd.date_range(start_date, end_date, freq="D").strftime("%Y-%m-%d")
    idx = pd.MultiIndex.from_product([SITES, dates, BLOCKS], names=["Site", "Date", "Block"])
    return idx.to_frame(index=False)


def validate_prediction_df(pred: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    # Purpose: enforce the submission contract so metrics are meaningful.
    # Effect: raises ValueError on contract violations; returns normalized df otherwise.
    missing = [c for c in COLS if c not in pred.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Required: {list(COLS)}")

    df = pred[list(COLS)].copy()
    df["Site"] = df["Site"].astype(str)
    df["Date"] = _to_date_str(df["Date"])
    df["Block"] = pd.to_numeric(df["Block"], errors="raise").astype(int)

    # Allowed values
    bad_sites = sorted(set(df["Site"]) - set(SITES))
    if bad_sites:
        raise ValueError(f"Invalid Site values: {bad_sites}. Allowed: {list(SITES)}")
    bad_blocks = sorted(set(df["Block"]) - set(BLOCKS))
    if bad_blocks:
        raise ValueError(f"Invalid Block values: {bad_blocks}. Allowed: {list(BLOCKS)}")

    # Uniqueness and coverage
    key_cols = ["Site", "Date", "Block"]
    if df.duplicated(key_cols).any():
        dups = df[df.duplicated(key_cols, keep=False)].sort_values(key_cols).head(20)
        raise ValueError(f"Duplicate (Site,Date,Block) rows found. Example:\n{dups}")

    grid = expected_grid(start_date, end_date)
    merged = grid.merge(df, on=key_cols, how="left", validate="one_to_one")
    if merged["ED Enc"].isna().any() or merged["ED Enc Admitted"].isna().any():
        missing_rows = merged[merged["ED Enc"].isna() | merged["ED Enc Admitted"].isna()][key_cols].head(40)
        raise ValueError(
            f"Predictions missing required rows for the window. Example missing keys:\n{missing_rows}"
        )

    # Numeric + integer-like
    for c in ("ED Enc", "ED Enc Admitted"):
        merged[c] = pd.to_numeric(merged[c], errors="raise")
        if not np.all(np.isfinite(merged[c].to_numpy())):
            raise ValueError(f"Non-finite values found in {c}")
        # allow floats that are actually integers (common after merges)
        if not np.all(np.isclose(merged[c] % 1, 0)):
            bad = merged.loc[~np.isclose(merged[c] % 1, 0), ["Site", "Date", "Block", c]].head(20)
            raise ValueError(f"Non-integer values found in {c}. Example:\n{bad}")
        merged[c] = merged[c].round().astype(int)

    # Constraints
    if (merged["ED Enc"] < 0).any() or (merged["ED Enc Admitted"] < 0).any():
        bad = merged[(merged["ED Enc"] < 0) | (merged["ED Enc Admitted"] < 0)].head(20)
        raise ValueError(f"Negative predictions found. Example:\n{bad}")
    if (merged["ED Enc Admitted"] > merged["ED Enc"]).any():
        bad = merged[merged["ED Enc Admitted"] > merged["ED Enc"]].head(20)
        raise ValueError(f"Admitted > Total violations found. Example:\n{bad}")

    return merged


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.sum(np.abs(y_true)))
    if denom <= 0:
        return float("nan")
    return float(np.sum(np.abs(y_true - y_pred)) / denom)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y = y_true.astype(float)
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - (ss_res / ss_tot)


def _metric_pack(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "wape": wape(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": r2(y_true, y_pred),
    }


def score_window(
    truth_blocks: pd.DataFrame,
    pred_df: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> Dict[str, object]:
    # Normalize / validate predictions
    pred = validate_prediction_df(pred_df, start_date, end_date)

    # Slice + join truth to required grid
    truth = truth_blocks.copy()
    truth = truth[(truth["Date"] >= start_date) & (truth["Date"] <= end_date)].copy()
    truth = truth[["Site", "Date", "Block", "ED Enc", "ED Enc Admitted"]]

    key_cols = ["Site", "Date", "Block"]
    joined = pred.merge(
        truth,
        on=key_cols,
        how="left",
        suffixes=("_pred", "_true"),
        validate="one_to_one",
    )
    if joined["ED Enc_true"].isna().any():
        raise ValueError("Truth missing rows for this window (unexpected).")

    # Overall metrics
    yt_total = joined["ED Enc_true"].to_numpy()
    yp_total = joined["ED Enc_pred"].to_numpy()
    yt_adm = joined["ED Enc Admitted_true"].to_numpy()
    yp_adm = joined["ED Enc Admitted_pred"].to_numpy()

    overall = {
        "total": _metric_pack(yt_total, yp_total),
        "admitted": _metric_pack(yt_adm, yp_adm),
        # Primary selection metric:
        "primary_admitted_wape": wape(yt_adm, yp_adm),
    }

    # Breakdowns
    by_site = []
    for s in SITES:
        sub = joined[joined["Site"] == s]
        by_site.append(
            {
                "Site": s,
                "total_wape": wape(sub["ED Enc_true"].to_numpy(), sub["ED Enc_pred"].to_numpy()),
                "admitted_wape": wape(sub["ED Enc Admitted_true"].to_numpy(), sub["ED Enc Admitted_pred"].to_numpy()),
                "total_rmse": rmse(sub["ED Enc_true"].to_numpy(), sub["ED Enc_pred"].to_numpy()),
                "admitted_rmse": rmse(sub["ED Enc Admitted_true"].to_numpy(), sub["ED Enc Admitted_pred"].to_numpy()),
            }
        )
    by_block = []
    for b in BLOCKS:
        sub = joined[joined["Block"] == b]
        by_block.append(
            {
                "Block": int(b),
                "total_wape": wape(sub["ED Enc_true"].to_numpy(), sub["ED Enc_pred"].to_numpy()),
                "admitted_wape": wape(sub["ED Enc Admitted_true"].to_numpy(), sub["ED Enc Admitted_pred"].to_numpy()),
                "total_rmse": rmse(sub["ED Enc_true"].to_numpy(), sub["ED Enc_pred"].to_numpy()),
                "admitted_rmse": rmse(sub["ED Enc Admitted_true"].to_numpy(), sub["ED Enc Admitted_pred"].to_numpy()),
            }
        )

    return {
        "overall": overall,
        "by_site": pd.DataFrame(by_site).sort_values("Site").reset_index(drop=True),
        "by_block": pd.DataFrame(by_block).sort_values("Block").reset_index(drop=True),
    }


def folds_2025() -> List[Fold]:
    return [
        Fold(1, "2024-12-31", "2025-01-01", "2025-02-28", "Train≤Dec2024 → Valid Jan-Feb2025"),
        Fold(2, "2025-02-28", "2025-03-01", "2025-04-30", "Train≤Feb2025 → Valid Mar-Apr2025"),
        Fold(3, "2025-04-30", "2025-05-01", "2025-06-30", "Train≤Apr2025 → Valid May-Jun2025"),
        Fold(4, "2025-06-30", "2025-07-01", "2025-08-31", "Train≤Jun2025 → Valid Jul-Aug2025"),
    ]


def evaluate_cv_from_csvs(
    truth_blocks: pd.DataFrame,
    fold_to_pred_csv: Dict[int, str],
) -> pd.DataFrame:
    rows = []
    for f in folds_2025():
        pred = pd.read_csv(fold_to_pred_csv[f.period_id])
        scored = score_window(truth_blocks, pred, f.test_start, f.test_end)
        rows.append(
            {
                "period_id": f.period_id,
                "window": f"{f.test_start}..{f.test_end}",
                "primary_admitted_wape": scored["overall"]["primary_admitted_wape"],
                "total_wape": scored["overall"]["total"]["wape"],
                "admitted_wape": scored["overall"]["admitted"]["wape"],
                "total_rmse": scored["overall"]["total"]["rmse"],
                "admitted_rmse": scored["overall"]["admitted"]["rmse"],
                "total_r2": scored["overall"]["total"]["r2"],
                "admitted_r2": scored["overall"]["admitted"]["r2"],
            }
        )
    out = pd.DataFrame(rows).sort_values("period_id").reset_index(drop=True)
    out.loc["mean"] = {
        "period_id": "mean",
        "window": "",
        **{c: float(out[c].mean()) for c in out.columns if c not in ("period_id", "window")},
    }
    return out


# ---- Mode B: end-to-end runner contract ----
# predict_fold(train_df_hourly, test_start, test_end) -> submission-like df
def evaluate_cv_runner(
    df_hourly: pd.DataFrame,
    predict_fold,
) -> pd.DataFrame:
    truth_blocks = hourly_to_blocks_truth(df_hourly)
    rows = []
    for f in folds_2025():
        train = df_hourly[df_hourly["Date"] <= pd.to_datetime(f.train_end)].copy()
        pred_df = predict_fold(train, f.test_start, f.test_end)
        scored = score_window(truth_blocks, pred_df, f.test_start, f.test_end)
        rows.append(
            {
                "period_id": f.period_id,
                "window": f"{f.test_start}..{f.test_end}",
                "primary_admitted_wape": scored["overall"]["primary_admitted_wape"],
                "total_wape": scored["overall"]["total"]["wape"],
                "admitted_wape": scored["overall"]["admitted"]["wape"],
                "total_rmse": scored["overall"]["total"]["rmse"],
                "admitted_rmse": scored["overall"]["admitted"]["rmse"],
                "total_r2": scored["overall"]["total"]["r2"],
                "admitted_r2": scored["overall"]["admitted"]["r2"],
            }
        )
    out = pd.DataFrame(rows).sort_values("period_id").reset_index(drop=True)
    out.loc["mean"] = {
        "period_id": "mean",
        "window": "",
        **{c: float(out[c].mean()) for c in out.columns if c not in ("period_id", "window")},
    }
    return out
```

---

## What “good” looks like (quick sanity checklist)

- **Mean admitted WAPE** decreases across iterations (that’s your primary).
- **Per-site admitted WAPE** doesn’t explode on any single site (common failure: Site B dominates; others get ignored).
- **By-block** errors are stable (common failure: block allocation drifts).
- **RMSE and R²** improve but don’t override WAPE decisions (they’re diagnostics; WAPE is what we optimize).

---

## Suggested workflow for every pipeline

1. For each fold, train using data **≤ train_end**.
2. Produce a CSV **exactly** matching the submission schema for that fold’s `[test_start..test_end]`.
3. Run the evaluator to get a fold table + mean row.
4. Use **mean admitted WAPE** to rank pipelines; use the other metrics + breakdowns to debug.

---

## Pipeline convergence analysis (cross-pipeline diagnostic)

When all pipelines have been scored, compute the **coefficient of variation** of their mean admitted WAPE scores:

$$CV = \frac{\sigma(\text{pipeline WAPEs})}{\mu(\text{pipeline WAPEs})}$$

| CV Range | Interpretation | Action |
|----------|---------------|--------|
| < 0.05 | **Converged** — dataset predictive ceiling reached | Focus on ensemble post-processing, not new pipelines |
| 0.05 - 0.15 | **Partial convergence** — some diversity remains useful | Ensemble will help; investigate outlier pipelines |
| > 0.15 | **Divergent** — pipelines capture different signals | Strong ensemble gains expected; keep all pipelines |

Also compute pairwise prediction correlations between pipelines. If all pairs show `corr > 0.95`, the pipelines are making the same errors and ensemble gains will be marginal.

See `master_strategy_2.md` §7.2 for the full convergence argument and actionable implications.
