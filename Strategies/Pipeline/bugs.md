# Bug Tracker & Implementation Plan — Post-Run Analysis (2026-02-13)

**Context:** Run with target encodings, cross-block lags, residual model, and mixed-effects changes. Results compared against previous run (committed at `35e6cc1`).

---

## BUG-1: Pipeline B — Cross-Block Features Regression (+0.43pp Admitted WAPE)

### Diagnosis

Pipeline B regressed from 0.2786 → 0.2829 admitted WAPE. Initial hypothesis was TE min_lag=16 leaking, but **the TE functions are dead code** — `add_target_encodings_to_bucket()` is defined in `features.py` but never called from `run_pipeline.py` (line 69 imports only `add_static_features, build_bucket_data`).

The actual culprit is the **cross-block features inside `build_bucket_data()`** (lines 223–234 of `features.py`). These use `shift(h + k)` with horizon-adaptive shifts, but:
- For Bucket 1 (`h ∈ [1,15]`), `shift(h + k)` with `k = lags[0] = 16` means `shift(17..31)` — the Block 3 data is only 17–31 days old. This is correct leakage-wise (block 3 happened before the as-of date), but the signal may be too noisy at short horizons for Block 0's overnight volumes.
- Regression is concentrated on Site D (+1.77pp), Site A (+1.27pp), and Block 0 (+1.46pp).

### Root Cause

The cross-block features add 10+ new columns (`xblock_b3_total_{16,21,28,56,91,182,364}`, `xblock_b3_roll_mean_{7,14,28}`) for Block 0 rows, but these are all NaN for non-Block-0 rows. With horizon expansion, the expanded dataset is ~13× larger per bucket. Adding sparse cross-block features to this already-expanded dataset may increase noise-to-signal, especially for Bucket 1 where there are more horizon rows per observation.

### Fix Options (pick one)

#### Option A: Remove cross-block features from B entirely (safest)

**File:** `Pipelines/Pipeline B/features.py`

Delete or comment out lines 172–177 (Block 3 pre-extraction) and lines 223–234 (cross-block feature computation inside `build_bucket_data`).

Specifically:

1. Remove Block 3 pre-extraction (lines 172–176):
```python
# DELETE these lines inside build_bucket_data():
    _b3_total_by_site: dict[str, pd.Series] = {}
    for _s in df["site"].unique():
        b3 = df.loc[(df["site"] == _s) & (df["block"] == 3)].sort_values("date")
        _b3_total_by_site[_s] = b3["total_enc"]
```

2. Remove cross-block computation inside the `for h in h_list:` loop (lines 223–234):
```python
# DELETE these lines:
            # ── Cross-block lags: Block 0 ← lagged Block 3 (evening decay) ──
            if _blk == 0 and _site in _b3_total_by_site:
                b3_t = _b3_total_by_site[_site]
                for k in lags:
                    expanded.loc[idx, f"xblock_b3_total_{k}"] = (
                        b3_t.shift(h + k).values
                    )
                b3_shifted = b3_t.shift(h + min_lag)
                for w in [7, 14, 28]:
                    expanded.loc[idx, f"xblock_b3_roll_mean_{w}"] = (
                        b3_shifted.rolling(w, min_periods=1).mean().values
                    )
```

#### Option B: Only add cross-block features for Buckets 2–3 (compromise)

**File:** `Pipelines/Pipeline B/features.py`, inside `build_bucket_data()`.

Wrap the cross-block block in a bucket guard:

```python
            # Only add cross-block for longer-horizon buckets (safer lags)
            if bucket_id >= 2 and _blk == 0 and _site in _b3_total_by_site:
                # ... existing cross-block code ...
```

This preserves cross-block signal where the lag gap (≥31 days) is large enough to be reliable, while dropping it for Bucket 1 where `h + k` can be as small as 17.

#### Option C: Wire up the TE functions (as originally intended) + revert cross-block

The TE functions `add_target_encodings()` and `add_target_encodings_to_bucket()` exist but are never called. If the intent was to use them:

**File:** `Pipelines/Pipeline B/run_pipeline.py`, after line 77 (bucket_data_map build loop):

```python
    # ── Step 3b: Add target encodings per bucket ──────────────────────
    from features import add_target_encodings_to_bucket
    for bid in [1, 2, 3]:
        bucket_data_map[bid] = add_target_encodings_to_bucket(
            bucket_data_map[bid], base_df, bid,
        )
        print(f"  Bucket {bid}: +TE features → {len(bucket_data_map[bid].columns)} cols")
```

**BUT** this reintroduces the TE min_lag=16 concern for Bucket 1. To be safe, override Bucket 1's min_lag for TE specifically:

**File:** `Pipelines/Pipeline B/features.py`, `add_target_encodings_to_bucket()`:

Change line 323 from:
```python
    min_lag = cfg.BUCKETS[bucket_id]["min_lag"]
```
To:
```python
    # Use conservative lag for TE to avoid near-leak (Bucket 1 min_lag=16 is too short
    # for 90-day trailing means that could include validation-adjacent data)
    min_lag = max(cfg.BUCKETS[bucket_id]["min_lag"], 63)
```

### Recommendation

**Go with Option A** (remove cross-block from B). Pipeline B's strength is horizon-adaptive lags, not cross-block signal. Cross-block features work well in Pipeline A (which doesn't expand by horizon), but the expansion in B creates too many sparse NaN columns that dilute the gradient. The cross-block signal is already captured by the same-block rolling features in B's expanded form.

### Verification

After fix, re-run:
```powershell
cd "Pipelines/Pipeline B"
python run_pipeline.py --mode cv --skip-tune
```

Expected: Admitted WAPE returns to ~0.2786 (matching pre-change baseline).

---

## BUG-2: Pipeline D — Mixed-Effects Model Never Executes (Falls Back to Per-Series GLMs)

### Diagnosis

`model_type.txt` in `fold_0/` reads `per_series`. Folds 1–4 don't even have `model_type.txt` files — only `coefficients.csv`.

**Root cause is in `predict.py`**, not `training.py`:

```
17:from training import (
18:    train_all_models,       ← imports per-series directly
19:    save_models,
20:    ...
21:)
```

Line 173: `models = train_all_models(train_df, fourier_config, alpha, verbose=False)`
Line 232: `models = train_all_models(train_df, fourier_config, alpha, verbose=True)`

`predict.py` imports and calls `train_all_models()` directly — **completely bypassing** the `train_models()` unified entry point (training.py line 385) that does the mixed-effects attempt + fallback. The `train_models()` function was added to `training.py` but the call site in `predict.py` was never updated.

The `model_type.txt = "per_series"` in `fold_0/` comes from `save_models()` being called at `predict.py:204` and `predict.py:254`, which still receives the per-series dict.

### Fix

**File:** `Pipelines/Pipeline D/predict.py`

#### Step 1: Update import (line 17–22)

Change:
```python
from training import (
    train_all_models,
    save_models,
    largest_remainder_round,
    wape,
)
```
To:
```python
from training import (
    train_models,
    train_all_models,
    predict_mixed_effects,
    save_models,
    largest_remainder_round,
    wape,
)
```

#### Step 2: Update `train_and_predict_fold()` (line 173)

Change:
```python
    models = train_all_models(train_df, fourier_config, alpha, verbose=False)
```
To:
```python
    models = train_models(train_df, fourier_config, alpha, verbose=True)
```

#### Step 3: Update `generate_final_forecast()` (line 232)

Change:
```python
    models = train_all_models(train_df, fourier_config, alpha, verbose=True)
```
To:
```python
    models = train_models(train_df, fourier_config, alpha, verbose=True)
```

#### Step 4: Update `predict_window()` to handle both model types

The current `predict_window()` (lines 29–97) assumes per-series dict keyed by `(site, block)`. It needs a branch for mixed-effects:

Replace `predict_window()` with:

```python
def predict_window(
    master_df: pd.DataFrame,
    models: dict,
    forecast_start: str,
    forecast_end: str,
    fourier_config: list[dict] | None = None,
) -> pd.DataFrame:
    """Generate raw predictions for all (site, block) on a date window."""
    model_type = models.get("model_type", "per_series")

    if model_type == "mixed_effects":
        return _predict_window_mixed(master_df, models, forecast_start, forecast_end, fourier_config)
    return _predict_window_per_series(master_df, models, forecast_start, forecast_end, fourier_config)


def _predict_window_mixed(
    master_df: pd.DataFrame,
    models: dict,
    forecast_start: str,
    forecast_end: str,
    fourier_config: list[dict] | None = None,
) -> pd.DataFrame:
    """Predict using unified mixed-effects models."""
    start = pd.Timestamp(forecast_start)
    end = pd.Timestamp(forecast_end)

    forecast_df = master_df[
        (master_df["date"] >= start) & (master_df["date"] <= end)
    ].copy()

    if forecast_df.empty:
        return pd.DataFrame()

    pred_total = predict_mixed_effects(models["total_model"], forecast_df, target="total")
    pred_rate = predict_mixed_effects(models["rate_model"], forecast_df, target="rate")
    pred_admitted = pred_total * pred_rate

    rows = []
    for i in range(len(forecast_df)):
        rows.append({
            "site": forecast_df.iloc[i]["site"],
            "date": forecast_df.iloc[i]["date"],
            "block": forecast_df.iloc[i]["block"],
            "pred_total": pred_total[i],
            "pred_rate": pred_rate[i],
            "pred_admitted": pred_admitted[i],
        })
    return pd.DataFrame(rows)


def _predict_window_per_series(
    master_df: pd.DataFrame,
    models: dict,
    forecast_start: str,
    forecast_end: str,
    fourier_config: list[dict] | None = None,
) -> pd.DataFrame:
    """Predict using per-series GLMs (original logic)."""
    # ... existing predict_window body (lines 40–97) ...
```

#### Step 5: Handle potential `mixedlm` convergence failure

`train_mixed_effects_total()` and `train_mixed_effects_rate()` catch exceptions and return `None`, triggering fallback to per-series. But the `vc_formula` syntax may be causing silent issues.

In `training.py`, the `vc_formula` uses `C(site_block)`:
```python
vc_formula={"site_block": "0 + C(site_block)"},
```

This requires `site_block` to be in the dataframe's columns AND in the formula namespace. Verify that `df["site_block"]` is created before the `smf.mixedlm()` call — it is (line 157), so this should work. But if the formula parsing fails silently, the `except Exception` catches it and falls back.

**Add logging to confirm:** In `training.py`, `train_mixed_effects_total()`, after line 193, add:
```python
        import traceback
        if verbose:
            print(f"  [Mixed-effects total_enc] FAILED: {e}")
            traceback.print_exc()
```

### Verification

```powershell
cd "Pipelines/Pipeline D"
python run_pipeline.py --mode fold --fold-id 1 --skip-tune
```

Check:
1. Console output should show `Attempting mixed-effects model (single pooled model)...`
2. `output/models/fold_1/model_type.txt` should read `mixed_effects`
3. If it still falls back, the traceback will reveal why

---

## BUG-3: Pipeline E — TE Features Add No Value (Marginal Regression +0.13pp)

### Diagnosis

Pipeline E went from 0.2766 → 0.2779 admitted WAPE after adding target encodings. By-site analysis shows uniform tiny regressions across all sites (none improved).

**Why:** Pipeline E's factor features (`factor_*_pred`, `factor_*_momentum`, `factor_*_deviation_yearly`, `factor_*_roll_mean_*`) already capture site-level volume baselines. The TE features (`te_site_mean_total_enc`, `te_site_block_mean_total`, etc.) are computing trailing 90-day means of the same `total_enc` and `admitted_enc` targets — highly collinear with what the factor features + lag features already provide. Adding 8 collinear features adds split noise without new signal.

The `te_site_month_mean` feature (computed per-fold from training data) is the only potentially useful one, but it's a static mean per (site, month) — essentially a lookup table the tree can already learn from `site_enc × month` interactions.

### Fix

**File:** `Pipelines/Pipeline E/features.py`

#### Option A: Remove TE entirely (recommended)

In `add_all_base_features()` (line 346–352), remove the `add_target_encodings` call:

Change:
```python
def add_all_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add static features + target lag features + cross-block lags + target encodings."""
    df = add_static_features(df)
    df = add_target_lag_features(df)
    df = add_cross_block_lag_features(df)
    df = add_target_encodings(df)
    return df
```
To:
```python
def add_all_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add static features + target lag features + cross-block lags."""
    df = add_static_features(df)
    df = add_target_lag_features(df)
    df = add_cross_block_lag_features(df)
    return df
```

Also remove the `compute_fold_target_encodings()` call from wherever it's invoked in the training/run_pipeline script. Search for it:

```powershell
rg "compute_fold_target_encodings" "Pipelines/Pipeline E/"
```

If called in `run_pipeline.py` or `training.py`, remove those call sites too.

#### Option B: Keep only `te_site_month_mean` (compromise)

If you want to preserve the fold-specific encoding, keep `compute_fold_target_encodings()` but remove `add_target_encodings()` from the base feature pipeline. The per-fold site×month mean is the only feature that couldn't trivially be reconstructed by the tree from existing features.

### Verification

```powershell
cd "Pipelines/Pipeline E"
python run_pipeline.py --mode cv --skip-tune
```

Expected: Admitted WAPE returns to ~0.2766 (matching pre-change baseline).

---

## BUG-4: Stale `evaluation_report.md`

### Diagnosis

`Pipelines/Eval/output/evaluation_report.md` shows numbers from the previous run (committed at `35e6cc1`). The working-tree CSVs (`leaderboard.csv`, `per_fold_comparison.csv`, etc.) contain the current run's data, but the markdown report was not regenerated.

**Root cause:** The Eval pipeline has no markdown report generator. The `compare.py:save_reports()` function (line 269) saves CSVs and JSON but not the `.md` file. The report was hand-written or generated externally.

### Fix

**File:** `Pipelines/Eval/compare.py`

Add a `generate_markdown_report()` function and call it from `save_reports()`.

Add after `save_reports()` (line 319):

```python
def generate_markdown_report(results: Dict[str, Dict]) -> str:
    """Generate evaluation_report.md from current results."""
    from datetime import date

    lb = build_leaderboard(results)
    pf = per_fold_comparison(results)
    conv = convergence_analysis(results)
    corr = pairwise_correlation(results)

    lines = []
    lines.append("# DSU-2026 Pipeline Evaluation Report\n")
    lines.append(f"**Generated:** {date.today().isoformat()}")
    lines.append("**Evaluation protocol:** 4× 2-month forward validation windows (per `Strategies/eval.md`)")
    lines.append("**Primary metric:** Mean Admitted WAPE (lower is better)")
    lines.append(f"**Pipelines evaluated:** {', '.join(sorted(results.keys()))}\n")
    lines.append("---\n")

    # ── Leaderboard ──
    lines.append("## Leaderboard (ranked by Mean Admitted WAPE)\n")
    lines.append("| Rank | Pipeline | Admitted WAPE | Total WAPE | Admitted RMSE | Total RMSE | Admitted R² | Total R² |")
    lines.append("|------|----------|:------------:|:----------:|:-------------:|:----------:|:-----------:|:--------:|")
    for rank, row in lb.iterrows():
        name = row["pipeline"]
        bold = "**" if rank == 1 else ""
        lines.append(
            f"| {rank} | {bold}{name}{bold} | {bold}{row['admitted_wape']:.4f}{bold} | "
            f"{row['total_wape']:.4f} | {row['admitted_rmse']:.3f} | {row['total_rmse']:.3f} | "
            f"{row['admitted_r2']:.3f} | {row['total_r2']:.3f} |"
        )
    winner = lb.iloc[0]["pipeline"]
    w_wape = lb.iloc[0]["admitted_wape"]
    lines.append(f"\n**Pipeline {winner} wins** with a mean admitted WAPE of {w_wape:.4f}.\n")
    lines.append("---\n")

    # ── Per-Fold ──
    if len(pf) > 0:
        lines.append("## Per-Fold Breakdown (Admitted WAPE)\n")
        fold_windows = {
            1: "Jan–Feb 2025", 2: "Mar–Apr 2025",
            3: "May–Jun 2025", 4: "Jul–Aug 2025",
        }
        pipe_names = sorted([c for c in pf.columns if c != "best"])
        header = "| Fold | Window | " + " | ".join(pipe_names) + " | Best |"
        sep_row = "|------|--------|" + "|".join([":-----:" for _ in pipe_names]) + "|:----:|"
        lines.append(header)
        lines.append(sep_row)
        for fold_id, row in pf.iterrows():
            best = row["best"]
            cells = []
            for p in pipe_names:
                val = row[p]
                if p == best:
                    cells.append(f"**{val:.4f}**")
                else:
                    cells.append(f"{val:.4f}")
            window = fold_windows.get(fold_id, "")
            lines.append(f"| {fold_id} | {window} | " + " | ".join(cells) + f" | **{best}** |")
        lines.append("")
        lines.append("---\n")

    # ── By-Site ──
    lines.append("## By-Site Analysis (Admitted WAPE, averaged across folds)\n")
    pipe_names_sorted = sorted(results.keys())
    header = "| Site | " + " | ".join(pipe_names_sorted) + " |"
    sep_row = "|------|" + "|".join([":-----:" for _ in pipe_names_sorted]) + "|"
    lines.append(header)
    lines.append(sep_row)
    # Collect by-site data
    site_data = {}
    for name, res in results.items():
        if not res["by_site"].empty:
            for _, row in res["by_site"].iterrows():
                site = row["Site"]
                if site not in site_data:
                    site_data[site] = {}
                site_data[site][name] = row["admitted_wape"]
    for site in sorted(site_data.keys()):
        vals = site_data[site]
        best_p = min(vals, key=vals.get)
        cells = []
        for p in pipe_names_sorted:
            v = vals.get(p, float("nan"))
            if p == best_p:
                cells.append(f"**{v:.4f}**")
            else:
                cells.append(f"{v:.4f}")
        lines.append(f"| {site} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("---\n")

    # ── By-Block ──
    lines.append("## By-Block Analysis (Admitted WAPE, averaged across folds)\n")
    block_labels = {0: "0 (00:00–05:59)", 1: "1 (06:00–11:59)", 2: "2 (12:00–17:59)", 3: "3 (18:00–23:59)"}
    header = "| Block (6h window) | " + " | ".join(pipe_names_sorted) + " |"
    sep_row = "|-------------------|" + "|".join([":-----:" for _ in pipe_names_sorted]) + "|"
    lines.append(header)
    lines.append(sep_row)
    block_data = {}
    for name, res in results.items():
        if not res["by_block"].empty:
            for _, row in res["by_block"].iterrows():
                blk = row["Block"]
                if blk not in block_data:
                    block_data[blk] = {}
                block_data[blk][name] = row["admitted_wape"]
    for blk in sorted(block_data.keys()):
        vals = block_data[blk]
        best_p = min(vals, key=vals.get)
        cells = []
        for p in pipe_names_sorted:
            v = vals.get(p, float("nan"))
            if p == best_p:
                cells.append(f"**{v:.4f}**")
            else:
                cells.append(f"{v:.4f}")
        label = block_labels.get(blk, str(blk))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("---\n")

    # ── Convergence ──
    lines.append("## Convergence Analysis\n")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Number of pipelines | {conv['n_pipelines']} |")
    if not np.isnan(conv.get("cv", float("nan"))):
        lines.append(f"| Mean WAPE (across pipelines) | {conv['mean_wape']:.4f} |")
        lines.append(f"| Std WAPE | {conv['std_wape']:.4f} |")
        lines.append(f"| **Coefficient of Variation (CV)** | **{conv['cv']:.4f}** |")
        if conv["cv"] < 0.05:
            interp_label = "Converged"
        elif conv["cv"] < 0.15:
            interp_label = "Partial Convergence"
        else:
            interp_label = "Divergent"
        lines.append(f"| Interpretation | **{interp_label}** |")
    lines.append("")
    lines.append("---\n")

    # ── Pairwise Correlation ──
    if corr is not None:
        lines.append("## Pairwise Prediction Correlation\n")
        names = list(corr.columns)
        header = "|   | " + " | ".join(names) + " |"
        sep_row = "|---|" + "|".join([":-----:" for _ in names]) + "|"
        lines.append(header)
        lines.append(sep_row)
        for name in names:
            cells = [f"{corr.loc[name, n]:.3f}" for n in names]
            lines.append(f"| **{name}** | " + " | ".join(cells) + " |")
        lines.append("")

    return "\n".join(lines) + "\n"
```

Then update `save_reports()` to call it. Add at the end of `save_reports()` (before the last print):

```python
    # Markdown report
    md_report = generate_markdown_report(results)
    md_path = out / "evaluation_report.md"
    md_path.write_text(md_report, encoding="utf-8")
    print(f"  Saved: {md_path}")
```

### Verification

```powershell
cd "Pipelines/Eval"
python run_eval.py
```

Check that `output/evaluation_report.md` now matches `output/leaderboard.csv` numbers.

---

## Execution Order

| Priority | Bug | Impact | Effort | Dependencies |
|----------|-----|--------|--------|--------------|
| **1** | BUG-2 (D mixed-effects) | High — D is 11pp behind leader, mixed-effects was the planned fix | Medium | None |
| **2** | BUG-1 (B cross-block) | Medium — B dropped from 2nd to 3rd | Low (delete code) | None |
| **3** | BUG-3 (E TE removal) | Low — only +0.13pp | Low (delete call) | None |
| **4** | BUG-4 (stale report) | Housekeeping | Medium (new function) | Run after all pipeline fixes |

BUG-1, BUG-2, BUG-3 are independent — can be fixed in parallel. BUG-4 should be done last since you'll want to regenerate the report after re-running all pipelines with fixes applied.

### Full Re-run Sequence

```powershell
# 1. Fix all code changes
# 2. Re-run affected pipelines
cd "Pipelines/Pipeline D" && python run_pipeline.py --mode cv --skip-tune
cd "Pipelines/Pipeline B" && python run_pipeline.py --mode cv --skip-tune
cd "Pipelines/Pipeline E" && python run_pipeline.py --mode cv --skip-tune

# 3. Re-run centralized eval (regenerates all CSVs + markdown report)
cd "Pipelines/Eval" && python run_eval.py
```

### Expected Post-Fix Leaderboard

| Pipeline | Current WAPE | Expected Post-Fix |
|----------|-------------|-------------------|
| A | 0.2794 | 0.2794 (unchanged) |
| B | 0.2829 | ~0.2786 (revert to baseline) |
| C | 0.2910 | 0.2910 (unchanged) |
| D | 0.3888 | ~0.35–0.37 (speculative — mixed-effects should help but won't close the full gap) |
| E | 0.2779 | ~0.2766 (revert to baseline) |
