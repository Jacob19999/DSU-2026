# Pipeline A: Global GBDT — Audit Report

**Audited against:** `Strategies/Implementation plan/Pipeline_A_Implementation.md`  
**Date:** 2026-02-12  
**Status:** 3 bugs, 5 warnings, 4 misalignments

---

## BUGS

### BUG-1: Missing guard for `is_covid_era` when `covid_policy="exclude"` — `step_03_train.py` ~line 97

**Severity: HIGH (crash risk)**

When `covid_policy="exclude"`, the code directly indexes `train_data["is_covid_era"]` without checking whether the column exists. If the master parquet lacks this column, this crashes with a `KeyError`.

`step_05_predict.py` handles this correctly with a guard:
```python
if "is_covid_era" in train_fit.columns:
    mask = train_fit["is_covid_era"].astype(bool)
    train_fit = train_fit[~mask].copy()
else:
    print("  [WARN] covid_policy='exclude' but 'is_covid_era' not in columns; skipping exclusion.")
```

**Impact:** Runtime crash during Optuna tuning (step 04 tries `covid_policy="exclude"` as a search candidate).

**Fix:** Add the same guard from step_05 into `train_fold()`.

---

### BUG-2: Standalone mode does not load tuned `covid_policy` from disk — `step_05_predict.py`

**Severity: MEDIUM (silent wrong result)**

`generate_final_forecast` loads tuned A1/A2 params from disk but **never** loads `best_covid_policy.txt`. The function defaults to `covid_policy="downweight"`.

When run standalone (`__main__`), the tuned covid policy is silently ignored.

**Impact:** If Optuna determined `covid_policy="exclude"` was optimal, the standalone final forecast silently uses "downweight" instead, producing a suboptimal submission.

**Fix:** Add covid_policy disk-loading logic alongside the A1/A2 param loading:
```python
if covid_policy == "downweight":
    pol_path = cfg.MODEL_DIR / "best_covid_policy.txt"
    if pol_path.exists():
        covid_policy = pol_path.read_text().strip()
```

---

### BUG-3: `cat_features` list not filtered against actual `feature_cols` — `step_03_train.py` ~line 87, `step_05_predict.py` ~line 59

**Severity: LOW (crash risk)**

`cat_features = ["site_enc", "block", "site_x_dow", "site_x_month"]` is hardcoded and passed directly to `LGBMRegressor.fit(categorical_feature=...)` without intersecting with `feature_cols`. If any of these columns are absent (e.g., `site_x_dow` not created upstream), LightGBM will raise an error.

**Fix:** Filter: `cat_features = [c for c in cat_features if c in feature_cols]`

---

## WARNINGS

### WARN-1: Top-N reason category selection uses all-time data (minor leakage) — `step_02_feature_eng.py` ~line 177

The selection of **which** N categories to track is computed across the entire dataset, including future dates. The actual share **values** are properly lagged by 63 days, so leakage is limited to category selection.

**Impact:** Minimal — dominant reason categories are unlikely to shift dramatically across the 8-month validation window.

---

### WARN-2: `bfill` on external enrichment features can use future data — `step_02_feature_eng.py` ~line 162

After `ffill`, leading NaNs (before the first available observation) are filled via `bfill`, which pulls values backward from the future. For fold 1, if there are NaN gaps in CDC ILI or AQI that ffill can't reach, bfill could pull validation-period data into the training window.

**Impact:** Low — only affects leading NaN gaps in external features, not target-derived features.

---

### WARN-3: Weather climatology uses global monthly means — `step_02_feature_eng.py` ~line 147

The monthly climatology for weather imputation is computed across all years including future data. Same pattern as WARN-2.

**Impact:** Minimal — weather patterns are external and cyclic.

---

### WARN-4: String-based date comparison for fold filtering — `step_06_evaluate.py` ~lines 68-69

Both `truth["Date"]` and `fold["val_start/val_end"]` are strings. Lexicographic comparison works for ISO dates, but is fragile to format deviations.

---

### WARN-5: Admitted sum may drift after `admitted <= total` enforcement

`largest_remainder_round` is applied per (Site, Date) to both `pred_total` and `pred_admitted`, then `pred_admitted` is clipped to `min(admitted, total)`. This clipping can reduce individual blocks, meaning the daily admitted sum no longer matches the originally-rounded sum. Reasonable tradeoff, but worth noting.

---

## MISALIGNMENT

### MIS-1: Two sample weight formulas vs. plan's single formula — `step_02_feature_eng.py` ~lines 221-222

- `sample_weight_a1` = covid_weight × total_enc.clip(1) — matches plan
- `sample_weight_a2` = covid_weight × **admitted_enc**.clip(1) — deviates from plan

**Assessment:** Reasonable deviation (A2 predicts admit_rate, so weighting by admitted volume is sensible). Plan should be updated.

---

### MIS-2: A2 search space fixed to `"regression"` — `step_04_tune.py` ~line 84

Plan states search space includes objective (tweedie/poisson). A2 hardcodes `"regression"`.

**Assessment:** Sensible — A2 targets admit_rate ∈ [0,1]; tweedie/poisson are inappropriate for a proportion.

---

### MIS-3: Column name casing differs from plan — `step_01_data_loading.py` ~line 22

Plan says title-case identifiers (Site, Date, Block). Code uses lowercase (`site`, `date`, `block`).

**Assessment:** Not a bug — pipeline uses lowercase internally, converts to title-case only for submission CSVs.

---

### MIS-4: Aggregate encodings only encode `total_enc`, not `admit_rate`

All three aggregate encodings (`site_month_block_mean`, `site_dow_mean`, `site_month_mean`) encode `total_enc`. No admit_rate aggregate encodings for model A2. Could improve A2 performance.

---

## VERIFIED CORRECT

| Check | Status |
|-------|--------|
| Fold dates match eval.md | ✅ PASS |
| All lags >= 63 (no leakage) | ✅ PASS |
| ROLLING_SHIFT = 63 | ✅ PASS |
| Aggregate encodings computed per-fold on train only | ✅ PASS |
| Current-period shares excluded from features | ✅ PASS |
| Raw reason counts excluded from features | ✅ PASS |
| admitted <= total constraint enforced | ✅ PASS |
| 976-row count for Sept-Oct forecast | ✅ PASS |
| Output schema matches eval.md | ✅ PASS |
| Largest-remainder rounding per (Site, Date) | ✅ PASS |

---

## PRIORITY

1. **BUG-1** — Fix immediately (will crash during Optuna tuning)
2. **BUG-2** — Fix before generating final submission
3. **BUG-3** — Defensive fix, low effort
