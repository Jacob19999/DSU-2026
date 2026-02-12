# Pipeline E: Reason-Mix Latent Factor Model — Audit Report

**Audited against:** `Strategies/Implementation plan/Pipeline_E_Implementation.md`  
**Date:** 2026-02-12  
**Status:** 2 bugs, 6 warnings, 8 misalignments

---

## BUGS

### BUG-1: `largest_remainder_round` can break sum invariant after clipping — `training.py` ~lines 46-59

**Severity: MINOR**

The `deficit < 0` branch decrements floored values, potentially pushing some to -1. The final `np.maximum(floored, 0)` clips them back to 0, but silently breaks the sum invariant — the rounded values may no longer sum to `target_sum`.

```python
elif deficit < 0:
    idx = np.argsort(remainders)[:(-deficit)]
    floored[idx] -= 1       # can push to -1
return np.maximum(floored, 0)  # clips, breaking sum guarantee
```

**Fix:** After `np.maximum(floored, 0)`, re-check and redistribute any remaining deficit.

---

### BUG-2: `shift()` assumes contiguous daily grid — no date-gap validation — `factor_forecasting.py` ~lines 42-43, `features.py` ~lines 142-153

**Severity: MINOR (latent)**

All lag/rolling features use positional `shift(lag)`, which only equals a calendar-day lag if there are zero date gaps within each `(site, block)` group. `data_loader.py` asserts total row count but does **not** verify date contiguity per group. If the parquet ever has a missing day, every lag feature silently shifts by the wrong amount.

**Fix:** Add contiguity assertion:
```python
for (site, blk), grp in df.groupby(["site", "block"]):
    diffs = grp["date"].diff().dropna()
    assert (diffs == pd.Timedelta(days=1)).all(), f"Date gap in {site}/{blk}"
```

---

## WARNINGS

### WARN-1: No sample weight on early-stopping eval sets — `training.py` ~lines 167-188, `predict.py` ~lines 82-104

Training uses `sample_weight` (volume-based + COVID down-weight) but early-stopping evaluation is unweighted. Stopping criterion optimizes a different loss surface than the training objective. Could cause premature or delayed stopping.

---

### WARN-2: Train/predict factor-value distribution mismatch — `factor_forecasting.py` ~lines 176-188

For training rows, `factor_i_pred` = actual PCA values. For validation/forecast rows, `factor_i_pred` = GBDT-predicted values. The final model trains on exact factor values but predicts using noisy estimates. This is standard two-stage practice but introduces systematic train/test distribution shift.

---

### WARN-3: Extremely expensive Optuna — factor pipeline re-runs per trial — `training.py` ~lines 334-404

Each Optuna trial calls `train_fold()` across all 4 folds. Each `train_fold` invocation re-runs the full factor pipeline (PCA fit + factor lag engineering + factor forecast model training). At 100 trials × 4 folds = 400 full factor-pipeline executions. Consider caching the factor-enriched DataFrames outside the objective.

---

### WARN-4: 60 factor features may cause overfitting — `factor_forecasting.py`, `features.py`

Code includes **all** factor lag/rolling/momentum features for all 5 factors: 5 factors × (1 pred + 6 lags + 3 rolling + 1 momentum + 1 deviation) = **60 factor features**. The plan specifies only 5 feature types (factor_pred, factor_momentum, factor_deviation_yearly, factor_lag_63, factor_roll_mean_28) = 25 features. The additional 35 highly-correlated features risk diluting importance and overfitting.

---

### WARN-5: Static features assume pre-existing calendar columns — `features.py` ~lines 69-76

`add_static_features` directly references `df["dow"]`, `df["day_of_year"]`, `df["month"]`, `df["year"]` without creating them. Must exist in the parquet. If the schema changes, this fails with KeyError.

---

### WARN-6: Holiday date ranges hardcoded to 2017-2026 — `features.py` ~lines 83-84

Xmas/July4 lists end at 2025, `holidays.US` covers through 2026. If the dataset extends, `days_until_next` will return NaN for dates near end-of-2025 looking forward.

---

## MISALIGNMENT

### MIS-1: `FACTOR_LAG_DAYS` and `FACTOR_ROLLING_SHIFT` differ from plan values (INTENTIONAL)

| Parameter | Plan | Code |
|-----------|------|------|
| `FACTOR_LAG_DAYS` | `[7,14,28,56,91,182,364]` | `[63,70,77,91,182,364]` |
| `FACTOR_ROLLING_SHIFT` | `7` | `63` |

The plan has an internal contradiction: config section lists short lags, but CRITICAL note says "For v1 safe approach, use lags >= 63 only." **Code correctly implements the safe constraint.** Plan config should be updated to match.

---

### MIS-2: Factor momentum formula adapted for safe lags (INTENTIONAL) — `factor_forecasting.py` ~lines 57-59

Plan: `factor_lag_7 - factor_lag_14`. Code: `factor_lag_63 - factor_lag_70`. This is the safe-lag equivalent (same 7-day delta, shifted back). Correct adaptation.

---

### MIS-3: `factor_x_site` interaction feature MISSING — `features.py` ~lines 113-119

**Severity: MEDIUM**

Plan explicitly requires `factor_x_site` interaction features. Code creates `holiday_x_block`, `weekend_x_block`, `site_x_dow`, `site_x_month` but **no** `factor_x_site`. Confirmed absent in saved model feature lists.

**Fix:**
```python
for i in range(cfg.N_FACTORS):
    df[f"factor_{i}_x_site"] = df[f"factor_{i}_pred"] * df["site_enc"]
```

---

### MIS-4: `share_matrix.parquet` never saved to disk — `share_matrix.py`

Plan says "Save share_matrix.parquet." The function returns the DataFrame but never writes it. Blocks reproducibility for debugging.

---

### MIS-5: Factor config Optuna tuning not implemented — `config.py` ~line 122

`OPTUNA_N_TRIALS_FACTOR = 30` is defined but never used. Optuna only has two stages (total_enc params and admit_rate params). Factor method, n_factors, and factor forecast params are **never tuned**. Dead config constant.

---

### MIS-6: Factor forecast MAE missing from evaluation diagnostics — `evaluate.py` ~lines 164-175

Plan requires "factor forecast MAE" in evaluation diagnostics. The evaluate module reports factor-feature importance counts but does **not** compute or report factor forecast MAE. This metric is only printed transiently during training and not persisted.

---

### MIS-7: PCA/NMF factor model not saved to disk — `training.py`, `predict.py`

Plan says "Save factor model and loadings." Code saves LightGBM boosters but never pickles the PCA/NMF model or its loadings. `print_factor_loadings` helper exists in `factor_extraction.py` but is never called. Prevents post-hoc factor interpretation and reproducibility.

---

### MIS-8: More factor features than plan specifies — `features.py`

Plan: 5 factor feature types × 5 factors = 25 features. Code: 12 feature types × 5 factors = **60 features**. The extra 35 are additional lags and rolling windows not in the plan. See also WARN-4.

---

## VERIFIED CORRECT

| Check | Status |
|-------|--------|
| Fold dates match eval.md | ✅ PASS |
| Factor extraction is fold-aware (PCA fit on train only) | ✅ PASS |
| All factor lags >= 63 (safe, no leakage) | ✅ PASS |
| All target lags >= 63 (no leakage) | ✅ PASS |
| ROLLING_SHIFT = 63 | ✅ PASS |
| Non-negativity constraint enforced | ✅ PASS |
| Integer output enforced | ✅ PASS |
| admitted <= total enforced | ✅ PASS |
| 976-row assertion for Sept-Oct 2025 | ✅ PASS |
| Share computation correct (each_reason / row_total) | ✅ PASS |
| `event_count` included as feature from parquet | ✅ PASS |
| OOF predictions saved | ✅ PASS |

---

## PRIORITY

1. **MIS-3** — Add `factor_x_site` interaction (plan requirement, easy fix)
2. **MIS-5** — Implement factor config tuning or remove dead constant
3. **MIS-7** — Pickle PCA/NMF models for reproducibility
4. **BUG-2** — Add date-contiguity assertion
5. **WARN-4/MIS-8** — Consider restricting factor features to plan-specified subset
