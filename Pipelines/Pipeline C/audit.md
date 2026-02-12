# Pipeline C: Hierarchical Reconciliation — Audit Report

**Audited against:** `Strategies/Implementation plan/Pipeline_C_Implementation.md`  
**Date:** 2026-02-12  
**Status:** 4 bugs, 8 warnings, 3 misalignments

---

## BUGS

### BUG-1: Dead consistency-check code — `step_08_evaluate.py` ~lines 237-242

**Severity: MEDIUM**

`consistency_violations` is initialized to 0, the loop body never modifies it, and it is never printed or returned. The plan requires a "daily sum consistency check" — this is scaffolded but entirely inert.

```python
consistency_violations = 0
for (site, date), grp in all_merged.groupby(["Site", "Date"]):
    pred_sum = grp["ED Enc_pred"].sum()
    # Each pred was rounded by largest_remainder, so sum should be consistent
    # (We just count anomalous patterns, not exact violations since truth may differ)
```

**Fix:** Actually implement the check or remove the dead code.

---

### BUG-2: Optuna Phase 2 uses default daily params instead of Phase 1 best — `step_06_tune.py` ~lines 117-121, 179-183

**Severity: HIGH**

The `_objective_share` function receives no precomputed daily predictions. Inside it, `train_daily_fold(daily_df, fold, save=False)` trains daily models with **default** hyperparameters — not the best params discovered in Phase 1. The share model is tuned against a suboptimal daily baseline.

**Impact:** Best share params found may not be optimal when paired with the best daily params.

**Fix:** Pass Phase 1 best daily params (or precomputed daily predictions) into Phase 2.

---

### BUG-3: Final forecast uses single share model for both total and admitted — `step_07_predict.py` ~lines 304-322

**Severity: HIGH**

```python
share_preds_out[f"pred_share_total_b{b}"] = pred_probs[:, b]
share_preds_out[f"pred_share_admitted_b{b}"] = pred_probs[:, b]
```

Both total and admitted share columns get **identical** probabilities from a single model weighted by `total_enc`. But fold training (`step_05_train_shares.py` ~lines 92-118) trains **two** separate models — one weighted by `total_enc`, another by `admitted_enc`.

**Impact:** Final submission's admitted share allocation is driven by total-encounter patterns, inconsistent with fold validation behavior.

**Fix:** Train and use a separate admitted share model in `generate_final_forecast`.

---

### BUG-4: Admitted block sums break after clamping with no re-allocation — `step_07_predict.py` ~lines 101-105

**Severity: MEDIUM**

```python
block_admitted_int = largest_remainder_round(block_admitted_raw)
block_admitted_int = np.minimum(block_admitted_int, block_total_int)
```

`largest_remainder_round` guarantees sum preservation. But `np.minimum` can reduce individual blocks, breaking the sum. The "freed" admitted counts are never redistributed. The plan's daily sum consistency requirement is violated for admitted.

---

## WARNINGS

### WARN-1: Optuna objectives re-read parquet every trial, every fold — `step_06_tune.py` ~lines 73-74, 132-133

With 100 daily trials + 50 share trials × 4 folds = **600 parquet reads** during tuning. Should load once and pass into the objective closure.

---

### WARN-2: Optuna Phase 1 re-trains the share model on every trial — `step_06_tune.py` ~lines 60-65

No `precomputed_shares` is passed from `run_tuning`. Since Phase 1 only tunes daily params, share predictions are constant and should be precomputed once. Roughly 2x unnecessary runtime.

---

### WARN-3: `allocate_daily_to_blocks` uses `iterrows()` — `step_07_predict.py` ~line 73

Row-by-row DataFrame iteration. Very slow. Tolerable for CV but adds up during tuning (600 calls).

---

### WARN-4: Final forecast share model has no early stopping — `step_07_predict.py` ~lines 305-311

During fold training, share model uses `early_stopping(30)`. In the final forecast, no `eval_set` and no early stopping. Model trains for full `n_estimators`, risking overfitting.

---

### WARN-5: Final share model not saved to disk — `step_07_predict.py` ~lines 341-342

Daily total and rate models are saved, but `share_model` is never persisted. Final submission cannot be reproduced from saved artifacts alone.

---

### WARN-6: `admit_share_lag_*` features computed but never used — `step_03_feature_eng_shares.py` ~line 32

`admit_share_lag_*` columns are computed but `get_share_feature_columns` only matches `share_lag_*`. These columns consume memory and compute time but are never used.

---

### WARN-7: Allocation error decomposition only for total WAPE, not admitted — `step_08_evaluate.py` ~lines 206-208

Decomposition only computed for total WAPE. Since admitted WAPE is the primary metric, the admitted decomposition should also be reported.

---

### WARN-8: R-squared not computed — `step_08_evaluate.py`

The eval.md contract specifies R-squared as a secondary diagnostic. Pipeline C does not compute or report it.

---

## MISALIGNMENT

### MIS-1: Share lags are narrow-format instead of plan-specified wide-format — `step_03_feature_eng_shares.py` ~lines 28-33

**Severity: HIGH**

Plan specifies `share_lag_63_b0-b3, share_lag_364_b0-b3` — wide-format where each row has **all four blocks'** lagged shares. Code creates only `share_lag_63`, etc. — the lagged share for **that row's own block only**. The softmax classifier can't learn cross-block distributional shifts.

---

### MIS-2: Daily rate model is never tuned — `step_06_tune.py` ~line 233

Plan says "Phase 1: tune daily models (100 trials)" — plural. Only the total model is tuned; rate model always uses defaults.

---

### MIS-3: No eval.md hard-constraint validation — `step_08_evaluate.py`

The eval.md contract defines hard-constraint checks (no missing combos, no duplicates, non-negativity, admitted <= total, integer-valued). Pipeline C does not call `validate_prediction_df()` or implement equivalent checks.

---

## VERIFIED CORRECT

| Check | Status |
|-------|--------|
| Fold dates match eval.md | ✅ PASS |
| All config constants match plan (SITES, BLOCKS, N_BLOCKS, COVID, SHARE_MODEL_TYPE) | ✅ PASS |
| All lags >= 63 (daily and share models) | ✅ PASS |
| ROLLING_SHIFT_DAILY = 63, ROLLING_SHIFT_SHARES = 63 | ✅ PASS |
| Daily aggregation logic correct (sum across blocks) | ✅ PASS |
| admit_rate derivation correct (0/0 → 0.0, clip [0,1]) | ✅ PASS |
| Share computation correct (block_total / daily_total, zero-day → 0.25) | ✅ PASS |
| Share sums validated ≈ 1.0 (tolerance 1e-6) | ✅ PASS |
| Aggregate encodings computed per-fold on train only | ✅ PASS |
| Largest-remainder rounding per (site, date) | ✅ PASS |

---

## PRIORITY

1. **BUG-3** — Final forecast missing admitted share model (inconsistent with CV)
2. **BUG-2** — Optuna Phase 2 using default daily params (suboptimal share tuning)
3. **MIS-1** — Narrow vs wide share lags (limits share model's learning capacity)
4. **BUG-4** — Admitted sum break after clamping
5. **BUG-1** — Dead consistency check code
