# Pipeline B: Direct Multi-Step GBDT — Audit Report

**Audited against:** `Strategies/Implementation plan/Pipeline_B_Implementation.md`  
**Date:** 2026-02-12  
**Status:** 3 bugs, 5 warnings, 6 misalignments

---

## BUGS

### BUG-1: Validation predictions use averaged horizons instead of exact horizon — `training.py` ~lines 203-227

**Severity: CRITICAL**

In `train_fold`, when building prediction features for the validation window, `build_bucket_data` is called with ALL horizons in the bucket for ALL valid dates. Each `(site, date, block)` tuple receives predictions at every horizon in the bucket, which are then **averaged**:

```python
preds = (
    preds.groupby(["site", "date", "block"], as_index=False)
    .agg(pred_total=("pred_total", "mean"),
         pred_admitted=("pred_admitted", "mean"),
         bucket=("bucket", "first"))
)
```

For a date that is truly 35 days into the future, the model generates predictions with `days_ahead=31, 32, 33, ..., 59` and averages them. This does NOT reflect real submission behavior.

Contrast with `predict.py` ~line 97, which **correctly** filters to the exact horizon per date:
```python
mask = (pred_data["date"] - train_end).dt.days == pred_data["days_ahead"]
pred_data = pred_data[mask].copy()
```

**Impact:** CV WAPE is computed on averaged-across-horizon predictions, not exact-horizon predictions. CV metrics are unreliable and could mask horizon-specific weaknesses.

**Fix:** Add the same `mask` filter in `train_fold` after building prediction features (~line 209), and remove the groupby averaging.

---

### BUG-2: Optuna objective evaluates on the early-stopping set — `training.py` ~lines 420-422, 460-462

**Severity: MODERATE**

In both `_objective_total` and `_objective_rate`, the Optuna trial score is WAPE on `es_df` — the **same** data used for LightGBM early stopping:

```python
m_total, _ = train_bucket(train_df, es_df, fcols, params, None)
# Quick WAPE on ES set
preds = m_total.predict(es_df[fcols]).clip(0)
fold_wapes.append(wape(es_df["total_enc"].values, preds))
```

Early stopping already selected the optimal iteration for `es_df`. The Optuna score on the same data is **optimistically biased**.

**Impact:** Optuna may select hyperparameters that overfit to the ES split rather than generalizing to unseen data.

**Fix:** Evaluate on the actual fold validation window, or use a 3-way split (train / ES / Optuna-eval).

---

### BUG-3: Evaluate bucket bins exclude day 62 — `evaluate.py` ~lines 86-88

**Severity: MODERATE**

```python
merged["_bucket"] = pd.cut(
    merged["_days_ahead"],
    bins=[0, 15, 30, 61],
    labels=[1, 2, 3],
).astype("Int64")
```

`pd.cut` with `bins=[0, 15, 30, 61]` creates interval `(30, 61]` for Bucket 3. But config.py defines `h_max=62`. Fold 4 has a 62-day validation window, so `days_ahead=62` rows get `NaN` bucket and are silently excluded from per-bucket reporting.

**Fix:** Change to `bins=[0, 15, 30, 62]`.

---

## WARNINGS

### WARN-1: COVID down-weighting silently skipped if column missing — `data_loader.py` ~line 55

If `is_covid_era` is not a column in the parquet, `covid_mask = False` and all rows get weight 1.0. No warning is emitted. The code never derives the mask from config dates as a fallback.

---

### WARN-2: Rate model sample weight uses `admitted_enc`, not `total_enc` — `data_loader.py` ~line 60

Plan says `max(total_enc, 1) * (0.1 if covid else 1.0)` uniformly. Code creates a second weight column for the rate model using `max(admitted_enc, 1)`. This deviates from the plan.

---

### WARN-3: Optuna rate tuning redundantly trains total model — `training.py` ~lines 458-459

`train_bucket` trains **both** models, but only the rate model is evaluated. The already-tuned total model is wastefully retrained every trial. 50 rate trials × 4 folds × 3 buckets = 600 wasted total model fits.

---

### WARN-4: Confusing sanity assertion — `predict.py` ~lines 155-156

```python
assert (submission["ED Enc Admitted"] >= submission["ED Enc"]).sum() == 0 or \
       (submission["ED Enc Admitted"] <= submission["ED Enc"]).all(), "admitted > total"
```

Functionally correct but unnecessarily confusing. Should be simplified to the second condition only.

---

### WARN-5: `wape()` duplicated across `training.py` and `evaluate.py`

Identical implementations. Should live in a shared utility to avoid maintenance drift.

---

## MISALIGNMENT

### MIS-1: Bucket 3 bounds differ from plan — `config.py` ~line 53

| Parameter | Plan | Code |
|-----------|------|------|
| h_max | 61 | **62** |
| min_lag | 62 | **63** |

Code is actually more correct (Fold 4 spans 62 days), but disagrees with the stated plan.

---

### MIS-2: Optuna rate trials = 50, plan says 100 — `config.py` ~line 112

Plan specifies "100 per bucket-model pair." Code gives rate model only 50 trials.

---

### MIS-3: Rate model Optuna search space differs from plan — `training.py` ~lines 441-449

Plan: `n_estimators [800,3000]`, `max_depth [4,8]`. Code: `n_estimators [500,2000]`, `max_depth [3,7]`. May be intentional (rate is simpler), but deviates.

---

### MIS-4: No `get_fold_data` function in `data_loader.py`

Plan says data_loader should expose `get_fold_data`. Fold splitting is handled inside `training.py:_prepare_bucket_fold` instead.

---

### MIS-5: Event features explicitly excluded — `features.py` ~lines 264-268

`event_name` and `event_type` are in `_EXCLUDE_COLS` rather than being encoded. If the parquet has meaningful event annotations, this signal is lost.

---

### MIS-6: Rolling features only computed for `total_enc`, not `admit_rate` — `features.py` ~line 186

Only `total_enc` gets rolling stats. The admit_rate model has lag features but no rolling summary statistics.

---

## VERIFIED CORRECT

| Check | Status |
|-------|--------|
| Fold dates match eval.md | ✅ PASS |
| Lag safety check on import (min(lags) >= min_lag per bucket) | ✅ PASS |
| `days_ahead` included as feature | ✅ PASS |
| admitted <= total constraint enforced | ✅ PASS |
| Largest-remainder rounding per (site, date) | ✅ PASS |
| Output schema matches eval.md | ✅ PASS |
| run_pipeline.py modes (cv, submit, fold) | ✅ PASS |

---

## PRIORITY

1. **BUG-1** — CRITICAL: Fix horizon-averaging in validation. CV metrics are unreliable until fixed.
2. **BUG-2** — Fix Optuna eval on early-stopping set (optimistic bias in tuning).
3. **BUG-3** — Trivial fix: change `61` to `62` in bucket bins.
