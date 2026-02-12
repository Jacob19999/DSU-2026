# Pipeline D: GLM/GAM with Fourier Seasonality — Audit Report

**Audited against:** `Strategies/Implementation plan/Pipeline_D_Implementation.md`  
**Date:** 2026-02-12  
**Status:** 2 bugs, 6 warnings, 3 misalignments

---

## BUGS

### BUG-1: Weather climatology references non-existent `"month"` column — `data_loader.py` ~line 68

**Severity: LATENT CRASH**

```python
clim = df.groupby(["site", "month"])[col].transform("mean")
```

The groupby references `"month"` but this column is never created in `data_loader.py`. If any weather column still has NaN after ffill/bfill, this will raise a **KeyError**.

**Impact:** If weather data is fully populated after ffill/bfill, this code path never executes, but it's a ticking time bomb.

**Fix:**
```python
clim = df.groupby(["site", df["date"].dt.month])[col].transform("mean")
```

---

### BUG-2: Intercept (const) is L2-regularized — `training.py` ~lines 61-76

**Severity: MEDIUM (systematic bias)**

`alpha=0.1` is passed as a scalar, penalizing **all** parameters equally, **including the intercept** (`const` column inserted at position 0 in `features.py`). For Poisson GLM with log link, regularizing the intercept shrinks the baseline rate toward exp(0)=1, biasing predictions when the true mean is far from 1. Same issue in `train_rate_model`.

**Fix:** Pass a per-parameter alpha vector with 0 for the intercept:
```python
alpha_vec = np.array([0.0] + [alpha] * (X_train.shape[1] - 1))
result = model.fit_regularized(alpha=alpha_vec, L1_wt=cfg.GLM_L1_WT, maxiter=cfg.GLM_MAXITER)
```

---

## WARNINGS

### WARN-1: Weekly Fourier order=3 + DOW dummies = exact multicollinearity — `features.py`

**Severity: MEDIUM**

Weekly Fourier at order=3 on 7 discrete dayofweek values produces 6 features (sin/cos × 3 harmonics). DOW dummies (drop_first=True) also produce 6 features. Together with the intercept, these **span the same 7-dimensional space** — exact multicollinearity. L2 regularization prevents a singular matrix, but the redundant parameters waste capacity and make coefficient interpretation meaningless.

**Recommendation:** Either drop DOW dummies or reduce weekly Fourier order to 1-2.

---

### WARN-2: Convergence warnings silently suppressed — `training.py` ~line 69

```python
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    result = model.fit_regularized(...)
```

Both `train_total_model` and `train_rate_model` suppress **all** warnings during fitting. If `fit_regularized` fails to converge, the user will never know. Returned coefficients could be garbage.

**Recommendation:** Capture and log ConvergenceWarnings rather than silencing them.

---

### WARN-3: Weather NaN filled with 0.0 instead of climatology — `predict.py` ~lines 69-70

```python
if X.isna().any().any():
    X = X.fillna(0.0)
```

For Sept-Oct 2025 forecast, weather columns will be NaN and get filled with 0.0. Temperature = 0 is physically wrong for September. Should use monthly climatological means.

---

### WARN-4: `MAX_TOTAL_FACTOR` safety rail defined but never enforced — `config.py` ~line 82

```python
MAX_TOTAL_FACTOR = 1.5  # Safety rail: cap predictions at factor × historical max
```

Never referenced anywhere. A runaway Poisson prediction (exp of large eta) could produce unrealistically high values.

---

### WARN-5: Left-join for scoring could produce NaN WAPE — `predict.py` ~lines 185-189

`how="left"` merge means unmatched submission rows get NaN actuals. `np.abs(NaN - pred) = NaN`, making the entire WAPE = NaN. Should use `how="inner"` or drop NaN rows before scoring.

---

### WARN-6: Fragile weekly/annual period threshold — `features.py` ~line 73

```python
if period <= 7.5:
    t = dates.dt.dayofweek.values.astype(float)
else:
    t = dates.dt.dayofyear.values.astype(float)
```

Hardcoded two-branch assumption. If tuning ever introduces an intermediate period (e.g., 14), it would incorrectly use `dayofyear` as `t`.

---

## MISALIGNMENT

### MIS-1: Ablation studies not implemented — `tuning.py`

Plan specifies testing `covid_policy`, `trend_type`, and `family` ablations. The current tuning only searches `weekly_order × annual_order × alpha`. No ablation for:
- covid_policy (different weight values or exclusion)
- trend_type (linear vs sqrt vs piecewise)
- family (Poisson vs Negative Binomial for overdispersion)

---

### MIS-2: Config constants defined but never referenced

| Constant | Defined | Used? |
|----------|---------|-------|
| `CLIP_TOTAL_MIN = 0` | `config.py:80` | Never — `predict.py` hardcodes `.clip(lower=0)` |
| `ADMIT_RATE_CLIP = (0.0, 1.0)` | `config.py:81` | Never — hardcoded `.clip(0, 1)` |
| `MAX_TOTAL_FACTOR = 1.5` | `config.py:82` | Never referenced anywhere |

---

### MIS-3: `GLM_FAMILY` and `ADMIT_MODEL_TYPE` not configurable

Plan specifies these as config-level settings. They are hardcoded in `training.py`. If ablation studies were implemented, these would need to be parameterizable.

---

## VERIFIED CORRECT

| Check | Status |
|-------|--------|
| Fold dates match eval.md | ✅ PASS |
| Fourier term computation (weekly: dayofweek, annual: dayofyear) | ✅ PASS |
| NO lagged target features used | ✅ PASS (Pipeline D's key property) |
| GLM family/link (Poisson+log for total, Binomial+logit for rate) | ✅ PASS |
| L2 regularization via fit_regularized (L1_wt=0.0) | ✅ PASS |
| var_weights=total_enc for rate model (binomial n-trials) | ✅ PASS |
| Constraint enforcement (non-negative, admitted<=total, integer) | ✅ PASS |
| Largest-remainder rounding per (site, date) | ✅ PASS |
| Output schema matches eval.md | ✅ PASS |
| COVID weighting via freq_weights | ✅ PASS |
| OOF predictions saved for ensemble stacking | ✅ PASS |
| run_pipeline.py mode routing (cv, submit, fold, tune) | ✅ PASS |

---

## PRIORITY

1. **BUG-1** — Trivial one-liner fix (replace `"month"` with `df["date"].dt.month`)
2. **BUG-2** — Per-parameter alpha vector (systematic bias fix)
3. **WARN-1** — Drop DOW dummies or reduce Fourier order (multicollinearity)
4. **WARN-2** — Stop suppressing convergence warnings
5. **WARN-3** — Use climatology for weather imputation instead of 0.0
