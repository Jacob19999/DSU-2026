# Site D Transfer Learning — Implementation Plan

**Status:** DRAFT  
**Motivation:** Site D admitted WAPE is ~0.47–0.49 across all top pipelines (E, A, B) — roughly 2× worse than Sites A/B/C (~0.24). The global model approach treats all sites equally, but Site D has structurally different dynamics: lowest volume (16.4% of total), lowest admission rate (18.3% vs 29–35%), and highest "other" reason-category share (52.5%). Global models are gradient-dominated by Sites A/B (60%+ of rows) and learn compromised splits.  
**Core Idea:** Split training into two explicit stages — an ABC teacher model and a D-specific student model — connected by feature transfer, not weight sharing. Each runs independently but the student inherits learned representations from the teacher.  
**Location:** Implemented as an enhancement to existing pipelines (primarily E and A), not a new pipeline.  
**Data Dependency:** `master_block_history.parquet` (same as all pipelines)

---

## 1. Problem Diagnosis (Why Global Models Fail on Site D)

### 1.1 Volume Imbalance

| Site | % of Rows | Avg Block Volume | Gradient Influence |
|------|-----------|------------------|--------------------|
| A | 26.3% | ~27 | Medium |
| B | 36.0% | ~37 | **Dominant** |
| C | 21.3% | ~25 | Medium |
| D | 16.4% | ~19 | **Drowned out** |

LightGBM's volume-weighted loss amplifies this: Site B rows produce ~2× the gradient magnitude of Site D rows. Tree splits optimize for the A/B/C population; Site D gets leftover leaf predictions.

### 1.2 Admission Rate Gap

| Site | Admit Rate | Implication |
|------|-----------|-------------|
| A | 34.8% | ~1 in 3 admitted |
| C | 32.7% | ~1 in 3 admitted |
| B | 27.9% | ~1 in 4 admitted |
| D | **18.3%** | ~1 in 5.5 admitted |

A single admit-rate model trained globally learns a compromise around ~29%. For Site D, this means systematic overprediction of admitted counts and underprediction of total encounters.

### 1.3 Reason-Mix Noise

Site D has 52.5% of visits in the "other" catch-all category (vs 44–50% for other sites). This means:
- PCA/NMF factors (Pipeline E) are noisier for Site D
- Reason-share features carry less discriminative signal
- The global model may over-index on factor features that are informative for A/B/C but noisy for D

### 1.4 Current Mitigation (Enhancement B) — Why It's Not Enough

Pipeline A already implements a hybrid-residual model for Site D (`step_03_train.py` lines 181–223). This trains a small LightGBM on the global model's Site D residuals with shrinkage weight 0.5. Results:
- Site D admitted WAPE for Pipeline A: **0.4771**
- Site D admitted WAPE for Pipeline E: **0.4810**

The residual model helps but is fundamentally limited — it corrects a global model that was never optimized for D. The global model's errors on D are **structurally biased** (wrong baseline, wrong block distribution, wrong admit-rate), not random noise that a residual model can easily fix.

---

## 2. Architecture: Two-Stage Transfer Learning

```
┌────────────────────────────────────────────────────────────┐
│  STAGE 1: ABC Teacher                                       │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Train on Sites A, B, C only                          │  │
│  │  Models: T1 (total_enc), T2 (admit_rate)              │  │
│  │  Full feature set (lags, rolling, calendar, factors)  │  │
│  └──────────┬───────────────────────────────────────────┘  │
│             │                                               │
│             ▼  Transfer artifacts                           │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  1. ABC predictions on D's historical data (meta-feat)│  │
│  │  2. ABC feature importances (feature selection guide) │  │
│  │  3. ABC aggregate encodings (baseline priors)         │  │
│  │  4. ABC admit-rate distribution (guardrail bounds)    │  │
│  └──────────┬───────────────────────────────────────────┘  │
│             │                                               │
│             ▼                                               │
│  STAGE 2: Site D Student                                    │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Train on Site D only                                 │  │
│  │  Models: S1 (total_enc), S2 (admit_rate)              │  │
│  │  Features: D's own + transferred meta-features        │  │
│  │  Regularization: aggressive (small dataset)           │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  MERGE: ABC predictions from T1/T2, D predictions from S1/S2│
│  → Standard submission CSV                                  │
└────────────────────────────────────────────────────────────┘
```

### Key Principle: No Weight Sharing, Only Feature Transfer

We do **not** use `init_model` or warm-starting. LightGBM's continued boosting from a pre-trained model is fragile with distribution shift (D's volume distribution is shifted left by ~40% vs ABC). Instead, the teacher's knowledge flows through **features** — its predictions on D's history become an informed prior that the student can learn to correct.

---

## 3. Stage 1: ABC Teacher Model

### 3.1 Training Data

```python
# Filter to Sites A, B, C only
train_abc = df_fold[train_mask & (df_fold["site"].isin(["A", "B", "C"]))]
```

- **Row count**: ~38K–42K rows per fold (vs ~45K–48K for global model) — still large enough for robust GBDT training
- **COVID policy**: Same as parent pipeline (downweight or exclude)
- **Sample weights**: Same volume-based weighting, but now Site B represents ~43% instead of ~36% — acceptable since we're only modeling A/B/C

### 3.2 Feature Set

Same as the parent pipeline's full feature set. No modifications needed — the ABC model is a standard GBDT, just trained on a subset of sites.

### 3.3 Model Training

```python
# Model T1: total_enc (Tweedie) — ABC only
model_t1 = lgb.LGBMRegressor(**params_a1)
model_t1.fit(
    X_train_abc, y_train_abc_total,
    sample_weight=w_abc_total,
    eval_set=[(X_val_abc, y_val_abc_total)],  # ABC val only for early stopping
    callbacks=[lgb.early_stopping(50, verbose=False)],
)

# Model T2: admit_rate (regression) — ABC only
model_t2 = lgb.LGBMRegressor(**params_a2)
model_t2.fit(
    X_train_abc, y_train_abc_rate,
    sample_weight=w_abc_rate,
    eval_set=[(X_val_abc, y_val_abc_rate)],
    callbacks=[lgb.early_stopping(50, verbose=False)],
)
```

### 3.4 Transfer Artifact Generation

After T1/T2 are trained, generate all transfer artifacts for Site D:

```python
# 1. ABC teacher predictions on Site D's FULL history (train + val)
#    These become meta-features for the student model
X_all_d = df_fold[df_fold["site"] == "D"][feature_cols]
teacher_pred_total_d = model_t1.predict(X_all_d).clip(0)
teacher_pred_rate_d  = model_t2.predict(X_all_d).clip(0, 1)

df_fold.loc[df_fold["site"] == "D", "teacher_pred_total"] = teacher_pred_total_d
df_fold.loc[df_fold["site"] == "D", "teacher_pred_rate"]  = teacher_pred_rate_d

# 2. Derived transfer features
df_fold.loc[df_fold["site"] == "D", "teacher_pred_admitted"] = (
    teacher_pred_total_d * teacher_pred_rate_d
)

# 3. Teacher residual on D's training history (how wrong is ABC model on D?)
#    The student can learn the systematic bias pattern
mask_d_train = (df_fold["site"] == "D") & train_mask
if mask_d_train.sum() > 0:
    actual_total_d = df_fold.loc[mask_d_train, "total_enc"]
    teacher_total_d = df_fold.loc[mask_d_train, "teacher_pred_total"]
    df_fold.loc[mask_d_train, "teacher_residual_total"] = actual_total_d - teacher_total_d
    
    # Rolling teacher error (how has the ABC model been trending on D?)
    d_train_sorted = df_fold.loc[mask_d_train].sort_values(["block", "date"])
    for w in [7, 14, 28]:
        df_fold.loc[mask_d_train, f"teacher_residual_roll_{w}"] = (
            d_train_sorted.groupby("block")["teacher_residual_total"]
            .transform(lambda s: s.shift(63).rolling(w, min_periods=1).mean())
        )

# 4. ABC cross-site aggregate priors (what is "normal" at the ABC population level?)
abc_train = df_fold[train_mask & df_fold["site"].isin(["A", "B", "C"])]
abc_month_block_mean = abc_train.groupby(["month", "block"])["total_enc"].mean()
abc_dow_mean = abc_train.groupby(["dow"])["total_enc"].mean()
abc_admit_rate_mean = abc_train.groupby(["month", "block"]).apply(
    lambda g: g["admitted_enc"].sum() / g["total_enc"].sum()
)
# Map these onto Site D rows as "what ABC sites do on similar days"
```

### Eval Notes — Stage 1
- [ ] **ABC model quality**: Print per-site WAPE for A, B, C. Should be similar or slightly better than the global model (no gradient competition from D).
- [ ] **Teacher prediction range on D**: Print min/max/mean of `teacher_pred_total` for Site D. Expected: overpredicts (ABC avg volume > D avg volume). This systematic overprediction IS the signal the student learns to correct.
- [ ] **Teacher residual distribution on D**: Print mean, std, skew of `teacher_residual_total`. Expected: negative mean (ABC model overpredicts D volume). If mean ≈ 0, the teacher is already well-calibrated for D and transfer adds less value.

---

## 4. Stage 2: Site D Student Model

### 4.1 Training Data

```python
# Site D only
train_d = df_fold[train_mask & (df_fold["site"] == "D")]
val_d   = df_fold[val_mask & (df_fold["site"] == "D")]
```

- **Row count**: ~9K–11K training rows per fold (D is 16.4% of data, minus NaN burn-in)
- This is small for GBDT — **aggressive regularization is critical**

### 4.2 Feature Set: D's Own + Transfer Features

```python
# D's own features (same as parent pipeline, minus site categorical)
D_OWN_FEATURES = [
    # Block identifier
    "block",
    # Target lags (D-specific history)
    "lag_63", "lag_70", "lag_77", "lag_91", "lag_182", "lag_364",
    # Rolling stats (D-specific)
    "roll_mean_7", "roll_mean_14", "roll_mean_28", "roll_mean_56", "roll_mean_91",
    "roll_std_7", "roll_std_14", "roll_std_28",
    # Trend deltas
    "delta_7_28", "delta_28_91", "delta_lag_63_70",
    # Calendar (deterministic, always safe)
    "dow", "day", "week_of_year", "month", "quarter", "day_of_year",
    "is_weekend", "is_halloween",
    "dow_sin", "dow_cos", "doy_sin", "doy_cos", "month_sin", "month_cos",
    "days_since_epoch", "year_frac",
    # Holidays
    "is_us_holiday", "days_to_nearest_holiday",
    # School
    "school_in_session",
    # Events
    "event_count",
    # COVID
    "is_covid_era",
    # Target encodings (D-specific baselines)
    "te_site_block_mean_total", "te_site_block_mean_admitted",
    "te_site_admit_rate", "te_site_dow_mean",
]

# Transfer features (from Stage 1)
TRANSFER_FEATURES = [
    # Teacher predictions (informed prior — "what ABC patterns predict for D")
    "teacher_pred_total",
    "teacher_pred_rate",
    "teacher_pred_admitted",
    # Teacher rolling error (how biased is ABC model on D recently?)
    "teacher_residual_roll_7",
    "teacher_residual_roll_14",
    "teacher_residual_roll_28",
    # ABC population priors (what is "normal" for comparable days across ABC?)
    "abc_month_block_mean_total",
    "abc_dow_mean_total",
    "abc_month_block_admit_rate",
]

# For Pipeline E: also include factor features, but REDUCED set
# (factors are noisier for D — only include the top 2-3 by importance)
FACTOR_TRANSFER_FEATURES = [
    "factor_0_pred", "factor_1_pred",       # Top 2 factors only
    "factor_0_momentum", "factor_1_momentum",
]

STUDENT_FEATURE_COLS = D_OWN_FEATURES + TRANSFER_FEATURES
# + FACTOR_TRANSFER_FEATURES if running under Pipeline E
```

**Why this feature set:**
- D's own lags/rolling provide the autocorrelation signal
- `teacher_pred_total` is the most important transfer feature — it encodes "what the ABC temporal patterns predict for this (block, date) combo". The student learns the delta.
- `teacher_residual_roll_*` tell the student "the ABC model has been overpredicting D by ~X recently" — a bias correction signal
- `abc_*_mean` provide population-level baselines the student can anchor to
- Factor features are limited to top 2 (noisier for D, risk of overfitting with small dataset)

### 4.3 Regularization (Critical)

~10K rows is small for LightGBM. Defaults from Pipeline A will overfit badly.

```python
STUDENT_LGBM_PARAMS_S1 = {
    "objective": "tweedie",
    "tweedie_variance_power": 1.5,
    "n_estimators": 800,        # Reduced from 1500
    "num_leaves": 20,           # Much smaller than default 31
    "max_depth": 4,             # Shallow trees
    "min_child_samples": 40,    # High minimum leaf size
    "learning_rate": 0.02,      # Slower learning
    "subsample": 0.7,           # More dropout
    "colsample_bytree": 0.6,    # Feature subsampling
    "reg_lambda": 10.0,         # Strong L2 (doubled from Pipeline A)
    "reg_alpha": 1.0,           # Add L1 for sparsity
    "min_split_gain": 0.1,      # Don't split unless gain is meaningful
    "verbosity": -1,
}

STUDENT_LGBM_PARAMS_S2 = {
    "objective": "regression",
    "n_estimators": 600,
    "num_leaves": 15,
    "max_depth": 4,
    "min_child_samples": 40,
    "learning_rate": 0.02,
    "subsample": 0.7,
    "colsample_bytree": 0.6,
    "reg_lambda": 10.0,
    "reg_alpha": 1.0,
    "min_split_gain": 0.1,
    "verbosity": -1,
}
```

### 4.4 Training Loop

```python
def train_student_d(
    df_fold: pd.DataFrame,
    train_mask: pd.Series,
    val_mask: pd.Series,
    teacher_models: dict,  # {"t1": model_t1, "t2": model_t2}
    feature_cols_parent: list,
    params_s1: dict = None,
    params_s2: dict = None,
) -> dict:
    """Train Site D student models with transfer features."""
    
    p_s1 = (params_s1 or STUDENT_LGBM_PARAMS_S1).copy()
    p_s2 = (params_s2 or STUDENT_LGBM_PARAMS_S2).copy()
    
    # Generate transfer features
    df_fold = generate_transfer_features(df_fold, train_mask, teacher_models, feature_cols_parent)
    
    # Split — Site D only
    mask_d_train = train_mask & (df_fold["site"] == "D")
    mask_d_val   = val_mask & (df_fold["site"] == "D")
    
    train_d = df_fold.loc[mask_d_train].dropna(subset=[f"lag_{LAG_DAYS[-1]}"])
    val_d   = df_fold.loc[mask_d_val]
    
    student_features = [f for f in STUDENT_FEATURE_COLS if f in train_d.columns]
    
    X_train = train_d[student_features]
    X_val   = val_d[student_features]
    
    # Inner validation split for early stopping (last 20% of D training)
    # Using the main val set for early stopping would leak across the fold boundary
    n_inner = max(int(len(X_train) * 0.2), 50)
    X_tr, X_iv = X_train.iloc[:-n_inner], X_train.iloc[-n_inner:]
    y_tr_total, y_iv_total = (
        train_d["total_enc"].iloc[:-n_inner],
        train_d["total_enc"].iloc[-n_inner:],
    )
    y_tr_rate, y_iv_rate = (
        train_d["admit_rate"].iloc[:-n_inner],
        train_d["admit_rate"].iloc[-n_inner:],
    )
    
    # Weights (COVID-aware)
    w_s1 = train_d["sample_weight_a1"].iloc[:-n_inner].values
    w_s2 = train_d["sample_weight_a2"].iloc[:-n_inner].values
    
    # Model S1: total_enc for Site D
    model_s1 = lgb.LGBMRegressor(**p_s1)
    model_s1.fit(
        X_tr, y_tr_total,
        sample_weight=w_s1,
        eval_set=[(X_iv, y_iv_total)],
        callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)],
    )
    
    # Model S2: admit_rate for Site D
    model_s2 = lgb.LGBMRegressor(**p_s2)
    model_s2.fit(
        X_tr, y_tr_rate,
        sample_weight=w_s2,
        eval_set=[(X_iv, y_iv_rate)],
        callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)],
    )
    
    # Predict
    pred_total_d = model_s1.predict(X_val).clip(0)
    pred_rate_d  = model_s2.predict(X_val).clip(0, 1)
    
    # Apply admit-rate guardrails (same bounds as Enhancement C)
    # ... (same logic as existing ADMIT_RATE_BOUNDS for Site D)
    
    return {
        "model_s1": model_s1,
        "model_s2": model_s2,
        "pred_total": pred_total_d,
        "pred_rate": pred_rate_d,
        "student_features": student_features,
    }
```

### 4.5 Blending: Teacher Fallback

As a safety net, blend the student prediction with the teacher prediction, tunable per fold:

```python
# Blend weight α ∈ [0, 1]: α=1 means fully student, α=0 means fully teacher
STUDENT_BLEND_ALPHA = 0.8  # Tune via inner CV

pred_total_d_final = (
    STUDENT_BLEND_ALPHA * student_pred_total
    + (1 - STUDENT_BLEND_ALPHA) * teacher_pred_total_on_val_d
)
```

**Why blend:** The student has a small training set. In folds where D's recent history is unrepresentative (e.g., a local event distorted volumes), the teacher's ABC-derived patterns provide a stable fallback. The blend weight should be tuned — if the student consistently outperforms (α → 1.0 is optimal), disable blending.

### Eval Notes — Stage 2
- [ ] **Student training size**: Print rows per fold after NaN drops. Expected ~8K–10K. If < 5K, increase `min_child_samples` and reduce `num_leaves` further.
- [ ] **Early stopping**: Verify it triggers before max `n_estimators`. If not, reduce `n_estimators`.
- [ ] **Feature importance**: Print top 15 features by gain. **Critical check**: `teacher_pred_total` should be in the top 5 — this validates transfer is working. If it's not in top 10, the student is ignoring the teacher and transfer adds no value.
- [ ] **Overfitting check**: Compare train WAPE vs inner-val WAPE. Gap should be < 3× (e.g., train=0.10, val=0.30 is acceptable; train=0.02, val=0.40 is overfitting).
- [ ] **Blend tuning**: Print Site D WAPE for α ∈ {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}. Optimal α should be > 0.5 (student > teacher) for this to be worthwhile.

---

## 5. Merge & Post-Processing

### 5.1 Combine Predictions

```python
# Sites A, B, C — predictions from Teacher model (T1/T2)
pred_abc = teacher_predictions  # Standard pipeline output for ABC

# Site D — predictions from Student model (S1/S2), optionally blended with teacher
pred_d = student_predictions  # Or blended version

# Merge into single submission
submission = pd.concat([pred_abc, pred_d]).sort_values(["Site", "Date", "Block"])
```

### 5.2 Post-Processing (Same as Current)

- Largest-remainder rounding per (Site, Date)
- Enforce `ED Enc Admitted ≤ ED Enc`
- Clip negatives to 0
- All values integer

### 5.3 Grid Validation

Same eval.md contract — 4 sites × val_days × 4 blocks, no gaps, no duplicates.

---

## 6. Integration Points

### 6.1 Where This Fits in Pipeline A

Modify `step_03_train.py` → `train_fold()`:

```python
def train_fold(df, fold, ...):
    # --- EXISTING: Split data ---
    # ... (unchanged)
    
    # --- NEW: Stage 1 — Train ABC teacher ---
    teacher_result = train_teacher_abc(df_fold, train_mask, val_mask, ...)
    
    # --- NEW: Stage 2 — Train D student with transfer ---
    student_result = train_student_d(df_fold, train_mask, val_mask,
                                      teacher_models=teacher_result["models"], ...)
    
    # --- EXISTING (modified): Merge predictions ---
    # Replace global model predictions for Site D with student predictions
    # Keep teacher predictions for Sites A/B/C
    
    # --- EXISTING: Post-processing, save ---
    # ... (unchanged)
```

This **replaces** the current Enhancement B (hybrid-residual) for Site D. The residual model becomes unnecessary because the student model is directly optimized for D instead of patching a misfit global model.

### 6.2 Where This Fits in Pipeline E

Same integration pattern in `training.py` → `train_fold()`. The key difference: Pipeline E's factor features are included in the teacher model (Stage 1) but **reduced to top 2-3 factors** in the student model (Stage 2) due to Site D's factor noise.

### 6.3 Model Files Per Fold

```
models/
  fold_{k}_teacher_t1.txt       # ABC teacher total_enc
  fold_{k}_teacher_t2.txt       # ABC teacher admit_rate
  fold_{k}_student_s1.txt       # Site D student total_enc
  fold_{k}_student_s2.txt       # Site D student admit_rate
```

This replaces:
```
models/
  fold_{k}_model_a1.txt         # (was global model)
  fold_{k}_model_a2.txt         # (was global model)
  fold_{k}_residual_d.txt       # (Enhancement B — no longer needed)
```

---

## 7. Hyperparameter Tuning

### 7.1 Two-Stage Optuna

**Stage 1 (ABC teacher):** Use the parent pipeline's existing Optuna config. The ABC teacher is essentially the same model minus Site D — no new tuning needed. Use existing `best_params_a1.json` / `best_params_a2.json`.

**Stage 2 (D student):** Separate Optuna search, optimizing Site D WAPE only:

```python
def objective_student(trial):
    params_s1 = {
        "objective": "tweedie",
        "tweedie_variance_power": trial.suggest_float("tvp", 1.1, 1.9),
        "n_estimators": trial.suggest_int("n_est", 400, 1200),
        "num_leaves": trial.suggest_int("leaves", 10, 31),
        "max_depth": trial.suggest_int("depth", 3, 6),
        "min_child_samples": trial.suggest_int("min_child", 20, 80),
        "learning_rate": trial.suggest_float("lr", 0.01, 0.05, log=True),
        "reg_lambda": trial.suggest_float("lambda", 3.0, 20.0),
        "reg_alpha": trial.suggest_float("alpha", 0.0, 5.0),
        "subsample": trial.suggest_float("subsample", 0.5, 0.8),
        "colsample_bytree": trial.suggest_float("colsample", 0.4, 0.8),
    }
    
    blend_alpha = trial.suggest_float("blend_alpha", 0.3, 1.0)
    
    # Score across all 4 folds — Site D WAPE only
    wapes_d = []
    for fold in FOLDS:
        # ... train teacher, generate transfer, train student, predict
        site_d_wape = compute_site_d_admitted_wape(...)
        wapes_d.append(site_d_wape)
    
    return np.mean(wapes_d)
```

**Trials:** 50 trials (smaller search space than full pipeline — fewer rows means faster training). Estimated: ~30–60 minutes.

### 7.2 Collateral Damage Check

After tuning the student, verify ABC WAPE is unchanged:

```python
# ABC predictions come from teacher model (unchanged by student tuning)
assert abc_wape_new == abc_wape_old, "ABC predictions must be identical"
```

This is guaranteed by design — the teacher runs first and ABC predictions are never modified by the student.

---

## 8. Expected Impact

### 8.1 Quantitative Targets

| Metric | Current (Pipeline E) | Target | Rationale |
|--------|---------------------|--------|-----------|
| Site D Admitted WAPE | 0.4810 | **0.40–0.44** | 8–17% relative improvement |
| Sites A/B/C Admitted WAPE | ~0.24–0.30 | **Unchanged** | Teacher = similar to current global model for ABC |
| Overall Admitted WAPE | 0.2779 | **0.265–0.275** | ~1–5% overall improvement |

### 8.2 Why These Targets Are Realistic

- The current global model's Site D error is dominated by **bias** (systematic over/underprediction), not variance. A D-specific model eliminates this bias.
- ~10K training rows is sufficient for a well-regularized GBDT with ~50 features. Kaggle M5 competition showed single-store LightGBM models with similar row counts achieving strong performance.
- Transfer features (teacher predictions) provide an informed prior that compensates for the smaller training set — the student doesn't need to learn temporal patterns from scratch, it learns **corrections to the teacher's estimate**.

### 8.3 Risk: When This Doesn't Help

- If Site D's errors are primarily **noise** (irreducible variance from low volume), not bias → The student model can't improve, and the extra complexity adds nothing. **Diagnostic**: If `teacher_residual_total` has near-zero mean and high variance, bias correction won't help.
- If ~10K rows aren't enough even with regularization → Overfitting. **Diagnostic**: Train/val WAPE gap > 3× consistently across folds.
- If transfer features don't rank in top 10 importance → The student is ignoring the teacher, meaning D's patterns are too different from ABC for transfer to work. **Diagnostic**: Feature importance analysis.

**Fallback:** If the student model doesn't beat the current Enhancement B residual approach, revert to Enhancement B. The implementation should include a comparison flag.

---

## 9. Ablation Plan

Run these comparisons on the 4-fold CV to validate each component:

| Variant | Description | Measures |
|---------|-------------|----------|
| **Baseline** | Current Pipeline E (global + Enhancement B residual) | Site D WAPE = 0.4810 |
| **V1: ABC/D split only** | Separate models, no transfer features | Does D-specific training help at all? |
| **V2: + Teacher predictions** | Add `teacher_pred_total/rate/admitted` | Does transfer learning add signal? |
| **V3: + Teacher residuals** | Add `teacher_residual_roll_*` | Does error-correction signal help? |
| **V4: + ABC priors** | Add `abc_month_block_mean_*` etc. | Do population baselines help? |
| **V5: + Blend tuning** | Optimize `STUDENT_BLEND_ALPHA` per fold | Does teacher fallback improve stability? |
| **V6: Full (V2-V5 + Optuna)** | Everything above with tuned student HPs | Final production version |

Run V1 first — if D-specific training alone doesn't beat Enhancement B, the remaining variants are unlikely to help and we should investigate the root cause differently.

---

## 10. File Changes

### Modified Files

| File | Change |
|------|--------|
| `Pipeline A/step_03_train.py` | Replace Enhancement B residual with two-stage training loop |
| `Pipeline A/config.py` | Add `STUDENT_LGBM_PARAMS_S1`, `STUDENT_LGBM_PARAMS_S2`, `STUDENT_BLEND_ALPHA` |
| `Pipeline E/training.py` | Same two-stage split (if extending to Pipeline E) |
| `Pipeline E/config.py` | Same new constants |

### New Files

| File | Purpose |
|------|---------|
| `Pipeline A/transfer_learning.py` | Contains `train_teacher_abc()`, `train_student_d()`, `generate_transfer_features()`, `blend_predictions()` |
| `Pipeline E/transfer_learning.py` | Pipeline E variant (adds reduced factor feature handling) |

### Removed Behavior

- Enhancement B hybrid-residual model (`residual_d.txt` model files) — superseded by the student model
- Zero-inflation classifier (Enhancement C) — keep as optional; can be applied on top of student predictions if still beneficial

---

## 11. Execution Timeline

| Phase | Task | Estimated Time |
|-------|------|---------------|
| 1 | Implement `transfer_learning.py` for Pipeline A | 2–3 hours |
| 2 | Run V1 ablation (split only, no transfer) | 20 min |
| 3 | Run V2 ablation (+ teacher predictions) | 20 min |
| 4 | If V2 > Baseline: implement full V6, Optuna tuning | 2–3 hours |
| 5 | Port to Pipeline E | 1 hour |
| 6 | Re-run `Pipelines/Eval/run_eval.py` for updated leaderboard | 10 min |

**Total if successful:** ~6–8 hours  
**Early exit (V1 fails):** ~3 hours, revert to Enhancement B

---

## 12. Dependencies

No new libraries. Uses existing:
- `lightgbm` (GBDT training)
- `numpy`, `pandas` (data manipulation)
- `optuna` (tuning, optional)
- `scikit-learn` (LogisticRegression for zero-inflation, if kept)
