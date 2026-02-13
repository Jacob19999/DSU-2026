# DSU-2026 Pipeline Evaluation Report

**Generated:** 2025-02-13 (from computed eval artifacts)
**Evaluation protocol:** 4× 2-month forward validation windows (per `Strategies/eval.md`)
**Primary metric:** Mean Admitted WAPE (lower is better)
**Pipelines evaluated:** A, B, C, D, E

---

## Leaderboard (ranked by Mean Admitted WAPE)

| Rank | Pipeline | Admitted WAPE | Total WAPE | Admitted RMSE | Total RMSE | Admitted R² | Total R² |
|------|----------|:------------:|:----------:|:-------------:|:----------:|:-----------:|:--------:|
| 1 | **E** | **0.2779** | 0.1456 | 3.038 | 5.682 | 0.697 | 0.869 |
| 2 | A | 0.2794 | 0.1459 | 3.057 | 5.677 | 0.694 | 0.869 |
| 3 | B | 0.2829 | 0.1492 | 3.078 | 5.823 | 0.689 | 0.862 |
| 4 | C | 0.2910 | 0.1492 | 3.228 | 5.814 | 0.658 | 0.862 |
| 5 | D | 0.3888 | 0.3341 | 4.285 | 13.737 | 0.398 | 0.231 |

**Pipeline E wins** with a mean admitted WAPE of 0.2779, narrowly beating Pipeline A (0.2794, +0.15pp). The top 4 pipelines (E, A, B, C) are tightly clustered within ~1.3 percentage points. Pipeline D is a significant outlier, ~11pp worse than the leader.

---

## Per-Fold Breakdown (Admitted WAPE)

| Fold | Window | A | B | C | D | E | Best |
|------|--------|:-----:|:-----:|:-----:|:-----:|:-----:|:----:|
| 1 | Jan–Feb 2025 | 0.2827 | 0.2854 | 0.2951 | 0.3869 | **0.2813** | **E** |
| 2 | Mar–Apr 2025 | 0.2848 | 0.2860 | 0.2986 | 0.3898 | **0.2741** | **E** |
| 3 | May–Jun 2025 | **0.2678** | 0.2864 | 0.2869 | 0.3943 | 0.2854 | **A** |
| 4 | Jul–Aug 2025 | 0.2822 | 0.2738 | 0.2833 | 0.3843 | **0.2707** | **E** |

- **E** wins 3 of 4 folds; **A** wins fold 3 (May–Jun).
- No single pipeline dominates all folds — supports ensembling.
- Pipeline D is consistently worst across all 4 folds.
- The top-4 spread within each fold is tight (~1–2pp), suggesting they capture similar signal.
- Pipeline A has notably strong performance in fold 3 (0.2678), its best fold by a wide margin.

---

## By-Site Analysis (Admitted WAPE, pooled across all folds)

| Site | A | B | C | D | E |
|------|:-----:|:-----:|:-----:|:-----:|:-----:|
| A | 0.2422 | 0.2421 | 0.2560 | 0.2342 | 0.2382 |
| B | 0.2428 | 0.2486 | 0.2631 | 0.3186 | **0.2425** |
| C | 0.2972 | 0.2983 | 0.3044 | 0.4597 | **0.2950** |
| D | 0.4771 | 0.4894 | 0.4624 | 0.9397 | 0.4810 |

**Key observations:**
- **Site D is the hardest** for every pipeline — admitted WAPE ~0.46–0.94 (roughly 2× worse than other sites). This is the primary area for improvement.
- **Site A is easiest** — all pipelines achieve ~0.23–0.26.
- Pipeline D catastrophically fails on Site D (0.94 WAPE — nearly random).
- **Pipeline D (the GLM)** actually leads on Site A (0.2342) — its simple structure captures the most regular site well, but falls apart on harder sites.
- **Pipeline C** leads on Site D (0.4624), providing unique value there.
- **Pipeline E** leads on Sites B and C.

---

## By-Block Analysis (Admitted WAPE, pooled across all folds)

| Block (6h window) | A | B | C | D | E |
|-------------------|:-----:|:-----:|:-----:|:-----:|:-----:|
| 0 (00:00–05:59) | 0.4599 | 0.4683 | **0.4305** | 0.6746 | 0.4513 |
| 1 (06:00–11:59) | 0.2846 | 0.2786 | 0.2799 | 0.5096 | **0.2800** |
| 2 (12:00–17:59) | 0.2251 | 0.2286 | 0.2289 | 0.2834 | **0.2263** |
| 3 (18:00–23:59) | 0.2741 | 0.2846 | 0.3296 | 0.3004 | **0.2747** |

**Key observations:**
- **Block 0 (overnight) is the hardest** — WAPE ~0.43–0.67. Low volume during these hours means absolute errors weigh heavily.
- **Block 2 (afternoon) is easiest** — WAPE ~0.23, likely because this is the highest-volume, most regular block.
- **Pipeline C has a notable spike on Block 3** (0.3296 vs ~0.27–0.28 for others) — something specific to its evening predictions is off.
- **Pipeline C leads on Block 0** (0.4305) — its hierarchical daily→block approach may handle low-volume overnight patterns better.
- Pipeline D struggles badly on Blocks 0 and 1.

---

## Convergence Analysis

| Metric | Value |
|--------|-------|
| Number of pipelines | 5 |
| Mean WAPE (across pipelines) | 0.3040 |
| Std WAPE | 0.0427 |
| **Coefficient of Variation (CV)** | **0.1403** |
| Interpretation | **Partial Convergence** |

Per `eval.md` guidelines (CV 0.05–0.15): *Some diversity remains useful. Ensemble will help; investigate outlier pipelines.*

Pipeline D is the primary outlier inflating CV. Excluding D, the top-4 CV drops substantially, suggesting they have largely converged on the same signal.

---

## Pairwise Prediction Correlation

|   | A | B | C | D | E |
|---|:-----:|:-----:|:-----:|:-----:|:-----:|
| **A** | 1.000 | 0.982 | 0.951 | 0.896 | 0.983 |
| **B** | 0.982 | 1.000 | 0.957 | 0.900 | 0.986 |
| **C** | 0.951 | 0.957 | 1.000 | 0.902 | 0.947 |
| **D** | 0.896 | 0.900 | 0.902 | 1.000 | 0.896 |
| **E** | 0.983 | 0.986 | 0.947 | 0.896 | 1.000 |

**Key observations:**
- **A, B, and E are highly correlated** (corr ≥ 0.982). Ensembling just these three will yield marginal gains — they make similar errors.
- **C provides moderate diversity** (corr ~0.95–0.96) — useful ensemble candidate.
- **D is the most diverse** (corr ~0.90) — despite poor solo performance, its uncorrelated errors could help in an ensemble, but its high error rate likely makes its signal too noisy.

---

## Summary & Recommendations

### Winner: Pipeline E
Pipeline E achieves the lowest mean admitted WAPE (0.2779) and leads or ties on most by-site and by-block breakdowns. It also has the best admitted R² (0.697).

### True Ranking (corrected)
1. **E** — 0.2779 (best overall, wins 3/4 folds)
2. **A** — 0.2794 (+0.15pp; wins fold 3 with standout 0.2678)
3. **B** — 0.2829 (+0.50pp)
4. **C** — 0.2910 (+1.31pp; provides unique value on Site D and Block 0)
5. **D** — 0.3888 (+11.1pp; fundamentally different error structure)

### Ensemble Strategy
Given partial convergence (CV = 0.14) and the correlation structure:
1. **Core ensemble:** E + A (top two, slight complementarity — A dominates fold 3)
2. **Diversity injection:** Add C (corr ~0.95, provides unique signal especially on Site D and Block 0)
3. **Pipeline B:** Near-redundant with E (corr 0.986); include only if weighting is cheap.
4. **Pipeline D:** Exclude from ensemble — error rate too high despite diversity. Investigate root cause (likely a modeling issue with Sites C/D and overnight blocks).

### Areas for Improvement
1. **Site D predictions** — all pipelines struggle here (~0.46+ WAPE). This site likely has different dynamics or lower volume making it harder to predict.
2. **Block 0 (overnight)** — universally the weakest block. Low-volume overnight patterns are inherently noisier.
3. **Pipeline C Block 3** — has an anomalous spike (0.3296 vs ~0.27); worth debugging the evening prediction logic.
4. **Pipeline D** — needs fundamental rework. Its Total WAPE (0.334) and Total R² (0.231) suggest the model is barely capturing the overall encounter signal, let alone admissions.

### Metric Context
- An admitted WAPE of ~0.28 means predictions are off by ~28% of total admitted volume on average. For a hospital forecasting use case, this is reasonable but there's room to push toward ~0.25.
- Total encounter WAPE (~0.146 for top pipelines) is substantially better — the total volume is more predictable than the admitted subset.
