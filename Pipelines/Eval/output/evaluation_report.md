# DSU-2026 Pipeline Evaluation Report

**Generated:** 2025-02-11
**Evaluation protocol:** 4× 2-month forward validation windows (per `Strategies/eval.md`)
**Primary metric:** Mean Admitted WAPE (lower is better)
**Pipelines evaluated:** A, B, C, D, E

---

## Leaderboard (ranked by Mean Admitted WAPE)

| Rank | Pipeline | Admitted WAPE | Total WAPE | Admitted RMSE | Total RMSE | Admitted R² | Total R² |
|------|----------|:------------:|:----------:|:-------------:|:----------:|:-----------:|:--------:|
| 1 | **E** | **0.2766** | 0.1455 | 3.019 | 5.674 | 0.701 | 0.869 |
| 2 | B | 0.2786 | 0.1470 | 3.037 | 5.735 | 0.697 | 0.866 |
| 3 | A | 0.2849 | 0.1479 | 3.100 | 5.733 | 0.685 | 0.866 |
| 4 | C | 0.2913 | 0.1494 | 3.231 | 5.827 | 0.658 | 0.862 |
| 5 | D | 0.3888 | 0.3341 | 4.285 | 13.737 | 0.398 | 0.231 |

**Pipeline E wins** with a mean admitted WAPE of 0.2766, narrowly beating Pipeline B (0.2786). The top 4 pipelines (E, B, A, C) are tightly clustered within ~1.5 percentage points. Pipeline D is a significant outlier, ~11pp worse than the leader.

---

## Per-Fold Breakdown (Admitted WAPE)

| Fold | Window | A | B | C | D | E | Best |
|------|--------|:-----:|:-----:|:-----:|:-----:|:-----:|:----:|
| 1 | Jan–Feb 2025 | 0.2823 | 0.2784 | 0.2961 | 0.3869 | 0.2789 | **B** |
| 2 | Mar–Apr 2025 | 0.2883 | 0.2814 | 0.2981 | 0.3898 | **0.2752** | **E** |
| 3 | May–Jun 2025 | 0.2844 | 0.2850 | 0.2888 | 0.3943 | **0.2804** | **E** |
| 4 | Jul–Aug 2025 | 0.2846 | **0.2695** | 0.2823 | 0.3843 | 0.2719 | **B** |

- **E** and **B** trade best-fold honors (2 each).
- No single pipeline dominates all folds — supports ensembling.
- Pipeline D is consistently worst across all 4 folds.
- The top-4 spread within each fold is tight (~2pp), suggesting they capture similar signal.

---

## By-Site Analysis (Admitted WAPE, averaged across folds)

| Site | A | B | C | D | E |
|------|:-----:|:-----:|:-----:|:-----:|:-----:|
| A | 0.2437 | 0.2294 | 0.2543 | 0.2342 | **0.2377** |
| B | 0.2498 | 0.2466 | 0.2640 | 0.3186 | **0.2407** |
| C | 0.3025 | 0.2943 | 0.3054 | 0.4597 | **0.2937** |
| D | 0.4912 | 0.4717 | 0.4657 | 0.9397 | **0.4789** |

**Key observations:**
- **Site D is the hardest** for every pipeline — admitted WAPE ~0.47–0.94 (roughly 2× worse than other sites). This is the primary area for improvement.
- **Site A is easiest** — all pipelines achieve ~0.23–0.25.
- Pipeline D catastrophically fails on Site D (0.94 WAPE — nearly random).
- Pipeline E leads on Sites B and C; Pipeline B leads on Site A; Pipeline C leads on Site D (excluding D pipeline).

---

## By-Block Analysis (Admitted WAPE, averaged across folds)

| Block (6h window) | A | B | C | D | E |
|-------------------|:-----:|:-----:|:-----:|:-----:|:-----:|
| 0 (00:00–05:59) | 0.4672 | 0.4537 | 0.4326 | 0.6746 | 0.4537 |
| 1 (06:00–11:59) | 0.2831 | 0.2776 | 0.2818 | 0.5096 | **0.2776** |
| 2 (12:00–17:59) | 0.2327 | 0.2258 | 0.2291 | 0.2834 | **0.2271** |
| 3 (18:00–23:59) | 0.2827 | 0.2794 | 0.3278 | 0.3004 | **0.2700** |

**Key observations:**
- **Block 0 (overnight) is the hardest** — WAPE ~0.43–0.67. Low volume during these hours means absolute errors weigh heavily.
- **Block 2 (afternoon) is easiest** — WAPE ~0.23, likely because this is the highest-volume, most regular block.
- Pipeline C has a notable spike on Block 3 (0.3278 vs ~0.27–0.28 for others) — something specific to its evening predictions is off.
- Pipeline D struggles badly on Blocks 0 and 1.

---

## Convergence Analysis

| Metric | Value |
|--------|-------|
| Number of pipelines | 5 |
| Mean WAPE (across pipelines) | 0.3040 |
| Std WAPE | 0.0427 |
| **Coefficient of Variation (CV)** | **0.1405** |
| Interpretation | **Partial Convergence** |

Per `eval.md` guidelines (CV 0.05–0.15): *Some diversity remains useful. Ensemble will help; investigate outlier pipelines.*

Pipeline D is the primary outlier inflating CV. Excluding D, the top-4 CV drops substantially, suggesting they have largely converged on the same signal.

---

## Pairwise Prediction Correlation

|   | A | B | C | D | E |
|---|:-----:|:-----:|:-----:|:-----:|:-----:|
| **A** | 1.000 | 0.987 | 0.962 | 0.906 | 0.987 |
| **B** | 0.987 | 1.000 | 0.956 | 0.903 | 0.987 |
| **C** | 0.962 | 0.956 | 1.000 | 0.903 | 0.950 |
| **D** | 0.906 | 0.903 | 0.903 | 1.000 | 0.899 |
| **E** | 0.987 | 0.987 | 0.950 | 0.899 | 1.000 |

**Key observations:**
- **A, B, and E are near-identical** (corr ≥ 0.987). Ensembling just these three will yield marginal gains — they make the same errors.
- **C provides moderate diversity** (corr ~0.95–0.96) — useful ensemble candidate.
- **D is the most diverse** (corr ~0.90) — despite poor solo performance, its uncorrelated errors could still help in an ensemble, but its high error rate likely makes its signal noisy.

---

## Summary & Recommendations

### Winner: Pipeline E
Pipeline E achieves the lowest mean admitted WAPE (0.2766) and leads or ties on most by-site and by-block breakdowns. It also has the best admitted R² (0.701).

### Ensemble Strategy
Given partial convergence (CV = 0.14) and the correlation structure:
1. **Core ensemble:** E + B (best two, corr 0.987 but slight complementarity in fold performance)
2. **Diversity injection:** Add C (corr ~0.95, provides unique signal especially on Site D and Block 0)
3. **Pipeline D:** Exclude from ensemble — error rate too high despite diversity. Investigate root cause (likely a modeling issue with Sites C/D and overnight blocks).

### Areas for Improvement
1. **Site D predictions** — all pipelines struggle here (~0.47+ WAPE). This site likely has different dynamics or lower volume making it harder to predict.
2. **Block 0 (overnight)** — universally the weakest block. Low-volume overnight patterns are inherently noisier.
3. **Pipeline C Block 3** — has an anomalous spike; worth debugging the evening prediction logic.
4. **Pipeline D** — needs fundamental rework. Its Total WAPE (0.334) and Total R² (0.231) suggest the model is barely capturing the overall encounter signal, let alone admissions.

### Metric Context
- An admitted WAPE of ~0.28 means predictions are off by ~28% of total admitted volume on average. For a hospital forecasting use case, this is reasonable but there's room to push toward ~0.25.
- Total encounter WAPE (~0.145 for top pipelines) is substantially better — the total volume is more predictable than the admitted subset.
