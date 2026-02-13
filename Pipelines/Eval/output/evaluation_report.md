# DSU-2026 Pipeline Evaluation Report

**Generated:** 2026-02-13
**Evaluation protocol:** 4× 2-month forward validation windows (per `Strategies/eval.md`)
**Primary metric:** Mean Admitted WAPE (lower is better)
**Pipelines evaluated:** A, B, C, D, E

---

## Leaderboard (ranked by Mean Admitted WAPE)

| Rank | Pipeline | Admitted WAPE | Total WAPE | Admitted RMSE | Total RMSE | Admitted R² | Total R² |
|------|----------|:------------:|:----------:|:-------------:|:----------:|:-----------:|:--------:|
| 1 | **A** | **0.2794** | 0.1454 | 3.054 | 5.672 | 0.694 | 0.869 |
| 2 | B | 0.2829 | 0.1492 | 3.078 | 5.823 | 0.689 | 0.862 |
| 3 | C | 0.2910 | 0.1492 | 3.228 | 5.814 | 0.658 | 0.862 |
| 4 | E | 0.2928 | 0.1582 | 3.122 | 6.053 | 0.680 | 0.851 |
| 5 | D | 0.3888 | 0.3341 | 4.285 | 13.737 | 0.398 | 0.231 |

**Pipeline A wins** with a mean admitted WAPE of 0.2794.

---

## Per-Fold Breakdown (Admitted WAPE)

| Fold | Window | A | B | C | D | E | Best |
|------|--------|:-----:|:-----:|:-----:|:-----:|:-----:|:----:|
| 1 | Jan–Feb 2025 | **0.2816** | 0.2854 | 0.2951 | 0.3869 | 0.2974 | **A** |
| 2 | Mar–Apr 2025 | **0.2845** | 0.2860 | 0.2986 | 0.3898 | 0.2934 | **A** |
| 3 | May–Jun 2025 | **0.2696** | 0.2864 | 0.2869 | 0.3943 | 0.2991 | **A** |
| 4 | Jul–Aug 2025 | 0.2817 | **0.2738** | 0.2833 | 0.3843 | 0.2812 | **B** |

---

## By-Site Analysis (Admitted WAPE, averaged across folds)

| Site | A | B | C | D | E |
|------|:-----:|:-----:|:-----:|:-----:|:-----:|
| A | 0.2422 | 0.2421 | 0.2560 | **0.2342** | 0.2374 |
| B | **0.2447** | 0.2486 | 0.2631 | 0.3186 | 0.2477 |
| C | 0.2971 | 0.2983 | 0.3044 | 0.4597 | **0.2962** |
| D | 0.4711 | 0.4894 | **0.4624** | 0.9397 | 0.6102 |

---

## By-Block Analysis (Admitted WAPE, averaged across folds)

| Block (6h window) | A | B | C | D | E |
|-------------------|:-----:|:-----:|:-----:|:-----:|:-----:|
| 0 (00:00–05:59) | 0.4591 | 0.4683 | **0.4305** | 0.6746 | 0.5409 |
| 1 (06:00–11:59) | 0.2798 | **0.2786** | 0.2799 | 0.5096 | 0.2847 |
| 2 (12:00–17:59) | 0.2281 | 0.2286 | 0.2289 | 0.2834 | **0.2264** |
| 3 (18:00–23:59) | **0.2749** | 0.2846 | 0.3296 | 0.3004 | 0.2884 |

---

## Convergence Analysis

| Metric | Value |
|--------|-------|
| Number of pipelines | 5 |
| Mean WAPE (across pipelines) | 0.3070 |
| Std WAPE | 0.0412 |
| **Coefficient of Variation (CV)** | **0.1343** |
| Interpretation | **Partial Convergence** |

---

## Pairwise Prediction Correlation

|   | A | B | C | D | E |
|---|:-----:|:-----:|:-----:|:-----:|:-----:|
| **A** | 1.000 | 0.983 | 0.955 | 0.899 | 0.980 |
| **B** | 0.983 | 1.000 | 0.957 | 0.900 | 0.984 |
| **C** | 0.955 | 0.957 | 1.000 | 0.902 | 0.952 |
| **D** | 0.899 | 0.900 | 0.902 | 1.000 | 0.910 |
| **E** | 0.980 | 0.984 | 0.952 | 0.910 | 1.000 |

