"""
Eval: Cross-pipeline comparison and convergence analysis.

Scores all discovered pipelines, ranks by mean admitted WAPE,
and computes convergence diagnostics (CV, pairwise prediction correlation).
See eval.md §Pipeline convergence analysis.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from . import config as cfg
    from .evaluator import build_truth, evaluate_pipeline, wape
except ImportError:
    import config as cfg  # type: ignore[no-redef]
    from evaluator import build_truth, evaluate_pipeline, wape  # type: ignore[no-redef]


# ═══════════════════════════════════════════════════════════════════════════════
#  Score all pipelines
# ═══════════════════════════════════════════════════════════════════════════════

def score_all_pipelines(
    pipeline_names: Optional[List[str]] = None,
) -> Dict[str, Dict]:
    """
    Evaluate every discovered pipeline (or a subset).
    Returns {pipeline_name: result_dict} for pipelines with predictions.
    """
    truth = build_truth()

    if pipeline_names is None:
        discovered = cfg.discover_pipelines()
        pipeline_names = sorted(discovered.keys())

    if not pipeline_names:
        print("  No pipelines with predictions found!")
        return {}

    results = {}
    for name in pipeline_names:
        print(f"\n{'-'*60}")
        print(f"  Scoring Pipeline {name} ...")
        res = evaluate_pipeline(name, truth)
        if res is not None:
            results[name] = res
        else:
            print(f"  Pipeline {name}: no valid predictions found.")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  Leaderboard (ranking table)
# ═══════════════════════════════════════════════════════════════════════════════

def build_leaderboard(results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Build a ranked comparison table across all scored pipelines.
    Primary ranking: mean admitted WAPE (ascending).
    """
    rows = []
    for name, res in results.items():
        ft = res["fold_table"]
        mean_row = ft[ft["fold_id"] == "mean"].iloc[0]
        rows.append({
            "pipeline":          name,
            "n_folds":           res["n_folds"],
            "admitted_wape":     mean_row["primary_admitted_wape"],
            "total_wape":        mean_row["total_wape"],
            "admitted_rmse":     mean_row["admitted_rmse"],
            "total_rmse":        mean_row["total_rmse"],
            "admitted_mae":      mean_row["admitted_mae"],
            "total_mae":         mean_row["total_mae"],
            "admitted_r2":       mean_row["admitted_r2"],
            "total_r2":          mean_row["total_r2"],
        })

    lb = pd.DataFrame(rows).sort_values("admitted_wape").reset_index(drop=True)
    lb.index = lb.index + 1  # 1-based rank
    lb.index.name = "rank"
    return lb


# ═══════════════════════════════════════════════════════════════════════════════
#  Convergence analysis (eval.md §Pipeline convergence analysis)
# ═══════════════════════════════════════════════════════════════════════════════

def convergence_analysis(results: Dict[str, Dict]) -> Dict[str, object]:
    """
    Compute coefficient of variation of pipeline WAPEs and interpret.
    Requires ≥ 2 scored pipelines.
    """
    wapes = {name: res["mean_admitted_wape"] for name, res in results.items()}

    if len(wapes) < 2:
        return {
            "n_pipelines": len(wapes),
            "wapes": wapes,
            "cv": float("nan"),
            "interpretation": "Need ≥ 2 pipelines for convergence analysis.",
        }

    vals = np.array(list(wapes.values()))
    mu = float(np.mean(vals))
    sigma = float(np.std(vals, ddof=0))
    cv = sigma / mu if mu > 0 else float("nan")

    if cv < 0.05:
        interp = "CONVERGED -- dataset predictive ceiling likely reached. Focus on ensemble post-processing."
    elif cv < 0.15:
        interp = "PARTIAL CONVERGENCE -- some diversity remains. Ensemble will help; investigate outlier pipelines."
    else:
        interp = "DIVERGENT -- pipelines capture different signals. Strong ensemble gains expected."

    return {
        "n_pipelines":    len(wapes),
        "wapes":          wapes,
        "mean_wape":      mu,
        "std_wape":       sigma,
        "cv":             cv,
        "interpretation": interp,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Pairwise prediction correlation
# ═══════════════════════════════════════════════════════════════════════════════

def pairwise_correlation(results: Dict[str, Dict]) -> Optional[pd.DataFrame]:
    """
    Compute pairwise Pearson correlation of predicted admitted counts.
    High corr (> 0.95) means pipelines make similar errors → ensemble gains marginal.
    """
    if len(results) < 2:
        return None

    # Build a matrix: rows = (Site, Date, Block), cols = pipeline predictions
    pred_series = {}
    for name, res in results.items():
        joined = res["_all_joined"]
        # Use admitted predictions keyed by (Site, Date, Block)
        keyed = joined.set_index(["Site", "Date", "Block"])["ED Enc Admitted_pred"]
        pred_series[name] = keyed

    pred_df = pd.DataFrame(pred_series)
    # Only keep rows present in ALL pipelines
    pred_df = pred_df.dropna()

    if len(pred_df) < 10:
        return None

    corr = pred_df.corr()
    return corr


# ═══════════════════════════════════════════════════════════════════════════════
#  Per-fold comparison (which pipeline wins each fold?)
# ═══════════════════════════════════════════════════════════════════════════════

def per_fold_comparison(results: Dict[str, Dict]) -> pd.DataFrame:
    """Build a table showing admitted WAPE per pipeline × fold."""
    rows = []
    for name, res in results.items():
        ft = res["fold_table"]
        for _, row in ft.iterrows():
            if row["fold_id"] == "mean":
                continue
            rows.append({
                "pipeline": name,
                "fold_id":  int(row["fold_id"]),
                "admitted_wape": row["primary_admitted_wape"],
                "total_wape":    row["total_wape"],
            })
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    pivot = df.pivot(index="fold_id", columns="pipeline", values="admitted_wape")
    pivot["best"] = pivot.idxmin(axis=1)
    return pivot


# ═══════════════════════════════════════════════════════════════════════════════
#  Print full report
# ═══════════════════════════════════════════════════════════════════════════════

def print_comparison_report(results: Dict[str, Dict]) -> None:
    """Print a comprehensive cross-pipeline comparison to stdout."""
    sep = "=" * 72

    print(f"\n{sep}")
    print("  DSU-2026: CROSS-PIPELINE EVALUATION REPORT")
    print(sep)

    # Leaderboard
    lb = build_leaderboard(results)
    print(f"\n LEADERBOARD (ranked by mean admitted WAPE, {len(lb)} pipelines):\n")
    print(lb.to_string())

    # Per-fold comparison
    pf = per_fold_comparison(results)
    if len(pf) > 0:
        print(f"\n PER-FOLD ADMITTED WAPE:\n")
        print(pf.to_string())

    # Per-pipeline detail
    for name in sorted(results.keys()):
        res = results[name]
        print(f"\n{'-'*60}")
        print(f"  PIPELINE {name} -- {res['n_folds']}-fold detail")
        print(f"{'-'*60}")

        print("\n  Per-fold:")
        ft = res["fold_table"]
        cols_show = ["fold_id", "window", "primary_admitted_wape", "total_wape",
                     "admitted_rmse", "total_rmse"]
        cols_avail = [c for c in cols_show if c in ft.columns]
        print(ft[cols_avail].to_string(index=False))

        if not res["by_site"].empty:
            print("\n  By-site:")
            print(res["by_site"].to_string(index=False))

        if not res["by_block"].empty:
            print("\n  By-block:")
            print(res["by_block"].to_string(index=False))

    # Convergence
    conv = convergence_analysis(results)
    print(f"\n{'-'*60}")
    print(f"  CONVERGENCE ANALYSIS")
    print(f"{'-'*60}")
    print(f"  Pipelines scored : {conv['n_pipelines']}")
    if not np.isnan(conv.get("cv", float("nan"))):
        print(f"  Mean WAPE        : {conv['mean_wape']:.4f}")
        print(f"  Std WAPE         : {conv['std_wape']:.4f}")
        print(f"  CV               : {conv['cv']:.4f}")
        print(f"  >> {conv['interpretation']}")
    else:
        print(f"  {conv['interpretation']}")

    # Pairwise correlation
    corr = pairwise_correlation(results)
    if corr is not None:
        print(f"\n  Pairwise admitted prediction correlation:")
        print(corr.to_string())
        # Flag high-correlation pairs
        names = list(corr.columns)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                c = corr.iloc[i, j]
                if c > 0.95:
                    print(f"  WARNING: {names[i]} vs {names[j]}: corr={c:.3f} (> 0.95 -- similar errors, limited ensemble gain)")

    print(f"\n{sep}\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  Save reports to disk
# ═══════════════════════════════════════════════════════════════════════════════

def save_reports(results: Dict[str, Dict]) -> None:
    """Persist all evaluation artifacts to Pipelines/Eval/output/."""
    cfg.ensure_dirs()
    out = cfg.OUTPUT_DIR

    # Leaderboard CSV
    lb = build_leaderboard(results)
    lb.to_csv(out / "leaderboard.csv")
    print(f"  Saved: {out / 'leaderboard.csv'}")

    # Per-fold comparison CSV
    pf = per_fold_comparison(results)
    if len(pf) > 0:
        pf.to_csv(out / "per_fold_comparison.csv")
        print(f"  Saved: {out / 'per_fold_comparison.csv'}")

    # Convergence JSON
    conv = convergence_analysis(results)
    conv_safe = {k: v for k, v in conv.items()}
    with open(out / "convergence.json", "w") as f:
        json.dump(conv_safe, f, indent=2, default=str)
    print(f"  Saved: {out / 'convergence.json'}")

    # Pairwise correlation CSV
    corr = pairwise_correlation(results)
    if corr is not None:
        corr.to_csv(out / "pairwise_correlation.csv")
        print(f"  Saved: {out / 'pairwise_correlation.csv'}")

    # Per-pipeline detail
    for name, res in results.items():
        pipe_dir = out / f"pipeline_{name}"
        pipe_dir.mkdir(parents=True, exist_ok=True)

        res["fold_table"].to_csv(pipe_dir / "fold_results.csv", index=False)
        if not res["by_site"].empty:
            res["by_site"].to_csv(pipe_dir / "by_site.csv", index=False)
        if not res["by_block"].empty:
            res["by_block"].to_csv(pipe_dir / "by_block.csv", index=False)

        # JSON summary
        summary = {
            "pipeline":          name,
            "n_folds":           res["n_folds"],
            "mean_admitted_wape": res["mean_admitted_wape"],
            "mean_total_wape":   res["mean_total_wape"],
        }
        with open(pipe_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"  Saved: {pipe_dir}/")
