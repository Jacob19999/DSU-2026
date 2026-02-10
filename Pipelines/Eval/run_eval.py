"""
Eval: CLI entry point — score one or all pipelines.

Usage:
    # Score all discovered pipelines + cross-pipeline comparison
    python -m Pipelines.Eval.run_eval

    # Score specific pipeline(s) only
    python -m Pipelines.Eval.run_eval --pipelines A B

    # Score all but skip saving reports to disk
    python -m Pipelines.Eval.run_eval --no-save

    # Score single pipeline with full detail
    python -m Pipelines.Eval.run_eval --pipelines A --detail
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure Eval package is importable when run as module or script
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as cfg
from compare import (
    build_leaderboard,
    convergence_analysis,
    pairwise_correlation,
    per_fold_comparison,
    print_comparison_report,
    save_reports,
    score_all_pipelines,
)
from evaluator import build_truth, evaluate_pipeline


def _print_single_pipeline(res: dict) -> None:
    """Detailed report for a single pipeline."""
    sep = "=" * 62
    name = res["pipeline"]

    print(f"\n{sep}")
    print(f"  PIPELINE {name} -- EVALUATION REPORT ({res['n_folds']}-fold)")
    print(sep)

    # Primary metric
    print(f"\n  OVERALL (mean across {res['n_folds']} folds):")
    print(f"    Primary Admitted WAPE : {res['mean_admitted_wape']:.4f}")
    print(f"    Total WAPE           : {res['mean_total_wape']:.4f}")

    # Per-fold table
    ft = res["fold_table"]
    print(f"\n  PER-FOLD:")
    cols = ["fold_id", "window", "primary_admitted_wape", "total_wape",
            "admitted_rmse", "total_rmse", "admitted_r2", "total_r2"]
    cols_avail = [c for c in cols if c in ft.columns]
    print(ft[cols_avail].to_string(index=False))

    # By-site
    if not res["by_site"].empty:
        print(f"\n  BY-SITE (aggregated across all folds):")
        print(res["by_site"].to_string(index=False))

    # By-block
    if not res["by_block"].empty:
        print(f"\n  BY-BLOCK (aggregated across all folds):")
        print(res["by_block"].to_string(index=False))

    # Sanity warnings
    if not res["by_site"].empty:
        site_wapes = res["by_site"]["admitted_wape"].dropna()
        if len(site_wapes) >= 2:
            best, worst = site_wapes.min(), site_wapes.max()
            if worst > 2 * best:
                print(f"\n  WARNING: worst site WAPE ({worst:.4f}) > 2x best ({best:.4f}) -- site imbalance")

    print(sep)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DSU-2026 Centralized Pipeline Evaluation",
    )
    parser.add_argument(
        "--pipelines", "-p", nargs="+", default=None,
        help="Pipeline names to score (e.g. A B C). Default: all discovered.",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Skip saving reports to disk.",
    )
    parser.add_argument(
        "--detail", action="store_true",
        help="Print full per-pipeline detail (useful with --pipelines).",
    )
    args = parser.parse_args()

    # Discover what's available
    discovered = cfg.discover_pipelines()
    if not discovered:
        print("No pipelines with prediction CSVs found!")
        print("Expected directories:")
        for name, dirs in cfg.PIPELINE_PRED_DIRS.items():
            for d in dirs:
                print(f"  Pipeline {name}: {d}")
        sys.exit(1)

    print(f"Discovered pipelines with predictions: {sorted(discovered.keys())}")
    target_pipelines = args.pipelines or sorted(discovered.keys())

    # Validate requested pipelines exist
    missing = [p for p in target_pipelines if p not in discovered]
    if missing:
        print(f"WARNING: No predictions found for pipeline(s): {missing}")
        target_pipelines = [p for p in target_pipelines if p in discovered]
        if not target_pipelines:
            print("Nothing to evaluate!")
            sys.exit(1)

    # Score
    results = score_all_pipelines(target_pipelines)
    if not results:
        print("No pipelines produced valid scores.")
        sys.exit(1)

    # Report
    if len(results) == 1 or args.detail:
        for res in results.values():
            _print_single_pipeline(res)

    if len(results) >= 2:
        print_comparison_report(results)
    elif len(results) == 1:
        name = list(results.keys())[0]
        print(f"\n  Only 1 pipeline scored ({name}). "
              f"Score more pipelines for cross-pipeline comparison.")

    # Save
    if not args.no_save:
        print(f"\nSaving reports to {cfg.OUTPUT_DIR} ...")
        save_reports(results)

    print("\nDone.")


if __name__ == "__main__":
    main()
