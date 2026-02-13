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

    # Markdown report
    md_report = generate_markdown_report(results)
    md_path = out / "evaluation_report.md"
    md_path.write_text(md_report, encoding="utf-8")
    print(f"  Saved: {md_path}")


def generate_markdown_report(results: Dict[str, Dict]) -> str:
    """Generate evaluation_report.md from current results."""
    from datetime import date

    lb = build_leaderboard(results)
    pf = per_fold_comparison(results)
    conv = convergence_analysis(results)
    corr = pairwise_correlation(results)

    lines = []
    lines.append("# DSU-2026 Pipeline Evaluation Report\n")
    lines.append(f"**Generated:** {date.today().isoformat()}")
    lines.append("**Evaluation protocol:** 4\u00d7 2-month forward validation windows (per `Strategies/eval.md`)")
    lines.append("**Primary metric:** Mean Admitted WAPE (lower is better)")
    lines.append(f"**Pipelines evaluated:** {', '.join(sorted(results.keys()))}\n")
    lines.append("---\n")

    # ── Leaderboard ──
    lines.append("## Leaderboard (ranked by Mean Admitted WAPE)\n")
    lines.append("| Rank | Pipeline | Admitted WAPE | Total WAPE | Admitted RMSE | Total RMSE | Admitted R\u00b2 | Total R\u00b2 |")
    lines.append("|------|----------|:------------:|:----------:|:-------------:|:----------:|:-----------:|:--------:|")
    for rank, row in lb.iterrows():
        name = row["pipeline"]
        bold = "**" if rank == 1 else ""
        lines.append(
            f"| {rank} | {bold}{name}{bold} | {bold}{row['admitted_wape']:.4f}{bold} | "
            f"{row['total_wape']:.4f} | {row['admitted_rmse']:.3f} | {row['total_rmse']:.3f} | "
            f"{row['admitted_r2']:.3f} | {row['total_r2']:.3f} |"
        )
    winner = lb.iloc[0]["pipeline"]
    w_wape = lb.iloc[0]["admitted_wape"]
    lines.append(f"\n**Pipeline {winner} wins** with a mean admitted WAPE of {w_wape:.4f}.\n")
    lines.append("---\n")

    # ── Per-Fold ──
    if len(pf) > 0:
        lines.append("## Per-Fold Breakdown (Admitted WAPE)\n")
        fold_windows = {
            1: "Jan\u2013Feb 2025", 2: "Mar\u2013Apr 2025",
            3: "May\u2013Jun 2025", 4: "Jul\u2013Aug 2025",
        }
        pipe_names = sorted([c for c in pf.columns if c != "best"])
        header = "| Fold | Window | " + " | ".join(pipe_names) + " | Best |"
        sep_row = "|------|--------|" + "|".join([":-----:" for _ in pipe_names]) + "|:----:|"
        lines.append(header)
        lines.append(sep_row)
        for fold_id, row in pf.iterrows():
            best = row["best"]
            cells = []
            for p in pipe_names:
                val = row[p]
                if p == best:
                    cells.append(f"**{val:.4f}**")
                else:
                    cells.append(f"{val:.4f}")
            window = fold_windows.get(fold_id, "")
            lines.append(f"| {fold_id} | {window} | " + " | ".join(cells) + f" | **{best}** |")
        lines.append("")
        lines.append("---\n")

    # ── By-Site ──
    lines.append("## By-Site Analysis (Admitted WAPE, averaged across folds)\n")
    pipe_names_sorted = sorted(results.keys())
    header = "| Site | " + " | ".join(pipe_names_sorted) + " |"
    sep_row = "|------|" + "|".join([":-----:" for _ in pipe_names_sorted]) + "|"
    lines.append(header)
    lines.append(sep_row)
    site_data = {}
    for name, res in results.items():
        if not res["by_site"].empty:
            for _, row in res["by_site"].iterrows():
                site = row["Site"]
                if site not in site_data:
                    site_data[site] = {}
                site_data[site][name] = row["admitted_wape"]
    for site in sorted(site_data.keys()):
        vals = site_data[site]
        best_p = min(vals, key=vals.get)
        cells = []
        for p in pipe_names_sorted:
            v = vals.get(p, float("nan"))
            if p == best_p:
                cells.append(f"**{v:.4f}**")
            else:
                cells.append(f"{v:.4f}")
        lines.append(f"| {site} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("---\n")

    # ── By-Block ──
    lines.append("## By-Block Analysis (Admitted WAPE, averaged across folds)\n")
    block_labels = {0: "0 (00:00\u201305:59)", 1: "1 (06:00\u201311:59)", 2: "2 (12:00\u201317:59)", 3: "3 (18:00\u201323:59)"}
    header = "| Block (6h window) | " + " | ".join(pipe_names_sorted) + " |"
    sep_row = "|-------------------|" + "|".join([":-----:" for _ in pipe_names_sorted]) + "|"
    lines.append(header)
    lines.append(sep_row)
    block_data = {}
    for name, res in results.items():
        if not res["by_block"].empty:
            for _, row in res["by_block"].iterrows():
                blk = row["Block"]
                if blk not in block_data:
                    block_data[blk] = {}
                block_data[blk][name] = row["admitted_wape"]
    for blk in sorted(block_data.keys()):
        vals = block_data[blk]
        best_p = min(vals, key=vals.get)
        cells = []
        for p in pipe_names_sorted:
            v = vals.get(p, float("nan"))
            if p == best_p:
                cells.append(f"**{v:.4f}**")
            else:
                cells.append(f"{v:.4f}")
        label = block_labels.get(blk, str(blk))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("---\n")

    # ── Convergence ──
    lines.append("## Convergence Analysis\n")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Number of pipelines | {conv['n_pipelines']} |")
    if not np.isnan(conv.get("cv", float("nan"))):
        lines.append(f"| Mean WAPE (across pipelines) | {conv['mean_wape']:.4f} |")
        lines.append(f"| Std WAPE | {conv['std_wape']:.4f} |")
        lines.append(f"| **Coefficient of Variation (CV)** | **{conv['cv']:.4f}** |")
        if conv["cv"] < 0.05:
            interp_label = "Converged"
        elif conv["cv"] < 0.15:
            interp_label = "Partial Convergence"
        else:
            interp_label = "Divergent"
        lines.append(f"| Interpretation | **{interp_label}** |")
    lines.append("")
    lines.append("---\n")

    # ── Pairwise Correlation ──
    if corr is not None:
        lines.append("## Pairwise Prediction Correlation\n")
        names = list(corr.columns)
        header = "|   | " + " | ".join(names) + " |"
        sep_row = "|---|" + "|".join([":-----:" for _ in names]) + "|"
        lines.append(header)
        lines.append(sep_row)
        for name in names:
            cells = [f"{corr.loc[name, n]:.3f}" for n in names]
            lines.append(f"| **{name}** | " + " | ".join(cells) + " |")
        lines.append("")

    return "\n".join(lines) + "\n"
