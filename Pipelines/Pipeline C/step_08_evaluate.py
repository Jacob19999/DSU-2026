"""
Step 8: Evaluate Pipeline C predictions against the eval.md contract.

Loads fold CSVs from output/, aggregates ground truth from raw visits CSV,
and produces an evaluation report with overall, per-fold, per-site, per-block,
and hierarchical decomposition diagnostics.
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd

import config as cfg


# ── Metrics ──────────────────────────────────────────────────────────────────

def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.sum(np.abs(y_true)))
    return float(np.sum(np.abs(y_true - y_pred)) / denom) if denom > 0 else float("nan")


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


# ── Ground truth builder ────────────────────────────────────────────────────

def _build_truth() -> pd.DataFrame:
    """Aggregate raw visits to (Site, Date, Block) grain — same as eval.md."""
    print(f"  Loading ground truth from {cfg.RAW_VISITS_CSV} ...")
    raw = pd.read_csv(cfg.RAW_VISITS_CSV)
    raw["Date"] = pd.to_datetime(raw["Date"])
    raw["Block"] = (raw["Hour"] // 6).astype(int)
    truth = (
        raw.groupby(["Site", "Date", "Block"], as_index=False)
        .agg({"ED Enc": "sum", "ED Enc Admitted": "sum"})
    )
    truth["Date"] = truth["Date"].dt.strftime("%Y-%m-%d")
    return truth


def _build_truth_daily() -> pd.DataFrame:
    """Aggregate raw visits to (Site, Date) grain for daily-level diagnostics."""
    raw = pd.read_csv(cfg.RAW_VISITS_CSV)
    raw["Date"] = pd.to_datetime(raw["Date"])
    truth = (
        raw.groupby(["Site", "Date"], as_index=False)
        .agg({"ED Enc": "sum", "ED Enc Admitted": "sum"})
    )
    truth["Date"] = truth["Date"].dt.strftime("%Y-%m-%d")
    return truth


# ── Score one fold ──────────────────────────────────────────────────────────

def _score_fold(truth: pd.DataFrame, fold: dict) -> dict | None:
    csv_path = cfg.OUTPUT_DIR / f"fold_{fold['id']}_predictions.csv"
    if not csv_path.exists():
        print(f"  WARNING: {csv_path} not found — skipping fold {fold['id']}")
        return None

    pred = pd.read_csv(csv_path)
    pred["Date"] = pd.to_datetime(pred["Date"]).dt.strftime("%Y-%m-%d")

    truth_w = truth[
        (truth["Date"] >= fold["val_start"]) & (truth["Date"] <= fold["val_end"])
    ]

    merged = pred.merge(truth_w, on=["Site", "Date", "Block"],
                        suffixes=("_pred", "_true"), how="inner")

    if len(merged) == 0:
        print(f"  WARNING: No matching truth rows for fold {fold['id']}")
        return None

    yt = merged["ED Enc_true"].values
    yp = merged["ED Enc_pred"].values
    at = merged["ED Enc Admitted_true"].values
    ap = merged["ED Enc Admitted_pred"].values

    return {
        "fold_id": fold["id"],
        "window": f"{fold['val_start']}..{fold['val_end']}",
        "total_wape": wape(yt, yp),
        "admitted_wape": wape(at, ap),
        "total_rmse": rmse(yt, yp),
        "admitted_rmse": rmse(at, ap),
        "total_mae": mae(yt, yp),
        "admitted_mae": mae(at, ap),
        "n_rows": len(merged),
        "_merged": merged,
    }


# ── Daily-level diagnostic ──────────────────────────────────────────────────

def _compute_daily_wape(truth_daily: pd.DataFrame, fold: dict) -> dict | None:
    """Compute daily-level WAPE from block-level predictions (sum blocks → daily)."""
    csv_path = cfg.OUTPUT_DIR / f"fold_{fold['id']}_predictions.csv"
    if not csv_path.exists():
        return None

    pred = pd.read_csv(csv_path)
    pred["Date"] = pd.to_datetime(pred["Date"]).dt.strftime("%Y-%m-%d")

    pred_daily = (
        pred.groupby(["Site", "Date"], as_index=False)
        .agg({"ED Enc": "sum", "ED Enc Admitted": "sum"})
    )

    truth_w = truth_daily[
        (truth_daily["Date"] >= fold["val_start"]) & (truth_daily["Date"] <= fold["val_end"])
    ]

    merged = pred_daily.merge(truth_w, on=["Site", "Date"],
                              suffixes=("_pred", "_true"), how="inner")

    if len(merged) == 0:
        return None

    return {
        "daily_total_wape": wape(merged["ED Enc_true"].values, merged["ED Enc_pred"].values),
        "daily_admitted_wape": wape(merged["ED Enc Admitted_true"].values,
                                   merged["ED Enc Admitted_pred"].values),
    }


# ── Main entry point ────────────────────────────────────────────────────────

def evaluate() -> dict:
    """Score all folds, print full report, save JSON."""
    truth = _build_truth()
    truth_daily = _build_truth_daily()

    fold_results = []
    daily_diagnostics = []
    for fold in cfg.FOLDS:
        res = _score_fold(truth, fold)
        if res is not None:
            fold_results.append(res)
        dd = _compute_daily_wape(truth_daily, fold)
        if dd is not None:
            daily_diagnostics.append(dd)

    if not fold_results:
        print("  No fold predictions found!")
        return {}

    # ── Aggregate ────────────────────────────────────────────────────────
    mean_tw = np.mean([r["total_wape"] for r in fold_results])
    mean_aw = np.mean([r["admitted_wape"] for r in fold_results])
    mean_tr = np.mean([r["total_rmse"] for r in fold_results])
    mean_ar = np.mean([r["admitted_rmse"] for r in fold_results])

    all_merged = pd.concat([r["_merged"] for r in fold_results], ignore_index=True)

    # Daily-level diagnostics
    mean_daily_tw = np.mean([d["daily_total_wape"] for d in daily_diagnostics]) if daily_diagnostics else float("nan")
    mean_daily_aw = np.mean([d["daily_admitted_wape"] for d in daily_diagnostics]) if daily_diagnostics else float("nan")
    alloc_delta = mean_tw - mean_daily_tw if daily_diagnostics else float("nan")

    # ── Report ───────────────────────────────────────────────────────────
    sep = "=" * 65
    print(f"\n{sep}")
    print(" PIPELINE C: HIERARCHICAL RECONCILIATION — EVALUATION REPORT")
    print(sep)

    print(f"\n OVERALL ({len(fold_results)}-fold mean):")
    print(f"   Primary Admitted WAPE : {mean_aw:.4f}")
    print(f"   Total WAPE           : {mean_tw:.4f}")
    print(f"   Admitted RMSE        : {mean_ar:.2f}")
    print(f"   Total RMSE           : {mean_tr:.2f}")

    print(f"\n PER-FOLD:")
    for r in fold_results:
        print(f"   Fold {r['fold_id']} ({r['window']}): "
              f"admitted_wape={r['admitted_wape']:.4f}  "
              f"total_wape={r['total_wape']:.4f}  rows={r['n_rows']}")

    print(f"\n BY-SITE:")
    for site in cfg.SITES:
        sub = all_merged[all_merged["Site"] == site]
        if len(sub) == 0:
            continue
        tw = wape(sub["ED Enc_true"].values, sub["ED Enc_pred"].values)
        aw = wape(sub["ED Enc Admitted_true"].values, sub["ED Enc Admitted_pred"].values)
        print(f"   Site {site}: admitted_wape={aw:.4f}  total_wape={tw:.4f}")

    print(f"\n BY-BLOCK:")
    for blk in cfg.BLOCKS:
        sub = all_merged[all_merged["Block"] == blk]
        if len(sub) == 0:
            continue
        tw = wape(sub["ED Enc_true"].values, sub["ED Enc_pred"].values)
        aw = wape(sub["ED Enc Admitted_true"].values, sub["ED Enc Admitted_pred"].values)
        print(f"   Block {blk}: admitted_wape={aw:.4f}  total_wape={tw:.4f}")

    print(f"\n HIERARCHICAL DECOMPOSITION DIAGNOSTIC:")
    print(f"   Daily-Level Total WAPE (before allocation): {mean_daily_tw:.4f}")
    print(f"   Block-Level Total WAPE (after allocation):  {mean_tw:.4f}")
    print(f"   Allocation Error Contribution (Δ):          {alloc_delta:+.4f}")

    # Read share model type from saved config
    share_type_path = cfg.MODEL_DIR / "best_share_type.txt"
    share_type_used = share_type_path.read_text().strip() if share_type_path.exists() else cfg.SHARE_MODEL_TYPE
    print(f"   Share Model Type Used:                      {share_type_used}")

    # Sanity checks
    site_wapes = []
    for site in cfg.SITES:
        sub = all_merged[all_merged["Site"] == site]
        if len(sub) > 0:
            site_wapes.append(wape(sub["ED Enc Admitted_true"].values,
                                   sub["ED Enc Admitted_pred"].values))
    max_min_ratio = max(site_wapes) / min(site_wapes) if site_wapes and min(site_wapes) > 0 else float("nan")

    block_wapes = []
    for blk in cfg.BLOCKS:
        sub = all_merged[all_merged["Block"] == blk]
        if len(sub) > 0:
            block_wapes.append(wape(sub["ED Enc Admitted_true"].values,
                                     sub["ED Enc Admitted_pred"].values))
    block_avg = np.mean(block_wapes) if block_wapes else float("nan")
    b0_ratio = block_wapes[0] / block_avg if block_wapes and block_avg > 0 else float("nan")

    fold_wapes = [r["admitted_wape"] for r in fold_results]
    fold_cv = np.std(fold_wapes) / np.mean(fold_wapes) if fold_wapes else float("nan")

    # Daily sum consistency: admitted <= total per block; count violations
    consistency_violations = int((all_merged["ED Enc Admitted_pred"] > all_merged["ED Enc_pred"]).sum())
    if consistency_violations > 0:
        print(f"   WARNING: {consistency_violations} block(s) with admitted > total (consistency violation)")

    print(f"\n SANITY CHECKS:")
    print(f"   Max site WAPE / Min site WAPE ratio: {max_min_ratio:.2f} (should be < 2.0)")
    print(f"   Block 0 WAPE / Block avg WAPE ratio: {b0_ratio:.2f} (flag if > 2.0)")
    print(f"   Per-fold WAPE std / mean:             {fold_cv:.2f} (flag if > 0.30)")
    print(f"   Admitted <= total violations:         {consistency_violations} (should be 0)")
    print(sep)

    # ── Persist report ───────────────────────────────────────────────────
    report = {
        "mean_admitted_wape": mean_aw,
        "mean_total_wape": mean_tw,
        "mean_admitted_rmse": mean_ar,
        "mean_total_rmse": mean_tr,
        "mean_daily_total_wape": mean_daily_tw,
        "allocation_error_delta": alloc_delta,
        "consistency_violations": consistency_violations,
        "share_model_type": share_type_used,
        "per_fold": [
            {k: v for k, v in r.items() if k != "_merged"} for r in fold_results
        ],
    }
    cfg.ensure_dirs()
    out_path = cfg.OUTPUT_DIR / "evaluation_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved → {out_path}")

    return report


if __name__ == "__main__":
    evaluate()
