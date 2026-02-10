"""
Step 6: Evaluate Pipeline A predictions against the eval.md contract.

Loads fold CSVs from output/, aggregates ground truth from the raw
visits CSV, and produces an evaluation report with overall, per-fold,
per-site, and per-block breakdowns.
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd

import config as cfg

# ── Metrics (mirrors eval.md reference evaluator) ────────────────────────────


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.sum(np.abs(y_true)))
    return float(np.sum(np.abs(y_true - y_pred)) / denom) if denom > 0 else float("nan")


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true.astype(float) - y_pred) ** 2))
    ss_tot = float(np.sum((y_true.astype(float) - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


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


# ── Score one fold ──────────────────────────────────────────────────────────


def _score_fold(truth: pd.DataFrame, fold: dict) -> dict | None:
    csv_path = cfg.OUTPUT_DIR / f"fold_{fold['id']}_predictions.csv"
    if not csv_path.exists():
        print(f"  WARNING: {csv_path} not found -- skipping fold {fold['id']}")
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
        "total_r2": r2(yt, yp),
        "admitted_r2": r2(at, ap),
        "n_rows": len(merged),
        "_merged": merged,   # for breakdowns
    }


# ── Main entry point ────────────────────────────────────────────────────────


def evaluate() -> dict:
    """Score all folds, print report, save JSON."""
    truth = _build_truth()

    fold_results = []
    for fold in cfg.FOLDS:
        res = _score_fold(truth, fold)
        if res is not None:
            fold_results.append(res)

    if not fold_results:
        print("  No fold predictions found!")
        return {}

    # ── Aggregate ────────────────────────────────────────────────────────
    mean_tw = np.mean([r["total_wape"] for r in fold_results])
    mean_aw = np.mean([r["admitted_wape"] for r in fold_results])
    mean_tr = np.mean([r["total_rmse"] for r in fold_results])
    mean_ar = np.mean([r["admitted_rmse"] for r in fold_results])

    all_merged = pd.concat([r["_merged"] for r in fold_results], ignore_index=True)

    # ── Report ───────────────────────────────────────────────────────────
    sep = "=" * 62
    print(f"\n{sep}")
    print(" PIPELINE A: GLOBAL GBDT -- EVALUATION REPORT")
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
              f"total_wape={r['total_wape']:.4f}  "
              f"rows={r['n_rows']}")

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

    print(sep)

    # ── Persist report ───────────────────────────────────────────────────
    report = {
        "mean_admitted_wape": mean_aw,
        "mean_total_wape": mean_tw,
        "mean_admitted_rmse": mean_ar,
        "mean_total_rmse": mean_tr,
        "per_fold": [
            {k: v for k, v in r.items() if k != "_merged"} for r in fold_results
        ],
    }
    cfg.ensure_dirs()
    out_path = cfg.OUTPUT_DIR / "evaluation_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved -> {out_path}")

    return report


if __name__ == "__main__":
    evaluate()
