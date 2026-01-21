from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

# Allow running without installing the package (repo-local `src/` layout)
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dsu_forecast.modeling.features import add_lag_features, add_rolling_mean_features
from dsu_forecast.paths import artifacts_dir, outputs_dir


def _largest_remainder_round(values: np.ndarray, target_sum: int) -> np.ndarray:
    """
    Round non-negative floats to ints summing exactly to target_sum.
    """
    values = np.clip(values, 0, None)
    floor = np.floor(values).astype(int)
    remainder = values - floor
    current = int(floor.sum())
    need = target_sum - current
    if need == 0:
        return floor
    idx = np.argsort(-remainder)  # descending
    out = floor.copy()
    if need > 0:
        out[idx[:need]] += 1
    else:
        # need < 0: subtract from smallest remainders where possible
        idx2 = np.argsort(remainder)  # ascending
        k = -need
        for j in idx2:
            if k <= 0:
                break
            if out[j] > 0:
                out[j] -= 1
                k -= 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=str(outputs_dir() / "feature_table.parquet"))
    ap.add_argument("--block_forecast", default=str(outputs_dir() / "forecast_sep_oct_2025.csv"))
    ap.add_argument("--forecast_start", default="2025-09-01")
    ap.add_argument("--forecast_end", default="2025-10-31")
    ap.add_argument("--artifacts", default=str(artifacts_dir()))
    ap.add_argument("--out", default=str(outputs_dir() / "forecast_sep_oct_2025_reconciled.csv"))
    args = ap.parse_args()

    art = Path(args.artifacts)
    booster_total = xgb.Booster()
    booster_total.load_model(str(art / "booster_daily_total.json"))
    booster_rate = xgb.Booster()
    booster_rate.load_model(str(art / "booster_daily_admit_rate.json"))
    feature_cols = json.loads((art / "feature_cols_daily.json").read_text(encoding="utf-8"))

    # Build daily feature frame for horizon
    block = pd.read_parquet(Path(args.features))
    block["Date"] = pd.to_datetime(block["Date"])
    # Create daily table by aggregating blocks (similar to train_daily_models)
    sum_cols = [c for c in block.columns if c.endswith("_sum")]
    mean_cols = [c for c in block.columns if c.endswith("_mean")] + ["event_intensity", "event_count", "nws_severity_index", "nws_alerts_started", "nws_alerts_active"]
    mean_cols = [c for c in mean_cols if c in block.columns]
    cmix_cols = [c for c in block.columns if c.startswith("cmix_") and c.endswith("_share")]
    cal_cols = ["dow", "month", "day", "is_weekend", "doy", "doy_sin", "doy_cos", "is_us_holiday", "is_month_start", "is_month_end"]
    llm_cols = [c for c in block.columns if c.startswith("llm_")]

    agg: dict[str, str] = {}
    for c in sum_cols:
        agg[c] = "sum"
    for c in mean_cols + cmix_cols:
        agg[c] = "mean"
    for c in cal_cols:
        if c in block.columns:
            agg[c] = "first"
    for c in llm_cols:
        agg[c] = "first"
    for c in ["total_enc", "admitted_enc"]:
        if c in block.columns:
            agg[c] = "sum"

    daily = block.groupby(["Site", "Date"], as_index=False).agg(agg).sort_values(["Site", "Date"])
    daily["admit_rate"] = (daily["admitted_enc"] / daily["total_enc"].clip(lower=1.0)).clip(0, 1)

    daily = add_lag_features(daily, group_cols=["Site"], sort_cols=["Date"], target_cols=["total_enc", "admitted_enc", "admit_rate"], lags=[1, 7, 14, 28])
    daily = add_rolling_mean_features(daily, group_cols=["Site"], sort_cols=["Date"], target_cols=["total_enc", "admitted_enc", "admit_rate"], windows=[7, 28])

    horizon_daily = daily[(daily["Date"] >= pd.to_datetime(args.forecast_start)) & (daily["Date"] <= pd.to_datetime(args.forecast_end))].copy()
    X = horizon_daily[feature_cols].fillna(0.0)
    dm = xgb.DMatrix(X, feature_names=feature_cols)
    pred_daily_total = np.clip(booster_total.predict(dm), 0, None)
    pred_daily_rate = np.clip(booster_rate.predict(dm), 0, 1)
    pred_daily_adm = np.minimum(pred_daily_total, pred_daily_total * pred_daily_rate)

    horizon_daily["pred_daily_total"] = pred_daily_total
    horizon_daily["pred_daily_adm"] = pred_daily_adm

    # Read block forecast and scale within each site/day to match daily
    blk = pd.read_csv(Path(args.block_forecast))
    blk["Date"] = pd.to_datetime(blk["Date"])
    blk["ED Enc"] = pd.to_numeric(blk["ED Enc"])
    blk["ED Enc Admitted"] = pd.to_numeric(blk["ED Enc Admitted"])

    merged = blk.merge(horizon_daily[["Site", "Date", "pred_daily_total", "pred_daily_adm"]], on=["Site", "Date"], how="left")

    out_rows: list[pd.DataFrame] = []
    for (site, dt), g in merged.groupby(["Site", "Date"], sort=False):
        tgt_total = float(g["pred_daily_total"].iloc[0]) if pd.notna(g["pred_daily_total"].iloc[0]) else float(g["ED Enc"].sum())
        tgt_adm = float(g["pred_daily_adm"].iloc[0]) if pd.notna(g["pred_daily_adm"].iloc[0]) else float(g["ED Enc Admitted"].sum())

        cur_total = float(g["ED Enc"].sum())
        cur_adm = float(g["ED Enc Admitted"].sum())

        scale_total = (tgt_total / cur_total) if cur_total > 0 else 1.0
        scale_adm = (tgt_adm / cur_adm) if cur_adm > 0 else 1.0

        scaled_total = np.clip(g["ED Enc"].values.astype(float) * scale_total, 0, None)
        scaled_adm = np.clip(g["ED Enc Admitted"].values.astype(float) * scale_adm, 0, None)
        # Enforce admitted <= total per-block (pre-round)
        scaled_adm = np.minimum(scaled_adm, scaled_total)

        # Integer rounding with exact daily sums
        total_int = _largest_remainder_round(scaled_total, int(round(tgt_total)))
        adm_int = _largest_remainder_round(np.minimum(scaled_adm, total_int.astype(float)), int(round(min(tgt_adm, tgt_total))))

        gg = g.copy()
        gg["ED Enc"] = total_int
        gg["ED Enc Admitted"] = np.minimum(adm_int, total_int)
        out_rows.append(gg[["Site", "Date", "Block", "ED Enc", "ED Enc Admitted"]])

    out = pd.concat(out_rows, ignore_index=True)
    out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")
    out = out.sort_values(["Site", "Date", "Block"])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"[OK] wrote {out_path} rows={len(out):,}")


if __name__ == "__main__":
    main()

