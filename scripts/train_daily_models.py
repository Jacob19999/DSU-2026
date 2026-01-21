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
from dsu_forecast.modeling.metrics import mae, wape
from dsu_forecast.paths import artifacts_dir, outputs_dir


def _daily_aggregate(block_df: pd.DataFrame) -> pd.DataFrame:
    df = block_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    # Aggregate block-level covariates to daily
    group = ["Site", "Date"]

    sum_cols = [c for c in df.columns if c.endswith("_sum")]
    mean_cols = [c for c in df.columns if c.endswith("_mean")] + ["event_intensity", "event_count", "nws_severity_index", "nws_alerts_started", "nws_alerts_active"]
    mean_cols = [c for c in mean_cols if c in df.columns]

    cmix_cols = [c for c in df.columns if c.startswith("cmix_") and c.endswith("_share")]

    agg_spec: dict[str, str] = {}
    for c in sum_cols:
        agg_spec[c] = "sum"
    for c in mean_cols + cmix_cols:
        agg_spec[c] = "mean"

    # Calendar features are duplicated across blocks; take first
    cal_cols = ["dow", "month", "day", "is_weekend", "doy", "doy_sin", "doy_cos", "is_us_holiday", "is_month_start", "is_month_end"]
    for c in cal_cols:
        if c in df.columns:
            agg_spec[c] = "first"

    # LLM daily already daily; take first
    llm_cols = [c for c in df.columns if c.startswith("llm_")]
    for c in llm_cols:
        agg_spec[c] = "first"

    # Targets
    if "total_enc" in df.columns:
        agg_spec["total_enc"] = "sum"
    if "admitted_enc" in df.columns:
        agg_spec["admitted_enc"] = "sum"

    daily = df.groupby(group, as_index=False).agg(agg_spec).sort_values(group)
    return daily


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=str(outputs_dir() / "feature_table.parquet"))
    ap.add_argument("--artifacts", default=str(artifacts_dir()))
    ap.add_argument("--n_folds", type=int, default=4)
    ap.add_argument("--val_days", type=int, default=28)
    args = ap.parse_args()

    df = pd.read_parquet(Path(args.features))
    daily = _daily_aggregate(df)
    daily_train = daily[daily["total_enc"].notna() & daily["admitted_enc"].notna()].copy()
    daily_train["admit_rate"] = (daily_train["admitted_enc"] / daily_train["total_enc"].clip(lower=1.0)).clip(0, 1)

    daily_train = add_lag_features(
        daily_train,
        group_cols=["Site"],
        sort_cols=["Date"],
        target_cols=["total_enc", "admitted_enc", "admit_rate"],
        lags=[1, 7, 14, 28],
    )
    daily_train = add_rolling_mean_features(
        daily_train,
        group_cols=["Site"],
        sort_cols=["Date"],
        target_cols=["total_enc", "admitted_enc", "admit_rate"],
        windows=[7, 28],
    )

    drop = {"total_enc", "admitted_enc"}
    feature_cols = [c for c in daily_train.columns if c not in drop and pd.api.types.is_numeric_dtype(daily_train[c]) and daily_train[c].notna().any()]
    daily_train = daily_train.dropna(subset=feature_cols)

    # Simple rolling splits
    dates = sorted(daily_train["Date"].unique())
    max_date = pd.to_datetime(dates[-1])
    splits = []
    for i in range(args.n_folds, 0, -1):
        val_end = max_date - pd.Timedelta(days=(i - 1) * args.val_days)
        val_start = val_end - pd.Timedelta(days=args.val_days - 1)
        train_end = val_start - pd.Timedelta(days=1)
        splits.append((train_end, val_end))

    metrics: dict[str, dict[str, float]] = {}
    for k, (train_end, val_end) in enumerate(splits, 1):
        tr = daily_train[daily_train["Date"] <= train_end]
        va = daily_train[(daily_train["Date"] > train_end) & (daily_train["Date"] <= val_end)]

        X_tr = tr[feature_cols]
        X_va = va[feature_cols]

        m_total = xgb.XGBRegressor(
            n_estimators=1200,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=2.0,
            min_child_weight=3.0,
            objective="count:poisson",
            tree_method="hist",
            random_state=42,
            n_jobs=0,
        )
        m_total.fit(X_tr, tr["total_enc"].values, verbose=False)

        m_rate = xgb.XGBRegressor(
            n_estimators=1200,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=2.0,
            min_child_weight=3.0,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=42,
            n_jobs=0,
        )
        m_rate.fit(X_tr, tr["admit_rate"].values, verbose=False)

        dm = xgb.DMatrix(X_va, feature_names=feature_cols)
        pred_total = np.clip(m_total.get_booster().predict(dm), 0, None)
        pred_rate = np.clip(m_rate.get_booster().predict(dm), 0, 1)
        pred_adm = np.minimum(pred_total, pred_total * pred_rate)

        fold_key = f"daily_fold_{k}_{train_end.date()}_{val_end.date()}"
        metrics[fold_key] = {
            "total_wape": wape(va["total_enc"].values, pred_total),
            "adm_wape": wape(va["admitted_enc"].values, pred_adm),
            "total_mae": mae(va["total_enc"].values, pred_total),
            "adm_mae": mae(va["admitted_enc"].values, pred_adm),
        }
        print(f"[{fold_key}] total_wape={metrics[fold_key]['total_wape']:.4f} adm_wape={metrics[fold_key]['adm_wape']:.4f}")

    # Fit final
    X_all = daily_train[feature_cols]
    m_total = xgb.XGBRegressor(objective="count:poisson", n_estimators=1200, max_depth=6, learning_rate=0.03, subsample=0.9, colsample_bytree=0.9, reg_lambda=2.0, min_child_weight=3.0, tree_method="hist", random_state=42, n_jobs=0)
    m_total.fit(X_all, daily_train["total_enc"].values, verbose=False)
    m_rate = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=1200, max_depth=6, learning_rate=0.03, subsample=0.9, colsample_bytree=0.9, reg_lambda=2.0, min_child_weight=3.0, tree_method="hist", random_state=42, n_jobs=0)
    m_rate.fit(X_all, daily_train["admit_rate"].values, verbose=False)

    art = Path(args.artifacts)
    art.mkdir(parents=True, exist_ok=True)
    m_total.get_booster().save_model(str(art / "booster_daily_total.json"))
    m_rate.get_booster().save_model(str(art / "booster_daily_admit_rate.json"))
    (art / "feature_cols_daily.json").write_text(json.dumps(feature_cols, indent=2), encoding="utf-8")
    (art / "backtest_metrics_daily.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[OK] saved daily models to {art}")


if __name__ == "__main__":
    main()

