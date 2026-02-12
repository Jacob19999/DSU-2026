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


def _prep(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    # Targets exist only for historical rows
    df_train = df[df["total_enc"].notna() & df["admitted_enc"].notna()].copy()
    df_train["total_enc"] = df_train["total_enc"].astype(float)
    df_train["admitted_enc"] = df_train["admitted_enc"].astype(float)
    df_train["admit_rate"] = (df_train["admitted_enc"] / df_train["total_enc"].clip(lower=1.0)).clip(0, 1)

    # Lags/rolling on targets and admit_rate (strong signal)
    df_train = add_lag_features(
        df_train,
        group_cols=["Site", "Block"],
        sort_cols=["Date"],
        target_cols=["total_enc", "admitted_enc", "admit_rate"],
        lags=[1, 7, 14, 28],
    )
    df_train = add_rolling_mean_features(
        df_train,
        group_cols=["Site", "Block"],
        sort_cols=["Date"],
        target_cols=["total_enc", "admitted_enc", "admit_rate"],
        windows=[7, 28],
    )

    # Feature columns: numeric, excluding targets
    drop = {"total_enc", "admitted_enc"}
    numeric_cols = [c for c in df_train.columns if c not in drop and pd.api.types.is_numeric_dtype(df_train[c])]
    # Some numeric cols may be all-NA; keep and let XGBoost handle, but drop constant all-NA for stability.
    feature_cols = [c for c in numeric_cols if df_train[c].notna().any()]
    return df_train, feature_cols


def _time_splits(df: pd.DataFrame, *, n_folds: int = 4, val_days: int = 28) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    dates = sorted(df["Date"].unique())
    max_date = pd.to_datetime(dates[-1])
    splits: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    # Build folds ending before max_date, stepping backwards by val_days.
    for i in range(n_folds, 0, -1):
        val_end = max_date - pd.Timedelta(days=(i - 1) * val_days)
        val_start = val_end - pd.Timedelta(days=val_days - 1)
        train_end = val_start - pd.Timedelta(days=1)
        splits.append((train_end, val_end))
    return splits


def train_xgb_regressor(X: pd.DataFrame, y: np.ndarray, *, objective: str) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(
        n_estimators=1200,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=2.0,
        min_child_weight=3.0,
        objective=objective,
        tree_method="hist",
        random_state=42,
        n_jobs=0,
    )
    model.fit(X, y, verbose=False)
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=str(outputs_dir() / "feature_table.parquet"))
    ap.add_argument("--artifacts", default=str(artifacts_dir()))
    ap.add_argument("--n_folds", type=int, default=4)
    ap.add_argument("--val_days", type=int, default=28)
    args = ap.parse_args()

    feats_path = Path(args.features)
    df = pd.read_parquet(feats_path)

    df_train, feature_cols = _prep(df)
    df_train = df_train.dropna(subset=feature_cols)  # ensure no all-NA row after lag creation

    splits = _time_splits(df_train, n_folds=args.n_folds, val_days=args.val_days)
    metrics: dict[str, dict[str, float]] = {}

    for k, (train_end, val_end) in enumerate(splits, 1):
        train_mask = df_train["Date"] <= train_end
        val_mask = (df_train["Date"] > train_end) & (df_train["Date"] <= val_end)

        tr = df_train[train_mask]
        va = df_train[val_mask]

        X_tr = tr[feature_cols]
        X_va = va[feature_cols]

        m_total = train_xgb_regressor(X_tr, tr["total_enc"].values, objective="count:poisson")
        m_rate = train_xgb_regressor(X_tr, tr["admit_rate"].values, objective="reg:squarederror")

        pred_total = np.clip(m_total.predict(X_va), 0, None)
        pred_rate = np.clip(m_rate.predict(X_va), 0, 1)
        pred_adm = np.minimum(pred_total, pred_total * pred_rate)

        fold_key = f"fold_{k}_{train_end.date()}_{val_end.date()}"
        metrics[fold_key] = {
            "total_mae": mae(va["total_enc"].values, pred_total),
            "total_wape": wape(va["total_enc"].values, pred_total),
            "adm_mae": mae(va["admitted_enc"].values, pred_adm),
            "adm_wape": wape(va["admitted_enc"].values, pred_adm),
        }
        print(f"[{fold_key}] total_wape={metrics[fold_key]['total_wape']:.4f} adm_wape={metrics[fold_key]['adm_wape']:.4f}")

    # Final train on full history
    X_all = df_train[feature_cols]
    m_total = train_xgb_regressor(X_all, df_train["total_enc"].values, objective="count:poisson")
    m_rate = train_xgb_regressor(X_all, df_train["admit_rate"].values, objective="reg:squarederror")

    art_dir = Path(args.artifacts)
    art_dir.mkdir(parents=True, exist_ok=True)
    # Save raw boosters (more stable than sklearn wrapper load_model across sklearn versions)
    m_total.get_booster().save_model(str(art_dir / "booster_total.json"))
    m_rate.get_booster().save_model(str(art_dir / "booster_admit_rate.json"))

    (art_dir / "feature_cols.json").write_text(json.dumps(feature_cols, indent=2), encoding="utf-8")
    (art_dir / "backtest_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"[OK] saved models + metrics to {art_dir}")


if __name__ == "__main__":
    main()

