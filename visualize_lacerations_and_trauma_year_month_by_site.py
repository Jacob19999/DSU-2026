import os
import re
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


@dataclass(frozen=True)
class PlotConfig:
    facets_per_figure: int = 6  # 2 columns x 3 rows (matches existing faceted injury style)
    ncols: int = 2
    cmap_counts: str = "YlOrRd"
    cmap_rate: str = "YlGnBu"
    dpi: int = 300


MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def chunked(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv("Dataset/DSU-Dataset.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def filter_by_reason_pattern(df: pd.DataFrame, *, pattern: str) -> pd.DataFrame:
    rx = re.compile(pattern, re.IGNORECASE)
    mask = df["REASON_VISIT_NAME"].astype(str).apply(lambda s: bool(rx.search(s)))
    return df[mask].copy()


def prepare_time_fields(df: pd.DataFrame) -> pd.DataFrame:
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["YearMonth"] = df["Date"].dt.to_period("M").astype(str)
    return df


def pivot_year_month(df: pd.DataFrame, *, value_col: str, years: list[int]) -> pd.DataFrame:
    pivot = (
        df.pivot(index="Year", columns="Month", values=value_col)
        .reindex(index=years, fill_value=0)
        .reindex(columns=range(1, 13), fill_value=0)
    )
    pivot.columns = MONTH_NAMES
    return pivot


def create_faceted_year_month_heatmaps(
    *,
    agg: pd.DataFrame,
    sites: list[str],
    years: list[int],
    value_col: str,
    title: str,
    filename_prefix: str,
    cmap: str,
    annot_fmt: str,
    cbar_label: str,
    cfg: PlotConfig,
) -> None:
    os.makedirs("Visualizations", exist_ok=True)

    for chunk_idx, site_chunk in enumerate(chunked(sites, cfg.facets_per_figure), start=1):
        n = len(site_chunk)
        rows = max(1, (n + cfg.ncols - 1) // cfg.ncols)

        fig, axes = plt.subplots(rows, cfg.ncols, figsize=(18, 4 * rows + 2))
        fig.suptitle(title, fontsize=16, fontweight="bold", y=0.995)

        if rows == 1:
            axes = np.array(axes).reshape(1, -1)
        axes = axes.flatten()

        for ax_idx, site in enumerate(site_chunk):
            ax = axes[ax_idx]
            site_data = agg[agg["Site"] == site]
            pivot = pivot_year_month(site_data, value_col=value_col, years=years)

            if value_col == "Admission Rate %":
                total_enc = float(site_data["ED Enc"].sum())
                total_adm = float(site_data["ED Enc Admitted"].sum())
                total_rate = (total_adm / total_enc * 100.0) if total_enc > 0 else float("nan")
                subtitle = f"(Total Enc: {total_enc:,.0f}, Admit%: {total_rate:.1f})"
            else:
                total = float(pivot.to_numpy().sum())
                subtitle = f"(Total: {total:,.0f})"

            sns.heatmap(
                pivot,
                cmap=cmap,
                ax=ax,
                cbar_kws={"label": cbar_label},
                annot=True,
                fmt=annot_fmt,
                annot_kws={"size": 7},
                linewidths=0.5,
                cbar=True,
                square=False,
            )

            ax.set_title(f"{site}\n{subtitle}", fontweight="bold", pad=12, fontsize=11)
            ax.set_xlabel("Month", fontweight="bold")
            ax.set_ylabel("Year", fontweight="bold")

        for ax_idx in range(n, len(axes)):
            axes[ax_idx].set_visible(False)

        plt.tight_layout()
        out = f"Visualizations/{filename_prefix}_{chunk_idx:02d}.png"
        plt.savefig(out, dpi=cfg.dpi, bbox_inches="tight")
        print(f"Saved: {out}")
        plt.close(fig)


def create_site_correlation_heatmap(*, df: pd.DataFrame, title: str, out_path: str, cfg: PlotConfig) -> None:
    """
    Correlation of monthly volume across sites (Pearson r on YearMonth totals).
    """
    os.makedirs("Visualizations", exist_ok=True)

    by_site_month = (
        df.groupby(["Site", "YearMonth"], as_index=False)["ED Enc"]
        .sum()
        .pivot(index="YearMonth", columns="Site", values="ED Enc")
        .fillna(0)
    )

    if by_site_month.shape[1] < 2:
        print(f"Skipping correlation heatmap for '{title}' (need at least 2 sites).")
        return

    corr = by_site_month.corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_title(title, fontweight="bold", pad=14)
    sns.heatmap(
        corr,
        cmap="vlag",
        vmin=-1,
        vmax=1,
        center=0,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 8},
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Pearson r"},
        ax=ax,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=cfg.dpi, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


def run_focus(*, df: pd.DataFrame, label: str, reason_pattern: str, cfg: PlotConfig) -> None:
    focused = filter_by_reason_pattern(df, pattern=reason_pattern)
    focused = prepare_time_fields(focused)

    total = float(focused["ED Enc"].sum()) if not focused.empty else 0.0
    print(f"\n[{label}] Total encounters: {total:,.0f} (pattern={reason_pattern!r})")

    if focused.empty:
        print(f"[{label}] No matching records found; skipping.")
        return

    years = sorted(focused["Year"].unique().tolist())

    agg = (
        focused.groupby(["Site", "Year", "Month"], as_index=False)
        .agg({"ED Enc": "sum", "ED Enc Admitted": "sum"})
        .assign(
            **{
                "Admission Rate %": lambda x: np.where(
                    x["ED Enc"] > 0, x["ED Enc Admitted"] / x["ED Enc"] * 100.0, np.nan
                )
            }
        )
    )

    site_totals = agg.groupby("Site", as_index=False)["ED Enc"].sum().sort_values("ED Enc", ascending=False)
    sites = site_totals["Site"].tolist()
    print(f"[{label}] Sites with volume: {len(sites)}")

    safe_label = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")

    create_faceted_year_month_heatmaps(
        agg=agg,
        sites=sites,
        years=years,
        value_col="ED Enc",
        title=f"{label} Encounters by Year and Month (Faceted by Site)",
        filename_prefix=f"{safe_label}_year_month_by_site_faceted_counts",
        cmap=cfg.cmap_counts,
        annot_fmt=".0f",
        cbar_label="Encounters",
        cfg=cfg,
    )

    create_faceted_year_month_heatmaps(
        agg=agg,
        sites=sites,
        years=years,
        value_col="Admission Rate %",
        title=f"{label} Admission Rate (%) by Year and Month (Faceted by Site)",
        filename_prefix=f"{safe_label}_year_month_by_site_faceted_admission_rate",
        cmap=cfg.cmap_rate,
        annot_fmt=".1f",
        cbar_label="Admission Rate (%)",
        cfg=cfg,
    )

    create_site_correlation_heatmap(
        df=focused,
        title=f"{label}: Site-to-Site Correlation (Monthly Volume)",
        out_path=f"Visualizations/{safe_label}_site_monthly_correlation.png",
        cfg=cfg,
    )


def main() -> None:
    sns.set_style("whitegrid")
    plt.rcParams["font.size"] = 10

    cfg = PlotConfig()
    df = load_dataset()

    # Keep patterns intentionally tight to match what you asked for.
    run_focus(df=df, label="Lacerations", reason_pattern=r"\bLACERATION(S)?\b", cfg=cfg)
    run_focus(df=df, label="Trauma", reason_pattern=r"\bTRAUMA\b", cfg=cfg)

    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

