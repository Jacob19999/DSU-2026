import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def _load_injury_df() -> pd.DataFrame:
    df = pd.read_csv(
        "Dataset/DSU-Dataset.csv",
        usecols=["Site", "Date", "REASON_VISIT_NAME", "ED Enc"],
    )
    df["Date"] = pd.to_datetime(df["Date"])

    reason_to_category = pd.read_csv("reason_categories.csv")
    reason_to_category_dict = dict(
        zip(reason_to_category["REASON_VISIT_NAME"], reason_to_category["Category"])
    )
    df["Reason_Category"] = df["REASON_VISIT_NAME"].map(reason_to_category_dict)

    injury_df = df[df["Reason_Category"] == "Injury/Trauma"].copy()
    injury_df["Year"] = injury_df["Date"].dt.year
    injury_df["Month"] = injury_df["Date"].dt.month
    injury_df["YearMonth"] = injury_df["Date"].dt.to_period("M").astype(str)

    injury_df["ED Enc"] = pd.to_numeric(injury_df["ED Enc"], errors="coerce").fillna(0)
    return injury_df


def _top_injury_types(injury_df: pd.DataFrame, top_n: int) -> list[str]:
    vol = (
        injury_df.groupby("REASON_VISIT_NAME", as_index=False)["ED Enc"]
        .sum()
        .sort_values("ED Enc", ascending=False)
    )
    return vol.head(top_n)["REASON_VISIT_NAME"].tolist()


def _site_monthly_pivot(
    injury_df: pd.DataFrame,
    site: str,
    injury_types: list[str],
    *,
    normalize_row_share: bool,
) -> pd.DataFrame:
    df_site = injury_df[
        (injury_df["Site"] == site)
        & (injury_df["REASON_VISIT_NAME"].isin(injury_types))
    ].copy()

    pivot = df_site.pivot_table(
        index="YearMonth",
        columns="REASON_VISIT_NAME",
        values="ED Enc",
        aggfunc="sum",
        fill_value=0,
    )

    # Ensure stable ordering and include any missing columns.
    pivot = pivot.reindex(columns=injury_types, fill_value=0)

    pivot.index = pd.to_datetime(pivot.index)
    pivot = pivot.sort_index()

    if normalize_row_share:
        row_sum = pivot.sum(axis=1).replace(0, np.nan)
        pivot = pivot.div(row_sum, axis=0).fillna(0)

    return pivot


def _corr_from_pivot(pivot: pd.DataFrame) -> pd.DataFrame:
    # Pearson correlation across injury-type time series (monthly points).
    return pivot.corr(method="pearson").fillna(0)


def _plot_corr_heatmap_grid(
    site_to_corr: dict[str, pd.DataFrame],
    *,
    out_path: str,
    title: str,
) -> None:
    sns.set_style("white")
    Path(os.path.dirname(out_path) or ".").mkdir(parents=True, exist_ok=True)

    sites = sorted(site_to_corr.keys())
    n_sites = len(sites)
    nrows = int(np.ceil(n_sites / 2))
    ncols = 2 if n_sites > 1 else 1

    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 18), squeeze=False)
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.995)

    mappable = None
    for idx, site in enumerate(sites):
        ax = axes[idx // ncols][idx % ncols]
        corr = site_to_corr[site]

        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        hm = sns.heatmap(
            corr,
            mask=mask,
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
            center=0,
            square=True,
            linewidths=0.25,
            linecolor="#f0f0f0",
            cbar=False,
            ax=ax,
        )
        mappable = hm.collections[0]
        ax.set_title(f"Site {site}", fontweight="bold", pad=10)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", labelrotation=60, labelsize=8)
        ax.tick_params(axis="y", labelsize=8)

    # Hide unused axes (if odd number of sites).
    for j in range(n_sites, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    if mappable is not None:
        # Put the colorbar outside the subplot grid so it doesn't overlap any heatmaps.
        plt.subplots_adjust(left=0.08, right=0.88, top=0.94, bottom=0.06, wspace=0.18, hspace=0.18)
        cbar_ax = fig.add_axes([0.90, 0.20, 0.02, 0.60])
        cbar = fig.colorbar(mappable, cax=cbar_ax)
        cbar.set_label("Pearson r", fontweight="bold")

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _write_interactive_html(
    site_to_corr: dict[str, pd.DataFrame],
    *,
    out_path: str,
    title: str,
) -> None:
    try:
        import plotly.graph_objects as go
    except Exception as e:  # pragma: no cover
        print(f"Plotly import failed ({e}); skipping interactive HTML.")
        return

    sites = sorted(site_to_corr.keys())
    if not sites:
        return

    fig = go.Figure()
    buttons = []

    for i, site in enumerate(sites):
        corr = site_to_corr[site]
        fig.add_trace(
            go.Heatmap(
                z=corr.values,
                x=corr.columns.tolist(),
                y=corr.index.tolist(),
                zmin=-1,
                zmax=1,
                colorscale="RdBu",
                reversescale=True,
                colorbar=dict(title="Pearson r") if i == 0 else None,
                visible=i == 0,
            )
        )
        buttons.append(
            dict(
                label=f"Site {site}",
                method="update",
                args=[
                    {"visible": [j == i for j in range(len(sites))]},
                    {"title": f"{title} — Site {site}"},
                ],
            )
        )

    fig.update_layout(
        title=f"{title} — Site {sites[0]}",
        updatemenus=[
            dict(
                type="dropdown",
                x=0.01,
                y=1.08,
                xanchor="left",
                yanchor="top",
                buttons=buttons,
            )
        ],
        height=900,
        width=950,
        margin=dict(l=80, r=40, t=120, b=80),
    )

    Path(os.path.dirname(out_path) or ".").mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path, include_plotlyjs="cdn")


def main() -> None:
    os.makedirs("Visualizations", exist_ok=True)

    print("Loading dataset + filtering to Injury/Trauma...")
    injury_df = _load_injury_df()
    print(f"Total injury encounters: {injury_df['ED Enc'].sum():,.0f}")

    TOP_N = 15
    NORMALIZE_ROW_SHARE = True  # correlate composition changes, not raw volume

    injury_types = _top_injury_types(injury_df, TOP_N)
    sites = sorted(injury_df["Site"].dropna().unique().tolist())
    print(f"Sites: {sites}")
    print(f"Using top {TOP_N} injury types by total volume.")

    site_to_corr: dict[str, pd.DataFrame] = {}
    for site in sites:
        pivot = _site_monthly_pivot(
            injury_df,
            site,
            injury_types,
            normalize_row_share=NORMALIZE_ROW_SHARE,
        )
        site_to_corr[site] = _corr_from_pivot(pivot)

    title = (
        "Injury-Type Correlation by Site over Year-Month\n"
        + ("(based on monthly share of injury encounters)" if NORMALIZE_ROW_SHARE else "(based on raw monthly counts)")
    )

    out_png = "Visualizations/injury_type_corr_site_yearmonth.png"
    _plot_corr_heatmap_grid(site_to_corr, out_path=out_png, title=title)
    print(f"Saved: {out_png}")

    out_html = "Visualizations/injury_type_corr_site_yearmonth.html"
    _write_interactive_html(site_to_corr, out_path=out_html, title=title.replace("\n", " "))
    print(f"Saved: {out_html}")


if __name__ == "__main__":
    main()

