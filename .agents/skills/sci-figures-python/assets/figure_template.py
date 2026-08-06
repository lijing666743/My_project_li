"""Reproducible SCI figure template.

Fill `load_data` and `plot_figure` with project-specific logic. Do not use
synthetic data for manuscript figures unless the figure is explicitly a toy
example and is labeled as such.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


KEVIN_DATA_FIGURE_COLORS = [
    "#3D5487",  # kevinCold2
    "#E64A36",  # kevinWarm1
    "#4DBAD6",  # kevinCold1
    "#F29C80",  # kevinWarm2
    "#2AA198",  # kevinTeal
    "#8E63B0",  # kevinPurple
    "#D9A441",  # kevinGold
    "#5AB769",  # kevinGreen
    "#6B7280",  # kevinGray
    "#111827",  # kevinBlack
]

MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
LINESTYLES = ["-", "--", "-.", ":"]


def setup_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "lines.linewidth": 1.5,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linestyle": "--",
            "legend.frameon": True,
            "legend.framealpha": 0.9,
            "legend.edgecolor": "0.75",
            "savefig.dpi": 450,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )


def load_data(data_path: Path) -> pd.DataFrame:
    """Load source data. Replace with the repo's real data contract."""
    if not data_path.exists():
        raise FileNotFoundError(f"Missing source data: {data_path}")
    return pd.read_csv(data_path)


def plot_figure(df: pd.DataFrame) -> plt.Figure:
    """Create a single- or multi-curve figure from real data.

    Required columns: x, y. Optional column: series.
    Do not add smoothing or error bars here unless the project supplies a
    documented transformation and real uncertainty columns.
    """
    required = {"x", "y"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Source data is empty")

    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    if "series" in df.columns:
        series_order = df["series"].drop_duplicates().tolist()
        for idx, series in enumerate(series_order):
            sub = df.loc[df["series"] == series].sort_values("x")
            ax.plot(
                sub["x"],
                sub["y"],
                color=KEVIN_DATA_FIGURE_COLORS[idx % len(KEVIN_DATA_FIGURE_COLORS)],
                marker=MARKERS[idx % len(MARKERS)],
                linestyle=LINESTYLES[(idx // len(MARKERS)) % len(LINESTYLES)],
                markersize=3.2,
                label=str(series),
            )
        ax.legend(loc="best")
    else:
        sub = df.sort_values("x")
        ax.plot(
            sub["x"],
            sub["y"],
            color=KEVIN_DATA_FIGURE_COLORS[0],
            marker=MARKERS[0],
            markersize=3.2,
            label=None,
        )
    ax.set_xlabel("X unit")
    ax.set_ylabel("Y unit")
    return fig


def save_figure(fig: plt.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"))
    fig.savefig(output_base.with_suffix(".svg"))


def main() -> None:
    setup_plot_style()
    data_path = Path("data/figure_source.csv")
    output_base = Path("figures/figure_name")
    df = load_data(data_path)
    fig = plot_figure(df)
    save_figure(fig, output_base)


if __name__ == "__main__":
    main()