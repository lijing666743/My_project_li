"""Plot CA-GAT-MAPPO Long Smoke training diagnostics.

The default input is the archived single-seed Long Smoke metrics file.  The
script only reads the CSV and writes PNG figures under ``results/figures``;
it never starts training or changes experiment artifacts.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "long_smoke"
    / "small_seed42"
    / "metrics.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "figures"
MOVING_AVERAGE_WINDOW = 10
FIGURE_SIZE = (6.4, 4.2)
FIGURE_DPI = 300
FONT_FAMILY = "Arial"
TEXT_COLOR = "#222222"
GRID_COLOR = "#B8B8B8"

plt.rcParams.update(
    {
        "font.family": FONT_FAMILY,
        "font.size": 9.0,
        "axes.titlesize": 11.0,
        "axes.labelsize": 10.0,
        "axes.labelcolor": TEXT_COLOR,
        "axes.titlecolor": TEXT_COLOR,
        "axes.edgecolor": "#666666",
        "axes.linewidth": 0.8,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "legend.fontsize": 8.5,
        "legend.labelcolor": TEXT_COLOR,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.unicode_minus": False,
        "savefig.facecolor": "white",
        "savefig.dpi": FIGURE_DPI,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

COLOR_BLUE = "#0072B2"
COLOR_ORANGE = "#D55E00"
COLOR_GREEN = "#009E73"
COLOR_PURPLE = "#CC79A7"


def _read_rows(input_path: Path) -> list[dict[str, str]]:
    """Read and validate the CSV schema needed by the requested figures."""

    if not input_path.is_file():
        raise FileNotFoundError(f"Metrics CSV does not exist: {input_path}")

    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Metrics CSV has no header: {input_path}")

        required_columns = {
            "record_type",
            "series_index",
            "episode_index",
            "update_index",
            "ppo_epoch_index",
            "reward",
            "completion_component",
            "expiration_penalty",
            "actor_loss",
            "critic_loss",
            "entropy",
            "ratio",
        }
        missing_columns = required_columns.difference(reader.fieldnames)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Metrics CSV is missing required columns: {missing}")

        rows = list(reader)

    if not rows:
        raise ValueError(f"Metrics CSV is empty: {input_path}")
    return rows


def _as_int(row: Mapping[str, str], column: str) -> int:
    value = row.get(column, "").strip()
    if not value:
        raise ValueError(f"Missing integer value in column ''{column}''")
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Invalid integer in column ''{column}'': {value!r}") from exc


def _as_float(row: Mapping[str, str], column: str) -> float:
    value = row.get(column, "").strip()
    if not value:
        raise ValueError(f"Missing numeric value in column ''{column}''")
    try:
        number = float(value)
    except ValueError as exc:
        raise ValueError(f"Invalid numeric value in column ''{column}'': {value!r}") from exc
    if not (float("-inf") < number < float("inf")):
        raise ValueError(f"Non-finite numeric value in column ''{column}'': {value!r}")
    return number


def _trailing_moving_average(values: Sequence[float], window: int) -> list[float]:
    """Return a trailing arithmetic mean, using available points at the start."""

    if window < 1:
        raise ValueError("Moving-average window must be positive")

    averages: list[float] = []
    running_sum = 0.0
    for index, value in enumerate(values):
        running_sum += value
        if index >= window:
            running_sum -= values[index - window]
        denominator = min(index + 1, window)
        averages.append(running_sum / denominator)
    return averages


def _style_axes(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, pad=12.0, fontweight="semibold")
    ax.set_xlabel(xlabel, labelpad=6.0)
    ax.set_ylabel(ylabel, labelpad=6.0)
    ax.grid(True, which="major", color=GRID_COLOR, alpha=0.45, linewidth=0.65)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#666666")
    ax.spines["bottom"].set_color("#666666")
    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=3.5,
        width=0.75,
        pad=3.0,
        labelsize=8.5,
    )
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))


def _add_legend(
    ax: plt.Axes,
    handles: Sequence[object] | None = None,
    labels: Sequence[str] | None = None,
    *,
    loc: str = "best",
    ncol: int | None = None,
) -> None:
    """Place a compact legend inside the axes, away from the title area."""

    if handles is None or labels is None:
        handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        loc=loc,
        ncol=min(2, len(labels)) if ncol is None else ncol,
        frameon=True,
        framealpha=0.9,
        facecolor="white",
        edgecolor="#D0D0D0",
        borderpad=0.55,
        handlelength=2.4,
        columnspacing=1.5,
        borderaxespad=0.4,
    )


def _save_figure(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Keep the requested 6.4 x 4.2 inch canvas while reserving enough room
    # around the title, labels, and in-axes legend.
    fig.tight_layout(rect=(0.04, 0.04, 0.96, 0.95), pad=1.25)
    fig.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(fig)


def _episode_records(rows: Iterable[Mapping[str, str]]) -> list[Mapping[str, str]]:
    records = [row for row in rows if row.get("record_type", "").strip() == "episode"]
    records.sort(key=lambda row: _as_int(row, "episode_index"))
    if not records:
        raise ValueError("Metrics CSV contains no episode records")
    return records


def _ppo_records(rows: Iterable[Mapping[str, str]]) -> list[Mapping[str, str]]:
    records = [row for row in rows if row.get("record_type", "").strip() == "ppo_epoch"]
    records.sort(
        key=lambda row: (
            _as_int(row, "update_index"),
            _as_int(row, "ppo_epoch_index"),
            _as_int(row, "series_index"),
        )
    )
    if not records:
        raise ValueError("Metrics CSV contains no ppo_epoch records")
    return records


def plot_training_curves(
    input_path: Path = DEFAULT_INPUT_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> list[Path]:
    """Generate the four requested diagnostic figures from one metrics CSV."""

    rows = _read_rows(input_path)
    episode_rows = _episode_records(rows)
    ppo_rows = _ppo_records(rows)

    episode_x = [_as_int(row, "episode_index") for row in episode_rows]
    rewards = [_as_float(row, "reward") for row in episode_rows]
    completion = [_as_float(row, "completion_component") for row in episode_rows]
    expiration = [_as_float(row, "expiration_penalty") for row in episode_rows]
    reward_moving_average = _trailing_moving_average(rewards, MOVING_AVERAGE_WINDOW)

    # PPO diagnostics are ordered by update and inner PPO epoch.  An ordinal
    # axis keeps all logged epoch-level points visible without averaging them.
    ppo_x = list(range(1, len(ppo_rows) + 1))
    actor_loss = [_as_float(row, "actor_loss") for row in ppo_rows]
    critic_loss = [_as_float(row, "critic_loss") for row in ppo_rows]
    entropy = [_as_float(row, "entropy") for row in ppo_rows]
    ratio = [_as_float(row, "ratio") for row in ppo_rows]

    output_paths = [
        output_dir / "reward_curve.png",
        output_dir / "completion_penalty_curve.png",
        output_dir / "ppo_loss_curve.png",
        output_dir / "policy_diagnostics_curve.png",
    ]

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.plot(
        episode_x,
        rewards,
        color=COLOR_BLUE,
        linewidth=0.95,
        alpha=0.62,
        label="Episode reward (raw)",
        zorder=2,
    )
    ax.plot(
        episode_x,
        reward_moving_average,
        color=COLOR_ORANGE,
        linewidth=2.15,
        marker="o",
        markersize=3.0,
        markeredgewidth=0.45,
        markeredgecolor="white",
        markevery=MOVING_AVERAGE_WINDOW,
        label=f"{MOVING_AVERAGE_WINDOW}-episode moving average",
        zorder=3,
    )
    _style_axes(ax, "Training reward", "Episode index", "Episode reward")
    _add_legend(ax, loc="lower right", ncol=1)
    _save_figure(fig, output_paths[0])

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.plot(
        episode_x,
        completion,
        color=COLOR_GREEN,
        linewidth=1.55,
        marker="o",
        markersize=2.8,
        markeredgewidth=0.35,
        markeredgecolor="white",
        markevery=MOVING_AVERAGE_WINDOW,
        label="Completion component",
    )
    ax.plot(
        episode_x,
        expiration,
        color=COLOR_ORANGE,
        linewidth=1.55,
        linestyle="--",
        marker="s",
        markersize=2.7,
        markeredgewidth=0.35,
        markeredgecolor="white",
        markevery=MOVING_AVERAGE_WINDOW,
        label="Expiration penalty (magnitude)",
    )
    _style_axes(ax, "Reward components", "Episode index", "Component value")
    _add_legend(ax, loc="best", ncol=1)
    _save_figure(fig, output_paths[1])

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.plot(ppo_x, actor_loss, color=COLOR_BLUE, linewidth=1.0, alpha=0.88, label="Actor loss")
    ax.plot(ppo_x, critic_loss, color=COLOR_ORANGE, linewidth=1.0, alpha=0.88, label="Critic loss")
    _style_axes(ax, "PPO loss diagnostics", "PPO epoch diagnostic index", "Loss")
    _add_legend(ax, loc="best", ncol=1)
    _save_figure(fig, output_paths[2])

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    entropy_line, = ax.plot(ppo_x, entropy, color=COLOR_PURPLE, linewidth=1.0, alpha=0.9, label="Entropy")
    _style_axes(ax, "PPO policy diagnostics", "PPO epoch diagnostic index", "Entropy")

    # Ratio is logged on a much narrower scale than entropy.  A secondary
    # y-axis preserves every raw point while keeping both diagnostics legible.
    ax2 = ax.twinx()
    ratio_line, = ax2.plot(ppo_x, ratio, color=COLOR_GREEN, linewidth=1.0, alpha=0.9, label="Ratio")
    ax2.set_ylabel("Policy ratio", labelpad=6.0)
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax2.tick_params(
        axis="y",
        which="major",
        direction="out",
        length=3.5,
        width=0.75,
        pad=3.0,
        labelsize=8.5,
        colors=TEXT_COLOR,
    )
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color("#666666")
    ax2.spines["right"].set_linewidth(0.8)
    ax2.grid(False)
    _add_legend(
        ax,
        [entropy_line, ratio_line],
        ["Entropy", "Ratio"],
        loc="best",
        ncol=2,
    )
    _save_figure(fig, output_paths[3])

    return output_paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Metrics CSV path (default: {DEFAULT_INPUT_PATH})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Figure output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_paths = plot_training_curves(args.input.resolve(), args.output_dir.resolve())
    print(f"Read metrics: {args.input.resolve()}")
    for output_path in output_paths:
        print(f"Wrote figure: {output_path.resolve()}")


if __name__ == "__main__":
    main()
