from __future__ import annotations

from pathlib import Path

from .config import RunConfig


def plot_dashboard(rows: list[dict[str, object]], config: RunConfig) -> Path:
    if not rows:
        raise ValueError("No metrics rows to plot.")
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to create dashboard PNG output.") from exc

    plot_dir = Path("plots")
    plot_dir.mkdir(parents=True, exist_ok=True)
    path = plot_dir / f"{config.run_id}_dashboard.png"

    epochs = [int(row["epoch"]) for row in rows]
    losses = [float(row["loss"]) for row in rows]
    rewards = [float(row["reward"]) for row in rows]
    kpis = [float(row["kpi"]) for row in rows]

    fig, axes = plt.subplots(2, 2, figsize=(9, 6), constrained_layout=True)
    axes[0, 0].plot(epochs, losses, marker="o")
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")

    axes[0, 1].plot(epochs, rewards, marker="o", color="#0072B2")
    axes[0, 1].set_title("Reward")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Reward")

    axes[1, 0].plot(epochs, kpis, marker="o", color="#009E73")
    axes[1, 0].set_title("Key KPI")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("KPI")

    axes[1, 1].axis("off")
    summary = (
        f"run_id: {config.run_id}\n"
        f"epochs: {config.epochs}\n"
        f"episodes/epoch: {config.episodes_per_epoch}\n"
        f"seed: {config.seed}\n"
        f"alpha: {config.alpha}\n"
        f"epsilon: {config.epsilon}"
    )
    axes[1, 1].text(0.0, 1.0, summary, va="top", family="monospace")

    fig.suptitle("Training Dashboard")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path
