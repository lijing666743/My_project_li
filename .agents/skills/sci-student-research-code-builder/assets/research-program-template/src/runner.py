from __future__ import annotations

import random
from dataclasses import asdict

from .config import RunConfig


def _reset_scenario(epoch: int) -> dict[str, float]:
    return {"difficulty": 1.0 + epoch * 0.05}


def _run_episode(state: dict[str, float], episode: int, config: RunConfig) -> tuple[float, float]:
    rng = random.Random(config.seed + int(state["epoch"]) * 1009 + episode)
    progress = (episode + 1) / config.episodes_per_epoch
    jitter = rng.uniform(-0.01, 0.01)
    reward = max(0.0, progress * (1.0 - config.epsilon) / state["difficulty"] + jitter)
    loss = max(1e-9, state["difficulty"] / ((episode + 1) * 10.0) - jitter)
    return loss, reward


def run_training(config: RunConfig) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for epoch in range(config.epochs):
        state = _reset_scenario(epoch)
        state["epoch"] = float(epoch)
        episode = 0
        losses: list[float] = []
        rewards: list[float] = []
        while episode < config.episodes_per_epoch:
            loss, reward = _run_episode(state, episode, config)
            losses.append(loss)
            rewards.append(reward)
            episode += 1
        mean_loss = sum(losses) / len(losses)
        mean_reward = sum(rewards) / len(rewards)
        kpi = mean_reward / (mean_loss + 1e-9)
        row = {
            **asdict(config),
            "epoch": epoch,
            "episodes_per_epoch": config.episodes_per_epoch,
            "loss": mean_loss,
            "reward": mean_reward,
            "kpi": kpi,
        }
        rows.append(row)
        print(
            f"epoch={epoch} seed={config.seed} lr={config.alpha:.4g} eps={config.epsilon:.3f} "
            f"loss={mean_loss:.4f} reward={mean_reward:.4f} kpi={kpi:.4f}"
        )
    return rows
