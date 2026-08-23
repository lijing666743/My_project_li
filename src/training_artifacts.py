"""Artifact writer for completed CA-GAT-MAPPO training diagnostics.

This module deliberately sits outside the Trainer lifecycle. It consumes the
Trainer's immutable result after training has completed and writes auditable
diagnostic artifacts without changing environment, reward, PPO, or collection
semantics.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .config import RunConfig


LOSS_TYPE = "PPO clipped-surrogate actor + half-MSE critic"
REWARD_SCALE = "dimensionless fixed-reference normalized composite"
REWARD_NEUTRAL_BASELINE = 0.0
REWARD_SMOOTHING_WINDOW_EPISODES = 5
MIN_SIGNAL_EPISODES = 10
MIN_SIGNAL_PPO_EPOCHS = 10

TRAINING_METRIC_COLUMNS = (
    "run_id",
    "method_id",
    "scenario_id",
    "seed",
    "git_commit",
    "config_hash",
    "record_type",
    "series_index",
    "environment_steps",
    "episode_index",
    "environment_seed",
    "completed_boundary",
    "transition_count",
    "reward",
    "reward_mean",
    "reward_positive",
    "penalty_total",
    "completion_component",
    "expiration_penalty",
    "workload_penalty",
    "energy_penalty",
    "completed_task_count",
    "expired_task_count",
    "update_index",
    "ppo_epoch_index",
    "policy_version_before",
    "policy_version_after",
    "alpha",
    "critic_learning_rate",
    "ppo_clip_epsilon",
    "loss",
    "loss_type",
    "actor_loss",
    "critic_loss",
    "entropy",
    "total_loss",
    "ratio",
    "actor_grad_norm_before_clip",
    "critic_grad_norm_before_clip",
    "signal_gate_status",
)


@dataclass(frozen=True)
class TrainingArtifactOutcome:
    """Paths and headline metrics emitted after one completed training run."""

    artifacts: tuple[str, ...]
    smoke_gate_status: str
    signal_gate_status: str
    reward_mean: float
    actor_loss: float
    critic_loss: float
    entropy: float


def _base_record(config: RunConfig, signal_gate_status: str) -> dict[str, Any]:
    return {
        "run_id": config.run_id,
        "method_id": config.method_id,
        "scenario_id": config.scenario_id,
        "seed": config.seed,
        "git_commit": config.git_commit,
        "config_hash": config.config_hash,
        "record_type": None,
        "series_index": None,
        "environment_steps": None,
        "episode_index": None,
        "environment_seed": None,
        "completed_boundary": None,
        "transition_count": None,
        "reward": None,
        "reward_mean": None,
        "reward_positive": None,
        "penalty_total": None,
        "completion_component": None,
        "expiration_penalty": None,
        "workload_penalty": None,
        "energy_penalty": None,
        "completed_task_count": None,
        "expired_task_count": None,
        "update_index": None,
        "ppo_epoch_index": None,
        "policy_version_before": None,
        "policy_version_after": None,
        "alpha": None,
        "critic_learning_rate": None,
        "ppo_clip_epsilon": None,
        "loss": None,
        "loss_type": None,
        "actor_loss": None,
        "critic_loss": None,
        "entropy": None,
        "total_loss": None,
        "ratio": None,
        "actor_grad_norm_before_clip": None,
        "critic_grad_norm_before_clip": None,
        "signal_gate_status": signal_gate_status,
    }


def _reward_penalty_total(reward: Any) -> float:
    return math.fsum(
        (
            reward.expiration_penalty.total,
            reward.workload_penalty.total,
            reward.energy_penalty.total,
        )
    )


def _reward_identity_holds(reward: Any) -> bool:
    expected = reward.completion_component.total - _reward_penalty_total(reward)
    return math.isclose(
        reward.reward.total,
        expected,
        rel_tol=1.0e-9,
        abs_tol=1.0e-9,
    )


def _episode_records(
    config: RunConfig,
    training: Any,
    signal_gate_status: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    environment_steps = 0
    for series_index, episode in enumerate(training.episodes, start=1):
        environment_steps += episode.transition_count
        reward = episode.reward
        record = _base_record(config, signal_gate_status)
        record.update(
            {
                "record_type": "episode",
                "series_index": series_index,
                "environment_steps": environment_steps,
                "episode_index": episode.episode_index,
                "environment_seed": episode.environment_seed,
                "completed_boundary": episode.completed_boundary,
                "transition_count": episode.transition_count,
                "reward": reward.reward.total,
                "reward_mean": reward.reward.mean,
                "reward_positive": reward.completion_component.total,
                "penalty_total": _reward_penalty_total(reward),
                "completion_component": reward.completion_component.total,
                "expiration_penalty": reward.expiration_penalty.total,
                "workload_penalty": reward.workload_penalty.total,
                "energy_penalty": reward.energy_penalty.total,
                "completed_task_count": reward.completed_task_count.total,
                "expired_task_count": reward.expired_task_count.total,
            }
        )
        records.append(record)
    return records


def _ppo_epoch_records(
    config: RunConfig,
    training: Any,
    signal_gate_status: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    mappo = config.training.mappo
    series_index = 0
    for update in training.updates:
        update_environment_steps = min(
            (update.update_index + 1) * mappo.rollout_length_slots,
            training.total_environment_transitions,
        )
        for epoch in update.output.epoch_diagnostics:
            series_index += 1
            record = _base_record(config, signal_gate_status)
            record.update(
                {
                    "record_type": "ppo_epoch",
                    "series_index": series_index,
                    "environment_steps": update_environment_steps,
                    "update_index": update.update_index,
                    "ppo_epoch_index": epoch.epoch_index,
                    "policy_version_before": update.rollout_policy_version,
                    "policy_version_after": update.policy_version_after_update,
                    "alpha": mappo.actor_learning_rate,
                    "critic_learning_rate": mappo.critic_learning_rate,
                    "ppo_clip_epsilon": mappo.ppo_clip_epsilon,
                    "loss": epoch.total_loss,
                    "loss_type": LOSS_TYPE,
                    "actor_loss": epoch.actor_loss,
                    "critic_loss": epoch.critic_loss,
                    "entropy": epoch.entropy_mean,
                    "total_loss": epoch.total_loss,
                    "ratio": epoch.ratio_mean,
                    "actor_grad_norm_before_clip": epoch.actor_grad_norm_before_clip,
                    "critic_grad_norm_before_clip": epoch.critic_grad_norm_before_clip,
                }
            )
            records.append(record)
    return records


def _all_finite(values: Iterable[float]) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def _diagnostic_checks(config: RunConfig, training: Any) -> dict[str, bool]:
    epoch_rewards = [episode.reward.reward.total for episode in training.episodes]
    ppo_epochs = [
        epoch
        for update in training.updates
        for epoch in update.output.epoch_diagnostics
    ]
    ppo_values = [
        value
        for epoch in ppo_epochs
        for value in (
            epoch.actor_loss,
            epoch.critic_loss,
            epoch.entropy_mean,
            epoch.total_loss,
            epoch.ratio_mean,
            epoch.actor_grad_norm_before_clip,
            epoch.critic_grad_norm_before_clip,
        )
    ]
    return {
        "environment_budget_reached": (
            training.total_environment_transitions
            == config.training.mappo.max_training_environment_steps
        ),
        "reward_count_matches_transitions": (
            training.reward.transition_count
            == training.total_environment_transitions
        ),
        "reward_values_are_finite": _all_finite(epoch_rewards),
        "reward_formula_identity_holds": (
            _reward_identity_holds(training.reward)
            and all(_reward_identity_holds(episode.reward) for episode in training.episodes)
        ),
        "ppo_update_present": training.ppo_update_count > 0,
        "ppo_update_accounting_matches": (
            training.ppo_update_count == len(training.updates)
            and training.optimized_transitions
            == training.ppo_update_count * config.training.mappo.rollout_length_slots
        ),
        "ppo_epoch_count_matches": (
            len(ppo_epochs)
            == training.ppo_update_count * config.training.mappo.update_epochs
        ),
        "ppo_values_are_finite": bool(ppo_values) and _all_finite(ppo_values),
        "critic_loss_is_nonnegative": bool(ppo_epochs)
        and all(epoch.critic_loss >= 0.0 for epoch in ppo_epochs),
        "entropy_is_nonnegative": bool(ppo_epochs)
        and all(epoch.entropy_mean >= 0.0 for epoch in ppo_epochs),
        "ratio_is_positive": bool(ppo_epochs)
        and all(epoch.ratio_mean > 0.0 for epoch in ppo_epochs),
    }


def _signal_gate_status(training: Any, checks: Mapping[str, bool]) -> str:
    signal_checks = (
        "reward_values_are_finite",
        "reward_formula_identity_holds",
        "ppo_update_present",
        "ppo_values_are_finite",
        "critic_loss_is_nonnegative",
        "entropy_is_nonnegative",
        "ratio_is_positive",
    )
    if not all(checks[name] for name in signal_checks):
        return "signal-fail"
    ppo_epoch_count = sum(
        len(update.output.epoch_diagnostics) for update in training.updates
    )
    if (
        training.completed_episode_count < MIN_SIGNAL_EPISODES
        or ppo_epoch_count < MIN_SIGNAL_PPO_EPOCHS
    ):
        return "insufficient-horizon"
    return "signal-watch"


def _csv_text(records: list[dict[str, Any]]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=TRAINING_METRIC_COLUMNS,
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(records)
    return stream.getvalue()


def _json_lines(records: list[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
        for record in records
    )


def _training_summary(
    config: RunConfig,
    training: Any,
    checks: Mapping[str, bool],
    smoke_gate_status: str,
    signal_gate_status: str,
    dashboard_csv: str,
) -> dict[str, Any]:
    reward = training.reward
    ppo = training.ppo
    return {
        "run_id": config.run_id,
        "method_id": config.method_id,
        "scenario_id": config.scenario_id,
        "seed": config.seed,
        "validation_stage": "smoke-training",
        "smoke_training_gate_status": smoke_gate_status,
        "signal_gate_status": signal_gate_status,
        "checks": dict(checks),
        "training": {
            "training_device": config.training.mappo.training_device,
            "total_environment_transitions": training.total_environment_transitions,
            "optimized_transitions": training.optimized_transitions,
            "unused_final_tail_transitions": training.unused_final_tail_transitions,
            "started_episode_count": training.started_episode_count,
            "completed_episode_count": training.completed_episode_count,
            "ppo_update_count": training.ppo_update_count,
            "ppo_epoch_diagnostic_count": ppo.actor_loss.count,
            "initial_policy_version": training.initial_policy_version,
            "final_policy_version": training.final_policy_version,
        },
        "reward": {
            "neutral_baseline": REWARD_NEUTRAL_BASELINE,
            "scale": REWARD_SCALE,
            "clipping": "none",
            "transition_count": reward.transition_count,
            "total": reward.reward.total,
            "mean_per_transition": reward.reward.mean,
            "positive_total": reward.completion_component.total,
            "penalty_total": _reward_penalty_total(reward),
            "completion_component_total": reward.completion_component.total,
            "expiration_penalty_total": reward.expiration_penalty.total,
            "workload_penalty_total": reward.workload_penalty.total,
            "energy_penalty_total": reward.energy_penalty.total,
        },
        "ppo": {
            "loss_type": LOSS_TYPE,
            "actor_loss_mean": ppo.actor_loss.mean,
            "critic_loss_mean": ppo.critic_loss.mean,
            "entropy_mean": ppo.entropy.mean,
            "total_loss_mean": ppo.total_loss.mean,
            "ratio_mean": ppo.ratio.mean,
            "actor_grad_norm_before_clip_mean": ppo.actor_grad_norm_before_clip.mean,
            "critic_grad_norm_before_clip_mean": ppo.critic_grad_norm_before_clip.mean,
        },
        "dashboard": {
            "source_csv": dashboard_csv,
            "reward_series": "episode total reward",
            "ppo_series": "four diagnostics per completed PPO update",
            "reward_smoothing": (
                "trailing arithmetic mean, window up to "
                f"{REWARD_SMOOTHING_WINDOW_EPISODES} episodes, min_periods=1"
            ),
        },
        "claim_boundary": (
            "single-seed, short-horizon smoke diagnostic; validates execution, "
            "finite training signals, logging, and plotting only; it is not evidence "
            "of convergence, stability, superiority, generalization, or manuscript readiness"
        ),
    }


def _trailing_mean(values: list[float], window: int) -> list[float]:
    means: list[float] = []
    for index in range(len(values)):
        start = max(0, index + 1 - window)
        means.append(math.fsum(values[start : index + 1]) / (index + 1 - start))
    return means


def _plot_dashboard(csv_text: str, output_path: Path, summary: Mapping[str, Any]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "matplotlib is required to generate the training dashboard"
        ) from exc

    rows = list(csv.DictReader(io.StringIO(csv_text)))
    episode_rows = [row for row in rows if row["record_type"] == "episode"]
    ppo_rows = [row for row in rows if row["record_type"] == "ppo_epoch"]
    if not episode_rows or not ppo_rows:
        raise ValueError("dashboard requires episode reward and PPO epoch rows")

    colors = ("#3D5487", "#E64A36", "#4DBAD6", "#2AA198", "#8E63B0")
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "lines.linewidth": 1.5,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linestyle": "--",
            "legend.frameon": True,
            "legend.framealpha": 0.9,
        }
    )

    episode_x = [int(row["series_index"]) for row in episode_rows]
    rewards = [float(row["reward"]) for row in episode_rows]
    reward_smooth = _trailing_mean(rewards, REWARD_SMOOTHING_WINDOW_EPISODES)
    positives = [float(row["reward_positive"]) for row in episode_rows]
    penalties = [float(row["penalty_total"]) for row in episode_rows]
    ppo_x = [int(row["series_index"]) for row in ppo_rows]
    actor_loss = [float(row["actor_loss"]) for row in ppo_rows]
    critic_loss = [float(row["critic_loss"]) for row in ppo_rows]
    entropy = [float(row["entropy"]) for row in ppo_rows]
    ratio = [float(row["ratio"]) for row in ppo_rows]

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.0))
    ax = axes[0, 0]
    ax.plot(episode_x, rewards, color=colors[0], marker="o", label="raw")
    ax.plot(
        episode_x,
        reward_smooth,
        color=colors[1],
        linestyle="--",
        label=f"trailing mean (window={REWARD_SMOOTHING_WINDOW_EPISODES})",
    )
    ax.axhline(REWARD_NEUTRAL_BASELINE, color="0.35", linewidth=1.0, linestyle=":")
    ax.set(title="Episode reward", xlabel="Episode", ylabel="Total reward")
    ax.legend(loc="best")

    ax = axes[0, 1]
    ax.plot(episode_x, positives, color=colors[3], marker="o", label="positive")
    ax.plot(episode_x, penalties, color=colors[1], marker="s", label="penalty total")
    ax.set(title="Reward components", xlabel="Episode", ylabel="Component total")
    ax.legend(loc="best")

    ax = axes[1, 0]
    ax.plot(ppo_x, actor_loss, color=colors[0], marker="o", label="actor loss")
    ax.set(title="PPO losses", xlabel="PPO epoch diagnostic", ylabel="Actor loss")
    critic_axis = ax.twinx()
    critic_axis.plot(
        ppo_x,
        critic_loss,
        color=colors[1],
        marker="s",
        label="critic loss",
    )
    critic_axis.set_ylabel("Critic loss")
    handles = ax.get_lines() + critic_axis.get_lines()
    ax.legend(handles, [line.get_label() for line in handles], loc="best")

    ax = axes[1, 1]
    ax.plot(ppo_x, entropy, color=colors[2], marker="o", label="entropy")
    ax.set(title="Policy diagnostics", xlabel="PPO epoch diagnostic", ylabel="Entropy")
    ratio_axis = ax.twinx()
    ratio_axis.plot(ppo_x, ratio, color=colors[4], marker="s", label="ratio")
    ratio_axis.axhline(
        1.0,
        color="0.35",
        linewidth=1.0,
        linestyle=":",
        label="_nolegend_",
    )
    ratio_axis.set_ylabel("Policy ratio")
    handles = [ax.get_lines()[0], ratio_axis.get_lines()[0]]
    ax.legend(handles, [line.get_label() for line in handles], loc="best")

    fig.suptitle(
        "CA-GAT-MAPPO smoke training diagnostics\n"
        f"seed={summary['seed']}, transitions={summary['training']['total_environment_transitions']}, "
        f"signal={summary['signal_gate_status']}"
    )
    fig.text(
        0.5,
        0.002,
        "Diagnostic only: single seed and short horizon; no convergence or performance claim.",
        ha="center",
        va="bottom",
        fontsize=8,
        color="0.3",
    )
    fig.tight_layout(rect=(0.0, 0.045, 1.0, 0.93))
    try:
        fig.savefig(output_path, format="png", dpi=220, bbox_inches="tight")
    finally:
        plt.close(fig)


def _stage_text(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8", newline="")
    return temporary


def write_cagat_mappo_training_artifacts(
    config: RunConfig,
    training: Any,
) -> TrainingArtifactOutcome:
    """Validate a completed result and atomically publish smoke diagnostics."""

    if not isinstance(config, RunConfig):
        raise TypeError("config must be a RunConfig")
    from .models.ca_gat_mappo_trainer import CAGATMAPPOTrainingResult

    if not isinstance(training, CAGATMAPPOTrainingResult):
        raise TypeError("training must be a CAGATMAPPOTrainingResult")

    checks = _diagnostic_checks(config, training)
    signal_gate_status = _signal_gate_status(training, checks)
    smoke_gate_status = "pass" if all(checks.values()) else "fail"
    records = _episode_records(config, training, signal_gate_status)
    records.extend(_ppo_epoch_records(config, training, signal_gate_status))
    csv_text = _csv_text(records)

    paths = config.artifact_paths()
    snapshot_path = Path(paths["config_snapshot"])
    raw_path = Path(paths["raw_metrics"])
    aggregate_path = Path(paths["aggregate_metrics"])
    csv_path = Path(paths["dashboard_csv"])
    dashboard_path = Path(paths["dashboard_png"])
    summary = _training_summary(
        config,
        training,
        checks,
        smoke_gate_status,
        signal_gate_status,
        str(csv_path),
    )

    staged: list[tuple[Path, Path]] = []
    dashboard_temporary = dashboard_path.with_name(
        f".{dashboard_path.name}.{os.getpid()}.tmp"
    )
    dashboard_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        staged.extend(
            (
                (
                    snapshot_path,
                    _stage_text(
                        snapshot_path,
                        json.dumps(
                            config.snapshot_dict(),
                            ensure_ascii=False,
                            indent=2,
                            allow_nan=False,
                        )
                        + "\n",
                    ),
                ),
                (raw_path, _stage_text(raw_path, _json_lines(records))),
                (
                    aggregate_path,
                    _stage_text(
                        aggregate_path,
                        json.dumps(
                            summary,
                            ensure_ascii=False,
                            indent=2,
                            sort_keys=True,
                            allow_nan=False,
                        )
                        + "\n",
                    ),
                ),
                (csv_path, _stage_text(csv_path, csv_text)),
            )
        )
        _plot_dashboard(csv_text, dashboard_temporary, summary)
        staged.append((dashboard_path, dashboard_temporary))
        for destination, temporary in staged:
            temporary.replace(destination)
    finally:
        for _destination, temporary in staged:
            if temporary.exists():
                temporary.unlink()
        if dashboard_temporary.exists():
            dashboard_temporary.unlink()

    artifacts = tuple(
        str(path)
        for path in (
            snapshot_path,
            raw_path,
            aggregate_path,
            csv_path,
            dashboard_path,
        )
    )
    return TrainingArtifactOutcome(
        artifacts=artifacts,
        smoke_gate_status=smoke_gate_status,
        signal_gate_status=signal_gate_status,
        reward_mean=training.reward.reward.mean,
        actor_loss=training.ppo.actor_loss.mean,
        critic_loss=training.ppo.critic_loss.mean,
        entropy=training.ppo.entropy.mean,
    )


__all__ = [
    "LOSS_TYPE",
    "TRAINING_METRIC_COLUMNS",
    "TrainingArtifactOutcome",
    "write_cagat_mappo_training_artifacts",
]
