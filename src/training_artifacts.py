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
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Iterable, Mapping

from .artifacts import atomic_write_bytes_group, require_artifact_targets_absent
from .config import RunConfig, compute_route_entropy_schedule
from .models.ca_gat_mappo_route_telemetry import (
    ROUTE_TELEMETRY_SCHEMA_VERSION,
    TRAJECTORY_CREDIT_EVENT_TYPES,
    TRAJECTORY_CREDIT_SCHEMA_VERSION,
    RouteOutcomeAssociation,
    aggregate_route_outcomes,
    ROUTE_ACTION_CATEGORIES,
    RouteTelemetry,
)


LOSS_TYPE = "PPO clipped-surrogate actor + half-MSE critic"
REWARD_SCALE = "dimensionless fixed-reference normalized composite"
REWARD_NEUTRAL_BASELINE = 0.0
REWARD_SMOOTHING_WINDOW_EPISODES = 5
MIN_SIGNAL_EPISODES = 10
MIN_SIGNAL_PPO_EPOCHS = 10

TRAINING_DIAGNOSTICS_SCHEMA_VERSION = 4
ROUTE_TELEMETRY_ROLLING_WINDOW_PPO_EPOCHS = 4
_ROUTE_TELEMETRY_COLUMNS = (
    "route_telemetry_schema_version",
    *RouteTelemetry.field_names()[1:],
)
_ROUTE_SAMPLE_COLUMNS = (
    "route_sample_batch_index",
    "route_sample_time_index",
    "route_sample_agent_index",
    "route_sample_source_uav",
    "route_sample_category",
    "route_sample_legal_remote_destinations",
    "route_sample_remote_probability_by_destination",
    "route_sample_local_probability",
    "route_sample_defer_probability",
    "route_sample_total_remote_probability_mass",
    "route_sample_best_remote_probability",
    "route_sample_number_of_legal_remote_destinations",
    "route_sample_selected_route_action_index",
    "route_sample_selected_destination_uav",
    "route_sample_old_route_log_prob",
    "route_sample_new_route_log_prob",
    "route_sample_log_ratio",
    "route_sample_ratio",
    "route_sample_approx_kl",
    "route_sample_clip_indicator",
    "route_sample_advantage",
    "route_sample_return_target",
    "route_sample_td_residual",
)
_ROUTE_OUTCOME_COLUMNS = tuple(
    f"route_outcome_{item.name}" for item in fields(RouteOutcomeAssociation)
)
_BRANCH_ACTIVITY_COLUMNS = (
    "branch_activity_rollout_timestep",
    "branch_activity_matrix",
)
_AGENT_CREDIT_COLUMNS = (
    "agent_value_mean",
    "agent_td_residual_mean",
    "agent_advantage_mean",
    "agent_return_mean",
    "same_timestep_advantage_equality_rate",
    "route_category_agent_advantage_mean",
    "per_head_critic_loss",
)
_ROUTE_ROLLING_FIELDS = (
    ("route_entropy_rolling_mean", "route_entropy_mean"),
    ("remote_selection_rate_rolling_mean", "remote_selection_rate_given_legal_remote"),
    ("mean_local_probability_rolling_mean", "mean_local_probability"),
    ("mean_best_remote_probability_rolling_mean", "mean_best_remote_probability"),
    ("local_minus_best_remote_logit_margin_rolling_mean", "mean_local_minus_best_remote_logit_margin"),
    ("route_head_total_grad_norm_rolling_mean", "route_head_total_grad_norm"),
    ("route_active_advantage_rolling_mean", "route_active_advantage_mean"),
)

_ROUTE_ROLLING_COLUMNS = tuple(name for name, _ in _ROUTE_ROLLING_FIELDS)

TRAINING_METRIC_COLUMNS = (
    "run_id",
    "method_id",
    "scenario_id",
    "seed",
    "git_commit",
    "config_hash",
    "diagnostics_schema_version",
    "actor_ratio_mode",
    "agent_credit_mode",
    "record_type",
    "telemetry_scope",
    "series_index",
    "environment_steps",
    "collected_environment_steps",
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
    "route_entropy_coefficient",
    "route_entropy_schedule_progress",
    "route_entropy_loss_contribution",
    "other_branch_entropy_loss_contribution",
    "global_entropy_loss_contribution",
    "total_loss",
    "ratio",
    "approx_kl",
    "clip_fraction",
    "actor_grad_norm_before_clip",
    "critic_grad_norm_before_clip",
    "route_telemetry_enabled",
    *_ROUTE_TELEMETRY_COLUMNS,
    *_ROUTE_ROLLING_COLUMNS,
    *_ROUTE_SAMPLE_COLUMNS,
    *_ROUTE_OUTCOME_COLUMNS,
    *_BRANCH_ACTIVITY_COLUMNS,
    *_AGENT_CREDIT_COLUMNS,
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


class TrajectoryCreditArtifactError(RuntimeError):
    """Raised when the append-only trajectory-credit sidecar is invalid."""


_FORBIDDEN_ROUTE_OBSERVATION_KEYS = {
    "channel_imag",
    "channel_real",
    "future_csi",
    "instantaneous_channel",
    "stale_csi",
    "true_channel",
}
_TRAJECTORY_REQUIRED_FIELDS = {
    "route": {
        "route_step",
        "rollout_index",
        "source_uav",
        "route_category",
        "selected_destination_uav",
        "route_action_index",
        "route_status",
        "source_queue_proxy",
        "legal_route_mask",
        "selected_link_proxy_status",
        "selected_link_proxy",
        "helper_queue_proxy_status",
        "helper_queue_proxy",
    },
    "terminal": {
        "route_status",
        "terminal_kind",
        "outcome",
        "observed_transition_step",
        "terminal_step",
        "source_uav",
        "first_tx_step",
        "last_tx_step",
        "first_cpu_step",
        "last_cpu_step",
        "reward_decomposition",
        "reward_ledger",
        "workload_ledger",
        "energy_ledger",
    },
    "credit": {
        "route_step",
        "rollout_index",
        "source_uav",
        "route_category",
        "selected_destination_uav",
        "route_action_index",
        "credit_status",
        "ppo_update_index",
        "ppo_epoch_index",
        "policy_version_before",
        "policy_version_after",
        "advantage",
        "td_residual",
        "return_target",
        "na_reason",
    },
}


def _nested_mapping_keys(value: Any) -> tuple[str, ...]:
    keys: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            keys.append(str(key).lower())
            keys.extend(_nested_mapping_keys(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            keys.extend(_nested_mapping_keys(child))
    return tuple(keys)


def _trajectory_route_key(record: Mapping[str, Any]) -> tuple[int, int, int] | None:
    value = record.get("route_event_key")
    if value is None:
        return None
    if (
        not isinstance(value, list)
        or len(value) != 3
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise TrajectoryCreditArtifactError(
            "route_event_key must be null or a three-integer JSON array"
        )
    return (value[0], value[1], value[2])


def _validate_trajectory_credit_record(
    config: RunConfig,
    record: Mapping[str, Any],
) -> tuple[str, tuple[int, int], tuple[int, int, int] | None]:
    if not isinstance(record, Mapping):
        raise TrajectoryCreditArtifactError("trajectory-credit record must be a mapping")
    if record.get("schema_version") != TRAJECTORY_CREDIT_SCHEMA_VERSION:
        raise TrajectoryCreditArtifactError("trajectory-credit schema version mismatch")
    event_type = record.get("event_type")
    if event_type not in TRAJECTORY_CREDIT_EVENT_TYPES:
        raise TrajectoryCreditArtifactError("unknown trajectory-credit event type")
    missing = sorted(_TRAJECTORY_REQUIRED_FIELDS[str(event_type)] - set(record))
    if missing:
        raise TrajectoryCreditArtifactError(
            f"{event_type} event is missing required fields: {missing}"
        )
    for name, expected in (
        ("run_id", config.run_id),
        ("config_hash", config.config_hash),
        ("git_commit", config.git_commit),
    ):
        if record.get(name) != expected:
            raise TrajectoryCreditArtifactError(
                f"trajectory-credit {name} differs from the run identity"
            )
    episode_id = record.get("episode_id")
    task_id = record.get("task_id")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in (episode_id, task_id)
    ):
        raise TrajectoryCreditArtifactError(
            "episode_id and task_id must be non-negative integers"
        )
    task_key = (episode_id, task_id)
    route_key = _trajectory_route_key(record)
    if route_key is not None and route_key[:2] != task_key:
        raise TrajectoryCreditArtifactError(
            "route_event_key episode/task identity is inconsistent"
        )
    if event_type in {"route", "credit"} and route_key is None:
        raise TrajectoryCreditArtifactError(
            f"{event_type} event requires a route_event_key"
        )
    if event_type == "route":
        if record.get("route_step") != route_key[2]:
            raise TrajectoryCreditArtifactError(
                "route event step differs from route_event_key"
            )
        forbidden = _FORBIDDEN_ROUTE_OBSERVATION_KEYS.intersection(
            _nested_mapping_keys(record)
        )
        if forbidden:
            raise TrajectoryCreditArtifactError(
                "route event contains non-causal channel fields: "
                + ", ".join(sorted(forbidden))
            )
        if record.get("route_status") != "applied" or record.get(
            "route_category"
        ) not in {"local", "remote"}:
            raise TrajectoryCreditArtifactError(
                "route events must describe applied local/remote routing"
            )
    elif event_type == "terminal":
        if record.get("terminal_kind") not in {"lifecycle", "collection_censored"}:
            raise TrajectoryCreditArtifactError("unknown terminal event kind")
        decomposition = record.get("reward_decomposition")
        reward_ledger = record.get("reward_ledger")
        workload = record.get("workload_ledger")
        energy = record.get("energy_ledger")
        if not all(
            isinstance(value, Mapping)
            for value in (decomposition, reward_ledger, workload, energy)
        ):
            raise TrajectoryCreditArtifactError(
                "terminal reward and conservation ledgers must be mappings"
            )
        for name in ("cumulative_workload_raw", "cumulative_workload_penalty"):
            if name not in decomposition:
                raise TrajectoryCreditArtifactError(
                    f"terminal reward decomposition lacks {name}"
                )
        for name in (
            "tx_energy_attribution_method",
            "tx_energy_attribution_status",
            "allocated_task_tx_energy",
            "unattributed_tx_energy",
        ):
            if name not in energy:
                raise TrajectoryCreditArtifactError(
                    f"terminal energy ledger lacks {name}"
                )
        if (
            energy.get("tx_energy_attribution_method") != "bits_pro_rata"
            or energy.get("tx_energy_attribution_status")
            != "diagnostic_allocation"
        ):
            raise TrajectoryCreditArtifactError(
                "terminal TX energy attribution semantics are invalid"
            )
    elif record.get("credit_status") not in {"linked", "tail_not_optimized"}:
        raise TrajectoryCreditArtifactError("unknown credit event status")
    json.dumps(record, ensure_ascii=False, allow_nan=False)
    return str(event_type), task_key, route_key


class TrajectoryCreditArtifactWriter:
    """Durably append validated V1 events to a fresh-run-only JSONL sidecar."""

    def __init__(self, config: RunConfig) -> None:
        if not isinstance(config, RunConfig):
            raise TypeError("config must be a RunConfig")
        if not config.training.mappo.trajectory_credit_telemetry_enabled:
            raise TrajectoryCreditArtifactError(
                "trajectory-credit writer requires enabled telemetry"
            )
        self.config = config
        self.path = Path(config.artifact_paths()["trajectory_credit_events"])
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self.path.open("x", encoding="utf-8", newline="\n"):
                pass
        except FileExistsError as exc:
            raise TrajectoryCreditArtifactError(
                f"trajectory-credit sidecar already exists: {self.path}"
            ) from exc
        self._counts = {name: 0 for name in TRAJECTORY_CREDIT_EVENT_TYPES}
        self._route_keys: set[tuple[int, int, int]] = set()
        self._terminal_keys: set[tuple[int, int]] = set()
        self._credit_keys: set[tuple[int, int, int]] = set()

    def write(self, record: Mapping[str, Any]) -> None:
        event_type, task_key, route_key = _validate_trajectory_credit_record(
            self.config, record
        )
        if event_type == "route":
            assert route_key is not None
            if route_key in self._route_keys:
                raise TrajectoryCreditArtifactError("duplicate route event identity")
            self._route_keys.add(route_key)
        elif event_type == "terminal":
            if task_key in self._terminal_keys:
                raise TrajectoryCreditArtifactError("duplicate terminal task identity")
            self._terminal_keys.add(task_key)
        else:
            assert route_key is not None
            if route_key in self._credit_keys:
                raise TrajectoryCreditArtifactError("duplicate credit route identity")
            self._credit_keys.add(route_key)
        encoded = json.dumps(
            dict(record),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        with self.path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self._counts[event_type] += 1

    def summary(self) -> dict[str, Any]:
        return {
            "schema_version": TRAJECTORY_CREDIT_SCHEMA_VERSION,
            "path": str(self.path),
            "event_counts": dict(self._counts),
            "route_credit_cardinality_match": self._route_keys == self._credit_keys,
        }


def inspect_trajectory_credit_artifact(config: RunConfig) -> dict[str, Any]:
    """Read back and validate identity, uniqueness, and route-credit cardinality."""

    path = Path(config.artifact_paths()["trajectory_credit_events"])
    if not path.is_file():
        raise TrajectoryCreditArtifactError(
            f"trajectory-credit sidecar is absent: {path}"
        )
    counts = {name: 0 for name in TRAJECTORY_CREDIT_EVENT_TYPES}
    route_keys: set[tuple[int, int, int]] = set()
    terminal_keys: set[tuple[int, int]] = set()
    credit_keys: set[tuple[int, int, int]] = set()
    task_keys: set[tuple[int, int]] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.endswith("\n") or not line.strip():
                raise TrajectoryCreditArtifactError(
                    f"invalid JSONL framing at line {line_number}"
                )
            try:
                record = json.loads(line)
                event_type, task_key, route_key = _validate_trajectory_credit_record(
                    config, record
                )
            except (json.JSONDecodeError, ValueError, TypeError) as exc:
                raise TrajectoryCreditArtifactError(
                    f"invalid trajectory-credit record at line {line_number}"
                ) from exc
            task_keys.add(task_key)
            counts[event_type] += 1
            if event_type == "route":
                assert route_key is not None
                if route_key in route_keys:
                    raise TrajectoryCreditArtifactError("duplicate route event identity")
                route_keys.add(route_key)
            elif event_type == "terminal":
                if task_key in terminal_keys:
                    raise TrajectoryCreditArtifactError("duplicate terminal task identity")
                terminal_keys.add(task_key)
            else:
                assert route_key is not None
                if route_key in credit_keys:
                    raise TrajectoryCreditArtifactError("duplicate credit route identity")
                credit_keys.add(route_key)
    if route_keys != credit_keys:
        raise TrajectoryCreditArtifactError(
            "route and credit event identities do not match one-to-one"
        )
    if task_keys != terminal_keys:
        raise TrajectoryCreditArtifactError(
            "every observed task must have exactly one terminal event"
        )
    return {
        "schema_version": TRAJECTORY_CREDIT_SCHEMA_VERSION,
        "path": str(path),
        "event_counts": counts,
        "task_count": len(task_keys),
        "route_credit_cardinality_match": True,
        "terminal_cardinality_match": True,
    }


def _base_record(config: RunConfig, signal_gate_status: str) -> dict[str, Any]:
    ratio_mode = config.training.mappo.actor_ratio_mode
    actor_ratio_mode = getattr(ratio_mode, "value", ratio_mode)
    credit_mode = config.training.mappo.agent_credit_mode
    agent_credit_mode = getattr(credit_mode, "value", credit_mode)
    return {
        "run_id": config.run_id,
        "method_id": config.method_id,
        "scenario_id": config.scenario_id,
        "seed": config.seed,
        "git_commit": config.git_commit,
        "config_hash": config.config_hash,
        "diagnostics_schema_version": TRAINING_DIAGNOSTICS_SCHEMA_VERSION,
        "actor_ratio_mode": actor_ratio_mode,
        "agent_credit_mode": agent_credit_mode,
        "record_type": None,
        "telemetry_scope": None,
        "series_index": None,
        "environment_steps": None,
        "collected_environment_steps": None,
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
        "route_entropy_coefficient": None,
        "route_entropy_schedule_progress": None,
        "route_entropy_loss_contribution": None,
        "other_branch_entropy_loss_contribution": None,
        "global_entropy_loss_contribution": None,
        "total_loss": None,
        "ratio": None,
        "approx_kl": None,
        "clip_fraction": None,
        "actor_grad_norm_before_clip": None,
        "critic_grad_norm_before_clip": None,
        "route_telemetry_enabled": None,
        **{column: None for column in _ROUTE_TELEMETRY_COLUMNS},
        **{column: None for column in _ROUTE_ROLLING_COLUMNS},
        **{column: None for column in _ROUTE_SAMPLE_COLUMNS},
        **{column: None for column in _ROUTE_OUTCOME_COLUMNS},
        **{column: None for column in _BRANCH_ACTIVITY_COLUMNS},
        **{column: None for column in _AGENT_CREDIT_COLUMNS},
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


def _add_route_rolling_aggregates(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Add trailing per-PPO-epoch means without replacing raw telemetry."""

    window = ROUTE_TELEMETRY_ROLLING_WINDOW_PPO_EPOCHS
    for index, record in enumerate(records):
        start = max(0, index + 1 - window)
        rows = records[start : index + 1]
        for output_name, source_name in _ROUTE_ROLLING_FIELDS:
            values = [
                float(row[source_name])
                for row in rows
                if row[source_name] is not None
            ]
            record[output_name] = (
                None if not values else math.fsum(values) / len(values)
            )
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
                    "telemetry_scope": "per_ppo_epoch",
                    "series_index": series_index,
                    "environment_steps": update_environment_steps,
                    "collected_environment_steps": epoch.collected_environment_steps,
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
                    "route_entropy_coefficient": epoch.route_entropy_coefficient,
                    "route_entropy_schedule_progress": epoch.route_entropy_schedule_progress,
                    "route_entropy_loss_contribution": epoch.route_entropy_loss_contribution,
                    "other_branch_entropy_loss_contribution": epoch.other_branch_entropy_loss_contribution,
                    "global_entropy_loss_contribution": epoch.global_entropy_loss_contribution,
                    "total_loss": epoch.total_loss,
                    "ratio": epoch.ratio_mean,
                    "approx_kl": epoch.approx_kl,
                    "clip_fraction": epoch.clip_fraction,
                    "actor_grad_norm_before_clip": epoch.actor_grad_norm_before_clip,
                    "critic_grad_norm_before_clip": epoch.critic_grad_norm_before_clip,
                    "route_telemetry_enabled": epoch.route_telemetry is not None,
                }
            )
            if epoch.route_telemetry is not None:
                telemetry_record = epoch.route_telemetry.record()
                telemetry_record["route_telemetry_schema_version"] = (
                    telemetry_record.pop("schema_version")
                )
                record.update(telemetry_record)
            if epoch.agent_credit_telemetry is not None:
                credit = epoch.agent_credit_telemetry
                record.update(
                    {
                        "agent_value_mean": list(credit.per_agent_value_mean),
                        "agent_td_residual_mean": list(
                            credit.per_agent_td_residual_mean
                        ),
                        "agent_advantage_mean": list(
                            credit.per_agent_advantage_mean
                        ),
                        "agent_return_mean": list(credit.per_agent_return_mean),
                        "same_timestep_advantage_equality_rate": (
                            credit.same_timestep_advantage_equality_rate
                        ),
                        "route_category_agent_advantage_mean": {
                            key: list(values)
                            for key, values in (
                                credit.route_category_agent_advantage_mean.items()
                            )
                        },
                        "per_head_critic_loss": list(
                            credit.per_head_critic_loss
                        ),
                    }
                )
            records.append(record)
    return _add_route_rolling_aggregates(records)


def _all_finite(values: Iterable[float]) -> bool:

    return all(math.isfinite(float(value)) for value in values)

def _route_sample_records(
    config: RunConfig,
    training: Any,
    signal_gate_status: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    series_index = 0
    mappo = config.training.mappo
    for update in training.updates:
        environment_steps = min(
            (update.update_index + 1) * mappo.rollout_length_slots,
            training.total_environment_transitions,
        )
        for epoch in update.output.epoch_diagnostics:
            series_index += 1
            telemetry = epoch.route_telemetry
            if telemetry is None:
                continue
            for sample in getattr(telemetry, "samples", ()) or ():
                record = _base_record(config, signal_gate_status)
                record.update({
                    "record_type": "route_sample",
                    "telemetry_scope": "per_route_active_sample",
                    "series_index": series_index,
                    "environment_steps": environment_steps,
                    "update_index": update.update_index,
                    "ppo_epoch_index": epoch.epoch_index,
                    "policy_version_before": update.rollout_policy_version,
                    "policy_version_after": update.policy_version_after_update,
                    "ppo_clip_epsilon": mappo.ppo_clip_epsilon,
                    "route_telemetry_enabled": True,
                    "route_telemetry_schema_version": telemetry.schema_version,
                })
                record.update(sample.record())
                records.append(record)
    return records


def _route_outcome_records(
    config: RunConfig,
    training: Any,
    signal_gate_status: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for series_index, outcome in enumerate(
        getattr(training, "route_outcomes", ()) or (), start=1
    ):
        if not isinstance(outcome, RouteOutcomeAssociation):
            continue
        record = _base_record(config, signal_gate_status)
        record.update({
            "record_type": "route_outcome",
            "telemetry_scope": "route_decision_to_task_outcome",
            "series_index": series_index,
            "route_telemetry_enabled": True,
            "route_telemetry_schema_version": ROUTE_TELEMETRY_SCHEMA_VERSION,
        })
        record.update(outcome.record())
        records.append(record)
    return records


def _branch_activity_records(
    config: RunConfig,
    training: Any,
    signal_gate_status: str,
) -> list[dict[str, Any]]:
    """Persist one complete [A,7] activity matrix per rollout timestep."""

    records: list[dict[str, Any]] = []
    series_index = 0
    mappo = config.training.mappo
    for update in training.updates:
        telemetry = update.output.epoch_diagnostics[0].route_telemetry
        if telemetry is None:
            continue
        environment_steps = min(
            (update.update_index + 1) * mappo.rollout_length_slots,
            training.total_environment_transitions,
        )
        for timestep, matrix in enumerate(telemetry.active_branch_matrix):
            series_index += 1
            record = _base_record(config, signal_gate_status)
            record.update(
                {
                    "record_type": "branch_activity",
                    "telemetry_scope": "per_rollout_timestep",
                    "series_index": series_index,
                    "environment_steps": environment_steps,
                    "update_index": update.update_index,
                    "policy_version_before": update.rollout_policy_version,
                    "policy_version_after": update.policy_version_after_update,
                    "route_telemetry_enabled": True,
                    "route_telemetry_schema_version": telemetry.schema_version,
                    "branch_activity_rollout_timestep": timestep,
                    "branch_activity_matrix": [list(row) for row in matrix],
                }
            )
            records.append(record)
    return records


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
    stability_values = [
        value
        for epoch in ppo_epochs
        for value in (epoch.approx_kl, epoch.clip_fraction)
        if value is not None
    ]
    route_telemetry = [
        epoch.route_telemetry
        for epoch in ppo_epochs
        if epoch.route_telemetry is not None
    ]
    schedule_rows = [
        (update, epoch)
        for update in training.updates
        for epoch in update.output.epoch_diagnostics
        if epoch.collected_environment_steps is not None
    ]
    schedule_group_presence_is_valid = all(
        sum(
            epoch.collected_environment_steps is not None
            for epoch in update.output.epoch_diagnostics
        )
        in (0, config.training.mappo.update_epochs)
        for update in training.updates
    )
    schedule_identity_holds = all(
        math.isclose(
            epoch.global_entropy_loss_contribution,
            epoch.route_entropy_loss_contribution
            + epoch.other_branch_entropy_loss_contribution,
            rel_tol=1.0e-7,
            abs_tol=1.0e-9,
        )
        for _, epoch in schedule_rows
    )
    schedule_reconstruction_holds = all(
        epoch.collected_environment_steps
        == (update.update_index + 1)
        * config.training.mappo.rollout_length_slots
        and math.isclose(
            epoch.route_entropy_coefficient,
            compute_route_entropy_schedule(
                config.training.mappo,
                epoch.collected_environment_steps,
            ).coefficient,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        and math.isclose(
            epoch.route_entropy_schedule_progress,
            compute_route_entropy_schedule(
                config.training.mappo,
                epoch.collected_environment_steps,
            ).progress,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        for update, epoch in schedule_rows
    )
    schedule_presence_is_valid = (
        not config.training.mappo.entropy_coefficient_schedule_enabled
        or bool(schedule_rows)
    )
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
        "ppo_stability_telemetry_is_finite": _all_finite(stability_values),
        "route_telemetry_schema_is_valid": all(
            item.schema_version == ROUTE_TELEMETRY_SCHEMA_VERSION
            for item in route_telemetry
        ),
        "route_entropy_schedule_telemetry_identity_holds": (
            schedule_group_presence_is_valid
            and schedule_identity_holds
            and schedule_reconstruction_holds
            and schedule_presence_is_valid
        ),
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
    csv_records = []
    for record in records:
        normalized = dict(record)
        for name in (
            "branch_activity_matrix",
            "agent_value_mean",
            "agent_td_residual_mean",
            "agent_advantage_mean",
            "agent_return_mean",
            "route_category_agent_advantage_mean",
            "per_head_critic_loss",
        ):
            value = normalized.get(name)
            if value is not None:
                normalized[name] = json.dumps(
                    value,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
        csv_records.append(normalized)
    writer.writerows(csv_records)
    return stream.getvalue()


def _json_lines(records: list[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
        for record in records
    )


def read_training_metrics_csv_text(csv_text: str) -> tuple[dict[str, str], ...]:
    """Read legacy or current metrics with absent newer fields as blanks."""

    if not isinstance(csv_text, str):
        raise TypeError("csv_text must be a string")
    rows = []
    for row in csv.DictReader(io.StringIO(csv_text)):
        normalized = {column: row.get(column, "") or "" for column in TRAINING_METRIC_COLUMNS}
        normalized["diagnostics_schema_version"] = (
            row.get("diagnostics_schema_version") or "1"
        )
        rows.append(normalized)
    return tuple(rows)


def _mean_present(values: Iterable[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return None if not present else math.fsum(present) / len(present)


def _route_telemetry_summary(training: Any) -> dict[str, Any]:
    epochs = [
        epoch
        for update in training.updates
        for epoch in update.output.epoch_diagnostics
    ]
    present = [epoch.route_telemetry for epoch in epochs if epoch.route_telemetry is not None]
    unique_rollouts = [
        update.output.epoch_diagnostics[0].route_telemetry
        for update in training.updates
        if update.output.epoch_diagnostics[0].route_telemetry is not None
    ]
    count_fields = (
        "route_branch_active_count",
        "legal_remote_route_count",
        "route_selected_local_count",
        "route_selected_remote_count",
        "route_selected_defer_count",
    )
    curve_fields = (
        "route_entropy_mean",
        "remote_selection_rate_given_legal_remote",
        "mean_local_probability",
        "mean_best_remote_probability",
        "mean_local_minus_best_remote_logit_margin",
        "route_head_total_grad_norm",
        "route_active_advantage_mean",
        "mean_defer_probability_all_route_active",
        "mean_total_remote_probability_mass",
        "mean_local_minus_total_remote_probability",
        "mean_local_minus_total_remote_logit_margin",
    )
    return {
        "schema_version": ROUTE_TELEMETRY_SCHEMA_VERSION,
        "per_ppo_epoch": {
            "record_type": "ppo_epoch",
            "telemetry_scope": "per_ppo_epoch",
            "present_count": len(present),
            "missing_count": len(epochs) - len(present),
            "sample_counts_repeat_across_the_four_epochs_of_one_update": True,
        },
        "rolling_window": {
            "window_ppo_epochs": ROUTE_TELEMETRY_ROLLING_WINDOW_PPO_EPOCHS,
            "semantics": "trailing rows, valid values only, NA when the window has no valid value",
            "columns": list(_ROUTE_ROLLING_COLUMNS),
        },
        "final_summary": {
            "unique_rollout_sample_totals": {
                name: sum(getattr(item, name) for item in unique_rollouts)
                for name in count_fields
            },
            "mean_over_valid_ppo_epochs": {
                name: _mean_present(getattr(item, name) for item in present)
                for name in curve_fields
            },
            "latest_ppo_epoch": None if not present else present[-1].record(),
        },
        "per_route_active_sample": {
            "record_type": "route_sample",
            "telemetry_scope": "per_route_active_sample",
            "present_count": sum(len(getattr(item, "samples", ()) or ()) for item in present),
        },
        "per_branch_ppo": {
            "branches": list(config_branch for config_branch in (
                "route",
                "tx_select",
                "resource_group",
                "resource_width",
                "power_level",
                "cpu_queue",
                "cpu_frequency",
            )),
            "statistics": [
                "active_count",
                "ratio_mean",
                "ratio_median",
                "ratio_std",
                "approx_kl_mean",
                "clip_fraction",
                "surrogate_contribution_mean",
                "advantage_mean",
            ],
            "present_ppo_epoch_count": sum(
                item.branch_ppo_dynamics is not None for item in present
            ),
        },
        "branch_activity": {
            "record_type": "branch_activity",
            "telemetry_scope": "per_rollout_timestep",
            "matrix_shape": "[A,7]",
            "present_count": sum(
                len(item.active_branch_matrix) for item in unique_rollouts
            ),
        },
        "route_outcomes": aggregate_route_outcomes(getattr(training, "route_outcomes", ()) or ()),
        "na_semantics": (
            "CSV blank and JSON null mean no valid sample or legacy checkpoint telemetry; "
            "zero is retained only for an observed numeric zero"
        ),
        "checkpoint_v1_compatibility": (
            "route telemetry is intentionally excluded from Checkpoint V1 payload fields; "
            "restored legacy epoch diagnostics use NA and the checkpoint identity is unchanged"
        ),
    }


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
    ppo_epochs = [
        epoch
        for update in training.updates
        for epoch in update.output.epoch_diagnostics
    ]
    schedule_epochs = [
        epoch
        for epoch in ppo_epochs
        if epoch.collected_environment_steps is not None
    ]
    latest_schedule_epoch = None if not schedule_epochs else schedule_epochs[-1]
    return {
        "run_id": config.run_id,
        "method_id": config.method_id,
        "scenario_id": config.scenario_id,
        "seed": config.seed,
        "diagnostics_schema_version": TRAINING_DIAGNOSTICS_SCHEMA_VERSION,
        "validation_stage": "smoke-training",
        "smoke_training_gate_status": smoke_gate_status,
        "signal_gate_status": signal_gate_status,
        "checks": dict(checks),
        "training": {
            "training_device": config.training.mappo.training_device,
            "actor_ratio_mode": getattr(
                config.training.mappo.actor_ratio_mode,
                "value",
                config.training.mappo.actor_ratio_mode,
            ),
            "agent_credit_mode": getattr(
                config.training.mappo.agent_credit_mode,
                "value",
                config.training.mappo.agent_credit_mode,
            ),
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
            "approx_kl_mean": _mean_present(epoch.approx_kl for epoch in ppo_epochs),
            "clip_fraction_mean": _mean_present(epoch.clip_fraction for epoch in ppo_epochs),
            "actor_grad_norm_before_clip_mean": ppo.actor_grad_norm_before_clip.mean,
            "critic_grad_norm_before_clip_mean": ppo.critic_grad_norm_before_clip.mean,
            "route_entropy_schedule_latest": (
                None
                if latest_schedule_epoch is None
                else {
                    "route_entropy_coefficient": (
                        latest_schedule_epoch.route_entropy_coefficient
                    ),
                    "route_entropy_schedule_progress": (
                        latest_schedule_epoch.route_entropy_schedule_progress
                    ),
                    "route_entropy_loss_contribution": (
                        latest_schedule_epoch.route_entropy_loss_contribution
                    ),
                    "other_branch_entropy_loss_contribution": (
                        latest_schedule_epoch.other_branch_entropy_loss_contribution
                    ),
                    "global_entropy_loss_contribution": (
                        latest_schedule_epoch.global_entropy_loss_contribution
                    ),
                    "collected_environment_steps": (
                        latest_schedule_epoch.collected_environment_steps
                    ),
                }
            ),
            "agent_credit_latest": (
                None
                if not ppo_epochs
                or ppo_epochs[-1].agent_credit_telemetry is None
                else {
                    "per_agent_value_mean": list(
                        ppo_epochs[-1].agent_credit_telemetry.per_agent_value_mean
                    ),
                    "per_agent_td_residual_mean": list(
                        ppo_epochs[-1].agent_credit_telemetry.per_agent_td_residual_mean
                    ),
                    "per_agent_advantage_mean": list(
                        ppo_epochs[-1].agent_credit_telemetry.per_agent_advantage_mean
                    ),
                    "per_agent_return_mean": list(
                        ppo_epochs[-1].agent_credit_telemetry.per_agent_return_mean
                    ),
                    "same_timestep_advantage_equality_rate": (
                        ppo_epochs[-1]
                        .agent_credit_telemetry
                        .same_timestep_advantage_equality_rate
                    ),
                    "route_category_agent_advantage_mean": {
                        key: list(values)
                        for key, values in (
                            ppo_epochs[-1]
                            .agent_credit_telemetry
                            .route_category_agent_advantage_mean.items()
                        )
                    },
                    "per_head_critic_loss": list(
                        ppo_epochs[-1].agent_credit_telemetry.per_head_critic_loss
                    ),
                }
            ),
        },
        "route_telemetry": _route_telemetry_summary(training),
        "dashboard": {
            "source_csv": dashboard_csv,
            "reward_series": "episode total reward",
            "ppo_series": "four diagnostics per completed PPO update",
            "reward_smoothing": (
                "trailing arithmetic mean, window up to "
                f"{REWARD_SMOOTHING_WINDOW_EPISODES} episodes, min_periods=1"
            ),
            "route_telemetry_smoothing": (
                "trailing arithmetic mean over up to "
                f"{ROUTE_TELEMETRY_ROLLING_WINDOW_PPO_EPOCHS} PPO epoch rows; "
                "NA values are excluded"
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


def _optional_float_series(
    rows: Iterable[Mapping[str, str]],
    name: str,
) -> list[float]:
    return [
        math.nan if row.get(name) in {None, ""} else float(row[name])
        for row in rows
    ]


def _has_finite(values: Iterable[float]) -> bool:
    return any(math.isfinite(value) for value in values)


def _plot_dashboard(
    csv_text: str,
    output_stream: io.BytesIO,
    summary: Mapping[str, Any],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "matplotlib is required to generate the training dashboard"
        ) from exc

    rows = list(read_training_metrics_csv_text(csv_text))
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
    approx_kl = _optional_float_series(ppo_rows, "approx_kl")
    clip_fraction = _optional_float_series(ppo_rows, "clip_fraction")
    route_x = [int(row["environment_steps"]) for row in ppo_rows]
    route_entropy = _optional_float_series(ppo_rows, "route_entropy_mean")
    route_entropy_rolling = _optional_float_series(ppo_rows, "route_entropy_rolling_mean")
    remote_rate = _optional_float_series(ppo_rows, "remote_selection_rate_given_legal_remote")
    local_probability = _optional_float_series(ppo_rows, "mean_local_probability")
    best_remote_probability = _optional_float_series(ppo_rows, "mean_best_remote_probability")
    logit_margin = _optional_float_series(ppo_rows, "mean_local_minus_best_remote_logit_margin")
    route_grad = _optional_float_series(ppo_rows, "route_head_total_grad_norm")
    route_advantage = _optional_float_series(ppo_rows, "route_active_advantage_mean")

    fig, axes = plt.subplots(4, 2, figsize=(11.0, 13.0))
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
    if _has_finite(approx_kl):
        ratio_axis.plot(ppo_x, approx_kl, color=colors[3], label="approx KL")
    if _has_finite(clip_fraction):
        ratio_axis.plot(ppo_x, clip_fraction, color=colors[1], label="clip fraction")
    ratio_axis.set_ylabel("Policy ratio")
    handles = [ax.get_lines()[0], *ratio_axis.get_lines()]
    ax.legend(handles, [line.get_label() for line in handles], loc="best")

    route_series = (
        route_entropy,
        remote_rate,
        local_probability,
        best_remote_probability,
        logit_margin,
        route_grad,
        route_advantage,
    )
    if not any(_has_finite(values) for values in route_series):
        for route_ax in axes[2:, :].flat:
            route_ax.text(
                0.5,
                0.5,
                "Route telemetry unavailable (legacy/disabled)",
                ha="center",
                va="center",
                transform=route_ax.transAxes,
            )
            route_ax.set_axis_off()
    else:
        ax = axes[2, 0]
        ax.plot(route_x, route_entropy, color=colors[2], marker="o", label="route entropy")
        ax.plot(route_x, route_entropy_rolling, color=colors[2], linestyle="--", label="entropy rolling")
        ax.set(title="Route entropy / remote selection", xlabel="Environment steps", ylabel="Entropy")
        rate_axis = ax.twinx()
        rate_axis.plot(route_x, remote_rate, color=colors[1], marker="s", label="remote rate")
        rate_axis.set_ylabel("P(remote | legal remote)")
        handles = ax.get_lines() + rate_axis.get_lines()
        ax.legend(handles, [line.get_label() for line in handles], loc="best")

        ax = axes[2, 1]
        ax.plot(route_x, local_probability, color=colors[0], marker="o", label="P(local)")
        ax.plot(route_x, best_remote_probability, color=colors[3], marker="s", label="P(best remote)")
        ax.set(title="Legal-remote route probabilities", xlabel="Environment steps", ylabel="Probability")
        ax.legend(loc="best")

        ax = axes[3, 0]
        ax.plot(route_x, logit_margin, color=colors[4], marker="o", label="local - best remote logit")
        ax.axhline(0.0, color="0.35", linewidth=1.0, linestyle=":")
        ax.set(title="Route margin / gradient", xlabel="Environment steps", ylabel="Logit margin")
        grad_axis = ax.twinx()
        grad_axis.plot(route_x, route_grad, color=colors[1], marker="s", label="route-head grad norm")
        grad_axis.set_ylabel("Gradient norm")
        handles = ax.get_lines()[:1] + grad_axis.get_lines()
        ax.legend(handles, [line.get_label() for line in handles], loc="best")

        ax = axes[3, 1]
        ax.plot(route_x, route_advantage, color=colors[0], marker="o", label="route-active advantage")
        ax.axhline(0.0, color="0.35", linewidth=1.0, linestyle=":")
        ax.set(title="Route-active advantage", xlabel="Environment steps", ylabel="Advantage")
        ax.legend(loc="best")

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
        fig.savefig(output_stream, format="png", dpi=220, bbox_inches="tight")
    finally:
        plt.close(fig)


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
    records.extend(_route_sample_records(config, training, signal_gate_status))
    records.extend(_route_outcome_records(config, training, signal_gate_status))
    records.extend(_branch_activity_records(config, training, signal_gate_status))
    csv_text = _csv_text(records)

    paths = config.artifact_paths()
    snapshot_path = Path(paths["config_snapshot"])
    raw_path = Path(paths["raw_metrics"])
    aggregate_path = Path(paths["aggregate_metrics"])
    csv_path = Path(paths["dashboard_csv"])
    dashboard_path = Path(paths["dashboard_png"])
    trajectory_summary = None
    trajectory_path = Path(paths["trajectory_credit_events"])
    if config.training.mappo.trajectory_credit_telemetry_enabled:
        trajectory_summary = inspect_trajectory_credit_artifact(config)
    artifact_paths = (
        snapshot_path,
        raw_path,
        aggregate_path,
        csv_path,
        dashboard_path,
    )
    require_artifact_targets_absent(
        artifact_paths,
        group_name="CA-GAT-MAPPO training artifact group",
    )
    summary = _training_summary(
        config,
        training,
        checks,
        smoke_gate_status,
        signal_gate_status,
        str(csv_path),
    )
    if trajectory_summary is not None:
        summary["trajectory_credit_telemetry"] = trajectory_summary

    dashboard_stream = io.BytesIO()
    _plot_dashboard(csv_text, dashboard_stream, summary)
    atomic_write_bytes_group(
        (
            (
                snapshot_path,
                (
                    json.dumps(
                        config.snapshot_dict(),
                        ensure_ascii=False,
                        indent=2,
                        allow_nan=False,
                    )
                    + "\n"
                ).encode("utf-8"),
            ),
            (raw_path, _json_lines(records).encode("utf-8")),
            (
                aggregate_path,
                (
                    json.dumps(
                        summary,
                        ensure_ascii=False,
                        indent=2,
                        sort_keys=True,
                        allow_nan=False,
                    )
                    + "\n"
                ).encode("utf-8"),
            ),
            (csv_path, csv_text.encode("utf-8")),
            (dashboard_path, dashboard_stream.getvalue()),
        ),
        group_name="CA-GAT-MAPPO training artifact group",
    )

    artifacts = tuple(str(path) for path in artifact_paths)
    if trajectory_summary is not None:
        artifacts = (*artifacts, str(trajectory_path))
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
    "TrajectoryCreditArtifactError",
    "TrajectoryCreditArtifactWriter",
    "inspect_trajectory_credit_artifact",
    "write_cagat_mappo_training_artifacts",
]
