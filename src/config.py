"""Typed experiment configuration and reproducibility metadata.

This module is the single configuration entry point for the interactive launcher,
direct CLI calls, tests, and future environment backends.  It intentionally uses
only the Python standard library so that the configuration layer remains usable
before the environment dependencies are added.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import platform
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, TypeVar, get_args, get_origin, get_type_hints


CONFIG_VERSION = "section4.v1"
DEFAULT_SEED = 42
SUPPORTED_MODES = (
    "environment_sanity",
    "gate0",
    "random",
    "heuristic",
    "baseline",
    "rl",
    "evaluation",
    "ablation",
    "plot",
)
SUPPORTED_METHODS = (
    "environment",
    "random",
    "heuristic",
    "ca_gat_mappo",
    "factorized_action_gat_qmix",
)
SUPPORTED_SCENARIOS = ("small", "medium", "large")

STREAM_IDS = {
    "reset_mobility": 10,
    "task_arrival": 20,
    "task_workload": 30,
    "channel_fading": 40,
    "csi_error": 50,
    "interference_measurement": 60,
    "python_utility": 70,
    "torch_initialization": 100,
    "torch_policy_sampling": 110,
    "qmix_epsilon": 120,
}

_T = TypeVar("_T")


class ConfigError(ValueError):
    """Raised when a config file or CLI override violates the schema."""


@dataclass(frozen=True)
class BoundsConfig:
    x_min_m: float = 0.0
    x_max_m: float = 1000.0
    y_min_m: float = 0.0
    y_max_m: float = 1000.0


@dataclass(frozen=True)
class BuildingConfig:
    name: str
    x_min_m: float
    x_max_m: float
    y_min_m: float
    y_max_m: float
    height_m: float


@dataclass(frozen=True)
class EnvironmentConfig:
    """Frozen Section 4 environment values used by the future backend."""

    episode_horizon: int = 500
    slot_duration_s: float = 0.020
    height_m: float = 80.0
    bounds: BoundsConfig = field(default_factory=BoundsConfig)
    boundary_rule: str = "coordinate-wise specular reflection"
    reset_safe_distance_m: float = 50.0
    velocity_mean_mps: tuple[float, float] = (0.0, 0.0)
    velocity_std_mps: tuple[float, float] = (1.0, 1.0)
    gauss_markov_alpha: float = 0.85
    candidate_neighbor_radius_m: float = 500.0
    total_bandwidth_hz: float = 20e6
    ru_count: int = 20
    resource_group_count: int = 5
    carrier_frequency_hz: float = 3.5e9
    reference_transmit_power_w: float = 1.0
    reference_cpu_frequency_hz: float = 2.0e9
    reference_cpu_coefficient: float = 1.0e-28
    reference_initial_energy_j: float = 30.0
    reference_rate_bps: float = 20e6
    antenna_gain_dbi: float = 2.0
    noise_psd_dbm_hz: float = -174.0
    receiver_noise_figure_db: float = 7.0
    building_loss_db: float = 15.0
    los_rician_k_db: float = 6.0
    shadowing_std_db: float = 4.0
    shadowing_correlation_distance_m: float = 50.0
    fixed_csi_aoi_slots: int = 3
    csi_error_std_db: float = 2.0
    interference_ema_beta: float = 0.8
    initial_interference_w: float = 0.0
    outage_threshold_db: float = -3.0
    minimum_task_slack_slots: int = 10
    maximum_task_slack_slots: int = 150
    uav_count: int = 4
    arrival_probabilities: tuple[float, ...] = (0.04, 0.05, 0.06, 0.05)
    profile_assignment: tuple[str, ...] = (
        "Balanced",
        "Resource-poor",
        "Compute-rich",
        "Energy-limited",
    )
    building_layout: tuple[BuildingConfig, ...] = field(
        default_factory=lambda: (
            BuildingConfig("B1", 150.0, 250.0, 150.0, 350.0, 120.0),
            BuildingConfig("B2", 400.0, 500.0, 150.0, 350.0, 120.0),
            BuildingConfig("B3", 550.0, 650.0, 650.0, 850.0, 120.0),
            BuildingConfig("B4", 750.0, 850.0, 650.0, 850.0, 120.0),
        )
    )
    profile_ratios: Mapping[str, tuple[float, float, float, float]] = field(
        default_factory=lambda: {
            "Resource-poor": (0.7, 0.8, 0.8, 1.2),
            "Balanced": (1.0, 1.0, 1.0, 1.0),
            "Compute-rich": (1.5, 1.0, 1.1, 0.9),
            "Energy-limited": (1.0, 0.8, 0.5, 1.1),
            "Communication-rich": (1.0, 1.5, 1.1, 1.0),
        }
    )

    @property
    def ru_bandwidth_hz(self) -> float:
        return self.total_bandwidth_hz / self.ru_count

    @property
    def ru_per_group(self) -> int:
        return self.ru_count // self.resource_group_count


@dataclass(frozen=True)
class ActionConfig:
    """Seven fixed factorized action branches from Sections 3 and 4."""

    route_fixed_actions: tuple[str, ...] = ("idle", "local", "defer")
    tx_select_idle_action: str = "idle"
    resource_group_count: int = 5
    resource_width_options: tuple[int, ...] = (1, 2)
    power_levels: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0)
    cpu_queue_idle_action: str = "idle"
    cpu_frequency_levels: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0)
    resource_group_numbering: str = "one_based_contiguous"
    sampling_order: tuple[str, ...] = (
        "route",
        "tx_select",
        "resource_group",
        "resource_width",
        "power_level",
        "cpu_queue",
        "cpu_frequency",
    )
    canonical_inactive_values: Mapping[str, Any] = field(
        default_factory=lambda: {
            "route": "idle",
            "tx_select": "idle",
            "resource_group": "idle",
            "resource_width": 1,
            "power_level": 0.0,
            "cpu_queue": "idle",
            "cpu_frequency": 0.0,
        }
    )

    @property
    def resource_groups(self) -> tuple[tuple[int, ...], ...]:
        width = self.resource_group_count
        return tuple(
            tuple(range(index * width + 1, (index + 1) * width + 1))
            for index in range(self.resource_group_count)
        )


@dataclass(frozen=True)
class ReproducibilityConfig:
    seed_policy: str = "single_seed_default_42"
    stream_ids: Mapping[str, int] = field(default_factory=lambda: dict(STREAM_IDS))
    snapshot_enabled: bool = True
    backend_schema_version: str = CONFIG_VERSION
    python_version: str = "TO VERIFY AT IMPLEMENTATION PREFLIGHT"
    numpy_version: str = "TO VERIFY AT IMPLEMENTATION PREFLIGHT"
    torch_version: str = "TO VERIFY AT IMPLEMENTATION PREFLIGHT"
    cuda_version: str = "TO VERIFY AT IMPLEMENTATION PREFLIGHT"
    miniconda_version: str = "TO VERIFY AT IMPLEMENTATION PREFLIGHT"


@dataclass(frozen=True)
class MAPPOConfig:
    encoder_hidden_dimension: int = 128
    gat_layer_count: int = 1
    attention_head_count: int = 4
    gru_hidden_dimension: int = 128
    actor_learning_rate: float = 3.0e-4
    critic_learning_rate: float = 3.0e-4
    optimizer: str = "Adam"
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ppo_clip_epsilon: float = 0.20
    entropy_coefficient: float = 0.01
    value_coefficient: float = 0.50
    rollout_length_slots: int = 256
    recurrent_chunk_length_slots: int = 32
    sequence_minibatch_size: int = 8
    update_epochs: int = 4
    gradient_clip_norm: float = 0.5
    max_training_episodes: int = 1000
    max_training_environment_steps: int = 500000
    evaluation_interval_steps: int = 50000
    checkpoint_interval_steps: int = 50000


@dataclass(frozen=True)
class QMIXConfig:
    encoder_hidden_dimension: int = 128
    gat_layer_count: int = 1
    attention_head_count: int = 4
    gru_hidden_dimension: int = 128
    learning_rate: float = 5.0e-4
    gamma: float = 0.99
    optimizer: str = "Adam"
    replay_capacity_sequences: int = 10000
    recurrent_sequence_length_slots: int = 32
    sequence_batch_size: int = 32
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 100000
    target_update_interval_updates: int = 2000
    max_training_episodes: int = 1000
    max_training_environment_steps: int = 500000
    checkpoint_interval_steps: int = 50000


@dataclass(frozen=True)
class TrainingConfig:
    mappo: MAPPOConfig = field(default_factory=MAPPOConfig)
    qmix: QMIXConfig = field(default_factory=QMIXConfig)
    formal_rl_enabled: bool = False


@dataclass(frozen=True)
class EvaluationConfig:
    train_seeds: tuple[int, ...] = (42, 43, 44, 45, 46)
    evaluation_seeds: tuple[int, ...] = (1042, 1043, 1044, 1045, 1046)
    inference_rule_mappo: str = "masked_argmax"
    inference_rule_qmix: str = "masked_greedy"
    update_network: bool = False
    metric_statistics: tuple[str, ...] = ("mean", "std", "valid_sample_count")


@dataclass(frozen=True)
class OutputConfig:
    logs_dir: str = "logs"
    plots_dir: str = "plots"
    dashboard_logs_dir: str = "dashboard_logs"
    snapshot_filename: str = "config_snapshot.yaml"
    raw_metrics_filename: str = "raw_metrics.jsonl"
    aggregate_metrics_filename: str = "aggregate_metrics.json"


@dataclass(frozen=True)
class RunConfig:
    """Canonical, typed configuration shared by every execution path."""

    config_version: str = CONFIG_VERSION
    mode: str = "environment_sanity"
    method_id: str = "environment"
    scenario_id: str = "small"
    seed: int = DEFAULT_SEED
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    action: ActionConfig = field(default_factory=ActionConfig)
    reproducibility: ReproducibilityConfig = field(default_factory=ReproducibilityConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # Provenance is attached after schema parsing and is deliberately excluded
    # from the resolved config hash.  It is still present in snapshots.
    source_config_path: str | None = field(default=None, init=False, repr=False, compare=False)
    cli_overrides: Mapping[str, Any] = field(default_factory=dict, init=False, repr=False, compare=False)
    interactive_overrides: Mapping[str, Any] = field(default_factory=dict, init=False, repr=False, compare=False)
    git_branch: str = field(default="unknown", init=False, repr=False, compare=False)
    git_commit: str = field(default="unknown", init=False, repr=False, compare=False)
    git_dirty: bool = field(default=False, init=False, repr=False, compare=False)

    def resolved_dict(self) -> dict[str, Any]:
        """Return only typed final values used to compute ``config_hash``."""

        return _jsonable(asdict(self), exclude={
            "source_config_path",
            "cli_overrides",
            "interactive_overrides",
            "git_branch",
            "git_commit",
            "git_dirty",
        })

    def to_dict(self) -> dict[str, Any]:
        """Alias used by callers that need the canonical resolved mapping."""

        return self.resolved_dict()

    @property
    def config_hash(self) -> str:
        payload = json.dumps(self.resolved_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def run_id(self) -> str:
        return (
            f"{self.method_id}__{self.scenario_id}__seed-{self.seed}__"
            f"cfg-{self.config_hash[:12]}__git-{self.git_commit[:12]}"
        )

    @property
    def derived_stream_ids(self) -> dict[str, int]:
        return dict(self.reproducibility.stream_ids)

    def snapshot_dict(self) -> dict[str, Any]:
        """Return a serializable snapshot with provenance and run metadata."""

        snapshot = self.resolved_dict()
        snapshot["_metadata"] = {
            "source_config_path": self.source_config_path,
            "cli_overrides": _jsonable(dict(self.cli_overrides)),
            "interactive_overrides": _jsonable(dict(self.interactive_overrides)),
            "run_id": self.run_id,
            "config_hash": self.config_hash,
            "git_branch": self.git_branch,
            "git_commit": self.git_commit,
            "git_dirty": self.git_dirty,
            "derived_stream_ids": self.derived_stream_ids,
            "runtime_versions": {
                "python": self.reproducibility.python_version,
                "numpy": self.reproducibility.numpy_version,
                "torch": self.reproducibility.torch_version,
                "cuda": self.reproducibility.cuda_version,
                "miniconda": self.reproducibility.miniconda_version,
            },
        }
        return snapshot

    def artifact_paths(self) -> dict[str, str]:
        """Return contract paths without creating files or directories."""

        run_dir = Path(self.output.logs_dir) / self.run_id
        return {
            "config_snapshot": str(run_dir / self.output.snapshot_filename),
            "raw_metrics": str(run_dir / self.output.raw_metrics_filename),
            "aggregate_metrics": str(run_dir / self.output.aggregate_metrics_filename),
            "dashboard_csv": str(Path(self.output.dashboard_logs_dir) / f"{self.run_id}_metrics.csv"),
            "figure_input": str(Path(self.output.plots_dir) / f"{self.run_id}_figure_input.csv"),
            "dashboard_png": str(Path(self.output.plots_dir) / f"{self.run_id}_dashboard.png"),
        }

    def validate(self) -> None:
        if self.config_version != CONFIG_VERSION:
            raise ConfigError(f"config_version must be {CONFIG_VERSION!r}, got {self.config_version!r}")
        if self.mode not in SUPPORTED_MODES:
            raise ConfigError(f"mode must be one of {SUPPORTED_MODES}, got {self.mode!r}")
        if self.method_id not in SUPPORTED_METHODS:
            raise ConfigError(f"method_id must be one of {SUPPORTED_METHODS}, got {self.method_id!r}")
        if self.scenario_id not in SUPPORTED_SCENARIOS:
            raise ConfigError(f"scenario_id must be one of {SUPPORTED_SCENARIOS}, got {self.scenario_id!r}")
        if self.seed < 0:
            raise ConfigError("seed must be a non-negative integer")
        expected_method = _default_method_for_mode(self.mode)
        if self.mode in {"environment_sanity", "gate0"} and self.method_id != "environment":
            raise ConfigError(f"mode {self.mode!r} requires method_id 'environment'")
        if self.mode == "random" and self.method_id != "random":
            raise ConfigError("mode 'random' requires method_id 'random'")
        if self.mode == "heuristic" and self.method_id != "heuristic":
            raise ConfigError("mode 'heuristic' requires method_id 'heuristic'")
        if self.mode == "rl" and self.method_id not in {"ca_gat_mappo", "factorized_action_gat_qmix"}:
            raise ConfigError("mode 'rl' requires a registered RL method")
        if self.mode == "baseline" and self.method_id != "factorized_action_gat_qmix":
            raise ConfigError("mode 'baseline' requires method_id 'factorized_action_gat_qmix'")
        if expected_method and self.mode in {"evaluation", "ablation", "plot"}:
            # These modes are early-stage menu entries.  Keep a single explicit
            # default while allowing a future method-specific evaluator.
            if self.method_id not in SUPPORTED_METHODS:
                raise ConfigError(f"unsupported method for mode {self.mode!r}")
        env = self.environment
        if env.episode_horizon <= 0 or env.slot_duration_s <= 0:
            raise ConfigError("episode_horizon and slot_duration_s must be positive")
        if env.uav_count <= 0 or env.height_m <= 0:
            raise ConfigError("uav_count and height_m must be positive")
        bounds = env.bounds
        if bounds.x_max_m <= bounds.x_min_m or bounds.y_max_m <= bounds.y_min_m:
            raise ConfigError("environment bounds must have positive width and height")
        if env.boundary_rule != "coordinate-wise specular reflection":
            raise ConfigError("boundary_rule must be 'coordinate-wise specular reflection'")
        if env.reset_safe_distance_m <= 0 or env.candidate_neighbor_radius_m <= 0:
            raise ConfigError("reset safe distance and candidate neighbor radius must be positive")
        if len(env.velocity_mean_mps) != 2 or len(env.velocity_std_mps) != 2:
            raise ConfigError("velocity mean/std must each contain x and y values")
        if any(value < 0 for value in env.velocity_std_mps):
            raise ConfigError("velocity standard deviations must be non-negative")
        if not 0.0 <= env.gauss_markov_alpha <= 1.0:
            raise ConfigError("gauss_markov_alpha must be in [0, 1]")
        if env.ru_count <= 0 or env.resource_group_count <= 0 or env.ru_count % env.resource_group_count:
            raise ConfigError("ru_count must be positive and divisible by resource_group_count")
        if env.total_bandwidth_hz <= 0 or env.carrier_frequency_hz <= 0:
            raise ConfigError("bandwidth and carrier frequency must be positive")
        if env.reference_transmit_power_w <= 0:
            raise ConfigError("reference_transmit_power_w must be positive")
        if env.shadowing_std_db < 0 or env.shadowing_correlation_distance_m <= 0:
            raise ConfigError("shadowing std must be non-negative and correlation distance positive")
        if env.csi_error_std_db < 0 or env.initial_interference_w < 0:
            raise ConfigError("CSI error std and initial interference must be non-negative")
        if env.building_loss_db < 0:
            raise ConfigError("building_loss_db must be non-negative")
        for building in env.building_layout:
            if (
                building.x_max_m <= building.x_min_m
                or building.y_max_m <= building.y_min_m
                or building.height_m <= 0
            ):
                raise ConfigError(f"building {building.name!r} has invalid bounds or height")
        if len(env.arrival_probabilities) != env.uav_count:
            raise ConfigError("arrival_probabilities length must equal environment.uav_count")
        if len(env.profile_assignment) != env.uav_count:
            raise ConfigError("profile_assignment length must equal environment.uav_count")
        if any(prob < 0.0 or prob > 1.0 for prob in env.arrival_probabilities):
            raise ConfigError("arrival_probabilities must be in [0, 1]")
        if env.fixed_csi_aoi_slots < 0:
            raise ConfigError("fixed_csi_aoi_slots must be non-negative")
        if not 0.0 <= env.interference_ema_beta < 1.0:
            raise ConfigError("interference_ema_beta must be in [0, 1)")
        if env.minimum_task_slack_slots <= 0 or env.maximum_task_slack_slots < env.minimum_task_slack_slots:
            raise ConfigError("task slack clip bounds are invalid")
        if self.action.resource_group_count != env.resource_group_count:
            raise ConfigError("action.resource_group_count must equal environment.resource_group_count")
        if tuple(self.action.sampling_order) != (
            "route", "tx_select", "resource_group", "resource_width", "power_level", "cpu_queue", "cpu_frequency"
        ):
            raise ConfigError("action.sampling_order must follow the frozen seven-branch order")
        if tuple(self.action.power_levels) != (0.0, 0.25, 0.5, 1.0):
            raise ConfigError("action.power_levels must be (0.0, 0.25, 0.5, 1.0)")
        if tuple(self.action.cpu_frequency_levels) != (0.0, 0.25, 0.5, 1.0):
            raise ConfigError("action.cpu_frequency_levels must be (0.0, 0.25, 0.5, 1.0)")
        if self.evaluation.update_network:
            raise ConfigError("evaluation.update_network must remain false")


SCENARIO_DEFAULTS: dict[str, dict[str, Any]] = {
    "small": {
        "uav_count": 4,
        "arrival_probabilities": [0.04, 0.05, 0.06, 0.05],
        "profile_assignment": ["Balanced", "Resource-poor", "Compute-rich", "Energy-limited"],
    },
    "medium": {
        "uav_count": 6,
        "arrival_probabilities": [0.03, 0.04, 0.05, 0.06, 0.05, 0.07],
        "profile_assignment": [
            "Balanced", "Resource-poor", "Compute-rich", "Energy-limited", "Communication-rich", "Balanced"
        ],
    },
    "large": {
        "uav_count": 8,
        "arrival_probabilities": [0.02, 0.04, 0.06, 0.08, 0.02, 0.04, 0.06, 0.08],
        "profile_assignment": [
            "Resource-poor", "Balanced", "Compute-rich", "Energy-limited", "Communication-rich",
            "Resource-poor", "Balanced", "Compute-rich",
        ],
    },
}


DEFAULT_CONFIG_DATA: dict[str, Any] = {
    "config_version": CONFIG_VERSION,
    "mode": "environment_sanity",
    "method_id": "environment",
    "scenario_id": "small",
    "seed": DEFAULT_SEED,
    "environment": asdict(EnvironmentConfig()),
    "action": asdict(ActionConfig()),
    "reproducibility": asdict(ReproducibilityConfig()),
    "training": asdict(TrainingConfig()),
    "evaluation": asdict(EvaluationConfig()),
    "output": asdict(OutputConfig()),
}


def _default_method_for_mode(mode: str) -> str | None:
    return {
        "environment_sanity": "environment",
        "gate0": "environment",
        "random": "random",
        "heuristic": "heuristic",
        "baseline": "factorized_action_gat_qmix",
        "rl": "ca_gat_mappo",
        "evaluation": "ca_gat_mappo",
        "ablation": "ca_gat_mappo",
        "plot": "environment",
    }.get(mode)


def load_run_config(
    config_path: str | Path | None = None,
    *,
    cli_overrides: Mapping[str, Any] | None = None,
    interactive_overrides: Mapping[str, Any] | None = None,
) -> RunConfig:
    """Load defaults, then file, CLI, and interactive overrides in that order."""

    cli = _normalize_overrides(cli_overrides or {}, source="CLI")
    interactive = _normalize_overrides(interactive_overrides or {}, source="interactive")
    merged = copy.deepcopy(DEFAULT_CONFIG_DATA)
    source_path = str(Path(config_path)) if config_path is not None else None
    file_data: Mapping[str, Any] = {}
    if config_path is not None:
        path = Path(config_path)
        if not path.exists() or not path.is_file():
            raise ConfigError(f"config file does not exist or is not a file: {path}")
        loaded_data = _load_mapping_file(path)
        # A snapshot produced by this module is also a valid input config;
        # provenance is metadata, not a second source of typed config leaves.
        file_data = {key: value for key, value in loaded_data.items() if key != "_metadata"}
        _validate_mapping_keys(file_data, RunConfig, path="config")
        merged = _deep_merge(merged, file_data)

    # Scenario values are derived from the frozen scenario table unless the
    # caller explicitly overrides the corresponding environment leaf.
    scenario_id = str(merged.get("scenario_id", "small"))
    if scenario_id not in SUPPORTED_SCENARIOS:
        raise ConfigError(f"scenario_id must be one of {SUPPORTED_SCENARIOS}, got {scenario_id!r}")
    explicit_env = set(file_data.get("environment", {}).keys()) if isinstance(file_data.get("environment", {}), Mapping) else set()
    explicit_env.update(_dotted_top_level_keys(cli, prefix="environment"))
    explicit_env.update(_dotted_top_level_keys(interactive, prefix="environment"))
    for key, value in SCENARIO_DEFAULTS[scenario_id].items():
        if key not in explicit_env:
            merged["environment"][key] = value

    merged = _apply_dotted_overrides(merged, cli)
    merged = _apply_dotted_overrides(merged, interactive)
    if "mode" in merged and "method_id" not in cli and "method_id" not in interactive:
        if "method_id" not in file_data or file_data.get("method_id") in (None, ""):
            merged["method_id"] = _default_method_for_mode(str(merged["mode"])) or merged.get("method_id")

    # Re-apply scenario defaults after mode/file merging only when scenario_id
    # itself was supplied by a higher-priority source.
    final_scenario = str(merged.get("scenario_id", "small"))
    if final_scenario not in SUPPORTED_SCENARIOS:
        raise ConfigError(f"scenario_id must be one of {SUPPORTED_SCENARIOS}, got {final_scenario!r}")
    if final_scenario != scenario_id:
        for key, value in SCENARIO_DEFAULTS[final_scenario].items():
            if key not in explicit_env:
                merged["environment"][key] = value

    config = _construct_run_config(merged)
    _attach_provenance(
        config,
        source_config_path=source_path,
        cli_overrides=cli,
        interactive_overrides=interactive,
    )
    config.validate()
    return config


def write_config_snapshot(config: RunConfig, path: str | Path | None = None) -> Path:
    """Write a real canonical snapshot when a caller has a successful run."""

    target = Path(path) if path is not None else Path(config.artifact_paths()["config_snapshot"])
    target.parent.mkdir(parents=True, exist_ok=True)
    # JSON is valid YAML 1.2 and avoids introducing an unconfirmed dependency.
    target.write_text(json.dumps(config.snapshot_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target


def _construct_run_config(mapping: Mapping[str, Any]) -> RunConfig:
    values = _coerce_dataclass_mapping(mapping, RunConfig, path="config")
    return RunConfig(**values)


def _attach_provenance(
    config: RunConfig,
    *,
    source_config_path: str | None,
    cli_overrides: Mapping[str, Any],
    interactive_overrides: Mapping[str, Any],
) -> None:
    branch, commit, dirty = _git_metadata()
    object.__setattr__(config, "source_config_path", source_config_path)
    object.__setattr__(config, "cli_overrides", dict(cli_overrides))
    object.__setattr__(config, "interactive_overrides", dict(interactive_overrides))
    object.__setattr__(config, "git_branch", branch)
    object.__setattr__(config, "git_commit", commit)
    object.__setattr__(config, "git_dirty", dirty)


def _validate_mapping_keys(mapping: Mapping[str, Any], cls: type[Any], *, path: str) -> None:
    if not isinstance(mapping, Mapping):
        raise ConfigError(f"{path} must be a mapping")
    hints = get_type_hints(cls)
    allowed = {item.name for item in fields(cls) if item.init}
    for key, value in mapping.items():
        if key not in allowed:
            raise ConfigError(f"unknown field {path}.{key}")
        annotation = hints.get(key, item_type(cls, key))
        nested = _nested_dataclass_type(annotation)
        if nested is not None and isinstance(value, Mapping):
            _validate_mapping_keys(value, nested, path=f"{path}.{key}")
        elif nested is not None and value is not None:
            raise ConfigError(f"{path}.{key} must be a mapping")


def _coerce_dataclass_mapping(mapping: Mapping[str, Any], cls: type[_T], *, path: str) -> dict[str, Any]:
    _validate_mapping_keys(mapping, cls, path=path)
    hints = get_type_hints(cls)
    result: dict[str, Any] = {}
    for item in fields(cls):
        if not item.init:
            continue
        if item.name not in mapping:
            continue
        annotation = hints.get(item.name, item.type)
        result[item.name] = _coerce_value(mapping[item.name], annotation, f"{path}.{item.name}")
    return result


def _coerce_value(value: Any, annotation: Any, path: str) -> Any:
    nested = _nested_dataclass_type(annotation)
    if nested is not None:
        if not isinstance(value, Mapping):
            raise ConfigError(f"{path} must be a mapping")
        return nested(**_coerce_dataclass_mapping(value, nested, path=path))
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin in (tuple,):
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{path} must be a list or tuple")
        element_type = args[0] if args and args[-1] is not Ellipsis else (args[0] if args else Any)
        if len(args) > 1 and args[-1] is not Ellipsis:
            if len(value) != len(args):
                raise ConfigError(f"{path} must contain {len(args)} values")
            return tuple(_coerce_value(item, item_type, f"{path}[{index}]") for index, (item, item_type) in enumerate(zip(value, args)))
        return tuple(_coerce_value(item, element_type, f"{path}[{index}]") for index, item in enumerate(value))
    if origin in (list, Sequence):
        if not isinstance(value, list):
            raise ConfigError(f"{path} must be a list")
        element_type = args[0] if args else Any
        return [_coerce_value(item, element_type, f"{path}[{index}]") for index, item in enumerate(value)]
    if origin in (dict, Mapping):
        if not isinstance(value, Mapping):
            raise ConfigError(f"{path} must be a mapping")
        key_type, value_type = args if len(args) == 2 else (Any, Any)
        return {
            _coerce_value(key, key_type, f"{path}.<key>"): _coerce_value(item, value_type, f"{path}.{key}")
            for key, item in value.items()
        }
    if annotation is Any or annotation is None:
        return value
    if annotation is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{path} must be a boolean")
        return value
    if annotation is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{path} must be an integer")
        return value
    if annotation is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{path} must be numeric")
        return float(value)
    if annotation is str:
        if not isinstance(value, str):
            raise ConfigError(f"{path} must be a string")
        return value
    return value


def _nested_dataclass_type(annotation: Any) -> type[Any] | None:
    if isinstance(annotation, type) and is_dataclass(annotation):
        return annotation
    return None


def item_type(cls: type[Any], name: str) -> Any:
    return next(item.type for item in fields(cls) if item.name == name)


def _normalize_overrides(overrides: Mapping[str, Any], *, source: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in overrides.items():
        if value is None:
            continue
        if not isinstance(key, str) or not key:
            raise ConfigError(f"{source} override keys must be non-empty strings")
        result[key] = value
    return result


def _dotted_top_level_keys(mapping: Mapping[str, Any], *, prefix: str) -> set[str]:
    return {key.split(".", 1)[1] for key in mapping if key.startswith(prefix + ".")}


def _apply_dotted_overrides(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for dotted_key, value in overrides.items():
        parts = dotted_key.split(".")
        if len(parts) == 1:
            if parts[0] not in result:
                raise ConfigError(f"unknown override field {dotted_key}")
            result[parts[0]] = value
            continue
        cursor: Any = result
        for part in parts[:-1]:
            if not isinstance(cursor, Mapping) or part not in cursor:
                raise ConfigError(f"unknown override field {dotted_key}")
            cursor = cursor[part]
        if not isinstance(cursor, dict) or parts[-1] not in cursor:
            raise ConfigError(f"unknown override field {dotted_key}")
        cursor[parts[-1]] = value
    return result


def _deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _load_mapping_file(path: Path) -> Mapping[str, Any]:
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ConfigError(f"config file is empty: {path}")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except ImportError:
            value = _parse_minimal_yaml(text)
        else:
            value = yaml.safe_load(text)
    if not isinstance(value, Mapping):
        raise ConfigError(f"config root must be a mapping: {path}")
    return value


def _parse_minimal_yaml(text: str) -> dict[str, Any]:
    """Parse the small YAML subset needed for project config files.

    PyYAML remains preferred when available.  This fallback handles nested
    mappings, lists, quoted strings, numeric values, booleans, and nulls.
    """

    raw_lines = []
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        content = line.split(" #", 1)[0].rstrip()
        indent = len(content) - len(content.lstrip(" "))
        raw_lines.append((indent, content.strip()))
    if not raw_lines:
        return {}

    root: Any = {}
    stack: list[tuple[int, Any]] = [(-1, root)]
    for index, (indent, content) in enumerate(raw_lines):
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1] if stack else root
        if content.startswith("- "):
            if not isinstance(parent, list):
                raise ConfigError("minimal YAML parser expected a list parent")
            parent.append(_parse_scalar(content[2:].strip()))
            continue
        if ":" not in content:
            raise ConfigError(f"unsupported YAML line: {content}")
        key, raw_value = content.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        if not isinstance(parent, dict):
            raise ConfigError(f"minimal YAML parser expected a mapping parent for {key!r}")
        if raw_value:
            parent[key] = _parse_scalar(raw_value)
        else:
            next_content = raw_lines[index + 1][1] if index + 1 < len(raw_lines) else ""
            child: Any = [] if next_content.startswith("- ") else {}
            parent[key] = child
            stack.append((indent, child))
    return root


def _parse_scalar(value: str) -> Any:
    if value in {"null", "Null", "NULL", "~"}:
        return None
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.startswith(("[", "{")):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(value)
            except (ValueError, SyntaxError) as exc:
                raise ConfigError(f"invalid inline YAML value: {value}") from exc
    if (value.startswith("'") and value.endswith("'")) or (value.startswith('"') and value.endswith('"')):
        return value[1:-1]
    try:
        if re.fullmatch(r"[-+]?\d+", value):
            return int(value)
        if re.fullmatch(r"[-+]?(?:\d+\.\d*|\d*\.\d+|\d+e[-+]?\d+|\d+\.\d*e[-+]?\d+)", value, re.I):
            return float(value)
    except ValueError:
        pass
    return value


def _git_metadata() -> tuple[str, str, bool]:
    def git(*args: str) -> str:
        try:
            completed = subprocess.run(
                ["git", *args],
                check=True,
                capture_output=True,
                text=True,
                cwd=Path(__file__).resolve().parents[1],
            )
        except (OSError, subprocess.CalledProcessError):
            return "unknown"
        return completed.stdout.strip() or "unknown"

    branch = git("branch", "--show-current")
    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")
    return branch, commit, status not in {"", "unknown"}


def _jsonable(value: Any, *, exclude: set[str] | None = None) -> Any:
    exclude = exclude or set()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item, exclude=exclude) for key, item in value.items() if str(key) not in exclude}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item, exclude=exclude) for item in value]
    if is_dataclass(value):
        return _jsonable(asdict(value), exclude=exclude)
    return value


__all__ = [
    "ActionConfig",
    "ConfigError",
    "DEFAULT_SEED",
    "EnvironmentConfig",
    "EvaluationConfig",
    "MAPPOConfig",
    "OutputConfig",
    "QMIXConfig",
    "RunConfig",
    "SUPPORTED_METHODS",
    "SUPPORTED_MODES",
    "SUPPORTED_SCENARIOS",
    "TrainingConfig",
    "load_run_config",
    "write_config_snapshot",
]
