"""No-training identity and configuration gates for matched short campaigns.

This module deliberately constructs only models and optimizers.  It never
creates an environment, samples an action, performs a PPO update, or writes a
checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import platform
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from .config import (
    CHECKPOINT_KIND_FINAL_COMPLETED,
    RunConfig,
    compute_mappo_periodic_checkpoint_steps,
    mappo_checkpoint_kind_at,
    validate_mappo_checkpoint_contract,
)
from .models.ca_gat_mappo import CAGATMAPPOActor, MAPPOCentralizedCritic
from .models.ca_gat_mappo_update import (
    CAGATMAPPOOptimizerBundle,
    build_ca_gat_mappo_optimizers,
    compute_route_choice_stability_loss,
)


MATCHED_GROUP_COEFFICIENTS = {"baseline": 0.0, "a1": 0.02, "a2": 0.05}
ALLOWED_CONFIG_DIFF_PATHS = (
    "training.mappo.route_choice_stability_coef",
)
SMOKE_BUDGET = 1500
SMOKE_CHECKPOINT_INTERVAL = 500
DIAGNOSTIC_BUDGET = 32000
DIAGNOSTIC_CHECKPOINT_INTERVAL = 4000
EXTENSION_BUDGET = 42500
EXTENSION_CHECKPOINT_INTERVAL = 4000
PERIODIC_PARTIAL_ROLLOUT_BEHAVIOR = "serialized_in_periodic_resume"
FINAL_PARTIAL_ROLLOUT_BEHAVIOR = (
    "record_unused_tail_then_clear_without_partial_ppo_update"
)
PROTECTED_UNTRACKED_PATHS = frozenset(
    {"knowledge/papers/1.pdf", "knowledge/papers/7121.pdf"}
)


class MatchedPreflightError(RuntimeError):
    """Raised when a no-training matched-campaign gate fails."""


@dataclass(frozen=True)
class MatchedBudgetSpec:
    """One frozen matched-campaign budget without changing trainer semantics."""

    name: str
    total_steps: int
    checkpoint_interval: int
    purpose: str
    fresh_campaign_required: bool


MATCHED_SHORT_BUDGET_SPECS = {
    "smoke": MatchedBudgetSpec(
        name="smoke",
        total_steps=SMOKE_BUDGET,
        checkpoint_interval=SMOKE_CHECKPOINT_INTERVAL,
        purpose="runtime_telemetry_and_safety_only",
        fresh_campaign_required=True,
    ),
    "diagnostic": MatchedBudgetSpec(
        name="diagnostic",
        total_steps=DIAGNOSTIC_BUDGET,
        checkpoint_interval=DIAGNOSTIC_CHECKPOINT_INTERVAL,
        purpose="matched_diagnostic_screening",
        fresh_campaign_required=True,
    ),
    "extension": MatchedBudgetSpec(
        name="extension",
        total_steps=EXTENSION_BUDGET,
        checkpoint_interval=EXTENSION_CHECKPOINT_INTERVAL,
        purpose="human_approved_fresh_matched_extension",
        fresh_campaign_required=True,
    ),
}


def joint_lossless_boundary(episode_horizon: int, rollout_capacity: int) -> int:
    """Return the least positive boundary aligned to episodes and PPO rollouts."""

    for name, value in (
        ("episode_horizon", episode_horizon),
        ("rollout_capacity", rollout_capacity),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise MatchedPreflightError(f"{name} must be a positive integer")
    return math.lcm(episode_horizon, rollout_capacity)


def next_joint_lossless_boundary(
    step: int, episode_horizon: int, rollout_capacity: int
) -> int:
    """Return the first joint boundary strictly after step."""

    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise MatchedPreflightError("step must be a non-negative integer")
    boundary = joint_lossless_boundary(episode_horizon, rollout_capacity)
    return (step // boundary + 1) * boundary


@dataclass(frozen=True)
class CheckpointPlan:
    """Pure checkpoint geometry report without creating checkpoint files."""

    total_steps: int
    episode_horizon: int
    episode_count: int
    rollout_capacity: int
    full_rollout_count: int
    optimized_transition_count: int
    rollout_remainder: int
    unused_final_tail_transitions: int
    optimized_cutoff_step: int
    checkpoint_interval: int
    periodic_milestones: tuple[int, ...]
    periodic_active_rollout_lengths: tuple[tuple[int, int], ...]
    final_step: int
    final_kind: str
    final_requires_episode_boundary: bool
    final_requires_rollout_boundary: bool
    final_completed_supported: bool
    final_tail_behavior: str
    periodic_partial_rollout_behavior: str
    final_partial_rollout_ppo_update: bool
    final_tail_marks_tasks_truncated: bool
    final_active_rollout_length: int

    @property
    def budget(self) -> int:
        """Backward-compatible alias for the declared collection budget."""

        return self.total_steps

    @property
    def periodic_steps(self) -> tuple[int, ...]:
        """Backward-compatible alias for periodic checkpoint milestones."""

        return self.periodic_milestones

    @property
    def episode_boundary_aligned(self) -> bool:
        return self.total_steps % self.episode_horizon == 0

    @property
    def rollout_boundary_aligned(self) -> bool:
        return self.rollout_remainder == 0

    @property
    def final_completed_runtime_compatible(self) -> bool:
        """Backward-compatible alias for the honest final support gate."""

        return self.final_completed_supported

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_steps": self.total_steps,
            "budget": self.total_steps,
            "collected_environment_steps": self.total_steps,
            "episode_horizon": self.episode_horizon,
            "episode_count": self.episode_count,
            "rollout_capacity": self.rollout_capacity,
            "full_rollout_count": self.full_rollout_count,
            "optimized_transition_count": self.optimized_transition_count,
            "optimized_environment_steps": self.optimized_transition_count,
            "rollout_remainder": self.rollout_remainder,
            "unused_final_tail_transitions": self.unused_final_tail_transitions,
            "optimized_cutoff_step": self.optimized_cutoff_step,
            "checkpoint_interval": self.checkpoint_interval,
            "periodic_milestones": list(self.periodic_milestones),
            "periodic_steps": list(self.periodic_milestones),
            "periodic_active_rollout_lengths": [
                {"step": step, "active_rollout_length": active_length}
                for step, active_length in self.periodic_active_rollout_lengths
            ],
            "final_step": self.final_step,
            "final_kind": self.final_kind,
            "episode_boundary_aligned": self.episode_boundary_aligned,
            "rollout_boundary_aligned": self.rollout_boundary_aligned,
            "final_requires_episode_boundary": self.final_requires_episode_boundary,
            "final_requires_rollout_boundary": self.final_requires_rollout_boundary,
            "final_completed_supported": self.final_completed_supported,
            "final_completed_runtime_compatible": self.final_completed_supported,
            "final_tail_behavior": self.final_tail_behavior,
            "periodic_partial_rollout_behavior": (
                self.periodic_partial_rollout_behavior
            ),
            "final_partial_rollout_ppo_update": (
                self.final_partial_rollout_ppo_update
            ),
            "final_tail_marks_tasks_truncated": (
                self.final_tail_marks_tasks_truncated
            ),
            "final_active_rollout_length": self.final_active_rollout_length,
        }


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _tensor_bytes(tensor: Tensor) -> bytes:
    contiguous = tensor.detach().to(device="cpu").contiguous()
    return contiguous.view(torch.uint8).numpy().tobytes(order="C")


def module_digest(module: nn.Module) -> str:
    """Hash qualified state names, shapes, dtypes, and contiguous raw bytes."""

    digest = hashlib.sha256()
    state = module.state_dict()
    for name in sorted(state):
        tensor = state[name]
        if not isinstance(tensor, Tensor):
            raise MatchedPreflightError(f"module state {name!r} is not a tensor")
        metadata = {
            "name": name,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
        }
        digest.update(_canonical_json(metadata))
        digest.update(b"\0")
        digest.update(_tensor_bytes(tensor))
        digest.update(b"\0")
    return digest.hexdigest()


def _optimizer_scalar(value: Any) -> Any:
    if isinstance(value, Tensor):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": _sha256_bytes(_tensor_bytes(value)),
        }
    if isinstance(value, tuple):
        return [_optimizer_scalar(item) for item in value]
    if isinstance(value, list):
        return [_optimizer_scalar(item) for item in value]
    if isinstance(value, Mapping):
        return {
            str(key): _optimizer_scalar(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise MatchedPreflightError(
        f"unsupported optimizer state value {type(value).__name__}"
    )


def _normalized_optimizer(
    optimizer: torch.optim.Optimizer,
    named_parameters: Mapping[int, str],
) -> dict[str, Any]:
    groups = []
    for group in optimizer.param_groups:
        parameter_names = []
        for parameter in group["params"]:
            name = named_parameters.get(id(parameter))
            if name is None:
                raise MatchedPreflightError(
                    "optimizer contains a parameter without a qualified name"
                )
            parameter_names.append(name)
        hyperparameters = {
            str(key): _optimizer_scalar(value)
            for key, value in sorted(group.items())
            if key != "params"
        }
        groups.append(
            {
                "parameters": parameter_names,
                "hyperparameters": hyperparameters,
            }
        )
    state: dict[str, Any] = {}
    for parameter, values in optimizer.state.items():
        name = named_parameters.get(id(parameter))
        if name is None:
            raise MatchedPreflightError(
                "optimizer state contains an unnamed parameter"
            )
        state[name] = _optimizer_scalar(values)
    return {
        "optimizer_class": type(optimizer).__qualname__,
        "param_groups": groups,
        "state": {name: state[name] for name in sorted(state)},
        "state_semantics": "canonical-empty-state" if not state else "initialized",
    }


def normalized_optimizer_digest(
    bundle: CAGATMAPPOOptimizerBundle,
    actor: CAGATMAPPOActor,
    critic: MAPPOCentralizedCritic,
) -> tuple[str, dict[str, Any]]:
    """Return a digest independent of Python parameter object identities."""

    actor_names = {id(value): f"actor.{name}" for name, value in actor.named_parameters()}
    critic_names = {
        id(value): f"critic.{name}" for name, value in critic.named_parameters()
    }
    normalized = {
        "actor": _normalized_optimizer(bundle.actor_optimizer, actor_names),
        "critic": _normalized_optimizer(bundle.critic_optimizer, critic_names),
    }
    return _sha256_bytes(_canonical_json(normalized)), normalized


def seed_global_rngs(seed: int) -> None:
    """Set all user-requested global RNG streams to one explicit seed."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise MatchedPreflightError("seed must be a non-negative integer")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_digests() -> dict[str, Any]:
    """Capture all requested RNG states without exposing the raw states."""

    cuda_states = (
        tuple(torch.cuda.get_rng_state_all()) if torch.cuda.is_available() else ()
    )
    return {
        "python": _sha256_bytes(pickle.dumps(random.getstate(), protocol=5)),
        "numpy": _sha256_bytes(pickle.dumps(np.random.get_state(), protocol=5)),
        "torch_cpu": _sha256_bytes(_tensor_bytes(torch.get_rng_state())),
        "torch_cuda": [
            _sha256_bytes(_tensor_bytes(state)) for state in cuda_states
        ],
        "torch_cuda_device_count": len(cuda_states),
    }


def capture_initial_identity(config: RunConfig) -> dict[str, Any]:
    """Construct one independent model/optimizer context and capture step-0 identity."""

    if not isinstance(config, RunConfig):
        raise TypeError("config must be a RunConfig")
    config.validate()
    seed_global_rngs(config.seed)
    device = torch.device(config.training.mappo.training_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise MatchedPreflightError("configured CUDA preflight requires CUDA availability")
    actor = CAGATMAPPOActor(config).to(device=device, dtype=torch.float32)
    critic = MAPPOCentralizedCritic(config).to(device=device, dtype=torch.float32)
    optimizers = build_ca_gat_mappo_optimizers(actor, critic, config)
    optimizer_digest, optimizer_normalized = normalized_optimizer_digest(
        optimizers, actor, critic
    )
    return {
        "actor_digest": module_digest(actor),
        "critic_digest": module_digest(critic),
        "optimizer_normalized_digest": optimizer_digest,
        "optimizer_state_semantics": {
            name: optimizer_normalized[name]["state_semantics"]
            for name in ("actor", "critic")
        },
        "rng_digests": rng_digests(),
    }


def validate_initial_identities(
    identities: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Fail fast unless all three independent step-0 identities match."""

    if set(identities) != set(MATCHED_GROUP_COEFFICIENTS):
        raise MatchedPreflightError(
            "initial identities must be provided for baseline, a1, and a2"
        )
    scalar_fields = (
        "actor_digest",
        "critic_digest",
        "optimizer_normalized_digest",
    )
    matched_fields = []
    for field in scalar_fields:
        try:
            values = [
                _canonical_json(identities[group][field])
                for group in MATCHED_GROUP_COEFFICIENTS
            ]
        except KeyError as exc:
            raise MatchedPreflightError(
                f"initial identity is missing {field}"
            ) from exc
        if len(set(values)) != 1:
            raise MatchedPreflightError(
                f"STOP_INITIAL_IDENTITY: {field} differs across matched groups"
            )
        matched_fields.append(field)
    rng_fields = (
        "python",
        "numpy",
        "torch_cpu",
        "torch_cuda",
        "torch_cuda_device_count",
    )
    for rng_field in rng_fields:
        qualified = f"rng_digests.{rng_field}"
        try:
            values = [
                _canonical_json(identities[group]["rng_digests"][rng_field])
                for group in MATCHED_GROUP_COEFFICIENTS
            ]
        except KeyError as exc:
            raise MatchedPreflightError(
                f"initial identity is missing {qualified}"
            ) from exc
        if len(set(values)) != 1:
            raise MatchedPreflightError(
                f"STOP_INITIAL_IDENTITY: {qualified} differs across matched groups"
            )
        matched_fields.append(qualified)
    return {"status": "PASS", "matched_fields": matched_fields}


def _flatten_config(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, Mapping):
        flattened: dict[str, Any] = {}
        for key in sorted(value):
            path = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten_config(value[key], path))
        return flattened
    return {prefix: value}


def canonical_config_differences(
    reference: RunConfig,
    candidate: RunConfig,
) -> tuple[dict[str, Any], ...]:
    """Return exact leaf-level canonical config differences."""

    left = _flatten_config(reference.resolved_dict())
    right = _flatten_config(candidate.resolved_dict())
    rows = []
    for path in sorted(set(left) | set(right)):
        if left.get(path) != right.get(path):
            rows.append(
                {"path": path, "reference": left.get(path), "candidate": right.get(path)}
            )
    return tuple(rows)


def validate_matched_configs(configs: Mapping[str, RunConfig]) -> dict[str, Any]:
    """Enforce the coefficient-only canonical config whitelist."""

    if set(configs) != set(MATCHED_GROUP_COEFFICIENTS):
        raise MatchedPreflightError("matched configs must be baseline, a1, and a2")
    for group, expected_coefficient in MATCHED_GROUP_COEFFICIENTS.items():
        config = configs[group]
        config.validate()
        mappo = config.training.mappo
        if not mappo.route_choice_stability_enabled:
            raise MatchedPreflightError(f"{group} stability must be enabled")
        if mappo.route_choice_entropy_floor_nats != 0.20:
            raise MatchedPreflightError(f"{group} entropy floor must equal 0.20")
        if mappo.route_choice_stability_coef != expected_coefficient:
            raise MatchedPreflightError(
                f"{group} stability coefficient differs from {expected_coefficient}"
            )
    reference = configs["baseline"]
    comparisons = {}
    for group in ("a1", "a2"):
        differences = canonical_config_differences(reference, configs[group])
        paths = tuple(item["path"] for item in differences)
        if paths != ALLOWED_CONFIG_DIFF_PATHS:
            raise MatchedPreflightError(
                f"STOP_CONFIG_DIFF: {group} differences are {paths!r}"
            )
        comparisons[group] = list(differences)
    return {
        "status": "PASS",
        "allowed_paths": list(ALLOWED_CONFIG_DIFF_PATHS),
        "comparisons": comparisons,
    }


def synthetic_stability_gate() -> dict[str, Any]:
    """Exercise the existing Candidate-A helper on fixed violating FP32 logits."""

    logits = torch.tensor([[[[0.0, 10.0, 0.0, -10.0, -10.0]]]], dtype=torch.float32)
    action_masks = torch.tensor(
        [[[[False, True, False, True, True]]]], dtype=torch.bool
    )
    local_domain = torch.tensor(
        [[[[False, True, False, False, False]]]], dtype=torch.bool
    )
    remote_domain = torch.tensor(
        [[[[False, False, False, True, True]]]], dtype=torch.bool
    )
    common = {
        "route_logits": logits,
        "route_action_masks": action_masks,
        "route_active": torch.tensor([[[True]]], dtype=torch.bool),
        "sequence_valid_mask": torch.tensor([[True]], dtype=torch.bool),
        "local_domain_mask": local_domain,
        "remote_domain_mask": remote_domain,
        "selected_route_indices": torch.tensor([[[1]]], dtype=torch.long),
        "entropy_floor_nats": 0.20,
    }
    before = rng_digests()
    outputs = {
        group: compute_route_choice_stability_loss(
            **common, coefficient=coefficient
        )
        for group, coefficient in MATCHED_GROUP_COEFFICIENTS.items()
    }
    after = rng_digests()
    unscaled = {
        group: float(output.unscaled_loss.detach().cpu().item())
        for group, output in outputs.items()
    }
    scaled = {
        group: float(output.scaled_loss.detach().cpu().item())
        for group, output in outputs.items()
    }
    if not all(value > 0.0 for value in unscaled.values()):
        raise MatchedPreflightError("synthetic unscaled stability loss must be positive")
    reference = unscaled["baseline"]
    if not np.isclose(scaled["baseline"], 0.0, rtol=0.0, atol=1.0e-8):
        raise MatchedPreflightError("baseline synthetic scaled loss must be zero")
    for group in ("a1", "a2"):
        expected = MATCHED_GROUP_COEFFICIENTS[group] * reference
        if not np.isclose(scaled[group], expected, rtol=1.0e-6, atol=1.0e-8):
            raise MatchedPreflightError(f"{group} synthetic loss is not proportional")
    if before != after:
        raise MatchedPreflightError("Candidate-A helper consumed an RNG stream")
    return {
        "status": "PASS",
        "unscaled_loss": unscaled,
        "scaled_loss": scaled,
        "rng_unchanged": True,
        "dtype": "torch.float32",
    }


def checkpoint_plan(config: RunConfig) -> CheckpointPlan:
    """Report both declared schedule and actual FINAL_COMPLETED feasibility."""

    validate_mappo_checkpoint_contract(config)
    mappo = config.training.mappo
    budget = mappo.max_environment_transitions
    final_kind = mappo_checkpoint_kind_at(config, budget)
    if final_kind != CHECKPOINT_KIND_FINAL_COMPLETED:
        raise MatchedPreflightError("configured final budget is not classified as final")
    episode_horizon = config.environment.episode_horizon
    rollout_capacity = mappo.rollout_length_slots
    episode_count, episode_remainder = divmod(budget, episode_horizon)
    full_rollout_count, rollout_remainder = divmod(budget, rollout_capacity)
    optimized_transition_count = full_rollout_count * rollout_capacity
    periodic_milestones = compute_mappo_periodic_checkpoint_steps(config)
    final_supported = episode_remainder == 0
    if not final_supported:
        final_tail_behavior = "unsupported_mid_episode_final"
    elif rollout_remainder:
        final_tail_behavior = FINAL_PARTIAL_ROLLOUT_BEHAVIOR
    else:
        final_tail_behavior = "no_unused_final_tail"
    return CheckpointPlan(
        total_steps=budget,
        episode_horizon=episode_horizon,
        episode_count=episode_count,
        rollout_capacity=rollout_capacity,
        full_rollout_count=full_rollout_count,
        optimized_transition_count=optimized_transition_count,
        rollout_remainder=rollout_remainder,
        unused_final_tail_transitions=rollout_remainder,
        optimized_cutoff_step=optimized_transition_count,
        checkpoint_interval=mappo.checkpoint_interval_steps,
        periodic_milestones=periodic_milestones,
        periodic_active_rollout_lengths=tuple(
            (step, step % rollout_capacity) for step in periodic_milestones
        ),
        final_step=budget,
        final_kind=final_kind,
        final_requires_episode_boundary=True,
        final_requires_rollout_boundary=False,
        final_completed_supported=final_supported,
        final_tail_behavior=final_tail_behavior,
        periodic_partial_rollout_behavior=PERIODIC_PARTIAL_ROLLOUT_BEHAVIOR,
        final_partial_rollout_ppo_update=False,
        final_tail_marks_tasks_truncated=False,
        final_active_rollout_length=0,
    )


def validate_exact_final_checkpoint_plan(config: RunConfig) -> CheckpointPlan:
    """Fail when FINAL_COMPLETED is requested away from an episode boundary."""

    plan = checkpoint_plan(config)
    if not plan.final_completed_runtime_compatible:
        raise MatchedPreflightError(
            "STOP_CHECKPOINT_PLAN: exact FINAL_COMPLETED at "
            f"{plan.final_step} is not expressible with episode_horizon="
            f"{config.environment.episode_horizon}"
        )
    return plan


def _git_text(arguments: Sequence[str]) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def _git_status() -> dict[str, Any]:
    lines = tuple(
        line for line in _git_text(["status", "--porcelain=v1", "--untracked-files=all"]).splitlines()
        if line
    )
    tracked = tuple(line for line in lines if not line.startswith("?? "))
    untracked = tuple(line[3:].replace("\\", "/") for line in lines if line.startswith("?? "))
    unexpected_untracked = tuple(sorted(set(untracked) - PROTECTED_UNTRACKED_PATHS))
    protected_present = tuple(sorted(set(untracked) & PROTECTED_UNTRACKED_PATHS))
    return {
        "porcelain": list(lines),
        "tracked_dirty": bool(tracked),
        "tracked_entries": list(tracked),
        "protected_untracked_paths_present": list(protected_present),
        "unexpected_untracked_paths": list(unexpected_untracked),
        "acceptable": not tracked and not unexpected_untracked,
    }


def _exclusive_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
    ) + "\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def run_matched_preflight(
    configs: Mapping[str, RunConfig],
    *,
    seed: int,
    expected_head: str,
    expected_branch: str,
    output_root: str | Path,
) -> dict[str, Any]:
    """Run the complete Stage-0 gate and write one immutable manifest."""

    root = Path(output_root)
    manifest_path = root / "matched_preflight_manifest.json"
    if manifest_path.exists():
        raise MatchedPreflightError(f"preflight manifest already exists: {manifest_path}")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "FAIL",
        "seed": seed,
        "expected_head": expected_head,
        "expected_branch": expected_branch,
    }
    error: Exception | None = None
    try:
        if any(config.seed != seed for config in configs.values()):
            raise MatchedPreflightError("all matched configs must use the requested seed")
        head = _git_text(["rev-parse", "HEAD"])
        branch = _git_text(["branch", "--show-current"])
        status = _git_status()
        payload["git"] = {
            "commit": head,
            "branch": branch,
            "dirty_status": status,
        }
        if head != expected_head or branch != expected_branch or not status["acceptable"]:
            raise MatchedPreflightError("Git provenance gate failed")
        config_gate = validate_matched_configs(configs)
        identities = {group: capture_initial_identity(config) for group, config in configs.items()}
        identity_gate = validate_initial_identities(identities)
        payload["config_whitelist"] = config_gate
        payload["initial_identity_gate"] = identity_gate
        payload["initial_identity"] = identities
        payload["synthetic_stability_gate"] = synthetic_stability_gate()
        plans = {
            group: checkpoint_plan(config).to_dict()
            for group, config in configs.items()
        }
        payload["checkpoint_plan"] = plans
        for group, config in configs.items():
            try:
                validate_exact_final_checkpoint_plan(config)
            except MatchedPreflightError as exc:
                raise MatchedPreflightError(f"{group}: {exc}") from exc
        destinations = {}
        for group, config in configs.items():
            paths = {
                **config.artifact_paths(),
                "live_training_diagnostics": str(
                    Path(config.output.logs_dir)
                    / config.run_id
                    / "live_training_diagnostics.jsonl"
                ),
            }
            destinations[group] = {
                "run_id": config.run_id,
                "paths_sha256": _sha256_bytes(_canonical_json(paths)),
                "paths": paths,
            }
        all_destination_paths = [
            path
            for item in destinations.values()
            for path in item["paths"].values()
        ]
        repeated_destinations = sorted(
            {
                path
                for path in all_destination_paths
                if all_destination_paths.count(path) > 1
            }
        )
        if repeated_destinations:
            raise MatchedPreflightError(
                "matched groups share output destinations: "
                + ", ".join(repeated_destinations)
            )
        collisions = sorted(
            path for path in all_destination_paths if Path(path).exists()
        )
        if collisions:
            raise MatchedPreflightError(
                "output destination collision: " + ", ".join(collisions)
            )
        payload.update(
            {
                "runtime": {
                    "python": sys.version,
                    "platform": platform.platform(),
                    "torch": torch.__version__,
                    "cuda_version": torch.version.cuda,
                    "cuda_available": torch.cuda.is_available(),
                    "gpu_names": [
                        torch.cuda.get_device_name(index)
                        for index in range(torch.cuda.device_count())
                    ],
                },
                "resolved_config_hashes": {
                    group: config.config_hash for group, config in configs.items()
                },
                "output_destination_identity": destinations,
                "status": "PASS",
            }
        )
    except Exception as exc:
        error = exc
        payload["error"] = f"{type(exc).__name__}: {exc}"
    _exclusive_json_write(manifest_path, payload)
    if error is not None:
        raise MatchedPreflightError(
            f"matched preflight failed; see {manifest_path}: {error}"
        ) from error
    return payload


__all__ = [
    "ALLOWED_CONFIG_DIFF_PATHS",
    "CheckpointPlan",
    "MATCHED_GROUP_COEFFICIENTS",
    "MatchedPreflightError",
    "canonical_config_differences",
    "capture_initial_identity",
    "checkpoint_plan",
    "module_digest",
    "normalized_optimizer_digest",
    "rng_digests",
    "run_matched_preflight",
    "seed_global_rngs",
    "synthetic_stability_gate",
    "validate_exact_final_checkpoint_plan",
    "validate_initial_identities",
    "validate_matched_configs",
]
