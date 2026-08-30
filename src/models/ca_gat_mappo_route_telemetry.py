"""Behavior-neutral route diagnostics for CA-GAT-MAPPO PPO updates.

The functions in this module only read already-computed policy statistics,
fixed rollout targets, action-mask snapshots, and existing gradients.  They do
not sample actions, advance random-number generators, build loss terms, run
backward, clip gradients, or step optimizers.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
import math
from statistics import median
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch
from torch import Tensor, nn

from ..config import ActorRatioMode, RunConfig, WorkloadTimingMode
from ..env.actions import ActionProposal
from ..env.reward import TaskWorkloadSnapshot
from ..env.tasks import Task, TaskOutcome
from .ca_gat_mappo import ACTION_BRANCH_ORDER
from .ca_gat_mappo_actions import (
    SequentialActionDistributionOutput,
    SequentialActionMaskBatch,
)


ROUTE_TELEMETRY_SCHEMA_VERSION = 3
TRAJECTORY_CREDIT_SCHEMA_VERSION = 1
TRAJECTORY_CREDIT_EVENT_TYPES = ("route", "terminal", "credit")
TRAJECTORY_LEDGER_TOLERANCE = 1.0e-9
ROUTE_GRADIENT_STATUSES = ("none", "zero", "finite_nonzero", "nonfinite")
ROUTE_ACTION_CATEGORIES = ("local", "remote", "defer")
ROUTE_DISTRIBUTION_GROUPS = (
    "route_active",
    "local_route",
    "remote_route",
    "defer_route",
    "legal_remote_local",
    "legal_remote_remote",
)
ROUTE_PPO_RECORD_FIELDS = (
    "route_ppo_epsilon_clip",
    "route_ppo_overall_count",
    "route_ppo_overall_old_log_prob_mean",
    "route_ppo_overall_new_log_prob_mean",
    "route_ppo_overall_log_ratio_mean",
    "route_ppo_overall_ratio_mean",
    "route_ppo_overall_approx_kl_mean",
    "route_ppo_overall_clip_indicator_fraction",
    "route_ppo_overall_clip_fraction",
    "local_route_ppo_count",
    "local_route_ppo_old_log_prob_mean",
    "local_route_ppo_new_log_prob_mean",
    "local_route_ppo_log_ratio_mean",
    "local_route_ppo_ratio_mean",
    "local_route_ppo_approx_kl_mean",
    "local_route_ppo_clip_indicator_fraction",
    "local_route_ppo_clip_fraction",
    "remote_route_ppo_count",
    "remote_route_ppo_old_log_prob_mean",
    "remote_route_ppo_new_log_prob_mean",
    "remote_route_ppo_log_ratio_mean",
    "remote_route_ppo_ratio_mean",
    "remote_route_ppo_approx_kl_mean",
    "remote_route_ppo_clip_indicator_fraction",
    "remote_route_ppo_clip_fraction",
    "defer_route_ppo_count",
    "defer_route_ppo_old_log_prob_mean",
    "defer_route_ppo_new_log_prob_mean",
    "defer_route_ppo_log_ratio_mean",
    "defer_route_ppo_ratio_mean",
    "defer_route_ppo_approx_kl_mean",
    "defer_route_ppo_clip_indicator_fraction",
    "defer_route_ppo_clip_fraction",
)
BRANCH_PPO_STATISTIC_NAMES = (
    "active_count",
    "ratio_mean",
    "ratio_median",
    "ratio_std",
    "approx_kl_mean",
    "clip_fraction",
    "surrogate_contribution_mean",
    "advantage_mean",
)
BRANCH_PPO_RECORD_FIELDS = tuple(
    f"{branch}_branch_ppo_{statistic}"
    for branch in ACTION_BRANCH_ORDER
    for statistic in BRANCH_PPO_STATISTIC_NAMES
)


class RouteTelemetryError(ValueError):
    """Raised when route telemetry inputs or derived invariants are invalid."""


@dataclass(frozen=True)
class GradientMeasurement:
    """One read-only gradient measurement with explicit missing semantics."""

    status: str
    norm: float | None
    mean: float | None = None
    signed_sum: float | None = None

    def __post_init__(self) -> None:
        if self.status not in ROUTE_GRADIENT_STATUSES:
            raise RouteTelemetryError("unknown route-head gradient status")
        if self.status == "none":
            if self.norm is not None:
                raise RouteTelemetryError("missing gradient must use an NA norm")
            return
        if self.status == "nonfinite":
            if self.norm is not None:
                raise RouteTelemetryError("nonfinite gradient must use an NA norm")
            return
        if self.norm is None or not math.isfinite(self.norm) or self.norm < 0.0:
            raise RouteTelemetryError("present gradient norm must be finite and non-negative")
        if (self.status == "zero") != (self.norm == 0.0):
            raise RouteTelemetryError("gradient status and norm disagree")


@dataclass(frozen=True)
class RouteHeadGradientTelemetry:
    """Weight, bias, and combined route-head gradient diagnostics."""

    weight: GradientMeasurement
    bias: GradientMeasurement
    total: GradientMeasurement
    rows: tuple[Any, ...] = ()


@dataclass(frozen=True)
class BranchPPODynamicsGroup:
    """Detached statistics for one active action branch."""

    active_count: int = 0
    ratio_mean: float | None = None
    ratio_median: float | None = None
    ratio_std: float | None = None
    approx_kl_mean: float | None = None
    clip_fraction: float | None = None
    surrogate_contribution_mean: float | None = None
    advantage_mean: float | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.active_count, bool)
            or not isinstance(self.active_count, int)
            or self.active_count < 0
        ):
            raise RouteTelemetryError("branch PPO active_count is invalid")
        values = tuple(
            getattr(self, name)
            for name in BRANCH_PPO_STATISTIC_NAMES
            if name != "active_count"
        )
        if self.active_count == 0:
            if any(value is not None for value in values):
                raise RouteTelemetryError(
                    "branch PPO statistics must be NA without active samples"
                )
            return
        if any(value is None or not math.isfinite(value) for value in values):
            raise RouteTelemetryError(
                "branch PPO statistics must be finite with active samples"
            )
        if self.clip_fraction is None or not 0.0 <= self.clip_fraction <= 1.0:
            raise RouteTelemetryError("branch PPO clip_fraction is outside [0, 1]")

    def record(self, branch: str) -> dict[str, object]:
        return {
            f"{branch}_branch_ppo_{name}": getattr(self, name)
            for name in BRANCH_PPO_STATISTIC_NAMES
        }


@dataclass(frozen=True)
class BranchPPODynamics:
    """All seven branch statistics from one existing PPO evaluation."""

    epsilon_clip: float
    groups: Mapping[str, BranchPPODynamicsGroup] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not math.isfinite(self.epsilon_clip) or not 0.0 < self.epsilon_clip < 1.0:
            raise RouteTelemetryError("branch PPO epsilon_clip is invalid")
        if tuple(self.groups) != ACTION_BRANCH_ORDER:
            raise RouteTelemetryError("branch PPO groups must use frozen branch order")

    def record(self) -> dict[str, object]:
        result: dict[str, object] = {}
        for branch in ACTION_BRANCH_ORDER:
            result.update(self.groups[branch].record(branch))
        return result


@dataclass(frozen=True)
class RouteTelemetry:
    """Detached per-PPO-epoch route telemetry with explicit valid counts."""

    schema_version: int
    route_branch_active_count: int
    legal_remote_route_count: int
    route_selected_local_count: int
    route_selected_remote_count: int
    route_selected_defer_count: int
    remote_selection_rate_given_legal_remote: float | None
    route_entropy_valid_sample_count: int
    route_entropy_mean: float | None
    legal_remote_probability_valid_sample_count: int
    mean_selected_route_probability: float | None
    mean_local_probability: float | None
    mean_best_remote_probability: float | None
    mean_local_minus_best_remote_probability: float | None
    mean_local_minus_best_remote_logit_margin: float | None
    route_active_advantage_valid_sample_count: int
    route_active_advantage_mean: float | None
    route_active_advantage_std: float | None
    route_active_return_mean: float | None
    route_active_return_std: float | None
    local_route_valid_sample_count: int
    local_route_advantage_mean: float | None
    local_route_return_mean: float | None
    remote_route_valid_sample_count: int
    remote_route_advantage_mean: float | None
    remote_route_return_mean: float | None
    defer_route_valid_sample_count: int
    defer_route_advantage_mean: float | None
    defer_route_return_mean: float | None
    legal_remote_local_valid_sample_count: int
    legal_remote_local_advantage_mean: float | None
    legal_remote_local_return_mean: float | None
    legal_remote_remote_valid_sample_count: int
    legal_remote_remote_advantage_mean: float | None
    legal_remote_remote_return_mean: float | None
    route_head_weight_grad_status: str
    route_head_weight_grad_norm: float | None
    route_head_bias_grad_status: str
    route_head_bias_grad_norm: float | None
    route_head_total_grad_status: str
    route_head_total_grad_norm: float | None
    route_probability_valid_sample_count: int = 0
    mean_selected_route_probability_all_route_active: float | None = None
    mean_local_probability_all_route_active: float | None = None
    mean_defer_probability_all_route_active: float | None = None
    mean_total_remote_probability_mass: float | None = None
    mean_number_of_legal_remote_destinations: float | None = None
    mean_local_minus_total_remote_probability: float | None = None
    mean_local_minus_total_remote_logit_margin: float | None = None
    mean_remote_probability_by_destination: Mapping[str, float] | None = None
    remote_probability_destination_valid_samples: Mapping[str, int] | None = None
    route_head_weight_grad_mean: float | None = None
    route_head_weight_grad_signed_sum: float | None = None
    route_head_bias_grad_mean: float | None = None
    route_head_bias_grad_signed_sum: float | None = None
    shared_trunk_grad_status: str = "none"
    shared_trunk_grad_norm: float | None = None
    shared_trunk_gradient_scope: str = "joint_actor_loss"
    shared_trunk_interference_status: str = "NOT OBSERVABLE WITHOUT ALGORITHM-PERTURBING EXTRA BACKWARD"
    route_head_gradient_rows: tuple[Any, ...] = ()
    samples: tuple[RouteSampleTelemetry, ...] = ()
    route_ppo_dynamics: RoutePPODynamics | None = None
    actor_ratio_mode: str = ActorRatioMode.JOINT.value
    branch_ppo_dynamics: BranchPPODynamics | None = None
    active_branch_matrix: tuple[tuple[tuple[bool, ...], ...], ...] = ()
    _advantage_distributions: Mapping[str, RouteDistributionSummary] = field(default_factory=dict, repr=False)
    _return_distributions: Mapping[str, RouteDistributionSummary] = field(default_factory=dict, repr=False)
    _td_residual_distributions: Mapping[str, RouteDistributionSummary] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.schema_version != ROUTE_TELEMETRY_SCHEMA_VERSION:
            raise RouteTelemetryError("route telemetry schema version is invalid")
        try:
            ActorRatioMode(self.actor_ratio_mode)
        except (TypeError, ValueError) as exc:
            raise RouteTelemetryError("actor_ratio_mode is invalid") from exc
        if self.branch_ppo_dynamics is not None and not isinstance(
            self.branch_ppo_dynamics, BranchPPODynamics
        ):
            raise TypeError("branch_ppo_dynamics must be BranchPPODynamics or None")
        if self.active_branch_matrix:
            agent_count = len(self.active_branch_matrix[0])
            if agent_count == 0 or any(
                len(time_step) != agent_count
                or any(
                    len(agent_row) != len(ACTION_BRANCH_ORDER)
                    or any(type(value) is not bool for value in agent_row)
                    for agent_row in time_step
                )
                for time_step in self.active_branch_matrix
            ):
                raise RouteTelemetryError(
                    "active_branch_matrix must have consistent [T,A,7] boolean rows"
                )
        count_names = (
            item.name
            for item in fields(self)
            if item.name.endswith("_count")
        )
        for name in count_names:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise RouteTelemetryError(f"{name} must be a non-negative integer")
        if self.route_entropy_valid_sample_count != self.route_branch_active_count:
            raise RouteTelemetryError("route entropy count must equal route-active count")
        if (
            self.route_active_advantage_valid_sample_count
            != self.route_branch_active_count
        ):
            raise RouteTelemetryError("route advantage count must equal route-active count")
        if (
            self.legal_remote_probability_valid_sample_count
            != self.legal_remote_route_count
        ):
            raise RouteTelemetryError("legal-remote probability count is inconsistent")
        if self.legal_remote_route_count > self.route_branch_active_count:
            raise RouteTelemetryError("legal-remote count exceeds route-active count")
        if self.route_selected_remote_count > self.legal_remote_route_count:
            raise RouteTelemetryError("remote selections exceed legal-remote samples")
        if (
            self.route_selected_local_count
            + self.route_selected_remote_count
            + self.route_selected_defer_count
            != self.route_branch_active_count
        ):
            raise RouteTelemetryError("route selection counts do not partition active samples")

        self._validate_optional_group(
            self.legal_remote_route_count,
            (
                "remote_selection_rate_given_legal_remote",
                "mean_selected_route_probability",
                "mean_local_probability",
                "mean_best_remote_probability",
                "mean_local_minus_best_remote_probability",
                "mean_local_minus_best_remote_logit_margin",
            ),
        )
        self._validate_optional_group(
            self.route_branch_active_count,
            (
                "route_entropy_mean",
                "route_active_advantage_mean",
                "route_active_advantage_std",
                "route_active_return_mean",
                "route_active_return_std",
            ),
        )
        for prefix in (
            "local_route",
            "remote_route",
            "defer_route",
            "legal_remote_local",
            "legal_remote_remote",
        ):
            self._validate_optional_group(
                getattr(self, f"{prefix}_valid_sample_count"),
                (f"{prefix}_advantage_mean", f"{prefix}_return_mean"),
            )
        if self.remote_selection_rate_given_legal_remote is not None and not (
            0.0 <= self.remote_selection_rate_given_legal_remote <= 1.0
        ):
            raise RouteTelemetryError("conditional remote selection rate is outside [0, 1]")
        for prefix in ("weight", "bias", "total"):
            GradientMeasurement(
                status=getattr(self, f"route_head_{prefix}_grad_status"),
                norm=getattr(self, f"route_head_{prefix}_grad_norm"),
            )

    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        names = tuple(
            item.name
            for item in fields(cls)
            if not item.name.startswith("_")
            and item.name not in {
                "samples",
                "route_ppo_dynamics",
                "actor_ratio_mode",
                "branch_ppo_dynamics",
                "active_branch_matrix",
            }
        )
        distribution_names = []
        for group in ROUTE_DISTRIBUTION_GROUPS:
            distribution_names.extend(
                f"{group}_advantage_{suffix}"
                for suffix in (
                    "valid_sample_count",
                    "mean",
                    "std",
                    "median",
                    "p25",
                    "p75",
                    "positive_fraction",
                    "negative_fraction",
                )
            )
            distribution_names.extend(
                f"{group}_return_target_{suffix}"
                for suffix in (
                    "valid_sample_count",
                    "mean",
                    "std",
                    "median",
                    "p25",
                    "p75",
                    "positive_fraction",
                    "negative_fraction",
                )
            )
            distribution_names.extend(
                f"{group}_td_residual_{suffix}"
                for suffix in (
                    "valid_sample_count",
                    "mean",
                    "std",
                    "median",
                    "p25",
                    "p75",
                    "positive_fraction",
                    "negative_fraction",
                )
            )
        return (
            names
            + tuple(ROUTE_PPO_RECORD_FIELDS)
            + tuple(BRANCH_PPO_RECORD_FIELDS)
            + tuple(distribution_names)
        )

    def record(self) -> dict[str, object]:
        """Return a deterministic field-ordered flat record for CSV/JSONL."""

        result = {
            name: getattr(self, name)
            for name in self.field_names()
            if hasattr(self, name)
        }
        result["schema_version"] = self.schema_version
        result["route_telemetry_schema_version"] = ROUTE_TELEMETRY_SCHEMA_VERSION
        result["actor_ratio_mode"] = self.actor_ratio_mode
        rows = []
        for row in self.route_head_gradient_rows or ():
            rows.append(row.record() if hasattr(row, "record") else row)
        result["route_head_gradient_rows"] = rows
        if self.route_ppo_dynamics is not None:
            result.update(self.route_ppo_dynamics.record())
        else:
            result.update({name: None for name in ROUTE_PPO_RECORD_FIELDS})
        if self.branch_ppo_dynamics is not None:
            result.update(self.branch_ppo_dynamics.record())
        else:
            result.update({name: None for name in BRANCH_PPO_RECORD_FIELDS})
        for prefix, summaries, include_quantiles in (
            ("advantage", self._advantage_distributions, True),
            ("return_target", self._return_distributions, True),
            ("td_residual", self._td_residual_distributions, True),
        ):
            for group in ROUTE_DISTRIBUTION_GROUPS:
                summary = (summaries or {}).get(group, RouteDistributionSummary())
                result.update(summary.record(
                    f"{group}_{prefix}", include_quantiles=include_quantiles
                ))
        return result

    def _validate_optional_group(
        self,
        count: int,
        names: Sequence[str],
    ) -> None:
        for name in names:
            value = getattr(self, name)
            if count == 0:
                if value is not None:
                    raise RouteTelemetryError(f"{name} must be NA without valid samples")
            elif value is None or not math.isfinite(value):
                raise RouteTelemetryError(f"{name} must be finite with valid samples")


def _gradient_measurement(parameter: nn.Parameter) -> GradientMeasurement:
    gradient = parameter.grad
    if gradient is None:
        return GradientMeasurement("none", None)
    detached = gradient.detach()
    if not torch.isfinite(detached).all():
        raise RouteTelemetryError("route-head gradient contains NaN or Inf")
    norm = float(torch.linalg.vector_norm(detached).cpu().item())
    return GradientMeasurement(
        "zero" if norm == 0.0 else "finite_nonzero",
        norm,
        mean=float(detached.mean().cpu().item()),
        signed_sum=float(detached.sum().cpu().item()),
    )


def measure_route_head_gradients(route_head: nn.Linear, route_domains: Any = None) -> RouteHeadGradientTelemetry:
    """Read existing route-head gradients without changing or clipping them."""

    if not isinstance(route_head, nn.Linear) or route_head.bias is None:
        raise TypeError("route_head must be a Linear layer with weight and bias")
    with torch.no_grad():
        weight = _gradient_measurement(route_head.weight)
        bias = _gradient_measurement(route_head.bias)
        present = tuple(
            item.norm for item in (weight, bias) if item.norm is not None
        )
        if not present:
            total = GradientMeasurement("none", None)
        else:
            total_norm = math.sqrt(math.fsum(value * value for value in present))
            total = GradientMeasurement(
                "zero" if total_norm == 0.0 else "finite_nonzero",
                total_norm,
            )
    return RouteHeadGradientTelemetry(
        weight=weight,
        bias=bias,
        total=total,
        rows=measure_route_head_gradient_rows(route_head, route_domains),
    )


def measure_route_head_gradient_rows(
    route_head: nn.Linear, route_domains: Any = None
) -> tuple[RouteHeadGradientRowTelemetry, ...]:
    """Map existing route-head row gradients to local/defer/remote semantics."""

    domains = list(route_domains or ())
    if domains and isinstance(domains[0], str):
        domains = [tuple(domains)]
    if not domains:
        domains = [tuple(range(route_head.out_features))]
    rows: list[RouteHeadGradientRowTelemetry] = []
    for row_index in range(route_head.out_features):
        values = [
            domain[row_index]
            for domain in domains
            if row_index < len(domain)
        ]
        destinations = tuple(sorted({
            value for value in values
            if isinstance(value, int) and not isinstance(value, bool)
        }, key=str))
        if any(type(value) is str and value == "local" for value in values):
            role = "local"
        elif any(type(value) is str and value == "defer" for value in values):
            role = "defer"
        elif destinations:
            role = "remote"

        elif any(type(value) is str and value == "idle" for value in values):
            role = "idle"
        else:
            role = "unknown"
        gradient = route_head.weight.grad[row_index] if route_head.weight.grad is not None else None
        measurement = GradientMeasurement("none", None)
        if gradient is not None:
            detached = gradient.detach()
            if not torch.isfinite(detached).all():
                measurement = GradientMeasurement("none", None)
            else:
                norm = float(torch.linalg.vector_norm(detached).cpu().item())
                measurement = GradientMeasurement(
                    "zero" if norm == 0.0 else "finite_nonzero", norm,
                    mean=float(detached.mean().cpu().item()),
                    signed_sum=float(detached.sum().cpu().item()),
                )
        bias_value = None
        if route_head.bias.grad is not None:
            bias_value = float(route_head.bias.grad[row_index].detach().cpu().item())
        rows.append(RouteHeadGradientRowTelemetry(
            row_index=row_index,
            semantic_role=role,
            destination_uavs=destinations,
            grad_status=measurement.status,
            grad_norm=measurement.norm,
            grad_mean=measurement.mean,
            grad_signed_sum=measurement.signed_sum,
            bias_grad=bias_value,
            parameter_update_direction_mean=-measurement.mean if measurement.mean is not None else None,
            parameter_update_direction_signed_sum=-measurement.signed_sum if measurement.signed_sum is not None else None,
            bias_update_direction=-bias_value if bias_value is not None else None,
        ))
    return tuple(rows)

def _count(mask: Tensor) -> int:

    return int(mask.sum().detach().cpu().item())


def _optional_mean(values: Tensor, mask: Tensor) -> float | None:
    selected = values.masked_select(mask)
    if selected.numel() == 0:
        return None
    return float(selected.mean().detach().cpu().item())


def _optional_population_std(values: Tensor, mask: Tensor) -> float | None:
    selected = values.masked_select(mask)
    if selected.numel() == 0:
        return None
    return float(selected.std(unbiased=False).detach().cpu().item())


def _flatten_proposals(
    proposals: Sequence[Sequence[Sequence[ActionProposal]]],
) -> tuple[ActionProposal, ...]:
    return tuple(
        proposal
        for batch in proposals
        for time_step in batch
        for proposal in time_step
    )


def collect_route_telemetry(
    *,
    policy: SequentialActionDistributionOutput,
    action_mask_batch: SequentialActionMaskBatch,
    proposals: Sequence[Sequence[Sequence[ActionProposal]]],
    advantage: Tensor,
    return_target: Tensor,
    sequence_valid_mask: Tensor,
    route_head: nn.Linear,
    old_route_log_prob: Tensor | None = None,
    old_branch_log_probs: Tensor | None = None,
    td_residual: Tensor | None = None,
    shared_trunk: Any = None,
    epsilon_clip: float = 0.2,
    actor_ratio_mode: str = ActorRatioMode.JOINT.value,
) -> RouteTelemetry:
    """Aggregate route telemetry without changing the PPO algorithm state."""

    if not isinstance(policy, SequentialActionDistributionOutput):
        raise TypeError("policy must be SequentialActionDistributionOutput")
    if not isinstance(action_mask_batch, SequentialActionMaskBatch):
        raise TypeError("action_mask_batch must be SequentialActionMaskBatch")
    try:
        ratio_mode = ActorRatioMode(actor_ratio_mode)
    except (TypeError, ValueError) as exc:
        raise RouteTelemetryError("actor_ratio_mode is invalid") from exc
    route_active = policy.active_branches["route"]
    if route_active.dtype != torch.bool or route_active.ndim != 3:
        raise RouteTelemetryError("route activity must be boolean [B,T,A]")
    batch, time, agents = route_active.shape
    if action_mask_batch.shape != (batch, time, agents):
        raise RouteTelemetryError("action-mask contracts differ from route axes")
    valid_credit_shapes = {(batch, time), (batch, time, agents)}
    if tuple(advantage.shape) not in valid_credit_shapes:
        raise RouteTelemetryError("advantage must have shape [B,T] or [B,T,A]")
    if tuple(return_target.shape) != tuple(advantage.shape):
        raise RouteTelemetryError(
            "return_target must share the scalar or per-agent advantage shape"
        )
    if td_residual is not None and tuple(td_residual.shape) != tuple(advantage.shape):
        raise RouteTelemetryError(
            "td_residual must share the scalar or per-agent advantage shape"
        )
    if (
        tuple(sequence_valid_mask.shape) != (batch, time)
        or sequence_valid_mask.dtype != torch.bool
    ):
        raise RouteTelemetryError("sequence_valid_mask must be boolean [B,T]")

    route_probabilities = policy.probabilities["route"].detach()
    route_logits = policy.raw_logits["route"].detach()
    route_masks = policy.action_masks["route"].detach().bool()
    route_indices = policy.action_indices["route"].detach().long()
    route_entropy = policy.branch_entropies["route"].detach()
    dimension = route_probabilities.shape[-1]
    expected_vector_shape = (batch, time, agents, dimension)
    if (
        tuple(route_probabilities.shape) != expected_vector_shape
        or tuple(route_logits.shape) != expected_vector_shape
        or tuple(route_masks.shape) != expected_vector_shape
        or tuple(route_indices.shape) != (batch, time, agents)
        or tuple(route_entropy.shape) != (batch, time, agents)
    ):
        raise RouteTelemetryError("route policy tensors have incompatible shapes")

    contracts = action_mask_batch.flattened()
    flattened_proposals = _flatten_proposals(proposals)
    if len(contracts) != batch * time * agents or len(flattened_proposals) != len(contracts):
        raise RouteTelemetryError("proposal or contract count differs from route axes")

    remote_domain_rows: list[list[bool]] = []
    local_domain_rows: list[list[bool]] = []
    selected_local_rows: list[bool] = []
    selected_remote_rows: list[bool] = []
    selected_defer_rows: list[bool] = []
    domains: list[tuple[object, ...]] = []
    source_uavs: list[object] = []
    for contract, proposal in zip(contracts, flattened_proposals):
        domain = tuple(contract.route_domain)
        if len(domain) != dimension:
            raise RouteTelemetryError("route domain differs from policy dimension")
        remote_row = [
            isinstance(value, int) and not isinstance(value, bool) for value in domain
        ]
        local_row = [type(value) is str and value == "local" for value in domain]
        if sum(local_row) != 1:
            raise RouteTelemetryError("route domain must contain exactly one local action")
        domains.append(domain)
        source_uavs.append(
            getattr(contract, "source_uav", getattr(contract, "uav_id", None))
        )
        remote_domain_rows.append(remote_row)
        local_domain_rows.append(local_row)
        selected = proposal.route
        selected_local_rows.append(type(selected) is str and selected == "local")
        selected_defer_rows.append(type(selected) is str and selected == "defer")
        selected_remote_rows.append(
            isinstance(selected, int) and not isinstance(selected, bool)
        )

    device = route_probabilities.device
    remote_domain = torch.tensor(remote_domain_rows, dtype=torch.bool, device=device).reshape(
        expected_vector_shape
    )
    local_domain = torch.tensor(local_domain_rows, dtype=torch.bool, device=device).reshape(
        expected_vector_shape
    )
    selected_local = torch.tensor(selected_local_rows, dtype=torch.bool, device=device).reshape(
        batch, time, agents
    )
    selected_remote = torch.tensor(selected_remote_rows, dtype=torch.bool, device=device).reshape(
        batch, time, agents
    )
    selected_defer = torch.tensor(selected_defer_rows, dtype=torch.bool, device=device).reshape(
        batch, time, agents
    )

    with torch.no_grad():
        valid_actor = sequence_valid_mask.to(device=device).unsqueeze(-1).expand_as(route_active)
        active = route_active.detach() & valid_actor
        selected_partition = selected_local | selected_remote | selected_defer
        if torch.any(active & ~selected_partition):
            raise RouteTelemetryError("active route sample is not local, remote, or defer")
        legal_remote_actions = route_masks & remote_domain
        legal_remote = active & torch.any(legal_remote_actions, dim=-1)
        if torch.any(active & selected_remote & ~legal_remote):
            raise RouteTelemetryError("selected remote lacks a legal remote action")

        selected_probability = torch.gather(
            route_probabilities, -1, route_indices.unsqueeze(-1)
        ).squeeze(-1)
        local_probability = (route_probabilities * local_domain.to(route_probabilities.dtype)).sum(dim=-1)
        defer_domain = torch.tensor(
            [[type(value) is str and value == "defer" for value in domain] for domain in domains],
            dtype=torch.bool,
            device=device,
        ).reshape(expected_vector_shape)
        defer_probability = (
            route_probabilities * defer_domain.to(route_probabilities.dtype)
        ).sum(dim=-1)
        legal_remote_probability = route_probabilities.masked_fill(
            ~legal_remote_actions, 0.0
        )
        total_remote_probability = legal_remote_probability.sum(dim=-1)
        best_remote_probability = route_probabilities.masked_fill(
            ~legal_remote_actions, -torch.inf
        ).max(dim=-1).values
        best_remote_logit = route_logits.masked_fill(
            ~legal_remote_actions, -torch.inf
        ).max(dim=-1).values
        legal_remote_logit_sum = torch.logsumexp(
            route_logits.masked_fill(~legal_remote_actions, -torch.inf), dim=-1
        )
        local_logit = (route_logits * local_domain.to(route_logits.dtype)).sum(dim=-1)
        probability_margin = local_probability - best_remote_probability
        logit_margin = local_logit - best_remote_logit
        total_logit_margin = local_logit - legal_remote_logit_sum
        if not torch.isfinite(best_remote_probability.masked_select(legal_remote)).all():
            raise RouteTelemetryError("best legal-remote probability is not finite")
        if not torch.isfinite(best_remote_logit.masked_select(legal_remote)).all():
            raise RouteTelemetryError("best legal-remote logit is not finite")

        expanded_advantage = advantage.detach().to(device=device)
        expanded_return = return_target.detach().to(device=device)
        if expanded_advantage.ndim == 2:
            expanded_advantage = expanded_advantage.unsqueeze(-1).expand_as(route_active)
            expanded_return = expanded_return.unsqueeze(-1).expand_as(route_active)
        raw_branch_active = torch.stack(
            [
                policy.active_branches[branch].detach().to(device=device)
                for branch in ACTION_BRANCH_ORDER
            ],
            dim=-1,
        )
        if tuple(raw_branch_active.shape) != (
            batch,
            time,
            agents,
            len(ACTION_BRANCH_ORDER),
        ):
            raise RouteTelemetryError("active branch tensor must have shape [B,T,A,7]")
        valid_branch_active = raw_branch_active & valid_actor.unsqueeze(-1)
        new_branch_log_probs = torch.stack(
            [
                policy.branch_log_probs[branch].detach().to(device=device)
                for branch in ACTION_BRANCH_ORDER
            ],
            dim=-1,
        )
        branch_ppo = compute_branch_ppo_dynamics(
            old_branch_log_probs,
            new_branch_log_probs,
            valid_branch_active,
            expanded_advantage.unsqueeze(-1).expand_as(new_branch_log_probs),
            epsilon_clip,
        )
        active_branch_matrix = tuple(
            tuple(
                tuple(bool(value) for value in agent_row)
                for agent_row in time_step
            )
            for time_step in raw_branch_active.reshape(
                batch * time,
                agents,
                len(ACTION_BRANCH_ORDER),
            ).detach().cpu().tolist()
        )
        td_values = td_residual.detach().to(device=device) if td_residual is not None else None
        if td_values is not None and td_values.ndim == 2:
            td_values = td_values.unsqueeze(-1).expand_as(route_active)
        local_group = active & selected_local
        remote_group = active & selected_remote
        defer_group = active & selected_defer
        legal_local_group = legal_remote & selected_local
        legal_remote_group = legal_remote & selected_remote
        category_masks = {
            "local": local_group,
            "remote": remote_group,
            "defer": defer_group,
        }
        gradients = measure_route_head_gradients(route_head, domains)
        trunk_measurement = measure_shared_trunk_gradients(
            shared_trunk if shared_trunk is not None else ()
        )

        legal_count = _count(legal_remote)
        active_count = _count(active)
        local_count = _count(local_group)
        remote_count = _count(remote_group)
        defer_count = _count(defer_group)
        legal_local_count = _count(legal_local_group)
        legal_selected_remote_count = _count(legal_remote_group)
        old_route = old_route_log_prob
        if old_route is None and old_branch_log_probs is not None:
            old_route = old_branch_log_probs[..., 0]
        if old_route is not None:
            old_route = old_route.detach().to(device=device).reshape(batch, time, agents)
        new_route = policy.branch_log_probs["route"].detach().to(device=device).reshape(
            batch, time, agents
        )
        route_ppo = compute_route_only_ppo_dynamics(
            old_route, new_route, active, category_masks, epsilon_clip
        )

        advantage_distributions = {}
        return_distributions = {}
        td_distributions = {}
        group_masks = {
            "route_active": active,
            "local_route": local_group,
            "remote_route": remote_group,
            "defer_route": defer_group,
            "legal_remote_local": legal_local_group,
            "legal_remote_remote": legal_remote_group,
        }
        for name, mask in group_masks.items():
            advantage_distributions[name] = summarize_distribution(expanded_advantage, mask)
            return_distributions[name] = summarize_distribution(expanded_return, mask)
            td_distributions[name] = (
                summarize_distribution(td_values, mask)
                if td_values is not None
                else RouteDistributionSummary()
            )

        def mean_active(values: Tensor) -> float | None:
            return _optional_mean(values, active)

        def mean_legal(values: Tensor) -> float | None:
            return _optional_mean(values, legal_remote)

        destination_sum: dict[str, float] = {}
        destination_count: dict[str, int] = {}
        flat_remote = legal_remote_actions.reshape(-1, dimension)
        flat_probability = route_probabilities.reshape(-1, dimension)
        flat_legal = legal_remote.reshape(-1)
        for flat_index in range(flat_probability.shape[0]):
            if not bool(flat_legal[flat_index]):
                continue
            for row_index in range(dimension):
                if bool(flat_remote[flat_index, row_index]):
                    key = str(domains[flat_index][row_index])
                    destination_sum[key] = destination_sum.get(key, 0.0) + float(
                        flat_probability[flat_index, row_index].item()
                    )
                    destination_count[key] = destination_count.get(key, 0) + 1

        samples: list[RouteSampleTelemetry] = []
        flat_active = active.reshape(-1)
        flat_indices = route_indices.reshape(-1)
        flat_selected = selected_probability.reshape(-1)
        flat_old = old_route.reshape(-1) if old_route is not None else None
        flat_new = new_route.reshape(-1)
        flat_adv = expanded_advantage.reshape(-1)
        flat_ret = expanded_return.reshape(-1)
        flat_td = td_values.reshape(-1) if td_values is not None else None
        flat_total_remote = total_remote_probability.reshape(-1)
        flat_local = local_probability.reshape(-1)
        flat_defer = defer_probability.reshape(-1)
        flat_best = best_remote_probability.reshape(-1)
        for flat_index in torch.nonzero(flat_active, as_tuple=False).reshape(-1).tolist():
            b = flat_index // (time * agents)
            remainder = flat_index % (time * agents)
            t = remainder // agents
            a = remainder % agents
            action_index = int(flat_indices[flat_index].item())
            domain = domains[flat_index]
            selected_value = flattened_proposals[flat_index].route
            category = (
                "local" if bool(selected_local.reshape(-1)[flat_index])
                else "remote" if bool(selected_remote.reshape(-1)[flat_index])
                else "defer"
            )
            legal_indices = [
                index for index, value in enumerate(domain)
                if isinstance(value, int) and not isinstance(value, bool) and bool(legal_remote_actions.reshape(-1, dimension)[flat_index, index])
            ]
            destination_probabilities = {
                str(domain[index]): float(flat_probability[flat_index, index].item())
                for index in legal_indices
            }
            old_value = float(flat_old[flat_index].item()) if flat_old is not None else None
            new_value = float(flat_new[flat_index].item())
            log_ratio = new_value - old_value if old_value is not None else None
            ratio = math.exp(log_ratio) if log_ratio is not None else None
            samples.append(RouteSampleTelemetry(
                batch_index=b,
                time_index=t,
                agent_index=a,
                source_uav=source_uavs[flat_index],
                category=category,
                legal_remote_destinations=tuple(domain[index] for index in legal_indices),
                remote_probability_by_destination=destination_probabilities,
                local_probability=float(flat_local[flat_index].item()),
                defer_probability=float(flat_defer[flat_index].item()),
                total_remote_probability_mass=float(flat_total_remote[flat_index].item()),
                best_remote_probability=(
                    float(flat_best[flat_index].item()) if legal_indices else None
                ),
                number_of_legal_remote_destinations=len(legal_indices),
                selected_route_action_index=action_index,
                selected_destination_uav=(
                    source_uavs[flat_index] if category == "local"
                    else selected_value if category == "remote" else None
                ),
                old_route_log_prob=old_value,
                new_route_log_prob=new_value,
                route_log_ratio=log_ratio,
                route_ratio=ratio,
                route_approx_kl=(ratio - 1.0 - log_ratio) if ratio is not None else None,
                route_clip_indicator=(
                    (ratio < 1.0 - epsilon_clip or ratio > 1.0 + epsilon_clip)
                    if ratio is not None else None
                ),
                advantage=float(flat_adv[flat_index].item()),
                return_target=float(flat_ret[flat_index].item()),
                td_residual=float(flat_td[flat_index].item()) if flat_td is not None else None,
            ))

        return RouteTelemetry(
            schema_version=ROUTE_TELEMETRY_SCHEMA_VERSION,
            route_branch_active_count=active_count,
            legal_remote_route_count=legal_count,
            route_selected_local_count=local_count,
            route_selected_remote_count=remote_count,
            route_selected_defer_count=defer_count,
            remote_selection_rate_given_legal_remote=(
                None if legal_count == 0 else remote_count / legal_count
            ),
            route_entropy_valid_sample_count=active_count,
            route_entropy_mean=_optional_mean(route_entropy, active),
            legal_remote_probability_valid_sample_count=legal_count,
            mean_selected_route_probability=_optional_mean(
                selected_probability, legal_remote
            ),
            mean_local_probability=_optional_mean(local_probability, legal_remote),
            mean_best_remote_probability=_optional_mean(
                best_remote_probability, legal_remote
            ),
            mean_local_minus_best_remote_probability=_optional_mean(
                probability_margin, legal_remote
            ),
            mean_local_minus_best_remote_logit_margin=_optional_mean(
                logit_margin, legal_remote
            ),
            route_active_advantage_valid_sample_count=active_count,
            route_active_advantage_mean=_optional_mean(expanded_advantage, active),
            route_active_advantage_std=_optional_population_std(
                expanded_advantage, active
            ),
            route_active_return_mean=_optional_mean(expanded_return, active),
            route_active_return_std=_optional_population_std(expanded_return, active),
            local_route_valid_sample_count=local_count,
            local_route_advantage_mean=_optional_mean(expanded_advantage, local_group),
            local_route_return_mean=_optional_mean(expanded_return, local_group),
            remote_route_valid_sample_count=remote_count,
            remote_route_advantage_mean=_optional_mean(expanded_advantage, remote_group),
            remote_route_return_mean=_optional_mean(expanded_return, remote_group),
            defer_route_valid_sample_count=defer_count,
            defer_route_advantage_mean=_optional_mean(expanded_advantage, defer_group),
            defer_route_return_mean=_optional_mean(expanded_return, defer_group),
            legal_remote_local_valid_sample_count=legal_local_count,
            legal_remote_local_advantage_mean=_optional_mean(
                expanded_advantage, legal_local_group
            ),
            legal_remote_local_return_mean=_optional_mean(
                expanded_return, legal_local_group
            ),
            legal_remote_remote_valid_sample_count=legal_selected_remote_count,
            legal_remote_remote_advantage_mean=_optional_mean(
                expanded_advantage, legal_remote_group
            ),
            legal_remote_remote_return_mean=_optional_mean(
                expanded_return, legal_remote_group
            ),
            route_head_weight_grad_status=gradients.weight.status,
            route_head_weight_grad_norm=gradients.weight.norm,
            route_head_bias_grad_status=gradients.bias.status,
            route_head_bias_grad_norm=gradients.bias.norm,
            route_head_total_grad_status=gradients.total.status,
            route_head_total_grad_norm=gradients.total.norm,
            route_probability_valid_sample_count=active_count,
            mean_selected_route_probability_all_route_active=mean_active(selected_probability),
            mean_local_probability_all_route_active=mean_active(local_probability),
            mean_defer_probability_all_route_active=mean_active(defer_probability),
            mean_total_remote_probability_mass=mean_active(total_remote_probability),
            mean_number_of_legal_remote_destinations=_optional_mean(
                legal_remote_actions.sum(dim=-1).float(), active
            ),
            mean_local_minus_total_remote_probability=mean_active(
                local_probability - total_remote_probability
            ),
            mean_local_minus_total_remote_logit_margin=mean_legal(total_logit_margin),
            mean_remote_probability_by_destination={
                key: destination_sum[key] / destination_count[key]
                for key in destination_sum if destination_count[key]
            },
            remote_probability_destination_valid_samples=destination_count,
            route_head_gradient_rows=gradients.rows,
            route_ppo_dynamics=route_ppo,
            actor_ratio_mode=ratio_mode.value,
            branch_ppo_dynamics=branch_ppo,
            active_branch_matrix=active_branch_matrix,
            _advantage_distributions=advantage_distributions,
            _return_distributions=return_distributions,
            _td_residual_distributions=td_distributions,
            shared_trunk_grad_status=trunk_measurement.status,
            shared_trunk_grad_norm=trunk_measurement.norm,
            route_head_weight_grad_mean=gradients.weight.mean,
            route_head_weight_grad_signed_sum=gradients.weight.signed_sum,
            route_head_bias_grad_mean=gradients.bias.mean,
            route_head_bias_grad_signed_sum=gradients.bias.signed_sum,
            samples=tuple(samples),
        )



@dataclass(frozen=True)
class RouteDistributionSummary:
    """Distribution summary for a fixed, already-selected route group."""

    count: int = 0
    mean: float | None = None
    std: float | None = None
    median: float | None = None
    p25: float | None = None
    p75: float | None = None
    positive_fraction: float | None = None
    negative_fraction: float | None = None

    def record(self, prefix: str, *, include_quantiles: bool = True) -> dict[str, object]:
        result = {
            f"{prefix}_valid_sample_count": self.count,
            f"{prefix}_mean": self.mean,
            f"{prefix}_std": self.std,
            f"{prefix}_median": self.median,
        }
        if include_quantiles:
            result.update({
                f"{prefix}_p25": self.p25,
                f"{prefix}_p75": self.p75,
                f"{prefix}_positive_fraction": self.positive_fraction,
                f"{prefix}_negative_fraction": self.negative_fraction,
            })
        return result


def summarize_distribution(values: Tensor, mask: Tensor | None = None) -> RouteDistributionSummary:
    selected = values.detach().float()
    if mask is not None:
        selected = selected.masked_select(mask.to(device=selected.device, dtype=torch.bool))
    selected = selected[torch.isfinite(selected)]
    if selected.numel() == 0:
        return RouteDistributionSummary()
    ordered = selected.sort().values
    return RouteDistributionSummary(
        count=int(selected.numel()),
        mean=float(selected.mean().cpu().item()),
        std=float(selected.std(unbiased=False).cpu().item()),
        median=float(median(selected.cpu().tolist())),
        p25=float(torch.quantile(selected, 0.25).cpu().item()),
        p75=float(torch.quantile(selected, 0.75).cpu().item()),
        positive_fraction=float((selected > 0).float().mean().cpu().item()),
        negative_fraction=float((selected < 0).float().mean().cpu().item()),
    )


@dataclass(frozen=True)
class RoutePPODynamicsGroup:
    count: int = 0
    old_log_prob_mean: float | None = None
    new_log_prob_mean: float | None = None
    log_ratio_mean: float | None = None
    ratio_mean: float | None = None
    approx_kl_mean: float | None = None
    clip_indicator_fraction: float | None = None
    clip_fraction: float | None = None

    def record(self, prefix: str) -> dict[str, object]:
        return {
            f"{prefix}_count": self.count,
            f"{prefix}_old_log_prob_mean": self.old_log_prob_mean,
            f"{prefix}_new_log_prob_mean": self.new_log_prob_mean,
            f"{prefix}_log_ratio_mean": self.log_ratio_mean,
            f"{prefix}_ratio_mean": self.ratio_mean,
            f"{prefix}_approx_kl_mean": self.approx_kl_mean,
            f"{prefix}_clip_indicator_fraction": self.clip_indicator_fraction,
            f"{prefix}_clip_fraction": self.clip_fraction,
        }


@dataclass(frozen=True)
class RoutePPODynamics:
    epsilon_clip: float
    overall: RoutePPODynamicsGroup
    groups: Mapping[str, RoutePPODynamicsGroup] = field(default_factory=dict)

    def record(self) -> dict[str, object]:
        result: dict[str, object] = {"route_ppo_epsilon_clip": self.epsilon_clip}
        result.update(self.overall.record("route_ppo_overall"))
        for category in ROUTE_ACTION_CATEGORIES:
            result.update(self.groups.get(category, RoutePPODynamicsGroup()).record(
                f"{category}_route_ppo"
            ))
        return result


@dataclass(frozen=True)
class RouteHeadGradientRowTelemetry:
    row_index: int
    semantic_role: str
    destination_uavs: tuple[object, ...] = ()
    grad_status: str = "none"
    grad_norm: float | None = None
    grad_mean: float | None = None
    grad_signed_sum: float | None = None
    bias_grad: float | None = None
    parameter_update_direction_mean: float | None = None
    parameter_update_direction_signed_sum: float | None = None
    bias_update_direction: float | None = None

    def record(self) -> dict[str, object]:
        return {
            "row_index": self.row_index,
            "semantic_role": self.semantic_role,
            "destination_uavs": list(self.destination_uavs),
            "grad_status": self.grad_status,
            "grad_norm": self.grad_norm,
            "grad_mean": self.grad_mean,
            "grad_signed_sum": self.grad_signed_sum,
            "bias_grad": self.bias_grad,
            "parameter_update_direction_mean": self.parameter_update_direction_mean,
            "parameter_update_direction_signed_sum": self.parameter_update_direction_signed_sum,
            "bias_update_direction": self.bias_update_direction,
        }


@dataclass(frozen=True)
class RouteSampleTelemetry:
    batch_index: int
    time_index: int
    agent_index: int
    source_uav: object
    category: str
    legal_remote_destinations: tuple[object, ...]
    remote_probability_by_destination: Mapping[str, float]
    local_probability: float | None
    defer_probability: float | None
    total_remote_probability_mass: float | None
    best_remote_probability: float | None
    number_of_legal_remote_destinations: int
    selected_route_action_index: int
    selected_destination_uav: object
    old_route_log_prob: float | None
    new_route_log_prob: float | None
    route_log_ratio: float | None
    route_ratio: float | None
    route_approx_kl: float | None
    route_clip_indicator: bool | None
    advantage: float | None
    return_target: float | None
    td_residual: float | None

    def __post_init__(self) -> None:
        if self.number_of_legal_remote_destinations != len(self.legal_remote_destinations):
            raise RouteTelemetryError("legal remote destination count is inconsistent")
        if self.total_remote_probability_mass is not None:
            total = sum(self.remote_probability_by_destination.values())
            if not math.isclose(total, self.total_remote_probability_mass, rel_tol=1e-6, abs_tol=1e-6):
                raise RouteTelemetryError("total remote probability is not the legal-destination sum")

    def record(self) -> dict[str, object]:
        return {
            "route_sample_batch_index": self.batch_index,
            "route_sample_time_index": self.time_index,
            "route_sample_agent_index": self.agent_index,
            "route_sample_source_uav": self.source_uav,
            "route_sample_category": self.category,
            "route_sample_legal_remote_destinations": list(self.legal_remote_destinations),
            "route_sample_remote_probability_by_destination": dict(self.remote_probability_by_destination),
            "route_sample_local_probability": self.local_probability,
            "route_sample_defer_probability": self.defer_probability,
            "route_sample_total_remote_probability_mass": self.total_remote_probability_mass,
            "route_sample_best_remote_probability": self.best_remote_probability,
            "route_sample_number_of_legal_remote_destinations": self.number_of_legal_remote_destinations,
            "route_sample_selected_route_action_index": self.selected_route_action_index,
            "route_sample_selected_destination_uav": self.selected_destination_uav,
            "route_sample_old_route_log_prob": self.old_route_log_prob,
            "route_sample_new_route_log_prob": self.new_route_log_prob,
            "route_sample_log_ratio": self.route_log_ratio,
            "route_sample_ratio": self.route_ratio,
            "route_sample_approx_kl": self.route_approx_kl,
            "route_sample_clip_indicator": self.route_clip_indicator,
            "route_sample_advantage": self.advantage,
            "route_sample_return_target": self.return_target,
            "route_sample_td_residual": self.td_residual,
        }


def _branch_ppo_group(
    old: Tensor,
    new: Tensor,
    active: Tensor,
    advantage: Tensor,
    epsilon_clip: float,
) -> BranchPPODynamicsGroup:
    selected_old = old.masked_select(active)
    selected_new = new.masked_select(active)
    selected_advantage = advantage.masked_select(active)
    if selected_old.numel() == 0:
        return BranchPPODynamicsGroup()
    log_ratio = selected_new - selected_old
    ratio = torch.exp(log_ratio)
    if not torch.isfinite(ratio).all():
        raise RouteTelemetryError("branch PPO ratio contains NaN or Inf")
    clipped_ratio = torch.clamp(
        ratio,
        1.0 - epsilon_clip,
        1.0 + epsilon_clip,
    )
    surrogate = torch.minimum(
        ratio * selected_advantage,
        clipped_ratio * selected_advantage,
    )
    clipped = ratio != clipped_ratio
    ratio_values = ratio.detach().cpu().tolist()
    return BranchPPODynamicsGroup(
        active_count=int(ratio.numel()),
        ratio_mean=float(ratio.mean().cpu().item()),
        ratio_median=float(median(ratio_values)),
        ratio_std=float(ratio.std(unbiased=False).cpu().item()),
        approx_kl_mean=float(
            ((ratio - 1.0) - log_ratio).mean().cpu().item()
        ),
        clip_fraction=float(clipped.float().mean().cpu().item()),
        surrogate_contribution_mean=float(surrogate.mean().cpu().item()),
        advantage_mean=float(selected_advantage.mean().cpu().item()),
    )


def compute_branch_ppo_dynamics(
    old_branch_log_probs: Tensor | None,
    new_branch_log_probs: Tensor | None,
    active_branch_indicators: Tensor,
    expanded_advantage: Tensor,
    epsilon_clip: float,
) -> BranchPPODynamics | None:
    """Summarize each branch from detached tensors already used by PPO."""

    if old_branch_log_probs is None or new_branch_log_probs is None:
        return None
    if (
        not isinstance(active_branch_indicators, Tensor)
        or active_branch_indicators.dtype != torch.bool
        or active_branch_indicators.ndim != 4
        or active_branch_indicators.shape[-1] != len(ACTION_BRANCH_ORDER)
    ):
        raise RouteTelemetryError(
            "active_branch_indicators must be boolean [B,T,A,7]"
        )
    expected_shape = active_branch_indicators.shape
    for name, tensor in (
        ("old_branch_log_probs", old_branch_log_probs),
        ("new_branch_log_probs", new_branch_log_probs),
        ("expanded_advantage", expanded_advantage),
    ):
        if not isinstance(tensor, Tensor) or tensor.shape != expected_shape:
            raise RouteTelemetryError(f"{name} must have shape [B,T,A,7]")
    new = new_branch_log_probs.detach().float()
    old = old_branch_log_probs.detach().to(device=new.device, dtype=new.dtype)
    advantage = expanded_advantage.detach().to(
        device=new.device,
        dtype=new.dtype,
    )
    active = active_branch_indicators.detach().to(device=new.device)
    finite = torch.isfinite(old) & torch.isfinite(new) & torch.isfinite(advantage)
    active = active & finite
    groups = {
        branch: _branch_ppo_group(
            old[..., index],
            new[..., index],
            active[..., index],
            advantage[..., index],
            epsilon_clip,
        )
        for index, branch in enumerate(ACTION_BRANCH_ORDER)
    }
    return BranchPPODynamics(epsilon_clip=epsilon_clip, groups=groups)


def _ppo_group(old: Tensor, new: Tensor, mask: Tensor, epsilon_clip: float) -> RoutePPODynamicsGroup:
    selected_old = old.masked_select(mask)
    selected_new = new.masked_select(mask)
    if selected_old.numel() == 0:
        return RoutePPODynamicsGroup()
    log_ratio = selected_new - selected_old
    ratio = torch.exp(log_ratio)
    approx_kl = (ratio - 1.0) - log_ratio
    clipped = (ratio < 1.0 - epsilon_clip) | (ratio > 1.0 + epsilon_clip)
    return RoutePPODynamicsGroup(
        count=int(ratio.numel()),
        old_log_prob_mean=float(selected_old.mean().item()),
        new_log_prob_mean=float(selected_new.mean().item()),
        log_ratio_mean=float(log_ratio.mean().item()),
        ratio_mean=float(ratio.mean().item()),
        approx_kl_mean=float(approx_kl.mean().item()),
        clip_indicator_fraction=float(clipped.float().mean().item()),
        clip_fraction=float(clipped.float().mean().item()),
    )


def compute_route_only_ppo_dynamics(
    old_route_log_prob: Tensor | None,
    new_route_log_prob: Tensor | None,
    route_active_mask: Tensor,
    category_masks: Mapping[str, Tensor],
    epsilon_clip: float,
) -> RoutePPODynamics | None:
    if old_route_log_prob is None or new_route_log_prob is None:
        return None
    active = route_active_mask.detach().bool()
    old = old_route_log_prob.detach().float()
    new = new_route_log_prob.detach().float()
    finite = torch.isfinite(old) & torch.isfinite(new)
    active = active & finite
    groups = {
        category: _ppo_group(old, new, active & category_masks[category].detach().bool(), epsilon_clip)
        for category in ROUTE_ACTION_CATEGORIES
    }
    return RoutePPODynamics(
        epsilon_clip=epsilon_clip,
        overall=_ppo_group(old, new, active, epsilon_clip),
        groups=groups,
    )


def measure_shared_trunk_gradients(actor_or_parameters: Any) -> GradientMeasurement:
    if hasattr(actor_or_parameters, "named_parameters"):
        parameters = [
            parameter
            for name, parameter in actor_or_parameters.named_parameters()
            if name.startswith(("self_encoder", "public_encoder", "graph_encoder", "gru"))
        ]
    else:
        parameters = list(actor_or_parameters or ())
    gradients = [
        parameter.grad.detach().reshape(-1)
        for parameter in parameters
        if getattr(parameter, "grad", None) is not None
    ]
    if not gradients:
        return GradientMeasurement("none", None)
    joined = torch.cat(gradients)
    if not torch.isfinite(joined).all():
        return GradientMeasurement("nonfinite", None)
    norm = float(torch.linalg.vector_norm(joined).cpu().item())
    return GradientMeasurement(
        "zero" if norm == 0.0 else "finite_nonzero",
        norm,
        mean=float(joined.mean().cpu().item()),
        signed_sum=float(joined.sum().cpu().item()),
    )


@dataclass(frozen=True)
class RouteOutcomeAssociation:
    episode_index: int
    task_id: object
    source_uav: object
    category: str
    selected_destination_uav: object
    route_slot: int | None
    tx_start_slot: int | None
    tx_complete_slot: int | None
    cpu_service_start_slot: int | None
    outcome: str
    completion_slot: int | None
    end_to_end_latency_slots: int | None
    end_to_end_latency_s: float | None
    immediate_reward: float | None = None
    workload_penalty_component: float | None = None
    completion_component: float | None = None
    expiration_component: float | None = None
    energy_component: float | None = None
    reward_component_scope: str = "team_level"
    route_decision_sequence: int = 1
    route_decision_slot: int | None = None
    selected_route_action_index: int | None = None
    terminal_slot: int | None = None
    terminal_age_slots: int | None = None

    def record(self) -> dict[str, object]:
        return {
            f"route_outcome_{item.name}": getattr(self, item.name)
            for item in fields(self)
        }


def _route_items(value: Any) -> list[Mapping[str, Any]]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        return [value]
    if isinstance(value, (list, tuple)):
        return [item for item in value if isinstance(item, Mapping)]
    return []


def _route_category(value: object) -> str:
    if type(value) is str and value == "local":
        return "local"
    if isinstance(value, int) and not isinstance(value, bool):
        return "remote"
    return "defer"


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


@dataclass
class _RouteDecisionState:
    episode_index: int
    task_id: object
    decision_sequence: int
    source_uav: object = None
    category: str = "defer"
    destination: object = None
    route_slot: int | None = None
    route_decision_slot: int | None = None
    selected_route_action_index: int | None = None
    tx_start_slot: int | None = None
    tx_complete_slot: int | None = None
    cpu_service_start_slot: int | None = None
    completion_slot: int | None = None
    terminal_slot: int | None = None
    terminal_age_slots: int | None = None
    outcome: str | None = None
    arrival_slot: int | None = None
    tx_bits: float = 0.0
    reward: dict[str, float | None] = field(default_factory=dict)


class RouteOutcomeTracker:
    """Join route decisions to later service/terminal snapshots without changing them."""

    def __init__(self, slot_duration_s: float | None = None) -> None:
        self.slot_duration_s = slot_duration_s
        self._states: dict[tuple[int, object, int], _RouteDecisionState] = {}
        self._decision_keys: list[tuple[int, object, int]] = []
        self._task_decision_keys: dict[tuple[int, object], list[tuple[int, object, int]]] = {}

    @staticmethod
    def _reward_terms(info: Mapping[str, Any]) -> dict[str, float | None]:
        raw = info.get("reward", {})
        if not isinstance(raw, Mapping):
            return {}
        return {
            "immediate_reward": _safe_float(raw.get("reward")),
            "workload_penalty_component": _safe_float(raw.get(
                "workload_penalty", raw.get("workload_penalty_component")
            )),
            "completion_component": _safe_float(raw.get("completion_component")),
            "expiration_component": _safe_float(raw.get(
                "expiration_penalty", raw.get("expiration_component")
            )),
            "energy_component": _safe_float(raw.get(
                "energy_penalty", raw.get("energy_component")
            )),
        }

    def _task_states(self, task_key: tuple[int, object]) -> list[_RouteDecisionState]:
        return [self._states[key] for key in self._task_decision_keys.get(task_key, ())]

    @staticmethod
    def _optional_int(value: object) -> int | None:
        return int(value) if isinstance(value, int) and not isinstance(value, bool) else None

    @staticmethod
    def _normalized_outcome(value: object, default: str) -> str:
        raw = str(value if value is not None else default)
        if raw in {"done", "completed"}:
            return "completed"
        if raw in {"expired", "truncated"}:
            return raw
        return raw

    @staticmethod
    def _set_terminal(
        state: _RouteDecisionState,
        outcome: str,
        item: Mapping[str, Any],
        observed_slot: int | None,
    ) -> None:
        state.outcome = outcome
        state.arrival_slot = item.get("arrival_slot", state.arrival_slot)
        if outcome in {"expired", "truncated"}:
            state.completion_slot = None
            state.terminal_slot = observed_slot
        else:
            completion_slot = item.get("completion_slot", observed_slot)
            state.completion_slot = completion_slot
            state.terminal_slot = completion_slot
        if state.arrival_slot is not None and state.terminal_slot is not None:
            state.terminal_age_slots = int(state.terminal_slot) - int(state.arrival_slot)

    def observe_step(
        self,
        episode_index: int,
        info: Mapping[str, Any],
        *,
        route_action_indices: Mapping[int, int] | None = None,
    ) -> None:
        if not isinstance(info, Mapping):
            return
        try:
            slot = int(info.get("slot", info.get("slot_index")))
        except (TypeError, ValueError):
            slot = None
        reward = self._reward_terms(info)
        service = info.get("service", {})
        if not isinstance(service, Mapping):
            service = {}
        for route in _route_items(service.get("routing", service.get("routing_records"))):
            task_id = route.get("task_id")
            if task_id is None:
                continue
            proposal_record = route.get("proposal")
            if isinstance(proposal_record, Mapping):
                proposed_route = proposal_record.get(
                    "destination", proposal_record.get("route")
                )
            else:
                proposed_route = proposal_record
            destination = route.get("destination", proposed_route)
            if (
                (type(proposed_route) is str and proposed_route in {"local", "defer", "idle"})
                or (isinstance(proposed_route, int) and not isinstance(proposed_route, bool))
            ):
                category_value = proposed_route
            else:
                category_value = destination
            task_key = (episode_index, task_id)
            decision_keys = self._task_decision_keys.setdefault(task_key, [])
            decision_sequence = len(decision_keys) + 1
            key = (episode_index, task_id, decision_sequence)
            state = _RouteDecisionState(
                episode_index=episode_index,
                task_id=task_id,
                decision_sequence=decision_sequence,
                source_uav=route.get("uav_id", route.get("source_uav")),
                category=_route_category(category_value),
                destination=destination if _route_category(category_value) == "remote" else None,
                route_slot=slot,
                route_decision_slot=slot,
                selected_route_action_index=self._optional_int(route.get(
                    "selected_route_action_index",
                    route.get(
                        "selected_action_index",
                        route.get("action_index", (route_action_indices or {}).get(
                            route.get("uav_id", route.get("source_uav"))
                        )),
                    ),
                )),
                reward=dict(reward),
            )
            self._states[key] = state
            decision_keys.append(key)
            self._decision_keys.append(key)
        raw_task_services = service.get("task_services")
        if raw_task_services is None:
            raw_task_services = service.get("links")
        task_service_items: list[Mapping[str, Any]] = []
        for link_or_item in _route_items(raw_task_services):
            nested = link_or_item.get("task_services")
            task_service_items.extend(
                _route_items(nested) if nested is not None else [link_or_item]
            )
        for item in task_service_items:
            task_key = (episode_index, item.get("task_id"))
            states = self._task_states(task_key)
            if not states:
                continue
            amount = _safe_float(item.get(
                "amount", item.get("served_bits", item.get("bits"))
            )) or 0.0
            data_bits = _safe_float(item.get("data_bits", item.get("required_bits")))
            for state in states:
                if state.tx_start_slot is None:
                    state.tx_start_slot = slot
                state.tx_bits += amount
                if data_bits is not None and state.tx_bits >= data_bits:
                    state.tx_complete_slot = slot
        for item in _route_items(service.get("cpu_services", service.get("cpu"))):
            states = self._task_states((episode_index, item.get("task_id")))
            for state in states:
                if state.cpu_service_start_slot is None:
                    state.cpu_service_start_slot = slot
        for section, default_outcome in (
            ("settled_tasks", "completed"), ("truncated_tasks", "truncated")
        ):
            for item in _route_items(service.get(section)):
                states = self._task_states((episode_index, item.get("task_id")))
                if not states:
                    continue
                outcome = self._normalized_outcome(item.get("outcome"), default_outcome)
                for state in states:
                    self._set_terminal(state, outcome, item, slot)
                    if state.category == "remote" and state.tx_complete_slot is None:
                        cpu_entry_slot = item.get("cpu_entry_slot")
                        if isinstance(cpu_entry_slot, int) and not isinstance(cpu_entry_slot, bool):
                            state.tx_complete_slot = cpu_entry_slot
        metrics = info.get("metrics", {})
        if isinstance(metrics, Mapping):
            for item in _route_items(metrics.get("expired_tasks")):
                states = self._task_states((episode_index, item.get("task_id")))
                for state in states:
                    self._set_terminal(state, "expired", item, slot)

    def finalize(self) -> tuple[RouteOutcomeAssociation, ...]:
        result = []
        for key in sorted(self._decision_keys, key=lambda item: (item[0], str(item[1]), item[2])):
            state = self._states[key]
            outcome = state.outcome or "truncated"
            latency_slots = None
            if outcome == "completed" and state.arrival_slot is not None and state.completion_slot is not None:
                latency_slots = int(state.completion_slot) - int(state.arrival_slot)
            result.append(RouteOutcomeAssociation(
                episode_index=state.episode_index,
                task_id=state.task_id,
                source_uav=state.source_uav,
                category=state.category,
                selected_destination_uav=(
                    state.destination if state.category == "remote"
                    else state.source_uav if state.category == "local" else None
                ),
                route_slot=state.route_slot,
                tx_start_slot=state.tx_start_slot,
                tx_complete_slot=state.tx_complete_slot,
                cpu_service_start_slot=state.cpu_service_start_slot,
                outcome=outcome,
                completion_slot=state.completion_slot,
                end_to_end_latency_slots=latency_slots,
                end_to_end_latency_s=(
                    latency_slots * self.slot_duration_s
                    if latency_slots is not None and self.slot_duration_s is not None
                    else None
                ),
                immediate_reward=state.reward.get("immediate_reward"),
                workload_penalty_component=state.reward.get("workload_penalty_component"),
                completion_component=state.reward.get("completion_component"),
                expiration_component=state.reward.get("expiration_component"),
                energy_component=state.reward.get("energy_component"),
                reward_component_scope="team_level",
                route_decision_sequence=state.decision_sequence,
                route_decision_slot=state.route_decision_slot,
                selected_route_action_index=state.selected_route_action_index,
                terminal_slot=state.terminal_slot,
                terminal_age_slots=state.terminal_age_slots,
            ))
        return tuple(result)


@dataclass(frozen=True)
class TrajectoryPreStepCapture:
    """Actor-safe route context captured before one environment transition."""

    episode_id: int
    route_step: int
    rollout_index: int
    route_candidates: Mapping[int, Mapping[str, Any]]
    workload_snapshots: Mapping[int, TaskWorkloadSnapshot]

    def __post_init__(self) -> None:
        for name in ("episode_id", "route_step", "rollout_index"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise RouteTelemetryError(f"{name} must be a non-negative integer")


@dataclass
class _TrajectoryTaskLedger:
    episode_id: int
    task_id: int
    source_uav: int
    initial_task_snapshot: dict[str, Any]
    latest_task_snapshot: dict[str, Any]
    route_event_key: tuple[int, int, int] | None = None
    route_event: dict[str, Any] | None = None
    cumulative_workload_raw: float = 0.0
    cumulative_workload_penalty: float = 0.0
    workload_residual_raw: float = 0.0
    workload_residual_penalty: float = 0.0
    completion_component: float = 0.0
    expiration_penalty: float = 0.0
    cumulative_tx_bits: float = 0.0
    cumulative_cpu_cycles: float = 0.0
    allocated_task_tx_energy: float = 0.0
    exact_task_cpu_energy: float = 0.0
    first_tx_step: int | None = None
    last_tx_step: int | None = None
    first_cpu_step: int | None = None
    last_cpu_step: int | None = None
    terminal_emitted: bool = False


class TrajectoryCreditTracker:
    """Fresh-run-only task/route/reward/PPO sidecar with no control influence."""

    def __init__(
        self,
        config: RunConfig,
        emit: Callable[[Mapping[str, Any]], None],
    ) -> None:
        if not isinstance(config, RunConfig):
            raise TypeError("config must be a RunConfig")
        if not config.training.mappo.trajectory_credit_telemetry_enabled:
            raise RouteTelemetryError("trajectory credit tracker requires enabled telemetry")
        if not callable(emit):
            raise TypeError("emit must be callable")
        self.config = config
        self._emit = emit
        self._tasks: dict[tuple[int, int], _TrajectoryTaskLedger] = {}
        self._route_events: dict[tuple[int, int, int], dict[str, Any]] = {}
        self._route_by_transition_agent: dict[tuple[int, int], tuple[int, int, int]] = {}
        self._terminal_task_keys: set[tuple[int, int]] = set()
        self._credit_route_keys: set[tuple[int, int, int]] = set()
        self._event_counts = {name: 0 for name in TRAJECTORY_CREDIT_EVENT_TYPES}
        self._team_workload_raw = 0.0
        self._team_workload_penalty = 0.0
        self._team_completion_component = 0.0
        self._team_expiration_penalty = 0.0
        self._workload_numeric_residual_raw = 0.0
        self._workload_numeric_residual_penalty = 0.0
        self._measured_tx_energy = 0.0
        self._allocated_tx_energy = 0.0
        self._unattributed_tx_energy = 0.0
        self._measured_cpu_energy = 0.0
        self._attributed_cpu_energy = 0.0
        self._unattributed_cpu_energy = 0.0
        self._production_measured_energy = 0.0
        self._energy_numeric_residual = 0.0

    def _common(
        self,
        event_type: str,
        episode_id: int,
        task_id: int,
        route_event_key: tuple[int, int, int] | None,
    ) -> dict[str, Any]:
        return {
            "schema_version": TRAJECTORY_CREDIT_SCHEMA_VERSION,
            "event_type": event_type,
            "run_id": self.config.run_id,
            "config_hash": self.config.config_hash,
            "git_commit": self.config.git_commit,
            "episode_id": episode_id,
            "task_id": task_id,
            "route_event_key": None if route_event_key is None else list(route_event_key),
        }

    def _write(self, record: Mapping[str, Any]) -> None:
        event_type = record.get("event_type")
        if event_type not in TRAJECTORY_CREDIT_EVENT_TYPES:
            raise RouteTelemetryError("unknown trajectory-credit event type")
        self._emit(record)
        self._event_counts[str(event_type)] += 1

    def _ledger_for_task(self, episode_id: int, task: Task) -> _TrajectoryTaskLedger:
        key = (episode_id, task.task_id)
        snapshot = task.snapshot()
        ledger = self._tasks.get(key)
        if ledger is None:
            ledger = _TrajectoryTaskLedger(
                episode_id=episode_id,
                task_id=task.task_id,
                source_uav=task.source_uav,
                initial_task_snapshot=dict(snapshot),
                latest_task_snapshot=dict(snapshot),
            )
            self._tasks[key] = ledger
        else:
            ledger.latest_task_snapshot = dict(snapshot)
        return ledger

    @staticmethod
    def _selected_link_proxy(observation: Any, destination: int) -> dict[str, Any]:
        edge = observation.edge_history
        quality_values = edge.historical_quality[destination]
        quality_mask = edge.quality_valid_mask[destination]
        valid_quality = [
            float(value)
            for value, valid in zip(quality_values, quality_mask)
            if bool(valid)
        ]
        return {
            "candidate_neighbor": bool(observation.candidate_neighbor_mask[destination]),
            "visible": bool(edge.visible_mask[destination]),
            "estimated_distance_m": float(edge.estimated_distance_m[destination]),
            "historical_quality_mean": (
                math.fsum(valid_quality) / len(valid_quality) if valid_quality else None
            ),
            "historical_quality_valid_count": len(valid_quality),
            "csi_valid": bool(edge.csi_valid_mask[destination]),
            "csi_aoi_slots": (
                int(edge.csi_aoi_slots[destination])
                if bool(edge.csi_valid_mask[destination]) else None
            ),
            "message_aoi_slots": (
                int(edge.message_aoi_slots[destination])
                if bool(edge.message_aoi_valid_mask[destination]) else None
            ),
            "message_aoi_valid": bool(edge.message_aoi_valid_mask[destination]),
            "last_effective_rate_bps": (
                float(edge.last_effective_rate_bps[destination])
                if bool(edge.last_rate_valid_mask[destination]) else None
            ),
            "last_rate_valid": bool(edge.last_rate_valid_mask[destination]),
            "outage_rate": (
                float(edge.outage_rate[destination])
                if bool(edge.outage_valid_mask[destination]) else None
            ),
            "outage_valid": bool(edge.outage_valid_mask[destination]),
        }

    @staticmethod
    def _helper_queue_proxy(observation: Any, destination: int) -> dict[str, Any]:
        public = observation.neighbor_public
        return {
            "valid": bool(public.valid_mask[destination]),
            "task_count": int(public.cpu_load_task_count[destination]),
            "remaining_cycles": float(public.cpu_load_remaining_cycles[destination]),
            "message_aoi_slots": (
                int(public.message_aoi_slots[destination])
                if bool(public.message_aoi_valid_mask[destination]) else None
            ),
            "message_aoi_valid": bool(public.message_aoi_valid_mask[destination]),
        }

    def capture_pre_step(
        self,
        *,
        episode_id: int,
        rollout_index: int,
        observations: Sequence[Any],
        proposals: Sequence[ActionProposal],
        route_action_indices: Mapping[int, int],
        environment: Any,
    ) -> TrajectoryPreStepCapture:
        if not observations:
            raise RouteTelemetryError("trajectory capture requires observations")
        route_step = int(observations[0].slot)
        lifecycle = getattr(environment, "lifecycle", None)
        tasks = getattr(lifecycle, "tasks", None)
        if not isinstance(tasks, Mapping):
            raise RouteTelemetryError("telemetry requires environment lifecycle tasks")
        for task in tasks.values():
            self._ledger_for_task(episode_id, task)
        proposal_by_uav = {proposal.uav_id: proposal for proposal in proposals}
        candidates: dict[int, Mapping[str, Any]] = {}
        workload_snapshots: dict[int, TaskWorkloadSnapshot] = {}
        for observation in observations:
            proposal = proposal_by_uav.get(observation.uav_id)
            if proposal is None:
                raise RouteTelemetryError("missing proposal for trajectory capture")
            destination = proposal.route
            is_remote = isinstance(destination, int) and not isinstance(destination, bool)
            if destination != "local" and not is_remote:
                continue
            queue = observation.private_queues.unbound
            if not queue.head_valid_mask:
                continue
            task_id = int(queue.head_task_id)
            task = tasks.get(task_id)
            if not isinstance(task, Task):
                raise RouteTelemetryError("route head task is absent from lifecycle")
            route_domain = tuple(observation.action_masks.route_domain)
            route_mask = tuple(bool(value) for value in observation.action_masks.route_mask)
            legal_remote = [
                value
                for value, legal in zip(route_domain, route_mask)
                if legal and isinstance(value, int) and not isinstance(value, bool)
            ]
            selected_destination = int(destination) if is_remote else observation.uav_id
            candidates[observation.uav_id] = {
                "task_id": task_id,
                "source_uav": observation.uav_id,
                "route_category": "remote" if is_remote else "local",
                "selected_destination_uav": selected_destination,
                "route_action_index": int(route_action_indices[observation.uav_id]),
                "task_snapshot": task.snapshot(),
                "source_queue_proxy": queue.snapshot(),
                "legal_remote_destinations": list(legal_remote),
                "legal_route_mask": list(route_mask),
                "selected_link_proxy_status": "observed" if is_remote else "not_applicable_local",
                "selected_link_proxy": (
                    self._selected_link_proxy(observation, selected_destination)
                    if is_remote else None
                ),
                "helper_queue_proxy_status": "observed" if is_remote else "not_applicable_local",
                "helper_queue_proxy": (
                    self._helper_queue_proxy(observation, selected_destination)
                    if is_remote else None
                ),
            }
            workload_snapshots[task_id] = TaskWorkloadSnapshot.from_task(task)
        return TrajectoryPreStepCapture(
            episode_id=episode_id,
            route_step=route_step,
            rollout_index=rollout_index,
            route_candidates=candidates,
            workload_snapshots=workload_snapshots,
        )

    @staticmethod
    def _mapping_items(value: Any) -> tuple[Mapping[str, Any], ...]:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            return ()
        return tuple(item for item in value if isinstance(item, Mapping))

    def _observe_routes(
        self,
        capture: TrajectoryPreStepCapture,
        tasks: Mapping[int, Task],
        service: Mapping[str, Any],
    ) -> None:
        for route in self._mapping_items(service.get("routing")):
            if route.get("applied") is not True:
                continue
            source_uav = route.get("uav_id", route.get("source_uav"))
            if isinstance(source_uav, bool) or not isinstance(source_uav, int):
                raise RouteTelemetryError("applied route lacks source_uav")
            candidate = capture.route_candidates.get(source_uav)
            if candidate is None:
                raise RouteTelemetryError("applied route lacks pre-step candidate")
            task_id = route.get("task_id")
            if task_id != candidate["task_id"]:
                raise RouteTelemetryError("applied route task differs from pre-step head")
            task = tasks.get(int(task_id))
            if not isinstance(task, Task):
                raise RouteTelemetryError("applied route task is absent after step")
            key = (capture.episode_id, int(task_id), capture.route_step)
            ledger = self._ledger_for_task(capture.episode_id, task)
            if ledger.route_event_key is not None or key in self._route_events:
                raise RouteTelemetryError("task produced more than one formal route event")
            task_before = candidate["task_snapshot"]
            task_after = task.snapshot()
            event = self._common("route", capture.episode_id, int(task_id), key)
            event.update({
                "route_step": capture.route_step,
                "rollout_index": capture.rollout_index,
                "source_uav": source_uav,
                "route_category": candidate["route_category"],
                "selected_destination_uav": candidate["selected_destination_uav"],
                "route_action_index": candidate["route_action_index"],
                "route_status": "applied",
                "task_arrival_slot": task_before["arrival_slot"],
                "task_deadline_slot": task_before["deadline_slot"],
                "task_data_bits_initial": task_before["data_bits"],
                "task_cpu_cycles_initial": task_before["cpu_cycles"],
                "remaining_bits_at_route": task_before["remaining_bits"],
                "remaining_cycles_at_route": task_before["remaining_cycles"],
                "deadline_slack_slots_at_route": (
                    int(task_before["deadline_slot"]) - capture.route_step + 1
                ),
                "binding_slot": task_after["binding_slot"],
                "service_eligible_slot": task_after["service_eligible_slot"],
                "task_status_after_route": task_after["status"],
                "source_queue_proxy": candidate["source_queue_proxy"],
                "legal_remote_destinations": candidate["legal_remote_destinations"],
                "legal_route_mask": candidate["legal_route_mask"],
                "selected_link_proxy_status": candidate["selected_link_proxy_status"],
                "selected_link_proxy": candidate["selected_link_proxy"],
                "helper_queue_proxy_status": candidate["helper_queue_proxy_status"],
                "helper_queue_proxy": candidate["helper_queue_proxy"],
                "capture_source": "pre_step_observation_plus_post_step_routing",
            })
            ledger.route_event_key = key
            ledger.route_event = dict(event)
            self._route_events[key] = dict(event)
            transition_agent = (capture.rollout_index, source_uav)
            if transition_agent in self._route_by_transition_agent:
                raise RouteTelemetryError("transition agent produced duplicate route identity")
            self._route_by_transition_agent[transition_agent] = key
            self._write(event)

    def _observe_workload(
        self,
        capture: TrajectoryPreStepCapture,
        environment: Any,
        info: Mapping[str, Any],
        tasks: Mapping[int, Task],
        service: Mapping[str, Any],
    ) -> None:
        calculator = getattr(environment, "reward_calculator", None)
        primitive = getattr(calculator, "_urgent_workload_for_task", None)
        if not callable(primitive):
            raise RouteTelemetryError("production workload primitive is unavailable")
        reward = info.get("reward")
        if not isinstance(reward, Mapping):
            raise RouteTelemetryError("trajectory ledger requires reward terms")
        team_raw = _safe_float(reward.get("urgent_workload_s"))
        team_penalty = _safe_float(reward.get("workload_penalty"))
        team_completion = _safe_float(reward.get("completion_component"))
        team_expiration = _safe_float(reward.get("expiration_penalty"))
        if any(
            value is None
            for value in (
                team_raw,
                team_penalty,
                team_completion,
                team_expiration,
            )
        ):
            raise RouteTelemetryError("production workload terms are unavailable")
        arrival = info.get("arrival")
        arrival_items = (
            self._mapping_items(arrival.get("tasks"))
            if isinstance(arrival, Mapping) else ()
        )
        arrival_ids = {item.get("task_id") for item in arrival_items}
        truncated_ids = {
            item.get("task_id")
            for item in self._mapping_items(service.get("truncated_tasks"))
        }
        route_pre_mode = (
            WorkloadTimingMode(self.config.environment.workload_timing_mode)
            is WorkloadTimingMode.ROUTE_SLOT_PRE_ROUTE
        )
        coefficient = (
            calculator.weights.workload
            / calculator.references.workload_reference_s
        )
        task_raw_parts: list[float] = []
        task_penalty_parts: list[float] = []
        for task_id in sorted(tasks):
            task = tasks[task_id]
            if task_id in arrival_ids:
                continue
            if task.outcome is not TaskOutcome.NONE and task_id not in truncated_ids:
                continue
            ledger = self._ledger_for_task(capture.episode_id, task)
            snapshot = (
                capture.workload_snapshots.get(task_id)
                if route_pre_mode
                and ledger.route_event_key
                == (capture.episode_id, task_id, capture.route_step)
                else None
            )
            raw = float(primitive(task, capture.route_step, snapshot))
            penalty = coefficient * raw
            task_raw_parts.append(raw)
            task_penalty_parts.append(penalty)
            if ledger.route_event_key is None:
                ledger.workload_residual_raw += raw
                ledger.workload_residual_penalty += penalty
            else:
                ledger.cumulative_workload_raw += raw
                ledger.cumulative_workload_penalty += penalty
        task_raw = math.fsum(task_raw_parts)
        task_penalty = math.fsum(task_penalty_parts)
        self._team_workload_raw += team_raw
        self._team_workload_penalty += team_penalty
        self._team_completion_component += team_completion
        self._team_expiration_penalty += team_expiration
        self._workload_numeric_residual_raw += team_raw - task_raw
        self._workload_numeric_residual_penalty += team_penalty - task_penalty

    def _observe_energy_and_service(
        self,
        capture: TrajectoryPreStepCapture,
        info: Mapping[str, Any],
        tasks: Mapping[int, Task],
        service: Mapping[str, Any],
    ) -> None:
        slot_tx = 0.0
        slot_cpu = 0.0
        for link in self._mapping_items(service.get("links")):
            energy = _safe_float(link.get("transmit_energy_j"))
            if energy is None or energy < 0.0:
                raise RouteTelemetryError("link TX energy must be finite and non-negative")
            slot_tx += energy
            positive: list[tuple[int, float]] = []
            for item in self._mapping_items(link.get("task_services")):
                amount = _safe_float(item.get("amount"))
                task_id = item.get("task_id")
                if amount is None or amount < 0.0:
                    raise RouteTelemetryError("TX task service must be non-negative")
                if amount <= 0.0:
                    continue
                task = tasks.get(int(task_id))
                if not isinstance(task, Task):
                    raise RouteTelemetryError("TX service task is absent")
                positive.append((int(task_id), amount))
                ledger = self._ledger_for_task(capture.episode_id, task)
                ledger.cumulative_tx_bits += amount
                if ledger.first_tx_step is None:
                    ledger.first_tx_step = capture.route_step
                ledger.last_tx_step = capture.route_step
            total_bits = math.fsum(amount for _task_id, amount in positive)
            allocated_parts: list[float] = []
            if total_bits > 0.0:
                for task_id, amount in positive:
                    allocation = energy * amount / total_bits
                    allocated_parts.append(allocation)
                    self._tasks[(capture.episode_id, task_id)].allocated_task_tx_energy += allocation
            allocated = math.fsum(allocated_parts)
            self._measured_tx_energy += energy
            self._allocated_tx_energy += allocated
            self._unattributed_tx_energy += energy - allocated
        for item in self._mapping_items(service.get("cpu")):
            energy = _safe_float(item.get("cpu_energy_j"))
            cycles = _safe_float(item.get("service_cycles"))
            if energy is None or energy < 0.0 or cycles is None or cycles < 0.0:
                raise RouteTelemetryError("CPU service telemetry must be non-negative")
            slot_cpu += energy
            task_id = item.get("task_id")
            self._measured_cpu_energy += energy
            if task_id is None:
                self._unattributed_cpu_energy += energy
                continue
            task = tasks.get(int(task_id))
            if not isinstance(task, Task):
                raise RouteTelemetryError("CPU service task is absent")
            ledger = self._ledger_for_task(capture.episode_id, task)
            ledger.exact_task_cpu_energy += energy
            ledger.cumulative_cpu_cycles += cycles
            self._attributed_cpu_energy += energy
            if cycles > 0.0:
                if ledger.first_cpu_step is None:
                    ledger.first_cpu_step = capture.route_step
                ledger.last_cpu_step = capture.route_step
        reward = info.get("reward")
        production_energy = (
            _safe_float(reward.get("actual_energy_j"))
            if isinstance(reward, Mapping) else None
        )
        if production_energy is None:
            raise RouteTelemetryError("production energy term is unavailable")
        self._production_measured_energy += production_energy
        self._energy_numeric_residual += production_energy - slot_tx - slot_cpu

    def _workload_ledger_snapshot(self) -> dict[str, Any]:
        routed_raw = math.fsum(
            item.cumulative_workload_raw for item in self._tasks.values()
        )
        routed_penalty = math.fsum(
            item.cumulative_workload_penalty for item in self._tasks.values()
        )
        residual_raw = math.fsum(
            item.workload_residual_raw for item in self._tasks.values()
        )
        residual_penalty = math.fsum(
            item.workload_residual_penalty for item in self._tasks.values()
        )
        raw_conservation = self._team_workload_raw - math.fsum((
            routed_raw,
            residual_raw,
            self._workload_numeric_residual_raw,
        ))
        penalty_conservation = self._team_workload_penalty - math.fsum((
            routed_penalty,
            residual_penalty,
            self._workload_numeric_residual_penalty,
        ))
        return {
            "cumulative_team_workload_raw": self._team_workload_raw,
            "cumulative_routed_task_workload_raw": routed_raw,
            "cumulative_workload_residual_raw": residual_raw,
            "cumulative_numeric_residual_raw": self._workload_numeric_residual_raw,
            "raw_conservation_residual": raw_conservation,
            "cumulative_team_workload_penalty": self._team_workload_penalty,
            "cumulative_routed_task_workload_penalty": routed_penalty,
            "cumulative_workload_residual_penalty": residual_penalty,
            "cumulative_numeric_residual_penalty": self._workload_numeric_residual_penalty,
            "penalty_conservation_residual": penalty_conservation,
            "status": (
                "pass"
                if abs(raw_conservation) <= TRAJECTORY_LEDGER_TOLERANCE
                and abs(penalty_conservation) <= TRAJECTORY_LEDGER_TOLERANCE
                else "residual"
            ),
        }

    def _energy_ledger_snapshot(self) -> dict[str, Any]:
        tx_residual = self._measured_tx_energy - math.fsum((
            self._allocated_tx_energy,
            self._unattributed_tx_energy,
        ))
        cpu_residual = self._measured_cpu_energy - math.fsum((
            self._attributed_cpu_energy,
            self._unattributed_cpu_energy,
        ))
        production_residual = self._production_measured_energy - math.fsum((
            self._measured_tx_energy,
            self._measured_cpu_energy,
            self._energy_numeric_residual,
        ))
        return {
            "tx_energy_attribution_method": "bits_pro_rata",
            "tx_energy_attribution_status": "diagnostic_allocation",
            "production_measured_tx_energy": self._measured_tx_energy,
            "allocated_task_tx_energy": self._allocated_tx_energy,
            "unattributed_tx_energy": self._unattributed_tx_energy,
            "tx_conservation_residual": tx_residual,
            "cpu_energy_attribution_status": "exact_task_id",
            "production_measured_cpu_energy": self._measured_cpu_energy,
            "attributed_task_cpu_energy": self._attributed_cpu_energy,
            "unattributed_cpu_energy": self._unattributed_cpu_energy,
            "cpu_conservation_residual": cpu_residual,
            "production_measured_total_energy": self._production_measured_energy,
            "production_energy_numeric_residual": self._energy_numeric_residual,
            "total_conservation_residual": production_residual,
            "status": (
                "pass"
                if max(abs(tx_residual), abs(cpu_residual), abs(production_residual))
                <= TRAJECTORY_LEDGER_TOLERANCE
                else "residual"
            ),
        }

    def _reward_ledger_snapshot(self) -> dict[str, Any]:
        task_completion = math.fsum(
            item.completion_component for item in self._tasks.values()
        )
        task_expiration = math.fsum(
            item.expiration_penalty for item in self._tasks.values()
        )
        completion_residual = self._team_completion_component - task_completion
        expiration_residual = self._team_expiration_penalty - task_expiration
        return {
            "cumulative_team_completion_component": self._team_completion_component,
            "cumulative_task_completion_component": task_completion,
            "completion_conservation_residual": completion_residual,
            "cumulative_team_expiration_penalty": self._team_expiration_penalty,
            "cumulative_task_expiration_penalty": task_expiration,
            "expiration_conservation_residual": expiration_residual,
            "status": (
                "pass"
                if max(abs(completion_residual), abs(expiration_residual))
                <= TRAJECTORY_LEDGER_TOLERANCE
                else "residual"
            ),
        }

    def _emit_terminal(
        self,
        *,
        ledger: _TrajectoryTaskLedger,
        task_snapshot: Mapping[str, Any],
        observed_transition_step: int,
        terminal_kind: str,
        terminal_step: int | None,
        outcome: str | None,
        calculator: Any,
    ) -> None:
        task_key = (ledger.episode_id, ledger.task_id)
        if ledger.terminal_emitted or task_key in self._terminal_task_keys:
            raise RouteTelemetryError("task produced more than one terminal event")
        completion_component = (
            calculator.weights.completion / calculator.references.task_count_reference
            if outcome == TaskOutcome.DONE.value else 0.0
        )
        expiration_penalty = (
            calculator.weights.expiration / calculator.references.task_count_reference
            if outcome == TaskOutcome.EXPIRED.value else 0.0
        )
        energy_scale = (
            calculator.weights.energy / calculator.references.active_energy_reference_j
        )
        route = ledger.route_event or {}
        ledger.completion_component = completion_component
        ledger.expiration_penalty = expiration_penalty
        completion_slot = task_snapshot.get("completion_slot")
        latency_slots = (
            int(completion_slot) - int(task_snapshot["arrival_slot"])
            if outcome == TaskOutcome.DONE.value and completion_slot is not None
            else None
        )
        event = self._common(
            "terminal", ledger.episode_id, ledger.task_id, ledger.route_event_key
        )
        event.update({
            "route_status": "routed" if ledger.route_event_key is not None else "never_routed",
            "terminal_kind": terminal_kind,
            "outcome": outcome,
            "observed_transition_step": observed_transition_step,
            "terminal_step": terminal_step,
            "source_uav": ledger.source_uav,
            "destination_uav": task_snapshot.get("destination"),
            "route_category": route.get("route_category"),
            "route_step": route.get("route_step"),
            "task_arrival_slot": task_snapshot.get("arrival_slot"),
            "task_deadline_slot": task_snapshot.get("deadline_slot"),
            "binding_slot": task_snapshot.get("binding_slot"),
            "service_eligible_slot": task_snapshot.get("service_eligible_slot"),
            "cpu_entry_slot": task_snapshot.get("cpu_entry_slot"),
            "completion_slot": completion_slot,
            "e2e_delay_slots": latency_slots,
            "e2e_delay_s": (
                latency_slots * self.config.environment.slot_duration_s
                if latency_slots is not None else None
            ),
            "first_tx_step": ledger.first_tx_step,
            "last_tx_step": ledger.last_tx_step,
            "tx_complete_step": (
                task_snapshot.get("cpu_entry_slot")
                if route.get("route_category") == "remote" else None
            ),
            "first_cpu_step": ledger.first_cpu_step,
            "last_cpu_step": ledger.last_cpu_step,
            "final_remaining_bits": task_snapshot.get("remaining_bits"),
            "final_remaining_cycles": task_snapshot.get("remaining_cycles"),
            "cumulative_tx_bits": ledger.cumulative_tx_bits,
            "cumulative_cpu_cycles": ledger.cumulative_cpu_cycles,
            "reward_decomposition": {
                "completion_component": completion_component,
                "completion_component_signed": completion_component,
                "expiration_penalty": expiration_penalty,
                "expiration_component_signed": -expiration_penalty,
                "cumulative_workload_raw": ledger.cumulative_workload_raw,
                "cumulative_workload_penalty": ledger.cumulative_workload_penalty,
                "workload_residual_raw": ledger.workload_residual_raw,
                "workload_residual_penalty": ledger.workload_residual_penalty,
                "workload_attribution_status": (
                    "task_decomposed_with_pre_route_residual"
                    if ledger.route_event_key is not None else "unrouted_residual"
                ),
                "exact_task_cpu_energy": ledger.exact_task_cpu_energy,
                "cpu_energy_penalty": energy_scale * ledger.exact_task_cpu_energy,
                "allocated_task_tx_energy": ledger.allocated_task_tx_energy,
                "tx_energy_penalty_diagnostic": (
                    energy_scale * ledger.allocated_task_tx_energy
                ),
                "tx_energy_attribution_method": "bits_pro_rata",
                "tx_energy_attribution_status": "diagnostic_allocation",
            },
            "workload_ledger": self._workload_ledger_snapshot(),
            "energy_ledger": self._energy_ledger_snapshot(),
            "reward_ledger": self._reward_ledger_snapshot(),
        })
        ledger.terminal_emitted = True
        ledger.latest_task_snapshot = dict(task_snapshot)
        self._terminal_task_keys.add(task_key)
        self._write(event)

    def _observe_terminals(
        self,
        capture: TrajectoryPreStepCapture,
        environment: Any,
        info: Mapping[str, Any],
        tasks: Mapping[int, Task],
        service: Mapping[str, Any],
    ) -> None:
        calculator = environment.reward_calculator
        for section in ("settled_tasks", "truncated_tasks"):
            for task_snapshot in self._mapping_items(service.get(section)):
                task_id = int(task_snapshot["task_id"])
                task = tasks.get(task_id)
                if not isinstance(task, Task):
                    raise RouteTelemetryError("terminal task is absent from lifecycle")
                ledger = self._ledger_for_task(capture.episode_id, task)
                outcome = str(task_snapshot.get("outcome"))
                terminal_step = (
                    int(info.get("boundary_slot"))
                    if outcome == TaskOutcome.TRUNCATED.value
                    else int(task_snapshot["completion_slot"])
                    if outcome == TaskOutcome.DONE.value
                    and task_snapshot.get("completion_slot") is not None
                    else capture.route_step
                )
                self._emit_terminal(
                    ledger=ledger,
                    task_snapshot=task_snapshot,
                    observed_transition_step=capture.route_step,
                    terminal_kind="lifecycle",
                    terminal_step=terminal_step,
                    outcome=outcome,
                    calculator=calculator,
                )

    def observe_step(
        self,
        capture: TrajectoryPreStepCapture,
        environment: Any,
        info: Mapping[str, Any],
    ) -> None:
        if not isinstance(capture, TrajectoryPreStepCapture):
            raise TypeError("capture must be TrajectoryPreStepCapture")
        if not isinstance(info, Mapping) or info.get("slot") != capture.route_step:
            raise RouteTelemetryError("trajectory step info differs from pre-step capture")
        lifecycle = getattr(environment, "lifecycle", None)
        tasks = getattr(lifecycle, "tasks", None)
        if not isinstance(tasks, Mapping):
            raise RouteTelemetryError("telemetry requires environment lifecycle tasks")
        typed_tasks = {
            int(task_id): task
            for task_id, task in tasks.items()
            if isinstance(task, Task)
        }
        for task in typed_tasks.values():
            self._ledger_for_task(capture.episode_id, task)
        service = info.get("service")
        if not isinstance(service, Mapping):
            raise RouteTelemetryError("trajectory step requires service telemetry")
        self._observe_routes(capture, typed_tasks, service)
        self._observe_workload(capture, environment, info, typed_tasks, service)
        self._observe_energy_and_service(capture, info, typed_tasks, service)
        self._observe_terminals(capture, environment, info, typed_tasks, service)

    def observe_ppo_update(
        self,
        output: Any,
        *,
        update_index: int,
        policy_version_before: int,
        policy_version_after: int,
        rollout_start_index: int,
        rollout_length: int,
    ) -> None:
        epochs = getattr(output, "epoch_diagnostics", ())
        if not epochs or getattr(epochs[0], "epoch_index", None) != 0:
            raise RouteTelemetryError("canonical PPO epoch 0 telemetry is unavailable")
        route_telemetry = getattr(epochs[0], "route_telemetry", None)
        samples = (
            getattr(route_telemetry, "samples", ())
            if route_telemetry is not None else ()
        )
        expected = {
            key
            for key, event in self._route_events.items()
            if rollout_start_index
            <= int(event["rollout_index"])
            < rollout_start_index + rollout_length
        }
        emitted: set[tuple[int, int, int]] = set()
        chunk_length = self.config.training.mappo.recurrent_chunk_length_slots
        for sample in samples:
            local_index = (
                int(sample.batch_index) * chunk_length + int(sample.time_index)
            )
            global_index = rollout_start_index + local_index
            key = self._route_by_transition_agent.get(
                (global_index, int(sample.agent_index))
            )
            if key is None:
                continue
            route = self._route_events[key]
            if (
                sample.category != route["route_category"]
                or sample.selected_destination_uav
                != route["selected_destination_uav"]
                or sample.selected_route_action_index != route["route_action_index"]
            ):
                raise RouteTelemetryError("PPO route sample differs from route identity")
            if key in self._credit_route_keys or key in emitted:
                raise RouteTelemetryError("route transition produced duplicate PPO credit")
            event = self._common("credit", key[0], key[1], key)
            event.update({
                "route_step": key[2],
                "rollout_index": route["rollout_index"],
                "source_uav": route["source_uav"],
                "route_category": route["route_category"],
                "selected_destination_uav": route["selected_destination_uav"],
                "route_action_index": route["route_action_index"],
                "credit_status": "linked",
                "ppo_update_index": update_index,
                "ppo_epoch_index": 0,
                "policy_version_before": policy_version_before,
                "policy_version_after": policy_version_after,
                "advantage": sample.advantage,
                "td_residual": sample.td_residual,
                "return_target": sample.return_target,
                "old_route_log_prob": sample.old_route_log_prob,
                "new_route_log_prob": sample.new_route_log_prob,
                "credit_source": "epoch0_route_telemetry",
                "na_reason": None,
            })
            self._write(event)
            emitted.add(key)
            self._credit_route_keys.add(key)
        if emitted != expected:
            missing = sorted(expected - emitted)
            raise RouteTelemetryError(
                f"canonical PPO credit linkage is incomplete; missing={missing}"
            )

    def finalize_collection(
        self,
        *,
        environment: Any,
        episode_id: int,
        observed_transition_step: int,
        optimized_transitions: int,
    ) -> Mapping[str, Any]:
        lifecycle = getattr(environment, "lifecycle", None)
        tasks = getattr(lifecycle, "tasks", {})
        calculator = getattr(environment, "reward_calculator", None)
        if calculator is None:
            raise RouteTelemetryError("reward calculator is unavailable at finalization")
        for task in tasks.values():
            if not isinstance(task, Task):
                continue
            ledger = self._ledger_for_task(episode_id, task)
            if not ledger.terminal_emitted:
                if task.is_terminal:
                    raise RouteTelemetryError(
                        "lifecycle-terminal task lacks its terminal event"
                    )
                self._emit_terminal(
                    ledger=ledger,
                    task_snapshot=task.snapshot(),
                    observed_transition_step=observed_transition_step,
                    terminal_kind="collection_censored",
                    terminal_step=None,
                    outcome=None,
                    calculator=calculator,
                )
        for key, route in sorted(
            self._route_events.items(), key=lambda item: item[1]["rollout_index"]
        ):
            if key in self._credit_route_keys:
                continue
            if int(route["rollout_index"]) < optimized_transitions:
                raise RouteTelemetryError("optimized route lacks canonical PPO credit")
            event = self._common("credit", key[0], key[1], key)
            event.update({
                "route_step": key[2],
                "rollout_index": route["rollout_index"],
                "source_uav": route["source_uav"],
                "route_category": route["route_category"],
                "selected_destination_uav": route["selected_destination_uav"],
                "route_action_index": route["route_action_index"],
                "credit_status": "tail_not_optimized",
                "ppo_update_index": None,
                "ppo_epoch_index": None,
                "policy_version_before": None,
                "policy_version_after": None,
                "advantage": None,
                "td_residual": None,
                "return_target": None,
                "old_route_log_prob": None,
                "new_route_log_prob": None,
                "credit_source": None,
                "na_reason": "tail_not_optimized",
            })
            self._write(event)
            self._credit_route_keys.add(key)
        if set(self._route_events) != self._credit_route_keys:
            raise RouteTelemetryError("route/credit cardinality differs at finalization")
        return self.summary()

    def summary(self) -> Mapping[str, Any]:
        return {
            "schema_version": TRAJECTORY_CREDIT_SCHEMA_VERSION,
            "event_counts": dict(self._event_counts),
            "task_count": len(self._tasks),
            "route_event_count": len(self._route_events),
            "terminal_event_count": len(self._terminal_task_keys),
            "credit_event_count": len(self._credit_route_keys),
            "workload_ledger": self._workload_ledger_snapshot(),
            "energy_ledger": self._energy_ledger_snapshot(),
            "reward_ledger": self._reward_ledger_snapshot(),
        }


def aggregate_route_outcomes(
    outcomes: Iterable[RouteOutcomeAssociation],
) -> dict[str, object]:
    items = list(outcomes)
    result: dict[str, object] = {}
    for category in ROUTE_ACTION_CATEGORIES:
        subset = [item for item in items if item.category == category]
        completed = [item for item in subset if item.outcome == "completed"]
        latencies = [
            float(item.end_to_end_latency_slots)
            for item in completed if item.end_to_end_latency_slots is not None
        ]
        reward_means = {}
        for name in (
            "immediate_reward", "workload_penalty_component",
            "completion_component", "expiration_component", "energy_component"
        ):
            values = [
                float(getattr(item, name)) for item in subset
                if getattr(item, name) is not None
            ]
            reward_means[name] = sum(values) / len(values) if values else None
        result[category] = {
            "count": len(subset),
            "completed": sum(item.outcome == "completed" for item in subset),
            "expired": sum(item.outcome == "expired" for item in subset),
            "truncated": sum(item.outcome == "truncated" for item in subset),
            "completion_rate": len(completed) / len(subset) if subset else None,
            "mean_latency_slots_completed": (
                sum(latencies) / len(latencies) if latencies else None
            ),
            "valid_latency_count": len(latencies),
            "team_level_reward_means": reward_means,
        }
    return result

__all__ = [
    "BRANCH_PPO_RECORD_FIELDS",
    "BranchPPODynamics",
    "BranchPPODynamicsGroup",
    "GradientMeasurement",
    "ROUTE_GRADIENT_STATUSES",
    "ROUTE_TELEMETRY_SCHEMA_VERSION",
    "TRAJECTORY_CREDIT_EVENT_TYPES",
    "TRAJECTORY_CREDIT_SCHEMA_VERSION",
    "TRAJECTORY_LEDGER_TOLERANCE",
    "RouteHeadGradientTelemetry",
    "RouteTelemetry",
    "RouteTelemetryError",
    "collect_route_telemetry",
    "measure_route_head_gradients",
    "ROUTE_ACTION_CATEGORIES",
    "ROUTE_DISTRIBUTION_GROUPS",
    "ROUTE_PPO_RECORD_FIELDS",
    "RouteDistributionSummary",
    "RouteHeadGradientRowTelemetry",
    "RouteOutcomeAssociation",
    "RouteOutcomeTracker",
    "RoutePPODynamics",
    "RoutePPODynamicsGroup",
    "RouteSampleTelemetry",
    "TrajectoryCreditTracker",
    "TrajectoryPreStepCapture",
    "aggregate_route_outcomes",
    "compute_route_only_ppo_dynamics",
    "compute_branch_ppo_dynamics",
    "measure_route_head_gradient_rows",
    "measure_shared_trunk_gradients",
    "summarize_distribution",
]
