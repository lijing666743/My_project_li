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
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import Tensor, nn

from ..env.actions import ActionProposal
from .ca_gat_mappo_actions import (
    SequentialActionDistributionOutput,
    SequentialActionMaskBatch,
)


ROUTE_TELEMETRY_SCHEMA_VERSION = 3
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
    _advantage_distributions: Mapping[str, RouteDistributionSummary] = field(default_factory=dict, repr=False)
    _return_distributions: Mapping[str, RouteDistributionSummary] = field(default_factory=dict, repr=False)
    _td_residual_distributions: Mapping[str, RouteDistributionSummary] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.schema_version != ROUTE_TELEMETRY_SCHEMA_VERSION:
            raise RouteTelemetryError("route telemetry schema version is invalid")
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
            and item.name not in {"samples", "route_ppo_dynamics"}
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
                for suffix in ("valid_sample_count", "mean", "std", "median")
            )
            distribution_names.extend(
                f"{group}_td_residual_{suffix}"
                for suffix in ("valid_sample_count", "mean", "std", "median")
            )
        return names + tuple(ROUTE_PPO_RECORD_FIELDS) + tuple(distribution_names)

    def record(self) -> dict[str, object]:
        """Return a deterministic field-ordered flat record for CSV/JSONL."""

        result = {
            name: getattr(self, name)
            for name in self.field_names()
            if hasattr(self, name)
        }
        result["schema_version"] = self.schema_version
        result["route_telemetry_schema_version"] = ROUTE_TELEMETRY_SCHEMA_VERSION
        rows = []
        for row in self.route_head_gradient_rows or ():
            rows.append(row.record() if hasattr(row, "record") else row)
        result["route_head_gradient_rows"] = rows
        if self.route_ppo_dynamics is not None:
            result.update(self.route_ppo_dynamics.record())
        else:
            result.update({name: None for name in ROUTE_PPO_RECORD_FIELDS})
        for prefix, summaries, include_quantiles in (
            ("advantage", self._advantage_distributions, True),
            ("return_target", self._return_distributions, False),
            ("td_residual", self._td_residual_distributions, False),
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
) -> RouteTelemetry:
    """Aggregate route telemetry without changing the PPO algorithm state."""

    if not isinstance(policy, SequentialActionDistributionOutput):
        raise TypeError("policy must be SequentialActionDistributionOutput")
    if not isinstance(action_mask_batch, SequentialActionMaskBatch):
        raise TypeError("action_mask_batch must be SequentialActionMaskBatch")
    route_active = policy.active_branches["route"]
    if route_active.dtype != torch.bool or route_active.ndim != 3:
        raise RouteTelemetryError("route activity must be boolean [B,T,A]")
    batch, time, agents = route_active.shape
    if action_mask_batch.shape != (batch, time, agents):
        raise RouteTelemetryError("action-mask contracts differ from route axes")
    if tuple(advantage.shape) != (batch, time):
        raise RouteTelemetryError("advantage must have shape [B,T]")
    if tuple(return_target.shape) != (batch, time):
        raise RouteTelemetryError("return_target must have shape [B,T]")
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

        expanded_advantage = advantage.detach().to(device=device).unsqueeze(-1).expand_as(route_active)
        expanded_return = return_target.detach().to(device=device).unsqueeze(-1).expand_as(route_active)
        td_values = (
            td_residual.detach().to(device=device).unsqueeze(-1).expand_as(route_active)
            if td_residual is not None
            else None
        )
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
    "GradientMeasurement",
    "ROUTE_GRADIENT_STATUSES",
    "ROUTE_TELEMETRY_SCHEMA_VERSION",
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
    "aggregate_route_outcomes",
    "compute_route_only_ppo_dynamics",
    "measure_route_head_gradient_rows",
    "measure_shared_trunk_gradients",
    "summarize_distribution",
]
