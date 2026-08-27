"""Behavior-neutral route diagnostics for CA-GAT-MAPPO PPO updates.

The functions in this module only read already-computed policy statistics,
fixed rollout targets, action-mask snapshots, and existing gradients.  They do
not sample actions, advance random-number generators, build loss terms, run
backward, clip gradients, or step optimizers.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import math
from typing import Sequence

import torch
from torch import Tensor, nn

from ..env.actions import ActionProposal
from .ca_gat_mappo_actions import (
    SequentialActionDistributionOutput,
    SequentialActionMaskBatch,
)


ROUTE_TELEMETRY_SCHEMA_VERSION = 2
ROUTE_GRADIENT_STATUSES = ("none", "zero", "finite_nonzero")


class RouteTelemetryError(ValueError):
    """Raised when route telemetry inputs or derived invariants are invalid."""


@dataclass(frozen=True)
class GradientMeasurement:
    """One read-only gradient measurement with explicit missing semantics."""

    status: str
    norm: float | None

    def __post_init__(self) -> None:
        if self.status not in ROUTE_GRADIENT_STATUSES:
            raise RouteTelemetryError("unknown route-head gradient status")
        if self.status == "none":
            if self.norm is not None:
                raise RouteTelemetryError("missing gradient must use an NA norm")
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

    def __post_init__(self) -> None:
        if self.schema_version != ROUTE_TELEMETRY_SCHEMA_VERSION:
            raise RouteTelemetryError("route telemetry schema version is invalid")
        count_names = (
            name
            for name in self.field_names()
            if name.endswith("_count")
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
        return tuple(item.name for item in fields(cls))

    def record(self) -> dict[str, object]:
        """Return a deterministic field-ordered flat record for CSV/JSONL."""

        return {name: getattr(self, name) for name in self.field_names()}

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
    )


def measure_route_head_gradients(route_head: nn.Linear) -> RouteHeadGradientTelemetry:
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
    return RouteHeadGradientTelemetry(weight=weight, bias=bias, total=total)


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
) -> RouteTelemetry:
    """Aggregate one current-policy PPO epoch without touching training state."""

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

    route_probabilities = policy.probabilities["route"]
    route_logits = policy.raw_logits["route"]
    route_masks = policy.action_masks["route"]
    route_indices = policy.action_indices["route"]
    route_entropy = policy.branch_entropies["route"]
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
    if len(contracts) != batch * time * agents or len(flattened_proposals) != len(
        contracts
    ):
        raise RouteTelemetryError("proposal or contract count differs from route axes")

    remote_domain_rows: list[list[bool]] = []
    local_domain_rows: list[list[bool]] = []
    selected_local_rows: list[bool] = []
    selected_remote_rows: list[bool] = []
    selected_defer_rows: list[bool] = []
    for contract, proposal in zip(contracts, flattened_proposals):
        domain = contract.route_domain
        if len(domain) != dimension:
            raise RouteTelemetryError("route domain differs from policy dimension")
        remote_row = [
            isinstance(value, int) and not isinstance(value, bool) for value in domain
        ]
        local_row = [type(value) is str and value == "local" for value in domain]
        if sum(local_row) != 1:
            raise RouteTelemetryError("route domain must contain exactly one local action")
        remote_domain_rows.append(remote_row)
        local_domain_rows.append(local_row)
        selected = proposal.route
        selected_local_rows.append(type(selected) is str and selected == "local")
        selected_defer_rows.append(type(selected) is str and selected == "defer")
        selected_remote_rows.append(
            isinstance(selected, int) and not isinstance(selected, bool)
        )

    device = route_probabilities.device
    remote_domain = torch.tensor(
        remote_domain_rows, dtype=torch.bool, device=device
    ).reshape(expected_vector_shape)
    local_domain = torch.tensor(
        local_domain_rows, dtype=torch.bool, device=device
    ).reshape(expected_vector_shape)
    selected_local = torch.tensor(
        selected_local_rows, dtype=torch.bool, device=device
    ).reshape(batch, time, agents)
    selected_remote = torch.tensor(
        selected_remote_rows, dtype=torch.bool, device=device
    ).reshape(batch, time, agents)
    selected_defer = torch.tensor(
        selected_defer_rows, dtype=torch.bool, device=device
    ).reshape(batch, time, agents)

    with torch.no_grad():
        valid_actor = sequence_valid_mask.to(device=device).unsqueeze(-1).expand_as(
            route_active
        )
        active = route_active.detach() & valid_actor
        selected_partition = selected_local | selected_remote | selected_defer
        if torch.any(active & ~selected_partition):
            raise RouteTelemetryError("active route sample is not local, remote, or defer")

        legal_remote_actions = route_masks.detach() & remote_domain
        legal_remote = active & torch.any(legal_remote_actions, dim=-1)
        if torch.any(active & selected_remote & ~legal_remote):
            raise RouteTelemetryError("selected remote lacks a legal remote action")

        selected_probability = torch.gather(
            route_probabilities.detach(), -1, route_indices.detach().unsqueeze(-1)
        ).squeeze(-1)
        local_probability = (
            route_probabilities.detach() * local_domain.to(route_probabilities.dtype)
        ).sum(dim=-1)
        best_remote_probability = route_probabilities.detach().masked_fill(
            ~legal_remote_actions, -torch.inf
        ).max(dim=-1).values
        best_remote_logit = route_logits.detach().masked_fill(
            ~legal_remote_actions, -torch.inf
        ).max(dim=-1).values
        local_logit = (
            route_logits.detach() * local_domain.to(route_logits.dtype)
        ).sum(dim=-1)
        probability_margin = local_probability - best_remote_probability
        logit_margin = local_logit - best_remote_logit
        if not torch.isfinite(best_remote_probability.masked_select(legal_remote)).all():
            raise RouteTelemetryError("best legal-remote probability is not finite")
        if not torch.isfinite(best_remote_logit.masked_select(legal_remote)).all():
            raise RouteTelemetryError("best legal-remote logit is not finite")

        expanded_advantage = advantage.detach().to(device=device).unsqueeze(-1).expand_as(
            route_active
        )
        expanded_return = return_target.detach().to(device=device).unsqueeze(-1).expand_as(
            route_active
        )
        local_group = active & selected_local
        remote_group = active & selected_remote
        defer_group = active & selected_defer
        legal_local_group = legal_remote & selected_local
        legal_remote_group = legal_remote & selected_remote
        gradients = measure_route_head_gradients(route_head)

        legal_count = _count(legal_remote)
        active_count = _count(active)
        local_count = _count(local_group)
        remote_count = _count(remote_group)
        defer_count = _count(defer_group)
        legal_local_count = _count(legal_local_group)
        legal_selected_remote_count = _count(legal_remote_group)
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
            route_entropy_mean=_optional_mean(route_entropy.detach(), active),
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
        )


__all__ = [
    "GradientMeasurement",
    "ROUTE_GRADIENT_STATUSES",
    "ROUTE_TELEMETRY_SCHEMA_VERSION",
    "RouteHeadGradientTelemetry",
    "RouteTelemetry",
    "RouteTelemetryError",
    "collect_route_telemetry",
    "measure_route_head_gradients",
]
