"""PPO objective and loss computation for CA-GAT-MAPPO.

This module only combines already-computed current policy statistics with
fixed rollout snapshots.  It does not perform recurrent re-evaluation,
backpropagation, parameter updates, gradient clipping, or training.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from ..config import RunConfig


class PPOObjectiveError(ValueError):
    """Raised when PPO statistics violate the frozen objective contract."""


def _finite_coefficient(
    value: float,
    name: str,
    *,
    strictly_positive: bool,
    less_than_one: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PPOObjectiveError(f"{name} must be a finite real scalar")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise PPOObjectiveError(f"{name} must be finite")
    if strictly_positive:
        if resolved <= 0.0:
            raise PPOObjectiveError(f"{name} must be positive")
    elif resolved < 0.0:
        raise PPOObjectiveError(f"{name} must be non-negative")
    if less_than_one and resolved >= 1.0:
        raise PPOObjectiveError(f"{name} must be less than one")
    return resolved


def _floating_tensor(tensor: Tensor, name: str, rank: int) -> Tensor:
    if not isinstance(tensor, Tensor) or not tensor.is_floating_point():
        raise PPOObjectiveError(f"{name} must be a floating tensor")
    if tensor.ndim != rank:
        shape = "[T,A]" if rank == 2 else "[T]"
        raise PPOObjectiveError(f"{name} must have shape {shape}")
    if tensor.numel() == 0:
        raise PPOObjectiveError(f"{name} cannot be empty")
    if not torch.isfinite(tensor).all():
        raise PPOObjectiveError(f"{name} contains NaN or Inf")
    return tensor


def _sequence_mask(tensor: Tensor, time_steps: int, device: torch.device) -> Tensor:
    if not isinstance(tensor, Tensor) or tensor.dtype != torch.bool:
        raise PPOObjectiveError("sequence_valid_mask must be a boolean tensor")
    if tensor.ndim != 1 or tensor.shape[0] != time_steps:
        raise PPOObjectiveError("sequence_valid_mask must have shape [T]")
    if tensor.device != device:
        raise PPOObjectiveError("sequence_valid_mask must share the statistics device")
    valid_count = int(tensor.sum().item())
    if valid_count == 0:
        raise PPOObjectiveError("sequence_valid_mask has no valid positions")
    expected = torch.arange(time_steps, device=device) < valid_count
    if not torch.equal(tensor, expected):
        raise PPOObjectiveError(
            "sequence_valid_mask must describe a contiguous prefix with trailing padding"
        )
    return tensor


def _masked_mean(values: Tensor, mask: Tensor, name: str) -> Tensor:
    if values.shape != mask.shape:
        raise PPOObjectiveError(f"{name} values and mask must have identical shapes")
    selected = values.masked_select(mask)
    if selected.numel() == 0:
        raise PPOObjectiveError(f"{name} has no valid positions")
    result = selected.mean()
    if not torch.isfinite(result):
        raise PPOObjectiveError(f"{name} reduction produced NaN or Inf")
    return result


@dataclass(frozen=True)
class PPOObjectiveDiagnostics:
    """Detached scalar diagnostics for one PPO objective evaluation."""

    valid_timestep_count: int
    valid_actor_position_count: int
    ratio_mean: float
    ratio_min: float
    ratio_max: float
    clipped_fraction: float
    surrogate_mean: float
    value_squared_error_mean: float
    entropy_mean: float

    def __post_init__(self) -> None:
        if self.valid_timestep_count <= 0 or self.valid_actor_position_count <= 0:
            raise PPOObjectiveError("diagnostic valid counts must be positive")
        scalars = (
            self.ratio_mean,
            self.ratio_min,
            self.ratio_max,
            self.clipped_fraction,
            self.surrogate_mean,
            self.value_squared_error_mean,
            self.entropy_mean,
        )
        if not all(math.isfinite(value) for value in scalars):
            raise PPOObjectiveError("PPO diagnostics contain NaN or Inf")
        if not 0.0 <= self.clipped_fraction <= 1.0:
            raise PPOObjectiveError("clipped_fraction must lie in [0, 1]")


@dataclass(frozen=True)
class CAGATMAPPOLossOutput:
    """Auditable PPO matrices, scalar losses, and detached diagnostics."""

    log_ratio: Tensor
    ratio: Tensor
    clipped_ratio: Tensor
    expanded_advantage: Tensor
    unclipped_surrogate: Tensor
    clipped_surrogate: Tensor
    surrogate: Tensor
    actor_valid_mask: Tensor
    critic_valid_mask: Tensor
    clipped_objective: Tensor
    actor_loss: Tensor
    critic_loss: Tensor
    entropy_mean: Tensor
    total_loss: Tensor
    epsilon_clip: float
    value_coefficient: float
    entropy_coefficient: float
    diagnostics: PPOObjectiveDiagnostics

    def __post_init__(self) -> None:
        matrices = (
            self.log_ratio,
            self.ratio,
            self.clipped_ratio,
            self.expanded_advantage,
            self.unclipped_surrogate,
            self.clipped_surrogate,
            self.surrogate,
        )
        if any(not isinstance(tensor, Tensor) or tensor.ndim != 2 for tensor in matrices):
            raise PPOObjectiveError("PPO actor statistics must be [T,A] tensors")
        matrix_shape = matrices[0].shape
        if any(tensor.shape != matrix_shape for tensor in matrices[1:]):
            raise PPOObjectiveError("PPO actor statistics must share one [T,A] shape")
        if self.actor_valid_mask.shape != matrix_shape or self.actor_valid_mask.dtype != torch.bool:
            raise PPOObjectiveError("actor_valid_mask must be boolean [T,A]")
        if (
            self.critic_valid_mask.ndim != 1
            or self.critic_valid_mask.shape[0] != matrix_shape[0]
            or self.critic_valid_mask.dtype != torch.bool
        ):
            raise PPOObjectiveError("critic_valid_mask must be boolean [T]")
        for tensor in matrices:
            if not tensor.is_floating_point() or not torch.isfinite(tensor).all():
                raise PPOObjectiveError("PPO actor statistics contain NaN or Inf")
        losses = (
            self.clipped_objective,
            self.actor_loss,
            self.critic_loss,
            self.entropy_mean,
            self.total_loss,
        )
        if any(
            not isinstance(tensor, Tensor)
            or tensor.ndim != 0
            or not torch.isfinite(tensor)
            for tensor in losses
        ):
            raise PPOObjectiveError("PPO objective outputs must be finite scalars")
        if not torch.allclose(self.actor_loss, -self.clipped_objective):
            raise PPOObjectiveError("actor_loss must equal -clipped_objective")


def compute_ppo_objective_and_loss(
    *,
    new_joint_log_prob: Tensor,
    old_joint_log_prob: Tensor,
    advantage: Tensor,
    current_value: Tensor,
    return_target: Tensor,
    entropy: Tensor,
    sequence_valid_mask: Tensor,
    epsilon_clip: float,
    value_coefficient: float,
    entropy_coefficient: float,
) -> CAGATMAPPOLossOutput:
    """Compute the frozen PPO objective from proposal-policy statistics.

    Policy inputs use explicit ``[T,A]`` axes.  The raw shared-team advantage,
    current centralized value, return target, and sequence mask use ``[T]``.
    Rollout quantities are detached fixed targets; gradients remain connected
    only to current log-probabilities, current values, and current entropy.
    """

    epsilon = _finite_coefficient(
        epsilon_clip,
        "epsilon_clip",
        strictly_positive=True,
        less_than_one=True,
    )
    value_weight = _finite_coefficient(
        value_coefficient,
        "value_coefficient",
        strictly_positive=False,
    )
    entropy_weight = _finite_coefficient(
        entropy_coefficient,
        "entropy_coefficient",
        strictly_positive=False,
    )

    new_log_prob = _floating_tensor(new_joint_log_prob, "new_joint_log_prob", 2)
    old_log_prob = _floating_tensor(old_joint_log_prob, "old_joint_log_prob", 2)
    raw_advantage = _floating_tensor(advantage, "advantage", 1)
    values = _floating_tensor(current_value, "current_value", 1)
    targets = _floating_tensor(return_target, "return_target", 1)
    active_entropy = _floating_tensor(entropy, "entropy", 2)

    time_steps, agent_count = new_log_prob.shape
    if time_steps == 0 or agent_count == 0:
        raise PPOObjectiveError("policy statistics cannot have empty time or agent axes")
    if old_log_prob.shape != new_log_prob.shape:
        raise PPOObjectiveError("new and old joint log-prob must share shape [T,A]")
    if active_entropy.shape != new_log_prob.shape:
        raise PPOObjectiveError("entropy must share the policy shape [T,A]")
    for name, tensor in (
        ("advantage", raw_advantage),
        ("current_value", values),
        ("return_target", targets),
    ):
        if tensor.shape != (time_steps,):
            raise PPOObjectiveError(f"{name} must align with the policy time axis [T]")
    floating_inputs = (
        old_log_prob,
        raw_advantage,
        values,
        targets,
        active_entropy,
    )
    if any(tensor.device != new_log_prob.device for tensor in floating_inputs):
        raise PPOObjectiveError("all PPO statistics must share one device")
    if any(tensor.dtype != new_log_prob.dtype for tensor in floating_inputs):
        raise PPOObjectiveError("all PPO statistics must share one floating dtype")

    critic_valid_mask = _sequence_mask(
        sequence_valid_mask,
        time_steps,
        new_log_prob.device,
    )
    actor_valid_mask = critic_valid_mask.reshape(time_steps, 1).expand(
        time_steps,
        agent_count,
    )

    fixed_old_log_prob = old_log_prob.detach()
    fixed_advantage = raw_advantage.detach()
    fixed_return_target = targets.detach()
    log_ratio = new_log_prob - fixed_old_log_prob
    if not torch.isfinite(log_ratio).all():
        raise PPOObjectiveError("new-old log-prob difference contains NaN or Inf")
    ratio = torch.exp(log_ratio)
    if not torch.isfinite(ratio).all():
        raise PPOObjectiveError("exp(log-ratio) produced NaN or Inf")
    clipped_ratio = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)

    expanded_advantage = fixed_advantage.reshape(time_steps, 1).expand(
        time_steps,
        agent_count,
    )
    unclipped_surrogate = ratio * expanded_advantage
    clipped_surrogate = clipped_ratio * expanded_advantage
    surrogate = torch.minimum(unclipped_surrogate, clipped_surrogate)
    if not torch.isfinite(unclipped_surrogate).all() or not torch.isfinite(
        clipped_surrogate
    ).all():
        raise PPOObjectiveError("PPO surrogate produced NaN or Inf")
    clipped_objective = _masked_mean(
        surrogate,
        actor_valid_mask,
        "clipped_objective",
    )
    actor_loss = -clipped_objective

    value_squared_error = (values - fixed_return_target).square()
    critic_loss = 0.5 * _masked_mean(
        value_squared_error,
        critic_valid_mask,
        "critic_loss",
    )
    entropy_mean = _masked_mean(active_entropy, actor_valid_mask, "entropy_mean")
    total_loss = (
        actor_loss
        + value_weight * critic_loss
        - entropy_weight * entropy_mean
    )
    if not torch.isfinite(total_loss):
        raise PPOObjectiveError("total_loss produced NaN or Inf")

    valid_ratios = ratio.masked_select(actor_valid_mask).detach()
    valid_surrogate = surrogate.masked_select(actor_valid_mask).detach()
    valid_value_error = value_squared_error.masked_select(critic_valid_mask).detach()
    clipped_positions = (ratio != clipped_ratio) & actor_valid_mask
    diagnostics = PPOObjectiveDiagnostics(
        valid_timestep_count=int(critic_valid_mask.sum().item()),
        valid_actor_position_count=int(actor_valid_mask.sum().item()),
        ratio_mean=float(valid_ratios.mean().cpu().item()),
        ratio_min=float(valid_ratios.min().cpu().item()),
        ratio_max=float(valid_ratios.max().cpu().item()),
        clipped_fraction=float(
            clipped_positions.sum().to(dtype=torch.float32).div(
                actor_valid_mask.sum()
            ).cpu().item()
        ),
        surrogate_mean=float(valid_surrogate.mean().cpu().item()),
        value_squared_error_mean=float(valid_value_error.mean().cpu().item()),
        entropy_mean=float(entropy_mean.detach().cpu().item()),
    )

    return CAGATMAPPOLossOutput(
        log_ratio=log_ratio,
        ratio=ratio,
        clipped_ratio=clipped_ratio,
        expanded_advantage=expanded_advantage,
        unclipped_surrogate=unclipped_surrogate,
        clipped_surrogate=clipped_surrogate,
        surrogate=surrogate,
        actor_valid_mask=actor_valid_mask,
        critic_valid_mask=critic_valid_mask,
        clipped_objective=clipped_objective,
        actor_loss=actor_loss,
        critic_loss=critic_loss,
        entropy_mean=entropy_mean,
        total_loss=total_loss,
        epsilon_clip=epsilon,
        value_coefficient=value_weight,
        entropy_coefficient=entropy_weight,
        diagnostics=diagnostics,
    )


def compute_configured_ppo_objective_and_loss(
    *,
    new_joint_log_prob: Tensor,
    old_joint_log_prob: Tensor,
    advantage: Tensor,
    current_value: Tensor,
    return_target: Tensor,
    entropy: Tensor,
    sequence_valid_mask: Tensor,
    config: RunConfig,
) -> CAGATMAPPOLossOutput:
    """Use the Section 4 values already centralized in ``RunConfig``."""

    if not isinstance(config, RunConfig):
        raise TypeError("config must be a RunConfig")
    config.validate()
    mappo = config.training.mappo
    return compute_ppo_objective_and_loss(
        new_joint_log_prob=new_joint_log_prob,
        old_joint_log_prob=old_joint_log_prob,
        advantage=advantage,
        current_value=current_value,
        return_target=return_target,
        entropy=entropy,
        sequence_valid_mask=sequence_valid_mask,
        epsilon_clip=mappo.ppo_clip_epsilon,
        value_coefficient=mappo.value_coefficient,
        entropy_coefficient=mappo.entropy_coefficient,
    )


__all__ = [
    "CAGATMAPPOLossOutput",
    "PPOObjectiveDiagnostics",
    "PPOObjectiveError",
    "compute_configured_ppo_objective_and_loss",
    "compute_ppo_objective_and_loss",
]
