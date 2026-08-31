"""Deterministic recurrent PPO parameter updates for CA-GAT-MAPPO.

This module is deliberately limited to adapting one already-finalized full
rollout into the frozen 8-by-32 recurrent minibatch and applying four PPO
epochs.  It does not collect rollouts, run a trainer, save checkpoints, expose
a CLI, or start an experiment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.optim import Adam

from ..config import ActorRatioMode, AgentCreditMode, RouteCreditMode, RunConfig
from ..env.actions import ActionProposal
from .ca_gat_mappo import (
    ACTION_BRANCH_ORDER,
    ActorTensorBatch,
    CAGATMAPPOActor,
    CentralizedStateTensorBatch,
    MAPPOCentralizedCritic,
    MAPPOTensorSpec,
)
from .ca_gat_mappo_actions import (
    CAGATMAPPOActionDistribution,
    SequentialActionDistributionOutput,
    SequentialActionMaskBatch,
)
from .ca_gat_mappo_gae import (
    compute_rollout_gae,
    compute_route_specific_advantage,
)
from .ca_gat_mappo_ppo import (
    CAGATMAPPOLossOutput,
    compute_configured_ppo_objective_and_loss,
)
from .ca_gat_mappo_route_telemetry import (
    RouteTelemetry,
    collect_route_telemetry,
)
from .ca_gat_mappo_rollout import CAGATMAPPORolloutBuffer


class RecurrentPPOUpdateError(ValueError):
    """Raised when the frozen recurrent PPO update contract is violated."""


_FROZEN_GEOMETRY = (256, 32, 8, 4)


def _validate_frozen_contract(config: RunConfig) -> None:
    if not isinstance(config, RunConfig):
        raise TypeError("config must be a RunConfig")
    config.validate()
    mappo = config.training.mappo
    geometry = (
        mappo.rollout_length_slots,
        mappo.recurrent_chunk_length_slots,
        mappo.sequence_minibatch_size,
        mappo.update_epochs,
    )
    if geometry != _FROZEN_GEOMETRY:
        raise RecurrentPPOUpdateError(
            "recurrent PPO geometry must remain rollout=256, chunk=32, "
            "minibatch=8, epochs=4"
        )


def _cat_actor_batches(batches: tuple[ActorTensorBatch, ...]) -> ActorTensorBatch:
    return ActorTensorBatch(
        self_features=torch.cat([item.self_features for item in batches], dim=0),
        neighbor_public_features=torch.cat(
            [item.neighbor_public_features for item in batches], dim=0
        ),
        edge_features=torch.cat([item.edge_features for item in batches], dim=0),
        neighbor_mask=torch.cat([item.neighbor_mask for item in batches], dim=0),
        action_masks={
            branch: torch.cat([item.action_masks[branch] for item in batches], dim=0)
            for branch in ACTION_BRANCH_ORDER
        },
        action_indices={
            branch: torch.cat([item.action_indices[branch] for item in batches], dim=0)
            for branch in ACTION_BRANCH_ORDER
        },
        episode_starts=torch.cat([item.episode_starts for item in batches], dim=0),
    )


@dataclass(frozen=True)
class RecurrentPPOMinibatch:
    """One deterministic, padding-free view of a finalized 256-slot rollout."""

    spec: MAPPOTensorSpec
    rollout_indices: Tensor
    actor_batch: ActorTensorBatch
    action_mask_batch: SequentialActionMaskBatch
    centralized_batch: CentralizedStateTensorBatch
    proposals: tuple[tuple[tuple[ActionProposal, ...], ...], ...]
    initial_hidden: Tensor
    stored_hidden_in: Tensor
    active_branch_indicators: Tensor
    old_joint_log_prob: Tensor
    old_value: Tensor
    advantage: Tensor
    return_target: Tensor
    sequence_valid_mask: Tensor
    agent_credit_mode: str
    old_branch_log_probs: Tensor | None = None
    td_residual: Tensor | None = None
    route_advantage: Tensor | None = None
    bootstrap_mask: Tensor | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.spec, MAPPOTensorSpec):
            raise TypeError("spec must be a MAPPOTensorSpec")
        shape = self.actor_batch.validate(self.spec)
        if shape != (8, 32, self.spec.uav_count):
            raise RecurrentPPOUpdateError("actor minibatch must have shape [8,32,A]")
        self.action_mask_batch.validate(shape)
        if self.centralized_batch.validate(self.spec) != (8, 32):
            raise RecurrentPPOUpdateError("critic minibatch must have shape [8,32,F]")
        try:
            credit_mode = AgentCreditMode(self.agent_credit_mode)
        except (TypeError, ValueError) as exc:
            raise RecurrentPPOUpdateError("agent_credit_mode is invalid") from exc
        credit_shape = (
            (8, 32)
            if credit_mode is AgentCreditMode.TEAM
            else (8, 32, self.spec.uav_count)
        )
        expected = {
            "rollout_indices": (8, 32),
            "initial_hidden": (8, self.spec.uav_count, self.spec.gru_hidden_dimension),
            "stored_hidden_in": (
                8,
                32,
                self.spec.uav_count,
                self.spec.gru_hidden_dimension,
            ),
            "active_branch_indicators": (
                8,
                32,
                self.spec.uav_count,
                len(ACTION_BRANCH_ORDER),
            ),
            "old_joint_log_prob": (8, 32, self.spec.uav_count),
            "old_value": credit_shape,
            "advantage": credit_shape,
            "return_target": credit_shape,
            "sequence_valid_mask": (8, 32),
        }
        for name, expected_shape in expected.items():
            tensor = getattr(self, name)
            if not isinstance(tensor, Tensor) or tuple(tensor.shape) != expected_shape:
                raise RecurrentPPOUpdateError(f"{name} has an invalid shape")
            if tensor.device.type != "cpu":
                raise RecurrentPPOUpdateError(f"{name} must remain a CPU snapshot")
        optional = {
            "old_branch_log_probs": (8, 32, self.spec.uav_count, len(ACTION_BRANCH_ORDER)),
            "td_residual": credit_shape,
            "route_advantage": (8, 32, self.spec.uav_count),
        }
        for name, expected_shape in optional.items():
            tensor = getattr(self, name)
            if tensor is not None and (not isinstance(tensor, Tensor) or tuple(tensor.shape) != expected_shape or tensor.device.type != "cpu"):
                raise RecurrentPPOUpdateError(f"{name} has an invalid optional snapshot shape")
            if tensor is not None and (tensor.dtype != torch.float32 or tensor.requires_grad or not torch.isfinite(tensor).all()):
                raise RecurrentPPOUpdateError(f"{name} must be a detached finite CPU float32 snapshot")
        if self.bootstrap_mask is not None and (
            not isinstance(self.bootstrap_mask, Tensor)
            or tuple(self.bootstrap_mask.shape) != (8, 32)
            or self.bootstrap_mask.device.type != "cpu"
            or self.bootstrap_mask.dtype != torch.bool
        ):
            raise RecurrentPPOUpdateError(
                "bootstrap_mask must be a detached boolean [8,32] CPU snapshot"
            )
        if self.rollout_indices.dtype != torch.long:
            raise RecurrentPPOUpdateError("rollout_indices must use torch.long")
        expected_indices = torch.arange(256, dtype=torch.long).reshape(8, 32)
        if not torch.equal(self.rollout_indices, expected_indices):
            raise RecurrentPPOUpdateError(
                "chunks must preserve deterministic contiguous rollout order"
            )
        if self.sequence_valid_mask.dtype != torch.bool or not torch.all(
            self.sequence_valid_mask
        ):
            raise RecurrentPPOUpdateError("the frozen minibatch has no padding")
        if self.active_branch_indicators.dtype != torch.bool:
            raise RecurrentPPOUpdateError("active_branch_indicators must be boolean")
        floating = (
            self.initial_hidden,
            self.stored_hidden_in,
            self.old_joint_log_prob,
            self.old_value,
            self.advantage,
            self.return_target,
        )
        if any(
            tensor.dtype != torch.float32
            or tensor.requires_grad
            or not torch.isfinite(tensor).all()
            for tensor in floating
        ):
            raise RecurrentPPOUpdateError(
                "rollout and target statistics must be detached finite CPU float32"
            )
        if not torch.equal(self.initial_hidden, self.stored_hidden_in[:, 0]):
            raise RecurrentPPOUpdateError(
                "each chunk must start from stored hidden_in[t_start]"
            )
        episode_starts = self.actor_batch.episode_starts
        reset_hidden = self.stored_hidden_in.masked_select(
            episode_starts.unsqueeze(-1).expand_as(self.stored_hidden_in)
        )
        if torch.count_nonzero(reset_hidden).item() != 0:
            raise RecurrentPPOUpdateError(
                "episode-start hidden snapshots must be exactly zero"
            )
        if len(self.proposals) != 8 or any(len(chunk) != 32 for chunk in self.proposals):
            raise RecurrentPPOUpdateError("proposal snapshots must have shape [8][32][A]")
        if any(
            len(time_step) != self.spec.uav_count
            for chunk in self.proposals
            for time_step in chunk
        ):
            raise RecurrentPPOUpdateError("proposal snapshots have a ragged agent axis")


def build_recurrent_ppo_minibatch(
    buffer: CAGATMAPPORolloutBuffer,
    config: RunConfig,
) -> RecurrentPPOMinibatch:
    """Build the sole frozen minibatch without shuffle, padding, or burn-in."""

    _validate_frozen_contract(config)
    if not isinstance(buffer, CAGATMAPPORolloutBuffer):
        raise TypeError("buffer must be a CAGATMAPPORolloutBuffer")
    expected_spec = MAPPOTensorSpec.from_config(config)
    if buffer.spec != expected_spec:
        raise RecurrentPPOUpdateError("rollout buffer and config tensor specs differ")
    if not buffer.finalized:
        raise RecurrentPPOUpdateError("rollout must be finalized before PPO update")
    if len(buffer) != 256:
        raise RecurrentPPOUpdateError("finalized rollout must contain exactly 256 slots")

    chunks = tuple(buffer.get_sequence(index * 32, 32) for index in range(8))
    full_rollout = buffer.get_sequence(0, 256)
    gae = compute_rollout_gae(full_rollout, config)
    route_advantage = None
    mappo = config.training.mappo
    if RouteCreditMode(mappo.route_credit_mode) is RouteCreditMode.ROUTE_SPECIFIC_GAE:
        route_advantage = compute_route_specific_advantage(
            base_td_residual=gae.td_residual,
            bootstrap_mask=gae.bootstrap_mask,
            sequence_mask=gae.sequence_mask,
            gamma=mappo.gamma,
            route_gae_lambda=mappo.route_gae_lambda,
        )
    actor_batches = tuple(chunk.actor_batch for chunk in chunks)
    actor_batch = _cat_actor_batches(actor_batches)
    centralized_batch = CentralizedStateTensorBatch(
        torch.cat([chunk.centralized_batch.features for chunk in chunks], dim=0)
    )
    action_mask_batch = SequentialActionMaskBatch(
        tuple(chunk.action_mask_batch.contracts[0] for chunk in chunks)
    )
    proposals = tuple(chunk.proposals[0] for chunk in chunks)
    result = RecurrentPPOMinibatch(
        spec=expected_spec,
        rollout_indices=torch.arange(256, dtype=torch.long).reshape(8, 32),
        actor_batch=actor_batch,
        action_mask_batch=action_mask_batch,
        centralized_batch=centralized_batch,
        proposals=proposals,
        initial_hidden=torch.cat([chunk.initial_hidden for chunk in chunks], dim=0),
        stored_hidden_in=torch.stack([chunk.hidden_in for chunk in chunks], dim=0),
        active_branch_indicators=torch.stack(
            [chunk.active_branch_indicators for chunk in chunks], dim=0
        ),
        old_joint_log_prob=torch.stack(
            [chunk.old_joint_log_prob for chunk in chunks], dim=0
        ),
        old_value=torch.stack([chunk.old_value for chunk in chunks], dim=0),
        advantage=gae.advantage.reshape(
            (8, 32) + tuple(gae.advantage.shape[1:])
        ).clone(),
        return_target=gae.return_target.reshape(
            (8, 32) + tuple(gae.return_target.shape[1:])
        ).clone(),
        sequence_valid_mask=gae.sequence_mask.reshape(8, 32).clone(),
        agent_credit_mode=AgentCreditMode(
            config.training.mappo.agent_credit_mode
        ).value,
        old_branch_log_probs=torch.stack([chunk.old_branch_log_probs for chunk in chunks], dim=0),
        td_residual=gae.td_residual.reshape(
            (8, 32) + tuple(gae.td_residual.shape[1:])
        ).clone(),
        route_advantage=(
            None
            if route_advantage is None
            else route_advantage.reshape(8, 32, expected_spec.uav_count).clone()
        ),
        bootstrap_mask=gae.bootstrap_mask.reshape(8, 32).clone(),
    )
    return result


@dataclass(frozen=True)
class BatchedCAGATMAPPOLossOutput:
    """The existing PPO output with explicit recurrent [B,L,A] views."""

    flattened: CAGATMAPPOLossOutput
    batch_size: int
    sequence_length: int
    agent_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.flattened, CAGATMAPPOLossOutput):
            raise TypeError("flattened must be a CAGATMAPPOLossOutput")
        if tuple(self.flattened.ratio.shape) != (
            self.batch_size * self.sequence_length,
            self.agent_count,
        ):
            raise RecurrentPPOUpdateError("flattened PPO output has incompatible axes")

    def _actor_view(self, tensor: Tensor) -> Tensor:
        return tensor.reshape(self.batch_size, self.sequence_length, self.agent_count)

    def _critic_view(self, tensor: Tensor) -> Tensor:
        return tensor.reshape(
            (self.batch_size, self.sequence_length) + tuple(tensor.shape[1:])
        )

    def _branch_view(self, tensor: Tensor) -> Tensor:
        return tensor.reshape(
            self.batch_size,
            self.sequence_length,
            self.agent_count,
            len(ACTION_BRANCH_ORDER),
        )

    @property
    def log_ratio(self) -> Tensor:
        return self._actor_view(self.flattened.log_ratio)

    @property
    def ratio(self) -> Tensor:
        return self._actor_view(self.flattened.ratio)

    @property
    def clipped_ratio(self) -> Tensor:
        return self._actor_view(self.flattened.clipped_ratio)

    @property
    def expanded_advantage(self) -> Tensor:
        return self._actor_view(self.flattened.expanded_advantage)

    @property
    def surrogate(self) -> Tensor:
        return self._actor_view(self.flattened.surrogate)

    @property
    def branch_ratio(self) -> Tensor | None:
        value = self.flattened.branch_ratio
        return None if value is None else self._branch_view(value)

    @property
    def branch_surrogate(self) -> Tensor | None:
        value = self.flattened.branch_surrogate
        return None if value is None else self._branch_view(value)

    @property
    def branch_advantage(self) -> Tensor | None:
        value = self.flattened.branch_advantage
        return None if value is None else self._branch_view(value)

    @property
    def active_branch_indicators(self) -> Tensor | None:
        value = self.flattened.active_branch_indicators
        return None if value is None else self._branch_view(value)

    @property
    def active_branch_count(self) -> Tensor | None:
        value = self.flattened.active_branch_count
        return None if value is None else self._actor_view(value)

    @property
    def actor_valid_mask(self) -> Tensor:
        return self._actor_view(self.flattened.actor_valid_mask)

    @property
    def critic_valid_mask(self) -> Tensor:
        return self._critic_view(self.flattened.critic_valid_mask)

    @property
    def clipped_objective(self) -> Tensor:
        return self.flattened.clipped_objective

    @property
    def actor_loss(self) -> Tensor:
        return self.flattened.actor_loss

    @property
    def critic_loss(self) -> Tensor:
        return self.flattened.critic_loss

    @property
    def entropy_mean(self) -> Tensor:
        return self.flattened.entropy_mean

    @property
    def total_loss(self) -> Tensor:
        return self.flattened.total_loss

    @property
    def diagnostics(self):
        return self.flattened.diagnostics

    @property
    def route_entropy_coefficient(self) -> float:
        return self.flattened.route_entropy_coefficient

    @property
    def route_entropy_schedule_progress(self) -> float:
        return self.flattened.route_entropy_schedule_progress

    @property
    def route_entropy_loss_contribution(self) -> Tensor:
        return self.flattened.route_entropy_loss_contribution

    @property
    def other_branch_entropy_loss_contribution(self) -> Tensor:
        return self.flattened.other_branch_entropy_loss_contribution

    @property
    def global_entropy_loss_contribution(self) -> Tensor:
        return self.flattened.global_entropy_loss_contribution

    @property
    def collected_environment_steps(self) -> int | None:
        return self.flattened.collected_environment_steps


def compute_configured_batched_ppo_objective_and_loss(
    *,
    new_joint_log_prob: Tensor,
    old_joint_log_prob: Tensor,
    advantage: Tensor,
    current_value: Tensor,
    return_target: Tensor,
    entropy: Tensor,
    sequence_valid_mask: Tensor,
    config: RunConfig,
    new_branch_log_probs: Tensor | None = None,
    old_branch_log_probs: Tensor | None = None,
    active_branch_indicators: Tensor | None = None,
    route_entropy: Tensor | None = None,
    collected_environment_steps: int | None = None,
    route_advantage: Tensor | None = None,
) -> BatchedCAGATMAPPOLossOutput:
    """Apply the passed PPO objective uniformly over all B-by-L positions."""

    _validate_frozen_contract(config)
    if not isinstance(new_joint_log_prob, Tensor) or new_joint_log_prob.ndim != 3:
        raise RecurrentPPOUpdateError("new_joint_log_prob must have shape [B,L,A]")
    batch_size, sequence_length, agent_count = new_joint_log_prob.shape
    if (batch_size, sequence_length) != (8, 32) or agent_count <= 0:
        raise RecurrentPPOUpdateError("batched PPO policy axes must be [8,32,A]")
    for name, tensor in (
        ("old_joint_log_prob", old_joint_log_prob),
        ("entropy", entropy),
    ):
        if not isinstance(tensor, Tensor) or tensor.shape != new_joint_log_prob.shape:
            raise RecurrentPPOUpdateError(f"{name} must have shape [8,32,A]")
    if route_entropy is not None and (
        not isinstance(route_entropy, Tensor)
        or route_entropy.shape != new_joint_log_prob.shape
    ):
        raise RecurrentPPOUpdateError(
            "route_entropy must have shape [8,32,A] or be NA"
        )
    credit_mode = AgentCreditMode(config.training.mappo.agent_credit_mode)
    expected_credit_shape = (
        (8, 32)
        if credit_mode is AgentCreditMode.TEAM
        else (8, 32, agent_count)
    )
    for name, tensor in (
        ("advantage", advantage),
        ("current_value", current_value),
        ("return_target", return_target),
    ):
        if not isinstance(tensor, Tensor) or tensor.shape != expected_credit_shape:
            raise RecurrentPPOUpdateError(
                f"{name} shape differs from agent_credit_mode"
            )
    if route_advantage is not None and (
        not isinstance(route_advantage, Tensor)
        or tuple(route_advantage.shape) != (8, 32, agent_count)
    ):
        raise RecurrentPPOUpdateError(
            "route_advantage must have shape [8,32,A] or be NA"
        )
    if (
        not isinstance(sequence_valid_mask, Tensor)
        or sequence_valid_mask.shape != (8, 32)
        or sequence_valid_mask.dtype != torch.bool
        or not torch.all(sequence_valid_mask)
    ):
        raise RecurrentPPOUpdateError("all 8x32 positions must be real and valid")

    ratio_mode = ActorRatioMode(config.training.mappo.actor_ratio_mode)
    flattened_new_branches = None
    flattened_old_branches = None
    flattened_active_branches = None
    if ratio_mode is ActorRatioMode.BRANCH_SPECIFIC:
        expected_branch_shape = (
            8,
            32,
            agent_count,
            len(ACTION_BRANCH_ORDER),
        )
        for name, tensor in (
            ("new_branch_log_probs", new_branch_log_probs),
            ("old_branch_log_probs", old_branch_log_probs),
        ):
            if not isinstance(tensor, Tensor) or tuple(tensor.shape) != expected_branch_shape:
                raise RecurrentPPOUpdateError(
                    f"{name} must have shape [8,32,A,7]"
                )
        if (
            not isinstance(active_branch_indicators, Tensor)
            or tuple(active_branch_indicators.shape) != expected_branch_shape
            or active_branch_indicators.dtype != torch.bool
        ):
            raise RecurrentPPOUpdateError(
                "active_branch_indicators must be boolean [8,32,A,7]"
            )
        assert new_branch_log_probs is not None
        assert old_branch_log_probs is not None
        flattened_new_branches = new_branch_log_probs.reshape(
            256, agent_count, len(ACTION_BRANCH_ORDER)
        )
        flattened_old_branches = old_branch_log_probs.reshape(
            256, agent_count, len(ACTION_BRANCH_ORDER)
        )
        flattened_active_branches = active_branch_indicators.reshape(
            256, agent_count, len(ACTION_BRANCH_ORDER)
        )

    flattened = compute_configured_ppo_objective_and_loss(
        new_joint_log_prob=new_joint_log_prob.reshape(256, agent_count),
        old_joint_log_prob=old_joint_log_prob.reshape(256, agent_count),
        advantage=advantage.reshape(
            (256,) if credit_mode is AgentCreditMode.TEAM else (256, agent_count)
        ),
        current_value=current_value.reshape(
            (256,) if credit_mode is AgentCreditMode.TEAM else (256, agent_count)
        ),
        return_target=return_target.reshape(
            (256,) if credit_mode is AgentCreditMode.TEAM else (256, agent_count)
        ),
        entropy=entropy.reshape(256, agent_count),
        sequence_valid_mask=sequence_valid_mask.reshape(256),
        config=config,
        new_branch_log_probs=flattened_new_branches,
        old_branch_log_probs=flattened_old_branches,
        active_branch_indicators=flattened_active_branches,
        route_entropy=(
            None
            if route_entropy is None
            else route_entropy.reshape(256, agent_count)
        ),
        collected_environment_steps=collected_environment_steps,
        route_advantage=(
            None
            if route_advantage is None
            else route_advantage.reshape(256, agent_count)
        ),
    )
    return BatchedCAGATMAPPOLossOutput(
        flattened=flattened,
        batch_size=batch_size,
        sequence_length=sequence_length,
        agent_count=agent_count,
    )


def _module_device_dtype(module: nn.Module, name: str) -> tuple[torch.device, torch.dtype]:
    tensors = tuple(module.parameters()) + tuple(module.buffers())
    if not tensors:
        raise RecurrentPPOUpdateError(f"{name} has no tensors")
    devices = {tensor.device for tensor in tensors}
    if len(devices) != 1:
        raise RecurrentPPOUpdateError(f"{name} tensors must use one device")
    floating_dtypes = {tensor.dtype for tensor in tensors if tensor.is_floating_point()}
    if len(floating_dtypes) != 1:
        raise RecurrentPPOUpdateError(f"{name} floating tensors must use one dtype")
    dtype = next(iter(floating_dtypes))
    if dtype != torch.float32:
        raise RecurrentPPOUpdateError("the frozen network update dtype is torch.float32")
    return next(iter(devices)), dtype


@dataclass(frozen=True)
class CAGATMAPPOOptimizerBundle:
    """Separate Adam optimizers and their disjoint trainable parameter sets."""

    actor_optimizer: Adam
    critic_optimizer: Adam
    actor_parameters: tuple[nn.Parameter, ...]
    critic_parameters: tuple[nn.Parameter, ...]

    def validate(
        self,
        actor: CAGATMAPPOActor,
        critic: MAPPOCentralizedCritic,
        config: RunConfig,
    ) -> None:
        _validate_frozen_contract(config)
        if not isinstance(self.actor_optimizer, Adam) or not isinstance(
            self.critic_optimizer, Adam
        ):
            raise RecurrentPPOUpdateError("actor and critic optimizers must be Adam")
        expected_actor = tuple(parameter for parameter in actor.parameters() if parameter.requires_grad)
        expected_critic = tuple(parameter for parameter in critic.parameters() if parameter.requires_grad)
        if tuple(map(id, self.actor_parameters)) != tuple(map(id, expected_actor)):
            raise RecurrentPPOUpdateError("actor optimizer parameter set is incomplete")
        if tuple(map(id, self.critic_parameters)) != tuple(map(id, expected_critic)):
            raise RecurrentPPOUpdateError("critic optimizer parameter set is incomplete")
        if set(map(id, self.actor_parameters)) & set(map(id, self.critic_parameters)):
            raise RecurrentPPOUpdateError("actor and critic parameters must not overlap")
        actor_optimized = tuple(
            parameter
            for group in self.actor_optimizer.param_groups
            for parameter in group["params"]
        )
        critic_optimized = tuple(
            parameter
            for group in self.critic_optimizer.param_groups
            for parameter in group["params"]
        )
        if tuple(map(id, actor_optimized)) != tuple(map(id, self.actor_parameters)):
            raise RecurrentPPOUpdateError("actor Adam parameter IDs differ from actor set")
        if tuple(map(id, critic_optimized)) != tuple(map(id, self.critic_parameters)):
            raise RecurrentPPOUpdateError("critic Adam parameter IDs differ from critic set")
        mappo = config.training.mappo
        for name, optimizer, learning_rate in (
            ("actor", self.actor_optimizer, mappo.actor_learning_rate),
            ("critic", self.critic_optimizer, mappo.critic_learning_rate),
        ):
            if len(optimizer.param_groups) != 1:
                raise RecurrentPPOUpdateError(f"{name} Adam must use one parameter group")
            group = optimizer.param_groups[0]
            expected = {
                "lr": learning_rate,
                "betas": (mappo.adam_beta1, mappo.adam_beta2),
                "eps": mappo.adam_eps,
                "weight_decay": mappo.weight_decay,
            }
            if any(group[key] != value for key, value in expected.items()):
                raise RecurrentPPOUpdateError(
                    f"{name} Adam hyperparameters differ from RunConfig"
                )


def build_ca_gat_mappo_optimizers(
    actor: CAGATMAPPOActor,
    critic: MAPPOCentralizedCritic,
    config: RunConfig,
) -> CAGATMAPPOOptimizerBundle:
    """Build the two independent Adam optimizers from ``RunConfig`` only."""

    _validate_frozen_contract(config)
    if not isinstance(actor, CAGATMAPPOActor):
        raise TypeError("actor must be a CAGATMAPPOActor")
    if not isinstance(critic, MAPPOCentralizedCritic):
        raise TypeError("critic must be a MAPPOCentralizedCritic")
    expected_spec = MAPPOTensorSpec.from_config(config)
    if actor.spec != expected_spec or critic.spec != expected_spec:
        raise RecurrentPPOUpdateError("actor, critic, and config tensor specs must agree")
    actor_device, actor_dtype = _module_device_dtype(actor, "actor")
    critic_device, critic_dtype = _module_device_dtype(critic, "critic")
    if actor_device != critic_device or actor_dtype != critic_dtype:
        raise RecurrentPPOUpdateError("actor and critic must use the same device and dtype")
    actor_parameters = tuple(parameter for parameter in actor.parameters() if parameter.requires_grad)
    critic_parameters = tuple(parameter for parameter in critic.parameters() if parameter.requires_grad)
    if not actor_parameters or not critic_parameters:
        raise RecurrentPPOUpdateError("actor and critic need trainable parameters")
    if set(map(id, actor_parameters)) & set(map(id, critic_parameters)):
        raise RecurrentPPOUpdateError("actor and critic trainable parameters overlap")
    mappo = config.training.mappo
    common = {
        "betas": (mappo.adam_beta1, mappo.adam_beta2),
        "eps": mappo.adam_eps,
        "weight_decay": mappo.weight_decay,
    }
    bundle = CAGATMAPPOOptimizerBundle(
        actor_optimizer=Adam(actor_parameters, lr=mappo.actor_learning_rate, **common),
        critic_optimizer=Adam(critic_parameters, lr=mappo.critic_learning_rate, **common),
        actor_parameters=actor_parameters,
        critic_parameters=critic_parameters,
    )
    bundle.validate(actor, critic, config)
    return bundle


@dataclass(frozen=True)
class RecurrentPPOEvaluation:
    """Current-policy proposal statistics and centralized values for one minibatch."""

    policy: SequentialActionDistributionOutput
    current_value: Tensor
    agent_credit_mode: str

    def __post_init__(self) -> None:
        if self.policy.mode != "evaluation":
            raise RecurrentPPOUpdateError("PPO update must re-evaluate stored proposals")
        if tuple(self.policy.joint_log_prob.shape[:2]) != (8, 32):
            raise RecurrentPPOUpdateError("policy evaluation must retain [8,32,A] axes")
        try:
            credit_mode = AgentCreditMode(self.agent_credit_mode)
        except (TypeError, ValueError) as exc:
            raise RecurrentPPOUpdateError("agent_credit_mode is invalid") from exc
        agent_count = self.policy.joint_log_prob.shape[-1]
        expected = (
            (8, 32)
            if credit_mode is AgentCreditMode.TEAM
            else (8, 32, agent_count)
        )
        if tuple(self.current_value.shape) != expected:
            raise RecurrentPPOUpdateError(
                "current critic value shape differs from agent_credit_mode"
            )


@dataclass(frozen=True)
class _DeviceRecurrentPPOMinibatch:
    actor_batch: ActorTensorBatch
    action_mask_batch: SequentialActionMaskBatch
    centralized_batch: CentralizedStateTensorBatch
    proposals: tuple[tuple[tuple[ActionProposal, ...], ...], ...]
    initial_hidden: Tensor
    active_branch_indicators: Tensor
    old_joint_log_prob: Tensor
    old_value: Tensor
    advantage: Tensor
    return_target: Tensor
    sequence_valid_mask: Tensor
    old_branch_log_probs: Tensor | None = None
    td_residual: Tensor | None = None
    route_advantage: Tensor | None = None
    bootstrap_mask: Tensor | None = None


def _prepare_device_minibatch(
    minibatch: RecurrentPPOMinibatch,
    device: torch.device,
    dtype: torch.dtype,
) -> _DeviceRecurrentPPOMinibatch:
    return _DeviceRecurrentPPOMinibatch(
        actor_batch=minibatch.actor_batch.to(device, dtype=dtype),
        action_mask_batch=minibatch.action_mask_batch,
        centralized_batch=minibatch.centralized_batch.to(device, dtype=dtype),
        proposals=minibatch.proposals,
        initial_hidden=minibatch.initial_hidden.to(device=device, dtype=dtype),
        active_branch_indicators=minibatch.active_branch_indicators.to(device=device),
        old_joint_log_prob=minibatch.old_joint_log_prob.to(device=device, dtype=dtype),
        old_value=minibatch.old_value.to(device=device, dtype=dtype),
        advantage=minibatch.advantage.to(device=device, dtype=dtype),
        return_target=minibatch.return_target.to(device=device, dtype=dtype),
        sequence_valid_mask=minibatch.sequence_valid_mask.to(device=device),
        old_branch_log_probs=(minibatch.old_branch_log_probs.to(device=device, dtype=dtype)
                              if minibatch.old_branch_log_probs is not None else None),
        td_residual=(minibatch.td_residual.to(device=device, dtype=dtype)
                     if minibatch.td_residual is not None else None),
        route_advantage=(minibatch.route_advantage.to(device=device, dtype=dtype)
                         if minibatch.route_advantage is not None else None),
        bootstrap_mask=(minibatch.bootstrap_mask.to(device=device)
                        if minibatch.bootstrap_mask is not None else None),
    )


@dataclass(frozen=True)
class AgentCreditPPOEpochTelemetry:
    """Detached Fix-5 statistics derived from an existing PPO evaluation."""

    per_agent_value_mean: tuple[float, ...]
    per_agent_td_residual_mean: tuple[float, ...]
    per_agent_advantage_mean: tuple[float, ...]
    per_agent_return_mean: tuple[float, ...]
    same_timestep_advantage_equality_rate: float
    route_category_agent_advantage_mean: Mapping[
        str, tuple[float | None, ...]
    ]
    per_head_critic_loss: tuple[float, ...]

    def __post_init__(self) -> None:
        agent_count = len(self.per_agent_value_mean)
        if agent_count == 0:
            raise RecurrentPPOUpdateError("agent credit telemetry cannot be empty")
        vectors = (
            self.per_agent_value_mean,
            self.per_agent_td_residual_mean,
            self.per_agent_advantage_mean,
            self.per_agent_return_mean,
            self.per_head_critic_loss,
        )
        if any(len(values) != agent_count for values in vectors):
            raise RecurrentPPOUpdateError("agent credit telemetry axes disagree")
        if any(not math.isfinite(value) for values in vectors for value in values):
            raise RecurrentPPOUpdateError("agent credit telemetry contains NaN or Inf")
        if not 0.0 <= self.same_timestep_advantage_equality_rate <= 1.0:
            raise RecurrentPPOUpdateError(
                "same-timestep advantage equality rate must lie in [0, 1]"
            )
        if set(self.route_category_agent_advantage_mean) != {
            "local", "remote", "defer"
        }:
            raise RecurrentPPOUpdateError("route category telemetry is incomplete")
        for values in self.route_category_agent_advantage_mean.values():
            if len(values) != agent_count or any(
                value is not None and not math.isfinite(value) for value in values
            ):
                raise RecurrentPPOUpdateError(
                    "route category per-agent advantage telemetry is invalid"
                )


def _collect_agent_credit_epoch_telemetry(
    *,
    current_value: Tensor,
    td_residual: Tensor,
    advantage: Tensor,
    return_target: Tensor,
    sequence_valid_mask: Tensor,
    route_telemetry: RouteTelemetry | None,
    per_head_critic_loss: tuple[float, ...],
    same_timestep_advantage_equality_rate: float | None,
) -> AgentCreditPPOEpochTelemetry:
    tensors = (current_value, td_residual, advantage, return_target)
    if any(not isinstance(tensor, Tensor) or tensor.ndim != 3 for tensor in tensors):
        raise RecurrentPPOUpdateError(
            "role-decomposed telemetry requires [B,L,A] values"
        )
    shape = current_value.shape
    if any(tensor.shape != shape for tensor in tensors[1:]):
        raise RecurrentPPOUpdateError("agent credit telemetry value axes disagree")
    if sequence_valid_mask.shape != shape[:2] or sequence_valid_mask.dtype != torch.bool:
        raise RecurrentPPOUpdateError("agent credit telemetry mask is invalid")
    agent_count = shape[-1]

    def per_agent_mean(tensor: Tensor) -> tuple[float, ...]:
        detached = tensor.detach()
        return tuple(
            float(
                detached[..., agent_index]
                .masked_select(sequence_valid_mask)
                .mean()
                .cpu()
                .item()
            )
            for agent_index in range(agent_count)
        )

    category_values: dict[str, list[list[float]]] = {
        category: [[] for _ in range(agent_count)]
        for category in ("local", "remote", "defer")
    }
    if route_telemetry is not None:
        for sample in route_telemetry.samples:
            category_values[sample.category][sample.agent_index].append(
                sample.advantage
            )
    category_means = {
        category: tuple(
            None if not values else math.fsum(values) / len(values)
            for values in rows
        )
        for category, rows in category_values.items()
    }
    if same_timestep_advantage_equality_rate is None:
        raise RecurrentPPOUpdateError(
            "role-decomposed objective did not report advantage equality"
        )
    return AgentCreditPPOEpochTelemetry(
        per_agent_value_mean=per_agent_mean(current_value),
        per_agent_td_residual_mean=per_agent_mean(td_residual),
        per_agent_advantage_mean=per_agent_mean(advantage),
        per_agent_return_mean=per_agent_mean(return_target),
        same_timestep_advantage_equality_rate=(
            same_timestep_advantage_equality_rate
        ),
        route_category_agent_advantage_mean=category_means,
        per_head_critic_loss=per_head_critic_loss,
    )


@dataclass(frozen=True)
class RecurrentPPOEpochDiagnostics:
    """Detached diagnostics for one of the four fixed PPO epochs."""

    epoch_index: int
    actor_loss: float
    critic_loss: float
    entropy_mean: float
    total_loss: float
    ratio_mean: float
    actor_grad_norm_before_clip: float
    critic_grad_norm_before_clip: float
    clip_max_norm: float
    approx_kl: float | None = None
    clip_fraction: float | None = None
    route_entropy_coefficient: float | None = None
    route_entropy_schedule_progress: float | None = None
    route_entropy_loss_contribution: float | None = None
    other_branch_entropy_loss_contribution: float | None = None
    global_entropy_loss_contribution: float | None = None
    collected_environment_steps: int | None = None
    route_telemetry: RouteTelemetry | None = None
    agent_credit_telemetry: AgentCreditPPOEpochTelemetry | None = None

    def __post_init__(self) -> None:
        if self.epoch_index not in range(4):
            raise RecurrentPPOUpdateError("epoch_index must lie in [0, 3]")
        values = (
            self.actor_loss,
            self.critic_loss,
            self.entropy_mean,
            self.total_loss,
            self.ratio_mean,
            self.actor_grad_norm_before_clip,
            self.critic_grad_norm_before_clip,
            self.clip_max_norm,
        )
        if not all(math.isfinite(value) for value in values):
            raise RecurrentPPOUpdateError("update diagnostics contain NaN or Inf")
        if self.approx_kl is not None and not math.isfinite(self.approx_kl):
            raise RecurrentPPOUpdateError("approx_kl must be finite or NA")
        if self.clip_fraction is not None and not (
            math.isfinite(self.clip_fraction) and 0.0 <= self.clip_fraction <= 1.0
        ):
            raise RecurrentPPOUpdateError("clip_fraction must lie in [0, 1] or be NA")
        entropy_schedule_values = (
            self.route_entropy_coefficient,
            self.route_entropy_schedule_progress,
            self.route_entropy_loss_contribution,
            self.other_branch_entropy_loss_contribution,
            self.global_entropy_loss_contribution,
            self.collected_environment_steps,
        )
        present_count = sum(value is not None for value in entropy_schedule_values)
        if present_count not in (0, len(entropy_schedule_values)):
            raise RecurrentPPOUpdateError(
                "route entropy schedule telemetry must be all present or all NA"
            )
        if present_count:
            assert self.route_entropy_coefficient is not None
            assert self.route_entropy_schedule_progress is not None
            assert self.route_entropy_loss_contribution is not None
            assert self.other_branch_entropy_loss_contribution is not None
            assert self.global_entropy_loss_contribution is not None
            assert self.collected_environment_steps is not None
            scalar_values = (
                self.route_entropy_coefficient,
                self.route_entropy_schedule_progress,
                self.route_entropy_loss_contribution,
                self.other_branch_entropy_loss_contribution,
                self.global_entropy_loss_contribution,
            )
            if not all(math.isfinite(value) for value in scalar_values):
                raise RecurrentPPOUpdateError(
                    "route entropy schedule telemetry contains NaN or Inf"
                )
            if (
                self.route_entropy_coefficient < 0.0
                or self.route_entropy_loss_contribution < -1.0e-8
                or self.other_branch_entropy_loss_contribution < -1.0e-8
                or self.global_entropy_loss_contribution < -1.0e-8
            ):
                raise RecurrentPPOUpdateError(
                    "route entropy coefficients and contributions must be non-negative"
                )
            if not 0.0 <= self.route_entropy_schedule_progress <= 1.0:
                raise RecurrentPPOUpdateError(
                    "route entropy schedule progress must lie in [0, 1]"
                )
            if (
                isinstance(self.collected_environment_steps, bool)
                or not isinstance(self.collected_environment_steps, int)
                or self.collected_environment_steps < 0
            ):
                raise RecurrentPPOUpdateError(
                    "collected environment steps must be a non-negative integer"
                )
            if not math.isclose(
                self.global_entropy_loss_contribution,
                self.route_entropy_loss_contribution
                + self.other_branch_entropy_loss_contribution,
                rel_tol=1.0e-7,
                abs_tol=1.0e-9,
            ):
                raise RecurrentPPOUpdateError(
                    "global entropy contribution must equal route plus other branches"
                )
        if self.route_telemetry is not None and not isinstance(
            self.route_telemetry, RouteTelemetry
        ):
            raise TypeError("route_telemetry must be RouteTelemetry or None")
        if self.agent_credit_telemetry is not None and not isinstance(
            self.agent_credit_telemetry, AgentCreditPPOEpochTelemetry
        ):
            raise TypeError(
                "agent_credit_telemetry must be AgentCreditPPOEpochTelemetry or None"
            )


@dataclass(frozen=True)
class RecurrentPPOUpdateOutput:
    """Detached summary of one completed four-epoch full-rollout update."""

    epoch_diagnostics: tuple[RecurrentPPOEpochDiagnostics, ...]
    chunk_count: int
    chunk_length: int
    valid_transition_count: int
    old_policy_snapshot_preserved: bool

    def __post_init__(self) -> None:
        if len(self.epoch_diagnostics) != 4:
            raise RecurrentPPOUpdateError("exactly four epoch diagnostics are required")
        if (self.chunk_count, self.chunk_length, self.valid_transition_count) != (
            8,
            32,
            256,
        ):
            raise RecurrentPPOUpdateError("update output geometry differs from 8x32")
        if not self.old_policy_snapshot_preserved:
            raise RecurrentPPOUpdateError("old policy snapshot changed during update")


class CAGATMAPPORecurrentPPOUpdater:
    """Apply the frozen recurrent PPO update to one finalized full rollout."""

    def __init__(
        self,
        actor: CAGATMAPPOActor,
        critic: MAPPOCentralizedCritic,
        config: RunConfig,
        *,
        optimizers: CAGATMAPPOOptimizerBundle | None = None,
        action_distribution: CAGATMAPPOActionDistribution | None = None,
        route_telemetry_enabled: bool = True,
    ) -> None:
        _validate_frozen_contract(config)
        if not isinstance(actor, CAGATMAPPOActor):
            raise TypeError("actor must be a CAGATMAPPOActor")
        if not isinstance(critic, MAPPOCentralizedCritic):
            raise TypeError("critic must be a MAPPOCentralizedCritic")
        expected_spec = MAPPOTensorSpec.from_config(config)
        if actor.spec != expected_spec or critic.spec != expected_spec:
            raise RecurrentPPOUpdateError("actor, critic, and config specs differ")
        actor_device, actor_dtype = _module_device_dtype(actor, "actor")
        critic_device, critic_dtype = _module_device_dtype(critic, "critic")
        if actor_device != critic_device or actor_dtype != critic_dtype:
            raise RecurrentPPOUpdateError("actor and critic must share one training device")
        self.actor = actor
        self.critic = critic
        self.config = config
        self.device = actor_device
        self.dtype = actor_dtype
        self.optimizers = (
            build_ca_gat_mappo_optimizers(actor, critic, config)
            if optimizers is None
            else optimizers
        )
        if not isinstance(self.optimizers, CAGATMAPPOOptimizerBundle):
            raise TypeError("optimizers must be a CAGATMAPPOOptimizerBundle")
        self.optimizers.validate(actor, critic, config)
        self.action_distribution = (
            CAGATMAPPOActionDistribution(actor, config)
            if action_distribution is None
            else action_distribution
        )
        if not isinstance(self.action_distribution, CAGATMAPPOActionDistribution):
            raise TypeError("action_distribution must be CAGATMAPPOActionDistribution")
        if self.action_distribution.actor is not actor:
            raise RecurrentPPOUpdateError("action distribution must own the updated actor")
        if not isinstance(route_telemetry_enabled, bool):
            raise TypeError("route_telemetry_enabled must be boolean")
        self.route_telemetry_enabled = route_telemetry_enabled

    def _evaluate_prepared(
        self,
        minibatch: _DeviceRecurrentPPOMinibatch,
    ) -> RecurrentPPOEvaluation:
        policy = self.action_distribution.evaluate_actions(
            minibatch.actor_batch,
            minibatch.action_mask_batch,
            minibatch.proposals,
            minibatch.initial_hidden,
        )
        for branch in ACTION_BRANCH_ORDER:
            if not torch.equal(
                policy.action_masks[branch], minibatch.actor_batch.action_masks[branch]
            ):
                raise RecurrentPPOUpdateError(
                    f"re-evaluated {branch} mask differs from rollout snapshot"
                )
        active = torch.stack(
            [policy.active_branches[branch] for branch in ACTION_BRANCH_ORDER], dim=-1
        )
        if not torch.equal(active, minibatch.active_branch_indicators):
            raise RecurrentPPOUpdateError(
                "re-evaluated active branches differ from rollout snapshot"
            )
        if policy.proposals != minibatch.proposals:
            raise RecurrentPPOUpdateError("proposal re-evaluation changed stored proposals")
        raw_value = self.critic(minibatch.centralized_batch)
        credit_mode = AgentCreditMode(
            self.config.training.mappo.agent_credit_mode
        )
        current_value = (
            raw_value.squeeze(-1)
            if credit_mode is AgentCreditMode.TEAM
            else raw_value
        )
        return RecurrentPPOEvaluation(
            policy=policy,
            current_value=current_value,
            agent_credit_mode=credit_mode.value,
        )

    def evaluate_minibatch(
        self,
        minibatch: RecurrentPPOMinibatch,
    ) -> RecurrentPPOEvaluation:
        """Synchronously transfer and deterministically re-evaluate one minibatch."""

        if not isinstance(minibatch, RecurrentPPOMinibatch):
            raise TypeError("minibatch must be a RecurrentPPOMinibatch")
        prepared = _prepare_device_minibatch(minibatch, self.device, self.dtype)
        return self._evaluate_prepared(prepared)

    @staticmethod
    def _finite_grad_norm(
        parameters: tuple[nn.Parameter, ...],
        max_norm: float,
        name: str,
    ) -> float:
        try:
            norm = torch.nn.utils.clip_grad_norm_(
                parameters,
                max_norm=max_norm,
                error_if_nonfinite=True,
            )
        except RuntimeError as error:
            raise RecurrentPPOUpdateError(
                f"{name} gradient norm is NaN or Inf"
            ) from error
        value = float(norm.detach().cpu().item())
        if not math.isfinite(value):
            raise RecurrentPPOUpdateError(f"{name} gradient norm is NaN or Inf")
        return value

    def update(
        self,
        buffer: CAGATMAPPORolloutBuffer,
        *,
        collected_environment_steps: int | None = None,
    ) -> RecurrentPPOUpdateOutput:
        """Run exactly four epochs over the sole deterministic 8-by-32 minibatch."""
        if collected_environment_steps is not None and (
            isinstance(collected_environment_steps, bool)
            or not isinstance(collected_environment_steps, int)
            or collected_environment_steps < 0
        ):
            raise RecurrentPPOUpdateError(
                "collected_environment_steps must be a non-negative integer"
            )
        if (
            self.config.training.mappo.entropy_coefficient_schedule_enabled
            and collected_environment_steps is None
        ):
            raise RecurrentPPOUpdateError(
                "enabled route entropy schedule requires collected environment steps"
            )

        minibatch = build_recurrent_ppo_minibatch(buffer, self.config)
        prepared = _prepare_device_minibatch(minibatch, self.device, self.dtype)
        fixed_targets: dict[str, Tensor] = {
            "old_joint_log_prob": prepared.old_joint_log_prob.detach().clone(),
            "old_value": prepared.old_value.detach().clone(),
            "advantage": prepared.advantage.detach().clone(),
            "return_target": prepared.return_target.detach().clone(),
        }
        if prepared.route_advantage is not None:
            fixed_targets["route_advantage"] = (
                prepared.route_advantage.detach().clone()
            )
        mappo = self.config.training.mappo
        epoch_diagnostics: list[RecurrentPPOEpochDiagnostics] = []
        for epoch_index in range(mappo.update_epochs):
            self.optimizers.actor_optimizer.zero_grad(
                set_to_none=mappo.zero_grad_set_to_none
            )
            self.optimizers.critic_optimizer.zero_grad(
                set_to_none=mappo.zero_grad_set_to_none
            )
            evaluation = self._evaluate_prepared(prepared)
            new_branch_log_probs = None
            if ActorRatioMode(mappo.actor_ratio_mode) is ActorRatioMode.BRANCH_SPECIFIC:
                new_branch_log_probs = torch.stack(
                    [
                        evaluation.policy.branch_log_probs[branch]
                        for branch in ACTION_BRANCH_ORDER
                    ],
                    dim=-1,
                )
            loss = compute_configured_batched_ppo_objective_and_loss(
                new_joint_log_prob=evaluation.policy.joint_log_prob,
                old_joint_log_prob=prepared.old_joint_log_prob,
                advantage=prepared.advantage,
                current_value=evaluation.current_value,
                return_target=prepared.return_target,
                entropy=evaluation.policy.entropy,
                sequence_valid_mask=prepared.sequence_valid_mask,
                config=self.config,
                new_branch_log_probs=new_branch_log_probs,
                old_branch_log_probs=prepared.old_branch_log_probs,
                active_branch_indicators=prepared.active_branch_indicators,
                route_entropy=evaluation.policy.branch_entropies["route"],
                collected_environment_steps=collected_environment_steps,
                route_advantage=prepared.route_advantage,
            )
            loss.total_loss.backward()
            route_telemetry = None
            if self.route_telemetry_enabled:
                route_telemetry = collect_route_telemetry(
                    policy=evaluation.policy,
                    action_mask_batch=prepared.action_mask_batch,
                    proposals=prepared.proposals,
                    advantage=(
                        prepared.route_advantage
                        if prepared.route_advantage is not None
                        else prepared.advantage
                    ),
                    return_target=prepared.return_target,
                    sequence_valid_mask=prepared.sequence_valid_mask,
                    route_head=self.actor.action_heads["route"],
                    old_branch_log_probs=prepared.old_branch_log_probs,
                    td_residual=prepared.td_residual,
                    shared_trunk=self.actor,
                    epsilon_clip=self.config.training.mappo.ppo_clip_epsilon,
                    actor_ratio_mode=mappo.actor_ratio_mode,
                    base_advantage=prepared.advantage,
                    route_advantage=prepared.route_advantage,
                    branch_advantage=loss.branch_advantage,
                    bootstrap_mask=prepared.bootstrap_mask,
                    effective_route_gae_lambda=(
                        mappo.route_gae_lambda
                        if RouteCreditMode(mappo.route_credit_mode)
                        is RouteCreditMode.ROUTE_SPECIFIC_GAE
                        else mappo.gae_lambda
                    ),
                )
            agent_credit_telemetry = None
            if (
                AgentCreditMode(mappo.agent_credit_mode)
                is AgentCreditMode.ROLE_DECOMPOSED
            ):
                if prepared.td_residual is None:
                    raise RecurrentPPOUpdateError(
                        "role-decomposed update requires per-agent TD residual"
                    )
                agent_credit_telemetry = _collect_agent_credit_epoch_telemetry(
                    current_value=evaluation.current_value,
                    td_residual=prepared.td_residual,
                    advantage=prepared.advantage,
                    return_target=prepared.return_target,
                    sequence_valid_mask=prepared.sequence_valid_mask,
                    route_telemetry=route_telemetry,
                    per_head_critic_loss=loss.diagnostics.per_head_critic_loss,
                    same_timestep_advantage_equality_rate=(
                        loss.diagnostics.same_timestep_advantage_equality_rate
                    ),
                )
            actor_grad_norm = self._finite_grad_norm(
                self.optimizers.actor_parameters,
                mappo.gradient_clip_norm,
                "actor",
            )
            critic_grad_norm = self._finite_grad_norm(
                self.optimizers.critic_parameters,
                mappo.gradient_clip_norm,
                "critic",
            )
            self.optimizers.actor_optimizer.step()
            self.optimizers.critic_optimizer.step()
            for name, expected in fixed_targets.items():
                if not torch.equal(getattr(prepared, name), expected):
                    raise RecurrentPPOUpdateError(
                        f"fixed rollout target {name} changed during PPO epochs"
                    )
            epoch_diagnostics.append(
                RecurrentPPOEpochDiagnostics(
                    epoch_index=epoch_index,
                    actor_loss=float(loss.actor_loss.detach().cpu().item()),
                    critic_loss=float(loss.critic_loss.detach().cpu().item()),
                    entropy_mean=float(loss.entropy_mean.detach().cpu().item()),
                    total_loss=float(loss.total_loss.detach().cpu().item()),
                    ratio_mean=loss.diagnostics.ratio_mean,
                    actor_grad_norm_before_clip=actor_grad_norm,
                    critic_grad_norm_before_clip=critic_grad_norm,
                    clip_max_norm=mappo.gradient_clip_norm,
                    approx_kl=loss.diagnostics.approx_kl,
                    clip_fraction=loss.diagnostics.clipped_fraction,
                    route_entropy_coefficient=(
                        None
                        if collected_environment_steps is None
                        else loss.route_entropy_coefficient
                    ),
                    route_entropy_schedule_progress=(
                        None
                        if collected_environment_steps is None
                        else loss.route_entropy_schedule_progress
                    ),
                    route_entropy_loss_contribution=(
                        None
                        if collected_environment_steps is None
                        else float(
                            loss.route_entropy_loss_contribution.detach().cpu().item()
                        )
                    ),
                    other_branch_entropy_loss_contribution=(
                        None
                        if collected_environment_steps is None
                        else float(
                            loss.other_branch_entropy_loss_contribution.detach()
                            .cpu()
                            .item()
                        )
                    ),
                    global_entropy_loss_contribution=(
                        None
                        if collected_environment_steps is None
                        else float(
                            loss.global_entropy_loss_contribution.detach().cpu().item()
                        )
                    ),
                    collected_environment_steps=collected_environment_steps,
                    route_telemetry=route_telemetry,
                    agent_credit_telemetry=agent_credit_telemetry,
                )
            )
        return RecurrentPPOUpdateOutput(
            epoch_diagnostics=tuple(epoch_diagnostics),
            chunk_count=8,
            chunk_length=32,
            valid_transition_count=256,
            old_policy_snapshot_preserved=True,
        )


__all__ = [
    "AgentCreditPPOEpochTelemetry",
    "BatchedCAGATMAPPOLossOutput",
    "CAGATMAPPOOptimizerBundle",
    "CAGATMAPPORecurrentPPOUpdater",
    "RecurrentPPOEpochDiagnostics",
    "RecurrentPPOMinibatch",
    "RecurrentPPOEvaluation",
    "RecurrentPPOUpdateError",
    "RecurrentPPOUpdateOutput",
    "build_ca_gat_mappo_optimizers",
    "build_recurrent_ppo_minibatch",
    "compute_configured_batched_ppo_objective_and_loss",
]
