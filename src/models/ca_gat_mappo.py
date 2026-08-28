"""CA-GAT-MAPPO networks and tensor adapters for Stage 07B-1.

This module stops at deterministic neural-network forward interfaces. It does
not sample actions, create rollout storage, calculate GAE/PPO losses, own an
optimizer, or run training. Actor tensors are built only from the existing
ActorObservation boundary; centralized tensors are built separately from the
existing decision-time CentralizedState boundary.

Public layouts keep batch, time, and agent axes explicit:
actor features [B,T,A,F], graph features [B,T,A,N,F], logits [B,T,A,D],
and recurrent hidden input/output [B,A,H]. The critic consumes [B,T,S] and
returns [B,T,1].
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from ..config import AgentCreditMode, RunConfig, STREAM_IDS
from ..env.actions import ActionProposal
from ..env.observation import ActorObservation, BRANCH_ORDER, QueueSummary
from ..env.state import CentralizedState


class MAPPONetworkError(ValueError):
    """Raised when a Stage 07B-1 tensor violates the frozen interface."""


ACTION_BRANCH_ORDER = tuple(BRANCH_ORDER)
ACTION_BRANCH_DEPENDENCIES: Mapping[str, tuple[str, ...]] = {
    "route": (),
    "tx_select": (),
    "resource_group": ("tx_select",),
    "resource_width": ("tx_select", "resource_group"),
    "power_level": ("tx_select", "resource_group", "resource_width"),
    "cpu_queue": (),
    "cpu_frequency": ("cpu_queue",),
}
_ACTIVE_TASK_STATUSES = ("unbound", "local", "tx", "cpu")
_QUEUE_KINDS = ("unbound", "local", "tx", "cpu")


def _derived_torch_seed(master_seed: int) -> int:
    """Derive Torch initialization from the frozen stream ID 100."""

    if isinstance(master_seed, bool) or not isinstance(master_seed, int) or master_seed < 0:
        raise MAPPONetworkError("master seed must be a non-negative integer")
    state = np.random.SeedSequence(
        [master_seed, STREAM_IDS["torch_initialization"]]
    ).generate_state(1, dtype=np.uint64)
    return int(state[0] % np.uint64(np.iinfo(np.int64).max))


def _action_dimensions(config: RunConfig) -> dict[str, int]:
    count = config.environment.uav_count
    return {
        "route": count + 2,
        "tx_select": count,
        "resource_group": config.environment.resource_group_count + 1,
        "resource_width": len(config.action.resource_width_options),
        "power_level": len(config.action.power_levels),
        "cpu_queue": count + 1,
        "cpu_frequency": len(config.action.cpu_frequency_levels),
    }


@dataclass(frozen=True)
class MAPPOTensorSpec:
    """Dimensions derived from the environment schema and MAPPO defaults."""

    uav_count: int
    ru_count: int
    self_feature_dim: int
    neighbor_public_feature_dim: int
    edge_feature_dim: int
    centralized_state_dim: int
    action_dimensions: Mapping[str, int]
    encoder_hidden_dimension: int
    gat_layer_count: int
    attention_head_count: int
    gru_hidden_dimension: int

    @classmethod
    def from_config(cls, config: RunConfig) -> "MAPPOTensorSpec":
        if not isinstance(config, RunConfig):
            raise TypeError("config must be a RunConfig")
        config.validate()
        env = config.environment
        model = config.training.mappo
        if tuple(config.action.sampling_order) != ACTION_BRANCH_ORDER:
            raise MAPPONetworkError("config action order differs from the frozen order")
        if model.gat_layer_count != 1:
            raise MAPPONetworkError("Stage 07B-1 requires one CA-GATv2 layer")
        if model.encoder_hidden_dimension % model.attention_head_count != 0:
            raise MAPPONetworkError("encoder width must be divisible by attention heads")
        if model.gru_hidden_dimension != model.encoder_hidden_dimension:
            raise MAPPONetworkError("graph and GRU widths must match the frozen 128 interface")

        count = env.uav_count
        ru_count = env.ru_count
        action_dimensions = _action_dimensions(config)
        previous_dim = sum(action_dimensions.values()) + 2 * count + 7
        self_dim = 9 + 16 + 16 * count + 2 + 2 * count + previous_dim
        public_dim = 17
        edge_dim = 10 + 6 * ru_count

        fixed_state_dim = (
            1
            + 13 * count
            + 7 * count * count
            + 3 * count * ru_count
            + 6 * count * count * ru_count
        )
        aggregate_state_dim = 10 * count + 5 * count * count
        building_dim = 6 * len(env.building_layout)
        return cls(
            uav_count=count,
            ru_count=ru_count,
            self_feature_dim=self_dim,
            neighbor_public_feature_dim=public_dim,
            edge_feature_dim=edge_dim,
            centralized_state_dim=fixed_state_dim + aggregate_state_dim + building_dim,
            action_dimensions=action_dimensions,
            encoder_hidden_dimension=model.encoder_hidden_dimension,
            gat_layer_count=model.gat_layer_count,
            attention_head_count=model.attention_head_count,
            gru_hidden_dimension=model.gru_hidden_dimension,
        )


@dataclass(frozen=True)
class ActorTensorBatch:
    """One explicit recurrent actor tensor batch."""

    self_features: Tensor
    neighbor_public_features: Tensor
    edge_features: Tensor
    neighbor_mask: Tensor
    action_masks: Mapping[str, Tensor]
    action_indices: Mapping[str, Tensor]
    episode_starts: Tensor

    def validate(self, spec: MAPPOTensorSpec) -> tuple[int, int, int]:
        if self.self_features.ndim != 4:
            raise MAPPONetworkError("self_features must be [batch,time,agent,feature]")
        batch, time, agents, features = self.self_features.shape
        if agents != spec.uav_count or features != spec.self_feature_dim:
            raise MAPPONetworkError("self_features differ from the configured tensor spec")
        if tuple(self.neighbor_public_features.shape) != (
            batch,
            time,
            agents,
            spec.uav_count,
            spec.neighbor_public_feature_dim,
        ):
            raise MAPPONetworkError("neighbor_public_features have an invalid shape")
        if tuple(self.edge_features.shape) != (
            batch,
            time,
            agents,
            spec.uav_count,
            spec.edge_feature_dim,
        ):
            raise MAPPONetworkError("edge_features have an invalid shape")
        if tuple(self.neighbor_mask.shape) != (batch, time, agents, spec.uav_count):
            raise MAPPONetworkError("neighbor_mask has an invalid shape")
        if tuple(self.episode_starts.shape) != (batch, time, agents):
            raise MAPPONetworkError("episode_starts must be [batch,time,agent]")
        for name, tensor in (
            ("self_features", self.self_features),
            ("neighbor_public_features", self.neighbor_public_features),
            ("edge_features", self.edge_features),
        ):
            if not tensor.is_floating_point() or not torch.isfinite(tensor).all():
                raise MAPPONetworkError(f"{name} must be a finite floating tensor")
        if self.neighbor_mask.dtype != torch.bool or self.episode_starts.dtype != torch.bool:
            raise MAPPONetworkError("graph and episode masks must be boolean")
        if tuple(self.action_masks) != ACTION_BRANCH_ORDER:
            raise MAPPONetworkError("action_masks must contain exactly seven ordered branches")
        if tuple(self.action_indices) != ACTION_BRANCH_ORDER:
            raise MAPPONetworkError("action_indices must contain exactly seven ordered branches")
        for branch in ACTION_BRANCH_ORDER:
            dimension = spec.action_dimensions[branch]
            mask = self.action_masks[branch]
            indices = self.action_indices[branch]
            if tuple(mask.shape) != (batch, time, agents, dimension):
                raise MAPPONetworkError(f"{branch} mask has an invalid shape")
            if mask.dtype != torch.bool or not torch.all(torch.any(mask, dim=-1)):
                raise MAPPONetworkError(f"{branch} mask has no legal fallback")
            if tuple(indices.shape) != (batch, time, agents) or indices.dtype != torch.long:
                raise MAPPONetworkError(f"{branch} indices must be long [batch,time,agent]")
            if torch.any(indices < 0) or torch.any(indices >= dimension):
                raise MAPPONetworkError(f"{branch} index lies outside its domain")
            legal = torch.gather(mask, -1, indices.unsqueeze(-1)).squeeze(-1)
            if not torch.all(legal):
                raise MAPPONetworkError(f"{branch} action context violates its mask")
        return batch, time, agents

    def to(
        self,
        device: torch.device | str,
        *,
        dtype: torch.dtype | None = None,
    ) -> "ActorTensorBatch":
        target_dtype = self.self_features.dtype if dtype is None else dtype
        return ActorTensorBatch(
            self_features=self.self_features.to(device=device, dtype=target_dtype),
            neighbor_public_features=self.neighbor_public_features.to(
                device=device, dtype=target_dtype
            ),
            edge_features=self.edge_features.to(device=device, dtype=target_dtype),
            neighbor_mask=self.neighbor_mask.to(device=device),
            action_masks={key: value.to(device=device) for key, value in self.action_masks.items()},
            action_indices={
                key: value.to(device=device) for key, value in self.action_indices.items()
            },
            episode_starts=self.episode_starts.to(device=device),
        )

    def repeat_batch(self, repeats: int) -> "ActorTensorBatch":
        if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats <= 0:
            raise MAPPONetworkError("repeats must be a positive integer")

        def repeat(tensor: Tensor) -> Tensor:
            return tensor.repeat((repeats,) + (1,) * (tensor.ndim - 1))

        return ActorTensorBatch(
            self_features=repeat(self.self_features),
            neighbor_public_features=repeat(self.neighbor_public_features),
            edge_features=repeat(self.edge_features),
            neighbor_mask=repeat(self.neighbor_mask),
            action_masks={key: repeat(value) for key, value in self.action_masks.items()},
            action_indices={key: repeat(value) for key, value in self.action_indices.items()},
            episode_starts=repeat(self.episode_starts),
        )


def stack_actor_time(steps: Sequence[ActorTensorBatch]) -> ActorTensorBatch:
    """Concatenate step batches along the explicit time dimension."""

    copied = tuple(steps)
    if not copied:
        raise MAPPONetworkError("at least one actor step is required")
    return ActorTensorBatch(
        self_features=torch.cat([item.self_features for item in copied], dim=1),
        neighbor_public_features=torch.cat(
            [item.neighbor_public_features for item in copied], dim=1
        ),
        edge_features=torch.cat([item.edge_features for item in copied], dim=1),
        neighbor_mask=torch.cat([item.neighbor_mask for item in copied], dim=1),
        action_masks={
            branch: torch.cat([item.action_masks[branch] for item in copied], dim=1)
            for branch in ACTION_BRANCH_ORDER
        },
        action_indices={
            branch: torch.cat([item.action_indices[branch] for item in copied], dim=1)
            for branch in ACTION_BRANCH_ORDER
        },
        episode_starts=torch.cat([item.episode_starts for item in copied], dim=1),
    )


@dataclass(frozen=True)
class CentralizedStateTensorBatch:
    """Deterministic centralized critic tensor batch."""

    features: Tensor

    def validate(self, spec: MAPPOTensorSpec) -> tuple[int, int]:
        if self.features.ndim != 3:
            raise MAPPONetworkError("centralized features must be [batch,time,feature]")
        batch, time, dimension = self.features.shape
        if dimension != spec.centralized_state_dim:
            raise MAPPONetworkError("centralized feature dimension differs from tensor spec")
        if not self.features.is_floating_point() or not torch.isfinite(self.features).all():
            raise MAPPONetworkError("centralized features must be finite floating tensors")
        return batch, time

    def to(
        self,
        device: torch.device | str,
        *,
        dtype: torch.dtype | None = None,
    ) -> "CentralizedStateTensorBatch":
        target_dtype = self.features.dtype if dtype is None else dtype
        return CentralizedStateTensorBatch(
            self.features.to(device=device, dtype=target_dtype)
        )

    def repeat_batch(self, repeats: int) -> "CentralizedStateTensorBatch":
        if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats <= 0:
            raise MAPPONetworkError("repeats must be a positive integer")
        return CentralizedStateTensorBatch(self.features.repeat(repeats, 1, 1))


def stack_centralized_time(
    steps: Sequence[CentralizedStateTensorBatch],
) -> CentralizedStateTensorBatch:
    copied = tuple(steps)
    if not copied:
        raise MAPPONetworkError("at least one centralized state is required")
    return CentralizedStateTensorBatch(
        torch.cat([item.features for item in copied], dim=1)
    )

class ActorObservationTensorizer:
    """Whitelist-only adapter from ActorObservation to Torch tensors."""

    def __init__(self, config: RunConfig) -> None:
        if not isinstance(config, RunConfig):
            raise TypeError("config must be a RunConfig")
        config.validate()
        self.config = config
        self.spec = MAPPOTensorSpec.from_config(config)
        env = config.environment
        self._max_tasks = max(1, env.uav_count * env.episode_horizon)
        self._max_task_bits = float(env.task_data_bits_max)
        self._max_task_cycles = float(env.task_data_bits_max * env.task_cycles_per_bit_max)
        self._max_queue_bits = self._max_tasks * self._max_task_bits
        self._max_queue_cycles = self._max_tasks * self._max_task_cycles
        self._time_scale = float(max(1, env.episode_horizon))
        self._slack_scale = float(max(1, env.maximum_task_slack_slots))
        self._distance_scale = math.sqrt(
            (env.bounds.x_max_m - env.bounds.x_min_m) ** 2
            + (env.bounds.y_max_m - env.bounds.y_min_m) ** 2
            + env.height_m**2
        )
        noise_psd_w_hz = 10.0 ** ((env.noise_psd_dbm_hz - 30.0) / 10.0)
        self._noise_reference_w = max(
            np.finfo(np.float64).tiny,
            noise_psd_w_hz
            * env.ru_bandwidth_hz
            * 10.0 ** (env.receiver_noise_figure_db / 10.0),
        )

    def encode_step(
        self,
        observations: Sequence[ActorObservation],
        proposals: Sequence[ActionProposal] | None = None,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        episode_start: bool | None = None,
    ) -> ActorTensorBatch:
        copied = tuple(observations)
        count = self.spec.uav_count
        if len(copied) != count or any(
            not isinstance(item, ActorObservation) for item in copied
        ):
            raise MAPPONetworkError("one ActorObservation per configured UAV is required")
        if tuple(item.uav_id for item in copied) != tuple(range(count)):
            raise MAPPONetworkError("observations must use stable zero-based UAV order")
        if len({item.slot for item in copied}) != 1:
            raise MAPPONetworkError("all observations must describe one decision slot")
        selected = (
            tuple(proposals)
            if proposals is not None
            else tuple(self._first_legal_proposal(item) for item in copied)
        )
        if len(selected) != count:
            raise MAPPONetworkError("proposals must align with observations")
        for observation, proposal in zip(copied, selected):
            if not observation.action_masks.is_legal(proposal):
                raise MAPPONetworkError("proposal violates its observation masks")

        self_values = np.stack([self._self_vector(item) for item in copied])
        public_values = np.stack([self._public_matrix(item) for item in copied])
        edge_values = np.stack([self._edge_matrix(item) for item in copied])
        graph_masks = np.stack([item.candidate_neighbor_mask for item in copied])

        action_masks: dict[str, Tensor] = {}
        action_indices: dict[str, Tensor] = {}
        for branch in ACTION_BRANCH_ORDER:
            masks = np.stack(
                [
                    observation.action_masks.mask_for(branch, proposal)
                    for observation, proposal in zip(copied, selected)
                ]
            )
            indices = np.asarray(
                [
                    self._branch_index(
                        observation, branch, getattr(proposal, branch)
                    )
                    for observation, proposal in zip(copied, selected)
                ],
                dtype=np.int64,
            )
            action_masks[branch] = torch.as_tensor(
                masks[None, None], dtype=torch.bool, device=device
            )
            action_indices[branch] = torch.as_tensor(
                indices[None, None], dtype=torch.long, device=device
            )

        start = copied[0].slot == 0 if episode_start is None else episode_start
        if not isinstance(start, (bool, np.bool_)):
            raise MAPPONetworkError("episode_start must be boolean")
        result = ActorTensorBatch(
            self_features=torch.as_tensor(
                self_values[None, None], dtype=dtype, device=device
            ),
            neighbor_public_features=torch.as_tensor(
                public_values[None, None], dtype=dtype, device=device
            ),
            edge_features=torch.as_tensor(
                edge_values[None, None], dtype=dtype, device=device
            ),
            neighbor_mask=torch.as_tensor(
                graph_masks[None, None], dtype=torch.bool, device=device
            ),
            action_masks=action_masks,
            action_indices=action_indices,
            episode_starts=torch.full(
                (1, 1, count), bool(start), dtype=torch.bool, device=device
            ),
        )
        result.validate(self.spec)
        return result

    def _self_vector(self, observation: ActorObservation) -> np.ndarray:
        env = self.config.environment
        resources = observation.self_resources
        values: list[float] = [
            resources.residual_energy_j / env.reference_initial_energy_j,
            resources.initial_energy_j / env.reference_initial_energy_j,
            resources.residual_energy_ratio,
            resources.max_transmit_power_w / env.reference_transmit_power_w,
            resources.max_transmit_power_ratio,
            resources.max_cpu_frequency_hz / env.reference_cpu_frequency_hz,
            resources.max_cpu_frequency_ratio,
            resources.cpu_coefficient / env.reference_cpu_coefficient,
            resources.cpu_coefficient_ratio,
        ]
        queues = observation.private_queues
        values.extend(self._queue_vector(queues.unbound))
        values.extend(self._queue_vector(queues.local_cpu))
        for indexed in (queues.tx_by_destination, queues.cpu_by_source):
            for index in range(self.spec.uav_count):
                values.extend(
                    self._queue_vector(
                        QueueSummary(
                            task_count=int(indexed.task_count[index]),
                            remaining_bits=float(indexed.remaining_bits[index]),
                            remaining_cycles=float(indexed.remaining_cycles[index]),
                            head_task_id=int(indexed.head_task_id[index]),
                            head_remaining_bits=float(indexed.head_remaining_bits[index]),
                            head_remaining_cycles=float(indexed.head_remaining_cycles[index]),
                            head_slack_slots=int(indexed.head_slack_slots[index]),
                            head_valid_mask=bool(indexed.head_valid_mask[index]),
                        )
                    )
                )
        values.extend(
            [
                float(observation.arrival_rate_estimate),
                float(observation.arrival_rate_valid_mask),
            ]
        )
        values.extend(observation.candidate_neighbor_mask.astype(np.float64).tolist())
        values.extend(observation.service_queue_available.astype(np.float64).tolist())
        values.extend(self._previous_action_vector(observation))
        result = np.asarray(values, dtype=np.float64)
        if result.shape != (self.spec.self_feature_dim,):
            raise MAPPONetworkError("self-feature encoder produced the wrong dimension")
        return result

    def _queue_vector(self, summary: QueueSummary) -> list[float]:
        return [
            summary.task_count / self._max_tasks,
            summary.remaining_bits / self._max_queue_bits,
            summary.remaining_cycles / self._max_queue_cycles,
            (summary.head_task_id + 1) / self._max_tasks
            if summary.head_valid_mask
            else 0.0,
            summary.head_remaining_bits / self._max_task_bits,
            summary.head_remaining_cycles / self._max_task_cycles,
            summary.head_slack_slots / self._slack_scale
            if summary.head_valid_mask
            else 0.0,
            float(summary.head_valid_mask),
        ]

    def _previous_action_vector(self, observation: ActorObservation) -> list[float]:
        previous = observation.previous_action
        values: list[float] = [float(previous.valid)]
        for branch in ACTION_BRANCH_ORDER:
            domain = observation.action_masks.domain_for(branch)
            index = self._branch_index(
                observation, branch, getattr(previous.proposal, branch)
            )
            one_hot = [0.0] * len(domain)
            one_hot[index] = 1.0
            values.extend(one_hot)
        values.append(float(previous.communication_accepted))
        receiver = [0.0] * self.spec.uav_count
        if previous.communication_receiver_uav is not None:
            receiver[previous.communication_receiver_uav] = 1.0
        values.extend(receiver)
        values.extend(
            [
                previous.executed_ru_count / max(1, self.spec.ru_count),
                previous.ru_utilization_ratio,
                previous.executed_power_ratio,
            ]
        )
        cpu_source = [0.0] * self.spec.uav_count
        if previous.cpu_source_uav is not None:
            cpu_source[previous.cpu_source_uav] = 1.0
        values.extend(cpu_source)
        values.extend(
            [float(previous.cpu_active), previous.executed_cpu_frequency_ratio]
        )
        return values

    def _public_matrix(self, observation: ActorObservation) -> np.ndarray:
        env = self.config.environment
        public = observation.neighbor_public
        span_x = max(1.0, env.bounds.x_max_m - env.bounds.x_min_m)
        span_y = max(1.0, env.bounds.y_max_m - env.bounds.y_min_m)
        velocity_scale = np.asarray(
            [
                max(1.0, abs(env.velocity_std_mps[0])),
                max(1.0, abs(env.velocity_std_mps[1])),
                1.0,
            ],
            dtype=np.float64,
        )
        result = np.zeros(
            (self.spec.uav_count, self.spec.neighbor_public_feature_dim),
            dtype=np.float64,
        )
        for neighbor in range(self.spec.uav_count):
            result[neighbor] = np.asarray(
                [
                    float(public.valid_mask[neighbor]),
                    public.relative_position_m[neighbor, 0] / span_x,
                    public.relative_position_m[neighbor, 1] / span_y,
                    public.relative_position_m[neighbor, 2] / max(1.0, env.height_m),
                    *(public.relative_velocity_mps[neighbor] / velocity_scale),
                    public.residual_energy_j[neighbor] / env.reference_initial_energy_j,
                    public.residual_energy_ratio[neighbor],
                    public.max_transmit_power_ratio[neighbor],
                    public.max_cpu_frequency_ratio[neighbor],
                    public.cpu_coefficient_ratio[neighbor],
                    public.cpu_load_task_count[neighbor] / self._max_tasks,
                    public.cpu_load_remaining_cycles[neighbor] / self._max_queue_cycles,
                    (
                        (public.message_source_slots[neighbor] + 1) / self._time_scale
                        if public.valid_mask[neighbor]
                        else 0.0
                    ),
                    (
                        (public.message_aoi_slots[neighbor] + 1) / self._time_scale
                        if public.message_aoi_valid_mask[neighbor]
                        else 0.0
                    ),
                    float(public.message_aoi_valid_mask[neighbor]),
                ],
                dtype=np.float64,
            )
        return result
    def _edge_matrix(self, observation: ActorObservation) -> np.ndarray:
        edge = observation.edge_history
        result = np.zeros(
            (self.spec.uav_count, self.spec.edge_feature_dim), dtype=np.float64
        )
        for neighbor in range(self.spec.uav_count):
            stale = edge.stale_csi[neighbor]
            row: list[float] = [
                float(edge.visible_mask[neighbor]),
                edge.estimated_distance_m[neighbor] / self._distance_scale,
            ]
            row.extend(stale.real.tolist())
            row.extend(stale.imag.tolist())
            row.extend(
                [
                    float(edge.csi_valid_mask[neighbor]),
                    edge.csi_aoi_slots[neighbor] / self._time_scale,
                ]
            )
            row.extend(
                (edge.interference_history_w[neighbor] / self._noise_reference_w).tolist()
            )
            row.extend(
                edge.interference_valid_mask[neighbor].astype(np.float64).tolist()
            )
            row.extend(
                [
                    (
                        (edge.message_aoi_slots[neighbor] + 1) / self._time_scale
                        if edge.message_aoi_valid_mask[neighbor]
                        else 0.0
                    ),
                    float(edge.message_aoi_valid_mask[neighbor]),
                ]
            )
            row.extend(edge.historical_quality[neighbor].tolist())
            row.extend(edge.quality_valid_mask[neighbor].astype(np.float64).tolist())
            row.extend(
                [
                    edge.last_effective_rate_bps[neighbor]
                    / self.config.environment.reference_rate_bps,
                    float(edge.last_rate_valid_mask[neighbor]),
                    edge.outage_rate[neighbor],
                    float(edge.outage_valid_mask[neighbor]),
                ]
            )
            result[neighbor] = np.asarray(row, dtype=np.float64)
        return result

    def _first_legal_proposal(self, observation: ActorObservation) -> ActionProposal:
        values: dict[str, str | int | float] = {}
        for branch in ACTION_BRANCH_ORDER:
            valid = np.flatnonzero(
                observation.action_masks.mask_for(branch, values)
            )
            if valid.size == 0:
                raise MAPPONetworkError(f"{branch} has no legal action")
            values[branch] = observation.action_masks.domain_for(branch)[int(valid[0])]
        return ActionProposal(
            uav_id=observation.uav_id,
            route=values["route"],
            tx_select=values["tx_select"],
            resource_group=values["resource_group"],
            resource_width=int(values["resource_width"]),
            power_level=float(values["power_level"]),
            cpu_queue=values["cpu_queue"],
            cpu_frequency=float(values["cpu_frequency"]),
        )

    @staticmethod
    def _branch_index(
        observation: ActorObservation,
        branch: str,
        raw_value: object,
    ) -> int:
        value = raw_value
        if branch == "cpu_queue" and value == "local":
            value = observation.uav_id
        if branch in {"power_level", "cpu_frequency"} and isinstance(value, (int, float)):
            value = float(value)
        for index, candidate in enumerate(
            observation.action_masks.domain_for(branch)
        ):
            if isinstance(candidate, float) and isinstance(value, (int, float)):
                if not isinstance(value, bool) and candidate == float(value):
                    return index
            elif type(candidate) is type(value) and candidate == value:
                return index
        raise MAPPONetworkError(f"{branch} value is outside the observation domain")


class CentralizedStateTensorizer:
    """Adapter from the official pre-action CentralizedState interface."""

    def __init__(self, config: RunConfig) -> None:
        if not isinstance(config, RunConfig):
            raise TypeError("config must be a RunConfig")
        config.validate()
        self.config = config
        self.spec = MAPPOTensorSpec.from_config(config)
        env = config.environment
        self._max_tasks = max(1, env.uav_count * env.episode_horizon)
        self._max_task_bits = float(env.task_data_bits_max)
        self._max_task_cycles = float(env.task_data_bits_max * env.task_cycles_per_bit_max)
        self._time_scale = float(max(1, env.episode_horizon))
        self._distance_scale = math.sqrt(
            (env.bounds.x_max_m - env.bounds.x_min_m) ** 2
            + (env.bounds.y_max_m - env.bounds.y_min_m) ** 2
            + env.height_m**2
        )
        noise_psd_w_hz = 10.0 ** ((env.noise_psd_dbm_hz - 30.0) / 10.0)
        self._noise_reference_w = max(
            np.finfo(np.float64).tiny,
            noise_psd_w_hz
            * env.ru_bandwidth_hz
            * 10.0 ** (env.receiver_noise_figure_db / 10.0),
        )

    def encode_step(
        self,
        state: CentralizedState,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> CentralizedStateTensorBatch:
        vector = self._state_vector(state)
        result = CentralizedStateTensorBatch(
            torch.as_tensor(vector[None, None], dtype=dtype, device=device)
        )
        result.validate(self.spec)
        return result

    def _state_vector(self, state: CentralizedState) -> np.ndarray:
        if not isinstance(state, CentralizedState):
            raise TypeError("state must be a CentralizedState")
        count = self.spec.uav_count
        env = self.config.environment
        if state.uav_ids.shape != (count,) or state.true_channel.shape != (
            count,
            count,
            self.spec.ru_count,
        ):
            raise MAPPONetworkError("centralized state differs from configured scenario")

        span_x = max(1.0, env.bounds.x_max_m - env.bounds.x_min_m)
        span_y = max(1.0, env.bounds.y_max_m - env.bounds.y_min_m)
        positions = state.positions_m.copy()
        positions[:, 0] = (positions[:, 0] - env.bounds.x_min_m) / span_x
        positions[:, 1] = (positions[:, 1] - env.bounds.y_min_m) / span_y
        positions[:, 2] = positions[:, 2] / max(1.0, env.height_m)
        velocity_scale = np.asarray(
            [
                max(1.0, abs(env.velocity_std_mps[0])),
                max(1.0, abs(env.velocity_std_mps[1])),
                1.0,
            ]
        )

        values: list[float] = [state.slot / self._time_scale]
        values.extend(positions.reshape(-1).tolist())
        values.extend((state.velocities_mps / velocity_scale).reshape(-1).tolist())
        values.extend(
            (state.max_transmit_power_w / env.reference_transmit_power_w).tolist()
        )
        values.extend(
            (state.max_cpu_frequency_hz / env.reference_cpu_frequency_hz).tolist()
        )
        values.extend((state.cpu_coefficient / env.reference_cpu_coefficient).tolist())
        values.extend((state.residual_energy_j / env.reference_initial_energy_j).tolist())
        values.extend(state.true_arrival_probabilities.tolist())
        values.extend(state.candidate_neighbor_mask.astype(float).reshape(-1).tolist())
        values.extend((state.distances_m / self._distance_scale).reshape(-1).tolist())
        values.extend(state.true_channel.real.reshape(-1).tolist())
        values.extend(state.true_channel.imag.reshape(-1).tolist())
        values.extend(state.blocked_links.astype(float).reshape(-1).tolist())
        values.extend(
            (state.shadowing_db / max(1.0, env.shadowing_std_db)).reshape(-1).tolist()
        )
        values.extend(state.path_loss_db.reshape(-1).tolist())
        values.extend(state.stale_csi.real.reshape(-1).tolist())
        values.extend(state.stale_csi.imag.reshape(-1).tolist())
        values.extend(state.csi_valid_mask.astype(float).reshape(-1).tolist())
        values.extend((state.csi_aoi_slots / self._time_scale).reshape(-1).tolist())
        values.extend(
            (state.interference_history_w / self._noise_reference_w).reshape(-1).tolist()
        )
        values.extend(state.interference_valid_mask.astype(float).reshape(-1).tolist())
        values.extend(((state.message_aoi_slots + 1) / self._time_scale).tolist())
        values.extend(state.message_aoi_valid_mask.astype(float).tolist())
        values.extend(
            (state.historical_denominator_w / self._noise_reference_w).reshape(-1).tolist()
        )
        values.extend(state.historical_quality.reshape(-1).tolist())
        values.extend(state.quality_valid_mask.astype(float).reshape(-1).tolist())
        values.extend(self._task_aggregates(state))
        values.extend(self._queue_aggregates(state))
        if len(state.buildings) != len(env.building_layout):
            raise MAPPONetworkError("building snapshot differs from config")
        for building in state.buildings:
            values.extend(
                [
                    1.0,
                    (building.x_min_m - env.bounds.x_min_m) / span_x,
                    (building.x_max_m - env.bounds.x_min_m) / span_x,
                    (building.y_min_m - env.bounds.y_min_m) / span_y,
                    (building.y_max_m - env.bounds.y_min_m) / span_y,
                    building.height_m / max(1.0, env.height_m),
                ]
            )

        result = np.asarray(values, dtype=np.float64)
        if result.shape != (self.spec.centralized_state_dim,):
            raise MAPPONetworkError("centralized encoder produced the wrong dimension")
        if not np.all(np.isfinite(result)):
            raise MAPPONetworkError("centralized encoder produced NaN or Inf")
        return result

    def _task_aggregates(self, state: CentralizedState) -> list[float]:
        count = self.spec.uav_count
        aggregate = np.zeros((count, 10), dtype=np.float64)
        destination_counts = np.zeros((count, count), dtype=np.float64)
        slack_min = np.full(count, np.inf, dtype=np.float64)
        for task in state.active_tasks:
            source = task.source_uav
            aggregate[source, 0] += 1.0
            aggregate[source, 1] += task.data_bits / self._max_task_bits
            aggregate[source, 2] += task.cpu_cycles / self._max_task_cycles
            aggregate[source, 3] += task.remaining_bits / self._max_task_bits
            aggregate[source, 4] += task.remaining_cycles / self._max_task_cycles
            slack_min[source] = min(slack_min[source], task.deadline_slot - state.slot)
            try:
                status_index = _ACTIVE_TASK_STATUSES.index(task.status)
            except ValueError as exc:
                raise MAPPONetworkError("active task has terminal or unknown status") from exc
            aggregate[source, 6 + status_index] += 1.0
            if task.destination_uav is not None:
                destination_counts[source, task.destination_uav] += 1.0
        aggregate[:, 0] /= self._max_tasks
        aggregate[:, 1:5] /= self._max_tasks
        aggregate[:, 5] = np.where(
            np.isfinite(slack_min),
            slack_min / max(1, self.config.environment.maximum_task_slack_slots),
            0.0,
        )
        aggregate[:, 6:10] /= self._max_tasks
        destination_counts /= self._max_tasks
        return [*aggregate.reshape(-1).tolist(), *destination_counts.reshape(-1).tolist()]

    def _queue_aggregates(self, state: CentralizedState) -> list[float]:
        count = self.spec.uav_count
        values = np.zeros((len(_QUEUE_KINDS), count, count), dtype=np.float64)
        for queue in state.queues:
            kind = _QUEUE_KINDS.index(queue.kind)
            destination = (
                queue.source_uav
                if queue.destination_uav is None
                else queue.destination_uav
            )
            values[kind, queue.source_uav, destination] = (
                len(queue.task_ids) / self._max_tasks
            )
        return values.reshape(-1).tolist()

class CAGATv2Layer(nn.Module):
    """One ego-centered channel/context-aware GATv2 layer."""

    def __init__(self, hidden_dimension: int, edge_dimension: int, heads: int) -> None:
        super().__init__()
        if hidden_dimension % heads != 0:
            raise MAPPONetworkError("hidden dimension must be divisible by heads")
        self.hidden_dimension = hidden_dimension
        self.edge_dimension = edge_dimension
        self.heads = heads
        self.head_dimension = hidden_dimension // heads
        self.query = nn.ModuleList(
            [nn.Linear(hidden_dimension, self.head_dimension) for _ in range(heads)]
        )
        self.edge_projection = nn.ModuleList(
            [nn.Linear(edge_dimension, self.head_dimension) for _ in range(heads)]
        )
        pair_dimension = hidden_dimension + self.head_dimension
        self.key = nn.ModuleList(
            [nn.Linear(pair_dimension, self.head_dimension) for _ in range(heads)]
        )
        self.value = nn.ModuleList(
            [nn.Linear(pair_dimension, self.head_dimension) for _ in range(heads)]
        )
        self.attention_vector = nn.Parameter(torch.empty(heads, self.head_dimension))
        nn.init.xavier_uniform_(self.attention_vector)
        self.output_projection = nn.Linear(hidden_dimension, hidden_dimension)
        self.normalization = nn.LayerNorm(hidden_dimension)

    def forward(
        self,
        self_state: Tensor,
        neighbor_public_state: Tensor,
        edge_features: Tensor,
        neighbor_mask: Tensor,
    ) -> Tensor:
        if self_state.ndim != 2:
            raise MAPPONetworkError("CA-GAT self state must be [item,hidden]")
        items, hidden = self_state.shape
        if neighbor_public_state.ndim != 3:
            raise MAPPONetworkError("CA-GAT public state must be [item,neighbor,hidden]")
        neighbors = neighbor_public_state.shape[1]
        if hidden != self.hidden_dimension:
            raise MAPPONetworkError("CA-GAT self hidden dimension mismatch")
        if tuple(neighbor_public_state.shape) != (items, neighbors, hidden):
            raise MAPPONetworkError("CA-GAT public hidden shape mismatch")
        if tuple(edge_features.shape) != (items, neighbors, self.edge_dimension):
            raise MAPPONetworkError("CA-GAT edge shape mismatch")
        if tuple(neighbor_mask.shape) != (items, neighbors) or neighbor_mask.dtype != torch.bool:
            raise MAPPONetworkError("CA-GAT neighbor mask mismatch")

        outputs: list[Tensor] = []
        for head in range(self.heads):
            edge_state = self.edge_projection[head](edge_features)
            neighbor_pair = torch.cat([neighbor_public_state, edge_state], dim=-1)
            query = self.query[head](self_state)
            key = self.key[head](neighbor_pair)
            value = self.value[head](neighbor_pair)
            self_edge = torch.zeros(
                (items, self.head_dimension),
                device=self_state.device,
                dtype=self_state.dtype,
            )
            self_pair = torch.cat([self_state, self_edge], dim=-1)
            self_key = self.key[head](self_pair)
            self_value = self.value[head](self_pair)
            self_score = torch.sum(
                self.attention_vector[head]
                * torch.nn.functional.leaky_relu(query + self_key),
                dim=-1,
                keepdim=True,
            )
            neighbor_score = torch.sum(
                self.attention_vector[head]
                * torch.nn.functional.leaky_relu(query[:, None, :] + key),
                dim=-1,
            )
            scores = torch.cat([self_score, neighbor_score], dim=-1)
            valid = torch.cat(
                [
                    torch.ones((items, 1), dtype=torch.bool, device=self_state.device),
                    neighbor_mask,
                ],
                dim=-1,
            )
            scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
            attention = torch.softmax(scores, dim=-1)
            all_values = torch.cat([self_value[:, None, :], value], dim=1)
            outputs.append(torch.sum(attention[..., None] * all_values, dim=1))
        aggregate = torch.cat(outputs, dim=-1)
        update = torch.nn.functional.leaky_relu(self.output_projection(aggregate))
        return self.normalization(self_state + update)


@dataclass(frozen=True)
class ActorNetworkOutput:
    """Raw seven-head logits, validated masks, and recurrent output."""

    raw_logits: Mapping[str, Tensor]
    action_masks: Mapping[str, Tensor]
    recurrent_features: Tensor
    hidden_out: Tensor

    @property
    def logits(self) -> Mapping[str, Tensor]:
        return self.raw_logits

    def masked_logits(self) -> dict[str, Tensor]:
        """Apply masks without sampling or constructing categorical objects."""

        return {
            branch: logits.masked_fill(~self.action_masks[branch], -torch.inf)
            for branch, logits in self.raw_logits.items()
        }


class CAGATMAPPOActor(nn.Module):
    """Shared CA-GATv2 to GRU to seven-raw-logit actor."""

    def __init__(self, config: RunConfig) -> None:
        super().__init__()
        self.spec = MAPPOTensorSpec.from_config(config)
        hidden = self.spec.encoder_hidden_dimension
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(_derived_torch_seed(config.seed))
            self.self_encoder = nn.Sequential(
                nn.Linear(self.spec.self_feature_dim, hidden),
                nn.LeakyReLU(),
            )
            self.public_encoder = nn.Sequential(
                nn.Linear(self.spec.neighbor_public_feature_dim, hidden),
                nn.LeakyReLU(),
            )
            self.graph_encoder = CAGATv2Layer(
                hidden,
                self.spec.edge_feature_dim,
                self.spec.attention_head_count,
            )
            self.gru = nn.GRU(
                hidden,
                self.spec.gru_hidden_dimension,
                num_layers=1,
                batch_first=True,
            )
            dimensions = self.spec.action_dimensions
            self.branch_embeddings = nn.ModuleDict(
                {
                    branch: nn.Embedding(dimensions[branch], hidden)
                    for branch in (
                        "tx_select",
                        "resource_group",
                        "resource_width",
                        "cpu_queue",
                    )
                }
            )
            self.action_heads = nn.ModuleDict(
                {
                    "route": nn.Linear(hidden, dimensions["route"]),
                    "tx_select": nn.Linear(hidden, dimensions["tx_select"]),
                    "resource_group": nn.Linear(2 * hidden, dimensions["resource_group"]),
                    "resource_width": nn.Linear(3 * hidden, dimensions["resource_width"]),
                    "power_level": nn.Linear(4 * hidden, dimensions["power_level"]),
                    "cpu_queue": nn.Linear(hidden, dimensions["cpu_queue"]),
                    "cpu_frequency": nn.Linear(2 * hidden, dimensions["cpu_frequency"]),
                }
            )

    def initial_hidden(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise MAPPONetworkError("batch_size must be a positive integer")
        parameter = next(self.parameters())
        return torch.zeros(
            (batch_size, self.spec.uav_count, self.spec.gru_hidden_dimension),
            device=parameter.device if device is None else device,
            dtype=parameter.dtype if dtype is None else dtype,
        )

    def branch_logits(
        self,
        branch: str,
        recurrent_features: Tensor,
        action_indices: Mapping[str, Tensor],
    ) -> Tensor:
        """Return one head's logits under the frozen predecessor dependencies."""

        if branch not in ACTION_BRANCH_DEPENDENCIES:
            raise MAPPONetworkError(f"unknown action branch {branch!r}")
        if recurrent_features.ndim != 4:
            raise MAPPONetworkError(
                "recurrent_features must be [batch,time,agent,hidden]"
            )
        batch, time, agents, hidden = recurrent_features.shape
        if agents != self.spec.uav_count or hidden != self.spec.gru_hidden_dimension:
            raise MAPPONetworkError("recurrent_features differ from the actor spec")
        if (
            not recurrent_features.is_floating_point()
            or not torch.isfinite(recurrent_features).all()
        ):
            raise MAPPONetworkError("recurrent_features must be finite floating tensors")

        context_parts = [recurrent_features]
        for dependency in ACTION_BRANCH_DEPENDENCIES[branch]:
            if dependency not in action_indices:
                raise MAPPONetworkError(
                    f"{branch} requires the selected {dependency} indices"
                )
            indices = action_indices[dependency]
            if tuple(indices.shape) != (batch, time, agents) or indices.dtype != torch.long:
                raise MAPPONetworkError(
                    f"{dependency} indices must be long [batch,time,agent]"
                )
            if indices.device != recurrent_features.device:
                raise MAPPONetworkError(
                    f"{dependency} indices and recurrent features must share a device"
                )
            dimension = self.spec.action_dimensions[dependency]
            if torch.any(indices < 0) or torch.any(indices >= dimension):
                raise MAPPONetworkError(f"{dependency} index lies outside its domain")
            context_parts.append(self.branch_embeddings[dependency](indices))

        context = (
            context_parts[0]
            if len(context_parts) == 1
            else torch.cat(context_parts, dim=-1)
        )
        logits = self.action_heads[branch](context)
        expected = (batch, time, agents, self.spec.action_dimensions[branch])
        if tuple(logits.shape) != expected:
            raise MAPPONetworkError(f"{branch} logits have an invalid shape")
        if not torch.isfinite(logits).all():
            raise MAPPONetworkError(f"{branch} logits contain NaN or Inf")
        return logits

    def forward(
        self,
        batch: ActorTensorBatch,
        hidden_in: Tensor | None = None,
    ) -> ActorNetworkOutput:
        batch_size, time, agents = batch.validate(self.spec)
        if hidden_in is None:
            hidden = self.initial_hidden(
                batch_size,
                device=batch.self_features.device,
                dtype=batch.self_features.dtype,
            )
        else:
            expected = (batch_size, agents, self.spec.gru_hidden_dimension)
            if tuple(hidden_in.shape) != expected or not hidden_in.is_floating_point():
                raise MAPPONetworkError("hidden_in must be [batch,agent,128]")
            if hidden_in.device != batch.self_features.device:
                raise MAPPONetworkError("hidden_in and actor inputs must share a device")
            if hidden_in.dtype != batch.self_features.dtype:
                raise MAPPONetworkError("hidden_in and actor inputs must share a dtype")
            if not torch.isfinite(hidden_in).all():
                raise MAPPONetworkError("hidden_in contains NaN or Inf")
            hidden = hidden_in

        item_count = batch_size * time * agents
        self_state = self.self_encoder(
            batch.self_features.reshape(item_count, self.spec.self_feature_dim)
        )
        public_state = self.public_encoder(
            batch.neighbor_public_features.reshape(
                item_count,
                self.spec.uav_count,
                self.spec.neighbor_public_feature_dim,
            )
        )
        spatial = self.graph_encoder(
            self_state,
            public_state,
            batch.edge_features.reshape(
                item_count, self.spec.uav_count, self.spec.edge_feature_dim
            ),
            batch.neighbor_mask.reshape(item_count, self.spec.uav_count),
        ).reshape(
            batch_size,
            time,
            agents,
            self.spec.encoder_hidden_dimension,
        )

        outputs: list[Tensor] = []
        for step in range(time):
            keep = (~batch.episode_starts[:, step]).to(hidden.dtype).unsqueeze(-1)
            hidden = hidden * keep
            step_output, next_hidden = self.gru(
                spatial[:, step].reshape(
                    batch_size * agents, 1, self.spec.encoder_hidden_dimension
                ),
                hidden.reshape(1, batch_size * agents, self.spec.gru_hidden_dimension),
            )
            outputs.append(
                step_output.reshape(batch_size, agents, self.spec.gru_hidden_dimension)
            )
            hidden = next_hidden.reshape(
                batch_size, agents, self.spec.gru_hidden_dimension
            )
        recurrent = torch.stack(outputs, dim=1)

        logits = {
            branch: self.branch_logits(branch, recurrent, batch.action_indices)
            for branch in ACTION_BRANCH_ORDER
        }
        for branch in ACTION_BRANCH_ORDER:
            if logits[branch].shape != batch.action_masks[branch].shape:
                raise MAPPONetworkError(f"{branch} logits and mask shapes differ")
            if not torch.isfinite(logits[branch]).all():
                raise MAPPONetworkError(f"{branch} logits contain NaN or Inf")
        return ActorNetworkOutput(
            raw_logits=logits,
            action_masks=batch.action_masks,
            recurrent_features=recurrent,
            hidden_out=hidden,
        )


class MAPPOCentralizedCritic(nn.Module):
    """Centralized value network with a legacy scalar or per-agent head."""

    def __init__(self, config: RunConfig) -> None:
        super().__init__()
        self.spec = MAPPOTensorSpec.from_config(config)
        self.agent_credit_mode = AgentCreditMode(
            config.training.mappo.agent_credit_mode
        )
        self.output_dimension = (
            1
            if self.agent_credit_mode is AgentCreditMode.TEAM
            else self.spec.uav_count
        )
        hidden = self.spec.encoder_hidden_dimension
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(_derived_torch_seed(config.seed))
            self.value_network = nn.Sequential(
                nn.Linear(self.spec.centralized_state_dim, hidden),
                nn.LeakyReLU(),
                nn.LayerNorm(hidden),
                nn.Linear(hidden, self.output_dimension),
            )

    def forward(self, batch: CentralizedStateTensorBatch | Tensor) -> Tensor:
        features = batch.features if isinstance(batch, CentralizedStateTensorBatch) else batch
        CentralizedStateTensorBatch(features).validate(self.spec)
        values = self.value_network(features)
        if values.shape != (*features.shape[:2], self.output_dimension):
            raise MAPPONetworkError(
                "critic output head count differs from agent_credit_mode"
            )
        if not torch.isfinite(values).all():
            raise MAPPONetworkError("critic value contains NaN or Inf")
        return values


CentralizedCritic = MAPPOCentralizedCritic


__all__ = [
    "ACTION_BRANCH_DEPENDENCIES",
    "ACTION_BRANCH_ORDER",
    "ActorNetworkOutput",
    "ActorObservationTensorizer",
    "ActorTensorBatch",
    "CAGATMAPPOActor",
    "CAGATv2Layer",
    "CentralizedCritic",
    "CentralizedStateTensorBatch",
    "CentralizedStateTensorizer",
    "MAPPONetworkError",
    "MAPPOCentralizedCritic",
    "MAPPOTensorSpec",
    "stack_actor_time",
    "stack_centralized_time",
]
