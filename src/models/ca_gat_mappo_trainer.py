"""Environment-driven production trainer for the frozen CA-GAT-MAPPO contract.

Collection lifecycle and accounting live here; action sampling, rollout
storage, GAE, PPO losses, and optimization remain delegated to gated modules.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import copy
from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import Tensor

from ..config import (
    MAPPO_INITIAL_POLICY_VERSION,
    RunConfig,
    require_formal_rl_enabled,
    validate_mappo_training_device,
)
from ..env.environment import ResetResult, StepResult, U2UMECEnvironment
from ..env.randomness import derive_training_episode_seed
from .ca_gat_mappo import (
    ActorObservationTensorizer,
    CAGATMAPPOActor,
    CentralizedStateTensorizer,
    MAPPOCentralizedCritic,
)
from .ca_gat_mappo_actions import (
    CAGATMAPPOActionDistribution,
    SequentialActionMaskBatch,
)
from .ca_gat_mappo_rollout import CAGATMAPPORolloutBuffer
from .ca_gat_mappo_update import (
    CAGATMAPPORecurrentPPOUpdater,
    RecurrentPPOUpdateOutput,
)


class CAGATMAPPOTrainerError(RuntimeError):
    """Raised when production collection violates the frozen lifecycle."""


_REWARD_FIELDS = (
    "completed_task_count",
    "expired_task_count",
    "urgent_workload_s",
    "actual_energy_j",
    "normalized_completed",
    "normalized_expired",
    "normalized_workload",
    "normalized_energy",
    "completion_component",
    "expiration_penalty",
    "workload_penalty",
    "energy_penalty",
)


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CAGATMAPPOTrainerError(f"{name} must be numeric")
    converted = float(value)
    if not math.isfinite(converted):
        raise CAGATMAPPOTrainerError(f"{name} must be finite")
    return converted


@dataclass(frozen=True)
class ScalarTrainingDiagnostics:
    """Immutable count/sum/mean for one scalar training diagnostic."""

    count: int
    total: float
    mean: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.count, bool)
            or not isinstance(self.count, int)
            or self.count < 0
        ):
            raise CAGATMAPPOTrainerError("diagnostic count must be non-negative")
        if not math.isfinite(self.total) or not math.isfinite(self.mean):
            raise CAGATMAPPOTrainerError("diagnostic values must be finite")
        expected = 0.0 if self.count == 0 else self.total / self.count
        if not math.isclose(
            self.mean, expected, rel_tol=1.0e-12, abs_tol=1.0e-12
        ):
            raise CAGATMAPPOTrainerError(
                "diagnostic mean does not match count and total"
            )


@dataclass(frozen=True)
class CAGATMAPPORewardDiagnostics:
    """Reward and component aggregates sourced from real StepResult records."""

    reward: ScalarTrainingDiagnostics
    completed_task_count: ScalarTrainingDiagnostics
    expired_task_count: ScalarTrainingDiagnostics
    urgent_workload_s: ScalarTrainingDiagnostics
    actual_energy_j: ScalarTrainingDiagnostics
    normalized_completed: ScalarTrainingDiagnostics
    normalized_expired: ScalarTrainingDiagnostics
    normalized_workload: ScalarTrainingDiagnostics
    normalized_energy: ScalarTrainingDiagnostics
    completion_component: ScalarTrainingDiagnostics
    expiration_penalty: ScalarTrainingDiagnostics
    workload_penalty: ScalarTrainingDiagnostics
    energy_penalty: ScalarTrainingDiagnostics

    def __post_init__(self) -> None:
        counts = {
            getattr(self, name).count for name in ("reward", *_REWARD_FIELDS)
        }
        if len(counts) != 1:
            raise CAGATMAPPOTrainerError("reward diagnostic counts must agree")

    @property
    def transition_count(self) -> int:
        return self.reward.count


@dataclass(frozen=True)
class CAGATMAPPOEpisodeDiagnostics:
    """Collection accounting for one started training episode."""

    episode_index: int
    environment_seed: int
    transition_count: int
    completed_boundary: bool
    reward: CAGATMAPPORewardDiagnostics

    def __post_init__(self) -> None:
        if self.episode_index < 0 or self.environment_seed < 0:
            raise CAGATMAPPOTrainerError(
                "episode index and seed must be non-negative"
            )
        if (
            self.transition_count <= 0
            or self.reward.transition_count != self.transition_count
        ):
            raise CAGATMAPPOTrainerError(
                "episode transition accounting is inconsistent"
            )
        if not isinstance(self.completed_boundary, bool):
            raise CAGATMAPPOTrainerError("completed_boundary must be boolean")


@dataclass(frozen=True)
class CAGATMAPPOUpdateDiagnostics:
    """One successful full-rollout update and its policy-version barrier."""

    update_index: int
    rollout_policy_version: int
    policy_version_after_update: int
    output: RecurrentPPOUpdateOutput

    def __post_init__(self) -> None:
        if self.update_index < 0 or self.rollout_policy_version < 0:
            raise CAGATMAPPOTrainerError(
                "update index and policy version must be non-negative"
            )
        if self.policy_version_after_update != self.rollout_policy_version + 1:
            raise CAGATMAPPOTrainerError(
                "one successful update must increment policy version once"
            )
        if not isinstance(self.output, RecurrentPPOUpdateOutput):
            raise TypeError("output must be a RecurrentPPOUpdateOutput")


@dataclass(frozen=True)
class CAGATMAPPOLossDiagnostics:
    """Aggregate PPO diagnostics across all four-epoch update records."""

    actor_loss: ScalarTrainingDiagnostics
    critic_loss: ScalarTrainingDiagnostics
    entropy: ScalarTrainingDiagnostics
    total_loss: ScalarTrainingDiagnostics
    ratio: ScalarTrainingDiagnostics
    actor_grad_norm_before_clip: ScalarTrainingDiagnostics
    critic_grad_norm_before_clip: ScalarTrainingDiagnostics


@dataclass(frozen=True)
class CAGATMAPPOTrainingResult:
    """Immutable production collection/update result without artifact writes."""

    total_environment_transitions: int
    optimized_transitions: int
    unused_final_tail_transitions: int
    started_episode_count: int
    completed_episode_count: int
    ppo_update_count: int
    initial_policy_version: int
    final_policy_version: int
    episode_seeds: tuple[int, ...]
    reward: CAGATMAPPORewardDiagnostics
    ppo: CAGATMAPPOLossDiagnostics
    episodes: tuple[CAGATMAPPOEpisodeDiagnostics, ...]
    updates: tuple[CAGATMAPPOUpdateDiagnostics, ...]

    def __post_init__(self) -> None:
        counts = (
            self.total_environment_transitions,
            self.optimized_transitions,
            self.unused_final_tail_transitions,
            self.started_episode_count,
            self.completed_episode_count,
            self.ppo_update_count,
            self.initial_policy_version,
            self.final_policy_version,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for value in counts
        ):
            raise CAGATMAPPOTrainerError(
                "training result counts must be non-negative integers"
            )
        if (
            self.optimized_transitions + self.unused_final_tail_transitions
            != self.total_environment_transitions
        ):
            raise CAGATMAPPOTrainerError(
                "optimized plus unused transitions must equal collected transitions"
            )
        if (
            self.started_episode_count != len(self.episodes)
            or self.started_episode_count != len(self.episode_seeds)
        ):
            raise CAGATMAPPOTrainerError(
                "started episode accounting is inconsistent"
            )
        if self.completed_episode_count != sum(
            item.completed_boundary for item in self.episodes
        ):
            raise CAGATMAPPOTrainerError(
                "completed episode accounting is inconsistent"
            )
        if self.ppo_update_count != len(self.updates):
            raise CAGATMAPPOTrainerError("PPO update accounting is inconsistent")
        if (
            self.final_policy_version
            != self.initial_policy_version + self.ppo_update_count
        ):
            raise CAGATMAPPOTrainerError(
                "final policy version differs from successful update count"
            )
        if self.reward.transition_count != self.total_environment_transitions:
            raise CAGATMAPPOTrainerError(
                "global reward count differs from collected transitions"
            )

    @property
    def completed_episodes(self) -> int:
        """Return the exact count of episodes that reached a real boundary."""

        return self.completed_episode_count

    @property
    def collected_environment_transitions(self) -> int:
        """Return the exact number of real environment steps collected."""

        return self.total_environment_transitions


class _ScalarAccumulator:
    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0

    def add(self, value: Any, name: str) -> None:
        self.total = math.fsum((self.total, _finite(value, name)))
        self.count += 1

    def snapshot(self) -> ScalarTrainingDiagnostics:
        mean = 0.0 if self.count == 0 else self.total / self.count
        return ScalarTrainingDiagnostics(self.count, self.total, mean)


class _RewardAccumulator:
    def __init__(self) -> None:
        self._values = {
            name: _ScalarAccumulator() for name in ("reward", *_REWARD_FIELDS)
        }

    def add(self, step_result: StepResult) -> None:
        reward = _finite(step_result.reward, "StepResult.reward")
        info_reward = step_result.info.get("reward")
        if not isinstance(info_reward, Mapping):
            raise CAGATMAPPOTrainerError(
                "StepResult.info['reward'] must be a mapping"
            )
        diagnostic_reward = _finite(
            info_reward.get("reward"), "StepResult.info.reward.reward"
        )
        if not math.isclose(
            reward, diagnostic_reward, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise CAGATMAPPOTrainerError(
                "StepResult reward and diagnostic reward differ"
            )
        self._values["reward"].add(reward, "StepResult.reward")
        for name in _REWARD_FIELDS:
            self._values[name].add(
                info_reward.get(name), f"StepResult.info.reward.{name}"
            )

    def snapshot(self) -> CAGATMAPPORewardDiagnostics:
        return CAGATMAPPORewardDiagnostics(
            **{
                name: accumulator.snapshot()
                for name, accumulator in self._values.items()
            }
        )


def _aggregate_updates(
    updates: Sequence[CAGATMAPPOUpdateDiagnostics],
) -> CAGATMAPPOLossDiagnostics:
    names = (
        "actor_loss",
        "critic_loss",
        "entropy_mean",
        "total_loss",
        "ratio_mean",
        "actor_grad_norm_before_clip",
        "critic_grad_norm_before_clip",
    )
    values = {name: _ScalarAccumulator() for name in names}
    for update in updates:
        for epoch in update.output.epoch_diagnostics:
            for name in names:
                values[name].add(getattr(epoch, name), f"PPO.{name}")
    return CAGATMAPPOLossDiagnostics(
        actor_loss=values["actor_loss"].snapshot(),
        critic_loss=values["critic_loss"].snapshot(),
        entropy=values["entropy_mean"].snapshot(),
        total_loss=values["total_loss"].snapshot(),
        ratio=values["ratio_mean"].snapshot(),
        actor_grad_norm_before_clip=values[
            "actor_grad_norm_before_clip"
        ].snapshot(),
        critic_grad_norm_before_clip=values[
            "critic_grad_norm_before_clip"
        ].snapshot(),
    )


def build_training_episode_config(
    base_config: RunConfig,
    episode_index: int,
) -> RunConfig:
    """Return an episode-local config whose only resolved change is seed."""

    if not isinstance(base_config, RunConfig):
        raise TypeError("base_config must be a RunConfig")
    if (
        isinstance(episode_index, bool)
        or not isinstance(episode_index, int)
        or episode_index < 0
    ):
        raise CAGATMAPPOTrainerError(
            "episode_index must be a non-negative integer"
        )
    base_config.validate()
    episode_seed = derive_training_episode_seed(
        base_config.seed, episode_index
    )
    episode_config = copy.copy(base_config)
    object.__setattr__(episode_config, "seed", episode_seed)
    episode_config.validate()
    return episode_config


class CAGATMAPPOTrainer:
    """Collect real environment transitions and invoke the gated updater."""

    def __init__(
        self,
        config: RunConfig,
        *,
        actor: CAGATMAPPOActor | None = None,
        critic: MAPPOCentralizedCritic | None = None,
        environment_factory: Callable[[RunConfig], Any] | None = None,
        updater_factory: Callable[
            [
                CAGATMAPPOActor,
                MAPPOCentralizedCritic,
                RunConfig,
                CAGATMAPPOActionDistribution,
            ],
            Any,
        ]
        | None = None,
    ) -> None:
        if not isinstance(config, RunConfig):
            raise TypeError("config must be a RunConfig")
        require_formal_rl_enabled(config)
        device_name = validate_mappo_training_device(
            config, cuda_available=torch.cuda.is_available()
        )
        self.config = config
        self.device = torch.device(device_name)
        self.dtype = torch.float32
        self.actor = CAGATMAPPOActor(config) if actor is None else actor
        self.critic = (
            MAPPOCentralizedCritic(config) if critic is None else critic
        )
        if not isinstance(self.actor, CAGATMAPPOActor):
            raise TypeError("actor must be a CAGATMAPPOActor")
        if not isinstance(self.critic, MAPPOCentralizedCritic):
            raise TypeError("critic must be a MAPPOCentralizedCritic")
        self.actor.to(device=self.device, dtype=self.dtype)
        self.critic.to(device=self.device, dtype=self.dtype)
        self._validate_model_placement()

        self.actor_tensorizer = ActorObservationTensorizer(config)
        self.state_tensorizer = CentralizedStateTensorizer(config)
        self.action_distribution = CAGATMAPPOActionDistribution(
            self.actor, config
        )
        self.policy_generator = torch.Generator(device=self.device.type)
        self.policy_generator.manual_seed(
            self.action_distribution.policy_seed
        )
        self.rollout_buffer = CAGATMAPPORolloutBuffer(config)
        self.environment_factory = (
            U2UMECEnvironment
            if environment_factory is None
            else environment_factory
        )
        if not callable(self.environment_factory):
            raise TypeError("environment_factory must be callable")
        self.updater = (
            CAGATMAPPORecurrentPPOUpdater(
                self.actor,
                self.critic,
                config,
                action_distribution=self.action_distribution,
            )
            if updater_factory is None
            else updater_factory(
                self.actor,
                self.critic,
                config,
                self.action_distribution,
            )
        )
        if not callable(getattr(self.updater, "update", None)):
            raise TypeError("updater must expose update(buffer)")

        self.policy_version = MAPPO_INITIAL_POLICY_VERSION
        self._rollout_policy_version: int | None = None
        self._has_run = False

    def _validate_model_placement(self) -> None:
        placements: list[torch.device] = []
        for name, module in (("actor", self.actor), ("critic", self.critic)):
            parameters = tuple(module.parameters())
            if not parameters:
                raise CAGATMAPPOTrainerError(f"{name} has no parameters")
            devices = {parameter.device for parameter in parameters}
            if len(devices) != 1:
                raise CAGATMAPPOTrainerError(
                    f"{name} parameters span more than one device"
                )
            actual_device = next(iter(devices))
            if actual_device.type != self.device.type:
                raise CAGATMAPPOTrainerError(
                    f"{name} is not placed on explicit training_device"
                )
            if any(
                parameter.dtype != self.dtype for parameter in parameters
            ):
                raise CAGATMAPPOTrainerError(
                    f"{name} must use float32 parameters"
                )
            placements.append(actual_device)
        if placements[0] != placements[1]:
            raise CAGATMAPPOTrainerError(
                "actor and critic must share one actual training device"
            )
        self.device = placements[0]

    def _should_continue(
        self, completed_episodes: int, transitions: int
    ) -> bool:
        mappo = self.config.training.mappo
        return (
            completed_episodes < mappo.max_training_episodes
            and transitions < mappo.max_environment_transitions
        )

    def _start_episode(
        self, episode_index: int
    ) -> tuple[Any, ResetResult, int]:
        episode_config = build_training_episode_config(
            self.config, episode_index
        )
        environment = self.environment_factory(episode_config)
        reset_result = environment.reset()
        if not isinstance(reset_result, ResetResult):
            raise CAGATMAPPOTrainerError(
                "environment.reset() must return ResetResult"
            )
        observations = tuple(reset_result.observations)
        if len(observations) != self.actor.spec.uav_count:
            raise CAGATMAPPOTrainerError(
                "reset must return one observation per UAV"
            )
        if any(observation.slot != 0 for observation in observations):
            raise CAGATMAPPOTrainerError(
                "every real episode must reset at slot zero"
            )
        if reset_result.centralized_state.slot != 0:
            raise CAGATMAPPOTrainerError(
                "reset centralized state must use slot zero"
            )
        return environment, reset_result, episode_config.seed

    def _validate_step_result(
        self, slot: int, result: StepResult
    ) -> bool:
        if not isinstance(result, StepResult):
            raise CAGATMAPPOTrainerError(
                "environment.step() must return StepResult"
            )
        boundary = result.terminated or result.truncated
        if result.info.get("slot") != slot:
            raise CAGATMAPPOTrainerError(
                "StepResult.info slot differs from collected slot"
            )
        allowed = result.info.get("bootstrap_allowed")
        if not isinstance(allowed, bool) or allowed != (not boundary):
            raise CAGATMAPPOTrainerError(
                "bootstrap permission differs from episode boundary"
            )
        if boundary:
            if (
                result.observations is not None
                or result.centralized_state is not None
            ):
                raise CAGATMAPPOTrainerError(
                    "episode boundary must expose no fake next state"
                )
        elif (
            result.observations is None
            or result.centralized_state is None
        ):
            raise CAGATMAPPOTrainerError(
                "non-boundary transition requires the real next state"
            )
        return boundary

    @staticmethod
    def _execution_metadata(
        info: Mapping[str, Any],
    ) -> tuple[
        Sequence[Mapping[str, Any]],
        Mapping[str, Any],
    ]:
        executed = info.get("executed")
        if not isinstance(executed, Sequence) or isinstance(
            executed, (str, bytes)
        ):
            raise CAGATMAPPOTrainerError(
                "StepResult.info['executed'] must be a sequence"
            )
        if any(not isinstance(item, Mapping) for item in executed):
            raise CAGATMAPPOTrainerError(
                "executed summaries must be mappings"
            )
        summary = {
            "rejection": info.get("rejection", ()),
            "downgrade": info.get("downgrade", ()),
            "canonicalization": info.get("canonicalization", ()),
        }
        return executed, summary

    def _perform_update(
        self, updates: list[CAGATMAPPOUpdateDiagnostics]
    ) -> None:
        if not self.rollout_buffer.full:
            raise CAGATMAPPOTrainerError(
                "PPO update requires one full rollout"
            )
        if self._rollout_policy_version != self.policy_version:
            raise CAGATMAPPOTrainerError(
                "rollout contains more than one policy version"
            )
        frozen_version = self.policy_version
        self.rollout_buffer.finalize()
        if self.policy_version != frozen_version:
            raise CAGATMAPPOTrainerError(
                "policy version changed before PPO update"
            )
        generator_state = self.policy_generator.get_state().clone()
        output = self.updater.update(self.rollout_buffer)
        if not isinstance(output, RecurrentPPOUpdateOutput):
            raise CAGATMAPPOTrainerError(
                "updater must return RecurrentPPOUpdateOutput"
            )
        if not torch.equal(
            generator_state, self.policy_generator.get_state()
        ):
            raise CAGATMAPPOTrainerError(
                "PPO update consumed the policy sampling stream"
            )
        if self.policy_version != frozen_version:
            raise CAGATMAPPOTrainerError(
                "updater changed Trainer policy version"
            )
        self.policy_version += 1
        updates.append(
            CAGATMAPPOUpdateDiagnostics(
                update_index=len(updates),
                rollout_policy_version=frozen_version,
                policy_version_after_update=self.policy_version,
                output=output,
            )
        )
        self.rollout_buffer.clear()
        self._rollout_policy_version = None

    def train(self) -> CAGATMAPPOTrainingResult:
        """Execute the finite frozen lifecycle without writing artifacts."""

        if self._has_run:
            raise CAGATMAPPOTrainerError(
                "one Trainer instance represents exactly one training run"
            )
        self._has_run = True
        self.actor.train()
        self.critic.train()

        global_reward = _RewardAccumulator()
        episode_reward = _RewardAccumulator()
        episodes: list[CAGATMAPPOEpisodeDiagnostics] = []
        episode_seeds: list[int] = []
        updates: list[CAGATMAPPOUpdateDiagnostics] = []
        transitions = 0
        optimized = 0
        completed_episodes = 0
        episode_transitions = 0
        episode_index = 0

        environment, reset_result, episode_seed = self._start_episode(
            episode_index
        )
        episode_seeds.append(episode_seed)
        observations = reset_result.observations
        state = reset_result.centralized_state
        hidden = self.actor.initial_hidden(
            1, device=self.device, dtype=self.dtype
        )
        episode_active = True

        while self._should_continue(completed_episodes, transitions):
            slot = observations[0].slot
            if any(observation.slot != slot for observation in observations):
                raise CAGATMAPPOTrainerError(
                    "actor observations must describe one decision slot"
                )
            if state.slot != slot:
                raise CAGATMAPPOTrainerError(
                    "actor observations and critic state must share a slot"
                )
            episode_start = slot == 0
            if episode_start and episode_transitions != 0:
                raise CAGATMAPPOTrainerError(
                    "slot zero appeared without an episode boundary"
                )

            actor_batch = self.actor_tensorizer.encode_step(
                observations,
                device=self.device,
                dtype=self.dtype,
                episode_start=episode_start,
            )
            action_masks = SequentialActionMaskBatch.from_observations(
                observations
            )
            state_batch = self.state_tensorizer.encode_step(
                state, device=self.device, dtype=self.dtype
            )
            hidden_in = hidden.detach()
            step_policy_version = self.policy_version
            if len(self.rollout_buffer) == 0:
                self._rollout_policy_version = step_policy_version
            elif self._rollout_policy_version != step_policy_version:
                raise CAGATMAPPOTrainerError(
                    "policy changed during one rollout"
                )

            with torch.no_grad():
                action_output = self.action_distribution.sample_actions(
                    actor_batch,
                    action_masks,
                    hidden_in,
                    generator=self.policy_generator,
                )
                old_value: Tensor = self.critic(state_batch)[0, 0, 0]

            proposals = tuple(action_output.proposals[0][0])
            step_result = environment.step(proposals)
            boundary = self._validate_step_result(slot, step_result)
            if self.policy_version != step_policy_version:
                raise CAGATMAPPOTrainerError(
                    "policy version changed during transition collection"
                )

            bootstrap_value: Tensor | None = None
            if not boundary:
                assert step_result.centralized_state is not None
                next_state_batch = self.state_tensorizer.encode_step(
                    step_result.centralized_state,
                    device=self.device,
                    dtype=self.dtype,
                )
                with torch.no_grad():
                    bootstrap_value = self.critic(next_state_batch)[
                        0, 0, 0
                    ]

            executed, rejections = self._execution_metadata(
                step_result.info
            )
            self.rollout_buffer.append_step(
                slot=slot,
                actor_batch=actor_batch,
                action_mask_batch=action_masks,
                action_output=action_output,
                hidden_in=hidden_in,
                centralized_state=state_batch,
                old_value=old_value,
                reward=step_result.reward,
                terminated=step_result.terminated,
                truncated=step_result.truncated,
                episode_boundary=boundary,
                bootstrap_allowed=not boundary,
                bootstrap_value=bootstrap_value,
                executed_action_summary=executed,
                rejection_or_downgrade_summary=rejections,
            )
            global_reward.add(step_result)
            episode_reward.add(step_result)
            transitions += 1
            episode_transitions += 1
            hidden = action_output.hidden_out.detach()

            if boundary:
                episodes.append(
                    CAGATMAPPOEpisodeDiagnostics(
                        episode_index=episode_index,
                        environment_seed=episode_seed,
                        transition_count=episode_transitions,
                        completed_boundary=True,
                        reward=episode_reward.snapshot(),
                    )
                )
                completed_episodes += 1
                episode_active = False
                episode_transitions = 0
                episode_reward = _RewardAccumulator()
                if self._should_continue(completed_episodes, transitions):
                    episode_index += 1
                    (
                        environment,
                        reset_result,
                        episode_seed,
                    ) = self._start_episode(episode_index)
                    episode_seeds.append(episode_seed)
                    observations = reset_result.observations
                    state = reset_result.centralized_state
                    hidden = self.actor.initial_hidden(
                        1, device=self.device, dtype=self.dtype
                    )
                    episode_active = True
            else:
                assert step_result.observations is not None
                assert step_result.centralized_state is not None
                observations = step_result.observations
                state = step_result.centralized_state

            if self.rollout_buffer.full:
                self._perform_update(updates)
                optimized += (
                    self.config.training.mappo.rollout_length_slots
                )

        if episode_active and episode_transitions:
            episodes.append(
                CAGATMAPPOEpisodeDiagnostics(
                    episode_index=episode_index,
                    environment_seed=episode_seed,
                    transition_count=episode_transitions,
                    completed_boundary=False,
                    reward=episode_reward.snapshot(),
                )
            )

        unused_tail = len(self.rollout_buffer)
        if unused_tail:
            if self.rollout_buffer.finalized:
                raise CAGATMAPPOTrainerError(
                    "final partial rollout must not be finalized"
                )
            self.rollout_buffer.clear()
            self._rollout_policy_version = None
        rollout_length = self.config.training.mappo.rollout_length_slots
        if optimized != len(updates) * rollout_length:
            raise CAGATMAPPOTrainerError(
                "optimized transition accounting differs from updates"
            )
        if optimized + unused_tail != transitions:
            raise CAGATMAPPOTrainerError(
                "final transition accounting is inconsistent"
            )

        return CAGATMAPPOTrainingResult(
            total_environment_transitions=transitions,
            optimized_transitions=optimized,
            unused_final_tail_transitions=unused_tail,
            started_episode_count=len(episodes),
            completed_episode_count=completed_episodes,
            ppo_update_count=len(updates),
            initial_policy_version=MAPPO_INITIAL_POLICY_VERSION,
            final_policy_version=self.policy_version,
            episode_seeds=tuple(episode_seeds),
            reward=global_reward.snapshot(),
            ppo=_aggregate_updates(updates),
            episodes=tuple(episodes),
            updates=tuple(updates),
        )


__all__ = [
    "CAGATMAPPOLossDiagnostics",
    "CAGATMAPPOEpisodeDiagnostics",
    "CAGATMAPPORewardDiagnostics",
    "CAGATMAPPOTrainer",
    "CAGATMAPPOTrainerError",
    "CAGATMAPPOTrainingResult",
    "CAGATMAPPOUpdateDiagnostics",
    "ScalarTrainingDiagnostics",
    "build_training_episode_config",
]
