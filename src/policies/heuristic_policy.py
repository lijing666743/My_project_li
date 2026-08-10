"""Frozen deterministic heuristic baseline over actor-visible observations.

The policy implements the Section 4 Deadline-and-Historical-Link-Aware
Lexicographic Heuristic. It emits only the seven-branch proposal and
deliberately has no access to centralized state, current channel truth,
executor metadata, or environment RNG state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from ..config import RunConfig
from ..env.actions import ActionProposal
from ..env.observation import ActorObservation


class HeuristicPolicyError(ValueError):
    """Raised when an actor-visible heuristic input violates the frozen contract."""


@dataclass(frozen=True)
class HistoricalQualityAggregate:
    """Masked historical-quality result that keeps missing distinct from zero."""

    history_valid: bool
    mean: float | None


def masked_historical_quality(
    quality: Sequence[float] | np.ndarray,
    valid_mask: Sequence[bool] | np.ndarray,
    ru_indices: Iterable[int] | None = None,
) -> HistoricalQualityAggregate:
    """Return the arithmetic mean over valid candidate RUs only."""

    values = np.asarray(quality, dtype=np.float64)
    mask = np.asarray(valid_mask, dtype=np.bool_)
    if values.ndim != 1 or mask.shape != values.shape:
        raise HeuristicPolicyError(
            "historical quality and mask must be equal-length vectors"
        )
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise HeuristicPolicyError(
            "historical quality must be finite and non-negative"
        )

    if ru_indices is None:
        indices = tuple(range(values.shape[0]))
    else:
        indices = tuple(ru_indices)
        if any(
            isinstance(index, bool)
            or not isinstance(index, (int, np.integer))
            or not 0 <= int(index) < values.shape[0]
            for index in indices
        ):
            raise HeuristicPolicyError(
                "candidate RU index is outside the historical-quality vector"
            )
        indices = tuple(int(index) for index in indices)

    valid_values = [
        float(values[index]) for index in indices if bool(mask[index])
    ]
    if not valid_values:
        return HistoricalQualityAggregate(history_valid=False, mean=None)
    return HistoricalQualityAggregate(
        history_valid=True,
        mean=math.fsum(valid_values) / len(valid_values),
    )


class HeuristicPolicy:
    """Deadline- and historical-link-aware deterministic proposal policy."""

    method_id = "heuristic"
    policy_stream_id = None

    def __init__(self, config: RunConfig, *, policy_seed: int | None = None) -> None:
        if not isinstance(config, RunConfig):
            raise TypeError("config must be a RunConfig")
        config.validate()
        resolved_seed = config.seed if policy_seed is None else policy_seed
        if (
            isinstance(resolved_seed, bool)
            or not isinstance(resolved_seed, int)
            or resolved_seed < 0
        ):
            raise HeuristicPolicyError(
                "policy_seed must be a non-negative integer"
            )
        self.policy_seed = resolved_seed
        self._slot_duration_s = float(config.environment.slot_duration_s)
        self._sampling_order = tuple(config.action.sampling_order)
        self._canonical = dict(config.action.canonical_inactive_values)
        self._resource_groups = tuple(
            tuple(int(ru) - 1 for ru in group)
            for group in config.environment.resource_groups
        )

    @property
    def seed_material(self) -> dict[str, int | None]:
        """Record interface-compatible seed metadata without owning an RNG."""

        return {
            "policy_seed": self.policy_seed,
            "policy_stream_id": self.policy_stream_id,
        }

    def act(self, observation: ActorObservation) -> ActionProposal:
        """Choose one legal proposal in the frozen seven-branch order."""

        masks = observation.action_masks
        if tuple(masks.sampling_order) != self._sampling_order:
            raise HeuristicPolicyError(
                "observation sampling order differs from the frozen config"
            )

        context: dict[str, str | int | float] = {}
        route = self._select_route(observation, context)
        context["route"] = route
        tx_select = self._select_tx(observation, context)
        context["tx_select"] = tx_select
        resource_group = self._select_resource_group(observation, context)
        context["resource_group"] = resource_group
        resource_width = self._select_resource_width(observation, context)
        context["resource_width"] = resource_width
        power_level = self._select_power(observation, context)
        context["power_level"] = power_level
        cpu_queue = self._select_cpu_queue(observation, context)
        context["cpu_queue"] = cpu_queue
        cpu_frequency = self._select_cpu_frequency(observation, context)

        proposal = ActionProposal(
            uav_id=observation.uav_id,
            route=route,
            tx_select=tx_select,
            resource_group=resource_group,
            resource_width=int(resource_width),
            power_level=float(power_level),
            cpu_queue=cpu_queue,
            cpu_frequency=float(cpu_frequency),
        )
        if not masks.is_legal(proposal):
            raise HeuristicPolicyError(
                "frozen heuristic produced an illegal proposal"
            )
        return proposal

    def _select_route(
        self,
        observation: ActorObservation,
        context: dict[str, str | int | float],
    ) -> str | int:
        legal = self._legal_values(observation, "route", context)
        head = observation.private_queues.unbound
        if not bool(head.head_valid_mask):
            return self._require_value(
                "route", self._canonical["route"], legal
            )
        slack = int(head.head_slack_slots)
        if slack <= 1:
            return self._require_value("route", "defer", legal)
        local_legal = "local" in legal
        local_total_cycles = (
            float(observation.private_queues.local_cpu.remaining_cycles)
            + float(head.head_remaining_cycles)
        )
        local_capacity_cycles = (
            float(observation.self_resources.max_cpu_frequency_hz)
            * self._slot_duration_s
            * (slack - 1)
        )
        if local_legal and local_total_cycles <= local_capacity_cycles:
            return "local"
        remote_candidates = self._integer_actions(
            legal, exclude=observation.uav_id
        )
        if slack >= 3 and remote_candidates:
            return min(
                remote_candidates,
                key=lambda destination: self._historical_key(
                    self._destination_quality(observation, destination),
                    destination,
                ),
            )
        if local_legal:
            return "local"
        return self._require_value("route", "defer", legal)

    def _select_tx(
        self,
        observation: ActorObservation,
        context: dict[str, str | int | float],
    ) -> str | int:
        legal = self._legal_values(observation, "tx_select", context)
        candidates = self._integer_actions(
            legal, exclude=observation.uav_id
        )
        if not candidates:
            return self._require_value(
                "tx_select", self._canonical["tx_select"], legal
            )
        queues = observation.private_queues.tx_by_destination

        def key(destination: int) -> tuple[float, int, float, int]:
            if not bool(queues.head_valid_mask[destination]):
                raise HeuristicPolicyError(
                    "legal TX candidate lacks an actor-visible queue head"
                )
            aggregate = self._destination_quality(
                observation, destination
            )
            invalid_flag, negative_mean, identifier = (
                self._historical_key(aggregate, destination)
            )
            return (
                float(queues.head_slack_slots[destination]),
                invalid_flag,
                negative_mean,
                identifier,
            )

        return min(candidates, key=key)

    def _select_resource_group(
        self,
        observation: ActorObservation,
        context: dict[str, str | int | float],
    ) -> str | int:
        legal = self._legal_values(
            observation, "resource_group", context
        )
        if context["tx_select"] == self._canonical["tx_select"]:
            return self._require_value(
                "resource_group",
                self._canonical["resource_group"],
                legal,
            )
        destination = context["tx_select"]
        if isinstance(destination, bool) or not isinstance(destination, int):
            raise HeuristicPolicyError(
                "active communication requires an integer destination"
            )
        groups = self._integer_actions(legal)
        if not groups:
            raise HeuristicPolicyError(
                "active communication has no legal resource group"
            )

        def key(group: int) -> tuple[int, float, int]:
            if not 1 <= group <= len(self._resource_groups):
                raise HeuristicPolicyError(
                    "legal resource group is outside the frozen config"
                )
            aggregate = self._destination_quality(
                observation,
                destination,
                self._resource_groups[group - 1],
            )
            return self._historical_key(aggregate, group)

        return min(groups, key=key)

    def _select_resource_width(
        self,
        observation: ActorObservation,
        context: dict[str, str | int | float],
    ) -> int:
        legal = self._legal_values(
            observation, "resource_width", context
        )
        widths = self._integer_actions(legal)
        if not widths:
            raise HeuristicPolicyError(
                "resource width branch has no legal integer action"
            )
        if context["tx_select"] == self._canonical["tx_select"]:
            return int(
                self._require_value(
                    "resource_width",
                    self._canonical["resource_width"],
                    legal,
                )
            )
        return max(widths)

    def _select_power(
        self,
        observation: ActorObservation,
        context: dict[str, str | int | float],
    ) -> float:
        legal = self._legal_values(
            observation, "power_level", context
        )
        if context["tx_select"] == self._canonical["tx_select"]:
            return float(
                self._require_value(
                    "power_level",
                    self._canonical["power_level"],
                    legal,
                )
            )
        positive = [
            float(value)
            for value in legal
            if self._is_number(value) and float(value) > 0.0
        ]
        if positive:
            return max(positive)
        return float(
            self._require_value(
                "power_level", self._canonical["power_level"], legal
            )
        )

    def _select_cpu_queue(
        self,
        observation: ActorObservation,
        context: dict[str, str | int | float],
    ) -> str | int:
        legal = self._legal_values(observation, "cpu_queue", context)
        candidates = self._integer_actions(legal)
        if not candidates:
            return self._require_value(
                "cpu_queue", self._canonical["cpu_queue"], legal
            )
        queues = observation.private_queues.cpu_by_source

        def key(source: int) -> tuple[int, int, int]:
            if not bool(queues.head_valid_mask[source]):
                raise HeuristicPolicyError(
                    "legal CPU candidate lacks an actor-visible queue head"
                )
            return (
                int(queues.head_slack_slots[source]),
                int(queues.head_task_id[source]),
                source,
            )

        return min(candidates, key=key)

    def _select_cpu_frequency(
        self,
        observation: ActorObservation,
        context: dict[str, str | int | float],
    ) -> float:
        legal = self._legal_values(
            observation, "cpu_frequency", context
        )
        selected_queue = context["cpu_queue"]
        if selected_queue == self._canonical["cpu_queue"]:
            return float(
                self._require_value(
                    "cpu_frequency",
                    self._canonical["cpu_frequency"],
                    legal,
                )
            )
        source = (
            observation.uav_id
            if selected_queue == "local"
            else selected_queue
        )
        if isinstance(source, bool) or not isinstance(source, int):
            raise HeuristicPolicyError(
                "active CPU queue must identify an integer source UAV"
            )
        queues = observation.private_queues.cpu_by_source
        if not bool(queues.head_valid_mask[source]):
            raise HeuristicPolicyError(
                "selected CPU queue lacks an actor-visible head"
            )
        slack = int(queues.head_slack_slots[source])
        if slack <= 0:
            raise HeuristicPolicyError(
                "serviceable CPU head must have positive slot-start slack"
            )
        required_hz = (
            float(queues.head_remaining_cycles[source])
            / (slack * self._slot_duration_s)
        )
        maximum_hz = float(
            observation.self_resources.max_cpu_frequency_hz
        )
        positive = sorted(
            (
                float(value) * maximum_hz,
                float(value),
                index,
            )
            for index, value in enumerate(legal)
            if self._is_number(value) and float(value) > 0.0
        )
        if not positive:
            return float(
                self._require_value(
                    "cpu_frequency",
                    self._canonical["cpu_frequency"],
                    legal,
                )
            )
        sufficient = [
            item for item in positive if item[0] >= required_hz
        ]
        if sufficient:
            return min(
                sufficient, key=lambda item: (item[0], item[2])
            )[1]
        return max(
            positive, key=lambda item: (item[0], -item[2])
        )[1]

    @staticmethod
    def _historical_key(
        aggregate: HistoricalQualityAggregate,
        identifier: int,
    ) -> tuple[int, float, int]:
        if aggregate.history_valid:
            assert aggregate.mean is not None
            return (0, -float(aggregate.mean), identifier)
        return (1, 0.0, identifier)

    @staticmethod
    def _integer_actions(
        values: Sequence[object],
        *,
        exclude: int | None = None,
    ) -> list[int]:
        return [
            int(value)
            for value in values
            if not isinstance(value, bool)
            and isinstance(value, (int, np.integer))
            and (exclude is None or int(value) != exclude)
        ]

    @staticmethod
    def _is_number(value: object) -> bool:
        return not isinstance(value, bool) and isinstance(
            value, (int, float, np.number)
        )

    def _destination_quality(
        self,
        observation: ActorObservation,
        destination: int,
        ru_indices: Iterable[int] | None = None,
    ) -> HistoricalQualityAggregate:
        quality = observation.edge_history.historical_quality
        valid = observation.edge_history.quality_valid_mask
        if (
            isinstance(destination, bool)
            or not isinstance(destination, int)
            or not 0 <= destination < quality.shape[0]
        ):
            raise HeuristicPolicyError(
                "destination is outside the actor edge-history tensor"
            )
        return masked_historical_quality(
            quality[destination],
            valid[destination],
            ru_indices,
        )

    @staticmethod
    def _legal_values(
        observation: ActorObservation,
        branch: str,
        context: dict[str, str | int | float],
    ) -> list[object]:
        masks = observation.action_masks
        domain = masks.domain_for(branch)
        mask = np.asarray(
            masks.mask_for(branch, context), dtype=np.bool_
        )
        if mask.shape != (len(domain),) or not np.any(mask):
            raise HeuristicPolicyError(
                f"branch {branch!r} has no legal masked action"
            )
        return [
            value
            for value, allowed in zip(domain, mask)
            if bool(allowed)
        ]

    @staticmethod
    def _require_value(
        branch: str,
        value: object,
        legal: Sequence[object],
    ):
        if value not in legal:
            raise HeuristicPolicyError(
                f"frozen {branch} fallback {value!r} "
                "is not legal in the current mask"
            )
        return value


__all__ = [
    "HeuristicPolicy",
    "HeuristicPolicyError",
    "HistoricalQualityAggregate",
    "masked_historical_quality",
]
