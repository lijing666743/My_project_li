"""Fixed-reference post-settlement team reward accounting.

The calculator in this module is intentionally stateless across slots.  Its
normalization references and weights are supplied once at construction time;
no running maximum, batch statistic, or episode observation can change them.
Callers must invoke it after physical service and task settlement, but before
the current slot's arrivals are registered.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from .tasks import Task, TaskOutcome


class RewardError(ValueError):
    """Raised when reward inputs violate the frozen timing or numeric contract."""


def _finite(value: float, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    converted = float(value)
    if not math.isfinite(converted):
        raise RewardError(f"{name} must be finite")
    if positive and converted <= 0.0:
        raise RewardError(f"{name} must be positive")
    return converted


def _finite_nonnegative(value: float, name: str) -> float:
    converted = _finite(value, name)
    if converted < 0.0:
        raise RewardError(f"{name} must be non-negative")
    return converted


@dataclass(frozen=True)
class RewardReferences:
    """Episode-invariant physical references used by the frozen reward."""

    reference_rate_bps: float
    reference_cpu_frequency_hz: float
    workload_reference_s: float
    task_count_reference: float
    active_energy_reference_j: float

    def __post_init__(self) -> None:
        for name in (
            "reference_rate_bps",
            "reference_cpu_frequency_hz",
            "workload_reference_s",
            "task_count_reference",
            "active_energy_reference_j",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name, positive=True))


@dataclass(frozen=True)
class RewardWeights:
    """Fixed non-negative coefficients for completion and three penalties."""

    completion: float
    expiration: float
    workload: float
    energy: float

    def __post_init__(self) -> None:
        for name in ("completion", "expiration", "workload", "energy"):
            object.__setattr__(self, name, _finite_nonnegative(getattr(self, name), name))


@dataclass(frozen=True)
class RewardTerms:
    """Auditable scalar terms from one post-settlement/pre-arrival boundary."""

    slot: int
    completed_task_count: int
    expired_task_count: int
    urgent_workload_s: float
    actual_energy_j: float
    normalized_completed: float
    normalized_expired: float
    normalized_workload: float
    normalized_energy: float
    completion_component: float
    expiration_penalty: float
    workload_penalty: float
    energy_penalty: float
    reward: float

    def to_dict(self) -> dict[str, int | float]:
        """Return a deterministic JSON-safe audit record."""

        return {
            "slot": self.slot,
            "completed_task_count": self.completed_task_count,
            "expired_task_count": self.expired_task_count,
            "urgent_workload_s": self.urgent_workload_s,
            "actual_energy_j": self.actual_energy_j,
            "normalized_completed": self.normalized_completed,
            "normalized_expired": self.normalized_expired,
            "normalized_workload": self.normalized_workload,
            "normalized_energy": self.normalized_energy,
            "completion_component": self.completion_component,
            "expiration_penalty": self.expiration_penalty,
            "workload_penalty": self.workload_penalty,
            "energy_penalty": self.energy_penalty,
            "reward": self.reward,
        }


class RewardCalculator:
    """Evaluate the unique frozen reward with immutable normalization values."""

    def __init__(self, references: RewardReferences, weights: RewardWeights) -> None:
        if not isinstance(references, RewardReferences):
            raise TypeError("references must be a RewardReferences instance")
        if not isinstance(weights, RewardWeights):
            raise TypeError("weights must be a RewardWeights instance")
        self.references = references
        self.weights = weights

    def calculate(
        self,
        *,
        slot: int,
        all_tasks: Iterable[Task],
        settled_tasks: Iterable[Task],
        actual_energy_j: float,
    ) -> RewardTerms:
        """Calculate reward after settlement and before slot-end arrivals.

        ``all_tasks`` is the lifecycle's full task collection at that boundary.
        ``settled_tasks`` contains only tasks completed or expired in ``slot``.
        Current-slot arrivals must not yet be present in either collection.
        """

        if isinstance(slot, bool) or not isinstance(slot, int) or slot < 0:
            raise RewardError("slot must be a non-negative integer")
        energy = _finite_nonnegative(actual_energy_j, "actual_energy_j")
        tasks = self._unique_tasks(all_tasks, "all_tasks")
        settled = self._unique_tasks(settled_tasks, "settled_tasks")
        by_id = {task.task_id: task for task in tasks}
        for task in settled:
            if by_id.get(task.task_id) is not task:
                raise RewardError("settled_tasks must reference objects in all_tasks")
            if task.outcome not in {TaskOutcome.DONE, TaskOutcome.EXPIRED}:
                raise RewardError("settled_tasks may contain only done or expired tasks")

        active_tasks = tuple(task for task in tasks if not task.is_terminal)
        urgent_workload = math.fsum(
            self._urgent_workload_for_task(task, slot)
            for task in active_tasks
        )
        completed = sum(task.outcome is TaskOutcome.DONE for task in settled)
        expired = sum(task.outcome is TaskOutcome.EXPIRED for task in settled)

        refs = self.references
        normalized_completed = completed / refs.task_count_reference
        normalized_expired = expired / refs.task_count_reference
        normalized_workload = urgent_workload / refs.workload_reference_s
        normalized_energy = energy / refs.active_energy_reference_j

        completion_component = self.weights.completion * normalized_completed
        expiration_penalty = self.weights.expiration * normalized_expired
        workload_penalty = self.weights.workload * normalized_workload
        energy_penalty = self.weights.energy * normalized_energy
        reward = (
            completion_component
            - expiration_penalty
            - workload_penalty
            - energy_penalty
        )
        if not math.isfinite(reward):
            raise RewardError("reward must be finite")
        return RewardTerms(
            slot=slot,
            completed_task_count=completed,
            expired_task_count=expired,
            urgent_workload_s=urgent_workload,
            actual_energy_j=energy,
            normalized_completed=normalized_completed,
            normalized_expired=normalized_expired,
            normalized_workload=normalized_workload,
            normalized_energy=normalized_energy,
            completion_component=completion_component,
            expiration_penalty=expiration_penalty,
            workload_penalty=workload_penalty,
            energy_penalty=energy_penalty,
            reward=reward,
        )

    def _urgent_workload_for_task(self, task: Task, slot: int) -> float:
        remaining_bits = _finite_nonnegative(task.remaining_bits, "task.remaining_bits")
        remaining_cycles = _finite_nonnegative(task.remaining_cycles, "task.remaining_cycles")
        denominator = max(1, task.deadline_slot - slot + 1)
        return (
            remaining_bits / self.references.reference_rate_bps
            + remaining_cycles / self.references.reference_cpu_frequency_hz
        ) / denominator

    @staticmethod
    def _unique_tasks(tasks: Iterable[Task], name: str) -> tuple[Task, ...]:
        materialized = tuple(tasks)
        if any(not isinstance(task, Task) for task in materialized):
            raise TypeError(f"{name} must contain only Task instances")
        ids = [task.task_id for task in materialized]
        if len(ids) != len(set(ids)):
            raise RewardError(f"{name} contains duplicate task IDs")
        return tuple(sorted(materialized, key=lambda task: task.task_id))


def compute_reward(
    *,
    slot: int,
    all_tasks: Iterable[Task],
    settled_tasks: Iterable[Task],
    actual_energy_j: float,
    references: RewardReferences,
    weights: RewardWeights,
) -> RewardTerms:
    """Functional facade for callers that do not retain a calculator object."""

    return RewardCalculator(references, weights).calculate(
        slot=slot,
        all_tasks=all_tasks,
        settled_tasks=settled_tasks,
        actual_energy_j=actual_energy_j,
    )


__all__ = [
    "RewardCalculator",
    "RewardError",
    "RewardReferences",
    "RewardTerms",
    "RewardWeights",
    "compute_reward",
]
