"""Execution registry shared by interactive and direct launch paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from .config import RunConfig


Handler = Callable[[RunConfig], "RunResult"]


@dataclass(frozen=True)
class RunResult:
    """Artifact-aware lifecycle result returned by every registered handler."""

    status: str
    run_id: str
    mode: str
    method_id: str
    message: str
    artifacts: tuple[str, ...] = ()

    @property
    def is_success(self) -> bool:
        return self.status == "completed"

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "run_id": self.run_id,
            "mode": self.mode,
            "method_id": self.method_id,
            "message": self.message,
            "artifacts": list(self.artifacts),
        }


@dataclass(frozen=True)
class RegistryEntry:
    mode: str
    method_id: str
    handler: Handler


class Registry:
    """Map a validated ``(mode, method_id)`` pair to one backend handler."""

    def __init__(self) -> None:
        self._entries: dict[tuple[str, str], RegistryEntry] = {}

    def register(self, mode: str, method_id: str, handler: Handler) -> None:
        key = (mode, method_id)
        if key in self._entries:
            raise ValueError(f"registry entry already exists: {mode}/{method_id}")
        self._entries[key] = RegistryEntry(mode, method_id, handler)

    def resolve(self, mode: str, method_id: str) -> Handler:
        try:
            return self._entries[(mode, method_id)].handler
        except KeyError as exc:
            available = ", ".join(f"{m}/{i}" for m, i in sorted(self._entries))
            raise KeyError(f"no runner registered for {mode}/{method_id}; available: {available}") from exc

    def entries(self) -> tuple[RegistryEntry, ...]:
        return tuple(self._entries[key] for key in sorted(self._entries))

    def pairs(self) -> tuple[tuple[str, str], ...]:
        return tuple(sorted(self._entries))


def unavailable_handler(config: RunConfig) -> RunResult:
    """Return an explicit blocked state without fabricating run artifacts."""

    return RunResult(
        status="unavailable",
        run_id=config.run_id,
        mode=config.mode,
        method_id=config.method_id,
        message=(
            "The environment backend for this mode is not implemented yet. "
            "No metrics, dashboard, checkpoint, or synthetic result was generated."
        ),
    )


def environment_sanity_handler(config: RunConfig) -> RunResult:
    """Run the real environment reset/step sanity validation."""

    from .env.validation import run_environment_sanity

    outcome = run_environment_sanity(config)
    return RunResult(
        status=outcome.status,
        run_id=config.run_id,
        mode=config.mode,
        method_id=config.method_id,
        message=outcome.message,
    )


def gate0_handler(config: RunConfig) -> RunResult:
    """Run the explicit end-to-end G0-01..G0-21 test module."""

    from .env.validation import run_gate0_tests

    outcome = run_gate0_tests(config)
    return RunResult(
        status=outcome.status,
        run_id=config.run_id,
        mode=config.mode,
        method_id=config.method_id,
        message=outcome.message,
    )


def random_rollout_handler(config: RunConfig) -> RunResult:
    """Run one real masked-random episode through the unified backend."""

    from .policies.random_policy import RandomPolicy
    from .rollout import RolloutRunner

    try:
        outcome = RolloutRunner(config, RandomPolicy(config.seed)).run()
    except Exception as exc:
        return RunResult(
            status="failed",
            run_id=config.run_id,
            mode=config.mode,
            method_id=config.method_id,
            message=f"random policy rollout failed: {exc}",
        )
    tasks = outcome.summary["tasks"]
    return RunResult(
        status="completed",
        run_id=config.run_id,
        mode=config.mode,
        method_id=config.method_id,
        message=(
            "random policy rollout completed through U2UMECEnvironment: "
            f"generated={tasks['generated']}, completed={tasks['completed']}, "
            f"expired={tasks['expired']}, truncated={tasks['truncated']}"
        ),
        artifacts=outcome.artifacts,
    )


def heuristic_rollout_handler(config: RunConfig) -> RunResult:
    """Run one real deterministic heuristic episode through the unified backend."""

    from .policies.heuristic_policy import HeuristicPolicy
    from .rollout import RolloutRunner

    try:
        outcome = RolloutRunner(config, HeuristicPolicy(config)).run()
    except Exception as exc:
        return RunResult(
            status="failed",
            run_id=config.run_id,
            mode=config.mode,
            method_id=config.method_id,
            message=f"heuristic policy rollout failed: {exc}",
        )
    tasks = outcome.summary["tasks"]
    return RunResult(
        status="completed",
        run_id=config.run_id,
        mode=config.mode,
        method_id=config.method_id,
        message=(
            "heuristic policy rollout completed through U2UMECEnvironment: "
            f"generated={tasks['generated']}, completed={tasks['completed']}, "
            f"expired={tasks['expired']}, truncated={tasks['truncated']}"
        ),
        artifacts=outcome.artifacts,
    )


def build_default_registry() -> Registry:
    """Register the frozen menu surface with honest early-stage handlers."""

    registry = Registry()
    pairs: Iterable[tuple[str, str]] = (
        ("environment_sanity", "environment"),
        ("gate0", "environment"),
        ("random", "random"),
        ("heuristic", "heuristic"),
        ("baseline", "factorized_action_gat_qmix"),
        ("rl", "ca_gat_mappo"),
        ("rl", "factorized_action_gat_qmix"),
        ("evaluation", "environment"),
        ("evaluation", "random"),
        ("evaluation", "heuristic"),
        ("evaluation", "ca_gat_mappo"),
        ("evaluation", "factorized_action_gat_qmix"),
        ("ablation", "ca_gat_mappo"),
        ("plot", "environment"),
    )
    implemented_handlers: dict[tuple[str, str], Handler] = {
        ("environment_sanity", "environment"): environment_sanity_handler,
        ("gate0", "environment"): gate0_handler,
        ("random", "random"): random_rollout_handler,
        ("heuristic", "heuristic"): heuristic_rollout_handler,
    }
    for mode, method_id in pairs:
        handler = implemented_handlers.get((mode, method_id), unavailable_handler)
        registry.register(mode, method_id, handler)
    return registry


__all__ = [
    "Handler",
    "Registry",
    "RegistryEntry",
    "RunResult",
    "build_default_registry",
    "environment_sanity_handler",
    "gate0_handler",
    "heuristic_rollout_handler",
    "random_rollout_handler",
    "unavailable_handler",
]
