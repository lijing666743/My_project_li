"""Common runner entry point for all configuration execution paths."""

from __future__ import annotations

from dataclasses import dataclass

from .config import RunConfig, write_config_snapshot
from .registry import Registry, RunResult, build_default_registry


@dataclass
class Runner:
    """Dispatch a canonical ``RunConfig`` through one registry.

    The current wave deliberately contains no environment backend.  A future
    handler can return ``completed`` and call ``write_config_snapshot`` only
    after it has produced real raw metrics.  The default handlers return
    ``unavailable`` and therefore create no fake artifacts.
    """

    registry: Registry

    def __init__(self, registry: Registry | None = None) -> None:
        self.registry = registry or build_default_registry()

    def run(self, config: RunConfig) -> RunResult:
        config.validate()
        handler = self.registry.resolve(config.mode, config.method_id)
        result = handler(config)
        if not isinstance(result, RunResult):
            raise TypeError("registered runner handlers must return RunResult")
        return result

    def write_snapshot_after_success(self, config: RunConfig, result: RunResult):
        """Persist a config snapshot only for a genuinely completed run."""

        if not result.is_success:
            raise RuntimeError("cannot write a successful-run snapshot for a non-completed result")
        return write_config_snapshot(config)


def run_config(config: RunConfig, registry: Registry | None = None) -> RunResult:
    """Functional façade used by tests and future batch orchestration."""

    return Runner(registry=registry).run(config)


__all__ = ["Runner", "run_config"]
