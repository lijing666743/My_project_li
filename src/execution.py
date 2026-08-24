"""Explicit launch-time execution instructions kept outside ``RunConfig``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .config import ConfigError

if TYPE_CHECKING:
    from .config import RunConfig


@dataclass(frozen=True)
class ExecutionContext:
    """Non-scientific instructions that control one launcher invocation."""

    resume_from: Path | None = None
    evaluate_from: Path | None = None
    evaluation_device: str | None = None

    def __post_init__(self) -> None:
        if self.resume_from is not None and not isinstance(self.resume_from, Path):
            object.__setattr__(self, "resume_from", Path(self.resume_from))
        if self.evaluate_from is not None and not isinstance(self.evaluate_from, Path):
            object.__setattr__(self, "evaluate_from", Path(self.evaluate_from))

    @classmethod
    def from_resume_path(cls, resume_from: str | Path | None) -> "ExecutionContext":
        return cls(None if resume_from is None else Path(resume_from))

    @classmethod
    def from_paths(
        cls,
        *,
        resume_from: str | Path | None = None,
        evaluate_from: str | Path | None = None,
        evaluation_device: str | None = None,
    ) -> "ExecutionContext":
        return cls(
            resume_from=None if resume_from is None else Path(resume_from),
            evaluate_from=None if evaluate_from is None else Path(evaluate_from),
            evaluation_device=evaluation_device,
        )

    @classmethod
    def from_evaluation_path(
        cls,
        evaluate_from: str | Path | None,
        evaluation_device: str | None,
    ) -> "ExecutionContext":
        return cls.from_paths(
            evaluate_from=evaluate_from,
            evaluation_device=evaluation_device,
        )


def validate_execution_context(
    config: "RunConfig",
    context: ExecutionContext | None,
) -> ExecutionContext:
    """Fail fast on incompatible training/evaluation launch instructions."""

    resolved = context or ExecutionContext()
    if resolved.evaluate_from is not None:
        if resolved.resume_from is not None:
            raise ConfigError("--resume-from and --evaluate-from are mutually exclusive")
        if config.mode != "evaluation":
            raise ConfigError("--evaluate-from requires mode='evaluation'")
        if config.method_id != "ca_gat_mappo":
            raise ConfigError(
                "--evaluate-from requires method_id='ca_gat_mappo'"
            )
        if resolved.evaluation_device not in {"cpu", "cuda"}:
            raise ConfigError(
                "--evaluate-from requires explicit --evaluation-device cpu or cuda"
            )
        path = resolved.evaluate_from
        if not path.exists():
            raise ConfigError(f"evaluation checkpoint path does not exist: {path}")
        if not path.is_file():
            raise ConfigError(
                f"evaluation checkpoint path is not a regular file: {path}"
            )
        return resolved
    if config.mode == "evaluation":
        raise ConfigError("mode='evaluation' requires --evaluate-from")
    if resolved.evaluation_device is not None:
        raise ConfigError("--evaluation-device requires --evaluate-from")
    if resolved.resume_from is None:
        return resolved
    if config.mode != "rl":
        raise ConfigError("--resume-from requires mode='rl'")
    if config.method_id != "ca_gat_mappo":
        raise ConfigError("--resume-from requires method_id='ca_gat_mappo'")
    if config.launch_profile != "rl-formal":
        raise ConfigError("--resume-from requires launch_profile='rl-formal'")

    path = resolved.resume_from
    if not path.exists():
        raise ConfigError(f"checkpoint path does not exist: {path}")
    if not path.is_file():
        raise ConfigError(f"checkpoint path is not a regular file: {path}")
    return resolved


__all__ = ["ExecutionContext", "validate_execution_context"]
