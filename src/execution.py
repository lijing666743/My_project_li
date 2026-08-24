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

    def __post_init__(self) -> None:
        if self.resume_from is not None and not isinstance(self.resume_from, Path):
            object.__setattr__(self, "resume_from", Path(self.resume_from))

    @classmethod
    def from_resume_path(cls, resume_from: str | Path | None) -> "ExecutionContext":
        return cls(None if resume_from is None else Path(resume_from))


def validate_execution_context(
    config: "RunConfig",
    context: ExecutionContext | None,
) -> ExecutionContext:
    """Fail fast on unsupported resume requests before trainer construction."""

    resolved = context or ExecutionContext()
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
