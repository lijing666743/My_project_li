"""Fail-fast, atomic publication helpers for immutable run artifacts.

Checkpoint serialization keeps its own schema-aware writer. This module is
limited to ordinary result files such as snapshots, metrics, CSV inputs, and
dashboard images.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Iterable, Mapping


ORDINARY_ARTIFACT_KEYS = (
    "config_snapshot",
    "raw_metrics",
    "aggregate_metrics",
    "dashboard_csv",
    "figure_input",
    "dashboard_png",
)


class ArtifactError(RuntimeError):
    """Base error for immutable ordinary-artifact publication."""


class ArtifactConflictError(ArtifactError):
    """Raised before publication when any target path already exists."""


class ArtifactWriteError(ArtifactError):
    """Raised when staging or publishing a new artifact fails."""


def _path_entry_exists(path: Path) -> bool:
    """Return true for every existing directory entry, including broken links."""

    return os.path.lexists(path)


def require_artifact_targets_absent(
    paths: Iterable[str | Path],
    *,
    group_name: str = "artifact group",
) -> tuple[Path, ...]:
    """Validate a complete target group before any member is written."""

    targets = tuple(Path(path) for path in paths)
    if not targets:
        raise ValueError("artifact target group must not be empty")
    if len(set(targets)) != len(targets):
        raise ArtifactWriteError(f"{group_name} contains duplicate target paths")
    conflicts = tuple(path for path in targets if _path_entry_exists(path))
    if conflicts:
        rendered = ", ".join(str(path) for path in conflicts)
        raise ArtifactConflictError(
            f"{group_name} target already exists; refusing overwrite: {rendered}"
        )
    return targets


def ordinary_artifact_targets(paths: Mapping[str, str]) -> tuple[Path, ...]:
    """Return every ordinary path reserved for one canonical run identity."""

    return tuple(Path(paths[key]) for key in ORDINARY_ARTIFACT_KEYS)


def preflight_rollout_artifacts(paths: Mapping[str, str]) -> tuple[Path, ...]:
    """Reject an existing identity before a baseline environment is started."""

    return require_artifact_targets_absent(
        ordinary_artifact_targets(paths),
        group_name="rollout artifact group",
    )


def preflight_formal_training_artifacts(
    paths: Mapping[str, str],
    *,
    resume: bool,
) -> tuple[Path, ...]:
    """Apply fresh/resume conflict rules before formal Trainer construction."""

    ordinary = ordinary_artifact_targets(paths)
    final_checkpoint = Path(paths["final_checkpoint"])
    if resume:
        targets = (*ordinary, final_checkpoint)
        group_name = "formal resume final artifact group"
    else:
        run_directory = Path(paths["config_snapshot"]).parent
        targets = (run_directory, *ordinary[3:])
        group_name = "fresh formal run artifact group"
    return require_artifact_targets_absent(targets, group_name=group_name)


def atomic_write_bytes_group(
    artifacts: Iterable[tuple[str | Path, bytes]],
    *,
    group_name: str = "artifact group",
) -> tuple[Path, ...]:
    """Stage, fsync, and publish a complete immutable artifact group."""

    items = tuple((Path(path), content) for path, content in artifacts)
    targets = require_artifact_targets_absent(
        (path for path, _content in items),
        group_name=group_name,
    )
    if any(not isinstance(content, bytes) for _path, content in items):
        raise TypeError("artifact contents must be bytes")

    staged: list[tuple[Path, Path]] = []
    try:
        for target, content in items:
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w+b",
                prefix=f".{target.name}.",
                suffix=".tmp",
                dir=target.parent,
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                staged.append((target, temporary))
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())

        require_artifact_targets_absent(targets, group_name=group_name)
        for target, temporary in staged:
            if _path_entry_exists(target):
                raise ArtifactConflictError(
                    f"{group_name} target already exists; refusing overwrite: {target}"
                )
            os.replace(temporary, target)
            if not target.is_file():
                raise ArtifactWriteError(
                    f"atomic artifact publication did not create a file: {target}"
                )
    except ArtifactError:
        raise
    except Exception as exc:
        raise ArtifactWriteError(
            f"atomic {group_name} publication failed: {exc}"
        ) from exc
    finally:
        for _target, temporary in staged:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
    return targets


def atomic_write_text_group(
    artifacts: Iterable[tuple[str | Path, str]],
    *,
    group_name: str = "artifact group",
) -> tuple[Path, ...]:
    """Encode UTF-8 text and publish it through the immutable group writer."""

    return atomic_write_bytes_group(
        ((path, content.encode("utf-8")) for path, content in artifacts),
        group_name=group_name,
    )


def atomic_write_text(path: str | Path, content: str) -> Path:
    """Publish one UTF-8 text artifact without replacing an existing target."""

    return atomic_write_text_group(
        ((path, content),),
        group_name="text artifact",
    )[0]


__all__ = [
    "ArtifactConflictError",
    "ArtifactError",
    "ArtifactWriteError",
    "ORDINARY_ARTIFACT_KEYS",
    "atomic_write_bytes_group",
    "atomic_write_text",
    "atomic_write_text_group",
    "ordinary_artifact_targets",
    "preflight_formal_training_artifacts",
    "preflight_rollout_artifacts",
    "require_artifact_targets_absent",
]
