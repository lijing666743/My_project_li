"""Registry adapter for Validation Gate V1 Formal Evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping

from ..execution import ExecutionContext, validate_execution_context

if TYPE_CHECKING:
    from ..config import RunConfig
    from ..registry import RunResult


_ALLOWED_DISPATCH_OVERRIDE_KEYS = frozenset({"mode", "method_id"})


def formal_evaluation_handler(
    config: "RunConfig",
    *,
    execution_context: ExecutionContext | None = None,
) -> "RunResult":
    """Run the fixed Validation Gate V1 evaluator entry."""

    from ..registry import RunResult
    from .protocol import EvaluationProtocol
    from .runner import EvaluationWorkspaceLayout, FormalEvaluationRunner

    try:
        context = validate_execution_context(config, execution_context)
        _validate_cli_config_boundary(config)
        assert context.evaluate_from is not None
        assert context.evaluation_device is not None
        layout = EvaluationWorkspaceLayout.discover_formal_v1()
        protocol_path = layout.evaluator_root / "configs" / "formal_validation.json"
        protocol = EvaluationProtocol.from_path(protocol_path)
        evaluation = FormalEvaluationRunner(
            config,
            context.evaluate_from,
            evaluation_device=context.evaluation_device,
            protocol=protocol,
            workspace_layout=layout,
        ).run(write_artifacts=True)
    except Exception as exc:
        return RunResult(
            status="failed",
            run_id=config.run_id,
            mode=config.mode,
            method_id=config.method_id,
            message=f"formal evaluation failed before completion: {exc}",
        )
    return RunResult(
        status="completed",
        run_id=evaluation.evaluation_run_id,
        mode=config.mode,
        method_id=config.method_id,
        message=(
            "formal evaluation completed without training side effects: "
            f"evaluation_run_id={evaluation.evaluation_run_id}, "
            f"episodes={len(evaluation.episodes)}, "
            f"methods={','.join(evaluation.manifest['method_suite'])}"
        ),
        artifacts=evaluation.artifacts,
    )


def _validate_cli_config_boundary(config: "RunConfig") -> None:
    if config.source_config_path is not None:
        raise ValueError(
            "Validation Gate V1 does not accept a CLI configuration file"
        )
    for source_name, overrides in (
        ("CLI", config.cli_overrides),
        ("interactive", config.interactive_overrides),
    ):
        _reject_non_dispatch_overrides(source_name, overrides)


def _reject_non_dispatch_overrides(
    source_name: str,
    overrides: Mapping[str, object],
) -> None:
    forbidden = sorted(set(overrides) - _ALLOWED_DISPATCH_OVERRIDE_KEYS)
    if forbidden:
        raise ValueError(
            "Validation Gate V1 accepts only mode/method dispatch overrides; "
            f"forbidden {source_name} keys: {forbidden}"
        )


__all__ = ["formal_evaluation_handler"]
