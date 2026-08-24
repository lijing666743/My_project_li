"""Registry adapter for the Formal Evaluation Runner."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..execution import ExecutionContext, validate_execution_context

if TYPE_CHECKING:
    from ..config import RunConfig
    from ..registry import RunResult


def formal_evaluation_handler(
    config: "RunConfig",
    *,
    execution_context: ExecutionContext | None = None,
) -> "RunResult":
    """Run the one shared CLI/interactive formal evaluator entry."""

    from ..registry import RunResult
    from .runner import FormalEvaluationRunner

    context = validate_execution_context(config, execution_context)
    assert context.evaluate_from is not None
    assert context.evaluation_device is not None
    try:
        evaluation = FormalEvaluationRunner(
            config,
            context.evaluate_from,
            evaluation_device=context.evaluation_device,
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


__all__ = ["formal_evaluation_handler"]
