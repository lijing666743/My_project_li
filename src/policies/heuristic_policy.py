"""Explicit blocker for the currently under-specified heuristic baseline."""

from __future__ import annotations


MISSING_HEURISTIC_DECISIONS = (
    "route preference among local, defer, and legal remote destinations",
    "tx_select preference when multiple EDF queue heads are available",
    "resource_group and resource_width preference",
    "power_level preference",
    "cpu_queue preference when multiple EDF queue heads are available",
    "cpu_frequency preference",
)


class HeuristicSpecificationBlocker(RuntimeError):
    """Raised instead of inventing a heuristic that is absent from Section 4."""


def heuristic_blocker_message() -> str:
    missing = "; ".join(MISSING_HEURISTIC_DECISIONS)
    return (
        "HEURISTIC SPECIFICATION BLOCKER: Frozen Section 4 limits the baseline "
        "to EDF, legal destination/queue preference, and valid-mask fallback, "
        f"but does not uniquely define: {missing}. No heuristic rollout or "
        "artifact was generated."
    )


class HeuristicPolicy:
    """Non-instantiable placeholder until the frozen decision rules are unique."""

    method_id = "heuristic"
    policy_seed = None
    policy_stream_id = None

    def __init__(self) -> None:
        raise HeuristicSpecificationBlocker(heuristic_blocker_message())


__all__ = [
    "HeuristicPolicy",
    "HeuristicSpecificationBlocker",
    "MISSING_HEURISTIC_DECISIONS",
    "heuristic_blocker_message",
]
