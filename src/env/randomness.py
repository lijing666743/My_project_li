"""Deterministic NumPy RNG streams owned by the environment.

Each generator is derived exactly from ``SeedSequence([master_seed, stream_id])``
as frozen by Section 4.  Environment components receive generators explicitly;
they never seed or consume NumPy's global RNG.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..config import RunConfig


def make_rng(master_seed: int, stream_id: int) -> np.random.Generator:
    """Create one deterministic, independent NumPy generator."""

    if isinstance(master_seed, bool) or not isinstance(master_seed, int) or master_seed < 0:
        raise ValueError("master_seed must be a non-negative integer")
    if isinstance(stream_id, bool) or not isinstance(stream_id, int) or stream_id < 0:
        raise ValueError("stream_id must be a non-negative integer")
    return np.random.default_rng(np.random.SeedSequence([master_seed, stream_id]))


def rng_from_run_config(config: "RunConfig", stream_name: str) -> np.random.Generator:
    """Create the named stream from the canonical ``RunConfig`` metadata."""

    try:
        stream_id = config.reproducibility.stream_ids[stream_name]
    except KeyError as exc:
        raise KeyError(f"unknown reproducibility stream {stream_name!r}") from exc
    return make_rng(config.seed, stream_id)


__all__ = ["make_rng", "rng_from_run_config"]
