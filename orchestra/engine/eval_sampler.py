from __future__ import annotations
"""Deterministic production trace sampling for the eval harness.

Decides whether a given run should be auto-judged, controlling eval cost in
production by evaluating only a fraction of runs. The decision is a stable
function of ``run_id`` (no RNG), so the same run always yields the same verdict
and the sampled fraction is uniform across the run-id space.
"""
import hashlib

_MAX = 0xFFFFFFFF


def should_sample(run_id: str, rate: float) -> bool:
    """Return True if the run should be sampled for evaluation.

    Args:
        run_id: Unique run identifier (hashed for a stable [0,1) position).
        rate: Sampling fraction. ``>= 1.0`` samples all, ``<= 0.0`` samples none.
    """
    if rate >= 1.0:
        return True
    if rate <= 0.0:
        return False
    digest = hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:8]
    position = int(digest, 16) / _MAX
    return position < rate
