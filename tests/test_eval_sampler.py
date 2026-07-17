"""Tests for deterministic production trace sampler (Faz 4 Block E)."""
from orchestra.engine import eval_sampler


def test_rate_one_samples_all():
    assert eval_sampler.should_sample("any-run", 1.0) is True


def test_rate_zero_samples_none():
    assert eval_sampler.should_sample("any-run", 0.0) is False


def test_deterministic_for_same_input():
    a = eval_sampler.should_sample("run-42", 0.5)
    b = eval_sampler.should_sample("run-42", 0.5)
    assert a == b


def test_rate_half_roughly_half():
    ids = [f"run-{i}" for i in range(200)]
    sampled = sum(eval_sampler.should_sample(rid, 0.5) for rid in ids)
    frac = sampled / len(ids)
    assert 0.3 < frac < 0.7
