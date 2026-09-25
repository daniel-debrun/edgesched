from __future__ import annotations

import itertools
import random

import pytest

from edgesched.profile import (
    LinearProfile,
    NoiseModel,
    TableProfile,
    fit_linear,
    load_profile_json,
    measure,
)


def test_linear_profile_predict_and_scale():
    p = LinearProfile(0.002, 0.0005)
    assert p.predict(1) == pytest.approx(0.0025)
    assert p.predict(4) == pytest.approx(0.004)
    assert p.scaled(2.0).predict(4) == pytest.approx(0.008)
    with pytest.raises(ValueError):
        p.predict(0)
    with pytest.raises(ValueError):
        LinearProfile(-1.0, 0.0)


def test_table_profile_interpolates_and_extrapolates():
    t = TableProfile({1: 1.0, 4: 2.5, 8: 4.5})
    assert t.predict(1) == 1.0
    assert t.predict(2) == pytest.approx(1.5)
    assert t.predict(6) == pytest.approx(3.5)
    assert t.predict(16) == pytest.approx(8.5)  # slope of last segment
    assert TableProfile({2: 3.0}).predict(1) == 3.0
    with pytest.raises(ValueError):
        TableProfile({})


def test_fit_linear_recovers_coefficients():
    pts = {b: 0.003 + 0.0007 * b for b in (1, 2, 4, 8, 16)}
    fit = fit_linear(pts)
    assert fit.a == pytest.approx(0.003)
    assert fit.c == pytest.approx(0.0007)


def test_noise_model_is_seeded_and_has_tail():
    n = NoiseModel(sigma=0.0, tail_prob=0.5, tail_scale=3.0)
    rng = random.Random(0)
    vals = [n.sample(1.0, rng) for _ in range(200)]
    assert set(vals) == {1.0, 3.0}
    rng1, rng2 = random.Random(7), random.Random(7)
    m = NoiseModel(sigma=0.1)
    assert [m.sample(1.0, rng1) for _ in range(5)] == [m.sample(1.0, rng2) for _ in range(5)]


def test_measure_with_fake_clock_and_json_roundtrip(tmp_path):
    ticks = itertools.count()

    def clock():
        return next(ticks) * 1e-3

    def fn(x):
        for _ in range(2 + x):  # consume clock ticks: latency grows with batch
            clock()

    result = measure(fn, lambda b: b, [1, 2, 4], name="fake", warmup=1, iters=5, clock=clock)
    stats = result.stats()
    assert set(stats) == {1, 2, 4}
    assert stats[4]["p50"] > stats[1]["p50"]
    path = tmp_path / "fake.json"
    result.save(path)
    table = load_profile_json(path)
    lin = load_profile_json(path, kind="linear")
    assert table.predict(2) == pytest.approx(stats[2]["p50"])
    assert lin.predict(4) == pytest.approx(result.linear().predict(4))


def test_load_profile_json_ms_units(tmp_path):
    path = tmp_path / "p.json"
    path.write_text('{"unit": "ms", "points": {"1": 2.0, "4": 5.0}}')
    assert load_profile_json(path).predict(4) == pytest.approx(0.005)
