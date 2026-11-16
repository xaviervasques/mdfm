"""Tests of scripts/f2_H2sensor.py on synthetic inputs (no store, no target, no render is read)."""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import f2_H2sensor as H  # noqa: E402
from mdfm.data.canonical import Refused  # noqa: E402


def _rot(seed):
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    return q * np.sign(np.linalg.det(q))


def _walk(n=600, amp=300.0, seed=0):
    t = np.arange(n) / 100.0
    rng = np.random.default_rng(seed)
    gyro = np.stack([amp * np.sin(2 * np.pi * 0.9 * t), 20 * np.sin(2 * np.pi * 1.8 * t), rng.normal(0, 2, n)], 1)
    acc = np.stack([0.5 * np.sin(2 * np.pi * 1.8 * t), 1.0 + 0.3 * np.cos(2 * np.pi * 0.9 * t), rng.normal(0, .02, n)], 1)
    return acc, gyro


def test_features_rotation_invariant_and_scale():
    acc, gyro = _walk()
    R = _rot(3)
    a = H.unit_features(acc, gyro)
    b = H.unit_features(acc @ R.T, gyro @ R.T)
    for k in H.FEATURES:
        assert a[k] == pytest.approx(b[k], rel=1e-9, abs=1e-12)
    c = H.unit_features(acc, 0.85 * gyro)
    assert c["G_pc"] / a["G_pc"] == pytest.approx(0.85, rel=1e-9)
    assert c["G_p95"] / a["G_p95"] == pytest.approx(0.85, rel=1e-9)
    assert a["G_pc"] == pytest.approx(300.0 * np.sin(np.pi / 2 * 0.95), rel=0.03)
    assert H.unit_features(acc[:50], gyro[:50]) is None


def test_lowpass_runs_masks_and_keeps_low_band():
    t = np.arange(1000) / 100.0
    x = np.stack([np.sin(2 * np.pi * 1.0 * t) + 0.5 * np.sin(2 * np.pi * 30.0 * t)] * 6, 1)
    valid = np.ones(1000, bool)
    valid[500:503] = False
    valid[990:] = False
    y = H.lowpass_runs(x, valid)
    assert np.isnan(y[500:503]).all() and np.isnan(y[990:]).all()
    mid = slice(100, 400)
    assert np.abs(y[mid, 0] - np.sin(2 * np.pi * t[mid])).max() < 0.02
    z = H.lowpass_runs(x, valid, cutoff=None)
    assert np.array_equal(z[valid], x[valid]) and np.isnan(z[~valid]).all()


def test_fit_power_and_ratios():
    rng = np.random.default_rng(1)
    v = rng.uniform(0.5, 1.8, 300)
    cad = 100 * v ** 0.4 * np.exp(rng.normal(0, 0.03, 300))
    f = 200.0 * v ** 0.8 * np.exp(rng.normal(0, 0.01, 300))
    fit = H.fit_power(v, f)
    assert fit["b"] == pytest.approx(0.8, abs=0.02) and math.exp(fit["a"]) == pytest.approx(200.0, rel=0.02)
    q, s = H.ratios(fit, 1.2, 200.0 * (0.85 * 1.2) ** 0.8)
    assert s == pytest.approx(0.85, abs=0.01) and q == pytest.approx(0.85 ** 0.8, abs=0.01)
    fit2 = H.fit_power(v, f, cad)
    assert fit2["c"] is not None and fit2["n"] == 300
    flat = H.fit_power(v, np.full(300, 5.0) * np.exp(rng.normal(0, .01, 300)))
    assert math.isnan(H.ratios(flat, 1.0, 5.0)[1])


def test_boot_and_classify_and_reading():
    st = H.boot_median([0.85] * 20 + [0.86] * 20, n_boot=500)
    assert H.classify(st) == "reads smaller"
    assert H.classify(H.boot_median([1.0, 1.02, 0.99] * 10, n_boot=500)) == "reads same or larger"
    assert H.classify(H.boot_median([0.94] * 10, n_boot=500)) == "in between"
    none = {"gravity_sites": [], "gyro_kind_sites": [], "acc_kind_sites": [], "found": False}
    assert H.reading("reads same or larger", "reads same or larger", none)[0] == "NOT SUPPORTED"
    assert H.reading("reads smaller", "reads smaller", none)[0] == "PARTLY"
    assert H.reading("reads smaller", "reads smaller", {**none, "found": True})[0] == "SUPPORTED"
    assert H.reading("reads smaller", "in between", none)[0] == "PARTLY"


def test_sensor_cause():
    grav = {s: 1.0 for s in H.LEG}
    q = {s: {"G_pc": 0.85, "G_p95": 0.86, "A_dyn": 1.0, "A_p95": 1.0, "A_med": 1.0} for s in H.LEG}
    c = H.sensor_cause(grav, q)
    assert c["found"] and len(c["gyro_kind_sites"]) == 8 and not c["acc_kind_sites"]
    c2 = H.sensor_cause({**grav, "shank_L": 0.9}, {})
    assert c2["found"] and c2["gravity_sites"] == ["shank_L"]
    assert not H.sensor_cause(grav, {s: {k: 1.0 for k in H.FEATURES} for s in H.LEG})["found"]


def test_guards():
    locked = {"weargait:L1"}
    assert H.keep_row("weargait:A", {"weargait:A"}, locked, "False")
    assert not H.keep_row("weargait:B", {"weargait:A"}, locked)
    with pytest.raises(Refused):
        H.keep_row("weargait:L1", {"weargait:L1"}, locked)
    with pytest.raises(Refused):
        H.keep_row("weargait:A", {"weargait:A"}, locked, "True")
    dev = [f"weargait:D{i}" for i in range(14)]
    train = [f"weargait:T{i}" for i in range(54)]
    sets = {"dev14_sha256": H.person_hash(dev), "train54_sha256": H.person_hash(train)}
    assert H.check_sets(dev, train, locked, sets)["dev14"] == sorted(dev)
    with pytest.raises(AssertionError):
        H.check_sets(dev, train[:-1] + ["weargait:X"], locked, sets)
    with pytest.raises(Refused):
        H.check_sets(dev, train[:-1] + ["weargait:L1"], locked, sets)


def test_runs():
    assert H.runs(np.array([0, 1, 1, 0, 1], bool)) == [(1, 3), (4, 5)]


def test_within_exponent_removes_person_level():
    units = []
    for p, lvl, sc in (("a", 1.0, 1.0), ("b", 3.0, 1.5)):
        for v in (0.8, 1.0, 1.3):
            units.append({"person": p, "speed": v * sc,
                          "feat": {(H.CUTOFF_HZ, "shank_L"): {"G_pc": lvl * (v * sc) ** 0.6}}})
    assert H.within_exponent(units, {"a", "b"}, "shank_L", "G_pc") == pytest.approx(0.6, abs=1e-9)
