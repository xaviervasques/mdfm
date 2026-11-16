"""Tests of scripts/g3_wrist_bars.py (task WRS) on synthetic signals and rows: no target, record or checkpoint is read
except the prediction file's hash and the fold lists."""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import pytest

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(TREE, "scripts", "g3_wrist_bars.py")


@pytest.fixture(scope="module")
def W():
    for p in (os.path.join(TREE, "scripts"), TREE):
        if p not in sys.path:
            sys.path.insert(0, p)
    spec = importlib.util.spec_from_file_location("g3_wrist_bars_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def swing(T, f_stride, rate=100.0, amp=120.0, noise=5.0, seed=0, axis=(0.2, 0.9, 0.3)):
    """(T, 6) wrist: acc near 1 g with a step-frequency ripple, gyro an arm swing at f_stride about `axis`."""
    rng = np.random.default_rng(seed)
    t = np.arange(T) / rate
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    w = amp * np.sin(2 * np.pi * f_stride * t)
    gyro = w[:, None] * a[None] + noise * rng.standard_normal((T, 3))
    acc = np.zeros((T, 3))
    acc[:, 2] = 1.0 + 0.2 * np.sin(2 * np.pi * 2 * f_stride * t)
    acc += 0.02 * rng.standard_normal((T, 3))
    return np.column_stack([acc, gyro])


def test_prediction_hash_holds(W):
    assert W.check_prediction() == W.PREDICTION_SHA256


def test_prediction_hash_refuses_an_edit(W, tmp_path):
    p = tmp_path / "prediction.txt"
    p.write_text("x\n", encoding="utf-8")
    s = tmp_path / "prediction.sha256"
    s.write_text(W.sha256_file(str(p)) + "  prediction.txt\n", encoding="utf-8")
    with pytest.raises(AssertionError):
        W.check_prediction(str(p), str(s), W.PREDICTION_SHA256)
    p.write_text("y\n", encoding="utf-8")
    with pytest.raises(AssertionError):
        W.check_prediction(str(p), str(s), W.sha256_file(str(s)))


@pytest.mark.parametrize("f", [0.75, 0.9, 1.1, 1.3])
def test_cadence_recovers_the_swing_frequency(W, f):
    x = swing(800, f, seed=int(f * 100))
    t = W.pass_terms({"wrist_L": x, "wrist_R": swing(800, f, seed=7, axis=(1.0, 0.1, 0.0))}, 2.0, 6.0)
    assert t["n_wrists"] == 2
    assert abs(t["f_stride"] - f) / f < 0.01
    assert abs(t["f_step"] - 2 * t["f_stride"]) < 1e-12
    assert np.isfinite(t["A"]) and np.isfinite(t["G"]) and t["G"] > 50


def test_cadence_skips_a_wrist_with_too_few_valid_samples(W):
    x = swing(800, 0.9)
    bad = x.copy()
    bad[150:700] = np.nan
    t = W.pass_terms({"wrist_L": bad}, 2.0, 6.0)
    assert t["n_wrists"] == 0 and np.isnan(t["f_stride"]) and np.isnan(t["A"])
    t2 = W.pass_terms({"wrist_L": bad, "wrist_R": x}, 2.0, 6.0)
    assert t2["n_wrists"] == 1 and abs(t2["f_stride"] - 0.9) < 0.01


def test_sagittal_and_autocorr(W):
    assert W.sagittal(np.full((2, 3), 1.0)) is None
    g = swing(500, 1.0)[:, 3:]
    s = W.sagittal(g)
    assert s.shape == (500,) and abs(np.nanmean(s)) < 1e-9
    r = W.autocorr(s, 150)
    assert r[0] == pytest.approx(1.0)
    lag, val = W.peak_lag(r, 66, 150)
    assert abs(lag - 100) < 1.0 and val > 0.8
    r2 = W.autocorr(s[:150], 120)                       # few pairs at long lags: NaN there
    assert np.isnan(r2[100]) and np.isfinite(r2[40])
    assert np.isnan(W.peak_lag(np.full(10, np.nan), 2, 8)[0])


def test_walkway_cadence(W):
    assert W.walkway_cadence([1.0, 1.5, 2.0, 2.5, 9.0], 0.9, 2.6) == pytest.approx(2.0)
    assert np.isnan(W.walkway_cadence([1.0, 1.5], 0.0, 3.0))


def test_ols_and_impute(W):
    rng = np.random.default_rng(1)
    X = rng.standard_normal((50, 2))
    y = 0.5 + 2.0 * X[:, 0] - 1.0 * X[:, 1]
    X[3, 0] = np.nan
    c = W.ols_fit(X, y)
    assert np.allclose(c, [0.5, 2.0, -1.0])
    assert np.allclose(W.ols_predict(c, np.array([[1.0, 1.0]])), [1.5])
    a, b = W.impute(np.array([[1.0, np.nan], [3.0, 4.0], [np.inf, 6.0]]), np.array([[np.nan, np.nan]]))
    assert np.allclose(a, [[1.0, 5.0], [3.0, 4.0], [2.0, 6.0]]) and np.allclose(b, [[2.0, 5.0]])


def test_classical_a1(W):
    terms = pd.DataFrame({"f_stride": [0.9, np.nan], "f_step": [1.8, np.nan], "A": [1.0, 1.0], "G": [1.0, 1.0]})
    p = W.classical_a1(terms, 1.3)
    assert p["stride_m"][0] == 1.3 and np.isnan(p["stride_m"][1])
    assert p["speed_m_s"][0] == pytest.approx(1.17) and np.isnan(p["speed_m_s"][1])


def frame(values, line="stride_mae_cm"):
    """ROW_COLUMNS rows: one entering row per (person, value)."""
    from mdfm.baselines import comparison as CMP
    rows = []
    for p, vs in values.items():
        for v in vs:
            rows.append({"line": line, "variant": "x", "chosen": True, "person_id": p, "trial_id": "t", "task": "k",
                         "pass_id": "0", "foot": "L", "t0": 0.0, "t1": 1.0, "t_s": np.nan, "target": 1.0,
                         "prediction": 1.0, "matched": True, "enters": True, "value": v})
    return pd.DataFrame(rows, columns=CMP.ROW_COLUMNS)


PERSONS = [f"p{i}" for i in range(10)]


def test_premium_test_premium_loss_neither(W):
    rng = np.random.default_rng(3)
    base = {p: list(10.0 + rng.standard_normal(3)) for p in PERSONS}
    good = {p: [v - 4.0 for v in vs] for p, vs in base.items()}
    good2 = {p: [v - 4.2 for v in vs] for p, vs in base.items()}
    t = W.premium_test([frame(good), frame(good2)], [frame(base)], PERSONS, "stride_mae_cm")
    assert t["separation"] == "body" and t["premium"] and not t["loss"]
    assert t["margin"] == pytest.approx(max(0.1 * t["comparator_mean"], 0.4), rel=1e-9)
    t = W.premium_test([frame(base)], [frame(good), frame(good2)], PERSONS, "stride_mae_cm")
    assert t["loss"] and not t["premium"]
    small = {p: [v - 0.3 for v in vs] for p, vs in base.items()}    # below 0 but inside the 10 percent margin
    t = W.premium_test([frame(small)], [frame(base)], PERSONS, "stride_mae_cm")
    assert t["separation"] == "body" and t["interval"][1] < 0 and not t["premium"] and not t["loss"]


def test_comparisons_need_both_sets(W):
    rng = np.random.default_rng(4)
    base = {p: list(10.0 + rng.standard_normal(2)) for p in PERSONS}
    good = {p: [v - 5.0 for v in vs] for p, vs in base.items()}
    # the body is better on persons p0..p4 only: premium on the set of those five, not on all ten
    mixed = {p: ([v - 5.0 for v in vs] if int(p[1:]) < 5 else [v + 5.0 for v in vs]) for p, vs in base.items()}
    both = lambda d: pd.concat([frame(d, "stride_mae_cm"), frame(d, "speed_mae_m_s")], ignore_index=True)  # noqa: E731
    bars = {n: [both(base)] for g in W.GROUPS.values() for n in g}
    sets = {"dev12": PERSONS[:5], "dev14": PERSONS}
    comp, rd = W.comparisons({"d_raw": [both(good), both(good)], "d_gain": [both(mixed), both(mixed)]}, bars, sets)
    assert rd["d_raw"]["stride_mae_cm"] == "PREMIUM"
    assert comp["d_gain"]["stride_mae_cm"]["dev12"]["best"]["premium"]
    assert not comp["d_gain"]["stride_mae_cm"]["dev14"]["best"]["premium"]
    assert rd["d_gain"]["stride_mae_cm"] == "NO PREMIUM"
    assert set(comp["d_raw"]["speed_mae_m_s"]["dev14"]) == {"best", "classical", "supervised", "token"}


def test_best_bar_picks_the_lowest_seed_mean(W):
    bars = {"a": [frame({p: [5.0] for p in PERSONS})], "b": [frame({p: [4.0] for p in PERSONS}),
                                                         frame({p: [7.0] for p in PERSONS})],
            "c": [frame({p: [5.4] for p in PERSONS})]}
    assert W.best_bar(bars, ("a", "b", "c"), PERSONS, "stride_mae_cm") == "a"
    assert W.best_bar(bars, ("b", "c"), PERSONS, "stride_mae_cm") == "c"


def test_wrist_input_cuts_to_the_wrists(W):
    T = 50
    loc = np.array([0, 3, 4, 7], dtype=np.int64)
    rec = {"signal": np.ones((T, 4, 9), np.float32), "channel_mask": np.ones((4, 9), bool), "location_ids": loc,
           "sample_valid": np.ones((T, 4), bool), "source": "weargait_pd", "fps": 100.0}
    r = W.wrist_input(rec)
    assert list(np.asarray(r["location_ids"])) == [3, 4]
    assert r["signal"].shape == (T, 2, 9) and not r["channel_mask"][:, 6:].any()
    assert rec["signal"].shape == (T, 4, 9)                      # the record itself unchanged
    rec2 = dict(rec, location_ids=np.array([0, 1, 2, 7]))
    assert W.wrist_input(rec2) is None
    arr = W.wrist_arrays(r)
    assert set(arr) == {"wrist_L", "wrist_R"} and arr["wrist_L"].shape == (T, 6)


def test_sets_of(W):
    train, dev, dev12 = W.sets_of()
    assert len(train) == 54 and len(dev) == 14 and len(dev12) == 12
    assert not set(train) & set(dev) and set(dev) - set(dev12) == {"weargait:WHC003", "weargait:WHC026"}
