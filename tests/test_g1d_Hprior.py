"""Tests of the pure functions of scripts/g1d_Hprior.py (no data, no model)."""
import importlib.util
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("g1d_Hprior", os.path.join(HERE, "..", "scripts", "g1d_Hprior.py"))
H = importlib.util.module_from_spec(spec)
spec.loader.exec_module(H)


def _pose(sh_deg, el_deg):
    """22 joints: pelvis at 0, neck 0.5 m above, left shoulder at x 0.2, the upper arm lowered from the downward
    vertical by sh_deg toward +x, the forearm bent by el_deg further in the same plane."""
    J = np.zeros((22, 3))
    J[H.NECK] = [0, 0.5, 0]
    for side, sx in (("L", 0.2), ("R", -0.2)):
        s, e, w = H.ARM[side]
        J[s] = [sx, 0.45, 0]
        a = math.radians(sh_deg)
        u = np.array([math.sin(a) * np.sign(sx), -math.cos(a), 0.0])
        J[e] = J[s] + 0.3 * u
        b = math.radians(sh_deg + el_deg)
        f = np.array([math.sin(b) * np.sign(sx), -math.cos(b), 0.0])
        J[w] = J[e] + 0.25 * f
    return J


def test_arm_angles_known_pose():
    a = H.arm_angles(_pose(30.0, 45.0))
    assert np.allclose(a, [30.0, 30.0, 45.0, 45.0], atol=1e-6)
    b = H.arm_angles(np.stack([_pose(0.0, 0.0), _pose(90.0, 120.0)]))
    assert b.shape == (2, 4)
    assert np.allclose(b[1], [90.0, 90.0, 120.0, 120.0], atol=1e-6)


def test_arm_angles_rotation_invariant():
    J = _pose(40.0, 70.0)
    th = 0.7
    R = np.array([[math.cos(th), 0, math.sin(th)], [0, 1, 0], [-math.sin(th), 0, math.cos(th)]])
    assert np.allclose(H.arm_angles(J @ R.T + 3.0), H.arm_angles(J), atol=1e-6)


def test_pull_stats_unbiased_and_shrunk():
    rng = np.random.default_rng(0)
    t = rng.normal(40, 15, 5000)
    s = H.pull_stats(t, t, 20.0)
    assert abs(s["b"] - 1) < 1e-9 and s["share"] == 0.0 and s["rmse"] == 0.0
    m = 20.0
    d = t + 0.5 * (m - t)                       # half way to the reference
    s = H.pull_stats(d, t, m)
    assert abs(s["b"] - 0.5) < 1e-9 and abs(s["lam"] - 0.5) < 1e-9 and abs(s["share"] - 1.0) < 1e-9
    assert abs(s["sd_ratio"] - 0.5) < 1e-9
    d = t + rng.normal(0, 15, t.size)           # noise only: b near 1, share near 0
    s = H.pull_stats(d, t, m)
    assert abs(s["b"] - 1) < 0.05 and s["share"] < 0.05


def test_within_centres_each_window():
    x = np.array([1.0, 3.0, 10.0, 20.0, 30.0])
    w = np.array([0, 0, 1, 1, 1])
    assert np.allclose(H.within(x, w), [-1, 1, -10, 0, 10])


def test_draw_calibration_calibrated_and_narrow():
    rng = np.random.default_rng(1)
    n = 20000
    mu = rng.normal(0, 10, n)
    t = mu + rng.normal(0, 5, n)                # truth exchangeable with the draws
    D = mu[None] + rng.normal(0, 5, (16, n))
    c = H.draw_calibration(D, t, mu)
    assert abs(c["range_coverage"] - H.COVERAGE_CALIBRATED) < 0.01
    assert abs(c["sd_over_rmse"] - 1) < 0.05
    D = mu[None] + rng.normal(0, 0.5, (16, n))   # ten times too narrow
    c = H.draw_calibration(D, t, mu)
    assert c["range_coverage"] < 0.2 and c["median_abs_z"] > 3


def test_readings_and_verdict():
    assert H.reading_c1(0.3) == "UNDER_DISPERSED" and H.reading_c1(0.8) == "NOT_UNDER_DISPERSED"
    assert H.reading_c1(0.6) == "BETWEEN"
    assert H.reading_c2(0.7, 0.6) == "MAIN" and H.reading_c2(0.95, 0.6) == "NOT_MAIN"
    assert H.reading_c2(0.7, 0.1) == "NOT_MAIN" and H.reading_c2(0.85, 0.3) == "BETWEEN"
    assert H.reading_c3(0.9, 0.85, 0.75) == "PULLS" and H.reading_c3(0.9, 0.85, 0.86) == "FLAT"
    assert H.reading_c4(0.7, 0.7) == "PRIOR_NARROW" and H.reading_c4(1.0, 0.95) == "PRIOR_NOT_NARROW"
    assert H.verdict("UNDER_DISPERSED", "MAIN", "FLAT", "PRIOR_NARROW") == "supported"
    assert H.verdict("UNDER_DISPERSED", "NOT_MAIN", "PULLS", "PRIOR_NARROW") == "not_supported"
    assert H.verdict("UNDER_DISPERSED", "BETWEEN", "PULLS", "PRIOR_NARROW") == "partly"


def test_blocks_keep_takes_whole():
    import pandas as pd

    class G:
        windows = pd.DataFrame({"take_id": ["a", "a", "b", "b", "b", "c", "d"]})
    out = H.blocks(G(), [0, 1, 2, 3, 4, 5, 6], size=4)
    assert out == [[0, 1], [2, 3, 4, 5], [6]]
    assert sorted(i for b in out for i in b) == list(range(7))
