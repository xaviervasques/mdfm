"""Tests of scripts/g1d_Hstand.py (G1D-Hstand): synthetic inputs only, no data, no model."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(TREE, "scripts"))
import g1d_Hstand as H  # noqa: E402


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def test_twist_swing_pure_and_mixed():
    tw, sw = H.twist_swing(H.rot_y_np(np.array(0.3)))
    assert tw == pytest.approx(0.3) and sw == pytest.approx(0.0, abs=1e-9)
    tw, sw = H.twist_swing(rot_x(0.2))
    assert tw == pytest.approx(0.0, abs=1e-9) and sw == pytest.approx(0.2)
    E = rot_x(0.25) @ H.rot_y_np(np.array(-0.4))          # swing after twist: E = S T
    tw, sw = H.twist_swing(E)
    assert tw == pytest.approx(-0.4) and sw == pytest.approx(0.25)


def test_quintile_equal_counts():
    q = H.quintile(np.arange(10.0)[::-1])
    assert sorted(np.bincount(q)[1:].tolist()) == [2, 2, 2, 2, 2]
    assert q[-1] == 1 and q[0] == 5


def test_window_energy_static_and_moving():
    sites = ("head", "lower_back", "wrist_L", "wrist_R", "shank_L", "shank_R")
    x = np.zeros((200, 6, 6))
    x[:, :, 1] = 1.0                       # gravity on y, still
    x[:, :, 3:] = 2.0                      # a constant gyro bias
    e = H.window_energy(x, sites)
    assert e["energy_dps"] == pytest.approx(0.0, abs=1e-12) and e["still_share"] == 1.0
    x[:, 2, 3] += 100.0 * np.sin(np.linspace(0, 20, 200))
    e2 = H.window_energy(x, sites)
    assert e2["arm_dps"] > 30 and e2["leg_dps"] == pytest.approx(0.0, abs=1e-12) and e2["still_share"] < 1.0


def test_counterfactual_removal_and_persons():
    t = pd.DataFrame({"person_id": ["a", "a", "b", "b"], "n": [100, 100, 100, 300], "g1a_m": [0.1, 0.2, 0.05, 0.05]})
    full = (0.15 + 0.05) / 2
    c = H.counterfactual(t, np.array([1.0, 0.0, 1.0, 1.0]), n_boot=50)
    assert c["full_mm"] == pytest.approx(1000 * full)
    assert c["reweighted_mm"] == pytest.approx(1000 * (0.1 + 0.05) / 2)
    assert c["change_mm"] == pytest.approx(c["reweighted_mm"] - c["full_mm"])
    c = H.counterfactual(t, np.array([0.0, 0.0, 1.0, 1.0]), n_boot=50)
    assert c["persons_kept"] == 1 and c["reweighted_mm"] == pytest.approx(50.0)


def test_group_table_shares_sum_to_one():
    t = pd.DataFrame({"person_id": ["a", "b", "c"], "n": [100, 200, 300], "g1a_m": [0.1, 0.05, 0.02],
                      "q": [1, 2, 2]})
    rows = H.group_table(t, "q")
    assert sum(r["error_share"] for r in rows) == pytest.approx(1.0)
    assert sum(r["knot_share"] for r in rows) == pytest.approx(1.0)
    assert rows[0]["error_share"] == pytest.approx(10 / (10 + 10 + 6))


def test_axis_window_pure_vertical_twist_of_one_segment():
    from mdfm.body.smplh import PARENTS
    groups = {"arms": (18,), "root": (0,), "limbs": (18,)}
    n = 5
    rng = np.random.default_rng(0)
    J = rng.normal(size=(n, 22, 3))
    Rt = np.broadcast_to(np.eye(3), (n, 22, 3, 3)).copy()
    Rp = Rt.copy()
    Rp[:, 18] = H.rot_y_np(np.array(0.5)) @ Rt[:, 18]
    out = H.axis_window(J.copy(), Rp, J, Rt, groups, PARENTS)
    assert out["g1a_m"] == pytest.approx(0.0, abs=1e-12) and out["theta"] == pytest.approx(0.0, abs=1e-12)
    assert out["tw2_arms"] == pytest.approx(n * 0.25) and out["sw2_arms"] == pytest.approx(0.0, abs=1e-12)
    assert out["tw2_root"] == pytest.approx(0.0, abs=1e-12)
