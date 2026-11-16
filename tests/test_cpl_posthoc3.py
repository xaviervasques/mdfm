"""Tests of scripts/cpl_posthoc3.py on hand-made tables (no store, no record, no person)."""
import importlib.util
import math
import os

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(os.path.dirname(HERE), "scripts", "cpl_posthoc3.py")


def _mod():
    spec = importlib.util.spec_from_file_location("_t_cpl_posthoc3", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


M = _mod()
C = M.C
B = {"thigh": 0.5, "shank": 0.45, "ankle": 0.45, "foot": 0.6}
A = {"thigh": 3.6, "shank": 4.1, "ankle": 4.1, "foot": 4.3}


def _pop(rng, n_persons, prefix, gamma=1.0, n=20, noise=0.03, tread=0):
    """Strides whose segment ROMs follow log ROM = a + b log(gamma S / l) + 0.1 (log c - log 110) + noise."""
    rows = []
    for k in range(n_persons):
        stature = rng.uniform(1.5, 1.9)
        l = C.LEG_RATIO * stature
        for j in range(n):
            T = rng.uniform(0.9, 1.3)
            S = l * math.exp(rng.normal(0.35, 0.12))
            c = 120.0 / T
            row = {"person_id": f"{prefix}{k}", "t0": 0.0, "t1": T, "S": S, "stature_m": stature,
                   "treadmill": tread, "archive": "X"}
            for s in C.SEGMENTS:
                row[f"rom_{s}"] = math.exp(A[s] + B[s] * math.log(gamma * S / l) + 0.1 * (math.log(c) - C.C_CENTRE)
                                           + rng.normal(0, noise))
            rows.append(row)
    return C.complete(C.finish(pd.DataFrame(rows)))


def test_ref_row_recovers_gamma_and_h():
    rng = np.random.default_rng(3)
    am = _pop(rng, 30, "a")
    real = _pop(rng, 20, "r", gamma=0.9)
    Q = 0.88
    row = M.ref_row(C.Relations(am), real, C.PRIMARY, Q, 100, 1)
    assert row["gamma"] == pytest.approx(0.9, abs=0.01)
    assert row["h"] == pytest.approx(math.log(row["gamma"]) / math.log(Q), rel=1e-12)
    assert row["interval"][0] <= row["gamma"] <= row["interval"][1]
    assert row["h_interval"][0] <= row["h"] <= row["h_interval"][1]
    for s in C.SEGMENTS:
        assert row["single"][s]["gamma"] == pytest.approx(0.9, abs=0.02)


def test_leg_factor_scales_gamma_exactly():
    rng = np.random.default_rng(4)
    am = _pop(rng, 20, "a")
    real = _pop(rng, 10, "r", gamma=0.9)
    fit = C.Relations(am).fit()
    g0 = C.statistics(real, fit)["gamma_L"]
    for k in (0.97, 1.03, 0.51 / C.LEG_RATIO):
        g = C.statistics(M.with_leg(real, k), fit)["gamma_L"]
        assert g == pytest.approx(g0 * k, rel=1e-10)


def test_wrelations_with_equal_person_weights_equal_relations():
    rng = np.random.default_rng(5)
    am = _pop(rng, 12, "a")
    n = am.groupby("person_id")["S"].transform("size").to_numpy(np.float64)
    a, b = C.Relations(am).fit(), M.WRelations(am, 1.0 / n).fit()
    for k in a:
        np.testing.assert_allclose(a[k][0], b[k][0], rtol=1e-10, atol=1e-12)
        assert a[k][1] == pytest.approx(b[k][1], rel=1e-10)


def test_stream_weights():
    am = pd.DataFrame({"person_id": ["p", "p", "q"], "T": [1.0, 3.0, 2.0]})
    w = M.stream_weights(am, {"p": 100.0, "q": 50.0})
    # raw: p 0.01 + 0.03 = 0.04, q 0.04; mean 0.04
    np.testing.assert_allclose(w, [0.25, 0.75, 1.0])
    assert (w[0] + w[1] + w[2]) / 2 == pytest.approx(1.0)


def test_stream_shares():
    tk = pd.DataFrame({"person_id": ["p", "p", "q", "r"], "n_knots": [300, 100, 200, 50],
                       "treadmill": [1, 0, 0, 0], "task_name": ["walk", "jump", "walk", "walk"]})
    s = M.stream_shares(tk, ["p", "q", "r"], [0, 2])
    # take weights: p walk 0.75 / 3, p jump 0.25 / 3, q 1 / 3, r 1 / 3
    assert s["share_straight_forward"] == pytest.approx(1.75 / 3)
    assert s["treadmill_share_of_straight_forward"] == pytest.approx(0.75 / 1.75)
    assert s["share_task_walk"] == pytest.approx(2.75 / 3)
    assert s["treadmill_share_of_task_walk"] == pytest.approx(0.75 / 2.75)


def test_table_treadmill_shares():
    am = pd.DataFrame({"person_id": ["p", "p", "q"], "treadmill": [1, 0, 0]})
    s = M.table_treadmill_shares(am, np.array([1.0, 1.0, 2.0]))
    assert s["persons_equal"] == pytest.approx(0.25)
    assert s["stream"] == pytest.approx(0.25)
    assert s["persons_with_treadmill_strides"] == 1 and s["persons_with_overground_strides"] == 2


def test_contributions_add_up_for_one_stride():
    rng = np.random.default_rng(6)
    am = _pop(rng, 15, "a")
    real = _pop(rng, 1, "r", gamma=0.85, n=1)
    c = M.contributions(real, C.Relations(am).fit(), 0.88)
    assert c["sum_of_terms"] == pytest.approx(c["log_gamma_L"], abs=1e-12)
    assert sum(v["weight"] for v in c["sites"].values()) == pytest.approx(1.0)


def test_readings():
    row = {"h": 0.96, "h_interval": [0.69, 1.12]}
    assert M.size_reading(row, prefix="STREAM ") == "STREAM SIZE"
    assert M.size_reading({"h": 0.4, "h_interval": [0.1, 0.7]}) == "PART"
    assert M.size_reading({"h": 0.28, "h_interval": [-0.3, 0.8]}) == "NONE"
    assert M.band_reading(1.0) == "MATCH" and M.band_reading(1.10) == "ABOVE" and M.band_reading(0.9) == "BELOW"


def test_plan_hash():
    assert C.check_prediction(M.PLAN, M.PLAN_SHA_FILE, M.PLAN_SHA256) == M.PLAN_SHA256


def test_no_dashes_in_script():
    with open(SCRIPT, encoding="utf-8") as fh:
        text = fh.read()
    assert chr(0x2013) not in text and chr(0x2014) not in text
