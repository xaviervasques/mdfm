"""Tests of scripts/f2_H4root.py (task F2-H4root): the per-stride decomposition on synthetic bodies, the test A
share, test B ratios, the person guards and the queued job. Task F2D: test A on the pass speeds, the root ceiling,
the dev12 lines under a uniform factor and the rerun guard on the first run's constants. No real record is read."""
import importlib.util
import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.dirname(HERE)
if TREE not in sys.path:
    sys.path.insert(0, TREE)


def _load():
    key = "_test_f2_H4root"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, os.path.join(TREE, "scripts", "f2_H4root.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


M = _load()
RATE = 100.0
V = 1.2           # m/s
PERIOD = 1.0      # s per stride
STANCE = 0.6      # share of the cycle in stance


def _foot_z(t, offset):
    """World z of a foot whose stance starts at offset + k: fixed in stance, linear in swing (no slip)."""
    ph = np.mod(t - offset, PERIOD)
    k = np.floor((t - offset) / PERIOD)
    z_st = V * (offset + k * PERIOD + 0.5 * STANCE * PERIOD)
    sw = ph >= STANCE * PERIOD
    frac = (ph - STANCE * PERIOD) / ((1 - STANCE) * PERIOD)
    return np.where(sw, z_st + frac * V * PERIOD, z_st), (~sw).astype(float)


def _body(root_scale=1.0, T=600, windows=((0, 400), (200, 600))):
    t = np.arange(T) / RATE
    pel = np.zeros((T, 3))
    pel[:, 1] = 0.9
    pel[:, 2] = V * t
    J = np.repeat(pel[:, None, :], 22, axis=1).copy()
    contact = np.zeros((T, 2))
    for j, (side, off) in enumerate((("L", 0.0), ("R", 0.5))):
        z, c = _foot_z(t, off)
        x = -0.1 if side == "L" else 0.1
        ankle, hip, knee, footj = (7, 1, 4, 10) if side == "L" else (8, 2, 5, 11)
        J[:, ankle] = np.stack([np.full(T, x), np.full(T, 0.08), z], axis=1)
        J[:, footj] = J[:, ankle] + np.array([0.0, -0.05, 0.12])
        J[:, hip] = pel + np.array([x, 0.0, 0.0])
        J[:, knee] = 0.5 * (J[:, hip] + J[:, ankle])
        contact[:, j] = c
    rel = J - J[:, :1]
    Jw = rel + (pel * np.array([1.0, 1.0, root_scale]))[:, None, :]
    W = np.full((len(windows), T, 22, 3), np.nan)
    for w, (s, e) in enumerate(windows):
        W[w, s:e] = Jw[s:e]
    wof = np.zeros(T, np.int64)
    wof[300:] = 1
    return W, contact, wof


def test_decomposition_no_slip():
    W, c, wof = _body()
    d = M.stride_decomposition(W, c, wof, RATE)
    assert len(d) >= 6
    assert set(d["foot"]) == {"L", "R"}
    assert np.allclose(d["ankle_chord_m"], V * PERIOD, atol=1e-6)
    assert np.allclose(d["pelvis_chord_m"], V * PERIOD, atol=1e-6)
    assert np.allclose(d["rel_ankle_along_m"], 0.0, atol=1e-6)
    assert np.allclose(d["legs_m"], V * PERIOD, rtol=0.05)
    assert np.allclose(d["root_along_m_s"], V, atol=1e-6)
    # the abrupt contact edges of the synthetic gait put a few stance frames on a moving foot: small, positive
    assert (np.abs(d["slip_along_m_s"]) < 0.07).all()
    assert np.allclose(d["legs_along_m_s"] + d["slip_along_m_s"], d["root_along_m_s"], atol=1e-6)
    assert (d["leg_length_m"] > 0.5).all()


def test_decomposition_root_lags_legs():
    W, c, wof = _body(root_scale=0.8)
    d = M.stride_decomposition(W, c, wof, RATE)
    d0 = M.stride_decomposition(*_body(), RATE)
    assert np.allclose(d["pelvis_chord_m"], 0.8 * V * PERIOD, atol=1e-6)
    assert np.allclose(d["ankle_chord_m"], 0.8 * V * PERIOD, atol=1e-6)
    assert np.allclose(d["legs_m"], V * PERIOD, rtol=0.05)
    assert np.allclose(d["slip_along_m_s"] - d0["slip_along_m_s"], -0.2 * V, atol=1e-6)
    assert np.allclose(d["legs_along_m_s"], d0["legs_along_m_s"], atol=1e-9)


def test_decomposition_matches_gait_strides():
    from mdfm.readouts import gait
    W, c, wof = _body(root_scale=0.9)
    d = M.stride_decomposition(W, c, wof, RATE)
    s = gait.strides(W, gait.initial_contacts(c, RATE), wof, RATE)
    m = d.merge(s, on=["foot", "t0", "t1"])
    assert len(m) == len(s) == len(d)
    assert np.allclose(m["ankle_chord_m"], m["length_m"], atol=1e-9)
    assert (m["window_x"] == m["window_y"]).all()


def test_f_share_and_reading():
    d_root, d_legs, f = M.f_share(0.85, 0.75, 0.97, 0.875)
    assert math.isclose(d_root, math.log(0.85 / 0.97))
    assert math.isclose(d_legs, math.log(0.75 / 0.875))
    assert math.isclose(f, 1 - d_legs / d_root)
    assert math.isnan(M.f_share(0.85, -1.0, 0.97, 0.875)[2])
    assert M.reading_a(0.8, 0.1, 1.2) == "SUPPORTED"
    assert M.reading_a(0.6, -0.1, 1.2) == "PARTLY"
    assert M.reading_a(0.1, -0.5, 0.4) == "NOT SUPPORTED"
    assert M.reading_a(float("nan"), None, None) == "NOT EVALUATED"
    assert M.weaker(["SUPPORTED", "PARTLY"]) == "PARTLY"
    assert M.weaker(["SUPPORTED", "NOT SUPPORTED"]) == "NOT SUPPORTED"


def _events():
    rows = []
    for p in range(4):
        pid, tid = f"weargait:P{p}", f"P{p}_SelfPace"
        for i in range(5):
            t0, t1 = 1.0 + i, 2.0 + i
            rows.append(("base", "stride", tid, pid, "SelfPace", "L", "0", t0, t1, 120.0, 102.0, -18.0, "cm", "True"))
            rows.append(("model_legs", "stride", tid, pid, "SelfPace", "L", "0", t0, t1, 120.0, 96.0, -24.0, "cm", "True"))
        for i in range(6):
            t = 1.0 + i
            rows.append(("base", "ic_walkway", tid, pid, "SelfPace", "L", "", t, np.nan, 1000 * t, 1000 * t * 1.0 + 20,
                         20.0, "ms", "True"))
        rows.append(("base", "speed", tid, pid, "SelfPace", "", "0", 1.0, 6.0, 1.2, 1.02, -0.18, "m/s", "True"))
        rows.append(("model_legs", "speed", tid, pid, "SelfPace", "", "0", 1.0, 6.0, 1.2, 0.96, -0.24, "m/s", "True"))
    cols = ["row", "kind", "trial_id", "person_id", "task", "foot", "pass_id", "t0_s", "t1_s", "reference",
            "prediction", "signed_error", "unit", "matched"]
    return pd.DataFrame(rows, columns=cols)


def _write(tmp_path, df):
    p = tmp_path / "events.csv"
    df.to_csv(p, index=False, encoding="utf-8")
    return str(p)


def test_read_rows_guards(tmp_path):
    from mdfm.data.canonical import Refused
    ev = _events()
    path = _write(tmp_path, ev)
    allowed = {f"weargait:P{p}" for p in range(4)}
    got = M.read_rows(path, allowed, [], ["weargait:P0", "weargait:P1"])
    assert set(got["person_id"]) == {"weargait:P0", "weargait:P1"}
    assert got["matched"].dtype == bool
    with pytest.raises(Refused):
        M.read_rows(path, allowed, ["weargait:P3"], ["weargait:P0"])
    with pytest.raises(AssertionError):
        M.read_rows(path, {"weargait:P0"}, [], ["weargait:P0"])


def test_test_a_and_b(tmp_path):
    ev = M.read_rows(_write(tmp_path, _events()), None, [], [f"weargait:P{p}" for p in range(4)])
    pairs = M.paired_strides(ev)
    assert len(pairs) == 20
    sim = {"root": 0.97, "legs": 0.875}
    a = M.test_a(pairs, sim, n_boot=50, seed=1)
    assert math.isclose(a["g_root_real"], 0.85)
    assert math.isclose(a["g_legs_real"], 0.8)
    assert math.isclose(a["f"], M.f_share(0.85, 0.8, 0.97, 0.875)[2])
    assert math.isclose(a["f_ci"][0], a["f"]) and math.isclose(a["f_ci"][1], a["f"])
    b = M.test_b(M.stride_ratios(ev, "base"), n_boot=50, seed=1)
    assert b["strides"] == 20
    assert math.isclose(b["median_d"], 1.0)
    assert math.isclose(b["median_k"], 1.0)
    assert b["stride_follows_root"] and b["cadence_right"] and b["speed_error_equals_stride_error"]
    bl = M.test_b(M.stride_ratios(ev, "model_legs"), n_boot=50, seed=1)
    assert math.isclose(bl["median_s"], 0.8)


def test_match_decomposition():
    dec = pd.DataFrame({"trial_id": ["T", "T"], "foot": ["L", "L"], "t0": [1.0, 2.02], "t1": [2.02, 3.0],
                        "ankle_chord_m": [1.1, 1.2], "slip_along_m_s": [0.0, -0.1]})
    wk = pd.DataFrame({"trial_id": ["T", "T", "T"], "person_id": ["p"] * 3, "task": ["SelfPace"] * 3,
                       "foot": ["L", "L", "R"], "pass_id": ["0", "0", "0"], "t0": [1.0, 2.0, 1.5],
                       "t1": [2.0, 3.0, 2.5], "length_m": [1.2, 1.25, 1.2]})
    ps = pd.DataFrame({"trial_id": ["T"], "pass_id": ["0"], "speed_m_s": [1.3]})
    m = M.match_decomposition(dec, wk, ps, tol=0.15)
    assert list(m["matched"]) == [True, True, False]
    assert list(m["ankle_chord_m"][:2]) == [1.1, 1.2]
    assert (m["walk_speed_m_s"] == 1.3).all()


def test_job_command():
    c = M.job_command()
    assert c[0] == "scripts/queue_job.py"
    assert c[c.index("--cls") + 1] == "1"
    assert 70 <= int(c[c.index("--priority") + 1]) <= 79
    assert c[c.index("--requires") + 1] == "cache/runs/phaseA_root_full/ckpt_B_032044.pt"
    assert c[c.index("--group") + 1] == "F2"
    tail = c[c.index("--") + 1:]
    assert tail[tail.index("--memory-gb") + 1] == "12"
    assert tail[tail.index("--out") + 1].startswith("reports/bench/f2_scale/H4root/")


def test_speed_divisor_and_root_ceiling(tmp_path):
    """Task F2D: test A on the pass speeds (the second divisor of F2-CHAL A, not pre-registered) and the root gain
    if it lost on real input only what the legs lose, on a dump folder with synthetic events and decomposition."""
    ev = M.read_rows(_write(tmp_path, _events()), None, [], [f"weargait:P{p}" for p in range(4)])
    q = M.test_a_speed(ev, {"root": 0.97, "legs": 0.92}, n_boot=50, seed=1)
    assert math.isclose(q["g_root_speed_real"], 0.85) and math.isclose(q["g_legs_speed_real"], 0.8)
    assert math.isclose(q["f"], M.f_share(0.85, 0.8, 0.97, 0.92)[2])
    assert q["passes_root"] == 4 and q["persons"] == 4 and q["reading_if_registered"] in M.ORDER
    folder = tmp_path / "dump"
    folder.mkdir()
    _events().to_csv(folder / "events.csv", index=False, encoding="utf-8")
    dec = pd.DataFrame({"trial_id": ["P0_SelfPace"] * 2, "person_id": ["weargait:P0"] * 2, "matched": [True, True],
                        "slip_along_m_s": [0.02, 0.03], "root_along_m_s": [1.0, 1.0], "legs_along_m_s": [0.98, 0.97],
                        "rel_ankle_along_m": [0.0, 0.0], "leg_length_m": [0.77, 0.77], "ankle_chord_m": [1.0, 1.0],
                        "pelvis_chord_m": [1.0, 1.0], "legs_m": [0.98, 0.98], "walk_speed_m_s": [1.2, 1.2]})
    dec.to_csv(folder / "decomposition.csv", index=False, encoding="utf-8")
    sets = {"train54": [f"weargait:P{p}" for p in range(4)], "dev14": [], "dev12": []}
    sim = {"root": 0.97, "legs": 0.875}
    r = M.read_dump(str(folder), sets, [], sim, n_boot=30, seed=1, speed_sim={"root": 0.97, "legs": 0.92})
    c = r["root_specific_ceiling_train54"]
    assert math.isclose(c["stride"], 0.97 * math.exp(r["test_a_train54"]["D_legs"]))
    assert math.isclose(c["stride"], 0.97 * 0.8 / 0.875) and math.isclose(c["stride_now"], 0.85)
    assert math.isclose(c["speed"], 0.97 * 0.8 / 0.92) and math.isclose(c["speed_now"], 0.85)
    assert "test_a_train54_speed_divisor" in r and r["slip_reading"]["supports_H4"] is False
    r0 = M.read_dump(str(folder), sets, [], sim, n_boot=30, seed=1)      # the registered call: no speed block
    assert "test_a_train54_speed_divisor" not in r0 and "speed" not in r0["root_specific_ceiling_train54"]
    s = M.scaled_lines(M.paired_strides(ev), r, n_boot=20, seed=1)
    assert math.isclose(s["raw"]["S_cm"], 18.0) and math.isclose(s["raw"]["gain"], 0.85)
    assert math.isclose(s["sim_gain"]["factor"], 0.97 / 0.85)
    assert math.isclose(s["sim_gain"]["S_cm"], abs(102.0 * 0.97 / 0.85 - 120.0))
    assert math.isclose(s["root_specific_removed"]["gain"], c["stride"])


def test_insim_speed_gains_and_the_speed_bootstrap(tmp_path):
    """Review of F2D: the in-sim pass-speed divisors (root: median of model over true speed of the forward takes;
    legs: 1 + speed bias / mean true speed) on a temporary tree, and the person bootstrap of test A on the pass speeds:
    a point interval when every person is the same, a nondegenerate one around f when the persons differ."""
    import json
    name = "run_t_ckpt_B_000001"
    folder = os.path.dirname(M.insim_json(name, str(tmp_path)))
    os.makedirs(folder)
    with open(os.path.join(folder, "insim.json"), "w", encoding="utf-8") as f:
        json.dump({"checkpoint": {"sha256": "a" * 64},
                   "lines": {"forward": {"model": {"stride_bias_cm": {"point": -3.74}},
                                         "model_legs": {"stride_bias_cm": {"point": -13.2},
                                                        "speed_bias_m_s": {"point": -0.12}}}},
                   "counts": {"forward": {"true_stride_m": 1.10, "true_speed_m_s": 1.0}}}, f)
    pd.DataFrame({"forward": ["True", "True", "True", "False"], "model_speed_m_s": [0.97, 0.98, 0.99, 0.5],
                  "true_speed_m_s": [1.0, 1.0, 1.0, 1.0]}).to_csv(os.path.join(folder, "takes.csv"), index=False,
                                                                  encoding="utf-8")
    g = M.insim_speed_gains(name, str(tmp_path))
    assert math.isclose(g["root"], 0.98) and math.isclose(g["legs"], 1.0 - 0.12 / 1.0)
    # identical persons: every resample gives f, so the interval is the point
    ev = M.read_rows(_write(tmp_path, _events()), None, [], [f"weargait:P{p}" for p in range(4)])
    q = M.test_a_speed(ev, g, n_boot=50, seed=1)
    assert math.isclose(q["f"], M.f_share(0.85, 0.8, 0.98, 0.88)[2])
    assert math.isclose(q["f_ci"][0], q["f"]) and math.isclose(q["f_ci"][1], q["f"])
    # persons that differ: the interval opens around f
    rows = []
    for p, (gr, gl) in enumerate([(0.80, 0.78), (0.84, 0.80), (0.88, 0.84), (0.92, 0.85), (0.86, 0.79)]):
        pid, tid = f"weargait:P{p}", f"P{p}_SelfPace"
        for k in range(3):
            v = 1.0 + 0.1 * k
            rows.append(("base", "speed", tid, pid, "SelfPace", "", str(k), 1.0, 6.0, v, gr * v, (gr - 1) * v, "m/s",
                         "True"))
            rows.append(("model_legs", "speed", tid, pid, "SelfPace", "", str(k), 1.0, 6.0, v, gl * v, (gl - 1) * v,
                         "m/s", "True"))
    ev2 = M.read_rows(_write(tmp_path, pd.DataFrame(rows, columns=_events().columns)), None, [],
                      [f"weargait:P{p}" for p in range(5)])
    q2 = M.test_a_speed(ev2, g, n_boot=400, seed=1)
    assert math.isclose(q2["g_root_speed_real"], 0.86) and math.isclose(q2["g_legs_speed_real"], 0.80)
    lo, hi = q2["f_ci"]
    assert lo < q2["f"] < hi and hi - lo > 0.05
    assert q2["reading_if_registered"] == M.reading_a(q2["f"], lo, hi)


def test_rerun_asserts_the_constants_of_the_first_run(tmp_path):
    first = tmp_path / "results.json"
    consts = {"boot_n": 10000, "seed": 20261005, "level": 0.95, "f_support": 0.5, "k_band": [0.97, 1.03],
              "d_band": [0.98, 1.02], "sv_band": [0.97, 1.03], "slip_fraction": 0.5}
    first.write_text(__import__("json").dumps({"constants": consts, "prediction": M.REPORT}), encoding="utf-8")
    res = {"constants": {**consts, "k_band": (0.97, 1.03), "d_band": (0.98, 1.02), "sv_band": (0.97, 1.03)},
           "prediction": M.REPORT}
    assert M.assert_same_constants(res, str(first))
    with pytest.raises(AssertionError):
        M.assert_same_constants({**res, "constants": {**consts, "f_support": 0.4}}, str(first))
    rep = tmp_path / "r.md"
    rep.write_text("# t\n\n## Prediction\n\nabc\n\n## Results\nx\n", encoding="utf-8")
    import hashlib
    assert M.prediction_section_sha256(str(rep)) == hashlib.sha256(b"## Prediction\n\nabc\n").hexdigest()
    assert M.parse(["--results-out", "x"]).results_out == "x"


def test_exploratory_mae(tmp_path):
    ev = M.read_rows(_write(tmp_path, _events()), None, [], [f"weargait:P{p}" for p in range(4)])
    pairs = M.paired_strides(ev)
    x = M.exploratory_mae(pairs, {"root": 0.85, "legs": 0.8}, n_boot=20, seed=1)
    assert math.isclose(x["root_raw"]["mae_cm"], 18.0)
    assert math.isclose(x["legs_raw"]["mae_cm"], 24.0)
    assert x["root_over_insim_gain"]["mae_cm"] < 1e-9
    assert x["legs_over_insim_gain"]["mae_cm"] < 1e-9
    assert math.isclose(x["legs_over_insim_gain"]["gain"], 1.0)
