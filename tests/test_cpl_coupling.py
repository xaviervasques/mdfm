"""Tests of scripts/cpl_coupling.py on hand-made signals and tables (no store, no record, no person)."""
import importlib.util
import math
import os

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(os.path.dirname(HERE), "scripts", "cpl_coupling.py")


def _mod():
    spec = importlib.util.spec_from_file_location("_t_cpl_coupling", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


M = _mod()


def _rot(axis, ang):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K


def test_stride_rom_sinusoid_and_bias():
    fps, f0, A = 100.0, 0.9, 30.0
    t = np.arange(0, 1 / f0 + 1e-9, 1 / fps)
    w = A * 2 * math.pi * f0 * np.cos(2 * math.pi * f0 * t)          # angle A sin: excursion 2A
    rom, peak = M.stride_rom(w, 0, len(t) - 1, fps)
    assert rom == pytest.approx(2 * A, rel=0.01)
    assert peak == pytest.approx(A * 2 * math.pi * f0, rel=1e-6)
    rom_b, _ = M.stride_rom(w + 5.0, 0, len(t) - 1, fps)               # a constant bias is detrended
    assert rom_b == pytest.approx(rom, rel=1e-6)
    w2 = w.copy()
    w2[10] = np.nan
    assert math.isnan(M.stride_rom(w2, 0, len(t) - 1, fps)[0])
    assert math.isnan(M.stride_rom(w, 0, len(t) + 5, fps)[0])


def test_rom_phase_invariant():
    fps, f0, A = 100.0, 1.0, 20.0
    t = np.arange(0, 5, 1 / fps)
    w = A * 2 * math.pi * f0 * np.cos(2 * math.pi * f0 * t) + 0.3 * A * 2 * math.pi * 2 * f0 * np.cos(
        2 * math.pi * 2 * f0 * t + 0.4)
    roms = [M.stride_rom(w, k, k + 100, fps)[0] for k in (0, 13, 37, 71)]
    assert max(roms) - min(roms) < 0.02 * max(roms)


def test_axis_and_segment_table_with_mount():
    rng = np.random.default_rng(0)
    fps = 100.0
    t = np.arange(0, 6, 1 / fps)
    w_sag = 200 * np.sin(2 * math.pi * 0.9 * t)
    g = np.stack([0.1 * w_sag + rng.normal(0, 2, len(t)), rng.normal(0, 2, len(t)), w_sag], 1)
    Rm = _rot([0.3, 1.0, -0.2], 0.7)
    gm = g @ Rm.T                                                       # sensor mounted with a rotation
    win = [(0, 111), (111, 222), (222, 333)]
    rom_a, _, axis_a, plan_a = M.segment_table(g, win, fps)
    rom_b, _, axis_b, plan_b = M.segment_table(gm, win, fps)
    assert np.allclose(rom_a, rom_b, rtol=1e-6)
    assert plan_a > 0.95 and plan_b == pytest.approx(plan_a, rel=1e-9)
    assert abs(abs(axis_b @ (Rm @ axis_a)) - 1) < 1e-9


def test_lowpass_keeps_gait_band_and_handles_gaps():
    fps = 100.0
    t = np.arange(0, 10, 1 / fps)
    slow = np.sin(2 * math.pi * 1.0 * t)
    fast = np.sin(2 * math.pi * 30.0 * t)
    y = M.lowpass(np.stack([slow + fast, slow, slow], 1), fps)
    mid = slice(100, 900)
    assert np.abs(y[mid, 0] - slow[mid]).max() < 0.02
    x = np.stack([slow] * 3, 1)
    x[500:505] = np.nan
    x[600:610, 1] = np.nan                                              # a NaN in one column cuts the run
    z = M.lowpass(x, fps)
    assert np.isnan(z[500:505]).all() and np.isnan(z[600:610]).all()
    assert np.isfinite(z[100:490]).all() and np.isfinite(z[700:900]).all()


def test_style_filter():
    good = ["BMLrub/rub001/0001_normal_walk1_poses.npz", "BMLrub/rub002/0007_treadmill_fast_poses.npz",
            "KIT/3/walking_medium05_poses.npz", "KIT/11/WalkingStraightForwards01_poses.npz",
            "KIT/9/WalkingStraightForward03_poses.npz", "KIT/4/walking_2_poses.npz", "KIT/1/walking1_poses.npz",
            "KIT/2/walk_3m_straight_line01_poses.npz", "KIT/5/WalkingStraightForwards02_1_poses.npz"]
    bad = ["BMLrub/rub001/0003_circle_walk_poses.npz", "KIT/11/WalkingStraightBackwards01_poses.npz",
           "KIT/3/step_over_gap01_poses.npz", "KIT/3/walk_with_handrail_left01_poses.npz",
           "KIT/3/WalkInClockwiseCircle01_poses.npz", "KIT/3/walking_forward_4steps_right_01_poses.npz",
           "KIT/3/nordic_walking01_poses.npz", "KIT/3/RecoveryStepping_1_2_3_poses.npz"]
    assert all(M.is_straight_forward(x) for x in good)
    assert not any(M.is_straight_forward(x) for x in bad)


def _population(rng, n_persons, prefix, kappa=1.0, n=30, noise=0.03, treadmill=0):
    """Strides with log ROM_s = a_s + b_s log(S / l) + d_s (log c - log 110) + log kappa + noise."""
    coef = {"thigh": (3.9, 0.45, 0.05), "shank": (4.4, 0.40, 0.10), "ankle": (4.4, 0.40, 0.10),
            "foot": (4.6, 0.30, 0.08)}
    rows = []
    for k in range(n_persons):
        stature = rng.uniform(1.5, 1.9)
        l = M.LEG_RATIO * stature
        for j in range(n):
            T = rng.uniform(0.9, 1.3)
            c = 120 / T
            S = l * math.exp(rng.normal(0.35, 0.12))
            row = {"person_id": f"{prefix}{k}", "unit": f"{prefix}{k}_u{j // 10}", "t0": 0.0, "t1": T, "S": S,
                   "stature_m": stature, "treadmill": treadmill, "archive": "X", "task": "SelfPace"}
            for s, (a, b, d) in coef.items():
                row[f"rom_{s}"] = math.exp(a + b * math.log(S / l) + d * (math.log(c) - M.C_CENTRE)
                                           + math.log(kappa) + rng.normal(0, noise))
                row[f"peak_{s}"] = 100.0
            rows.append(row)
    return M.finish(pd.DataFrame(rows))


def test_relations_recover_kappa_and_gamma():
    rng = np.random.default_rng(1)
    am = _population(rng, 40, "a")
    real = _population(rng, 30, "r", kappa=0.93)
    rel = M.Relations(am)
    fit = rel.fit()
    assert fit["shank"][0][1] == pytest.approx(0.40, abs=0.03)
    st = M.statistics(real, fit)
    for s in M.PRIMARY:
        assert st[f"kappa_{s}"] == pytest.approx(0.93, abs=0.012)
    # gamma_L: the stride shift whose AMASS rotations match: log 0.93 / b per segment, weighted
    want = {s: math.log(0.93) / fit[s][0][1] for s in M.PRIMARY}
    w = {s: fit[s][0][1] ** 2 / fit[s][1] for s in M.PRIMARY}
    expect = math.exp(sum(w[s] * want[s] for s in M.PRIMARY) / sum(w.values()))
    assert st["gamma_L"] == pytest.approx(expect, rel=0.03)
    assert st["gamma_L"] < 0.9
    same = M.statistics(_population(rng, 30, "q"), fit)
    assert same["gamma_L"] == pytest.approx(1.0, abs=0.03)


def test_bootstrap_brackets_point():
    rng = np.random.default_rng(2)
    am = _population(rng, 25, "a", n=15)
    real = _population(rng, 20, "r", kappa=0.95, n=15)
    rel = M.Relations(am)
    out = M.interval_row(rel, real, 200)
    lo, hi = out["interval"]["gamma_L"]
    assert lo <= out["point"]["gamma_L"] <= hi
    assert hi < 1.0


def test_reading_rule():
    assert M.reading(0.88, [0.84, 0.92]) == "COUPLING EXPLAINS"
    assert M.reading(0.93, [0.88, 0.975]) == "PARTLY"
    assert M.reading(1.0, [0.96, 1.04]) == "COUPLING MATCHED"
    assert M.reading(1.0, [0.92, 1.04]) == "PARTLY"
    assert M.reading(1.10, [1.02, 1.2]) == "COUPLING LONG"
    assert M.reading(0.95, [0.90, 1.00]) == "PARTLY"


def test_person_split_and_medians():
    assert M.person_split("amass:x") == M.person_split("amass:x")
    assert {M.person_split(f"p{k}") for k in range(20)} == {0, 1}
    pm = M.person_median(["a", "a", "b", "b", "b"], [1.0, 3.0, np.nan, 2.0, 4.0])
    assert pm == {"a": 2.0, "b": 3.0}
    assert M.med_of_persons(pm) == 2.5


def test_check_prediction(tmp_path):
    p = tmp_path / "prediction.txt"
    p.write_text("registered\n", encoding="utf-8")
    sha = M.sha256_file(str(p))
    h = tmp_path / "prediction.sha256"
    h.write_text(f"{sha}  prediction.txt\n", encoding="utf-8")
    assert M.check_prediction(str(p), str(h), sha) == sha
    with pytest.raises(AssertionError):
        M.check_prediction(str(p), str(h), "0" * 64)
    p.write_text("changed\n", encoding="utf-8")
    with pytest.raises(AssertionError):
        M.check_prediction(str(p), str(h), sha)


def test_no_dashes_in_script():
    with open(SCRIPT, encoding="utf-8") as fh:
        text = fh.read()
    assert chr(0x2013) not in text and chr(0x2014) not in text
