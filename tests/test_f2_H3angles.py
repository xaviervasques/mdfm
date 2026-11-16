"""Tests of scripts/f2_H3angles.py (task F2-H3angles: leg excursion and cadence of the decoded walk) on synthetic
bodies and events, without a weargait file, a real checkpoint or the GPU:

- angle_series and stride_features recover known hip and knee angles of a planar leg, unchanged by a yaw and a
  translation of the body; a stitched body is read in the window of the stride's midpoint, another window when that
  one lacks a frame;
- stride_durations pairs each walkway stride with its two walkway IC rows and the decoded IC times; the run's IC
  offset cancels, a time scale of the decoded ICs is r_T;
- part_a_set and the readings: a pure length error at the walkway rhythm, carried by the legs, reads SPATIAL and
  LEGS; legs at the walkway length with a short root read ROOT; a decoded rhythm 0.85 times the walkway one reads
  TEMPORAL; the edges of C1, C2, Part A and Part B;
- the events reader refuses a lockbox person and a person outside the dev set before reading values;
- part_a end to end on a temporary tree (diag and in-sim folders);
- part_b on a synthetic dump: decoded real ranges short beyond the in-sim ratio read SUPPORTED, ranges at the in-sim
  ratio NOT_SUPPORTED, no dump NOT evaluated;
- the dump job command passes scripts/queue_job.py's checks (dry run, nothing written);
- task F2D: the descriptive block of Part B (speed and stride expectations, an older dump skips it without touching
  the registered reading) and the rerun guard on the first run's criteria;
- --dump with a tiny model on CPU (fakes for the checkpoint, the folds, the targets and the records) writes
  strides_real.csv with train54 and dev14 persons only; a lockbox record stops it; the in-sim part on one AMASS dev
  take when the dev body store is present.
"""
import hashlib
import importlib.util
import json
import math
import os

import numpy as np
import pandas as pd
import pytest

from mdfm import paths
from mdfm.data.canonical import Refused
from mdfm.gates import g3

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCRIPT = os.path.join(ROOT, "scripts", "f2_H3angles.py")
HAVE_MA = os.path.exists(paths.smplh_model("neutral"))


def _load(path, name):
    old = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        if old is None:
            os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        else:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = old
    return mod


H = _load(SCRIPT, "f2_h3angles_tested")
GG = H.GG()


@pytest.fixture(autouse=True)
def keep_alloc_conf():
    old = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    yield
    if old is None:
        os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    else:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = old


# ---------------------------------------------------------------- synthetic bodies
def leg_body(n=101, hip=(-10.0, 30.0), knee=(0.0, 60.0), yaw=0.0, shift=(0.0, 0.0, 0.0), side="L"):
    """(n, 22, 3) joints of a body whose `side` leg has a hip angle sweeping hip and a knee flexion sweeping knee
    (degrees, sinusoids over the n frames), the pelvis frame +x left, +y up, +z forward, then turned by yaw about y
    and shifted."""
    t = np.linspace(0.0, 2.0 * np.pi, n)
    th = np.radians(hip[0] + (hip[1] - hip[0]) * 0.5 * (1 - np.cos(t)))
    ka = np.radians(knee[0] + (knee[1] - knee[0]) * 0.5 * (1 - np.cos(2 * t)))
    J = np.zeros((n, 22, 3))
    J[:, 0] = [0.0, 1.0, 0.0]
    J[:, 1] = [0.1, 0.95, 0.0]
    J[:, 2] = [-0.1, 0.95, 0.0]
    J[:, 3] = [0.0, 1.1, 0.0]
    hi, ki, ai = (1, 4, 7) if side == "L" else (2, 5, 8)
    oi, oki, oai = (2, 5, 8) if side == "L" else (1, 4, 7)
    J[:, ki] = J[:, hi] + 0.4 * np.stack([0 * th, -np.cos(th), np.sin(th)], -1)
    J[:, ai] = J[:, ki] + 0.4 * np.stack([0 * th, -np.cos(th - ka), np.sin(th - ka)], -1)
    J[:, oki] = J[:, oi] + [0.0, -0.4, 0.0]
    J[:, oai] = J[:, oki] + [0.0, -0.4, 0.0]
    for j in range(22):
        if not J[:, j].any():
            J[:, j] = J[:, 0]
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return J @ R.T + np.asarray(shift)


def test_angle_series_recovers_known_angles():
    J = leg_body()
    s = H.angle_series(J, "L")
    assert np.allclose(s["hip_deg"].min(), -10.0, atol=1e-6) and np.allclose(s["hip_deg"].max(), 30.0, atol=1e-6)
    assert np.allclose(s["knee_deg"].min(), 0.0, atol=1e-5) and np.allclose(s["knee_deg"].max(), 60.0, atol=1e-6)
    assert np.allclose(s["leg_length_m"], 0.8)
    f = H.stride_features(J, None, "L", 0.0, 1.0)
    assert f["hip_range_deg"] == pytest.approx(40.0, abs=1e-6) and f["knee_range_deg"] == pytest.approx(60.0, abs=1e-5)
    assert f["thigh_range_deg"] == pytest.approx(40.0, abs=1e-6) and f["leg_length_m"] == pytest.approx(0.8)
    assert f["root_m"] == pytest.approx(0.0, abs=1e-12)


def test_angles_unchanged_by_yaw_and_translation_and_side():
    a = H.stride_features(leg_body(), None, "L", 0.0, 1.0)
    b = H.stride_features(leg_body(yaw=1.1, shift=(3.0, 0.0, -2.0)), None, "L", 0.0, 1.0)
    for k in ("hip_range_deg", "knee_range_deg", "thigh_range_deg", "leg_length_m", "ankle_fa_range_m"):
        assert a[k] == pytest.approx(b[k], abs=1e-6)
    r = H.stride_features(leg_body(side="R", hip=(-5.0, 25.0)), None, "R", 0.0, 1.0)
    assert r["hip_range_deg"] == pytest.approx(30.0, abs=1e-6)


def test_stride_features_reads_the_midpoint_window_or_a_whole_one():
    J1, J2 = leg_body(), leg_body(hip=(0.0, 20.0))
    W = np.full((2, 101, 22, 3), np.nan)
    W[0, :] = J1
    W[1, 40:] = J2[40:]
    wof = np.where(np.arange(101) < 30, 0, 1)
    f = H.stride_features(W, wof, "L", 0.0, 1.0)                 # midpoint window 1 lacks frames 0..39: window 0
    assert f["hip_range_deg"] == pytest.approx(40.0, abs=1e-6)
    W[1, :] = J2
    f = H.stride_features(W, wof, "L", 0.0, 1.0)
    assert f["hip_range_deg"] == pytest.approx(20.0, abs=1e-6)
    assert math.isnan(H.stride_features(W, wof, "L", 0.5, 2.0)["hip_range_deg"])     # off the clock


# ---------------------------------------------------------------- synthetic events
COLS = ["row", "kind", "trial_id", "person_id", "task", "foot", "pass_id", "t0_s", "t1_s", "reference", "prediction",
        "signed_error", "unit", "matched"]
DEV = ["weargait:WHC003", "weargait:WHC026", "weargait:DV01", "weargait:DV02", "weargait:DV03", "weargait:DV04"]
TRAIN = [f"weargait:TR{i:02d}" for i in range(1, 9)]
LOCK = ["weargait:LB01"]
SETS = {"dev14": DEV, "dev12": [p for p in DEV if p not in GG.EXCLUDED_FROM_12], "train54": TRAIN}


def events(persons, base=0.85, legs=0.75, tscale=1.0, offset_s=0.02, seed=0):
    """events.csv rows of synthetic persons (one trial each, 6 strides per foot): walkway ICs every 0.55 s alternating
    feet; decoded ICs at offset + tscale x walkway time (+ small noise); base strides base x walkway, legs strides
    legs x walkway; one pass speed per trial (base and legs scaled the same)."""
    rng = np.random.default_rng(seed)
    rows = []
    for p in persons:
        tid = p.split(":")[1] + "_SelfPace"
        ics = {"L": [1.0 + 1.1 * k for k in range(7)], "R": [1.55 + 1.1 * k for k in range(7)]}
        for foot, ts in ics.items():
            for t in ts:
                pred = offset_s + tscale * t + rng.normal(0, 0.003)
                rows.append(("base", "ic_walkway", tid, p, "SelfPace", foot, "", t, np.nan, 1000 * t, 1000 * pred,
                             1000 * (pred - t), "ms", "True"))
            for a, b in zip(ts[:-1], ts[1:]):
                x = 120.0 + rng.normal(0, 5)
                for row, g in (("base", base), ("model_legs", legs)):
                    y = g * x * (1 + rng.normal(0, 0.01))
                    rows.append((row, "stride", tid, p, "SelfPace", foot, "0", a, b, x, y, y - x, "cm", "True"))
        for row, g in (("base", base), ("model_legs", legs)):
            rows.append((row, "speed", tid, p, "SelfPace", "", "0", 1.0, 7.6, 1.1, g * 1.1, (g - 1) * 1.1, "m/s",
                         "True"))
    return pd.DataFrame(rows, columns=COLS)


def _ev(df):
    df = df.copy()
    df["is_matched"] = df["matched"].str.lower() == "true"
    return df


def test_stride_durations_offset_cancels_and_scale_is_r_T():
    d = H.stride_durations(_ev(events(DEV, tscale=1.0, offset_s=0.05)))
    assert d.attrs["ends_found"] == 1.0 and len(d) == len(DEV) * 12
    assert np.allclose(d["r_T"], 1.0, atol=0.01)
    d = H.stride_durations(_ev(events(DEV, tscale=0.85, offset_s=-0.03)))
    assert np.allclose(d["r_T"], 0.85, atol=0.01)


def test_spatial_legs_reading():
    d = _ev(events(DEV, base=0.85, legs=0.85 * 0.88))
    dur = H.stride_durations(d)
    st = d[(d["kind"] == "stride")].rename(columns={"reference": "x", "prediction": "y"})
    s = H.part_a_set(dur, st[st["row"] == "model_legs"], st[st["row"] == "base"], 0.88, 0.966, SETS["dev12"],
                     n_boot=300)
    assert s["r_T"]["point"] == pytest.approx(1.0, abs=0.01)
    assert s["D_L"]["point"] == pytest.approx(0.85, abs=0.01) and s["D_R"]["point"] == pytest.approx(0.88, abs=0.01)
    assert H.reading_c1(s) == "SPATIAL" and H.reading_c2(s) == "LEGS"


def test_root_and_temporal_readings():
    d = _ev(events(DEV, base=0.85, legs=0.88))
    dur = H.stride_durations(d)
    st = d[(d["kind"] == "stride")].rename(columns={"reference": "x", "prediction": "y"})
    s = H.part_a_set(dur, st[st["row"] == "model_legs"], st[st["row"] == "base"], 0.88, 0.966, SETS["dev12"],
                     n_boot=300)
    assert H.reading_c2(s) == "ROOT"
    d = _ev(events(DEV, base=0.85, legs=0.85, tscale=0.85))
    dur = H.stride_durations(d)
    st = d[(d["kind"] == "stride")].rename(columns={"reference": "x", "prediction": "y"})
    s = H.part_a_set(dur, st[st["row"] == "model_legs"], st[st["row"] == "base"], 1.0, 1.0, SETS["dev12"], n_boot=300)
    assert s["s_T"]["point"] == pytest.approx(1.0, abs=0.08) and H.reading_c1(s) == "TEMPORAL"


def blk(r_iv, sT, DL, DR):
    return {"r_T": {"interval": r_iv}, "s_T": {"point": sT}, "D_L": {"point": DL}, "D_R": {"point": DR}}


def test_reading_edges():
    assert H.reading_c1(blk([0.97, 1.03], 0.2, 1, 1)) == "SPATIAL"
    assert H.reading_c1(blk([0.969, 1.0], 0.0, 1, 1)) == "MIXED"
    assert H.reading_c1(blk([0.80, 0.86], 0.5, 1, 1)) == "TEMPORAL"
    assert H.reading_c1(blk([0.80, 0.86], 0.49, 1, 1)) == "MIXED"
    assert H.reading_c1(blk(None, 0.0, 1, 1)) == "MIXED"
    assert H.reading_c2(blk(None, 0, 0.92, 0.88)) == "LEGS"
    assert H.reading_c2(blk(None, 0, 0.92, 0.87)) == "BETWEEN"
    assert H.reading_c2(blk(None, 0, 0.96, 0.92)) == "ROOT"
    assert H.reading_c2(blk(None, 0, 0.95, 0.88)) == "BETWEEN"
    assert H.reading_a({"a": ("SPATIAL", "LEGS"), "b": ("SPATIAL", "LEGS")})[0] == "compatible"
    assert H.reading_a({"a": ("SPATIAL", "ROOT"), "b": ("MIXED", "ROOT")})[0] == "not_supported"
    assert H.reading_a({"a": ("TEMPORAL", "LEGS"), "b": ("TEMPORAL", "BETWEEN")})[0] == "not_supported"
    assert H.reading_a({"a": ("SPATIAL", "LEGS"), "b": ("SPATIAL", "BETWEEN")})[0] == "partly"
    assert H.reading_b(0.92, 0.99) == "SUPPORTED" and H.reading_b(0.92, 1.0) == "PARTLY"
    assert H.reading_b(0.96, 1.2) == "NOT_SUPPORTED" and H.reading_b(0.94, 0.9) == "PARTLY"
    assert H.reading_b(float("nan"), 0.9) == "NOT_EVALUATED"


def _write(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


def test_events_reader_guards_persons_first(tmp_path):
    p = str(tmp_path / "events.csv")
    _write(events(DEV + LOCK), p)
    with pytest.raises(Refused):
        H.read_events(p, DEV, LOCK)
    _write(events(DEV + TRAIN[:1]), p)
    with pytest.raises(AssertionError, match="outside"):
        H.read_events(p, DEV, LOCK)
    _write(events(DEV), p)
    assert set(H.read_events(p, DEV, LOCK)["person_id"]) == set(DEV)


def test_part_a_end_to_end_on_a_temporary_tree(tmp_path):
    tree = str(tmp_path)
    end = GG.End("v1_B", "run_t_ckpt_B_000001", "cache/runs/run_t/ckpt_B_000001.pt", True, 32)
    dd = GG.diag_dir(end.name, tree)
    _write(events(DEV, base=0.85, legs=0.85 * 0.88), os.path.join(dd, "events.csv"))
    with open(os.path.join(dd, "diag.json"), "w", encoding="utf-8") as f:
        json.dump({"checkpoint": {"sha256": "a" * 64}}, f)
    idir = GG.insim_dir(end.name, tree)
    os.makedirs(idir)
    with open(os.path.join(idir, "insim.json"), "w", encoding="utf-8") as f:
        json.dump({"checkpoint": {"sha256": "a" * 64},
                   "lines": {"forward": {"model": {"stride_bias_cm": {"point": -3.74}},
                                         "model_legs": {"stride_bias_cm": {"point": -13.2},
                                                        "speed_bias_m_s": {"point": -0.12}}}},
                   "counts": {"forward": {"true_stride_m": 1.10, "true_speed_m_s": 1.0}},
                   "windows": {"forward": {"pelvis_speed_ratio": {"median": 0.98}, "windows": 10}}}, f)
    _write(pd.DataFrame({"forward": ["True", "True"], "model_speed_m_s": [0.97, 0.98], "true_speed_m_s": [1.0, 1.0]}),
           os.path.join(idir, "takes.csv"))
    res = H.part_a(end, SETS, LOCK, n_boot=200, tree=tree)
    s = res["sets"]["dev12"]
    assert res["ends_found"] == 1.0 and s["C1"] == "SPATIAL" and s["C2"] == "LEGS"
    assert res["insim"]["model_legs"]["stride_gain"] == pytest.approx(0.88)
    assert res["speed_gains"]["dev12"]["base"] == pytest.approx(0.85)
    with open(os.path.join(idir, "insim.json"), encoding="utf-8") as f:
        doc = json.load(f)
    doc["checkpoint"]["sha256"] = "b" * 64
    with open(os.path.join(idir, "insim.json"), "w", encoding="utf-8") as f:
        json.dump(doc, f)
    with pytest.raises(ValueError, match="checkpoint"):
        H.part_a(end, SETS, LOCK, n_boot=50, tree=tree)


# ---------------------------------------------------------------- Part B
def dump_files(folder, real_factor, sim_factor=0.95, seed=0):
    rng = np.random.default_rng(seed)
    sim = []
    for k in range(6):
        for i in range(20):
            v = 0.7 + 0.6 * rng.random()
            th, tk = 20 + 15 * v + rng.normal(0, 1), 40 + 15 * v + rng.normal(0, 1)
            sim.append({"take_id": f"t{k}", "person_id": f"amass:P{k}", "scored": True, "foot": "L",
                        "true_speed_m_s": v, "true_hip_range_deg": th, "dec_hip_range_deg": sim_factor * th,
                        "true_knee_range_deg": tk, "dec_knee_range_deg": sim_factor * tk, "true_leg_length_m": 0.85,
                        "dec_leg_length_m": 0.84, "legs_m": 1.0, "true_ankle_m": 1.1})
    real = []
    for p in DEV + TRAIN:
        for i in range(10):
            v = 1.0 + 0.5 * rng.random()
            real.append({"trial_id": "x", "person_id": p, "foot": "R", "walkway_m": 1.3, "walkway_speed_m_s": v,
                         "dec_hip_range_deg": real_factor * (20 + 15 * v), "dec_knee_range_deg": real_factor * (40 + 15 * v),
                         "dec_ankle_m": 1.1, "dec_root_m": 1.1, "legs_m": 1.0, "dec_leg_length_m": 0.84})
    _write(pd.DataFrame(sim), os.path.join(folder, "strides_insim.csv"))
    _write(pd.DataFrame(real), os.path.join(folder, "strides_real.csv"))


def test_part_b_readings(tmp_path):
    f = str(tmp_path / "dump")
    assert not H.part_b(f, SETS, LOCK)["evaluated"]
    dump_files(f, real_factor=0.95 * 0.85)
    b = H.part_b(f, SETS, LOCK, n_boot=200)
    assert b["joints"]["hip"]["rho_sim"] == pytest.approx(0.95, abs=1e-9)
    assert b["joints"]["hip"]["sets"]["train54"]["ratio_to_sim"] == pytest.approx(0.85, abs=0.03)
    assert b["reading"] == "SUPPORTED"
    dump_files(f, real_factor=0.95)
    assert H.part_b(f, SETS, LOCK, n_boot=200)["reading"] == "NOT_SUPPORTED"
    real = pd.read_csv(os.path.join(f, "strides_real.csv"), encoding="utf-8")
    real.loc[0, "person_id"] = LOCK[0]
    _write(real, os.path.join(f, "strides_real.csv"))
    with pytest.raises(Refused):
        H.part_b(f, SETS, LOCK)


def full_dump_files(folder, real_arc=1.0, real_stride=1.0, seed=0):
    """A synthetic dump with every column of the real dump: in-sim truth arcs and sweep linear in the true speed,
    stride = 0.4 + 0.6 v, decoded = truth x 0.9; real strides real_stride x the in-sim stride at their speed, decoded
    arcs real_arc x 0.9 x the truth line at the walkway speed."""
    rng = np.random.default_rng(seed)
    sim, real = [], []
    lines = {"hip_range_deg": (20.0, 15.0), "knee_range_deg": (40.0, 15.0), "thigh_range_deg": (22.0, 14.0),
             "ankle_fa_range_m": (0.3, 0.2)}
    for k in range(6):
        for i in range(20):
            v = 0.7 + 0.6 * rng.random()
            row = {"take_id": f"t{k}", "person_id": f"amass:P{k}", "scored": True, "foot": "L", "true_speed_m_s": v,
                   "true_stride_m": 0.4 + 0.6 * v, "true_ankle_m": 0.4 + 0.6 * v, "legs_m": 0.9 * (0.4 + 0.6 * v),
                   "true_leg_length_m": 0.85, "dec_leg_length_m": 0.84}
            for c, (a, b) in lines.items():
                row[f"true_{c}"] = a + b * v
                row[f"dec_{c}"] = 0.9 * (a + b * v)
            sim.append(row)
    for p in DEV + TRAIN:
        for i in range(10):
            v = 0.8 + 0.4 * rng.random()
            row = {"trial_id": "x", "person_id": p, "foot": "R", "walkway_speed_m_s": v,
                   "walkway_m": real_stride * (0.4 + 0.6 * v), "dec_ankle_m": 0.85 * real_stride * (0.4 + 0.6 * v),
                   "dec_root_m": 0.85, "legs_m": 0.8, "dec_leg_length_m": 0.84}
            for c, (a, b) in lines.items():
                row[f"dec_{c}"] = real_arc * 0.9 * (a + b * v)
            real.append(row)
    _write(pd.DataFrame(sim), os.path.join(folder, "strides_insim.csv"))
    _write(pd.DataFrame(real), os.path.join(folder, "strides_real.csv"))


def test_part_b_descriptive_speed_and_stride_expectations(tmp_path):
    """Descriptive block of task F2D: real arcs at the in-sim decoded ratio read 1 on the speed line; when the real
    strides are 1.2 x the in-sim strides at the same speed, the stride line expects longer arcs and reads below 1;
    the absent columns of an older dump skip the block without touching the registered reading."""
    f = str(tmp_path / "dump")
    full_dump_files(f, real_arc=1.0, real_stride=1.0)
    d = H.part_b(f, SETS, LOCK, n_boot=100)["descriptive_dumps_2026_10_05"]
    assert d["thigh"]["ratio_to_sim"] == pytest.approx(1.0, abs=1e-6)
    assert d["real_stride_over_insim_stride_at_equal_speed"]["median_of_person_medians"] == pytest.approx(1.0)
    assert d["stride_based_expectation"]["hip_range_deg"]["ratio_to_sim"] == pytest.approx(1.0, abs=1e-6)
    assert d["speed_slopes"]["hip"]["real_decoded_over_sim_true"] == pytest.approx(0.9, abs=1e-6)
    full_dump_files(f, real_arc=1.0, real_stride=1.2)
    b = H.part_b(f, SETS, LOCK, n_boot=100)
    d = b["descriptive_dumps_2026_10_05"]
    assert d["real_stride_over_insim_stride_at_equal_speed"]["median_of_person_medians"] == pytest.approx(1.2)
    assert d["stride_based_expectation"]["hip_range_deg"]["ratio_to_sim"] < 0.95
    assert b["reading"] == "NOT_SUPPORTED"                          # the registered reading is on the speed line
    assert "Descriptive, no criterion" in H.summary_md(_summary_stub(b))
    dump_files(f, real_factor=0.95)                                  # an older dump without the new columns
    b = H.part_b(f, SETS, LOCK, n_boot=100)
    assert b["descriptive_dumps_2026_10_05"]["skipped"] and b["reading"] == "NOT_SUPPORTED"
    assert "skipped" in H.summary_md(_summary_stub(b))


LINES = {"hip_range_deg": (20.0, 15.0), "knee_range_deg": (40.0, 15.0), "thigh_range_deg": (22.0, 14.0),
         "ankle_fa_range_m": (0.3, 0.2)}


def k_of(i):
    """Person factor of the i-th person of DEV + TRAIN (0.80 to 0.95)."""
    return 0.8 + 0.05 * (i % 4)


def person_dump_files(folder, seed=0):
    """A synthetic dump for the remaining descriptive lines. In sim: the truth lines of LINES on the true speed
    (0.7 to 1.3 m/s), decoded arcs 0.9 x truth, decoded leg 0.84 m against a true 0.85 m. Real: speeds 0.5 to 1.5 m/s
    (part of them outside the in-sim 10th to 90th percentile band), walkway stride 0.4 + 0.6 v over t1 - t0 =
    stride / v, a person factor k: decoded arcs 0.9 k x the truth line, decoded ankle stride 0.85 k x the walkway
    stride (so the decoded speed is 0.85 k v), decoded leg 0.86 m."""
    rng = np.random.default_rng(seed)
    sim, real = [], []
    for k in range(6):
        for i in range(20):
            v = 0.7 + 0.6 * rng.random()
            row = {"take_id": f"t{k}", "person_id": f"amass:P{k}", "scored": True, "foot": "L", "true_speed_m_s": v,
                   "true_stride_m": 0.4 + 0.6 * v, "true_ankle_m": 0.4 + 0.6 * v, "legs_m": 0.9 * (0.4 + 0.6 * v),
                   "true_leg_length_m": 0.85, "dec_leg_length_m": 0.84}
            for c, (a, b) in LINES.items():
                row[f"true_{c}"] = a + b * v
                row[f"dec_{c}"] = 0.9 * (a + b * v)
            sim.append(row)
    for n, p in enumerate(DEV + TRAIN):
        kp = k_of(n)
        for i in range(10):
            v = 0.5 + 1.0 * rng.random()
            s = 0.4 + 0.6 * v
            row = {"trial_id": "x", "person_id": p, "foot": "R", "walkway_speed_m_s": v, "walkway_m": s,
                   "t0": 10.0 + i, "t1": 10.0 + i + s / v, "dec_ankle_m": 0.85 * kp * s, "dec_root_m": 0.85,
                   "legs_m": 0.8, "dec_leg_length_m": 0.86}
            for c, (a, b) in LINES.items():
                row[f"dec_{c}"] = 0.9 * kp * (a + b * v)
            real.append(row)
    _write(pd.DataFrame(sim), os.path.join(folder, "strides_insim.csv"))
    _write(pd.DataFrame(real), os.path.join(folder, "strides_real.csv"))


def _mpm(df, values):
    """Median over persons of each person's median of values (a column aligned with df)."""
    return float(df.assign(_v=values).groupby("person_id")["_v"].median().median())


def test_part_b_descriptive_sensitivity_band_persons_and_leg_length(tmp_path):
    """Review of F2D: the after-the-fact attainability ratio (arcs on the truth line at the decoded speed, over the
    truth line at the walkway speed), the in-sim speed band (counts and ratio), the sign of the person Pearson, the
    person stride ratio and the decoded leg lengths, each against a value computed here from the written files."""
    f = str(tmp_path / "dump")
    person_dump_files(f)
    b = H.part_b(f, SETS, LOCK, n_boot=100)
    d = b["descriptive_dumps_2026_10_05"]
    real = pd.read_csv(os.path.join(f, "strides_real.csv"), encoding="utf-8")
    sim = pd.read_csv(os.path.join(f, "strides_insim.csv"), encoding="utf-8")
    tr = real[real["person_id"].isin(TRAIN)]
    v_wk = tr["walkway_speed_m_s"].to_numpy()
    v_dec = tr["dec_ankle_m"].to_numpy() / (tr["t1"] - tr["t0"]).to_numpy()
    assert np.allclose(v_dec, 0.85 * np.array([k_of((DEV + TRAIN).index(p)) for p in tr["person_id"]]) * v_wk)
    # attainability: a decoded speed below the walkway speed gives a ratio below 1 (hip and knee lines of LINES)
    sens = d["ratio_if_arcs_followed_the_decoded_speed"]
    for j in ("hip", "knee"):
        a, bb = LINES[f"{j}_range_deg"]
        assert sens[j] == pytest.approx(_mpm(tr, (a + bb * v_dec) / (a + bb * v_wk)), abs=1e-9)
        assert sens[j] < 0.97
    # the in-sim speed band: only the real strides inside the in-sim 10th to 90th percentile of the true speed
    lo, hi = np.quantile(sim["true_speed_m_s"], [0.1, 0.9])
    inb = tr[(tr["walkway_speed_m_s"] >= lo) & (tr["walkway_speed_m_s"] <= hi)]
    band = d["in_sim_speed_band"]
    assert 0 < band["n_strides"] == len(inb) < len(tr) and band["n_persons"] == inb["person_id"].nunique()
    kb = _mpm(inb, [k_of((DEV + TRAIN).index(p)) for p in inb["person_id"]])
    assert band["hip"]["ratio_to_sim"] == pytest.approx(kb, abs=1e-9)          # rho_real 0.9 k over rho_sim 0.9
    assert band["hip"]["rho_real"] == pytest.approx(0.9 * kb, abs=1e-9)
    assert d["stride_speed_m_s"]["real_share_above_sim_p90"] == pytest.approx(float(np.mean(v_wk > hi)))
    # persons: a person with short arcs also has a short stride here, so the Pearson is +1
    ph = d["persons_hip_ratio_vs_stride_ratio"]
    assert ph["pearson"] == pytest.approx(1.0, abs=1e-9) and ph["n_persons"] == len(TRAIN)
    assert ph["median_stride_ratio"] == pytest.approx(0.85 * float(np.median([k_of(6 + i) for i in range(8)])))
    # decoded leg length: real, in-sim decoded and in-sim true kept apart
    assert d["leg_length_m"] == pytest.approx({"real_decoded_train54": 0.86, "sim_decoded": 0.84, "sim_true": 0.85})
    # the real stride at equal speed equals the in-sim one here
    assert d["real_stride_over_insim_stride_at_equal_speed"]["median_of_person_medians"] == pytest.approx(1.0)


def test_rerun_asserts_the_criteria_of_the_first_run(tmp_path, monkeypatch):
    first = str(tmp_path / "results.json")
    res = {"criteria": {"C1_band": [0.97, 1.03], "B_support_max": 0.92}, "bootstrap": {"n": 10000, "seed": 1},
           "prediction": {"report": "r", "written": "t"}}
    with open(first, "w", encoding="utf-8") as fh:
        json.dump(res, fh)
    assert H.assert_same_criteria({**res, "criteria": {"C1_band": (0.97, 1.03), "B_support_max": 0.92}}, first)
    with pytest.raises(AssertionError):
        H.assert_same_criteria({**res, "criteria": {"C1_band": [0.97, 1.03], "B_support_max": 0.95}}, first)
    rep = str(tmp_path / "rep.md")
    with open(rep, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# t\n\n## Prediction\n\ntext\n\n## Files\nlater\n")
    h = H.prediction_section_sha256(rep)
    assert h == hashlib.sha256("## Prediction\n\ntext\n".encode("utf-8")).hexdigest()
    assert "--results-out" in open(SCRIPT, encoding="utf-8").read()
    assert H.parse(["--results-out", str(tmp_path / "x")]).results_out == str(tmp_path / "x")


def _summary_stub(part_b):
    s = {"r_T": {"point": 1.0, "interval": [0.98, 1.01]}, "g_S_same": {"point": 0.85}, "s_T": {"point": 0.0,
         "interval": [-0.1, 0.1]}, "C1": "SPATIAL", "g_L": {"point": 0.75}, "in_sim": {"g_L_sim": 0.88, "g_S_sim": 0.97},
         "D_L": {"point": 0.85, "interval": None}, "g_S": {"point": 0.85}, "D_R": {"point": 0.88, "interval": None},
         "D_L_minus_D_R": {"point": -0.03, "interval": None}, "C2": "LEGS", "n_strides_r_T": 1, "n_persons_r_T": 1,
         "n_strides_legs": 1, "n_strides_base": 1}
    e = {"sets": {"dev12": s}, "checkpoint": {"path": "p", "sha256": "a"}, "events": "e", "ends_found": 1.0,
         "share_with_r_T": 1.0, "r_T_all_strides_dev12": 1.0,
         "speed_gains": {"dev12": {"base": 0.8, "model_legs": 0.7, "D_L_speed": 0.78, "D_R_speed": 0.82}},
         "insim": {"model_legs": {"speed_gain": 0.9}, "model": {"speed_gain": 0.97}},
         "r_T_descriptive_dev12": {"mean_of_person_means": 1.0, "stride_p10": 0.98, "stride_p90": 1.02,
                                   "share_within_10ms": 0.9, "strides": 10}}
    return {"part_a": {"v1_B": e}, "reading_a": {"reading": "compatible", "text": "t"}, "part_b": part_b,
            "sets": {"dev14_sha256": "a", "train54_sha256": "b"}, "moveatlas": {"head": "h"}, "run_started": "now",
            "seconds": 1.0}


def test_summary_has_no_dashes_and_the_label():
    s = {"r_T": {"point": 1.0, "interval": [0.98, 1.01]}, "g_S_same": {"point": 0.85}, "s_T": {"point": 0.0,
         "interval": [-0.1, 0.1]}, "C1": "SPATIAL", "g_L": {"point": 0.75}, "in_sim": {"g_L_sim": 0.88, "g_S_sim": 0.97},
         "D_L": {"point": 0.85, "interval": None}, "g_S": {"point": 0.85}, "D_R": {"point": 0.88, "interval": None},
         "D_L_minus_D_R": {"point": -0.03, "interval": None}, "C2": "LEGS", "n_strides_r_T": 1, "n_persons_r_T": 1,
         "n_strides_legs": 1, "n_strides_base": 1}
    e = {"sets": {"dev12": s, "dev14": s}, "checkpoint": {"path": "p", "sha256": "a"}, "events": "e", "ends_found": 1.0,
         "share_with_r_T": 1.0, "r_T_all_strides_dev12": 1.0,
         "speed_gains": {"dev12": {"base": 0.8, "model_legs": 0.7, "D_L_speed": 0.78, "D_R_speed": 0.82}},
         "insim": {"model_legs": {"speed_gain": 0.9}, "model": {"speed_gain": 0.97}},
         "r_T_descriptive_dev12": {"mean_of_person_means": 1.0, "stride_p10": 0.98, "stride_p90": 1.02,
                                   "share_within_10ms": 0.9, "strides": 10}}
    r = {"part_a": {"v1_B": e}, "reading_a": {"reading": "compatible", "text": "t"},
         "part_b": {"evaluated": False, "reason": "none"}, "sets": {"dev14_sha256": "a", "train54_sha256": "b"},
         "moveatlas": {"head": "h"}, "run_started": "now", "seconds": 1.0}
    text = H.summary_md(r)
    assert H.LABEL in text and chr(0x2014) not in text and chr(0x2013) not in text


def test_job_command_passes_queue_job_checks(capsys):
    qj = _load(os.path.join(paths.TREE, "scripts", "queue_job.py"), "queue_job_f2h3_tested")
    c = H.job_command()
    i = c.index("--")
    cmd = c[i + 1:]
    assert cmd[:2] == ["scripts/f2_H3angles.py", "--dump"]
    assert cmd[cmd.index("--out") + 1].startswith("reports/bench/f2_scale/H3angles/dump_")
    assert c[c.index("--cls") + 1] == "1" and 70 <= int(c[c.index("--priority") + 1]) <= 79
    assert c[c.index("--group") + 1] == "F2" and cmd[cmd.index("--memory-gb") + 1] == "12"
    assert qj.main(c[1:i] + ["--dry-run"] + c[i:]) == 0
    capsys.readouterr()


# ---------------------------------------------------------------- --dump with a tiny model
def _record(person, task, seed, lockbox=False):
    import model_fixture as MF
    st = MF.imu_stream(12.0, sites=tuple(range(13)), seed=seed)
    sig = np.array(st.values, copy=True)
    sig[..., 6:9] = 0.3
    T = sig.shape[0]
    return {"source": "weargait_pd", "subject_ns": person.replace("weargait:", "weargait_pd:"), "person_id": person,
            "key": task, "fps": 100.0, "signal": sig, "channel_mask": np.ones((13, 9), bool),
            "location_ids": np.arange(13), "sample_valid": np.ones((T, 13), bool), "acc_content": "with_gravity",
            "lockbox": lockbox, "tier": "TRAIN"}


def _targets(persons):
    tr, ics, st, ps = [], [], [], []
    for p in persons:
        tid = f"{p.split(':')[1]}_SelfPace"
        tr.append((tid, p, "SelfPace", 1200))
        L, R = [2.0, 3.1, 4.2, 5.3, 6.4], [2.55, 3.65, 4.75, 5.85]
        ics += [(tid, p, "SelfPace", "L", x) for x in L] + [(tid, p, "SelfPace", "R", x) for x in R]
        st += [(tid, p, "SelfPace", "0", "L", a, b, 1.2) for a, b in zip(L[:-1], L[1:])]
        st += [(tid, p, "SelfPace", "0", "R", a, b, 1.2) for a, b in zip(R[:-1], R[1:])]
        ps.append((tid, p, "SelfPace", "0", 2.0, 6.4, 9, 1.1))
    walk = pd.DataFrame(ics, columns=["trial_id", "person_id", "task", "foot", "t_s"])
    return g3.Targets(tuple(sorted(persons)), False,
                      pd.DataFrame(tr, columns=["trial_id", "person_id", "task", "samples"]), walk,
                      walk.assign(t_raw_s=walk["t_s"], offset_ms=0.0),
                      pd.DataFrame(st, columns=["trial_id", "person_id", "task", "pass_id", "foot", "t0", "t1",
                                                "length_m"]),
                      pd.DataFrame(ps, columns=["trial_id", "person_id", "task", "pass_id", "t0", "t1", "n_footfalls",
                                                "speed_m_s"]),
                      pd.DataFrame(columns=["trial_id", "offset_ms", "n_matched"]), {"lockbox_rows": 0})


@pytest.fixture()
def tiny(tmp_path, monkeypatch):
    if not HAVE_MA:
        pytest.skip("SMPL+H model not found")
    import torch

    from mdfm.data import sources
    from mdfm.eval import infer as I
    from mdfm.model import mfm as M
    torch.manual_seed(7)
    model = M.MFM(M.TINY)
    model.eval()
    model.requires_grad_(False)
    ck = str(tmp_path / "run_tiny" / "ckpt_B_000010.pt")
    loaded = I.Loaded(model=model, path=ck, name="run_tiny_ckpt_B_000010", sha256="d" * 64, weights="ema", step=10,
                      phase="B", phase_steps=(5, 5), config_json="{}", config_sha256="e" * 64,
                      persons={"weargait_pd": ["weargait:TR01"]}, persons_sha256={"all": "f" * 64}, hashes={},
                      drift=[])
    train, dev = ["weargait:TR01", "weargait:TR02"], ["weargait:DV01"]
    state = {"lockbox_record": False}

    def load_targets(persons, allow_lockbox=False, read_row=None, tasks=g3.TASKS):
        assert set(persons) == set(train + dev) and not allow_lockbox
        return _targets(persons)

    def imu_records(persons, trials, allow_lockbox=False, read_row=None, tasks=g3.TASKS):
        assert not allow_lockbox
        for i, r in enumerate(trials.itertuples()):
            yield r.trial_id, _record(r.person_id, r.task, i, lockbox=state["lockbox_record"] and i == 1)

    real = g3.controls
    monkeypatch.setattr(g3, "controls", lambda fold: train if fold == "train" else dev if fold == "dev" else real(fold))
    monkeypatch.setattr(g3, "lockbox_persons", lambda: LOCK)
    monkeypatch.setattr(g3, "load_targets", load_targets)
    monkeypatch.setattr(g3, "imu_records", imu_records)
    monkeypatch.setattr(GG, "N_DEV", 1)
    monkeypatch.setattr(GG, "N_TRAIN", 2)
    monkeypatch.setattr(GG, "EXCLUDED_FROM_12", ())
    monkeypatch.setattr(H, "F1_RESULTS", str(tmp_path / "absent.json"))
    monkeypatch.setattr(sources, "demographic", lambda source, ns: {"sex": "F", "age": 64.0})
    monkeypatch.setattr(I, "load_checkpoint", lambda path, weights="ema", device="cpu", cfg=None: loaded)
    monkeypatch.setattr(GG, "stamp_moveatlas", lambda: {"head": "test"})
    out = str(tmp_path / "dump")
    argv = ["--dump", "--checkpoint", ck, "--device", "cpu", "--refine-steps", "1", "--points", "16",
            "--batch-windows", "4", "--out", out]
    return {"argv": argv, "train": train, "dev": dev, "state": state, "out": out, "loaded": loaded}


def test_dump_real_with_a_tiny_model(tiny):
    out = H.dump(H.parse(tiny["argv"] + ["--no-insim"]), log=lambda *a: None)
    df = pd.read_csv(os.path.join(out, "strides_real.csv"), encoding="utf-8")
    assert set(df["person_id"]) == set(tiny["train"] + tiny["dev"]) and len(df) == 3 * 7
    assert set(df.loc[df["person_id"].isin(tiny["dev"]), "set"]) == {"dev14"}
    for k in H.FEATURES:
        assert f"dec_{k}" in df.columns
    assert np.isfinite(df["dec_hip_range_deg"]).all() and np.isfinite(df["legs_m"]).all()
    with open(os.path.join(out, "dump.json"), encoding="utf-8") as f:
        doc = json.load(f)
    assert doc["what"] == H.LABEL and doc["real"]["trials"] == 3 and "insim" not in doc


def test_dump_stops_on_a_lockbox_record(tiny):
    tiny["state"]["lockbox_record"] = True
    with pytest.raises(AssertionError, match="lockbox"):
        H.dump(H.parse(tiny["argv"] + ["--no-insim"]), log=lambda *a: None)
    assert not os.path.exists(os.path.join(tiny["out"], "strides_real.csv"))


def test_dump_insim_one_take_with_a_tiny_model(tiny):
    try:
        from mdfm.data import body_store
        body_store.BodyStore(split="dev")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"dev body store not available: {e}")
    rows, info = H.dump_insim(tiny["loaded"].model, set(), "cpu", refine_steps=1, points=16, batch_windows=4, limit=1,
                              log=lambda *a: None)
    df = pd.DataFrame(rows)
    assert info["takes"] == 1 and len(df) > 0
    assert np.isfinite(df["true_hip_range_deg"]).any() and np.isfinite(df["dec_knee_range_deg"]).any()
    assert set(df["scored"]) <= {True, False}
