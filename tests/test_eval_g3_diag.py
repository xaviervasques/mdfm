"""Tests of scripts/eval_g3_diag.py (task T04 of the weekend 2026-10-01: G3 dev diagnostics, the learning-curve table
and the speed and cadence quantiles), on CPU, without a record or a checkpoint file:

- the record transforms of the ablations: cut_sites keeps the sensors of the kept sites (None when nothing is cut,
  EMPTY when nothing is left), the sites of WHC003 and of the families are the canonical site ids, swap_shanks
  exchanges shank_L and shank_R (data, masks, validity) and undoes itself, lowpass removes 30 Hz and keeps 2 Hz on
  the acc and gyro channels of each valid run (sosfiltfilt at 12 Hz, order 4) and leaves the magnetometer, the
  invalid samples and short runs alone; variants holds base alone by default and every model row with --ablations;
- the readouts of a stitched trial body: legs_speed_body gives the values of eval_gait_insim.legs_speed on the window
  bodies; heel_strike_ics finds the heel strikes of a walking body held in a different frame per window, once each,
  also when windows decode the phase a few frames apart around their ownership boundaries;
- scoring on synthetic targets and predictions: the per-person values and the all14 means equal the harness's
  (g3.evaluate), the subsets leave out WHC003 and WHC026, the SelfPace stride MAE, the hurried speed bias and the
  signed biases have the signs of prediction minus reference, the Spearman of a speed error that falls with speed is
  -1 (no interval for one person); the G10 transfer readings at 1.3 and 1.6;
- the rerun check: within the tolerances it passes, beyond them it states that a code path diverged and names the
  person (a person without a matched stride in both runs agrees, on one side only diverges), another checkpoint
  file or raw weights are not evaluated;
- --curve: the folders of the run (not of a run whose name extends it), sorted by phase and step, dev before curve,
  raw weights apart; --quantiles: numpy quantiles, shares, no person-level value;
- the full default and --ablations runs (diag) with a tiny model (mdfm.model.mfm.TINY) on records made of
  tests/model_fixture.py streams: diag.json holds every row key of the schema (rows not run are null and listed),
  the subsets and per-person records, the code hash of _provenance.json covers its code_files, and
  scripts/weekend_branch.py reads the base row, the SelfPace per-person values, the hurried speed bias, the Spearman
  and the heel-strike block from it.
"""
import importlib.util
import json
import math
import os
import types

import numpy as np
import pandas as pd
import pytest

from mdfm import paths
from mdfm.gates import g3

import model_fixture as MF

TREE = paths.TREE
SCRIPT = os.path.join(TREE, "scripts", "eval_g3_diag.py")
HAVE_MA = os.path.exists(paths.smplh_model("neutral"))


def _load(path, name):
    old = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    try:                                                    # the scripts set the variable at import
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        if old is None:
            os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        else:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = old
    return mod


S = _load(SCRIPT, "eval_g3_diag_tested")


@pytest.fixture(autouse=True)
def keep_alloc_conf():
    old = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    yield
    if old is None:
        os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    else:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = old


@pytest.fixture()
def no_stamp(monkeypatch):
    monkeypatch.setattr(S, "stamp_moveatlas", lambda: {"head": "test", "note": "no git in the tests"})


# ---------------------------------------------------------------- records
def record(sites=tuple(range(13)), duration=12.0, seed=0, person="weargait:HC103", task="SelfPace"):
    st = MF.imu_stream(duration, sites=sites, seed=seed)
    sig = np.array(st.values, copy=True)
    sig[..., 6:9] = 0.3
    T = sig.shape[0]
    return {"source": "weargait_pd", "subject_ns": person.replace("weargait:", "weargait_pd:"), "person_id": person,
            "key": task, "fps": 100.0, "signal": sig, "channel_mask": np.ones((len(sites), 9), bool),
            "location_ids": np.array(sites), "sample_valid": np.ones((T, len(sites)), bool),
            "acc_content": "with_gravity"}


def test_sites_are_the_canonical_ids():
    from mdfm.factory.imu_render import SITES
    assert list(S.XSENS_SITES) == SITES[:13]
    assert sorted(S.site_ids(S.WHC003_SITES)) == [0, 1, 2, 3, 4, 11]
    assert sorted(S.site_ids(S.FAMILIES["family_lower_back"])) == [2]
    assert sorted(S.site_ids(S.FAMILIES["family_wrists"])) == [3, 4]
    assert sorted(S.site_ids(S.FAMILIES["family_legs"])) == [5, 6, 7, 8, 9, 10, 11, 12]


def test_cut_sites():
    rec = record()
    out = S.cut_sites(rec, S.site_ids(S.WHC003_SITES))
    assert out["signal"].shape[1] == 6 and list(out["location_ids"]) == [0, 1, 2, 3, 4, 11]
    np.testing.assert_array_equal(out["signal"][:, 5], rec["signal"][:, 11])
    assert out["channel_mask"].shape == (6, 9) and out["sample_valid"].shape == (rec["signal"].shape[0], 6)
    assert rec["signal"].shape[1] == 13                                     # the input is left as it was
    six = record(sites=(0, 1, 2, 3, 4, 11))
    assert S.cut_sites(six, S.site_ids(S.WHC003_SITES)) is None              # nothing cut: the base is reused
    assert S.cut_sites(six, S.site_ids(S.FAMILIES["family_lower_back"]))["signal"].shape[1] == 1
    assert S.cut_sites(record(sites=(0, 1)), S.site_ids(S.FAMILIES["family_legs"])) == S.EMPTY


def test_swap_shanks():
    rec = record()
    rec["sample_valid"][:10, 7] = False
    rec["channel_mask"][8, 0] = False
    out = S.swap_shanks(rec)
    np.testing.assert_array_equal(out["signal"][:, 7], rec["signal"][:, 8])
    np.testing.assert_array_equal(out["signal"][:, 8], rec["signal"][:, 7])
    assert not out["sample_valid"][:10, 8].any() and out["sample_valid"][:10, 7].all()
    assert not out["channel_mask"][7, 0] and out["channel_mask"][8, 0]
    np.testing.assert_array_equal(out["location_ids"], rec["location_ids"])
    back = S.swap_shanks(out)
    for k in ("signal", "channel_mask", "sample_valid"):
        np.testing.assert_array_equal(back[k], rec[k])
    assert S.swap_shanks(record(sites=(0, 1, 7))) is None


def test_lowpass():
    from scipy.signal import butter, sosfiltfilt

    from mdfm.data import window_store as WS
    T = 1500
    t = np.arange(T) / 100.0
    rec = record(duration=T / 100.0)
    slow, fast = np.sin(2 * np.pi * 2.0 * t), np.sin(2 * np.pi * 30.0 * t)
    rec["signal"][:, :, :6] = (slow + fast)[:, None, None]
    rec["sample_valid"][700:710, 3] = False          # sensor 3: two runs of 700 and 790 samples
    rec["sample_valid"][:, 4] = False
    rec["sample_valid"][100:110, 4] = True           # sensor 4: one run of 10 samples, not above padlen (15)
    rec["signal"][705, 3, :6] = 99.0
    out = S.lowpass(rec)
    assert out["signal"].dtype == rec["signal"].dtype
    x = out["signal"][:, 0, 0].astype(np.float64)
    mid = slice(200, 1300)
    fit = np.linalg.lstsq(np.stack([slow[mid], fast[mid], np.cos(2 * np.pi * 30 * t[mid]),
                                    np.cos(2 * np.pi * 2 * t[mid])], 1), x[mid], rcond=None)[0]
    assert abs(fit[0] - 1.0) < 0.02 and math.hypot(fit[1], fit[2]) < 0.02
    sos = butter(4, 12.0, btype="lowpass", fs=100.0, output="sos")
    pad = WS._padlen(sos, 100.0, 12.0)
    ref = sosfiltfilt(sos, (slow + fast)[:700], axis=0, padlen=pad)
    np.testing.assert_allclose(out["signal"][:700, 3, 0], ref, atol=1e-5)
    assert (out["signal"][705, 3, :6] == 99.0).all()                       # an invalid sample is left as it was
    np.testing.assert_array_equal(out["signal"][:, 4], rec["signal"][:, 4])  # a run under padlen is left
    np.testing.assert_array_equal(out["signal"][..., 6:9], rec["signal"][..., 6:9])


def test_variants():
    assert list(S.variants(False)) == ["base"]
    v = S.variants(True)
    assert set(v) == set(S.ROWS) - {"model_legs"}
    assert v["refine0"].refine == 0 and v["refine60"].refine == 60 and v["base"].transform is None
    rec = record()
    assert v["swap_whc026"].transform("weargait:HC103", rec) is None
    np.testing.assert_array_equal(v["swap_whc026"].transform("weargait:WHC026", rec)["signal"][:, 7],
                                  rec["signal"][:, 8])
    assert v["family_wrists"].transform("weargait:HC103", rec)["signal"].shape[1] == 2
    assert v["sites6"].transform("weargait:HC103", rec)["signal"].shape[1] == 6


def test_parse_modes():
    assert S.parse(["--quantiles"]).quantiles
    assert S.parse(["--curve", "r1"]).curve == "r1"
    a = S.parse(["--checkpoint", "x.pt", "--ablations"])
    assert a.ablations and a.batch_windows == 16 and a.refine_steps == 20
    for bad in ([], ["--quantiles", "--curve", "r"], ["--ablations", "--quantiles"]):
        with pytest.raises(SystemExit):
            S.parse(bad)


def test_default_out():
    ld = types.SimpleNamespace(name="run_ckpt_B_032044", weights="ema")
    assert S.default_out(ld, False, tree="T").replace("\\", "/") == "T/reports/gates/g3/diag_run_ckpt_B_032044"
    assert S.default_out(ld, True, tree="T").replace("\\", "/").endswith("diag_run_ckpt_B_032044_ablations")
    raw = types.SimpleNamespace(name="run_ckpt_B_032044", weights="model")
    assert os.path.basename(S.default_out(raw, False, tree="T")) == "diag_model_run_ckpt_B_032044"


# ---------------------------------------------------------------- readouts of a stitched body
def _windows_of(J, contact, starts, n):
    return [types.SimpleNamespace(joints=J[s:s + n], contact=contact[s:s + n]) for s in starts]


def test_legs_speed_body_equals_the_insim_readout():
    rng = np.random.default_rng(3)
    T, n, starts = 1700, 1000, [0, 500, 700]
    J = np.cumsum(0.01 * rng.standard_normal((T, 22, 3)), axis=0)
    wins = _windows_of(J, np.zeros((T, 2)), starts, n)
    for k, w in enumerate(wins):                         # each window in its own frame, with its own contacts
        w.joints = w.joints + np.array([0.3 * k, 0.0, -0.2 * k])
        w.contact = (rng.random((n, 2)) > 0.4).astype(np.float64)
    tb = g3.stitched_body("T_SelfPace", [w.joints for w in wins], [w.contact for w in wins], None, starts)
    got = S.legs_speed_body(tb)
    want = S.INSIM().legs_speed(wins, starts)
    assert got.shape == (T,)
    np.testing.assert_allclose(got, want, equal_nan=True)


def walking_body(T=1700, v=1.2, f=0.9, amp=0.35, phase_s=0.0):
    t = np.arange(T) / 100.0
    J = np.zeros((T, 22, 3))
    J[:, :, 0] = (v * t)[:, None]
    J[:, :, 1] = 0.9
    J[:, 7, 0] += amp * np.sin(2 * np.pi * f * (t + phase_s))
    J[:, 8, 0] += amp * np.sin(2 * np.pi * f * (t + phase_s) + np.pi)
    return J


def _rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def test_heel_strikes_in_window_frames():
    from mdfm.readouts import gait
    T, n, starts = 1700, 1000, [0, 500, 700]
    J = walking_body(T)
    wins = _windows_of(J, np.zeros((T, 2)), starts, n)
    for k, w in enumerate(wins):                         # a different frame per window: turned and moved
        w.joints = w.joints @ _rot_y(0.7 * k).T + np.array([5.0 * k, 0.0, -3.0 * k])
    tb = g3.stitched_body("T_SelfPace", [w.joints for w in wins], [w.contact for w in wins], None, starts)
    got = S.heel_strike_ics(tb)
    want = gait.heel_strikes(J)
    for foot in gait.FEET:
        assert len(got[foot]) == len(want[foot]) > 10
        np.testing.assert_allclose(got[foot], want[foot], atol=0.011)
        assert np.all(np.diff(got[foot]) >= gait.HS_DISTANCE_S - 1e-9)


def test_heel_strikes_at_window_boundaries():
    """Windows that decode the gait phase up to 30 ms apart: a strike near the ownership boundary of two windows can
    lie on the other window's side in both; it is still found once (owning-window-only stitching loses 27 of these
    strikes here)."""
    from mdfm.readouts import gait
    rng = np.random.default_rng(0)
    T, n, hop = 6000, 1000, 500
    starts = list(range(0, T - n + 1, hop))
    truth = gait.heel_strikes(walking_body(T))
    inner = lambda x: x[(x > 1.0) & (x < T / 100.0 - 1.0)]  # noqa: E731
    lost = doubled = extra = 0
    for _ in range(10):
        shifts = rng.uniform(-0.03, 0.03, size=len(starts))
        wins = [walking_body(T, phase_s=float(sh))[s:s + n] for s, sh in zip(starts, shifts)]
        tb = g3.stitched_body("T_SelfPace", wins, [np.zeros((n, 2))] * len(wins), None, starts)
        got = S.heel_strike_ics(tb)
        for foot in gait.FEET:
            tt, g = inner(truth[foot]), inner(got[foot])
            k = np.array([np.sum(np.abs(g - x) < 0.2) for x in tt])
            lost += int((k == 0).sum())
            doubled += int((k > 1).sum())
            extra += int(sum(np.min(np.abs(tt - y)) >= 0.2 for y in g))
            assert np.all(np.diff(got[foot]) >= gait.HS_DISTANCE_S - 1e-9)
    assert (lost, doubled, extra) == (0, 0, 0)


# ---------------------------------------------------------------- scoring on synthetic targets
PERSONS = ("weargait:HC103", "weargait:WHC003", "weargait:WHC026")


def targets(persons=PERSONS, tasks=("SelfPace", "HurriedPace"), speeds=None):
    """Two trials per person; L ICs at 2.0, 3.1, 4.2, 5.3 s, R at 2.55, 3.65, 4.75 s; strides of 1.2 m; one pass of
    2.0 to 5.3 s at the person's speed of the task."""
    speeds = speeds or {}
    tr, ics, st, ps = [], [], [], []
    for i, p in enumerate(persons):
        for task in tasks:
            tid = f"{S.short_id(p)}_{task}"
            tr.append((tid, p, task, 1200))
            L, R = [2.0, 3.1, 4.2, 5.3], [2.55, 3.65, 4.75]
            ics += [(tid, p, task, "L", x) for x in L] + [(tid, p, task, "R", x) for x in R]
            st += [(tid, p, task, "0", "L", a, b, 1.2) for a, b in zip(L[:-1], L[1:])]
            st += [(tid, p, task, "0", "R", a, b, 1.2) for a, b in zip(R[:-1], R[1:])]
            ps.append((tid, p, task, "0", 2.0, 5.3, 7, speeds.get((p, task), 1.0 + 0.4 * (task == "HurriedPace")
                                                                  + 0.1 * i)))
    walk = pd.DataFrame(ics, columns=["trial_id", "person_id", "task", "foot", "t_s"])
    ins = walk.copy()
    ins["t_raw_s"], ins["offset_ms"] = ins["t_s"], 0.0
    return g3.Targets(tuple(sorted(persons)), False,
                      pd.DataFrame(tr, columns=["trial_id", "person_id", "task", "samples"]), walk, ins,
                      pd.DataFrame(st, columns=["trial_id", "person_id", "task", "pass_id", "foot", "t0", "t1",
                                                "length_m"]),
                      pd.DataFrame(ps, columns=["trial_id", "person_id", "task", "pass_id", "t0", "t1", "n_footfalls",
                                                "speed_m_s"]),
                      pd.DataFrame(columns=["trial_id", "offset_ms", "n_matched"]), {"lockbox_rows": 0})


@pytest.fixture()
def dev_is(monkeypatch):
    """g3.controls('dev') answers the synthetic persons (g3.fit_offsets fits on dev persons only)."""
    real = g3.controls

    def fake(fold):
        return sorted(PERSONS) if fold == "dev" else real(fold) if fold != "train" else ["weargait:TRAIN01"]
    monkeypatch.setattr(g3, "controls", fake)


def predictions_with_errors(tg, stride_err, speed_err, ic_shift=0.02):
    """Predictions equal to the targets plus a stride error (m) and a speed error (m/s) per (person, task)."""
    st = tg.strides.copy()
    st["length_m"] = [r.length_m + stride_err(r.person_id, r.task) for r in st.itertuples()]
    sp = tg.passes.copy()
    sp["speed_m_s"] = [r.speed_m_s + speed_err(r.person_id, r.task, r.speed_m_s) for r in sp.itertuples()]
    ics = tg.walkway_ic[["trial_id", "foot", "t_s"]].copy()
    ics["t_s"] += ic_shift
    return g3.Predictions("base", ics, st[["trial_id", "foot", "t0", "t1", "length_m"]],
                          sp[["trial_id", "pass_id", "speed_m_s"]])


def test_row_entry_equals_the_harness_and_reads_the_subsets(dev_is):
    tg = targets()
    serr = {"weargait:HC103": 0.05, "weargait:WHC003": -0.20, "weargait:WHC026": 0.10}
    pred = predictions_with_errors(tg, lambda p, task: serr[p] * (2.0 if task == "HurriedPace" else 1.0),
                                   lambda p, task, v: -0.1 if task == "HurriedPace" else 0.02)
    summ, tables, offsets, unfitted = S.score(pred, tg)
    assert not unfitted and offsets[("base", "walkway")][0] == pytest.approx(0.02)
    row, pt = S.row_entry("base", summ, tables, list(tg.persons))
    for p in tg.persons:
        assert row["all14"]["per_person"][p]["stride_cm"] == pytest.approx(summ["stride_mae_cm"]["per_person"][p])
        assert row["all14"]["per_person"][p]["speed_ms"] == pytest.approx(summ["speed_mae_m_s"]["per_person"][p])
    assert row["all14"]["stride_mae_cm"] == pytest.approx(summ["stride_mae_cm"]["point"])
    assert row["all14"]["speed_mae_ms"] == pytest.approx(summ["speed_mae_m_s"]["point"])
    assert row["all14"]["stride_mape_pct"] == pytest.approx(summ["stride_mape_pct"]["point"])
    assert row["all14"]["n_persons"] == 3 and row["no_whc003"]["n_persons"] == 2 and row[S.SUB12]["n_persons"] == 1
    assert set(row[S.SUB12]["per_person"]) == {"weargait:HC103"}
    assert row[S.SUB12]["stride_mae_cm"] == pytest.approx(1.5 * 5.0)       # 5 cm SelfPace, 10 cm HurriedPace
    sp = row["selfpace_stride_mae_cm"]
    assert sp[S.SUB12]["point"] == pytest.approx(5.0) and sp[S.SUB12]["per_person"] == {"weargait:HC103": pytest.approx(5.0)}
    assert sp["all14"]["point"] == pytest.approx((5.0 + 20.0 + 10.0) / 3)
    hb = row["hurried_speed_bias_ms"]
    assert hb[S.SUB12]["point"] == pytest.approx(-0.1) and hb["all14"]["n_persons"] == 3
    sb = row["signed_bias"]
    assert sb["SelfPace"]["stride_cm"]["per_person"]["weargait:WHC003"] == pytest.approx(-20.0)
    assert sb["HurriedPace"]["speed_ms"]["subsets"]["all14"]["point"] == pytest.approx(-0.1)
    assert sb["SelfPace"]["speed_ms"]["subsets"]["no_whc003"]["point"] == pytest.approx(0.02)
    blk = S.ic_block(summ, tables, offsets, unfitted, None, "base")
    assert blk["median_ms"] == pytest.approx(0.0, abs=1e-6) and blk["offset_ms"] == pytest.approx(20.0)
    assert blk["insole"]["offset_ms"] == pytest.approx(20.0)


def test_spearman_of_an_error_that_falls_with_speed(dev_is):
    speeds = {(p, t): 0.8 + 0.2 * i + 0.5 * (t == "HurriedPace") for i, p in enumerate(PERSONS)
              for t in ("SelfPace", "HurriedPace")}
    tg = targets(speeds=speeds)
    pred = predictions_with_errors(tg, lambda p, task: 0.0, lambda p, task, v: -0.5 * (v - 0.8))
    summ, tables, _, _ = S.score(pred, tg)
    r = S.spearman(tables["speeds"], "all14", n_boot=200)
    assert r["rho"] == pytest.approx(-1.0) and r["n_passes"] == 6 and r["n_persons"] == 3
    assert r["interval"] is not None and r["interval"][1] <= -0.99
    r12 = S.spearman(tables["speeds"], S.SUB12, n_boot=50)
    assert r12["n_persons"] == 1 and r12["n_passes"] == 2
    one = S.spearman(pd.concat([tables["speeds"]] * 2, ignore_index=True), S.SUB12, n_boot=50)
    assert one["rho"] is not None and one["n_persons"] == 1 and one["interval"] is None   # no bootstrap of one person


def test_transfer_readings():
    base = {s: {"stride_mae_cm": 10.0} for s in S.SUBSETS}
    for v, reading in ((13.0, "at most 1.3"), (16.0, "between 1.3 and 1.6"), (16.5, "above 1.6"), (None, "n/a")):
        row = {s: {"stride_mae_cm": v} for s in S.SUBSETS}
        t = S.transfer(row, base)
        assert t[S.SUB12]["reading"] == reading and t[S.SUB12]["label"] == S.PREVIEW
        if v is not None:
            assert t["all14"]["ratio"] == pytest.approx(v / 10.0)


# ---------------------------------------------------------------- rerun check
def _reference(tree, name, sha, stride, speed, pp, weights="ema"):
    d = os.path.join(tree, "reports", "gates", "g3_sim", f"dev_{name}")
    os.makedirs(d)
    doc = {"checkpoint": {"name": name, "sha256": sha, "weights": weights},
           "inference": {"refine_steps": 20, "points_per_stream": 256, "batch_windows": 32, "seed": 20260929,
                         "shape_pooling": "per trial", "weights": weights},
           "lines": {"dev": {"model": {"stride_mae_cm": {"point": stride}, "speed_mae_m_s": {"point": speed}}}}}
    with open(os.path.join(d, "g3_sim.json"), "w", encoding="utf-8") as f:
        json.dump(doc, f)
    rows = [{"row": "dev", "predictor": "model", "line": "stride_mae_cm", "person_id": p, "value": v} for p, v in pp.items()]
    rows.append({"row": "dev", "predictor": "walkway", "line": "stride_mae_cm", "person_id": "weargait:HC103", "value": 0})
    pd.DataFrame(rows).to_csv(os.path.join(d, "per_person.csv"), index=False)


def _base_row(pp, speed):
    return {"all14": {"stride_mae_cm": float(np.mean([v for v in pp.values() if v is not None])), "speed_mae_ms": speed,
                      "per_person": {p: {"stride_cm": v} for p, v in pp.items()}}}


def test_rerun_check(tmp_path):
    tree = str(tmp_path)
    pp = {"weargait:HC103": 20.0, "weargait:WHC003": 40.0}
    _reference(tree, "r_ckpt_A_010681", "a" * 64, 30.0, 0.28, pp)
    inf = {"refine_steps": 20, "points_per_stream": 256, "batch_windows": 16, "seed": 20260929,
           "shape_pooling": "per trial", "weights": "ema"}
    ok = S.rerun_check(_base_row({"weargait:HC103": 20.4, "weargait:WHC003": 40.3}, 0.285), "r_ckpt_A_010681", "a" * 64,
                       "ema", inf, tree)
    assert ok["evaluated"] and ok["passed"] and ok["reading"].startswith("the base row matches")
    assert ok["inference_differences"] == {"batch_windows": [32, 16]}
    assert ok["max_person_stride_difference_cm"] == pytest.approx(0.4)
    bad = S.rerun_check(_base_row({"weargait:HC103": 21.5, "weargait:WHC003": 39.4}, 0.30), "r_ckpt_A_010681", "a" * 64,
                        "ema", inf, tree)
    assert bad["evaluated"] and not bad["passed"] and bad["reading"].startswith("a code path diverged")
    assert "speed MAE" in bad["reading"] and "HC103 (1.50)" in bad["reading"] and "WHC003" not in bad["reading"]
    # a person without a matched stride in both runs (absent from the reference's per_person.csv) agrees
    both = S.rerun_check(_base_row({"weargait:HC103": 20.2, "weargait:WHC003": 40.1, "weargait:WHC026": None}, 0.28),
                         "r_ckpt_A_010681", "a" * 64, "ema", inf, tree)
    assert both["passed"] and both["persons_without_stride_in_both"] == ["weargait:WHC026"]
    _reference(tree, "r2_ckpt_A_010681", "a" * 64, 30.0, 0.28, {**pp, "weargait:WHC026": 25.0})
    one = S.rerun_check(_base_row({"weargait:HC103": 20.2, "weargait:WHC003": 40.1, "weargait:WHC026": None}, 0.28),
                        "r2_ckpt_A_010681", "a" * 64, "ema", inf, tree)
    assert not one["passed"] and "WHC026 (n/a)" in one["reading"]                  # missing on one side only
    assert not S.rerun_check(_base_row(pp, 0.28), "r_ckpt_A_010681", "b" * 64, "ema", inf, tree)["evaluated"]
    assert S.rerun_check(_base_row(pp, 0.28), "r_ckpt_A_010681", "a" * 64, "model", inf, tree)["reading"].startswith(
        "not applicable")
    assert S.rerun_check(_base_row(pp, 0.28), "other", "a" * 64, "ema", inf, tree)["reading"].startswith("not evaluated")


# ---------------------------------------------------------------- --curve
def _g3json(tree, folder, sub, name, stem, weights="ema", stride=10.0, speed=0.2, ic=25.0, mm=20.0, md=10.0):
    d = os.path.join(tree, "reports", "gates", folder, sub)
    os.makedirs(d)
    doc = {"checkpoint": {"name": name, "weights": weights, "sha256": "c" * 64},
           "lines": {"dev": {"model": {"stride_mae_cm": {"point": stride, "n_persons": 14},
                                       "speed_mae_m_s": {"point": speed}, "stride_mape_pct": {"point": 12.0},
                                       "ic_median_ms_walkway": {"point": ic}}}},
           "offsets_ms": {"model|walkway": {"offset_ms": 80.0, "n_matched": 900}},
           "results_dev": [{"line": "ic_median_ms_walkway",
                            "extra": {"margin_model_ms": mm, "margin_detector_ms": md, "margin_ok": False}}]}
    with open(os.path.join(d, f"{stem}.json"), "w", encoding="utf-8") as f:
        json.dump(doc, f)


def test_curve(tmp_path, no_stamp):
    tree = str(tmp_path)
    _g3json(tree, "g3", "curve_r1_ckpt_B_004000", "r1_ckpt_B_004000", "g3_dev", stride=20.0)
    _g3json(tree, "g3", "curve_r1_ckpt_B_008000", "r1_ckpt_B_008000", "g3_dev", stride=99.0)
    _g3json(tree, "g3", "dev_r1_ckpt_B_008000", "r1_ckpt_B_008000", "g3_dev", stride=15.0)
    _g3json(tree, "g3", "dev_r1_ckpt_B_008000_model", "r1_ckpt_B_008000", "g3_dev", weights="model", stride=16.0)
    _g3json(tree, "g3", "curve_r1_ckpt_A_004000", "r1_ckpt_A_004000", "g3_sim", stride=40.0)
    _g3json(tree, "g3_sim", "dev_r1_ckpt_A_010681", "r1_ckpt_A_010681", "g3_sim", stride=28.8, ic=None)
    _g3json(tree, "g3", "dev_r1_full_ckpt_B_004000", "r1_full_ckpt_B_004000", "g3_dev", stride=1.0)  # another run
    os.makedirs(os.path.join(tree, "reports", "gates", "g3", "curve_r1_ckpt_B_012000"))           # no json yet
    out = S.curve("r1", tree=tree, log=lambda *a: None)
    with open(os.path.join(out, "curve.json"), encoding="utf-8") as f:
        c = json.load(f)
    assert os.path.basename(out) == "curve_r1"
    assert [(e["phase"], e["step"]) for e in c["steps"]] == [("A", 4000), ("A", 10681), ("B", 4000), ("B", 8000)]
    assert [e["stride_mae_cm"] for e in c["steps"]] == [40.0, 28.8, 20.0, 15.0]
    assert c["steps"][3]["source"] == "reports/gates/g3/dev_r1_ckpt_B_008000"
    assert [e["source"] for e in c["duplicates"]] == ["reports/gates/g3/curve_r1_ckpt_B_008000"]
    assert [e["stride_mae_cm"] for e in c["raw_weights"]] == [16.0]
    e = c["steps"][0]
    assert e["margin_ratio"] == pytest.approx(2.0) and e["ic_offset_ms"] == 80.0 and c["steps"][1]["ic_median_ms_walkway"] is None
    with open(os.path.join(out, "curve.md"), encoding="utf-8") as f:
        md = f.read()
    assert "| B | 8,000 | 15.00 |" in md and "\u2014" not in md
    assert os.path.exists(os.path.join(out, "_provenance.json"))


# ---------------------------------------------------------------- --quantiles
def test_quantile_tables_are_population_quantiles():
    rng = np.random.default_rng(0)
    amass = [{"subset": "KIT", "speed_m_s": float(v), "cadence_steps_min": float(c)}
             for v, c in zip(rng.uniform(0.5, 1.5, 200), rng.uniform(80, 130, 200))]
    amass.append({"subset": "KIT", "speed_m_s": None, "cadence_steps_min": 100.0})
    wg = [{"task": t, "person_id": f"weargait:P{i}", "speed_m_s": 1.0 + 0.4 * (t == "HurriedPace") + 0.01 * i,
           "cadence_steps_min": 105.0 + 15 * (t == "HurriedPace") + 0.1 * i}
          for i in range(30) for t in S.TASKS]
    q = S.quantile_tables(amass, wg)
    sp = [a["speed_m_s"] for a in amass if a["speed_m_s"] is not None]
    assert q["amass_train"]["speed_m_s"]["n"] == 200 and q["amass_train"]["cadence_steps_min"]["n"] == 201
    assert q["amass_train"]["speed_m_s"]["quantiles"]["0.50"] == pytest.approx(float(np.quantile(sp, 0.5)))
    w = q["weargait_train_walkway"]["HurriedPace"]
    assert w["n_trials"] == 30 and w["n_persons"] == 30 and q["weargait_train_walkway"]["both"]["n_trials"] == 60
    hq = w["speed_m_s"]["quantiles"]["0.50"]
    assert q["comparison"]["HurriedPace"]["speed_m_s"]["amass_share_at_or_below"]["0.50"] == pytest.approx(
        np.mean([x <= hq for x in sp]))
    assert "weargait:P1" not in json.dumps(q)                                # no person-level value


def test_amass_train_walks_are_train_archives():
    if not os.path.exists(S.AMASS_WALKS_CSV):
        pytest.skip("reports/amass_walks/takes.csv not found")
    rows, splits = S.amass_train_walks()
    assert set(splits.values()) == {"train"} and len(rows) > 0


# ---------------------------------------------------------------- the full run with a tiny model
TRIALS = (("weargait:HC103", "SelfPace"), ("weargait:HC103", "HurriedPace"), ("weargait:WHC003", "SelfPace"),
          ("weargait:WHC026", "HurriedPace"))


@pytest.fixture(scope="module")
def tiny_runs(tmp_path_factory):
    """diag() of the default and the --ablations run with a tiny model on CPU (fakes for the checkpoint, the dev
    context and the stamp)."""
    if not HAVE_MA:
        pytest.skip("SMPL+H model not found")
    import torch

    from mdfm.data import sources
    from mdfm.eval import infer as I
    from mdfm.model import mfm as M
    mp = pytest.MonkeyPatch()
    try:
        torch.manual_seed(5)
        model = M.MFM(M.TINY)
        model.eval()
        model.requires_grad_(False)
        ck = os.path.join(str(tmp_path_factory.mktemp("run_tiny")), "ckpt_A_000010.pt")
        loaded = I.Loaded(model=model, path=ck, name="run_tiny_ckpt_A_000010", sha256="d" * 64, weights="ema", step=10,
                          phase="A", phase_steps=(10, 30), config_json="{}", config_sha256="e" * 64,
                          persons={"amass": ["fake:amass:P01"]}, persons_sha256={"all": "f" * 64}, hashes={}, drift=[])
        tg = targets(persons=PERSONS)
        tg.trials = tg.trials[[(p, t) in TRIALS for p, t in zip(tg.trials["person_id"], tg.trials["task"])]]
        keep = set(tg.trials["trial_id"])
        for k in ("walkway_ic", "insole_ic", "strides", "passes"):
            df = getattr(tg, k)
            setattr(tg, k, df[df["trial_id"].isin(keep)].reset_index(drop=True))
        recs = []
        for i, (p, task) in enumerate(TRIALS):
            sites = (0, 1, 2, 3, 4, 11) if p.endswith("WHC003") else tuple(range(13))
            recs.append((f"{S.short_id(p)}_{task}", p, record(sites=sites, seed=i, person=p, task=task)))
        dev_ids = sorted(PERSONS)
        real = g3.controls
        mp.setattr(g3, "controls", lambda fold: dev_ids if fold == "dev" else ["weargait:TRAIN01"] if fold == "train"
                   else real(fold))
        mp.setattr(sources, "demographic", lambda source, ns: {"sex": "F", "age": 64.0})
        shank = g3.shank_predictions({tid: g3.shank_gyro(r) for tid, _, r in recs})
        offs = g3.fit_offsets([shank], tg)
        summ0, tables0 = g3.evaluate(tg, [shank], {k: v[0] for k, v in offs.items()})
        ctx = {"dev": dev_ids, "train": ["weargait:TRAIN01"], "targets": tg, "summ0": summ0, "tables0": tables0,
               "records": recs}
        mp.setattr(S, "dev_context", lambda: ctx)
        mp.setattr(S, "stamp_moveatlas", lambda: {"head": "test"})
        mp.setattr(S, "REFINE_ROWS", {"refine0": 0, "refine60": 2})
        mp.setattr(I, "load_checkpoint", lambda path, weights="ema", device="cpu", cfg=None: loaded)
        out = {}
        for mode in ("default", "ablations"):
            argv = ["--checkpoint", ck, "--device", "cpu", "--refine-steps", "1", "--points", "16", "--batch-windows",
                    "4", "--out", str(tmp_path_factory.mktemp(mode))] + (["--ablations"] if mode == "ablations" else [])
            out[mode] = S.diag(S.parse(argv), log=lambda *a: None)
        yield out
    finally:
        mp.undo()


def _diag(folder):
    with open(os.path.join(folder, "diag.json"), encoding="utf-8") as f:
        return json.load(f)


SUB_KEYS = ("stride_mae_cm", "speed_mae_ms", "stride_mape_pct", "n_persons", "per_person")


def _check_row(row, persons):
    for s in S.SUBSETS:
        assert all(k in row[s] for k in SUB_KEYS), s
        assert set(row[s]["per_person"]) == {p for p in persons if S.in_subset(p, s)}
        for v in row[s]["per_person"].values():
            assert set(v) >= {"stride_cm", "speed_ms", "selfpace_stride_cm"}
    for k in ("selfpace_stride_mae_cm", "hurried_speed_bias_ms", "spearman_speed_error"):
        assert set(row[k]) == set(S.SUBSETS), k
    assert isinstance(row["selfpace_stride_mae_cm"][S.SUB12]["per_person"], dict)


def test_default_run_writes_the_schema(tiny_runs):
    d = _diag(tiny_runs["default"])
    assert set(d["rows"]) == set(S.ROWS)
    assert sorted(d["rows_not_run"]) == sorted(set(S.ROWS) - {"base", "model_legs"})
    assert all(d["rows"][r] is None for r in d["rows_not_run"])
    for r in ("base", "model_legs"):
        _check_row(d["rows"][r], PERSONS)
    hs = d["rows"]["base"]["heel_strike_ic"]
    assert {"median_ms", "offset_ms", "margin_ratio"} <= set(hs)
    for k in ("selfpace_stride_mae_cm", "hurried_speed_bias_ms", "spearman_speed_error", "heel_strike_ic"):
        assert k in d
    assert d["rerun_check"]["evaluated"] is False
    ev = pd.read_csv(os.path.join(tiny_runs["default"], "events.csv"))
    assert set(ev["row"]) == {"base", "model_legs", "model_heel_strike"}
    assert set(ev["kind"]) >= {"stride", "speed", "ic_walkway"}
    for f in ("diag.md", "per_person.csv", "trials.csv", "_provenance.json"):
        assert os.path.exists(os.path.join(tiny_runs["default"], f))
    from mdfm.gates import common
    with open(os.path.join(tiny_runs["default"], "_provenance.json"), encoding="utf-8") as f:
        prov = json.load(f)
    assert prov["code_sha256"] == common.code_sha256([os.path.join(TREE, p) for p in prov["code_files"]])


def test_ablations_run_writes_every_row(tiny_runs):
    d = _diag(tiny_runs["ablations"])
    assert d["rows_not_run"] == [] and set(d["rows"]) == set(S.ROWS)
    for r in S.ROWS:
        _check_row(d["rows"][r], PERSONS)
    for r in S.FAMILIES:
        assert d["rows"][r]["label"] == S.PREVIEW and set(d["rows"][r]["transfer_ratio"]) == set(S.SUBSETS)
    assert d["rows"]["family_lower_back"]["trials_run"] == 4
    assert d["rows"]["sites6"]["trials_reused"] == 1                       # WHC003 holds the six sites already
    assert d["rows"]["swap_whc026"]["trials_run"] == 1 and d["rows"]["swap_whc026"]["trials_reused"] == 3
    trials = pd.read_csv(os.path.join(tiny_runs["ablations"], "trials.csv"))
    assert set(trials["row"]) == set(S.ROWS) - {"model_legs"}


def test_weekend_branch_reads_the_diag(tiny_runs):
    WB = _load(os.path.join(TREE, "scripts", "weekend_branch.py"), "weekend_branch_for_t04")
    paths_ = [os.path.join(tiny_runs["default"], "diag.json"), os.path.join(tiny_runs["ablations"], "diag.json")]
    dg = WB.Diag(paths_)
    assert not dg.errors and set(dg.rows) == set(S.ROWS)
    assert dg.rows["base"][1] == paths_[0] and dg.rows["lowpass12"][1] == paths_[1]   # null rows never shadow
    s = dg.g3set("base")
    assert set(s["subsets"]) == set(S.SUBSETS) and set(s["subsets"][S.SUB12]["S"]) == {"weargait:HC103"}
    point, pp = dg.selfpace("base", WB.SUB12)
    assert pp is not None and set(pp) == {"weargait:HC103"}
    d = _diag(tiny_runs["default"])
    assert WB.scalar(dg.extra("base", "hurried_speed_bias_ms")) == d["rows"]["base"]["hurried_speed_bias_ms"][S.SUB12]["point"]
    assert WB.scalar(dg.extra("base", "spearman_speed_error")) == d["rows"]["base"]["spearman_speed_error"][S.SUB12]["rho"]
    hs = dg.extra("base", "heel_strike_ic")
    assert isinstance(hs, dict) and "margin_ratio" in hs and "median_ms" in hs
