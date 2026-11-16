"""scripts/f2_H1shape.py (task F2-H1shape) on synthetic inputs: the prediction hash, the demographic row filter and
its lockbox refusal, the rest-mesh stature and the beta_0 shift, the regressions, the criteria and the verdict, and
the queued job command. Task F2D (2026-10-05): a shift beyond SHIFT_MAX is reported and never clamped, the height
screen, and the dump with a tiny model on the CPU, where one trial's height cannot be reached: that trial is skipped
and recorded and the dump continues. The stature tests read the SMPL+H model files of the MoveAtlas tree (read only);
no target, record or demographic value of a real person is read."""
import importlib.util
import json
import math
import os

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name):
    spec = importlib.util.spec_from_file_location(f"_test_{name}", os.path.join(ROOT, "scripts", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _load("f2_H1shape")
from mdfm import paths  # noqa: E402
from mdfm.data.canonical import Refused  # noqa: E402

HAVE_SMPLH = os.path.exists(paths.smplh_model("male"))


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


# ---------------------------------------------------------------- prediction
DOC = "# t\n\n## Prediction\n\nWritten at 10:00.\n- C1 holds when b > 0.5\n\n## Results\nlater\n"


def test_prediction_hash(tmp_path):
    p = str(tmp_path / "r.md")
    write(p, DOC)
    sha = M.prediction_sha(DOC)
    assert M.check_prediction(p, expected=sha) == sha
    write(p, DOC.replace("## Results\nlater", "## Results\nnumbers added after"))   # later sections are free
    assert M.check_prediction(p, expected=sha) == sha
    write(p, DOC.replace("b > 0.5", "b > 0.3"))                                     # a criterion edited: stop
    with pytest.raises(AssertionError):
        M.check_prediction(p, expected=sha)
    write(p, "# no section\n")
    with pytest.raises(AssertionError):
        M.check_prediction(p, expected=sha)
    assert M.prediction_sha(DOC.replace("\n", "\r\n")) == sha


def test_script_constant_matches_report():
    assert M.check_prediction() == M.PREDICTION_SHA256


# ---------------------------------------------------------------- demographics
def test_table_id():
    assert M.table_id("weargait:HC103") == "HC103"
    assert M.table_id("weargait:NLS901 (control)") == "NLS901"
    with pytest.raises(ValueError):
        M.table_id("gait120:S01")


def test_read_demographics_filters_rows(tmp_path):
    p = str(tmp_path / "CONTROLS.csv")
    write(p, "title line\nSubject ID,Age,Sex,Height (in),Weight (kg)\nHC1,60,Male,70,80\n"
             "HC2\nHC3,55,Female,64,60\nHC4,50,Male,,70\n")
    out = M.read_demographics(["weargait:HC1", "weargait:HC3", "weargait:HC4", "weargait:HC9"], [], files=[p])
    assert set(out) == {"weargait:HC1", "weargait:HC3", "weargait:HC4"}       # HC2 (a short row) never indexed
    assert out["weargait:HC1"]["height_m"] == pytest.approx(70 * 0.0254)
    assert out["weargait:HC3"]["sex"] == "F" and out["weargait:HC1"]["sex"] == "M"
    assert math.isnan(out["weargait:HC4"]["height_m"])
    with pytest.raises(Refused):
        M.read_demographics(["weargait:HC1"], ["weargait:HC1"], files=[p])
    write(p, "Subject ID,Sex,Height (in)\nHC1,Male,70\nHC1,Male,71\n")
    with pytest.raises(AssertionError):
        M.read_demographics(["weargait:HC1"], [], files=[p])


# ---------------------------------------------------------------- shapes
@pytest.mark.skipif(not HAVE_SMPLH, reason="SMPL+H model files absent")
def test_stature_and_shift():
    z = np.zeros(16)
    for g in ("female", "male", "neutral"):
        st, leg = M.stature_of(z, g)
        assert M.STATURE_RANGE_M[0] < float(st) < M.STATURE_RANGE_M[1]
        assert 0.35 * float(st) < float(leg) < 0.6 * float(st)
    two = M.stature_of(np.zeros((2, 16)), "male")[0]
    assert two.shape == (2,)
    b0 = np.zeros(16)
    st0 = float(M.stature_of(b0, "male")[0])
    for target in (st0 + 0.08, st0 - 0.06):
        bb, d = M.shift_beta0(b0, 1, target)
        assert abs(float(M.stature_of(bb, 1)[0]) - target) < M.SHIFT_TOL_M
        assert np.allclose(bb[1:], b0[1:]) and bb[0] == pytest.approx(d)
    with pytest.raises(ValueError):
        M.shift_beta0(b0, 1, float("nan"))
    with pytest.raises(ValueError):
        M.shift_beta0(b0, 1, 3.5)


# ---------------------------------------------------------------- statistics
def test_ols_and_boot():
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    z = rng.normal(size=200)
    y = 0.3 + 1.0 * x - 0.5 * z + 0.01 * rng.normal(size=200)
    c = M.ols(y, np.column_stack([x, z]))
    assert c == pytest.approx([0.3, 1.0, -0.5], abs=0.01)
    b = M.boot_coef(y, x, 1, n_boot=200, seed=1)
    assert b["interval"][0] < b["point"] < b["interval"][1]
    assert math.isnan(M.ols([1.0, 2.0], [1.0, 1.0])[1])
    assert M.r2(y, np.column_stack([x, z])) > 0.99


def test_verdict_table():
    assert M.verdict_of(True, True, True) == "supported"
    assert M.verdict_of(False, True, True) == "supported"
    assert M.verdict_of(True, False, True) == "partly"
    assert M.verdict_of(True, True, False) == "partly"
    assert M.verdict_of(False, False, True) == "partly"
    assert M.verdict_of(False, True, False) == "not_supported"
    assert M.verdict_of(False, False, False) == "not_supported"


def _table(rng, n, shape_effect):
    """A synthetic person table: h_ref 1.70, heights spread, r = 0.97 (h_ref / h)^shape_effect with noise."""
    h = rng.uniform(1.55, 1.95, n)
    S = 120 + 60 * (h - 1.75) + rng.normal(0, 8, n)
    r = 0.97 * (1.70 / h) ** shape_effect * np.exp(rng.normal(0, 0.01, n))
    return pd.DataFrame({"person_id": [f"p{i}" for i in range(n)], "r": r, "S_cm": S, "height_m": h,
                         "sex": "M", "h_ref_m": 1.70, "x": np.log(1.70 / h)})


def test_criteria_direction():
    rng = np.random.default_rng(3)
    t = _table(rng, 54, 1.0)
    c = M.criteria(t, g_s=0.97 * float(np.median(1.70 / t["height_m"])), g_sim=0.97, n_boot=300, seed=2)
    assert c["C1"]["holds"] and c["C3"]["holds"] and c["C2"]["holds"]
    assert c["C1"]["point"] == pytest.approx(1.0, abs=0.1)
    t0 = _table(rng, 54, 0.0)
    c0 = M.criteria(t0, g_s=0.85, g_sim=0.97, n_boot=300, seed=2)
    assert not c0["C1"]["holds"] and not c0["C3"]["holds"]
    t1 = t.copy()
    t1.loc[0, "x"] = np.nan
    assert M.criteria(t1, 0.85, 0.97, n_boot=50)["n_dropped"] == 1


def test_person_table():
    st = pd.DataFrame({"person_id": ["a", "a", "a", "b"], "x": [100.0, 120.0, 110.0, 130.0],
                       "y": [85.0, 100.0, 99.0, 120.0]})
    prior = {"female": {"stature_m": 1.6}, "male": {"stature_m": 1.75}, "neutral": {"stature_m": 1.68}}
    t = M.person_table(st, {"a": {"height_m": 1.80, "sex": "M"}, "b": {"height_m": 1.60, "sex": "F"}}, prior)
    a = t.set_index("person_id")
    assert a.loc["a", "r"] == pytest.approx(np.median([0.85, 100 / 120, 0.9]))
    assert a.loc["a", "x"] == pytest.approx(math.log(1.75 / 1.80))
    assert a.loc["b", "x"] == pytest.approx(0.0)
    assert a.loc["a", "S_cm"] == pytest.approx(110.0)


# ---------------------------------------------------------------- the job
def test_job_command():
    c = M.job_command()
    assert c[0] == "scripts/queue_job.py"
    v = lambda k: c[c.index(k) + 1]  # noqa: E731
    assert v("--cls") == "1" and 70 <= int(v("--priority")) <= 79 and v("--group") == "F2D"
    assert v("--name").endswith("_r2")                 # the first name failed and a logged name never reruns
    assert v("--requires") == M.ENDS[M.DUMP_END][1]
    tail = c[c.index("--") + 1:]
    assert tail[:2] == ["scripts/f2_H1shape.py", "--dump"]
    assert tail[tail.index("--memory-gb") + 1] == "12"
    assert tail[tail.index("--out") + 1].startswith("reports/bench/f2_scale/H1shape/")


# ---------------------------------------------------------------- a shift beyond SHIFT_MAX (job of 2026-10-05)
@pytest.mark.skipif(not HAVE_SMPLH, reason="SMPL+H model files absent")
def test_shift_beyond_the_bound_is_reported_not_clamped():
    """The failure of f2_h1shape_dump_v1_B: a target that needs |shift| > SHIFT_MAX raises ShiftOutOfRange (still a
    ValueError) with the required shift; counterfactual_shape turns it into a skipped trial with that shift, no cf
    betas and no clamped value."""
    b0 = np.zeros(16)
    st0 = float(M.stature_of(b0, "female")[0])
    per_unit = float(M.stature_of(np.eye(16)[0], "female")[0]) - st0
    target = st0 + per_unit * (M.SHIFT_MAX + 0.7)                  # needs about SHIFT_MAX + 0.7
    with pytest.raises(M.ShiftOutOfRange) as e:
        M.shift_beta0(b0, "female", target)
    assert isinstance(e.value, ValueError)
    assert e.value.required == pytest.approx(M.SHIFT_MAX + 0.7, abs=0.05) and e.value.target_m == target
    cf = M.counterfactual_shape(b0, 0, target)                      # gender code 0: female
    assert cf["status"] == "skipped_shift" and cf["cf_beta"] is None
    assert cf["required_shift"] == pytest.approx(M.SHIFT_MAX + 0.7, abs=0.05)
    assert math.isnan(cf["shift"]) and math.isnan(cf["stature_cf_m"])
    ok = M.counterfactual_shape(b0, 0, st0 - 0.05)
    assert ok["status"] == "done" and abs(ok["stature_cf_m"] - (st0 - 0.05)) < M.SHIFT_TOL_M
    assert abs(ok["shift"]) <= M.SHIFT_MAX and ok["required_shift"] == ok["shift"]
    assert M.counterfactual_shape(b0, 0, float("nan"))["status"] == "no_height"
    far = M.counterfactual_shape(b0, 0, 0.4)                        # beyond the search bound too: still skipped
    assert far["status"] == "skipped_shift"


@pytest.mark.skipif(not HAVE_SMPLH, reason="SMPL+H model files absent")
def test_height_screen_and_feet_inches_reading():
    reach = M.reachable_statures()
    for g in ("female", "male"):
        lo, hi = reach[g]
        assert M.STATURE_RANGE_M[0] - 0.2 < lo < 1.5 < 1.9 < hi
    lo_f = reach["female"][0]
    demo = {"a": {"height_m": lo_f - 0.05, "sex": "F"}, "b": {"height_m": 1.65, "sex": "F"},
            "c": {"height_m": float("nan"), "sex": "M"}, "d": {"height_m": 1.80, "sex": "M"}}
    assert M.height_flags(demo, ["a", "b", "c", "d", "e"], reach) == ["a"]
    assert M.ft_in_reading(47.0) == 55.0 and math.isnan(M.ft_in_reading(510.0))
    assert math.isnan(M.ft_in_reading(35.0)) and math.isnan(M.ft_in_reading(float("nan")))
    assert math.isnan(M.ft_in_reading(64.5))


# ---------------------------------------------------------------- the dump with a tiny model, one trial skipped
HAVE_NEUTRAL = os.path.exists(paths.smplh_model("neutral"))


def _record(person, task, seed):
    import model_fixture as MF
    st = MF.imu_stream(12.0, sites=tuple(range(13)), seed=seed)
    sig = np.array(st.values, copy=True)
    sig[..., 6:9] = 0.3
    T = sig.shape[0]
    return {"source": "weargait_pd", "subject_ns": person.replace("weargait:", "weargait_pd:"), "person_id": person,
            "key": task, "fps": 100.0, "signal": sig, "channel_mask": np.ones((13, 9), bool),
            "location_ids": np.arange(13), "sample_valid": np.ones((T, 13), bool), "acc_content": "with_gravity",
            "lockbox": False, "tier": "TRAIN"}


def _targets(persons):
    from mdfm.gates import g3
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


@pytest.mark.skipif(not (HAVE_SMPLH and HAVE_NEUTRAL), reason="SMPL+H model files absent")
def test_dump_skips_an_unreachable_height_and_continues(tmp_path, monkeypatch):
    """The dump decodes every trial: a person whose reported height needs a shift beyond SHIFT_MAX keeps the base
    decode, gets no counterfactual decode and is recorded with the required shift and the reported height; the next
    trials still run; read_dump pairs the base strides with the counterfactual trials and lists the skip."""
    import torch

    from mdfm.data import sources
    from mdfm.eval import infer as I
    from mdfm.gates import g3
    from mdfm.model import mfm as MM
    torch.manual_seed(7)
    model = MM.MFM(MM.TINY)
    model.eval()
    model.requires_grad_(False)
    ck = str(tmp_path / "run_tiny" / "ckpt_B_000010.pt")
    loaded = I.Loaded(model=model, path=ck, name="run_tiny_ckpt_B_000010", sha256="d" * 64, weights="ema", step=10,
                      phase="B", phase_steps=(5, 5), config_json="{}", config_sha256="e" * 64,
                      persons={"weargait_pd": ["weargait:TR01"]}, persons_sha256={"all": "f" * 64}, hashes={},
                      drift=[])
    train, dev, lock = ["weargait:TR01", "weargait:TR02"], ["weargait:DV01"], ["weargait:LB01"]
    sets = {"train54": train, "dev14": dev, "dev12": dev}
    heights = {"weargait:TR01": 1.66, "weargait:TR02": 0.90, "weargait:DV01": float("nan")}
    seen = []

    def imu_records(persons, trials, allow_lockbox=False, read_row=None, tasks=g3.TASKS):
        assert not allow_lockbox
        for i, r in enumerate(trials.itertuples()):
            seen.append(r.trial_id)
            yield r.trial_id, _record(r.person_id, r.task, i)

    monkeypatch.setattr(M, "load_sets", lambda f1_results=None: (sets, lock))
    monkeypatch.setattr(M, "read_demographics",
                        lambda persons, locked, files=None, weight=False: {p: {"height_m": heights[p], "sex": "F"}
                                                                            for p in persons})
    monkeypatch.setattr(g3, "load_targets", lambda persons, allow_lockbox=False, read_row=None, tasks=g3.TASKS:
                        _targets(persons))
    monkeypatch.setattr(g3, "imu_records", imu_records)
    monkeypatch.setattr(sources, "demographic", lambda source, ns: {"sex": "F", "age": 64.0})
    monkeypatch.setattr(I, "load_checkpoint", lambda path, weights="ema", device="cpu", cfg=None: loaded)
    monkeypatch.setattr(M.GC(), "stamp_moveatlas", lambda: {"head": "test"})
    out = str(tmp_path / "dump")
    M.dump(M.parse(["--dump", "--checkpoint", ck, "--device", "cpu", "--refine-steps", "1", "--points", "16",
                    "--batch-windows", "4", "--out", out]))
    assert len(seen) == 3                                           # the trial after the skip ran too
    sh = pd.read_csv(os.path.join(out, "shape.csv"), encoding="utf-8")
    st = dict(zip(sh["person_id"], sh["cf_status"]))
    assert st == {"weargait:TR01": "done", "weargait:TR02": "skipped_shift", "weargait:DV01": "no_height"}
    row = sh[sh["person_id"] == "weargait:TR02"].iloc[0]
    assert math.isnan(row["beta0_shift"]) and row["beta0_shift_required"] < -M.SHIFT_MAX
    sk = pd.read_csv(os.path.join(out, "skipped_cf.csv"), encoding="utf-8")
    assert list(sk.columns) == M.SKIPPED_COLUMNS and len(sk) == 1
    s = sk.iloc[0]
    assert s["trial_id"] == "TR02_SelfPace" and s["reported_height_m"] == pytest.approx(0.90)
    assert s["reported_height_in"] == pytest.approx(0.90 / M.IN_TO_M) and s["required_beta0_shift"] < -M.SHIFT_MAX
    with open(os.path.join(out, "dump.json"), encoding="utf-8") as f:
        doc = json.load(f)
    assert doc["partial"] is False and doc["cf_skipped"]["n_trials"] == 1
    assert doc["cf_skipped"]["trials"][0]["trial_id"] == "TR02_SelfPace"
    cf = pd.read_csv(os.path.join(out, "events_cf.csv"), encoding="utf-8", dtype={"matched": str})
    # the walkway rows of every trial are listed; only the trial with a counterfactual decode can match one
    assert not (cf.loc[cf["trial_id"] != "TR01_SelfPace", "matched"].str.lower() == "true").any()
    assert cf.loc[cf["trial_id"] != "TR01_SelfPace", "prediction"].isna().all()
    base = pd.read_csv(os.path.join(out, "events_base.csv"), encoding="utf-8")
    assert {"TR01_SelfPace", "TR02_SelfPace", "DV01_SelfPace"} <= set(base["trial_id"])
    r = M.read_dump(out, sets, lock, {p: {"height_m": h} for p, h in heights.items()}, 0.97, n_boot=20, seed=1,
                    flagged=["weargait:TR02"])
    assert r["status"] == "read" and r["cf_skipped"]["train54_trials"] == ["TR02_SelfPace"]
    assert r["D2"]["strides_base_paired"] <= r["D2"]["strides_base_all"]
    with pytest.raises(Refused):
        M.read_dump(out, sets, ["weargait:TR02"], {}, 0.97, n_boot=20, seed=1)


# ---------------------------------------------------------------- read_dump on a hand-written dump (review of F2D)
EV_COLS = ["row", "kind", "trial_id", "person_id", "task", "foot", "pass_id", "t0_s", "t1_s", "reference",
           "prediction", "signed_error", "unit", "matched"]
HW_TRAIN = ["weargait:TR01", "weargait:TR02", "weargait:TR03", "weargait:TR04", "weargait:TR05"]
HW_SETS = {"train54": HW_TRAIN, "dev14": ["weargait:DV01"], "dev12": ["weargait:DV01"]}
# reported heights; TR02 is the person whose counterfactual needs a shift beyond the bound
HW_HEIGHTS = {"weargait:TR01": 1.60, "weargait:TR02": 1.194, "weargait:TR03": 1.70, "weargait:TR04": 1.80,
              "weargait:TR05": 1.75, "weargait:DV01": 1.65}


def _stride_rows(row, trial, person, ys, x=100.0):
    """Matched stride rows of one trial: one per (foot, pass) key, walkway x cm, decoded y cm."""
    keys = [("L", "0"), ("R", "0"), ("L", "1"), ("R", "1")]
    return [(row, "stride", trial, person, "SelfPace", f, p, 1.0 + i, 2.0 + i, x, y, y - x, "cm", "True")
            for i, ((f, p), y) in enumerate(zip(keys, ys))]


def hand_dump(folder, with_status=True, skipped_trials=("TR02_SelfPace",)):
    """A dump folder written by hand. train54: TR01 and TR03 have both decodes (cf_status done), TR02 is skipped
    (base only), TR04 and TR05 have a shape row and no stride; DV01 (dev14) has both and must stay out of D2.
    Base ratios: TR01 0.80, 0.82; TR03 0.84, 0.86; TR02 0.60, 0.62, 0.64 (median paired 0.83, over every train54
    trial 0.80). Counterfactual ratios: TR01 0.86, 0.88; TR03 0.90, 0.92 (median 0.89; +6 cm on every paired stride).
    Decoded statures follow 1.70 + 0.2 (height - 1.70) except TR02, decoded at 1.70 m."""
    os.makedirs(folder, exist_ok=True)
    base = (_stride_rows("base", "TR01_SelfPace", "weargait:TR01", [80.0, 82.0])
            + _stride_rows("base", "TR03_SelfPace", "weargait:TR03", [84.0, 86.0])
            + _stride_rows("base", "TR02_SelfPace", "weargait:TR02", [60.0, 62.0, 64.0])
            + _stride_rows("base", "DV01_SelfPace", "weargait:DV01", [99.0, 99.0]))
    cf = (_stride_rows("cf", "TR01_SelfPace", "weargait:TR01", [86.0, 88.0])
          + _stride_rows("cf", "TR03_SelfPace", "weargait:TR03", [90.0, 92.0])
          + _stride_rows("cf", "DV01_SelfPace", "weargait:DV01", [110.0, 110.0])
          # the skipped trial's walkway rows are listed in the cf file but never matched (no cf decode)
          + [("cf", "stride", "TR02_SelfPace", "weargait:TR02", "SelfPace", "L", "0", 1.0, 2.0, 100.0, "", "", "cm",
              "False")])
    pd.DataFrame(base, columns=EV_COLS).to_csv(os.path.join(folder, "events_base.csv"), index=False,
                                               encoding="utf-8", lineterminator="\n")
    pd.DataFrame(cf, columns=EV_COLS).to_csv(os.path.join(folder, "events_cf.csv"), index=False, encoding="utf-8",
                                             lineterminator="\n")
    status = {"TR01": "done", "TR02": "skipped_shift", "TR03": "done", "TR04": "done", "TR05": "done", "DV01": "done"}
    rows = []
    for t, s in status.items():
        p = f"weargait:{t}"
        h = HW_HEIGHTS[p]
        st = 1.70 if t == "TR02" else 1.70 + 0.2 * (h - 1.70)
        r = {"trial_id": f"{t}_SelfPace", "person_id": p, "stature_dec_m": st, "leg_dec_m": 0.53 * st,
             "beta0_shift": float("nan") if s == "skipped_shift" else 0.1, "has_height": True}
        if with_status:
            r["cf_status"] = s
        rows.append(r)
    pd.DataFrame(rows).to_csv(os.path.join(folder, "shape.csv"), index=False, encoding="utf-8", lineterminator="\n")
    sk = [{"trial_id": t, "person_id": "weargait:" + t.split("_")[0], "required_beta0_shift": -6.68,
           "shift_max": M.SHIFT_MAX, "reported_height_m": 1.194, "reported_height_in": 47.0, "stature_dec_m": 1.70,
           "gender_code": 0} for t in skipped_trials]
    pd.DataFrame(sk, columns=M.SKIPPED_COLUMNS).to_csv(os.path.join(folder, "skipped_cf.csv"), index=False,
                                                       encoding="utf-8", lineterminator="\n")
    doc = {"partial": False, "cf_skipped": {"shift_max": M.SHIFT_MAX, "n_trials": 1, "trials": [
        {"trial_id": "TR02_SelfPace", "person_id": "weargait:TR02", "required_beta0_shift": -6.68}]}}
    with open(os.path.join(folder, "dump.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f)


def test_read_dump_pairs_the_base_with_the_counterfactual_trials(tmp_path):
    """D2 reads the cf decode against the base decode of the same trials (the skipped trial out of both sides), with
    the all-trial base gain beside; the two readings can differ; D1 without the flagged person is computed without
    her. Every expected value is set by hand_dump."""
    folder = str(tmp_path / "dump")
    hand_dump(folder)
    demo = {p: {"height_m": h} for p, h in HW_HEIGHTS.items()}
    r = M.read_dump(folder, HW_SETS, [], demo, 0.97, n_boot=50, seed=1, flagged=["weargait:TR02", "weargait:DV01"])
    d2 = r["D2"]
    assert d2["g_base"] == pytest.approx(0.83) and d2["g_base_all_trials"] == pytest.approx(0.80)
    assert d2["g_cf"] == pytest.approx(0.89) and d2["gain_change"] == pytest.approx(0.06)
    assert d2["needed"] == pytest.approx(0.5 * (0.97 - 0.83)) and d2["holds"] is False
    assert d2["needed_against_all_trials"] == pytest.approx(0.5 * (0.97 - 0.80))
    assert d2["gain_change_against_all_trials"] == pytest.approx(0.09) and d2["holds_against_all_trials"] is True
    assert (d2["strides_base_paired"], d2["strides_base_all"], d2["strides_cf"]) == (4, 7, 4)
    assert d2["mean_cf_minus_base_cm_paired"] == pytest.approx(6.0)
    assert r["cf_skipped"]["train54_trials"] == ["TR02_SelfPace"] and r["n_train_persons"] == 5
    # D1 as registered keeps TR02; the post hoc line leaves out the flagged train54 person only (DV01 is not in D1)
    assert r["D1"]["n_persons"] == 5 and r["D1"]["point"] < 0.1
    ph = r["posthoc_D1_without_flagged"]
    assert ph["left_out"] == ["weargait:TR02"] and ph["n_persons"] == 4 and ph["point"] == pytest.approx(0.2)
    assert r["median_beta0_shift"] == pytest.approx(0.1)            # the skipped trial's NaN shift is left out
    # a dump written before the skip rule (no cf_status): every trial with a height is a cf trial, so the paired
    # base is the all-trial base
    old = str(tmp_path / "old")
    hand_dump(old, with_status=False)
    d2o = M.read_dump(old, HW_SETS, [], demo, 0.97, n_boot=20, seed=1)["D2"]
    assert d2o["g_base"] == pytest.approx(0.80) and d2o["strides_base_paired"] == 7
    assert "posthoc_D1_without_flagged" not in M.read_dump(old, HW_SETS, [], demo, 0.97, n_boot=20, seed=1)
    # skipped_cf.csv and dump.json must name the same trials
    bad = str(tmp_path / "bad")
    hand_dump(bad, skipped_trials=("TR01_SelfPace",))
    with pytest.raises(AssertionError, match="different trials"):
        M.read_dump(bad, HW_SETS, [], demo, 0.97, n_boot=20, seed=1)
