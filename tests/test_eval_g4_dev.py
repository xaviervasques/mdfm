"""Tests of scripts/eval_g4_dev.py, the G4 development estimate of a checkpoint (weekend task T06; rule
docs/rules/G4_RULE.md, DRAFT; harness mdfm/gates/g4.py):

- the truth given as the model's prediction reads r = 1 and RMSE 0 on the GRF line and r = 1 on the EMG line,
  beside the Newton floor and the template, through run_dev and g4.evaluate (the g4 trial readers replaced by
  synthetic trials: no record), with the rule's lines of 2026-10-05 as the sealed read applies them: the EMG
  trials under 1 s left out by the harness before the model is called, the same samples for the template and
  the model, the emg_r_gain line at delta 0.03 deciding emg_r's pass, the GRF row without the low-peak plate
  series, a patient without a recorded body mass dropped from the GRF rows and listed but kept for the metric-3D
  round trip; the model's (bw, contact) pair scored through g4.flight_predictor, the sealed read's flight code;
- a lockbox stroke patient or gait120 person in a development set is refused before any record opens
  (canonical.load counted: zero calls); a lockbox person's record is refused by the predictor before its root is
  read; template persons among the scored persons are refused;
- records left out: a record under the 1 s window minimum (infer.MIN_S) gives no window, the model is not called,
  and it is counted per source and reason (pose of a gait120 EMG trial, pose and camera of a stroke trial, a trial
  without a valid pelvis); left-out trials keep every row on the same samples (a GRF trial left out scores no
  sample for the Newton floor either; an EMG trial left out is gone for the template and the model alike, and the
  summary is g4's on the kept trials);
- the disjointness check: the persons/*.txt lists of a checkpoint folder (sidecars left out), a scored person in a
  list refused, a checkpoint without the folder refused; the development sets of the script against the lists of
  the seed-1 run folder when it is present;
- the predictor on tests/model_fixture.py windows with a tiny model on the CPU: per-frame ForceHead GRF (flight
  frames at 0), the BW before the flight rule and the contact probabilities whose flight frames are the model's
  own flag (a model whose flag differs is refused), grf_contact of a stroke trial equal to grf through
  g4.flight_predictor, and EMG log-envelopes of the four muscles at every pose frame; the pose path of a record
  with its root attached (two windows, each frame from its most central window, no root read); the metric-3D
  round trip of a walking skeleton through one seeded camera (deterministic under its seed), and of a person
  without GRF trials (metric3d_person);
- helpers: frame assignment, the hold fill, the linear frame interpolation, the camera seed;
- the stroke set of a stage-2 checkpoint (task S2.10): --stroke-set fold reads the 8 patients of the proposed stroke
  dev fold (their SHA-256 pinned and equal to the fold table's, S047 among them, all among the 40), a changed fold
  table, a lockbox person in it or the pinned SHA-256 differing is refused, rows of another group are not read, a
  fold person outside the non-lockbox stroke patients is refused (review REVS210); main() with the fold on a stub
  checkpoint whose training lists hold the other 32 scores the 8 (the predictor and the trials replaced), the
  default set on that checkpoint is refused with the hint, and a checkpoint trained on one of the 8 is refused before
  anything is written; the default output folder carries the suffix _strokefold.
"""
import collections
import hashlib
import importlib.util
import os

import numpy as np
import pytest
import torch

from mdfm import paths
from mdfm.data import canonical as C
from mdfm.data import sources as SRC
from mdfm.data import windows as W
from mdfm.gates import g4
from mdfm.model import mfm as M

import model_fixture as MF

TREE = paths.TREE
SCRIPT = os.path.join(TREE, "scripts", "eval_g4_dev.py")
HAVE_MANIFEST = os.path.exists(C.manifest_path())
HAVE_MA = os.path.exists(paths.smplh_model("neutral"))
needs_manifest = pytest.mark.skipif(not HAVE_MANIFEST, reason="tiers records file not found")
needs_ma = pytest.mark.skipif(not HAVE_MA, reason="SMPL+H model not found")
RUN1_PERSONS = os.path.join(TREE, "cache", "runs", "phaseA_root_full", "persons")


def _script():
    spec = importlib.util.spec_from_file_location("eval_g4_dev_script", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S = _script()


# ---------------------------------------------------------------- synthetic trials
def walking_world(T=300, fps=100.0, speed=1.2, phase=0.0):
    """World joints (T, 15, 3) m y up of a skeleton moving along +z at `speed` (0: stepping in place, the stance
    ankle still): the fixture skeleton at 0.5 m per torso unit, feet on the floor, a vertical bob of the body,
    ankles alternately lifted by up to 6 cm."""
    t = np.arange(T) / fps
    base = MF.SKELETON15.astype(np.float64) * 0.5
    base = base - np.array([0.0, base[:, 1].min(), 0.0])                     # feet on the floor
    X = np.repeat(base[None], T, 0)
    X[:, :, 2] += speed * t[:, None]
    X[:, :13, 1] += 0.02 * np.sin(4 * np.pi * t + phase)[:, None]           # the ankles do not bob
    lift = 0.06 * np.clip(np.sin(2 * np.pi * t + phase), 0, None)
    for j, s in ((13, 1.0), (14, -1.0)):                                      # ankles of CANONICAL_15
        X[:, j, 1] += lift if s > 0 else 0.06 * np.clip(-np.sin(2 * np.pi * t + phase), 0, None)
    return X


def grf_trial(person, key, seed=0, T=300):
    rng = np.random.default_rng(seed)
    t = np.arange(T) / 100.0
    obs = np.stack([0.6 + 0.4 * np.sin(2 * np.pi * t + rng.uniform(0, 1)),
                    0.6 - 0.4 * np.sin(2 * np.pi * t + rng.uniform(0, 1))], 1)
    return g4.GRFTrial(person=person, key=key, fps=100.0, t=t, world=walking_world(T, speed=0.0, phase=seed), obs=obs,
                       obs_valid=np.ones((T, 2), bool), mass_kg=70.0, mass_from="c3d", sex="M")


def emg_trial(person, key, seed=0, T=220, noise=0.1):
    """Synthetic EMGTrial at 100 Hz (no fps field: the harness reads the rate from the times): four phase-locked
    log-envelopes with Gaussian noise of the given SD."""
    rng = np.random.default_rng(seed)
    phase = np.linspace(0.0, 1.0, T)
    obs = np.stack([np.sin(2 * np.pi * (phase + m / 4)) + noise * rng.standard_normal(T) - 3 for m in range(4)], 1)
    return g4.EMGTrial(person=person, key=key, t=np.arange(T) / 100.0, frames=np.arange(T), phase=phase, obs=obs,
                       valid=np.ones((T, 4), bool))


class TruthPredictor:
    """The truth as the model's prediction (the stub of a G4Predictor in run_dev): grf_contact gives the plate
    truth as BW with both contact probabilities at 0.9 (never in flight); grf, the model's own flight flag
    applied, is never what run_dev scores (it raises); metric3d_person records the persons asked."""

    def __init__(self):
        self.metric3d_persons = set()
        self.m3d = []
        self.m3d_persons_asked = []

    def grf(self, trial):
        raise AssertionError("run_dev scores grf_contact through g4.flight_predictor, not grf")

    def grf_contact(self, trial):
        return trial.obs.copy(), np.full(trial.obs.shape, 0.9)

    def emg(self, trial):
        return trial.obs.copy()

    def metric3d_person(self, person):
        self.m3d_persons_asked.append(person)


SPECIAL = {}                # per test: no_mass (a stroke person without a mass), short (EMG trials under 1 s),
                            # low_peak ((person, key, foot) whose plate peak is lowered to 0.4 BW), noise (EMG)


@pytest.fixture()
def synthetic_trials(monkeypatch):
    stroke = ["fake:stroke:S901", "fake:stroke:S902", "fake:stroke:S903"]
    dev = ["fake:gait120:S901", "fake:gait120:S902", "fake:gait120:S903"]
    train = ["fake:gait120:S951", "fake:gait120:S952"]
    SPECIAL.clear()

    def fake_grf(persons, allow_lockbox=False, stats=None, steps_mass=False, left_out=None, no_mass=None):
        assert not allow_lockbox and not steps_mass
        out = []
        for i, p in enumerate(persons):
            for k in range(2):
                tr = grf_trial(p, f"trial{k}", seed=10 * i + k)
                if p == SPECIAL.get("no_mass"):                  # as g4.grf_trials does without a recorded mass
                    if left_out is not None:
                        left_out.append(g4._left("grf", g4.STROKE, p, tr.key, "force", "no_body_mass", len(tr.t),
                                                 tr.fps))
                    if no_mass is not None:
                        no_mass.append(g4.GRFTrial(**{**tr.__dict__, "obs_valid": np.zeros_like(tr.obs_valid)}))
                    continue
                lp = SPECIAL.get("low_peak")
                if lp and (p, tr.key) == lp[:2]:
                    j = "LR".index(lp[2])
                    tr.obs[:, j] *= 0.4 / tr.obs[:, j].max()
                out.append(tr)
        return out

    def fake_emg(persons, allow_lockbox=False, stats=None, keep_record=False):
        assert not allow_lockbox
        return [emg_trial(p, f"level_walking_trial0{k}", seed=10 * i + k,
                          T=92 if (p, k) in SPECIAL.get("short", ()) else 220, noise=SPECIAL.get("noise", 0.1))
                for i, p in enumerate(persons) for k in range(1, 3)]

    monkeypatch.setattr(g4, "grf_trials", fake_grf)
    monkeypatch.setattr(g4, "emg_trials", fake_emg)
    return stroke, dev, train


# ---------------------------------------------------------------- truth as the prediction
@needs_manifest
def test_truth_as_the_prediction_gives_r_1_and_rmse_0(synthetic_trials):
    stroke, dev, train = synthetic_trials
    SPECIAL["noise"] = 0.4                                   # a template well below r = 1: the truth gains on it
    able = ["fake:stroke_ab:S801"]
    res = S.run_dev(TruthPredictor(), stroke, dev, train, able, metric3d=False)
    summ = res["summary"]
    assert summ["grf_rmse_pct_bw"]["model"]["point"] == pytest.approx(0.0, abs=1e-12)
    assert summ["grf_r"]["model"]["point"] == pytest.approx(1.0, abs=1e-12)
    assert summ["emg_r"]["model"]["point"] == pytest.approx(1.0, abs=1e-12)
    for line, names in (("grf_rmse_pct_bw", ("newton", "model")), ("grf_r", ("newton", "model")),
                        ("grf_r_gain", ("model",)), ("emg_r", ("template", "model"))):
        for n in names:
            s = summ[line][n]
            assert s["n_persons"] == 3 and np.isfinite(s["point"]) and all(np.isfinite(s["interval"])), (line, n)
    assert summ["grf_rmse_pct_bw"]["newton"]["point"] > 0
    assert summ["grf_r_gain"]["model"]["point"] == pytest.approx(1.0 - summ["grf_r"]["newton"]["point"])
    assert (res["person_grf"]["model_rmse_pct_bw"] == 0).all() and np.allclose(res["person_grf"]["model_r"], 1.0)
    assert np.allclose(res["person_emg"]["model_r"], 1.0) and (res["person_emg"]["template_r"] < 1.0).all()
    assert set(res["trial_grf"]["foot"]) == {"L", "R"} and (res["trial_grf"]["model_rmse_pct_bw"] == 0).all()
    ab = res["able"]["summary"]
    assert ab["grf_rmse_pct_bw"]["model"]["point"] == pytest.approx(0.0, abs=1e-12)
    assert ab["grf_r"]["model"]["n_persons"] == 1
    # the EMG gain over the template: the median of per-person differences, and emg_r's pass depends on it
    pe = res["person_emg"]
    assert (pe["pairs_scored"] == pe["pairs"]).all() and res["trial_emg"]["scored"].all()
    gain = summ["emg_r_gain"]["model"]
    assert gain["n_persons"] == 3 and gain["point"] == pytest.approx(np.median(1.0 - pe["template_r"].to_numpy()))
    lines = S.gate_lines(summ, "r" * 64, "c" * 64, "p" * 64, "t" * 64)          # the script's gate rows
    assert [r.line for r in lines] == ["grf_rmse", "grf_r_gain", "emg_r", "emg_r_gain"]
    assert lines[3].pass_line == g4.EMG_GAIN == 0.03 and lines[3].extra["template_persons_sha256"] == "t" * 64
    assert all(r.checkpoint_sha256 == "c" * 64 and r.train_persons_sha256 == "p" * 64 for r in lines)
    assert lines[0].point == pytest.approx(0.0, abs=1e-12)
    assert gain["point"] >= g4.EMG_GAIN and [r.verdict for r in lines[2:]] == ["pass", "pass"]
    # no plate series below 0.5 BW: the secondary row is the primary row
    low = res["low_peak"]
    assert low["series_left_out"] == 0 and len(low["series"]) == 0 and low["patients"] == []
    for line in ("grf_rmse_pct_bw", "grf_r", "grf_r_gain"):
        assert low["summary"][line] == summ[line]
    assert res["grf_persons_no_mass"] == [] and res["left_out"]["n"] == 0
    assert res["metric3d"]["mpjpe_mm"]["n_persons"] == 0


@needs_manifest
def test_the_rule_lines_of_the_development_estimate(synthetic_trials):
    """The rule's lines on the development rows: EMG trials under 1 s left out by the harness before the model is
    called (counted, reason pose_shorter_than_min_s, line emg); a model below the template keeps emg_r above its
    pass line but undetermined, the gain line killed; a plate series below 0.5 BW out of the secondary row only;
    a patient without a recorded mass dropped from the GRF rows and listed (reason no_body_mass, line grf), and
    its trials still given to the metric-3D round trip when it runs."""
    stroke, dev, train = synthetic_trials
    SPECIAL.update(no_mass=stroke[2], short={(dev[0], 2), (dev[1], 2)}, low_peak=(stroke[0], "trial1", "R"))
    seen = []

    class Worse(TruthPredictor):
        def emg(self, trial):
            seen.append((trial.person, trial.key))
            assert len(trial.t) >= 100, "the model was called on a trial under 1 s"
            return trial.obs + 0.5 * np.sin(np.arange(len(trial.t)))[:, None]

    worse = Worse()
    res = S.run_dev(worse, stroke, dev, train, [], metric3d=False)
    assert sorted(seen) == sorted([(dev[0], "level_walking_trial01"), (dev[1], "level_walking_trial01"),
                                   (dev[2], "level_walking_trial01"), (dev[2], "level_walking_trial02")])
    assert worse.m3d_persons_asked == []                                     # no metric-3D step asked
    truth = TruthPredictor()                                                 # the metric-3D line needs no mass
    S.run_dev(truth, stroke, dev, train, [], metric3d=True)
    assert truth.m3d_persons_asked == [stroke[2]]
    assert res["emg_trials_all"] == 6 and S.n_trials(res["trial_emg"]) == 4
    lo = res["left_out"]
    assert lo["n"] == 4 and lo["by_line"] == {"emg": 2, "grf": 2}
    assert lo["counts"] == {"gait120": {"pose_shorter_than_min_s": 2}, "stroke_gait_vancriekinge": {"no_body_mass": 2}}
    assert {(r["person_id"], r["reason"]) for r in lo["records"]} == {
        (dev[0], "pose_shorter_than_min_s"), (dev[1], "pose_shorter_than_min_s"), (stroke[2], "no_body_mass")}
    assert "no_body_mass" in lo["reasons"] and "pose_shorter_than_min_s" in lo["reasons"]
    assert res["grf_persons_no_mass"] == [stroke[2]] and list(res["person_grf"]["person_id"]) == stroke[:2]
    summ = res["summary"]
    lines = S.gate_lines(summ, "r" * 64)
    assert lines[2].point > g4.EMG_PASS and summ["emg_r_gain"]["model"]["point"] < g4.EMG_GAIN
    assert [r.verdict for r in lines[2:]] == ["undetermined", "kill"]
    assert [r.verdict for r in g4.gate_results(summ, "model", "r" * 64)][2] == "pass"   # the line without its gain
    low = res["low_peak"]
    assert low["series_left_out"] == 1 and low["patients"] == [stroke[0]]
    assert tuple(low["series"].iloc[0][["person_id", "key", "foot"]]) == (stroke[0], "trial1", "R")
    assert low["series"].iloc[0]["peak_bw"] == pytest.approx(0.4)
    pg, pl = res["person_grf"].set_index("person_id"), low["person_grf"].set_index("person_id")
    assert pl.loc[stroke[0], "samples"] == pg.loc[stroke[0], "samples"] - low["samples_left_out"]
    assert pl.loc[stroke[1], "samples"] == pg.loc[stroke[1], "samples"]
    assert low["summary"]["grf_rmse_pct_bw"]["newton"]["point"] != pytest.approx(
        summ["grf_rmse_pct_bw"]["newton"]["point"])


@needs_manifest
def test_the_development_row_applies_the_sealed_read_flight_code(synthetic_trials):
    """run_dev scores the model's (bw, contact) pair through g4.flight_predictor, the sealed read's code (rule
    section 2 (a)): 0 BW on both feet where both contact probabilities are below 0.5 (one foot below is not
    flight), the frames counted; the per-trial table and the able-bodied row through the same code."""
    stroke, dev, train = synthetic_trials

    class Flying(TruthPredictor):
        def grf_contact(self, trial):
            p = np.full(trial.obs.shape, 0.9)
            p[:30] = 0.2                                    # both feet: flight
            p[30:40, 0] = 0.2                               # one foot only: not flight
            return trial.obs.copy(), p

    res = S.run_dev(Flying(), stroke, dev, train, ["fake:stroke_ab:S801"], metric3d=False)
    flown = {"newton": g4.newton_predict, "model": lambda tr: g4.apply_flight(*Flying().grf_contact(tr))[0]}
    want = g4.grf_person_table(res["gtr"], flown)
    np.testing.assert_allclose(res["person_grf"]["model_rmse_pct_bw"], want["model_rmse_pct_bw"])
    np.testing.assert_allclose(res["person_grf"]["model_r"], want["model_r"])
    assert (res["person_grf"]["model_rmse_pct_bw"] > 1.0).all()
    assert res["stats"]["grf_model_flight_frames_plate"] == 30 * len(res["gtr"]) == 180
    assert res["stats"]["grf_model_frames_plate"] == sum(len(tr.t) for tr in res["gtr"])
    tg = res["trial_grf"]
    assert (tg["model_rmse_pct_bw"] > 1.0).all()
    for tr in res["gtr"]:
        y = flown["model"](tr)
        assert (y[:30] == 0).all() and (y[30:] == tr.obs[30:]).all()
    ab = res["able"]
    want_ab = g4.grf_person_table(ab["trials"], flown)
    assert ab["summary"]["grf_rmse_pct_bw"]["model"]["point"] == pytest.approx(want_ab["model_rmse_pct_bw"].mean())
    assert ab["stats"]["grf_model_flight_frames_plate"] == 30 * len(ab["trials"]) == 60


@needs_manifest
def test_template_persons_among_the_scored_persons_are_refused(synthetic_trials):
    stroke, dev, train = synthetic_trials
    with pytest.raises(C.Refused):
        S.run_dev(TruthPredictor(), stroke, dev, train + dev[:1], [], metric3d=False)


# ---------------------------------------------------------------- lockbox refusals
@needs_manifest
@pytest.mark.parametrize("which", ["stroke", "gait120", "able_bodied"])
def test_a_lockbox_person_is_refused_before_any_record_opens(monkeypatch, which):
    calls = []
    real = C.load

    def counting(*a, **k):
        calls.append(a)
        return real(*a, **k)

    monkeypatch.setattr(C, "load", counting)
    locked_stroke, locked_gait = g4.stroke_patients(lockbox=True), g4.gait120_persons("lockbox")
    assert locked_stroke and locked_gait
    stroke = locked_stroke[:1] if which == "stroke" else []
    gait = locked_gait[:1] if which == "gait120" else []
    able = locked_stroke[-1:] if which == "able_bodied" else []
    with pytest.raises(C.Refused, match="lockbox"):
        S.run_dev(TruthPredictor(), stroke, gait, [], able, metric3d=True)
    assert calls == []


def _paired(person, source="gait120", T=1200, key="level_walking_trial01", root=True):
    X = walking_world(T)
    c3 = (X - X[:, :1]) / 0.5
    rec = {"coords": c3.astype(np.float32), "coord_mask": np.ones(3, bool), "joint_mask": np.ones(15, bool),
           "valid_mask": np.ones((T, 15), bool), "fps": 100.0, "scale": 0.5, "dim": 3, "modality": "3d_meter",
           "source": source, "subject_ns": person, "person_id": person, "key": key}
    pr = SRC.PairedRecord(source=source, person_id=person, subject_ns=person, key=key, streams={"pose3d": rec},
                          t={"pose3d": np.arange(T) / 100.0}, roles={"pose3d": "input"},
                          meta={"pose3d": {"layout": "mocap3d", "unit_to_si": 1.0}})
    if root:
        W.attach_root(pr, X[:, 0], np.ones(T, bool))
    return pr, X


def test_the_predictor_refuses_a_lockbox_record_before_its_root_is_read():
    from mdfm.data import window_store as WS
    locked = sorted(WS.lockbox_persons())
    assert locked
    reads = []
    pred = S.G4Predictor(model=None, attach_root=lambda pr: reads.append(pr) or pr)
    pr, _ = _paired(locked[0], root=False)
    with pytest.raises(C.Refused):
        pred.predict_pose(pr, ("grf",))
    with pytest.raises(C.Refused):
        pred.metric3d_trial(pr)
    assert reads == []


# ---------------------------------------------------------------- records left out
def test_a_record_shorter_than_a_window_is_left_out_and_counted(no_tables):
    from mdfm.eval import infer as I
    assert I.spans(99, 100.0) == [] and I.spans(100, 100.0) == [(0, 100)]      # the 1 s minimum, infer.MIN_S
    pred = S.G4Predictor(model=None, attach_root=no_root_read)               # no window: the model is never called
    pr, _ = _paired("gait120:FIX03", T=92, key="level_walking_trial05")
    assert pred.predict_pose(pr, ("grf",)) is None
    want = {"source": "gait120", "person_id": "gait120:FIX03", "key": "level_walking_trial05", "stream": "pose",
            "reason": "pose_shorter_than_a_window", "samples": 92, "fps": 100.0, "seconds": 0.92, "min_samples": 100}
    assert pred.left_out == [want] and pred.log == []
    T = 92
    trial = g4.EMGTrial(person=pr.person_id, key=pr.key, t=pr.t["pose3d"], frames=np.arange(T),
                        phase=np.linspace(0.0, 1.0, T), obs=np.zeros((T, 4)), valid=np.ones((T, 4), bool), record=pr)
    y = pred.emg(trial)
    assert y.shape == (T, 4) and np.isnan(y).all() and pred.left_out == [want, want]
    rep = S.left_out_report(pred.left_out)
    assert rep["n"] == 2 and rep["counts"] == {"gait120": {"pose_shorter_than_a_window": 2}}
    assert rep["by_line"] == {"emg": 2} and list(rep["records"][0]) == list(S.LEFT_OUT_COLUMNS)
    with pytest.raises(ValueError):
        pred.leave_out(pr, "pose", "too_short", 92, 100.0)                   # an unknown reason is refused


def test_a_short_stroke_trial_reads_nan_and_its_camera_is_left_out(monkeypatch, no_tables):
    person = "stroke_gait:stroke:S991"
    pr1, X = _paired(person, source="stroke_gait_vancriekinge", T=60, key="gait_FIX2")
    pr2, _ = _paired(person, source="stroke_gait_vancriekinge", T=60, key="gait_FIX3")
    pr2.streams["pose3d"]["valid_mask"][:, 0] = False                        # the pelvis never valid
    seen = []

    def fake_paired(source, persons, allow_lockbox=False, stats=None):
        assert not allow_lockbox and list(persons) == [person]
        seen.append(source)
        return iter([pr1, pr2])

    monkeypatch.setattr(g4, "paired_records", fake_paired)
    pred = S.G4Predictor(model=None, attach_root=no_root_read, metric3d_persons=[person])
    for pr in (pr1, pr2):
        trial = g4.GRFTrial(person=person, key=pr.key, fps=100.0, t=pr.t["pose3d"], world=X, obs=np.zeros((60, 2)),
                            obs_valid=np.ones((60, 2), bool), mass_kg=70.0, mass_from="c3d", sex="M")
        y = pred.grf(trial)
        assert y.shape == (60, 2) and np.isnan(y).all() and pred.flight(trial) is None
        bw, contact = pred.grf_contact(trial)                                # NaN in both: no sample scored
        assert bw.shape == contact.shape == (60, 2) and np.isnan(bw).all() and np.isnan(contact).all()
        assert np.isnan(g4.flight_predictor(pred.grf_contact)(trial)).all()
    assert seen == [g4.STROKE] and pred.log == [] and pred.m3d == []
    got = [(r["key"], r["stream"], r["reason"], r["samples"], r["fps"]) for r in pred.left_out]
    assert got == [("gait_FIX2", "pose", "pose_shorter_than_a_window", 60, 100.0),
                   ("gait_FIX2", "camera", "camera_shorter_than_a_window", 18, 30.0),
                   ("gait_FIX3", "pose", "pose_shorter_than_a_window", 60, 100.0),
                   ("gait_FIX3", "camera", "camera_no_pelvis", 60, 100.0)]
    rep = S.left_out_report(pred.left_out, stroke=[person])
    assert rep["counts"] == {"stroke_gait_vancriekinge": {"camera_no_pelvis": 1, "camera_shorter_than_a_window": 1,
                                                          "pose_shorter_than_a_window": 2}}
    assert rep["by_line"] == {"grf": 2, "metric3d": 2}
    assert S.left_out_report(pred.left_out, able=[person])["by_line"] == {"grf_able_bodied": 2, "metric3d": 2}


class LeavingOutPredictor(TruthPredictor):
    """The truth as the prediction, except the (person, key) trials named: they read NaN and are recorded as left
    out, as G4Predictor records a record without a window."""

    def __init__(self, out):
        super().__init__()
        self.out = set(out)
        self.left_out = []

    def _left(self, source, trial):
        k = (trial.person, trial.key)
        if k not in self.out:
            return False
        if not any((r["person_id"], r["key"]) == k for r in self.left_out):     # grf() is called once per table
            n = len(trial.t)
            self.left_out.append({"source": source, "person_id": trial.person, "key": trial.key, "stream": "pose",
                                  "reason": "pose_shorter_than_a_window", "samples": n, "fps": 100.0,
                                  "seconds": n / 100.0, "min_samples": 100})
        return True

    def grf_contact(self, trial):
        if self._left(g4.STROKE, trial):                                        # as G4Predictor: NaN in both
            return np.full(trial.obs.shape, np.nan), np.full(trial.obs.shape, np.nan)
        return super().grf_contact(trial)

    def emg(self, trial):
        return np.full(trial.obs.shape, np.nan) if self._left(g4.GAIT120, trial) else trial.obs.copy()


@needs_manifest
def test_left_out_trials_keep_every_row_on_the_same_samples(synthetic_trials):
    stroke, dev, train = synthetic_trials
    out = [(stroke[0], "trial1"), (dev[0], "level_walking_trial01"), (dev[2], "level_walking_trial01"),
           (dev[2], "level_walking_trial02")]
    res = S.run_dev(LeavingOutPredictor(out), stroke, dev, train, [], metric3d=False)
    # GRF: the trial left out has no scored sample, for the Newton floor as for the model
    pg = res["person_grf"].set_index("person_id")
    assert pg.loc[stroke[0], "samples"] * 2 == pg.loc[stroke[1], "samples"] > 0
    kept = [tr for tr in res["gtr"] if (tr.person, tr.key) == (stroke[0], "trial0")]
    alone = g4.grf_person_table(kept, {"newton": g4.newton_predict})
    assert pg.loc[stroke[0], "newton_rmse_pct_bw"] == pytest.approx(alone["newton_rmse_pct_bw"].iloc[0])
    assert res["summary"]["grf_rmse_pct_bw"]["model"]["point"] == pytest.approx(0.0, abs=1e-12)
    tg = res["trial_grf"]
    assert (tg.loc[(tg["person_id"] == stroke[0]) & (tg["key"] == "trial1"), "samples"] == 0).all()
    # EMG: the trials left out are gone for the template and the model alike; g4's tables on the kept trials
    te = res["trial_emg"]
    assert res["emg_trials_all"] == 6 and S.n_trials(te) == 3
    assert not set(zip(te["person_id"], te["key"])) & set(out)
    assert te["template_r"].notna().all() and np.allclose(te["model_r"], 1.0)
    pe = res["person_emg"].set_index("person_id")
    assert list(pe.index) == [dev[0], dev[1]] and pe.loc[dev[0], "pairs"] == 4 and pe.loc[dev[1], "pairs"] == 8
    want = g4.summary(res["person_grf"], g4.emg_person_table(te, ["template", "model"]), ["newton", "model"],
                      ["template", "model"], emg_gain=True)
    for n in ("template", "model"):
        assert res["summary"]["emg_r"][n] == want["emg_r"][n] and want["emg_r"][n]["n_persons"] == 2
    assert res["summary"]["emg_r_gain"] == want["emg_r_gain"] and want["emg_r_gain"]["model"]["n_persons"] == 2
    assert res["emg_template_all_trials"]["n_persons"] == 3
    assert S.unscored(dev, res["person_emg"], "template_r") == [dev[2]]
    assert S.unscored(stroke, res["person_grf"], "newton_rmse_pct_bw") == []
    lines = g4.gate_results(res["summary"], "model", "r" * 64)
    assert lines[2].n_persons == 2 and lines[2].point == pytest.approx(1.0)
    # counted per source and reason, and per line
    lo = res["left_out"]
    assert lo["n"] == 4 and lo["counts"] == {"gait120": {"pose_shorter_than_a_window": 3},
                                             "stroke_gait_vancriekinge": {"pose_shorter_than_a_window": 1}}
    assert lo["by_line"] == {"emg": 3, "grf": 1}


@needs_manifest
def test_nothing_left_out_keeps_g4_evaluate_rows(synthetic_trials):
    stroke, dev, train = synthetic_trials
    res = S.run_dev(LeavingOutPredictor([]), stroke, dev, train, [], metric3d=False)
    assert res["left_out"]["n"] == 0 and res["left_out"]["counts"] == {} and res["emg_trials_all"] == 6
    assert S.n_trials(res["trial_emg"]) == 6 and res["emg_template_all_trials"] == res["summary"]["emg_r"]["template"]


# ---------------------------------------------------------------- disjointness
def test_scored_persons_disjoint_from_the_checkpoint_lists(tmp_path):
    run = tmp_path / "run"
    (run / "persons").mkdir(parents=True)
    (run / "persons" / "gait120.txt").write_text("gait120:S002\ngait120:S003\n", encoding="utf-8")
    (run / "persons" / "stroke_gait_vancriekinge.txt").write_text("stroke_gait:able_bodied:S001\n", encoding="utf-8")
    (run / "persons" / "._gait120.txt").write_bytes(b"\x00\x05\x16\x07")
    lists = S.person_lists(run / "ckpt_B_000010.pt")
    assert lists == {"gait120": frozenset({"gait120:S002", "gait120:S003"}),
                     "stroke_gait_vancriekinge": frozenset({"stroke_gait:able_bodied:S001"})}
    scored = {"gait120_dev": ["gait120:S021"], "stroke_dev": ["stroke_gait:stroke:S003"]}
    assert S.assert_disjoint(scored, lists) == {"gait120_dev": 0, "stroke_dev": 0}
    with pytest.raises(C.Refused, match="gait120:S003"):
        S.assert_disjoint({"gait120_dev": ["gait120:S021", "gait120:S003"]}, lists)
    with pytest.raises(C.Refused, match="able_bodied:S001"):
        S.assert_disjoint({"able_bodied_dev": ["stroke_gait:able_bodied:S001"]}, {"checkpoint:x": lists[
            "stroke_gait_vancriekinge"]})
    with pytest.raises(C.Refused):
        S.person_lists(tmp_path / "elsewhere" / "ckpt_B_000010.pt")


@needs_manifest
@pytest.mark.skipif(not os.path.isdir(RUN1_PERSONS), reason="seed-1 run folder not found")
def test_the_development_sets_are_disjoint_from_the_seed_1_lists():
    lists = S.person_lists(os.path.join(os.path.dirname(RUN1_PERSONS), "ckpt_B_002000.pt"))
    scored = {"stroke_dev": g4.stroke_patients(lockbox=False), "gait120_dev": g4.gait120_persons("dev"),
              "able_bodied_dev": S.able_bodied_dev()}
    assert len(scored["stroke_dev"]) == 40 and len(scored["gait120_dev"]) == 19 and len(scored["able_bodied_dev"]) == 28
    assert S.assert_disjoint(scored, lists) == dict.fromkeys(scored, 0)
    counts = S.check_sets({**scored, "gait120_template": g4.gait120_persons("train")})
    assert counts["gait120_template"] == 77


# ---------------------------------------------------------------- helpers
def test_frames_go_to_the_most_central_window():
    t = np.arange(1600) / 100.0
    win, rel = S.assign_frames(t, [0.0, 5.0], [1000, 1000], 100.0)
    assert (win[t < 7.495] == 0).all() and (win[(t > 7.505) & (t <= 14.99 + 1e-9)] == 1).all()
    assert (win[t > 14.996] == -1).all()
    np.testing.assert_allclose(rel[win == 1], t[win == 1] - 5.0, atol=1e-9)
    w2, _ = S.assign_frames([0.0, 2.0], [0.0, 0.0], [300, 300], 100.0)
    assert list(w2) == [0, 0]                                                # a tie keeps the earlier window


def test_hold_fill_and_frame_interpolation():
    x = np.zeros((6, 3, 3))
    x[:, :, 0] = np.arange(6)[:, None]
    x[2, 1] = np.nan
    x[4:, 2] = np.nan
    y = S.hold_fill(x)
    assert y[2, 1, 0] in (1.0, 3.0) and (y[4:, 2, 0] == 3.0).all() and np.isfinite(y).all()
    x[:, 0] = np.nan
    assert S.hold_fill(x) is None
    t = np.array([0.0, 0.01, 0.02])
    v = np.array([[0.0], [1.0], [np.nan]])
    np.testing.assert_allclose(S.interp_frames(v, t, [0.0, 0.005, 0.01])[:, 0], [0.0, 0.5, 1.0])
    assert np.isnan(S.interp_frames(v, t, [0.015])).all()


def test_camera_seed_is_the_sha256_of_the_trial_key():
    h = hashlib.sha256("stroke_gait:stroke:S003\ngait_BWA6\nG4v1".encode("utf-8")).hexdigest()
    assert S.camera_seed("stroke_gait:stroke:S003", "gait_BWA6") == int(h[:16], 16) & ((1 << 63) - 1)
    assert S.camera_seed("stroke_gait:stroke:S004", "gait_BWA6") != S.camera_seed("stroke_gait:stroke:S003",
                                                                                 "gait_BWA6")


# ---------------------------------------------------------------- the predictor on the CPU
@pytest.fixture()
def no_tables(monkeypatch):
    monkeypatch.setattr(SRC, "demographic", lambda source, ns: {"sex": "M", "age": 61.0})


def no_root_read(pr):
    raise AssertionError("the root track is already attached: no root read")


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(5)
    model = M.MFM(M.TINY)
    model.eval()
    model.requires_grad_(False)
    return model


@needs_ma
def test_the_predictor_runs_on_the_model_fixture_on_the_cpu(tiny):
    pred = S.G4Predictor(tiny, refine_steps=2, points=64, batch_windows=2, seed=1)
    wd = MF.make_pose_window(3.0, seed=0, person="gait120:FIX01", source="gait120", sex=1)
    st = wd.streams[0]
    bodies = pred.infer([wd], "gait120:FIX01/fixture", "pose")
    assert len(bodies) == 1 and bodies[0].rot6d.shape[0] == len(st.t)
    t = wd.t0 + st.t.astype(np.float64)
    res, win = pred.frames([wd], bodies, t, ("grf", "emg", "joints15"))
    T = len(t)
    assert (win == 0).all() and res["grf"].shape == (T, 2) and res["emg"].shape == (T, 4)
    assert res["joints15"].shape == (T, 15, 3) and np.isfinite(res["joints15"]).all()
    assert np.isfinite(res["grf"]).all() and (res["grf"][res["flight"]] == 0).all()
    # the sealed read's pair: the BW before the flight rule and the contact probabilities; the harness's flight
    # code on them gives the model's own flight flag and its GRF exactly
    assert res["bw"].shape == res["contact"].shape == (T, 2) and np.isfinite(res["bw"]).all()
    assert ((res["contact"] > 0) & (res["contact"] < 1)).all()
    y, fl = g4.apply_flight(res["bw"], res["contact"])
    np.testing.assert_array_equal(fl, res["flight"])
    np.testing.assert_array_equal(y, res["grf"])
    modelled = pred.emg_modelled
    assert modelled.any() and np.isfinite(res["emg"][:, modelled]).all() and np.isnan(res["emg"][:, ~modelled]).all()
    assert pred.log[-1]["kind"] == "pose" and pred.log[-1]["windows"] == 1
    pred2 = S.G4Predictor(tiny, refine_steps=2, points=64, batch_windows=2, seed=1)
    again, _ = pred2.frames([wd], pred2.infer([wd], "gait120:FIX01/fixture", "pose"), t, ("grf",))
    np.testing.assert_array_equal(again["grf"], res["grf"])                  # deterministic under its seed

    class FlippedFlight:
        """The model with its own flight flag inverted: the contact probabilities no longer read it."""

        def __getattr__(self, name):
            return getattr(tiny, name)

        def render(self, body, batch, nuis):
            out = tiny.render(body, batch, nuis)
            out["force"]["flight"] = ~out["force"]["flight"]
            return out

    bad = S.G4Predictor(FlippedFlight(), refine_steps=2, points=64, batch_windows=2, seed=1)
    with pytest.raises(ValueError, match="flight flag"):
        bad.frames([wd], bodies, t, ("grf",))


@needs_ma
def test_the_pose_path_of_a_record_with_its_root(tiny, no_tables):
    pred = S.G4Predictor(tiny, refine_steps=1, points=32, batch_windows=1, seed=2, attach_root=no_root_read)
    pr, _ = _paired("gait120:FIX02", T=1200)
    res = pred.predict_pose(pr, ("grf", "emg"))
    assert res["grf"].shape == (1200, 2) and res["emg"].shape == (1200, 4) and np.isfinite(res["grf"]).all()
    win = res["windows"]
    assert list(np.unique(win)) == [0, 1] and win[0] == 0 and win[-1] == 1     # spans (0, 1000) and (200, 1000)
    assert (win[:599] == 0).all() and (win[701:] == 1).all()
    assert pred.log[-1]["windows"] == 2 and 0.0 <= pred.log[-1]["flight_share"] <= 1.0
    stream, t64 = S.pose_stream(pr)
    assert stream.meta["layout"] == "mocap3d" and stream.meta["source"] == "gait120" and "root" in stream.meta
    np.testing.assert_allclose(stream.meta["root"], pr.meta["pose3d"]["root"])
    np.testing.assert_array_equal(g4.apply_flight(res["bw"], res["contact"])[0], res["grf"])


@needs_ma
def test_grf_contact_of_a_stroke_trial_is_the_sealed_read_pair(tiny, no_tables, monkeypatch):
    """grf_contact of a GRF trial reads the person's pairs once (as grf does) and gives (bw, contact) at the trial's
    pose frames; g4.flight_predictor on it equals grf, the model's own flight flag applied; the metric-3D trials of
    a person without GRF trials run through metric3d_person."""
    person = "stroke_gait:stroke:S992"
    pr, X = _paired(person, source="stroke_gait_vancriekinge", T=400, key="gait_FIX4")
    reads = []

    def fake_paired(source, persons, allow_lockbox=False, stats=None):
        assert not allow_lockbox and list(persons) == [person] and source == g4.STROKE
        reads.append(source)
        return iter([pr])

    monkeypatch.setattr(g4, "paired_records", fake_paired)
    pred = S.G4Predictor(tiny, refine_steps=1, points=32, batch_windows=2, seed=4, attach_root=no_root_read)
    trial = g4.GRFTrial(person=person, key=pr.key, fps=100.0, t=pr.t["pose3d"], world=X, obs=np.zeros((400, 2)),
                        obs_valid=np.ones((400, 2), bool), mass_kg=70.0, mass_from="c3d", sex="M")
    bw, contact = pred.grf_contact(trial)
    assert bw.shape == contact.shape == (400, 2) and np.isfinite(bw).all() and np.isfinite(contact).all()
    stats = collections.Counter()
    y = g4.flight_predictor(pred.grf_contact, stats)(trial)
    np.testing.assert_array_equal(y, pred.grf(trial))
    assert stats["grf_model_frames_plate"] == 400
    assert stats["grf_model_flight_frames_plate"] == int(pred.flight(trial).sum())
    assert reads == [g4.STROKE] and pred.m3d == []                           # read once; not a metric-3D person
    pred.metric3d_person(person)
    assert reads == [g4.STROKE] * 2 and len(pred.m3d) == 1 and pred.m3d[0]["key"] == pr.key


@needs_ma
def test_the_metric3d_round_trip_through_one_seeded_camera(tiny, no_tables):
    pred = S.G4Predictor(tiny, refine_steps=1, points=32, batch_windows=2, seed=3, attach_root=no_root_read)
    pr, X = _paired("stroke_gait:stroke:S990", source="stroke_gait_vancriekinge", T=400, key="gait_FIX1")
    world = g4.world_joints(pr.streams["pose3d"], pr.meta["pose3d"]["root"].astype(np.float64),
                            pr.meta["pose3d"]["root_valid"])
    np.testing.assert_allclose(world, X, atol=1e-5)
    seed = S.camera_seed(pr.person_id, pr.key)
    a = S.camera_stream(world, pr.t["pose3d"], seed, pr.source)
    b = S.camera_stream(world, pr.t["pose3d"], seed, pr.source)
    c = S.camera_stream(world, pr.t["pose3d"], seed + 1, pr.source)
    assert a[0].fps == 30.0 and a[0].meta["layout"] == "camera1" and a[2]["frames"] == len(a[1]) == 120
    np.testing.assert_array_equal(a[0].values, b[0].values)
    assert a[2]["camera_KRt"] == b[2]["camera_KRt"] != c[2]["camera_KRt"]
    entry = pred.metric3d_trial(pr)
    assert entry is not None and entry["pred"].shape == (400, 15, 3) and pred.log[-1]["kind"] == "camera"
    m_trial, m_person = S.metric3d_tables(pred.m3d)
    assert len(m_trial) == 1 and len(m_person) == 1
    assert np.isfinite(m_person["mpjpe_mm"]).all() and np.isfinite(m_person["pa_mpjpe_mm"]).all()
    assert m_person["pa_mpjpe_mm"].iloc[0] <= m_person["mpjpe_mm"].iloc[0] + 1e-6
    line = S.mean_line(m_person["mpjpe_mm"])
    assert line["n_persons"] == 1 and np.isfinite(line["point"])


def test_yaw_alignment_undoes_a_turn_about_y():
    X = walking_world(50)
    from mdfm.gates import g1
    P = g1.turn_y(X - X[:, :1], 0.7)
    out = S.yaw_aligned(P, X, np.zeros(50, np.int64))
    np.testing.assert_allclose(out, X - X[:, :1], atol=1e-9)
    assert g4.mpjpe(out, X - X[:, :1]) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------- the stroke set of a stage-2 checkpoint (task S2.10)
def _fold_rows():
    """{fold: sorted ids} of the stroke patients of folds_proposed.csv (group stroke)."""
    import csv
    out = {"dev": [], "train": []}
    with open(S.STROKE_FOLD_CSV, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["source"] == g4.STROKE and r["group"] == "stroke":
                out[r["fold"]].append(r["person_id"])
    return {k: sorted(v) for k, v in out.items()}


needs_fold = pytest.mark.skipif(not os.path.exists(S.STROKE_FOLD_CSV), reason="stage-2 fold table not found")


@needs_manifest
@needs_fold
def test_the_stroke_fold_is_the_8_of_the_proposed_fold():
    ids, info = S.stroke_set("fold")
    assert len(ids) == 8 and ids == sorted(ids) and ids == _fold_rows()["dev"]
    assert info["sha256"] == S.STROKE_FOLD_SHA256 == SRC.person_hash(ids) and info["set"] == "fold"
    assert "stroke_gait:stroke:S047" in ids                       # the patient without a pose and plate pair
    pool, info_all = S.stroke_set("all")
    assert len(pool) == 40 and set(ids) <= set(pool) and info_all["set"] == "all"
    assert pool == list(g4.stroke_patients(lockbox=False))       # the default set is the run before S2.10
    assert not set(pool) & set(g4.stroke_patients(lockbox=True))
    assert set(_fold_rows()["train"]) == set(pool) - set(ids)     # the 32 a stage-2 run trains on
    with pytest.raises(ValueError):
        S.stroke_set("some")


@needs_fold
def test_a_changed_stroke_fold_is_refused(tmp_path, monkeypatch):
    import shutil
    rows = open(S.STROKE_FOLD_CSV, encoding="utf-8").read().splitlines()
    dev = _fold_rows()["dev"]
    swapped = [r.replace(f",{dev[0]},dev", f",{dev[0]},train") for r in rows]
    bad = tmp_path / "folds_proposed.csv"
    bad.write_text("\n".join(swapped) + "\n", encoding="utf-8")
    shutil.copy(S.STROKE_FOLD_SUMMARY, tmp_path / "summary.json")
    with pytest.raises(ValueError, match="hashes"):
        S.stroke_fold(str(bad), str(tmp_path / "summary.json"))
    with pytest.raises(ValueError, match="hashes"):                 # the pinned SHA-256 alone also refuses
        S.stroke_fold(expected="0" * 64)
    # rows of another group or source in the dev fold are not stroke patients of the fold
    extra = tmp_path / "folds_extra.csv"
    extra.write_text(chr(10).join(rows + [f"{g4.STROKE},able_bodied,stroke_gait:able_bodied:S901,dev",
                                          "weargait_pd,PD,weargait:NLS901,dev"]) + chr(10), encoding="utf-8")
    assert S.stroke_fold(str(extra), str(tmp_path / "summary.json"))[0] == dev
    monkeypatch.setattr(S, "lockbox_names", lambda: frozenset({dev[3]}))
    with pytest.raises(C.Refused, match="lockbox"):
        S.stroke_fold()


@needs_fold
def test_a_fold_person_outside_the_non_lockbox_stroke_patients_is_refused(monkeypatch):
    folds = _fold_rows()
    pool = sorted(set(folds["train"]) | set(folds["dev"]))
    real = g4.stroke_patients                                          # the lockbox list stays the real one
    monkeypatch.setattr(g4, "stroke_patients", lambda lockbox=False: real(lockbox=True) if lockbox else list(pool))
    assert S.stroke_set("fold")[0] == folds["dev"]                    # the fold inside the pool: read
    monkeypatch.setattr(g4, "stroke_patients", lambda lockbox=False: real(lockbox=True) if lockbox else
                        [p for p in pool if p != folds["dev"][0]])
    with pytest.raises(ValueError, match="outside the non-lockbox stroke patients"):
        S.stroke_set("fold")


class _MainPredictor(TruthPredictor):
    """TruthPredictor with the attributes main() reads after run_dev."""

    def __init__(self, model, **kw):
        super().__init__()
        self.log, self.read_stats, self.left_out = [], collections.Counter(), []
        self.emg_modelled = [True] * len(g4.MUSCLES)
        self.win_s, self.hop_s = 10.0, 5.0

    def flight(self, trial):
        return np.zeros(len(trial.t), dtype=bool)


def _main_env(tmp_path, monkeypatch, trained_stroke):
    """A stub checkpoint in a run folder whose persons lists hold trained_stroke and two fake gait120 persons;
    load_checkpoint and the predictor replaced; gait120 sets fake (the synthetic_trials fixture fakes the trials)."""
    from mdfm.eval import infer as I
    run = tmp_path / "runs" / "s2run"
    (run / "persons").mkdir(parents=True)
    lists = {"stroke_gait_vancriekinge": sorted(trained_stroke), "gait120": ["fake:gait120:S990"]}
    for s, ids in lists.items():
        (run / "persons" / f"{s}.txt").write_text("".join(f"{p}\n" for p in ids), encoding="utf-8")
    ck = run / "ckpt_C_000020.pt"

    def load(path, weights="ema", device="cpu", **kw):
        return I.Loaded(model=torch.nn.Linear(1, 1), path=str(ck), name="s2run_ckpt_C_000020", sha256="a" * 64,
                        weights=weights, step=20, phase="C", phase_steps=(5, 10, 20), config_json="{}",
                        config_sha256="b" * 64, persons=lists, persons_sha256={"all": "c" * 64}, hashes={}, drift=[])

    fake = {"dev": ["fake:gait120:S901", "fake:gait120:S902", "fake:gait120:S903"],
            "train": ["fake:gait120:S951", "fake:gait120:S952"], "lockbox": []}
    monkeypatch.setattr(S.I, "load_checkpoint", load)
    monkeypatch.setattr(S, "G4Predictor", _MainPredictor)
    monkeypatch.setattr(g4, "gait120_persons", lambda fold: list(fake[fold]))
    return ck


@needs_manifest
@needs_fold
def test_main_with_the_fold_scores_the_8_and_refuses_a_checkpoint_trained_on_one(tmp_path, monkeypatch,
                                                                                  synthetic_trials):
    import json
    folds = _fold_rows()
    ck = _main_env(tmp_path, monkeypatch, folds["train"])
    base = ["--checkpoint", str(ck), "--device", "cpu", "--no-secondary", "--no-metric3d"]
    out = tmp_path / "out_fold"
    S.main(base + ["--stroke-set", "fold", "--out", str(out)])
    with open(out / "g4.json", encoding="utf-8") as f:
        doc = json.load(f)
    st = doc["sets"]
    assert st["stroke_set"]["set"] == "fold" and st["stroke_set"]["sha256"] == S.STROKE_FOLD_SHA256
    assert st["stroke_dev"] == 8 and st["stroke_dev_sha256"] == S.STROKE_FOLD_SHA256
    assert st["stroke_scored"] == 8 and st["stroke_unscored"] == []          # every fold patient has fake trials
    assert st["disjoint_from_training_lists"]["stroke_dev"] == 0
    import pandas as pd
    pp = pd.read_csv(out / "per_person_grf.csv", encoding="utf-8")
    assert sorted(pp["person_id"]) == folds["dev"]
    assert "stroke set fold" in (out / "g4.md").read_text(encoding="utf-8")
    # the default set reads all 40, of which the checkpoint trained on 32: refused, with the hint
    out_all = tmp_path / "out_all"
    with pytest.raises(C.Refused, match="--stroke-set fold"):
        S.main(base + ["--out", str(out_all)])
    assert not out_all.exists()
    # a checkpoint trained on one patient of the fold is refused before anything is written
    for who in (folds["dev"][0], "stroke_gait:stroke:S047"):
        ck2 = _main_env(tmp_path / who.replace(":", "_"), monkeypatch, folds["train"] + [who])
        out2 = tmp_path / ("out_" + who.replace(":", "_"))
        with pytest.raises(C.Refused, match=who):
            S.main(["--checkpoint", str(ck2), "--device", "cpu", "--no-secondary", "--no-metric3d",
                    "--stroke-set", "fold", "--out", str(out2)])
        assert not out2.exists()


def test_the_default_output_folder_of_the_fold():
    class L:
        name, weights = "run_ckpt_C_000020", "ema"
    assert S.default_out(L, "T").replace("\\", "/") == "T/reports/gates/g4/dev_run_ckpt_C_000020"
    assert S.default_out(L, "T", "fold").replace("\\", "/").endswith("dev_run_ckpt_C_000020_strokefold")
    L.weights = "model"
    assert S.default_out(L, "T", "fold").replace("\\", "/").endswith("dev_run_ckpt_C_000020_strokefold_model")
    a = S.parse(["--checkpoint", "x.pt"])
    assert a.stroke_set == "all"
    with pytest.raises(SystemExit):
        S.parse(["--checkpoint", "x.pt", "--stroke-set", "lockbox"])
