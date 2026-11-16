"""Tests of the inference adapters, mdfm/eval/infer.py (spec docs/MFM_MODEL_CODE_SPEC_2026-09-28.md, 2.7 and 6, plan
task B4.19), on a tiny model saved by mdfm/train/checkpoint.py in a temporary run folder:

- load_checkpoint goes through checkpoint.load (a tampered file is refused), restores the EMA or the raw weights
  strictly, freezes and audits the model, and reports the phase, step and training persons of the save;
- spans cover every sample of a record with full windows every hop, the last one aligned on the end;
- record_windows band-copies a whole IMU record before the cut, masks the magnetometer, carries the source's layout
  and the task of the record key, and returns windows of n / fps s;
- infer returns bodies on the 100 Hz clock (one knot per sample time), deterministic under its seed, pools the shape
  per group, draws the chosen joint in the pelvis-heading frame (the torch form equals mdfm.gates.g1.heading_points);
  a short window packed with a longer one keeps its own knots and equals the window run alone; every window's root
  sits on the decoder's anchor at its own centre knot;
- CODY windows and records are refused without the evaluation flag;
- G3Predictor returns a stitched g3.TrialBody that the G3 readouts accept;
- G1Predictor gives the prediction dict of mdfm.gates.g1 in blocks of whole takes, fit_conformal fits both wrist
  predictives on calib only, each factor equal to the harness's g1.fit_conformal of that key, and
  g1.score_window scores a prediction;
- scripts/eval_g3_dev.py: an IC predictor with no IC in the match window gets an unfitted offset (0, n 0), and a
  checkpoint that closes phase A with no matched stride reads 'undetermined' under the stop rule, never 'no';
- mdfm.eval.infer imports mdfm.gates while mdfm.gates.g1 imports mdfm.eval.conformal: both import orders work;
- saved conformal factors (B4.20): a fit on every calib window is saved to conformal_<sha12>.json beside the
  checkpoint, written once; a predictor never reads it unless asked (use_conformal_file), and a file of another
  checkpoint, of other predictive settings or of other calib inputs is refused; a fit on a subset of the calib
  windows or a non-finite factor is never saved;
- scripts/eval_g1_dev.py --secondary-rows: the four model rows of G1_RULE section 8 run through the row functions of
  scripts/probe_g1_decompose.py on a fake G1 set, the main pass reproduces the harness's G1a on the same blocks and
  seeds, analysis by synthesis is labelled partial, and the markdown of the rows renders.
"""
import copy
import math
import os
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch

from mdfm import paths
from mdfm.data import canonical as C
from mdfm.data import windows as W
from mdfm.eval import infer as I
from mdfm.gates import g1, g3
from mdfm.model import mfm as M
from mdfm.train import checkpoint as CK
from mdfm.train import config as CF
from mdfm.train import provenance as PV

import model_fixture as MF

HAVE_MANIFEST = os.path.exists(C.manifest_path())
HAVE_MA = os.path.exists(paths.smplh_model("neutral"))
needs_all = pytest.mark.skipif(not (HAVE_MANIFEST and HAVE_MA), reason="tiers records file or SMPL+H model not found")
TREE = paths.TREE


# ---------------------------------------------------------------- a tiny checkpoint
@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    torch.manual_seed(3)
    model = M.MFM(M.TINY)
    ema = copy.deepcopy(model)
    with torch.no_grad():
        for p in ema.parameters():
            p.add_(0.01 * torch.randn_like(p))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    persons = {"amass": ["fake:amass:P01", "fake:amass:P02"]}
    held = {"amass": ["fake:amass:P09"]}
    prov = {"format": PV.FORMAT, "hashes": PV.hashes({"name": "t"}), "persons": persons,
            "persons_sha256": PV.list_hashes(persons), "held_out": held, "held_out_sha256": PV.list_hashes(held),
            "report_mismatches": [], "extra": {"phase": "A", "step": 5}}
    cfg = replace(CF.RunConfig(name="tiny_infer", seed=3, status="test"), model=CF.ModelSection("tiny"))
    path = tmp_path_factory.mktemp("run_tiny") / "ckpt_A_000005.pt"
    sched = {"format": "mdfm-loop/1", "in_phase": "A", "step": 5, "steps": [10, 30], "tallies": {}}
    CK.save(path, model, ema, opt, sched, None, CK.rng_state(), cfg, prov, step=5)
    return {"path": path, "model": model, "ema": ema}


@pytest.fixture(scope="module")
def loaded(ckpt):
    return I.load_checkpoint(ckpt["path"])


@needs_all
def test_load_checkpoint_restores_the_weights_and_describes_the_run(ckpt, loaded):
    for name, mod in (("ema", ckpt["ema"]), ("model", ckpt["model"])):
        got = loaded if name == "ema" else I.load_checkpoint(ckpt["path"], weights=name)
        for (k, a), b in zip(got.model.state_dict().items(), mod.state_dict().values()):
            assert torch.equal(a, b), (name, k)
        assert not got.model.training and not any(p.requires_grad for p in got.model.parameters())
    assert loaded.weights == "ema" and loaded.step == 5 and loaded.phase == "A" and loaded.phase_steps == (10, 30)
    assert not loaded.phase_a_complete and replace(loaded, step=10).phase_a_complete
    assert loaded.name == f"{os.path.basename(os.path.dirname(ckpt['path']))}_ckpt_A_000005"
    with open(str(ckpt["path"]) + ".sha256", encoding="utf-8") as f:
        assert loaded.sha256 == f.read().split()[0]
    assert loaded.trained_persons() == {"fake:amass:P01", "fake:amass:P02"}
    d = loaded.describe()
    assert d["persons"] == {"amass": 2} and d["sha256"] == loaded.sha256 and "drift" in d
    with pytest.raises(ValueError):
        I.load_checkpoint(ckpt["path"], weights="opt")


@needs_all
def test_a_tampered_checkpoint_is_refused(ckpt, tmp_path):
    bad = tmp_path / "run" / "ckpt_A_000005.pt"
    bad.parent.mkdir()
    data = bytearray(open(ckpt["path"], "rb").read())
    data[-20] ^= 1
    bad.write_bytes(bytes(data))
    with open(str(ckpt["path"]) + ".sha256", encoding="utf-8") as f:
        (bad.parent / "ckpt_A_000005.pt.sha256").write_text(f.read(), encoding="utf-8")
    with pytest.raises(PV.ProvenanceError):
        I.load_checkpoint(bad)


# ---------------------------------------------------------------- windows
def test_spans_cover_every_sample_with_full_windows():
    for n in (3338, 2500, 1000, 1001, 4400):
        sp = I.spans(n, 100.0)
        cover = np.zeros(n, int)
        for s, k in sp:
            assert k == 1000 and 0 <= s and s + k <= n
            cover[s:s + k] += 1
        assert cover.min() >= 1
        starts = [s for s, _ in sp]
        assert all(b - a == 500 for a, b in zip(starts[:-2], starts[1:-1])) and starts[-1] + 1000 == n
    assert I.spans(700, 100.0) == [(0, 700)] and I.spans(50, 100.0) == []


def _record(T=2500, seed=0, person="weargait:TEST01", source="weargait_pd", key="SelfPace"):
    rng = np.random.default_rng(seed)
    t = np.arange(T) / 100.0
    sig = np.zeros((T, 13, 9), np.float32)
    for g in range(13):
        sig[:, g, 1] = 1.0 + 0.1 * np.sin(2 * np.pi * 1.0 * t + g)
        sig[:, g, 3:6] = 60.0 * np.sin(2 * np.pi * 1.0 * t + g)[:, None] * np.array([1.0, 0.3, 0.2])
        sig[:, g, 6:9] = 0.4                                     # magnetometer, masked by the reader
    sig[:, :, :6] += 0.01 * rng.standard_normal((T, 13, 6)).astype(np.float32)
    return {"source": source, "subject_ns": person.replace("weargait:", "weargait_pd:"), "person_id": person,
            "key": key, "fps": 100.0, "signal": sig, "channel_mask": np.ones((13, 9), bool),
            "location_ids": np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]), "sample_valid": np.ones((T, 13), bool),
            "acc_content": "with_gravity"}


@pytest.fixture()
def no_tables(monkeypatch):
    from mdfm.data import sources
    monkeypatch.setattr(sources, "demographic", lambda source, ns: {"sex": "F", "age": 64.0})


def test_record_windows_band_copy_the_record_then_cut_it(no_tables):
    from mdfm.data.window_store import band_copy
    rec = _record()
    wins, sp, fps = I.record_windows(rec)
    assert fps == 100.0 and sp == [(0, 1000), (500, 1000), (1000, 1000), (1500, 1000)] and len(wins) == 4
    values, cm, valid, meta = W.arrays("imu", W.whitelist(rec))
    full = band_copy(W.Stream("imu", "input", 100.0, np.arange(2500) / 100.0, values, cm, valid, meta))
    walk = W.task_vocabulary()["ids"]["walk"]
    for wd, (s, n) in zip(wins, sp):
        st = wd.streams[0]
        assert wd.duration == pytest.approx(10.0) and wd.t0 == pytest.approx(s / 100.0) and wd.cov["task"] == walk
        assert wd.cov["sex"] == 0 and wd.source == "weargait_pd" and not wd.synthetic
        assert st.meta["layout"] == "xsens13" and not st.chan_mask[:, 6:9].any() and (st.values[..., 6:9] == 0).all()
        np.testing.assert_allclose(st.t, np.arange(n) / 100.0, atol=1e-5)
        np.testing.assert_array_equal(st.values, full.values[s:s + n])


def _imu_windows(n=2, duration=2.0, person="p0"):
    return [W.Window("synthetic", person, f"k{i}", False, 0.0, duration, [MF.imu_stream(duration, seed=i)],
                     {"sex": 1, "age_band": W.AGE_UNKNOWN, "task": 0}) for i in range(n)]


@needs_all
def test_infer_returns_bodies_on_the_100_hz_clock(loaded):
    wins = _imu_windows()
    st = {}
    a = I.infer(loaded.model, wins, refine_steps=2, points=16, samples=2, decoder_draws=3, seed=7, stats=st)
    b = I.infer(loaded.model, wins, refine_steps=2, points=16, samples=2, decoder_draws=3, seed=7)
    for x, y in zip(a, b):
        assert x.joints.shape == (200, 22, 3) and x.rot6d.shape == (200, 22, 6) and x.transl.shape == (200, 3)
        assert x.contact.shape == (200, 2) and x.pos_logvar.shape == (200, 22) and x.draws.shape == (200, 6, 3)
        assert ((x.contact >= 0) & (x.contact <= 1)).all() and np.isfinite(x.joints).all() and x.gender == 1
        np.testing.assert_array_equal(x.joints, y.joints)
        np.testing.assert_array_equal(x.draws, y.draws)
    assert st["windows"] == 2 and st["refine_first"] is not None and st["seconds"] > 0
    c = I.infer(loaded.model, wins, refine_steps=0, seed=7)
    assert c[0].draws is None and not np.array_equal(c[0].joints, a[0].joints)
    # the knot joints are the forward kinematics of the decoded knots
    from mdfm.body.rotations import rot6d_to_matrix
    from mdfm.body.smplh import fk
    rest = loaded.model.skeleton.rest(torch.as_tensor(a[0].beta)[None], torch.tensor([1]))[0]
    pos, _ = fk(rot6d_to_matrix(torch.as_tensor(a[0].rot6d)), torch.as_tensor(a[0].transl), rest)
    np.testing.assert_allclose(pos.numpy(), a[0].joints, atol=1e-5)


@needs_all
def test_windows_of_different_lengths_give_their_own_knots(loaded):
    """A short window (1.37 s) packed in a chunk with a 2 s one returns its own 137 knots, equal to the window
    run alone, and a window alone in the last chunk too (padding and chunk cuts)."""
    durs = (2.0, 1.37, 2.0)
    wins = [W.Window("synthetic", "p0", f"k{i}", False, 0.0, d, [MF.imu_stream(d, seed=i)],
                     {"sex": 1, "age_band": W.AGE_UNKNOWN, "task": 0}) for i, d in enumerate(durs)]
    both = I.infer(loaded.model, wins, refine_steps=0, batch_windows=2)
    assert [b.joints.shape[0] for b in both] == [200, 137, 200]
    for wd, b in zip(wins, both):
        alone = I.infer(loaded.model, [wd], refine_steps=0)[0]
        assert alone.joints.shape == b.joints.shape and alone.contact.shape == b.contact.shape
        np.testing.assert_allclose(b.joints, alone.joints, atol=1e-4)
        np.testing.assert_allclose(b.contact, alone.contact, atol=1e-4)


@needs_all
def test_infer_anchors_every_window_at_its_centre(loaded):
    """Windows of 2 s, 1.37 s and 2 s in one chunk: each body's root sits on the decoder's anchor (x, z) at its own
    centre knot lengths // 2 (201, 138 and 201 knots), whatever the other windows of the chunk (the integrated root
    of reports/bench/root_motion_diagnosis.md, fix 1), and moves away from it elsewhere (the EMA weights are not 0)."""
    durs = (2.0, 1.37, 2.0)
    wins = [W.Window("synthetic", "p0", f"k{i}", False, 0.0, d, [MF.imu_stream(d, seed=i)],
                     {"sex": 1, "age_band": W.AGE_UNKNOWN, "task": 0}) for i, d in enumerate(durs)]
    bodies = I.infer(loaded.model, wins, refine_steps=0, batch_windows=3)
    anchor = loaded.model.decoder.anchor_xz.cpu().numpy()
    for b, L in zip(bodies, (201, 138, 201)):
        c = L // 2
        np.testing.assert_allclose(b.transl[c, [0, 2]], anchor, atol=1e-5)
        assert np.abs(b.transl[:, [0, 2]] - anchor).max() > 1e-4


@needs_all
def test_the_shape_is_pooled_per_group(loaded):
    wins = _imu_windows(3)
    alone = I.infer(loaded.model, wins, refine_steps=0)
    pooled = I.infer(loaded.model, wins, refine_steps=0, groups=["a", "b", "a"], batch_windows=2)
    assert not np.allclose(alone[0].beta, alone[2].beta)
    np.testing.assert_allclose(pooled[0].beta, pooled[2].beta, atol=1e-6)        # pooled across chunks
    np.testing.assert_allclose(pooled[1].beta, alone[1].beta, atol=1e-5)
    with pytest.raises(ValueError):
        I.infer(loaded.model, wins, refine_steps=0, groups=["a"])


def test_heading_points_equal_the_g1_form():
    rng = np.random.default_rng(0)
    x, p, yaw = rng.standard_normal((50, 3)), rng.standard_normal((50, 3)), rng.uniform(-math.pi, math.pi, 50)
    got = I.heading_points(torch.as_tensor(x), torch.as_tensor(p), torch.as_tensor(yaw)).numpy()
    np.testing.assert_allclose(got, g1.heading_points(x, p, yaw), atol=1e-12)


@needs_all
def test_cody_windows_and_records_are_refused_without_the_eval_flag(loaded, no_tables):
    for person in ("cody_2:P1", "adult:P01", "cody_canary:mdfm-v1"):
        with pytest.raises(C.Refused):
            I.infer(loaded.model, _imu_windows(1, person=person), refine_steps=0)
    with pytest.raises(C.Refused):
        I.record_windows(_record(person="cody_1:X", source="cody_1"))
    with pytest.raises(C.Refused):
        I.G3Predictor(loaded, refine_steps=0)("T_SelfPace", _record(person="adult:P02"))
    out = I.infer(loaded.model, _imu_windows(1, person="adult:P01"), refine_steps=0, allow_eval=True)
    assert out[0].joints.shape == (200, 22, 3)


# ---------------------------------------------------------------- G3
@needs_all
def test_g3_predictor_returns_a_stitched_trial_body(loaded, no_tables):
    pred = I.G3Predictor(loaded, refine_steps=1, points=16, batch_windows=4, cache={})
    rec = g3.model_input(_record())
    tb = pred("TEST01_SelfPace", rec)
    assert tb.trial_id == "TEST01_SelfPace" and tb.joints.shape == (4, 2500, 22, 3) and tb.contact.shape == (2500, 2)
    assert np.isfinite(tb.contact).all() and tb.window_of_frame.min() == 0 and tb.window_of_frame.max() == 3
    assert np.isnan(tb.joints[0, 1000:]).all() and np.isfinite(tb.joints[3, 1500:]).all()
    assert pred.log[-1]["windows"] == 4 and pred("TEST01_SelfPace", rec) is tb and len(pred.log) == 1
    targets = g3.Targets(("weargait:TEST01",), False, None, None, None, None,
                         pd.DataFrame([("TEST01_SelfPace", "weargait:TEST01", "SelfPace", "0", 3.0, 20.0, 12, 1.1)],
                                      columns=["trial_id", "person_id", "task", "pass_id", "t0", "t1", "n_footfalls",
                                               "speed_m_s"]), None)
    p = g3.body_predictions([tb], targets)
    assert len(p.speeds) == 1 and np.isfinite(p.speeds["speed_m_s"]).all()


# ---------------------------------------------------------------- G1
class FakeG1Set:
    """Two takes of two 1.5 s windows each, inputs of the G1 contract, a standing truth."""

    def __init__(self, split="calib", n=150):
        self.split = split
        rows = []
        for t, take in enumerate(("A/take_a_poses.npz", "A/take_b_poses.npz")):
            for k in range(2):
                rows.append({"window": len(rows), "take_id": take, "archive": "A", "subject_ns": f"amass:A:S{t}",
                             "person_id": f"amass:person:A:S{t}", "gender": "female" if t else "male", "task": 0, "k": k,
                             "n": n})
        self.windows = pd.DataFrame(rows)
        self.n = n

    def __len__(self):
        return len(self.windows)

    def inputs(self, i):
        rng = np.random.default_rng(int(i))
        n = self.n
        imu = np.zeros((n, 6, 6), np.float32)
        imu[..., 1] = 1.0
        imu += 0.02 * rng.standard_normal(imu.shape).astype(np.float32)
        nc = len(np.arange(0, (n - 1) / 100 + 1e-9, 1 / 30))
        cam = MF.SKELETON15[None, :, :2].repeat(nc, 0) + 0.01 * rng.standard_normal((nc, 15, 2)).astype(np.float32)
        return {"imu": imu, "imu_valid": np.ones((n, 6), bool), "cam": cam.astype(np.float32),
                "cam_valid": np.ones((nc, 15), bool), "cam_fps": 30.0, "cam_scale": 120.0, "cam_params": np.zeros(21)}

    def truth(self, i):
        J = np.zeros((self.n, 22, 3))
        J[:, :, 1] = np.linspace(0.0, 1.0, 22)[None]
        J[:, g1.WRIST_JOINT] = [0.3, 0.9, 0.1]
        return J, np.zeros(self.n)


def test_g1_windows_follow_the_training_layouts():
    ws = I.g1_windows(FakeG1Set(), 1)
    assert set(ws) == set(I.G1_CONFIGS)
    six, wrist, cam = ws["six"].streams[0], ws["wrist"].streams[0], ws["camera"].streams[0]
    assert six.meta["layout"] == "g1_six" and len(six.meta["site_ids"]) == 6 and six.values.shape == (150, 6, 9)
    assert wrist.meta["layout"] == "wrist_L" and wrist.values.shape == (150, 1, 9)
    assert cam.modality == "pose2d" and cam.meta["layout"] == "camera1" and cam.fps == 30.0
    assert ws["six"].cov["sex"] == 1 and ws["six"].duration == pytest.approx(1.5)
    bad = FakeG1Set()
    bad.windows.loc[1, "person_id"] = "cody_3:P1"
    with pytest.raises(C.Refused):
        I.g1_windows(bad, 1)


@needs_all
def test_g1_predictor_blocks_draws_and_conformal(loaded):
    cal = FakeG1Set("calib")
    pred = I.G1Predictor(loaded, refine_steps=1, points=16, samples=2, decoder_draws=2, batch_windows=4, block_windows=3)
    assert pred.block(cal, 0) == [0, 1] and pred.block(cal, 3) == [2, 3]
    p = pred.predict(cal, 1)
    assert p["joints_six"].shape == (150, 22, 3) and p["joints_camera"].shape == (150, 22, 3)
    assert p["wrist"].shape == (150, 4, 3) and p["wrist_six"].shape == (150, 4, 3)
    assert len(pred.log) == 1 and pred.log[0]["takes"] == 1
    assert pred.train_persons == {"fake:amass:P01", "fake:amass:P02"} and pred.sha256 == loaded.sha256
    with pytest.raises(ValueError):
        pred.conformal
    fits = I.fit_conformal(cal, pred)
    assert set(fits) == {"wrist", "wrist_six"} and len(pred.log) == 2
    for f in fits.values():
        assert math.isfinite(f["c"]) and f["n_frames"] == 600 and f["coverage_at_c"] >= 0.9
    with pytest.raises(C.Refused):
        I.fit_conformal(FakeG1Set("dev"), pred)
    pred.c_wrist, pred.c_six = fits["wrist"]["c"], fits["wrist_six"]["c"]
    J, yaw = cal.truth(2)
    m = g1.score_window(J, yaw, pred.predict(cal, 2), pred.conformal)
    assert all(np.isfinite(m[k]) for k in g1.METRICS if k != "degenerate")
    wide = I.G1Predictor(loaded, refine_steps=0, samples=2, decoder_draws=0, block_windows=8, configs=("six",))
    assert wide.block(cal, 0) == [0, 1, 2, 3]
    wide.subset = [2, 3]
    assert wide.block(cal, 2) == [2, 3]
    q = wide.predict(cal, 2)
    assert q["wrist_six"].shape == (150, 2, 3) and q["wrist"] is None and q["joints_camera"] is None


@needs_all
def test_fit_conformal_equals_the_harness_fit_of_each_key(loaded):
    """infer.fit_conformal (both wrist predictives in one pass) gives, for each key, what the harness's own
    g1.fit_conformal gives for that key alone on the same predictor."""
    cal = FakeG1Set("calib")
    pred = I.G1Predictor(loaded, refine_steps=1, points=16, samples=2, decoder_draws=2, batch_windows=4,
                         block_windows=4, configs=("six", "wrist"))
    both = I.fit_conformal(cal, pred)
    for key in ("wrist", "wrist_six"):
        one = g1.fit_conformal(cal, pred, key)
        assert both[key]["n_frames"] == one["n_frames"] and both[key]["n_degenerate"] == one["n_degenerate"]
        assert both[key]["c"] == pytest.approx(one["c"], rel=1e-9)
        assert both[key]["coverage_at_c"] == pytest.approx(one["coverage_at_c"], abs=1e-12)


def _script(name):
    import importlib.util
    old = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    try:                                                            # the script sets the variable at import
        spec = importlib.util.spec_from_file_location(f"{name}_script", os.path.join(TREE, "scripts", f"{name}.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        if old is None:
            os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        else:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = old
    return mod


@needs_all
def test_eval_g3_dev_offsets_and_the_stop_rule():
    """scripts/eval_g3_dev.py: an IC predictor that never comes within the match window of a reference IC gets
    g3.UNFITTED (NaN, n 0) and is listed, a matching one gets g3.fit_offset's; the stop rule reads
    'undetermined' (None), never 'no', for a checkpoint that closes phase A with no matched stride."""
    S = _script("eval_g3_dev")
    person = g3.controls("dev")[0]
    tid = g3.trial_id(person, "SelfPace")
    ref = pd.DataFrame({"trial_id": [tid] * 3, "person_id": [person] * 3, "task": ["SelfPace"] * 3,
                        "foot": ["L", "L", "R"], "t_s": [1.0, 2.1, 1.55]})
    tg = g3.Targets((person,), False, None, ref, ref.copy(), None, None, None)
    near = g3.Predictions("near", pd.DataFrame({"trial_id": [tid] * 3, "foot": ["L", "L", "R"],
                                                "t_s": [1.04, 2.14, 1.59]}))
    far = g3.Predictions("far", pd.DataFrame({"trial_id": [tid], "foot": ["L"], "t_s": [9.0]}))
    none = g3.Predictions("none", pd.DataFrame(columns=["trial_id", "foot", "t_s"]))
    off, unfitted = S.fit_offsets([near, far, none, g3.Predictions("no_ics")], tg)
    assert off[("near", "walkway")][0] == pytest.approx(0.04) and off[("near", "insole")][1] == 3
    for key in (("far", "walkway"), ("none", "insole")):
        assert math.isnan(off[key][0]) and off[key][1] == 0 and not g3.offset_fitted(off[key][0])
    assert unfitted == ["far|walkway", "far|insole", "none|walkway", "none|insole"]
    assert not any(k[0] == "no_ics" for k in off)
    other = g3.controls("train")[0]
    bad = g3.Targets((other,), False, None, ref.assign(person_id=other), ref.assign(person_id=other), None, None, None)
    with pytest.raises(ValueError, match="dev controls only"):
        S.fit_offsets([near], bad)
    assert S.stop_decision(True, 41.0) is True and S.stop_decision(True, 40.0) is False
    assert S.stop_decision(True, float("nan")) is None and S.stop_decision(True, None) is None
    assert S.stop_decision(False, 90.0) is False and S.stop_decision(False, float("nan")) is False
    assert S.stop_text(None).startswith("undetermined")


def test_infer_and_the_gates_import_in_either_order():
    for code in ("import mdfm.eval.infer, mdfm.gates.g1", "import mdfm.gates.g1, mdfm.eval.infer"):
        r = subprocess.run([sys.executable, "-c", code], cwd=TREE, capture_output=True, text=True, timeout=120)
        assert r.returncode == 0, r.stderr[-2000:]


# ---------------------------------------------------------------- saved conformal factors (B4.20)
def _calib(manifest="a"):
    cal = FakeG1Set("calib")
    cal.manifest = {"manifest_sha256": manifest * 64, "persons_sha256": "b" * 64}
    return cal


@needs_all
def test_saved_conformal_factors_round_trip_and_refuse_another_checkpoint(loaded, tmp_path):
    import json
    from mdfm.gates import common
    cal = _calib()
    pred = I.G1Predictor(loaded, refine_steps=1, points=16, samples=2, decoder_draws=2, batch_windows=4,
                         block_windows=4, configs=("six", "wrist"))
    fits = I.fit_conformal(cal, pred)
    rec = I.conformal_record(pred, fits, cal, extra={"script": "test"})
    assert rec["format"] == I.CONFORMAL_FORMAT and rec["checkpoint"]["sha256"] == loaded.sha256
    assert rec["calib"]["windows"] == 4 and rec["calib"]["archives"] == ["A"] and rec["calib"]["knots"] == 600
    assert rec["settings"]["refine_steps"] == 1 and rec["settings"]["shape_pooling"] == "per take"
    run = tmp_path / "run_x"
    path = I.conformal_path(run / "ckpt_B_000005.pt", loaded.sha256)
    assert path == os.path.join(str(run), f"conformal_{loaded.sha256[:12]}.json")
    p = I.save_conformal(path, rec)
    with pytest.raises(FileExistsError):
        I.save_conformal(p, rec)
    raw = open(p, "rb").read()
    assert b"\r\n" not in raw and raw.endswith(b"\n")
    # never automatic: a new predictor has no factor until it is asked to read the file
    other = I.G1Predictor(loaded, refine_steps=1, points=16, samples=2, decoder_draws=2)
    with pytest.raises(ValueError):
        other.conformal
    doc = other.use_conformal_file(p, cal)
    assert other.conformal == {"wrist": pytest.approx(fits["wrist"]["c"]),
                               "wrist_six": pytest.approx(fits["wrist_six"]["c"])}
    assert doc["file_sha256"] == common.sha256_file(p) and other.conformal_file["file_sha256"] == doc["file_sha256"]
    # a file of another checkpoint is refused, by the predictor and by read_conformal
    bad = json.loads(raw.decode("utf-8"))
    bad["checkpoint"]["sha256"] = "0" * 64
    p2 = tmp_path / "other.json"
    p2.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(C.Refused, match="its own checkpoint"):
        other.use_conformal_file(p2)
    with pytest.raises(C.Refused):
        I.read_conformal(p, "f" * 64)
    # other settings of the wrist predictives, other calib inputs, a calib set of another split
    for kw in ({"refine_steps": 0}, {"samples": 3}, {"pool": False}):
        args = dict(refine_steps=1, points=16, samples=2, decoder_draws=2) | kw
        with pytest.raises(C.Refused, match="other settings"):
            I.G1Predictor(loaded, **args).use_conformal_file(p)
    with pytest.raises(C.Refused, match="calib inputs"):
        other.use_conformal_file(p, _calib("c"))
    dev = FakeG1Set("dev")
    dev.manifest = dict(cal.manifest)
    with pytest.raises(C.Refused):
        other.use_conformal_file(p, dev)
    # never saved: a fit on part of the calib windows, a non-finite factor, a set other than calib
    with pytest.raises(ValueError, match="every calib window"):
        I.conformal_record(pred, fits, cal, windows=[0, 1])
    with pytest.raises(ValueError, match="finite"):
        I.conformal_record(pred, {"wrist": dict(fits["wrist"], c=float("inf")), "wrist_six": fits["wrist_six"]}, cal)
    with pytest.raises(ValueError):
        I.conformal_record(pred, fits, FakeG1Set("dev"))


@needs_all
def test_eval_g1_dev_secondary_rows_run_through_the_t05_functions(loaded):
    S = _script("eval_g1_dev")
    T = S.t05()
    a = S.parse(["--checkpoint", "x", "--refine-steps", "1", "--points", "16", "--batch-windows", "4",
                 "--block-windows", "4", "--abs-steps", "2", "--abs-limit-takes", "1"])
    assert S.check_args(a) is None
    cal, dset = _calib(), FakeG1Set("dev")
    pred = I.G1Predictor(loaded, refine_steps=1, points=16, samples=2, decoder_draws=2, batch_windows=4,
                         block_windows=4, seed=a.seed)
    fits = I.fit_conformal(cal, pred)
    pred.c_wrist, pred.c_six = fits["wrist"]["c"], fits["wrist_six"]["c"]
    table = g1.score(dset, pred)
    eye = np.broadcast_to(np.eye(3), (150, 22, 3, 3)).copy()
    truth_of = lambda i: {"R_local": eye, "betas": np.zeros(16), "gender": 1}  # noqa: E731
    sec, tabs = S.secondary_model_rows(T, loaded, cal, dset, [0, 1, 2, 3], [0, 1, 2, 3], table, a,
                                       truth_of=truth_of, draws={"samples": 2, "decoder_draws": 2})
    R = sec["rows"]
    assert set(R) == {"zero_refinement_steps", "truth_rerendered", "truth_rerendered_kinematic", "acc_only_wrist",
                      "analysis_by_synthesis"}
    for k in ("zero_refinement_steps", "truth_rerendered", "truth_rerendered_kinematic"):
        assert math.isfinite(R[k]["point_mm"]) and R[k]["n_windows"] == 4 and R[k]["partial"] is None
        assert R[k]["delta_of"] == "g1a_m"
    ab = R["analysis_by_synthesis"]
    assert ab["partial"] == "first 1 takes of each dev archive (2 windows)" and math.isfinite(ab["point_mm"])
    assert ab["reference"]["n_windows"] == 2 and ab["delta_of"] == "reference_m"
    acc = R["acc_only_wrist"]
    assert 0.0 <= acc["coverage"] <= 1.0 and acc["n_windows"] == 4 and math.isfinite(acc["conformal"]["c"])
    assert acc["acc_gyro_same_windows"]["windows"] == 4
    # the main pass of the rows reproduces the harness's G1a: same blocks, seeds and settings
    h = sec["harness_check"]
    assert h["windows"] == 4 and h["max_window_difference_mm"] < 1e-3, h
    assert sec["truth_check"] is None and set(tabs) == {"secondary_windows", "acc_only_windows", "abs_windows"}
    md = "\n".join(S.secondary_markdown(sec, 1, S.fmt))
    assert "zero refinement steps" in md and "partial: first 1 takes" in md and "\u2014" not in md
