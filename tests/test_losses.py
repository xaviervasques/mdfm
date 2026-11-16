"""Tests of the stage-1 loss terms and the label-free check, mdfm/train/losses.py (spec
docs/MFM_MODEL_CODE_SPEC_2026-09-28.md, section 4 and 2.6, task B4.16).

- The label-free check finds label keys at any depth (dict keys, dataclass and namedtuple fields, plain
  attributes, DataFrame columns, Series index labels, structured and object arrays; manifest dicts and
  nested keys; acronym boundaries as PDLabel, HCGroup, OFFphase; diagnoses, sublabel), passes the model
  batch (mfm.Points.group admitted only as a stream-window index), synthetic windows and one real window per
  pilot source of the window store; stage1_losses and targets_from_windows run it; CODY windows and batches
  are refused.
- A no-label grep over mdfm/model and mdfm/train (text and AST) finds no read of a label, group, cohort,
  severity, diagnosis, phase or manifest field (the one admitted read: mfm.Points.group), and mdfm/train
  never imports mdfm.gates.
- Every term is zero or minimal at truth on a synthetic batch (clean renders of a KIT walk of a train
  archive: xsens13 with insoles, a camera, 3D mocap with a root track): the kernel likelihoods at the truth
  body and the rendered nuisance lie below those of perturbed bodies; direct supervision is 0 with zero
  gradient at the truth knots; nuis_aux is 0 at the rendered nuisance; the KL terms, the nuisance prior,
  consistency, subject and amortization are minimal at their centres; the plausibility parts vanish inside
  the limits, above the floor and without horizontal foot speed in contact; HSIC is 0 for independent
  summaries, positive for source-coded ones and inactive in strata with fewer than two sources, and equals
  the explicit matrix form tr(K H L H) / n^2. Noisy default-mixture renders of train body-store windows keep
  every kernel term lowest at the truth, and targets_from_windows finds the sampler's render pairs.
- Shape (reports/bench/phaseA_diagnosis.md, C4): on windows with a body truth only the shape terms move the
  shape posterior (the gradient of the pose terms and the kernels on mu_b is exactly 0, that of sup_beta is
  not; without the truth mask the kernels reach it), and sup_pos poses both bodies on the true skeleton, so
  its squared part is 0 at the truth knots whatever beta is drawn. sup_root scales the horizontal and the
  vertical root apart (SIGMAS root_xz and root_y), and LossConfig.sigmas reaches every supervision term.
- Root motion (reports/bench/root_motion_diagnosis.md, fixes 3 to 6): sup_root_rate is 0 with zero gradient at the
  true grid rates and scales the velocity and the yaw rate apart; sup_deriv is invariant to a constant heading of
  the decoded body (its window-frame form was not); on a circle walk with a growing heading error the root terms
  no longer reward a shorter path (their window-frame form did); heading_observable reads the channel masks.
- The kernel terms sum to -MFM.log_lik; WEIGHTS equal the weight column of the spec's section 4 table;
  stage1_losses weighs each phase as that table (root, EMG, HSIC and amortization off in phase A, force on
  synthetic windows only in phase A, KL annealed over the first 10 percent of phase A) and backpropagates.
  Targets marking a body truth that the batch lacks (a mixed batch) and a body without contact labels
  raise instead of switching supervision off.
- The JEPA term (task J2a): jepa_per_window is the smooth L1 (beta 1) per masked patch averaged over the
  features, then over each window's patches, 0 elsewhere, and is not one of TERMS.

The model tests need the MoveAtlas SMPL+H model files; the walk tests also the AMASS take (cache or raw
archive, read-only); the real-window test the window store (skipped without them).
"""
import ast
import math
import os
import re
from collections import namedtuple
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import pytest
import torch

from mdfm import paths
from mdfm.body.rotations import (axis_angle_to_matrix, matrix_to_axis_angle, matrix_to_rot6d, rot6d_to_matrix, rot_y,
                                 yaw_angle)
from mdfm.data.canonical import Refused
from mdfm.model import mfm as M
from mdfm.model.decoder import BodyOut
from mdfm.model.posterior import subject_pool
from mdfm.model.prior import gaussian_kl
from mdfm.train import losses as L

import model_fixture as FX

F64 = torch.float64
HAVE_MA = os.path.exists(paths.smplh_model("neutral"))
needs_ma = pytest.mark.skipif(not HAVE_MA, reason="MoveAtlas SMPL+H model not found")
WALK = ("KIT", "KIT/11/WalkingStraightForwards01_poses.npz")
N_KNOTS = 300
ROOT_SHIFT = (1.0, 0.1, -2.0)
SCAN = ("mdfm/model", "mdfm/train")


# ---------------------------------------------------------------- fixtures
def _toy_windows(root=False):
    """The model_fixture windows: synthetic (every stream), a weargait_pd window (every stream, a real
    source), a gait120 pose window; root: the pose3d streams carry a raw root track."""
    wins = [FX.make_window(2.0, seed=0), FX.make_window(1.5, seed=3, person="p1", sex=0, source="weargait_pd"),
            FX.make_pose_window(1.0, seed=5, person="p2")]
    if root:
        rng = np.random.default_rng(7)
        for wd in wins:
            for s in wd.streams:
                if s.modality == "pose3d":
                    n = len(s.t)
                    s.meta["root"] = (np.array([0.5, 0.9, 1.0]) + 0.3 * rng.standard_normal((n, 3))).astype(np.float32)
                    s.meta["root_valid"] = np.ones(n, bool)
    return wins


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    return M.MFM(M.TINY)


def _walk_bodies(n_windows, n=N_KNOTS):
    """BodyWindows of the first n knots of the KIT walk (a train archive), repeated n_windows times: floor at
    y = 0, pelvis x and z at 0 at the centre knot, contacts by the body store's rule."""
    from mdfm.body import frames
    from mdfm.body.kinematics import KnotBatch
    from mdfm.body.smplh import SMPLHBody, SkeletonBank, gender_code
    from mdfm.data import amass_index
    from mdfm.data.amass import load_sequence
    from mdfm.data.body_store import BodyWindows
    from mdfm.data.prep import amass_to_body_clock
    if os.path.exists(amass_index.INDEX_PATH):
        assert amass_index.check_member(*WALK) == "train"
    seq = load_sequence(*WALK)
    r6, tr, betas = amass_to_body_clock(seq, rate=100.0, cutoff=12.0, body=SMPLHBody(seq["gender"], dtype=F64))
    g = gender_code(seq["gender"])
    r6, tr = r6[:n], tr[:n].clone()
    rest = SkeletonBank(F64).rest(betas[None].to(F64), torch.tensor([g]))[0]
    w0, tt = torch.zeros(n, dtype=torch.int64), torch.arange(n, dtype=F64) / 100.0
    joints = KnotBatch(r6[None], tr[None], rest[None]).pose(w0, tt)[0]
    tr[:, 1] -= frames.floor_height(joints)
    tr[:, [0, 2]] -= joints[n // 2, 0, [0, 2]]
    kin = KnotBatch(r6[None], tr[None], rest[None]).kinematics(w0, tt)
    contact = frames.contact_labels(kin.p[None], vel=kin.v[None], floor=0.0)[0]
    rep = lambda x: x[None].expand((n_windows,) + tuple(x.shape)).clone()  # noqa: E731
    return BodyWindows(rep(r6.float()), rep(tr.float()), rep(contact), torch.ones(n_windows, n, dtype=torch.bool),
                       torch.full((n_windows,), n), rep(betas.float()), torch.full((n_windows,), g),
                       np.array([f"test:kit:{i}" for i in range(n_windows)]), torch.zeros(n_windows, dtype=torch.int64),
                       torch.zeros(n_windows, dtype=torch.int64), torch.zeros(n_windows, dtype=torch.int64)), rest


def _true_nuisance(model, batch, tg):
    """The rendered nuisance in the model's parameterization: IMU mounts (R0 dR) and biases at the rendered
    sites, the camera (rotation, focal, distance of the model's target, the window-mean true pelvis at the
    pose2d points, along the optical axis); other fields at the heads' prior centres."""
    from mdfm.kernels.nuisance import CameraNuisance, IMUNuisance, identity6d
    W, d = batch.n_windows, model.cfg.d
    fam = batch.win["imu_family"]
    with torch.no_grad():
        c0 = M.Nuisance(model.heads.imu(torch.zeros(W, 14, d), fam), model.heads.camera(torch.zeros(W, d)),
                        model.heads.pose3d(torch.zeros(W, d)))
        R0 = model.heads.priors().rotation(fam).double()
        res = (matrix_to_rot6d(R0.transpose(-1, -2) @ tg.imu_mount.double()) - identity6d(F64)).float()
        known = tg.imu_known.any(-1)
        imu = IMUNuisance(torch.where(known[..., None], res, c0.imu.mount_res6d), c0.imu.offset_res,
                          tg.imu_bias_acc.clone(), tg.imu_bias_gyro.clone(), c0.imu.log_scale, c0.imu.gain_res)
        kin_t, sl, _ = model.kinematics(batch.truth, batch)
        target = torch.zeros(W, 3)
        if "pose2d" in sl:
            p2 = batch.points["pose2d"]
            target = target.index_add(0, p2.w, kin_t.p[sl["pose2d"], 0]) / \
                torch.bincount(p2.w, minlength=W).clamp_min(1)[:, None]
        depth = (tg.cam_R @ target[..., None])[..., 0] + tg.cam_t
        on = tg.cam_known
        cam = CameraNuisance(torch.where(on[:, None], matrix_to_rot6d(tg.cam_R), c0.camera.rot6d),
                             torch.where(on, torch.log(depth[:, 2].clamp_min(1e-3)), c0.camera.log_distance),
                             torch.where(on, torch.log(tg.cam_focal), c0.camera.log_focal), c0.camera.log_sigma_px)
    return M.Nuisance(imu, cam, c0.pose3d_log_sigma), depth[on, :2]


@pytest.fixture(scope="module")
def walk(tiny):
    """Clean renders of one KIT walk body (3 s) under three layouts: xsens13 IMU with insole totals (window
    0), a camera (window 1), 3D mocap with a raw root track turned by the render's lab heading and shifted
    (window 2); all kernel points; the targets and the rendered nuisance."""
    from mdfm.body.kinematics import KnotBatch
    from mdfm.body.smplh import SkeletonBank
    from mdfm.factory import layouts as LY
    from mdfm.factory.synth import RenderSpec, SynthSampler
    lays = LY.load_layouts()
    specs = [RenderSpec(lays["xsens13_insoles"], clean=True), RenderSpec(LY.concrete(lays["camera1"], 0), clean=True),
             RenderSpec(lays["mocap3d"], clean=True)]
    bodies, rest = _walk_bodies(len(specs))
    wins = SynthSampler(layouts=lays, skel=SkeletonBank(torch.float32), dtype=torch.float32,
                        device="cpu").render(bodies, specs, 0)
    s = wins[2].streams[0]
    assert s.modality == "pose3d"
    kb = KnotBatch(bodies.rot6d[:1].double(), bodies.transl[:1].double(), rest[None].double())
    pel = kb.pose(torch.zeros(len(s.t), dtype=torch.int64), torch.as_tensor(s.t, dtype=F64))[0][:, 0]
    turn = rot_y(torch.tensor(float(s.truth["heading"]), dtype=F64))
    s.meta["root"] = ((pel @ turn.T) + torch.tensor(ROOT_SHIFT, dtype=F64)).float().numpy()
    s.meta["root_valid"] = np.ones(len(s.t), bool)
    batch = M.batch_from_windows(wins, gen=0)
    tg = L.targets_from_windows(wins, T=batch.T)
    nuis, lateral = _true_nuisance(tiny, batch, tg)
    return {"windows": wins, "batch": batch, "targets": tg, "nuis": nuis, "lateral": lateral}


def _perturbed(truth, scale, gen):
    return truth._replace(rot6d=truth.rot6d + scale * torch.randn(truth.rot6d.shape, generator=gen),
                          transl=truth.transl + 0.25 * scale * torch.randn(truth.transl.shape, generator=gen))


def _truth_out(batch, tg, rot6d=None, transl=None):
    """BodyOut at the batch's decoded knots holding the truth knots, contact logits +-20 at the truth
    contacts, pos_logvar -8, the truth's root grid (losses.truth_root_grid)."""
    w, k = batch.knots
    tr = batch.truth
    c = tg.contact[w, k]
    return BodyOut(tr.rot6d[w, k] if rot6d is None else rot6d, tr.transl[w, k] if transl is None else transl,
                   20.0 * (2 * c - 1), torch.full((len(w), 22), -8.0), w, k,
                   L.truth_root_grid(tr, batch.rate, batch.lengths))


# ---------------------------------------------------------------- label-free check
LabelFree = namedtuple("LabelFree", ["coords", "subjectGroup"])


@dataclass
class Holder:
    rows: list
    extra: object = None


class Plain:
    def __init__(self):
        self.values = np.zeros(3)
        self.phase = "ON"


def test_label_keys_named_like_labels_are_found_at_any_depth():
    for name in ("label", "Labels", "analog_labels", "label_prefix_stripped", "subject_group", "subjectGroup",
                 "UPDRS-Label", "manifest", "phase_on", "Diagnosis", "cohort_id", "severity", "tier", "lockbox",
                 "aux_streams", "notes", "scope", "lane",
                 # acronym boundaries, plural and derived forms, a stem inside a word
                 "PDLabel", "UPDRSLabel", "HCGroup", "OFFphase", "medPhase", "diagnoses", "diagnostic", "Diagnosed",
                 "sublabel", "pdCohortId", "HYseverity", "GROUPS"):
        assert L.label_key(name), name
    for name in ("p3_scale", "imu_family", "site_ids", "source_id", "lengths", "root_valid", "t_first_s", "flight",
                 "contact_prob", "channels", "muscle", "task", "sex", "gender", "person_id", "age_band", "key",
                 "unit_to_si", "signal_kind", "nuis_desc", "n_patches", "stems", "w", "image_plane", "frontier",
                 "telescope_id", "IMUsites", "IDs", "p3_zup", "acc_content", "mag_unit", "space"):
        assert not L.label_key(name), name
    record = {"coords": np.zeros((4, 15, 3), np.float32), "fps": 100.0,
              "manifest": {"group": "control", "tier": "RELEASE", "person_id": "x"}}
    assert L.label_paths(record) == ["manifest", "manifest.group", "manifest.tier"]
    structured = np.zeros(3, dtype=[("t", "f4"), ("severity", "i4")])
    objects = np.array([{"ok": 1}, {"cohort": 2}], dtype=object)
    nest = {"a": [({"b": LabelFree(np.zeros(2), 1)},)], "frame": pd.DataFrame({"x": [1], "group": [2]}),
            "row": pd.Series({"fps": 100.0, "label": "PD"}), "arr": structured, "obj": objects,
            "holder": Holder([{"deep": {"Diagnosis_code": 3}}], Plain())}
    found = set(L.label_paths(nest))
    assert found == {"a[0][0].b.subjectGroup", "frame.group", "row.label", "arr.severity", "obj[1].cohort",
                     "holder.rows[0].deep.Diagnosis_code", "holder.extra.phase"}, found
    with pytest.raises(L.LabelLeak):
        L.assert_label_free({"fine": 1}, nest)
    L.assert_label_free({"fine": 1}, None, [np.zeros(3)], torch.zeros(2))


def test_the_batch_passes_and_injected_keys_trip():
    wins = _toy_windows(root=True)
    batch = M.batch_from_windows(wins, points_per_stream=32, gen=0)
    L.assert_label_free(batch, wins)                                  # Points.group admitted as stream-window ids
    tg = L.targets_from_windows(wins, T=batch.T)
    L.assert_label_free(tg)
    cases = [
        (replace(batch, win=dict(batch.win, label=torch.zeros(3))), "win.label"),
        (replace(batch, win=dict(batch.win, subject_cohort=torch.zeros(3))), "win.subject_cohort"),
    ]
    imu = batch.points["imu"]
    leak = replace(imu, extra={"severity_score": torch.zeros(imu.w.numel())})
    cases.append((replace(batch, points=dict(batch.points, imu=leak)), "points.imu.extra.severity_score"))
    cases.append((replace(batch, points=dict(batch.points, imu=replace(imu, group=imu.group.float()))),
                  "points.imu.group"))
    spanning = torch.zeros_like(imu.group)                              # one id over two windows: not an index
    assert imu.w.unique().numel() > 1
    cases.append((replace(batch, points=dict(batch.points, imu=replace(imu, group=spanning))), "points.imu.group"))
    for bad, path in cases:
        with pytest.raises(L.LabelLeak, match=re.escape(path)):
            L.assert_label_free(bad)
    # window meta: a manifest dict with its group column, and a nested truth key
    meta = replace(wins[0].streams[0], meta=dict(wins[0].streams[0].meta, manifest={"group": "control"}))
    leaky = replace(wins[0], streams=[meta] + wins[0].streams[1:])
    assert L.label_paths(leaky) == ["streams[0].meta.manifest", "streams[0].meta.manifest.group"]
    deep = replace(wins[1].streams[1], truth={"camera": {"R": np.eye(3)}, "extra": {"Diagnosis": 1}})
    with pytest.raises(L.LabelLeak, match=re.escape("streams[1].truth.extra.Diagnosis")):
        L.targets_from_windows([wins[0], replace(wins[1], streams=[wins[1].streams[0], deep])])
    with pytest.raises(L.LabelLeak):
        L.targets_from_windows([leaky])
    # stage1_losses checks the batch and the targets before anything else
    with pytest.raises(L.LabelLeak):
        L.stage1_losses(None, cases[0][0], None, None, {}, "A")
    with pytest.raises(L.LabelLeak):
        L.stage1_losses(None, batch, None, None, {}, "A", targets=replace(tg, kin={"label": 1}))


def test_cody_windows_and_batches_are_refused():
    wins = _toy_windows()
    for source, person in (("cody_3", "cody_3:P01"), ("synthetic", "cody_1:P02"), ("CODY_2", "p9")):
        bad = replace(wins[0], source=source, person_id=person)
        with pytest.raises(Refused):
            L.targets_from_windows([wins[1], bad])
        assert L.targets_from_windows([bad], allow_eval=True).has_truth.tolist() == [False]
    batch = M.batch_from_windows(wins, points_per_stream=16, gen=0)
    with pytest.raises(Refused):
        L.stage1_losses(None, replace(batch, sources=tuple(batch.sources) + ("cody_3",)), None, None, {}, "B")


def test_real_windows_of_every_pilot_source_pass_the_check():
    from mdfm.data import window_store as WS
    root = WS.store_root()
    if not os.path.exists(os.path.join(root, "store.json")):
        pytest.skip("window store not built")
    store = WS.WindowStore(root)
    try:
        seen = {}
        for src, g in store.recordings.groupby("source"):
            wd = store.window(int(g.index[0]), float(g["span_t0"].iloc[0]), 10.0)
            L.assert_label_free(wd)
            seen[src] = sorted(s.modality for s in wd.streams)
        tg = L.targets_from_windows([store.window(int(g.index[0]), None, 5.0)
                                     for _, g in store.recordings.groupby("source")])
        assert not bool(tg.has_truth.any()) and not bool(tg.imu_known.any()) and set(tg.pair.tolist()) == {-1}
    finally:
        store.close()
    assert {"weargait_pd", "gait120", "stroke_gait_vancriekinge", "nonan_gaitprint", "nonan_gaitprint_young",
            "sixmwt_lifespan"} <= set(seen), seen


@needs_ma
def test_real_windows_score_every_observed_stream(tiny):
    """One window per pilot source of the window store through forward and stage1_losses (phase B): the
    kernel terms of the streams each window holds are nonzero and finite, the others 0; no supervision."""
    from mdfm.data import window_store as WS
    root = WS.store_root()
    if not os.path.exists(os.path.join(root, "store.json")):
        pytest.skip("window store not built")
    store = WS.WindowStore(root)
    try:
        wins = [store.window(int(g.index[0]), float(g["span_t0"].iloc[0]), 4.0)
                for _, g in store.recordings.groupby("source")]
    finally:
        store.close()
    tiny.eval()
    batch = M.batch_from_windows(wins, points_per_stream=32, gen=0)
    tg = L.targets_from_windows(wins, T=batch.T)
    with torch.no_grad():
        post = tiny.encode(batch)
        body, pred = L.forward(tiny, batch, post, torch.Generator().manual_seed(1))
        out = L.stage1_losses(tiny, batch, post, body, pred, "B", targets=tg)
    term = {"imu": "imu_nll", "pose3d": "pose3d_nll", "force": "force_nll", "emg": "emg_nll"}
    for i, wd in enumerate(wins):
        mods = {s.modality for s in wd.streams}
        for m, t in term.items():
            assert (float(out[t][i]) != 0.0) == (m in mods), (wd.source, t, float(out[t][i]))
        has_root = any(s.modality == "pose3d" and s.meta.get("root") is not None for s in wd.streams)
        assert (float(out["root_nll"][i]) != 0.0) == has_root, wd.source
    assert all(bool(torch.isfinite(out[t]).all()) for t in L.TERMS + L.DIAGNOSTICS)
    for t in ("sup_rot", "sup_pos", "sup_root", "sup_deriv", "sup_contact", "sup_beta", "nuis_aux", "consistency"):
        assert float(out[t].abs().max()) == 0.0, t


# ---------------------------------------------------------------- the no-label grep
STEM = r"(?:label\w*|group|cohort\w*|severity\w*|diagnos\w*|phase\w*|manifest\w*)"
STRING_READ = re.compile(r"(?:\[\s*|\.get\(\s*|\.pop\(\s*|\.setdefault\(\s*|getattr\(\s*[\w.\[\]]+\s*,\s*"
                         r"|hasattr\(\s*[\w.\[\]]+\s*,\s*)[\"']" + STEM + r"[\"']|[\"']" + STEM
                         + r"[\"']\s+(?:not\s+)?in\s", re.I)
# the one field whose name matches: mfm.Points.group, the stream-window index of the kernel points
ALLOWED_ATTRIBUTES = {("mdfm/model/mfm.py", "group")}


def label_reads(text, rel):
    """Line numbers of reads of a label-like field in a source text: subscripts, get, pop, setdefault,
    getattr, hasattr and membership with such a string (text and AST), attributes of such a name (AST)."""
    hits = {n for n, line in enumerate(text.splitlines(), 1) if STRING_READ.search(line)}
    stem = re.compile(STEM, re.I)
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Attribute) and stem.fullmatch(node.attr) and (rel, node.attr) not in ALLOWED_ATTRIBUTES:
            hits.add(node.lineno)
        elif isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) \
                and isinstance(node.slice.value, str) and stem.fullmatch(node.slice.value):
            hits.add(node.lineno)
        elif isinstance(node, ast.Call):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            if name in ("get", "pop", "setdefault", "getattr", "hasattr") and any(
                    isinstance(a, ast.Constant) and isinstance(a.value, str) and stem.fullmatch(a.value)
                    for a in node.args):
                hits.add(node.lineno)
        elif isinstance(node, ast.Compare) and isinstance(node.left, ast.Constant) \
                and isinstance(node.left.value, str) and stem.fullmatch(node.left.value) \
                and any(isinstance(op, (ast.In, ast.NotIn)) for op in node.ops):
            hits.add(node.lineno)
    return sorted(hits)


def test_no_label_grep_over_model_and_train():
    probe = ('a = batch["label"]\nb = rec.get("cohort")\nc = win.severity\nd = getattr(w, "phase")\n'
             'e = "manifest" in rec\nf = p.group\ng = p.groups\nh = task_id + sex\n')
    assert label_reads(probe, "mdfm/train/probe.py") == [1, 2, 3, 4, 5, 6]
    assert label_reads(probe, "mdfm/model/mfm.py") == [1, 2, 3, 4, 5]
    hits, scanned = [], 0
    for folder in SCAN:
        for name in sorted(os.listdir(os.path.join(paths.TREE, folder))):
            if not name.endswith(".py") or name.startswith("._"):
                continue
            rel = f"{folder}/{name}"
            with open(os.path.join(paths.TREE, folder, name), encoding="utf-8") as f:
                text = f.read()
            scanned += 1
            hits += [f"{rel}:{n}" for n in label_reads(text, rel)]
            if folder == "mdfm/train":                                   # the import graph: never mdfm.gates
                for node in ast.walk(ast.parse(text)):
                    if isinstance(node, ast.Import):
                        assert not any("gates" in a.name.split(".") for a in node.names), rel
                    elif isinstance(node, ast.ImportFrom):
                        assert "gates" not in (node.module or "").split(".") and \
                            not any(a.name == "gates" for a in node.names), rel
    assert scanned >= 12 and not hits, hits


# ---------------------------------------------------------------- kernel likelihoods
@needs_ma
def test_kernel_terms_sum_to_the_model_likelihood(tiny):
    batch = M.batch_from_windows(_toy_windows(root=True), points_per_stream=48, gen=1)
    with torch.no_grad():
        post = tiny.encode(batch)
        body, pred = L.forward(tiny, batch, post, torch.Generator().manual_seed(2))
        terms = L.kernel_terms(tiny, pred, batch)
        ll = tiny.log_lik(pred, batch)
        per = tiny.nll_terms(pred, batch)
    assert set(terms) == set(L.KERNEL_TERMS)
    total = sum(terms.values())
    assert torch.allclose(total, -ll, rtol=1e-6, atol=1e-3), (total, -ll)
    assert torch.allclose(terms["pose3d_nll"] + terms["root_nll"], per["pose3d"], rtol=1e-6, atol=1e-3)
    for m in ("imu", "pose2d", "force", "emg"):
        assert torch.allclose(terms[f"{m}_nll"], per[m])
    assert bool((terms["root_nll"] != 0).all()) and bool((terms["emg_nll"][:2] != 0).all())
    assert float(terms["emg_nll"][2]) == 0.0 and float(terms["imu_nll"][2]) == 0.0


# ---------------------------------------------------------------- minimal at truth
@needs_ma
def test_kernel_likelihoods_are_minimal_at_the_truth_body(tiny, walk):
    batch, nuis = walk["batch"], walk["nuis"]
    assert float(walk["lateral"].abs().max()) < 1e-4          # the model's camera target is on the true axis
    gen = torch.Generator().manual_seed(3)
    with torch.no_grad():
        at = L.kernel_terms(tiny, tiny.render(batch.truth, batch, nuis), batch)
        # window: term present there
        present = {"imu_nll": 0, "force_nll": 0, "pose2d_nll": 1, "pose3d_nll": 2, "root_nll": 2}
        for term, i in present.items():
            assert float(at[term][i]) != 0.0, term
        for scale in (0.002, 0.005, 0.02):
            off = L.kernel_terms(tiny, tiny.render(_perturbed(batch.truth, scale, gen), batch, nuis), batch)
            for term, i in present.items():
                assert float(at[term][i]) < float(off[term][i]), (term, scale, float(at[term][i]), float(off[term][i]))
        # the rendered nuisance is also the best one: turning every mount or the camera raises the likelihood
        turned = replace(nuis.imu, mount_res6d=nuis.imu.mount_res6d + 0.05)
        worse = L.kernel_terms(tiny, tiny.render(batch.truth, batch, M.Nuisance(turned, nuis.camera,
                                                                               nuis.pose3d_log_sigma)), batch)
        assert float(worse["imu_nll"][0]) > float(at["imu_nll"][0])
        cam = replace(nuis.camera, log_distance=nuis.camera.log_distance + 0.05)
        worse = L.kernel_terms(tiny, tiny.render(batch.truth, batch, M.Nuisance(nuis.imu, cam,
                                                                               nuis.pose3d_log_sigma)), batch)
        assert float(worse["pose2d_nll"][1]) > float(at["pose2d_nll"][1])


@needs_ma
def test_supervision_is_zero_with_zero_gradient_at_the_truth_knots(tiny, walk):
    batch, tg = walk["batch"], walk["targets"]
    tr, W = batch.truth, batch.n_windows
    lengths = batch.lengths.float()
    has = torch.ones(W, dtype=torch.bool)
    w, k = batch.knots
    rot = tr.rot6d[w, k].clone().requires_grad_(True)
    tra = tr.transl[w, k].clone().requires_grad_(True)
    out = _truth_out(batch, tg, rot, tra)
    se, band = L.sup_pos_parts(out, tr, batch.gender, tiny.skeleton, lengths, has)
    zero = L.sup_rot(out, tr, lengths, has) + se + L.sup_root(out, tr, lengths, has)
    zero = zero + sum(L.sup_root_rate_parts(out, tr, batch.lengths, has, batch.rate))
    assert float(zero.detach().abs().max()) < 1e-6, zero
    grads = torch.autograd.grad(zero.sum(), [rot, tra])
    assert max(float(g.abs().max()) for g in grads) < 1e-6
    # the band NLL is minimal where exp(pos_logvar) matches the error: at 0 error, the lower bound
    assert torch.isfinite(band).all()
    contact = L.sup_contact(out, tg.contact, lengths, has)
    flipped = L.sup_contact(out._replace(contact_logit=-out.contact_logit), tg.contact, lengths, has)
    assert float(contact.max()) < 1e-5 and float(flipped.min()) > 1e3
    # the store's contacts (float64 kinematics) equal the rule on the float32 truth kinematics, the fallback of
    # stage1_losses, up to knots on a threshold
    assert float((L.truth_contacts(tiny, batch, w, k) == tg.contact[w, k]).float().mean()) >= 0.99
    # derivatives: the truth body through the same kinematics gives 0 and zero gradient
    kin_t = L.truth_kinematics(tiny, batch)
    w_pt, _, wt_pt = L.point_index(batch)
    body = tr._replace(rot6d=tr.rot6d.clone().requires_grad_(True))
    kin = tiny.kinematics(body, batch)[0]
    d = L.sup_deriv(kin, kin_t, w_pt, wt_pt, lengths, has)
    assert float(d.detach().abs().max()) < 1e-6
    assert float(torch.autograd.grad(d.sum(), body.rot6d)[0].abs().max()) < 1e-6
    # and positive off truth, in every supervised window
    gen = torch.Generator().manual_seed(4)
    pert = _perturbed(tr, 0.01, gen)
    off = _truth_out(batch, tg, pert.rot6d[w, k], pert.transl[w, k])
    se_off, _ = L.sup_pos_parts(off, tr, batch.gender, tiny.skeleton, lengths, has)
    for x in (L.sup_rot(off, tr, lengths, has), se_off, L.sup_root(off, tr, lengths, has),
              L.sup_deriv(tiny.kinematics(pert, batch)[0], kin_t, w_pt, wt_pt, lengths, has)):
        assert bool((x > 1.0).all()), x
    # sup_beta: the mean part vanishes at the true betas (zero gradient on mu)
    mu = tr.beta.clone().requires_grad_(True)
    nll = L.sup_beta(mu, torch.zeros(W, 16), tr.beta, has)
    assert float(torch.autograd.grad(nll.sum(), mu)[0].abs().max()) == 0.0
    assert bool((L.sup_beta(tr.beta + 0.3, torch.zeros(W, 16), tr.beta, has) > nll.detach()).all())
    none = L.sup_beta(mu.detach(), torch.zeros(W, 16), tr.beta, torch.zeros(W, dtype=torch.bool))
    assert float(none.abs().max()) == 0.0


def _toy_truth(lengths=(50, 30), seed=0):
    """A KnotBody truth of len(lengths) windows (random rotations, translations and betas) and a BodyOut at
    every knot holding it."""
    from mdfm.model.decoder import all_knots
    g = torch.Generator().manual_seed(seed)
    W, T = len(lengths), max(lengths)
    r6 = matrix_to_rot6d(axis_angle_to_matrix(0.3 * torch.randn(W, T, 22, 3, generator=g)))
    tr = torch.randn(W, T, 3, generator=g)
    truth = M.KnotBody(r6, tr, None, torch.randn(W, 16, generator=g), torch.tensor([0, 1][:W]),
                       torch.tensor(lengths))
    w, k = all_knots(torch.tensor(lengths))
    out = BodyOut(r6[w, k], tr[w, k], torch.zeros(len(w), 2), torch.full((len(w), 22), -8.0), w, k)
    return truth, out


def test_sup_root_scales_horizontal_and_vertical_apart():
    truth, out = _toy_truth()
    lengths, has = truth.lengths.float(), torch.ones(2, dtype=torch.bool)
    for axis, sd, part in ((0, L.SIGMAS["root_xz"], 0), (2, L.SIGMAS["root_xz"], 0), (1, L.SIGMAS["root_y"], 1)):
        shift = torch.zeros(3)
        shift[axis] = sd
        off = out._replace(transl=out.transl + shift)
        parts = L.sup_root_parts(off, truth, lengths, has)
        assert torch.allclose(parts[part], 0.5 * lengths, rtol=1e-5), (axis, parts)
        assert float(parts[1 - part].abs().max()) < 1e-6
        assert torch.allclose(L.sup_root(off, truth, lengths, has), 0.5 * lengths, rtol=1e-5)
    assert L.SIGMAS == {"rot": 0.1, "pos": 0.02, "root_xz": 0.50, "root_y": 0.02, "vel": 0.1, "acc": 2.0,
                        "root_vel": 0.1, "root_yaw_rate": 0.05}
    assert L.SIGMAS["root_xz"] == 25 * L.SIGMAS["root_y"]
    # the scales passed in: 0.20 m horizontal at sigma 0.4 costs a quarter as much
    off = out._replace(transl=out.transl + torch.tensor([0.2, 0.0, 0.0]))
    assert torch.allclose(L.sup_root_parts(off, truth, lengths, has, sigma_xz=0.4)[0], 0.125 * lengths, rtol=1e-5)
    assert L.sup_root_parts(off, truth, lengths, torch.tensor([True, False]))[0][1] == 0.0


@needs_ma
def test_sup_pos_ignores_the_drawn_beta(tiny, walk):
    """At the truth knots the squared part of sup_pos is 0 whatever beta is drawn: both bodies are posed on the
    true skeleton, so the term scores the pose and never the shape (reports/bench/phaseA_diagnosis.md, C4)."""
    batch, tg = walk["batch"], walk["targets"]
    tr, W = batch.truth, batch.n_windows
    lengths, has = batch.lengths.float(), torch.ones(W, dtype=torch.bool)
    out = _truth_out(batch, tg)
    se, _ = L.sup_pos_parts(out, tr, batch.gender, tiny.skeleton, lengths, has)
    assert float(se.abs().max()) < 1e-6
    # end to end: a draw at the truth knots with betas far from the truth leaves sup_pos at its band part
    tiny.eval()
    with torch.no_grad():
        post = tiny.encode(batch)
        for beta in (tr.beta + 1.5, tr.beta - 2.0):
            dense = tiny.dense(out, beta, batch)
            kin = tiny.kinematics(dense, batch)
            body = L.BodyDraw(torch.zeros_like(post.mu_z), beta, out, dense, kin)
            pred = tiny.render(dense, batch, post.nuis, kin)
            res = L.stage1_losses(tiny, batch, post, body, pred, "A", targets=tg)
            assert float((res["sup_pos"] - res["sup_pos_band"]).abs().max()) < 1e-6
            assert float(res["sup_rot"].abs().max()) < 1e-6 and float(res["sup_root"].abs().max()) < 1e-6
            assert float(res["sup_root_rate"].abs().max()) < 1e-6
            assert torch.allclose(res["sup_root"], res["sup_root_xz"] + res["sup_root_y"])
            assert torch.allclose(res["sup_root_rate"], res["sup_root_rate_u"] + res["sup_root_rate_yaw"])


@needs_ma
def test_on_truth_windows_only_the_shape_terms_move_the_shape_posterior(tiny, walk):
    """Synthetic walk windows with a body truth: the gradient of sup_rot + sup_pos + sup_root + sup_deriv +
    imu_nll + plausibility with respect to the shape posterior mean is exactly 0, that of sup_beta is not;
    without the truth mask the kinematics pose the draw on the drawn betas and the kernels reach mu_b."""
    from dataclasses import replace as dreplace
    batch, tg = walk["batch"], walk["targets"]
    tiny.eval()
    truth_windows = L.truth_mask(batch, tg)
    assert bool(truth_windows.all())
    pose_terms = ("sup_rot", "sup_pos", "sup_root", "sup_root_rate", "sup_deriv", "imu_nll", "plausibility")
    grads = {}
    for name, tw in (("masked", truth_windows), ("unmasked", None)):
        post = tiny.encode(batch)
        post = dreplace(post, mu_b=post.mu_b.detach().clone().requires_grad_(True))
        body, pred = L.forward(tiny, batch, post, torch.Generator().manual_seed(3), truth_windows=tw)
        out = L.stage1_losses(tiny, batch, post, body, pred, "A", targets=tg)
        assert float(out["imu_nll"][0].detach()) != 0.0 and float(out["sup_deriv"].detach().min()) > 0.0
        pose = sum(out[t].sum() for t in pose_terms)
        g = torch.autograd.grad(pose, post.mu_b, retain_graph=True, allow_unused=True)[0]
        grads[name] = 0.0 if g is None else float(g.abs().max())
        gb = torch.autograd.grad(out["sup_beta"].sum(), post.mu_b)[0]
        assert float(gb.abs().max()) > 0.0
    assert grads["masked"] == 0.0, grads
    assert grads["unmasked"] > 0.0, grads


def test_the_config_sigmas_reach_the_supervision_terms():
    """LossConfig carries SIGMAS; the pilot config names every weight and sigma of losses.WEIGHTS and losses.SIGMAS
    with their values (root-motion diagnosis: root_xz 0.50, root_vel 0.1, root_yaw_rate 0.05, sup_root_rate 1 in
    both phases), and config.problems refuses a config without the new keys."""
    from dataclasses import replace as dreplace
    from mdfm.train import config as CF
    cfg = L.LossConfig()
    assert cfg.sigmas == L.SIGMAS and cfg.sigmas is not L.SIGMAS
    rc = CF.load(os.path.join(paths.TREE, "configs", "stage1_pilot.yaml"))
    assert CF.problems(rc) == []
    assert dict(rc.losses.sigmas) == L.SIGMAS
    assert {k: tuple(v) for k, v in rc.losses.weights.items()} == L.WEIGHTS
    for key in ("root_vel", "root_yaw_rate"):
        bad = dreplace(rc, losses=dreplace(rc.losses, sigmas={k: v for k, v in L.SIGMAS.items() if k != key}))
        assert any("sigmas" in p for p in CF.problems(bad)), key
    bad = dreplace(rc, losses=dreplace(rc.losses, weights={k: v for k, v in rc.losses.weights.items()
                                                            if k != "sup_root_rate"}))
    assert any("weights" in p for p in CF.problems(bad))


@needs_ma
def test_nuisance_aux_is_zero_at_the_rendered_nuisance(tiny, walk):
    batch, tg, nuis = walk["batch"], walk["targets"], walk["nuis"]
    with torch.no_grad():
        pred = tiny.render(batch.truth, batch, nuis)
        at = L.nuis_aux(tiny, nuis, batch, tg, pred)
        assert float(at.abs().max()) < 1e-6, at
        assert bool(tg.imu_known[0].any()) and bool(tg.cam_known[1])
        turned = replace(nuis.imu, mount_res6d=nuis.imu.mount_res6d + 0.05, bias_acc=nuis.imu.bias_acc + 0.01)
        n2 = M.Nuisance(turned, replace(nuis.camera, log_focal=nuis.camera.log_focal + 0.1), nuis.pose3d_log_sigma)
        off = L.nuis_aux(tiny, n2, batch, tg, tiny.render(batch.truth, batch, n2))
    assert float(off[0]) > 1.0 and float(off[1]) > 0.5 and float(off[2]) == 0.0, off
    assert float(L.nuis_aux(tiny, n2, batch, None).abs().max()) == 0.0


@needs_ma
def test_plausibility_at_truth_sits_below_perturbed_bodies(tiny, walk):
    batch, tg = walk["batch"], walk["targets"]
    tr, W = batch.truth, batch.n_windows
    lengths = batch.lengths.float()
    w_pt, t_pt, wt_pt = L.point_index(batch)
    w, k = batch.knots
    logits = torch.zeros(W, batch.T, 2)
    logits[w, k] = 20.0 * (2 * tg.contact[w, k] - 1)
    with torch.no_grad():
        body = tr._replace(contact_logit=logits)
        at = L.plausibility(_truth_out(batch, tg), body, tiny.kinematics(body, batch)[0], w_pt, t_pt, wt_pt, lengths,
                            batch.rate)
        assert float(at["plaus_limits"].abs().max()) == 0.0                 # the walk stays inside the limits
        assert float(at["plaus_penetration"].max()) < 0.01                   # the joint floor of the take
        spread = L.bone_spread(tiny.kinematics(body, batch)[0].p, w_pt, W)
        assert float(spread.max()) < 1e-5                                  # one rest skeleton per window
        gen = torch.Generator().manual_seed(6)
        for scale in (0.005, 0.02):
            pb = _perturbed(tr, scale, gen)._replace(contact_logit=logits)
            off = L.plausibility(_truth_out(batch, tg, pb.rot6d[w, k], pb.transl[w, k]), pb,
                                 tiny.kinematics(pb, batch)[0], w_pt, t_pt, wt_pt, lengths, batch.rate)
            ok = bool((off["plausibility"] > at["plausibility"]).all())
            assert ok, (scale, off["plausibility"], at["plausibility"])


# ---------------------------------------------------------------- centres of the other terms
@needs_ma
def test_kl_terms_and_the_nuisance_prior_are_minimal_at_their_centres(tiny):
    tiny.eval()
    batch = M.batch_from_windows(_toy_windows(), points_per_stream=32, gen=0)
    W, P = batch.tokens.lat_idx.shape
    mask = batch.tokens.lat_idx >= 0
    gen = torch.Generator().manual_seed(5)
    z = torch.randn(W, P, 32, generator=gen) * mask[..., None]
    beta = torch.randn(W, 16, generator=gen)
    cond = batch.cond(beta)
    with torch.no_grad():
        mu_p, ls_p = tiny.prior.params(z, cond, mask)
    with torch.no_grad():
        kl0 = tiny.prior.kl(mu_p, 2 * ls_p, z, cond, free_bits=0.0, mask=mask)
        floor = tiny.prior.kl(mu_p, 2 * ls_p, z, cond, mask=mask)
        off = tiny.prior.kl(mu_p + 0.5, 2 * ls_p, z, cond, free_bits=0.0, mask=mask)
    assert float(kl0.abs().max()) < 1e-5
    assert torch.allclose(floor, 0.02 * 32 * mask.sum(1).float(), rtol=1e-5)
    assert bool((off > kl0 + 1.0).all())
    # the shape KL of a diagonal q is minimal at the prior mean with precision diag(Sigma^-1)
    sp = tiny.shape_prior
    g = batch.gender
    prec = torch.diagonal(torch.cholesky_inverse(sp.chol[g]), dim1=-2, dim2=-1)
    mu = sp.mean[g].clone().requires_grad_(True)
    lv = (-torch.log(prec)).clone().requires_grad_(True)
    kb = sp.kl(mu, lv, g)
    gm, gl = torch.autograd.grad(kb.sum(), [mu, lv])
    assert float(gm.abs().max()) < 1e-4 and float(gl.abs().max()) < 1e-4
    assert bool((sp.kl(mu.detach() + 0.2, lv.detach(), g) > kb.detach()).all())
    assert bool((sp.kl(mu.detach(), lv.detach() + 0.3, g) > kb.detach()).all())
    # nuisance prior: minimal at the prior centre (zero residuals and biases) on windows with an IMU
    d = tiny.cfg.d
    c0 = tiny.heads.imu(torch.zeros(W, 14, d), batch.win["imu_family"])
    res = c0.mount_res6d.detach().clone().requires_grad_(True)
    ba = c0.bias_acc.detach().clone().requires_grad_(True)
    nuis = M.Nuisance(replace(c0, mount_res6d=res, bias_acc=ba), None, None)
    npr = L.nuis_prior(tiny, nuis, batch)
    gr, gb = torch.autograd.grad(npr.sum(), [res, ba])
    assert float(gr.abs().max()) < 1e-6 and float(gb.abs().max()) < 1e-6
    off = L.nuis_prior(tiny, M.Nuisance(replace(c0, mount_res6d=res.detach() + 0.05), None, None), batch)
    imu_windows = batch.win["imu_present"].any(-1)
    assert bool((off[imu_windows] > npr.detach()[imu_windows]).all()) and bool((off[~imu_windows] == 0).all())


def test_consistency_subject_and_amortization_vanish_at_agreement():
    gen = torch.Generator().manual_seed(11)
    W, P, D = 6, 5, 32
    mu = torch.randn(W, P, D, generator=gen)
    lv = 0.3 * torch.randn(W, P, D, generator=gen)
    mask = torch.ones(W, P, dtype=torch.bool)
    mask[4, 3:] = False
    pair = torch.tensor([0, 0, 1, 1, 1, -1])
    same_mu, same_lv = mu.clone(), lv.clone()
    for a, b in ((1, 0), (3, 2), (4, 2)):
        same_mu[a], same_lv[a] = mu[b], lv[b]
    same_mu[4, 3:] = 9.0                                                # padding patches do not count
    assert float(L.consistency(same_mu, same_lv, mask, pair).abs().max()) < 1e-5
    c = L.consistency(mu, lv, mask, pair)
    assert bool((c[:5] > 1.0).all()) and float(c[5]) == 0.0
    skl = lambda a, b: float(((gaussian_kl(mu[a], lv[a], mu[b], 0.5 * lv[b])  # noqa: E731
                               + gaussian_kl(mu[b], lv[b], mu[a], 0.5 * lv[a])) * (mask[a] & mask[b])[..., None]).sum())
    assert math.isclose(float(c[0]), skl(0, 1) / 2, rel_tol=1e-5)
    assert math.isclose(float(c[2]), (skl(2, 3) + skl(2, 4)) / 2, rel_tol=1e-5)
    assert math.isclose(float(c.sum()), skl(0, 1) + skl(2, 3) + skl(2, 4) + skl(3, 4), rel_tol=1e-5)
    assert float(L.consistency(mu, lv, mask, None).abs().max()) == 0.0
    # subject: persons with two or three windows; agreement gives 0, disagreement a positive KL, lone windows 0
    mu_b, lv_b = torch.randn(W, 16, generator=gen), 0.2 * torch.randn(W, 16, generator=gen)
    person = torch.tensor([7, 7, 3, 3, 3, 5])
    agree_mu, agree_lv = mu_b.clone(), lv_b.clone()
    agree_mu[1], agree_lv[1] = mu_b[0], lv_b[0]
    agree_mu[3:5], agree_lv[3:5] = mu_b[2], lv_b[2]
    s0 = L.subject(agree_mu, agree_lv, person)
    assert float(s0.abs().max()) < 1e-5
    s = L.subject(mu_b, lv_b, person)
    assert bool((s[:5] > 0.1).all()) and float(s[5]) == 0.0
    # against the product of Gaussians itself, agreement would cost 8 (n - 1 - log n) nats per window
    pm, pl = subject_pool(agree_mu, agree_lv, person)
    prod = gaussian_kl(agree_mu, agree_lv, pm, 0.5 * pl).sum(-1)
    n = torch.tensor([2.0, 2.0, 3.0, 3.0, 3.0])
    assert torch.allclose(prod[:5], 8 * (n - 1 - torch.log(n)), rtol=1e-4)
    # amortization: 0 at the refined means, gradient on the amortized means only
    Post = namedtuple("Post", ["mu_z", "logvar_z", "mask"])
    amu = mu.clone().requires_grad_(True)
    alv = lv.clone().requires_grad_(True)
    ref = mu.clone().requires_grad_(True)
    assert float(L.amortization(Post(amu, alv, mask), ref).detach().abs().max()) == 0.0
    a = L.amortization(Post(amu, alv, mask), ref + 0.1)
    expect = (0.01 * torch.exp(-lv) * mask[..., None]).sum((1, 2))
    assert torch.allclose(a, expect, rtol=1e-5)
    a.sum().backward()
    assert amu.grad is not None and float(amu.grad.abs().sum()) > 0
    assert alv.grad is None and ref.grad is None


def test_hsic_activation_rule():
    gen = torch.Generator().manual_seed(21)
    base = torch.randn(8, 1, 32, generator=gen)
    mask = torch.ones(16, 1, dtype=torch.bool)
    # two sources holding the same summaries: empirically independent, HSIC 0
    mu = torch.cat([base, base])
    src = torch.tensor([1] * 8 + [2] * 8)
    task = torch.zeros(16, dtype=torch.int64)
    h, active = L.hsic(mu, mask, src, task)
    assert bool(active.all()) and float(h.abs().max()) < 1e-6
    # source-coded summaries: positive
    coded = torch.cat([base, base + 3.0])
    h, _ = L.hsic(coded, mask, src, task)
    assert float(h.min()) > 0.05 and torch.allclose(h, h[0].expand(16))
    # a stratum with one source is inactive, whatever its summaries; the other stratum keeps its value
    task2 = torch.tensor([0] * 4 + [1] * 8 + [0] * 4)
    src2 = torch.tensor([1] * 4 + [1] * 8 + [2] * 4)
    h2, active2 = L.hsic(coded, mask, src2, task2)
    assert active2.tolist() == [True] * 4 + [False] * 8 + [True] * 4
    assert float(h2[4:12].abs().max()) == 0.0 and float(h2[0]) > 0.05
    assert torch.allclose(h2[0], L.hsic_gaussian(coded[active2].mean(1), src2[active2]))
    # synthetic windows: one code of their own, or left out of the term
    syn = torch.tensor([True] * 8 + [False] * 8)
    src3 = torch.tensor([0] * 8 + [2] * 8)
    h3, a3 = L.hsic(coded, mask, src3, task, syn, include_synthetic=True)
    assert bool(a3.all()) and float(h3.min()) > 0.05
    h4, a4 = L.hsic(coded, mask, src3, task, syn, include_synthetic=False)
    assert not bool(a4.any()) and float(h4.abs().max()) == 0.0
    # the summaries are the means over valid patches: padding does not move them
    padded = torch.cat([coded, torch.full((16, 1, 32), 50.0)], 1)
    pmask = torch.cat([mask, torch.zeros(16, 1, dtype=torch.bool)], 1)
    assert torch.allclose(L.hsic(padded, pmask, src, task)[0], L.hsic(coded, mask, src, task)[0])
    # gradient flows to the summaries
    x = coded.clone().requires_grad_(True)
    L.hsic(x, mask, src, task)[0].sum().backward()
    assert float(x.grad.abs().sum()) > 0


def _hsic_matrices(x, codes):
    """Biased HSIC tr(K H L H) / n^2 with explicit n x n centring matrices (numpy): K exp(-d^2 / m), m the
    lower median of the off-diagonal squared distances (torch.median's convention); L exp(-|e_a - e_b|^2 / 2)
    on one-hot codes."""
    n = len(x)
    d2 = ((x[:, None] - x[None]) ** 2).sum(-1)
    off = np.sort(d2[~np.eye(n, dtype=bool)])
    K = np.exp(-d2 / off[(len(off) - 1) // 2])
    onehot = np.eye(codes.max() + 1)[codes]
    Lk = np.exp(-((onehot[:, None] - onehot[None]) ** 2).sum(-1) / 2.0)
    H = np.eye(n) - np.full((n, n), 1.0 / n)
    return np.trace(K @ H @ Lk @ H) / n ** 2


def test_hsic_equals_the_matrix_form():
    rng = np.random.default_rng(3)
    for trial in range(20):
        n = int(rng.integers(3, 30))
        x = rng.standard_normal((n, 32))
        codes = rng.integers(0, 4, n)
        if trial % 2:
            x = x + 2.0 * codes[:, None]
        ref = _hsic_matrices(x, codes)
        got = float(L.hsic_gaussian(torch.tensor(x), torch.tensor(codes)))
        assert math.isclose(got, ref, rel_tol=1e-9, abs_tol=1e-15), (trial, got, ref)


def test_plausibility_parts_vanish_inside_the_limits():
    lo, hi = L.joint_limits(F64)
    mid = axis_angle_to_matrix((lo + hi) / 2)                          # (22, 3, 3) inside every limit
    r6 = matrix_to_rot6d(mid)[None].repeat(3, 1, 1)
    r6[1, 4] = matrix_to_rot6d(axis_angle_to_matrix(torch.tensor([float(hi[4, 0]) + 0.1, 0.0, 0.0], dtype=F64)))
    # the root is taken in its heading frame: a pure yaw of the root costs nothing
    r6[2, 0] = matrix_to_rot6d(rot_y(torch.tensor(2.0, dtype=F64)) @ mid[0])
    r6.requires_grad_(True)
    h = L.limit_hinge(r6, lo, hi)
    hv = h.detach()
    assert float(hv[0]) == 0.0 and float(hv[2]) < 1e-18
    lh = matrix_to_axis_angle(axis_angle_to_matrix(torch.tensor([float(hi[4, 0]) + 0.1, 0.0, 0.0], dtype=F64)))
    excess = (torch.relu(lo[4] - lh) + torch.relu(lh - hi[4]))
    assert math.isclose(float(hv[1]), float((excess ** 2).sum()) / (2 * L.LIMIT_SD ** 2), rel_tol=1e-6)
    assert float(hv[1]) > 1.0
    g = torch.autograd.grad(h.sum(), r6)[0]
    assert float(g[0].abs().max()) == 0.0 and float(g[1].abs().max()) > 0
    # the log map equals matrix_to_axis_angle and keeps finite gradients at the identity
    aa = torch.randn(200, 3, dtype=F64, generator=torch.Generator().manual_seed(2))
    angle = torch.rand(200, 1, dtype=F64, generator=torch.Generator().manual_seed(3)) * 3.0
    aa = aa / aa.norm(dim=-1, keepdim=True) * angle
    R = axis_angle_to_matrix(aa)
    assert torch.allclose(L.log_map(R), matrix_to_axis_angle(R), atol=1e-6)
    eye = torch.eye(3, dtype=F64).repeat(2, 1, 1).requires_grad_(True)
    assert torch.isfinite(torch.autograd.grad(L.log_map(eye).sum(), eye)[0]).all()
    # penetration: 0 above the floor, (depth / sd)^2 / 2 per joint below it
    p = torch.zeros(2, 22, 3)
    p[:, :, 1] = 0.01
    p[1, 10, 1] = -0.02
    pen = L.penetration(p)
    assert float(pen[0]) == 0.0 and math.isclose(float(pen[1]), 0.02 ** 2 / (2 * L.PENETRATION_SD ** 2), rel_tol=1e-5)
    # skate: 0 without contact or without horizontal speed; contact enters detached
    v = torch.zeros(3, 22, 3)
    v[:, :, 1] = 0.5                                                   # vertical speed only
    v[2, [7, 10], 0] = 0.1                                             # left foot sliding at 0.1 m/s
    contact = torch.tensor([[1.0, 1.0], [0.0, 0.0], [1.0, 0.0]], requires_grad=True)
    sk = L.skate(v, contact)
    assert float(sk[0]) == 0.0 and float(sk[1]) == 0.0
    assert math.isclose(float(sk[2]), 0.1 ** 2 / (2 * L.SKATE_SD ** 2), rel_tol=1e-5)
    assert not sk.requires_grad
    # the per-window sums: L_w times the mean over the window's knots or weighted points
    x, w = torch.tensor([1.0, 3.0, 5.0]), torch.tensor([0, 0, 1])
    lengths = torch.tensor([10.0, 4.0, 7.0])
    assert L.knot_estimate(x, w, lengths, 3).tolist() == [20.0, 20.0, 0.0]
    assert L.point_estimate(x, w, torch.tensor([1.0, 3.0, 2.0]), lengths, 3).tolist() == [25.0, 20.0, 0.0]


# ---------------------------------------------------------------- the total
@needs_ma
def test_stage1_losses_weigh_the_phases_and_backpropagate(tiny):
    tiny.train()
    torch.manual_seed(1)
    batch = M.batch_from_windows(_toy_windows(root=True), points_per_stream=48, gen=0)
    post = tiny.encode(batch)
    body, pred = L.forward(tiny, batch, post, torch.Generator().manual_seed(2))
    outs = {}
    for phase, progress in (("A", 0.0), ("A", 0.05), ("A", 0.5), ("B", 0.0)):
        torch.manual_seed(3)                                          # the prior's condition dropout
        outs[phase, progress] = L.stage1_losses(tiny, batch, post, body, pred, phase, progress=progress)
    syn = L.is_synthetic(batch)
    assert syn.tolist() == [True, False, False]
    vals = {key: {k: v.detach() for k, v in o.items()} for key, o in outs.items()}
    for (phase, progress), out in vals.items():
        assert set(out) == set(L.TERMS) | set(L.DIAGNOSTICS) | {"per_window", "total"}
        assert all(out[t].shape == (3,) and bool(torch.isfinite(out[t]).all()) for t in L.TERMS + L.DIAGNOSTICS)
        wts = L.term_weights(phase, progress)
        expect = torch.zeros(3)
        for t in L.TERMS:
            x = out[t].detach()
            if phase == "A" and t == "force_nll":
                x = torch.where(syn, x, torch.zeros_like(x))
            expect = expect + wts[t] * x
        assert torch.allclose(out["per_window"].detach(), expect, rtol=1e-5, atol=1e-2)
        assert math.isclose(float(out["total"]), float(out["per_window"].mean()), rel_tol=1e-6)
    a0, a5, a50, b = (L.term_weights(*k) for k in (("A", 0.0), ("A", 0.05), ("A", 0.5), ("B", 0.0)))
    assert a0["kl_z"] == 0.0 and a5["kl_z"] == 0.5 and a50["kl_z"] == 1.0 and b["kl_z"] == 1.0
    assert a0["kl_beta"] == 0.0 and a5["kl_beta"] == 0.5
    for t in ("root_nll", "emg_nll", "hsic", "amortization"):
        assert a50[t] == 0.0 and b[t] > 0.0
    assert a50["nuis_aux"] == 0.1 and b["nuis_aux"] == 0.0 and b["sup_deriv"] == 0.1 and b["consistency"] == 0.5
    out = vals["B", 0.0]
    assert bool((out["root_nll"] != 0).all()) and float(out["force_nll"][1]) != 0.0
    assert float(out["hsic_active"].sum()) == 3.0 and float(out["hsic"][0]) > 0     # task 0 holds three sources
    assert float(out["bone_spread"].max()) < 1e-5
    # a real window's force counts in phase B only
    fa = vals["A", 0.5]
    assert math.isclose(float(out["per_window"][1] - fa["per_window"][1]),
                        float(out["force_nll"][1] + out["root_nll"][1] + out["emg_nll"][1] + 0.1 * out["hsic"][1]
                              - 0.1 * fa["nuis_aux"][1]), rel_tol=1e-4, abs_tol=1e-2)
    tiny.zero_grad(set_to_none=True)
    outs["B", 0.0]["total"].backward()
    grads = [p.grad for p in tiny.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert float(tiny.decoder.out.weight.grad.abs().sum()) > 0
    assert float(tiny.latent_head.proj.weight.grad.abs().sum()) > 0
    tiny.zero_grad(set_to_none=True)
    tiny.eval()


def test_weights_are_the_spec_table():
    """WEIGHTS against the weight column of the section 4 table of the spec (read-only): '<w>, A and B',
    '<w>, B', '<w>, A only', force '1, B (synthetic force from A)', the KL rows annealed to 1 in both."""
    with open(os.path.join(paths.TREE, "docs", "MFM_MODEL_CODE_SPEC_2026-09-28.md"), encoding="utf-8") as f:
        section = f.read().split("## 4. Losses", 1)[1].split("## 5.", 1)[0]
    rows = {}
    for line in section.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if line.startswith("| ") and len(cells) == 4 and cells[0] not in ("term",):
            rows[cells[0]] = cells[3]
    assert set(rows) == set(L.TERMS), set(rows) ^ set(L.TERMS)
    for term, text in rows.items():
        if text.startswith("annealed") or text == "same schedule":
            expect = (1.0, 1.0)
        else:
            w, when = text.split(",", 1)
            w, when = float(w), when.strip()
            expect = {"A and B": (w, w), "A only": (w, 0.0), "B": (0.0, w)}.get(when)
            if when.startswith("B (synthetic force from A)"):
                expect = (w, w)                                         # masked to synthetic windows in A
        assert L.WEIGHTS[term] == expect, (term, text, L.WEIGHTS[term])
    assert L.SYNTHETIC_ONLY_IN_A == ("force_nll",) and L.ANNEALED == ("kl_z", "kl_beta") and L.KL_ANNEAL == 0.1


def test_the_jepa_term_is_a_mean_smooth_l1_per_window():
    """jepa_per_window (task J2a; docs/proposals/JEPA_EXPLORATION_DRAFT.md, 2.6): smooth L1 with beta 1 (0.5 x^2
    below 1, |x| - 0.5 above) averaged over the features, then over each window's masked patches; 0 for a window
    without one; not one of TERMS (its weight is the config's jepa.weight)."""
    pred = torch.tensor([[0.0, 0.0], [2.0, 0.0], [0.5, -0.5]])
    target = torch.zeros(3, 2)
    window = torch.tensor([0, 0, 2])
    out = L.jepa_per_window(pred, target, window, 4)
    assert out.dtype == torch.float32 and out.tolist() == pytest.approx([0.375, 0.0, 0.125, 0.0])
    assert L.jepa_per_window(pred[:0], target[:0], window[:0], 3).tolist() == [0.0, 0.0, 0.0]
    p = pred.clone().requires_grad_(True)
    L.jepa_per_window(p, target, window, 4).sum().backward()
    assert p.grad.reshape(-1).tolist() == pytest.approx([0.0, 0.0, 0.25, 0.0, 0.25, -0.25])
    assert "jepa" not in L.TERMS and "jepa" not in L.WEIGHTS and L.JEPA_BETA == 1.0


@needs_ma
def test_targets_with_a_truth_the_batch_lacks_raise(tiny, walk):
    """A mixed batch (a synthetic window with a body, a real one without) carries no Batch.truth: targets that
    mark the synthetic window's truth raise instead of switching its supervision off without a word."""
    wins = walk["windows"][:1] + [FX.make_window(1.5, seed=3, person="p1", sex=0, source="weargait_pd")]
    batch = M.batch_from_windows(wins, points_per_stream=32, gen=0)
    tg = L.targets_from_windows(wins, T=batch.T)
    assert batch.truth is None and tg.has_truth.tolist() == [True, False]
    with pytest.raises(ValueError, match="direct supervision"):
        L.truth_mask(batch, tg)
    tiny.eval()
    with torch.no_grad():
        post = tiny.encode(batch)
        body, pred = L.forward(tiny, batch, post, torch.Generator().manual_seed(1))
        with pytest.raises(ValueError, match="direct supervision"):
            L.stage1_losses(tiny, batch, post, body, pred, "B", targets=tg)
        bare = L.stage1_losses(tiny, batch, post, body, pred, "B")
    assert float(bare["sup_rot"].abs().max()) == 0.0
    with pytest.raises(ValueError, match="has_truth"):
        L.truth_mask(walk["batch"], replace(walk["targets"], has_truth=torch.ones(2, dtype=torch.bool)))
    no_contact = dict(walk["windows"][0].body)
    del no_contact["contact"]
    with pytest.raises(ValueError, match="contact"):
        L.targets_from_windows([replace(walk["windows"][0], body=no_contact)])


@needs_ma
def test_noisy_renders_of_store_windows_score_best_at_truth(tiny):
    """Default-mixture renders with device noise of train body-store windows (SynthSampler.sample, a third
    rendered twice): the render pairs of targets_from_windows are the sampler's; supervision is 0 at the truth
    knots; every kernel term present is lower at the truth body with the rendered nuisance than at bodies
    perturbed by 0.005 and 0.02."""
    from mdfm.data.body_store import BodyStore, store_root
    from mdfm.factory.synth import SynthSampler
    if not os.path.isdir(os.path.join(store_root(), "train")):
        pytest.skip("body store not built")
    wins, pairs = SynthSampler(store=BodyStore(split="train"), device="cpu").sample(np.random.default_rng(5), 6,
                                                                                    pair_share=0.34)
    batch = M.batch_from_windows(wins, points_per_stream=128, gen=0)
    tg = L.targets_from_windows(wins, T=batch.T)
    expect = [-1] * len(wins)
    for n, (a, b) in enumerate(pairs):
        expect[a] = expect[b] = n
    assert pairs and tg.pair.tolist() == expect
    tr, lengths, has = batch.truth, batch.lengths.float(), tg.has_truth
    w, k = batch.knots
    out = _truth_out(batch, tg)
    se, _ = L.sup_pos_parts(out, tr, batch.gender, tiny.skeleton, lengths, has)
    for x in (L.sup_rot(out, tr, lengths, has), se, L.sup_root(out, tr, lengths, has)):
        assert float(x.abs().max()) < 1e-6
    assert float((L.truth_contacts(tiny, batch, w, k) == tg.contact[w, k]).float().mean()) >= 0.99
    nuis, _ = _true_nuisance(tiny, batch, tg)
    gen = torch.Generator().manual_seed(9)
    with torch.no_grad():
        at = L.kernel_terms(tiny, tiny.render(tr, batch, nuis), batch)
        seen = 0
        for scale in (0.005, 0.02):
            off = L.kernel_terms(tiny, tiny.render(_perturbed(tr, scale, gen), batch, nuis), batch)
            for t in L.KERNEL_TERMS:
                present = at[t] != 0
                seen += int(present.sum())
                assert bool((at[t][present] < off[t][present]).all()), (t, scale, at[t], off[t])
    assert seen >= 2 * len(wins)


@needs_ma
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_gpu_terms_are_float32_under_bf16_autocast():
    torch.manual_seed(0)
    model = M.MFM(M.TINY).cuda().eval()
    wins = _toy_windows(root=True)
    batch = M.batch_from_windows(wins, points_per_stream=48, gen=0).to("cuda")
    tg = L.targets_from_windows(wins, T=batch.T)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        post = model.encode(batch)
        body, pred = L.forward(model, batch, post, torch.Generator(device="cuda").manual_seed(1))
        out = L.stage1_losses(model, batch, post, body, pred, "B", targets=tg)
    for t in L.TERMS + L.DIAGNOSTICS + ("per_window",):
        assert out[t].dtype == torch.float32 and out[t].device.type == "cuda", t
        assert bool(torch.isfinite(out[t]).all()), t
    out["total"].backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


@needs_ma
def test_refined_steps_add_the_amortization_term(tiny):
    from mdfm.model.refine import refine, straight_through
    tiny.eval()
    batch = M.batch_from_windows(_toy_windows(), points_per_stream=32, gen=0)
    post = tiny.encode(batch)
    st = straight_through(post, refine(tiny, post, batch, 2))
    body, pred = L.forward(tiny, batch, st, torch.Generator().manual_seed(2))
    out = L.stage1_losses(tiny, batch, st, body, pred, "B", amortized=post)
    assert bool((out["amortization"] > 0).all())
    tiny.zero_grad(set_to_none=True)
    out["amortization"].sum().backward()
    assert float(tiny.latent_head.proj.weight.grad.abs().sum()) > 0
    tiny.zero_grad(set_to_none=True)
    without = L.stage1_losses(tiny, batch, st, body, pred, "B")
    assert float(without["amortization"].abs().max()) == 0.0


@needs_ma
def test_synthetic_batch_end_to_end_with_targets(tiny, walk):
    tiny.eval()
    batch, tg = walk["batch"], walk["targets"]
    assert tg.pair.tolist() == [0, 0, 0]              # one body under three layouts: one render group
    with torch.no_grad():
        post = tiny.encode(batch)
        body, pred = L.forward(tiny, batch, post, torch.Generator().manual_seed(8))
        out = L.stage1_losses(tiny, batch, post, body, pred, "A", targets=tg, progress=1.0)
        bare = L.stage1_losses(tiny, batch, post, body, pred, "A", progress=1.0)
        kt = replace(tg, kin=L.truth_kinematics(tiny, batch))
        with_kin = L.stage1_losses(tiny, batch, post, body, pred, "A", targets=kt)
    for t in ("sup_rot", "sup_pos", "sup_root", "sup_deriv", "sup_contact", "sup_beta", "consistency"):
        assert bool((out[t] > 0).all()), t
    assert float(out["nuis_aux"][0]) > 0 and float(out["nuis_aux"][1]) > 0 and float(out["nuis_aux"][2]) == 0.0
    # without targets: the same supervision with the contact rule on the truth kinematics, no pair, no aux
    assert torch.allclose(bare["sup_contact"], out["sup_contact"], rtol=1e-5)
    assert torch.allclose(bare["sup_deriv"], out["sup_deriv"], rtol=1e-5)
    assert float(bare["consistency"].abs().max()) == 0.0 and float(bare["nuis_aux"].abs().max()) == 0.0
    # targets carrying the truth kinematics give the same sup_deriv
    assert torch.allclose(with_kin["sup_deriv"], out["sup_deriv"], rtol=1e-6)
    # the supervision scales of the config reach the terms: doubled sigmas quarter the squared parts
    wide = L.LossConfig(sigmas={k: 2 * v for k, v in L.SIGMAS.items()})
    with torch.no_grad():
        w2 = L.stage1_losses(tiny, batch, post, body, pred, "A", targets=tg, progress=1.0, cfg=wide)
    for t in ("sup_rot", "sup_root", "sup_root_xz", "sup_root_y", "sup_root_rate", "sup_root_rate_u",
              "sup_root_rate_yaw", "sup_deriv"):
        assert torch.allclose(w2[t], out[t] / 4, rtol=1e-4), t
    assert bool((out["sup_root_rate"] > 0).all())
    assert torch.allclose(w2["sup_pos"] - w2["sup_pos_band"], (out["sup_pos"] - out["sup_pos_band"]) / 4, rtol=1e-4)


# ---------------------------------------------------------------- root motion (reports/bench/root_motion_diagnosis.md)
def _rest22(seed=0):
    """Rest joints (22, 3) float64 without the SMPL+H files: a tree of 5 to 25 cm bones, the pelvis 0.9 m up."""
    from mdfm.body.smplh import PARENTS
    g = torch.Generator().manual_seed(seed)
    bones = torch.randn(22, 3, generator=g, dtype=F64)
    bones = bones / bones.norm(dim=-1, keepdim=True) * (0.05 + 0.2 * torch.rand(22, 1, generator=g, dtype=F64))
    pos = [torch.tensor([0.0, 0.9, 0.0], dtype=F64)]
    for j in range(1, 22):
        pos.append(pos[PARENTS[j]] + bones[j])
    return torch.stack(pos)


def _walk_truth(T=1001, omega=0.4, v=1.2, sway=True, seed=0):
    """A synthetic walk in the window frame, float64: a circle of speed v and yaw rate omega about the centre knot
    T // 2 (pelvis x, z and yaw 0 there), the joints swinging at 1 Hz (sway), rest joints _rest22."""
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(T, dtype=F64) / 100.0
    tau = t - t[T // 2]
    psi = omega * tau
    X = torch.stack([v / omega * (1 - torch.cos(psi)), v / omega * torch.sin(psi)], -1) if omega else \
        torch.stack([torch.zeros_like(tau), v * tau], -1)
    aa = torch.zeros(T, 22, 3, dtype=F64)
    if sway:
        amp = 0.3 * torch.randn(22, 3, generator=g, dtype=F64)
        aa[:, 1:] = amp[1:] * torch.sin(2 * math.pi * 1.0 * t)[:, None, None]
    R = axis_angle_to_matrix(aa)
    R[:, 0] = rot_y(psi)
    rest = _rest22(seed)
    transl = torch.stack([X[:, 0], torch.zeros_like(tau), X[:, 1]], -1)       # pelvis = transl + rest[0]
    return matrix_to_rot6d(R), transl, rest


def _truth_body(r6, tr, lengths=None):
    T = r6.shape[0]
    lengths = torch.tensor([T]) if lengths is None else lengths
    return M.KnotBody(r6[None], tr[None], None, torch.zeros(1, 16, dtype=F64), torch.tensor([2]), lengths)


def test_sup_root_rate_is_zero_at_truth_and_scales_apart():
    """At the true grid rates (losses.truth_root_grid) the term and its gradient are 0; a heading-frame velocity
    error of sigma_u alone and a yaw-rate error of sigma_w alone each cost 0.5 L_w, on windows with truth only; a
    body without a root grid raises."""
    r6, tr, _ = _walk_truth()
    truth = M.KnotBody(torch.stack([r6, r6]), torch.stack([tr, tr]), None, torch.zeros(2, 16, dtype=F64),
                       torch.tensor([2, 2]), torch.tensor([1001, 617]))
    lengths = truth.lengths
    has = torch.tensor([True, False])
    tg = L.truth_root_grid(truth, 100.0, lengths)
    u = tg.u.clone().requires_grad_(True)
    om = tg.omega.clone().requires_grad_(True)
    w, k = tg.w[:3], tg.k[:3]
    out = BodyOut(r6[k], tr[k], torch.zeros(3, 2), torch.zeros(3, 22), w, k, tg._replace(u=u, omega=om))
    ru, rw = L.sup_root_rate_parts(out, truth, lengths, has, 100.0)
    assert float((ru + rw).detach().abs().max()) == 0.0
    gu, gw = torch.autograd.grad((ru + rw).sum(), [u, om])
    assert float(gu.abs().max()) == 0.0 and float(gw.abs().max()) == 0.0
    su, sw = L.SIGMAS["root_vel"], L.SIGMAS["root_yaw_rate"]
    off = out._replace(root=tg._replace(u=tg.u + torch.tensor([su, 0.0], dtype=F64)))
    ru, rw = L.sup_root_rate_parts(off, truth, lengths, has, 100.0)
    assert float(ru[0]) == pytest.approx(0.5 * 1001, rel=1e-9) and float(rw.abs().max()) == 0.0 and float(ru[1]) == 0.0
    off = out._replace(root=tg._replace(omega=tg.omega - sw))
    ru, rw = L.sup_root_rate_parts(off, truth, lengths, has, 100.0)
    assert float(rw[0]) == pytest.approx(0.5 * 1001, rel=1e-9) and float(ru.abs().max()) == 0.0
    # the rates of the circle: 0.4 rad/s and 1.2 m/s forward (interval means of a chord: within 1e-4)
    inside = tg.valid & (tg.w == 0) & (tg.k >= 5) & (tg.k <= 995)            # full intervals
    assert float((tg.omega[inside] - 0.4).abs().max()) < 1e-9
    assert float((tg.u[inside] - torch.tensor([0.0, 1.2], dtype=F64)).abs().max()) < 2e-4
    with pytest.raises(ValueError, match="root grid"):
        L.sup_root_rate_parts(out._replace(root=None), truth, lengths, has, 100.0)
    assert L.SIGMAS["root_vel"] == 0.1 and L.SIGMAS["root_yaw_rate"] == 0.05 and L.WEIGHTS["sup_root_rate"] == (1.0, 1.0)


def _window_frame_deriv(kin, kin_t, w, weight, lengths, has, sigma_v=L.SIGMA_VEL, sigma_a=L.SIGMA_ACC):
    """sup_deriv before reports/bench/root_motion_diagnosis.md: the window-frame joint velocities and
    accelerations."""
    keep = has[w]
    ev = ((kin.v[keep] - kin_t.v[keep]) ** 2).sum((-1, -2)) / (2 * sigma_v ** 2)
    ea = ((kin.a[keep] - kin_t.a[keep]) ** 2).sum((-1, -2)) / (2 * sigma_a ** 2)
    return L.point_estimate(ev + ea, w[keep], weight[keep], lengths, len(lengths))


def _kin(r6, tr, rest, t):
    from mdfm.body.kinematics import KnotBatch
    kb = KnotBatch(r6[None], tr[None], rest[None], 100.0, torch.tensor([r6.shape[0]]))
    return kb.kinematics(torch.zeros(len(t), dtype=torch.int64), t)


def test_sup_deriv_is_heading_invariant():
    """Rotating a decoded body by a constant yaw about +y (every root rotation and the translation, about the
    world origin) leaves sup_deriv unchanged to 1e-6 (float64), while a change of the relative joint motion changes
    it; the window-frame form of the phase A runs changes under the same rotation (root-motion diagnosis C2)."""
    r6, tr, rest = _walk_truth(T=601)
    t = torch.arange(20, 580, 7, dtype=F64) / 100.0
    kin_t = _kin(r6, tr, rest, t)
    # a decoded body that differs from the truth: a slower walk and a bent knee
    R = rot6d_to_matrix(r6).clone()
    R[:, 4] = R[:, 4] @ axis_angle_to_matrix(torch.tensor([0.3, 0.0, 0.0], dtype=F64))
    tr_d = tr * 0.9
    lengths, has = torch.tensor([601]), torch.tensor([True])
    w, wt = torch.zeros(len(t), dtype=torch.int64), torch.ones(len(t), dtype=F64)
    base = L.sup_deriv(_kin(matrix_to_rot6d(R), tr_d, rest, t), kin_t, w, wt, lengths, has)
    old = _window_frame_deriv(_kin(matrix_to_rot6d(R), tr_d, rest, t), kin_t, w, wt, lengths, has)
    assert float(base) > 1.0
    for theta in (0.7, -2.5, 3.0):
        Q = rot_y(torch.tensor(theta, dtype=F64))
        Rr = R.clone()
        Rr[:, 0] = Q @ R[:, 0]
        tr_r = ((tr_d + rest[0]) @ Q.T) - rest[0]                  # the pelvis turned about the world origin
        kin = _kin(matrix_to_rot6d(Rr), tr_r, rest, t)
        assert float((kin.p[:, 0] - (_kin(matrix_to_rot6d(R), tr_d, rest, t).p[:, 0] @ Q.T)).abs().max()) < 1e-9
        turned = L.sup_deriv(kin, kin_t, w, wt, lengths, has)
        assert float(turned) == pytest.approx(float(base), rel=1e-6, abs=1e-6), theta
        assert abs(float(_window_frame_deriv(kin, kin_t, w, wt, lengths, has)) - float(old)) > 0.1 * float(old)
    R2 = R.clone()
    R2[:, 5] = R2[:, 5] @ axis_angle_to_matrix(torch.tensor([0.4, 0.0, 0.0], dtype=F64))
    other = L.sup_deriv(_kin(matrix_to_rot6d(R2), tr_d, rest, t), kin_t, w, wt, lengths, has)
    assert abs(float(other) - float(base)) > 1e-3 * float(base)


def _decoded_walk(truth_r6, truth_tr, rest, s, omega_err, rate=100.0):
    """A decoded body equal to the truth but for a heading error that ramps from 0 at the centre (yaw rate error
    omega_err) and a displacement (and grid velocities) scaled by s, built through the decoder's own operator:
    (BodyOut at every knot with its RootGrid, dense rot6d, dense transl)."""
    from mdfm.model.decoder import GRID_KNOTS, RootGrid, all_knots, grid_of, hermite_on_grid, integrate_root
    T = truth_r6.shape[0]
    lengths = torch.tensor([T])
    truth = _truth_body(truth_r6, truth_tr, lengths)
    tg = L.truth_root_grid(truth, rate, lengths)
    g = grid_of(lengths)
    h = GRID_KNOTS / rate
    omega = tg.omega + omega_err
    u = s * tg.u
    psi_g, X_g, V = integrate_root(omega, u, g, h)
    w, k = all_knots(lengths)
    psi = hermite_on_grid(psi_g, omega, g, w, k, h)
    X = hermite_on_grid(X_g, V, g, w, k, h)
    R = rot6d_to_matrix(truth_r6).clone()
    R[:, 0] = rot_y(psi) @ rot_y(-yaw_angle(R[:, 0])) @ R[:, 0]
    c = T // 2
    tr = truth_tr.clone()
    tr[:, 0] = truth_tr[c, 0] + X[:, 0]
    tr[:, 2] = truth_tr[c, 2] + X[:, 1]
    r6 = matrix_to_rot6d(R)
    root = RootGrid(g.w, g.k, psi_g, omega, u, X_g, tg.valid)
    return BodyOut(r6, tr, torch.zeros(T, 2, dtype=F64), torch.zeros(T, 22, dtype=F64), w, k, root), truth


def _shrink_optima(omega, err, scales):
    """{new, old, deriv_window}: the displacement scale s (of scales) minimizing, over decoded bodies equal to a
    walk of yaw rate omega but for a heading error growing at err rad/s from the centre and their displacement and
    grid velocities scaled by s (_decoded_walk), the root terms at the phase A weights: sup_root (sigma_xz 0.50) +
    sup_root_rate + 0.1 sup_deriv (heading frame); the terms of the phase A runs, sup_root at sigma_xz 0.20 + 0.1
    window-frame sup_deriv; and the window-frame sup_deriv alone."""
    r6, tr, rest = _walk_truth(T=1001, omega=omega)
    t = torch.arange(10, 991, 3, dtype=F64) / 100.0
    kin_t = _kin(r6, tr, rest, t)
    lengths, has = torch.tensor([1001]), torch.tensor([True])
    w_pt, wt = torch.zeros(len(t), dtype=torch.int64), torch.ones(len(t), dtype=F64)
    new, old, dw = [], [], []
    for s in scales:
        out, truth = _decoded_walk(r6, tr, rest, s, err)
        kin = _kin(out.rot6d, out.transl, rest, t)
        xz, y = L.sup_root_parts(out, truth, lengths, has, sigma_xz=L.SIGMAS["root_xz"])
        ru, rw = L.sup_root_rate_parts(out, truth, lengths, has, 100.0)
        new.append(float(xz + y + ru + rw + 0.1 * L.sup_deriv(kin, kin_t, w_pt, wt, lengths, has)))
        xz0, y0 = L.sup_root_parts(out, truth, lengths, has, sigma_xz=0.20)
        dw.append(float(_window_frame_deriv(kin, kin_t, w_pt, wt, lengths, has)))
        old.append(float(xz0 + y0) + 0.1 * dw[-1])
    pick = lambda x: scales[int(np.argmin(x))]                     # noqa: E731
    return {"new": pick(new), "old": pick(old), "deriv_window": pick(dw)}


def test_root_terms_do_not_reward_shrinkage_under_heading_error():
    """A circle walk of 1.2 m/s at 0.4 rad/s and a straight walk as truths; decoded bodies equal to them but for a
    heading error that ramps from 0 at the centre to 45 deg at 5 s (their path following their own heading, as the
    integrated root decodes it), their displacement and grid velocities scaled by s in 0.50 to 1.50 by 0.02: the s
    that minimizes sup_root + sup_root_rate + 0.1 sup_deriv (the phase A weights) lies within 0.95 to 1.05. Under the
    terms of the phase A runs (window-frame sup_deriv, sigma_xz 0.20, no rate term) the window-frame sup_deriv, the
    term that carried most of the truth's reward (root-motion diagnosis C2), has its optimum below 0.95 on both
    walks, and the total below 0.95 on the straight walk; on the circle an over-turning body walks a shorter chord,
    which moves the window-frame position optimum above 1, and the old total stays below the new one."""
    scales = [round(0.5 + 0.02 * i, 2) for i in range(51)]
    err = (math.pi / 4) / 5.0                                     # 45 deg at 5 s from the centre
    circle, straight = _shrink_optima(0.4, err, scales), _shrink_optima(0.0, err, scales)
    for name, o in (("circle", circle), ("straight", straight)):
        assert 0.95 <= o["new"] <= 1.05, (name, o)
        assert o["deriv_window"] < 0.95, (name, o)
    assert straight["old"] < 0.95, straight
    assert circle["old"] < circle["new"], circle
    # the heading error is real in the decoded bodies: 45 deg at 5 s
    r6, tr, rest = _walk_truth(T=1001, omega=0.4)
    out, _ = _decoded_walk(r6, tr, rest, 1.0, err)
    assert float(out.root.psi[-1] - 0.4 * 5.0) == pytest.approx(math.pi / 4, abs=1e-6)


def test_heading_observable_reads_the_input_channel_masks():
    """A window with a valid gyroscope channel or a pose point observes its heading; an accelerometer-only window
    does not; no label is read."""
    wins = _toy_windows()
    acc_only = FX.make_window(1.0, seed=9, person="p3")
    imu = next(s for s in acc_only.streams if s.modality == "imu")
    cm = imu.chan_mask.copy()
    cm[:, 3:6] = False
    values = imu.values.copy()
    values[..., 3:6] = 0.0
    acc_only = replace(acc_only, streams=[replace(imu, chan_mask=cm, values=values)])
    batch = M.batch_from_windows(wins + [acc_only], points_per_stream=32, gen=0)
    assert L.heading_observable(batch).tolist() == [True, True, True, False]
    L.assert_label_free(batch)


def test_the_heading_terms_skip_an_undefined_heading():
    """Where the truth's root lies near the vertical (its forward axis within 26 deg of it: lying, crawling,
    swimming) the heading is undefined: truth_root_grid marks those grid points invalid, sup_root_rate ignores any
    decoded rate there, and sup_deriv compares both bodies in the window frame there (the truth's yaw for both)."""
    r6, tr, rest = _walk_truth(T=601)
    R = rot6d_to_matrix(r6).clone()
    pitch = axis_angle_to_matrix(torch.tensor([1.4, 0.0, 0.0], dtype=F64))          # 80 deg: +z near vertical
    R[200:400, 0] = R[200:400, 0] @ pitch
    r6s = matrix_to_rot6d(R)
    truth = _truth_body(r6s, tr)
    lengths, has = truth.lengths, torch.tensor([True])
    assert not bool(L.heading_defined(R[300, 0])) and bool(L.heading_defined(R[100, 0]))
    tg = L.truth_root_grid(truth, 100.0, lengths)
    steep = (tg.k >= 195) & (tg.k <= 405)
    assert not bool(tg.valid[steep].any()) and bool(tg.valid[(tg.k < 190) & (tg.k >= 0)].all())
    w, k = tg.w[:2], tg.k[:2]
    inside = (tg.k >= 0) & (tg.k < 601)                                           # a decoded grid's own mask
    out = BodyOut(r6s[k], tr[k], torch.zeros(2, 2), torch.zeros(2, 22), w, k, tg._replace(valid=inside))
    garbage = tg._replace(u=torch.where(steep[:, None], tg.u + 5.0, tg.u), omega=torch.where(steep, tg.omega + 30.0,
                                                                                             tg.omega), valid=inside)
    assert float(sum(L.sup_root_rate_parts(out._replace(root=garbage), truth, lengths, has, 100.0)).abs().max()) == 0.0
    t = torch.tensor([3.0, 1.0], dtype=F64)                                        # a steep knot and an upright one
    kin_t = _kin(r6s, tr, rest, t)
    Rd = R.clone()
    Rd[:, 0] = rot_y(torch.tensor(0.3, dtype=F64)) @ R[:, 0]                         # a decoded body turned by 0.3 rad
    kin = _kin(matrix_to_rot6d(Rd), tr * 0.9, rest, t)
    wt = torch.ones(1, dtype=F64)
    for i in range(2):
        sel = lambda x: type(x)(*(y[i:i + 1] for y in x))                           # noqa: E731
        got = L.sup_deriv(sel(kin), sel(kin_t), torch.zeros(1, dtype=torch.int64), wt, lengths, has)
        window = _window_frame_deriv(sel(kin), sel(kin_t), torch.zeros(1, dtype=torch.int64), wt, lengths, has)
        if i == 0:
            assert float(got) == pytest.approx(float(window), rel=1e-9)             # the window frame where steep
        else:
            assert abs(float(got) - float(window)) > 1e-3 * float(window)          # each body's heading elsewhere
