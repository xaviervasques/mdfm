"""Tests of the checkpoint migration, mdfm/train/migrate.py (spec docs/MFM_STAGE2_CODE_SPEC_DRAFT.md, section 4.2
and the S2.2 row of section 7).

- A tiny model (every weight perturbed, so that every row matters) migrated to two more sources, two more tasks
  (vocabulary version 2), one more layout, device, camera and mount family computes on windows of old names what
  the old model computes (posterior, nuisance, prior log-density, nuisance log-prior, decoded knots, source
  offsets; float32 CPU, 1e-6), the model and its EMA alike; old codes include 'none' and 'unknown' rows.
- A window with a new task gets the prior condition that the old model gives task 'unknown' (copy_unknown); with
  task_rows zero the new rows are zero.
- The codes of the stage-1 windows decide the nuisance rows: an 'unknown' code among them copies the table's
  'unknown' row into the new rows, none leaves them at zero; the descriptor coding equals
  tokenize.nuisance_descriptors under STAGE1 (fixture streams and every store and layout stream), and recode maps
  the old codes onto the new ones. The layout pseudo-streams hold the codes of synthetic windows that the factory
  renders for every layout family (two draws each, CPU). A coding of batch_from_windows that differs from the one
  mfm.batch_from_windows gives under the same names is refused.
- plan_for: a source of other names than STAGE1 needs the codes of its windows, counted under its names.
- Refusals: a dropped, renamed or moved old name (sources, layouts, tasks); an old mount buffer that differs from
  the new priors; new mount families without priors; a state that does not fit the old names; an output in the
  source's run folder or in a non-empty folder; a drift_log in the source's run folder.
- The EMA, the JEPA target (jepa.target_tokenizer) and the AdamW moments migrate with the model: old rows kept, new
  rows zero, step kept, and the migrated state loads into an optimizer over loop.param_groups of the new model,
  whose order param_names reproduces.
- migrate() on a tiny checkpoint writes a new run folder with the lineage, leaves the source folder unchanged, and
  model_of loads the result strictly; the code counts and task_rows it is given reach its plan and the stored
  tables (an 'unknown' layout and device among the codes, task rows zero); the migrated checkpoint, of other names
  than STAGE1, migrates again only with codes counted under its names.
- cache/runs/phaseA_root_full/ckpt_A_010681.pt (skipped when absent): loaded with checkpoint.load (drift_log in
  the test's folder), migrated to S2B_DRAFT, loaded strictly into the S2B model; equal prior log-densities and
  decoded knots on fixed stage-1 training windows of the window store, the audit passing, the lineage plan that of
  the codes counted on the store and the layouts, nothing written beside the checkpoint.
"""
import inspect
import math
import os
from dataclasses import replace

import numpy as np
import pytest
import torch

from mdfm import paths
from mdfm.data import canonical as C
from mdfm.data import windows as W
from mdfm.factory import layouts as LY
from mdfm.kernels.nuisance import MountPriors
from mdfm.model import mfm as M
from mdfm.model import tokenize as TK
from mdfm.model.prior import Cond
from mdfm.train import checkpoint as CK
from mdfm.train import config as CF
from mdfm.train import loop as LP
from mdfm.train import migrate as MG
from mdfm.train import names as NM
from mdfm.train import provenance as PV

import model_fixture as MF

TINY = replace(M.TINY, skeleton=False)
S1 = NM.STAGE1
NEW = NM.NameTables(sources=S1.sources + ("pads", "gaitndd"), vocabulary_path=NM.VOCABULARY_V2,
                    layouts=S1.layouts + ("pads_wrists",), devices=S1.devices + ("pads_device",),
                    cameras=S1.cameras + ("modys",), mount_families=S1.mount_families + ("pads_mount",))
NEWER = replace(NEW, sources=NEW.sources + ("gaitrec",), layouts=NEW.layouts + ("insole8",))
JEPA = {"width": 32, "depth": 1, "heads": 4}
TOL = 1e-6
HAVE_MANIFEST = os.path.exists(C.manifest_path())
HAVE_MA = os.path.exists(paths.smplh_model("neutral"))
needs_all = pytest.mark.skipif(not (HAVE_MANIFEST and HAVE_MA), reason="tiers records file or SMPL+H model not found")
REAL = os.path.join(paths.CACHE, "runs", "phaseA_root_full", "ckpt_A_010681.pt")
# the archived records state REAL was born on (scripts/archive_records_state.py)
STATE_CA5A = "ca5a2d9eaada02b70f834d20ebc77f19de7b89719a0aed32e352faed0abdb8d8"
STORE = os.path.join(paths.CACHE, "window_store", "store.json")


def _perturb(model, seed, scale=0.05):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(scale * torch.randn(p.shape, generator=g, dtype=p.dtype))
    return model


def _old_model(seed=0, jepa=None):
    torch.manual_seed(seed)
    return _perturb(MG.build_model(TINY, S1, jepa=jepa), seed + 100).eval()


def _new_priors(names=NEW):
    return MG.extend_priors(MountPriors.load(), names.mount_families)[0]


def _new_model(state, names=NEW, jepa=None):
    m = MG.build_model(TINY, names, _new_priors(names), jepa)
    m.load_state_dict(state, strict=True)
    return m.eval()


def _windows():
    d = 2.0
    odd_imu = MF.imu_stream(d, seed=7, layout="not_a_layout")                    # layout 'unknown', device 'unknown'
    odd_imu.meta.pop("device")
    p2 = MF.pose2d_stream(d, seed=8)
    p2.meta["camera"] = "handheld"                                               # camera 'unknown'
    nonan = MF.imu_stream(d, seed=9, sites=(0, 1, 2), layout="nonan3")           # device from the family rule
    nonan.meta.pop("device")
    cov = {"sex": 2, "age_band": W.AGE_UNKNOWN, "task": 11}
    return [MF.make_window(d, seed=0), MF.make_window(d, seed=1, sex=0, task=3),
            MF.make_pose_window(d, seed=2), MF.make_pose_window(d, seed=3, source="stroke_gait_vancriekinge", task=0),
            W.Window("weargait_pd", "q1", "k1", False, 0.0, d, [odd_imu, p2], cov),
            W.Window("nonan_gaitprint", "q2", "k2", False, 0.0, d, [nonan], dict(cov, task=0))]


def _new_name_window():
    d = 2.0
    imu = MF.imu_stream(d, seed=11, sites=(3, 4), layout="pads_wrists")
    imu.meta["device"] = "pads_device"
    imu.meta["acc_content"] = "linear_no_gravity"
    p2 = MF.pose2d_stream(d, seed=12)
    p2.meta["camera"] = "modys"
    return W.Window("pads", "q3", "k3", False, 0.0, d, [imu, p2],
                    {"sex": 1, "age_band": W.AGE_UNKNOWN, "task": 12})


def _assert_equal(eq, tol=TOL):
    bad = {k: v for k, v in eq.items() if k not in ("n_windows", "n_knots") and not v <= tol}
    assert not bad, bad


# ---------------------------------------------------------------- the model of a NameTables
def test_build_model_under_stage1_is_todays_model():
    torch.manual_seed(0)
    a = M.MFM(TINY).state_dict()
    torch.manual_seed(0)
    b = MG.build_model(TINY, S1).state_dict()
    assert list(a) == list(b)
    assert all(a[k].shape == b[k].shape and torch.equal(a[k], b[k]) for k in a)
    m = MG.build_model(TINY, NEW, _new_priors(), JEPA)
    sd = m.state_dict()
    assert sd["prior.task.weight"].shape[0] == len(NEW.tasks()) == 14
    assert sd["source_offsets.offsets"].shape[0] == len(NEW.sources)
    for f, n in (("layout", len(NEW.layouts)), ("device", len(NEW.devices)), ("camera", len(NEW.cameras))):
        i = TK.FIELD_NAMES.index(f)
        assert sd[f"tokenizer.nuisance.tables.{i}.weight"].shape[0] == n + 2
        assert sd[f"jepa.target_tokenizer.nuisance.tables.{i}.weight"].shape[0] == n + 2
    assert sd["heads.mp_R0"].shape[0] == len(NEW.mount_families) and m.heads.families == NEW.mount_families
    assert list(sd) == list(_old_model(jepa=JEPA).state_dict())         # same entries in the same order


def test_param_names_follow_the_optimizer_order_of_the_loop():
    for model in (_old_model(jepa=JEPA), MG.build_model(TINY, NEW, _new_priors(), JEPA)):
        byname = dict(model.named_parameters())
        groups = LP.param_groups(model, 0.05)
        names = MG.param_names(model)
        assert [[id(byname[n]) for n in g] for g in names] == [[id(p) for p in g["params"]] for g in groups]
        assert not any(n.startswith("jepa.target_") for g in names for n in g)


# ---------------------------------------------------------------- coding of the windows
def test_descriptor_codes_equal_the_tokenizer_and_recode_maps_old_codes():
    pl = MG.plan(S1, NEW)
    f1, fn = S1.nuis_fields(), NEW.nuis_fields()
    assert f1 == TK.NUIS_FIELDS
    n = 0
    for w in _windows():
        for s in w.streams:
            if s.role != "input":
                continue
            old = TK.nuisance_descriptors(s)
            assert np.array_equal(MG.nuisance_descriptors(s, f1), old)
            assert torch.equal(MG.recode(torch.from_numpy(old), pl), torch.from_numpy(MG.nuisance_descriptors(s, fn)))
            n += 1
    assert n == 11
    # 'none' and 'unknown' move to the new len and len + 1
    li = TK.FIELD_NAMES.index("layout")
    wins = _windows()
    old = TK.nuisance_descriptors(wins[4].streams[0])
    assert set(old[:, li]) == {len(S1.layouts) + 1}
    new = MG.nuisance_descriptors(wins[4].streams[0], fn)
    assert set(new[:, li]) == {len(NEW.layouts) + 1}
    assert MG.nuisance_descriptors(wins[2].streams[0], fn)[0, li] == len(NEW.layouts)     # pose3d without layout
    nw = _new_name_window()
    d = MG.nuisance_descriptors(nw.streams[0], fn)
    assert set(d[:, li]) == {NEW.layouts.index("pads_wrists")}
    assert set(d[:, TK.FIELD_NAMES.index("device")]) == {NEW.devices.index("pads_device")}
    d2 = MG.nuisance_descriptors(nw.streams[1], fn)
    assert d2[0, TK.FIELD_NAMES.index("camera")] == NEW.cameras.index("modys")
    b = MG.batch_from_windows(wins, NEW)
    assert b.tokens.nuis_desc.shape == M.batch_from_windows(wins).tokens.nuis_desc.shape


def test_a_coding_that_differs_from_the_batch_coding_is_refused(monkeypatch):
    wins = _windows()
    assert torch.equal(MG.batch_from_windows(wins, S1).tokens.nuis_desc, M.batch_from_windows(wins).tokens.nuis_desc)
    real = MG.nuisance_descriptors
    ci = TK.FIELD_NAMES.index("camera")

    def shifted(stream, nuis_fields):
        d = real(stream, nuis_fields).copy()
        d[:, ci] = 0
        return d

    monkeypatch.setattr(MG, "nuisance_descriptors", shifted)
    with pytest.raises(ValueError, match="camera"):
        MG.batch_from_windows(wins, S1)
    if "nuis_fields" in inspect.signature(M.batch_from_windows).parameters:     # task S2.3 codes under NEW itself
        with pytest.raises(ValueError, match="camera"):
            MG.batch_from_windows(wins, NEW)
    else:                                                                       # before S2.3: the codes replaced
        assert not MG.batch_from_windows(wins, NEW).tokens.nuis_desc[:, ci].any()


# ---------------------------------------------------------------- the plan and its refusals
def test_plan_rows():
    pl = MG.plan(S1, NEW)
    assert pl.tables["sources"].rows[-2:] == (("zero", None), ("zero", None))
    u = S1.tasks().index("unknown")
    assert pl.tables["tasks"].rows[-2:] == (("copy", u), ("copy", u))
    lay = pl.tables["nuis.layout"]
    a = len(S1.layouts)
    assert (lay.old_rows, lay.new_rows) == (a + 2, a + 3)
    assert lay.rows[a:] == (("zero", None), ("old", a), ("old", a + 1))
    assert lay.code_map()[a] == a + 1 and lay.code_map()[a + 1] == a + 2
    assert pl.tables["mount_families"].rows[-1] == ("prior", "pads_mount")
    assert MG.plan(S1, NEW, task_rows="zero").tables["tasks"].rows[-1] == ("zero", None)
    assert MG.plan(S1, S1).tables["nuis.layout"].is_identity()
    assert pl.sha256() == MG.plan(S1, NEW).sha256() != MG.plan(S1, NEW, "zero").sha256()
    with pytest.raises(ValueError):
        MG.plan(S1, NEW, task_rows="mean")
    with pytest.raises(ValueError):
        MG.plan(S1, NEW, trained_unknown=("rate",))


def test_a_dropped_renamed_or_moved_old_name_is_refused(tmp_path):
    bad = {
        "dropped layout": replace(NEW, layouts=NEW.layouts[1:]),
        "renamed device": replace(NEW, devices=("generic", "xsens", "pads_device")),
        "moved source": replace(NEW, sources=("synthetic", "gait120", "weargait_pd") + NEW.sources[3:]),
        "camera inserted first": replace(NEW, cameras=("modys", "generic")),
        "mount family moved": replace(NEW, mount_families=("generic", "pads_mount", "xsens13")),
    }
    v1 = open(NM.resolve(NM.VOCABULARY_V1), encoding="utf-8").read()
    assert "tasks: [walk, turn," in v1
    swapped = tmp_path / "vocab_swapped.yaml"
    swapped.write_text(v1.replace("tasks: [walk, turn,", "tasks: [turn, walk,"), encoding="utf-8", newline="\n")
    bad["tasks moved"] = replace(NEW, vocabulary_path=str(swapped))
    for what, names in bad.items():
        with pytest.raises(ValueError):
            MG.plan(S1, names)
        assert what


def test_an_old_mount_buffer_that_differs_or_missing_priors_are_refused():
    model = _old_model()
    pl = MG.plan(S1, NEW)
    pr = _new_priors()
    R0 = pr.R0.clone()
    R0[1, 0, 0, 0] += 1e-3                                       # xsens13, an old family
    with pytest.raises(ValueError, match="xsens13"):
        MG.apply(model.state_dict(), pl, replace(pr, R0=R0))
    with pytest.raises(ValueError):
        MG.apply(model.state_dict(), pl, None)
    with pytest.raises(ValueError):
        MG.apply(model.state_dict(), pl, MG.extend_priors(MountPriors.load(), ("generic", "xsens13", "other"))[0])
    st = MG.apply(model.state_dict(), pl, pr)
    gen = MountPriors.load().families.index("generic")
    for f in ("R0", "sigma_tilt", "bias_acc_sd", "fitted"):
        x = st["heads.mp_" + f]
        assert torch.equal(x[:2], model.state_dict()["heads.mp_" + f])
        want = getattr(MountPriors.load(), f)[gen]
        assert torch.equal(x[2], want if want.dtype == torch.bool else want.float())
    with pytest.raises(ValueError):                              # a state that does not fit the old names
        MG.apply(st, pl, pr)


# ---------------------------------------------------------------- equal outputs on old names
def test_the_migrated_tiny_model_computes_what_the_old_one_computes():
    model = _old_model(seed=1)
    ema = _perturb(LP.make_ema(model), 7).eval()
    pr = _new_priors()
    pl = MG.plan(S1, NEW)
    wins = _windows()
    for old in (model, ema):
        new = _new_model(MG.apply(old.state_dict(), pl, pr))
        eq = MG.equivalence(old, new, wins, S1, NEW)
        _assert_equal(eq)
        assert eq["n_windows"] == 6 and eq["n_knots"] > 0
        assert eq["prior_log_prob"] == 0.0 and eq["knots.rot6d"] == 0.0
    # the nuisance outputs depend on the windows (the heads were perturbed), so equality is not trivial
    with torch.no_grad():
        post = model.encode(M.batch_from_windows(wins))
    assert float(post.nuis.imu.mount_res6d.std()) > 0 and float(post.nuis.pose3d_log_sigma.std()) > 0
    # a window of new names runs through the new model
    new = _new_model(MG.apply(model.state_dict(), pl, pr))
    out = new.encode(MG.batch_from_windows(wins + [_new_name_window()], NEW))
    assert torch.isfinite(out.mu_z).all() and torch.isfinite(out.nuis.imu.mount_res6d).all()


def test_a_new_task_gets_the_condition_of_unknown():
    model = _old_model(seed=2)
    pr = _new_priors()
    beta = torch.randn(4, 16, generator=torch.Generator().manual_seed(3))
    sex = torch.tensor([0, 1, 2, 1])
    u = S1.tasks().index("unknown")
    new_tasks = torch.tensor([NEW.tasks().index("rest_lying"), NEW.tasks().index("rest_sitting")] * 2)
    assert new_tasks.tolist() == [12, 13, 12, 13]
    want = model.prior.condition(Cond(sex, torch.full((4,), u), beta), dropout=False)
    new = _new_model(MG.apply(model.state_dict(), MG.plan(S1, NEW), pr))
    got = new.prior.condition(Cond(sex, new_tasks, beta), dropout=False)
    assert torch.equal(got, want)
    zero = _new_model(MG.apply(model.state_dict(), MG.plan(S1, NEW, task_rows="zero"), pr))
    assert torch.equal(zero.prior.task.weight[12:], torch.zeros(2, zero.prior.d))
    got0 = zero.prior.condition(Cond(sex, new_tasks, beta), dropout=False)
    assert torch.allclose(got0, want - model.prior.task.weight[u], atol=1e-6)
    # every old task keeps its condition
    old_tasks = torch.arange(len(S1.tasks()))
    s = torch.zeros(len(old_tasks), dtype=torch.int64)
    b = torch.zeros(len(old_tasks), 16)
    assert torch.equal(new.prior.condition(Cond(s, old_tasks, b), dropout=False),
                       model.prior.condition(Cond(s, old_tasks, b), dropout=False))


# ---------------------------------------------------------------- the codes of the stage-1 windows decide
def _pseudo(modality, layout=None, fps=100.0, **meta):
    from types import SimpleNamespace
    m = dict(meta)
    if layout is not None:
        m["layout"] = layout
    return SimpleNamespace(modality=modality, fps=fps, meta=m, source="x")


def test_the_stage1_codes_decide_the_new_nuisance_rows():
    store = [_pseudo("imu", "xsens13", site_ids=[0, 1]), _pseudo("pose3d", "mocap3d")]
    lays = [_pseudo("pose2d", "camera1", fps=30.0, camera="generic")]
    clean = MG.stage1_codes(streams=(store, lays))
    assert clean["store"]["layout"] == {"xsens13": 2, "mocap3d": 1} and clean["layouts"]["camera"] == {"generic": 1}
    assert MG.trained_unknown(clean) == ()
    odd = MG.stage1_codes(streams=(store + [_pseudo("imu", "nowhere", site_ids=[3])], lays))
    assert odd["store"]["layout"]["unknown"] == 1 and odd["store"]["device"]["unknown"] == 1
    assert MG.trained_unknown(odd) == ("layout", "device")
    model = _old_model(seed=4)
    pr = _new_priors()
    a = len(S1.layouts)
    li, di, ci = (TK.FIELD_NAMES.index(f) for f in ("layout", "device", "camera"))
    for codes, copied in ((clean, ()), (odd, ("layout", "device"))):
        pl = MG.plan(S1, NEW, trained_unknown=MG.trained_unknown(codes))
        st = MG.apply(model.state_dict(), pl, pr)
        for f, i, n in (("layout", li, a), ("device", di, len(S1.devices)), ("camera", ci, len(S1.cameras))):
            old = model.state_dict()[f"tokenizer.nuisance.tables.{i}.weight"]
            new = st[f"tokenizer.nuisance.tables.{i}.weight"]
            assert torch.equal(new[:n], old[:n]) and torch.equal(new[-2:], old[-2:])
            want = old[n + 1] if f in copied else torch.zeros_like(old[0])
            assert all(torch.equal(r, want) for r in new[n:-2]), f


@pytest.mark.skipif(not os.path.exists(STORE), reason="stage-1 window store not built")
def test_the_store_and_layout_codes_are_the_tokenizer_codes():
    ss, ls = MG.store_streams(), MG.layout_streams()
    assert len(ss) > 0 and len(ls) > 0
    f1 = S1.nuis_fields()
    for s in ss + ls:
        assert np.array_equal(MG.nuisance_descriptors(s, f1), TK.nuisance_descriptors(s))
    codes = MG.stage1_codes(streams=(ss, ls))
    tu = MG.trained_unknown(codes)
    assert all((codes["store"][f].get("unknown", 0) + codes["layouts"][f].get("unknown", 0) > 0) == (f in tu)
               for f in NM.NUIS_TABLES)
    assert sum(codes["store"]["kind"].values()) == sum(len(MG.nuisance_descriptors(s, f1)) for s in ss)


def _walk(K=400):
    """BodyWindows of one procedural walk of K knots at 100 Hz in the window frame (1 m/s along +z), the walk of
    tests/test_tokenize.py."""
    from mdfm.body.rotations import axis_angle_to_matrix, matrix_to_rot6d
    from mdfm.data.body_store import BodyWindows
    t = torch.arange(K, dtype=torch.float64) / 100.0
    ph = 2 * math.pi * 0.9 * t
    aa = torch.zeros(K, 22, 3, dtype=torch.float64)
    aa[:, 1, 0], aa[:, 2, 0] = 0.4 * torch.sin(ph), -0.4 * torch.sin(ph)
    aa[:, 4, 0], aa[:, 5, 0] = 0.6 * (1 - torch.cos(ph)) / 2, 0.6 * (1 + torch.cos(ph)) / 2
    aa[:, 16, 2], aa[:, 17, 2] = -1.2, 1.2
    aa[:, 18, 0], aa[:, 19, 0] = -0.3 * torch.sin(ph), 0.3 * torch.sin(ph)
    rot6d = matrix_to_rot6d(axis_angle_to_matrix(aa)).float()
    transl = torch.stack([torch.zeros_like(t), 0.92 + 0.02 * torch.sin(2 * ph), t - t[K // 2]], -1).float()
    zero = torch.zeros(1, dtype=torch.int64)
    return BodyWindows(rot6d[None], transl[None], torch.zeros(1, K, 2, dtype=torch.bool),
                       torch.ones(1, K, dtype=torch.bool), torch.tensor([K]), torch.zeros(1, 16), torch.tensor([2]),
                       np.array(["test:walk"]), zero, zero.clone(), zero.clone())


@pytest.mark.skipif(not (HAVE_MA and os.path.exists(LY.LAYOUTS_PATH)),
                    reason="SMPL+H model or reports/layouts/layouts.json not found")
def test_the_layout_streams_hold_the_codes_of_rendered_synthetic_windows():
    from mdfm.body.smplh import SkeletonBank
    from mdfm.factory.synth import SynthSampler
    f1 = S1.nuis_fields()
    pseudo = {}
    for s in MG.layout_streams():
        pseudo.setdefault((s.meta["layout"], s.modality), set()).update(
            map(tuple, MG.nuisance_descriptors(s, f1).tolist()))
    layouts = LY.load_layouts()
    sampler = SynthSampler(layouts=layouts, skel=SkeletonBank(torch.float32), dtype=torch.float32, device="cpu")
    body = _walk()
    seen = set()
    for k, (name, lay) in enumerate(layouts.items()):
        for j in range(2):
            w = sampler.render(body, [LY.concrete(lay, 11 + j)], 100 + 2 * k + j)[0]
            for s in w.streams:
                if s.role != "input":
                    continue
                key = (name, s.modality)
                assert key in pseudo, key
                got = set(map(tuple, TK.nuisance_descriptors(s).tolist()))
                assert got <= pseudo[key], (key, sorted(got - pseudo[key]))
                seen.add(key)
    assert seen == set(pseudo)


def test_plan_for_needs_the_codes_of_a_source_of_other_names():
    store = [_pseudo("imu", "pads_wrists", site_ids=[3]), _pseudo("pose2d", "camera1", fps=30.0, camera="modys")]
    with pytest.raises(ValueError, match="not STAGE1"):
        MG.plan_for(NEW, NEWER)
    with pytest.raises(ValueError, match="codes counted under"):
        MG.plan_for(NEW, NEWER, codes=MG.stage1_codes(S1, streams=(store, [])))
    with pytest.raises(ValueError, match="codes counted under"):
        MG.plan_for(S1, NEW, codes={k: v for k, v in MG.stage1_codes(streams=(store, [])).items()
                                    if k != "names_sha256"})
    codes = MG.stage1_codes(NEW, streams=(store, []))
    assert codes["names_sha256"] == NEW.sha256()
    assert codes["store"]["layout"] == {"pads_wrists": 1, "camera1": 1} and codes["store"]["camera"]["modys"] == 1
    assert codes["store"]["device"].get("unknown") == 1                  # no device and no family rule for pads_wrists
    pl, got = MG.plan_for(NEW, NEWER, codes=codes)
    assert got is codes and pl.trained_unknown == ("device",)
    assert pl.sha256() == MG.plan(NEW, NEWER, "copy_unknown", ("device",)).sha256()
    assert pl.tables["nuis.layout"].rows[len(NEW.layouts)] == ("zero", None)
    assert MG.plan_for(S1, NEW, "zero", MG.stage1_codes(streams=(store, [])))[0].sha256() == \
        MG.plan(S1, NEW, "zero", ("layout", "device", "camera")).sha256()


# ---------------------------------------------------------------- EMA, JEPA target and optimizer moments
def test_ema_jepa_target_and_optimizer_moments_migrate_with_the_model():
    model = _old_model(seed=5, jepa=JEPA)
    model.jepa.sync(model.tokenizer, model.encoder)
    _perturb(model.jepa.target_tokenizer, 9)
    opt = torch.optim.AdamW(LP.param_groups(model, 0.05), lr=1e-3, foreach=True)
    g = torch.Generator().manual_seed(6)
    for _ in range(2):
        for p in model.parameters():
            if p.requires_grad:
                p.grad = torch.randn(p.shape, generator=g)
        opt.step()
    pr = _new_priors()
    pl = MG.plan(S1, NEW)
    new = _new_model(MG.apply(model.state_dict(), pl, pr), jepa=JEPA)
    # the JEPA target's tables follow the plan of the tokenizer's
    for f in NM.NUIS_TABLES:
        i = TK.FIELD_NAMES.index(f)
        old_t = model.jepa.target_tokenizer.nuisance.tables[i].weight
        new_t = new.jepa.target_tokenizer.nuisance.tables[i].weight
        a = old_t.shape[0] - 2
        assert torch.equal(new_t[:a], old_t[:a]) and torch.equal(new_t[-2:], old_t[-2:])
        assert not new_t[a:-2].any()
        assert not torch.equal(new_t, new.tokenizer.nuisance.tables[i].weight)    # the target's own values
    # optimizer moments: old rows kept, new rows zero (the copied task rows too), step kept
    sd = MG.apply_optimizer(opt.state_dict(), model, new, pl)
    names = [n for gr in MG.param_names(new) for n in gr]
    old_sd = opt.state_dict()
    old_names = [n for gr in MG.param_names(model) for n in gr]
    assert names == old_names and len(sd["state"]) == len(old_sd["state"]) == len(names)
    for i, n in enumerate(names):
        a, b = old_sd["state"][i], sd["state"][i]
        assert torch.equal(a["step"], b["step"])
        t = MG.table_of(n)
        for k in ("exp_avg", "exp_avg_sq"):
            if t is None:
                assert torch.equal(a[k], b[k]), n
            else:
                tp = pl.tables[t]
                old_rows = [j for j, (kind, _) in enumerate(tp.rows) if kind == "old"]
                assert torch.equal(b[k][old_rows], a[k][[tp.rows[j][1] for j in old_rows]]), n
                other = [j for j in range(tp.new_rows) if j not in old_rows]
                assert other and not b[k][other].any(), n
    assert {MG.table_of(n) for n in names} - {None} == {"sources", "tasks", "nuis.layout", "nuis.device",
                                                         "nuis.camera"}
    opt2 = torch.optim.AdamW(LP.param_groups(new, 0.05), lr=1e-3, foreach=True)
    opt2.load_state_dict(sd)
    opt3 = torch.optim.AdamW(LP.param_groups(new, 0.05), lr=1e-3, foreach=True)
    LP.load_optimizer_state(opt3, sd)
    assert len(opt3.state) == len(names)
    # the old state does not fit the new model unmigrated
    with pytest.raises(ValueError):
        LP.load_optimizer_state(torch.optim.AdamW(LP.param_groups(new, 0.05), lr=1e-3, foreach=True), old_sd)
    # EMA: the copy without the JEPA module migrates with the same plan
    ema = _perturb(LP.make_ema(model), 8).eval()
    st = MG.apply(ema.state_dict(), pl, pr)
    assert not any(k.startswith("jepa.") for k in st)
    _assert_equal(MG.equivalence(ema, _new_model(st), _windows(), S1, NEW))


# ---------------------------------------------------------------- migrate() on a tiny checkpoint
def _provenance():
    persons = {"amass": ["fake:amass:P01"], "gait120": ["fake:gait120:P01", "fake:gait120:P02"]}
    held = {"gait120": ["fake:gait120:P09"]}
    return {"format": PV.FORMAT, "hashes": PV.hashes({"name": "t"}), "persons": persons,
            "persons_sha256": PV.list_hashes(persons), "held_out": held, "held_out_sha256": PV.list_hashes(held),
            "report_mismatches": [], "extra": {"phase": "B"}}


def _listing(folder):
    return sorted((n, os.stat(os.path.join(folder, n)).st_size, os.stat(os.path.join(folder, n)).st_mtime_ns)
                  for n in os.listdir(folder))


@needs_all
def test_migrate_writes_a_new_run_folder_with_its_lineage(tmp_path):
    model = _old_model(seed=6, jepa=JEPA)
    model.jepa.sync(model.tokenizer, model.encoder)
    ema = _perturb(LP.make_ema(model), 10).eval()
    opt = torch.optim.AdamW(LP.param_groups(model, 0.05), lr=1e-3, foreach=True)
    for p in model.parameters():
        if p.requires_grad:
            p.grad = torch.ones_like(p)
    opt.step()
    cfg = CF.RunConfig(model=CF.ModelSection("tiny"), jepa=CF.JepaSection(enabled=True, pred_width=32, pred_depth=1,
                                                                          pred_heads=4))
    src = tmp_path / "src" / "ckpt_B_000004.pt"
    CK.save(src, model, ema, opt, None, None, CK.rng_state(), cfg, _provenance(), step=4)
    before = _listing(tmp_path / "src")
    codes = MG.stage1_codes(streams=([], []))
    out = tmp_path / "new" / "ckpt_B_000004_s2.pt"
    with pytest.raises(ValueError):
        MG.migrate(src, tmp_path / "src" / "m.pt", drift_log=tmp_path / "d.jsonl", new=NEW, codes=codes)
    with pytest.raises(ValueError):
        MG.migrate(src, out, drift_log=tmp_path / "src" / "d.jsonl", new=NEW, codes=codes)
    (tmp_path / "busy").mkdir()
    (tmp_path / "busy" / "x.txt").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError):
        MG.migrate(src, tmp_path / "busy" / "m.pt", drift_log=tmp_path / "d.jsonl", new=NEW, codes=codes)
    res = MG.migrate(src, out, drift_log=tmp_path / "d.jsonl", new=NEW, codes=codes)
    assert _listing(tmp_path / "src") == before
    for name in ("ckpt_B_000004_s2.pt", "ckpt_B_000004_s2.pt.sha256", "provenance.json", "config.json",
                 "config.sha256", "persons"):
        assert (tmp_path / "new" / name).exists(), name
    ck = CK.load(out, drift_log=tmp_path / "d2.jsonl")
    lin = MG.lineage(ck)
    assert lin["migrated_from"]["sha256"] == PV.sha256_file(src) == res["source_sha256"]
    assert lin["migrated_from"]["step"] == 4 and ck["step"] == 4
    assert lin["names_sha256"] == {"old": S1.sha256(), "new": NEW.sha256()}
    assert lin["plan_sha256"] == MG.plan(S1, NEW).sha256() == res["plan_sha256"]
    assert lin["mount_origin"]["pads_mount"] == "generic" and lin["mount_origin"]["xsens13"] == "measured"
    assert ck["provenance"]["extra"]["phase"] == "B" and ck["provenance"]["persons"] == _provenance()["persons"]
    assert MG.names_of(ck["config"]) == NEW and MG.names_of(CK.load(src, drift_log=tmp_path / "d3.jsonl")["config"]) == S1
    PV.audit(ck["model"], ck["ema"], ck["opt"])
    new_model = MG.model_of(ck, "model")
    assert new_model.jepa is not None
    _assert_equal(MG.equivalence(model.eval(), new_model, _windows(), S1, NEW))
    _assert_equal(MG.equivalence(ema, MG.model_of(ck, "ema"), _windows(), S1, NEW))
    opt2 = torch.optim.AdamW(LP.param_groups(new_model, 0.05), lr=1e-3, foreach=True)
    opt2.load_state_dict(ck["opt"])
    with pytest.raises(ValueError):                    # migrating again into the same folder is refused
        MG.migrate(src, out, drift_log=tmp_path / "d.jsonl", new=NEW, codes=codes)
    # the codes and task_rows given reach the plan and the stored tables: an 'unknown' layout and device among the
    # codes copy those 'unknown' rows into the new rows, the camera's stay zero, the new task rows are zero
    odd = MG.stage1_codes(streams=([_pseudo("imu", "nowhere", site_ids=[3])], []))
    assert MG.trained_unknown(odd) == ("layout", "device")
    out2 = tmp_path / "new_zero" / "ckpt_B_000004_s2.pt"
    res2 = MG.migrate(src, out2, drift_log=tmp_path / "d.jsonl", new=NEW, task_rows="zero", codes=odd)
    ck2 = CK.load(out2, drift_log=tmp_path / "d4.jsonl")
    want = MG.plan(S1, NEW, "zero", ("layout", "device"))
    lin2 = MG.lineage(ck2)
    assert lin2["plan"] == want.to_dict() and lin2["plan_sha256"] == want.sha256() == res2["plan_sha256"]
    assert res2["trained_unknown"] == ("layout", "device") and lin2["codes"] == odd
    nt = len(S1.tasks())
    for st, old_sd, prefixes in ((ck2["model"], model.state_dict(), ("tokenizer.", "jepa.target_tokenizer.")),
                                 (ck2["ema"], ema.state_dict(), ("tokenizer.",))):
        assert torch.equal(st["prior.task.weight"][:nt], old_sd["prior.task.weight"])
        assert st["prior.task.weight"].shape[0] == len(NEW.tasks()) and not st["prior.task.weight"][nt:].any()
        for pre in prefixes:
            for f in NM.NUIS_TABLES:
                key = f"{pre}nuisance.tables.{TK.FIELD_NAMES.index(f)}.weight"
                o, n = old_sd[key], st[key]
                a = o.shape[0] - 2
                assert n.shape[0] > o.shape[0] and torch.equal(n[:a], o[:a]) and torch.equal(n[-2:], o[-2:]), key
                row = o[a + 1] if f in ("layout", "device") else torch.zeros_like(o[0])
                assert row.any() == (f != "camera") and all(torch.equal(r, row) for r in n[a:-2]), key
    _assert_equal(MG.equivalence(model.eval(), MG.model_of(ck2, "model"), _windows(), S1, NEW))
    # the migrated checkpoint holds names other than STAGE1: it migrates again only with codes under its names
    out3 = tmp_path / "newer" / "ckpt_B_000004_s3.pt"
    with pytest.raises(ValueError, match="not STAGE1"):
        MG.migrate(out, out3, drift_log=tmp_path / "d.jsonl", new=NEWER)
    with pytest.raises(ValueError, match="codes counted under"):
        MG.migrate(out, out3, drift_log=tmp_path / "d.jsonl", new=NEWER, codes=codes)
    assert not (tmp_path / "newer").exists()
    codes_new = MG.stage1_codes(NEW, streams=([_pseudo("imu", "pads_wrists", site_ids=[3])], []))
    res3 = MG.migrate(out, out3, drift_log=tmp_path / "d.jsonl", new=NEWER, codes=codes_new)
    ck3 = CK.load(out3, drift_log=tmp_path / "d5.jsonl")
    want3 = MG.plan(NEW, NEWER, "copy_unknown", MG.trained_unknown(codes_new))
    assert res3["plan_sha256"] == want3.sha256() == MG.lineage(ck3)["plan_sha256"]
    assert MG.names_of(ck3["config"]) == NEWER
    _assert_equal(MG.equivalence(model.eval(), MG.model_of(ck3, "model"), _windows(), S1, NEWER))
    _assert_equal(MG.equivalence(ema, MG.model_of(ck3, "ema"), _windows(), S1, NEWER))


# ---------------------------------------------------------------- the phase A end of the pilot
@needs_all
@pytest.mark.skipif(not (os.path.exists(REAL) and os.path.exists(STORE)),
                    reason="cache/runs/phaseA_root_full/ckpt_A_010681.pt or the stage-1 window store absent")
def test_the_phase_a_checkpoint_migrates_to_the_s2b_names(tmp_path):
    # read under the archived records state the checkpoint was born on (plan step M7; the live records moved with
    # the tiers rerun A23); skipped when the archived file is absent from this machine
    arch = PV.records_archive()
    assert STATE_CA5A in arch, f"records state {STATE_CA5A} is not in {PV.ARCHIVE_JSON}"
    if not os.path.exists(PV.archive_path(arch[STATE_CA5A])):
        pytest.skip(f"the archived records file of state {STATE_CA5A[:12]} is absent from this machine "
                    f"({PV.archive_path(arch[STATE_CA5A])}; scripts/archive_records_state.py)")
    folder = os.path.dirname(REAL)
    before = _listing(folder)
    wins = MG.stage1_windows(per_source=1, duration=4.0)
    assert len(wins) == 6 and {w.source for w in wins} == set(S1.sources[1:])
    res = MG.migrate(REAL, tmp_path / "run" / "ckpt_A_010681_s2b.pt", drift_log=tmp_path / "drift_log.jsonl",
                     new=MG.S2B_DRAFT, keep_source=True, records_state=STATE_CA5A)
    src = res.pop("source")
    assert res["source_sha256"].startswith("ad224565")
    ck = CK.load(res["out"], drift_log=tmp_path / "drift_log_migrated.jsonl", records_state=STATE_CA5A)
    assert "records_sha256" not in ck["drift"] and MG.lineage(ck)["source_records_state"] == STATE_CA5A
    PV.audit(ck["model"], ck["ema"], ck["opt"])
    ck["opt"] = None
    lin = MG.lineage(ck)
    assert lin["migrated_from"]["sha256"] == res["source_sha256"]
    codes = lin["codes"]
    assert codes["names_sha256"] == S1.sha256() and codes["store_streams"] > 0 and codes["layout_streams"] > 0
    assert codes["folds"] == ["train"] and res["trained_unknown"] == MG.trained_unknown(codes)
    want = MG.plan(S1, MG.S2B_DRAFT, "copy_unknown", MG.trained_unknown(codes))
    assert lin["plan"] == want.to_dict() and lin["plan_sha256"] == want.sha256() == res["plan_sha256"]
    for weights in ("ema", "model"):
        old = MG.model_of(src, weights)
        new = MG.model_of(ck, weights)
        eq = MG.equivalence(old, new, wins, S1, MG.S2B_DRAFT)
        _assert_equal(eq)
        u = S1.tasks().index("unknown")
        beta = torch.zeros(2, 16)
        sex = torch.tensor([0, 1])
        assert torch.equal(new.prior.condition(Cond(sex, torch.tensor([12, 13]), beta), dropout=False),
                           old.prior.condition(Cond(sex, torch.tensor([u, u]), beta), dropout=False))
        del old, new
    assert _listing(folder) == before
