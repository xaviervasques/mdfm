"""Tests of the layout families and the synthetic renderer (spec docs/MFM_MODEL_CODE_SPEC_2026-09-28.md, 2.4 and
task B4.10): layouts equal real records of their sources (sites, rate, channel mask); the IMU before band_copy
equals the reference renderer imu_render.render with noise off (float64: 1e-4 g and 1e-2 deg/s; float32 on the
GPU: the bound it meets, 5e-3 g and 5e-2 deg/s); synthetic and real streams share rate, filter and decimation
after band_copy; every Window validates; token counts 82, 122, 164, 527 and 1,094; 262,144 timestamps under
4 GB; mixture draws; the record reader refuses CODY and EVAL sources; the IMU nuisance truth rebuilds the
noisy render; force totals are invalid in flight; the stacked band copies equal one copy per stream.

Bodies: the standing KIT take of tests/amass_fixture.py (a 10 s window) and the KIT walk of
tests/test_imu_batch.py (5 s), train archives, turned into BodyWindows here (floor at y = 0, pelvis x and z at 0
at the centre knot), so that no test needs the body store.
"""
import json
import os

import numpy as np
import pytest
import torch

from amass_fixture import TAKE, assert_train_take
from mdfm import paths
from mdfm.factory import layouts as L

F64 = torch.float64
HAVE_MA = os.path.exists(paths.smplh_model("neutral"))
needs_ma = pytest.mark.skipif(not HAVE_MA, reason="MoveAtlas SMPL+H model not found")
needs_index = pytest.mark.skipif(not os.path.exists(os.path.join(paths.CACHE, "manifest", "record_index.csv.gz")),
                                 reason="cache/manifest/record_index.csv.gz not built")
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
WALK = ("KIT", "KIT/11/WalkingStraightForwards01_poses.npz")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ALL_SITES = L.Layout("all_sites", (L.StreamLayout("imu", "input", (100.0,), tuple(L.SITES), ("111111000",) * 14,
                                                  "with_gravity"),), mount="generic", device="generic")


# ---------------------------------------------------------------- fixtures
def _window(bank, archive, member, start, n, K=1000):
    """One BodyWindows row of knots start to start + n of a take: floor at y = 0 (frames.floor_height), pelvis
    x and z at 0 at the centre knot, held after n knots."""
    from mdfm.body import frames
    from mdfm.body.kinematics import KnotBatch
    from mdfm.body.smplh import SMPLHBody, gender_code
    from mdfm.data.amass import load_sequence
    from mdfm.data.prep import amass_to_body_clock
    seq = load_sequence(archive, member)
    r6, tr, betas = amass_to_body_clock(seq, rate=100.0, cutoff=12.0, body=SMPLHBody(seq["gender"], dtype=F64))
    g = gender_code(seq["gender"])
    r6, tr = r6[start:start + n], tr[start:start + n].clone()
    rest = bank.rest(betas[None].to(F64), torch.tensor([g]))[0]
    kb = KnotBatch(r6[None], tr[None], rest[None])
    joints = kb.pose(torch.zeros(n, dtype=torch.int64), torch.arange(n, dtype=F64) / 100.0)[0]
    tr[:, 1] -= frames.floor_height(joints)
    tr[:, [0, 2]] -= joints[n // 2, 0, [0, 2]]
    R6 = torch.cat([r6, r6[-1:].expand(K - n, 22, 6)]) if n < K else r6
    TR = torch.cat([tr, tr[-1:].expand(K - n, 3)]) if n < K else tr
    return {"rot6d": R6.float(), "transl": TR.float(), "n": n, "betas": betas.float(), "gender": g}


def _bodies(rows):
    from mdfm.data.body_store import BodyWindows
    K = rows[0]["rot6d"].shape[0]
    n = len(rows)
    lengths = torch.tensor([r["n"] for r in rows], dtype=torch.int64)
    return BodyWindows(torch.stack([r["rot6d"] for r in rows]), torch.stack([r["transl"] for r in rows]),
                       torch.zeros(n, K, 2, dtype=torch.bool), torch.arange(K)[None] < lengths[:, None], lengths,
                       torch.stack([r["betas"] for r in rows]), torch.tensor([r["gender"] for r in rows]),
                       np.array([f"test:kit:{k}" for k in range(n)]), torch.zeros(n, dtype=torch.int64),
                       torch.arange(n, dtype=torch.int64), torch.zeros(n, dtype=torch.int64))


@pytest.fixture(scope="module")
def layouts():
    return L.load_layouts()


@pytest.fixture(scope="module")
def bank():
    from mdfm.body.smplh import SkeletonBank
    return SkeletonBank(F64)


@pytest.fixture(scope="module")
def stand(bank):
    assert_train_take()
    return _window(bank, *TAKE, start=50, n=1000)


@pytest.fixture(scope="module")
def walk(bank):
    from mdfm.data import amass_index
    if os.path.exists(amass_index.INDEX_PATH):
        assert amass_index.check_member(*WALK) == "train"
    return _window(bank, *WALK, start=0, n=500)


@pytest.fixture(scope="module")
def real_last():
    """The last admitted recording of every source the families read (layouts.read_real)."""
    return L.read_real(last=True)


@pytest.fixture(scope="module")
def sampler(layouts, bank):
    from mdfm.factory.synth import SynthSampler
    return SynthSampler(layouts=layouts, skel=bank, dtype=torch.float32, device=DEVICE)


# ---------------------------------------------------------------- layouts, draws and tokens
def test_site_offsets_equal_site_table_exactly():
    from mdfm.body.smplh import PARENTS
    from mdfm.factory.imu_render import SITE_JOINTS, SITES, site_offsets, site_table
    g = torch.Generator().manual_seed(0)
    rest = torch.zeros(3, 22, 3, dtype=F64)
    for j in range(1, 22):
        rest[:, j] = rest[:, PARENTS[j]] + 0.2 * torch.randn(3, 3, generator=g, dtype=F64)
    joints, off = site_offsets(rest)
    assert joints.tolist() == SITE_JOINTS and off.shape == (3, 14, 3)
    for w in range(3):
        table = site_table(rest[w])
        for k, s in enumerate(SITES):
            assert table[s][0] == int(joints[k]) and torch.equal(table[s][1], off[w, k])


def test_token_counts_of_the_families(layouts):
    expect = {"wrist_acc": 82, "lower_back": 122, "wrist_L": 122, "nonan3": 164, "camera1": 82, "mocap3d": 82,
              "mocap3d_plate": 82, "g1_six": 527, "xsens13": 1094, "xsens13_insoles": 1094}
    for name, n in expect.items():
        lay = L.concrete(layouts[name], 0)
        assert L.token_count(lay, 10.0) == n, name
        assert L.token_count(lay, 9.99) == n, name       # a full window spans 999 knot intervals
    rnd = layouts["random_sites"]
    one = L.Layout("r", (L.StreamLayout("imu", "input", (100.0,), ("pocket",), ("111111000",), "with_gravity"),))
    assert L.token_count(one, 10.0) == 122 and rnd.k_sites == (1, 14)
    with pytest.raises(ValueError):
        L.token_count(rnd, 10.0)
    assert L.n_patches(0.9) == 4 and L.n_patches(30.0) == 40


def test_mixture_draws_match_within_two_percent(layouts):
    gen = np.random.default_rng(1)
    names = []
    ks = []
    for _ in range(10000):
        lay = L.sample_layout(gen, layouts=layouts)
        assert lay.concrete
        names.append(lay.name)
        if lay.name == "random_sites":
            s = lay.imu
            ks.append(len(s.sites))
            assert list(s.sites) == [x for x in L.SITES if x in s.sites] and s.fps[0] in (100.0, 200.0)
        if lay.name == "camera1":
            assert lay.streams[0].fps[0] in (25.0, 30.0, 50.0, 60.0)
    total = sum(L.DEFAULT_MIXTURE.values())
    for name, w in L.DEFAULT_MIXTURE.items():
        assert abs(names.count(name) / 10000 - w / total) < 0.02, name
    assert min(ks) == 1 and max(ks) == 14
    custom = {"xsens13": 3.0, "camera1": 1.0}
    got = [L.sample_layout(gen, custom, layouts).name for _ in range(10000)]
    assert abs(got.count("xsens13") / 10000 - 0.75) < 0.02 and set(got) == set(custom)


def test_layouts_json_holds_every_family_and_the_record_keys(layouts):
    with open(L.LAYOUTS_PATH, encoding="utf-8") as f:
        doc = json.load(f)
    assert doc["format"] == L.FORMAT and list(layouts) == list(L.FAMILY_RULES)
    for name, lay in layouts.items():
        rule = L.FAMILY_RULES[name]
        assert lay.mount == rule.get("mount") and lay.device == rule.get("device")
        for (modality, role, source, subset), s in zip(rule["streams"], lay.streams):
            assert (s.modality, s.role) == (modality, role)
            if source:
                assert s.record["source"] == source and s.record["subject_ns"] and s.record["key"]
            if subset:
                assert s.sites == subset
            if modality == "imu":
                assert not s.chan_mask[:, 6:].any()     # magnetometer never kept
    assert layouts["nonan3"].imu.fps == (200.0,) and layouts["nonan3"].imu.channels == "acc"
    assert layouts["wrist_acc"].imu.channels == "acc" and layouts["xsens13"].imu.channels == "acc+gyro"
    assert layouts["mocap3d_plate"].stream("force").plate and not layouts["xsens13_insoles"].stream("force").plate
    assert layouts["xsens13_insoles"].stream("force").unit == "unknown"
    assert layouts["mocap3d_plate"].stream("force").unit == "N"


def test_device_families_follow_device_noise_and_the_priors():
    from mdfm.factory.imu_render import Device
    from mdfm.kernels.nuisance import MountPriors
    fam = L.device_families()
    with open(L.DEVICE_NOISE_PATH, encoding="utf-8") as f:
        dn = json.load(f)
    x = fam["xsens13"]
    for row in dn["per_site"]:
        k = L.SITES.index(row["site"])
        assert np.array_equal(x.noise_acc[k], row["acc_sigma_g"])
        assert np.array_equal(x.noise_gyro[k], row["gyro_sigma_dps"])
        assert np.array_equal(x.bias_gyro_sd[k], row["gyro_bias_sd_dps"])
    pr = MountPriors.load()
    f = pr.families.index("xsens13")
    assert np.allclose(pr.noise_acc[f].numpy(), x.noise_acc) and np.allclose(pr.noise_gyro[f].numpy(), x.noise_gyro)
    assert np.allclose(pr.bias_gyro_sd[f].numpy(), x.bias_gyro_sd)
    d = Device()
    g = fam["generic"]
    assert (g.noise_acc == d.acc_noise_g).all() and (g.noise_gyro == d.gyro_noise_dps).all()
    assert g.noise_scale == L.GENERIC_NOISE_SCALE and x.noise_scale == (1.0, 1.0)


def test_prior_mounts_equal_the_prior_draw():
    from mdfm.factory.synth import prior_mounts
    from mdfm.kernels.nuisance import MountPriors
    pr = MountPriors.load()
    for fam in ("xsens13", "generic"):
        sites = ["head", "wrist_L", "shank_R", "foot_L"]
        ref = pr.sample_mounts(torch.Generator().manual_seed(3), fam, sites, n=2)
        z = torch.randn(2, 4, 3, generator=torch.Generator().manual_seed(3), dtype=F64)
        ids = [L.SITES.index(s) for s in sites] * 2
        got = prior_mounts(pr, [pr.families.index(fam)] * 8, ids, z.reshape(8, 3)).reshape(2, 4, 3, 3)
        assert (got - ref).abs().max() < 1e-12


@needs_ma
@needs_index
def test_layouts_equal_other_real_records_of_their_sources(layouts, real_last):
    real = real_last
    rows = L.check_families(layouts, real)
    assert rows and all(r["verdict"] == "equal" for r in rows), [r for r in rows if r["verdict"] != "equal"]
    first = {(s.record["source"], s.modality): s.record for lay in layouts.values() for s in lay.streams if s.record}
    for key, (_, _, _, rec) in real.items():
        if key in first:
            assert (rec["subject_ns"], rec["key"]) != (first[key]["subject_ns"], first[key]["key"]), key
    assert real[("nonan_gaitprint", "imu")][0] == 200.0 and real[("stroke_gait_vancriekinge", "force")][0] == 1000.0


def test_the_record_reader_refuses_cody():
    from mdfm.data.canonical import Refused
    for name in ("cody_1", "cody_2", "cody_3"):
        with pytest.raises(Refused):
            L.record_streams(name, {"pose2d"})
        with pytest.raises(Refused):
            L.record_streams(name, {"imu"}, last=True)


@needs_index
def test_the_record_reader_refuses_eval_sources():
    """An EVAL source is refused by the index selection, before any record is read (no evaluation flag)."""
    from mdfm.data.canonical import Refused
    for name, kind in (("care_pd", "pose3d"), ("kaggle_fog", "imu")):
        with pytest.raises(Refused):
            L.record_streams(name, {kind})
        with pytest.raises(Refused):
            L.record_streams(name, {kind}, last=True)


# ---------------------------------------------------------------- renderer
def _reference(bodies, windows, bank, device=DEVICE):
    """imu_render.render (float64 BodyTrajectory, on `device`) at the exact sample times and drawn mounts of
    each window."""
    from mdfm.body.smplh import SMPLHBody
    from mdfm.body.trajectory import BodyTrajectory
    from mdfm.factory.imu_render import SITES, render
    out = []
    for i, w in enumerate(windows):
        st = w.streams[0]
        g = int(bodies.gender[i])
        n = int(bodies.lengths[i])
        rest = bank.rest(bodies.betas[i:i + 1].to(F64), torch.tensor([g]))[0].to(device)
        on = lambda x: x.to(device=device, dtype=F64)  # noqa: E731
        traj = BodyTrajectory(SMPLHBody(("female", "male")[g], dtype=F64).to(device), on(bodies.rot6d[i, :n]),
                              on(bodies.transl[i, :n]), on(bodies.betas[i]), rest=rest)
        t64 = st.truth["t_first_s"] + np.arange(len(st.t)) / st.fps
        assert np.abs(t64 - st.t).max() < 1e-6
        mount = {s: on(torch.as_tensor(st.truth["mount"][k])) for k, s in enumerate(SITES)}
        sig, _, _ = render(traj, on(torch.from_numpy(t64)), sites=SITES, mount=mount)
        out.append(sig.numpy())
    return out


@needs_ma
def test_imu_before_band_copy_equals_the_reference_renderer(layouts, bank, stand, walk):
    from mdfm.factory.synth import RenderSpec, SynthSampler
    bodies = _bodies([walk, stand])
    specs = [RenderSpec(ALL_SITES, None, True)] * 2
    ref = None
    worst = {}
    for dt, dev in ((F64, DEVICE), (torch.float32, DEVICE)):
        sy = SynthSampler(layouts=layouts, skel=bank, dtype=dt, device=dev)
        ws = sy.render(bodies, specs, 11, band=False)
        if ref is None:
            ref = _reference(bodies, ws, bank)
        ea = max(float(np.abs(w.streams[0].values[..., 0:3] - r[..., 0:3]).max()) for w, r in zip(ws, ref))
        eg = max(float(np.abs(w.streams[0].values[..., 3:6] - r[..., 3:6]).max()) for w, r in zip(ws, ref))
        worst[(str(dt), dev)] = (ea, eg)
        assert all(w.streams[0].values.shape == (len(w.streams[0].t), 14, 9) for w in ws)
    assert max(float(np.abs(r[..., 3:6]).max()) for r in ref) > 100.0          # a walk: rates of hundreds of deg/s
    print("worst |synthetic - reference| (acc g, gyro deg/s):", worst)
    assert bank.J_template.dtype == F64 and bank.J_template.device.type == "cpu"     # the samplers copied it
    for (dt, dev), (ea, eg) in worst.items():
        if dt == str(F64):
            assert ea < 1e-4 and eg < 1e-2, (dt, dev, ea, eg)
        else:
            assert ea < 5e-3 and eg < 5e-2, (dt, dev, ea, eg)
    # a given mount is used as is
    M = torch.linalg.qr(torch.randn(14, 3, 3, generator=torch.Generator().manual_seed(2), dtype=F64))[0]
    M = M * torch.sign(torch.linalg.det(M))[:, None, None]
    sy = SynthSampler(layouts=layouts, skel=bank, dtype=F64, device=DEVICE)
    w = sy.render(_bodies([walk]), [RenderSpec(ALL_SITES, M, True)], 5, band=False)[0]
    assert np.abs(w.streams[0].truth["mount"] - M.numpy()).max() < 1e-7
    r = _reference(_bodies([walk]), [w], bank)[0]
    assert np.abs(w.streams[0].values[..., 0:6] - r[..., 0:6]).max() < 1e-2


@needs_ma
def test_imu_truth_rebuilds_the_noisy_render(layouts, bank, walk):
    """Every nuisance on: the stream truth (offset = site_table + offset_res, mount, biases) given to the
    single-sensor kernel kernels.imu.imu leaves the white noise of the truth levels (mean within 4 standard
    errors of 0, sd within 15 percent), per site and channel; masked channels read 0."""
    from mdfm.body.smplh import SMPLHBody
    from mdfm.body.trajectory import BodyTrajectory
    from mdfm.factory.imu_render import site_table
    from mdfm.factory.synth import RenderSpec, SynthSampler
    from mdfm.kernels.imu import G0, imu
    sy = SynthSampler(layouts=layouts, skel=bank, dtype=F64, device=DEVICE)
    bodies = _bodies([walk])
    n, g = int(bodies.lengths[0]), int(bodies.gender[0])
    rest = bank.rest(bodies.betas[:1].to(F64), bodies.gender[:1])[0]
    traj = BodyTrajectory(SMPLHBody(("female", "male")[g], dtype=F64), bodies.rot6d[0, :n].to(F64),
                          bodies.transl[0, :n].to(F64), bodies.betas[0].to(F64), rest=rest)
    table = site_table(rest)
    for name in ("g1_six", "nonan3"):
        lay = L.concrete(layouts[name], 0)
        st = sy.render(bodies, [RenderSpec(lay)], 13, band=False)[0].streams[0]
        tr = st.truth
        t = torch.from_numpy(tr["t_first_s"] + np.arange(len(st.t)) / st.fps)
        sd = np.concatenate([tr["noise_acc"], tr["noise_gyro"]], 1)
        for k, s in enumerate(lay.imu.sites):
            j, off0 = table[s]
            off = off0 + torch.as_tensor(tr["offset_res"][k], dtype=F64)
            assert np.abs(off.numpy() - tr["offset"][k]).max() < 1e-6 and np.abs(tr["offset_res"][k]).max() > 0
            acc, gyro = imu(traj, t, j, off, mount=torch.as_tensor(tr["mount"][k], dtype=F64))
            ref = np.concatenate([acc.numpy() / G0 + tr["bias_acc"][k], np.rad2deg(gyro.numpy()) + tr["bias_gyro"][k]], 1)
            for c in range(6):
                if not st.chan_mask[k, c]:
                    assert (st.values[:, k, c] == 0).all() and tr[("bias_acc", "bias_gyro")[c // 3]][k, c % 3] == 0
                    continue
                r = st.values[:, k, c] - ref[:, c]
                assert abs(r.mean()) < 4 * sd[k, c] / np.sqrt(len(r)), (name, s, c, r.mean())
                assert abs(r.std() / sd[k, c] - 1) < 0.15, (name, s, c, r.std(), sd[k, c])


@needs_ma
def test_every_family_renders_windows_that_validate(layouts, sampler, stand, walk):
    from mdfm.data import windows as W
    from mdfm.data.window_store import band_plan
    for name, fam in layouts.items():
        lay = L.concrete(fam, 3) if fam.k_sites is None else L.concrete(fam, 4)
        ws = sampler.render(_bodies([stand, walk]), [lay, lay], 7)
        for w in ws:
            W.validate(w)
            assert w.synthetic and w.source == "synthetic" and w.body["lengths"] in (1000, 500)
            assert L.window_tokens(w) == L.token_count(lay, w.duration)
            for s, sl in zip(w.streams, lay.streams):
                plan = band_plan(s.modality, sl.fps[0])
                assert s.fps == plan.fps_out and s.meta["layout"] == name
                assert s.truth["v"].shape == s.truth["a"].shape == (len(s.t), 22, 3)
                assert np.allclose(np.diff(s.t), plan.k / sl.fps[0], atol=1e-5)
                assert s.t[0] >= 0 and s.t[-1] < w.duration
                if s.modality == "imu":
                    assert s.meta["site_ids"] == sl.site_ids and not s.chan_mask[:, 6:].any()
                    assert np.array_equal(s.chan_mask, sl.chan_mask)
                    assert (s.values[:, ~sl.chan_mask] == 0).all()
        full = ws[0]
        assert full.duration == pytest.approx(9.99)
        if name in ("xsens13", "xsens13_insoles"):
            assert L.window_tokens(full) == 1094
        if name == "g1_six":
            assert L.window_tokens(full) == 527
        if name == "nonan3":
            assert L.window_tokens(full) == 164 and full.streams[0].fps == 100.0
        if name in ("wrist_acc", "camera1", "mocap3d", "mocap3d_plate"):
            assert L.window_tokens(full) == 82
        if name in ("lower_back", "wrist_L"):
            assert L.window_tokens(full) == 122


@needs_ma
@needs_index
def test_synthetic_and_real_streams_share_rate_filter_and_decimation(layouts, sampler, stand, real_last):
    from mdfm.data.window_store import band_plan
    real = real_last
    for name, lay in layouts.items():
        if lay.k_sites is not None or name == "camera1":
            continue
        w = sampler.render(_bodies([stand]), [lay], 1)[0]
        rule = L.FAMILY_RULES[name]
        for (modality, _, source, _), s, sl in zip(rule["streams"], w.streams, lay.streams):
            fps_real = real[(source, modality)][0]
            p_real, p_syn = band_plan(modality, fps_real), band_plan(modality, sl.fps[0])
            assert (p_real.kind, p_real.k, p_real.fps_out) == (p_syn.kind, p_syn.k, p_syn.fps_out) == \
                (p_syn.kind, p_syn.k, s.fps), (name, modality)
    nonan = band_plan("imu", real[("nonan_gaitprint", "imu")][0])
    assert (nonan.kind, nonan.k, nonan.fps_out) == ("lowpass", 2, 100.0)
    plate = band_plan("force", real[("stroke_gait_vancriekinge", "force")][0])
    assert (plate.k, plate.fps_out) == (10, 100.0)


@needs_ma
def test_render_is_deterministic_and_pairs_resample_the_nuisance(layouts, sampler, stand, walk):
    bodies = _bodies([walk, stand])
    lays = [L.concrete(layouts["g1_six"], 0), L.concrete(layouts["camera1"], 0)]
    a = sampler.render(bodies, lays, 21)
    b = sampler.render(bodies, lays, 21)
    for wa, wb in zip(a, b):
        for sa, sb in zip(wa.streams, wb.streams):
            assert np.array_equal(sa.values, sb.values) and np.array_equal(sa.valid, sb.valid)
    pairs = sampler.render_pair(bodies, lays, 22)
    assert len(pairs) == 2
    (p, q), (c, d) = pairs
    assert p.key == q.key and np.array_equal(p.body["rot6d"], q.body["rot6d"])
    assert p.streams[0].meta["site_ids"] == q.streams[0].meta["site_ids"] and p.streams[0].fps == q.streams[0].fps
    assert np.abs(p.streams[0].truth["mount"] - q.streams[0].truth["mount"]).max() > 1e-3
    assert np.abs(p.streams[0].truth["bias_gyro"] - q.streams[0].truth["bias_gyro"]).max() > 0
    assert c.streams[0].fps == d.streams[0].fps
    assert np.abs(c.streams[0].truth["camera"]["R"] - d.streams[0].truth["camera"]["R"]).max() > 1e-6


def test_keep_in_view_frames_a_long_walk():
    from mdfm.factory.camera_render import sample_camera
    from mdfm.factory.synth import keep_in_view
    from mdfm.kernels.camera import project
    T = 300
    x = torch.linspace(-6.5, 6.5, T, dtype=F64)
    joints = [[0.0, 0.9, 0.0], [0.0, 1.45, 0.0], [0.0, 1.7, 0.0]] + [[0.2, 0.9 - 0.06 * k, 0.1] for k in range(12)]
    body = torch.tensor(joints, dtype=F64)
    pos = body[None].repeat(T, 1, 1)
    pos[..., 2] += x[:, None]                       # walking along +z through the target
    gen = torch.Generator().manual_seed(4)
    dollied = 0
    for _ in range(20):
        cam = sample_camera(gen, pos[:, 0].mean(0))
        cam2, d = keep_in_view(cam, pos)
        uv, depth = project(pos, cam2.K, cam2.R, cam2.t)
        assert (depth >= 0.5 - 1e-9).all()
        assert (uv[..., 0] >= 0.05 * 1280 - 1e-6).all() and (uv[..., 0] <= 0.95 * 1280 + 1e-6).all()
        assert (uv[..., 1] >= 0.05 * 720 - 1e-6).all() and (uv[..., 1] <= 0.95 * 720 + 1e-6).all()
        assert d >= 0.0
        dollied += d > 0
    assert dollied == 20                            # a 13 m walk never fits the sampled 3 to 6 m camera
    still = pos[:1].repeat(10, 1, 1)
    cam = sample_camera(gen, still[:, 0].mean(0))
    assert keep_in_view(cam, still)[1] == 0.0       # a still body 3 to 6 m away already fits


@needs_ma
def test_force_streams_newton_floor_gain_and_plate(layouts, sampler, stand, walk):
    ins = L.concrete(layouts["xsens13_insoles"], 0)
    w = sampler.render(_bodies([stand]), [ins], 3)[0]
    f = w.streams[1]
    assert f.modality == "force" and f.role == "target" and f.meta["unit"] == "unknown"
    assert np.array_equal(f.chan_mask, ins.stream("force").chan_mask) and np.array_equal(f.t, w.streams[0].t)
    bw = f.truth["bw"].sum(1)
    assert abs(np.median(bw) - 1.0) < 0.05                       # standing: one body weight on the feet
    total = f.values[..., 8].sum(1)
    assert np.allclose(total[5:-5], bw[5:-5] * f.truth["gain"], rtol=0.05, atol=0.02 * f.truth["gain"])
    plate = L.concrete(layouts["mocap3d_plate"], 0)
    w = sampler.render(_bodies([walk]), [plate], 9)[0]
    p = w.streams[1]
    foot = p.truth["plate"]["foot"]
    assert p.fps == 100.0 and p.meta["unit"] == "N"
    assert p.chan_mask[foot, 8] and not p.chan_mask[1 - foot].any() and (p.values[:, 1 - foot] == 0).all()
    on = p.truth["on_plate"]
    assert on.any() and not on.all()
    assert p.values[on, foot, 8].max() > 0.3 * p.truth["gain"]    # a stance on the plate
    far = np.convolve(on.astype(float), np.ones(11), "same") == 0   # beyond the filter's reach of any plate sample
    assert np.abs(p.values[far, foot, 8]).max() < 0.02 * p.truth["gain"]


@needs_ma
def test_force_is_invalid_in_flight(layouts, sampler, stand):
    """kernels.force.newton_bw leaves the split undefined in flight and asks the caller to mask: the standing
    body has no flight sample and every force sample valid; the same body held 0.5 m above the floor is in
    flight throughout, so its force totals are invalid and read 0, before and after band_copy."""
    from mdfm.data import windows as W
    ins = L.concrete(layouts["xsens13_insoles"], 0)
    up = dict(stand, transl=stand["transl"].clone())
    up["transl"][:, 1] += 0.5
    for band in (False, True):
        ground, air = sampler.render(_bodies([stand, up]), [ins, ins], 3, band=band)
        for w in (ground, air):
            W.validate(w)
        f, a = ground.streams[1], air.streams[1]
        assert not f.truth["flight"].any() and f.valid.all() and (f.values[..., 8] != 0).all()
        assert a.truth["flight"].all() and not a.valid.any() and (a.values == 0).all()
        assert len(a.truth["flight"]) == len(a.t)


def test_band_copies_equal_one_band_copy_per_stream():
    """synth.band_copies stacks the streams of one modality, rate, length and channel count into one
    band_copy call: each part equals band_copy of its stream alone, with different groups and valid gaps
    (a flight gap, a run too short to filter) in the stack."""
    from mdfm.data import window_store as WS
    from mdfm.data import windows as W
    from mdfm.factory.synth import band_copies
    rng = np.random.default_rng(0)

    def stream(modality, fps, T, G, gaps=()):
        cm = np.zeros((G, 9), bool)
        if modality == "imu":
            cm[:, 0:6] = True
        else:
            cm[:, 8] = True
        valid = np.ones((T, G), bool)
        for a, b in gaps:
            valid[a:b] = False
        v = np.where(cm[None] & valid[..., None], rng.standard_normal((T, G, 9)), 0.0).astype(np.float32)
        t = (rng.uniform(0.0, 1.0 / fps) + np.arange(T) / fps).astype(np.float32)
        return W.Stream(modality, "input" if modality == "imu" else "target", float(fps), t, v, cm, valid,
                        {"source": "synthetic"}, {"k": len(gaps)})

    streams = [stream("imu", 100, 500, 3), stream("imu", 100, 500, 3, [(100, 140), (150, 200)]),
               stream("imu", 100, 500, 2, [(10, 20)]), stream("imu", 200, 500, 3), stream("imu", 100, 400, 3),
               stream("force", 1000, 3000, 2, [(1000, 1500)]), stream("force", 1000, 3000, 2)]
    got = band_copies(streams)
    for s, (b, info) in zip(streams, got):
        a, ia = WS.band_copy(s, return_info=True)
        assert np.array_equal(a.values, b.values) and np.array_equal(a.valid, b.valid) and np.array_equal(a.t, b.t)
        assert a.fps == b.fps and np.array_equal(ia.idx, info.idx) and b.truth is s.truth
    short = got[1][0]
    assert not short.valid[140:150].any() and short.valid[:100].all() and short.valid[200:].all()
    assert got[5][0].fps == 100.0 and not got[5][0].valid[100:150].any()


@needs_ma
@needs_cuda
def test_262144_timestamps_render_under_4_gb(layouts, bank, stand):
    from mdfm.factory.synth import SynthSampler, concat_bodies
    sy = SynthSampler(layouts=layouts, skel=bank, dtype=torch.float32, device="cuda")
    one = _bodies([stand])
    bodies = one
    while int((bodies.lengths - 1).sum()) < 262144:
        bodies = concat_bodies(bodies, bodies) if len(bodies.lengths) < 132 else concat_bodies(bodies, one)
    lay = L.concrete(layouts["xsens13"], 0)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    ws = sy.render(bodies, [lay] * len(bodies.lengths), 0, band=False)      # the host-side band copies use no GPU
    torch.cuda.synchronize()
    peak = (torch.cuda.max_memory_allocated() - base) / 1e9
    assert sum(len(w.streams[0].t) for w in ws) >= 262144
    assert peak < 4.0, peak


class _Store:
    """A stand-in for data.body_store.BodyStore: sample() cycles through fixed body windows."""

    def __init__(self, rows):
        import pandas as pd
        self.rows = rows
        self.takes = pd.DataFrame({"take_id": [f"KIT/test_{k}_poses.npz" for k in range(len(rows))]})

    def sample(self, gen, n, win_s=10.0):
        from mdfm.factory.synth import select_bodies
        pick = gen.integers(len(self.rows), size=n)
        return select_bodies(_bodies(self.rows), pick)


@needs_ma
def test_draw_and_sample_render_the_mixture_with_pairs(layouts, bank, stand, walk):
    from mdfm.data import windows as W
    from mdfm.factory.synth import SynthSampler
    sy = SynthSampler(store=_Store([stand, walk]), layouts=layouts, skel=bank, device=DEVICE,
                      mixture={"xsens13": 1.0, "wrist_acc": 1.0, "mocap3d": 1.0})
    ws, pairs = sy.sample(np.random.default_rng(5), 12, pair_share=0.25)
    assert len(ws) == 15 and len(pairs) == 3
    for i, j in pairs:
        a, b = ws[i], ws[j]
        assert j >= 12 and a.key == b.key and a.body["take_id"] == b.body["take_id"]
        assert [s.modality for s in a.streams] == [s.modality for s in b.streams]
        assert a.streams[0].meta["layout"] == b.streams[0].meta["layout"]
    for w in ws:
        W.validate(w)
        assert w.body["take_id"].startswith("KIT/test_") and w.streams[0].meta["layout"] in sy.mixture
    assert set(sy.timing) == set(sy.PHASES) and all(v > 0 for v in sy.timing.values())
    sy.reset_timing()
    again, _ = sy.sample(np.random.default_rng(5), 12, pair_share=0.25)
    assert all(np.array_equal(x.streams[0].values, y.streams[0].values) for x, y in zip(ws, again))
