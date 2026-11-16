"""mdfm.data.windows: native sample times, the record-field whitelist (property test on one real
recording per pilot source with injected label fields), fps refusal, tails, validation, covariates and
the task vocabulary."""
import collections
import copy
import hashlib
import os

import numpy as np
import pytest

from mdfm import paths
from mdfm.data import canonical as C
from mdfm.data import sources as S
from mdfm.data import windows as W

HAVE_INDEX = (os.path.exists(C.index_path()) and os.path.exists(C.manifest_path())
              and os.path.isdir(f"{paths.MOVEATLAS}/datasets"))
needs_index = pytest.mark.skipif(not HAVE_INDEX, reason="record index or MoveAtlas not found")

# injected into every record before cut: none of these names or values may reach a window
INJECTED = {"label": "LEAK_label", "labels": ["LEAK_labels"], "group": "LEAK_group", "cohort": "LEAK_cohort",
            "diagnosis": "LEAK_diag", "severity": "LEAK_sev", "phase": "LEAK_phase", "tier": "LEAK_tier",
            "lockbox": True, "scope": "LEAK_scope", "lane": "LEAK_lane", "analog_labels": ["LEAK_analog"],
            "label_prefix_stripped": True, "notes": "LEAK_notes", "aux_streams": [{"label": "LEAK_aux"}],
            "manifest": {"group": "LEAK_manifest_group", "tier": "LEAK_manifest_tier"}, "age": 44.0}


@pytest.fixture
def no_tables(monkeypatch):
    """Fake records read no demographic table (covariates then fall back to the record's own age)."""
    monkeypatch.setattr(S, "demographic", lambda source, subject_ns: {})


def _fake_paired(n_imu=2550, fps_imu=100.0, n_force=None, fps_force=100.0, t_force0=0.0, fps="native"):
    rng = np.random.default_rng(0)
    imu = {"signal": rng.normal(size=(n_imu, 1, 9)).astype(np.float32), "location_ids": np.array([2]),
           "channel_mask": np.ones((1, 9), bool), "fps": fps_imu if fps == "native" else fps, "source": "sixmwt_lifespan",
           "subject_ns": "sixmwt_lifespan:S99", "task": "6mwt", "age": 34.0}
    streams, t, roles = {"imu": imu}, {"imu": np.arange(n_imu) / fps_imu}, {"imu": "input"}
    if n_force:
        totals = np.zeros((2, 9), bool)
        totals[:, 8] = True
        streams["force"] = {"force": rng.normal(size=(n_force, 2, 9)).astype(np.float32), "fps": fps_force,
                            "sensor_mask": totals}
        t["force"] = t_force0 + np.arange(n_force) / fps_force
        roles["force"] = "target"
    return S.PairedRecord(source="sixmwt_lifespan", person_id="sixmwt_lifespan:S99", subject_ns="sixmwt_lifespan:S99",
                          key="6mwt", streams=streams, t=t, roles=roles)


# ---------------------------------------------------------------- vocabulary and covariates
def test_task_vocabulary_is_the_spec_list_and_hashed():
    v = W.task_vocabulary()
    assert v["tasks"] == ["walk", "turn", "stand", "sit_stand", "stairs", "slope", "transfer", "free_living", "adl",
                          "sport_dance", "other", "unknown"]
    with open(W.VOCABULARY, "rb") as f:
        assert v["sha256"] == hashlib.sha256(f.read()).hexdigest()
    assert v["status"] == "DRAFT" and set(S.PILOT) <= set(v["rules"])
    assert W.task_name("gait120", "stair_descent_trial03") == "stairs"
    assert W.task_name("weargait_pd", "Balance") == "stand" and W.task_name("weargait_pd", "TUG") == "transfer"
    assert W.task_name("amass", "KIT/3/walking_slow02_poses.npz") == "walk"
    assert W.task_name("nobody", "anything") == "unknown"


@needs_index
def test_every_admitted_recording_has_a_task():
    df = S.rows()
    names = {(s, r): W.task_name(s, r) for s, r in zip(df["source"], df["recording"])}
    unknown = sorted(k for k, v in names.items() if v == "unknown")
    assert not unknown, unknown[:5]


def test_age_band_edges():
    assert [W.age_band(a) for a in (0, 9.9, 10, 25, 69.9, 70, 95)] == [0, 0, 1, 2, 6, 7, 7]
    assert W.age_band(None) == W.age_band(float("nan")) == W.age_band("x") == W.age_band(-1) == 8


# ---------------------------------------------------------------- cut on fake records
def test_native_times_tails_and_offsets(no_tables):
    pr = _fake_paired(n_imu=2550, n_force=25500, fps_force=1000.0, t_force0=-0.12)
    stats = collections.Counter()
    ws = W.cut(pr, stats=stats)
    assert [w.t0 for w in ws] == [0.0, 10.0, 20.0] and ws[-1].duration == pytest.approx(5.5)
    for w in ws:
        W.validate(w)
        for s in w.streams:
            native = pr.t[s.modality]
            got = s.t.astype(np.float64) + w.t0
            i0 = np.searchsorted(native, w.t0)
            assert np.allclose(got, native[i0:i0 + len(got)], atol=1e-5)          # native samples, no resampling
            src = pr.streams[s.modality]["signal" if s.modality == "imu" else "force"][i0:i0 + len(got)]
            assert np.array_equal(s.values, np.where(s.chan_mask[None], src, 0))
    assert stats["windows"] == 3 and stats["tails_dropped"] == 0
    # a 0.5 s tail is dropped and counted; a record shorter than 1 s gives no window
    stats = collections.Counter()
    assert len(W.cut(_fake_paired(n_imu=2050), stats=stats)) == 2
    assert stats["tails_dropped"] == 1 and stats["tail_s_dropped"] == pytest.approx(0.5)
    stats = collections.Counter()
    assert W.cut(_fake_paired(n_imu=90), stats=stats) == [] and stats["records_short"] == 1
    # the target samples before the first input sample never enter a window
    assert all(s.t[0] >= 0 for w in ws for s in w.streams)


def test_record_without_fps_is_refused(no_tables):
    for bad in (float("nan"), None, 0.0, float("inf")):
        with pytest.raises(ValueError, match="fps"):
            W.cut(_fake_paired(fps=bad))


def test_force_sample_valid_is_honoured(no_tables):
    """A force record's sample_valid (the weargait zero gaps of sources.pair) marks samples invalid and zeroes
    them; a sample_valid of the wrong shape raises."""
    pr = _fake_paired(n_imu=300, n_force=300)
    sv = np.ones((300, 2), bool)
    sv[:50] = False
    sv[120, 1] = False
    pr.streams["force"]["sample_valid"] = sv
    w = W.cut(pr)[0]
    W.validate(w)
    f = next(s for s in w.streams if s.modality == "force")
    assert np.array_equal(f.valid, sv) and not f.values[:50].any() and not f.values[120, 1].any()
    assert f.values[50:, :, 8].any()
    pr.streams["force"]["sample_valid"] = sv[:, :1]
    with pytest.raises(ValueError, match="sample_valid"):
        W.cut(pr)


def test_magnetometer_masked_and_zeroed(no_tables):
    pr = _fake_paired()
    w = W.cut(pr)[0]
    imu = w.streams[0]
    assert not imu.chan_mask[:, 6:].any() and not imu.values[..., 6:].any() and imu.chan_mask[:, :6].all()
    assert w.cov == {"sex": 2, "age_band": 3, "task": W.task_vocabulary()["ids"]["walk"]}     # age from the record


def test_validate_rejects_contract_violations(no_tables):
    w = W.cut(_fake_paired())[0]
    W.validate(w)
    cases = []
    for mutate in (lambda s: s.values.__setitem__((0, 0, 7), 1.0),                 # masked magnetometer non-zero
                   lambda s: s.values.__setitem__((0, 0, 0), np.nan),
                   lambda s: s.meta.__setitem__("group", "x"),
                   lambda s: setattr(s, "values", s.values.astype(np.float64)),
                   lambda s: setattr(s, "t", s.t[::-1].copy()),
                   lambda s: setattr(s, "truth", {"mounts": 0})):
        bad = copy.deepcopy(w)
        mutate(bad.streams[0])
        cases.append(bad)
    bad = copy.deepcopy(w)
    bad.cov = dict(bad.cov, label=1)
    cases.append(bad)
    bad = copy.deepcopy(w)
    bad.cov = dict(bad.cov, sex=5)
    cases.append(bad)
    bad = copy.deepcopy(w)
    bad.streams[0].role = "target"
    cases.append(bad)
    for bad in cases:
        with pytest.raises(ValueError):
            W.validate(bad)


# ---------------------------------------------------------------- whitelist on real records
def _values(obj):
    if hasattr(obj, "__dataclass_fields__"):
        for name in obj.__dataclass_fields__:
            yield from _values(getattr(obj, name))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _values(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _values(v)
    else:
        yield obj


ALLOWED = ({f for f in W.Window.__dataclass_fields__} | {f for f in W.Stream.__dataclass_fields__} | set(W.META_KEYS)
           | {"sex", "age_band", "task", "muscle", "side"})


@needs_index
@pytest.mark.parametrize("source", S.PILOT)
def test_no_field_outside_the_whitelist_survives(source):
    df = S.rows(source)
    spec = S.SOURCES[source]
    # the shortest recording of the source with at least 1.5 s of input and as many stream kinds as any
    df = df.assign(input_s=np.where(df["stream"] == spec.streams[0].kind, df["n_frames"] / df["fps"], 0.0))
    per = df.groupby(["subject_ns", "recording"]).agg(kinds=("stream", "nunique"), input_s=("input_s", "sum"))
    per = per[per["input_s"] >= 1.5].sort_values(["kinds", "input_s"], ascending=[False, True])
    target = per.index[0]
    pr = next(S.paired(source, recordings={target}))
    for rec in pr.streams.values():
        rec.update(INJECTED)
        rec["manifest"] = dict(INJECTED["manifest"])
    ws = W.cut(pr)
    assert ws and set(pr.streams) == {s.kind for s in spec.streams if s.kind in pr.streams}
    for w in ws:
        W.validate(w)
        keys = W.keys_at_any_depth(w)
        assert keys <= ALLOWED, sorted(keys - ALLOWED)
        assert not W.forbidden_keys(w)
        leaked = [v for v in _values(w) if isinstance(v, str) and v.startswith("LEAK")]
        assert not leaked, leaked
        assert w.person_id == pr.person_id and w.key == pr.key and not w.synthetic
        for s in w.streams:
            assert s.role == pr.roles[s.modality]
            if s.modality == "pose3d":
                assert s.meta["unit_to_si"] == 1.0 and s.meta["scale"] > 0
            if s.modality == "force":
                assert s.meta["unit"] == ("unknown" if source == "weargait_pd" else "N")
            if s.modality == "force" and source == "weargait_pd" and w is ws[0]:
                assert not s.valid[0].any()          # every insole record starts with a zero gap (sources.pair)


# ---------------------------------------------------------------- root track (B4.13b)
SPEC_META = {"layout": "mocap3d", "unit_to_si": 1.0}        # shared by records, as a source spec would share it


def _fake_pose(n=350, fps=100.0):
    """A stroke-like recording: pose3d at 100 Hz (input) and force at 1000 Hz (target); a root track with a
    NaN run (140 to 149) and rows marked invalid (200 to 209)."""
    rng = np.random.default_rng(21)
    pose = {"coords": rng.normal(scale=0.5, size=(n, 15, 3)).astype(np.float32), "joint_mask": np.ones(15, bool),
            "coord_mask": np.ones(3, bool), "valid_mask": np.ones((n, 15), bool), "scale": 0.5, "dim": 3,
            "modality": "3d_meter", "fps": fps, "source": "stroke_gait_vancriekinge",
            "subject_ns": "stroke_gait:able_bodied:S901", "task": "gait_SUBJ9_1"}
    totals = np.zeros((2, 9), bool)
    totals[:, 8] = True
    force = {"force": rng.normal(size=(10 * n, 2, 9)).astype(np.float32), "fps": 10 * fps, "sensor_mask": totals}
    pr = S.PairedRecord(source="stroke_gait_vancriekinge", person_id="stroke_gait:able_bodied:S901",
                        subject_ns="stroke_gait:able_bodied:S901", key="gait_SUBJ9_1",
                        streams={"pose3d": pose, "force": force},
                        t={"pose3d": np.arange(n) / fps, "force": np.arange(10 * n) / (10 * fps)},
                        roles={"pose3d": "input", "force": "target"}, meta={"pose3d": SPEC_META})
    t = np.arange(n) / fps
    root = np.stack([1.2 * t - 3.0, 0.95 + 0.02 * np.sin(2 * np.pi * 2 * t), 0.4 + 0.01 * t], axis=1)
    root[140:150, 1] = np.nan
    valid = np.ones(n, bool)
    valid[200:210] = False
    return pr, root, valid


def test_attach_root_cleans_and_leaves_shared_meta_alone(no_tables):
    pr, root, valid = _fake_pose()
    arrays_before = {k: v.copy() for k, v in pr.streams["pose3d"].items() if isinstance(v, np.ndarray)}
    out = W.attach_root(pr, root, valid)
    assert out is pr and set(W.ROOT_KEYS) <= set(W.META_KEYS)
    m = pr.meta["pose3d"]
    ok = valid & np.isfinite(root).all(axis=1)
    assert m["root"].dtype == np.float32 and m["root"].shape == (350, 3) and m["root_valid"].dtype == bool
    assert np.array_equal(m["root_valid"], ok) and ok.sum() == 330
    assert np.array_equal(m["root"][ok], root[ok].astype(np.float32)) and not m["root"][~ok].any()
    assert m["layout"] == "mocap3d" and SPEC_META == {"layout": "mocap3d", "unit_to_si": 1.0}   # shared dict untouched
    assert all(np.array_equal(pr.streams["pose3d"][k], v) for k, v in arrays_before.items())
    for bad in ((root[:-1], valid), (root, valid[:-1]), (root[:, :2], valid)):
        with pytest.raises(ValueError, match="pose samples"):
            W.attach_root(_fake_pose()[0], *bad)
    no_pose = _fake_paired()
    with pytest.raises(ValueError, match="pose3d"):
        W.attach_root(no_pose, np.zeros((len(no_pose.t["imu"]), 3)), np.ones(len(no_pose.t["imu"]), bool))


def test_cut_slices_the_root_with_the_pose_samples(no_tables):
    pr, root, valid = _fake_pose()
    plain = W.cut(copy.deepcopy(pr), win_s=1.0, stride_s=0.6)
    W.attach_root(pr, root, valid)
    ws = W.cut(pr, win_s=1.0, stride_s=0.6)
    assert len(ws) == len(plain) == 5
    full = pr.meta["pose3d"]
    for w, p in zip(ws, plain):
        W.validate(w)
        pose = next(s for s in w.streams if s.modality == "pose3d")
        force = next(s for s in w.streams if s.modality == "force")
        assert "root" not in force.meta and "root_valid" not in force.meta
        i0 = int(round(w.t0 * 100))
        T = len(pose.t)
        assert np.array_equal(pose.meta["root"], full["root"][i0:i0 + T])
        assert np.array_equal(pose.meta["root_valid"], full["root_valid"][i0:i0 + T])
        # everything else equals the cut without a root track
        pp = next(s for s in p.streams if s.modality == "pose3d")
        assert {k: v for k, v in pose.meta.items() if k not in W.ROOT_KEYS} == pp.meta and "root" not in pp.meta
        assert np.array_equal(pose.values, pp.values) and np.array_equal(pose.t, pp.t)
    # the window holding the NaN run and the invalid rows has them invalid and zero
    w = ws[2]
    pose = next(s for s in w.streams if s.modality == "pose3d")
    assert w.t0 == pytest.approx(1.2) and not pose.meta["root_valid"][20:30].any() and pose.meta["root_valid"][:20].all()
    assert not pose.meta["root"][20:30].any()


def test_validate_rejects_root_violations(no_tables):
    pr, root, valid = _fake_pose()
    W.attach_root(pr, root, valid)
    w = W.cut(pr, win_s=1.0, stride_s=1.0)[1]
    W.validate(w)
    k = next(i for i, s in enumerate(w.streams) if s.modality == "pose3d")
    kf = next(i for i, s in enumerate(w.streams) if s.modality == "force")
    cases = [lambda w: w.streams[k].meta.pop("root_valid"),
             lambda w: w.streams[k].meta.__setitem__("root", w.streams[k].meta["root"].astype(np.float64)),
             lambda w: w.streams[k].meta.__setitem__("root", w.streams[k].meta["root"][:-1]),
             lambda w: w.streams[k].meta.__setitem__("root_valid", w.streams[k].meta["root_valid"].astype(np.uint8)),
             lambda w: w.streams[k].meta["root"].__setitem__((0, 0), np.inf),
             lambda w: (w.streams[k].meta["root_valid"].__setitem__(0, False),
                        w.streams[k].meta["root"].__setitem__((0, 0), 1.0)),
             lambda w: w.streams[kf].meta.update(root=np.zeros((len(w.streams[kf].t), 3), np.float32),
                                                 root_valid=np.ones(len(w.streams[kf].t), bool))]
    for mutate in cases:
        bad = copy.deepcopy(w)
        mutate(bad)
        with pytest.raises(ValueError, match="root"):
            W.validate(bad)
    # a root of the wrong length on the record is refused by cut
    pr2, _, _ = _fake_pose()
    pr2.meta = {"pose3d": dict(SPEC_META, root=np.zeros((10, 3), np.float32), root_valid=np.ones(10, bool))}
    with pytest.raises(ValueError, match="root"):
        W.cut(pr2)
