"""Tests of the open G1 lockbox path (mdfm/gates/g1_open.py; docs/rules/G1_RULE_OPEN_2026-11-13.md, registered as G1
version 2): the rule chosen by the read declaration, the SHA-256 checks of the rule and of the files it adopts, the
lockbox permission and the logged read, the sets and halves of the rule, the fit of a person by the store's rule and
its cache, the outputs of scripts/read_g1_open_lockbox.py, and a v1 declaration left on its own path.

Fixtures only: no lockbox, EVAL, CODY or care_pd record, no index row of a lockbox person and no file of the tree's
caches or stores is read. The real rule registry is read (never written); every read is logged in a temporary log
against a temporary copy of the registry; the gates-only store is moved to a temporary folder; the lockbox member file
is a fixture with its own SHA-256 patched in; the takes, the fits, the halves and the development and calibration
inputs are synthetic (lockbox_takes, fit_person, check_halves and the deny lists are replaced), so the real path from
the declaration to the lines runs on made-up persons. The tree's lockbox read log is compared byte for byte before and
after every test that reaches the logging step.
"""
import csv
import json
import os
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mdfm import paths
from mdfm.data import sources as S
from mdfm.data.canonical import Refused
from mdfm.gates import common
from mdfm.gates import g1 as G
from mdfm.gates import g1_open as GO
from mdfm.train import provenance as PV

GAIT, STROKE = GO.SOURCES
OPEN_SHA = "00c45fcfa4eef33508e8f4585dbfb6b2dbf088ba8fa8fe561b5adc750901e4bf"
V1_SHA = "c3aceff307fff29528c4b0cf90eed2a499af6a92822ba9e824a2df48ccf76da1"
DRAFT_SHA = "f9e099219d06e0a53f1b9a8d7578800e9f4fb3c86a869786ed786c2044f0d504"


def _log_bytes():
    return open(common.READ_LOG, "rb").read() if os.path.exists(common.READ_LOG) else None


@pytest.fixture(autouse=True)
def _tree_log_untouched():
    before = _log_bytes()
    yield
    assert _log_bytes() == before


def _decl(kind="open", **over):
    d = {"format": GO.DECLARATION_FORMAT, "gate": "G1",
         "checkpoint": {"file": "runs/x/ckpt_B_000001.pt", "sha256": "a" * 64}, "config_sha256": "b" * 64,
         "body": {"name": "anny", "version": "0.6.2", "joint_map": "mdfm/synth/openbody.py"},
         "corpus": {"sources": ["gait120", "cmu_mocap"] if kind == "open" else ["amass", "gait120"], "archives": []},
         "train_persons": {"gait120": ["gait120:T900"]},
         "train_persons_sha256": PV.list_hashes({"gait120": ["gait120:T900"]})["all"],
         "inference": {"mode": GO.INFERENCE_MODE}, "sets": {}}
    d.update(over)
    return d


def _registry_copy(tmp_path, drop=None):
    rows = [r for r in common.registry() if not (drop and (r["gate"], r["version"]) == drop)]
    p = str(tmp_path / ("gate_rule_hashes.csv" if drop is None else f"gate_rule_hashes_without_{'_'.join(drop)}.csv"))
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(common.RULE_COLUMNS), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    return p


# ---------------------------------------------------------------- rule selection by declaration
def test_an_open_declaration_reads_the_open_rule_and_checks_what_it_adopts():
    r = GO.declared_rule(_decl("open"))
    assert r["kind"] == "open" and r["sha256"] == OPEN_SHA
    assert common.rel(r["path"]) == "docs/rules/G1_RULE_OPEN_2026-11-13.md"
    assert [(a["path"], a["sha256"]) for a in r["adopts"]] == [("docs/rules/G1_RULE_2026-11-12.md", V1_SHA),
                                                              ("docs/rules/G1_RULE.md", DRAFT_SHA)]
    assert common.rule_status(r["text"]) == ("approved", "2026-11-14")


def test_a_declaration_with_amass_keeps_the_2026_11_12_version():
    r = GO.declared_rule(_decl("v1"))
    assert r["kind"] == "v1" and r["sha256"] == V1_SHA
    assert common.rel(r["path"]) == "docs/rules/G1_RULE_2026-11-12.md"
    assert [(a["path"], a["sha256"]) for a in r["adopts"]] == [("docs/rules/G1_RULE.md", DRAFT_SHA)]
    # an AMASS archive named without the source is AMASS as well
    assert GO.declaration_kind(_decl("open", corpus={"sources": ["gait120"], "archives": ["KIT"]})) == "v1"
    assert GO.declaration_kind(_decl("open")) == "open"


def test_an_unregistered_open_rule_is_refused(tmp_path):
    reg = _registry_copy(tmp_path, drop=("G1", "2"))
    with pytest.raises(Refused, match="not registered"):
        GO.declared_rule(_decl("open"), registry_path=reg)
    assert GO.declared_rule(_decl("v1"), registry_path=reg)["sha256"] == V1_SHA     # v1 stays readable


def test_an_edited_rule_or_an_edited_adopted_file_is_refused(tmp_path, monkeypatch):
    src = open(GO.RULE_PATH, "rb").read()
    copy = tmp_path / "G1_RULE_OPEN_copy.md"
    copy.write_bytes(src)
    reg = _registry_copy(tmp_path)
    with open(reg, "a", encoding="utf-8", newline="") as f:
        csv.writer(f, lineterminator="\n").writerow(["G1", "3", common.rel(str(copy)), common.sha256_bytes(src), "owner",
                                                     "2026-11-14", "2026-11-14T00:00:00", "test copy"])
    monkeypatch.setattr(GO, "RULE_PATH", str(copy))
    assert GO.declared_rule(_decl("open"), registry_path=reg)["sha256"] == OPEN_SHA
    copy.write_bytes(src.replace(b"24 gait120", b"25 gait120", 1))
    with pytest.raises(Refused, match="changed since its registration"):
        GO.declared_rule(_decl("open"), registry_path=reg)
    monkeypatch.setattr(GO, "RULE_PATH", os.path.join(paths.TREE, "docs", "rules", "G1_RULE_OPEN_2026-11-13.md"))
    draft = tmp_path / "G1_RULE.md"
    draft.write_bytes(open(GO.DRAFT_PATH, "rb").read() + b"\n")
    monkeypatch.setattr(GO, "DRAFT_PATH", str(draft))
    with pytest.raises(Refused, match="adopts"):
        GO.declared_rule(_decl("open"))
    with pytest.raises(Refused, match="adopts"):
        GO.declared_rule(_decl("v1"))


def test_a_malformed_declaration_is_refused(tmp_path):
    p = tmp_path / "d.json"
    for bad in ({"format": "other"}, {k: v for k, v in _decl().items() if k != "train_persons"},
                _decl(gate="G3")):
        p.write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(Refused):
            GO.load_declaration(str(p))
    p.write_text(json.dumps(_decl()), encoding="utf-8")
    d, sha = GO.load_declaration(str(p))
    assert d == _decl() and sha == common.sha256_file(str(p))


# ---------------------------------------------------------------- the sets and the halves
def _lockbox_csv(tmp_path, n_gait=24, n_stroke=10, stroke_group="stroke"):
    p = tmp_path / "lockbox_persons.csv"
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["source", "lane", "person_id", "group", "rule"])
        w.writerow(["amass", "body", "amass:person:FIXTURE:1", "FIXTURE", "x"])
        w.writerows(["gait120", "body", f"gait120:L{i:03d}", "all", "x"] for i in range(n_gait))
        w.writerows(["stroke_gait_vancriekinge", "body", f"stroke_gait:stroke:L{i:03d}", stroke_group, "x"]
                    for i in range(n_stroke))
        w.writerow(["weargait_pd", "body", "weargait:FIXTURE", "PD", "x"])
    return str(p)


def test_the_lockbox_sets_are_the_rule_s_24_and_10_persons(tmp_path, monkeypatch):
    p = _lockbox_csv(tmp_path)
    with pytest.raises(Refused, match="SHA-256"):
        GO.lockbox_sets(p)                                       # the fixture is not lockbox v1.2
    monkeypatch.setattr(GO, "LOCKBOX_SHA256", common.sha256_file(p))
    got = GO.lockbox_sets(p)
    assert {s: len(v) for s, v in got.items()} == {GAIT: 24, STROKE: 10} == GO.LOCKBOX_COUNTS
    assert not any(x.startswith(("amass", "weargait")) for v in got.values() for x in v)
    for kw in (dict(n_gait=23), dict(n_stroke=11), dict(stroke_group="able_bodied")):
        q = _lockbox_csv(tmp_path, **kw)
        monkeypatch.setattr(GO, "LOCKBOX_SHA256", common.sha256_file(q))
        with pytest.raises(Refused):
            GO.lockbox_sets(q)


def test_the_halves_follow_the_seeded_rule_and_the_declaration_names_their_sha256(tmp_path, monkeypatch):
    pools = {GAIT: [f"gait120:D{i:03d}" for i in range(19)], STROKE: [f"stroke_gait:stroke:D{i:03d}" for i in range(40)]}
    hv = {"halves": {"dev": {}, "calib": {}}}
    for s in GO.SOURCES:
        hv["halves"]["dev"][s], hv["halves"]["calib"][s] = GO.halves(s, pools[s])
    hv["sha256"] = {k: {s: S.person_hash(v) for s, v in d.items()} for k, d in hv["halves"].items()}
    assert {k: {s: len(v) for s, v in d.items()} for k, d in hv["halves"].items()} == \
        {"dev": {GAIT: 10, STROKE: 20}, "calib": {GAIT: 9, STROKE: 20}}
    lock = {GAIT: ["gait120:L000"], STROKE: ["stroke_gait:stroke:L000"]}
    ck = tmp_path / "ckpt.pt"
    ck.write_bytes(b"checkpoint")
    sha = common.sha256_file(str(ck))
    pred = SimpleNamespace(sha256=sha, train_persons=frozenset(), loaded=None)
    monkeypatch.setattr(GO, "LOCKBOX_SHA256", "9" * 64)
    good = _decl(checkpoint={"file": str(ck), "sha256": sha},
                 sets={"lockbox_csv_sha256": "9" * 64, "halves_sha256": hv["sha256"]})
    info = GO.check_declaration(good, pred, hv, lock)
    assert info["checkpoint_sha256"] == sha and info["train_persons_sha256"] == good["train_persons_sha256"]
    other = json.loads(json.dumps(hv["sha256"]))
    other["dev"][GAIT] = "0" * 64
    for bad in (dict(sets={"lockbox_csv_sha256": "9" * 64, "halves_sha256": other}),
                dict(sets={"lockbox_csv_sha256": "8" * 64, "halves_sha256": hv["sha256"]}),
                dict(inference={"mode": "ema_s20_pool"}),
                dict(checkpoint={"file": str(ck), "sha256": "a" * 64}),
                dict(corpus={"sources": ["gait120", "shida_pd_gait"], "archives": []}),
                dict(corpus={"sources": ["gait120", "cody_3"], "archives": []}),
                dict(train_persons_sha256="f" * 64)):
        with pytest.raises(Refused):
            GO.check_declaration({**good, **bad}, pred, hv, lock)
    # a training person among the lockbox or the halves is refused before any read
    for p in (lock[STROKE][0], hv["halves"]["calib"][GAIT][0]):
        lists = {"gait120": ["gait120:T900", p]}
        with pytest.raises(Refused, match="training persons"):
            GO.check_declaration({**good, "train_persons": lists,
                                  "train_persons_sha256": PV.list_hashes(lists)["all"]}, pred, hv, lock)
    # the checkpoint's own lists must equal the declared ones
    pred2 = SimpleNamespace(sha256=sha, train_persons=frozenset(), loaded=SimpleNamespace(persons={"gait120": ["x"]}))
    with pytest.raises(Refused, match="checkpoint's training lists"):
        GO.check_declaration(good, pred2, hv, lock)


# ---------------------------------------------------------------- the lockbox permission
def test_no_lockbox_shard_set_or_record_without_a_logged_read(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "GATES_STORE", str(tmp_path / "gates").replace("\\", "/"))
    log, reg = str(tmp_path / "reads.csv"), _registry_copy(tmp_path)
    root = paths.gates_store_dir(GO.VERSION)
    with pytest.raises(Refused):
        GO.build_shard("lockbox", GAIT, [], root)
    with pytest.raises(Refused):
        GO.OpenG1Set("lockbox")
    made_up = {"read_at": "2026-11-14T00:00:00", "gate": "G1", "checkpoint_sha256": "a" * 64,
               "rule_sha256": OPEN_SHA}
    for row in (None, made_up, {**made_up, "gate": "G3"}):
        with pytest.raises(Refused):
            GO.OpenLockboxSet(root, row, {GAIT: [], STROKE: []}, registry_path=reg, log_path=log)
        with pytest.raises(Refused):
            GO.build_lockbox(row, root, {GAIT: [], STROKE: []}, registry_path=reg, log_path=log)
    # a row logged under the v1 rule opens nothing on the open path
    v1row = common.log_lockbox_read("G1", "a" * 64, V1_SHA, log_path=log, registry_path=reg)
    with pytest.raises(Refused, match="another rule"):
        GO.build_lockbox(v1row, root, {GAIT: [], STROKE: []}, registry_path=reg, log_path=log)
    # the raw reader honours its lockbox flag for mdfm.gates callers only
    from mdfm.data import open_mocap_raw as OR
    row = {"source": GAIT, "subject_ns": "gait120:L000", "person_id": "gait120:L000", "key": "x", "tier": "RELEASE",
           "lockbox": "True"}
    with pytest.raises(Refused, match="mdfm/gates"):
        OR.load_take(row, allow_lockbox=True)
    with pytest.raises(Refused, match="lockbox row"):
        OR.load_take(row)


def test_the_lockbox_landmarks_are_the_store_builder_s(monkeypatch):
    from mdfm.data import open_mocap_raw as OR
    rng = np.random.default_rng(0)
    fake = {"landmarks": rng.normal(size=(240, 15, 3)), "toes": rng.normal(size=(240, 2, 3)), "fps": 120.0,
            "up_axis": "+y", "labels": None}
    seen = []
    monkeypatch.setattr(OR, "load_take", lambda row, measure_up=True, allow_lockbox=False:
                        seen.append(allow_lockbox) or {k: (v.copy() if hasattr(v, "copy") else v) for k, v in fake.items()})
    B = GO.store_builder()
    a, b = GO.lockbox_landmarks({"source": GAIT}), B.landmarks({"source": GAIT})
    assert seen == [True, False]
    assert np.array_equal(a[0], b[0], equal_nan=True) and np.array_equal(a[1], b[1]) and a[2] == b[2]


def test_fit_person_runs_the_store_s_fit_and_caches_in_the_gates_store(tmp_path, monkeypatch):
    """The real stage 1 and stage 2 on synthetic landmarks (openfit's own forward model, a smooth walk of two takes):
    the fitted takes, their cache under the gates-only store, a second call served from the cache, another fit
    provenance fitted again, a cache outside the store refused."""
    from mdfm.body import openbody as BO
    from mdfm.body import openfit as OF
    from mdfm.model.prior import PhenotypePrior
    monkeypatch.setattr(paths, "GATES_STORE", str(tmp_path / "gates").replace("\\", "/"))
    torch.manual_seed(0)
    basis, prior = BO.load_basis(), PhenotypePrior()
    n = 120
    t = torch.arange(n, dtype=torch.float32)[:, None] / 100.0
    lm = {}
    for k in range(2):
        truth = OF.Fit(n, basis, prior, u0=torch.tensor([0.3, 0.2, -0.4, 0.1, 0.6, 0.0]))
        with torch.no_grad():
            base = 0.15 * torch.randn(22, 3)
            truth.aa.copy_(base[None] + 0.1 * torch.sin(2 * torch.pi * t)[..., None] * torch.ones(1, 22, 3))
            truth.aa[:, list(OF.UNOBSERVED)] = 0.0
            truth.aa[:, 0] = torch.tensor([0.0, 0.3 * k, 0.0])
            truth.tr.copy_(torch.cat([0.0 * t, 0.9 + 0.0 * t, 1.1 * t], 1))
            target, _, _ = truth.predict()
        lm[f"walk_{k}"] = (target.numpy().astype(np.float64), np.ones((n, OF.N_LM), bool),
                           {"fps_native": 100.0, "n_native": n, "up_axis": "+y", "pelvis_neck_m": 0.5,
                            "frame_cut": None, "absurd": 0, "far": 0})
    monkeypatch.setattr(GO, "lockbox_landmarks", lambda tk: lm[tk["key"]])
    takes = [{"source": STROKE, "subject_ns": "stroke_gait:stroke:L000", "person_id": "stroke_gait:stroke:L000",
              "key": k, "fps": "100.0", "n_frames": str(n)} for k in sorted(lm)]
    cache = os.path.join(paths.gates_store_dir(GO.VERSION), "lockbox", "_fits", STROKE, "L000")
    with pytest.raises(Refused, match="gates-only store"):
        GO.fit_person(STROKE, takes[0]["person_id"], takes, str(tmp_path / "outside"), "1" * 64)
    got = GO.fit_person(STROKE, takes[0]["person_id"], takes, cache, "1" * 64)
    assert got["stage1"] and not got["errors"] and [f["key"] for f in got["takes"]] == ["walk_0", "walk_1"]
    for f in got["takes"]:
        assert f["rot6d"].shape == (n, 22, 6) and f["transl"].shape == (n, 3) and f["rest"].shape == (22, 3)
        assert f["rms_m"] < 0.02 and f["take_id"] == f"{STROKE}/stroke_gait:stroke:L000/{f['key']}"
    assert sorted(os.listdir(cache)) == ["person.json", "walk_0.npz", "walk_1.npz"]
    monkeypatch.setattr(OF, "fit_take", lambda *a, **k: pytest.fail("refitted a cached take"))
    monkeypatch.setattr(OF, "fit_person", lambda *a, **k: pytest.fail("refitted a cached person"))
    again = GO.fit_person(STROKE, takes[0]["person_id"], takes, cache, "1" * 64)
    assert all(np.array_equal(a["rot6d"], b["rot6d"]) for a, b in zip(got["takes"], again["takes"]))
    with pytest.raises(BaseException, match="refitted"):
        GO.fit_person(STROKE, takes[0]["person_id"], takes, cache, "2" * 64)
    st = GO.FittedStore(GO.LOCKBOX, cache, got["takes"], 1.0)
    assert len(st) == 2 and np.allclose(st.rest_joints[0], got["takes"][0]["rest"])


# ---------------------------------------------------------------- a full read on fixtures
def _rest():
    from mdfm.body import openbody as BO
    ph = np.array([0.5, 0.75, 0.5, 0.5, 0.5, 0.5])
    with torch.no_grad():
        return BO.load_basis(torch.float64).rest(torch.as_tensor(ph)[None])[0].numpy(), ph


def _fitted(source, pid, k, rest, ph, n=160, rms=0.005):
    from mdfm.body.rotations import matrix_to_rot6d
    t = np.arange(n) / 100.0
    r6 = np.broadcast_to(matrix_to_rot6d(torch.eye(3, dtype=torch.float64)).numpy(), (n, 22, 6)).astype(np.float32)
    tr = np.stack([0.02 * np.sin(2 * np.pi * t), 0.95 + 0.01 * np.cos(4 * np.pi * t), 1.1 * t + 0.1 * k], 1)
    key = f"walk_{k}"
    return {"take_id": f"{source}/{pid}/{key}", "source": source, "subject_ns": pid, "person_id": pid, "key": key,
            "fps_native": 100.0, "n_native": n, "rot6d": r6.copy(), "transl": tr, "contact": np.zeros((n, 2), np.uint8),
            "treadmill": 0, "belt_x": 0.0, "belt_z": 0.0, "floor_m": 0.0, "rms_m": rms, "phenotype": ph, "rest": rest}


class _Model:
    """A predictor that predicts a fixed pose and a fixed wrist predictive (the G1 predictor contract)."""
    name = "fixture model"

    def __init__(self, sha, lists, pose, draws):
        self.sha256, self.pose, self.draws, self.c_wrist = sha, pose, draws, None
        self.loaded = SimpleNamespace(persons={s: list(v) for s, v in lists.items()})
        self.train_persons = frozenset(p for v in lists.values() for p in v)
        self.train_persons_sha256 = PV.list_hashes(lists)["all"]

    def predict(self, gset, i):
        return {"joints_six": self.pose, "joints_camera": self.pose, "wrist": self.draws, "wrist_six": None}

    @property
    def conformal(self):
        return {"wrist": float(self.c_wrist), "wrist_six": None}


@pytest.fixture
def world(tmp_path, monkeypatch):
    """Two lockbox gait120 persons and one stroke patient (the rule's counts patched down), halves of two and two
    persons per source with their dev and calib inputs built by the real build_shard on synthetic fits, a temporary
    registry, read log and gates-only store, a declaration naming all of it."""
    monkeypatch.setattr(paths, "GATES_STORE", str(tmp_path / "gates").replace("\\", "/"))
    reg, log = _registry_copy(tmp_path), str(tmp_path / "lockbox_reads.csv")
    rest, ph = _rest()
    lock = {GAIT: ["gait120:L000", "gait120:L001"], STROKE: ["stroke_gait:stroke:L000"]}
    p = tmp_path / "lockbox.csv"
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["source", "lane", "person_id", "group", "rule"])
        w.writerows([s, "body", x, "stroke" if s == STROKE else "all", "x"] for s, v in lock.items() for x in v)
    monkeypatch.setattr(GO, "LOCKBOX_CSV", str(p))
    monkeypatch.setattr(GO, "LOCKBOX_SHA256", common.sha256_file(str(p)))
    monkeypatch.setattr(GO, "LOCKBOX_COUNTS", {GAIT: 2, STROKE: 1})
    halves = {"dev": {GAIT: ["gait120:D000", "gait120:D001"], STROKE: ["stroke_gait:stroke:D000", "stroke_gait:stroke:D001"]},
              "calib": {GAIT: ["gait120:D002", "gait120:D003"], STROKE: ["stroke_gait:stroke:D002", "stroke_gait:stroke:D003"]}}
    hv = {"halves": halves, "sha256": {k: {s: S.person_hash(v) for s, v in d.items()} for k, d in halves.items()}}
    hpath = tmp_path / "halves.json"
    hpath.write_text(json.dumps(hv), encoding="utf-8")
    monkeypatch.setattr(GO, "check_halves", lambda h: h["halves"] == halves)
    deny = PV.DenyLists(frozenset(x for v in lock.values() for x in v), frozenset(), frozenset(), frozenset(),
                        frozenset(), "canary:none")
    monkeypatch.setattr(PV, "deny_lists", lambda: deny)
    inputs = str(tmp_path / "inputs")
    for split, d in halves.items():
        for s, ids in d.items():
            st = GO.FittedStore(split, "fixture", [_fitted(s, x, k, rest, ph) for k, x in enumerate(ids)], 1.0)
            GO.build_shard(split, s, ids, inputs, render=True, store=st)
        G.write_manifest(split, inputs, list(GO.SOURCES))
    ck = tmp_path / "ckpt_B_000001.pt"
    ck.write_bytes(b"a fixture checkpoint")
    sha = common.sha256_file(str(ck))
    lists = {GAIT: ["gait120:T900"], "cmu_mocap": ["cmu_mocap:T1"]}
    decl = _decl(checkpoint={"file": str(ck), "sha256": sha}, train_persons=lists,
                 train_persons_sha256=PV.list_hashes(lists)["all"],
                 sets={"lockbox_csv_sha256": GO.LOCKBOX_SHA256, "halves_sha256": hv["sha256"]})
    dpath = tmp_path / "declaration.json"
    dpath.write_text(json.dumps(decl), encoding="utf-8")
    pose = np.zeros((22, 3))
    pose[:, 1] = np.linspace(0.0, 0.6, 22)
    draws = np.random.default_rng(1).normal(scale=0.15, size=(64, 3))
    model = _Model(sha, lists, pose, draws)
    prior = G.PopulationPrior(pose + 0.01, draws * 1.2, None,
                              {"source": "open train body store", "train_persons": ["cmu_mocap:P1"],
                               "train_persons_sha256": S.person_hash(["cmu_mocap:P1"])})
    calls = []

    def takes(lk):
        assert common.read_log(log), "lockbox rows listed before the read was logged"
        calls.append("takes")
        return [{"source": s, "subject_ns": x, "person_id": x, "key": "walk_0", "tier": "RELEASE", "lockbox": "True",
                 "group": ""} for s in GO.SOURCES for x in lk[s]], {}

    def fit(source, pid, tk, cache_dir, fit_sha, log=None):
        assert common.read_log(log_p), "a lockbox person fitted before the read was logged"
        assert paths.in_gates_store(cache_dir)
        calls.append(("fit", pid, cache_dir))
        over = 0.5 if pid == "gait120:L001" else 0.005             # one person's only take over the bound
        return {"takes": [_fitted(source, pid, 0, rest, ph, rms=over)], "errors": [], "stage1": True}
    log_p = log
    monkeypatch.setattr(GO, "lockbox_takes", takes)
    monkeypatch.setattr(GO, "fit_person", fit)
    monkeypatch.setattr(GO, "residual_bound_m", lambda: 0.0167)
    return SimpleNamespace(reg=reg, log=log, decl=str(dpath), decl_obj=decl, halves=str(hpath), inputs=inputs,
                           model=model, prior=prior, calls=calls, lock=lock, sha=sha, tmp=tmp_path)


def _read(w, **kw):
    return GO.read_lockbox(w.decl, w.model, w.prior, reason=kw.pop("reason", "fixture read"), inputs_root=w.inputs,
                           halves_path=w.halves, registry_path=w.reg, log_path=w.log, **kw)


def test_a_fixture_read_logs_one_row_then_fits_and_scores_the_lockbox(world):
    r = _read(world)
    rows = common.read_log(world.log)
    assert len(rows) == 1
    row = rows[0]
    assert (row["gate"], row["checkpoint_sha256"], row["rule_sha256"]) == ("G1", world.sha, OPEN_SHA)
    assert row["inputs_sha256"] == r.plan_sha256 == common.sha256_json(r.plan)
    assert r.declaration_sha256 in row["note"] and row["reread"] == "0"
    assert row["code_sha256"] == G.code_sha256(GO.READ_CODE)
    # the plan logged before the read names the declaration, the rule, the sets, the halves and the conformal factors
    assert r.plan["declaration_sha256"] == common.sha256_file(world.decl)
    assert r.plan["rule"]["sha256"] == OPEN_SHA and r.plan["checkpoint_sha256"] == world.sha
    assert r.plan["lockbox"]["n"] == {GAIT: 2, STROKE: 1}
    assert r.plan["halves_sha256"] == world.decl_obj["sets"]["halves_sha256"]
    assert set(r.plan["conformal"]) == {"model", "prior"}
    # the fits and the shards live in the gates-only store, after the log
    assert world.calls[0] == "takes" and [c[1] for c in world.calls[1:]] == world.lock[GAIT] + world.lock[STROKE]
    assert all(paths.in_gates_store(c[2]) for c in world.calls[1:])
    assert paths.in_gates_store(r.root) and os.path.exists(os.path.join(r.root, "lockbox", "manifest.json"))
    # the residual bound refuses the over-bound take, its person is unscored and listed
    assert r.fits[GAIT]["over_bound"] == 1 and r.fits[GAIT]["unscored"] == ["gait120:L001"]
    assert sorted(r.table["person_id"].unique()) == ["gait120:L000", "stroke_gait:stroke:L000"]
    assert [x.line for x in r.results] == ["G1a six IMUs MPJPE", "G1b single wrist coverage",
                                          "G1c single camera PA-MPJPE"]
    assert all(x.rule_sha256 == OPEN_SHA and x.checkpoint_sha256 == world.sha for x in r.results)
    assert all(x.train_persons_sha256 == world.decl_obj["train_persons_sha256"] for x in r.results)
    assert sorted(r.per_source["archive"]) == [GAIT, STROKE]
    assert len(r.dev_results) == 3 and len(r.prior_results) == 3
    wc = r.extra["window_counts"]
    assert {k: {s: v["persons"] for s, v in d.items()} for k, d in wc.items()} == \
        {"dev": {GAIT: 2, STROKE: 2}, "calib": {GAIT: 2, STROKE: 2}, "lockbox": {GAIT: 1, STROKE: 1}}
    # a second read of the same checkpoint and rule is refused; a reread with its reason is logged as one
    with pytest.raises(Refused, match="second read"):
        _read(world)
    assert len(common.read_log(world.log)) == 1
    r2 = _read(world, reread=True, reason="fixture reread")
    rows = common.read_log(world.log)
    assert len(rows) == 2 and rows[1]["reread"] == "1" and rows[1]["reason"] == "fixture reread"
    assert [x.point for x in r2.results] == [x.point for x in r.results]


def test_refusals_before_the_log_leave_the_log_empty(world, tmp_path, monkeypatch):
    # a v1 declaration does not run on the open path
    d = dict(world.decl_obj, corpus={"sources": ["amass", "gait120"], "archives": []})
    p = tmp_path / "v1.json"
    p.write_text(json.dumps(d), encoding="utf-8")
    with pytest.raises(Refused, match="2026-11-12"):
        GO.read_lockbox(str(p), world.model, world.prior, reason="x", inputs_root=world.inputs,
                        halves_path=world.halves, registry_path=world.reg, log_path=world.log)
    # an unregistered open rule, a prior trained on a scored person, a lockbox root outside the store
    with pytest.raises(Refused):
        GO.read_lockbox(world.decl, world.model, world.prior, reason="x", inputs_root=world.inputs,
                        halves_path=world.halves, registry_path=_registry_copy(tmp_path, drop=("G1", "2")),
                        log_path=world.log)
    bad = G.PopulationPrior(world.prior.mean_pose, world.prior.draws, None,
                            {"source": "open train body store", "train_persons": ["gait120:D000"],
                             "train_persons_sha256": S.person_hash(["gait120:D000"])})
    with pytest.raises(Refused, match="prior"):
        _read(world.__class__(**{**world.__dict__, "prior": bad}))
    with pytest.raises(Refused, match="gates-only store"):
        _read(world, root=str(tmp_path / "elsewhere"))
    assert not os.path.exists(world.log) or common.read_log(world.log) == []
    assert world.calls == []


def _script():
    import importlib.util
    spec = importlib.util.spec_from_file_location("read_g1_open_lockbox",
                                                  os.path.join(paths.TREE, "scripts", "read_g1_open_lockbox.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_the_read_script_writes_its_outputs_once_and_refuses_before_loading(world, tmp_path, monkeypatch):
    R = _script()
    r = _read(world)
    base = str(tmp_path / "out")
    folder, n = R.out_dir(base, world.sha, OPEN_SHA, True, world.log)
    assert n == 1 and folder.replace("\\", "/").endswith(f"{world.sha}/reread_1")
    with pytest.raises(SystemExit, match="--reread"):
        R.out_dir(base, world.sha, OPEN_SHA, False, world.log)
    first, _ = R.out_dir(base, "e" * 64, OPEN_SHA, False, world.log)
    texts = R.render(r)
    written = R.write_outputs(first, texts, {"script": R.SCRIPT_REL})
    assert sorted(os.listdir(first)) == sorted(R.FILES + (R.PROVENANCE,))
    doc = json.load(open(os.path.join(first, "g1.json"), encoding="utf-8"))
    assert [x["line"] for x in doc["lines"]] == [x.line for x in r.results] and doc["plan_sha256"] == r.plan_sha256
    prov = json.load(open(os.path.join(first, R.PROVENANCE), encoding="utf-8"))
    assert prov["outputs"] == written == {k: common.sha256_file(os.path.join(first, k)) for k in R.FILES}
    with pytest.raises(FileExistsError):
        R.write_outputs(first, texts, {})
    with pytest.raises(SystemExit, match="exists"):
        R.out_dir(base, "e" * 64, OPEN_SHA, False, world.log)
    # a v1 declaration and an empty reason are refused before any checkpoint loads
    from mdfm.eval import infer
    monkeypatch.setattr(infer, "load_checkpoint", lambda *a, **k: pytest.fail("a checkpoint loaded"))
    d = dict(world.decl_obj, corpus={"sources": ["amass"], "archives": []})
    p = tmp_path / "v1.json"
    p.write_text(json.dumps(d), encoding="utf-8")
    assert R.main(["--declaration", str(p), "--reason", "x", "--out", base]) == 2
    assert R.main(["--declaration", world.decl, "--reason", " ", "--out", base]) == 2


# ---------------------------------------------------------------- v1 unchanged
def test_the_v1_path_is_unchanged():
    assert common.rel(G.RULE_PATH) == "docs/rules/G1_RULE.md" and G.PARAMS["min_window_s"] == 2.0
    assert G.VERSION == "g1_v1" and GO.VERSION == "g1_open_v1" and GO.MIN_WINDOW_S == 1.0
    assert GO.RULE_V1_PATH != G.RULE_PATH

    class P:
        name, sha256 = "p", "e" * 64
        train_persons = frozenset(["amass:person:KIT:1"])
        train_persons_sha256 = "0" * 64
    with pytest.raises(Refused):                    # the draft is not registered: the v1 lockbox stays shut
        G.run("lockbox", P())
    with pytest.raises(Refused):
        G.G1Set("lockbox", allow_lockbox=True)      # the flag is honoured for mdfm/gates callers only
