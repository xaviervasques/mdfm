"""Tests of task S2.3 of docs/MFM_STAGE2_CODE_SPEC_DRAFT.md (section 4.1): the name tables of a run
(mdfm/train/names.py, task S2.1) read from the run config by the existing modules: the optional names section of
mdfm/train/config.py, MFMConfig.names and the Batch's nuisance vocabularies of mdfm/model/mfm.py, the nuisance
vocabularies of mdfm/model/tokenize.py, the vocabulary path and task count of mdfm/data/windows.py, and the source
registry (specs) of mdfm/data/sources.py and mdfm/data/window_store.py.

Done-when of the build-order table: every stage-1 config keeps its SHA-256 (configs/stage1_pilot.yaml, _s2, _f2l2,
_jepa and _f2l2_jepa at their written lambda and at the smoke's weight 1.0, _f2l4, _200m; pinned here from the
outputs of the S23 fingerprint before the first edit), and a model built from STAGE1 has today's state-dict keys,
shapes and dtypes (the TINY and PILOT key lists pinned from the same fingerprint) and, under one seed, today's
initial values. The bit-for-bit tiny CPU runs of each config, before and after, are the fingerprint of the task
report (reports/weekend_2026-10-01/tasks/S23.md).

Also: the section is left out of the canonical JSON at its defaults, round trips and is checked; a model of grown
tables grows exactly the rows the tables name; the model refuses names that disagree with its sources, its mount
priors or the motion prior's 'unknown' row, and a batch tokenized under other nuisance vocabularies; descriptor codes
follow the named vocabularies ('none' and 'unknown' after the last name); task ids follow the named vocabulary file;
a store builds a source of given specs under a given vocabulary, and the default refusals stand.

Review of S23 (task FIXS23): problems() refused a names section away from its defaults until task S2.7 wired the
trainer (config.NAMES_WIRED, True since S2.7; the tests set it False to check the refusal); a store records the SHA-256 of a given vocabulary (source.json, store.json) and
WindowStore refuses a source built under another vocabulary than the reader's; a relative vocabulary path reads
under the tree; a task id past the version-1 count builds under its vocabulary; the folds of given specs.

    PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 <conda>/envs/mdfm/python.exe \
        -m pytest -q -p no:cacheprovider tests/test_names_wiring.py
"""
import csv
import hashlib
import json
import os
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch

from mdfm import paths
from mdfm.data import sources as S
from mdfm.data import window_store as WS
from mdfm.data import windows as W
from mdfm.model import mfm as M
from mdfm.model import prior as P
from mdfm.model import tokenize as TK
from mdfm.train import config as CF
from mdfm.train import names as N

import model_fixture as FX

TREE = paths.TREE
# config SHA-256 of the files, recorded by the S23 fingerprint before the first edit (2026-10-06)
CONFIG_SHA256 = {
    "stage1_pilot.yaml": "7f761b1ac425a2dfefc4e546908fa766e2545d915f9c2a88dcb8d199f140b2b5",
    "stage1_pilot_s2.yaml": "a81f22e914d39dfd9cacee422dcfd3cd2f05cf36d9df7741abf47e8500c7470c",
    "stage1_pilot_f2l2.yaml": "26503b1edfe1fe3f9f2161d0687ed482fc6d790fbe458a23af3d73a3f312ca6a",
    "stage1_pilot_jepa.yaml": "ca9d1dd4b8f53cdd216bfbd2d5f3dc5c53bbb530cbc7f902d244da219fc6e166",
    "stage1_pilot_f2l2_jepa.yaml": "a819c5ec821a34ad9f5919ecdfa475ccd63d8d6ea2ae73f9b60455aab3626b73",
    "stage1_pilot_f2l4.yaml": "f9296067811308de569d023fc5f572b0ce5056e6ede7e6ba56be9f27d0d4701c",
    "stage1_200m.yaml": "627d6e53b94eda199b932558167872da8e638f975696ff76a775eecee83f3efe",
}
# at the smoke's weight 1.0 (tests/test_e1_configs.py, tests/test_walkway_anchor.py)
WEIGHT1_SHA256 = {"stage1_pilot_jepa.yaml": "b87dbc00594a7f01755585fe4781e1d214c6c647a419e7e1f1269c670ef48de6",
                  "stage1_pilot_f2l2_jepa.yaml": "a819c5ec821a34ad9f5919ecdfa475ccd63d8d6ea2ae73f9b60455aab3626b73"}
# state-dict entries and SHA-256 (first 16 hex) of the 'key shape dtype' lines, before the first edit
STATE_TINY = (170, "91b2970fe69cd51d")
STATE_PILOT = (389, "be23d62b4e8b93a0")

V2 = replace(N.STAGE1, vocabulary_path=N.VOCABULARY_V2)
GROWN = replace(N.STAGE1, sources=N.STAGE1.sources + ("pads", "clinical_gait_multipatho"),
                vocabulary_path=N.VOCABULARY_V2, layouts=N.STAGE1.layouts + ("pads_wrists",),
                devices=N.STAGE1.devices + ("pads_watch",), cameras=N.STAGE1.cameras + ("modys_camera",))


def _load(name):
    return CF.load(os.path.join(TREE, "configs", name))


def _state_lines(model):
    sd = model.state_dict()
    blob = "\n".join(f"{k} {tuple(v.shape)} {v.dtype}" for k, v in sd.items())
    return len(sd), hashlib.sha256(blob.encode()).hexdigest()[:16]


def _no_skeleton(cfg, **kw):
    return replace(cfg, skeleton=False, **kw)


def _grown_model(seed=0):
    torch.manual_seed(seed)
    return M.MFM(_no_skeleton(M.TINY, sources=GROWN.sources, names=GROWN))


def _vocab_file(tmp_path, tasks, name="vocab.yaml"):
    p = tmp_path / name                         # one name per content: windows.task_vocabulary caches by path
    p.write_text("version: 9\nstatus: DRAFT\ntasks: [" + ", ".join(tasks) + "]\ndefault: unknown\nsources: {}\n",
                 encoding="utf-8", newline="\n")
    return str(p)


# ---------------------------------------------------------------- configs
@pytest.mark.parametrize("name", sorted(CONFIG_SHA256))
def test_every_stage1_config_keeps_its_sha256(name):
    cfg = _load(name)
    assert CF.sha256(cfg) == CONFIG_SHA256[name]
    if name in WEIGHT1_SHA256:
        assert CF.sha256(replace(cfg, jepa=replace(cfg.jepa, weight=1.0))) == WEIGHT1_SHA256[name]
    assert cfg.names == CF.NamesSection() and "names" not in CF.to_dict(cfg)
    assert CF.name_tables(cfg) is N.STAGE1
    assert CF.real_sources(cfg) == CF.REAL_SOURCES
    mc = CF.model_config(cfg)
    assert mc == cfg.model.mfm_config() and mc.names is None and mc.sources == M.SOURCES


def test_the_section_defaults_are_stage1():
    sec = CF.NamesSection()
    assert sec.tables() == N.STAGE1 and sec.tables().sha256() == N.STAGE1.sha256()
    assert CF.to_dict(sec) == N.STAGE1.to_dict()
    assert CF.names_problems(sec) == []


def test_the_section_is_left_out_at_its_defaults_and_round_trips():
    base = _load("stage1_pilot.yaml")
    d = CF.to_dict(base)
    for written in ({}, N.STAGE1.to_dict(), {"vocabulary_path": N.VOCABULARY_V1}):
        same = CF.from_dict(dict(d, names=written))
        assert same == base and CF.sha256(same) == CF.sha256(base)
    v2 = replace(base, names=CF.NamesSection(vocabulary_path=N.VOCABULARY_V2))
    dv = CF.to_dict(v2)
    assert dv["names"] == V2.to_dict() and {k: v for k, v in dv.items() if k != "names"} == d
    assert CF.from_json(CF.canonical_json(v2)) == v2 and CF.sha256(v2) != CF.sha256(base)
    assert CF.names_problems(v2.names) == [] and CF.problems(v2) == []          # wired since task S2.7
    with pytest.raises(ValueError, match="unknown keys"):
        CF.from_dict(dict(d, names={"vocabulary": N.VOCABULARY_V2}))
    with pytest.raises(ValueError):
        CF.from_dict(dict(d, names={"sources": ["synthetic", 3]}))


def test_a_yaml_file_with_a_names_section_loads_once_wired(tmp_path, monkeypatch):
    with open(os.path.join(TREE, "configs", "stage1_pilot.yaml"), encoding="utf-8") as f:
        text = f.read()
    p = tmp_path / "with_names.yaml"
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(text + f"\nnames:\n  vocabulary_path: {N.VOCABULARY_V2}\n")
    monkeypatch.setattr(CF, "NAMES_WIRED", False)                  # the refusal before task S2.7 wired the trainer
    with pytest.raises(ValueError, match=r"not wired into train/loop.py \(task S2.7\)"):
        CF.load(p)
    monkeypatch.setattr(CF, "NAMES_WIRED", True)
    cfg = CF.load(p)
    assert CF.name_tables(cfg) == V2
    assert CF.name_tables(cfg).tasks()[11:] == ("unknown", "rest_lying", "rest_sitting")
    mc = CF.model_config(cfg)
    assert mc.names == V2 and mc.sources == M.SOURCES
    assert replace(mc, names=None) == cfg.model.mfm_config()
    assert replace(cfg, names=CF.NamesSection()) == _load("stage1_pilot.yaml")


@pytest.mark.parametrize("bad, match", [
    ({"sources": ()}, "empty"),
    ({"sources": ("weargait_pd", "synthetic")}, "synthetic"),
    ({"layouts": ("xsens13", "xsens13")}, "duplicate"),
    ({"mount_families": ("xsens13",)}, "generic"),
    ({"vocabulary_path": "docs/rules/no_such_vocabulary.yaml"}, "FileNotFoundError"),
])
def test_the_names_section_is_checked(bad, match):
    cfg = replace(_load("stage1_pilot.yaml"), names=CF.NamesSection(**bad))
    probs = CF.names_problems(cfg.names)
    assert any(p.startswith("names") and match in p for p in probs), probs
    assert all(p in CF.problems(cfg) for p in probs) and CF.NAMES_NOT_WIRED not in CF.problems(cfg)
    with pytest.raises(ValueError):
        CF.check(cfg)


def test_a_vocabulary_that_moves_unknown_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(CF, "NAMES_WIRED", True)
    tasks = list(N.STAGE1.tasks())
    moved = tasks[:-1] + ["rest_lying", "unknown"]
    cfg = replace(_load("stage1_pilot.yaml"), names=CF.NamesSection(vocabulary_path=_vocab_file(tmp_path, moved)))
    assert any("unknown" in p for p in CF.problems(cfg))
    kept = tasks + ["rest_lying"]
    ok = replace(cfg, names=CF.NamesSection(vocabulary_path=_vocab_file(tmp_path, kept, "kept.yaml")))
    assert not CF.problems(ok)


def test_mixture_sources_come_from_the_names_section(monkeypatch):
    base = _load("stage1_pilot.yaml")
    more = replace(base, mixture=replace(base.mixture, sources=CF.REAL_SOURCES + ("pads",)))
    assert any("sources outside" in p for p in CF.problems(more))
    named = replace(more, names=CF.NamesSection(sources=N.STAGE1.sources + ("pads",)))
    assert not CF.problems(named)                                     # wired since task S2.7
    monkeypatch.setattr(CF, "NAMES_WIRED", False)
    assert CF.problems(named) == [CF.NAMES_NOT_WIRED]
    monkeypatch.setattr(CF, "NAMES_WIRED", True)
    assert not CF.problems(named)
    assert CF.real_sources(named) == CF.REAL_SOURCES + ("pads",)
    assert CF.model_config(named).sources == M.SOURCES + ("pads",)


def test_a_non_default_names_section_is_refused_until_the_trainer_reads_it(monkeypatch):
    """Review of S23: train/loop.py builds MFM(cfg.model.mfm_config()) and collate codes the stage-1 vocabularies,
    so a names section that only changes the vocabulary (no batch refusal at MFM._check) would train the stage-1
    tables under another config hash. problems() refuses every section away from the defaults until task S2.7
    wires the trainer and sets NAMES_WIRED."""
    with open(os.path.join(TREE, "mdfm", "train", "loop.py"), encoding="utf-8") as f:
        loop = f.read()
    assert CF.NAMES_WIRED == ("model_config(" in loop), \
        "config.NAMES_WIRED is set with the trainer's wiring (train/loop.py reading model_config, task S2.7)"
    monkeypatch.setattr(CF, "NAMES_WIRED", False)
    base = _load("stage1_pilot.yaml")
    for nm in (CF.NamesSection(vocabulary_path=N.VOCABULARY_V2),
               CF.NamesSection(vocabulary_path=N.VOCABULARY_V2, layouts=GROWN.layouts, devices=GROWN.devices,
                               cameras=GROWN.cameras),
               CF.NamesSection(sources=N.STAGE1.sources + ("pads",))):
        cfg = replace(base, names=nm)
        assert CF.names_problems(nm) == [] and CF.problems(cfg) == [CF.NAMES_NOT_WIRED]
        with pytest.raises(ValueError, match="not wired"):
            CF.check(cfg)
    same = CF.from_dict(dict(CF.to_dict(base), names=N.STAGE1.to_dict()))      # the defaults written in full
    assert CF.problems(same) == [] and CF.check(same) == base
    monkeypatch.setattr(CF, "NAMES_WIRED", True)
    assert CF.problems(replace(base, names=CF.NamesSection(vocabulary_path=N.VOCABULARY_V2))) == []


# ---------------------------------------------------------------- the model
def test_a_model_built_from_stage1_has_todays_state_dict():
    for cfg, pinned in ((M.TINY, STATE_TINY), (M.PILOT, STATE_PILOT)):
        torch.manual_seed(0)
        today = M.MFM(_no_skeleton(cfg))
        torch.manual_seed(0)
        named = M.MFM(_no_skeleton(cfg, names=N.STAGE1))
        assert _state_lines(today) == _state_lines(named) == pinned
        a, b = today.state_dict(), named.state_dict()
        assert list(a) == list(b) and all(torch.equal(a[k], b[k]) for k in a)
        assert named.names is N.STAGE1 and today.names is None
        assert named.nuis_fields == today.nuis_fields == TK.NUIS_FIELDS
        assert tuple(named.heads.families) == N.STAGE1.mount_families


def test_a_model_of_grown_tables_grows_the_named_rows_only():
    torch.manual_seed(0)
    base = M.MFM(_no_skeleton(M.TINY))
    grown = _grown_model()
    a, b = base.state_dict(), grown.state_dict()
    assert list(a) == list(b)
    changed = {k: (tuple(a[k].shape), tuple(b[k].shape)) for k in a if a[k].shape != b[k].shape}
    f = list(TK.FIELD_NAMES)
    d = M.TINY.d
    assert changed == {
        "prior.task.weight": ((12, M.TINY.prior_d), (14, M.TINY.prior_d)),
        f"tokenizer.nuisance.tables.{f.index('layout')}.weight": ((13, d), (14, d)),
        f"tokenizer.nuisance.tables.{f.index('device')}.weight": ((4, d), (5, d)),
        f"tokenizer.nuisance.tables.{f.index('camera')}.weight": ((3, d), (4, d)),
        "source_offsets.offsets": ((7, 15, 3), (9, 15, 3)),
    }
    assert grown.nuis_fields["layout"][-1] == "pads_wrists" and grown.prior.n_tasks == 14


def test_the_model_refuses_names_that_disagree(tmp_path):
    with pytest.raises(ValueError, match="sources"):
        M.MFM(_no_skeleton(M.TINY, names=GROWN))                      # cfg.sources left at stage 1
    with pytest.raises(ValueError, match="mount"):
        fam = replace(N.STAGE1, mount_families=N.STAGE1.mount_families + ("pads_wrists",))
        M.MFM(_no_skeleton(M.TINY, names=fam))
    moved = list(N.STAGE1.tasks())[:-1] + ["rest_lying", "unknown"]
    with pytest.raises(ValueError, match="unknown"):
        M.MFM(_no_skeleton(M.TINY, names=replace(N.STAGE1, vocabulary_path=_vocab_file(tmp_path, moved))))
    assert P.TASKS.index("unknown") == P.TASK_UNKNOWN == V2.tasks().index("unknown")


def test_a_batch_carries_its_nuisance_vocabularies():
    model = _grown_model()
    wins = [FX.make_window(1.5, seed=0), FX.make_pose_window(1.0, seed=5, person="p2")]
    old = M.batch_from_windows(wins, sources=GROWN.sources, points_per_stream=32, gen=0)
    assert old.nuis_fields is None
    with pytest.raises(ValueError, match="nuisance"):
        model.encode(old)
    new = M.batch_from_windows(wins, sources=GROWN.sources, points_per_stream=32, gen=0, nuis_fields=model.nuis_fields)
    post = model.encode(new)
    assert torch.isfinite(post.mu_z).all()
    col = list(TK.FIELD_NAMES).index("camera")
    imu = new.tokens.nuis_desc[:, list(TK.FIELD_NAMES).index("kind")] == 0
    assert (new.tokens.nuis_desc[imu, col] == len(GROWN.cameras)).all()        # 'none' after the last camera
    assert (old.tokens.nuis_desc[imu, col] == len(N.STAGE1.cameras)).all()
    other = [c for c in range(len(TK.FIELD_NAMES)) if c != col]       # xsens13 keeps its layout and device codes
    assert torch.equal(new.tokens.nuis_desc[imu][:, other], old.tokens.nuis_desc[imu][:, other])
    torch.manual_seed(0)
    base = M.MFM(_no_skeleton(M.TINY))
    with pytest.raises(ValueError, match="nuisance"):
        base.encode(M.batch_from_windows(wins, points_per_stream=32, gen=0, nuis_fields=model.nuis_fields))
    same = M.batch_from_windows(wins, points_per_stream=32, gen=0, nuis_fields=N.STAGE1.nuis_fields())
    plain = M.batch_from_windows(wins, points_per_stream=32, gen=0)
    assert torch.equal(same.tokens.nuis_desc, plain.tokens.nuis_desc)
    base.encode(same)


# ---------------------------------------------------------------- tokenizer
def test_nuisance_fields_are_checked():
    assert TK.nuisance_fields(None) is TK.NUIS_FIELDS and TK.nuisance_fields(TK.NUIS_FIELDS) is TK.NUIS_FIELDS
    assert TK.nuisance_fields(GROWN.nuis_fields()) == GROWN.nuis_fields()
    good = dict(TK.NUIS_FIELDS)
    reordered = dict(reversed(list(good.items())))
    for bad in (reordered, dict(good, kind=("imu_site", "pose3d", "pose2d")), dict(good, layout=("a", "a")),
                dict(good, camera=()), {k: v for k, v in good.items() if k != "site"},
                dict(good, rate=good["rate"] + ("200_up",))):
        with pytest.raises(ValueError):
            TK.nuisance_fields(bad)
    assert set(TK.NAMED_FIELDS) == set(N.NUIS_TABLES)


def test_descriptor_codes_follow_the_named_vocabularies():
    s = FX.imu_stream(1.0, layout="pads_wrists")
    lay = list(TK.FIELD_NAMES).index("layout")
    assert TK.nuisance_descriptors(s)[0, lay] == len(N.STAGE1.layouts) + 1                 # unknown under stage 1
    assert TK.nuisance_descriptors(s, GROWN.nuis_fields())[0, lay] == GROWN.layouts.index("pads_wrists")
    x = FX.imu_stream(1.0)
    assert np.array_equal(TK.nuisance_descriptors(x), TK.nuisance_descriptors(x, N.STAGE1.nuis_fields()))
    g = TK.nuisance_descriptors(x, GROWN.nuis_fields())
    o = TK.nuisance_descriptors(x)
    assert g[0, lay] == o[0, lay] == 0                                                     # xsens13 keeps its code
    none = replace(x, meta=dict(x.meta, layout=None))
    assert TK.nuisance_descriptors(none, GROWN.nuis_fields())[0, lay] == len(GROWN.layouts)
    torch.manual_seed(0)
    a = TK.Tokenizer(16, 32)
    torch.manual_seed(0)
    b = TK.Tokenizer(16, 32, nuis_fields=N.STAGE1.nuis_fields())
    sa, sb = a.state_dict(), b.state_dict()
    assert list(sa) == list(sb) and all(torch.equal(sa[k], sb[k]) for k in sa)
    c = TK.Tokenizer(16, 32, nuis_fields=GROWN.nuis_fields())
    assert c.nuisance.tables[lay].num_embeddings == len(GROWN.layouts) + 2


# ---------------------------------------------------------------- windows
def test_task_ids_follow_the_named_vocabulary():
    v2 = N.resolve(N.VOCABULARY_V2)
    assert W.task_name("imu_inthewild_iprognosis", "sess3") == "unknown"
    assert W.task_id("imu_inthewild_iprognosis", "sess3") == P.TASK_UNKNOWN
    assert W.task_name("imu_inthewild_iprognosis", "sess3", path=v2) == V2.task_name("imu_inthewild_iprognosis",
                                                                                      "sess3") == "free_living"
    assert W.task_id("imu_inthewild_iprognosis", "sess3", path=v2) == V2.task_id("imu_inthewild_iprognosis", "sess3")
    for source, key in (("amass", "CMU/01/01_01_poses.npz"), ("weargait_pd", "SelfPace"), ("gait120", "x")):
        assert W.task_id(source, key) == W.task_id(source, key, path=W.VOCABULARY) == N.STAGE1.task_id(source, key)
    rec = {"key": "sess3", "subject_ns": "x"}
    assert W.covariates(rec, "imu_inthewild_iprognosis")["task"] == P.TASK_UNKNOWN
    assert W.covariates(rec, "imu_inthewild_iprognosis", vocabulary=v2)["task"] == V2.tasks().index("free_living")


def test_validate_reads_the_task_count():
    w = FX.make_pose_window(1.0, seed=5)
    W.validate(w)
    w12 = replace(w, cov=dict(w.cov, task=12))
    with pytest.raises(ValueError, match="cov values"):
        W.validate(w12)
    assert W.validate(w12, n_tasks=len(V2.tasks())) is w12
    with pytest.raises(ValueError, match="cov values"):
        W.validate(replace(w, cov=dict(w.cov, task=14)), n_tasks=14)


# ---------------------------------------------------------------- sources and the window store
@pytest.fixture
def no_tables(monkeypatch):
    """Fake records read no demographic table."""
    monkeypatch.setattr(S, "demographic", lambda source, subject_ns: {})


SPEC = replace(S.SOURCES["stroke_gait_vancriekinge"], name="imu_inthewild_iprognosis", groups=None)
SPECS = {"imu_inthewild_iprognosis": SPEC}
PERSON = "imu_inthewild_iprognosis:S901"         # a fake id: absent from the index and the lockbox


def test_specs_replace_the_pilot_registry():
    assert S._names(None) == list(S.PILOT)
    with pytest.raises(KeyError, match="not a pilot source"):
        S._names(["imu_inthewild_iprognosis"])
    assert S._names(None, SPECS) == ["imu_inthewild_iprognosis"]
    assert S._names("imu_inthewild_iprognosis", SPECS) == ["imu_inthewild_iprognosis"]
    with pytest.raises(KeyError, match="not a source of the given specs"):
        S._names(["gait120"], SPECS)


def _fake_paired(key="sess3", seconds=3.0):
    rng = np.random.default_rng(7)
    n = int(seconds * 100)
    pose = {"coords": rng.normal(scale=0.5, size=(n, 15, 3)).astype(np.float32), "joint_mask": np.ones(15, bool),
            "coord_mask": np.ones(3, bool), "valid_mask": np.ones((n, 15), bool), "scale": 0.5, "dim": 3,
            "modality": "3d_meter", "fps": 100.0, "source": "imu_inthewild_iprognosis", "subject_ns": PERSON,
            "task": key}
    return S.PairedRecord(source="imu_inthewild_iprognosis", person_id=PERSON, subject_ns=PERSON, key=key,
                          streams={"pose3d": pose}, t={"pose3d": np.arange(n) / 100.0}, roles={"pose3d": "input"},
                          meta={"pose3d": {"layout": "mocap3d", "unit_to_si": 1.0}})


def _tasks(root):
    with open(os.path.join(root, "imu_inthewild_iprognosis", "records.csv"), encoding="utf-8", newline="") as f:
        return [int(r["task"]) for r in csv.DictReader(f)]


def test_a_store_builds_a_source_of_given_specs_under_a_given_vocabulary(tmp_path, no_tables, monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("the index was read")
    monkeypatch.setattr(S, "rows", refuse)
    with pytest.raises(KeyError, match="not a pilot source"):
        WS.build_source("imu_inthewild_iprognosis", out=str(tmp_path / "x"), paired=[_fake_paired()],
                        folds={PERSON: "train"})
    v1 = WS.build_source("imu_inthewild_iprognosis", out=str(tmp_path / "v1"), paired=[_fake_paired()],
                         folds={PERSON: "train"}, specs=SPECS)
    v2 = WS.build_source("imu_inthewild_iprognosis", out=str(tmp_path / "v2"), paired=[_fake_paired()],
                         folds={PERSON: "train"}, specs=SPECS, vocabulary=N.resolve(N.VOCABULARY_V2))
    assert v1["records"] == v2["records"] == 1
    assert _tasks(tmp_path / "v1") == [P.TASK_UNKNOWN]
    assert _tasks(tmp_path / "v2") == [V2.tasks().index("free_living")]
    with pytest.raises(KeyError, match="not a source of the given specs"):
        WS.build(sources=["gait120"], out=str(tmp_path / "y"), specs=SPECS)
    assert not os.path.exists(tmp_path / "x") and not os.path.exists(tmp_path / "y")


# ---------------------------------------------------------------- review of S23: the store's vocabulary and folds
def _refuse_index(*a, **k):
    raise AssertionError("the index was read")


def _column(root, name):
    with open(os.path.join(root, "imu_inthewild_iprognosis", "records.csv"), encoding="utf-8", newline="") as f:
        return [r[name] for r in csv.DictReader(f)]


def _source_json(root):
    with open(os.path.join(root, "imu_inthewild_iprognosis", "source.json"), encoding="utf-8") as f:
        return json.load(f)


def _fake_rows(seen):
    """A fake index of ten fake imu_inthewild_iprognosis persons (PERSON first) and two fake gait120 persons; the
    stage-1 call (specs None) is refused."""
    persons = [PERSON] + [f"imu_inthewild_iprognosis:S9{i:02d}" for i in range(2, 11)]

    def rows(sources=None, specs=None):
        seen.append(specs)
        if specs is None:
            raise AssertionError("the stage-1 index was read")
        return pd.DataFrame({"source": ["imu_inthewild_iprognosis"] * len(persons) + ["gait120"] * 2,
                             "person_id": persons + ["gait120:G901", "gait120:G902"]})
    return rows, persons


def test_the_store_records_its_vocabulary_and_the_reader_checks_it(tmp_path, no_tables, monkeypatch):
    monkeypatch.setattr(S, "rows", _refuse_index)
    monkeypatch.chdir(tmp_path)                    # a relative vocabulary path is read under the tree, not the cwd
    assert WS.vocabulary_file(None) == WS.vocabulary_file(N.VOCABULARY_V1) == W.VOCABULARY
    v1, v2 = str(tmp_path / "v1"), str(tmp_path / "v2")
    a = WS.build_source("imu_inthewild_iprognosis", out=v1, paired=[_fake_paired()], folds={PERSON: "train"},
                        specs=SPECS)
    b = WS.build_source("imu_inthewild_iprognosis", out=v2, paired=[_fake_paired()], folds={PERSON: "train"},
                        specs=SPECS, vocabulary=N.VOCABULARY_V2)
    assert "task_vocabulary" not in a and "task_vocabulary" not in _source_json(v1)
    assert b["task_vocabulary"] == _source_json(v2)["task_vocabulary"] == {
        "sha256": V2.vocabulary_sha256(), "version": 2, "tasks": len(V2.tasks())}
    assert _tasks(v2) == [V2.tasks().index("free_living")] and _tasks(v1) == [P.TASK_UNKNOWN]
    s1, s2 = WS.write_store_json(v1), WS.write_store_json(v2)
    assert "task_vocabulary_sha256" not in s1["sources"]["imu_inthewild_iprognosis"]
    assert s2["sources"]["imu_inthewild_iprognosis"]["task_vocabulary_sha256"] == V2.vocabulary_sha256()
    assert WS.WindowStore(v1).vocabulary_sha256 == N.VOCABULARY_V1_SHA256 == N.STAGE1.vocabulary_sha256()
    with pytest.raises(ValueError, match="task vocabulary"):
        WS.WindowStore(v2)                                             # a stage-1 reader
    assert WS.WindowStore(v2, vocabulary_sha256=V2.vocabulary_sha256()).vocabulary_sha256 == V2.vocabulary_sha256()
    with pytest.raises(ValueError, match="task vocabulary"):
        WS.WindowStore(v1, vocabulary_sha256=V2.vocabulary_sha256())  # built with specs and no vocabulary


def test_a_task_id_past_the_version1_count_builds_under_its_vocabulary(tmp_path, no_tables, monkeypatch):
    monkeypatch.setattr(S, "rows", _refuse_index)
    p = tmp_path / "rest_vocab.yaml"                  # the version-2 tasks, the fake 'sess' keys mapped to rest_lying
    p.write_text("version: 9\nstatus: DRAFT\ntasks: [" + ", ".join(V2.tasks()) + "]\ndefault: unknown\n"
                 "sources:\n  imu_inthewild_iprognosis:\n    - {task: rest_lying, match: '^sess'}\n",
                 encoding="utf-8", newline="\n")
    rest = V2.tasks().index("rest_lying")
    assert rest >= len(N.STAGE1.tasks())
    w = W.cut(_fake_paired(), vocabulary=str(p))[0]
    assert w.cov["task"] == rest
    with pytest.raises(ValueError, match="cov values"):
        W.validate(w)                                                  # the version-1 task count
    WS.build_source("imu_inthewild_iprognosis", out=str(tmp_path / "rest"), paired=[_fake_paired()],
                    folds={PERSON: "train"}, specs=SPECS, vocabulary=str(p))
    assert _tasks(tmp_path / "rest") == [rest]


def test_folds_draw_the_sources_of_given_specs(monkeypatch):
    seen = []
    rows, persons = _fake_rows(seen)
    monkeypatch.setattr(S, "rows", rows)
    assert SPEC.fold_pool is None
    f = S.folds(specs=SPECS)
    assert seen == [SPECS]
    assert set(f["source"]) == {"imu_inthewild_iprognosis"} and sorted(f["person_id"]) == sorted(persons)
    n_hold = S._round_half_up(SPEC.fold_share * len(persons))
    assert n_hold == 2 and (f["fold"] == SPEC.fold).sum() == n_hold
    assert (f["fold"] == "train").sum() == len(persons) - n_hold
    assert f.attrs["sha256"] == S.fold_hashes(f)
    assert S.folds(specs=SPECS).equals(f)                              # seeded per source


def test_a_store_draws_the_folds_of_given_specs(tmp_path, no_tables, monkeypatch):
    seen = []
    rows, _ = _fake_rows(seen)
    monkeypatch.setattr(S, "rows", rows)
    f = S.folds(specs=SPECS)
    fold = dict(zip(f["person_id"], f["fold"]))[PERSON]
    WS.build_source("imu_inthewild_iprognosis", out=str(tmp_path / "a"), paired=[_fake_paired()], specs=SPECS)
    assert _column(tmp_path / "a", "fold") == [fold]

    def paired(source, **kw):
        assert source == "imu_inthewild_iprognosis" and kw["specs"] is SPECS
        return iter([_fake_paired()])
    monkeypatch.setattr(S, "paired", paired)
    out = WS.build(out=str(tmp_path / "b"), specs=SPECS, vocabulary=N.VOCABULARY_V2)
    assert list(out) == ["imu_inthewild_iprognosis"] and None not in seen
    assert _column(tmp_path / "b", "fold") == [fold] and _tasks(tmp_path / "b") == [V2.tasks().index("free_living")]
    with open(tmp_path / "b" / "store.json", encoding="utf-8") as fh:
        doc = json.load(fh)
    assert doc["sources"]["imu_inthewild_iprognosis"]["task_vocabulary_sha256"] == V2.vocabulary_sha256()
