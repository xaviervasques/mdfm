"""scripts/probe_stage2_sources.py (weekend task T19): the lockbox is refused before any file opens, the
proposed weargait_pd PD and stroke dev folds are deterministic and disjoint from the lockbox, and the weargait
site table counts held, missing and never-held sites per recording and per person.

The synthetic tests read no file of MoveAtlas or of this tree (the gaitpdb test writes its own zip in a pytest
temporary folder). The tests marked needs_index read the record index (identifier columns
only: source, modality, subject_ns, key, person_id, group, tier, lockbox, container) with every container
open refused; no record of any person is read. The tests marked needs_report read the CSV and JSON outputs
of the 2026-10-01 run in reports/stage2_sources/.
"""
import builtins
import contextlib
import importlib.util
import io
import json
import os
import zipfile

import numpy as np
import pandas as pd
import pytest

from mdfm import paths
from mdfm.data import canonical as C
from mdfm.data import sources as S

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load():
    spec = importlib.util.spec_from_file_location("probe_stage2_sources_t19_test",
                                                  os.path.join(TREE, "scripts", "probe_stage2_sources.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


P = _load()

HAVE_INDEX = (os.path.exists(C.index_path()) and os.path.exists(C.manifest_path())
              and os.path.isdir(f"{paths.MOVEATLAS}/datasets"))
needs_index = pytest.mark.skipif(not HAVE_INDEX, reason="record index or MoveAtlas not found")
REPORT = os.path.join(TREE, "reports", "stage2_sources")
needs_report = pytest.mark.skipif(not os.path.exists(os.path.join(REPORT, "summary.json")),
                                  reason="reports/stage2_sources not written")

# configs/canonical_imu.yaml of MoveAtlas, locations (the 14-location taxonomy; P.locations() reads it)
SITES = ["head", "trunk", "lower_back", "wrist_L", "wrist_R", "thigh_L", "thigh_R", "shank_L", "shank_R",
         "ankle_L", "ankle_R", "foot_L", "foot_R", "pocket"]
ALL13 = SITES[:-1]
ID_COLS = ["source", "manifest_source", "modality", "subject_ns", "key", "person_id", "group", "tier", "lane",
           "lockbox", "container", "record", "cody"]


@contextlib.contextmanager
def no_file_opens(monkeypatch):
    """Inside the block any container or file open raises AssertionError (the index is already in memory)."""
    def refuse(*a, **k):
        raise AssertionError(f"a file was opened: {a[:1]}")
    with monkeypatch.context() as m:
        m.setattr(zipfile, "ZipFile", refuse)
        m.setattr(builtins, "open", refuse)
        m.setattr(io, "open", refuse)
        yield


def _fake_index(rows):
    """An index() frame of identifier columns built from dicts (the columns canonical.load and the script use)."""
    base = {"manifest_source": "", "lane": "body", "tier": "RELEASE", "lockbox": False, "cody": False,
            "record": 0, "n_frames": 100, "fps": 100.0, "rec_n_frames": -1, "rule": "", "status": "ok"}
    df = pd.DataFrame([dict(base, **r) for r in rows])
    df["manifest_source"] = [m or s for m, s in zip(df["manifest_source"], df["source"])]
    return df


def _install(monkeypatch, df):
    monkeypatch.setattr(C, "_INDEX", df)
    monkeypatch.setattr(C, "_POS", None)
    monkeypatch.setattr(C, "_CODY_NS", frozenset())


# ---------------------------------------------------------------- lockbox refused before any file opens
def _wear_rows():
    return [
        {"source": "weargait_pd", "modality": "imu", "subject_ns": "weargait:WPD900", "person_id": "weargait:WPD900",
         "key": "SelfPace", "group": "PD", "lockbox": True, "container": "datasets/imu/weargait_pd/lock.npz"},
        {"source": "weargait_pd", "modality": "imu", "subject_ns": "weargait:WPD901", "person_id": "weargait:WPD901",
         "key": "SelfPace", "group": "PD", "container": "datasets/imu/weargait_pd/open.npz"},
        {"source": "weargait_pd", "modality": "force", "subject_ns": "weargait:WPD900",
         "person_id": "weargait:WPD900", "key": "SelfPace", "group": "PD", "lockbox": True,
         "container": "datasets/imu/weargait_pd/lock_force.npz"},
    ]


def test_reader_check_refuses_a_lockbox_row_without_opening(monkeypatch):
    df = _fake_index(_wear_rows())
    _install(monkeypatch, df)
    with no_file_opens(monkeypatch):
        out = P.reader_checks(df, "weargait_pd", [])
    assert out == {"lockbox_rows_tried": 1, "lockbox_rows_refused": 1, "load_rows_compared": 0,
                   "load_rows_equal": 0}
    # the index decides: the caller's copy of the row saying lockbox False changes nothing
    row = df[df["lockbox"]].iloc[0].to_dict()
    with no_file_opens(monkeypatch):
        with pytest.raises(C.Refused):
            C.load(dict(row, lockbox=False))


def test_selection_and_reader_never_reach_a_lockbox_container(monkeypatch):
    df = _fake_index(_wear_rows() + [
        {"source": "weargait_pd", "modality": "imu", "subject_ns": "weargait:WPD902", "person_id": "weargait:WPD902",
         "key": "SelfPace", "group": "PD", "container": ""},                                   # unresolved
        {"source": "pads", "modality": "imu", "subject_ns": "pads:001", "person_id": "pads:001", "key": "t1",
         "group": "all", "container": "datasets/imu/pads/other.npz"},                          # another source
    ])
    _install(monkeypatch, df)
    sel = P.selection(df, "weargait_pd")
    assert list(sel["person_id"]) == ["weargait:WPD901"]
    assert sel[["container", "record"]].values.tolist() == \
        C._select(sources="weargait_pd", lane="body")[["container", "record"]].values.tolist()
    opened = []

    def record_open(path, *a, **k):
        opened.append(str(path).replace("\\", "/").rsplit("/", 1)[-1])
        raise FileNotFoundError(path)

    for quick in (False, True):
        opened.clear()
        st = {"load_errors": 0}
        with monkeypatch.context() as m:
            m.setattr(zipfile, "ZipFile", record_open)
            got = list(P.read_source(df, "weargait_pd", quick, st))
        assert got == []
        # only the allowed record's container was asked for (the open itself fails here: no file exists)
        assert opened and set(opened) == {"open.npz"}, (quick, opened)
        assert not any(name.startswith("lock") for name in opened)


def test_a_refusal_stops_the_reader_instead_of_counting_as_a_read_error(monkeypatch):
    # canonical.Refused is a PermissionError, hence an OSError: the reader re-raises it, never reads around it
    df = _fake_index(_wear_rows() + [
        {"source": "weargait_pd", "modality": "imu", "subject_ns": "weargait:WPD903", "person_id": "weargait:WPD903",
         "key": "SelfPace", "group": "PD", "tier": "EVAL", "container": "datasets/imu/weargait_pd/eval.npz"}])
    _install(monkeypatch, df)
    st = {"load_errors": 0}
    with no_file_opens(monkeypatch):
        with pytest.raises(C.Refused):
            list(P.read_source(df, "weargait_pd", False, st))
    assert st == {"load_errors": 0}


def test_a_refusal_in_quick_mode_or_in_the_row_by_row_fallback_stops_the_reader(monkeypatch):
    # the two other paths of read_source that call canonical.load: --quick, and the fallback after an iterate
    # failure that is not a refusal; a Refused raised by canonical.load must propagate on both
    df = _fake_index(_wear_rows())
    _install(monkeypatch, df)

    def refuse(row, *a, **k):
        raise C.Refused(f"refused: {row['person_id']}")

    def iterate_fails(*a, **k):
        raise FileNotFoundError("container missing")    # an OSError that is not a refusal

    monkeypatch.setattr(C, "load", refuse)
    monkeypatch.setattr(C, "iterate", iterate_fails)
    for quick in (True, False):
        st = {"load_errors": 0}
        with no_file_opens(monkeypatch):
            with pytest.raises(C.Refused):
                list(P.read_source(df, "weargait_pd", quick, st))
        assert st["load_errors"] == 0 and "load_error_last" not in st, quick
        assert ("iterate_stopped" in st) == (not quick)   # the full run reached the fallback before the refusal


def test_selection_drops_eval_excluded_cody_other_lanes_and_duplicates():
    # the rows canonical._select drops with its flags at False. canonical._select itself is not called here:
    # naming a source that holds an EVAL or CODY row refuses at the call, so the rule is checked on selection()
    w = {"source": "weargait_pd", "modality": "imu", "key": "SelfPace", "group": "PD"}
    df = _fake_index(_wear_rows() + [
        dict(w, subject_ns="weargait:WPD904", person_id="weargait:WPD904", tier="EVAL", container="e.npz"),
        dict(w, subject_ns="weargait:WPD905", person_id="weargait:WPD905", tier="EXCLUDED_V1", container="x.npz"),
        dict(w, subject_ns="weargait:WPD906", person_id="weargait:WPD906", cody=True, container="c.npz"),
        dict(w, subject_ns="weargait:WPD907", person_id="weargait:WPD907", lane="none", container="n.npz"),
        dict(w, source="weargait_alias", manifest_source="weargait_pd", subject_ns="weargait:WPD908",
             person_id="weargait:WPD908", container="m.npz"),            # named through manifest_source: kept
        dict(w, subject_ns="weargait:WPD901", person_id="weargait:WPD901", key="Again",
             container="datasets/imu/weargait_pd/open.npz"),               # same container and record: read once
    ])
    sel = P.selection(df, "weargait_pd")
    assert sorted(sel["person_id"]) == ["weargait:WPD901", "weargait:WPD908"]
    assert list(sel.loc[sel["person_id"] == "weargait:WPD901", "key"]) == ["SelfPace"]


def test_compare_helpers():
    # NaN equals NaN once serialized; a list is one value; time stamps, durations and the folder are dropped
    assert P._jkey(float("nan")) == P._jkey(float("nan"))
    assert P._flat({"a": {"b": 1, "c": [1, 2]}, "d": {}}) == {"a.b": 1, "a.c": [1, 2], "d": {}}
    old = "[2026-10-01T10:36:19] start, BelowNormal priority set: True, quick False, out D:\\x\\stage2_sources"
    new = "[2026-10-06T02:53:11] start, BelowNormal priority set: True, quick False, out D:/x/rerun"
    assert P._log_norm(old) == P._log_norm(new)
    assert P._log_norm("[t] pads: {'records': 8250} in 18 s; checks") == P._log_norm("[u] pads: {'records': 8250} in 13 s; checks")
    assert P._log_norm("[t] pads: {'records': 8250} in 18 s") != P._log_norm("[t] pads: {'records': 8251} in 18 s")
    assert P.OUTPUTS == sorted(P.OUTPUTS) and len(P.OUTPUTS) == 18 and "_provenance.json" in P.OUTPUTS
    # paths in compare.json: tree-relative, or absolute on another drive (no ValueError from relpath)
    assert P._rel(os.path.join(P.TREE, "reports", "stage2_sources")) == "reports/stage2_sources"
    if os.name == "nt":
        other = "D:" if P.TREE.upper().startswith("C") else "C:"
        assert P._rel(other + "\\scratch\\c1") == other + "/scratch/c1"
    # summary.md blocks: prose echoed, table rows counted only (an old run's rows may hold lockbox aggregates)
    assert P._md_text(["Runtime: 1 s.", "| a | b |", "|---|---|", "| 15 | 2 |"]) == \
        "Runtime: 1 s. / (3 table rows, not echoed)"
    assert P._md_text(["| pads | 15 |"]) == "(1 table rows, not echoed)"
    assert P._md_text(["a", "b"]) == "a / b" and P._md_text([]) == ""
    assert P._md_text(["", "| k | v |", ""]) == "(1 table rows, not echoed)"     # blank lines are not prose


@needs_index
def test_every_stage2_source_refuses_its_lockbox_rows_before_any_open(monkeypatch):
    idx = C.index()
    ids = idx[ID_COLS]            # identifier columns only; canonical.load finds the full row itself
    for source in P.SOURCES:
        sel = P.selection(idx, source)
        assert not sel["lockbox"].any() and not sel["cody"].any()
        assert not sel["tier"].isin(["EVAL", "EXCLUDED_V1"]).any()
        assert sel[["container", "record"]].values.tolist() == \
            C._select(sources=source, lane="body")[["container", "record"]].values.tolist()
        with no_file_opens(monkeypatch):
            out = P.reader_checks(ids, source, [])
        assert out["lockbox_rows_tried"] == 1 and out["lockbox_rows_refused"] == 1, source


# ---------------------------------------------------------------- proposed dev folds
def _fold_index():
    rows = []
    for i in range(10):                                  # the pool: RELEASE, non-lockbox, with an IMU record
        p = f"weargait:WPD{i:03d}"
        rows.append({"source": "weargait_pd", "modality": "imu", "subject_ns": p, "person_id": p, "key": "k",
                     "group": "PD", "container": "c.npz"})
    for i in range(3):                                   # lockbox persons
        p = f"weargait:WPD{100 + i:03d}"
        rows.append({"source": "weargait_pd", "modality": "imu", "subject_ns": p, "person_id": p, "key": "k",
                     "group": "PD", "lockbox": True, "container": "c.npz"})
    rows += [
        {"source": "weargait_pd", "modality": "force", "subject_ns": "weargait:WPD200",
         "person_id": "weargait:WPD200", "key": "k", "group": "PD", "container": "c.npz"},     # no IMU row
        {"source": "weargait_pd", "modality": "imu", "subject_ns": "weargait:WPD201",
         "person_id": "weargait:WPD201", "key": "k", "group": "PD", "container": ""},          # unresolved
        {"source": "weargait_pd", "modality": "imu", "subject_ns": "weargait:WPD202",
         "person_id": "weargait:WPD202", "key": "k", "group": "PD", "tier": "INTERNAL", "container": "c.npz"},
        {"source": "weargait_pd", "modality": "imu", "subject_ns": "weargait:HC001",
         "person_id": "weargait:HC001", "key": "k", "group": "control", "container": "c.npz"},  # a control
    ]
    for i in range(13):                                  # 13 stroke persons: 20 percent is 2.6
        p = f"stroke:S{i + 3}"
        rows.append({"source": "stroke_gait_vancriekinge", "modality": "pose", "subject_ns": p, "person_id": p,
                     "key": "k", "group": "stroke", "container": "s.npz"})
    rows.append({"source": "stroke_gait_vancriekinge", "modality": "pose", "subject_ns": "stroke:S99",
                 "person_id": "stroke:S99", "key": "k", "group": "stroke", "lockbox": True, "container": "s.npz"})
    return _fake_index(rows)


def test_proposed_folds_draw_like_the_stage1_folds():
    idx = _fold_index()
    folds, info = P.proposed_folds(idx)
    # 20 percent of 13 is 2.6: the stroke pool tells rounding (3) from truncation (int() gives 2)
    assert int(P.FOLD_SHARE * 13) == 2
    for (source, group, _), pool, n_dev in ((P.FOLDS[0], [f"weargait:WPD{i:03d}" for i in range(10)], 2),
                                            (P.FOLDS[1], sorted(f"stroke:S{i + 3}" for i in range(13)), 3)):
        f = folds[(folds["source"] == source) & (folds["group"] == group)]
        assert list(f["person_id"]) == sorted(pool)
        dev = sorted(f.loc[f["fold"] == "dev", "person_id"])
        assert len(dev) == n_dev == S._round_half_up(P.FOLD_SHARE * len(pool))
        # mdfm.data.sources.folds: rng seeded by SHA-256 of "<seed>:<key>", first n of the permutation
        perm = np.random.default_rng(S._seed(S.SEED, f"{source}:{group}")).permutation(len(pool))
        assert dev == sorted(sorted(pool)[i] for i in perm[:n_dev])
        v = info[f"{source}:{group}"]
        assert (v["persons"], v["dev"], v["train"], v["dev_in_lockbox"]) == (len(pool), n_dev, len(pool) - n_dev, 0)
        assert v["sha256_dev"] == S.person_hash(dev)
        assert v["sha256_pool"] == S.person_hash(pool)
        assert v["sha256_train"] == S.person_hash(sorted(set(pool) - set(dev)))


def test_proposed_folds_deterministic_and_lockbox_free():
    idx = _fold_index()
    a, ia = P.proposed_folds(idx)
    b, ib = P.proposed_folds(idx.sample(frac=1.0, random_state=7).reset_index(drop=True))
    pd.testing.assert_frame_equal(a, b)
    assert ia == ib
    lock = set(idx.loc[idx["lockbox"], "person_id"])
    assert lock and not lock & set(a["person_id"])
    # no RELEASE-only filter slip: INTERNAL, unresolved, force-only and control persons stay out
    assert not {"weargait:WPD200", "weargait:WPD201", "weargait:WPD202", "weargait:HC001"} & set(a["person_id"])


@needs_index
def test_real_proposed_folds_deterministic_and_disjoint_from_the_lockbox():
    idx = C.index()[ID_COLS]
    a, ia = P.proposed_folds(idx)
    b, ib = P.proposed_folds(idx.iloc[::-1].reset_index(drop=True))
    pd.testing.assert_frame_equal(a, b)
    assert ia == ib
    for source, group, modality in P.FOLDS:
        f = a[(a["source"] == source) & (a["group"] == group)]
        lock = set(idx.loc[(idx["source"] == source) & idx["lockbox"], "person_id"])
        assert lock and not lock & set(f["person_id"])
        v = ia[f"{source}:{group}"]
        assert v["dev_in_lockbox"] == 0 and v["dev"] == S._round_half_up(P.FOLD_SHARE * v["persons"])
        assert set(f["person_id"]) <= set(idx.loc[(idx["source"] == source) & (idx["group"] == group)
                                                  & (idx["modality"] == modality), "person_id"])


@needs_index
@needs_report
def test_real_proposed_folds_match_the_hashes_of_the_report():
    with open(os.path.join(REPORT, "summary.json"), encoding="utf-8") as fh:
        rep = json.load(fh)["folds"]
    _, info = P.proposed_folds(C.index()[ID_COLS])
    for key, v in info.items():
        for k in ("persons", "dev", "train", "sha256_pool", "sha256_dev", "sha256_train", "dev_in_lockbox"):
            assert v[k] == rep[key][k], (key, k)
    saved = pd.read_csv(os.path.join(REPORT, "folds_proposed.csv"), dtype=str, keep_default_na=False,
                        encoding="utf-8")
    for key, v in rep.items():
        source, group = key.split(":", 1)
        s = saved[(saved["source"] == source) & (saved["group"] == group)]
        assert S.person_hash(s.loc[s["fold"] == "dev", "person_id"]) == v["sha256_dev"]


# ---------------------------------------------------------------- weargait site table
def _site_rows():
    rows = []

    def rec(pid, group, key, valid):
        for site, share in valid.items():
            rows.append({"source": "weargait_pd", "group": group, "person_id": pid, "subject_ns": pid, "key": key,
                         "site": site, "valid_share": share})

    full = {s: 1.0 for s in ALL13}
    rec("weargait:WPD001", "PD", "SelfPace", full)
    rec("weargait:WPD001", "PD", "HurriedPace", full)
    # thigh_L listed below the held share, thigh_R exactly at it; the second recording lacks lower_back
    rec("weargait:WPD002", "PD", "SelfPace", dict(full, thigh_L=0.49, thigh_R=0.5))
    rec("weargait:WPD002", "PD", "HurriedPace", {s: 1.0 for s in ALL13 if s != "lower_back"})
    six = {s: 1.0 for s in ("head", "trunk", "lower_back", "wrist_L", "wrist_R", "foot_L")}
    rec("weargait:HC001", "control", "SelfPace", six)
    rec("weargait:HC001", "control", "HurriedPace", six)
    rows.append({"source": "pads", "group": "all", "person_id": "pads:001", "subject_ns": "pads:001",
                 "key": "t1", "site": "wrist_L", "valid_share": 0.0})                  # another source: ignored
    return pd.DataFrame(rows)


def _site_index():
    return _fake_index([
        {"source": "weargait_pd", "modality": "force", "subject_ns": "weargait:WPD001",
         "person_id": "weargait:WPD001", "key": "SelfPace", "group": "PD", "container": "f.npz"},
        {"source": "weargait_pd", "modality": "force", "subject_ns": "weargait:WPD002",
         "person_id": "weargait:WPD002", "key": "SelfPace", "group": "PD", "lockbox": True,
         "container": "f.npz"},                                     # a lockbox force row never counts
    ])


def test_weargait_site_table(monkeypatch):
    monkeypatch.setattr(S, "folds", lambda *a, **k: pd.DataFrame(
        {"source": ["weargait_pd"], "person_id": ["weargait:HC001"], "fold": ["dev"]}))
    prop = pd.DataFrame({"source": "weargait_pd", "group": "PD", "person_id": ["weargait:WPD001", "weargait:WPD002"],
                         "fold": ["dev", "train"]})
    R, Pt = P.weargait_sites(_site_rows(), None, _site_index(), SITES, prop)
    assert len(R) == 6 and set(R["person_id"]) == {"weargait:WPD001", "weargait:WPD002", "weargait:HC001"}
    r = R.set_index(["person_id", "key"])
    assert r.loc[("weargait:WPD002", "SelfPace"), "n_sites_listed"] == 13
    assert r.loc[("weargait:WPD002", "SelfPace"), "n_sites_held"] == 12
    assert r.loc[("weargait:WPD002", "SelfPace"), "sites_listed_not_held"] == "thigh_L"
    assert r.loc[("weargait:WPD002", "SelfPace"), "min_site_valid_share"] == 0.49
    assert r.loc[("weargait:WPD002", "HurriedPace"), "sites_missing"] == "lower_back"
    assert "pocket" not in r.loc[("weargait:HC001", "SelfPace"), "sites_missing"]
    assert bool(r.loc[("weargait:WPD001", "SelfPace"), "insole_record"])
    assert not r.loc[("weargait:WPD001", "HurriedPace"), "insole_record"]
    assert not r.loc[("weargait:WPD002", "SelfPace"), "insole_record"]
    p = Pt.set_index("person_id")
    assert p.loc["weargait:WPD001", "recordings_all_13_held"] == 2
    assert p.loc["weargait:WPD001", "n_sites_held_in_every_recording"] == 13
    assert (p.loc["weargait:WPD001", "proposed_fold"], p.loc["weargait:WPD001", "stage1_fold"]) == ("dev", "")
    assert p.loc["weargait:WPD001", "insole_recordings"] == 1
    assert p.loc["weargait:WPD002", "n_sites_held_in_every_recording"] == 11
    assert p.loc["weargait:WPD002", "sites_never_held"] == ""
    assert p.loc["weargait:WPD002", "sites_held_in_some_not_all"] == "lower_back 1|thigh_L 1"
    assert p.loc["weargait:WPD002", "layouts"] == 2
    assert p.loc["weargait:HC001", "sites_never_held"] == "thigh_L|thigh_R|shank_L|shank_R|ankle_L|ankle_R|foot_R"
    assert (p.loc["weargait:HC001", "stage1_fold"], p.loc["weargait:HC001", "proposed_fold"]) == ("dev", "")
    w = P.weargait_summary(R, Pt, SITES)
    assert (w["persons"], w["recordings"], w["held_share_threshold"]) == (3, 6, 0.5)
    assert w["PD"] == {"persons": 2, "persons_all_13_held_in_every_recording": 1, "persons_with_a_site_never_held": 0,
                       "recordings": 4, "recordings_all_13_held": 2}
    assert w["control"]["persons_with_a_site_never_held"] == 1
    assert w["site_never_held_persons"]["foot_R"] == 1 and w["site_never_held_persons"]["lower_back"] == 0
    assert sum(x["recordings"] for x in w["layouts"]) == 6


def test_imu_facts_site_shares_and_up_axis():
    T = 200
    x = np.zeros((T, 2, 9), np.float32)
    x[:, 0, 2] = 1.0                                     # head: +1 g along z at rest
    x[:, 1, 1] = -1.0                                    # pocket: +1 g along -y
    cm = np.zeros((2, 9), bool)
    cm[0, :] = True
    cm[1, :3] = True
    sv = np.ones((T, 2), bool)
    sv[: T // 2, 1] = False
    x[: T // 2, 1, :3] = np.nan                         # invalid samples may hold NaN
    rec = {"signal": x, "channel_mask": cm, "location_ids": np.array([0, 13]), "sample_valid": sv}
    f, per_site = P.imu_facts(rec, SITES)
    by = {d["site"]: d for d in per_site}
    assert by["head"]["valid_share"] == 1.0 and by["pocket"]["valid_share"] == 0.5
    assert (by["head"]["up"], by["pocket"]["up"]) == ("+z", "-y")
    assert (by["head"]["channels"], by["pocket"]["channels"]) == ("acc+gyro+mag", "acc")
    assert f["acc_class"] == "with_gravity" and f["sites"] == "head|pocket"
    assert f["nonfinite"] == 3 * T // 2 and f["nonfinite_where_valid"] == 0
    assert json.loads(f["sig"])["acc_content"].startswith("absent")
    # linear acceleration without gravity: no up axis, class gravity_removed
    lin = np.zeros((T, 1, 9), np.float32)
    lin[:, 0, 0] = np.sin(np.linspace(0, 20 * np.pi, T))
    f2, ps2 = P.imu_facts({"signal": lin, "channel_mask": np.r_[[True] * 6, [False] * 3][None],
                           "location_ids": np.array([3]), "sample_valid": None, "acc_content": "linear_no_gravity"},
                          SITES)
    assert f2["acc_class"] == "gravity_removed" and ps2[0]["up"] == "" and f2["acc_content"] == "linear_no_gravity"


def test_gaitpdb_sex_code_decoded_from_training_persons_only(monkeypatch, tmp_path):
    # a person outside the training portion whose row would make code 1 ambiguous must not enter the decoding
    base = "gait-in-parkinsons-disease-1.0.0/"
    txt = "ID\tStudy\tGroup\tGender\tAge\tHeight\tWeight\nGaCo01\tGa\t2\t1\t60\t1.70\t70\n" \
          "GaPt02\tGa\t1\t2\t65\t1.60\t60\nGaPt09\tGa\t1\t1\t70\t1.65\t65\n"
    html = ("<tr><td>GaCo01</td><td>a</td><td>b</td><td>c</td><td>male</td></tr>"
            "<tr><td>GaPt02</td><td>a</td><td>b</td><td>c</td><td>female</td></tr>"
            "<tr><td>GaPt09</td><td>a</td><td>b</td><td>c</td><td>female</td></tr>")
    raw = tmp_path / "datasets" / "imu" / "physionet_gaitpdb" / "raw"
    raw.mkdir(parents=True)
    with zipfile.ZipFile(raw / "gaitpdb.zip", "w") as z:
        z.writestr(base + "demographics.txt", txt)
        z.writestr(base + "demographics.html", html)
    monkeypatch.setattr(paths, "MOVEATLAS", str(tmp_path).replace("\\", "/"))
    out, note = P.demo_gaitpdb(["gaitpdb:GaCo01", "gaitpdb:GaPt02"])
    assert sorted(out) == ["gaitpdb:GaCo01", "gaitpdb:GaPt02"]
    assert (out["gaitpdb:GaCo01"]["sex"], out["gaitpdb:GaPt02"]["sex"]) == ("M", "F")
    assert "'1->M': 1" in note and "'2->F': 1" in note and "1->F" not in note


def _facts(rows):
    cols = ["source", "group", "modality", "subject_ns", "person_id", "key", "diag_field", "diag_value"]
    return pd.DataFrame([dict({"modality": "imu", "key": "k", "diag_field": "", "diag_value": ""}, **r)
                         for r in rows]).reindex(columns=cols).fillna("")


def test_diagnosis_counts_the_pads_source_table_over_training_persons_only(monkeypatch, tmp_path):
    # review fix R1: pads_subject_table.csv lists every pads person; persons outside the training portion (here
    # pads:900 with a condition of its own and pads:901 with PD) must not enter source_table_training_persons
    def pads(pid, cond, n=1):
        return [{"source": "pads", "group": "all", "subject_ns": pid, "person_id": pid, "key": f"t{i}",
                 "diag_field": "condition", "diag_value": cond} for i in range(n)]
    F = _facts(pads("pads:001", "PD", 2) + pads("pads:002", "Essential Tremor") + pads("pads:003", "PD") + [
        {"source": "imu_inthewild_iprognosis", "group": "all", "subject_ns": "ipr:1", "person_id": "ipr:1"},
        {"source": "modys_video", "group": "all", "modality": "pose", "subject_ns": "modys:1", "person_id": "modys:1"},
        {"source": "gaitrec", "group": "HC", "subject_ns": "gaitrec:1", "person_id": "gaitrec:1",
         "diag_field": "label", "diag_value": "HC"},
        {"source": "stroke_gait_vancriekinge", "group": "stroke", "modality": "pose", "subject_ns": "stroke:S3",
         "person_id": "stroke:S3", "diag_field": "group", "diag_value": "stroke"},
        {"source": "clinical_gait_multipatho", "group": "HS", "subject_ns": "cgm:HS_1", "person_id": "cgm:HS_1"}])
    ma = tmp_path / "datasets"
    (ma / "imu" / "pads" / "processed").mkdir(parents=True)
    pd.DataFrame({"subject_ns": ["pads:001", "pads:002", "pads:003", "pads:900", "pads:901"],
                  "condition": ["PD", "Essential Tremor", "PD", "Multiple Sclerosis", "PD"]}).to_csv(
        ma / "imu" / "pads" / "processed" / "pads_subject_table.csv", index=False, lineterminator="\n")
    (ma / "imu" / "imu_inthewild_iprognosis" / "raw").mkdir(parents=True)
    with zipfile.ZipFile(ma / "imu" / "imu_inthewild_iprognosis" / "raw"
                         / "i-PROGNOSIS_DS2.10_Sub1_IMUdataPhonecalls.zip", "w") as z:
        z.writestr("dataset_description.txt", "Data of 3 PD patients and\n2 Healthy controls.")
    (ma / "pose" / "modys_video").mkdir(parents=True)
    (ma / "pose" / "modys_video" / "datacard.md").write_text("# modys\n- population: children with dyskinetic CP\n",
                                                             encoding="utf-8")
    monkeypatch.setattr(paths, "MOVEATLAS", str(tmp_path).replace("\\", "/"))
    D = P.diagnosis(F, None, {})
    p = D[D["source"] == "pads"]
    assert dict(zip(p["value"], p["training_persons"])) == {"Essential Tremor": 1, "PD": 2}
    assert dict(zip(p["value"], p["training_records"])) == {"Essential Tremor": 1, "PD": 3}
    assert list(p["source_table_training_persons"]) == list(p["training_persons"])
    assert "Multiple Sclerosis" not in " ".join(map(str, D.to_numpy().ravel()))
    assert all("rows of the training persons only (3)" in n for n in p["note"])
    assert "source_table_persons" not in D.columns and "source_table_training_persons" in D.columns
    i = D[D["source"] == "imu_inthewild_iprognosis"].iloc[0]
    assert "3 PD and 2 healthy controls" in i["note"] and i["training_persons"] == 1


def test_demo_gaitndd_parses_training_rows_only(monkeypatch, tmp_path):
    # review fix R1 for gaitndd: a row outside the training portion is never parsed into a record. The output
    # is keyed by the training subjects either way, so the test watches the row dicts the function builds
    raw = tmp_path / "datasets" / "imu" / "gaitndd" / "raw"
    raw.mkdir(parents=True)
    txt = ("ID\tGROUP\tAGE(YRS)\tHEIGHT(meters)\tWeight(kg)\tgender\n"
           "park1\tpark\t77\t1.86\t111\tm\nals1\tals\t52\t1.80\t70\tf\nhunt9\thunt\t60\t1.70\t80\tf\n")
    with zipfile.ZipFile(raw / "gaitndd.zip", "w") as z:
        z.writestr("gait-in-neurodegenerative-disease-database-1.0.0/subject-description.txt", txt)
    monkeypatch.setattr(paths, "MOVEATLAS", str(tmp_path).replace("\\", "/"))
    built = []

    def spy(*a, **k):
        d = builtins.dict(*a, **k)
        built.append(d)
        return d

    monkeypatch.setattr(P, "dict", spy, raising=False)    # module globals come before builtins in name lookup
    out, note = P.demo_gaitndd(["gaitndd:park1", "gaitndd:als1"])
    assert sorted(out) == ["gaitndd:als1", "gaitndd:park1"]
    assert (out["gaitndd:park1"]["sex"], out["gaitndd:park1"]["age"], out["gaitndd:als1"]["sex"]) == ("M", 77.0, "F")
    assert out["gaitndd:als1"]["height_m"] == 1.8 and out["gaitndd:park1"]["mass_kg"] == 111.0
    ids = sorted(d.get("ID") for d in built if "ID" in d)
    assert ids == ["als1", "park1"]                       # hunt9 was never parsed
    assert note.endswith("subject-description.txt")


@needs_index
@needs_report
def test_report_site_table_holds_only_non_lockbox_persons():
    idx = C.index()[ID_COLS]
    w = idx[(idx["source"] == "weargait_pd") & (idx["modality"] == "imu")]
    lock = set(w.loc[w["lockbox"], "person_id"])
    want = set(w.loc[~w["lockbox"] & (w["container"] != ""), "person_id"])
    t = pd.read_csv(os.path.join(REPORT, "weargait_sites.csv"), dtype=str, keep_default_na=False, encoding="utf-8")
    assert set(t["person_id"]) == want and not lock & set(t["person_id"])
    assert set(t.loc[t["proposed_fold"] != "", "group"]) == {"PD"}
