"""Tests of scripts/study_s1.py on a synthetic modys_video tree (no MoveAtlas file is read).

The synthetic tree holds 10 children, 7 of them lockbox. Lockbox members of the zips hold bytes that are not a
DeepLabCut CSV, and the item cells of lockbox rows on the sheet hold a sentinel string. The tests check that no
lockbox member is read, that no item cell of a lockbox row is decoded, that the header is refused before any item
cell when a region column differs, that the lockbox assertion fires, and the probe and posture rules of S1 section 2.
"""
import csv
import hashlib
import importlib.util
import io
import math
import os
import sys
import zipfile

import numpy as np
import pytest

openpyxl = pytest.importorskip("openpyxl")

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _script():
    spec = importlib.util.spec_from_file_location("study_s1_t", os.path.join(TREE, "scripts", "study_s1.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S = _script()
F = S.F
LOCK = [3, 4, 5, 6, 7, 8, 9]
TRAIN = [1, 2, 10]
# video -> (child, time, scorers)
VIDEOS = {}
_v = 1
for c in range(1, 11):
    for t in ("T0", "T12"):
        VIDEOS[_v] = (c, t, (1, 2) if t == "T0" else (1, 3))
        _v += 1


def dlc_csv(T=60, seed=0):
    rng = np.random.default_rng(seed)
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["scorer"] + ["DLC_model"] * 3 * len(F.POINTS))
    w.writerow(["bodyparts"] + [p for p in F.POINTS for _ in range(3)])
    w.writerow(["coords"] + ["x", "y", "likelihood"] * len(F.POINTS))
    for i in range(T):
        row = [i]
        for p in F.POINTS:
            row += [f"{100 + rng.normal():.3f}", f"{100 + rng.normal():.3f}", "0.99"]
        w.writerow(row)
    return buf.getvalue().encode("utf-8")


def build(tmp_path, rename=None):
    here, root = tmp_path / "mdfm", tmp_path / "moveatlas"
    raw = root / "datasets" / "pose" / "modys_video" / "raw"
    for d in (here / "reports" / "lockbox", here / "reports" / "modys_subjects", here / "reports" / "nuisance", raw):
        d.mkdir(parents=True)
    with open(here / "reports" / "lockbox" / "lockbox_persons.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["source", "lane", "person_id", "group", "rule"])
        w.writerow(["other", "body", "other:p1", "all", "x"])
        for c in LOCK:
            w.writerow(["modys_video", "body", f"modys:child{c:02d}", "all", "x"])
    with open(here / "reports" / "modys_subjects" / "modys_video_person_map.csv", "w", encoding="utf-8",
              newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["subject_ns", "take", "video", "ID", "time", "person_ns", "n_frames"])
        for v, (c, t, _) in VIDEOS.items():
            w.writerow([f"modys:{v:03d}", "x", v, c, t, f"modys:child{c:02d}", 60])
    (here / "reports" / "nuisance" / "frame_rates_facts.json").write_text('{"modys_source_level_fps": 25.0}',
                                                                         encoding="utf-8")
    wb = openpyxl.Workbook()
    ws = wb.active
    cols = ["ID", "video", "time", "scorer"]
    for n in S.ITEM_COLUMNS:
        cols += [n, n + "_max"]
    if rename:
        cols = [rename.get(c, c) for c in cols]
    ws.append(cols)
    for v, (c, t, scs) in VIDEOS.items():
        for s in scs:
            vals = [c, v, t, s]
            for n in S.ITEM_COLUMNS:
                vals += ["LOCKBOX" if c in LOCK else (v + s) % 5, 4]
            ws.append(vals)
    wb.save(raw / S.SHEET)
    for z in S.ZIPS:
        with zipfile.ZipFile(raw / z, "w") as zf:
            for v, (c, _, _) in VIDEOS.items():
                zf.writestr(f"{z[:-4]}/{v:03d}_DLC_model.csv", b"not a csv" if c in LOCK else dlc_csv(seed=v))
    return str(here), str(root)


@pytest.fixture
def tree(tmp_path, monkeypatch):
    here, root = build(tmp_path)
    with open(os.path.join(here, S.LOCKBOX_CSV), "rb") as f:
        monkeypatch.setattr(S, "LOCKBOX_SHA256", hashlib.sha256(f.read()).hexdigest())
    return here, root


def test_identity_excludes_the_lockbox_children(tree):
    here, root = tree
    ident = S.identity(root, here)
    assert ident["children"] == [f"modys:child{c:02d}" for c in TRAIN]
    assert {ident["child_of"][v] for v in ident["train_videos"]} == set(ident["children"])
    assert len(ident["lock_videos"]) == 14


def test_no_lockbox_member_is_read(tree, monkeypatch):
    here, root = tree
    read = []
    orig = zipfile.ZipFile.read

    def spy(self, name, pwd=None):
        read.append(name)
        return orig(self, name, pwd)
    monkeypatch.setattr(zipfile.ZipFile, "read", spy)
    ident = S.identity(root, here)
    read.clear()
    recs = S.load_records(root, ident, log=lambda m: None)
    assert len(recs) == 2 * 2 * len(TRAIN)
    assert all(int(os.path.basename(n)[:3]) in ident["train_videos"] for n in read)


def test_no_item_cell_of_a_lockbox_row_is_decoded(tree, monkeypatch):
    here, root = tree
    ident = S.identity(root, here)
    decoded = []
    orig = S._cell_value

    def spy(c, ss):
        decoded.append(c.get("r"))
        return orig(c, ss)
    monkeypatch.setattr(S, "_cell_value", spy)
    lab = S.load_labels(root, ident)
    lock_rows, r = set(), 2
    for v, (c, t, scs) in VIDEOS.items():
        for _ in scs:
            if c in LOCK:
                lock_rows.add(r)
            r += 1
    import re
    item_refs = [x for x in decoded if int(re.sub(r"[A-Z]+", "", x)) > 1
                 and re.match(r"[A-Z]+", x).group(0) not in ("A", "B", "C", "D")]
    assert item_refs and not {int(re.sub(r"[A-Z]+", "", x)) for x in item_refs} & lock_rows
    assert {x["child"] for x in lab} == set(ident["children"])
    assert len(lab) == 2 * 2 * len(TRAIN)


def test_header_with_another_region_column_is_refused_before_any_item(tmp_path, monkeypatch):
    here, root = build(tmp_path, rename={"D_TRUNK_R_tD": "D_NECK_R_tD"})
    with open(os.path.join(here, S.LOCKBOX_CSV), "rb") as f:
        monkeypatch.setattr(S, "LOCKBOX_SHA256", hashlib.sha256(f.read()).hexdigest())
    with pytest.raises(AssertionError, match="region columns differ"):
        S.identity(root, here)


def test_lockbox_guard_fires():
    g = S.LockboxGuard(frozenset({"modys:child03"}))
    g.check({"modys:child01"}, "x")
    with pytest.raises(AssertionError, match="lockbox child reached the analysis"):
        g.check({"modys:child01", "modys:child03"}, "x")


def test_parser_refuses_a_lockbox_video_before_reading():
    class NoRead:
        def read(self, name):
            raise RuntimeError("a member was read")
    with pytest.raises(AssertionError, match="refused"):
        S.parse_dlc(NoRead(), "x/005_DLC.csv", {1, 2}, {5})


def test_lockbox_file_hash_is_checked(tmp_path):
    here, _ = build(tmp_path)
    with pytest.raises(AssertionError, match="lockbox v1.2"):
        S.lockbox_children(here)


def test_item_values_999_and_side_pooling():
    items = {n: 2 for n in S.ITEM_COLUMNS}
    items["D_RAP_R_tD"] = 4
    items["D_LLP_R_tA"] = 999
    reg, raw = S.region_items(items)
    assert reg[("dystonia", "duration", "proximal_arm")] == 3.0
    assert math.isnan(reg[("dystonia", "amplitude", "proximal_leg")])
    assert math.isnan(reg[("dystonia", "subscale", "all")])
    assert reg[("choreoathetosis", "subscale", "all")] == 20.0
    with pytest.raises(AssertionError):
        S.item_value(5)


def test_probe_rule_version_2():
    a, b = S.ZIPS
    res = S.probe_reading({a: [1.2] * 9 + [0.5], b: [0.6] * 10})
    assert res["reading"] == {a: "lying", b: "sitting"} and res["settled"]
    res = S.probe_reading({a: [1.2] * 9 + [math.nan], b: [0.6] * 10})     # a record without q: not on its side
    assert res["fraction"][a] == pytest.approx(0.9) and res["settled"]
    res = S.probe_reading({a: [1.2] * 8 + [math.nan, 0.5], b: [0.6] * 10})
    assert not res["settled"]
    res = S.probe_reading({a: [0.6] * 10, b: [0.6] * 10})
    assert res["reading"] is None and not res["settled"]


def test_probe_q_reads_only_frames_with_all_six_points():
    T = 10
    xy = np.zeros((T, len(F.POINTS), 2))
    like = np.ones((T, len(F.POINTS)))
    ix = {n: i for i, n in enumerate(F.POINTS)}
    for s, dx in (("1", -10), ("2", 10)):
        xy[:, ix["shoulder" + s]] = [dx, -100]
        xy[:, ix["hip" + s]] = [dx, 0]
        xy[:, ix["knee" + s]] = [dx, 50]
    like[:5, ix["knee1"]] = 0.4
    q, n = S.probe_q({"names": F.POINTS, "xy": xy, "like": like})
    assert q == pytest.approx(0.5) and n == 5


def test_posture_needs_the_viewing_and_both_steps(tmp_path, monkeypatch):
    out = tmp_path / "ms"
    out.mkdir()
    monkeypatch.setattr(S, "OUT_STRUCT", str(out))
    with pytest.raises(AssertionError, match="viewing.md is absent"):
        S.stage_posture(type("A", (), {"moveatlas": str(tmp_path)})())
    (out / "viewing.md").write_text("posture data_stickfigure_coordinates_lying.zip: sitting\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="no posture for each zip"):
        S.read_viewing(str(out / "viewing.md"))
