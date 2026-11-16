"""Tests of the provenance of a stage-1 run, mdfm/train/provenance.py (spec docs/MFM_MODEL_CODE_SPEC_2026-09-28.md,
2.6, task B4.17).

- Hashes are read from the reports at run time: the lockbox and records SHA-256 of
  reports/manifest_tiers/hashes.json and reports/lockbox/lockbox.sha256 (a lockbox list or a hash file that
  disagrees raises), the MoveAtlas HEAD from its .git files equal to the commit reports/moveatlas_state
  checked, the rule files, the config and a code hash that moves when a source file changes.
- The training person lists of the stores on this machine hash to the values their reports hold, and pass
  the guard; their held-out persons are disjoint from them.
- The guard aborts on an injected lockbox person, EVAL person, EXCLUDED_V1 person, CODY name, CODY namespace,
  CODY row id, the canary and a held-out person; a guard or reader that lets the canary through raises.
- The state_dict audit passes on the tiny model, its EMA and its optimizer state, and trips on an injected
  shapedirs buffer, J_template or J_dirs in the model, the EMA or the optimizer state (MoveAtlas SMPL+H files
  needed).
- Tier hash history (T16), on temporary records files: an entry recomputes records_sha256 by the rule of
  scripts/build_tiers_and_lockbox.py (and the live records file hashes to the value of hashes.json); rows out
  of the build's order or an unknown column stop it; the history is hash-checked and never rewritten. The
  records drift policy tolerates an fps change in a source outside the run and refuses an fps change in a
  training or held-out source, an n_frames change, a tier, lockbox flag, group or person change, a row added
  or removed, a run source absent from the records, a checkpoint state the history does not hold and a live
  file that does not hash to hashes.json.
- Archived records states (scripts/archive_records_state.py): a state is named only when its archived file hashes
  to its record and to the history entry, and the live lockbox equals the state's; under a named state the
  unchanged policy compares the run's state with it.
- The records states of the pilot runs (cache/runs/*/provenance.json): the history holds them, the policy passes
  under the archived state they were born on and tolerates the batch 0002 state, the live state after A23 stays
  refused by default, and on a copy of the live records with the batch 0002 change (fps filled for
  li_taati and remap_open) it tolerates exactly those two sources while an fps change on an amass row raises.
"""
import csv
import gzip
import json
import os
from dataclasses import replace

import pytest
import torch

from mdfm import paths
from mdfm.data import canonical as C
from mdfm.train import provenance as PV

HAVE_MANIFEST = os.path.exists(C.manifest_path())
needs_manifest = pytest.mark.skipif(not HAVE_MANIFEST, reason="tiers records file not found")
HAVE_MA = os.path.exists(paths.smplh_model("neutral"))
needs_ma = pytest.mark.skipif(not HAVE_MA, reason="MoveAtlas SMPL+H model not found")
HAVE_STORES = os.path.exists(os.path.join(paths.CACHE, "window_store", "store.json")) and \
    os.path.isdir(os.path.join(paths.CACHE, "body_store", "train"))
needs_stores = pytest.mark.skipif(not HAVE_STORES, reason="window store or body store not built")


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------- hashes
@needs_manifest
def test_hashes_are_read_from_the_reports():
    h = PV.hashes()
    tiers = json.loads(_read(PV.TIERS_JSON))
    assert h["lockbox_sha256"] == _read(PV.LOCKBOX_SHA).split()[0] == tiers["lockbox_sha256"]
    assert h["lockbox_sha256"] == PV.sha256_file(PV.LOCKBOX_CSV)
    assert h["records_sha256"] == tiers["records_sha256"]
    assert h["records_file_sha256"] == tiers["records_file_sha256"] == PV.sha256_file(C.manifest_path())
    body, win = json.loads(_read(PV.BODY_JSON)), json.loads(_read(PV.WINDOW_JSON))
    assert h["body_store_report"]["train_persons_sha256"] == body["person_lists"]["train"]["store_sha256"]
    assert h["window_store_report"]["person_sha256"] == win["person_sha256"]
    assert set(h["rules_sha256"]) == {n for n in os.listdir(PV.RULES_DIR) if not n.startswith(".")}
    if os.path.isdir(os.path.join(paths.MOVEATLAS, ".git")) and os.path.exists(PV.STATE_JSON):
        state = json.loads(_read(PV.STATE_JSON))
        assert h["moveatlas_head"] == state["head_sha"] and h["moveatlas_head"].startswith(state["expect"])
        assert h["moveatlas_head_is_checked"]
    assert h["config_sha256"] is None
    assert PV.hashes(cfg={"a": 1})["config_sha256"] == PV.sha256_bytes(b'{"a":1}')


def test_a_lockbox_list_that_disagrees_with_its_hash_file_raises(tmp_path, monkeypatch):
    bad = tmp_path / "lockbox.sha256"
    bad.write_text("0" * 64 + "  lockbox_persons.csv\n", encoding="utf-8")
    monkeypatch.setattr(PV, "LOCKBOX_SHA", str(bad))
    with pytest.raises(PV.ProvenanceError):
        PV.lockbox_sha256()
    copy = tmp_path / "lockbox_persons.csv"
    copy.write_text(_read(PV.LOCKBOX_CSV) + "x,y\n", encoding="utf-8")
    monkeypatch.setattr(PV, "LOCKBOX_SHA", os.path.join(paths.TREE, "reports", "lockbox", "lockbox.sha256"))
    monkeypatch.setattr(PV, "LOCKBOX_CSV", str(copy))
    with pytest.raises(PV.ProvenanceError):
        PV.lockbox_sha256()


def test_code_hash_moves_with_a_source_file(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "pkg" / "b.py").write_text("y = 2\n", encoding="utf-8")
    (tmp_path / "._a.py").write_text("sidecar", encoding="utf-8")
    h0 = PV.code_sha256(str(tmp_path))
    assert h0 == PV.code_sha256(str(tmp_path))
    (tmp_path / "._a.py").write_text("changed sidecar", encoding="utf-8")
    assert PV.code_sha256(str(tmp_path)) == h0
    (tmp_path / "pkg" / "b.py").write_text("y = 3\n", encoding="utf-8")
    assert PV.code_sha256(str(tmp_path)) != h0
    assert len(PV.code_sha256()) == 64


# ---------------------------------------------------------------- person lists
def test_list_hashes_follow_the_report_rules():
    from mdfm.data import amass_index
    from mdfm.data.window_store import person_hash
    lists = {"b": ["x2", "x1", "x1"], "a": ["y"]}
    h = PV.list_hashes(lists)
    assert h["b"] == person_hash(["x1", "x2"]) == PV.sha256_bytes(b"x1\nx2")
    assert h["all"] == PV.sha256_bytes("a\ty\nb\tx1\nb\tx2".encode("utf-8"))
    assert amass_index.person_list_sha256(["x2", "x1"]) == PV.sha256_bytes(b"x1\nx2\n")


@needs_stores
@needs_manifest
def test_training_lists_of_the_stores_match_their_reports_and_pass_the_guard():
    from mdfm.data.body_store import BodyStore
    from mdfm.data.window_store import WindowStore
    bs, ws = BodyStore(split="train"), WindowStore()
    lists = PV.person_lists(bs, ws)
    assert set(lists) == {PV.AMASS} | set(ws.person_lists())
    assert len(lists[PV.AMASS]) == len(bs.persons)
    assert PV.report_mismatches(lists) == []
    held = PV.held_out_persons(ws)
    both = {p for v in lists.values() for p in v} & {p for v in held.values() for p in v}
    assert not both and held[PV.AMASS] and set(held) > {PV.AMASS}
    PV.assert_clean(lists, held_out=held)
    rec = PV.record(None, bs, ws)
    assert rec["report_mismatches"] == [] and rec["persons"] == lists
    assert rec["persons_sha256"] == PV.list_hashes(lists)
    assert rec["hashes"]["window_store"]["person_sha256"] == ws.person_hashes()
    with pytest.raises(C.Refused):
        dev = BodyStore.__new__(BodyStore)
        dev.split, dev.takes = "dev", bs.takes
        PV.person_lists(dev, None)


# ---------------------------------------------------------------- the guard
def _lockbox_person():
    with open(PV.LOCKBOX_CSV, encoding="utf-8", newline="") as f:
        return next(csv.DictReader(f))["person_id"]


@needs_manifest
def test_injected_lockbox_eval_cody_and_canary_ids_are_refused():
    deny = PV.deny_lists()
    clean = {"gait120": ["fake:gait120:P01", "fake:gait120:P02"], "amass": ["fake:amass:P01"]}
    PV.assert_clean(clean)
    injected = {"lockbox": _lockbox_person(), "EVAL": sorted(deny.eval)[0], "EXCLUDED_V1": sorted(deny.excluded)[0],
                "CODY": "cody_1:P001", "CODY namespace": "adult:P001", "CODY row": sorted(deny.cody)[0],
                "canary": deny.canary}
    for why, pid in injected.items():
        persons = dict(clean, gait120=clean["gait120"] + [pid])
        with pytest.raises(C.Refused) as e:
            PV.assert_clean(persons)
        assert pid in str(e.value), why
        assert PV.problems([("gait120", pid)]), why
    for pairs in ([("x", "cody_2:A")], [("cody_3", "p")], ["Cody_external:q"]):
        with pytest.raises(C.Refused):
            PV.assert_clean(pairs)
    with pytest.raises(C.Refused):
        PV.assert_clean(clean, held_out={"gait120": ["fake:gait120:P02"]})
    assert "canary" in PV.problems([deny.canary])[0]


@needs_manifest
def test_a_guard_or_reader_that_lets_the_canary_through_is_refused(monkeypatch):
    PV.canary_check()
    monkeypatch.setattr(PV, "problems", lambda persons, held_out=None, deny=None, tier=PV.RELEASE: [])
    with pytest.raises(PV.ProvenanceError):
        PV.assert_clean(["fake:x"])
    monkeypatch.undo()
    monkeypatch.setattr(PV, "reader_check", lambda row, **kw: row)
    with pytest.raises(PV.ProvenanceError):
        PV.assert_clean(["fake:x"])


# ---------------------------------------------------------------- audit
@needs_ma
def test_the_audit_passes_the_tiny_model_and_trips_on_smplh_arrays(monkeypatch):
    from mdfm.body import backend as BK
    from mdfm.body.smplh import SMPLHBody, SkeletonBank
    from mdfm.model import mfm as M
    # the v1 array audit, behind the reference flag (plan step M2)
    monkeypatch.setenv(BK.REFERENCE_ENV, "1")
    monkeypatch.setenv(BK.MODEL_BODY_ENV, BK.SMPLH)
    torch.manual_seed(0)
    model = M.MFM(M.TINY)
    ema = M.MFM(M.TINY)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss = sum((p.float() ** 2).sum() for p in model.parameters() if p.requires_grad)
    loss.backward()
    opt.step()
    states = PV.audit(model, ema, opt)
    assert set(states) == {"model", "ema", "opt"} and states["opt"]["state"]
    bank = SkeletonBank(torch.float64)
    inject = {"shapedirs": SMPLHBody("male").shapedirs.float(), "J_template": bank.J_template[0].float(),
              "J_dirs": bank.J_dirs[1].float()}
    sd, esd, osd = model.state_dict(), ema.state_dict(), opt.state_dict()
    ids = list(osd["state"])
    for k, (name, x) in enumerate(inject.items()):
        sd[f"decoder.extra_{name}"] = x.clone()
        esd[f"heads.buf_{name}"] = x.clone()
        osd["state"][ids[k]] = dict(osd["state"][ids[k]], exp_avg=x.clone())
    with pytest.raises(PV.ProvenanceError) as e:
        PV.audit(sd, esd, osd)
    msg = "\n".join(e.value.problems)
    for k, name in enumerate(inject):
        assert f"model.decoder.extra_{name}:" in msg and f"ema.heads.buf_{name}:" in msg, name
        assert f"opt.state.{ids[k]}.exp_avg:" in msg, name
    assert PV.audit_state_dict({"x": torch.zeros(6890, 3)})


def test_the_default_audit_is_the_open_body_licence_record_and_reads_no_smplh(monkeypatch):
    """Plan step M2: by default the audit screens shapes and names and checks the open body's licence record
    (the three licence SHA-256 of reports/openbody/licence_audit.json); it never opens an SMPL+H file, and the SMPL+H
    array audit is refused without the reference flag."""
    from mdfm import paths
    from mdfm.body import backend as BK
    from mdfm.body import openbody as BO
    from mdfm.model import mfm as M
    monkeypatch.delenv(BK.REFERENCE_ENV, raising=False)
    monkeypatch.delenv(BK.MODEL_BODY_ENV, raising=False)

    def no_smplh(*a, **k):
        raise AssertionError("the open audit read an SMPL+H file")
    monkeypatch.setattr(paths, "smplh_model", no_smplh)
    model = M.MFM(replace(M.TINY, skeleton=False))
    states = PV.audit(model)
    assert set(states) == {"model", "ema", "opt"}
    assert PV.audit_state_dict({"x": torch.zeros(6890, 3)}) and PV.audit_state_dict({"a.shapedirs": torch.zeros(2)})
    with pytest.raises(BK.ReferenceRefused):
        PV.smplh_arrays()
    with pytest.raises(BK.ReferenceRefused):
        PV.audit(model, body=BK.SMPLH)
    rec = BO.licence_record()
    assert BO.check_licence(rec) == []
    key = next(iter(rec["licences_sha256"]))
    bad = dict(rec, licences_sha256=dict(rec["licences_sha256"], **{key: "0" * 64}))
    assert any(key in p for p in BO.check_licence(bad))
    assert BO.check_licence(dict(rec, topology="smplx"))
    PV.open_licence_problems.cache_clear()
    monkeypatch.setattr(BO, "check_licence", lambda record=None: ["a changed licence text"])
    with pytest.raises(PV.ProvenanceError, match="licence"):
        PV.audit(model)
    PV.open_licence_problems.cache_clear()


# ---------------------------------------------------------------- tier hash history and records drift (T16)
COLS = ["source", "manifest_source", "modality", "subject_ns", "person_id", "key", "fps", "n_frames", "tier", "lane",
        "scope", "lockbox", "group"]
RUNS = ("phaseA_root_full", "pilot_v1_s2")
# archived records states (scripts/archive_records_state.py, reports/provenance_history/records_archive.json): the
# state the two runs were born on and the state of batch 0002 they were read under before the tiers rerun A23
STATE_CA5A = "ca5a2d9eaada02b70f834d20ebc77f19de7b89719a0aed32e352faed0abdb8d8"
STATE_0A45 = "0a453d1ce2335cee32374f410987446ab84f5d146dddc07349eccfa30bc680bf"


def _order(rows):
    return sorted(rows, key=lambda r: (r["source"], r["modality"], r["subject_ns"], r["key"]))


def _rows():
    """Records in the build's order: two run sources (amass, gait120), two others (li_taati, remap_open)."""
    out = []
    for s, fps in (("amass", "120.0"), ("gait120", "100.0"), ("li_taati", ""), ("remap_open", "")):
        for k in range(3):
            out.append({"source": s, "manifest_source": s, "modality": "pose", "subject_ns": f"{s}:S{k}",
                        "person_id": f"fake:{s}:P{k}", "key": f"take{k}", "fps": fps, "n_frames": str(100 + k),
                        "tier": "RELEASE", "lane": "body", "scope": "in", "lockbox": "False", "group": "all"})
    return _order(out)


def _edit(rows, source, first=False, **values):
    """A copy of rows with the rows of source (only its first with first) given values."""
    out, done = [], False
    for r in rows:
        r = dict(r)
        if r["source"] == source and not (first and done):
            r.update(values)
            done = True
        out.append(r)
    return out


def _write(path, rows, cols=COLS):
    with gzip.open(path, "wt", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return str(path)


def _entry(tmp_path, rows, name):
    return PV.tier_hash_entry(_write(tmp_path / f"{name}.csv.gz", rows))


def _build_hash(rows):
    """records_sha256 by the rule of scripts/build_tiers_and_lockbox.py, written out again here."""
    import hashlib
    h = hashlib.sha256()
    for r in rows:
        h.update(("\t".join(str(r[c]) for c in COLS) + "\n").encode())
    return h.hexdigest()


def test_the_tier_hash_entry_recomputes_the_build_records_hash(tmp_path):
    rows = _rows()
    e = _entry(tmp_path, rows, "a")
    assert e["records_sha256"] == _build_hash(rows) and e["records"] == 12 and e["records_columns"] == COLS
    assert e["records_file_sha256"] == PV.sha256_file(tmp_path / "a.csv.gz")
    assert sorted(e["sources"]) == ["amass", "gait120", "li_taati", "remap_open"]
    li = e["sources"]["li_taati"]
    assert li["rows"] == 3 and li["values"]["fps"] == {"": 3} and set(li["columns_sha256"]) == set(PV.OTHER_COLUMNS)
    assert _entry(tmp_path, rows, "b")["sources"] == e["sources"]                 # the file name plays no part
    moved = _entry(tmp_path, _edit(rows, "li_taati", fps="30.0"), "c")
    assert moved["sources"]["li_taati"]["tier_sha256"] == li["tier_sha256"]
    assert moved["sources"]["li_taati"]["columns_sha256"]["fps"] != li["columns_sha256"]["fps"]
    assert moved["sources"]["amass"] == e["sources"]["amass"]
    with pytest.raises(PV.ProvenanceError):
        _entry(tmp_path, rows[::-1], "reversed")                                  # not the build's order
    with pytest.raises(PV.ProvenanceError):
        _entry(tmp_path, rows[:1] + rows, "twice")                                # a repeated identity
    with pytest.raises(PV.ProvenanceError):
        PV.tier_hash_entry(_write(tmp_path / "x.csv.gz", [dict(r, extra="1") for r in rows], COLS + ["extra"]))
    with pytest.raises(PV.ProvenanceError):
        PV.tier_hash_entry(_write(tmp_path / "y.csv.gz", rows, [c for c in COLS if c != "group"]))


@needs_manifest
def test_the_live_records_file_hashes_to_the_records_sha256_of_hashes_json():
    e = PV.current_tier_entry()
    t = PV.tiers()
    assert e["records_sha256"] == t["records_sha256"] and e["records"] == t["records"]
    assert e["records_file_sha256"] == t["records_file_sha256"] and e["records_columns"] == t["records_columns"]
    assert PV.current_tier_entry() is e                                            # once per file and process


def test_the_history_is_hash_checked_and_never_rewritten(tmp_path):
    e = dict(_entry(tmp_path, _rows(), "a"), lockbox_sha256="L" * 64, written="2026-10-05T00:00:00")
    doc, added = PV.add_tier_entry(None, e)
    assert added and doc["format"] == PV.HISTORY_FORMAT and doc["tier_columns"] == list(PV.TIER_COLUMNS)
    held = doc["entries"][e["records_sha256"]]
    assert held["entry_sha256"] == PV.entry_sha256(held) and "entry_sha256" not in e
    path = tmp_path / "history.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    assert PV.tier_hash_history(str(path)) == {e["records_sha256"]: held}
    assert PV.tier_hash_history(str(tmp_path / "absent.json")) == {}
    doc2, added = PV.add_tier_entry(json.loads(path.read_text(encoding="utf-8")), dict(e, written="later"))
    assert not added and doc2["entries"][e["records_sha256"]] == held             # an equal entry is kept as held
    with pytest.raises(PV.ProvenanceError):
        PV.add_tier_entry(doc2, dict(e, records=13))                              # another content, same key
    bad = json.loads(path.read_text(encoding="utf-8"))
    bad["entries"][e["records_sha256"]]["sources"]["li_taati"]["rows"] = 4
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(PV.ProvenanceError):
        PV.tier_hash_history(str(path))
    bad = dict(doc, tier_columns=list(PV.TIER_COLUMNS[:-1]))
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(PV.ProvenanceError):
        PV.tier_hash_history(str(path))


def _prov(old, persons=None, held=None):
    persons = {"amass": ["fake:amass:P0"], "gait120": ["fake:gait120:P0"]} if persons is None else persons
    held = {"gait120": ["fake:gait120:P2"]} if held is None else held
    return {"hashes": {"records_sha256": old["records_sha256"], "records_file_sha256": old["records_file_sha256"],
                       "lockbox_sha256": "L" * 64}, "persons": persons, "held_out": held}


def _drift(prov, old, new, history=None):
    now = {"records_sha256": new["records_sha256"], "records_file_sha256": new["records_file_sha256"],
           "lockbox_sha256": "L" * 64}
    hist = {old["records_sha256"]: old} if history is None else history
    return PV.records_drift(prov, now=now, history=hist, current=new)


def test_the_records_drift_policy_tolerates_only_fps_outside_the_run(tmp_path):
    rows = _rows()
    old = _entry(tmp_path, rows, "old")
    prov = _prov(old)
    assert PV.run_sources(prov) == ["amass", "gait120"]
    assert _drift(prov, old, old) is None                                          # unchanged state
    new = _entry(tmp_path, _edit(rows, "li_taati", fps="30.0"), "li")
    ev = _drift(prov, old, new)
    assert ev["records_sha256"] == [old["records_sha256"], new["records_sha256"]]
    assert ev["run_sources"] == ["amass", "gait120"] and ev["lockbox_sha256"] == "L" * 64
    (ch,) = ev["changes"]
    assert ch["source"] == "li_taati" and ch["kind"] == "columns" and ch["columns"] == ["fps"] and ch["tolerated"]
    assert ch["values"]["fps"] == {"before": {"": 3}, "after": {"30.0": 3}} and ch["rows"] == [3, 3]
    both = _entry(tmp_path, _edit(_edit(rows, "li_taati", fps="30.0"), "remap_open", fps="30.0"), "both")
    assert [c["source"] for c in _drift(prov, old, both)["changes"]] == ["li_taati", "remap_open"]
    refused = {
        "fps of a training source": _edit(rows, "amass", first=True, fps="60.0"),
        "n_frames outside the run": _edit(rows, "li_taati", first=True, n_frames="7"),
        "tier": _edit(rows, "li_taati", first=True, tier="EVAL"),
        "lockbox flag": _edit(rows, "li_taati", first=True, lockbox="True"),
        "group": _edit(rows, "remap_open", first=True, group="PD"),
        "person": _edit(rows, "li_taati", first=True, person_id="fake:li_taati:P9"),
        "manifest_source": _edit(rows, "li_taati", first=True, manifest_source="li"),
        "row removed": [r for r in rows if r["key"] != "take2" or r["source"] != "li_taati"],
        "row added": _order(rows + [dict(rows[-1], key="take9")]),
        "source added": _order(rows + [dict(rows[-1], source="new_src", subject_ns="new_src:S0")]),
    }
    for why, edited in refused.items():
        with pytest.raises(PV.ProvenanceError) as e:
            _drift(prov, old, _entry(tmp_path, edited, "r"))
        assert e.value.problems, why
    with pytest.raises(PV.ProvenanceError) as e:
        _drift(prov, old, _entry(tmp_path, _edit(rows, "amass", fps="100.0"), "am"))
    assert "amass" in str(e.value) and "per-sample fps rule" in str(e.value)
    # a held-out source belongs to the run: its fps stays fatal
    gait = _entry(tmp_path, _edit(rows, "gait120", fps="2000.0"), "g")
    with pytest.raises(PV.ProvenanceError):
        _drift(_prov(old, persons={"amass": ["fake:amass:P0"]}), old, gait)
    with pytest.raises(PV.ProvenanceError):                                        # a run source with no row
        _drift(_prov(old, persons={"amass": ["fake:amass:P0"], "nowhere": ["x"]}), old, new)
    with pytest.raises(PV.ProvenanceError):                                        # the history lacks the state
        _drift(prov, old, new, history={})
    with pytest.raises(PV.ProvenanceError):                                        # another records file
        _drift(prov, dict(old, records_file_sha256="0" * 64), new)
    with pytest.raises(PV.ProvenanceError):                                        # another lockbox in the history
        _drift(prov, dict(old, lockbox_sha256="0" * 64), new)
    with pytest.raises(PV.ProvenanceError):                                        # live file against hashes.json
        PV.records_drift(prov, now={"records_sha256": "0" * 64, "records_file_sha256": new["records_file_sha256"]},
                         history={old["records_sha256"]: old}, current=new)
    with pytest.raises(PV.ProvenanceError):                                        # history entry of the live state
        _drift(prov, old, new, history={old["records_sha256"]: old, new["records_sha256"]: dict(new, records=1)})


def test_an_archived_records_state_is_checked_before_it_is_named(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "CACHE", str(tmp_path))
    folder = tmp_path / "manifest" / "archive"
    folder.mkdir(parents=True)
    rows = _rows()
    born = dict(_entry(tmp_path, rows, "born"), lockbox_sha256="L" * 64)
    path = _write(folder / "named.csv.gz", _edit(rows, "li_taati", fps="30.0"))
    named = dict(PV.tier_hash_entry(path), lockbox_sha256="L" * 64)
    doc, _ = PV.add_tier_entry(None, born)
    doc, _ = PV.add_tier_entry(doc, named)
    hist = doc["entries"]
    st = named["records_sha256"]
    arch = {st: {"records_sha256": st, "records_file_sha256": named["records_file_sha256"], "records": 12,
                 "cache_path": "manifest/archive/named.csv.gz", "history_entry_sha256": hist[st]["entry_sha256"]}}
    got = PV.archived_state(st, archive=arch, history=hist)
    assert got["records_sha256"] == st and got["lockbox_sha256"] == "L" * 64 and got["records"] == 12
    now = {"records_sha256": "0" * 64, "records_file_sha256": "1" * 64, "lockbox_sha256": "L" * 64}
    h, _ = PV.hashes_at(st, now, archive=arch, history=hist)
    assert h["records_sha256"] == st and h["records_file_sha256"] == named["records_file_sha256"]
    assert now["records_sha256"] == "0" * 64                                       # the given hashes are not changed
    # a run born on the named state reads it exactly; one born on another state passes the unchanged policy only
    assert PV.records_drift(_prov(named), now=now, history=hist, state=st, archive=arch) is None
    ev = PV.records_drift(_prov(born), now=now, history=hist, state=st, archive=arch)
    assert [c["source"] for c in ev["changes"]] == ["li_taati"] and ev["records_sha256"] == [born["records_sha256"], st]
    with pytest.raises(PV.ProvenanceError):                                        # the run's own source moved
        PV.records_drift(_prov(born, persons={"li_taati": ["fake:li_taati:P0"]}), now=now, history=hist, state=st,
                         archive=arch)
    with pytest.raises(PV.ProvenanceError):                                        # not in the archive record
        PV.archived_state(born["records_sha256"], archive=arch, history=hist)
    with pytest.raises(PV.ProvenanceError):                                        # the history lacks the state
        PV.archived_state(st, archive=arch, history={born["records_sha256"]: hist[born["records_sha256"]]})
    with pytest.raises(PV.ProvenanceError):                                        # another lockbox: never relaxed
        PV.hashes_at(st, dict(now, lockbox_sha256="M" * 64), archive=arch, history=hist)
    with pytest.raises(PV.ProvenanceError):                                        # a record of another file
        PV.archived_state(st, archive={st: dict(arch[st], records_file_sha256="2" * 64)}, history=hist)
    _write(folder / "named.csv.gz", rows)                                          # the file replaced
    with pytest.raises(PV.ProvenanceError):
        PV.archived_state(st, archive=arch, history=hist)
    os.remove(folder / "named.csv.gz")
    with pytest.raises(FileNotFoundError):                                         # absent from this machine
        PV.archived_state(st, archive=arch, history=hist)


def _run_provenance():
    out = {}
    for run in RUNS:
        p = os.path.join(paths.CACHE, "runs", run, "provenance.json")
        if os.path.exists(p):
            out[run] = json.loads(_read(p))
    return out


@pytest.fixture(scope="module")
def batch_0002(tmp_path_factory):
    """The live records with the batch 0002 change applied (fps 30.0 where li_taati and remap_open have none),
    their entry, and the entry of one amass fps changed on top; None when the live records are already rebuilt
    (their records_sha256 is no longer that of the pilot runs) or a run folder is missing."""
    runs = _run_provenance()
    if not HAVE_MANIFEST or not runs:
        return None
    live = PV.current_tier_entry()
    if any(p["hashes"]["records_sha256"] != live["records_sha256"] for p in runs.values()):
        return None
    with gzip.open(C.manifest_path(), "rt", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    d = tmp_path_factory.mktemp("batch_0002")
    fixed = [dict(r, fps="30.0") if r["source"] in ("li_taati", "remap_open") and r["fps"] == "" else r for r in rows]
    amass = next(i for i, r in enumerate(fixed) if r["source"] == "amass")
    worse = list(fixed)
    worse[amass] = dict(fixed[amass], fps="1.0")
    return {"live": live, "fixed": PV.tier_hash_entry(_write(d / "fixed.csv.gz", fixed, list(rows[0]))),
            "amass": PV.tier_hash_entry(_write(d / "amass.csv.gz", worse, list(rows[0])))}


@needs_manifest
def test_the_pilot_runs_keep_a_records_state_the_policy_reads(batch_0002):
    runs = _run_provenance()
    if not runs:
        pytest.skip("no pilot run folder with a provenance.json")
    arch = PV.records_archive()
    for s in (STATE_CA5A, STATE_0A45):
        assert s in arch, f"records state {s} is not in {PV.ARCHIVE_JSON}"
        if not os.path.exists(PV.archive_path(arch[s])):
            pytest.skip(f"the archived records file of state {s[:12]} is absent from this machine "
                        f"({PV.archive_path(arch[s])}; scripts/archive_records_state.py)")
    hist = PV.tier_hash_history()
    live = PV.tiers()["records_sha256"]
    for run, prov in runs.items():
        assert {"amass", "weargait_pd"} <= set(PV.run_sources(prov)), run
        assert not {"li_taati", "remap_open"} & set(PV.run_sources(prov)), run
        if hist:
            born = prov["hashes"]["records_sha256"]
            held = hist[born]
            assert held["records_file_sha256"] == prov["hashes"]["records_file_sha256"], run
            assert held["lockbox_sha256"] == prov["hashes"]["lockbox_sha256"], run
            assert born == STATE_CA5A, run
            # read under the archived state it was born on: no drift
            assert PV.records_drift(prov, state=born) is None, run
            # under the batch 0002 state: the unchanged policy tolerates the fps of two sources outside the run
            ev = PV.records_drift(prov, state=STATE_0A45)
            assert [(c["source"], c["columns"]) for c in ev["changes"]] == [("li_taati", ["fps"]),
                                                                             ("remap_open", ["fps"])], run
            # the live state after A23 (13 sources added, amass fps filled) stays refused by default
            if live not in (STATE_CA5A, STATE_0A45):
                with pytest.raises(PV.ProvenanceError) as e:
                    PV.records_drift(prov)
                assert any("rows added" in x for x in e.value.problems), run
        if batch_0002 is None:
            continue
        old = batch_0002["live"]
        here = dict(prov, hashes=dict(prov["hashes"], lockbox_sha256="L" * 64))
        ev = _drift(here, old, batch_0002["fixed"])
        assert [(c["source"], c["columns"]) for c in ev["changes"]] == [("li_taati", ["fps"]), ("remap_open", ["fps"])]
        assert all(set(c["values"]["fps"]["after"]) == {"30.0"} for c in ev["changes"])
        with pytest.raises(PV.ProvenanceError) as e:
            _drift(here, old, batch_0002["amass"])
        assert "amass" in str(e.value)
