"""Tests of the public movement gallery (scripts/render_skeleton_catalogue.py --public, 2026-11-10).

- provenance is measured at export: CaptureReads records every file a build opens under a capture store (and the
  calls of the body store and of the AMASS loader), and the data file carries it ('provenance');
- public_provenance leaves out a clip built on a capture store and every clip whose level-0 chain holds one, or a
  clip without a provenance record, and stops when the CAPTURED table names a clip the measure calls generated
  (positive controls);
- public_scan finds local paths, file names of the tree, internal version tags and task codes, the dashes and the
  hashed words, acronyms and phrases (positive controls through stand-in digests, so that no listed word is written
  here), and passes a clean text;
- public_blob keeps the keys the page draws and refuses a mesh, shape or model key;
- the manifest carries the fields of MoveAtlas's site/gen_movement_gallery.py, and a public build from two exported
  procedural clips writes a page and a manifest that pass the scan.
Bodies are procedural exam clips (no AMASS read, no lockbox, EVAL, CODY or care_pd record); files go to pytest's
temporary folder only.
"""
import base64
import gzip
import hashlib
import importlib.util
import json
import os
import re
from types import SimpleNamespace

import pytest

from mdfm import paths

HAVE_MA = os.path.exists(paths.smplh_model("neutral"))


def _script():
    spec = importlib.util.spec_from_file_location("render_skeleton_catalogue_script_public",
                                                  os.path.join(paths.TREE, "scripts", "render_skeleton_catalogue.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def S():
    return _script()


def _sha(t):
    return hashlib.sha256(t.lower().encode("utf-8")).hexdigest()


def _meta(cid, captured=False, level0=None, prov=True):
    m = {"clip_id": cid, "level0": level0}
    if prov:
        m["provenance"] = {"captured": captured, "stores": ["store"] if captured else [], "n_reads": int(captured),
                           "reads": ["store:take.bin"] if captured else []}
    return m


# ---------------------------------------------------------------- provenance measured at export
def test_capture_reads_records_reads_under_a_store(S, tmp_path):
    store, other = tmp_path / "store", tmp_path / "other"
    store.mkdir()
    other.mkdir()
    (store / "take.bin").write_bytes(b"x")
    (other / "free.bin").write_bytes(b"y")
    rec = S.CaptureReads(roots={"store": os.path.normcase(os.path.abspath(str(store)))})
    with open(other / "free.bin", "rb") as f:                # not recording: nothing kept
        f.read()
    rec.start()
    with open(other / "free.bin", "rb") as f:
        f.read()
    assert rec.stop() == [] and rec.last == []
    rec.start()
    with open(store / "take.bin", "rb") as f:
        f.read()
    os.listdir(str(store))
    reads = rec.stop()
    assert reads == ["store:.", "store:take.bin"] and rec.last == reads
    from mdfm.data import amass as AM
    from mdfm.data import body_store as BS
    assert getattr(BS.BodyStore.__init__, "_capture_wrapped", False)
    assert getattr(AM.load_sequence, "_capture_wrapped", False)
    rec.start()
    with pytest.raises(Exception):
        AM.load_sequence("NO_SUCH_ARCHIVE", "x_poses.npz")   # refused by the index, recorded as a call all the same
    assert "api:amass.load_sequence" in rec.stop()


def test_capture_roots_cover_the_stores(S):
    roots = S.capture_roots()
    from mdfm.data import body_store as BS
    norm = os.path.normcase(os.path.abspath(BS.store_root()))
    assert roots["body_store"] == norm and "moveatlas_datasets" in roots and "amass_cache" in roots
    assert all(f"window_store_{k}" in roots for k in paths.WINDOW_STORE_KINDS)


@pytest.mark.skipif(not HAVE_MA, reason="MoveAtlas SMPL+H model not found")
def test_export_writes_the_measured_provenance(S, tmp_path):
    """A build that reads a file under a capture store is written 'captured'; the same build without the read is
    'generated' (positive control of the export's record)."""
    import dataclasses
    from mdfm.synth import exam as EX
    s = {x.clip_id: x for x in EX.exam_set()}["tremor_E1_rest_g0"]
    store = tmp_path / "store"
    store.mkdir()
    (store / "take.bin").write_bytes(b"take")
    rec = S.CaptureReads(roots={"store": os.path.normcase(os.path.abspath(str(store)))})

    def reading_build():
        with open(store / "take.bin", "rb") as f:
            f.read()
        return s.build()
    out = str(tmp_path / "cat").replace("\\", "/")
    code = S.CodeState([s], []).of(s)
    for spec, captured in ((s, False), (dataclasses.replace(s, build=reading_build), True)):
        meta, bad = S.export_clip(spec, out, S.SET_CATALOGUE, module_checks=True, code=code, recorder=rec)
        assert bad is None
        with gzip.open(S.data_path(out, s.clip_id), "rb") as f:
            d = json.loads(f.read().decode("utf-8"))
        assert d["provenance"]["captured"] is captured and d["code"]["export_format"] == S.EXPORT_FORMAT
        assert S.own_provenance(d) == ("motion_capture" if captured else S.GENERATED)
        assert d["provenance"]["reads"] == (["store:take.bin"] if captured else [])


# ---------------------------------------------------------------- the provenance rule of the gallery
def test_public_provenance_leaves_out_captured_chains(S):
    metas = [_meta("walk_x", captured=True), _meta("sym_a", level0="walk_x"), _meta("sym_b", level0="sym_a"),
             _meta("gen_l0"), _meta("sym_c", level0="gen_l0"), _meta("noprov", prov=False),
             _meta("sym_d", level0="missing_l0")]
    specs = [SimpleNamespace(clip_id=m["clip_id"], level0=m["level0"]) for m in metas]
    specs.append(SimpleNamespace(clip_id="missing_l0", level0=None))
    own, out = S.public_provenance(metas, specs)
    assert own["walk_x"] == "motion_capture" and own["noprov"] == "unrecorded" and own["sym_c"] == S.GENERATED
    assert out == {"walk_x": "motion_capture_take", "sym_a": "level0_chain_motion_capture",
                   "sym_b": "level0_chain_motion_capture", "noprov": "provenance_unrecorded",
                   "sym_d": "level0_chain_unrecorded"}


def test_public_provenance_distrusts_a_missed_read(S):
    """A clip the CAPTURED table names whose build recorded no capture read stops the build."""
    # open-body port, wave 7: the table names no clip of the catalogue (the prefix 'mocap_'); its rule is read on
    # an id of that form, and the former captured references are generated now
    assert not re.match(S.CAPTURED, "ref_walk_E8") and not re.match(S.CAPTURED, "freezing_E8_g2_akinesia")
    metas = [_meta("mocap_walk_E8", captured=False)]
    with pytest.raises(SystemExit, match="not trusted"):
        S.public_provenance(metas, [SimpleNamespace(clip_id="mocap_walk_E8", level0=None)])
    own, out = S.public_provenance([_meta("mocap_walk_E8", captured=True)],
                                   [SimpleNamespace(clip_id="mocap_walk_E8", level0=None)])
    assert out == {"mocap_walk_E8": "motion_capture_take"}


def test_left_out_reasons_and_note(S):
    specs = [SimpleNamespace(clip_id=c, level0=None) for c in ("a", "b", "c", "d", "e", "f", "g")]
    refused = {"c": ["background"], "d": [S.SKELETON_FAIL + "skate"]}
    hidden = {"d": [S.SKELETON_FAIL + "skate"], "e": [S.INVISIBLE], "f": [S.LONE_LEVEL0]}
    left = S.left_out_reasons(specs, {"a"}, refused, {}, hidden, {"b": "motion_capture_take"})
    assert left == {"motion_capture_take": ["b"], "validity_check_failed": ["c"], "skeleton_check_failed": ["d"],
                    "invisible_on_skeleton": ["e"], "level0_without_shown_symptom": ["f"], "no_data": ["g"]}
    assert list(left) == [k for k in S.LEFT_OUT_REASONS if k in left]
    note = S.public_note(left)
    assert note.startswith("6 clips of the program's catalogue are not in this gallery: 1 built on a motion-capture")
    assert S.public_note({}) == "Every clip of the program's catalogue is shown."


# ---------------------------------------------------------------- the text scan
def test_public_scan_positive_controls(S, monkeypatch):
    # the drive letters are joined at run time: the public tree's redactions would rewrite the paths in clear
    bad = {"local path": ["open " + "F:" + "/MD_FM/synth_viewer now", "see " + "D:" + "\\MoveAtlas\\x",
                          "in /Users/someone/x"],
           "file name of the tree": ["built by scripts/render_x.py", "read skeleton_text.py", "SYNTHETIC_a_skeleton",
                                     "the rows of docs/x"],
           "internal version tag": ["since exam_v2", "physio-v6"],
           "internal task code": ["task SKELV3", "wave SH1d", "decision OD4", "SP5 rule"],
           "dash": ["a \u2014 b", "1\u20132"], "code identifier": ["a fist (handtasks.FIST) then"]}
    for kind, texts in bad.items():
        for t in texts:
            assert kind in [k for k, _ in S.public_scan_text(t)], (kind, t)
    monkeypatch.setattr(S, "PUBLIC_WORD_SHA256", frozenset({_sha("zebra")}))
    monkeypatch.setattr(S, "PUBLIC_ACRONYM_SHA256", frozenset({_sha("qq")}))
    monkeypatch.setattr(S, "PUBLIC_PHRASE_SHA256", frozenset({_sha("blue heron")}))
    assert [k for k, _ in S.public_scan_text("a Zebra here")] == ["word naming a writing tool or its model"]
    assert [k for k, _ in S.public_scan_text("made by QQ")] == ["abbreviation of the field"]
    assert S.public_scan_text("made by qq") == []
    assert [k for k, _ in S.public_scan_text("a Blue  heron")] == ["phrase naming the field or a writing tool"]


def test_public_scan_passes_a_clinical_text_and_reads_the_data(S):
    clean = ("Tremor, rest, left forearm: MDS-UPDRS 3.17 grade 2, 5 Hz, segment E1, clip tremor_E1_rest_g2; SARA 8, "
             "UDysRS part 3, doi:10.1145/3130800.3130883, MANO/SMPL+H, the same said again.")
    assert S.public_scan_text(clean) == []
    assert len(S.PUBLIC_WORD_SHA256) >= 10
    for lst in (S.PUBLIC_WORD_SHA256, S.PUBLIC_ACRONYM_SHA256, S.PUBLIC_PHRASE_SHA256, S.PUBLIC_TASK_PREFIX_SHA256):
        assert all(re.fullmatch(r"[0-9a-f]{64}", d) for d in lst)
    blob = base64.b64encode(gzip.compress(json.dumps({"rule": "see scripts/x"}).encode("utf-8"))).decode("ascii")
    page = f'<p>{clean}</p><script type="text/plain" class="clipdata" data-clip="c1">{blob}</script>'
    hits = S.public_scan(page, json.dumps({"clips": ["c1"]}))
    assert list(hits) == ["data c1"] and hits["data c1"][0][0] == "file name of the tree"


# ---------------------------------------------------------------- the public data and manifest
def test_public_blob_keeps_drawing_keys_only(S):
    d = {"label": "SYNTHETIC", "notice": "internal", "clip_id": "c1", "heading_deg": 0.0, "scene_boxes_m": [],
         "walk": None, "floor": {"floor_y_m": -0.01, "rule": "viewer.sole_offset_m()"}, "code": {"module": "x"},
         "provenance": {"captured": False}, "operator": "tremor", "since": "exam_v2",
         "skeleton": {"n_frames": 1, "positions_mm_delta": [0, 0, 0], "names": ["pelvis"],
                      "landmarks": {"nose": {"joint": 15, "offset_m": [0, 0.07, 0.1], "rule": "a vertex"}}}}
    out = json.loads(gzip.decompress(S.public_blob(gzip.compress(json.dumps(d).encode("utf-8")))))
    assert set(out) == {"label", "notice", "clip_id", "heading_deg", "scene_boxes_m", "walk", "floor", "skeleton"}
    assert out["notice"] == S.PUBLIC_NOTICE and out["floor"] == {"floor_y_m": -0.01}
    assert out["skeleton"]["landmarks"] == {"nose": {"joint": 15, "offset_m": [0, 0.07, 0.1]}}
    d["skeleton"]["vertices"] = [[0, 0, 0]]
    with pytest.raises(RuntimeError, match="mesh, shape or model"):
        S.public_blob(gzip.compress(json.dumps(d).encode("utf-8")))


def test_public_index_drops_versions_marks_and_code_refs(S):
    idx = {"clips": [{"id": "c1", "since": "exam_v2", "text": {
        "params": "closing into a fist (handtasks.FIST) from 5 s", "operator": "tremor",
        "provenance": [{"about": "5 Hz", "meaning": "a value the generator chose", "marks": ["module"]}]}}]}
    c = S.public_index(idx)["clips"][0]
    assert "since" not in c and c["text"]["params"] == "closing into a fist from 5 s"
    assert c["text"]["provenance"] == [{"about": "5 Hz", "meaning": "a value the generator chose"}]


def test_public_manifest_fields(S):
    """The fields MoveAtlas's site/gen_movement_gallery.py reads: public, clips, provenance of each, the numbers of
    symptom and level 0 clips, groups {name, n_clips, scales}, left_out {reason: [ids]}, created_utc, page_bytes,
    page_sha256."""
    TX = S.TX
    g1, g2 = TX.GROUP_ORDER[0], TX.GROUP_ORDER[2]

    def clip(cid, fam, level0, item):
        return {"id": cid, "family": fam, "level0": level0, "n_frames": 30,
                "text": {"item": item, "scale": "", "operator": "x", "grade": "", "params": ""}}
    idx = {"family_order": [g1, g2], "clips": [clip("a0", g1, None, "TETRAS 1"), clip("a1", g1, "a0", "SARA 2"),
                                               clip("b0", g2, None, "3.4 Finger tapping")]}
    left = {"motion_capture_take": ["w"], "level0_chain_motion_capture": ["w1", "w2"]}
    man = S.public_manifest(idx, left, 123, "ab" * 32, 99, created="2026-11-10T00:00:00Z")
    assert man["public"] is True and man["clips"] == ["a0", "a1", "b0"]
    assert man["provenance"] == {"a0": "generated", "a1": "generated", "b0": "generated"}
    assert (man["n_symptom_clips"], man["n_level0_clips"]) == (1, 2)
    assert man["groups"] == [{"name": TX.GROUP_NAMES[g1], "n_clips": 2, "scales": ["TETRAS", "SARA"]},
                             {"name": TX.GROUP_NAMES[g2], "n_clips": 1, "scales": ["MDS-UPDRS"]}]
    assert man["left_out"] == left and man["n_left_out"] == 3 and man["created_utc"] == "2026-11-10T00:00:00Z"
    assert (man["page_bytes"], man["page_sha256"], man["frames"]) == (123, "ab" * 32, 90)
    # open-body port, wave 7 (design section 9): the body model's credit is Anny's; plan phase E (section 21): the
    # motion data under CC BY 4.0, no non-commercial or no-derivatives clause
    assert man["licence"] == S.PUBLIC_LICENCE and man["licence"].startswith("CC BY 4.0 ")
    assert "NC" not in man["licence"].split() and "ND" not in man["licence"] and "CC BY 4.0" in S.PUBLIC_NOTICE
    assert "generator" in man["credit"]
    assert "Anny" in man["credit"] and "CC0 1.0" in man["credit"] and "Apache License 2.0" in man["credit"]
    assert "SMPL" not in json.dumps(man) and "Max Planck" not in json.dumps(man)
    assert S.public_scan_text(json.dumps(man, ensure_ascii=False)) == []


def test_scales_named(S):
    TX = S.TX
    assert TX.scales_named("3.17 Rest tremor amplitude") == {"MDS-UPDRS"}
    assert TX.scales_named(["BFM severity x provoking factor", "DIS trunk"]) == {"BFMDRS", "DIS"}
    assert TX.scales_named("S-FMDRS trunk") == {"S-FMDRS"} and TX.scales_named("asterixis (Conn)") == {"Conn"}
    assert TX.scales_named("UDysRS part 3; AIMS item 5") == {"UDysRS", "AIMS"} and TX.scales_named("level 0") == set()


@pytest.mark.skipif(not HAVE_MA, reason="MoveAtlas SMPL+H model not found")
def test_public_build_from_exported_clips(S, tmp_path, monkeypatch):
    """--public --page-only --data-from: the page and the manifest of two exported procedural clips (their provenance
    measured), with the public texts, the scan clean, the sha256 of the page in the manifest; refused without
    --page-only."""
    from mdfm.synth import exam as EX
    root = str(tmp_path / "viewer").replace("\\", "/")
    monkeypatch.setattr(S.V, "VIEWER_ROOT", root)
    src, out = f"{root}/data_set", f"{root}/public"
    sel = [s for s in EX.exam_set() if s.clip_id in ("tremor_E1_rest_g0", "tremor_E1_rest_g2")]
    rec = S.CaptureReads()
    state = S.CodeState(sel, [])
    for s in sel:
        assert S.export_clip(s, src, S.SET_CATALOGUE, module_checks=True, code=state.of(s), recorder=rec)[1] is None
    monkeypatch.setattr(S, "set_specs", lambda set_name, modules: sel)
    # the exam set's bodies carry no background layer: the seated level 0 stands still for seconds, which the statue
    # check refuses since the page checks refuse (2026-11-11); this test reads the gallery, the checks are reported
    monkeypatch.setattr(S, "SKELETON_REPORTED_ONLY", S.SKELETON_CHECKS)
    S.main(["--set", "catalogue", "--page-only", "--public", "--out", out, "--data-from", src, "--no-record"])
    assert sorted(os.listdir(out)) == sorted([S.PUBLIC_PAGE, S.PUBLIC_MANIFEST])
    with open(f"{out}/{S.PUBLIC_MANIFEST}", encoding="utf-8") as f:
        man = json.load(f)
    with open(f"{out}/{S.PUBLIC_PAGE}", "rb") as f:
        raw = f.read()
    assert man["page_sha256"] == hashlib.sha256(raw).hexdigest() and man["page_bytes"] == len(raw)
    assert sorted(man["clips"]) == ["tremor_E1_rest_g0", "tremor_E1_rest_g2"]
    assert (man["n_symptom_clips"], man["n_level0_clips"], man["left_out"]) == (1, 1, {})
    html = raw.decode("utf-8")
    assert "stays internal" not in html and "<title>SYNTHETIC Movement gallery</title>" in html
    assert S.PUBLIC_LICENCE in html and "Anny" in html and "CC BY-NC" not in html and "SMPL" not in html   # wave 7
    assert "not been reviewed by a movement-disorder neurologist" in html and "Clips not shown: Every clip" in html
    assert S.public_scan(html, json.dumps(man, ensure_ascii=False)) == {}
    for m in S.CLIPDATA.finditer(html):
        d = json.loads(gzip.decompress(base64.b64decode(m.group(2))))
        assert "code" not in d and "provenance" not in d and "operator" not in d
    with pytest.raises(SystemExit, match="--page-only"):
        S.main(["--set", "catalogue", "--public", "--out", out, "--data-from", src])
