"""Tests of the skeleton export of the synthetic patient (task SKELV): mdfm/synth/skeleton.py and
scripts/render_skeleton_catalogue.py.

- counts: every clip carries the 52 SMPL+H joints and the two head landmarks (22 joints without hands); the hands
  are 'animated' for a hands.HandBody and 'rest' otherwise; floor(span x 30) + 1 frames at t_k = k / 30;
- the bone list: parents of the 22 body joints are smplh.PARENTS, every finger chain starts at its wrist, the
  landmarks hang on the head joint, every parent precedes its child; bone lengths stay those of the rest skeleton;
- the positions are the body's: finite, and at a frame on a knot the 22 body joints are Body.joints() there;
- why the hands and the head: a pronation-supination moves the hand joints and not the wrist, a neck rotation moves
  the nose landmark and not the head joint;
- a level-0 clip and its graded partner share the time base (30 fps from t = 0);
- no vertex, face, shape or model array in a record or a data file; positions round-trip at 1 mm;
- the page's structure (scripts/skeleton_text.py, in English since the owner's decision of 2026-11-03) covers every
  spec of the exam set: its family, scale item and series have a label, its segment a title, its operator and grade
  texts pass as the spec writes them; the page embeds its clips, carries SYNTHETIC and no em-dash;
- wave SH1d: a catalogue data file carries the declared jerk events of its truth and the floor over its standing
  frames, the exam export keeps its median floor.
Bodies are procedural (exam specs whose base is the procedural sit or stand, no AMASS read, no lockbox, EVAL, CODY
or care_pd record); files go to pytest's temporary folder only.
"""
import dataclasses
import gzip
import importlib.util
import json
import math
import os
import re

import numpy as np
import pytest

from mdfm import paths

HAVE_MA = os.path.exists(paths.smplh_model("neutral"))
pytestmark = pytest.mark.skipif(not HAVE_MA, reason="MoveAtlas SMPL+H model not found")

FORBIDDEN_KEYS = ("vert", "face", "betas", "shape", "weights", "posedirs", "regressor", "template", "kintree")


def _script():
    spec = importlib.util.spec_from_file_location("render_skeleton_catalogue_script",
                                                  os.path.join(paths.TREE, "scripts", "render_skeleton_catalogue.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def specs():
    from mdfm.synth import exam
    return {s.clip_id: s for s in exam.exam_set()}


@pytest.fixture(scope="module")
def built(specs):
    """(body, truth, skeleton) of a few procedural clips."""
    from mdfm.synth import skeleton as SK
    out = {}
    for cid in ("tremor_E1_rest_g0", "tremor_E1_rest_g2", "brady_E4_ps_level0", "brady_E4_ps_interruptions_g2",
                "brady_E4a_ft_level0", "dystonia_E7_level0", "dystonia_E7_neck_rotation_left_g2"):
        b, t = specs[cid].build()
        out[cid] = (b, t, SK.skeleton(b))
    return out


def test_counts_and_hand_pose(built):
    from mdfm.synth import skeleton as SK
    for cid, (b, _, sk) in built.items():
        assert sk.n_joints == 52 and sk.n_points == 54, cid
        assert sk.positions.shape == (sk.n_frames, 54, 3)
        assert sk.names[:52] == SK.JOINT_NAMES and sk.landmarks == ("head_centre", "nose")
    assert built["brady_E4a_ft_level0"][2].hand_pose == "animated"
    assert built["tremor_E1_rest_g2"][2].hand_pose == "rest"
    b = built["tremor_E1_rest_g2"][0]
    sk22 = SK.skeleton(b, hands=False, landmarks=False)
    assert sk22.n_points == 22 and sk22.hand_pose is None and not sk22.hands


def test_frame_count_and_clock(built):
    for cid, (b, _, sk) in built.items():
        assert sk.fps == 30.0
        assert sk.n_frames == math.floor(b.duration * 30 + 1e-9) + 1, cid
        assert abs((sk.n_frames - 1) / 30 - b.duration) < 1 / 30, cid
        np.testing.assert_allclose(sk.times(), np.arange(sk.n_frames) / 30.0)
    b, _, sk = built["tremor_E1_rest_g2"]
    assert b.duration == 45.0 and sk.n_frames == 1351


def test_bone_list(built):
    from mdfm.body.smplh import PARENTS
    from mdfm.synth import hands as H
    from mdfm.synth import skeleton as SK
    sk = built["brady_E4a_ft_level0"][2]
    par = sk.parents
    assert par[:22] == tuple(PARENTS) and par[0] == -1
    assert all(0 <= par[j] < j for j in range(1, 52))
    for side in H.SIDES:
        for f in H.FINGERS:
            j = H.hand_joint(side, f, 0)
            assert par[j] == H.WRIST[side] and par[j + 1] == j and par[j + 2] == j + 1
    assert par[52:] == (SK.HEAD, SK.HEAD)
    bones = sk.bones()
    assert len(bones) == 53 and (0, -1) not in bones and all(p >= 0 for _, p in bones)
    assert bones == SK.bones(par)


def test_bone_lengths_rigid(built):
    """Forward kinematics keeps every bone at its rest length, the hand bones included."""
    for cid in ("brady_E4a_ft_level0", "tremor_E1_rest_g2"):
        sk = built[cid][2]
        P = sk.positions
        for j, p in sk.bones():
            L = np.linalg.norm(P[:, j] - P[:, p], axis=-1)
            assert L.max() - L.min() < 1e-9, (cid, j)
            assert L.min() > 0.005, (cid, j)


def test_positions_finite_and_on_knots(built):
    for cid, (b, _, sk) in built.items():
        assert np.all(np.isfinite(sk.positions)), cid
        J = b.joints().numpy()
        for k in range(0, sk.n_frames, 3):                  # t_k = k / 30 lands on knot 10 k / 3 of the 100 Hz clock
            knot = 10 * k // 3
            np.testing.assert_allclose(sk.positions[k, :22], J[knot], atol=1e-9, err_msg=cid)


def test_pronation_moves_the_hand_not_the_wrist(built):
    sk = built["brady_E4_ps_level0"][2]
    P = sk.positions
    assert np.ptp(P[:, 20], axis=0).max() < 1e-6                     # the left wrist joint stays put
    assert np.ptp(P[:, 22:37], axis=0).max() > 0.01                  # the left hand turns
    assert np.ptp(P[:, 37:52], axis=0).max() < 1e-6                  # the right hand does not


def test_neck_rotation_moves_the_nose_not_the_head_joint(built):
    a = built["dystonia_E7_level0"][2].positions
    b = built["dystonia_E7_neck_rotation_left_g2"][2].positions
    k = a.shape[0] - 1
    head_move = np.linalg.norm(a[k, 15] - b[k, 15])
    nose_move = np.linalg.norm(a[k, 53] - b[k, 53])
    assert head_move < 0.02 and nose_move > 3 * head_move, (head_move, nose_move)
    na, nb = a[k, 53] - a[k, 15], b[k, 53] - b[k, 15]
    ang = math.degrees(math.atan2(nb[0], nb[2]) - math.atan2(na[0], na[2]))
    assert ang > 10.0                                                  # turned toward the left (+x)
    from mdfm.synth.skeleton import head_landmarks
    np.testing.assert_allclose(np.linalg.norm(nb), np.linalg.norm(head_landmarks()["nose"]), atol=1e-9)


def test_partner_shares_the_time_base(specs, built):
    for cid in ("tremor_E1_rest_g2", "brady_E4_ps_interruptions_g2"):
        l0 = specs[cid].level0
        a, b = built[cid][2], built[l0][2]
        assert a.fps == b.fps == 30.0
        n = min(a.n_frames, b.n_frames)
        np.testing.assert_array_equal(a.times()[:n], b.times()[:n])
    assert built["tremor_E1_rest_g2"][2].n_frames == built["tremor_E1_rest_g0"][2].n_frames


def _keys(x, out):
    if isinstance(x, dict):
        for k, v in x.items():
            out.append(str(k))
            _keys(v, out)
    elif isinstance(x, list):
        for v in x:
            _keys(v, out)
    return out


def test_record_has_no_vertex_and_round_trips(built):
    from mdfm.synth import skeleton as SK
    sk = built["brady_E4a_ft_level0"][2]
    for delta in (False, True):
        rec = SK.record(sk, mm=1, delta=delta)
        keys = _keys(rec, [])
        assert not [k for k in keys if any(f in k.lower() for f in FORBIDDEN_KEYS)]
        flat = rec["positions_mm_delta" if delta else "positions_mm"]
        assert len(flat) == sk.n_frames * sk.n_points * 3 and all(isinstance(v, int) for v in flat[:500])
        P = SK.decode_positions(rec)
        assert np.abs(P - sk.positions).max() <= 0.0005 + 1e-12
        np.testing.assert_array_equal(P, SK.quantize_mm(sk.positions) / 1000.0)
    with pytest.raises(ValueError):
        SK.quantize_mm(np.array([[[np.nan, 0.0, 0.0]]]))


def test_data_file_and_page(specs, tmp_path):
    S = _script()
    out = str(tmp_path / "skeleton_test").replace("\\", "/")
    for cid in ("tremor_E1_rest_g0", "tremor_E1_rest_g2"):
        meta, bad = S.export_clip(specs[cid], out)
        assert bad is None and meta["level0"] == specs[cid].level0
    p = S.data_path(out, "tremor_E1_rest_g2")
    assert os.path.basename(p).startswith("SYNTHETIC_") and p.endswith(".json.gz")
    with gzip.open(p, "rb") as f:
        d = json.loads(f.read().decode("utf-8"))
    keys = _keys({k: v for k, v in d.items() if k != "notice"}, [])
    assert not [k for k in keys if any(f in k.lower() for f in FORBIDDEN_KEYS)]
    assert d["label"] == "SYNTHETIC" and d["skeleton"]["n_points"] == 54 and d["level0"] == "tremor_E1_rest_g0"
    assert len(d["scene_boxes_m"]) == 4 and d["walk"] is None
    assert d["caption"][0].startswith(f"{specs['tremor_E1_rest_g2'].operator} | MDS-UPDRS 3.17 grade 2")
    assert specs["tremor_E1_rest_g2"].operator.startswith("tremor, rest, left forearm")   # the exam's own words
    sel = [specs["tremor_E1_rest_g0"], specs["tremor_E1_rest_g2"]]
    metas, blobs = S.load_all(out, sel)
    page, nbytes, total, notes = S.build_page(out, metas, blobs, [])
    assert os.path.basename(page).startswith("SYNTHETIC_") and notes == {}
    with open(page, encoding="utf-8") as f:
        html = f.read()
    assert html.count('class="clipdata"') == 2 and "SYNTHETIC" in html and '<html lang="en">' in html
    code = re.sub(r'(<[^>]*class="clipdata"[^>]*>)[^<]*', r"\1", html)   # the page without its embedded data
    #   (base64 text, where any three letters can occur)
    assert "\u2014" not in html and "__INDEX_JSON__" not in html and "cdn" not in code.lower()


def test_page_structure_covers_the_exam_set(specs):
    from mdfm.synth import exam
    S = _script()
    for s in specs.values():
        named = s.item in S.ITEM_NAMES or s.item.startswith(("TWSTRS ", "DIS "))     # a scale's own name heads
        assert named and s.family in S.FAMILY_NAMES and s.segment in exam.SEGMENTS, s.clip_id
        S.series_name(S.series_key(s.clip_id), s.operator)                     # raises without a label
        meta = {"clip_id": s.clip_id, "family": s.family, "segment": s.segment,
                "segment_title": exam.segment_title(s), "item": s.item, "operator": s.operator, "grade": s.grade,
                "grade_text": s.grade_text, "params_text": s.params_text, "level0": s.level0, "since": s.since}
        text, notes = S.page_texts(meta)
        assert not notes and text["operator"] == s.operator and text["grade"] == s.grade_text, s.clip_id
        assert text["segment"] == exam.segment_title(s) and text["chip"], s.clip_id
        assert all(ord(ch) < 128 for v in text.values() for ch in str(v)), s.clip_id    # no accented letter
    assert S.series_key("tremor_E1_rest_g1b") == "tremor_E1_rest"
    assert S.series_key("freezing_E8_g2_akinesia") == "freezing_E8_akinesia"
    assert S.series_key("dystonia_E7_trunk_lateral_left_g3_duration2") == "dystonia_E7_trunk_lateral_left_duration2"
    assert S.series_name("dystonia_E7_trunk_lateral_left_duration2") == \
        "lateral flexion to the left, grade 3, DIS duration 2"
    assert S.series_name("tremor_E2_postural", "tremor, postural, left hand") == "postural, left hand"
    assert S.grade_chip({"clip_id": "tremor_E1_rest_g1a", "grade": 1, "grade_text": ""}) == \
        "1 (above the estimator floor)"
    assert S.grade_chip({"clip_id": "dystonia_E7_start_arm_abduction30_elbow100", "grade": "level",
                         "grade_text": "level 30 and 100 degrees"}) == "30 and 100 degrees"


def _reference(monkeypatch):
    """The synthetic patient on the SMPL+H reference, behind its flag: the family modules these tests build (jerks,
    ataxia, physio) still read the SMPL+H mesh until their wave of docs/OPEN_BODY_PORT_DESIGN.md ports them."""
    from mdfm.synth import openbody as OB
    monkeypatch.setenv(OB.BACKEND_ENV, OB.SMPLH_REFERENCE)
    monkeypatch.setenv(OB.REFERENCE_ENV, "1")


def test_catalogue_export_carries_events_and_standing_floor(specs, tmp_path, monkeypatch, request):
    """Wave SH1d: the catalogue export writes the declared jerk events of the truth ('events_s': jerks.event_span of
    each event, on the clip's clock) and the floor over the standing frames; the exam export keeps the median rule
    and no events. On the reference until waves 3 and 7 port jerks and physio, whose mesh caches are not keyed by
    the body model yet (cleared here, after the open-body tests of this process)."""
    _reference(monkeypatch)
    from mdfm.synth import involuntary, jerks, physio

    def clear():
        for mod in (physio, involuntary, jerks):
            for obj in list(vars(mod).values()):
                if callable(getattr(obj, "cache_clear", None)):
                    obj.cache_clear()
    clear()
    request.addfinalizer(clear)
    S = _script()
    from mdfm.synth import catalogue as CAT
    from mdfm.synth import jerks as JK
    s = {x.clip_id: x for x in CAT.module_specs("jerks")}["jerk_myoclonus_E1_generalized_f3_a2"]
    body, truth = s.build()
    fake = dataclasses.replace(s, build=lambda: (body, truth))
    out = str(tmp_path / "cat").replace("\\", "/")
    meta, bad = S.export_clip(fake, out, S.SET_CATALOGUE, module_checks=True, code=S.CodeState([s], []).of(s))
    spans = [[round(a, 4), round(b, 4)] for a, b in (JK.event_span(e) for e in truth["params"]["events"])]
    assert bad is None and len(spans) == truth["params"]["n_events"] > 0 and meta["events_s"] == spans
    with gzip.open(S.data_path(out, s.clip_id), "rb") as f:
        d = json.loads(f.read().decode("utf-8"))
    assert d["events_s"] == spans and d["code"]["export_format"] == S.EXPORT_FORMAT == "skeleton_v7_open_body"
    assert d["floor"]["standing_frames"] > 0.5 * d["skeleton"]["n_frames"] and "still" in d["floor"]["rule"]
    fl = d["floor"]
    assert abs(fl["floor_y_m"] - (fl["lowest_joint_standing_min_m"] - fl["sole_offset_m"])) < 1e-12
    exam_out = str(tmp_path / "exam").replace("\\", "/")
    meta, bad = S.export_clip(specs["tremor_E1_rest_g0"], exam_out)
    assert bad is None and "events_s" not in meta and "standing_frames" not in meta["floor"]
    assert meta["floor"]["rule"].startswith("median over the frames")
