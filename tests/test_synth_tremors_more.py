"""Tests of the gap wave of the tremor family (mdfm/synth/tremors.py, module docstring 'Gap wave', task tremor_gaps):
catalogue rows T04, T05, T08, T10, T13, T16, P29, P30 and M07.

- the clip list: the new clips since 'catalogue_v1', family tremor, ids apart from the exam set's, appended after the
  clips that were there, their level-0 partners in the list or the exam set, on the same clock;
- every new clip: finite, no new joint-limit violation, no skate added on the feet it checks, the hand limits held,
  no hand inside the face or the examiner's finger on the reach;
- re-measurement against the level 0: frequency (spectral peak) and peak-to-peak in the grade's band, per row: the
  terminal crescendo (T04, the amplitude near the targets against mid-path), the kinetic tremor riding on the
  pronation-supination (T05), the ellipse of the mixed head tremor (T08), the knees, pelvis and neck of the standing
  tremor with the feet planted and the unsteadiness of grade 3 (T10), both hands at rest (T13), the clenched fist on
  the supported forearm, the tremor only once the fist is closed (T16), the forefoot about the heel, which stays on
  the floor, the knee nearly still, on the heel-down level 0 (P29), the tremor on the walking arm read blind with the
  swing band removed (P30), the twitches of orthostatic myoclonus counted on the pelvis, on whole-knot intervals
  inside their rate band (M07);
- determinism under the seed.
Every body is procedural, in the neutral shape; no lockbox, EVAL, CODY or care_pd record is read.
"""
import math
import os

import numpy as np
import pytest
import torch

from mdfm import paths

HAVE_MA = os.path.exists(paths.smplh_model("neutral"))
pytestmark = pytest.mark.skipif(not HAVE_MA, reason="MoveAtlas SMPL+H model not found")
torch.set_num_threads(2)

GAP = ("tremor_E3_essential_terminal_tetras2", "tremor_E3_essential_terminal_tetras3",
       "tremor_E4_kinetic_ps_tetras2", "tremor_E4_kinetic_ps_tetras3",
       "tremor_E1_head_mixed_g2", "tremor_E1_head_mixed_g3",
       "tremor_E1_essential_rest_g1a", "tremor_E1_essential_rest_g2",
       "isometric_tremor_E4b_fist_g0", "isometric_tremor_E4b_fist_present",
       "tremor_E1_leg_rest_g0", "tremor_E1_leg_rest_g1a", "tremor_E1_leg_rest_g2", "tremor_E1_leg_rest_g3",
       "tremor_E8_walk_g0", "tremor_E8_walk_rest_g2", "tremor_E8_walk_rest_g3",
       "standing_tremor_E10_posture_g0", "standing_tremor_E10_tetras9_g1", "standing_tremor_E10_tetras9_g2",
       "standing_tremor_E10_tetras9_g3",
       "orthostatic_myoclonus_E7_present")
BANDED = [c for c in GAP if c.startswith("tremor_") and not c.endswith("_g0")]


@pytest.fixture(scope="module")
def TR():
    from mdfm.synth import tremors
    return tremors


@pytest.fixture(scope="module")
def specs(TR):
    return {s.clip_id: s for s in TR.specs()}


@pytest.fixture(scope="module")
def exam_specs():
    from mdfm.synth import exam
    return {s.clip_id: s for s in exam.exam_set()}


@pytest.fixture(scope="module")
def built(specs, exam_specs):
    """{clip_id: (body, truth)} of the gap clips and of their level-0 partners."""
    out = {}
    for cid in GAP:
        out[cid] = specs[cid].build()
    for cid in {specs[c].level0 for c in GAP if specs[c].level0} - set(out):
        out[cid] = (specs.get(cid) or exam_specs[cid]).build()
    return out


def _rb(TR, specs, built, cid):
    s = specs[cid]
    body, truth = built[cid]
    return TR.readback(s, body, truth, built[s.level0][0] if s.level0 else None)


# ---------------------------------------------------------------- the clip list
def test_the_gap_clips_are_appended_to_the_list(TR, specs, exam_specs):
    from mdfm.synth.exam import SEGMENTS
    ids = [s.clip_id for s in TR.specs()]
    assert ids[-len(GAP):] == list(GAP)                      # appended after the clips that were there
    assert not set(GAP) & set(exam_specs)
    for cid in GAP:
        s = specs[cid]
        assert s.since == "catalogue_v1" and s.family == "tremor", cid
        assert s.segment in SEGMENTS or s.segment_title, cid
        assert s.params_text and s.grade_text and s.views
        text = s.params_text + s.grade_text + s.operator
        assert chr(0x2013) not in text and chr(0x2014) not in text  # no en or em dash
        if s.level0 is not None:
            assert s.level0 in specs or s.level0 in exam_specs, cid
    # the cm-band read of tests/test_synth_tremors.py takes the graded tremor_* ids: the word scale (TETRAS 9) and the
    # presence clips carry other prefixes
    for cid in GAP:
        if cid.startswith(("standing_tremor_", "isometric_tremor_", "orthostatic_myoclonus_")):
            assert specs[cid].item in ("level 0", "presence", "TETRAS 9"), cid


@pytest.mark.parametrize("cid", GAP)
def test_every_gap_clip_is_valid(cid, specs, built):
    s = specs[cid]
    body, truth = built[cid]
    assert torch.isfinite(body.rot6d).all() and torch.isfinite(body.transl).all()
    v = truth["validity"]
    assert v["limits_new"] == 0, (cid, v["limits_worst_deg"])
    assert v["skate_added_m_s"] < 1e-6, cid
    if v.get("hand_limits") is not None:
        assert v["hand_limits"]["ok"], cid
    if s.segment == "E3":
        hh = v["hand_head"]
        assert hh["max_depth_cm"] < 0.3 and hh["examiner_max_depth_cm"] < 0.3, (cid, hh)
    if v.get("balance") is not None:
        assert v["balance"]["ok"], cid
    if s.level0 is not None:
        l0, _ = built[s.level0]
        assert l0.T == body.T and l0.rate == body.rate, cid
    assert set(truth["sources"]) <= set(__import__("mdfm.synth.tremors", fromlist=["SOURCES"]).SOURCES), cid


@pytest.mark.parametrize("cid", BANDED)
def test_banded_clips_read_back_in_their_band(cid, TR, specs, built):
    body, truth = built[cid]
    p = truth["params"]
    rb = _rb(TR, specs, built, cid)
    lo, hi = TR.SCALES[p["scale"]][p["grade"]]
    for side, r in rb.items():
        lb = p["limbs"][side]
        assert abs(r["frequency_hz"] - lb["frequency_hz"]) < 0.05, (cid, side, r)
        assert lo < r["pp_cm_max"] < hi, (cid, side, r)
        assert abs(r["pp_cm_max"] / lb["cm_pp"] - 1.0) < 0.05, (cid, side, r)
        assert lb["chord_cm_at_full_envelope"][0] == pytest.approx(lb["cm_pp_at_unit_waveform"], rel=1e-6)


# ---------------------------------------------------------------- T04
@pytest.mark.parametrize("cid", ["tremor_E3_essential_terminal_tetras2", "tremor_E3_essential_terminal_tetras3"])
def test_the_terminal_crescendo_rises_near_the_targets(cid, TR, RC_mod, built):
    """The amplitude within the plateau of a target against mid-path (the catalogue's 'last 10 cm against
    mid-path'), on the fingertip against the clip's own centre."""
    body, truth = built[cid]
    p = truth["params"]
    base = TR.reach_base("left")[0]
    base = TR.centred_body(base, p["limbs"]["left"], truth["arrays"]["left_centre_rad"])
    D = RC_mod.fingertip(body, "left") - RC_mod.fingertip(base, "left")
    d = truth["arrays"]["target_distance_m"]
    env = truth["arrays"]["left_envelope"]
    f = p["limbs"]["left"]["frequency_hz"]
    n = int(round(100 / f)) + 1                                   # one period: mid-path is short at the reach's speed
    near, mid = [], []
    for k in range(0, body.T - n, 3):
        if env[k:k + n].min() > 0 and d[k:k + n].max() < TR.TERMINAL["plateau_cm"] / 100.0 + 0.01:
            near.append(TR.M.chord_pp(D[k:k + n]))
        far = (TR.TERMINAL["plateau_cm"] + TR.TERMINAL["near_cm"]) / 100.0
        if env[k:k + n].min() > TR.TERMINAL["floor"] - 1e-9 and d[k:k + n].min() > far:    # mid-path, gate on
            mid.append(TR.M.chord_pp(D[k:k + n]))
    ratio = np.median(near) / np.median(mid)
    assert len(near) > 20 and len(mid) > 5
    assert 0.8 / TR.TERMINAL["floor"] < ratio < 1.25 / TR.TERMINAL["floor"], ratio
    last10 = (d < TR.TERMINAL["near_cm"] / 100.0) & (env > 0)
    assert env[last10].max() == pytest.approx(1.0) and env[(d > 0.15) & (env > 0)].max() == pytest.approx(
        TR.TERMINAL["floor"], abs=1e-9)
    assert p["reach"]["profile"] == "terminal" and p["point"] == "index_tip"


@pytest.fixture(scope="module")
def RC_mod():
    from mdfm.synth import reach
    return reach


# ---------------------------------------------------------------- T05
@pytest.mark.parametrize("cid", ["tremor_E4_kinetic_ps_tetras2", "tremor_E4_kinetic_ps_tetras3"])
def test_the_simple_kinetic_tremor_rides_on_pronation_supination(cid, TR, built):
    """The tremor only during the task, the task the level 0's (the forearm turning at 2 Hz under it), the tremor
    read blind on the task joint with the task band removed (task_removed_readback, the forearm's frame: the
    shoulder's share above 3 cm is not in it)."""
    from mdfm.synth import hands as H
    body, truth = built[cid]
    l0, _ = built[TR.PS_LEVEL0]
    p = truth["params"]
    lb = p["limbs"]["left"]
    a, z = p["task"]["window_s"]
    t = body.times().numpy()
    tip = H.body_points(body, [H.tip_vertices()["left"]["middle"]])[:, 0]
    tip0 = H.body_points(l0, [H.tip_vertices()["left"]["middle"]])[:, 0]
    off = (t < a - TR.RAMP_S / 2) | (t > z + TR.RAMP_S / 2)
    assert np.abs(tip[off] - tip0[off]).max() < 1e-9               # no tremor outside the movement
    assert torch.equal(body.rot6d[:, 18], l0.rot6d[:, 18])          # the forearm's turn is the level 0's
    # the task read at the thumb tip, off the forearm's axis (open-body port, wave 2: the open body's canonical hand
    # lies along the forearm, and the middle fingertip near the axis turns at twice the task's rate)
    thumb = H.body_points(body, [H.tip_vertices()["left"]["thumb"]])[:, 0]
    assert abs(TR.M.dominant_frequency(thumb, 100.0, (1.0, 3.0)) - p["task"]["task_hz"]) < 0.1
    w = (a + 0.6, z - 0.6)
    blind = TR.task_removed_readback(body, hz=lb["frequency_hz"], window_s=w)
    blind0 = TR.task_removed_readback(l0, hz=lb["frequency_hz"], window_s=w)
    assert abs(blind["frequency_hz"] - lb["frequency_hz"]) < 0.05
    assert blind0["pp_cm_median"] < 0.15 * blind["pp_cm_median"]   # the task band out (its 4 Hz harmonic left)
    if TR.postural_shoulder_weight(lb["cm_pp"]) == 0.0:
        assert blind["pp_cm_median"] == pytest.approx(lb["cm_pp"], rel=0.10)
    else:
        # the forearm-frame share of a level with the shoulder joining: 0.52 on the open body at TETRAS 3
        # (scripts/openbody_wave2_checks.py; above 0.6 on SMPL+H, open-body port wave 2)
        assert 0.45 * lb["cm_pp"] < blind["pp_cm_median"] < lb["cm_pp"]


# ---------------------------------------------------------------- T08
@pytest.mark.parametrize("cid", ["tremor_E1_head_mixed_g2", "tremor_E1_head_mixed_g3"])
def test_the_mixed_head_tremor_runs_an_ellipse(cid, TR, built):
    body, truth = built[cid]
    l0, _ = built["dystonia_E1_level0"]
    changed = [j for j in range(22) if not torch.equal(body.rot6d[:, j], l0.rot6d[:, j])]
    assert changed == [12, 15]
    D = TR.RC.nose_point(body) - TR.RC.nose_point(l0)
    k = slice(200, 1300)
    x = D[k] - D[k].mean(0)
    s = np.linalg.svd(x, full_matrices=False)[1]
    assert s[1] / s[0] == pytest.approx(TR.MIXED_HEAD["pitch_share"], abs=0.08)      # minor over major axis
    ya, pa = truth["arrays"]["left_angle_rad"][0], truth["arrays"]["pitch_angle_rad"][0]
    f = truth["params"]["limbs"]["left"]["frequency_hz"]
    lag = int(round(100.0 / f / 4))                                                   # a quarter period
    c = [np.corrcoef(ya[200:1200], pa[200 + j:1200 + j])[0, 1] for j in (0, lag)]
    assert abs(c[0]) < 0.1 and c[1] > 0.95, c
    assert max(truth["params"]["mixed"]["pitch"]["centre_shift_deg_max"]) == 0.0


# ---------------------------------------------------------------- T13
@pytest.mark.parametrize("cid", ["tremor_E1_essential_rest_g1a", "tremor_E1_essential_rest_g2"])
def test_the_essential_rest_tremor_moves_both_hands_at_the_wrist(cid, TR, built):
    body, truth = built[cid]
    l0, _ = built["tremor_E1_rest_g0"]
    changed = [j for j in range(22) if not torch.equal(body.rot6d[:, j], l0.rot6d[:, j])]
    assert changed == [20, 21]                                  # the wrists only, the forearms still
    assert np.abs(body.joints().numpy()[:, 20:22] - l0.joints().numpy()[:, 20:22]).max() < 1e-12
    lbs = truth["params"]["limbs"]
    assert {s: lb["frequency_hz"] for s, lb in lbs.items()} == {"left": TR.HZ["essential"],
                                                                 "right": TR.HZ["essential_right"]}
    assert truth["params"]["condition"] == "rest" and truth["params"]["tremor_type"] == "essential"


# ---------------------------------------------------------------- T16
def test_the_isometric_tremor_shakes_the_clenched_fist_only(TR, specs, built):
    """The forearm on the chair arm ('sit', the elbow flexed: not the outstretched arm of postural tremor), the hand
    open then closing into the fist as in the level 0, no tremor before the fist is closed."""
    from mdfm.synth.base import static_body
    from mdfm.synth.handtasks import FIST, TASKS
    body, truth = built["isometric_tremor_E4b_fist_present"]
    l0, t0 = built["isometric_tremor_E4b_fist_g0"]
    I = TR.ISOMETRIC
    assert torch.equal(body.hand6d, l0.hand6d)                  # the hand closing as in the level 0
    assert TASKS["hand_movements"]["closed"] is FIST
    sit = static_body("sit", 0.01)
    for j in (13, 16, 18):                                       # the arm of 'sit': the forearm on the chair arm
        assert torch.equal(body.rot6d[:, j], sit.rot6d[:1, j].expand(body.T, 6)), j
    J = body.joints().numpy()[0]
    upper, fore = J[18] - J[16], J[20] - J[18]
    assert math.degrees(math.acos(upper @ fore / np.linalg.norm(upper) / np.linalg.norm(fore))) > 60.0
    assert abs(fore[1]) / np.linalg.norm(fore) < math.sin(math.radians(20.0))   # the forearm near horizontal
    t = body.times().numpy()
    share = truth["arrays"]["fist_share"]
    closed = I["fist_at_s"] + I["close_s"]
    assert share[t < I["fist_at_s"]].min() == 1.0 and share[t >= closed].max() < 1e-12
    env = truth["arrays"]["left_envelope"]
    assert env[t <= closed].max() == 0.0 and env[t >= closed + TR.RAMP_S].min() == 1.0
    before = t <= closed
    assert (body.rot6d[before] - l0.rot6d[before]).abs().max() < 1e-9     # the tremor waits for the clenched fist
    assert np.abs(body.joints().numpy()[before] - l0.joints().numpy()[before]).max() < 1e-12
    assert truth["grade"] == 1 and truth["scale"] == "presence" and "force" in truth["params"]["not_shown"]
    rb = _rb(TR, specs, built, "isometric_tremor_E4b_fist_present")["left"]
    assert abs(rb["frequency_hz"] - I["hz"]) < 0.05
    assert rb["pp_cm_max"] == pytest.approx(truth["params"]["isometric"]["level_cm"], rel=0.05)
    changed = [j for j in range(22) if not torch.equal(body.rot6d[:, j], l0.rot6d[:, j])]
    assert changed == [20]
    assert specs["isometric_tremor_E4b_fist_present"].segment_title == "fist clenched, forearm supported"


# ---------------------------------------------------------------- P29
def test_the_leg_rest_level0_puts_the_heel_down(TR, built):
    """'sit' rests the foot on its ball; the level 0 dorsiflexes the ankle until the heel is on the floor."""
    from mdfm.synth import contact as CT
    l0, t0 = built["tremor_E1_leg_rest_g0"]
    p = t0["params"]
    # the seated base: the heel off the floor, beyond the mesh tolerance (open-body port, wave 2: 1.1 mm on the open
    # body, above 5 mm on SMPL+H)
    assert p["heel_gap_sit_mm"] > 1000 * CT.MESH_TOL_M
    assert abs(p["heel_gap_mm"]) <= 1000 * CT.MESH_TOL_M + 1e-6
    assert 0.0 < p["heel_down_deg"] < 10.0
    assert t0["validity"]["contact"]["ok"] and t0["validity"]["limits_new"] == 0
    sit, _ = built["tremor_E1_rest_g0"]
    changed = [j for j in range(22) if not torch.equal(l0.rot6d[:, j], sit.rot6d[:, j])]
    assert changed == [1, 4, 7]
    assert torch.equal(l0.rot6d[0], l0.rot6d[-1])                         # held


def _heel_low(body, knots):
    """(n,) m: the lowest skinned heel vertex (contact.heel_toe) of the left foot above the floor of 'sit'."""
    from mdfm.synth import contact as CT
    from mdfm.synth.base import static_body
    x = CT.skin_points(body.local_R().numpy()[knots], body.transl.numpy()[knots], body.betas.numpy(), body.gender,
                       "left")
    sit = static_body("sit", 0.01)
    floor = CT.skin_points(sit.local_R().numpy()[:1], sit.transl.numpy()[:1], sit.betas.numpy(), sit.gender,
                           "left")[0, :, 1].min()
    heel = CT.heel_toe(int(body.gender))["left"][0]
    return x[:, heel, 1].min(1) - floor


@pytest.mark.parametrize("cid", ["tremor_E1_leg_rest_g1a", "tremor_E1_leg_rest_g2", "tremor_E1_leg_rest_g3"])
def test_the_leg_rest_tremor_lifts_the_forefoot_about_the_planted_heel(cid, TR, specs, built):
    """On the heel-down level 0: the heel stays on the floor (its lowest skinned vertex within 1 mm at every knot of
    the mesh check, which covers every pose of the clip), the knee nearly still, the toe the largest chord, in the
    grade's band."""
    body, truth = built[cid]
    l0, _ = built["tremor_E1_leg_rest_g0"]
    assert specs[cid].level0 == "tremor_E1_leg_rest_g0"
    v = truth["validity"]
    assert v["contact"]["ok"] and v["continuity"]["ok"]
    p = truth["params"]
    kc = int(round(TR.LEG_REST["mesh_check_s"] * body.rate)) + 1
    gap = _heel_low(body, np.arange(kc))
    assert np.abs(gap).max() < 0.001, (gap.min(), gap.max())           # the heel on the floor, every knot
    lm = TR.leg_motion_pp_cm(body, l0, "left", np.arange(kc))
    lo, hi = TR.SCALES["MDS-UPDRS"][p["grade"]]
    cm = p["limbs"]["left"]["cm_pp"]
    assert max(lm, key=lm.get) == "toe" and lo < max(lm.values()) < hi, lm
    assert lm["toe"] == pytest.approx(cm, rel=0.02)
    assert lm["knee"] < 0.1 * lm["toe"] and lm["heel"] < 0.15 * lm["toe"], lm
    assert lm == pytest.approx(p["leg_motion_pp_cm"], rel=1e-9)
    fp = p["foot_pose"]
    assert abs(fp["heel_lift_max_m"]) < 0.001 and fp["toe_lift_max_m"] > 0.5 * cm / 100.0
    o = truth["arrays"]["ankle_offset_deg"]
    assert o.min() >= 0.0 and o[:20].min() < 0.05 * o.max()     # one-sided: the forefoot back on the floor each cycle
    assert truth["arrays"]["ankle_deg"] == pytest.approx(o + p["limbs"]["left"]["heel_down_deg"])
    changed = [j for j in range(22) if not torch.equal(body.rot6d[:, j], l0.rot6d[:, j])]
    assert changed == [1, 4, 7]                                 # the left leg only, the right foot untouched
    toe = TR.point_path(body, "toe", "left")
    assert np.ptp(toe[:, 1]) * 100 > 0.5 * cm                   # the motion is vertical at the toe


# ---------------------------------------------------------------- P30
@pytest.mark.parametrize("cid", ["tremor_E8_walk_rest_g2", "tremor_E8_walk_rest_g3"])
def test_the_walking_rest_tremor_reads_back_blind(cid, TR, built):
    body, truth = built[cid]
    l0, _ = built["tremor_E8_walk_g0"]
    lb = truth["params"]["limbs"]["left"]
    blind = TR.swing_removed_readback(body)
    blind0 = TR.swing_removed_readback(l0)
    assert abs(blind["frequency_hz"] - lb["frequency_hz"]) < 0.05
    assert blind["pp_cm_median"] == pytest.approx(lb["cm_pp"], rel=0.05)
    assert blind["pp_cm_max"] == pytest.approx(lb["cm_pp"], rel=0.10)
    assert blind0["pp_cm_median"] < 0.1 * blind["pp_cm_median"]     # the swing band is out
    changed = [j for j in range(22) if not torch.equal(body.rot6d[:, j], l0.rot6d[:, j])]
    assert changed == [18] and torch.equal(body.transl, l0.transl)


def test_the_walk_level0_is_the_gait_clip(TR, built):
    from mdfm.synth import gait
    g, _ = {s.clip_id: s for s in gait.specs()}[TR.WALK_SOURCE].build()
    b, t = built["tremor_E8_walk_g0"]
    assert torch.equal(g.rot6d, b.rot6d) and torch.equal(g.transl, b.transl)
    assert t["params"]["walk"]["source"] == TR.WALK_SOURCE


# ---------------------------------------------------------------- T10
@pytest.mark.parametrize("g", [1, 2, 3])
def test_the_standing_tremor_bobs_the_knees_with_the_feet_planted(g, TR, built):
    cid = f"standing_tremor_E10_tetras9_g{g}"
    body, truth = built[cid]
    L = TR.STANDING_LEVEL[g]
    rb = TR.readback(None, body, truth)
    for k in ("knee_left", "knee_right"):
        assert abs(rb[k]["frequency_hz"] - L["hz"]) < 0.05
        assert rb[k]["pp_median"] == pytest.approx(L["knee_pp_deg"], rel=0.03), (cid, rb[k])
    assert rb["neck"]["pp_median"] > L["trunk_cm"] and rb["pelvis"]["pp_median"] > 0.0
    J = body.joints().numpy()
    assert np.abs(J[:, [7, 8, 10, 11]] - J[:1, [7, 8, 10, 11]]).max() < 1e-5      # ankles and feet still
    x = J[:, 0, 0] - built["standing_tremor_E10_posture_g0"][0].joints().numpy()[:, 0, 0]
    assert (100 * np.ptp(x) > 2 * TR.OT_SHIFT_CM - 0.1) == L["shifts"]
    assert truth["params"]["words"] == TR.STANDING_WORDS[g]


def test_the_tetras_standing_posture(TR, built):
    body, truth = built["standing_tremor_E10_posture_g0"]
    st = truth["params"]["standing"]
    assert 12.0 <= st["knee_width_cm"] <= 18.5                     # knee centres inside 10 to 20 cm, clear of its edges
    assert st["ankle_width_cm_realized"] == pytest.approx(TR.STANDING["ankle_width_cm"], abs=0.01)
    assert all(10.0 <= k <= 20.0 for k in st["knee_deg"])
    assert [TR.STANDING_LEVEL[g]["knee_pp_deg"] for g in (1, 2, 3)] == sorted(
        TR.STANDING_LEVEL[g]["knee_pp_deg"] for g in (1, 2, 3))


def test_grade3_standing_tremor_is_unsteady_and_still_balanced(TR, built):
    """Grade 3 'impairs stance': the sway narrows the extrapolated centre of mass's margins well below grade 2's and
    the posture's, inside the balance bounds; grades 1 and 2 do not sway."""
    from mdfm.synth import jerks
    b3 = built["standing_tremor_E10_tetras9_g3"][1]
    b2 = built["standing_tremor_E10_tetras9_g2"][1]
    u = b3["params"]["unsteadiness"]
    assert u["sway"]["realized_head_cm_pp"] == pytest.approx(TR.STANDING_SWAY["head_cm_pp"], rel=0.02)
    m2 = min(b2["validity"]["balance"]["xcom_margin_posterior_m"], b2["validity"]["balance"]["xcom_margin_anterior_m"])
    assert u["xcom_margin_min_m"] < m2 - 0.01 and u["xcom_margin_min_m"] < u["posture_xcom_margin_min_m"] - 0.03
    assert u["xcom_margin_min_m"] >= jerks.BALANCE_MARGIN_M
    for g in (1, 2, 3):
        bl = built[f"standing_tremor_E10_tetras9_g{g}"][1]["validity"]["balance"]
        assert bl["ok"] and bl["friction_max"] <= jerks.FRICTION_MAX and bl["vertical_grf_min_bw"] >= jerks.GRF_MIN_BW
        dz = built[f"standing_tremor_E10_tetras9_g{g}"][1]["arrays"]["pelvis_dz_m"]
        assert (np.ptp(dz) > 0) == TR.STANDING_LEVEL[g]["shifts"]
    assert b2["params"]["unsteadiness"] is None


def test_grade3_standing_sway_exceeds_the_layered_level0(TR, built):
    """Wave SH1d part B, review M3c: grade 3's forward and backward sway turns the whole body about the ankles (the
    head travelling more than the pelvis, not a block slide), at least twice the head's travel that the standing
    background layer gives the level 0 (physio's layer of the series, laid by its own stand_sway), and the arms are
    held out from the sides; the layer lays the same sway on the whole series once physio leaves it out of SKIP."""
    from mdfm.synth import physio as PH
    b0, _ = built["standing_tremor_E10_posture_g0"]
    body, truth = built["standing_tremor_E10_tetras9_g3"]
    L = PH.layer_prefix(PH.make_layer(PH.series_seed("standing_tremor_E10_posture_g0"), "standing"), b0.T)
    lay = TR.ankle_sway(b0, L["sway_ap_m"])
    J0, Jl, J3 = b0.joints().numpy(), lay.joints().numpy(), body.joints().numpy()
    layer_head = np.ptp(Jl[:, 15, 2] - J0[:, 15, 2])
    own_head, own_pelvis = np.ptp(J3[:, 15, 2] - J0[:, 15, 2]), np.ptp(J3[:, 0, 2] - J0[:, 0, 2])
    assert own_head >= 2.0 * layer_head, (own_head, layer_head)
    assert own_head > 1.5 * own_pelvis                       # an inverted pendulum about the ankles
    g = truth["arrays"]["arm_guard_share"]
    assert g.max() <= 1.0 and g[len(g) // 2:].min() >= TR.STANDING_SWAY["arm_base_share"] - 1e-9
    for j, sg in ((16, 1.0), (17, -1.0)):                    # the upper arms out from the trunk
        out = J3[:, j + 2, 0] - J3[:, j, 0]
        ref = J0[:, j + 2, 0] - J0[:, j, 0]
        assert (sg * (out - ref))[len(g) // 2:].min() > 0.02


# ---------------------------------------------------------------- M07
def test_orthostatic_myoclonus_twitches_are_counted_on_the_pelvis(TR, specs, built):
    body, truth = built["orthostatic_myoclonus_E7_present"]
    l0, _ = built["tremor_E7_orthostatic_g0"]
    p = truth["params"]
    rb = _rb(TR, specs, built, "orthostatic_myoclonus_E7_present")
    assert rb["n_twitches"] == p["n_twitches"] >= 30
    for a, b in zip(rb["onsets_s"], p["onsets_s"]):
        assert 0.0 <= a - b <= 0.04
    lo, hi = TR.ORTHO_MYO["rate_hz"]
    iv = np.diff(p["onsets_s"])
    assert 1.0 / hi <= iv.min() and iv.max() <= 1.0 / lo                        # every realized rate in the band
    k = np.round(np.asarray(p["onsets_s"]) * body.rate)
    assert np.allclose(k / body.rate, p["onsets_s"], atol=1e-12)                # onsets on knots
    klo, khi = TR.ortho_myo_interval_knots(body.rate)
    assert np.diff(k).min() >= klo and np.diff(k).max() <= khi and len(set(np.diff(k))) > 5
    assert lo < rb["rate_hz_median"] < hi
    assert p["burst_ms"] < 50.0 and p["rise_s"] < 0.1
    free = truth["arrays"]["twitch_profile"] == 0.0
    assert torch.equal(body.rot6d[free], l0.rot6d[free]) and torch.equal(body.transl[free], l0.transl[free])
    J = body.joints().numpy()
    assert np.abs(J[:, [7, 8, 10, 11]] - J[:1, [7, 8, 10, 11]]).max() < 1e-5


# ---------------------------------------------------------------- determinism
@pytest.mark.parametrize("cid", ["tremor_E1_head_mixed_g2", "standing_tremor_E10_tetras9_g3",
                                 "orthostatic_myoclonus_E7_present", "tremor_E8_walk_rest_g2"])
def test_the_gap_build_is_deterministic(cid, specs, built):
    body, _ = specs[cid].build()
    b0, _ = built[cid]
    assert torch.equal(body.rot6d, b0.rot6d) and torch.equal(body.transl, b0.transl)


def test_another_seed_gives_other_twitches(TR):
    a, ta = TR.ortho_myoclonus(1)
    b, tb = TR.ortho_myoclonus(2)
    assert ta["params"]["onsets_s"] != tb["params"]["onsets_s"]
