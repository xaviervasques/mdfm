"""Tests of the abnormal postures of the symptom catalogue, mdfm/synth/postures.py (task CAT-POST, catalogue_v1):

- the photograph angles (Tinazzi et al. 2022 task force; skin landmarks of the skinned mesh) read the level-0 stand
  inside the normal bands of the trunk, and read back the held angle of each axial posture at its target, in its
  task force band, from the first knot; the thoracic camptocormia is reported unreachable inside the table; in a
  sagittal bend the arms hang (the humerus keeps the level 0's world direction), never riding the trunk;
- MDS-UPDRS 3.13: the stoop holds its total camptocormia angle and is undone at the request to stand straight for
  grade 2, only partly for grade 3; both feet stay where the base has them (skinned soles within 5 mm, no skate),
  no new joint-limit violation, no jump;
- cervical dystonia: the share of time with the posture present follows the TWSTRS duration score, the maximal
  excursion is the TWSTRS level and is reached; intermittent bouts drift in slowly and are corrected back quickly;
  the jerky pulls are irregular sawtooth events at 0.1 to 0.5 Hz, the course never above the maximal excursion nor
  flat on it; the shifts move the head 4 cm or more without tilting it; the geste antagoniste reduces the posture by
  the TWSTRS C share while the hand rests on the cheek: the right hand on the right cheek, the wrist lateral to the
  face and below the jaw line, beside the neck and never in front of it, the wrist's flexion and deviation inside
  the JARM ranges over the whole reach (wrist_bend_deg, its signs checked on rotated hands), the hand and the forearm
  off the trunk, the hand folding to the face without sweeping out in front and without a snap, the fingers curled and
  the palm turned toward the face (wave SH1c: the open hand read as a raised hand in the front view);
- limb dystonia: the equinovarus foot touches the floor (never below, within 5 mm) while the other foot stands; seated
  the heel rises while the foot keeps its place (no slide forward); standing the knee lift is capped and the pelvis
  hikes; the writing hand moves along the line, the hands never go through the table, the cramp's elbow lifts and
  nothing differs from the level 0 before the latency; the writing hand rests on the paper by the lowest point of its
  palm and its ring and little fingers and the pen's tip touches the paper within words, on the level 0s of both hands
  and on the cramp, overflow and mirror clips (table_report; wave SH1c: the hand hovered over the table), the forearm
  free above the table (forearm_gap_mm, as the texts say: review of wave SH1c); the table check runs again from the
  truth's record (table_report_of); the task-specific arm posture is absent at
  rest; the fixed dystonia does not change and its fist flexes the ring and little fingers most;
- generalized dystonia: the arms twist without large abductions; paroxysmal kinesigenic dyskinesia: the attack
  starts after lift-off and lasts its duration, the feet untouched, the folded arms unfolded once the trunk is near
  upright, each on its own clock with the skins apart (motions.unfold_to_stand, the validity's 'fold'), clear of the
  armrests and the thighs, and no arm joint snaps (a joint at its bound scales its own group of terms, the scale
  smoothed: fit_groups, smooth_scale), and no forearm or hand inside the trunk or a thigh, read on the skin by the
  winding number too (trunk_report, thigh_report, arm_part_gap, guard_rows; wave SH1c: the dystonic forearm lay
  inside the abdomen, a choreic hand inside the thigh while reach.clearance read it outside); the seated paroxysmal
  excerpt keeps its forearms off the armrests and every part of the body out of the backrest (backrest_report; wave
  SH1c: the right upper arm lay inside it);
- the SH2 dystonia table (limits_ext.py): the raised shoulder carries a hanging arm (cervical shoulder elevation,
  segmental dystonia), the equinovarus foot inverts past the healthy table's room and the toes curl, the mirror
  dystonia stays inside the table without a jump at its start (the table clearance's running maximum no longer
  wraps the clip's end onto its start), and so does the overflow dystonia (equal to its level 0 until the cramp), the
  geste's finger pads rest on the cheek 2 to 4 mm from the skin, the palm and the forearm at least 1 cm off, nothing
  inside the head;
- the clip set: unique ids, every level 0 in the exam set or here, the level 0 on the same time base, no NaN,
  determinism.
Every body is procedural in the neutral shape; no lockbox, EVAL, CODY or care_pd record is read.
"""
import os

import numpy as np
import pytest
import torch

from mdfm import paths

HAVE_MA = os.path.exists(paths.smplh_model("neutral"))
pytestmark = pytest.mark.skipif(not HAVE_MA, reason="MoveAtlas SMPL+H model not found")
torch.set_num_threads(2)


@pytest.fixture(scope="module")
def P():
    from mdfm.synth import postures
    return postures


@pytest.fixture(scope="module")
def stand():
    from mdfm.synth.base import static_body
    return static_body("stand", 4.0)


@pytest.fixture(scope="module")
def stand15():
    from mdfm.synth.base import static_body
    return static_body("stand", 15.0)


@pytest.fixture(scope="module")
def geste(P, stand15):
    """score -> (Body, truth) of geste_antagoniste on stand15 (rotation left, grade 2, seed 3), built once."""
    cache = {}

    def get(score):
        if score not in cache:
            cache[score] = P.geste_antagoniste(stand15, "rotation_left", 2, score, seed=3)
        return cache[score]
    return get


def _ok(v):
    """The checks of scripts/render_mesh_video.py failed_checks: no new limit violation, and each of contact,
    continuity and hand_limits ok when present."""
    bad = ["limits_new"] if v.get("limits_new") else []
    return bad + [k for k in ("contact", "continuity", "hand_limits") if v.get(k) is not None and not v[k]["ok"]]


def _on_the_paper(P, t, strict=True):
    """A table_report: the writing hand on its edge on the paper, the pen's tip on the paper within words and lifted by
    at most PEN['lift_m'] between them (strict: a level 0), or, under a posture, the hand resting on the paper by
    whatever of it or of the pen lies lowest; every resting hand on the table, nothing through the table or the
    paper."""
    tol = P.SCENE_TOL_M
    assert t["ok"] and t["over_paper"] and t["depth_mm"] <= 1000 * tol, t
    assert t["hand_gap_mm"][0] / 1000 >= -tol and t["pen_gap_mm"][0] / 1000 >= -tol
    if strict:
        assert -tol <= t["edge_gap_mm"][0] / 1000 and t["edge_gap_mm"][1] / 1000 <= P.WRITE_EDGE_GAP_M + tol
        assert t["pen_gap_mm"][1] / 1000 <= P.PEN["touch_m"] + P.PEN["lift_m"] + tol
        assert t["pen_gap_word_mm"][1] / 1000 <= P.PEN["touch_m"] + tol
    else:
        assert max(abs(x) for x in t["rest_low_mm"]) / 1000 <= tol
    for g in t["rest_gap_mm"].values():
        assert -tol <= g[0] / 1000 and g[1] / 1000 <= P.WRITE_REST_GAP_M + tol


def _finite(b):
    return bool(torch.isfinite(b.rot6d).all() and torch.isfinite(b.transl).all())


def _snaps(out, base, joints, thr=3.0):
    """{joint: largest step deg}: the joints of out that turn more than thr deg in one knot (300 deg/s at 100 Hz)
    beyond the base's own step at that knot."""
    from mdfm.synth.body import rotation_angle_deg

    def steps(b):
        L = b.local_R()
        return rotation_angle_deg(L[1:].transpose(-1, -2) @ L[:-1]).numpy()
    so, sb = steps(out), steps(base)
    return {j: float((so[:, j] - sb[:, j]).max()) for j in joints if ((so[:, j] - sb[:, j]) > thr).any()}


def _humerus_angle(base, out, s, e):
    """(T,) deg between the humerus (shoulder to elbow joint) of out and of base."""
    Jb, Jo = base.joints().numpy(), out.joints().numpy()
    ub, uo = Jb[:, e] - Jb[:, s], Jo[:, e] - Jo[:, s]
    c = (ub * uo).sum(-1) / np.linalg.norm(ub, axis=-1) / np.linalg.norm(uo, axis=-1)
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


def _from_vertical(body, s, e):
    J = body.joints().numpy()
    u = J[:, e] - J[:, s]
    return np.degrees(np.arccos(np.clip(-u[:, 1] / np.linalg.norm(u, axis=-1), -1.0, 1.0)))


# ---------------------------------------------------------------- photograph angles
def test_the_level0_stand_reads_normal_trunk_angles(P, stand):
    a = P.axial_angles(stand, knots=[0, stand.T - 1])
    for kind, key in (("lumbar", "tcc"), ("thoracic", "ucc"), ("lateral", "lateral")):
        assert P.band_of(kind, float(a[key][0])) == "normal"
    np.testing.assert_allclose(a["tcc"][0], a["tcc"][-1], atol=1e-9)
    assert set(P.landmark_ids()) == set(P.LANDMARKS)


@pytest.mark.parametrize("shape,deg,band", [("lumbar", 45.0, "camptocormia (lumbar fulcrum)"),
                                            ("lateral", 7.5, "lateral trunk flexion"),
                                            ("neck", 52.0, "antecollis")])   # NECK_OPEN_DEG (wave 4, D53)
def test_axial_postures_hold_their_task_force_angle(P, stand, shape, deg, band):
    out, tr = P.axial_posture(stand, shape, deg, 10.0 if shape == "lumbar" else 0.0)
    p = tr["params"]
    key = P.MEASURE_OF[shape]
    held = float(P.axial_angles(out, knots=[out.T - 1])[key][0])
    assert p["reachable"] and abs(held - deg) < 0.05 and P.band_of(shape, held) == band
    assert abs(float(tr["arrays"]["axial_" + key][0]) - held) < 0.01          # held from the first knot (review M10)
    assert _ok(tr["validity"]) == [] and tr["validity"]["skate_added_m_s"] < 1e-9 and _finite(out)
    if shape == "lumbar":                                    # the arms hang (review B1)
        for s_, e_ in ((16, 18), (17, 19)):
            assert _humerus_angle(stand, out, s_, e_).max() < 1.0


def test_thoracic_camptocormia_is_out_of_the_table(P, stand):
    amp, reach, got = P.solve_amplitude(stand, "thoracic", P.OPEN_BAND_FACTOR * P.TF_BANDS["thoracic"]["cut"], 5.0)
    assert not reach and got < P.TF_BANDS["thoracic"]["cut"]


# ---------------------------------------------------------------- MDS-UPDRS 3.13
@pytest.mark.parametrize("grade", [2, 3])
def test_parkinsonian_posture_holds_and_corrects_as_its_grade(P, stand, grade):
    out, tr = P.parkinsonian_posture(stand, grade, ask_s=1.5)
    p = tr["params"]
    tcc = tr["arrays"]["axial_tcc"]
    base = p["base_angles_deg"]["tcc"]
    assert abs(tcc[0] - P.PD_313[grade]["tcc"]) < 0.05 and p["reachable"]
    left = (tcc[-1] - base) / (tcc[0] - base)                 # the share of the stoop left after the request
    assert abs(left - (1.0 - P.PD_313[grade]["correct"])) < 0.08
    v = tr["validity"]
    assert _ok(v) == [] and v["skate_added_m_s"] < 1e-9 and v["contact"]["left"]["moved_low_max_m"] < 0.005
    J0, J1 = stand.joints(), out.joints()
    assert float((J1[:, [7, 8, 10, 11]] - J0[:, [7, 8, 10, 11]]).abs().max()) < 0.006   # the feet stay put
    assert p["stand"]["com_minus_ankles_m"]["max_error"] < 1e-6 and _finite(out)
    assert _humerus_angle(stand, out, 16, 18).max() < 1.0 and p["stand"]["arms_hang"]["left"]["share_min"] == 1.0


# ---------------------------------------------------------------- cervical dystonia
@pytest.mark.parametrize("dg", [1, 2, 4, 5])
def test_cervical_course_follows_the_twstrs_duration_and_level(P, stand15, dg):
    from mdfm.synth import grades as G
    lvl = G.twstrs_level_deg("rotation", 2, None)
    out, tr = P.cervical_course(stand15, [("neck", "rotation_left", "left", lvl)], duration_grade=dg, seed=7)
    share, exc = G.twstrs_duration(dg, "maximal")
    dev = P.joint_deviation(stand15, out, 15)
    assert abs(P.held_share(dev, lvl) - share) < 0.05
    assert dev.max() <= lvl + 1e-6
    if exc == "maximal":
        assert dev.max() >= 0.99 * lvl                     # a mobile spasm reaches the maximal excursion
    assert _ok(tr["validity"]) == [] and _finite(out)
    again, _ = P.cervical_course(stand15, [("neck", "rotation_left", "left", lvl)], duration_grade=dg, seed=7)
    other, _ = P.cervical_course(stand15, [("neck", "rotation_left", "left", lvl)], duration_grade=dg, seed=8)
    assert torch.equal(again.rot6d, out.rot6d) and not torch.equal(other.rot6d, out.rot6d)


def test_shoulder_elevation_course_is_measured_on_the_collar(P, stand15):
    from mdfm.synth import grades as G
    lv = G.twstrs_level_deg("shoulder_elevation", 2, P.region_room(stand15, "shoulder_elevation", "elevation", "left"))
    out, tr = P.cervical_course(stand15, [("shoulder_elevation", "elevation", "left", lv)], share=0.375, seed=4)
    p = tr["params"]
    assert p["measured_segment"] == 13 and abs(p["realized_share"] - 0.375) < 0.05 and p["realized_max_deg"] > 0.99 * lv


@pytest.mark.parametrize("grade", [2, 3])
def test_the_shoulder_rises_with_the_arm_hanging_under_it(P, stand, grade):
    """SH2 (limits_ext): the raised collar carries a hanging arm (review B2: the arm swung out with the collar)."""
    from mdfm.synth import grades as G
    lv = G.twstrs_level_deg("shoulder_elevation", grade,
                            P.region_room(stand, "shoulder_elevation", "elevation", "left", table=P.DYST))
    out, tr = P.cervical_course(stand, [("shoulder_elevation", "elevation", "left", lv)], share=1.0, seed=4)
    p, v = tr["params"], tr["validity"]
    assert _humerus_angle(stand, out, 16, 18).max() < 0.01 and p["arm_hangs"]["left"]["humerus_turn_deg_max"] < 0.01
    assert p["arm_hangs"]["left"]["shoulder_rise_cm_max"] > (2.0 if grade == 2 else 3.5)
    assert _ok(v) == [] and v["coupling"]["ok"] and v["limits_table"] == "dystonia"
    assert p["limits_table"]["table"] == "dystonia" and "limits_ext" in tr["sources"]
    assert p["realized_max_deg"] > 0.99 * lv                 # the collar holds its TWSTRS level


def test_the_segmental_dystonia_raises_the_shoulder_over_a_hanging_arm(P):
    out, tr = P.distribution("segmental", duration_s=3.0)
    p = tr["params"]
    assert p["arm_under_collar"]["left"]["humerus_turn_deg_max"] < 0.5
    assert p["arm_under_collar"]["left"]["shoulder_rise_cm_max"] > 2.0
    assert _ok(tr["validity"]) == [] and tr["validity"]["coupling"]["ok"] and _finite(out)


@pytest.mark.parametrize("kind", ["lateral", "anterior", "posterior"])
def test_head_shift_moves_the_head_without_tilting_it(P, stand, kind):
    out, tr = P.head_shift(stand, kind)
    p = tr["params"]
    assert p["shift_cm"] > 4.0 and p["head_tilt_deg"] < 0.01 and _ok(tr["validity"]) == []
    comp = p["shift_lateral_cm"] if kind == "lateral" else p["shift_forward_cm"]
    assert (comp > 0) == (kind in ("lateral", "anterior"))


@pytest.mark.parametrize("score", [0, 1, 2])
def test_geste_antagoniste_reduces_the_posture_by_the_trick_share(P, stand15, geste, score):
    out, tr = geste(score)
    p, v = tr["params"], tr["validity"]
    assert abs(p["reduction"] - P.TRICK[score]) < 0.1
    assert p["contact_gap_m"] < 0.005                        # the pads stay on their target by the cheek
    assert _ok(v) == [] and v["limits_new"] == 0 and _finite(out)
    k_before = int(round((P.TRICK_TIMES["reach_on_s"] - 0.5) * out.rate))
    torch.testing.assert_close(out.rot6d[:k_before, 17], stand15.rot6d[:k_before, 17], atol=1e-9, rtol=0)
    # reviews of 2026-11-02 and of the SH2 wave: nothing inside the head or neck; the pads 2 to 4 mm from the skin,
    # the palm and forearm 1 cm off or more
    assert v["hand_head"]["ok"] and v["hand_head"]["min_m"] >= 0.0
    assert abs(p["gap_solve"]["least_m"][-1] - P.GESTE_GAP_M) < P.GESTE_GAP_TOL_M
    sk = p["skin_m"]
    assert 0.002 <= sk["pads"][0] and sk["pads"][1] <= 0.004
    assert min(sk["palm_min"], sk["fore_min"]) >= P.GESTE_PALM_GAP_M
    # review of 2026-11-03 (the wrist bent past these ranges, inside the table's components): radial deviation <= 25
    # and dorsiflexion <= 70 deg (SOURCES jarm2022_wrist) while the hand rests, the whole range kept over the reach; the
    # hand and the forearm off the trunk
    w = p["wrist_deg"]
    assert w["held_deviation"][1] <= P.WRIST_RANGE_DEG["radial"]
    assert -w["held_flexion"][0] <= P.WRIST_RANGE_DEG["dorsal"]
    assert w["reach_load"] <= 1.0 and v["wrist_range"]["ok"]
    assert v["hand_trunk"]["min_m"] >= P.GESTE_TRUNK_GAP_M


def test_the_wrist_measure_reads_flexion_toward_the_palm_and_deviation_toward_the_thumb(P):
    """wrist_angles' signs on hands turned 20 deg from the forearm's line toward the palm and toward either side of
    the hand's transverse axis, on both sides (the thumb's side read from the rest joints: thumb knuckle against
    little-finger knuckle)."""
    from mdfm.body.rotations import axis_angle_to_matrix
    from mdfm.synth import hands as H
    J, _, _ = H._rest()
    for side in ("left", "right"):
        ax = P._hand_axes(side)
        ha, pn = ax[:, 0], ax[:, 1]
        r = np.cross(ha, pn)                                   # the hand's transverse axis, toward the thumb's side
        thumb = r * np.sign(r @ (J[H.hand_joint(side, "thumb", 0)] - J[H.hand_joint(side, "pinky", 0)]))
        for toward, want in ((pn, (20.0, 0.0)), (thumb, (0.0, 20.0)), (-thumb, (0.0, -20.0))):
            axis = np.cross(ha, toward)
            axis = axis / np.linalg.norm(axis)
            Rh = axis_angle_to_matrix(torch.tensor(np.radians(20.0) * axis)).numpy()
            fl, dv = P.wrist_angles(ha[None], Rh[None], side)
            np.testing.assert_allclose([float(fl[0]), float(dv[0])], want, atol=0.5)


def test_the_geste_hand_rests_on_its_own_cheek_from_the_side_clear_of_the_throat(P, geste):
    """Review of 2026-11-03 (the hand upright with the palm and the wrist in front of the neck): the right hand on
    the right cheek (the side away from the turn); the wrist lateral to the face, at or below the jaw line, beside
    the neck and never in front of it (its lateral offset in the neck's frame beyond the neck's half width by 1 cm),
    the palm and the wrist 1 cm or more from the front of the neck; the palm turned toward the face. The hand's long
    axis rises to the cheek from below (the second review of 2026-11-03: with it back toward the ear the wrist left
    its range), the hand frame being the one of least wrist bend that geste_frames ranks first."""
    out, tr = geste(1)
    p = tr["params"]
    h = p["hand_place"]
    assert p["hand"] == "right" and p["contact_vertex"] == P.cheek_contact_vertex("right")
    from mdfm.synth import hands as H
    J, v, _ = H._rest()
    assert v[p["contact_vertex"], 0] - J[15, 0] < -0.04                     # on the right cheek, not the chin
    assert h["wrist_lateral_m"] > h["face_half_m"] + 0.05 and h["wrist_below_jaw_m"] >= 0.0
    assert h["wrist_lateral_neck_m"] >= h["neck_half_m"] + 0.01
    assert min(h["throat_palm_m"], h["throat_wrist_m"]) >= P.GESTE_PALM_GAP_M
    assert 0.0 < h["axis_elevation_deg"] < 90.0 and h["palm_to_face_cos"] > 0.0
    # wave SH1c (in the front view the open hand beside the jaw, the back of the hand toward the viewer and the fingers
    # spread up, read as a raised hand): the fingers curl (GESTE_HAND) and the palm turns toward the face
    assert h["palm_to_face_cos"] > 0.5
    k = int(round(0.5 * (P.TRICK_TIMES["reach_on_s"] + P.TRICK_TIMES["reach_s"] + P.TRICK_TIMES["release_s"])
                  * out.rate))                               # while the hand rests on the cheek: the realized flexion
    comp = H.hand_components(out.hand_R()[k])               # (review of wave SH1c: not the module's own constants)
    fl = {f: [float(comp[H.hand_joint("right", f, lv)][0]) for lv in (0, 1)]
          for f in ("index", "middle", "ring", "pinky")}
    assert min(min(x) for x in fl.values()) >= 40.0          # curled, not the barely flexed open hand of before
    assert sum(fl["ring"]) >= sum(fl["index"]) and sum(fl["pinky"]) >= sum(fl["index"])   # the index leads the touch
    assert p["arm"]["score_deg"] < P.GESTE_MARGIN_DEG and p["arm"]["arms_inside"] >= 1
    assert p["frame"]["tried"] >= 1 and p["finger_deg"] in P.GESTE_FINGERS_DEG


@pytest.mark.parametrize("score", [0, 1, 2])
def test_the_geste_hand_folds_to_the_face_without_sweeping_out(P, stand15, geste, score):
    """Review of 2026-11-03 (on one blend of every arm row the arm rose nearly straight, the wrist 0.37 m in front of
    the shoulder joint, the middle fingertip travelling 1.51 to 1.55 m for 0.86 m in a straight line): the elbow folds
    ahead of the shoulder (GESTE_SHOULDER_SHARE); the fingertip's path at most GESTE_PATH_RATIO_MAX (1.4) times its
    straight line [module] and the wrist never more than GESTE_FORWARD_EXTRA_M (5 cm) further forward of the shoulder
    than while the hand rests [module] (since the wave SH1c the build takes the first arm that keeps both, or, where no
    arm of the tried frames keeps the second, the passing arm of least excess within GESTE_FORWARD_EXTRA_MAX_M, 6 cm,
    flagged 'forward_relaxed'); no arm joint turns more than 3 deg in one knot beyond the level 0's."""
    out, tr = geste(score)
    rp, arm = tr["params"]["reach_path"], tr["params"]["arm"]
    assert rp["ratio"] <= P.GESTE_PATH_RATIO_MAX
    extra = rp["wrist_forward_m"] - rp["wrist_forward_held_m"]
    assert abs(extra - arm["forward_extra_m"]) < 1e-9
    assert extra <= (P.GESTE_FORWARD_EXTRA_MAX_M if arm["forward_relaxed"] else P.GESTE_FORWARD_EXTRA_M)
    assert arm["forward_relaxed"] == (extra > P.GESTE_FORWARD_EXTRA_M)   # flagged exactly where the bound is relaxed
    from mdfm.synth import hands as H
    assert _snaps(out, H.hands_of(stand15), (14, 17, 19, 21)) == {}


# ---------------------------------------------------------------- limb dystonia
@pytest.mark.parametrize("standing", [False, True])
def test_the_dystonic_foot_touches_the_floor_and_the_other_foot_stands(P, standing):
    lv = P.FOOT_STAND if standing else None
    out, tr = P.foot_dystonia(3, standing=standing, levels=lv, duration_s=3.0, seed=2)
    v = tr["validity"]
    assert _ok(v) == [], _ok(v)
    assert v["contact"]["moved_low_min_m"] >= -0.005 and v["contact"]["moved_low_max_m"] <= 0.005
    assert v["contact"]["other_low_absmax_m"] < 0.005 and v["skate_added_m_s"] < 1e-9   # pose blend shapes: mm
    m = tr["params"]["foot_measured"]
    assert m["plantarflexion_deg"] > 5 and m["inversion_deg"] > 5 and m["adduction_deg"] > 2
    # SH2 (limits_ext) read on the anatomical table (open-body port, wave 4, D26): the healthy table now holds
    # jarm2022's 30 deg of inversion that SH2 widened the AMASS table's 22 to, and 35 deg of toe flexion (2.3 before);
    # the inversion term takes the DIS 3 share of that room, unscaled, and the measured component (the axis-angle of
    # the ankle's whole turn, its plantarflexion and adduction composed in) keeps more than 0.7 of it (15.6 deg seated,
    # 17.8 standing; reports/openbody/wave4_checks.json, foot)
    from mdfm.synth.body import joint_limits
    inv = [t for t in tr["params"]["terms"] if "inversion" in t["label"]][0]
    # the room is read on the base's ankle (foot_terms): standing since D125 the soft 'stand' holds the ankle
    # dorsiflexed by its hip and knee flexion, where the inversion's room reads 30.0022 deg (21.0016 at the share),
    # 30 on the straight stand the equality was written on; seated exact (wave 8 second part, D158)
    assert inv["deg"] == pytest.approx(-joint_limits().lo[7, 2] * P.DIS_SHARE[3], abs=0.01 if standing else 2.1e-5)
    assert tr["params"]["limit_scale_min"] == 1.0 and m["inversion_deg"] > 0.7 * inv["deg"]
    assert m["toe_flexion_deg"] > 15.0 if not standing else abs(m["toe_flexion_deg"]) < 1e-6
    assert tr["params"]["limits_table"]["table"] == "dystonia" and tr["validity"]["coupling"]["ok"]
    f = tr["params"]["foot"]
    assert f["knee_lift_deg_max"] > 0 and _finite(out)
    base = P.static_body("stand" if standing else "sit", 3.0)
    dz = 100 * float((out.joints()[:, 7, 2] - base.joints()[:, 7, 2]).max())
    if standing:                                             # the lift capped, the pelvis hiked (review M1)
        assert f["pelvis_hike_deg_max"] > 1.0 and f["knee_lift_deg_max"] < P.LIFT_CAP_DEG + 6.0 and dz < 10.0
        # Phase D: this foot rises from rest; where the hike reaches its cap the roll misses the floor and the foot
        # keeps the touchdown pin, otherwise it rolls (pin_roll_standing) with its lowest sole point on the floor.
        # Open-body port, wave 4 (D26): on SMPL+H the hike reached its cap and the touchdown pin held; on the open
        # body the hike stays under it (10.1 deg against HIKE_MAX_DEG 12) and the roll holds the floor
        pin = f["pin"]
        if pin["rule"].startswith("pin_contact_standing"):
            assert pin["roll_refused_floor_error_mm"] > P.ROLL_FLOOR_TOL_MM
        else:
            assert pin["rule"].startswith("pin_roll_standing") and pin["hike_at_cap_knots"] == 0
            assert pin["floor_error_after_mm"] <= P.ROLL_FLOOR_TOL_MM and pin["travel_after_mm"] <= 3.0
    else:                                                    # the heel rises, the foot keeps its place (review B6):
        # since wave SH1d part B (review M11) the place kept is the forefoot's contact (pin_contact), so the ankle
        # comes forward as the foot pivots about it instead of the toes sliding back: forward, and never past the
        # pinned contact point. Open-body port, wave 4 (D26): the bound was 6 cm over the 5.1 cm SMPL+H moved; the
        # open body's ankle comes 6.6 cm forward, its contact on the toes 19.8 cm in front of it
        from mdfm.synth import contact as CT
        from mdfm.synth import viewer as V
        vid = CT.sole_vertices()["left"][f["pin"]["vertex_in_cloud"]]
        reach = 100 * float(V.knot_vertices(base, [0])[0][vid, 2] - base.joints()[0, 7, 2])
        assert f["heel_rise_cm"] > 3.0 and f["pelvis_hike_deg_max"] == 0.0 and 0.0 < dz < reach
        assert f["pin"]["travel_after_mm"] <= 3.0


def test_writers_cramp_writes_along_the_line_keeps_the_hand_on_the_table_and_lifts_the_elbow(P):
    l0, t0 = P.writers_cramp(0, duration_s=6.0)
    out, tr = P.writers_cramp(2, duration_s=6.0, seed=1)
    p = tr["params"]
    assert p["realized_elbow_rise_cm"] > 1.0 and p["wcrs_writing_movement"] == 16
    assert _ok(tr["validity"]) == [] and l0.T == out.T and _finite(out)
    lat = P.WCRS_LATENCY_S[P.WCRS_ITEMS[2]["latency_factor"]]
    k = int(round((P.WRITE["start_s"] + lat) * out.rate))
    assert float(P.joint_deviation(l0, out, 21, segment=False)[:k].max()) < 1e-3        # nothing before the latency
    top = t0["params"]["writing"]["table_top_y"]
    for b in (l0, out):                                      # never through the table (review B4)
        assert float(P._arm_low(b, ("left", "right")).min()) > top - 5e-4
    _on_the_paper(P, t0["validity"]["table"])                # on the paper, the pen on it (wave SH1c)
    _on_the_paper(P, tr["validity"]["table"], strict=False)  # the cramped hand rests by its edge or the pen
    assert top == P.writing_surface("right")["table_top_y"]           # one table for every duration
    assert t0["params"]["writing"]["pen"]["knots_clamped"] == 0
    J = l0.joints().numpy()
    assert np.ptp(J[:, 21, 0]) > 0.03                        # the hand moves along the line (review B5)
    _, _, h = P.writing_table(l0, 0.0)
    assert 0.55 < h < 0.85                                   # a desk height above the joint floor
    # review of wave SH1c: the texts say the level 0's hand rests by its curled little finger with the forearm free
    # above the table; the cramp's elbow lifts from there. The table check runs again from the truth's record (the
    # layered body's check), and gives the clip's own result on the raw body
    from mdfm.synth import hands as H
    _, _, dom = H._rest()
    ids = P.hand_vertex_ids("right", P.EDGE_FINGERS)
    kk = np.arange(0, l0.T, 50)
    low = dom[ids[H.body_points(l0, ids, knots=kk)[..., 1].argmin(-1)]]
    assert set(low.tolist()) <= {H.hand_joint("right", "pinky", lv) for lv in range(3)}
    assert t0["validity"]["table"]["forearm_gap_mm"][0] > 1000 * P.WRITE_REST_GAP_M
    for t, b in ((t0, l0), (tr, out)):
        assert t["params"]["table_check"]["write_side"] == "right"
        assert P.table_report_of(b, t["params"]["table_check"]) == t["validity"]["table"]


def test_the_sheet_spans_the_longest_line(P):
    """Wave SH1c: the sheet is carried half of line_reach_m to the writer's right, and no line of writing_line runs
    further along than line_reach_m (the last word starts short of line_m plus a gap and runs past it)."""
    reach = P.line_reach_m()
    assert reach == P.LINE["line_m"] + P.LINE["gap_m"] + P.LINE["speed_m_s"] * P.LINE["word_s"][1]
    most = max(float(P.writing_line(1501, 100.0, P.as_rng(s))["along"].max()) for s in range(40))
    assert P.LINE["line_m"] < most <= reach + 1e-12


def test_mirror_dystonia_stays_inside_the_table(P):
    """The resting hand lies palm down; the mirror posture's pronation is fit on its own (fit_terms falls back to the
    base where it would leave the table), so no new violation. Open-body port, wave 4 (D26): on the AMASS table the
    resting wrist lay at its pronation bound and the pronation fell back to the base (its scale under 1); on the
    anatomical table the resting wrist's axial component leaves 82.7 deg toward the term's 15 deg (from its 37.7 deg
    the other way to jarm2022's 45 deg share), so the pronation applies whole."""
    out, tr = P.writers_cramp(2, duration_s=6.0, seed=1, mirror=True)
    p = tr["params"]
    assert _ok(tr["validity"]) == [] and tr["validity"]["limits_new"] == 0 and _finite(out)
    assert p["realized_wrist_deg"] > 5.0 and p["pronation_scale_held_max"] == 1.0
    l0, info, v0 = P.writing_level0("left", 6.0)
    assert _snaps(out, l0, range(13, 22)) == {}
    for v in (v0["table"], tr["validity"]["table"]):        # the left hand on the paper, the right on the table
        _on_the_paper(P, v)
    assert info["pen"]["knots_clamped"] == 0 and np.ptp(l0.joints().numpy()[:, 20, 0]) > 0.03


def test_the_left_writers_hand_never_meets_the_resting_right_hand(P, monkeypatch):
    """Review of wave SH1c: the line runs to the writer's right, so the left hand travels toward the resting right hand
    and the two passed through each other (on the catalogue's 15 s, late in the line). WRITE's rest_clear_m keeps them
    apart at every read of the clip and of its mirror dystonia; without it table_report refuses the level 0."""
    from mdfm.synth.motions import fold_gaps
    l0, _, v0 = P.writing_level0("left")
    out, tr = P.writers_cramp(2, seed=P.SEED + 81, mirror=True)
    for b, v in ((l0, v0), (out, tr["validity"])):
        _on_the_paper(P, v["table"])
        assert v["table"]["hands_gap_mm"] > 0.0
        assert float(fold_gaps(b, np.arange(0, b.T, 2), None, ("hands",))[0]["hands"].min()) > 0.0
    rt, _, v1 = P.writing_level0("right", 6.0)
    assert v1["table"]["ok"] and v1["table"]["hands_gap_mm"] > 0.0
    monkeypatch.setitem(P.WRITE, "rest_clear_m", 0.0)
    near, _ = P.writing_base(P.DUR_S, ("left", "right"), P.SEED, "left")
    bad = P.table_report(near, "left", ["right"], lift=P.pen_lift_share(near.T, near.rate))
    assert bad["hands_gap_mm"] < 0.0 and not bad["ok"]


def test_overflow_equals_its_level0_until_the_cramp_without_a_jump(P):
    """The SH1 lane's report on post_W_overflow (its body off its level 0 at the clip's start, then equal) came from
    clear_table's wrap of the clip's end onto its start (np.roll), replaced in the SH2 wave by an edge-padded
    running maximum: the overflow clip equals the writing level 0 until the cramp's latency, and no joint snaps."""
    l0, _ = P.writers_cramp(0, duration_s=6.0)
    out, tr = P.writers_cramp(2, duration_s=6.0, seed=1, overflow=True)
    assert _ok(tr["validity"]) == [] and tr["validity"]["limits_new"] == 0 and _finite(out) and l0.T == out.T
    lat = P.WCRS_LATENCY_S[P.WCRS_ITEMS[2]["latency_factor"]]
    k = int(round((P.WRITE["start_s"] + lat) * out.rate))
    assert float((out.joints()[:k] - l0.joints()[:k]).norm(dim=-1).max()) < 1e-6
    assert float((out.hand_R()[:k] - l0.hand_R()[:k]).abs().max()) < 1e-9
    assert tr["params"]["realized_head_deg"] > 1.0                 # the overflow to the neck is there by the clip's end
    assert _snaps(out, l0, range(9, 22)) == {}
    _on_the_paper(P, tr["validity"]["table"], strict=False)


def test_task_specific_arm_dystonia_is_absent_at_rest(P):
    out, tr = P.task_specific_arm(2, duration_s=8.0)
    p = tr["params"]
    assert p["rest_deviation_deg"] < 1e-3 and p["realized_deviation_deg"]["wrist flexion left"] > 5.0
    assert _ok(tr["validity"]) == [] and _finite(out)


def test_fixed_dystonia_does_not_vary(P):
    out, tr = P.distribution("fixed_fist", duration_s=3.0)
    assert torch.equal(out.rot6d, out.rot6d[:1].expand_as(out.rot6d))
    assert torch.equal(out.hand6d, out.hand6d[:1].expand_as(out.hand6d))
    assert tr["params"]["env_range"] == [1.0, 1.0] and _ok(tr["validity"]) == []
    from mdfm.synth import hands as H
    comp = H.hand_components(out.hand_R()[0])
    mcp = {f: float(comp[H.hand_joint("left", f, 0)][0]) for f in ("index", "middle", "ring", "pinky")}
    assert mcp["index"] < mcp["middle"] < mcp["ring"] and mcp["pinky"] > mcp["middle"]   # review M12


def test_the_fixed_fist_is_lifted_off_the_thigh(P):
    """Plan phase E: the clenched fist and the flexed wrist of the 'sit' hung the index tip 22.7 mm into the open
    body's thigh on the mesh; the left shoulder flexed by the least step that keeps the hand and forearm reach.CLEAR_M
    off the thighs (dystonia.arm_lift), as a term of the build (the table reads it)."""
    from mdfm.synth import dystonia as D
    from mdfm.synth.reach import CLEAR_M
    out, tr = P.distribution("fixed_fist", duration_s=1.0)
    lift = tr["params"]["arm_lift"]["sides"]["left"]
    assert lift["deg"] >= D.ARM_LIFT_STEP_DEG and lift["gap_before_mm"] < 1000 * CLEAR_M <= lift["gap_after_mm"]
    assert P.arm_part_gap(out, "left", [0], "thighs")[0] >= CLEAR_M and _ok(tr["validity"]) == []
    under = D.with_arm_flexion(out, "left", -D.ARM_LIFT_STEP_DEG)      # one step less: nearer than CLEAR_M
    assert P.arm_part_gap(under, "left", [0], "thighs")[0] < CLEAR_M
    assert any(t["label"].startswith("shoulder flexion left") for t in tr["params"]["terms"])


# ---------------------------------------------------------------- paroxysmal kinesigenic dyskinesia
def test_pkd_attack_follows_the_lift_off_and_leaves_the_feet(P):
    from mdfm.synth.motions import arise
    out, tr = P.paroxysmal_kinesigenic("dystonic", seed=1, standing_s=12.0)
    p = tr["params"]
    base, _ = arise(standing_s=12.0)
    assert p["attack_on_s"] > p["liftoff_s"] and p["pre_attack_deviation_deg"] < 1e-3
    assert abs(p["realized_attack_s"] - P.PKD["duration_s"]) < P.PKD["on_s"] + P.PKD["off_s"]
    legs = [1, 2, 4, 5, 7, 8, 10, 11]
    assert torch.equal(out.rot6d[:, legs], base.rot6d[:, legs]) and torch.equal(out.transl, base.transl)
    assert _ok(tr["validity"]) == [] and _finite(out)
    k = int(round((p["arms_unfold_s"][1] + 0.5) * out.rate))            # unfolded: the right forearm hangs (M6)
    assert float(_from_vertical(out, 19, 21)[k]) < 30.0 and float(_from_vertical(base, 19, 21)[k]) > 60.0
    assert float(_from_vertical(out, 16, 18)[k:].max()) < 35.0        # no abducted, scarecrow arm
    # review of 2026-11-02: the arms open only once the trunk is near upright, clear of the thighs and armrests
    flex = P.trunk_flexion_deg(base)
    k_unf = int(round(p["arms_unfold_s"][0] * out.rate))
    assert flex[k_unf] <= P.PKD["unfold_trunk_deg"] + 1e-9 and p["trunk_flexion_at_liftoff_deg"] > 20.0
    assert p["arms_unfold_s"][0] > p["liftoff_s"]
    v = tr["validity"]
    assert v["armrest"]["ok"] and v["hand_thigh"]["ok"] and v["hand_trunk"]["ok"]
    assert p["attack_on_s"] >= p["arms_unfold_s"][0]                     # no attack on the folded, leaning arms


def test_the_dystonic_pkd_attack_does_not_snap(P):
    """Review M2 of the SH2 wave: the fit's scale fell from 1 to 0.56 in one knot as the unfolding arm reached the
    shoulder's adduction bound, and the flexed forearm dropped in one knot; the clip's own build (seed, 15 s)."""
    from mdfm.synth.motions import arise
    out, tr = P.paroxysmal_kinesigenic("dystonic", seed=P.SEED + 90)
    base, _ = arise(standing_s=P.DUR_S)
    assert _snaps(out, base, range(1, 22)) == {} and _ok(tr["validity"]) == []
    assert [len(g) for g in tr["params"]["fit"]["groups"]] == [2, 2, 2]
    assert tr["params"]["limits_table"]["table"] == "dystonia"
    # wave SH1c: the left forearm lay inside the abdomen over the whole attack; the humerus's inward turn is held back
    # where it would (guard_rows), so that nothing of either arm enters the trunk after lift-off
    p, v = tr["params"], tr["validity"]
    assert v["hand_trunk"]["ok"] and p["trunk_guard"]["left"]["still_entering"] == 0
    k = np.arange(int(round(p["liftoff_s"] * out.rate)), out.T, P.SCENE_CHECK_STEP)
    assert P.trunk_report(out, k, ("left",))["left"]["min_mm"] >= 0.0
    assert p["realized_deviation_deg"]["internal rotation left"] > 5.0      # the arm still turns inward
    # the arms open each on its clock (one clock: the right hand through the left forearm), read on the final body
    # the lag is the smallest of FOLD_OPEN_LAGS whose rehearsal keeps the skins apart (D26, open-body port wave 6: on
    # SMPL+H's fold the one-clock opening put the right hand through the left forearm and the left arm waited; the open
    # body's fold, solved again, opens on one clock with the skins apart, so its lag is 0)
    from mdfm.synth.motions import FOLD_OPEN_CLEAR_M
    lag = p["arms_unfold_lag"]
    assert v["fold"]["ok"] and lag["tried"][0]["lag"] == 0.0
    ok0 = lag["tried"][0]["min_mm"] >= 1000.0 * FOLD_OPEN_CLEAR_M
    assert (lag["lag"] == 0.0) == ok0 and (ok0 or lag["lag_side"] in ("left", "right"))


def test_the_smoothed_scale_is_never_above_the_fit_and_moves_slowly(P):
    s = np.ones(200)
    s[80:] = 0.4
    s[120:125] = 0.0
    k = 15
    m = P.smooth_scale(s, k)
    assert (m <= s + 1e-12).all() and np.abs(np.diff(m)).max() <= 1.0 / (2 * k + 1) + 1e-12
    assert np.array_equal(P.smooth_scale(np.ones(50), k), np.ones(50))


def test_the_choreic_pkd_hand_stays_out_of_the_thigh(P):
    """Review of wave SH1c: on the clip's own build the right hand lay inside the right thigh in mid-attack, past the
    window of motions.unfold_check, while reach.clearance read it outside (its sign is the closest triangle's, and the
    thigh's mesh is open at its joints); arm_part_gap reads the winding number too, and the guard holds the arm back."""
    from mdfm.synth.motions import fold_gaps
    from mdfm.synth.reach import clearance
    out, tr = P.paroxysmal_kinesigenic("choreic", P.SEED + 90)
    p, v = tr["params"], tr["validity"]
    assert _ok(v) == [] and v["fold"]["ok"] and v["hand_thigh"]["ok"] and v["hand_trunk"]["ok"] and v["armrest"]["ok"]
    assert all(g["still_entering"] == 0 for g in p["thigh_guard"].values())
    k = np.arange(int(round(p["liftoff_s"] * out.rate)), out.T)
    gaps = fold_gaps(out, k, None, ("thigh_left", "thigh_right", "trunk_left", "trunk_right"))[0]
    assert min(float(g.min()) for g in gaps.values()) >= 0.0                     # at every knot from lift-off
    kk = k[::100]
    c = clearance(out, "right", kk, parts=("thighs",))["thighs"]
    assert (P.arm_part_gap(out, "right", kk, "thighs") <= np.minimum(c["hand"], c["fore"]) + 1e-12).all()
    assert min(p["realized_deviation_deg"][x] for x in ("chorea humerus rotation right", "chorea wrist right",
                                                        "chorea elbow right")) > 15.0   # the attack is still there


def test_choreic_pkd_leaves_the_level0_alone_before_its_attack(P):
    from mdfm.synth.motions import arise
    out, tr = P.paroxysmal_kinesigenic("choreic", seed=1, standing_s=4.0)
    base, _ = arise(standing_s=4.0)
    k = int(tr["params"]["arms_unfold_s"][0] * out.rate)               # chorea_term: 0 at a course of 0
    assert float((out.rot6d[:k] - base.rot6d[:k]).abs().max()) < 1e-9 and _ok(tr["validity"]) == []
    assert tr["params"]["limits_table"]["table"] == "healthy" and "limits_ext" not in tr["sources"]   # not dystonia
    assert tr["validity"]["hand_trunk"]["ok"]


# ---------------------------------------------------------------- time courses (review B3, M8, minor 1)
def test_jerky_pulls_are_slow_irregular_sawtooth_events_under_the_maximum(P):
    from mdfm.synth.common import as_rng
    env, info = P.mobile_envelope(1501, 100.0, as_rng(3), 1.0, "maximal", spasms=False, jerks=True)
    ev = info["jerk_events"]
    assert 2 <= len(ev) <= 8                                  # 0.1 to 0.5 Hz over 15 s
    gaps = np.diff([e[0] for e in ev])
    assert gaps.min() >= P.JERK_GAP_S[0] - 1e-9 and np.ptp(gaps) > 0.3          # irregular, at least 2 s apart
    assert all(P.JERK_PULL_S[0] <= e[1] <= P.JERK_PULL_S[1] < P.JERK_RETURN_S[0] <= e[2] for e in ev)  # sawtooth
    assert env.max() <= 1.0 + 1e-12                                           # never above the maximum


def test_intermittent_bouts_drift_in_slowly_and_are_corrected_quickly(P):
    from mdfm.synth.common import as_rng
    env, info = P.mobile_envelope(1501, 100.0, as_rng(7), 0.125, "submaximal")
    assert info["n_bouts"] >= 2
    for d, c in info["drift_correction_s"]:
        assert d > 2.0 * c and P.CORRECT_BACK_S[0] <= c <= P.CORRECT_BACK_S[1]


def test_spasms_reach_the_maximum_without_a_plateau(P):
    from mdfm.synth.common import as_rng
    env, info = P.mobile_envelope(1501, 100.0, as_rng(11), 1.0, "maximal")
    assert info["spasm_onsets_s"] and env.max() <= 1.0 + 1e-12 and env.max() > 0.999
    assert max(b - a for a, b in P._runs(env > 1.0 - 1e-4)) <= 5         # a peak, not a plateau


# ---------------------------------------------------------------- generalized and paroxysmal (review M3, M5)
def test_torsion_dystonia_twists_without_flapping(P):
    from mdfm.synth.base import static_body
    b = static_body("stand", 3.0)
    out, tr = P.torsion_dystonia(b, 3, seed=5)
    v = tr["validity"]
    assert _ok(v) == [] and v["skate_added_m_s"] < 1e-9 and _finite(out)
    for s_, e_ in ((16, 18), (17, 19)):
        assert float(_from_vertical(out, s_, e_).max()) < 35.0
    assert tr["params"]["foot_measured"]["inversion_deg"] > 5.0


def test_pnkd_arm_is_not_held_out(P):
    out, tr = P.paroxysmal_excerpt("pnkd", seed=2, duration_s=4.0)
    assert _ok(tr["validity"]) == [] and _finite(out)
    assert float(_from_vertical(out, 16, 18).max()) < 35.0
    assert {"chorea wrist left", "chorea wrist right"} <= {t["label"] for t in tr["params"]["terms"]}
    assert tr["validity"]["armrest"]["ok"]                   # review of 2026-11-02: the forearm inside the armrest
    # wave SH1c: on this build the passes of clear_armrests added up past the right collar's room, which left the table;
    # the raise stops at the room, and the right arm, which swings freely, takes no raise at all
    assert tr["validity"]["limits_new"] == 0 and tr["validity"]["coupling"]["ok"]
    assert tr["params"]["armrest_clearance"][-1]["right_still_entering"] == 0
    for sd in ("left", "right"):                             # at every knot, not only the sampled ones
        assert float(P.armrest_depth(out, sd, P.armrest_boxes("chair", out), np.arange(out.T)).max()) == 0.0
    assert tr["validity"]["backrest"]["ok"] and tr["validity"]["hand_trunk"]["ok"]
    from mdfm.synth.motions import posture_body                       # review of wave SH1c: at every knot
    assert P.trunk_report(out, np.arange(out.T), base=posture_body("sit", 4.0))["ok"]


def test_a_chorea_term_passes_through_zero_without_a_kink(P):
    """Review of wave SH1c: two straight lines meeting at a course of 0 changed the velocity in one knot at every zero
    crossing of a course. The share is 0 at 0, comes within 0.2 percent of each side's level at the course's ends and
    changes its slope gradually."""
    from mdfm.synth.base import static_body
    b = static_body("sit", 0.1)
    c = np.linspace(-1.0, 1.0, 4001)
    t = P.chorea_term(b, "wrist", "flexion", "extension", "left", P.CHOREA_SHARE, c, "x")
    ap = P.CHOREA_SHARE * P.region_room(b, "wrist", "flexion", "left")
    an = P.CHOREA_SHARE * P.region_room(b, "wrist", "extension", "left")
    assert an != ap
    assert abs(t.env[2000]) < 1e-12 and abs(t.env[-1] - 1.0) < 2e-3 and abs(t.env[0] + an / ap) < 2e-3
    slope = np.diff(t.env) / np.diff(c)
    assert float(np.abs(np.diff(slope)).max()) < 0.05 * abs(1.0 - an / ap)


def test_guard_rows_never_takes_the_whole_way_back_where_that_does_not_help(P):
    """Review of wave SH1c: where no share of the guarded rows ended an entry, guard_rows took the share 1 and held
    the PNKD's whole left arm at its base while the entry went on; now the share of least need, here 0: a need that
    the rows cannot change leaves them alone."""
    from mdfm.synth.base import static_body
    b = static_body("sit", 0.5)
    R = b.local_R().clone()
    R[:, 18] = P.axis_rotation("y", np.full(b.T, -0.3)) @ R[:, 18]
    out, w, info = P.guard_rows(b, R, (18,), lambda sub, kk: np.full(len(kk), 0.01))
    assert float(w.max()) == 0.0 and info["still_entering"] > 0 and info["least_need_knots"] > 0
    assert torch.equal(out, P.blend_components(R, b.local_R(), w, (18,)))


def test_the_pnkd_excerpt_keeps_the_arms_out_of_the_backrest(P):
    """Wave SH1c: on the clip's own build (seed and duration of the catalogue) the right upper arm entered the backrest
    as the trunk turned; the trunk's turn and the arm are held back where it would (guard_rows), so that no vertex of
    the body enters the backrest at any knot, and the forearms stay off the armrests. Review of wave SH1c: an armrest
    guard on the arm's rows held the left arm at its seated base at every knot (the entry came from the trunk's lean),
    and the left forearm entered the trunk; now the arm keeps at least half of the fit's elbow and wrist flexion and no
    forearm or hand enters the trunk at any knot."""
    out, tr = P.paroxysmal_excerpt("pnkd", P.SEED + 92)
    v, p = tr["validity"], tr["params"]
    assert _ok(v) == [] and v["backrest"]["ok"] and v["armrest"]["ok"] and v["backrest"]["depth_mm_max"] == 0.0
    assert all(g["still_entering"] == 0 for g in p["backrest_guard"].values())
    last = p["trunk_guard"][-1]
    assert v["hand_trunk"]["ok"] and last["step"] == 1 and all(last[sd]["knots"] == 0 for sd in ("left", "right"))
    assert v["hand_thigh"]["ok"]
    # plan phase E: the hands also held PNKD_THIGH_CLEAR_M off the thighs (the left hand passed 0.36 mm over the thigh
    # and the background layer carried it 3.80 mm inside); the last round, at every knot, finds nothing to hold
    assert all(last[sd + "_thigh"]["knots"] == 0 and last[sd + "_thigh"]["still_entering"] == 0
               for sd in ("left", "right"))
    assert min(v["hand_thigh"][sd]["min_mm"] for sd in ("left", "right")) >= 1000.0 * P.PNKD_THIGH_CLEAR_M - 1.0
    m = p["arm_rows_median_deg"]
    for j in ("18", "20"):                                   # the left elbow and wrist: the dystonic arm is there
        assert m["clip"][j] >= 0.5 * m["fit"][j] and m["fit"][j] > 10.0
    from mdfm.synth.motions import posture_body
    base = posture_body("sit", P.DUR_S)
    assert P.trunk_report(out, np.arange(out.T), base=base)["ok"]   # at every knot
    for sd in ("left", "right"):
        assert float(P.armrest_depth(out, sd, P.armrest_boxes("chair", out), np.arange(out.T)).max()) == 0.0
    # review of wave SH1c: the right arm swings freely over its armrest, so its collar is not raised (the raise read as
    # repeated shrugs that the clip does not declare); the arm itself held back keeps the attack, and the left collar,
    # under a forearm resting on its armrest, stays raised over the whole clip
    assert float(np.abs(tr["arrays"]["armrest_lift_right_deg"]).max()) == 0.0
    assert float(tr["arrays"]["armrest_lift_left_deg"].min()) > 0.0
    for j in ("17", "19", "21"):
        assert m["clip"][j] >= 0.5 * m["fit"][j]
    lo, hi = P.backrest_box("chair")
    from mdfm.synth import hands as H
    _, _, dom = H._rest()
    ids = np.flatnonzero(np.isin(dom, (13, 14, 16, 17, 18, 19, 20, 21)))
    X = H.body_points(out, ids)                              # the arms at every knot
    assert float(np.maximum(np.minimum(X - lo, hi - X).min(-1), 0.0).max()) == 0.0


# ---------------------------------------------------------------- the clip set
def test_the_clip_set_has_unique_ids_and_known_level0_partners(P):
    from mdfm.synth.exam import SEGMENTS, exam_set
    sp = P.specs()
    ids = [s.clip_id for s in sp]
    assert len(ids) == len(set(ids)) and all(s.since == P.CAT for s in sp)
    known = set(ids) | {s.clip_id for s in exam_set()}
    assert not set(ids) & {s.clip_id for s in exam_set()}
    for s in sp:
        assert s.level0 is None or s.level0 in known
        assert s.segment in SEGMENTS or s.segment_title
        assert s.params_text


@pytest.mark.parametrize("cid", ["post_E7_cd_rotation_g2_duration4", "post_E1_fixed_fist",
                                 "post_E2_task_specific_arm_g2", "post_W_writers_cramp_g1"])
def test_a_clip_and_its_level0_share_the_time_base(P, cid):
    from mdfm.synth.exam import exam_set
    every = {s.clip_id: s for s in exam_set()} | {s.clip_id: s for s in P.specs()}
    s = every[cid]
    b, tr = s.build()
    b0, _ = every[s.level0].build()
    assert b.T == b0.T and b.rate == b0.rate and _finite(b) and _ok(tr["validity"] or {}) == []
    v = tr["validity"] or {}                                 # the scene checks on the catalogue's own build (wave SH1c:
    scene = [k for k in ("table", "armrest", "backrest", "hand_thigh", "hand_trunk") if k in v]   # the mild cramp's
    assert [k for k in scene if not v[k]["ok"]] == []        # edge left the sheet on the 15 s build alone)
    if cid == "post_W_writers_cramp_g1":
        assert scene == ["table"]


# ---------------------------------------------------------------- wave SH1d part B (lane postures)
@pytest.fixture(scope="module")
def cat(P):
    return {s.clip_id: s for s in P.specs()}


FEET = ("post_E7_hemidystonia", "post_E7_multifocal", "post_E7_ped_excerpt", "post_E7_torsion_dystonia_sev2",
        "post_E7_off_period_foot", "post_E1_foot_dystonia_g2", "post_E1_foot_dystonia_g3")


@pytest.mark.parametrize("cid", FEET)
def test_a_dystonic_foot_does_not_skate_on_the_floor(P, cat, cid):
    """Review M11: the touching foot glided 15 to 125 mm while it touched (touch_floor restored only the height of
    the lowest sole point). pin_contact keeps the pinned sole point, once it touches down, within 3 mm of its place
    and the lowest sole point on the floor; read again on the skinned sole cloud of the built clip."""
    from mdfm.synth import contact as CT
    b, tr = cat[cid].build()
    f = tr["params"]["foot"]
    pin = f["pin"]
    assert pin["vertex_in_cloud"] is not None and pin["travel_after_mm"] <= 3.0, pin
    assert tr["validity"]["contact"]["ok"], tr["validity"]["contact"]       # on the floor, never under it
    C = CT.skin_points(b.local_R().numpy(), b.transl.numpy(), b.betas.numpy(), b.gender, f["side"])
    m = pin["vertex_in_cloud"]
    lowest = C[:, :, 1].min(1)
    touch = lowest - lowest.min() < 0.002                 # the knots the foot stands on the floor (standing: from the
    k0 = pin["pinned_from_s"] if "pinned_from_s" in pin else (pin.get("touchdown_s") or 0.0)   # touchdown, or rolled
    touch[:int(round(k0 * b.rate))] = False               # from the first active knot: Phase D)
    xz = C[:, m][touch][:, [0, 2]]
    assert 1000 * np.hypot(*(xz - xz[0]).T).max() <= 3.0, cid
    v = tr["validity"]
    assert _ok(v) == [], _ok(v)


@pytest.mark.parametrize("cid", ["post_E7_off_period_foot"])
def test_a_standing_dystonic_foot_rolls_about_its_lateral_border(P, cat, cid):
    """Phase D of docs/PLAN_2026-11-12.md: a standing foot that inverts and plantarflexes rolls about its lateral
    border, which stays on the floor (pin_roll_standing): the contact vertex lies on the lateral border of the rest
    foot, keeps its horizontal place from the first active knot (the off-period foot's onset slid 63 mm before:
    the page's skate check read 31.67 mm), the lowest sole point stays on the floor (never under it: the variants
    that pinned from the first knot sank the foot 12.6 mm) and the leg moves without a step. The feet held from their
    first knot keep the touchdown pin (pin_contact_standing)."""
    from mdfm.synth import contact as CT
    assert P._has_onset(np.r_[np.zeros(5), np.linspace(0.01, 1.0, 10)]) and not P._has_onset(np.full(10, 0.8))
    b, tr = cat[cid].build()
    f = tr["params"]["foot"]
    pin = f["pin"]
    assert pin["rule"].startswith("pin_roll_standing") and pin["vertex_in_cloud"] is not None, pin
    from mdfm.synth.motions import posture_body
    base = posture_body("stand", 0.01)
    border = P._lateral_border(base, f["side"])
    assert border[pin["vertex_in_cloud"]] and border.sum() < 0.2 * len(border), (pin, int(border.sum()))
    assert pin["travel_after_mm"] < 0.01 and pin["floor_error_after_mm"] < 0.01, pin
    C = CT.skin_points(b.local_R().numpy(), b.transl.numpy(), b.betas.numpy(), b.gender, f["side"])
    k0 = int(round(pin["pinned_from_s"] * b.rate))
    xz = C[k0:, pin["vertex_in_cloud"]][:, [0, 2]]
    assert 1000 * np.hypot(*(xz - xz[0]).T).max() < 0.05, cid      # held from the first active knot, not only once
    v = tr["validity"]
    assert v["contact"]["ok"] and v["contact"]["moved_low_min_m"] > -1e-6, v["contact"]   # never under the floor
    assert v["continuity"]["ok"] and v["continuity"]["max_step_deg"] < 1.0, v["continuity"]
    assert _ok(v) == [], _ok(v)


@pytest.mark.parametrize("kind", ["lateral", "anterior", "posterior"])
def test_the_head_shift_arms_hang_without_a_flick(P, stand15, kind):
    """Review M4: post_E7_cd_shift_lateral's right forearm flicked 45 mm in two knots (the hanging twist switched
    branch knot by knot); an arm no twist holds at some knot now takes none, and the collar and shoulder rows turn
    no faster than ARM_SPEED_MAX_DEG_S."""
    from mdfm.synth import hands as H
    out, tr = P.head_shift(stand15, kind, seed=3)
    a = tr["validity"]["arm_speed"]
    assert a["ok"] and a["deg_s_max"] <= P.ARM_SPEED_MAX_DEG_S, a
    tips = H.body_points(out, [H.tip_vertices()[sd][f] for sd in ("left", "right") for f in ("thumb", "index")])
    acc = np.linalg.norm(np.diff(tips, 2, axis=0), axis=-1) * out.rate ** 2
    assert acc.max() < 5.0, acc.max()


def test_the_twstrs_duration_score_is_read_from_the_realized_course(P, cat):
    """Review M17: the jerky clips held at 65 percent read one TWSTRS B score below their label (constant but mostly
    submaximal); the jerky course is now held at HOLD's maximal share and every cervical clip with a duration score
    reads it back from its realized course (twstrs_duration_of)."""
    import re
    assert P.JERK_HOLD == P.HOLD["maximal"]
    n = 0
    for cid, s in cat.items():
        m = re.search(r"duration(?: score)? (\d)", s.grade_text or "")
        if not cid.startswith("post_E7_cd_") or not m:
            continue
        _, tr = s.build()
        got = tr["params"]["twstrs_duration_realized"]
        assert got["score"] == int(m.group(1)), (cid, got)
        n += 1
    assert n >= 8
    assert "shoulder elevation 1" in cat["post_E7_cd_torticollis_complex"].grade_text

