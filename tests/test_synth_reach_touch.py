"""Tests of the face touch of mdfm/synth/reach.py (shared change SH15 of docs/proposals/SYNTH_SHARED_CHANGES_DRAFT.md),
the paths of the touching tics of mdfm/synth/jerks.py:

- the face targets: the nose tip, the chin below the mouth on the midline, the forehead above the nose, the cheek
  beside the nose on the hand's side, all on the front of the head;
- the pointing hand: hand_blend at 1 is the pointing hand of the reach (POINT_POSE), at 0 the rest, inside the hand
  limits at every share, the other hand at rest; it forms between HAND_FORM_START and HAND_FORM_END of the path;
- the paths from the seated rest (the forearms on the chair arms) and from the hanging arm while standing, to the
  nose, cheek, chin and forehead: the touch between 0 and 3 mm from the skin (near the 1.5 mm aimed at), nothing of
  the hand or forearm inside the head on the way, the trunk and the thighs clear, inside the joint-limit table, the
  collar held, the path from the rest posture to the touch posture with the last part on the straight approach line
  and the index along the pointing direction, the fingertip's speed one bell on a minimum-jerk share, the elbow 7 cm
  or more below the shoulder at the touch (no salute), the arm joints under the touch's speed cap on the clip's
  timing;
- the penetration check sees a finger pushed into the face; a gap search that does not converge raises; the existing
  reach keeps its solver (no bounds given).
Every body is procedural, in the neutral shape; no lockbox, EVAL, CODY or care_pd record is read.
"""
import os

import numpy as np
import pytest
import torch

from mdfm import paths

HAVE_MA = os.path.exists(paths.smplh_model("neutral"))
pytestmark = pytest.mark.skipif(not HAVE_MA, reason="MoveAtlas SMPL+H model not found")


@pytest.fixture(scope="module")
def RC():
    torch.set_num_threads(2)
    from mdfm.synth import reach
    return reach


@pytest.fixture(scope="module")
def all_paths(RC):
    return {(post, tg): RC.touch_path("right", tg, post) for post in RC.TOUCH_POSTURES for tg in RC.FACE_TARGETS}


def test_face_targets_lie_where_their_names_say(RC):
    from mdfm.synth import hands as H
    J, v, dom = H._rest()
    ids = {tg: RC.face_vertex(tg, "right") for tg in RC.FACE_TARGETS}
    assert ids["nose"] == RC.nose_vertex()
    assert all(dom[i] == RC.HEAD and v[i, 2] > J[RC.HEAD, 2] for i in ids.values())
    nose, chin, fore, cheek = (v[ids[k]] for k in ("nose", "chin", "forehead", "cheek"))
    assert abs(chin[0]) < 0.015 and abs(fore[0]) < 0.015
    assert nose[1] - chin[1] > 0.05                               # below the mouth
    assert fore[1] - nose[1] > 0.04 and fore[2] < nose[2]         # above the nose, behind its tip
    assert cheek[0] < -0.03 and abs(cheek[1] - nose[1]) < RC.FACE_TOL_M     # the right cheek at the nose's height
    assert RC.face_vertex("cheek", "left") != ids["cheek"] and v[RC.face_vertex("cheek", "left"), 0] > 0.03
    with pytest.raises(ValueError):
        RC.face_vertex("ear")


def test_the_pointing_hand_forms_inside_the_hand_limits(RC):
    from mdfm.synth import hands as H
    for side in ("left", "right"):
        R = RC.hand_blend(side, [0.0, 0.3, 0.7, 1.0])
        assert torch.allclose(R[-1], RC.point_rotations(side), atol=1e-12)
        assert torch.allclose(R[0], torch.eye(3, dtype=torch.float64).expand(30, 3, 3), atol=1e-15)
        other = slice(15, 30) if side == "left" else slice(0, 15)
        assert torch.allclose(R[:, other], torch.eye(3, dtype=torch.float64).expand(4, 15, 3, 3), atol=1e-15)
        rep = H.hand_limit_report(R)
        assert rep["ok"], rep["violations"]
    s = np.linspace(0.0, 1.0, 11)
    w = RC.hand_share(s)
    assert w[0] == 0.0 and np.all(np.diff(w) >= 0) and np.all(w[s >= RC.HAND_FORM_END] == 1.0)
    assert np.all(w[s <= RC.HAND_FORM_START] == 0.0) and 0.0 < RC.HAND_FORM_START < RC.HAND_FORM_END < 1.0


def test_touch_paths_touch_without_entering_and_stay_clear(RC, all_paths):
    for (post, tg), p in all_paths.items():
        c = p["check"]
        assert c["ok"], (post, tg, c)
        assert 0.0 <= c["touch_gap_mm"] <= 1000 * RC.SKIN_GAP_MAX_M, (post, tg, c)
        assert abs(p["gap_m"] - RC.SKIN_GAP_M) < RC.SKIN_GAP_TOL_M, (post, tg, p["gap_m"])
        assert c["head_min_mm"] >= 0.0 and c["head_min_at_s"] == 1.0, (post, tg, c)    # the touch is the closest
        for part in ("trunk", "thighs"):
            assert c[f"{part}_min_mm"] >= min(1000 * RC.CLEAR_M, c[f"{part}_rest_mm"]) - 0.1, (post, tg, c)
        assert p["ik_residual_mm"] < 1.0, (post, tg)


def test_touch_paths_run_inside_the_table_from_rest_to_touch(RC, all_paths):
    from mdfm.synth.body import limit_violations
    for (post, tg), p in all_paths.items():
        s = np.linspace(0.0, 1.0, 201)
        q = RC.touch_q(p, s)
        np.testing.assert_allclose(q[0], p["q_rest_deg"], atol=1e-9)
        np.testing.assert_allclose(q[-1], p["q_touch_deg"], atol=1e-9)
        assert np.all(np.diff(p["s"]) > 0) and p["s"][0] == 0.0 and abs(p["s"][-1] - 1.0) < 1e-12
        assert np.abs(q[:, 0:3] - p["q_rest_deg"][0:3]).max() < 1e-9                   # the collar held
        b = RC._arm_body("right", post, q, RC.hand_share(s))
        assert limit_violations(b.rot6d)["outside"] == 0, (post, tg)
        ch = RC.chain_on("right", post)
        sl = s >= p["blend_length_m"] / p["length_m"]                                 # the approach line
        P = ch.points(q[sl])
        tip = P["tip"].detach().numpy()
        a, d = p["tip_m"] - RC.TOUCH_APPROACH_M * p["dir"], p["dir"]
        off = (tip - a) - ((tip - a) @ d)[:, None] * d[None]
        assert np.linalg.norm(off, axis=1).max() < 0.002, (post, tg)                  # on the line within 2 mm
        u = (P["tip"] - P["index_mcp"]).detach().numpy()[-1]
        ang = np.degrees(np.arccos(u @ d / np.linalg.norm(u)))
        assert ang < 20.0 and abs(ang - p["dir_error_deg"]) < 1e-6, (post, tg, ang)   # pointing into the face


def test_the_fingertip_speed_is_one_bell_on_a_minimum_jerk_share(RC, all_paths):
    from mdfm.synth.motions import minimum_jerk
    for post in RC.TOUCH_POSTURES:
        p = all_paths[(post, "nose")]
        T = 101
        s = minimum_jerk(np.linspace(0.0, 1.0, T))
        b = RC._arm_body("right", post, RC.touch_q(p, s), RC.hand_share(s))
        tip = RC.fingertip(b, "right")
        runs = RC.speed_runs(tip, 100.0, 0.0, (T - 1) / 100.0)
        assert len(runs) == 1, (post, runs)
        v = np.linalg.norm(np.gradient(tip, 0.01, axis=0), axis=1)
        k = int(np.argmax(v))
        assert 0.3 * T < k < 0.7 * T, (post, k)


def test_the_check_sees_a_finger_inside_the_face(RC, all_paths):
    p = all_paths[("sit", "nose")]
    q = RC._solve_touch("right", "sit", p["tip_m"] + 0.01 * p["dir"], p["q_touch_deg"], p["dir"],
                        RC.TOUCH_POINT_WEIGHT)
    c = RC.clearance(RC._arm_body("right", "sit", q[None], [1.0]), "right", [0], parts=("head",))
    assert c["head"]["hand"][0] < -0.003
    c0 = RC.clearance(RC._arm_body("right", "sit", p["q_touch_deg"][None], [1.0]), "right", [0], parts=("head",))
    assert abs(c0["head"]["hand"][0] - p["gap_m"]) < 1e-9


def test_the_elbow_stays_below_the_shoulder_at_the_touch(RC, all_paths):
    """The review of the first build: a nearly level finger lifted the elbow to the IK's floor 5 cm below the shoulder
    (a salute); the finger now comes from below and every touch keeps the elbow TOUCH_ELBOW_DROP_M or more below the
    shoulder joint, clear of that floor, the forearm rising to the face."""
    for (post, tg), p in all_paths.items():
        c = p["check"]
        ch = RC.chain_on("right", post)
        P = ch.points(p["q_touch_deg"][None])
        sh, el, wr = (P[k][0].detach().numpy() for k in ("shoulder", "elbow", "wrist"))
        assert abs((sh[1] - el[1]) - c["elbow_drop_m"]) < 1e-12, (post, tg)
        assert c["elbow_drop_m"] >= RC.TOUCH_ELBOW_DROP_M > 0.05, (post, tg, c["elbow_drop_m"])
        assert wr[1] > el[1] + 0.05, (post, tg)                                       # the forearm rises to the face
        assert all(RC.TOUCH_DIR_DEG[k]["elevation"] >= 45.0 for k in RC.FACE_TARGETS)


def test_the_arm_turns_under_the_touch_cap_on_the_clip_timing(RC, all_paths):
    """On the minimum-jerk profile of touch_rise_s (the posture's fingertip speed) no arm joint turns faster than
    TOUCH_JOINT_SPEED_CAP_DEG_S (the first build's late shoulder flick); the collar is held; the reach from the hanging
    arm, longer, still lasts longer than from the chair arm."""
    from mdfm.synth import jerks as JK
    for (post, tg), p in all_paths.items():
        pk = p["check"]["joint_peak_deg_s"]
        assert pk == RC.touch_joint_peaks(p) and pk["collar"] < 1e-6, (post, tg, pk)
        assert max(pk.values()) <= RC.TOUCH_JOINT_SPEED_CAP_DEG_S, (post, tg, pk)
        rise = RC.touch_rise_s(p)
        assert abs(rise - RC.MJ_PEAK * p["length_m"] / RC.TOUCH_TIP_SPEED_M_S[post]) < 1e-12
        clip_rise, _ = JK.touch_timing(p)                                            # on the knot above: slower
        assert rise - 1e-9 <= clip_rise < rise + 0.01
        slower = RC.touch_joint_peaks(p, clip_rise)
        assert all(slower[k] <= pk[k] + 1e-9 for k in pk), (post, tg)
    for tg in RC.FACE_TARGETS:
        assert RC.touch_rise_s(all_paths[("stand", tg)]) > RC.touch_rise_s(all_paths[("sit", tg)]), tg


def test_a_gap_search_that_does_not_converge_raises(RC, monkeypatch):
    """The secant search moves the target after its last solve; when it stops unconverged the posture no longer
    matches the target, so touch_path refuses (the review's minor 5)."""
    monkeypatch.setattr(RC, "SKIN_GAP_TOL_M", 1e-12)
    with pytest.raises(ValueError, match="did not converge"):
        RC.touch_path.__wrapped__("right", "chin", "sit")


def test_the_reach_keeps_its_own_bounds(RC):
    ch = RC.chain("left")
    q0 = RC.target_posture("left")
    a = ch.solve(RC.targets("left")["nose_target"], q0, q0)
    b = ch.solve(RC.targets("left")["nose_target"], q0, q0, bounds=(ch.lo, ch.hi))
    np.testing.assert_array_equal(a, b)
    lo, hi = RC.touch_bounds("right", "sit")
    q = RC.chain_on("right", "sit").q_sit
    assert np.all(lo <= q) and np.all(q <= hi) and np.all(lo[0:3] == hi[0:3])
    with pytest.raises(ValueError):
        RC.chain_on("right", "sit_arms_forward")
