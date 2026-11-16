"""Tests of the flowing involuntary-movement family of the symptom catalogue (mdfm/synth/involuntary.py, task
CAT-INVOL): each operator's realized values read back from the body against its level-0 base match the grade
(amplitude as a share of the range of motion, share of time with movement, dominant frequency of the offsets or of
their second derivative inside the declared or published band, rates of the patterns, the count of body parts
involved, mirror ratio, counts of restless-leg movements, extents of pseudoathetosis), no new joint-limit violation
(and a positive control: an offset pushed past the table is scaled back inside), no foot skate where feet are
planted (and a positive control: a foot moved horizontally near the floor skates), the standing feet stay where they
are, the seated lifted foot never sinks, the hand joints stay inside the hand limits, no NaN, the level-0 partners
share their clips' time base, the same seed gives the same body and another seed another one, and the clip
specifications are complete. The findings of the clinical review of 2026-11-02 are checked too: the seated base
rests the hands on the thighs clear of the chair and the body, and keeps them there while the trunk moves; no arm
passes into the chair or the body after the contact guard; the seated trunk never leans back; a lifted foot never
jumps; the standing knees do not snap and the stepping pelvis does not bounce; neighbouring fingers stay together;
hemiballism flings outward, fast and far, and kicks; motor impersistence lapses are visible; constant akathisia
changes movement; a restless foot is up at most 1 s at a time. The seated knees kept apart (handoff 13: the tardive limb
clips' knees came close, then passed through each other): the closing clamp keeps the knees' summed inward swivel under
its ceiling, smoothly (to the second derivative) and leaves an outward knee as it is; the closing ceiling holds the
clearance on the seated posture however the closing is split; the knee guard lifts knees swivelled into each other to
the clearance at every knot with the hips, ankles and feet in place and only near the contact, and leaves knees that
stay apart bit for bit; a contact an outward knee swivel cannot open (the feet carried inward, the knees apart) is
reported, within the guard's total swivel and without a new joint-limit violation; the severe tardive limb movements
and the grade-4 seated chorea of the legs keep every knot's hard leg-to-leg gap (legcontact, every knot measured) at
the clearance, the tardive inward swivel fragments that would close the knees weighted to the ceiling (the clamp left
unused, no knee jolt), the outward swivels and arm amplitude as asked, the minimal step's and the catalogue's mild
step's swivels as asked, the chorea leg amplitude read after the guard; a truth made again (build_guarded) keeps the
knee clearance entry. The findings of the clinical review of 2026-11-03: the seated palms rest on the thighs on the
mesh (the patch of the palm's centre within PALM_CONTACT_M of the skin, physio.signed_distance) with the forearm
sloping down, as the solve in the tree gives them, also while the trunk moves; every flowing transition stays at its
speed cap, the standing grade-4 chorea's wrist under 3 m/s at 30 fps, faster than grade 2's and its amplitude still
at its level; the hands keep off the head, the neck and each other (chorea, piano-playing fingers, the opened
flapping posture); a step in place shifts the weight over the stance foot first (the centre of pressure of the body's
pendulum under it in single support), the heel rising before the ball, the ball up for most of the step and the ankle's
acceleration under its bound; the standing level 0 is not crouched; grade-3 pseudoathetosis
sinks and drifts out, the moving toes burst irregularly, the seated akathisia swing traces an arc, the stereotypy
captions print the clip's bouts, the seated grade-4 chorea kicks (the foot thrown only while airborne, no skate), a
seated moving hand lifts off its thigh. The review of wave SH1d: every contact-guarded clip carries the contact its
arms keep as a validity entry that refuses it (a fake record positive control; the catalogue's seated grade-4 trunk
chorea and grade-1 hemiballism, whose arms the guard had left 3.5 and 5.6 cm inside, now clear, the hands resting on
the thighs where no lift was needed); the seated rest leaves every palm vertex outside the skin. Bodies are procedural,
in the neutral
shape; clips are built shorter than the catalogue's where the test allows, to keep the run fast. No lockbox, EVAL, CODY
or care_pd record is read.
"""
import os

import numpy as np
import pytest
import torch

from mdfm import paths

HAVE_MA = os.path.exists(paths.smplh_model("neutral"))
pytestmark = pytest.mark.skipif(not HAVE_MA, reason="MoveAtlas SMPL+H model not found")
torch.set_num_threads(2)
SHORT = 8.0


@pytest.fixture(scope="module")
def I():
    from mdfm.synth import involuntary
    return involuntary


def _band(x):
    for i, (lo, hi) in enumerate(((0.0, 0.10), (0.10, 0.50), (0.50, 0.90), (0.90, 1.0001)), 1):
        if lo <= x < hi:
            return i
    return 0


def _clean(truth, planted=True):
    v = truth["validity"]
    assert v["limits_new"] == 0, v
    assert v["finite"]
    if planted:
        assert v["skate_added_m_s"] <= 1e-6, v
    assert v["lowest_foot_m"] >= v["lowest_foot_base_m"] - 0.005
    if "hand_limits" in v:
        assert v["hand_limits"]["ok"]


@pytest.fixture(scope="module")
def chorea_sit(I):
    return I.chorea("sit", 3, seconds=SHORT)


@pytest.fixture(scope="module")
def chorea_stand(I):
    return I.chorea("stand", 2, seconds=SHORT)


# ---------------------------------------------------------------- the engine
def test_soft_limit_and_channels(I):
    x = np.linspace(-200, 200, 4001)
    y = I.soft_limit(x, 30.0, 10.0)
    assert y.max() <= 30.0 * I.SOFT_MARGIN + 1e-9 and y.min() >= -10.0 * I.SOFT_MARGIN - 1e-9
    assert (np.diff(y) >= 0).all() and (np.diff(y[(x > -15) & (x < 45)]) > 0).all()
    small = np.abs(x) < 6.0
    assert np.allclose(y[small], x[small])
    from mdfm.synth.base import static_body
    b = static_body("sit", 0.1)
    for c in I.CHANNELS:
        assert I.channel_rom(c) > 0
        p, m = I.channel_room(b, c)
        assert p >= 0 and m >= 0


def test_limits_positive_control(I):
    from mdfm.synth.base import static_body
    from mdfm.synth.body import limit_violations
    b = static_body("stand", 2.0)
    sig = {"elbow_left": np.full(b.T, 400.0), "wrist_flex_left": np.linspace(-300, 300, b.T)}
    offs = I.offsets_of(b, sig)
    raw = b.with_rotations(torch.stack([I._rot(torch.as_tensor(offs.get(j, np.zeros((b.T, 3))))) @ b.local_R()[:, j]
                                        for j in range(22)], 1), list(offs))
    assert limit_violations(raw.rot6d, b.rot6d)["new"] > 0           # the raw offset leaves the table
    out, info = I.apply_offsets(b, offs)
    assert limit_violations(out.rot6d, b.rot6d)["new"] == 0           # the scaled one does not
    assert info["scaled_joints"]


def test_skate_positive_control(I):
    from mdfm.synth.base import static_body
    from mdfm.synth.body import contact_labels, validity
    from dataclasses import replace
    b = static_body("sit", 4.0)
    t = b.times().numpy()
    slide = 0.10 * np.sin(np.pi * np.clip(t - 1.0, 0, 2) / 2) ** 2
    R, _ = I.seated_legs(b, "left", lift=0.01 * slide / 0.1, fwd=slide)          # near the floor, ungated
    out = b.with_rotations(R, [1, 4, 7])
    out = replace(out, contact=contact_labels(out))
    assert validity(b, out)["skate_added_m_s"] > 0.01
    lift = 0.12 * np.sin(np.pi * np.clip(t - 1.0, 0, 2) / 2) ** 2               # lifted first, then travels
    prof, runs = I.airborne_travel(t, lift)
    assert len(runs) == 1 and (prof[lift <= I.AIR_M] == 0).all()
    R, _ = I.seated_legs(b, "left", lift=lift, fwd=0.10 * prof)
    out = b.with_rotations(R, [1, 4, 7])
    out = replace(out, contact=contact_labels(out))
    assert validity(b, out)["skate_added_m_s"] <= 1e-6
    v = np.abs(np.gradient(0.10 * prof, 1.0 / b.rate))                  # no faster than a 2 TRAVEL_MIN_S excursion
    assert v.max() <= 1.875 * 0.10 / I.TRAVEL_MIN_S + 1e-6


# ---------------------------------------------------------------- chorea
def test_chorea_seated_grade(I, chorea_sit):
    out, tr = chorea_sit
    _clean(tr)
    p = tr["params"]
    m = p["measured"]["regions"]
    level = I.AMP[I.UHDRS[3][0]]
    own = p["construction"]["arms_own"]                                 # the arms' own movement on their carriage
    for r in ("trunk", "neck", "arm_left", "arm_right"):
        c = p["construction"]["regions"][r]
        assert c["reachable"]
        a = own[r]["amplitude_share"] if r.startswith("arm") else m[r]["amplitude_share"]
        if r.startswith("arm"):                                         # the contact guard may trim it
            assert 0.8 * level <= a <= 1.02 * level
        else:
            assert a == pytest.approx(level, rel=0.02)
            assert _band(a) == 2                                        # moderate: the upper half of DIS band 2
        sh = own[r]["share"] if r.startswith("arm") else m[r]["share"]
        assert abs(sh - c["nominal_share"]) <= 0.07
        assert 1.0 <= m[r]["dominant_acc_hz"] <= 5.0                    # the 1 to 5 Hz band of the chorea index
    for leg in p["construction"]["legs"]:
        r = f"leg_{leg['side']}"
        assert m[r]["amplitude_share"] == pytest.approx(leg["realized"], abs=1e-6)
        # at most the level within the solve's reachable tolerance (open-body port, wave 3 follow-up, D45): on the
        # open body the grade-3 left leg's gain solve closes on a jump of the amplitude wider than that tolerance
        # (a kick or a lifted run switching on at its threshold) and reads 0.40074 of its ROM against the 0.4 asked
        assert leg["realized"] <= level * 1.01 + 1e-9
        assert leg["reachable"] == (abs(leg["realized"] - level) <= 0.01 * level + 1e-9)
    assert m["n_parts"] == 6
    for side in ("left", "right"):
        f = p["measured"][f"fingers_{side}"]
        assert 0.6 * level <= f["amplitude_share"] <= 1.02 * min(level, I.FINGER_LEVEL_CAP) and f["n_fingers"] >= 3
    # no arm inside the chair or the body, on the final body (open-body port, wave 3 follow-up, D44: the guard leaves
    # the right arm 4.5 cm in where its own choreic movement meets the body, and lift_off, the guards' last resort,
    # clears it; the guard's reading is the intermediate one, arm_contact the clip's, as the catalogue test reads it)
    ac = tr["validity"]["arm_contact"]
    assert ac["ok"] and max(ac["after_cm_max"].values()) <= 100 * I.GUARD_TOL_M + 1e-9, ac


def test_seated_base_hands_on_thighs(I):
    """The seated rest (the clinical review of 2026-11-03, M1: the former palms hovered 39 to 44 mm above the thigh
    skin, the forearms nearly level): on the mesh the palm's centre patch lies within PALM_CONTACT_M of the thigh's
    skin and no palm vertex presses deeper than PALM_PRESS_M; the forearm slopes down to the thigh; the base touches
    neither the chair nor the body; the hands stay on the thighs while the trunk moves."""
    from mdfm.synth import viewer as V
    from mdfm.synth.base import static_body
    from mdfm.synth.body import limit_violations
    b, s0 = I.base_body("sit", 0.1), static_body("sit", 0.1)
    assert limit_violations(b.rot6d, s0.rot6d)["new"] == 0
    J, J0 = b.joints().numpy()[0], s0.joints().numpy()[0]
    assert J[0, 2] - J0[0, 2] == pytest.approx(I.SEAT_FORWARD_M, abs=1e-9)
    for s, (w, el, h, k) in (("left", (20, 18, 1, 4)), ("right", (21, 19, 2, 5))):
        assert abs(J[w, 0] - J[h, 0]) < 0.12 and 0.0 < J[w, 1] - J[h, 1] < 0.25    # the wrist above its thigh
        fa = J[w] - J[el]
        # the forearm slopes down (open-body port, wave 3, D26): the review's complaint was the level forearm of the
        # armrests (0 deg); 25 deg read the SMPL+H rest (31, 34). On the open body the seated shoulder stands 0.33 m
        # over the thigh's top, the upper arm hangs near vertical (0.26 m) and the forearm reaches the palm's place
        # 15.3 deg down at wave 3's rest, 14.2 at the follow-up's (D43, HANDS_ON_THIGHS comment); 10 deg keeps a level
        # forearm failing
        assert np.degrees(np.arctan2(-fa[1], np.linalg.norm(fa[[0, 2]]))) >= 10.0
        u = (J[k] - J[h]) / np.linalg.norm(J[k] - J[h])
        d = J[w] - J[h]
        assert (d - (d @ u) * u)[1] < 0.16                                 # the review read 218 mm above the line
    rep = I.palm_rest_report(b, [0])
    assert rep["ok"], rep
    for s in ("left", "right"):                                         # a light rest, no palm vertex in the skin
        assert 1000 * I.PALM_GAP_M - 0.5 <= rep[s]["patch_median_mm_max"] <= 1000 * I.PALM_CONTACT_M, (s, rep[s])
        assert rep[s]["palm_min_mm"] >= 1000 * I.PALM_MIN_GAP_M - 0.05, (s, rep[s])
    boxes = I._boxes(I.SCENE["sit"], b)                                  # the base touches neither chair nor body
    Vk = V.knot_vertices(b, [0])
    for s in ("left", "right"):
        assert I._penetration(Vk, s, boxes, I._target_ctx(Vk, heads=True))[0] <= I.GUARD_TOL_M
    out, tr = I.chorea("sit", 2, regions=("trunk",), seconds=SHORT, seed=5)   # the hands stay on the thighs
    Jo = out.joints().numpy()
    Jb = b.joints().numpy()[0]
    for w, h in ((20, 1), (21, 2)):
        rel = np.linalg.norm((Jo[:, w] - Jo[:, h]) - (Jb[w] - Jb[h]), axis=-1)
        assert rel.max() < 0.02
    rep = I.palm_rest_report(out, range(0, out.T, 20))                  # the palms on the thighs on the mesh
    assert rep["ok"], rep


def test_the_seated_rest_is_the_solve(I):
    """HANDS_ON_THIGHS is what solve_hands_on_thighs returns (to its 0.01 deg rounding): the numbers come from the
    solve in the tree."""
    x, rec = I.solve_hands_on_thighs()
    for s in ("left", "right"):
        for k in I.HANDS_KEYS:
            assert x[s][k] == pytest.approx(I.HANDS_ON_THIGHS[s][k], abs=0.006), (s, k)
        assert rec[s]["patch_median_m"] >= I.PALM_GAP_M - 1e-9 and rec[s]["palm_min_m"] >= I.PALM_MIN_GAP_M - 1e-9
        assert min(rec[s]["patch_median_m"] - I.PALM_GAP_M, rec[s]["palm_min_m"] - I.PALM_MIN_GAP_M) < 2e-4   # lowest
        assert rec[s]["target_error_mm"] < 2.0                          # the table margin pulls it a little off


def test_chorea_by_region_and_grades(I):
    for g, region, parts in ((1, ("trunk",), {"trunk", "neck"}), (4, ("arms",), {"arm_left", "arm_right"})):
        out, tr = I.chorea("sit", g, regions=region, seconds=SHORT, seed=7)
        _clean(tr)
        m = tr["params"]["measured"]["regions"]
        passive = set() if "arms" in region else {"arm_left", "arm_right"}  # carried, the hands on the thighs
        assert {r for r in I.REGIONS if m[r]["involved"]} - passive == parts
        if "arms" not in region:
            assert "arms_own" not in tr["params"]["construction"]
        a, d = I.UHDRS[g]
        for r in parts:
            if r.startswith("arm"):
                own = tr["params"]["construction"]["arms_own"][r]["amplitude_share"]
                assert 0.8 * I.AMP[a] <= own <= 1.02 * I.AMP[a]
            else:
                assert m[r]["amplitude_share"] == pytest.approx(min(I.AMP[a], I.TRUNK_NECK_CAP), rel=0.02)
    out, tr = I.chorea("sit", 4, regions=("trunk",), seconds=SHORT, seed=7)      # capped trunk and neck
    m = tr["params"]["measured"]["regions"]
    assert m["trunk"]["amplitude_share"] == pytest.approx(I.TRUNK_NECK_CAP, rel=0.02)
    flex = I.channel_values(out, I.base_body("sit", SHORT), "trunk_flex")
    assert flex.min() >= -0.5                                            # seated, the trunk never leans back


def test_chorea_standing_feet_planted(I, chorea_stand):
    out, tr = chorea_stand
    _clean(tr)
    J, Jb = out.joints().numpy(), I.stand_soft(SHORT).joints().numpy()
    assert np.abs(J[:, [7, 8, 10, 11]] - Jb[:, [7, 8, 10, 11]]).max() < 1e-9      # the feet do not move
    m = tr["params"]["measured"]["regions"]
    assert m["pelvis"]["displacement_cm_max"] > 1.0                                  # the pelvis does
    assert out.contact.all() and tr["validity"]["contact_lost_share"] == 0.0


def test_standing_pelvis_no_snap(I):
    out, tr = I.chorea("stand", 4, regions=("legs", "trunk"), seconds=SHORT, seed=3)
    _clean(tr)
    base = I.stand_soft(SHORT)
    k = np.stack([I.channel_values(out, base, f"knee_{s}") for s in ("left", "right")], 1)
    acc = np.gradient(np.gradient(k, 0.01, axis=0), 0.01, axis=0)
    P = np.abs(np.fft.rfft(acc - acc.mean(0), axis=0)) ** 2
    f = np.fft.rfftfreq(len(k), 0.01)
    assert P[f > 8].sum() / P.sum() < 0.2                                # no knee snapping (review: 67 percent)
    assert np.abs(np.gradient(k, 0.01, axis=0)).max() < 250.0           # deg/s
    y = out.joints().numpy()[:, 0, 1] - base.joints().numpy()[:, 0, 1]
    assert y.max() <= 1e-9                                               # the pelvis never rides above the level 0
    out, tr = I.diphasic("stand", 3, seconds=SHORT)
    y = out.joints().numpy()[:, 0, 1] - base.joints().numpy()[:, 0, 1]
    assert y.max() <= 1e-9 and y.min() >= -0.015                         # stepping: no bounce, a dip under 1.5 cm


def test_the_standing_level0_knees(I):
    """The standing level 0 is not crouched (the clinical review of 2026-11-03: 20.1 to 20.5 deg between thigh and
    shank against 13.2, 12.1 to 17.1, on the motion-capture stance ref_stand_E7)."""
    J = I.stand_soft(0.1).joints().numpy()[0]
    for h, k, a in ((1, 4, 7), (2, 5, 8)):
        u, v = J[k] - J[h], J[a] - J[k]
        assert 12.1 <= np.degrees(np.arccos(u @ v / np.linalg.norm(u) / np.linalg.norm(v))) <= 17.1


def test_fingers_coupled(I):
    out, tr = I.piano_fingers(4, seconds=SHORT)
    _clean(tr)
    base = I.base_body("arms_forward", SHORT)
    for s in ("left", "right"):
        v = I.finger_values(out, base, s)
        for a, b in (("index", "middle"), ("middle", "ring"), ("ring", "pinky")):
            assert np.abs(v[a] - v[b]).max() <= I.FINGER_GAP_DEG + 1.0
        X = np.stack([v[f] for f in ("index", "middle", "ring", "pinky")], 1)
        assert np.abs(np.gradient(X, 0.01, axis=0)).max() < 1200.0     # review: up to 3000 deg/s
        assert tr["params"]["measured"][f"fingers_{s}"]["amplitude_share"] <= I.FINGER_LEVEL_CAP + 0.01
    assert tr["params"]["measured"]["regions"]["arm_left"]["involved"]  # grade 4: proximal arm chorea too


def test_seated_legs_no_pop(I):
    out, tr = I.chorea("sit", 3, regions=("legs",), seconds=SHORT, seed=4)
    _clean(tr)
    base = I.base_body("sit", SHORT)
    J, Jb = out.joints().numpy(), base.joints().numpy()
    for a in (7, 8):
        v = np.linalg.norm(np.gradient(J[:, a, [0, 2]] - Jb[:, a, [0, 2]], 0.01, axis=0), axis=-1)
        assert v.max() < 1.0                                             # m/s; review: 20 cm in 1 to 2 frames
    for s in ("left", "right"):
        w = I._region_offsets(out, base, I.LEG[s][:3])
        sp = np.linalg.norm(np.gradient(w, 0.01, axis=0).reshape(out.T, -1, 3), axis=-1).max(1)
        assert sp.max() < 3.5 * max(np.percentile(sp, 99), 60.0)


def test_chorea_determinism(I):
    a, _ = I.chorea("sit", 2, regions=("arms",), seconds=4.0, seed=11, fingers=False)
    b, _ = I.chorea("sit", 2, regions=("arms",), seconds=4.0, seed=11, fingers=False)
    c, _ = I.chorea("sit", 2, regions=("arms",), seconds=4.0, seed=12, fingers=False)
    assert torch.equal(a.rot6d, b.rot6d) and torch.equal(a.transl, b.transl)
    assert not torch.equal(a.rot6d, c.rot6d)


# ---------------------------------------------------------------- speed caps and contacts (review of 2026-11-03)
def _wrist_peak_m_s(body, fps=30.0):
    """The largest wrist speed (m/s) between frames at the page's fps (both wrists)."""
    from mdfm.synth import skeleton as SK
    n = int(np.floor(body.duration * fps + 1e-9)) + 1
    P, _ = SK.chain_at(body, np.arange(n) / fps, False)
    return float(max(np.linalg.norm(np.diff(P[:, w], axis=0), axis=1).max() * fps for w in (20, 21)))


@pytest.fixture(scope="module")
def chorea_stand4(I):
    return I.chorea("stand", 4, seconds=SHORT)


def test_the_flowing_movements_keep_their_speed_caps(I, chorea_stand, chorea_stand4):
    """The clinical review of 2026-11-03 (M9): grade 3 and 4 chorea moved at ballistic speed (the standing grade 4's
    wrist at 5.6 to 6.2 m/s). Every flowing transition of a channel group stays at its cap (the construction's
    realized peaks at the solved gain, the gain read again on the final signals), the wrist of the standing grade 4
    under 3 m/s at the page's 30 fps and faster than grade 2's, and grade 4's arm amplitude still at its level, well
    above grade 2's."""
    for out, tr in (chorea_stand, chorea_stand4):
        _clean(tr)
        for r, rec in tr["params"]["construction"]["regions"].items():
            sp = rec.get("speed")
            if not sp:
                continue
            for g, v in sp["peak_deg_s"].items():
                assert v <= 1.05 * sp["caps_deg_s"][g], (r, g, v)
            assert sp["gain_estimate"] == pytest.approx(rec["gain"], rel=0.05), r
    v2, v4 = _wrist_peak_m_s(chorea_stand[0]), _wrist_peak_m_s(chorea_stand4[0])
    assert v4 < 3.0 and v4 > v2, (v2, v4)
    own2 = chorea_stand[1]["params"]["construction"]["arms_own"]
    own4 = chorea_stand4[1]["params"]["construction"]["arms_own"]
    for r in ("arm_left", "arm_right"):
        assert own4[r]["amplitude_share"] >= 0.8 * I.AMP["marked"] > 2 * own2[r]["amplitude_share"]


def test_speed_caps_in_the_renderer(I):
    """A large flowing transition takes SPEED_PEAK times its excursion over its cap; a group's transitions follow one
    another; without caps the former rendering (the delayed rise shortened) is kept bit for bit."""
    t = np.arange(0.0, 6.0, 0.01)
    ev = [{"t0": 0.5, "rise": 0.25, "hold": 0.0, "ret": 0.0, "amps": {"sh_flex_left": 1.0, "elbow_left": 0.5},
           "delay": {"sh_flex_left": 0.0, "elbow_left": 0.05}, "flow": True, "bout_end": 5.0},
          {"t0": 0.6, "rise": 0.25, "hold": 0.0, "ret": 0.0, "amps": {"sh_abd_left": -1.0},
           "delay": {"sh_abd_left": 0.0}, "flow": True, "bout_end": 5.0}]
    ch = ["sh_flex_left", "sh_abd_left", "elbow_left"]
    caps = {c: I.SPEED_CAP_DEG_S[I.speed_group(c).rsplit("_", 1)[0]] for c in ch}
    dpu = {c: 100.0 for c in ch}                                         # 100 deg per unit
    X = I.render_fragments(ev, ch, t, speed=(caps, dpu))
    V = np.gradient(X * 100.0, 0.01, axis=0)
    sh = np.sqrt(V[:, 0] ** 2 + V[:, 1] ** 2)                            # the shoulder group's angular speed
    assert sh.max() <= 1.01 * I.SPEED_CAP_DEG_S["shoulder"]
    assert np.abs(V[:, 2]).max() <= 1.01 * I.SPEED_CAP_DEG_S["elbow"]
    on = np.flatnonzero(np.abs(V[:, 1]) > 1e-6)
    off = np.flatnonzero(np.abs(V[:, 0]) > 1e-6)
    assert t[on[0]] >= t[off[np.flatnonzero(t[off] < 3.0)[-1]]] - 0.011   # the abduction waits for the flexion
    X0 = I.render_fragments(ev, ch, t)
    assert np.abs(np.gradient(X0[:, 0] * 100.0, 0.01)).max() > 2 * I.SPEED_CAP_DEG_S["shoulder"]   # uncapped
    assert np.allclose(X[-1], 0.0, atol=1e-12) and np.allclose(X0[-1], 0.0)   # back to rest at the bout's end
    # a transition and its return that would not fit before the bout's end go a shorter way or not at all: at rest
    # from the bout's end on (the capped transitions spilled past the bouts, moving 0.72 to 0.93 of a clip at 0.63)
    late = [dict(ev[0], bout_end=1.2), dict(ev[1], t0=1.0, bout_end=1.2)]
    X = I.render_fragments(late, ch, t, speed=(caps, dpu))
    assert np.abs(X[t >= 1.2]).max() < 1e-12
    V = np.gradient(X * 100.0, 0.01, axis=0)
    assert np.sqrt(V[:, 0] ** 2 + V[:, 1] ** 2).max() <= 1.01 * I.SPEED_CAP_DEG_S["shoulder"]



def test_the_rate_governor_bounds_each_segment_and_keeps_the_poses(I, chorea_stand4):
    """Wave SH1d fix B: on fe89df3 the skeleton page read the grade-4 chorea and the step-4 dyskinesias over the plan's
    bounds of a limb (an upper arm at 258 to 323 deg/s, a carried elbow at 368, the standing trunk at 142), the caps of
    each channel group notwithstanding (the collar under the shoulder, the seated carriage, the guard, the pelvis).
    A body whose right shoulder swings through 60 deg in 0.15 s (about 750 deg/s at its peak): after govern_rates
    every reading of segment_rates is at most RATE_MARGIN of its bound (within GOVERN_TOL), the knots outside the
    warped windows are the body's own bit for bit, the swing keeps its excursion, the clip its length; a body under
    the bounds comes back as it is; the standing grade-4 chorea is under the bounds."""
    base = I.base_body("sit", 6.0)
    t = base.times().numpy()

    def mj(x):
        x = np.clip(x, 0.0, 1.0)
        return x ** 3 * (10.0 - 15.0 * x + 6.0 * x ** 2)
    ang = 60.0 * (mj((t - 2.0) / 0.15) - mj((t - 4.0) / 0.15))
    aa = np.zeros((len(t), 3))
    aa[:, 2] = -ang                                         # the right arm abducted
    R = base.local_R().clone()
    R[:, 17] = I._rot(aa) @ R[:, 17]
    body = base.with_rotations(R, [17])
    r0, _ = I._rate_ratio(body)
    assert r0.max() > 2.5
    out, info = I.govern_rates(body)
    r1, _ = I._rate_ratio(out)
    assert info["ok"] and r1.max() <= 1.0 + I.GOVERN_TOL, info
    assert out.T == body.T
    inside = np.zeros(body.T, dtype=bool)
    for w in info["windows"]:
        if w["fits"]:
            inside[int(round(w["s"][0] * body.rate)):int(round(w["s"][1] * body.rate)) + 1] = True
    assert inside.any() and not inside.all()
    assert torch.equal(out.rot6d[~torch.from_numpy(inside)], body.rot6d[~torch.from_numpy(inside)])
    turn = I._log_deg(out.local_R()[:, 17] @ base.local_R()[:, 17].transpose(-1, -2))
    assert np.abs(turn).max(0)[2] == pytest.approx(60.0, abs=0.1)
    same, info0 = I.govern_rates(base)
    assert same is base and info0["windows"] == [] and info0["ok"]
    r4, _ = I._rate_ratio(chorea_stand4[0])
    assert r4.max() <= 1.0 + I.GOVERN_TOL
    assert "rate_governor" in chorea_stand4[1]["params"]["construction"]

def _skin_gap(body, knots, a, b):
    """(n,) m: the smallest distance between the vertex sets a and b (ids) at the knots."""
    from scipy.spatial import cKDTree
    from mdfm.synth import viewer as V
    return np.array([cKDTree(v[b]).query(v[a])[0].min() for v in V.knot_vertices(body, knots)])


def test_hands_keep_off_the_head_and_each_other(I):
    """The clinical review of 2026-11-03 (M7): a choreic index tip came inside the jaw and the hands passed through
    each other (flapping 5 to 6 mm, chorea and dyskinesia down to 4 mm). The guard keeps each forearm and hand
    HEAD_CLEAR_M off the head and neck and PAIR_CLEAR_M off the other arm (its own measure, after the guard), and on
    the mesh no hand vertex comes within half the clearance of the head and neck nor of the other hand; the opened
    flap posture keeps the hands at least 30 mm apart through the flapping."""
    from mdfm.synth import viewer as V
    dom = V.dominant_joint("neutral")
    head = np.flatnonzero(np.isin(dom, (12, 15)))
    sets, _, _ = I._pair_geometry("arms")
    for out, tr in (I.chorea("sit", 4, regions=("trunk", "arms"), seconds=SHORT, seed=2),
                    I.piano_fingers(4, seconds=SHORT)):
        g = tr["params"]["construction"]["guard"]
        for side, x in g["sides"].items():
            assert x["after_cm_max"] <= 100 * I.GUARD_TOL_M + 1e-9, (side, x)
        ks = range(0, out.T, 5)
        for s in ("left", "right"):
            assert _skin_gap(out, ks, sets[s], head).min() >= 0.5 * I.HEAD_CLEAR_M
        assert _skin_gap(out, ks, sets["left"], sets["right"]).min() >= 0.5 * I.PAIR_CLEAR_M
    out, tr = I.stereotypy("flapping", 3, seconds=SHORT)
    l0 = I.base_body("flap", SHORT)
    for b in (l0, out):
        assert _skin_gap(b, range(0, b.T, 4), sets["left"], sets["right"]).min() >= 0.03


# ---------------------------------------------------------------- the other symptoms
def test_hemiballism(I):
    out, tr = I.hemiballism(2, seconds=SHORT)
    _clean(tr)
    m = tr["params"]["measured"]["regions"]
    assert _band(m["arm_left"]["amplitude_share"]) >= 3                  # large flings (DIS amplitude band 3 or 4)
    assert not m["arm_right"]["involved"] and not m["leg_right"]["involved"] and not m["trunk"]["involved"]
    assert m["leg_left"]["involved"]
    bout_s = sum(b - a for a, b in tr["params"]["construction"]["regions"]["arm_left"]["intervals_s"])
    assert 0.9 <= tr["params"]["flings"] / bout_s <= 1.7                  # about 1 to 1.5 flings per s (yahata1997)
    base = I.base_body("sit", SHORT)
    abd = I.channel_values(out, base, "sh_abd_left")
    room = I.channel_room(base, "sh_abd_left")[0]
    assert abd.min() >= -2.0                                             # outward only, never into the trunk
    assert np.percentile(abd, 99) >= 0.6 * room                           # flings reach most of the outward room
    assert np.abs(np.gradient(abd, 0.01)).max() >= 400.0                 # deg/s: a fling, not a gesture
    J, Jb = out.joints().numpy(), base.joints().numpy()
    assert (J[:, 7, 2] - Jb[:, 7, 2]).max() >= 0.12                       # the kick throws the foot forward


def test_impersistence_visible(I):
    out, tr = I.impersistence(3, seconds=SHORT)
    _clean(tr)
    lapses = tr["params"]["lapses"]
    assert lapses and all(10.0 <= x["sink_deg"] <= 20.0 and 15.0 <= x["drop_deg"] <= 25.0 for x in lapses)
    base = I.base_body("arms_forward", SHORT)
    J, Jb = out.joints().numpy(), base.joints().numpy()
    assert (Jb[:, 20, 1] - J[:, 20, 1]).max() >= 0.08 or (Jb[:, 21, 1] - J[:, 21, 1]).max() >= 0.08   # the arm sinks


def test_athetosis_and_choreoathetosis(I):
    out, tr = I.athetosis(2, 4, seconds=SHORT)
    _clean(tr)
    m = tr["params"]["measured"]
    for s in ("left", "right"):
        r = m["regions"][f"arm_{s}"]
        own = tr["params"]["construction"]["arms_own"][f"arm_{s}"]["amplitude_share"]
        assert 0.8 * I.DIS_LEVELS[2] <= own <= 1.02 * I.DIS_LEVELS[2]
        assert 0.25 <= r["dominant_hz"] <= 0.7                           # neilson1974: 0.3 to 0.6 Hz
        assert r["share"] >= 0.8
        # the palms on the thighs (review M1): the guard may take the fingers a little under the level
        share = m[f"fingers_{s}"]["amplitude_share"]
        assert I.GUARDED_FINGER_SHARE * I.DIS_LEVELS[2] <= share <= 1.02 * I.DIS_LEVELS[2]
    out2, tr2 = I.athetosis(2, 3, seconds=SHORT, chorea_mix=True)
    _clean(tr2)
    assert tr2["params"]["measured"]["regions"]["trunk"]["involved"]


def test_pseudoathetosis_extent(I):
    _, t1 = I.pseudoathetosis(1, seconds=SHORT)
    _, t3 = I.pseudoathetosis(3, seconds=SHORT)
    _clean(t1)
    _clean(t3)
    m1, m3 = t1["params"]["measured"], t3["params"]["measured"]
    assert not m1["regions"]["arm_left"]["involved"] and m1["fingers_left"]["n_fingers"] >= 3
    assert m3["regions"]["arm_left"]["involved"] and not m3["regions"]["arm_right"]["involved"]
    assert m3["regions"]["arm_left"]["dominant_hz"] <= 0.5


def test_lid_peak_dose_band(I):
    out, tr = I.lid_peak_dose("stand", 3, seconds=SHORT)
    _clean(tr)
    m = tr["params"]["measured"]["regions"]
    own = tr["params"]["construction"]["arms_own"]
    for r in ("trunk", "arm_left", "arm_right"):
        a = own[r]["amplitude_share"] if r.startswith("arm") else m[r]["amplitude_share"]
        assert 0.8 * I.STEPS[3][0] <= a <= 1.02 * I.STEPS[3][0]
        assert 1.0 <= m[r]["dominant_acc_hz"] <= 4.5                    # manson2000 1 to 3 Hz, hoff2001 1 to 4 Hz
    J, Jb = out.joints().numpy(), I.stand_soft(SHORT).joints().numpy()
    assert np.abs(J[:, [7, 8]] - Jb[:, [7, 8]]).max() < 1e-9


def test_diphasic(I):
    out, tr = I.diphasic("sit", 2, seconds=SHORT)
    _clean(tr)
    m = tr["params"]["measured"]["regions"]
    for s in ("left", "right"):
        assert m[f"leg_{s}"]["amplitude_share"] == pytest.approx(I.STEPS[2][0], rel=0.03)
        assert m[f"leg_{s}"]["dominant_hz"] == pytest.approx(I.DIPHASIC_HZ, abs=0.1)
    out, tr = I.diphasic("stand", 3, seconds=SHORT)
    _clean(tr)
    lift = out.joints().numpy()[:, [10, 11], 1] - I.stand_soft(SHORT).joints().numpy()[:, [10, 11], 1]   # the balls
    assert lift.min() > -1e-9 and lift.max() == pytest.approx(I.STEP_LIFT_M[3], rel=0.15)


def test_a_foot_lifts_only_over_a_full_shift(I):
    """On a bout's ramps (step_track): wherever a ball is up, the centre of pressure laid down lies under the other
    foot's centre and the stepping heel has its full pitch (the catalogue's diphasic stepping lifted a foot on the ramp
    with the shift partial); the pelvis shift is the pendulum's answer to that centre of pressure (lip_com: x -
    x''/omega0^2 gives it back away from the ends)."""
    t = np.arange(0.0, 6.0, 0.01)
    env = I.ramp_envelope(t, [(1.0, 5.0)], 0.6)
    ph = 2 * np.pi * 0.8 * t
    geo = {"half_width_m": 0.1, "com_gain": 0.85, "omega0": 3.0}
    lift, heel, dx, cop = I.step_track(ph, np.ones_like(t), env, 0.1, geo, 100.0)
    for i, sg in ((0, -1.0), (1, 1.0)):
        up = lift[:, i] > 1e-6
        assert up.any()
        assert np.all(sg * cop[up] >= 0.1 - 1e-9) and np.all(heel[up, i] >= I.HEEL_DEG - 1e-6)
    x = dx * geo["com_gain"]
    back = x - np.gradient(np.gradient(x, 0.01), 0.01) / geo["omega0"] ** 2
    assert np.abs(back - cop)[50:-50].max() < 0.003
    assert np.abs(np.gradient(dx, 0.01)).max() < 0.4                 # the first answer crossed 18 cm in 0.28 s


def test_a_step_in_place_shifts_the_weight_first(I):
    """The clinical review of 2026-11-03 (M12): the standing feet lifted with the pelvis swaying 3 cm unrelated to
    the steps, heel and toe together. Stepping (diphasic, akathisia's walking on the spot) now keeps the centre of
    pressure of the body's pendulum (hof2005 eq. 2 on the de Leva centre of mass) within COP_STANCE_TOL_M of the
    stance foot's centre while the other foot's ball is 1 cm or more up, the pelvis already toward the stance foot
    when the ball leaves, the heel risen before it, the foot points not sliding, and the pelvis and hands no faster
    than 0.35 m/s (the first answer of this wave carried them at 1.2 to 2.2 m/s). The review of wave SH1d: the steps
    had become flicks (the ball up 0.13 to 0.20 s, the ankle at 38 to 51 m/s^2): the ankle's acceleration now stays
    under STEP_ACCEL_CAP_M_S2 and a full step of the diphasic stepping keeps its ball 1 cm or more up for at least 0.4
    of its half cycle."""
    from mdfm.synth.body import foot_points
    for name, (out, tr) in (("diphasic", I.diphasic("stand", 3, seconds=SHORT)),
                            ("akathisia", I.akathisia("stand", 3, seconds=12.0, seed=3))):
        _clean(tr)
        base = I.stand_soft(out.duration)
        leg = tr["params"]["construction"]["legs"][0]
        st = leg["stance"]
        again = I.stance_report(out, base, omega0=leg["step_geometry"]["omega0"])
        assert st["ok"] and again["ok"]
        for s in ("left", "right"):
            assert again[s]["cop_to_stance_mm_max"] == pytest.approx(st[s]["cop_to_stance_mm_max"], abs=0.01)
            assert st[s]["single_support_share"] > 0.03, (name, st)
            assert st[s]["cop_to_stance_mm_max"] <= 1000 * I.COP_STANCE_TOL_M, (name, s, st[s])
            assert st[s]["ankle_rise_at_ball_off_mm_min"] >= 10.0, (name, s, st[s])     # heel before toe
        F, Fb = foot_points(out.joints()).numpy(), foot_points(base.joints()).numpy()
        J, Jb = out.joints().numpy(), base.joints().numpy()
        down = (J[:, [10, 11], 1] - Jb[:, [10, 11], 1]) < 1e-9
        assert np.abs((F - Fb)[..., [0, 2]][down]).max() < 1e-6, name        # a foot on its ball does not slide
        assert 0.04 <= leg["pelvis_shift_m_max"] <= 0.08, leg["pelvis_shift_m_max"]
        v = np.linalg.norm(np.gradient(J[:, [0, 20, 21]], 1.0 / out.rate, axis=0), axis=-1)
        assert v.max() < 0.35, (name, v.max())
        lat = I._lateral_frame(base)
        dx = ((J[:, 0] - Jb[:, 0]) * lat).sum(-1)
        n_full = 0
        for i, sg in ((0, -1.0), (1, 1.0)):                              # the left foot steps over a shift right
            h = J[:, 10 + i, 1] - Jb[:, 10 + i, 1]
            for a, b in I._runs(h > 1e-4):
                if h[a:b].max() < 0.9 * h.max():                       # a step on a bout's ramp
                    continue
                n_full += 1
                assert sg * dx[a] >= 0.015, (name, i, a, dx[a])        # shifted 1.5 cm or more before it leaves
        assert n_full >= 3, name
        acc = np.linalg.norm(np.diff(J[:, [7, 8]], 2, axis=0), axis=-1) * out.rate ** 2
        assert acc.max() <= I.STEP_ACCEL_CAP_M_S2, (name, acc.max())     # the review read 38 to 51 m/s^2
        if name == "diphasic":
            air = [(b - a) / out.rate for i in (0, 1) for a, b in I._runs(J[:, 10 + i, 1] - Jb[:, 10 + i, 1] >= 0.01)
                   if (J[a:b, 10 + i, 1] - Jb[a:b, 10 + i, 1]).max() >= 0.9 * I.STEP_LIFT_M[3]]
            assert air and np.median(air) >= 0.4 * 0.5 / I.DIPHASIC_HZ, air   # the review read 0.13 to 0.20 s


def test_the_arms_contact_is_a_validity_entry(I):
    """The review of wave SH1d (blocker 1): the guard's own reading of what it left inside (3.4 to 5.9 cm on 13
    catalogue builds) was recorded and refused nothing. arm_contact reads the last contact on the final body (the
    guard's, a seated lift's, or build_guarded's with the two arms' pair) and refuses above GUARD_TOL_M; the seated
    carriage's runs no swing or raise cleared are counted (positive controls on records)."""
    tol = 100 * I.GUARD_TOL_M
    g = {"sides": {"left": {"after_cm_max": 0.5}, "right": {"after_cm_max": tol + 0.1}}}
    e = I.arm_contact({"guard": g})
    assert e["read_after"] == "guard_arms" and not e["ok"]
    after = {"left": {"after_cm_max": 0.5}, "right": {"after_cm_max": tol - 0.1}}
    e = I.arm_contact({"guard": g, "contact_after": after})
    assert e["read_after"] == "lift_off" and e["ok"]
    e = I.arm_contact({"guard": g, "contact_after": after, "body_after": after,
                       "pair_after": {"after_cm_max": tol + 0.1}})
    assert e["read_after"] == "build_guarded" and not e["ok"]
    sw = {"left": {"runs": [{"clear": False}]}, "right": {"runs": [{"clear": True}]}}
    e = I.arm_contact({"guard": g, "contact_after": after, "arms": {"elbow_swivel": sw}})
    assert e["ok"] and e["carriage_runs_not_cleared"] == 1
    assert I.arm_contact({}) is None


@pytest.fixture(scope="module")
def cat_seated(I):
    sp = {s.clip_id: s for s in I.specs()}
    return {c: sp[c].build() for c in ("invol_chorea_E1_trunk_g3", "invol_chorea_E1_trunk_g4",
                                       "invol_hemiballism_E1_left_g1", "invol_choreoathetosis_E1_a3_d3")}


def test_a_seated_elbow_swings_out_with_the_hand_in_place(I):
    """The settling of 2026-11-10 (invol_choreoathetosis_E1_a3_d3, refused by its own arm contact: the trunk came
    onto a right forearm whose hand rested on the thigh, 0.3 mm beyond the tolerance, where neither the elbow's lift nor
    an outward turn of the shoulder cleared it): _swivel_arm swings the elbow outward about the shoulder-to-wrist line,
    the wrist's place and the hand's global rotation kept, the elbow moving along the body's lateral axis."""
    base = I.base_body("sit", 0.2)
    J0, G0 = base.fk()
    lat = torch.from_numpy(I._lateral_frame(base))
    for side in ("left", "right"):
        out = I._swivel_arm(base, side, np.full(base.T, 10.0))
        J, G = out.fk()
        _, sh, el, wr = I.ARM[side]
        assert float((J[:, wr] - J0[:, wr]).norm(dim=-1).max()) < 1e-9
        assert float((G[:, wr] - G0[:, wr]).abs().max()) < 1e-9
        moved = ((J[:, el] - J0[:, el]) * lat).sum(-1) * I.SIDE_SIGN[side]
        assert float(moved.min()) > 0.005, (side, moved)                 # outward, by millimetres
        assert float((J[:, sh] - J0[:, sh]).norm(dim=-1).max()) < 1e-9  # about the shoulder


def test_catalogue_seated_arms_keep_out_of_the_body(I, cat_seated):
    """The review of wave SH1d (blocker 1, major 4), on the catalogue's seeds: the seated grade-4 trunk chorea had its
    right forearm and thumb 0.8 to 3.4 cm in the flank and the thigh, the wrist 119 mm off its thigh, and the grade-1
    hemiballism its thumb 5.6 cm in the thigh, refused by nothing. Now every arm reading is under the tolerance
    (arm_contact) and every lift clears; on the trunk chorea, outside the runs that lift a hand (the carriage's raised
    wrists and lift_off's runs, 0.4 s around them; the elbow's outward swing keeps the hand, D42), the wrist stays
    where its thigh carries it (within 20 mm) and no palm vertex presses into the skin more
    than PALM_PRESS_M; at grade 3 the palms rest on the thighs there (palm_rest_report; at grade 4 the left wrist,
    turned to its range by the trunk's rotation, lets the palm roll up to 15 mm off). The settling of 2026-11-10: the
    choreoathetosis a3_d3, refused by its arms' contact (left 0.95 cm, right 0.83 cm: the other hand's clearance
    between the guard's knots, and the trunk onto the right forearm), clears with lift_off's runs (on the open
    body's follow-up rest, D43, without the elbow's outward swing)."""
    for cid, (out, tr) in cat_seated.items():
        _clean(tr)
        ac = tr["validity"]["arm_contact"]
        assert ac["ok"], (cid, ac)
        c = tr["params"]["construction"]
        for v in (c.get("lift") or {}).values():
            assert all(r["clear"] and r["kept"] for r in v["runs"]), (cid, v)
        if "choreoathetosis" in cid:
            # the settling of 2026-11-10 needed the elbow's outward swing for the trunk coming onto the right forearm;
            # on the follow-up's rest (D43) the elbow's skin lies 44.6 mm from the flank and lift_off clears the hands
            # with the elbow's lift and the shoulder's turn (the swing's own test: the elbow swings out, the hand kept)
            assert not c.get("lift_dropped") and ac["read_after"] == "lift_off"
            continue
        if "trunk" not in cid:
            continue
        base = I.base_body("sit", out.duration)
        keep = np.ones(out.T, bool)
        pad = int(round(0.4 * out.rate))
        # the runs that lift the hand off its thigh: the carriage's raised wrist targets and lift_off's runs. The
        # carriage's elbow swing alone (raise 0) turns the arm about the shoulder-to-wrist line with the wrist's place
        # and the hand's global rotation kept, so the hand is read there too (open-body port, wave 3 follow-up, D42:
        # on the open body the trunk chorea brings the flank onto the elbow, which rests 3.5 mm from it, in nine runs
        # of the grade-3 clip, and all the swing runs with their pads covered half of it; since the wrist's twist is
        # handed to the elbow, D41, the hand inside them stays where its thigh carries it)
        runs = [r["knots"] for v in c["arms"]["elbow_swivel"].values() for r in v["runs"] if r["raise_m"] > 0.0]
        runs += [r["knots"] for v in (c.get("lift") or {}).values() for r in v["runs"]]
        for k0, k1 in runs:
            keep[max(k0 - pad, 0):k1 + pad + 1] = False
        assert keep.mean() >= 0.75, (cid, keep.mean())
        J, Jb = out.joints().numpy(), base.joints().numpy()
        for w, h in ((20, 1), (21, 2)):
            rel = np.linalg.norm((J[:, w] - J[:, h]) - (Jb[:, w] - Jb[:, h]), axis=1)
            assert rel[keep].max() < 0.02, (cid, w, rel[keep].max())
        rep = I.palm_rest_report(out, [k for k in range(0, out.T, 20) if keep[k]])
        for s in ("left", "right"):
            assert rep[s]["palm_min_mm"] >= -1000 * I.PALM_PRESS_M, (cid, s, rep[s])
            assert rep[s]["patch_median_mm_max"] <= 1.5 * 1000 * I.PALM_CONTACT_M, (cid, s, rep[s])
        if cid.endswith("g3"):
            assert rep["ok"], (cid, rep)


def test_tardive(I):
    out, tr = I.td_rocking(3, seconds=SHORT)
    _clean(tr)
    b = I.base_body("sit", SHORT)
    (J, G), (Jb, Gb) = (x.numpy() for x in out.fk()), (x.numpy() for x in b.fk())
    v, vb = G[:, 9] @ np.array([0.0, 1.0, 0.0]), Gb[:, 9] @ np.array([0.0, 1.0, 0.0])    # the upper trunk's axis
    pitch = np.degrees(np.arctan2(v[:, 2], v[:, 1])) - np.degrees(np.arctan2(vb[:, 2], vb[:, 1]))
    assert (pitch.max() - pitch.min()) == pytest.approx(I.TD_ROCK_DEG[3], rel=0.12)
    assert pitch.min() >= -0.5                                           # forward from upright: never toward the back
    assert np.abs(J[:, [7, 8, 10, 11]] - Jb[:, [7, 8, 10, 11]]).max() < 1e-9          # the legs do not move
    rel = np.linalg.norm((J[:, 20] - J[:, 1]) - (Jb[:, 20] - Jb[:, 1]), axis=-1)
    assert rel.max() < 0.02                                              # the hands stay on the thighs
    out, tr = I.td_pelvic_thrust(3, seconds=SHORT)
    _clean(tr)
    m = tr["params"]["measured"]["regions"]["pelvis"]
    assert m["displacement_cm_pp_forward"] == pytest.approx(100 * I.TD_THRUST_M[3], rel=0.25)  # jittered cycles
    out, tr = I.td_limbs(2, seconds=SHORT)
    _clean(tr)
    m = tr["params"]["measured"]["regions"]
    own = tr["params"]["construction"]["arms_own"]["arm_left"]["amplitude_share"]
    assert 0.8 * I.STEPS[2][0] <= own <= 1.02 * I.STEPS[2][0]
    assert 0.7 <= m["arm_left"]["dominant_acc_hz"] <= 3.5


@pytest.mark.parametrize("kind,region", [("rocking", "trunk"), ("flapping", "arm_left"), ("nodding", "neck")])
def test_stereotypy_rate(I, kind, region):
    out, tr = I.stereotypy(kind, 3, seconds=SHORT)
    _clean(tr)
    r = tr["params"]["measured"]["regions"][region]
    assert r["dominant_hz"] == pytest.approx(I.STEREO[kind]["rate_hz"], rel=0.06)
    assert r["share"] >= 0.8
    if kind != "rocking":                                  # rocking splits its pitch with the pelvis
        assert r["amplitude_deg"] == pytest.approx(I.STEREO[kind]["deg_pp"], rel=0.25)   # jittered cycles


def test_akathisia(I):
    for posture in ("sit", "stand"):
        out, tr = I.akathisia(posture, 2, seconds=15.0, seed=3)
        _clean(tr)
        moving = np.zeros(out.T, bool)
        from mdfm.synth.base import static_body
        base = static_body("sit", 15.0) if posture == "sit" else I.stand_soft(15.0)
        for r in ("trunk", "leg_left", "leg_right"):
            W = I._region_offsets(out, base, I.REGIONS[r])
            moving |= I._speed_deg_s(W, out.rate) > I.V_MOVE_DEG_S
        share = I.close_mask(moving, out.rate).mean()
        nominal = sum(b - a for a, b in (x["interval_s"] for x in tr["params"]["bouts"])) / out.duration
        assert abs(share - nominal) <= 0.12, (posture, share, nominal)
        assert nominal >= 0.5                                           # BARS 2: at least half the observation
    out, tr = I.akathisia("sit", 3)                                      # constant: several movements, none filling it
    b = tr["params"]["bouts"]
    assert len({x["movement"] for x in b}) >= 2 and len(b) >= 4
    assert max(x["interval_s"][1] - x["interval_s"][0] for x in b) <= I.AKATHISIA_SEGMENT_S[1] + 2.0
    assert all(x["movement"] != y["movement"] for x, y in zip(b, b[1:]))
    assert tr["params"]["measured"]["regions"]["leg_left"]["involved"]


def test_restless_legs_count(I):
    out, tr = I.restless_legs(3, seconds=30.0)
    _clean(tr)
    n = len(tr["params"]["movements"])
    from mdfm.synth.base import static_body
    b = static_body("sit", 30.0)
    Jo, Jb = out.joints().numpy(), I.base_body("sit", 30.0).joints().numpy()
    for a in (7, 8):                                                     # the foot up at most 1 s at a time
        up = (Jo[:, a, 1] - Jb[:, a, 1]) > 0.05
        assert all((e - s_) / out.rate <= 1.0 + 1e-9 for s_, e in I._runs(up))
    runs = 0
    for s in ("left", "right"):
        W = I._region_offsets(out, b, I.REGIONS[f"leg_{s}"]).reshape(out.T, -1, 3)
        away = np.linalg.norm(W, axis=-1).max(1) > 0.5                   # the leg away from its level-0 posture
        runs += len(I._runs(away))
    assert runs == n == max(1, round(I.RLS_INDEX[3] * 30.0 / 3600))


def test_moving_toes(I):
    out, tr = I.moving_toes(3, seconds=SHORT)
    _clean(tr)
    from mdfm.synth import contact as CT
    b = I.base_body("sit", SHORT)
    m = tr["params"]["measured"]["regions"]["leg_left"]
    assert m["amplitude_channel"] == "toes_left"
    # open-body port, wave 3 (D26): the planted foot's toes rise only, and on the anatomical table their extension
    # room is 40 of a 75 deg ROM, so grade 3 reaches I.toes_level(3) = 0.97 x 40 / 75 (the soft limit's share), inside
    # DIS band 3 (50 to under 90 percent) and short of its centre, 0.70 on the AMASS table
    assert I.toes_level(3) == pytest.approx(I.SOFT_MARGIN * 40.0 / 75.0, abs=1e-6)
    assert m["amplitude_share"] == pytest.approx(I.toes_level(3), rel=0.02) and m["amplitude_share"] >= 0.5
    assert np.abs(out.joints().numpy()[:, [7, 10]] - b.joints().numpy()[:, [7, 10]]).max() < 1e-9
    low = CT.mesh_sole_low(out, knots=range(0, out.T, 10))[:, 0]
    low_b = CT.mesh_sole_low(b, knots=[0])[0, 0]
    assert low.min() >= low_b - 0.005                                   # the toes never pass below the floor


def test_mirror_movements(I):
    l0, t0 = I.mirror_movements(None)
    for g in (2, "2b", 4):
        out, tr = I.mirror_movements(g)
        assert out.T == l0.T
        v = tr["validity"]
        assert v["limits_new"] == 0 and v["hand_limits"]["ok"] and v["finite"]
        m = tr["params"]["measured"]
        if g == "2b":
            assert m["cycles_with_mirror"] == 3
        else:
            assert m["ratio_median"] == pytest.approx(I.MIRROR_RATIO[g], rel=0.03)
            assert m["cycles_with_mirror"] == 10 and abs(m["lag_s"]) <= 0.02
        assert torch.equal(out.hand6d[:, 15:30], l0.hand6d[:, 15:30])   # the voluntary hand is the level 0's


# ---------------------------------------------------------------- the minors of the review of 2026-11-03
def test_the_minor_findings(I):
    """Pseudoathetosis grade 3 sinks and drifts out (its 5 s means, not only a wander about the posture); the moving
    toes burst at irregular intervals; the seated akathisia swing traces an arc (the foot higher forward than back);
    the stereotypy captions print the clip's own bouts; the peak-dose dyskinesia has its own seed; the UDysRS
    anchors are slight, mild, moderate, severe and the texts say score."""
    out, tr = I.pseudoathetosis(3, seconds=I.CHOREA_S)
    _clean(tr)
    b = I.base_body("arms_forward", I.CHOREA_S)
    d = out.joints().numpy()[:, 20] - b.joints().numpy()[:, 20]
    last = d[-500:].mean(0)
    assert -last[1] >= 0.025 and last[0] >= 0.012, last                  # down (y) and out (the left: +x)
    out, tr = I.moving_toes(3, seconds=I.CHOREA_S)
    _clean(tr)
    x = I.channel_values(out, I.base_body("sit", I.CHOREA_S), "toes_left")
    on = np.array([a for a, _ in I._runs(x > 0.3 * x.max())]) / out.rate
    gaps = np.diff(on)
    assert len(gaps) >= 8 and gaps.std() / gaps.mean() >= 0.2           # irregular (the review: regular 1.35 Hz)
    out, tr = I.akathisia("sit", 3, seconds=I.AKATHISIA_S, seed=3)
    b = I.base_body("sit", I.AKATHISIA_S)
    J, Jb = out.joints().numpy(), b.joints().numpy()
    swings = [x for x in tr["params"]["bouts"] if x["movement"] == "swing"]
    assert swings
    for x in swings:
        k0, k1 = (int(round(v * out.rate)) for v in x["interval_s"])
        for a in (7, 8):
            fw, up = J[k0:k1, a, 2] - Jb[k0:k1, a, 2], J[k0:k1, a, 1] - Jb[k0:k1, a, 1]
            if fw.max() < 0.10:
                continue
            hi, lo = up[fw > 0.8 * fw.max()].mean(), up[np.abs(fw) < 0.02].mean()
            assert hi - lo >= 0.01, (x, hi, lo)                         # the foot rises as it swings forward
    sp = {s_.clip_id: s_ for s_ in I.specs()}
    for g in (1, 2, 3):                                                 # the clip's own bouts in its caption
        out, tr = I.stereotypy("flapping", g, seconds=I.CHOREA_S)
        iv = tr["params"]["construction"]["regions"]["arm_left"]["intervals_s"]
        assert I._bouts_text([tuple(v) for v in iv]) in sp[f"invol_stereotypy_flapping_g{g}"].params_text
    for kind in ("rocking", "nodding"):
        for g in (1, 2, 3):
            assert I._bouts_text(I.stereotypy_bouts(kind, g)) in sp[f"invol_stereotypy_{kind}_g{g}"].params_text
    assert I.LID_SEED != I.SEED
    assert list(I.UDYSRS_TERMS.values()) == ["slight", "mild", "moderate", "severe"]
    for s_ in sp.values():
        assert " step " not in s_.grade_text, s_.clip_id


def test_seated_chorea_kicks_at_grades_3_and_4(I):
    """The clinical review of 2026-11-03 (minor of M8): grade 4 leg chorea showed no kick and read as moderate. At
    grades 3 and 4 some seated foot excursions are thrown forward (the ankle well forward of its place, the knee
    extending), at grade 2 none; the legs keep their level and the knees their clearance."""
    fwd = {}
    for g in (2, 4):
        out, tr = I.chorea("sit", g, regions=("legs",), seconds=SHORT, seed=6)
        _clean(tr)
        b = I.base_body("sit", SHORT)
        J, Jb = out.joints().numpy(), b.joints().numpy()
        fwd[g] = float((J[:, [7, 8], 2] - Jb[:, [7, 8], 2]).max())
        v = np.linalg.norm(np.gradient(J[:, [7, 8]], 1.0 / out.rate, axis=0), axis=-1)
        assert v.max() < 1.5, (g, v.max())                              # a choreic kick, not a fling (KICK_MIN_S)
        assert tr["validity"]["knee_clearance"]["ok"]
        for leg in tr["params"]["construction"]["legs"]:
            assert leg["realized"] <= I.AMP[I.UHDRS[g][0]] + 1e-6
    assert fwd[4] >= 0.12 and fwd[4] >= fwd[2] + 0.1, fwd                # the kick's own travel: 130 mm on this
    #                                                                     build (66 mm through the common gate)


def test_supported_arm_lifts_off_the_thigh(I):
    """Seated, a flowing arm fragment flexes the elbow at least SUPPORTED_LIFT of the largest target the arm's other
    channels hold (lift_events: rolled or swung on the thigh, a resting hand pressed a thumb 5.9 cm into it), back
    to rest with them at the bout's end; the elbow's own larger target is kept."""
    rng = np.random.default_rng(3)
    chans = list(I.arm_ch("right"))
    ev = I.fragment_events([(0.5, 4.0), (5.0, 7.5)], chans, rng, I.arm_timing("chorea", "sit"), None,
                           I.arm_signs("right", "sit"))
    ev = sorted(ev, key=lambda e: e["t0"])
    asked = [e["amps"].get("elbow_right") for e in ev]                  # the elbow's own targets (None: not moved)
    I.lift_events(ev, "elbow_right")
    cur, own, bout = {}, 0.0, None
    for e, o in zip(ev, asked):
        if e["bout_end"] != bout:
            cur, own, bout = {}, 0.0, e["bout_end"]
        own = own if o is None else abs(o)
        cur.update({c: a for c, a in e["amps"].items() if c != "elbow_right"})
        want = I.SUPPORTED_LIFT * max((abs(v) for v in cur.values()), default=0.0)
        assert e["amps"]["elbow_right"] == pytest.approx(max(own, want)) and e["amps"]["elbow_right"] >= 0.0
    assert any(o is None or abs(o) < e["amps"]["elbow_right"] - 1e-9 for e, o in zip(ev, asked))   # lifted
    t = np.arange(0.0, 8.0, 0.01)
    X = I.render_fragments(ev, chans, t)
    el = X[:, chans.index("elbow_right")]
    assert el.min() >= -1e-9 and abs(el[-1]) < 1e-12                     # one-sided, back to rest


def test_hand_room(I):
    """Seated, the wrists' targets slide apart only where the thighs bring them closer than HAND_APPROACH_M to each
    other than at rest (the knees swinging in carried the thumbs within 5 mm of each other): zero at rest, and the
    separation never under the rest's less the approach and half the soft blend."""
    base = I.base_body("sit", 0.5)
    J = base.joints()
    lat = torch.from_numpy(I._lateral_frame(base))
    Wl, Wr = J[:, I.ARM["left"][3]].clone(), J[:, I.ARM["right"][3]].clone()
    assert float(I.hand_room(base, base, Wl, Wr).abs().max()) == 0.0
    sep0 = float(((Wl - Wr) * lat).sum(-1)[0])
    close = torch.linspace(0.0, 0.06, base.T, dtype=torch.float64)[:, None] * lat
    Wl2, Wr2 = Wl - 0.5 * close, Wr + 0.5 * close
    sl = I.hand_room(base, base, Wl2, Wr2)
    sep = ((Wl2 + sl - (Wr2 - sl)) * lat).sum(-1).numpy()
    assert sep.min() >= sep0 - I.HAND_APPROACH_M - 0.5 * I.HAND_ROOM_SOFT_M - 1e-9
    assert np.all(np.diff(sl.norm(dim=-1).numpy()) >= -1e-12)          # it grows as the thighs come together


def test_hand_room_keeps_the_thumbs_apart(I):
    """The thumb tips carried with the wrists' targets keep THUMB_ROOM_M of lateral separation less half the soft
    blend (the seated leg chorea's thumbs came within 4 mm of each other under the wrists' rule alone); where they lie
    farther apart the slide is the wrists' rule bit for bit."""
    base = I.base_body("sit", 0.5)
    J = base.joints()
    lat = torch.from_numpy(I._lateral_frame(base))
    Wl, Wr = J[:, I.ARM["left"][3]].clone(), J[:, I.ARM["right"][3]].clone()
    close = torch.linspace(0.0, 0.03, base.T, dtype=torch.float64)[:, None] * lat
    Wl2, Wr2 = Wl - 0.5 * close, Wr + 0.5 * close
    tip = 0.5 * torch.linspace(0.04, -0.01, base.T, dtype=torch.float64)[:, None] * lat   # 40 mm apart to crossing
    Tl, Tr = tip, -tip                                                   # only their separation counts
    old = I.hand_room(base, base, Wl2, Wr2)
    sl = I.hand_room(base, base, Wl2, Wr2, thumbs=(Tl, Tr))
    sep = ((Tl + sl - (Tr - sl)) * lat).sum(-1).numpy()
    assert sep.min() >= I.THUMB_ROOM_M - 0.5 * I.HAND_ROOM_SOFT_M - 1e-9
    far = ((Tl + old - (Tr - old)) * lat).sum(-1).numpy() >= I.THUMB_ROOM_M
    assert far.any() and torch.equal(sl[torch.from_numpy(far)], old[torch.from_numpy(far)])
    assert float((sl - old).norm(dim=-1).max()) > 0.0                    # the thumbs move the hands apart


def test_a_hemiballism_kick_is_bounded_in_speed(I):
    """A kick's lift rises at most BALLISM_KICK_SPEED shares per second (amp over rise: its minimum-jerk peak at most
    SPEED_PEAK times that), the hand resting on the kicking thigh carried with it (the grade 1 kick of amp 0.992
    over 0.229 s turned the resting upper arm at 264.9 deg/s on the page, against 250)."""
    t = np.arange(0.0, 20.0, 0.01)
    sig, n = I.kick_signals(t, [(0.6, 19.0)], np.random.default_rng(3), "left")
    assert n >= 5
    assert np.gradient(sig["lift"], 0.01).max() <= I.SPEED_PEAK * I.BALLISM_KICK_SPEED * 1.001


# ---------------------------------------------------------------- the seated knees kept apart (handoff 13)
def _full_gap(body):
    """legcontact over every knot, exact to 0.2 m (no knot culled at the clearance)."""
    from mdfm.synth.legcontact import gap_from_pose
    return gap_from_pose(body.local_R().numpy(), body.transl.numpy(), body.betas.numpy(), int(body.gender),
                         cull_m=0.2)


def _swivelled(I, base, inward_deg):
    """base with both knees swivelled inward by inward_deg {side: (T,)} (seated_legs)."""
    b = base
    for s, m in inward_deg.items():
        R, _ = I.seated_legs(b, s, swivel_deg=-I.SWIVEL_OUT[s] * np.asarray(m))
        b = b.with_rotations(R, list(I.LEG[s][:3]))
    return b


def test_the_closing_clamp_is_smooth_and_leaves_the_outward_swivel(I):
    t = np.linspace(0.0, 4.0, 401)
    il = 18.0 * np.sin(2 * np.pi * 0.5 * t)                             # one knee in and out
    ir = 12.0 * np.sin(2 * np.pi * 0.35 * t + 1.0)
    ceiling = 6.0
    kl, kr, cut = I.clamp_closing(il, ir, ceiling)
    S, S2 = il + ir, kl + kr
    assert (S2 < ceiling).all() and (cut >= 0).all() and np.allclose(S - S2, cut)
    free = S <= ceiling - I.CLOSING_BLEND_DEG
    assert np.array_equal(kl[free], il[free]) and np.array_equal(kr[free], ir[free])   # exactly as asked there
    assert np.array_equal(S2[free], S[free]) and (S2 <= S + 1e-12).all()
    for x, y in ((kl, il), (kr, ir)):                                    # an outward knee keeps (nearly) its swivel
        out = y < -1.0
        assert np.abs(x[out] - y[out]).max() < 0.2 * cut.max()

    def third(n):                                                        # the largest third difference on n samples
        tt = np.linspace(0.0, 4.0, n)
        out = I.clamp_closing(18.0 * np.sin(2 * np.pi * 0.5 * tt), 12.0 * np.sin(2 * np.pi * 0.35 * tt + 1.0), ceiling)
        return max(np.abs(np.diff(out[0], 3)).max(), np.abs(np.diff(out[1], 3)).max())
    # continuous to the second derivative: the third difference falls as the cube of the step (about 8 times for
    # half the step); a jump of the second derivative would fall as its square (about 4 times)
    assert third(4001) / third(8001) > 6.0


def test_the_closing_ceiling_keeps_the_clearance_however_it_is_split(I):
    base = I.base_body("sit", 0.1)
    c = I.closing_ceiling(base, 0)
    assert 0.0 < c < 40.0
    T = base.T
    keep = I.KNEE_CLEARANCE_M + I.KNEE_GUARD_MARGIN_M
    for a, o in I.CLOSING_SPLITS + ((0.25, 0.0), (0.75, 5.0), (0.5, 15.0)):   # the probe's splits, and others
        il = np.full(T, a * c + o)
        h = _full_gap(_swivelled(I, base, {"left": il, "right": c - il}))["hard_m"].min()
        assert h >= (keep - 1e-6 if (a, o) in I.CLOSING_SPLITS else I.KNEE_CLEARANCE_M), (a, o, h)
    # one degree more and the ceiling binds at its binding split (open-body port, wave 3, D26: on SMPL+H the even
    # split bound; on the open body the one-sided splits bind, the even split's hard gap at c + 1 is 16.1 mm and the
    # (1, 20) split's 8.8 mm, scripts/openbody_wave3_checks.py)
    h1 = min(_full_gap(_swivelled(I, base, {"left": np.full(T, a * (c + 1.0) + o),
                                            "right": np.full(T, (c + 1.0) - (a * (c + 1.0) + o))}))["hard_m"].min()
             for a, o in I.CLOSING_SPLITS)
    assert h1 < keep


def test_the_knee_guard_keeps_the_seated_knees_apart(I):
    """Both knees swivelled inward by a minimum-jerk bump that brings them into each other: after the guard every
    knot's hard gap (every knot measured) is at least the clearance, the hips, ankles and feet are where they were,
    the knots outside the guard's swivel keep their rows bit for bit; a body whose knees stay apart comes back as it
    is."""
    base = I.base_body("sit", 3.0)
    t = base.times().numpy()
    bump = I.minimum_jerk((t - 0.8) / 0.4) - I.minimum_jerk((t - 1.6) / 0.4)
    # open-body port, wave 3 (D26): the open body's seated knees stand farther apart; at the former bump's 12 deg of
    # closing (7 and 5) the hard gap stays 27.8 mm, at 20 deg (the same 7 to 5 split) it is -6.2 mm
    # (scripts/openbody_wave3_checks.py, knees)
    deep = _swivelled(I, base, {"left": 7.0 / 12.0 * 20.0 * bump, "right": 5.0 / 12.0 * 20.0 * bump})
    assert _full_gap(deep)["hard_m"].min() < 0.0                          # the knees pass into each other
    out, info = I.knee_guard(deep, base)
    g = _full_gap(out)
    assert info["ok"] and info["rounds"] >= 1 and info["limits_ok"]
    # the inner thighs (open-body port, wave 3 follow-up, D38): on SMPL+H they stood apart wherever the knees kept
    # their clearance (soft gap > 0); on the open body they meet at mid-thigh, 16 to 27 cm above the knee, at 10.8 deg
    # of closing, before the knees' ceiling (13.75 deg), and press 6.5 to 18.2 mm at the ceiling's splits
    # (scripts/openbody_legcontact_calibration.py, seated_knees): soft tissue the check allows to SOFT_TOL_M (30 mm);
    # the guard keeps the knees, the hard class
    from mdfm.synth.legcontact import SOFT_TOL_M
    assert g["hard_m"].min() >= I.KNEE_CLEARANCE_M and g["soft_m"].min() > -SOFT_TOL_M
    J, Jd = out.joints().numpy(), deep.joints().numpy()
    assert np.abs(J[:, [0, 1, 2, 7, 8, 10, 11]] - Jd[:, [0, 1, 2, 7, 8, 10, 11]]).max() < 1e-6
    assert (np.linalg.norm(J[:, 4] - J[:, 5], axis=-1) >= np.linalg.norm(Jd[:, 4] - Jd[:, 5], axis=-1) - 1e-9).all()
    moved = (out.rot6d - deep.rot6d).abs().amax((1, 2)) > 0
    assert moved.any() and not moved[:40].any() and not moved[-60:].any()   # the swivel is local
    assert torch.equal(out.rot6d[~moved], deep.rot6d[~moved])
    v = I.knee_clearance(out, base)
    assert v["ok"] and v["hard_min_m"] >= I.KNEE_CLEARANCE_M
    near = _swivelled(I, base, {"left": 1.0 * bump, "right": 1.0 * bump})   # the knees stay apart: no change
    same, info = I.knee_guard(near, base)
    assert info["rounds"] == 0 and torch.equal(same.rot6d, near.rot6d)


def test_tardive_limbs_keep_the_knees_apart_and_their_outward_swivel(I):
    """The tardive limb movements at the severe step (the review: the knees came close, then passed through each
    other): every knot's hard leg-to-leg gap at least the clearance, the inner thighs within SOFT_TOL_M, the closing
    under the posture's ceiling, each knee's outward swivel as asked, the arms at the step's level and the grade
    unchanged."""
    out, tr = I.td_limbs(4, seconds=SHORT)
    _clean(tr)
    p = tr["params"]
    g = _full_gap(out)
    # the inner thighs press within SOFT_TOL_M on the open body (D38, the knee guard test's comment): they meet before
    # the knees reach their clearance
    from mdfm.synth.legcontact import SOFT_TOL_M
    assert g["hard_m"].min() >= I.KNEE_CLEARANCE_M and g["soft_m"].min() > -SOFT_TOL_M
    kc = p["knee_closing"]
    assert kc["closing_deg_max"] <= kc["ceiling_deg"] + 1e-6 < kc["closing_deg_max_asked"]   # weighted to the ceiling
    assert kc["clamp_ceiling_deg"] == pytest.approx(kc["ceiling_deg"] + kc["blend_deg"])
    assert kc["closing_deg_max"] > kc["ceiling_deg"] - kc["blend_deg"]  # not further than the clearance needs
    assert sum(kc["fragments_weighted"].values()) > 0 and 0.0 <= kc["weight_min"] < 1.0
    assert kc["cut_share"] == 0.0                                         # the weights did it, the clamp left unused
    for leg in p["construction"]["legs"]:
        assert leg["outward_deg_max"] == pytest.approx(leg["outward_deg_max_asked"], rel=1e-3)
        assert leg["outward_deg_max_asked"] > 0.0
        assert leg["inward_deg_max"] < leg["inward_deg_max_asked"]
    assert tr["validity"]["knee_clearance"]["ok"] and tr["grade"] == 4
    own = p["construction"]["arms_own"]["arm_left"]["amplitude_share"]
    assert 0.8 * I.STEPS[4][0] <= own <= 1.02 * I.STEPS[4][0]
    J = out.joints().numpy()[:, [4, 5]]                                   # weighted, not stopped: no knee jolt
    assert np.linalg.norm(np.diff(J, 2, axis=0), axis=-1).max() * out.rate ** 2 < 15.0
    low, trl = I.td_limbs(1, seconds=SHORT)                               # the minimal step keeps its swivels
    kl = trl["params"]["knee_closing"]
    assert sum(kl["fragments_weighted"].values()) == 0 and kl["cut_share"] == 0.0
    assert all(x["knee_swivel_deg_max"] == x["knee_swivel_deg_max_asked"]
               for x in trl["params"]["construction"]["legs"])


def test_a_tardive_closing_under_the_ceiling_keeps_its_swivels(I):
    """The catalogue's mild tardive limb clip (invol_td_E1_limbs_s2, at its own length): its asked closing stays under
    the ceiling, so no fragment is weighted, the clamp and the guard leave it, and each knee swivels as asked (the
    review of the lane: weighted to the ceiling less the blend, it lost a share of its right knee's travel)."""
    out, tr = I.td_limbs(2)
    kc = tr["params"]["knee_closing"]
    assert kc["closing_deg_max_asked"] < kc["ceiling_deg"]
    assert sum(kc["fragments_weighted"].values()) == 0 and kc["cut_share"] == 0.0
    assert kc["closing_deg_max"] == kc["closing_deg_max_asked"]
    assert tr["params"]["construction"]["knee_guard"]["rounds"] == 0
    for x in tr["params"]["construction"]["legs"]:
        assert x["knee_swivel_deg_max"] == x["knee_swivel_deg_max_asked"]
        assert x["inward_deg_max"] == x["inward_deg_max_asked"] and x["outward_deg_max"] == x["outward_deg_max_asked"]
    assert tr["validity"]["knee_clearance"]["ok"]


def test_the_knee_guard_reports_a_contact_it_cannot_open(I):
    """Both feet carried 0.14 m inward (the review of the lane: the shanks meet with the knees apart, where an
    outward knee swivel does not open the contact): the guard stays within its total swivel, adds no joint-limit
    violation, keeps the hips, ankles and feet, and reports the contact (ok False, the unresolved knots those whose
    hard gap stays under the clearance) instead of forcing it (before the fix: 760 and 576 deg of swivel)."""
    from mdfm.synth.body import limit_violations
    base = I.base_body("sit", 2.0)
    t = base.times().numpy()
    bump = I.minimum_jerk((t - 0.4) / 0.4) - I.minimum_jerk((t - 1.4) / 0.4)
    b = base
    for s, sg in (("left", -1.0), ("right", 1.0)):
        R, _ = I.seated_legs(b, s, lat=sg * 0.14 * bump)
        b = b.with_rotations(R, list(I.LEG[s][:3]))
    h0 = _full_gap(b)["hard_m"]
    assert h0.min() < 0.0
    out, info = I.knee_guard(b, base)
    assert not info["ok"] and info["unresolved_knots"] and info["dropped_knots"] > 0
    assert info["swivel_out_deg_max_total"] <= I.KNEE_GUARD_MAX_DEG + 1e-9
    assert info["swivel_out_deg_max"]["left"] + info["swivel_out_deg_max"]["right"] <= 2 * I.KNEE_GUARD_MAX_DEG
    assert limit_violations(out.rot6d, b.rot6d)["new"] == 0              # nothing added to the table's violations
    J, Jb = out.joints().numpy(), b.joints().numpy()
    assert np.abs(J[:, [0, 1, 2, 7, 8, 10, 11]] - Jb[:, [0, 1, 2, 7, 8, 10, 11]]).max() < 1e-6
    assert np.linalg.norm(J[:, [4, 5]] - Jb[:, [4, 5]], axis=-1).max() < 0.05     # the knees barely moved
    h = _full_gap(out)["hard_m"]
    assert set(info["unresolved_knots"]) == set(np.flatnonzero(h < I.KNEE_CLEARANCE_M).tolist())
    assert I.knee_clearance(out, base)["ok"] is False


def test_a_rebuilt_truth_keeps_the_knee_clearance(I):
    """build_guarded makes the truth again (finish) after its arm passes: a seated clip that went through the knee
    guard carries the validity entry knee_clearance there too (the review: the levodopa dystonic clips had the guard's
    record and no entry)."""
    out, tr = I.lid_dystonic(2, seconds=SHORT)
    assert "knee_guard" in tr["params"]["construction"]
    v = tr["validity"]["knee_clearance"]
    assert v["ok"] and v == I.knee_clearance(out, I.base_body("sit", SHORT))


def test_seated_chorea_of_the_legs_keeps_the_knees_apart(I):
    out, tr = I.chorea("sit", 4, regions=("legs",), seconds=SHORT)
    _clean(tr)
    g = _full_gap(out)
    assert g["hard_m"].min() >= I.KNEE_CLEARANCE_M
    v, kg = tr["validity"]["knee_clearance"], tr["params"]["construction"]["knee_guard"]
    assert v["ok"] and kg["ok"]
    m = tr["params"]["measured"]["regions"]
    for leg in tr["params"]["construction"]["legs"]:                    # the amplitude read after the guard
        assert m[f"leg_{leg['side']}"]["amplitude_share"] == pytest.approx(leg["realized"], abs=1e-6)


# ---------------------------------------------------------------- the specifications
def test_specs(I):
    from mdfm.synth.exam import SEGMENTS, segment_title
    sp = I.specs()
    ids = [s.clip_id for s in sp]
    assert len(ids) == len(set(ids)) and len(sp) >= 90
    known = set(ids)
    for s in sp:
        assert s.since == "catalogue_v1"
        assert s.level0 is None or s.level0 in known
        assert s.segment in SEGMENTS or s.segment_title
        assert segment_title(s)
        assert s.params_text and s.grade_text and s.views
        assert "\u2014" not in s.params_text + s.grade_text + s.note


def test_level0_shares_time_base(I):
    sp = {s.clip_id: s for s in I.specs()}
    l0s = {k: v for k, v in sp.items() if v.level0 is None}
    durations = {}
    for cid, s in l0s.items():
        b, _ = s.build()
        durations[cid] = (b.T, b.rate)
    pick = ["invol_rls_E1_g1", "invol_stereotypy_flapping_g1", "invol_akathisia_E7_g1", "invol_mirror_E4a_g1",
            "invol_moving_toes_E1_left_a2"]
    for cid in pick:
        b, tr = sp[cid].build()
        assert (b.T, b.rate) == durations[sp[cid].level0], cid
        assert tr["validity"]["limits_new"] == 0
