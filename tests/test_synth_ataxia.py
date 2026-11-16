"""Tests of the cerebellar ataxia items of the symptom catalogue, mdfm/synth/ataxia.py (task CAT-PDATAX, revised by
task FIX-ataxia after the clinical review of 2026-11-02): every clip's realized values read back by the module's
measures match its grade (stance sway RMS per axis, the 3 Hz component carried in antiphase, the catching step
after the balance loss; the Romberg ratios and the bedside sign; the sitting trunk sway from the pelvis, the back
free; the titubation frequency and amplitude in its band at the visible head (the head centre and the nose, review
M13 of 2026-11-03) over a living background; the alternation's cycles,
time, rhythm and amplitude variability, interruptions and overshoots), the stance feet placed as stated on the
skinned mesh, the knees nearly straight, the legs never inside each other during a step, joints inside the SP1
table, no foot skate where the feet are planted, no NaN, the level-0 partner on the same time base, determinism.
Procedural bodies in the neutral shape only; no lockbox, EVAL, CODY or care_pd record is read."""
import math
import os

import numpy as np
import pytest
import torch

from mdfm import paths

HAVE_MA = os.path.exists(paths.smplh_model("neutral"))
pytestmark = pytest.mark.skipif(not HAVE_MA, reason="MoveAtlas SMPL+H model not found")
torch.set_num_threads(2)

CLIP_IDS_V1 = (["atax_SARA2_natural_g0", "atax_SARA2_natural_g3", "atax_SARA2_together_g0", "atax_SARA2_together_g2",
                "atax_SARA2_together_g3", "atax_SARA2_tandem_g0", "atax_SARA2_tandem_g1", "atax_ROMB_level0",
                "atax_ROMB_cerebellar", "atax_ROMB_sensory"] + [f"atax_SARA3_sitting_g{g}" for g in range(3)] +
               [f"atax_TITUB_g{g}" for g in range(5)] + [f"atax_SARA7_ddk_g{g}" for g in range(5)])
ROMBERG_SEEDS = {"healthy": 99, "cerebellar": 1, "sensory": 2, "sensory_bedside": 3}


@pytest.fixture(scope="module")
def A():
    from mdfm.synth import ataxia
    return ataxia


@pytest.fixture(scope="module")
def stances(A):
    return {k: A.stance_grade(*k, seed=A.SEED + 10 * k[1] + len(k[0])) for k in A.STANCE_LADDER}


def _finite(b):
    return bool(torch.isfinite(b.rot6d).all() and torch.isfinite(b.transl).all())


def _legs_inside(verts, tol=0.002, near=0.03):
    """Vertices of one shank and foot (dominant joints 4, 7, 10 or 5, 8, 11) lying inside the other, by the normal of
    the nearest vertex of the other leg (both directions)."""
    from scipy.spatial import cKDTree
    from mdfm.synth import raster as RS
    from mdfm.synth.viewer import dominant_joint, faces as body_faces
    faces = body_faces()                         # the selected body model's triangles (open-body port, wave 6)
    dom = dominant_joint("neutral")
    legs = [np.flatnonzero(np.isin(dom, (4, 7, 10))), np.flatnonzero(np.isin(dom, (5, 8, 11)))]
    fn = RS.face_normals(verts, faces)
    vn = np.zeros_like(verts)
    for i in range(3):
        np.add.at(vn, faces[:, i], fn)
    vn /= np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12
    n = 0
    for a, b in ((0, 1), (1, 0)):
        d, j = cKDTree(verts[legs[b]]).query(verts[legs[a]])
        s = ((verts[legs[a]] - verts[legs[b]][j]) * vn[legs[b]][j]).sum(1)
        n += int(((s < -tol) & (d < near)).sum())
    return n


def _knee_at_reach(reach):
    """deg: the knee flexion of a straight-rest leg whose hip lies reach of the leg's length from its ankle."""
    from mdfm.synth.body import rest_joints
    r = rest_joints().numpy()
    L1, L2 = float(np.linalg.norm(r[4] - r[1])), float(np.linalg.norm(r[7] - r[4]))
    d = reach * (L1 + L2)
    return 180.0 - math.degrees(math.acos((L1 ** 2 + L2 ** 2 - d ** 2) / (2.0 * L1 * L2)))


def test_stance_sway_is_read_back_at_its_level(A, stances):
    assert len({b.T for b, _ in stances.values()}) == 1
    kr = _knee_at_reach(A.REACH_ST)
    for (kind, g), (b, tr) in stances.items():
        p, v = tr["params"], tr["validity"]
        w = tuple(p["target_window_s"]) if p["fails"] else None
        m = A.measure_sway(b, w)
        assert m["ap_rms_cm"] == pytest.approx(p["target_rms_cm"]["ap"], rel=0.03), (kind, g)
        assert m["ml_rms_cm"] == pytest.approx(p["target_rms_cm"]["ml"], rel=0.03), (kind, g)
        assert _finite(b) and v["limits_new"] == 0 and v["continuity"]["ok"]
        assert v["skate_p95_m_s"] < 1e-6                                     # planted feet do not slide
        # nearly straight knees (D26, open-body port wave 6): the SMPL+H bound (4, 9) lay 1 deg under and 4 deg over
        # the farther knee its bent rest legs read at REACH_ST (5 deg, the module docstring); the open body's rest legs
        # are straight, so the farther knee is the law of cosines' at REACH_ST on the body's own thigh and shank
        # (8.9 deg), and the same bracket is read about it
        assert kr - 1.0 < p["stance"]["knee_deg"]["median"] < kr + 4.0, (kind, g)
        assert p["stance"]["xcom_margin_min_cm"] > 0.0                        # a held stance keeps the XcoM in
        travel = m["ankle_travel_m"]
        if p["fails"]:
            f = p["stance"]["failure"]
            assert travel[0] < 1e-9 and travel[1] > 0.05                      # the right foot steps out
            assert p["fail_at_s"] < 10.0
            assert f["xcom_exit_side"] == f["side"] == "right"
            assert -A.STEP_REACT_S - 1e-9 <= f["lift_delay_s"] <= A.STEP_REACT_MAX_S + 1e-9
            assert f["xcom_outside_before_search_s"] is None
            assert p["stance"]["arms_abduction_deg_max"] == pytest.approx(A.ARM_ABD_DEG, abs=0.5)
        else:
            assert max(travel) < 1e-9
            assert p["stance"]["arms_abduction_deg_max"] == 0.0
        if p["tremor3"]:
            assert m["ap_peak_2_5_hz"] == pytest.approx(A.TREMOR3_HZ, abs=0.3)
            assert m["ap_quotient_2_5_over_0_2"] > 0.01
        else:
            assert m["ap_quotient_2_5_over_0_2"] < 0.005
    lv = {k: math.hypot(t["params"]["target_rms_cm"]["ap"], t["params"]["target_rms_cm"]["ml"])
          for k, (_, t) in stances.items()}
    assert lv[("natural", 3)] > 2.5 * lv[("natural", 0)] and lv[("together", 2)] > 2.5 * lv[("together", 0)]


def test_the_3hz_component_moves_the_hips_more_than_the_head(A, stances):
    """Review F8: the 3 Hz part is not a rigid tilt (head over pelvis at the height ratio) but the legs about the
    ankles with the trunk in antiphase: the head's 3 Hz band amplitude is a fraction of the pelvis's. Review of
    2026-11-03: the hanging arms keep their orientation, so the wrists move with the shoulders, not with the pelvis;
    the pelvis's 3 Hz stays visible (over 4 mm peak to peak) at a moderate acceleration (its band sinusoid's peak
    under 0.25 g, against 0.6 g before). The cerebellar Romberg clip, whose 3 Hz component is the largest, is read
    the same way (review of 2026-11-04)."""
    from mdfm.synth.measure import band_amplitude_pp
    n = 0
    clips = dict(stances)
    clips["romberg_cerebellar"] = A.romberg("cerebellar", seed=A.SEED + ROMBERG_SEEDS["cerebellar"])
    for key, (b, tr) in clips.items():
        if not tr["params"]["tremor3"]:
            continue
        n += 1
        J = b.joints().numpy()
        pp = {j: band_amplitude_pp(J[:, j, 2] - J[0, j, 2], b.rate, A.TREMOR3_HZ, 0.5) for j in (0, 15, 16, 17, 20, 21)}
        assert pp[15] < 0.4 * pp[0], key
        assert max(pp[20], pp[21]) < 0.5 * pp[0], key                          # the wrists do not follow the pelvis
        for w, s in ((20, 16), (21, 17)):                                  # the wrist moves with its shoulder:
            rel = J[:, w, 2] - J[:, s, 2] - (J[0, w, 2] - J[0, s, 2])        # the band of the difference, so that an
            assert band_amplitude_pp(rel, b.rate, A.TREMOR3_HZ, 0.5) < 0.25 * pp[s], (key, w)   # antiphase fails
        assert pp[0] > 0.004 and (2 * math.pi * A.TREMOR3_HZ) ** 2 * pp[0] / 2 / 9.81 < 0.25, key
        assert pp[0] < 0.012, key                                         # the caption's 'small'
        assert tr["params"]["stance"]["tremor3_pattern"]["com_shear_bw_rms"] > 0.0
    assert n == 4


def test_a_catching_step_keeps_the_legs_apart(A, stances):
    """Review F1: during the swing no shank or foot vertex lies inside the other leg."""
    from mdfm.synth.viewer import knot_vertices
    for key, (b, tr) in stances.items():
        if not tr["params"]["fails"]:
            continue
        t0 = tr["params"]["stance"]["failure"]["lift_s"]
        ks = list(range(int(round(t0 * b.rate)), int(round((t0 + A.STEP_S) * b.rate)) + 1, 2))
        V = knot_vertices(b, ks)
        assert max(_legs_inside(v) for v in V) == 0, key


def test_stance_feet_are_placed_as_stated_on_the_mesh(A):
    from mdfm.synth import hands as H
    from mdfm.synth.contact import sole_vertices
    sv = sole_vertices(2)
    z = np.zeros(301)
    for kind in ("natural", "together", "tandem"):
        b, _ = A.stance_body(kind, z, z)
        L = H.body_points(b, sv["left"], knots=[0])[0]
        R = H.body_points(b, sv["right"], knots=[0])[0]
        assert abs(L[:, 1].min() - R[:, 1].min()) < 2e-3                     # both soles on one floor
        if kind == "together":
            assert L[:, 0].min() - R[:, 0].max() == pytest.approx(0.004, abs=0.004)
        if kind == "tandem":
            assert L[:, 2].min() - R[:, 2].max() == pytest.approx(0.005, abs=0.005)
            assert abs(L[:, 0].mean() - R[:, 0].mean()) < 0.02                # on one line
        from mdfm.synth.body import limit_violations
        assert limit_violations(b.rot6d)["outside"] == 0


@pytest.mark.parametrize("kind", ["healthy", "cerebellar", "sensory", "sensory_bedside"])
def test_romberg_ratio_and_tremor_are_read_back(A, kind):
    b, tr = A.romberg(kind, seed=A.SEED + ROMBERG_SEEDS[kind])
    p = tr["params"]
    m = p["remeasured"]
    assert m["ratio_rms"] == pytest.approx(A.ROMBERG[kind]["ratio"], rel=0.1)
    assert tr["validity"]["limits_new"] == 0 and _finite(b) and tr["validity"]["skate_p95_m_s"] < 1e-6
    assert [x["text"] for x in p["frame_text"]] == ["EYES OPEN", "EYES CLOSED"]
    assert p["frame_text"][1]["from_s"] == A.ROMBERG_CLOSE_S
    if A.ROMBERG[kind]["tremor3"]:
        assert m["open"]["ap_peak_2_5_hz"] == pytest.approx(3.0, abs=0.3)
    else:
        assert m["open"]["ap_quotient_2_5_over_0_2"] < 0.005
    travel = A.measure_sway(b)["ankle_travel_m"]
    if p["bedside_sign"]:                                         # the sign: a catching step after the closure
        f = p["stance"]["failure"]
        assert A.ROMBERG_CLOSE_S + A.ROMBERG_GROW_S < p["fail_at_s"] < A.ROMBERG_S - 2.0
        assert travel[1] > 0.05 and travel[0] < 1e-9 and f["xcom_exit_side"] == "right"
    else:
        assert max(travel) < 1e-9
        assert A.xcom_margin(b, "together")[1].max() < 0.0                # held: the XcoM stays in the base
    b2, _ = A.romberg(kind, seed=A.SEED + ROMBERG_SEEDS[kind])
    assert torch.equal(b.rot6d, b2.rot6d) and torch.equal(b.transl, b2.transl)


def test_the_bedside_sign_outgrows_the_posturographic_pair(A):
    """Review F2: with the eyes closed the bedside sensory clip sways more than the cerebellar one, with the eyes
    open less."""
    out = {k: A.romberg(k, seed=A.SEED + ROMBERG_SEEDS[k])[1]["params"]["remeasured"]
           for k in ("cerebellar", "sensory_bedside")}
    r = {k: {w: math.hypot(v[w]["ap_rms_cm"], v[w]["ml_rms_cm"]) for w in ("open", "closed")} for k, v in out.items()}
    assert r["sensory_bedside"]["open"] < r["cerebellar"]["open"]
    assert r["sensory_bedside"]["closed"] > r["cerebellar"]["closed"]


def test_sitting_sway_follows_the_grade_from_the_seat(A):
    from mdfm.synth.base import static_body
    out = {g: A.sitting_grade(g, seed=A.SEED + 30 + g) for g in A.SIT_LADDER}
    base = static_body("sit_arms_forward", A.SIT_S)
    assert len({b.T for b, _ in out.values()}) == 1
    for g, (b, tr) in out.items():
        p = tr["params"]
        m = p["remeasured"]
        want = A.SIT_LADDER[g]["line_deg"]
        assert math.hypot(m["ap_rms_in_deg"], m["ml_rms_in_deg"]) / math.sqrt(2) == pytest.approx(want, rel=0.05)
        assert tr["validity"]["limits_new"] == 0 and _finite(b)
        assert tr["validity"]["skate_p95_m_s"] < 1e-9 and tr["validity"]["contact_lost_share"] == 0.0
        J, Jb = b.joints().numpy(), base.joints().numpy()
        shift = np.array([0.0, 0.0, p["seat"]["seat_forward_m"]])
        assert np.abs(J[:, [7, 8, 10, 11]] - Jb[:, [7, 8, 10, 11]] - shift).max() < 1e-9   # the feet stay put
        assert not torch.allclose(b.rot6d[:, 0], base.rot6d[:, 0])        # the pelvis rocks on the seat
        assert p["seat"]["clearance_m"] >= A.SIT_BACK_CLEAR_M - 1e-9      # the back is free
    m = {g: tr["params"]["remeasured"] for g, (_, tr) in out.items()}
    assert m[0]["share_above"] == 0.0 and 0.05 < m[1]["share_above"] < 0.5 and m[2]["share_above"] > 0.5
    bouts = out[1][1]["params"]["bouts_s"]
    assert len(bouts) == 2 and all(e - s == pytest.approx(A.SIT_BOUT_S) for s, e in bouts)


def _page_head(body, fps=30.0):
    """{name: (N, 3) m} of the head centre and the nose relative to the pelvis on the skeleton page's points (the
    catalogue export's clock, skeleton.skeleton at 30 fps, rounded to the page's 1 mm points)."""
    from mdfm.synth import skeleton as SK
    sk = SK.skeleton(body, fps, hands=False)
    ix = {n: i for i, n in enumerate(sk.names)}
    P = SK.quantize_mm(sk.positions, 1) / 1000.0
    return {n: P[:, ix[n]] - P[:, ix["pelvis"]] for n in ("head_centre", "nose")}


@pytest.mark.parametrize("grade", [1, 2, 3, 4])
def test_titubation_frequency_and_head_amplitude_in_band(A, grade):
    """Review M13: levelled at the visible head, where a rater reads it: the nose's largest cycle at its target, under
    the band's upper edge, and the head centre's median cycle above its lower edge, at the body's rate and on the
    skeleton page's 30 fps points, by cycle and by 1 s chord (the synthesis's reading); the skull-base joint reads
    lower than the head centre, the head centre lower than the nose."""
    b, tr = A.titubation(grade, seed=A.SEED + 40 + grade)
    l0, _ = A.titubation(0, seed=A.SEED + 40)
    m = A.measure_titubation(l0, b)
    assert m["hz"] == pytest.approx(A.TITUBATION_HZ, abs=0.05)
    assert m["max_cm"] == pytest.approx(A.titubation_max_cm(grade), rel=0.01)
    assert tr["params"]["max_cm"] == A.titubation_max_cm(grade)
    lo, hi = A.titubation_band_cm(grade)
    assert tr["params"]["band_cm"] == [lo, hi] and tr["params"]["ftm_band_cm"] == list(A.FTM_BANDS_CM[grade - 1])
    assert lo == A.FTM_BANDS_CM[grade - 1][0] and hi == min(A.FTM_BANDS_CM[grade - 1][1], 2 * lo if lo else 1.0)
    pts = m["points"]
    for key in ("cycle", "window"):
        assert pts["nose"][f"{key}_max_cm"] < hi, key
        assert pts["head_centre"][f"{key}_median_cm"] > 1.1 * lo, key
    assert pts["head_joint"]["cycle_max_cm"] < pts["head_centre"]["cycle_max_cm"] < pts["nose"]["cycle_max_cm"]
    assert m["shoulders_pp_cm"] < 0.5 * pts["head_centre"]["band_pp_cm"]       # the head carries the sign
    r = A.titubation_readings(_page_head(l0), _page_head(b), 30.0)             # the page's points
    for key in ("cycle", "window"):
        assert r["nose"][f"{key}_max_cm"] < hi and r["head_centre"][f"{key}_median_cm"] > lo, key
    assert tr["validity"]["limits_new"] == 0 and _finite(b)
    m0 = A.measure_titubation(l0, l0)
    assert l0.T == b.T and m0["max_cm"] == 0.0 and m0["typical_cm"] == 0.0
    th, hd = tr["arrays"]["theta_trunk_rad"], tr["arrays"]["theta_head_rad"]
    lag = np.angle(np.vdot(np.fft.rfft(th), np.fft.rfft(hd)))           # review F13: the head lags the trunk
    assert math.degrees(abs(lag)) == pytest.approx(A.TITUB_LAG_DEG, abs=5.0)
    k = int(b.rate)                                                     # and the amplitude is not constant
    env = np.abs(th[k:-k]) / np.abs(th[k:-k]).max()
    peaks = [env[i:i + 33].max() for i in range(0, len(env) - 33, 33)]
    assert np.std(peaks) / np.mean(peaks) > 0.05


def test_the_titubation_level_0_is_alive(A):
    """Review F5: the level 0 breathes and sways (not a mannequin), the same background under every grade."""
    from mdfm.synth.base import static_body
    l0, tr = A.titubation(0, seed=A.SEED + 40)
    bg = tr["params"]["background"]
    assert bg["chest_rise_mm"] == pytest.approx(A.BREATH_CHEST_MM, rel=0.05)
    head = l0.joints().numpy()[:, 15] - static_body("sit", A.TITUB_S).joints().numpy()[:, 15]
    assert 0.3 < 100 * np.ptp(head[:, 2]) < 3.0                           # cm, forward-back
    b1, _ = A.titubation(1, seed=A.SEED + 41)
    assert torch.equal(b1.rot6d[:, [1, 2, 4, 5, 7, 8]], l0.rot6d[:, [1, 2, 4, 5, 7, 8]])


def test_alternation_rhythm_amplitude_and_time_follow_the_grade(A):
    from mdfm.synth.bradykinesia import task_angle
    out = {g: A.alternation(g, seed=A.SEED + 50 + g) for g in A.DDK_GRADES}
    assert len({b.T for b, _ in out.values()}) == 1
    cvs = []
    for g, (b, tr) in out.items():
        p = tr["params"]
        m = A.measure_alternation(task_angle(b, "pronation_supination", "left", 0), b.rate)
        assert m["n_cycles"] == (A.DDK_G4_CYCLES if g == 4 else A.DDK_N)
        assert m["time_s"] == pytest.approx(p["total_s"], abs=0.02)
        assert m["cv_rise"] == pytest.approx(p["realized_cv_amp"], abs=0.01)
        if p["realized_cv_peak_interval"] is not None:
            assert m["cv_interval"] == pytest.approx(p["realized_cv_peak_interval"], abs=0.02)
        assert m["interruptions"] == len(p["pauses_after_cycles"])
        assert tr["validity"]["limits_new"] == 0 and _finite(b)
        for c, d in zip(p["pauses_after_cycles"], p["hold_drift_deg"]):   # review F12: holds near the rest, drifting
            assert p["troughs_deg"][c - 1] <= A.DDK_HOLD_TROUGH * p["amplitude0_deg"] + 1e-9
            assert A.DDK_HOLD_DRIFT_DEG[0] <= d <= A.DDK_HOLD_DRIFT_DEG[1]
        if g in (2, 3):                                                    # and overshoots corrected back
            assert 0.2 * A.DDK_N <= len(p["overshoot_cycles"]) <= 0.4 * A.DDK_N
            assert all(A.DDK_OVERSHOOT_DEG[0] <= x <= A.DDK_OVERSHOOT_DEG[1] for x in p["overshoot_deg"])
        else:
            assert p["overshoot_cycles"] == []
        cvs.append(p["realized_cv_period"])
        if g <= 2:
            assert m["time_s"] < 10.0
        elif g == 3:
            assert m["time_s"] > 10.0
    assert cvs[:4] == sorted(cvs[:4])
    b2, _ = A.alternation(2, seed=A.SEED + 52)
    assert torch.equal(b2.rot6d, out[2][0].rot6d)


CLIP_IDS_SH7 = [f"atax_SARA8_heelshin_g{g}" for g in range(5)] + ["atax_SARA8_heelshin_right_g0"]


@pytest.fixture(scope="module")
def heelshin(A):
    return {g: A.heel_shin(g, seed=A.SEED + 80 + g) for g in A.HS_GRADES}


def test_the_supine_body_rests_on_the_couch(A):
    """SH7: the supine rest lies on the couch top with the upper back, the head, the arms and hands, the buttocks and
    the heels; nothing below it; inside the SP1 table; its heading defined (the feet toward +z)."""
    from mdfm.synth.body import limit_violations
    from mdfm.synth.viewer import heading_deg, knot_vertices
    R, info = A.supine_posture()
    gaps = info["lowest_above_couch_m"]
    for part in ("buttocks", "upper back", "head", "arms"):
        assert 0.0 <= gaps[part] < 1e-3, part
    # each leg rests by the heel or the calf, the module's rule (D26, open-body port wave 6: on SMPL+H the heels came
    # down first; on the open body the calves do, the heels 1.4 mm over the couch)
    assert 0.0 <= min(gaps["heels"], gaps["calves"]) < 1e-3
    assert gaps["thighs"] > gaps["calves"] >= 0.0                        # no soft tissue: reported, not hidden
    b = A.supine_body(1.0)
    V = knot_vertices(b, [0])[0]
    assert V[:, 1].min() == pytest.approx(A.couch_top_y(), abs=1e-9)
    assert limit_violations(b.rot6d)["outside"] == 0
    assert heading_deg(b) == pytest.approx(0.0, abs=1e-6)
    J = b.joints().numpy()[0]
    assert J[15, 2] < J[0, 2] < J[7, 2]                                   # head toward -z, feet toward +z
    assert abs(J[15, 1] - J[0, 1]) < 0.1                                  # lying, not standing


def test_the_couch_scene(A):
    """SH7: the 'exam_bed' couch: two boxes of viewer.box, its top at the body's resting level, COUCH_TOP_M above the
    room floor, the body over its top at rest; other kinds refused."""
    from mdfm.synth.viewer import knot_vertices
    b = A.supine_body(1.0)
    v, f = A.scene_mesh("exam_bed", b, 0.0)
    assert v.shape == (16, 3) and f.shape == (24, 3) and f.max() == 15
    (lo, hi), (blo, bhi) = A.couch_boxes(b)
    view = A.bed_view(b)
    assert hi[1] == pytest.approx(A.couch_top_y(), abs=1e-9)
    assert hi[1] - view["floor_y"] == pytest.approx(A.COUCH_TOP_M) and blo[1] == pytest.approx(view["floor_y"])
    assert view["heading_deg"] == 0.0 and hi[0] - lo[0] >= A.COUCH_WIDTH_M - 1e-9
    V = knot_vertices(b, [0])[0]
    assert (V[:, 0] > lo[0]).all() and (V[:, 0] < hi[0]).all() and (V[:, 2] > lo[2]).all() and (V[:, 2] < hi[2]).all()
    with pytest.raises(ValueError):
        A.scene_mesh("chair", b, 0.0)
    from mdfm.synth import exam as EX                                    # the route of the exam set's scenes
    ev, ef = EX.scene_mesh("exam_bed", b, view["floor_y"])
    assert np.array_equal(ev, v) and np.array_equal(ef, f)


def test_heel_shin_grades_are_read_back(A, heelshin):
    """SH7, SARA 8: three slides per clip; grade 0 and 1 keep the heel on the shin (1 with a wobble and a missed
    landing), grade 2 goes off it 1 to 3 times, grade 3 4 or more times, grade 4 never has it on the shin; the planned
    excursions are the ones read back, slide by slide; the landings past and short of the knee as planned."""
    assert len({b.T for b, _ in heelshin.values()}) == 1
    m = {g: tr["params"]["remeasured"] for g, (_, tr) in heelshin.items()}
    for g, (b, tr) in heelshin.items():
        p = tr["params"]
        assert m[g]["cycles"] == A.HS_CYCLES and len(m[g]["landings"]) == A.HS_CYCLES, g
        if g < 4:
            assert [x["off_shin"] for x in m[g]["slides"]] == [len(c["events"]) for c in p["plan"]], g
            assert all(x["on_shin_share"] > 0.4 for x in m[g]["slides"]), g
    assert m[0]["off_shin"] == 0 and m[1]["off_shin"] == 0
    assert m[0]["grade_from_counts"] is None and m[1]["grade_from_counts"] is None
    assert all(x["on_shin_share"] == 1.0 for g in (0, 1) for x in m[g]["slides"])           # contact kept
    assert 1 <= m[2]["off_shin"] <= 3 and m[2]["grade_from_counts"] == 2
    assert m[3]["off_shin"] >= 4 and m[3]["grade_from_counts"] == 3
    assert m[4]["no_contact_in_all_slides"] and m[4]["grade_from_counts"] == 4
    assert m[0]["placement_error_m_max"] < 0.005 and m[0]["lateral_rms_m_mean"] < 0.001
    assert m[1]["lateral_rms_m_mean"] > 2 * m[0]["lateral_rms_m_mean"]
    for g in (1, 2, 3, 4):                                               # the landing's error at the knee
        e = A.HS_GRADES[g]["error_m"]
        errs = [x["error_m"] for x in m[g]["landings"]]
        assert min(errs) > 0.5 * e and max(errs) < 1.5 * e, g
        plan = [c["landing"] for c in heelshin[g][1]["params"]["plan"]]
        assert [x["side_of_knee"] for x in m[g]["landings"]] == plan, g
    assert {"past", "short"} <= set(heelshin[2][1]["params"]["landings"])
    assert set(heelshin[3][1]["params"]["landings"]) == {"short"}


def test_heel_shin_validity_fits_a_lying_body(A, heelshin):
    """SH7: joint limits, continuity, the couch, the heel never below the shin's surface, the legs apart; the
    standing checks are named not applicable; the leg slid on never moves; the pelvis turns back to rest."""
    base = A.supine_body(A.heel_shin_duration())
    Jb = base.joints().numpy()
    for g, (b, tr) in heelshin.items():
        v = tr["validity"]
        assert _finite(b) and v["limits_new"] == 0 and v["continuity"]["ok"], g
        assert v["couch"]["ok"] and v["heel_on_shin"]["ok"] and v["legs_apart"]["ok"] and v["contact"]["ok"], g
        assert v["couch"]["min_mm"] >= -1000 * A.COUCH_YIELD_M
        rs, th = v["resting_slide"], v["thighs"]                           # the upper body keeps its place on the couch
        assert rs["ok"] and rs["upper_body_mm"] <= 1000 * A.HS_REST_SLIDE_M and min(rs["n_vertices"].values()) > 0, g
        assert rs["pelvis_mm"] > rs["upper_body_mm"] > rs["still_leg_mm"], g     # the pelvis turns on purpose
        assert th["ok"] and th["beyond_groin_min_mm"] >= -1000 * A.HS_THIGH_PRESS_M, g
        # the rest's thighs: pressed together on SMPL+H (rest_min_mm < 0), apart on the open body (design D36: its rest
        # thighs stand apart up to the crotch), D26, open-body port wave 6
        assert th["rest_min_mm"] > 0.0, g
        assert {"foot skate in stance", "sole contact with the floor"} <= set(v["not_applicable"])
        J = b.joints().numpy()
        assert np.abs(J[:, [2, 5, 8, 11]] - Jb[:, [2, 5, 8, 11]]).max() < 1e-9, g      # the right leg is still
        assert torch.allclose(b.rot6d[[0, -1]], base.rot6d[[0, -1]], atol=1e-9)       # from rest, back to rest
        # inside the table: the straight knee of the rest lies on jarm2022's knee extension bound, 0 deg (the AMASS
        # table's knee reached past it), so its margin reads 0 to float precision (D26, open-body port wave 6)
        from mdfm.synth.body import LIMIT_TOL_DEG
        assert tr["params"]["limit_margin_deg_min"] >= -LIMIT_TOL_DEG


def test_the_heel_starts_and_stops_without_a_jump(A, heelshin):
    """SH7 review: every arc of the heel's target leaves and lands with zero velocity (the lift, the landing, the
    correction, the return and the laying back), the approach's oscillation ends with zero slope at the landing."""
    for g, (b, tr) in heelshin.items():
        X, r = tr["arrays"]["heel_target_m"], b.rate
        v = np.linalg.norm(np.gradient(X, 1.0 / r, axis=0), axis=1)
        for c in tr["params"]["plan"]:
            for key in ("lift_s", "land_s", "correct_s", "knee_s", "return_s", "rest_s"):
                if key in c:
                    assert v[int(round(c[key] * r))] < 0.02, (g, c["cycle"], key)


def test_no_movement_of_an_ataxic_approach_is_faster_than_the_level_0(A):
    """SH7 review: the voluntary path (the heel's oscillation left out) of every grade's approach and correction, phase
    by phase, peaks no faster than the level 0's single arc."""
    from mdfm.synth import hands as H
    from mdfm.synth.common import as_rng
    base = A.supine_body(A.heel_shin_duration())
    t, r = base.times().numpy(), base.rate
    fr = A.shin_frame(base, "left")
    c_rest = H.body_points(base, A.heel_ids("left"), knots=[0])[0].mean(0)
    peak = {}
    for g in A.HS_GRADES:
        plan = A._hs_plan(g, as_rng(A.SEED + 80 + g))
        X, _, _ = A._hs_path(fr, c_rest, plan, g, np.zeros(len(t)), t, A.TR.HZ["intention"], r, oscillation=False)
        v = np.linalg.norm(np.gradient(X, 1.0 / r, axis=0), axis=1)
        peak[g] = max(v[(t >= c["lift_s"]) & (t <= c["knee_s"])].max() for c in plan)
    assert all(peak[g] <= peak[0] for g in A.HS_GRADES), peak


def test_the_slide_time_is_read_back(A, heelshin):
    """SH7 review: each slide's time, from the heel's last stop on the knee to its first stop at the slide's end, reads
    the planned slide time (the level 0 at SARA's about 1 s; the slow start and end of a smooth slide fall under the
    stop speed, so slightly less); the core between the tolerances is shorter."""
    for g, (b, tr) in heelshin.items():
        for x in tr["params"]["remeasured"]["slides"]:
            d = A.HS_GRADES[g]["slide_s"]
            assert 0.9 * d < x["duration_s"] <= d + 1.0 / b.rate, (g, x["duration_s"])
            assert x["core_duration_s"] < x["duration_s"], g
    assert all(abs(x["duration_s"] - A.HS_SLIDE_S_SARA) < 0.1 for x in heelshin[0][1]["params"]["remeasured"]["slides"])


def test_every_truth_source_resolves(A, heelshin, stances):
    """SH7 review: every source key of the heel-shin and stance truths is a key of the module's SOURCES."""
    for _, tr in list(heelshin.values()) + list(stances.values()):
        assert set(tr["sources"]) <= set(A.SOURCES), set(tr["sources"]) - set(A.SOURCES)
    assert A.SOURCES["buijink2012"]


def test_the_heel_touches_the_shin_where_planned(A, heelshin):
    """SH7: on the knee and on the slide the foot touches the other shin (clearance within 3 mm), off it the heel is
    clear; the heel's distance to the shin line stays under the off-shin threshold in grades 0 and 1."""
    surf = A.leg_surface(A.supine_body(A.heel_shin_duration()), "right")
    for g in (0, 1, 3):
        b, tr = heelshin[g]
        cm = tr["arrays"]["contact_planned"]
        gap = A._foot_clearance(b, "left", surf, np.flatnonzero(cm))
        assert cm.sum() > 100 and np.abs(gap[cm]).max() < 0.003, g
        if g < 2:
            on = tr["arrays"]["distance_m"][cm]
            assert on.max() < A.OFF_SHIN_M, g


def test_the_hypermetric_bound(A):
    """SH7: a landing past the knee by HS_PAST_MAX_M stays inside the table; the grades that land past are bounded by
    it. On the AMASS table the largest grade-3 error past the knee left the table (the reason grades 3 and 4 land
    short); on the anatomical table (jarm2022's hip flexion of 125 deg) it stays inside, and the split of the grades is
    kept (D26, open-body port wave 6)."""
    import math
    r = math.radians(A.HS_ERROR_DIR_DEG)
    s0 = A.HS_KNEE_BELOW_M
    assert A.placement_margin_deg(s0, 0.0) > 2.0
    assert A.placement_margin_deg(s0 - A.HS_PAST_MAX_M * math.cos(r), -A.HS_PAST_MAX_M * math.sin(r)) > 0.0
    e3 = (1 + A.HS_ERROR_JITTER) * A.HS_GRADES[3]["error_m"]
    assert A.placement_margin_deg(s0 - e3 * math.cos(r), -e3 * math.sin(r)) > 0.0
    for g, cfg in A.HS_GRADES.items():
        past = 0 < (1 + A.HS_ERROR_JITTER) * cfg["error_m"] <= A.HS_PAST_MAX_M
        assert past == (g in (1, 2)), g


def test_heel_shin_plan_is_deterministic(A):
    from mdfm.synth.common import as_rng
    for g in A.HS_GRADES:
        assert A._hs_plan(g, as_rng(A.SEED + 80 + g)) == A._hs_plan(g, as_rng(A.SEED + 80 + g))
    assert A.heel_shin_duration() == pytest.approx(
        math.ceil(max(A._hs_plan(g, as_rng(0))[-1]["rest_s"] for g in A.HS_GRADES) + A.HS_TAIL_S))


def test_the_right_side_level_0(A):
    """SH7: the right heel on the left shin, level 0, on the same time base."""
    sp = {x.clip_id: x for x in A.specs()}
    b, tr = sp["atax_SARA8_heelshin_right_g0"].build()
    m, v = tr["params"]["remeasured"], tr["validity"]
    assert tr["params"]["side"] == "right" and m["cycles"] == A.HS_CYCLES and m["off_shin"] == 0
    assert v["limits_new"] == 0 and v["contact"]["ok"] and v["continuity"]["ok"]
    assert b.T == int(round(A.heel_shin_duration() * b.rate)) + 1


def test_specs_are_complete_and_partnered(A):
    from mdfm.synth import exam as EX
    from mdfm.synth.viewer import VIEWS
    s = A.specs()
    ids = {x.clip_id for x in s}
    assert len(ids) == len(s)
    assert set(CLIP_IDS_V1) <= ids and "atax_ROMB_sensory_bedside" in ids     # the ids stay stable
    assert set(CLIP_IDS_SH7) <= ids
    for x in s:
        if x.segment == "SARA8":
            assert x.scene == "exam_bed" and x.segment_title and {"left", "right", "top"} <= set(x.views)
            assert x.level0 in (None, "atax_SARA8_heelshin_g0")
        assert x.since == "catalogue_v1" and (x.level0 is None or x.level0 in ids)
        assert x.segment in EX.SEGMENTS or x.segment_title
        assert x.caption_spec()["segment_text"].startswith(x.segment)
        assert all(v in VIEWS for v in x.views)
    sp = {x.clip_id: x for x in s}
    assert sp["atax_TITUB_g2"].views[1] in ("head_left", "head")
    for g in range(1, 5):                                     # review M13: the caption states the levelled reading
        x = sp[f"atax_TITUB_g{g}"]
        assert x.operator == "head and trunk titubation, seated"
        assert f"up to about {A.titubation_max_cm(g):.1f} cm peak to peak at the nose" in x.params_text
        assert "Fahn-Tolosa-Marin" in x.grade_text
    for cid in ("atax_ROMB_level0", "atax_ROMB_cerebellar", "atax_ROMB_sensory"):   # review: named posturographic
        assert "posturographic" in sp[cid].operator and "posturographic Romberg ratio" in sp[cid].grade_text, cid
    assert "no Romberg sign" in sp["atax_ROMB_cerebellar"].params_text
    assert "no bedside" not in sp["atax_ROMB_cerebellar"].operator              # its sway and tremor are visible
    assert sp["atax_ROMB_cerebellar"].operator.endswith("(posturographic sway, no Romberg sign)")
    assert sp["atax_ROMB_sensory"].operator.endswith("(posturographic sway, no bedside Romberg sign)")
    assert "below the threshold of a bedside Romberg sign" in sp["atax_ROMB_sensory"].params_text
    assert "the sway twice as large" in sp["atax_ROMB_sensory"].params_text
    for cid in ("atax_SARA8_heelshin_g0", "atax_SARA8_heelshin_right_g0"):   # review: the slide's end is stated
        assert "stopping short of the ankle" in sp[cid].params_text and "percent" not in sp[cid].params_text, cid
    for kind in ("natural", "together", "tandem"):                           # review: no doubled 'stance'
        op = sp[f"atax_SARA2_{kind}_g0"].operator
        assert op.endswith(", eyes open") and op.count("stance") <= 1, op
    for (kind, g), lad in A.STANCE_LADDER.items():                          # review: the 3 Hz component is named
        txt = sp[f"atax_SARA2_{kind}_g{g}"].params_text
        assert ("small 3 Hz forward and back postural tremor" in txt) == lad["tremor3"]
    for kind in ("healthy", "cerebellar", "sensory"):
        cid = "atax_ROMB_level0" if kind == "healthy" else f"atax_ROMB_{kind}"
        assert ("3 Hz forward and back postural tremor" in sp[cid].params_text) == A.ROMBERG[kind]["tremor3"], cid
    texts = [t for x in s for t in (x.operator, x.grade_text, x.params_text)]
    assert not [t for t in texts if "procedural" in t or "module value" in t or "_" in t]   # no provenance, no code
    for cid in ("atax_ROMB_sensory", "atax_ROMB_sensory_bedside"):
        a, _ = sp[cid].build()
        b, _ = sp[sp[cid].level0].build()
        assert a.T == b.T and a.rate == b.rate
