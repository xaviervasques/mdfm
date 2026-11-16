"""Tests of the tremor family and the reach-based limb signs of the symptom catalogue (mdfm/synth/tremors.py, task
CAT-TREM) and of the reach base motion (mdfm/synth/reach.py):

- the clip list: every spec since 'catalogue_v1', its segment known or titled, ids unique and apart from the exam
  set's, every level-0 partner a clip of the list or of the exam set;
- every clip: no NaN, no new joint-limit violation, no foot skate added, the hand limits held where hands are posed,
  the level-0 partner on the same clock (same knots, same rate);
- re-measurement: the dominant frequency (spectral peak) and the peak-to-peak chord of the wrist or nose
  displacement against the level-0 partner fall in the grade's band (frequency within 0.05 Hz of the asked one for
  a sine, inside the published band for the jittered waveforms); the dysmetria end point in the SARA 5 band; the
  decomposition's phases counted from the fingertip speed; the orthostatic frequency on the pelvis; the functional
  tremor's own frequency, its entrainment to the taps and its pause; the pill-rolling fingertip chord;
- the reach: the fingertip on its targets at the holds, never inside the face, the cited cycle time, three reaches,
  a near-straight fingertip path between the targets; the intention tremor larger near the target than in
  mid-movement; no joint of a reach clip stepping beyond its neighbourhood (reach.joint_step_excess under
  reach.STEP_EXCESS_MAX), the dysmetric correction joining the level 0's target posture with no step and the
  dysmetric arm turning about as fast as the level 0's joint by joint (wave SH1c: the snap at the end of the
  overshoot), the dysmetric movement taking the hold at the target; the correction of a miss to the side on its
  curve, beside the examiner's finger and never in it, the shoulder girdle moving steadily through it, the
  overshoot's fingertip at most twice as fast as the level 0's on its own movement to the target;
- the clinical review of the tremor family (2026-11-02): no reach clip inside the head or the examiner's finger
  (B1), the outstretched hand trembling up and down (B2), the thumb pad on the index pad (B3), the jerky cycles
  spread (M1), the intention tremor on both sides of the nose (M2), slower top grades (M3), the examiner's finger
  (M4), the orthostatic wide base and weight shifts (M5), the neck and head sharing the head tremor (M6), the
  dystonic arm's posture, pull and null point (M7), and the minor ones (envelope floor, functional pause,
  decomposition pace, the index at the nose, the wing posture, the right reach view);
- the levels: the band placement (geometric centres, the log placement of the open bands) and its inverse;
- the shared wave of the clinical review (2026-11-02) on the exam set (exam.py): the arms-outstretched ladder, the
  re-emergent tremor and the raise clips driven at the wrist and shoulder, the middle fingertip moving up and down
  inside each 3.15 band (SH4); the rest ladder clear of the chair, its grade 4 at the family's top level and
  frequency; the E3 clips on the scene 'chair_target' and the routes of exam.scene_mesh (SH6);
- determinism: the same seed gives the same body, another seed another one.
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


@pytest.fixture(scope="module")
def TR():
    from mdfm.synth import tremors
    return tremors


@pytest.fixture(scope="module")
def RC():
    from mdfm.synth import reach
    return reach


@pytest.fixture(scope="module")
def specs(TR):
    return {s.clip_id: s for s in TR.specs()}


@pytest.fixture(scope="module")
def exam_specs():
    from mdfm.synth import exam
    return {s.clip_id: s for s in exam.exam_set()}


@pytest.fixture(scope="module")
def built(specs, exam_specs):
    """{clip_id: (body, truth)} of every catalogue clip and of the exam level-0 partners they name (procedural)."""
    out = {}
    for cid, s in specs.items():
        out[cid] = s.build()
    for cid in {s.level0 for s in specs.values() if s.level0 and s.level0 not in specs}:
        out[cid] = exam_specs[cid].build()
    return out


def pytest_generate_tests(metafunc):
    if "cid" in metafunc.fixturenames:
        from mdfm.synth import tremors
        metafunc.parametrize("cid", [s.clip_id for s in tremors.specs()])


# ---------------------------------------------------------------- the clip list
def test_the_clip_list_is_catalogue_v1_with_known_segments_and_partners(TR, specs, exam_specs):
    from mdfm.synth.exam import SEGMENTS
    assert len(specs) == len(TR.specs()) >= 50
    assert not set(specs) & set(exam_specs)
    for s in specs.values():
        assert s.since == "catalogue_v1"
        assert s.segment in SEGMENTS or s.segment_title, s.clip_id
        assert s.params_text and s.grade_text and s.views
        if s.level0 is not None:
            assert s.level0 in specs or s.level0 in exam_specs, s.clip_id
    families = {s.family for s in specs.values()}
    assert families == {"tremor", "reach"}


def test_every_clip_is_finite_inside_the_table_and_shares_its_level0_clock(cid, specs, built):
    s = specs[cid]
    body, truth = built[cid]
    assert torch.isfinite(body.rot6d).all() and torch.isfinite(body.transl).all()
    v = truth["validity"]
    assert v["limits_new"] == 0, (cid, v["limits_worst_deg"])
    assert v["skate_added_m_s"] < 1e-6
    if v.get("hand_limits") is not None:
        assert v["hand_limits"]["ok"], cid
    if s.level0 is not None:
        l0, _ = built[s.level0]
        assert l0.T == body.T and l0.rate == body.rate, cid


# ---------------------------------------------------------------- re-measurement
def _oscillation_ids():
    from mdfm.synth import tremors
    return [s.clip_id for s in tremors.specs() if s.grade and s.clip_id.startswith("tremor_")
            and not any(k in s.clip_id for k in ("orthostatic", "functional", "pillrolling", "combined"))]


@pytest.mark.parametrize("oid", _oscillation_ids())
def test_oscillation_clips_read_back_in_their_band(oid, TR, specs, built):
    s = specs[oid]
    body, truth = built[oid]
    p = truth["params"]
    rb = TR.readback(s, body, truth, built[s.level0][0])
    scale, grade = p["scale"], p["grade"]
    jitter = p["limbs"][next(iter(p["limbs"]))]["waveform"] != "sine"
    for side, r in rb.items():
        lb = p["limbs"][side]
        f = lb["frequency_hz"]
        if jitter:
            band = {"holmes": (2.0, 5.0), "dystonic": (4.0, 10.0), "wing-beating": (2.0, 5.0)}[p["tremor_type"]]
            assert band[0] <= r["frequency_hz"] <= band[1], (oid, side, r)
        else:
            assert abs(r["frequency_hz"] - f) < 0.05, (oid, side, r)
        lo, hi = TR.SCALES[scale][grade]
        # 1e-3 cm: the read at the skinned fingertip of a level placed 0.11 mm under a band's upper edge (3.16 grade
        # 1 above the estimator floor, levelled at the index tip since wave SH1d part B) passes it by 0.15 um
        assert lo < r["pp_cm_max"] < hi + 1e-3, (oid, side, r)
        assert abs(r["pp_cm_max"] / lb["cm_pp"] - 1.0) < (0.15 if p["point"] == "nose" or jitter else 0.05) or \
            grade == 4, (oid, side, r)
        if not jitter and p.get("reach", {}).get("profile", "constant") == "constant":
            assert lo < r["pp_cm_median"] < hi and abs(r["pp_cm_median"] / lb["cm_pp"] - 1) < 0.05, (oid, side, r)
        chord = lb["chord_cm_at_full_envelope"]
        assert chord[0] == pytest.approx(lb["cm_pp_at_unit_waveform"], rel=1e-6)
        assert TR.grade_of(scale, lb["cm_pp"]) == grade or (scale == "TETRAS upper limb" and grade == 1.0)


def test_head_tremor_moves_the_neck_and_head_only(TR, built):
    for cid in ("tremor_E1_head_yaw_g3", "tremor_E1_head_pitch_g4"):
        body, truth = built[cid]
        l0, _ = built["dystonia_E1_level0"]
        changed = [j for j in range(22) if not torch.equal(body.rot6d[:, j], l0.rot6d[:, j])]
        assert changed == [12, 15]
        assert [d["weight"] for d in truth["params"]["limbs"]["left"]["drives"]] == [0.5, 0.5]
    assert built["tremor_E1_head_pitch_g4"][1]["params"]["limbs"]["left"]["frequency_hz"] == TR.HZ["essential_head_g4"]


def test_postural_tremor_of_the_outstretched_hand_moves_it_up_and_down(TR, built):
    """B2 of the clinical review: the hand flexes and extends at the wrist (the shoulder joining from 3 cm), so that
    the middle fingertip moves vertically, not sideways."""
    from mdfm.synth import hands as H
    for cid, l0 in (("tremor_E2_essential_tetras1p5", "tremor_E2_postural_g0"),
                    ("tremor_E2_essential_tetras3p5", "tremor_E2_postural_g0"),
                    ("tremor_E2_physiological_tetras1p5", "tremor_E2_postural_g0"),
                    ("tremor_E2f_functional_g3", "tremor_E2f_functional_g0")):
        body, truth = built[cid]
        b0, _ = built[l0]
        for side in truth["params"]["sides"]:
            v = H.tip_vertices()[side]["middle"]
            D = H.body_points(body, [v])[:, 0] - H.body_points(b0, [v])[:, 0]
            D = D - D.mean(0)
            ax = np.linalg.svd(D, full_matrices=False)[2][0]
            assert abs(ax[1]) > 0.85, (cid, side, ax)
        assert truth["params"]["limbs"][truth["params"]["sides"][0]]["drives"][0]["joint"] in (20, 21)


@pytest.mark.parametrize("did", ["reach_E3_dysmetria_past_sara1", "reach_E3_dysmetria_past_sara2",
                                 "reach_E3_dysmetria_past_sara3", "reach_E3_dysmetria_short_sara2"])
def test_dysmetria_end_points_read_back_in_the_sara5_band(did, TR, specs, built):
    body, truth = built[did]
    rb = TR.readback(specs[did], body, truth)
    g = truth["params"]["grade"]
    lo, hi = TR.SCALES["SARA 5"][g]
    level = TR.band_level("SARA 5", g)
    assert len(rb["endpoint_error_cm"]) == 3
    for e in rb["endpoint_error_cm"]:
        assert lo < e < hi and abs(e / level - 1) < 0.05, rb


def test_the_level0_reach_has_no_end_point_error(TR, RC, built):
    body, truth = built["tremor_E3_reach_left_g0"]
    tip = RC.fingertip(body, "left")
    for ph in truth["params"]["reach_phases"]:
        if ph["kind"] == "to_target":
            k = int(round(ph["t1_s"] * body.rate))
            assert RC.endpoint_error_cm(tip, body.rate, ph["t0_s"], ph["t1_s"] + 0.2, tip[k]) < 0.2


@pytest.mark.parametrize("did,n", [("reach_E3_decomposition_icars2", 2), ("reach_E3_decomposition_icars3", 3)])
def test_decomposition_counts_its_phases(did, n, TR, RC, specs, built):
    body, truth = built[did]
    rb = TR.readback(specs[did], body, truth)
    nr, cyc = RC.decomposed_pace("left")
    assert rb["phases_per_movement"] == [n] * (2 * nr)
    l0, t0 = built["tremor_E3_reach_left_g0"]
    assert TR.readback(specs[did], l0, dict(t0, params=dict(t0["params"], readback="decomposition")))[
        "phases_per_movement"] == [1] * 6
    # m4 of the clinical review: slower than the cited pace, pauses of 150 ms or more, the level 0's clock
    mv = [p for p in truth["params"]["reach_phases"] if p["kind"] == "to_nose"]
    mv0 = [p for p in t0["params"]["reach_phases"] if p["kind"] == "to_nose"]
    assert mv[0]["t1_s"] - mv[0]["t0_s"] > 1.4 * (mv0[0]["t1_s"] - mv0[0]["t0_s"])
    assert RC.PHASE_PAUSE_S >= 0.15 and body.T == l0.T


def test_the_combined_cerebellar_clip_reads_back_both_signs(TR, specs, built):
    body, truth = built["tremor_E3_cerebellar_combined"]
    rb = TR.readback(specs["tremor_E3_cerebellar_combined"], body, truth)
    r = rb["oscillation_highpass"]["left"]
    assert abs(r["frequency_hz"] - TR.HZ["intention"]) < 0.1
    assert 2.0 < r["pp_cm_max"] < 5.0
    assert all(5.0 < e < 15.0 for e in rb["dysmetria"]["endpoint_error_cm"])


def test_intention_tremor_rises_near_the_target(TR, built):
    body, truth = built["tremor_E3_intention_sara2"]
    l0, _ = built["tremor_E3_reach_left_g0"]
    D = TR.highpass(body.joints()[:, 20].numpy() - l0.joints()[:, 20].numpy(), 100.0, 2.0)   # the slow shift out
    env = truth["arrays"]["left_envelope"]
    for ph in truth["params"]["reach_phases"]:
        if ph["kind"] == "to_nose":
            a, b = int(round(ph["t0_s"] * 100)), int(round(ph["t1_s"] * 100))
            mid = slice(a + (b - a) * 2 // 5, a + (b - a) * 3 // 5)
            end = slice(b - 15, b + 15)
            assert TR.M.chord_pp(D[end]) > 1.5 * TR.M.chord_pp(D[mid])
            assert env[mid].max() < 0.6 and env[mid].min() < 0.4 and env[b] == pytest.approx(1.0, abs=1e-4)
            # m1 of the clinical review: the floor itself at mid-movement, a faster fade leaving a target
            m = a + int(np.argmin(env[a:b]))
            assert env[m] == pytest.approx(TR.INTENTION_FLOOR, abs=0.01)
            assert env[a + (b - a) // 4] < env[b - (b - a) // 4]


def test_intention_tremor_oscillates_around_the_nose(TR, RC, built):
    """M2 of the clinical review: at the nose the fingertip swings to both sides of the level-0 fingertip, not to one
    side only."""
    body, truth = built["tremor_E3_intention_sara3"]
    l0, _ = built["tremor_E3_reach_left_g0"]
    D = 100 * (RC.fingertip(body, "left") - RC.fingertip(l0, "left"))
    for ph in truth["params"]["reach_phases"]:
        if ph["kind"] == "hold_nose" and ph["cycle"] is not None:
            k = slice(int(round(ph["t0_s"] * 100)) - 10, int(round(ph["t1_s"] * 100)) + 10)
            x = D[k, 0]
            assert x.min() < -1.0 and x.max() > 1.0, (x.min(), x.max())


def test_jerky_cycles_spread_and_the_level_recurs(TR, built):
    """M1 of the clinical review: the cycle amplitudes spread over [A_MIN, 1] (Beta(2, 2)), a full cycle in every
    4 to 5 s; m3: Holmes periods jittered 15 percent or more."""
    for cid in ("tremor_E2_dystonic_g3", "tremor_E1_holmes_g3", "tremor_E2w_wing_tetras3p5",
                "tremor_E1_dystonic_head_g3"):
        body, truth = built[cid]
        for side, lb in truth["params"]["limbs"].items():
            assert lb["cycle_amplitude_median"] > lb["a_min"] + 0.25 * (1 - lb["a_min"]), (cid, lb)
            assert lb["full_cycles"] >= int(body.duration / TR.PEAK_EVERY_S[1]), (cid, lb)
    assert TR.PERIOD_CV["holmes"] >= 0.15 and TR.A_MIN["holmes"] <= 0.65


def test_dystonic_arm_tremor_has_its_posture_its_pull_and_its_null_point(TR, specs, built):
    """M7 and m8 of the clinical review: a dystonic wrist posture, fast phases toward the pull (the head: away from
    it, recorded), the tremor nearly gone while the wrist is let into the pull."""
    from mdfm.synth import hands as H
    body, truth = built["tremor_E2_dystonic_g3"]
    lb = truth["params"]["limbs"]["left"]
    assert lb["fast_sign"] == -1.0 and "toward" in truth["params"]["fast_phase"]
    assert built["tremor_E1_dystonic_head_g3"][1]["params"]["limbs"]["left"]["fast_sign"] == -1.0
    base, info, gain = TR.dystonic_arm_base("left")
    t = body.times().numpy()
    v = H.tip_vertices()["left"]["middle"]
    D = H.body_points(body, [v])[:, 0] - H.body_points(base, [v])[:, 0]
    a, z = info["null_window_s"]
    null = (t > a + info["null_ramp_s"] + 0.5) & (t < z - info["null_ramp_s"] - 0.5)
    full = t < a - 1.0
    assert TR.M.chord_pp(D[null]) < 0.25 * TR.M.chord_pp(D[full])
    rb = TR.readback(specs["tremor_E2_dystonic_g3"], body, truth)
    assert 3.0 < rb["left"]["pp_cm_max"] < 10.0


def test_the_wing_posture_holds_the_hands_at_nose_level(TR, built):
    """m6 of the clinical review (draft segment E2b): the fingertips near each other at nose height."""
    from mdfm.synth import hands as H
    body, truth = built["tremor_E2w_wing_g0"]
    nose = TR.RC.nose_point(body, knots=[0])[0]
    tl = H.body_points(body, [H.tip_vertices()["left"]["middle"]], knots=[0])[0, 0]
    tr = H.body_points(body, [H.tip_vertices()["right"]["middle"]], knots=[0])[0, 0]
    assert abs(tl[1] - nose[1]) < 0.04 and abs(tr[1] - nose[1]) < 0.04
    assert np.linalg.norm(tl - tr) < 0.12
    rep = TR._wing_solution()[1]
    assert all(r["tip_error_cm"] < 1.5 for r in rep.values())


def _hand_gap_mm(body, every=2):
    """mm: the least distance between a vertex of the left hand (wrist and fingers) and one of the right hand over
    every every-th knot of the skinned mesh."""
    from scipy.spatial import cKDTree
    from mdfm.synth import hands as H
    _, _, dom = H._rest()
    ids = {s: np.flatnonzero(np.isin(dom, (w,) + tuple(range(H.FIRST[s], H.FIRST[s] + 15))))
           for s, w in (("left", 20), ("right", 21))}
    P = H.body_points(body, np.r_[ids["left"], ids["right"]], knots=np.arange(0, body.T, every))
    n = len(ids["left"])
    return 1000.0 * min(float(cKDTree(Pk[n:]).query(Pk[:n])[0].min()) for Pk in P)


@pytest.mark.parametrize("wid", ["tremor_E2w_wing_g0", "tremor_E2w_wing_tetras2p5", "tremor_E2w_wing_tetras3",
                                 "tremor_E2w_wing_tetras3p5"])
def test_the_wing_hands_keep_their_room(wid, TR, built):
    """Wave SH1d part B, review M7: the two hands of the wing-beating clips never come within 20 mm of each other
    (they met at 1.7 mm at grade 3.5 with the fingertips 6 cm apart in the posture)."""
    assert TR.WING["tip_gap_cm"] == 10.0
    assert _hand_gap_mm(built[wid][0]) >= 20.0, wid


HAND_POINTS = ("middle_tip", "index_tip", "middle_tip_trunk", "fist_knuckle")


def test_every_hand_tremor_is_levelled_at_the_hand(TR, specs, built):
    """Rule D4 (decision of 2026-11-03, wave SH1d part B, review B1): every tremor that moves a hand is levelled and
    read at the hand (the middle fingertip; the index tip on the reach), never at the wrist joint, which read a band
    lower than the hand a rater watches (TETRAS 4c 2.5 read 6.7 cm at the index tip)."""
    arm = {16, 17, 18, 19, 20, 21, 13, 14}
    n = 0
    for cid, (body, truth) in built.items():
        p = truth["params"]
        if cid not in specs or not p.get("limbs") or "point" not in p:
            continue
        joints = {d["joint"] for lb in p["limbs"].values() for d in lb.get("drives", [])}
        if joints & arm:
            assert p["point"] in HAND_POINTS, (cid, p["point"])
            n += 1
    assert n >= 30
    for cid in ("tremor_E3_kinetic_g2", "tremor_E3_essential_tetras2p5", "tremor_E3_holmes_g3"):
        assert built[cid][1]["params"]["point"] == "index_tip", cid
    for cid in ("tremor_E1_holmes_g2", "tremor_E2_holmes_g3", "tremor_E2w_wing_tetras3"):
        assert built[cid][1]["params"]["point"] == "middle_tip", cid


def test_orthostatic_tremor_frequency_and_planted_feet(TR, specs, built):
    body, truth = built["tremor_E7_orthostatic_present"]
    l0, t0 = built["tremor_E7_orthostatic_g0"]
    rb = TR.readback(specs["tremor_E7_orthostatic_present"], body, truth, l0)
    assert abs(rb["frequency_hz"] - 15.7) < 0.05
    assert rb["pp_cm"] == pytest.approx(TR.OT_PP_CM, rel=0.05)
    assert truth["params"]["ankle_drift_mm"] < 0.01 and t0["params"]["ankle_drift_mm"] < 0.01
    assert bool(body.contact.all())
    # M5 of the clinical review: the visible signs, a wide base and weight shifts; the level 0 on its own stance
    assert truth["params"]["stance_width_cm"] == pytest.approx(TR.OT_STANCE_CM, abs=0.05)
    assert t0["params"]["stance_width_cm"] < TR.OT_STANCE_CM - 5.0
    x = body.joints()[:, 0, 0].numpy() - l0.joints()[:, 0, 0].numpy()
    assert 100 * (x.max() - x.min()) == pytest.approx(2 * TR.OT_SHIFT_CM, rel=0.02)
    assert len(truth["params"]["weight_shift_onsets_s"]) >= 3 and truth["validity"]["limits_new"] == 0


def test_functional_tremor_entrains_and_pauses(TR, specs, built):
    for g in (2, 3):
        cid = f"tremor_E2f_functional_g{g}"
        body, truth = built[cid]
        rb = TR.readback(specs[cid], body, truth, built["tremor_E2f_functional_g0"][0])
        own = truth["params"]["functional"]["own_hz_range"]
        assert own[0] - 0.3 < rb["own"]["frequency_hz"] < own[1] + 0.3
        assert abs(rb["own"]["frequency_hz"] - TR.HZ["tapping"]) > 1.0
        assert abs(rb["entrained"]["frequency_hz"] - TR.HZ["tapping"]) < 0.05
        assert TR.SCALES["MDS-UPDRS"][g][0] < rb["entrained"]["pp_cm_max"] < TR.SCALES["MDS-UPDRS"][g][1]
        assert rb["pause_pp_cm"] < 0.01
        # m2 of the clinical review: the pause starts after the onset of the ballistic movement; the amplitude
        # varies; the entrainment is not instantaneous
        F = truth["params"]["functional"]
        env = truth["arrays"]["envelope"]
        t = body.times().numpy()
        assert env[t <= F["ballistic_onset_s"] + 0.05].min() == pytest.approx(1.0)
        assert F["amplitude_share_range"][0] <= TR.FUNCTIONAL_AMP_MIN + 1e-9 and F["entrain_transition_s"] >= 2.0
    l0, _ = built["tremor_E2f_functional_g0"]
    b3, _ = built["tremor_E2f_functional_g3"]
    assert torch.equal(l0.rot6d[:, 21], b3.rot6d[:, 21])            # the tapping hand is the level 0's


@pytest.mark.parametrize("pid,g", [("tremor_E1_pillrolling_g1", 1), ("tremor_E1_pillrolling_supination_g2", 2)])
def test_pill_rolling_fingertip_reads_back(pid, g, TR, specs, built):
    body, truth = built[pid]
    rb = TR.readback(specs[pid], body, truth, built["tremor_E1_pillrolling_g0"][0])
    assert abs(rb["frequency_hz"] - TR.HZ["pill_rolling"]) < 0.05
    lo, hi = TR.SCALES["MDS-UPDRS"][g]
    assert lo < rb["pp_cm_max"] < hi
    assert rb["pp_cm_max"] == pytest.approx(truth["params"]["fingertip_cm_pp"], rel=0.05)
    assert truth["validity"]["hand_limits"]["ok"]
    # B3 of the clinical review: the thumb pad stays on the index pad (gap 0 to 3 mm) and slides along it
    c = truth["params"]["contact"]
    assert -0.05 <= c["gap_cm_range"][0] and c["gap_cm_range"][1] <= 0.3
    assert 0.5 <= c["contact_slide_cm"] <= 1.5
    l0 = built["tremor_E1_pillrolling_g0"][1]["params"]["thumb_index_gap_cm"]
    assert l0 == pytest.approx(TR.PILL["gap_cm"], abs=0.02)


# ---------------------------------------------------------------- the reach
def test_the_reach_touches_its_targets_at_the_cited_pace(RC, built):
    body, truth = built["tremor_E3_reach_left_g0"]
    tg = RC.targets("left")
    tip = RC.fingertip(body, "left")
    nose = RC.nose_point(body)
    holds = [p for p in truth["params"]["reach_phases"] if p["kind"] in ("hold_nose", "hold_target")
             and p["cycle"] is not None]
    assert len(holds) == 2 * RC.N_REACHES
    for p in holds:
        k = int(round(0.5 * (p["t0_s"] + p["t1_s"]) * body.rate))
        goal = tg["nose_target"] if p["kind"] == "hold_nose" else tg["target"]
        assert np.linalg.norm(tip[k] - goal) < 0.01
    assert np.linalg.norm(tip - nose, axis=1).min() > 0.005             # never inside the face
    cyc = [p for p in truth["params"]["reach_phases"] if p["cycle"] == 1]
    assert cyc[-1]["t1_s"] - cyc[0]["t0_s"] == pytest.approx(RC.CYCLE_S["left"])
    d = np.linalg.norm(tg["target"] - RC.targets("left")["shoulder"])
    assert d / tg["reach_m"] == pytest.approx(RC.TARGET_SHARE, abs=1e-6)


def test_the_finger_points_at_the_nose_and_the_arm_returns_smoothly(RC, built):
    """m5 of the clinical review: the index pointing at the face, not upright before the lips; the way back runs the
    same postures, so the hold at the target follows without a jump."""
    body, truth = built["tremor_E3_reach_left_g0"]
    assert truth["params"].get("reach_phases")
    from mdfm.synth import tremors as TR
    _, info = TR.reach_base("left")
    assert info["point_at_nose"]["index_from_vertical_deg"] > 35.0
    R = body.local_R()
    step = torch.rad2deg(torch.linalg.matrix_norm(R[1:] - R[:-1]) / math.sqrt(2)).numpy()     # deg per knot
    for p in truth["params"]["reach_phases"]:
        if p["kind"] == "hold_target" and p["cycle"] is not None:
            k = int(round(p["t0_s"] * body.rate))
            assert step[k - 2:k + 2].max() < 0.3, (p, step[k - 2:k + 2])  # the movement ends at rest, no jump


def test_no_reach_clip_steps_beyond_its_neighbourhood(RC, specs, built):
    """Wave SH1c: every E3 clip (the reach level 0s, the kinetic, essential, intention, Holmes and terminal tremors,
    the dysmetria, the decomposition and the combined cerebellar clip): no body joint steps between two knots more
    than STEP_EXCESS_MAX beyond the median step within STEP_NEIGHBOURS knots on either side (position mm, local
    rotation deg). Before the fix the dysmetric clips snapped back to the target posture in one knot at the hold."""
    n = 0
    for cid, s in specs.items():
        if s.segment != "E3":
            continue
        ex = RC.joint_step_excess(built[cid][0])
        for k, bound in RC.STEP_EXCESS_MAX.items():
            m = ex[k]
            assert m.max() < bound, (cid, k, float(m.max()), np.unravel_index(int(np.argmax(m)), m.shape))
        if "hand_deg" in ex:
            assert ex["hand_deg"].max() < RC.STEP_EXCESS_MAX["deg"], cid
        n += 1
    assert n >= 20


@pytest.mark.parametrize("knots,total_deg", [(1, 4.0), (2, 6.0), (3, 6.0)])
def test_the_step_scan_sees_a_snap(knots, total_deg, RC, built):
    """The read-back of the test above is not blind: the level-0 reach with its left shoulder turned by a few degrees
    at the hold, in one knot or spread over two or three (the review of wave SH1c: the largest neighbour, the first
    read, saw a snap of a few tens of degrees over two knots as nothing), steps beyond the bound at every knot of the
    snap, and nowhere else; a snap of one knot reads its whole turn."""
    from mdfm.body.rotations import axis_angle_to_matrix
    body, truth = built["tremor_E3_reach_left_g0"]
    hold = [p for p in truth["params"]["reach_phases"] if p["kind"] == "hold_target" and p["cycle"] == 1][0]
    k = int(round(hold["t0_s"] * body.rate))
    R = body.local_R().clone()
    turn = axis_angle_to_matrix(torch.tensor([0.0, 0.0, math.radians(total_deg / knots)], dtype=R.dtype))
    for i in range(knots):
        R[k + i:, 16] = R[k + i:, 16] @ turn
    ex = RC.joint_step_excess(body.with_rotations(R, [16]))
    rows = np.arange(k - 1, k - 1 + knots)
    assert ex["deg"][rows, 16].min() > RC.STEP_EXCESS_MAX["deg"], ex["deg"][rows, 16]
    assert ex["mm"][rows].max(1).min() > RC.STEP_EXCESS_MAX["mm"]
    assert np.delete(ex["deg"], rows, axis=0).max() < RC.STEP_EXCESS_MAX["deg"]
    if knots == 1:
        assert ex["deg"][k - 1, 16] > 2.0 * RC.STEP_EXCESS_MAX["deg"]


DYSMETRIA = ("reach_E3_dysmetria_past_sara1", "reach_E3_dysmetria_past_sara2", "reach_E3_dysmetria_past_sara3",
             "reach_E3_dysmetria_short_sara2", "tremor_E3_cerebellar_combined")


def _dysmetric_base(TR, built, cid):
    """(body, phases): the clip's reach base (the combined clip's without its tremor)."""
    body, truth = built[cid]
    p = truth["params"]
    if p.get("base_rebuild") == "reach":
        r = p["reach"]
        body = TR.reach_base("left", {int(k): v for k, v in r["endpoint_errors_m"].items()}, r["decomposition"])[0]
    return body, p["reach_phases"]


@pytest.mark.parametrize("did", DYSMETRIA)
def test_the_dysmetric_correction_joins_the_level0_target_posture(did, TR, built):
    """Wave SH1c: the dysmetric movement runs through the hold at the target that follows it (its 'movement_end_s'
    the end of that hold, the hold marked as taken by the correction) and its correction ends on the level 0's target
    posture with no joint speed, so that the movement back to the nose follows with no step (at most 0.3 deg of any
    joint between the knots around it, the level 0's own test above), and outside the dysmetric movements the clip's
    rows are the level 0's."""
    body, phases = _dysmetric_base(TR, built, did)
    l0, _ = built["tremor_E3_reach_left_g0"]
    R = body.local_R()
    step = torch.rad2deg(torch.linalg.matrix_norm(R[1:] - R[:-1]) / math.sqrt(2)).numpy()     # deg per knot
    own = np.zeros(body.T, bool)
    n = 0
    for p in phases:
        if p["kind"] == "to_target" and p.get("miss_m") is not None:
            hold = [h for h in phases if h["kind"] == "hold_target" and h["cycle"] == p["cycle"]][0]
            assert hold["taken_by_correction"] and p["movement_end_s"] == hold["t1_s"], (did, p, hold)
            a, b = int(round(p["t0_s"] * body.rate)), int(round(p["movement_end_s"] * body.rate))
            own[a:b + 1] = True
            assert step[b - 3:b + 3].max() < 0.3, (did, p["cycle"], step[b - 3:b + 3])
            assert step[a - 3:a + 3].max() < 0.3, (did, p["cycle"], step[a - 3:a + 3])
            d = p["movement_end_s"] - p["t0_s"]
            assert p["correction_s"] == pytest.approx((1.0 - TR.RC.PRIMARY_SHARE) * d)
            assert p["primary_end_s"] == pytest.approx(p["t0_s"] + TR.RC.PRIMARY_SHARE * d)
            n += 1
    assert n == 3
    assert torch.equal(body.rot6d[~own], l0.rot6d[~own])
    still = [j for j in range(22) if j not in TR.RC.ARM["left"]]                 # only the reaching arm's rows move
    assert torch.equal(body.rot6d[own][:, still], l0.rot6d[own][:, still])


@pytest.mark.parametrize("did", DYSMETRIA)
def test_the_dysmetric_arm_turns_about_as_fast_as_the_level0s(did, TR, built):
    """Wave SH1c: each arm joint (collar, shoulder, elbow, wrist) of each dysmetric movement turns at most 1.5 times
    as fast as the same joint of the level 0's movement to the target of the same cycle (the overshoot covers more
    ground in the movement and hold time); the first build turned the pointing hand away from the nose in a flick
    and snapped the shoulder at the hold, several times faster, and the second turned the collar of the grade 3
    overshoot several times faster than the level 0's collar (its test compared each joint with the level 0's
    fastest joint of the whole clip)."""
    body, phases = _dysmetric_base(TR, built, did)
    l0, t0 = built["tremor_E3_reach_left_g0"]
    arm = list(TR.RC.ARM["left"])

    def peak(b, t0_s, t1_s):
        a, z = int(round(t0_s * b.rate)), int(round(t1_s * b.rate))
        R = b.local_R()[a:z + 1, arm]
        rel = R[:-1].transpose(-1, -2) @ R[1:]
        c = ((rel.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0).clamp(-1.0, 1.0)
        return np.degrees(torch.arccos(c).numpy()).max(0) * b.rate

    def tip_peak(b, t0_s, t1_s):
        v = np.linalg.norm(np.gradient(TR.RC.fingertip(b, "left"), 1.0 / b.rate, axis=0), axis=-1)
        return float(v[int(round(t0_s * b.rate)):int(round(t1_s * b.rate)) + 1].max())
    n = 0
    for p in phases:
        if p["kind"] == "to_target" and p.get("miss_m") is not None:
            p0 = [q for q in t0["params"]["reach_phases"] if q["kind"] == "to_target" and q["cycle"] == p["cycle"]][0]
            mine, ref = peak(body, p["t0_s"], p["movement_end_s"]), peak(l0, p0["t0_s"], p0["t1_s"])
            # open-body port, wave 2: the level 0's joints no longer kink (reach._smooth_path), so its joint peaks fell
            # (scripts/openbody_reach_kink.py); the bound scales with the fingertip's own speed
            # ratio (at most 2, the test below), so that a joint may turn as much faster as the fingertip runs
            k = max(1.0, tip_peak(body, p["t0_s"], p["movement_end_s"]) / tip_peak(l0, p0["t0_s"], p0["t1_s"]))
            assert (mine < 1.5 * k * ref).all(), (did, p["cycle"], mine, ref, k)
            n += 1
    assert n == 3


@pytest.mark.parametrize("did", DYSMETRIA)
def test_the_dysmetric_correction_passes_beside_the_examiners_finger(did, RC, TR, built):
    """The review of wave SH1c: a miss off to the side is corrected on the curve of reach.correction_via (the overshoot
    drawn back along the movement before the side offset closes): the arm chain's fingertip (the point the IK places)
    within 1 mm of the curve (held to it by reach.BETWEEN['tip_weight']; at the nose posture's tip weight the grade 3
    fingertip left the curve by several millimetres mid-correction), and at the knot farthest from the straight line
    nearer the curve than half that distance; the hand and forearm never inside the examiner's finger (a rigid box:
    no depth allowed, where B1 allows 3 mm of skin); on the straight line the index grazed its underside. An
    undershoot keeps the line."""
    from mdfm.body.rotations import matrix_to_axis_angle
    body, phases = _dysmetric_base(TR, built, did)
    R = body.local_R()[:, list(RC.ARM["left"])]
    dense = np.linspace(0.0, 1.0, 2001)
    n = 0
    for p in phases:
        if p["kind"] != "to_target" or p.get("miss_m") is None:
            continue
        via = p["correction_via_m"]
        assert (via is None) == did.endswith("short_sara2"), (did, via)
        a, b = int(round(p["primary_end_s"] * body.rate)), int(round(p["movement_end_s"] * body.rate))
        q = np.degrees(matrix_to_axis_angle(R[a:b + 1]).numpy()).reshape(-1, 12)
        tip = RC.chain("left").points(q)["tip"].detach().numpy()
        curve = RC.fingertip_path(p["miss_m"], p["tip1_m"], dense, via)
        line = RC.fingertip_path(p["miss_m"], p["tip1_m"], dense)
        d_c = np.linalg.norm(tip[:, None] - curve[None], axis=-1).min(1)
        d_l = np.linalg.norm(tip[:, None] - line[None], axis=-1).min(1)
        assert d_c.max() < 0.001, (did, p["cycle"], d_c.max())
        if via is not None:
            k = int(np.argmax(d_l))
            assert d_c[k] < 0.5 * d_l[k], (did, p["cycle"], d_c[k], d_l[k])
        n += 1
    assert n == 3
    assert built[did][1]["validity"]["hand_head"]["examiner_max_depth_cm"] == 0.0, built[did][1]["validity"]


@pytest.mark.parametrize("did", DYSMETRIA[:4])
def test_the_overshoot_runs_at_most_twice_as_fast_as_the_level0(did, RC, built):
    """The primary of a dysmetric movement covers the longer path to the miss in reach.PRIMARY_SHARE of the level 0's
    movement and hold time (the shared clock), so its fingertip runs faster than the level 0's (reach.py's module
    docstring): the fingertip of the primary and of the correction at most twice the level 0's peak on its own
    movement to the target of the same cycle (the like-for-like peak: the level 0's raise and lower run faster), so
    that the overshoot reads as an overshoot and not as a flung arm."""
    def speed(b):
        return np.linalg.norm(np.gradient(RC.fingertip(b, "left"), 1.0 / b.rate, axis=0), axis=-1)

    def peak(v, rate, t0_s, t1_s):
        return float(v[int(round(t0_s * rate)):int(round(t1_s * rate)) + 1].max())
    body, truth = built[did]
    l0, t0 = built["tremor_E3_reach_left_g0"]
    v, v0 = speed(body), speed(l0)
    n = 0
    for p in truth["params"]["reach_phases"]:
        if p["kind"] == "to_target" and p.get("miss_m") is not None:
            p0 = [q for q in t0["params"]["reach_phases"] if q["kind"] == "to_target" and q["cycle"] == p["cycle"]][0]
            ref = peak(v0, l0.rate, p0["t0_s"], p0["t1_s"])
            for a, z in ((p["t0_s"], p["primary_end_s"]), (p["primary_end_s"], p["movement_end_s"])):
                assert peak(v, body.rate, a, z) < 2.0 * ref, (did, p["cycle"], a, z, peak(v, body.rate, a, z), ref)
            n += 1
    assert n == 3


@pytest.mark.parametrize("did", DYSMETRIA)
def test_the_shoulder_girdle_moves_steadily_through_the_correction(did, TR, built):
    """The review of wave SH1c: with the joint reference alone the collar took up the difference between the curved
    fingertip path of the correction and the linear joint blend, and the grade 3 shoulder girdle dipped mid-correction
    (the collar's angle from the target posture rising again by several degrees). Held to its blend
    (reach.BETWEEN['collar_weight']), the collar's angle from the target posture rises by at most 0.05 deg per knot
    over the correction and never exceeds its value at the correction's start by more than 0.1 deg."""
    body, phases = _dysmetric_base(TR, built, did)
    R = body.local_R()[:, TR.RC.ARM["left"][0]]
    n = 0
    for p in phases:
        if p["kind"] == "to_target" and p.get("miss_m") is not None:
            m, z = int(round(p["primary_end_s"] * body.rate)), int(round(p["movement_end_s"] * body.rate))
            rel = R[z].T @ R[m:z + 1]
            c = ((rel.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0).clamp(-1.0, 1.0)
            ang = np.degrees(torch.arccos(c).numpy())
            assert np.diff(ang).max() < 0.05 and ang.max() < ang[0] + 0.1, (did, p["cycle"], ang)
            n += 1
    assert n == 3


def test_no_reach_clip_enters_the_face_or_the_examiners_finger(TR, specs, built):
    """B1 of the clinical review: the hand and forearm (skinned) never 3 mm or more inside the head, and never inside
    the examiner's finger (reach.examiner_boxes) on any E3 clip (Phase D of docs/PLAN_2026-11-12.md: the bound was
    the skin allowance of 3 mm; the shoulder's arc of the two-phase decomposition took the index 0.9 mm into it,
    reach.DECOMP_LEADS)."""
    n = 0
    for cid, s in specs.items():
        if s.segment != "E3":
            continue
        hh = built[cid][1]["validity"]["hand_head"]
        assert hh["max_depth_cm"] < 0.3 and hh["examiner_max_depth_cm"] == 0.0, (cid, hh)
        n += 1
    assert n >= 20
    for cid in ("tremor_E3_kinetic_g3", "tremor_E3_kinetic_g4", "tremor_E3_essential_tetras3"):
        lb = built[cid][1]["params"]["limbs"]["left"]
        assert max(lb["room_shift_deg_max"]) > 0.5, cid                # the face moved the oscillation


def test_the_examiners_finger_scene(RC, built):
    """M4 of the clinical review: the examiner's finger at the target, tip toward the patient, touched by the level-0
    fingertip at the holds, the patient's hand never inside it."""
    body, truth = built["tremor_E3_reach_left_g0"]
    boxes = RC.examiner_boxes("left")
    tg = RC.targets("left")
    assert 0.0 < RC.box_sdf(tg["target"][None], boxes)[0] < 0.01
    v, f = RC.scene_mesh("chair_target", body, 0.0)
    from mdfm.synth import viewer as V
    cv, cf = V.chair_mesh(0.0)
    assert len(v) == len(cv) + 8 * len(boxes) and len(f) == len(cf) + 12 * len(boxes)
    assert truth["validity"]["hand_head"]["examiner_max_depth_cm"] < 0.3


def test_the_reach_fingertip_runs_nearly_straight_between_the_targets(RC, built):
    body, truth = built["tremor_E3_reach_left_g0"]
    tip = RC.fingertip(body, "left")
    for p in truth["params"]["reach_phases"]:
        if p["kind"] in ("to_nose", "to_target"):
            a, b = int(round(p["t0_s"] * 100)), int(round(p["t1_s"] * 100))
            P = tip[a:b + 1]
            u = (P[-1] - P[0]) / np.linalg.norm(P[-1] - P[0])
            off = P - P[0] - ((P - P[0]) @ u)[:, None] * u
            assert np.linalg.norm(off, axis=1).max() < 0.012
            assert len(RC.speed_runs(tip, 100.0, p["t0_s"], p["t1_s"])) == 1


def test_top_grades_are_slower_and_placed_near_their_lower_edge(TR, specs, built):
    """M3 of the clinical review: 4.5 Hz at grade 4 and TETRAS 3.5, 5 Hz at grade 3 and TETRAS 3; the open top band
    of a limb tremor at TOP_LEVEL_CM; m7: the right reach seen from the right."""
    for cid, hz in (("tremor_E3_kinetic_g4", 4.5), ("tremor_E3_kinetic_g3", 5.0),
                    ("tremor_E2_essential_tetras3p5", 4.5), ("tremor_E2_essential_tetras3", 5.0),
                    ("tremor_E3_essential_tetras3", 5.0), ("tremor_E3_kinetic_g2", 6.0)):
        assert built[cid][1]["params"]["limbs"]["left"]["frequency_hz"] == hz, cid
    assert built["tremor_E3_kinetic_g4"][1]["params"]["limbs"]["left"]["cm_pp"] == TR.TOP_LEVEL_CM["MDS-UPDRS"]
    assert "right" in specs["tremor_E3_reach_right_g0"].views


def test_the_right_reach_mirrors_the_left(RC, built):
    bl, _ = built["tremor_E3_reach_left_g0"]
    br, _ = built["tremor_E3_reach_right_g0"]
    tl, tr = RC.targets("left"), RC.targets("right")
    assert tl["target"][0] == pytest.approx(-tr["target"][0], abs=0.01)
    assert tl["reach_m"] == pytest.approx(tr["reach_m"], abs=0.01)
    assert br.duration > bl.duration                     # the dominant arm's cited cycle is the longer one


# ---------------------------------------------------------------- levels
def test_band_levels_follow_the_log_placement(TR):
    assert TR.band_level("TETRAS head", 2) == pytest.approx(math.sqrt(0.5 * 2.5))
    assert TR.band_level("TETRAS head", 1) == pytest.approx(0.5 / (math.sqrt(0.5 * 2.5) / 0.5))
    assert TR.band_level("TETRAS head", 4) == pytest.approx(5.0 * math.sqrt(2.5 * 5.0) / 2.5)
    assert TR.band_level("SARA 6", 3) == pytest.approx(5.0 * math.sqrt(10.0) / 2.0)
    assert TR.band_level("TETRAS upper limb", 1.5) == pytest.approx(1.0 / math.sqrt(3.0))
    assert TR.band_level("TETRAS upper limb", 1.0) < TR.band_level("TETRAS upper limb", 1.5)
    for scale, bands in TR.SCALES.items():
        for g in bands:
            if scale == "MDS-UPDRS" or (scale == "TETRAS upper limb" and g == 1.0):
                continue
            assert TR.grade_of(scale, TR.band_level(scale, g)) == g


# ---------------------------------------------------------------- determinism
@pytest.mark.parametrize("did", ["tremor_E2_essential_tetras2", "tremor_E2_dystonic_g3", "tremor_E3_intention_sara2",
                                 "tremor_E2f_functional_g2", "tremor_E1_pillrolling_supination_g2",
                                 "tremor_E7_orthostatic_present", "reach_E3_dysmetria_past_sara2",
                                 "tremor_E2w_wing_tetras3"])
def test_the_build_is_deterministic_under_its_seed(did, specs, built):
    body, _ = specs[did].build()
    b0, _ = built[did]
    assert torch.equal(body.rot6d, b0.rot6d) and torch.equal(body.transl, b0.transl)
    if hasattr(body, "hand6d"):
        assert torch.equal(body.hand6d, b0.hand6d)


def test_another_seed_gives_another_tremor(TR):
    from mdfm.synth.base import static_body
    base = static_body("sit_arms_forward", 5.0)
    a, _ = TR.limb_tremor(base, ["left"], 1.7, 6.0, seed=1)
    b, _ = TR.limb_tremor(base, ["left"], 1.7, 6.0, seed=2)
    c, _ = TR.limb_tremor(base, ["left"], 1.7, 6.0, seed=1)
    assert not torch.equal(a.rot6d, b.rot6d) and torch.equal(a.rot6d, c.rot6d)
    j, _ = TR.limb_tremor(base, ["left"], 1.7, 5.0, wave="dystonic", seed=1)
    k, _ = TR.limb_tremor(base, ["left"], 1.7, 5.0, wave="dystonic", seed=2)
    assert not torch.equal(j.rot6d, k.rot6d)


def test_a_tremor_that_cannot_fit_the_table_is_refused(TR):
    from mdfm.synth.base import static_body
    with pytest.raises(ValueError, match="does not fit"):
        TR.limb_tremor(static_body("sit", 2.0), ["left"], 60.0, 5.0, drive="head_pitch", point="nose")


# ---------------------------------------------------------------- the exam set, shared wave of the clinical review
E2_EXAM = [("tremor_E2_postural_g1b", "tremor_E2_postural_g0"), ("tremor_E2_postural_g1a", "tremor_E2_postural_g0"),
           ("tremor_E2_postural_g2", "tremor_E2_postural_g0"), ("tremor_E2_postural_g3", "tremor_E2_postural_g0"),
           ("tremor_E2_postural_g4", "tremor_E2_postural_g0"), ("tremor_E2_reemergent_g2", "tremor_E2_postural_g0"),
           ("tremor_E2_raise_postural_g2", "tremor_E2_raise_long_rest_g0"),
           ("tremor_E2_raise_reemergent_g2", "tremor_E2_raise_long_rest_g0")]


@pytest.fixture(scope="module")
def exam_built(exam_specs):
    """{clip_id: (body, truth)} of the exam's tremor clips of the shared wave and their level-0 partners."""
    ids = {c for pair in E2_EXAM for c in pair}
    ids |= {f"tremor_E1_rest_g{t}" for t in ("0", "1b", "1a", "2", "3", "4")}
    ids |= {f"tremor_E1_constancy_g{g}" for g in (1, 2, 3, 4)}
    return {c: exam_specs[c].build() for c in sorted(ids)}


@pytest.mark.parametrize("eid,l0", E2_EXAM)
def test_the_exam_postural_tremor_moves_the_hand_up_and_down(eid, l0, TR, exam_specs, exam_built):
    """SH4: every arms-outstretched exam tremor is driven at the wrist with the shoulder joining from 3 cm and read
    at the middle fingertip: against its level 0, the fingertip moves vertically, at the asked frequency, by the
    level of its 3.15 grade (the family's top level and frequency at grade 4); the version-2 elbow hinge, vertical
    in this posture, swept the hand sideways."""
    from mdfm.synth import tremor as T
    body, truth = exam_built[eid]
    b0, _ = exam_built[l0]
    p = truth["params"]
    assert p["driver"] == "postural" and p["point"] == "middle_tip" and truth["validity"]["limits_new"] == 0
    rb = T.readback(body, truth, b0)
    g = exam_specs[eid].grade
    lo, hi = TR.SCALES["MDS-UPDRS"][g]
    assert abs(rb["frequency_hz"] - p["frequency_hz"]) < 0.05 and lo < rb["pp_cm_max"] < hi, (eid, rb)
    assert abs(rb["pp_cm_max"] / p["cm_pp"] - 1) < 0.05 and abs(rb["axis"][1]) > 0.98, (eid, rb)
    assert p["cm_pp"] == pytest.approx(TR.level_cm("MDS-UPDRS", g, "below_floor" if eid.endswith("g1b") else
                                                   "above_floor"))
    if eid == "tremor_E2_postural_g4":
        assert p["cm_pp"] == TR.TOP_LEVEL_CM["MDS-UPDRS"] and p["frequency_hz"] == TR.TOP_HZ[4]
    if "reemergent" in eid:
        from mdfm.synth.grades import REEMERGENCE_MEAN_S
        assert p["intervals_s"][0][0] >= REEMERGENCE_MEAN_S - 1e-9


@pytest.mark.parametrize("tag", ["1b", "1a", "2", "3", "4"])
def test_the_exam_rest_tremor_keeps_clear_of_the_chair(tag, TR, exam_built):
    """The rest ladder's forearm lies on the armrest: every grade keeps the skinned forearm and hand clear of the
    chair (tremor.py, ChairRoom) and reads back at the hand in its band; since exam_v7 (wave SH1d, B1 and M8) the
    'rest' driver turns the forearm and the wrist, the elbow joining from grade 3 only; grade 4 at the family's top
    level and frequency, with no centre shift (version 2: the log level of grades.tremor_level_cm at the rest
    frequency, its extension extreme inside the armrest; exam_v5: shifted toward flexion)."""
    from mdfm.synth import tremor as T
    body, truth = exam_built[f"tremor_E1_rest_g{tag}"]
    b0, _ = exam_built["tremor_E1_rest_g0"]
    p, sc = truth["params"], truth["validity"]["scene_clearance"]
    assert sc["max_depth_cm"] == 0.0 and sc["min_clearance_cm"] >= T.CLEAR_GAP_CM - 0.005, sc
    rb = T.readback(body, truth, b0)
    g = int(tag[0])
    lo, hi = TR.SCALES["MDS-UPDRS"][g]
    assert abs(rb["frequency_hz"] - p["frequency_hz"]) < 0.05 and lo < rb["pp_cm_max"] < hi, rb
    assert abs(rb["pp_cm_max"] / p["cm_pp"] - 1) < 0.05
    assert p["driver"] == "rest" and p["point"] == "fingertips" and ("elbow" in p["angle_shares"]) == (g >= 3)
    if g == 4:
        assert p["cm_pp"] == pytest.approx(TR.TOP_LEVEL_CM["MDS-UPDRS"]) and p["frequency_hz"] == TR.TOP_HZ[4]
        assert p["shift_toward"] is None
    for cg in (1, 2, 3, 4):
        assert exam_built[f"tremor_E1_constancy_g{cg}"][1]["validity"]["scene_clearance"]["max_depth_cm"] == 0.0


def test_the_e3_clips_and_the_scene_routes(TR, RC, specs, exam_specs, built):
    """SH6: the E3 clips stand on the scene 'chair_target' (the chair and the examiner's finger at the reach target
    of reach.targets) through exam.scene_mesh; the routes 'chair_armless' (viewer.chair_mesh without armrests) and
    'exam_bed' (ataxia.scene_mesh) give their builders' meshes; any other kind raises."""
    from mdfm.synth import ataxia, exam as EX, viewer as V
    assert TR.E3_SCENE == "chair_target"
    e3 = [s for s in specs.values() if s.segment == "E3"]
    assert len(e3) >= 20 and all(s.scene == "chair_target" for s in e3)
    body, _ = built["tremor_E3_reach_left_g0"]
    v, f = EX.scene_mesh("chair_target", body, 0.0)
    rv, rf = RC.scene_mesh("chair_target", body, 0.0)
    assert np.array_equal(v, rv) and np.array_equal(f, rf)
    tip = v[-8:].mean(0)                                   # the examiner's finger box, at the left target
    assert np.linalg.norm(tip - RC.targets("left")["target"]) < 0.06
    av, _ = EX.scene_mesh("chair_armless", body, 0.0)
    assert np.array_equal(av, V.chair_mesh(0.0)[0][:len(av)]) and len(av) < len(V.chair_mesh(0.0)[0])
    bv, bf = EX.scene_mesh("exam_bed", body, 0.0)
    xv, xf = ataxia.scene_mesh("exam_bed", body, 0.0)
    assert np.array_equal(bv, xv) and np.array_equal(bf, xf)
    for kind in ("exam_bed_long_sitting", "sofa"):
        with pytest.raises(ValueError):
            EX.scene_mesh(kind, body, 0.0)


E1_TREMOR = [f"tremor_E1_rest_g{t}" for t in ("1b", "1a", "2", "3", "4")] + [f"tremor_E1_constancy_g{g}"
                                                                             for g in (1, 2, 3, 4)]


@pytest.mark.parametrize("eid", E1_TREMOR + [e for e, _ in E2_EXAM])
def test_every_exam_tremor_truth_records_both_reading_points(eid, exam_built):
    """Review of the shared wave, then wave SH1d (exam_v7, decision of 2026-11-03: every hand tremor levelled at
    the hand): 3.17 (E1) is levelled at the fingertip that moves most, 3.15 (E2) at the middle fingertip; every
    truth records the wrist, middle fingertip and fingertips readings against its base (tremor.point_readings),
    the levelled one matching the level; on the 3.17 ladder the levelled tip reads the clip's grade and, below
    grade 3, the wrist does not move (the forearm turns about its own axis); on E2 the wrist joint does not move
    below 3 cm."""
    _, truth = exam_built[eid]
    p = truth["params"]
    rd = p["readings"]
    assert p["wrist_pp_cm"] == rd["wrist"]["pp_cm_max"] and p["middle_tip_pp_cm"] == rd["middle_tip"]["pp_cm_max"]
    if eid.startswith("tremor_E1_"):
        assert p["point"] == "fingertips" and p["fingertips_pp_cm"] == rd["fingertips"]["pp_cm_max"]
        assert abs(p["fingertips_pp_cm"] / p["cm_pp"] - 1) < 0.05 and rd["fingertips"]["tip"] == p["level_tip"]
        assert p["fingertips_pp_cm"] >= p["middle_tip_pp_cm"] > p["wrist_pp_cm"]
        g = 2 if "constancy" in eid else int(eid[len("tremor_E1_rest_g")])
        assert rd["fingertips"]["grade_of_max"] == g, rd
        if g < 3:
            assert p["wrist_pp_cm"] < 1e-6
    else:
        assert p["point"] == "middle_tip" and abs(p["middle_tip_pp_cm"] / p["cm_pp"] - 1) < 0.01
        if p["cm_pp"] < 3.0:
            assert p["wrist_pp_cm"] == 0.0


def test_the_exam_v5_set_holds_the_clips_whose_body_changed(exam_specs, exam_built):
    """Review of the shared wave: the exam clips whose body the shared wave changed carry since 'exam_v5' (the
    module's rule since exam_v4) unless a later wave changed them again (wave SH1d, exam_v7: every E1 clip on the
    'rest' driver and the raise clips), the other exam clips keep their set (or a later one: the arising's exam_v6
    of the wave SH1b, exam.BODY_CHANGED_V6, holds no tremor). Every E2 tremor body changed (the 'postural'
    drive); an E1 tremor body changed only where its level, frequency or the chair's clearance moved it, so each E1
    clip is rebuilt by the version-2 operator (no clear, grades.tremor_level_cm at the rest frequency): exam_v5
    held exactly the ones whose body differed, and since exam_v7 every one differs. E2 is captioned at the left
    hand (driven at the wrist); the captions mark the module levels and frequencies of the top grades
    (tremors.TOP_LEVEL_CM, limb_hz), the fingertip where E1 is levelled with its drives, and the direction and
    reason of a centre shift (exam_v9, wave SH1d part C, holds the freezing clips only)."""
    from mdfm.synth import exam as EX, tremor as T
    v5 = {k for k, s in exam_specs.items() if s.since == EX.V5}
    v7 = set(EX.BODY_CHANGED_V7)
    assert v5 == set(EX.BODY_CHANGED_V5) - v7 and all(k.startswith("tremor_E") for k in v5)
    # exam_v10 (open-body port, wave 7): the references and the freezing clips off AMASS, none a tremor (wave 8, D153)
    assert {s.since for k, s in exam_specs.items() if k not in v5} <= {"exam_v2", "exam_v3", "exam_v4", "exam_v6",
                                                                       "exam_v7", "exam_v8", "exam_v9", "exam_v10"}
    assert not any(k.startswith("tremor_") for k in EX.BODY_CHANGED_V6)
    assert {e for e, _ in E2_EXAM} <= v5 | v7 and set(E1_TREMOR) <= v7
    for eid in E1_TREMOR:
        base = EX._static("sit", EX.DUR["E1"])
        if "constancy" in eid:
            cg = int(eid[-1])
            v2, _ = T.tremor(base, "left", grade=2, frequency_hz=EX.TREMOR_HZ["rest"], condition="rest",
                             constancy_grade=cg, seed=EX.SEED + 10 + cg)
        else:
            g = int(eid[len("tremor_E1_rest_g")])
            v2, _ = T.tremor(base, "left", grade=g, grade1="below_floor" if eid.endswith("1b") else "above_floor",
                             frequency_hz=EX.TREMOR_HZ["rest"], condition="rest", tremor_type="parkinsonian_rest",
                             seed=EX.SEED + g)
        body = exam_built[eid][0]
        same = torch.equal(v2.rot6d, body.rot6d) and torch.equal(v2.transl, body.transl)
        assert not same and exam_specs[eid].since == EX.V7, eid          # the 'rest' driver of exam_v7
    for k, s in exam_specs.items():
        if s.family == "tremor" and s.segment == "E2":
            assert "left hand" in s.operator and "forearm" not in s.operator, (k, s.operator)
    d4 = EX.describe(exam_specs["tremor_E2_postural_g4"], exam_built["tremor_E2_postural_g4"][1])
    assert "4.5 Hz (module)" in d4 and "at the middle fingertip (module level)" in d4
    t4 = exam_built["tremor_E2_postural_g4"][1]["params"]
    assert "TOP_LEVEL_CM" in t4["level_rule"] and "limb_hz" in t4["frequency_rule"]
    t1 = exam_built["tremor_E1_rest_g4"][1]
    d1 = EX.describe(exam_specs["tremor_E1_rest_g4"], t1)
    assert f"at {EX.TIP_WORDS[t1['params']['level_tip']]} (module level), forearm pronation-supination" in d1
    assert d1.endswith(", the elbow too") and "toward" not in d1
    t2 = exam_built["tremor_E1_rest_g2"][1]
    d2 = EX.describe(exam_specs["tremor_E1_rest_g2"], t2)
    assert "module" not in d2 and "toward" not in d2 and "elbow" not in d2
    assert f"at {EX.TIP_WORDS[t2['params']['level_tip']]}, forearm pronation-supination and wrist flexion-extension" in d2
    p = {"frequency_hz": 5.0, "cm_pp": 12.0, "centre_shift_deg": -3.0, "shift_toward": "extension"}
    assert "centred 3.0 deg toward extension (joint-limit table)" in EX.describe(
        exam_specs["tremor_E1_rest_g4"], {"operator": "tremor", "params": p})
    from mdfm.synth import viewer as V
    for eid, (_, truth) in exam_built.items():             # each caption fits the band, no ellipsis
        text = EX.describe(exam_specs[eid], truth)
        assert V.Annotator(960, 540, ("", text, ""), "front").lines[1] == text, (eid, text)
