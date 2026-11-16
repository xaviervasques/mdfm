"""Tests of the catalogue gap wave, lane hyper_gaps: the clips added to mdfm/synth/involuntary.py (catalogue rows L02
peak-dose dystonic dyskinesia, C07 dystonia in Huntington's disease, S04 tardive stereotypy) and mdfm/synth/jerks.py
(K06 tonic tics, F03 functional axial jerks). Each clip's realized values are read back from the body against its
level 0 and matched to the grade:

- the dystonic spells (twist_spells): inside their bouts, the distal arm channels lagging, the envelope at 1 while a
  posture is held, each arm posture unlike the last (also for the arm that keeps out of the way), the clean postures
  under the writhing, slower than the choreic fragments (burkhard1999: choreic above dystonic);
- L02 (seated) and C07 (standing): held postures read back (measure_holds) deviating from rest by the grade's level on
  the neck and the trunk (posture_signals, POSTURE_AXIAL_CAP), none in the choreic controls (the peak-dose choreic
  clip L01 at the same step, the HD background alone), each spell's deviation at its mid-hold recorded after the
  guards (realize_spells), the fingers curling or the fist clenching, the C07 background the same for every grade of
  the seed (legs and neck), no new joint-limit violation, no skate, the hands inside their limits, no arm in the
  chair or the body and no forearm in the other after the guards (the second arm keeps out of the way of the first);
  the catalogue's 20 s clips at step 4 and grade 4 built once: contact, varied postures per arm, the realized sizes;
- S04: the marching foot lifted by the step's height (its ball, the heel first) over the weight shifted onto the
  stance foot and set down on its spot at the steady rate, the rubbing palm sliding by the step's stroke along the
  thigh at its rate and staying on it without passing into it or hovering above it (the palm's centre within 10 mm of
  the skin on the mesh);
- K06: tonic tics held for their hold (a plateau read back), longer than the clonic ones, at the intensity's size
  (i4 over i3 by the share ratio), the trunk and the shoulder's abduction at their TONIC_WEIGHTS, the MRVS frequency
  and YGTSS number of the types shown, the hands clear of the thighs;
- F03: every jerk longer than a second, mostly trunk flexion (the head goes forward and down), the shapes varying,
  at least a pause apart, the counts per window of functional_windows (fewer and smaller jerks in the dual-task period
  stated in frame_text), the share of time moving of each duration grade, the severity's size; seated the hands stay
  over the thighs (no hand in a thigh); standing balanced with the feet still;
- the clip specifications (ids new and unique, since catalogue_v1, level 0 in the family, no dash) and the time base
  of each new clip equal to its level 0's.
Bodies are procedural, in the neutral shape; no lockbox, EVAL, CODY or care_pd record is read.
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
def I():
    from mdfm.synth import involuntary
    return involuntary


@pytest.fixture(scope="module")
def JK():
    from mdfm.synth import jerks
    return jerks


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
def sit20(I):
    return I.base_body("sit", I.CHOREA_S)


SHORT = 12.0                     # the dystonic clips are built shorter than the catalogue's 20 s to keep the run short


@pytest.fixture(scope="module")
def sit12(I):
    return I.base_body("sit", SHORT)


@pytest.fixture(scope="module")
def lid3(I):
    return I.lid_dystonic(3, seconds=SHORT)


@pytest.fixture(scope="module")
def hd(I):
    return {g: I.hd_dystonia(g, seconds=SHORT) for g in (0, 3)}


# ---------------------------------------------------------------- the dystonic engine
def test_twist_spells_shape_and_frequency(I, sit20):
    from mdfm.synth.measure import dominant_frequency
    t = sit20.times().numpy()
    rng = np.random.default_rng(3)
    iv = [(1.0, 9.0), (11.0, 19.0)]
    post = [{f"{k}_left": w for k, w in p.items()} for p in I.TWIST_ARM]
    raw, env, spells, env_d, clean = I.twist_spells(t, iv, rng, post)
    assert len(spells) >= 4
    for c in raw:                                                          # the writhing rides on the clean posture
        on = np.abs(clean[c]) > 1e-9
        assert np.all(np.abs(raw[c][on] / clean[c][on] - 1.0) <= 2.5 * I.TWIST["writhe"] + 1e-9)
    outside = np.ones_like(t, bool)
    for a, b in iv:
        outside &= ~((t >= a) & (t <= b))
    assert all(np.abs(x[outside]).max() < 1e-12 for x in raw.values()) and env[outside].max() < 1e-12
    for s in spells:
        a, b = s["held_s"]
        k = (t >= a + 0.01) & (t <= b - 0.01)
        assert np.allclose(env[k], 1.0)                                   # held: the envelope at 1
        assert I.TWIST["min_s"] - 1e-9 <= s["rise_s"] + s["hold_s"] + s["return_s"] <= 4.8 + I.TWIST["min_s"] + 0.3
        p = I.TWIST_ARM[s["posture"]]
        prox = next(c for c in p if c not in I.TWIST_DISTAL)
        dist = next(c for c in p if c in I.TWIST_DISTAL)
        k0 = int(round(s["t0_s"] * 100)) + int(round(0.5 * s["lag_s"] * 100))
        assert abs(raw[f"{prox}_left"][k0]) > 1e-6 and abs(raw[f"{dist}_left"][k0]) < 1e-12   # the hand lags
        k1 = (int(round((s["t0_s"] + s["rise_s"] + s["hold_s"] + s["return_s"]) * 100))
              - int(round(0.5 * s["lag_s"] * 100)))
        assert abs(raw[f"{prox}_left"][k1]) > 1e-6 and abs(raw[f"{dist}_left"][k1]) < 1e-12   # and comes back first
    for a, b in zip(spells, spells[1:]):
        assert a["posture"] != b["posture"]                               # each posture unlike the last
    # the second arm keeps out of the way: no midline posture while the first arm holds one
    mid = I.MIDLINE["lid"]
    left = I.twist_spells(t, iv, np.random.default_rng(4), post, midline=mid)[2]
    avoid = [(x["t0_s"], x["t0_s"] + x["rise_s"] + x["hold_s"] + x["return_s"]) for x in left if x["midline"]]
    right = I.twist_spells(t, iv, np.random.default_rng(5), post, midline=mid, avoid=avoid)[2]
    assert avoid and any(x["midline"] for x in right)
    for x in right:
        end = x["t0_s"] + x["rise_s"] + x["hold_s"] + x["return_s"]
        assert not (x["midline"] and any(x["t0_s"] < b_ and a_ < end for a_, b_ in avoid)), x
    for a, b in zip(right, right[1:]):
        assert a["posture"] != b["posture"]                               # also while keeping out of the way
    # slower than the choreic fragments: the acceleration's dominant frequency (burkhard1999)
    fr, _ = I.flow_raw(sit20, sorted(raw), iv, rng, "lid")
    acc = [dominant_frequency(np.gradient(np.gradient(np.stack([d[c] for c in sorted(d)], 1), 0.01, axis=0), 0.01,
                                          axis=0), 100.0, (0.1, 8.0)) for d in (raw, fr)]
    assert acc[0] < 1.5 and acc[0] < 0.5 * acc[1], acc


# ---------------------------------------------------------------- L02
def test_lid_dystonic_holds_and_levels(I, lid3, sit12):
    out, tr = lid3
    _clean(tr)
    p = tr["params"]
    level = I.STEPS[3][0]
    for side, cm in I.guard_after_cm(tr).items():                         # no arm inside the chair or the body
        assert cm <= 100 * I.GUARD_TOL_M + 1e-9, (side, cm)
    for r, rec in p["mix"].items():
        po = rec["postures"]
        axial = r in ("neck", "trunk")
        lv = min(level, I.POSTURE_AXIAL_CAP) if axial else level
        assert po["reachable"] and po["realized_before_table"] == pytest.approx(lv, rel=1e-3)   # one-sided
        assert rec["background"]["level"] == pytest.approx(min(I.TWIST_BG * level, I.TRUNK_NECK_CAP))
        assert po["posture_share"] >= 0.75 * I.STEPS[3][1] and po["held_half_share"] >= 0.5 * I.STEPS[3][1], po
        # each spell's deviation at its mid-hold, read back after the guards (realize_spells)
        X = {c: I.channel_values(out, sit12, c) for c in po["roms_deg"]}
        dev = np.max([np.abs(x) / I.channel_rom(c) for c, x in X.items()], axis=0)
        mids = [int(round(0.5 * sum(s["held_s"]) * out.rate)) for s in po["spells"]]
        assert [s["realized_share"] for s in po["spells"]] == pytest.approx([dev[k] for k in mids], abs=1e-9)
        assert po["realized_median"] == pytest.approx(float(np.median([dev[k] for k in mids])), abs=1e-9)
        if axial:                                     # held, and deviating from rest by the level (TWIST 'amp')
            h = I.measure_holds(out, sit12, r, lv)
            assert h["runs"] >= 0.6 * len(po["spells"]) and min(h["durations_s"]) >= I.HOLD_MIN_S, (r, h)
            assert float(np.median(h["run_dev_shares"])) == pytest.approx(lv, rel=0.25), (r, h)
            assert po["realized_median"] == pytest.approx(lv, rel=0.25), (r, po["realized_median"])
    m = p["measured"]
    for s in ("left", "right"):
        assert m[f"fingers_{s}"]["amplitude_share"] >= 0.5 * I.LID_DYST_FINGERS     # the fingers curl
    for r in ("neck", "trunk"):                       # peak to peak: the postures on both sides with the background
        lv = min(level, I.POSTURE_AXIAL_CAP)
        assert lv * 0.95 <= m["regions"][r]["amplitude_share"] <= 2 * lv + I.TWIST_BG * level + 0.02


def test_lid_choreic_control_holds_nothing(I, sit12):
    out, tr = I.lid_peak_dose("sit", 3, seconds=SHORT)
    for r in ("neck", "trunk", "arm_left", "arm_right"):
        assert I.measure_holds(out, sit12, r, I.STEPS[3][0])["runs"] == 0, r


# ---------------------------------------------------------------- C07
def test_hd_dystonia_grade_and_shared_background(I, hd):
    stand = I.base_body("stand", SHORT)
    (o0, t0), (o3, t3) = hd[0], hd[3]
    for tr in (t0, t3):
        _clean(tr)
        for side, cm in I.guard_after_cm(tr).items():
            assert cm <= 100 * I.GUARD_TOL_M + 1e-9, (side, cm)
    lv = I.AMP[I.UHDRS[3][0]]
    for r in ("trunk", "arm_left", "arm_right"):
        assert I.measure_holds(o0, stand, r, lv)["runs"] == 0, r            # the background alone holds nothing
    lt = min(lv, I.POSTURE_AXIAL_CAP)
    h = I.measure_holds(o3, stand, "trunk", lt)
    n_sp = len(t3["params"]["mix"]["trunk"]["postures"]["spells"])
    for r in ("trunk", "arm_left", "arm_right"):                         # present for the duration term's share
        assert t3["params"]["mix"][r]["postures"]["posture_share"] >= 0.75 * I.DUR[I.UHDRS[3][1]], r
    assert h["runs"] >= 0.6 * n_sp and float(np.median(h["run_dev_shares"])) == pytest.approx(lt, rel=0.25), h
    assert h["held_channel"] in ("trunk_lat", "trunk_rot")                 # bent sideways and turned
    assert t3["params"]["mix"]["trunk"]["postures"]["realized_median"] == pytest.approx(lt, rel=0.25)
    turned, n_arm = 0, 0
    for r in ("arm_left", "arm_right"):
        po = t3["params"]["mix"][r]["postures"]
        assert po["realized_before_table"] == pytest.approx(lv, rel=1e-3)
        rot = I.channel_values(o3, stand, f"sh_rot_{r.split('_')[1]}")
        mids = [int(round(0.5 * sum(s["held_s"]) * o3.rate)) for s in po["spells"]]
        turned += sum(rot[k] < -5.0 for k in mids)
        n_arm += len(mids)
        assert t3["params"]["measured"][f"fingers_{r.split('_')[1]}"]["amplitude_share"] >= 0.5 * min(
            I.HD_FIST, I.FIST_PER_LEVEL * lv)                                # the fist
    assert turned >= 0.6 * n_arm      # internal rotation at the holds (the guard may trim an arm a bowing trunk meets)
    # the same background for every grade of the seed: the neck moves identically (the legs follow the trunk's
    # balance, balance_standing)
    rows = [12, 15]
    assert torch.equal(o0.rot6d[:, rows], o3.rot6d[:, rows])
    J, Jb = o3.joints().numpy(), stand.joints().numpy()
    assert np.abs(J[:, [7, 8, 10, 11]] - Jb[:, [7, 8, 10, 11]]).max() < 1e-9     # both feet planted
    assert t0["params"]["mix"]["trunk"]["background"] == t3["params"]["mix"]["trunk"]["background"]
    m = t0["params"]["measured"]["regions"]
    assert m["trunk"]["amplitude_share"] == pytest.approx(min(I.AMP["mild"], I.TRUNK_NECK_CAP), rel=0.02)


# ---------------------------------------------------------------- the catalogue's 20 s dystonic clips at the top grade
@pytest.fixture(scope="module")
def top(I):
    sp = {s.clip_id: s for s in I.specs()}
    return {c: sp[c].build() for c in ("invol_lid_dystonic_E1_s4", "invol_hd_dystonia_E7_g4")}


ARM_REALIZED_SHARE = 0.55     # at the marked level 0.70 the shoulder's internal rotation and the pronation stop at
#                               their soft limits (0.45 to 0.49 of their ROM) and the guard trims the seated turned-in
#                               arm
ARM_WORST_SHARE = 0.6         # and the other seated arm at least this share of the level (0.68 and 0.85 since the
#                               postures turn the hand in by the forearm and the wrist, the review of wave SH1d)


def test_dystonic_top_grade_clips(I, top):
    for cid, (out, tr) in top.items():
        _clean(tr)
        for side, cm in I.guard_after_cm(tr).items():                     # no arm in the chair, the body or the other
            assert cm <= 100 * I.GUARD_TOL_M + 1e-9, (cid, side, cm)
        assert tr["validity"]["arm_contact"]["ok"], (cid, tr["validity"]["arm_contact"])   # and a refusal if any
        assert tr["validity"]["thigh_pair_cm"]["over_base_1cm_share"] < 0.01, cid   # the inner thighs, read back
        p = tr["params"]
        base = I.base_body(p["posture"], out.duration)
        level = p["level_share_of_rom"]
        lt = min(level, I.POSTURE_AXIAL_CAP)
        axial = ("neck", "trunk") if "lid" in cid else ("trunk",)
        for r in axial:
            po = p["mix"][r]["postures"]
            assert po["realized_median"] >= 0.6 * lt and po["realized_median"] == pytest.approx(lt, rel=0.25), (cid, r)
            h = I.measure_holds(out, base, r, lt)
            assert h["runs"] >= 0.6 * len(po["spells"]), (cid, r, h)
            assert float(np.median(h["run_dev_shares"])) == pytest.approx(lt, rel=0.25), (cid, r, h)
        best, worst = 0.0, 1.0
        for r in ("arm_left", "arm_right"):
            po = p["mix"][r]["postures"]
            seq = [s["posture"] for s in po["spells"]]
            assert len(set(seq)) >= 2 and all(a != b for a, b in zip(seq, seq[1:])), (cid, r, seq)   # varied
            best, worst = max(best, po["realized_median"]), min(worst, po["realized_median"])
        assert best >= ARM_REALIZED_SHARE * level, (cid, best)
        if "lid" in cid:            # the review of wave SH1d (major 3): the left arm had read a median 0.20 of 0.70
            assert worst >= ARM_WORST_SHARE * level, (cid, worst)
        # the arms come after the neck and the trunk
        assert sorted(p["mix"][r]["postures"]["order"] for r in ("arm_left", "arm_right")) == [2, 3]


def test_the_dystonic_hands_keep_apart(I):
    """The review of wave SH1d (major 2): in the catalogue's mild dystonic dyskinesia the two forearms and hands came
    within 0.3 to 1.2 mm of each other's skin from 15.9 to 17.0 s, the second guard pushing against armrests the drawn
    chair has not. The guards now read the drawn seated scene (SCENE['sit']): no arm inside (arm_contact) and the
    skins at least half of PAIR_CLEAR_M apart at every knot."""
    from scipy.spatial import cKDTree
    from mdfm.synth import viewer as V
    sp = {s.clip_id: s for s in I.specs()}
    out, tr = sp["invol_lid_dystonic_E1_s2"].build()
    _clean(tr)
    assert tr["validity"]["arm_contact"]["ok"], tr["validity"]["arm_contact"]
    for v in (tr["params"]["construction"].get("lift") or {}).values():
        assert all(r["clear"] for r in v["runs"]), v
    sets, _, _ = I._pair_geometry("arms")
    Vk = V.knot_vertices(out, np.arange(0, out.T, 5))
    gap = min(float(cKDTree(x[sets["right"]]).query(x[sets["left"]])[0].min()) for x in Vk)
    assert gap >= 0.5 * I.PAIR_CLEAR_M, gap


# ---------------------------------------------------------------- S04
@pytest.mark.parametrize("step", [2, 4])
def test_marching_in_place(I, step):
    out, tr = I.td_marching(step)
    _clean(tr)
    base = I.base_body("stand", I.CHOREA_S)
    J, Jb = out.joints().numpy(), base.joints().numpy()
    lift = J[:, [10, 11], 1] - Jb[:, [10, 11], 1]                          # the balls of the feet
    assert lift.min() > -1e-9 and (J[:, [7, 8], 1] - Jb[:, [7, 8], 1]).min() > -1e-9
    assert lift.max(0) == pytest.approx([I.TD_MARCH_LIFT_M[step]] * 2, rel=0.12)   # jittered cycles
    from mdfm.synth.body import foot_points
    F, Fb = foot_points(out.joints()).numpy(), foot_points(base.joints()).numpy()
    assert np.abs((F - Fb)[..., [0, 2]]).max() < 1e-6                    # set down on its spot, rolling on its ball
    flat = (np.abs(J[:, [7, 8], 1] - Jb[:, [7, 8], 1]) < 1e-9) & (lift < 1e-9)
    assert np.abs((J[:, [7, 8]] - Jb[:, [7, 8]])[..., [0, 2]][flat]).max() < 1e-6   # flat again where it was
    st = tr["params"]["construction"]["legs"][0]["stance"]                # the weight on the stance foot (hof2005)
    assert st["ok"]
    for s in ("left", "right"):
        assert st[s]["cop_to_stance_mm_max"] <= 1000 * I.COP_STANCE_TOL_M, st
        assert st[s]["ankle_rise_at_ball_off_mm_min"] >= 10.0, st
    m = tr["params"]["measured"]["regions"]
    for s in ("left", "right"):
        assert m[f"leg_{s}"]["dominant_hz"] == pytest.approx(I.TD_STEREO["marching"]["rate_hz"], abs=0.08)
        up = lift[:, 0 if s == "left" else 1] > 0.5 * I.TD_MARCH_LIFT_M[step]
        starts = np.array([a for a, _ in I._runs(up)]) / out.rate
        gaps = np.diff(starts)
        gaps = gaps[gaps < 2.0 / I.TD_STEREO["marching"]["rate_hz"]]      # inside the bouts
        assert len(gaps) >= 5 and gaps.std() / gaps.mean() < 0.12         # the same step repeated
    moving = I.close_mask(lift.max(1) > 0.005, out.rate, gap_s=1.5)
    assert moving.mean() == pytest.approx(I.STEPS[step][1], abs=0.12)
    assert not m["trunk"]["involved"]


@pytest.mark.parametrize("step", [2, 4])
def test_rubbing_the_thighs(I, step, sit20):
    out, tr = I.td_rubbing(step)
    _clean(tr)
    p = tr["params"]
    for s in ("left", "right"):
        assert p["stroke_measured_m_pp"][s] == pytest.approx(I.TD_RUB_PP_M[step], rel=0.12)   # jittered cycles
        assert tr["validity"]["hand_clearance"][s]["penetration_cm_max"] <= 100 * I.GUARD_TOL_M
        assert p["measured"]["regions"][f"arm_{s}"]["dominant_hz"] == pytest.approx(
            I.TD_STEREO["rubbing"]["rate_hz"], abs=0.08)
    rows = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15]
    assert torch.equal(out.rot6d[:, rows], sit20.rot6d[:, rows])          # only the arms move
    assert p["stroke_centre_m"] - 0.5 * I.TD_RUB_PP_M[step] * (1 + 2 * I.STEREO_JITTER) >= -I.RUB_BACK_M - 1e-12
    assert all(x["wrist_reach_clipped_share"] <= 0.02 for x in p["construction"]["legs"])  # the wrist reaches it
    #                                                                     (the last mm at the knee end; the gap below)
    ks = np.arange(0, out.T, 10)
    rep = I.palm_rest_report(out, ks)                                    # the palm's centre on the skin (review M1),
    assert rep["ok"], rep                                                # no palm vertex pressed in
    for s in ("left", "right"):                                          # never hovering more than 5 mm over its rest
        assert rep[s]["patch_median_mm_max"] <= 1000 * I.PALM_GAP_M + 5.0, (s, rep[s])
        assert rep[s]["patch_median_mm_min"] >= 1000 * I.PALM_GAP_M - 2.0, (s, rep[s])


# ---------------------------------------------------------------- K06
@pytest.fixture(scope="module")
def jspecs(JK):
    return {s.clip_id: s for s in JK.specs()}


@pytest.fixture(scope="module")
def jbuilt(jspecs):
    want = [c for c in jspecs if c.startswith(("jerk_tonic_tics", "jerk_functional"))] + ["jerk_E1_level0",
                                                                                         "jerk_E7_level0"]
    return {c: jspecs[c].build() for c in want}


def _sites(truth):
    return sorted({p["site"] for e in truth["params"]["events"] for p in e["parts"]})


def test_tonic_tics_hold_and_grades(JK, jspecs, jbuilt):
    base = jbuilt["jerk_E1_level0"][0]
    rise, fall = JK.timing("tic")
    peaks = {}
    for cid in ("jerk_tonic_tics_E1_f1_i3", "jerk_tonic_tics_E1_f1_i4", "jerk_tonic_tics_E1_f2_i3_mixed"):
        b, t = jbuilt[cid]
        s, p = jspecs[cid], t["params"]
        v = t["validity"]
        assert v["limits_new"] == 0 and v["skate_added_m_s"] < 1e-6 and v["contact_lost_share"] == 0.0
        assert v["hand_contact"]["ok"], v["hand_contact"]                   # no hand in a thigh
        ev = p["events"]
        m = JK.measure_jerks(b, base, JK.site_joints(_sites(t)))
        assert m["n"] == len(ev) == p["count"]
        share = JK.YGTSS_INTENSITY_SHARE[s.grade["ygtss_intensity"]]          # the tonic sites at their weights
        for ty in JK.TONIC_TYPES:
            for site, deg in p["peaks_deg"][ty].items():
                assert deg == pytest.approx(np.degrees(JK.peak_of(base, site, share * JK.TONIC_WEIGHTS[site], 50)),
                                            rel=1e-9)
        for e in ev:                                                        # read back: the trunk at its weight
            if e["label"] == "tonic_trunk_tense":
                k0, k1 = (np.array(JK.event_span(e)) * 100).round().astype(int)
                tf = JK.deviation_deg(b, base, [3, 6, 9]).sum(1)[k0:k1].max()      # the chain's total turn
                # the event's own size (jerks.vary_sizes, wave SH1d part B): the peak times its size factor
                want = max(abs(np.degrees(q["peak"])) for q in e["parts"] if q["site"] == "trunk_flexion")
                assert want <= p["peaks_deg"]["tonic_trunk_tense"]["trunk_flexion"] * e["size_factor"] * (1 + 1e-9)
                assert tf == pytest.approx(want, rel=0.05), tf
        assert JK.mrvs_frequency_grade(m["n"] * 60.0 / b.duration) == s.grade["mrvs_frequency"] == t["grade"][
            "mrvs_frequency"]
        assert t["grade"]["ygtss_number"] == s.grade["ygtss_number"] == JK.ygtss_number_grade(len(p["tic_types_shown"]))
        tonic = np.array([e["label"] in JK.TONIC_TYPES for e in ev])
        assert tonic.sum() == p["n_tonic"] and tonic.any()
        assert (m["duration_s"][tonic] >= rise + JK.TONIC_HOLD_S[0] + fall - 0.02).all()
        assert (m["duration_s"][~tonic] <= rise + fall + 0.02).all()      # the clonic ones are brief
        for e, held in zip([e for e in ev if e["label"] in JK.TONIC_TYPES], p["holds_s"]):
            q = e["parts"][0]
            joints = list(JK.SITES[q["site"]][0])
            dev = JK.deviation_deg(b, base, joints).max(1)
            k0, k1 = int(round(e["onset_s"] * 100)), int(round((e["onset_s"] + JK._event_dur(e["parts"])) * 100))
            seg = dev[k0:k1 + 1]
            plateau = (seg >= seg.max() - 1e-6).sum() - 1                 # knots at the peak, read back
            assert plateau / 100.0 == pytest.approx(held, abs=0.011), (cid, e["label"], plateau, held)
            assert JK.TONIC_HOLD_S[0] - 0.01 <= held <= JK.TONIC_HOLD_S[1] + 0.01
            peaks.setdefault(cid, {}).setdefault(e["label"], seg.max() / e["size_factor"])   # the median jerk
        # the arms the trunk carries (jerks.arms_rule, wave SH1d part B) are passive, not a tic's area
        areas, _ = JK.tic_areas(JK.apply_events(base, ev)[0], base)
        assert areas == p["areas"]
    ratio = JK.YGTSS_INTENSITY_SHARE[4] / JK.YGTSS_INTENSITY_SHARE[3]
    for lab in set(peaks["jerk_tonic_tics_E1_f1_i3"]) & set(peaks["jerk_tonic_tics_E1_f1_i4"]):
        r = peaks["jerk_tonic_tics_E1_f1_i4"][lab] / peaks["jerk_tonic_tics_E1_f1_i3"][lab]
        assert r == pytest.approx(ratio, rel=0.15), (lab, r)


# ---------------------------------------------------------------- F03
MOVING_SHARE = {1: (0.0, 0.35), 2: (0.3, 0.6)}
# the module's reading of the S-FMDRS duration anchors outside the dual task: 1 once or a few times (a third of the
# time at most), 2 frequent but intermittent (moving under 60 percent of the time, a pause between every two jerks)


def test_functional_axial_jerks(JK, jspecs, jbuilt):
    a, b_ = JK.DISTRACTION["window_s"]
    means = {}
    labels = set()
    durs = []
    for cid in [c for c in jbuilt if c.startswith("jerk_functional")]:
        b, t = jbuilt[cid]
        s, p = jspecs[cid], t["params"]
        standing = "E7" in cid
        base = jbuilt["jerk_E7_level0" if standing else "jerk_E1_level0"][0]
        v = t["validity"]
        assert v["limits_new"] == 0 and v["skate_added_m_s"] < 1e-6 and v["contact_lost_share"] == 0.0
        assert v["hand_contact"]["ok"], (cid, v["hand_contact"])            # seated, no hand in a thigh
        if standing:
            assert v["balance"]["ok"]
            assert float((b.joints()[:, [7, 8, 10, 11]] - base.joints()[:, [7, 8, 10, 11]]).abs().max()) < 1e-9
        else:                                                               # the hands stay where they rest
            Jb, Jo = base.joints().numpy(), b.joints().numpy()               # (within the table and the reach)
            assert np.linalg.norm(Jo[:, [20, 21]] - Jb[:, [20, 21]], axis=-1).max() < 0.01
            assert p["carry_arms"]["carried_share"] > 0.0
        assert p["frame_text"] == [{"from_s": a, "to_s": b_, "text": JK.DISTRACTION["text"]}]
        ev = p["events"]
        assert {q["type"] for e in ev for q in e["parts"]} == {"functional"}
        m = JK.measure_jerks(b, base, JK.site_joints(_sites(t)))
        assert m["n"] == len(ev)
        span = np.array([JK.event_span(e) for e in ev])
        assert np.abs(m["onset_s"] - span[:, 0]).max() <= 0.05
        assert (m["duration_s"] >= 1.0).all() and (np.array(p["bursts_s"]) >= 1.0).all()   # bursts over 1000 ms
        assert (span[1:, 0] - span[:-1, 1] >= JK.FUNCTIONAL["pause_s"] - 0.011).all()      # a pause between two
        durs += list(m["duration_s"])
        labels |= set(p["labels"])
        inside = (span[:, 0] >= a) & (span[:, 1] <= b_)
        assert inside.sum() == sum(p["dual_task_events"]) == p["count_dual_task"]
        wins = JK.functional_windows(JK.FUNCTIONAL_PER_30S[s.grade["sfmdrs_duration"]])
        assert [w["count"] for w in p["windows"]] == [w[4] for w in wins]
        assert inside.sum() == wins[1][4] and (~inside).sum() == wins[0][4] + wins[2][4] == p["count_outside"]
        outside_s = (wins[0][1] - wins[0][0]) + (wins[2][1] - wins[2][0])
        rate_out, rate_in = (~inside).sum() / outside_s, inside.sum() / (wins[1][1] - wins[1][0])
        assert rate_in <= JK.DISTRACTION["rate"] * rate_out + 1e-9, (cid, rate_in, rate_out)
        assert (~inside).sum() == pytest.approx(JK.FUNCTIONAL_PER_30S[s.grade["sfmdrs_duration"]] * outside_s / 30.0,
                                                abs=1.0)
        str_in = f"{wins[1][4]} jerk" if wins[1][4] else "no jerk"
        assert f"{(~inside).sum()} jerks in the" in s.params_text and str_in in s.params_text     # the caption
        moving = np.zeros(b.T, bool)                                        # the share of time moving, read back
        for x0, x1 in (span * 100).round().astype(int):
            moving[x0:x1] = True
        out_mask = np.zeros(b.T, bool)
        for w in (wins[0], wins[2]):
            out_mask[int(round(w[0] * 100)):int(round(w[1] * 100))] = True
        assert moving[out_mask].mean() == pytest.approx(p["moving_share_outside"], abs=0.01)
        dur = s.grade["sfmdrs_duration"]
        assert MOVING_SHARE[dur][0] <= p["moving_share_outside"] <= MOVING_SHARE[dur][1], \
            (cid, p["moving_share_outside"])
        fac = np.array([e["factor"] for e in ev])
        assert (fac[inside] <= JK.DISTRACTION["amp"] + 1e-9).all() and (fac[~inside] >= JK.FUNCTIONAL["amp"][0]).all()
        # mostly trunk flexion: the head forward and down at every jerk, the trunk never extends. The drop (open-body
        # port, wave 3 follow-up, D26, D39): a bow from upright lowers the head by the square of its forward excursion
        # f over twice the pivot's distance (f^2 / 2r, a rigid bow about the hip line, r the base's hip-to-head
        # distance, 0.64 m seated on the open body); the trunk and the neck flexing above the hips lower it more for
        # the same f, a slide or an extension does not. The 1 cm asked before read the AMASS rooms (the trunk's
        # forward room two to three times the anatomical one): on the anatomical table the severity-2 jerks at 0.6 of
        # their size and the dual task's at 0.6 of that move the head 4 to 5 cm forward and 0.6 to 0.7 cm down
        # (scripts/openbody_wave3_followup.py, functional)
        Jo, Jb = b.joints().numpy(), base.joints().numpy()
        r = float(np.linalg.norm(Jb[0, 15] - 0.5 * (Jb[0, 1] + Jb[0, 2])))
        for e, (k0, k1) in zip(ev, (span * 100).round().astype(int)):
            d = Jo[k0:k1, 15] - Jb[k0:k1, 15]
            f = d[:, 2].max()
            assert f > 0.02 and -d[:, 1].min() > f * f / (2.0 * r), (cid, f, d[:, 1].min())
        tf = JK.signed_angle_deg(b, base, 6, "x")
        assert tf.min() >= -0.5
        dev = JK.deviation_deg(b, base, [3, 6, 9]).sum(1)
        pk = np.array([dev[k0:k1].max() for k0, k1 in (span * 100).round().astype(int)])
        means[cid] = float(np.mean(pk[~inside] / fac[~inside]))
        if inside.any():
            assert pk[inside].max() < pk[~inside].max()
    assert len(labels) >= 3                                               # the shapes vary
    assert np.std(durs) / np.mean(durs) > 0.1
    r = means["jerk_functional_E1_axial_sev3_dur2"] / means["jerk_functional_E1_axial_sev2_dur2"]
    share = JK.FUNCTIONAL_SEVERITY_SHARE
    assert r == pytest.approx(share[3] / share[2], rel=0.25), r          # the weights vary per jerk


# ---------------------------------------------------------------- specifications and time bases
NEW_INVOL = ["invol_lid_dystonic_E1_s2", "invol_lid_dystonic_E1_s3", "invol_lid_dystonic_E1_s4",
             "invol_hd_dystonia_E7_g0", "invol_hd_dystonia_E7_g2", "invol_hd_dystonia_E7_g3",
             "invol_hd_dystonia_E7_g4"] + [f"invol_td_stereotypy_{k}_s{s}" for k in ("E7_marching", "E1_rubbing")
                                          for s in (2, 3, 4)]
NEW_JERK = ["jerk_tonic_tics_E1_f1_i3", "jerk_tonic_tics_E1_f1_i4", "jerk_tonic_tics_E1_f2_i3_mixed",
            "jerk_functional_E1_axial_sev2_dur2", "jerk_functional_E1_axial_sev3_dur2",
            "jerk_functional_E1_axial_sev2_dur1", "jerk_functional_E7_axial_sev2_dur2"]


def test_specs_new_ids(I, jspecs):
    from mdfm.synth.exam import SEGMENTS, exam_set
    exam_ids = {s.clip_id for s in exam_set()}
    isp = {s.clip_id: s for s in I.specs()}
    for ids, sp in ((NEW_INVOL, isp), (NEW_JERK, jspecs)):
        for cid in ids:
            s = sp[cid]
            assert cid not in exam_ids and s.since == "catalogue_v1" and s.level0 in sp
            assert s.segment in SEGMENTS and s.params_text and s.grade_text and s.views
            for txt in (s.params_text, s.grade_text, s.note, s.operator):
                assert chr(0x2014) not in txt and chr(0x2013) not in txt, cid
    assert isp["invol_lid_dystonic_E1_s3"].grade == 3 and isp["invol_hd_dystonia_E7_g4"].grade == 4


def test_time_bases(I, jspecs, jbuilt):
    isp = {s.clip_id: s for s in I.specs()}
    l0 = {k: isp[k].build()[0] for k in ("invol_E1_level0", "invol_E7_level0")}
    for cid in ("invol_td_stereotypy_E7_marching_s3", "invol_td_stereotypy_E1_rubbing_s3", "invol_lid_dystonic_E1_s2"):
        b, _ = isp[cid].build()
        assert (b.T, b.rate) == (l0[isp[cid].level0].T, l0[isp[cid].level0].rate), cid
    for cid in NEW_JERK:
        b = jbuilt[cid][0]
        b0 = jbuilt[jspecs[cid].level0][0]
        assert (b.T, b.rate) == (b0.T, b0.rate), cid
