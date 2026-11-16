"""Tests of the brief-jerk family of the symptom catalogue, mdfm/synth/jerks.py (task CAT-JERK):

- the engine: the pulse (zero outside its span, 1 at the end of the rise, held over the hold), the point process
  (the count placed exactly, the gaps kept, a count that cannot fit refused), the sites' anatomical directions (a
  forward arm rises, a palm-down hand rises on extension and drops on flexion, a shrug raises the shoulder, the head
  goes forward on flexion, a seated knee extension raises the foot), the dip and the pelvis turn keep both feet
  still;
- every clip of specs(): built, finite, no new joint-limit violation, no skate added and no contact lost on the
  checked feet, its level-0 partner on the same time base (the exam clip tremor_E2_raise_g0 for the outstretched
  arms), the same build under the same seed and another one under another seed;
- the re-measurement per operator read from the body against its base (measure_jerks, measure_clonus,
  measure_dips, tic_areas) matches the grade: the count and its UMRS, Conn or MRVS band, the onsets, the movement
  durations from the angular velocity, the lapse depth and drop time, the clonus beats, frequency, excursion and
  SCATS grade, the tic areas and types, the startle latency, habituation and stiffness, the segmental rhythm, the
  stimulus lock;
- a level 0 reads no jerk (control);
- the fixes of the clinical review (task FIX-jerks): hyperekplexia habituates poorly; standing clips keep their
  centre of mass over the feet by the hip strategy and pass their balance block; lapses are ballistic within their
  gravity bounds, the knuckles lapse with the wrists inside the hand limits; the focal elbow jerk is weighted and
  every rotation keeps under the speed cap; the myoclonus-dystonia jerks take the room of the body before the
  posture; the generalized jerk is led by the axial flexion and spreads from the neck down; the segmental burst
  leaves the arm still half of the period; the stimulus-sensitive jerks follow the tapped hand and every stimulus
  has a mark; the clonus foot is loaded on its ball, ends without a step and drifts; the tic captions and YGTSS
  number grades are those of the types shown; the standing squat is a real squat;
- the touching tics (shared change SH15): seated and standing clips whose complex tics touch the face with the right
  index (reach.touch_path), read back from the skinned mesh: the touch kept between 0 and 3 mm from the skin, nothing
  inside the head, the trunk and thighs clear, the taps counted and of their depth, the fingertip at its peak speed,
  the elbow below the shoulder, the hand in its limits and at rest outside the touches, the rows outside the touches
  those of the simple-tic engine, the YGTSS number of the discrete tics shown (the same when every touch counts as one
  tic and every tap as one), the share of the family's complexity ladder with the grade capped at 3;
- the healthy shrug (task sh1c, limits_ext.SHRUG_DECISION): a shrug raises the shoulder joint with the girdle and
  the arm hangs under it (the upper arm keeps its global rotation), its room the collar's, read against the table
  'shrug' (the healthy table refuses the turned-back shoulder, the shrug table holds it); every clip with a shrug
  records that table, and no clip's upper arm turns at a knot where a shrug alone moves.
- wave SH1d part B (review M6, M2): standing, the arms hang through a trunk or hip jerk (the upper arm within 10 deg
  of the level 0's angle to the vertical); seated, the wrists stay over their supports (carry_arms on every builder);
  each jerk of a kind draws its own size (a CV of at least 0.15 per clip), the rhythmic ones less, the rises jittered
  under the speed cap.
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


@pytest.fixture(scope="module")
def JK():
    torch.set_num_threads(2)
    from mdfm.synth import jerks
    return jerks


@pytest.fixture(scope="module")
def specs(JK):
    return {s.clip_id: s for s in JK.specs()}


@pytest.fixture(scope="module")
def built(specs):
    return {cid: s.build() for cid, s in specs.items()}


@pytest.fixture(scope="module")
def l0_E2():
    from mdfm.synth.exam import exam_set
    s = next(x for x in exam_set() if x.clip_id == "tremor_E2_raise_g0")
    return s.build()[0]


def _base(cid, specs, built, l0_E2):
    l0 = specs[cid].level0
    return l0_E2 if l0 == "tremor_E2_raise_g0" else built[l0][0]


def _sites(truth):
    return sorted({p["site"] for e in truth["params"]["events"] for p in e["parts"]})


# ---------------------------------------------------------------- the engine
def test_the_pulse_rises_holds_and_returns_on_its_span(JK):
    t = np.arange(0, 2.0, 0.01)
    p = JK.pulse(t, 0.5, 0.08, 0.12, hold_s=0.3)
    assert p[t <= 0.5].max() == 0.0 and p[t >= 0.5 + 0.08 + 0.3 + 0.12 - 1e-9].max() == 0.0
    k = int(round(0.58 / 0.01))
    assert p[k] == pytest.approx(1.0, abs=1e-12) and np.all(p[k:k + 30] == pytest.approx(1.0, abs=1e-12))
    assert np.abs(np.diff(p)).max() < 0.25                       # smooth: no step larger than a quarter


def test_the_point_process_places_the_count_with_its_gaps(JK):
    rng = np.random.default_rng(3)
    on = JK.place_events(37, 0.5, 29.5, 0.25, rng)
    assert len(on) == 37 and on[0] >= 0.5 and on[-1] + 0.25 <= 29.5 + 1e-9
    assert np.diff(on).min() >= 0.25 + JK.MIN_GAP_S - 1e-9
    assert all(abs(x * 100 - round(x * 100)) < 1e-9 for x in on)
    with pytest.raises(ValueError):
        JK.place_events(200, 0.0, 10.0, 0.25, rng)


def test_burst_and_movement_durations_come_from_the_cited_bands(JK):
    for kind, ty in JK.JERK_TYPES.items():
        lo, hi = ty["band_ms"]
        b = 1000 * JK.burst_s(kind)
        assert lo - 5 <= b <= hi + 5, kind
        rise, fall = JK.timing(kind)
        assert fall == pytest.approx(round(JK.RETURN_RATIO * rise * 100) / 100)
        assert rise == pytest.approx(JK.burst_s(kind) if ty["negative"] else JK.burst_s(kind) + JK.TWITCH_S)
    assert JK.timing("cortical") == (0.08, 0.12)                 # burst 40 ms (25 to 50 ms, geometric centre)
    assert 50 <= 1000 * JK.burst_s("negative") <= 200            # the silent period of asterixis


def _one(JK, base, site, peak, kind="cortical", onset=1.0):
    ev = [{"onset_s": onset, "parts": [JK.jerk_part(site, peak, kind)]}]
    return JK.apply_events(base, ev)[0]


def test_sites_move_in_their_anatomical_directions(JK):
    from mdfm.synth import skeleton as SK
    from mdfm.synth.hands import hand_joint
    from mdfm.synth.base import static_body
    arms, _ = JK.outstretched_base(4.0)
    k = int(round(1.08 * 100))                                    # the end of the rise of a jerk at 1.0 s
    t = np.array([3.0, 3.08])                                      # on the hold: a jerk at 3.0 s peaks at 3.08 s
    mid = hand_joint("left", "middle", 0)
    for site, sgn in (("wrist_extension_left", 1.0), ("wrist_flexion_left", -1.0), ("arm_raise_left", 1.0),
                      ("arm_drop_left", -1.0)):
        o = _one(JK, arms, site, math.radians(15.0), onset=3.0)
        P = SK.joints_at(o, t, hands=True)
        assert sgn * (P[1, mid, 1] - P[0, mid, 1]) > 0.01, site
    sit, stand = static_body("sit", 2.0), static_body("stand", 2.0)
    J0 = stand.joints()[k]
    o = _one(JK, stand, "shrug_left", math.radians(10))
    J = o.joints()[k]
    assert J[16, 1] - J0[16, 1] > 0.005
    J = _one(JK, stand, "neck_flexion", math.radians(20)).joints()[k]
    assert J[15, 2] - J0[15, 2] > 0.005
    J = _one(JK, stand, "shoulder_abduction_left", math.radians(20)).joints()[k]
    assert J[20, 0] - J0[20, 0] > 0.02                            # the left wrist moves out to the body's left (+x)
    J0s = sit.joints()[k]
    J = _one(JK, sit, "knee_extension_left", math.radians(20)).joints()[k]
    assert J[7, 2] - J0s[7, 2] > 0.10 and J[7, 1] > J0s[7, 1]        # the foot kicks forward and up
    J = _one(JK, sit, "elbow_flexion_left", math.radians(20)).joints()[k]
    assert J[20, 1] - J0s[20, 1] > 0.02


def test_the_dip_and_the_pelvis_turn_keep_the_feet_still(JK):
    from mdfm.synth.base import static_body
    st = static_body("stand", 2.0)
    for site, peak in (("knee_dip", 0.04), ("hip_flexion", math.radians(15.0))):
        o = _one(JK, st, site, peak, kind="propriospinal")
        J0, J = st.joints(), o.joints()
        assert float((J[:, [7, 8, 10, 11]] - J0[:, [7, 8, 10, 11]]).abs().max()) < 1e-9, site
        assert bool(torch.equal(o.rot6d[:50], st.rot6d[:50])), site          # bit for bit before the jerk
    o = _one(JK, st, "knee_dip", 0.04, kind="propriospinal")
    drop = (st.joints()[:, 0, 1] - o.joints()[:, 0, 1]).numpy()
    assert drop.max() == pytest.approx(0.04, abs=1e-9)
    lean = lambda b: (b.joints()[:, 12] - b.joints()[:, 0]).numpy()   # noqa: E731  the trunk keeps its lean in the dip
    assert np.abs(lean(o) - lean(st)).max() < 1e-9
    from mdfm.synth.measure import dominant_frequency            # noqa: F401  (import check)
    knee = JK.deviation_deg(o, st, [4])[:, 0]
    k0 = int(np.flatnonzero(knee > 0)[0])
    assert knee[k0] < 2.0                                           # continuous at the onset: no jump of the knee


# ---------------------------------------------------------------- every clip
def test_the_catalogue_specs_are_unique_and_resolve(JK, specs):
    from mdfm.synth.exam import SEGMENTS, exam_set
    exam_ids = {s.clip_id for s in exam_set()}
    assert len(specs) >= 50
    for cid, s in specs.items():
        assert s.since == "catalogue_v1" and cid.startswith("jerk_")
        assert s.segment in SEGMENTS or s.segment_title, cid
        assert s.params_text and s.grade_text and s.views, cid
        assert s.level0 is None or s.level0 in specs or s.level0 in exam_ids, cid
        assert cid not in exam_ids
    fams = {s.family for s in specs.values()}
    assert {"myoclonus", "negative_myoclonus", "tics", "startle", "clonus", "reference"} <= fams


def test_every_clip_builds_finite_and_valid(specs, built):
    for cid, (b, t) in built.items():
        assert bool(torch.isfinite(b.rot6d).all()) and bool(torch.isfinite(b.transl).all()), cid
        v = t.get("validity")
        if v is None:
            assert t["operator"] == "none", cid
            continue
        assert v["limits_new"] == 0, (cid, v["limits_worst_deg"])
        assert v["skate_added_m_s"] < 1e-6, cid
        assert v["contact_lost_share"] == 0.0, cid
        assert abs(v["lowest_foot_m"] - v["lowest_foot_base_m"]) < 1e-9, cid
        for k in ("balance", "hand_limits", "forefoot", "face_touch"):   # the blocks the render script refuses on
            assert v.get(k) is None or v[k]["ok"], (cid, k, v.get(k))
        if specs[cid].segment == "E7" and t["operator"] != "dystonic_posture":
            assert v.get("balance") is not None, cid                    # every standing jerk clip is balanced


def test_standing_jerks_keep_the_centre_of_mass_by_the_hip_strategy(JK, built):
    """The trunk flexion of a standing propriospinal jerk pushes the hips back; the centre of mass stays over its base
    place, the extrapolated centre of mass inside the feet, the friction and the vertical ground force inside their
    bounds (the review measured 3.5 friction and the XcoM 14.5 cm past the toes before the fix)."""
    from mdfm.kernels.force import com
    b, t = built["jerk_myoclonus_E7_propriospinal_f2_a3"]
    base = built["jerk_E7_level0"][0]
    bal = t["validity"]["balance"]
    assert bal["ok"] and bal["com_moved_forward_cm"] < 0.01
    assert min(bal["xcom_margin_posterior_m"], bal["xcom_margin_anterior_m"]) >= JK.BALANCE_MARGIN_M
    assert bal["friction_max"] <= JK.FRICTION_MAX and bal["vertical_grf_min_bw"] >= JK.GRF_MIN_BW
    back = t["arrays"]["hip_strategy_back_m"]
    k = int(np.argmax(back))
    assert back[k] > 0.03 and base.joints()[k, 0, 2] - b.joints()[k, 0, 2] > 0.03     # the pelvis went back
    c0, c1 = com(base.joints().numpy(), 2), com(b.joints().numpy(), 2)
    assert np.abs(c1[:, 2] - c0[:, 2]).max() < 1e-4
    assert float((b.joints()[:, [7, 8, 10, 11]] - base.joints()[:, [7, 8, 10, 11]]).abs().max()) < 1e-9
    o, prof, back0 = JK.apply_events(base, t["params"]["events"])                 # without: the trunk pivots
    assert not back0.any() and float(np.abs(com(o.joints().numpy(), 2)[:, 2] - c0[:, 2]).max()) > 0.03


def test_level0_partners_share_the_time_base(specs, built, l0_E2):
    for cid, s in specs.items():
        if s.level0 is None:
            continue
        b0 = _base(cid, specs, built, l0_E2)
        b = built[cid][0]
        assert b.T == b0.T and b.rate == b0.rate, cid
        if s.level0 == "tremor_E2_raise_g0" and "asterixis" not in cid:
            # the outstretched base is the exam clip's body: the knots before the first event are its knots
            k = int(round(min(e["onset_s"] for e in built[cid][1]["params"]["events"]) * b.rate))
            assert torch.equal(b.rot6d[:k], b0.rot6d[:k]), cid


def test_the_build_is_deterministic_under_its_seed(JK, specs, built):
    for cid in ("jerk_myoclonus_E1_multifocal_f3_a2", "jerk_tics_E1_bouts_60s", "jerk_asterixis_E2_conn3_sev2"):
        b2, _ = specs[cid].build()
        assert torch.equal(b2.rot6d, built[cid][0].rot6d) and torch.equal(b2.transl, built[cid][0].transl)
    from mdfm.synth.base import static_body
    a, _ = JK.build_myoclonus(static_body("sit", 30.0), "cortical", [["neck_flexion"], ["shrug_left"]], 3, 2,
                              (0.5, 29.5), 1, "myoclonus", "UMRS", ())
    c, _ = JK.build_myoclonus(static_body("sit", 30.0), "cortical", [["neck_flexion"], ["shrug_left"]], 3, 2,
                              (0.5, 29.5), 2, "myoclonus", "UMRS", ())
    assert not torch.equal(a.rot6d, c.rot6d)


# ---------------------------------------------------------------- re-measurement
def _jerk_clips(specs):
    return [c for c, s in specs.items() if s.family in ("myoclonus", "negative_myoclonus", "tics")
            and "posture_only" not in c]


def test_jerks_read_back_count_onsets_and_durations(JK, specs, built, l0_E2):
    for cid in _jerk_clips(specs):
        b, t = built[cid]
        base = _base(cid, specs, built, l0_E2)
        ev = t["params"]["events"]
        sites = _sites(t)
        # a touching clip's reaches (minimum jerk over about half a second to a second) set a peak speed whose 1e-3
        # hides the first knot of the slower tics: read with rel 1e-4 (the 0.1 deg/s floor); the onset of a touch
        # within one knot (the first knot of a minimum-jerk reach can turn under the floor), every other onset exact
        touch = bool(t["params"].get("touch"))
        m = JK.measure_jerks(b, base, JK.site_joints(sites), pelvis=any(JK.SITES[s][1] == "dip" for s in sites),
                             rel=1e-4 if touch else 1e-3)
        span = np.array([JK.event_span(e) for e in ev])
        assert m["n"] == len(ev), cid
        # open-body port, wave 3 (D26): a seated tic clip's onsets within one knot as well, its durations within three
        # (the return's last knot too). A minimum-jerk rise of n knots turns its first knot at about 5.3 / n^2 of its
        # peak speed; on the anatomical table the shrug's collar has jarm2022's 20 deg (40 to 61 on the AMASS
        # table), the seated trunk 25 deg of flexion, so the slower tics' first knot falls under the read's floor
        # next to the clip's fastest jerk (read late by 0.01 s at rel 1e-3, the tonic trunk tension's at 1e-4 as
        # well); the myoclonus and the lapses stay exact
        # wave 8 second part (D159): the standing tics as well (jerk_tics_E7_complexity_c4: a trunk jerk, two shrugs
        # and an orchestrated tic read 0.01 s late, the shrug's duration 0.03 s short): the same minimum-jerk first
        # knot under the floor of the clip's fastest jerk, on the anatomical table's collar and trunk rooms
        tic = specs[cid].family == "tics"
        tol = np.array([0.0101 if e["label"].startswith("complex:t") or tic else 1e-9 for e in ev])
        assert np.all(np.abs(np.asarray(m["onset_s"]) - span[:, 0]) <= tol), (cid, m["onset_s"], span[:, 0])
        np.testing.assert_allclose(m["duration_s"], span[:, 1] - span[:, 0], atol=0.031 if tic else 0.021,
                                   err_msg=cid)     # a seated tic: its first and last knot under the floor


def test_myoclonus_counts_and_amplitudes_match_their_umrs_grades(JK, specs, built):
    for cid, s in specs.items():
        if s.family != "myoclonus" or "posture_only" in cid:
            continue
        b, t = built[cid]
        p = t["params"]
        g = s.grade
        if p["process"] in ("poisson", "action"):
            assert JK.grade_of_count(p["count_per_30s"]) == g["frequency"], cid
        share = JK.UMRS_AMPLITUDE_SHARE[g["amplitude"]]
        for site, deg in p["peaks_deg"].items():
            w = (p.get("weights") or {}).get(site, 1.0)
            if JK.SITES[site][1] in ("parent", "world") and site in p["sites"]:
                # each event's part: the site's peak times the event's size factor (wave SH1d part B), under the cap
                parts = [(q, e["size_factor"]) for e in p["events"] for q in e["parts"] if q["site"] == site]
                assert p["sites"][site]["peak"] == pytest.approx(max(abs(math.degrees(q["peak"])) for q, _ in parts),
                                                                 rel=1e-9), (cid, site)
                assert all(abs(math.degrees(q["peak"])) <= deg * f * (1 + 1e-9) for q, f in parts), (cid, site)
                k = int(round(p["window_s"][0] * 100))
                ref = _base(cid, specs, built, None) if s.level0 != "tremor_E2_raise_g0" else None
                if ref is not None and p.get("room_cap_share"):          # the room of the body before the posture
                    pre = built[specs[s.level0].level0][0]
                    want = share * w * sum(JK.site_rooms(pre, site, k))
                    cap = p["room_cap_share"] * sum(JK.site_rooms(ref, site, k))
                    assert deg == pytest.approx(min(want, cap), rel=1e-6), (cid, site)
                    assert p["peak_rule"][site]["capped"] == (want > cap), (cid, site)
                elif ref is not None:
                    room = sum(JK.site_rooms(ref, site, k))
                    assert deg == pytest.approx(math.degrees(share * w * math.radians(room)), rel=1e-6), (cid, site)
        rise, fall = JK.timing(p["jerk_type"])
        if p["timing"] == "the type's":
            assert p["rise_s"] + p["fall_s"] == pytest.approx(rise + fall)
        for e in p["events"]:                                             # the speed cap, the rise of each part
            for q in e["parts"]:
                if JK.SITES[q["site"]][1] in ("parent", "world"):
                    assert q["rise_s"] >= (1.0 - JK.RISE_JITTER) * (p["rise_s"] if p["timing"] != "the type's"
                                                                    else rise) - 0.011, cid
                    v = JK.MJ_PEAK_SPEED * math.degrees(abs(q["peak"])) / q["rise_s"]
                    assert v <= JK.PEAK_SPEED_CAP_DEG_S + 1e-6, (cid, q["site"], v)


def test_segmental_myoclonus_is_rhythmic_at_its_cited_rate(JK, specs, built):
    for cid in [c for c in specs if "segmental" in c]:
        b, t = built[cid]
        m = JK.measure_jerks(b, built["jerk_E1_level0"][0], JK.site_joints(_sites(t)))
        f = 1.0 / np.median(np.diff(m["onset_s"]))
        assert JK.SEGMENTAL_HZ[0] <= f <= JK.SEGMENTAL_HZ[1], cid
        assert f == pytest.approx(1.0 / t["params"]["period_s"], abs=0.05)


def test_stimulus_sensitive_jerks_are_locked_to_the_stated_stimuli(JK, specs, built, l0_E2):
    for cid in [c for c in specs if "stimulus" in c]:
        b, t = built[cid]
        m = JK.measure_jerks(b, l0_E2, JK.site_joints(_sites(t)))
        st = np.array(t["params"]["stimuli_s"])
        assert m["n"] == len(st)
        np.testing.assert_allclose(m["onset_s"] - st, JK.STIMULUS_LATENCY_S, atol=1e-9)


def test_asterixis_lapses_match_conn_grade_depth_and_silent_period(JK, specs, built):
    base = built["jerk_asterixis_E2_level0"][0]
    for cid in [c for c in specs if c.startswith("jerk_asterixis_E2_conn")]:
        b, t = built[cid]
        g = specs[cid].grade
        wrists = [20, 21]
        m = JK.measure_jerks(b, base, JK.site_joints(_sites(t)))
        win = t["params"]["window_s"]
        assert JK.grade_of_count(m["n"] * 30.0 / (win[1] - win[0])) == g["conn"], cid
        dev = JK.deviation_deg(b, base, wrists)
        depth = JK.NEGATIVE_DEPTH_DEG[g["umrs_negative_severity"]]
        for e in t["params"]["events"]:                  # each lapse at the grade's depth times its size factor
            a_, c_ = JK.event_span(e)
            got = dev[int(round(a_ * 100)):int(round(c_ * 100)) + 1].max()
            want = max(abs(math.degrees(q["peak"])) for q in e["parts"])
            assert got == pytest.approx(want, abs=1e-6), cid
            assert want <= depth * e["size_factor"] * (1 + 1e-9), cid
        k = np.flatnonzero(dev.max(1) > 0)
        first = k[0] - 1
        x = dev.max(1)[first:first + 40]
        peak = int(np.argmax(x))
        lp = t["params"]["lapse"][next(s for s in t["params"]["lapse"] if s.startswith("wrist"))]
        assert lp["drop_s"] == pytest.approx(JK.burst_s("negative"))          # the drop lasts the silent period
        assert peak / 100.0 == pytest.approx(lp["drop_s"] + lp["catch_s"], abs=0.011), cid   # then the catch
        acc = np.diff(x, 2) * 1e4                                             # deg/s2 on the knots
        n1 = int(round(lp["drop_s"] * 100))
        assert (acc[:n1 - 1] > 0).all() and (acc[n1 + 1:peak - 1] < 0).all(), cid    # ballistic, then caught
        lim, catch = JK.LAPSE_ACCEL_DEG_S2["wrist"]
        assert np.abs(acc[:peak]).max() <= max(lim, catch) * 1.05, cid       # within the gravity bound of a hand
        assert lp["drop_acc"] <= lim + 1e-6
        fl = t["params"]["finger_lapse"]                                       # the knuckles lapse with the wrist
        assert fl["hand_limits"]["ok"] and len(fl["knuckles"]) == 4 * sum(
            1 for e in t["params"]["events"] for q in e["parts"] if q["site"].startswith("wrist"))
        lo = 0.67 * JK.FINGER_LAPSE_SHARE * JK.SIZE_CLIP[0] * min(JK.NEGATIVE_DEPTH_DEG.values())
        assert all(lo - 1e-9 <= r["deg"] <= 1.33 * JK.FINGER_LAPSE_MAX_DEG + 1e-9 for r in fl["knuckles"]), cid
        assert not torch.equal(b.hand6d, base.hand6d)
    lv = JK.measure_jerks(base, built["jerk_asterixis_E2_level0"][0], [20, 21])
    assert lv["n"] == 0                                                 # control: the level 0 reads no lapse


def test_leg_lapses_are_dips_of_the_graded_depth_with_the_feet_still(JK, specs, built):
    base = built["jerk_E7_level0"][0]
    for cid in [c for c in specs if "leg_lapses" in c]:
        b, t = built[cid]
        g = specs[cid].grade
        d = JK.measure_dips(b, base)
        assert d["n"] == t["params"]["n_events"] and JK.grade_of_count(d["n"]) == g["frequency"], cid
        np.testing.assert_allclose(d["depth_cm"], JK.LAPSE_DEPTH_CM[g["umrs_negative_severity"]] * np.asarray(
            [e["size_factor"] for e in t["params"]["events"]]), atol=1e-6)      # each lapse its own size
        assert float((b.joints()[:, [7, 8, 10, 11]] - base.joints()[:, [7, 8, 10, 11]]).abs().max()) < 1e-9
        # no faster than gravity: the pelvis falls at most at LAPSE_ACCEL_MAX and rises at most at DIP_ACCEL_MAX
        y = (base.joints()[:, 0, 1] - b.joints()[:, 0, 1]).numpy()       # the drop, m
        a = np.diff(y, 2) * 1e4
        lp = t["params"]["lapse"]["knee_dip"]
        assert lp["drop_s"] == pytest.approx(JK.burst_s("negative")) and lp["drop_acc"] <= JK.LAPSE_ACCEL_MAX
        assert a.max() <= JK.LAPSE_ACCEL_MAX * 1.02 and -a.min() <= max(JK.LAPSE_CATCH_ACCEL, JK.DIP_ACCEL_MAX) * 1.3
        bal = t["validity"]["balance"]
        assert bal["ok"] and bal["vertical_grf_min_bw"] >= 0.5          # the floor never pulls the body down


def test_tics_match_their_mrvs_frequency_areas_and_ygtss_number(JK, specs, built):
    for cid in [c for c in specs if specs[c].family == "tics"]:
        b, t = built[cid]
        s = specs[cid]
        p = t["params"]
        base = built[s.level0][0]
        clip_s = b.duration
        m = JK.measure_jerks(b, base, JK.site_joints(_sites(t)), pelvis="E7" in cid)
        assert JK.mrvs_frequency_grade(m["n"] * 60.0 / clip_s) == s.grade["mrvs_frequency"], cid
        # the arms the trunk carries or lets hang (arms_rule, wave SH1d part B) are passive, not a tic's area: the
        # areas are read on the events without that rule
        areas, _ = JK.tic_areas(JK.apply_events(base, p["events"], "E7" in cid)[0], base, standing="E7" in cid)
        assert areas == p["areas"] and JK.mrvs_area_grade(len(areas)) == p["mrvs_area_grade"], cid
        labels = {e["label"] for e in p["events"]}
        simple = {x for x in labels if not x.startswith("complex")}
        assert simple <= set(p["tic_types"]), cid
        if "number" in cid:
            assert len(simple) == JK.YGTSS_NUMBER_TYPES[s.grade["ygtss_number"]], cid
        if p.get("touch"):                              # the ladder's share, the grade capped (touching tics)
            assert p["n_complex"] == round(p["complexity_share"] * p["count"]), cid
            assert s.grade["ygtss_complexity"] <= JK.TOUCH_COMPLEXITY_MAX, cid
        elif s.grade.get("ygtss_complexity"):
            assert p["n_complex"] == round(JK.YGTSS_COMPLEX_SHARE[s.grade["ygtss_complexity"]] * p["count"]), cid
        # the reference grade is the one of the types shown (the review found 3 for 2 and 6 for 3 before the fix);
        # a touching tic is a discrete complex tic, counted with the simple ones (no orchestrated pattern)
        if p.get("touch"):
            assert t["grade"]["ygtss_number"] == s.grade["ygtss_number"] == JK.ygtss_number_grade(
                len(simple) + len(labels - simple)), cid
            lumped = len(simple) + len({x.split("_")[0] for x in labels - simple})       # one 'touching', one 'tapping'
            assert p["ygtss_number_lumped"] == JK.ygtss_number_grade(lumped) == s.grade["ygtss_number"], cid
        else:
            assert t["grade"]["ygtss_number"] == s.grade["ygtss_number"] == JK.ygtss_number_grade(
                len(simple), p["n_complex"], "E7" in cid), cid
        assert 0 <= t["grade"]["ygtss_number"] <= 5
        if s.params_text[0].isdigit() and "tic types" in s.params_text:   # the caption counts the types shown
            assert int(s.params_text.split()[0]) == len(p["tic_types_shown"]) == len(simple), cid


def test_the_standing_squat_tic_is_a_squat(JK, built):
    """The complex squat of the standing tic clip: the pelvis SQUAT_CM (12 cm) down, knees near 60 deg, lowered over
    at least 0.4 s, held, then up; the hips back by the hip strategy, the feet still (the review saw a 4 cm bob)."""
    b, t = built["jerk_tics_E7_complexity_c4"]
    base = built["jerk_E7_level0"][0]
    sq = [e for e in t["params"]["events"] if e["label"] == "complex:squat"]
    assert sq
    for e in sq:
        dip = next(q for q in e["parts"] if q["site"] == "knee_dip")
        assert dip["peak"] == pytest.approx(JK.SQUAT_CM / 100.0) and dip["rise_s"] >= 0.4 and dip["hold_s"] >= 0.3
        k = int(round((e["onset_s"] + dip["rise_s"] + 0.5 * dip["hold_s"]) * 100))
        knee = JK.deviation_deg(b, base, [4, 5])[k]
        assert (knee > 50.0).all() and (knee < 80.0).all(), knee
        assert base.joints()[k, 0, 2] - b.joints()[k, 0, 2] > 0.05                    # the hips back
    assert float((b.joints()[:, [7, 8, 10, 11]] - base.joints()[:, [7, 8, 10, 11]]).abs().max()) < 1e-9


def test_tic_bouts_wax_and_wane(JK, specs, built):
    b, t = built["jerk_tics_E1_bouts_60s"]
    p = t["params"]
    bouts = p["bouts_s"]
    for e in p["events"]:
        a, c = JK.event_span(e)
        assert any(lo - 1e-9 <= a and c <= hi + JK.timing("tic")[0] + JK.timing("tic")[1] + 1e-9 for lo, hi in bouts)
    fac = np.array([e["factor"] for e in p["events"]])
    assert fac.min() < 0.8 and fac.max() > 0.95                      # small at a bout's edges, full at its middle


def test_the_startle_latency_habituation_and_hyperekplexic_stiffness(JK, specs, built):
    for seg in ("E7", "E1"):                                             # standing, and seated (the seated normal new)
        res = {}
        base = built[f"jerk_{seg}_level0"][0]                            # read against the stimulus-free body
        for cond in ("normal", "hyperekplexia"):
            cid = f"jerk_startle_{seg}_{cond}"
            b, t = built[cid]
            m = JK.measure_jerks(b, base, JK.site_joints(_sites(t)), pelvis=True)
            st = np.array(t["params"]["stimuli_s"])
            assert m["n"] == len(st), cid
            np.testing.assert_allclose(m["onset_s"] - st, round(JK.STARTLE_SCM_MS / 10) / 100, atol=1e-9)
            dev = JK.deviation_deg(b, base, [12])[:, 0]
            res[cond] = np.array([dev[int(round(x * 100)):int(round(x * 100)) + 200].max() for x in st])
            k = int(round((st[0] + 0.06 + 0.13) * 100))
            held = dev[k:k + int(JK.STIFFNESS_S * 100)]
            if cond == "hyperekplexia":
                first = dev[int(round(st[0] * 100)):int(round(st[0] * 100)) + 200].max()
                assert held.min() == pytest.approx(first, rel=1e-9), cid       # the stiffness: the posture held
            else:
                assert dev[k + 30] < 0.05 * dev.max(), cid                     # the normal startle returns at once
        n, h = res["normal"], res["hyperekplexia"]
        fn = np.array([e["factor"] for e in built[f"jerk_startle_{seg}_normal"][1]["params"]["events"]])
        fh = np.array([e["factor"] for e in built[f"jerk_startle_{seg}_hyperekplexia"][1]["params"]["events"]])
        np.testing.assert_allclose(n / n[0], fn / fn[0], rtol=1e-6)         # habituation over the three trials,
        assert h[0] > 2.0 * n[0], seg                                       # times each trial's size factor
        np.testing.assert_allclose(h / h[0], fh / fh[0], rtol=1e-6)         # hyperekplexia habituates poorly
        assert h[-1] / h[0] >= 0.75 > 2 * n[-1] / n[0], seg
    for cid in (JK.startle_id(c, p) for c, p in JK.STARTLE_CLIPS):
        t = built[cid][1]
        assert [e["factor"] / e["size_factor"] for e in t["params"]["events"]] == pytest.approx(
            t["params"]["habituation"]), cid
        marks = t["params"]["stimulus_marks"]                              # each sound has its mark for the viewer
        assert [m["t_s"] for m in marks] == t["params"]["stimuli_s"] and {m["text"] for m in marks} == {"SOUND"}


def test_the_hyperekplexia_is_paired_with_the_normal_startle_to_the_same_sounds(JK, specs, built):
    """STARTLE_PAIRING: each hyperekplexia (standing and seated) has for level 0 the normal startle of its posture,
    built on the same base under the same seed, to the same sounds at the same times with the same marks; each normal
    startle has the stimulus-free body for level 0, so that the series root (physio.series_root) stays the neutral
    body; the two bodies agree bit for bit before the first response and differ by the exaggeration at every trial."""
    from mdfm.synth.physio import series_root
    assert [JK.startle_id(c, p) for c, p in JK.STARTLE_CLIPS] == [
        "jerk_startle_E7_normal", "jerk_startle_E7_hyperekplexia", "jerk_startle_E1_normal",
        "jerk_startle_E1_hyperekplexia"]
    for seg, posture, scene in (("E7", "stand", None), ("E1", "sit", "chair")):
        hyp, nor, neutral = (f"jerk_startle_{seg}_hyperekplexia", f"jerk_startle_{seg}_normal", f"jerk_{seg}_level0")
        assert specs[hyp].level0 == nor and specs[nor].level0 == neutral and specs[neutral].level0 is None
        assert specs[nor].segment == specs[hyp].segment == seg and specs[nor].scene == specs[hyp].scene == scene
        assert specs[nor].views == specs[hyp].views and specs[nor].params_text == specs[hyp].params_text
        assert specs[hyp].note == JK.STARTLE_PAIRING and specs[nor].note == ""
        assert series_root(hyp, specs) == series_root(nor, specs) == neutral
        (bh, th), (bn, tn), (b0, _) = built[hyp], built[nor], built[neutral]
        assert th["params"]["stimuli_s"] == tn["params"]["stimuli_s"] == list(JK.STIMULI_S)
        assert th["params"]["stimulus_marks"] == tn["params"]["stimulus_marks"]
        k1 = min(e["onset_s"] for e in th["params"]["events"])
        assert k1 == min(e["onset_s"] for e in tn["params"]["events"])
        k1 = int(round(k1 * bh.rate))
        for b in (bh, bn):                                       # nothing before the first response
            assert torch.equal(b.rot6d[:k1], b0.rot6d[:k1]) and torch.equal(b.transl[:k1], b0.transl[:k1])
        sign = JK.deviation_deg(bh, bn, [12])[:, 0]              # the exaggeration over the normal startle
        own = JK.deviation_deg(bn, b0, [12])[:, 0]
        for x in JK.STIMULI_S:
            w = slice(int(round(x * 100)), int(round(x * 100)) + 200)
            assert sign[w].max() > own[w].max() > 0.5, (seg, x)


def test_the_seated_normal_startle_is_a_valid_seated_body(JK, built):
    """The new seated normal startle (jerk_startle_E1_normal): no new joint-limit violation, no skate, no contact
    lost, no knee dip and the hips, legs and feet of the seated base (the seat holds the pelvis), no balance block (a
    seated body); the trunk and neck flex, the shoulders rise at each sound."""
    b, t = built["jerk_startle_E1_normal"]
    b0 = built["jerk_E1_level0"][0]
    v = t["validity"]
    assert v["limits_new"] == 0 and v["skate_added_m_s"] < 1e-6 and v["contact_lost_share"] == 0.0
    assert v.get("balance") is None and not t["params"]["standing"] and "knee_dip" not in t["params"]["peaks_deg"]
    legs = [0, 1, 2, 4, 5, 7, 8, 10, 11]
    assert float((b.joints()[:, legs] - b0.joints()[:, legs]).abs().max()) < 1e-9
    assert float((b.transl - b0.transl).abs().max()) < 1e-12
    for s in ("neck_flexion", "trunk_flexion", "shrug_left", "shrug_right"):
        assert t["params"]["peaks_deg"][s] > 0.5, s


def test_ankle_clonus_beats_frequency_and_scats_grade(JK, specs, built):
    base = JK.held_leg_base(20.0)
    for g in (0, 1, 2, 3):
        b, t = built[f"jerk_clonus_J2_scats{g}"]
        m = JK.measure_clonus(b, base)
        assert m["scats"] == g
        assert m["beats"] == t["params"]["beats_planted"]
        if g:
            assert m["frequency_hz"] == pytest.approx(JK.CLONUS_HZ, abs=0.1)
            assert m["pp_deg"] == pytest.approx(JK.CLONUS_PP_DEG, rel=JK.CLONUS_AMP_DRIFT + 0.03)   # it drifts
            assert m["first_beat_s"] == pytest.approx(t["params"]["first_beat_s"], abs=1e-9)
        assert bool(b.contact.all())                         # both feet on the floor: the ball of the loaded one
        assert t["validity"]["forefoot"]["ok"] and t["params"]["ball_moved_m"] < 1e-9


def test_the_clonus_foot_is_loaded_on_its_ball_and_the_beats_end_smoothly(JK, built):
    """Catalogue row O01, seated with the forefoot loaded: the ball of the foot on the floor, the heel 3 to 5 cm up,
    the knee 90 to 100 deg; the beats lift the heel and the knee; no step at the end of a sustained clonus (the
    review found a 4.65 deg jump in one knot at 19.5 s); the amplitude and the frequency drift."""
    base = JK.held_leg_base(20.0)
    sit = JK._sit_body(base.T)
    h0 = JK.heel_height_m(base, "left", [0])[0] - JK.heel_height_m(sit, "left", [0])[0]
    assert h0 == pytest.approx(JK.HEEL_UP_M, abs=1e-3)
    knee = JK.deviation_deg(base, sit, [4])[0, 0] + 90.0                      # 'sit' has the knee at 90 deg
    assert 90.0 <= knee <= 100.0
    for g in (1, 2, 3):
        b, t = built[f"jerk_clonus_J2_scats{g}"]
        p = t["params"]
        assert 3.0 <= p["heel_height_cm_range"][0] and p["heel_height_cm_range"][1] <= 6.5, p["heel_height_cm_range"]
        x = JK.signed_angle_deg(b, base, 7, "x") * -1.0
        step = np.abs(np.diff(x))
        k_on = int(round(p["beats_onset_s"] * 100))
        assert step[k_on + 20:].max() <= step[k_on + 20:k_on + 200].max() * 1.3 + 1e-9, g
        assert abs(x[-1] - p["stretch_deg"]) < 1e-6                           # back at the loaded angle
        kz = b.joints()[:, 4, 1].numpy()
        assert kz.max() - kz.min() > 0.003                                    # the knee bobs with the heel
        assert abs(p["drift"]["frequency_drift_hz"]) == JK.CLONUS_HZ_DRIFT
        assert abs(p["drift"]["amplitude_drift"]) == JK.CLONUS_AMP_DRIFT


def test_myoclonus_dystonia_jerks_ride_on_the_held_posture(JK, specs, built):
    for cid in ("jerk_mdyst_E7_neck_f3_a2", "jerk_mdyst_E1_right_arm_f3_a2"):
        b, t = built[cid]
        post = built[specs[cid].level0][0]
        l0 = built[specs[specs[cid].level0].level0][0]
        m = JK.measure_jerks(b, post, JK.site_joints(_sites(t)))
        assert m["n"] == t["params"]["n_events"] == 11
        held = JK.deviation_deg(post, l0, [15] if "neck" in cid else [19])[:, 0]
        assert held.min() > 10.0                                         # the posture is there throughout
        assert t["params"]["burst_ms"] == pytest.approx(100.0)          # the cited mean of 95 ms on a knot
        # the jerks take the room of the body before the posture (the review saw 5.3 deg of neck rotation). Open-body
        # port, wave 3 (D26): 8 deg read the AMASS rooms; on the anatomical table the a2 share (0.12) of jarm2022's
        # cervical rotation (60 deg) and extension (50) gives 7.2 and 6.0 deg, above the review's 5.3
        pk = t["params"]["peaks_deg"]
        for s in ("neck_rotation_left", "neck_rotation_right", "neck_flexion", "neck_extension"):
            if s in pk:
                assert pk[s] > 5.3, (cid, s, pk[s])
        assert t["params"]["room_cap_share"] == JK.ROOM_CAP_SHARE
    neck = built["jerk_mdyst_E7_neck_f3_a2"][1]["params"]
    assert {"neck_rotation_right", "neck_flexion", "neck_extension"} <= set(neck["peaks_deg"])   # against, across


def test_the_review_fixes_of_the_arm_jerks(JK, specs, built, l0_E2):
    """Focal elbow jerk weighted and paired with the arm raise (the hand up and in, 66.5 deg in 80 ms before the
    fix); the generalized jerk led by the neck and trunk flexion, spreading from the neck down; the segmental arm still
    for about half of each period; the stimulus-sensitive jerk on the tapped hand, every tap marked."""
    p = built["jerk_myoclonus_E2_cortical_focal_f3_a4"][1]["params"]
    assert p["peaks_deg"]["elbow_flexion_left"] < 30.0 and ["elbow_flexion_left", "arm_raise_left"] in p["repertoire"]
    for e in p["events"]:
        if e["label"] == "elbow_flexion_left+arm_raise_left":
            b = built["jerk_myoclonus_E2_cortical_focal_f3_a4"][0]
            k1 = int(round((e["onset_s"] + 0.08) * 100))
            d = (b.joints()[k1, 20] - l0_E2.joints()[k1, 20]).numpy()
            assert d[1] > 0.02 and d[0] < 0.0, d                         # the left wrist goes up and toward the midline
            break
    g = built["jerk_myoclonus_E1_generalized_f3_a2"][1]["params"]
    pk = g["peaks_deg"]
    assert pk["neck_flexion"] > pk["shoulder_abduction_left"] and pk["trunk_flexion"] > pk["shoulder_abduction_left"]
    parts = {q["site"]: q["delay_s"] for q in g["events"][0]["parts"]}
    assert parts["neck_flexion"] == 0.0 < parts["shoulder_abduction_left"] < parts["trunk_flexion"] < parts["hip_flexion"]
    s = built["jerk_myoclonus_E1_segmental_right_arm_a3"][1]
    prof = s["arrays"]["profile_shoulder_abduction_right"]
    k = slice(100, 2900)
    assert 0.40 <= float((np.abs(np.gradient(prof[k])) > 1e-9).mean()) <= 0.55       # moving about half of the time
    for ag in (2, 3):
        t = built[f"jerk_myoclonus_E2_stimulus_a{ag}"][1]
        p = t["params"]
        assert t["grade"] == specs[f"jerk_myoclonus_E2_stimulus_a{ag}"].grade
        for e, side, m in zip(p["events"], p["stimulus_sides"], p["stimulus_marks"]):
            assert e["label"].endswith(side) and m["text"] == f"TAP {side} hand" and m["t_s"] == e["stimulus_s"]


# ---------------------------------------------------------------- touching tics (shared change SH15)
TOUCH_CLIPS = ("jerk_tics_E1_touch_c2", "jerk_tics_E1_touch_c3", "jerk_tics_E7_touch_c2")


def _touch_events(t):
    return [e for e in t["params"]["events"] if e["label"].startswith("complex")]


def test_touching_tics_touch_the_face_without_entering_it(JK, specs, built):
    """Every touching tic read from the skinned mesh: on the skin the hand lies 0 to 3 mm from the head surface (the
    gap aimed at 1.5 mm), nothing of the hand or forearm enters the head on the way, the trunk and the thighs keep
    1 cm (or their distance at rest), the elbow lies 7 cm or more below the shoulder (no salute), the fingertip peaks
    at the posture's speed and the arm turns under the touch's cap."""
    from mdfm.synth import reach as RC
    for cid in TOUCH_CLIPS:
        b, t = built[cid]
        speed = RC.TOUCH_TIP_SPEED_M_S["stand" if "E7" in cid else "sit"]
        rb = t["params"]["touch_readback"]
        assert rb["ok"] and t["validity"]["face_touch"]["ok"], cid
        assert rb["n_touches"] == len(_touch_events(t)) == t["params"]["n_complex"] > 0, cid
        for r in rb["events"]:
            assert 0.0 <= r["gap_mm_min"] <= r["gap_mm_max"] <= 1000 * RC.SKIN_GAP_MAX_M, (cid, r)
            assert abs(r["gap_mm_min"] - 1000 * RC.SKIN_GAP_M) < 1000 * RC.SKIN_GAP_TOL_M + 0.05, (cid, r)
            assert r["head_min_mm"] >= 0.0 and r["knots_on_skin"] >= 25, (cid, r)
            assert min(r["trunk_min_mm"], r["thighs_min_mm"]) >= 1000 * RC.CLEAR_M, (cid, r)
            assert 0.9 * speed < r["tip_peak_speed_m_s"] <= 1.01 * speed, (cid, r)
            assert r["joint_peak_speed_deg_s"] <= RC.TOUCH_JOINT_SPEED_CAP_DEG_S, (cid, r)
            assert r["elbow_drop_cm"] >= 100 * RC.TOUCH_ELBOW_DROP_M, (cid, r)
        assert t["validity"]["hand_limits"]["ok"] and t["validity"]["limits_new"] == 0, cid
        if "E7" in cid:
            assert t["validity"]["balance"]["ok"], cid


def test_touching_tics_tap_and_hold_as_planned(JK, built):
    """A single touch holds TOUCH_HOLD_S on the skin; a repeated tap lifts the fingertip TAP_N times by TAP_DEPTH_M
    (read from the fingertip vertex); the reach lasts in proportion to the path's length at the posture's fingertip
    speed, on the knot at or above the path check's timing, longer from the hanging arm than from the chair arm."""
    from mdfm.synth import reach as RC
    for cid in TOUCH_CLIPS:
        b, t = built[cid]
        for r, e in zip(t["params"]["touch_readback"]["events"], _touch_events(t)):
            main = e["parts"][0]
            assert r["label"] == e["label"] and r["taps_read"] == r["taps_planned"] == len(e["parts"]) - 1, (cid, r)
            if e["label"].startswith("complex:tap"):
                assert r["taps_read"] == JK.TAP_N
                assert all(abs(x - 100 * JK.TAP_DEPTH_M) < 0.1 for x in r["tap_depth_cm"]), (cid, r)
            else:
                assert main["hold_s"] == JK.TOUCH_HOLD_S and r["knots_on_skin"] >= 30, (cid, r)
            path = RC.touch_path("right", JK.SITES[main["site"]][2], main["posture"])
            assert main["path_length_m"] == path["length_m"]
            assert (main["rise_s"], main["fall_s"]) == JK.touch_timing(path)
            speed = RC.TOUCH_TIP_SPEED_M_S[main["posture"]]
            assert main["rise_s"] >= JK.MJ_PEAK_SPEED * main["path_length_m"] / speed - 1e-9
            assert main["rise_s"] >= path["check"]["rise_s"] - 1e-9 and main["rise_s"] - path["check"]["rise_s"] < 0.01
    seated = built["jerk_tics_E1_touch_c2"][1]["params"]["touch"]["timing_s"]["nose"][0]
    standing = built["jerk_tics_E7_touch_c2"][1]["params"]["touch"]["timing_s"]["nose"][0]
    assert standing > seated


def test_touching_tics_leave_the_rest_to_the_engine(JK, specs, built):
    """Outside the touches the hand is at rest and, seated, every row is that of the simple tics alone; inside, the
    trunk, neck and head hold the posture the touch is solved from; the touch reads in the MRVS arms and hands areas
    (the collar held); the caption counts the types shown and the touching kinds."""
    from mdfm.body.rotations import matrix_to_rot6d
    from mdfm.synth import hands as HD
    for cid in TOUCH_CLIPS:
        b, t = built[cid]
        p = t["params"]
        standing = "E7" in cid
        assert isinstance(b, HD.HandBody)
        base = HD.hands_of(JK._static("stand" if standing else "sit"))
        o, _, _ = JK.apply_events(base, [e for e in p["events"] if not e["label"].startswith("complex")], standing,
                                  True)
        act = np.zeros(b.T, bool)
        for e in _touch_events(t):
            a, c = JK.event_span(e)
            act[int(round(a * 100)):int(round(c * 100)) + 1] = True
        off, on = torch.as_tensor(~act), torch.as_tensor(act)
        assert float((b.hand6d[off] - matrix_to_rot6d(torch.eye(3, dtype=torch.float64))).abs().max()) < 1e-12, cid
        if not standing:                                  # standing, the hip strategy acts on the whole clip
            assert torch.equal(b.rot6d[off], o.rot6d[off]), cid
        R0 = base.local_R()[0, [0, 3, 6, 9, 12, 15]]
        assert float((b.local_R()[on][:, [0, 3, 6, 9, 12, 15]] - R0).abs().max()) < 1e-12, cid
        assert JK.touch_posture(b, int(np.flatnonzero(act)[0])) == ("stand" if standing else "sit")
        assert {"arms", "hands"} <= set(p["areas"]), cid
        assert all(set(x["areas"]) == {"arms", "hands"} for x in p["touch"]["paths"].values()), cid
        assert p["frequent_orchestrated"] is False
        assert specs[cid].grade["ygtss_complexity"] == t["grade"]["ygtss_complexity"] == min(
            JK.TOUCH_COMPLEXITY_MAX, next(k for k, v in JK.YGTSS_COMPLEX_SHARE.items() if v == p["complexity_share"]))
        assert cid.endswith(f"_c{specs[cid].grade['ygtss_complexity']}") and specs[cid].views[1] == "right", cid
        assert specs[cid].params_text.startswith(f"{len(p['tic_types_shown'])} simple tic types and "
                                                 f"{len(p['touch']['kinds'])} touching tics"), cid
        assert set(p["touch"]["labels_shown"]) == {JK.touch_label(k) for k in p["touch"]["kinds"]}, cid


def test_a_touch_needs_a_known_rest(JK):
    """A touch that starts away from the rest of 'sit' or 'stand' is refused, and a tap stays on the approach line."""
    from mdfm.synth import reach as RC
    from mdfm.synth.base import static_body
    with pytest.raises(ValueError):
        JK.touch_posture(static_body("sit_arms_forward", 1.0), 0)
    assert JK.touch_posture(static_body("sit", 1.0), 0) == "sit"
    assert JK.TAP_DEPTH_M < RC.TOUCH_APPROACH_M


# ---------------------------------------------------------------- the healthy shrug (task sh1c)
@pytest.mark.parametrize("side,shoulder", [("left", 16), ("right", 17)])
def test_a_shrug_raises_the_shoulder_with_the_arm_hanging(JK, side, shoulder):
    from mdfm.body.rotations import geodesic
    from mdfm.synth import limits_ext as LX
    from mdfm.synth.base import static_body
    from mdfm.synth.body import room_deg
    stand = static_body("stand", 2.0)
    site = f"shrug_{side}"
    assert JK.KEEP_CHILD[site] == shoulder
    # open-body port, wave 3 (D26): the static stand's shoulders hold 80 deg of adduction from the T pose and the
    # anatomical table's bound is the hanging arm, 90 (the AMASS bound was 80.99 on the left), so the shrug must pass
    # 10 deg for the healthy table to refuse the hanging arm under it: 15 deg, inside the collar's 20 deg elevation
    o = _one(JK, stand, site, math.radians(15))
    k = int(round(1.08 * 100))                                    # the end of the jerk's rise
    J0, J = stand.joints()[k], o.joints()[k]
    assert J[shoulder, 1] - J0[shoulder, 1] > 0.005                # the shoulder joint rises with the girdle
    G0, G = stand.fk()[1], o.fk()[1]
    assert float(torch.rad2deg(geodesic(G[:, shoulder], G0[:, shoulder])).max()) < 1e-6     # the arm hangs
    c, sign = JK.SITES[site][0][0], JK.SITES[site][3]
    assert c == shoulder - 3                                       # the collar of that shoulder (13, 14)
    collar_room = room_deg(stand.local_R()[0, c], c, "z", sign, "parent")
    assert JK.site_rooms(stand, site, 0)[0] == pytest.approx(collar_room)       # the room is the collar's
    assert JK.limits_table([site]) == LX.SHRUG and JK.limits_table(["neck_flexion"]) == LX.HEALTHY
    v = JK.jerk_validity(stand, o, [site])
    assert v["limits_table"] == "shrug" and v["limits_new"] == 0 and v["coupling"]["ok"]
    assert LX.check(LX.HEALTHY, stand, o)["limits_new"] > 0        # the healthy table refuses the hanging arm


def test_every_clip_with_a_shrug_keeps_the_arm_hanging(specs, built):
    """At every knot where a shrug site moves and no other site does, the upper arms keep the global rotation of the
    clip's level 0 (the shrug before task sh1c turned them with the collar, up to the shrug's angle), and the clip
    records the table 'shrug'. Seated, since Phase D the shrugging side's arm is carried (CARRY_SHRUG: its wrist kept
    on the support, test_a_seated_shrug_keeps_the_forearm_on_its_support), so the standing clips are read here."""
    from mdfm.body.rotations import geodesic
    n = 0
    for cid, (b, t) in built.items():
        prof = {k[len("profile_"):]: np.asarray(v) for k, v in (t.get("arrays") or {}).items()
                if k.startswith("profile_")}
        sh = [k for k in prof if k.startswith("shrug_")]
        if not sh:
            continue
        assert t["params"]["limits_table"]["table"] == "shrug" and t["validity"]["limits_new"] == 0, cid
        if any((t["params"].get("carry_arms") or {}).get("shrug_carried_share", {}).values()):
            continue                                       # seated: carried (the test below)
        lv = specs[cid].level0
        if lv not in built or built[lv][0].T != b.T:
            continue
        b0 = built[lv][0]
        act = np.any([np.abs(prof[k]) > 0 for k in sh], 0)
        other = np.any([np.abs(v) > 0 for k, v in prof.items() if k not in sh] or [np.zeros(b.T, bool)], 0)
        ks = np.flatnonzero(act & ~other)
        if not len(ks):
            continue
        G, G0 = b.fk()[1][ks], b0.fk()[1][ks]
        for j in (16, 17):
            assert float(torch.rad2deg(geodesic(G[:, j], G0[:, j])).max()) < 1e-6, (cid, j)
        n += 1
    assert n >= 3                                          # the standing ones (5 with the seated before Phase D)


# ---------------------------------------------------------------- wave SH1d part B (lane jerks)
HANG_STANDING = ("jerk_myoclonus_E7_propriospinal_f2_a3", "jerk_myoclonus_E7_propriospinal_f3_a2",
                 "jerk_functional_E7_axial_sev2_dur2", "jerk_tics_E7_complexity_c4")
CARRY_SEATED = ("jerk_myoclonus_E1_propriospinal_f1_a4", "jerk_myoclonus_E1_propriospinal_f3_a3",
                "jerk_tics_E1_number_n3", "jerk_tonic_tics_E1_f1_i3", "jerk_tonic_tics_E1_f1_i4",
                "jerk_tonic_tics_E1_f2_i3_mixed", "jerk_functional_E1_axial_sev2_dur2")


def _upper_arm_behind_deg(J, side):
    s, e = (16, 18) if side == "left" else (17, 19)
    u = J[:, e] - J[:, s]
    return np.degrees(np.arctan2(-u[:, 2], -u[:, 1]))


@pytest.mark.parametrize("cid", HANG_STANDING)
def test_standing_arms_hang_through_a_trunk_jerk(cid, JK, specs, built):
    """Review M6 (wave SH1d part B): standing, the arms no longer ride on a trunk or hip flexion as one block (the
    upper arm swung 25 to 33 deg behind the vertical, the wrists 70 to 160 mm back): the upper arm keeps its angle to
    the vertical within 10 deg of the level 0 and the wrists do not go back by more than 2 cm."""
    b, t = built[cid]
    b0 = built[specs[cid].level0][0]
    J, J0 = b.joints().numpy(), b0.joints().numpy()
    for side, w in (("left", 20), ("right", 21)):
        d = np.abs(_upper_arm_behind_deg(J, side) - _upper_arm_behind_deg(J0, side))
        assert d.max() < 10.0, (cid, side, d.max())
        assert -(J[:, w, 2] - J0[:, w, 2]).min() < 0.02, cid
    assert t["params"]["carry_arms"]["mode"] == "hanging (standing)" and t["params"]["carry_arms"]["carried_share"] > 0
    assert t["validity"]["limits_new"] == 0 and t["validity"]["balance"]["ok"]


@pytest.mark.parametrize("cid", CARRY_SEATED)
def test_seated_hands_stay_over_the_thighs_through_a_trunk_jerk(cid, JK, specs, built):
    """Review M6 (wave SH1d part B): seated, the hands no longer drop over the front of the support while the trunk
    flexes (wrist drops of 32 to 106 mm): carry_arms, before only on the functional jerks, keeps each wrist within
    1 cm of its level-0 place, the arms clear of the chair and the body (hand_contact)."""
    b, t = built[cid]
    b0 = built[specs[cid].level0][0]
    J, J0 = b.joints().numpy(), b0.joints().numpy()
    prof = JK.site_profiles(b.T, t["params"]["events"], b.rate)
    act = np.any([np.abs(prof[s]) > 0 for s in JK.CARRY_SITES if s in prof], axis=0)
    for side, w, chain in (("left", 20, {13, 16, 18, 20}), ("right", 21, {14, 17, 19, 21})):
        own = np.any([np.abs(v) > 0 for s, v in prof.items() if set(JK.SITES[s][0]) & chain] + [np.zeros(b.T, bool)],
                     axis=0)
        m = act & ~own
        if m.any():
            assert np.linalg.norm(J[m, w] - J0[m, w], axis=-1).max() < 0.01, (cid, side)
    assert t["params"]["carry_arms"]["mode"] == "carried (seated)" and t["params"]["carry_arms"]["carried_share"] > 0
    assert t["validity"]["limits_new"] == 0
    assert JK.hand_contact(b, "chair")["ok"], cid


SHRUG_SEATED = ("jerk_tics_E1_intensity_i4", "jerk_tics_E1_number_n2", "jerk_myoclonus_E1_multifocal_f3_a2",
                "jerk_tics_E1_touch_c3")


@pytest.mark.parametrize("cid", SHRUG_SEATED)
def test_a_seated_shrug_keeps_the_forearm_on_its_support(cid, JK, specs, built):
    """Phase D of docs/PLAN_2026-11-12.md: seated, a shrug raised the hanging arm with its girdle as one block, the
    forearm off the armrest (the wrist up to 39 mm above its level-0 place on the grade-4 tic intensity); carry_arms
    now carries the arm of the shrugging side (CARRY_SHRUG): the wrist within 1 cm of its level-0 place wherever that
    shrug moves and no site of the arm's own joints does, the elbow taking the rise, the arm inside its table and clear
    of the chair and the body. A shrug faster than CARRY_SHRUG_MAX_DEG_S (the grade-5 intensity's) is not carried."""
    b, t = built[cid]
    b0 = built[specs[cid].level0][0]
    J, J0 = b.joints().numpy(), b0.joints().numpy()
    prof = JK.site_profiles(b.T, t["params"]["events"], b.rate)
    n = 0
    for side, w in (("left", 20), ("right", 21)):
        sh = JK.CARRY_SHRUG[side]
        if sh not in prof:
            continue
        own = np.any([np.abs(v) > 0 for s, v in prof.items() if set(JK.SITES[s][0]) & set(JK.ARM_SITE_JOINTS[side])]
                     + [np.zeros(b.T, bool)], axis=0)
        m = (np.abs(prof[sh]) > 0) & ~own
        if m.any():
            assert np.linalg.norm(J[m, w] - J0[m, w], axis=-1).max() < 0.01, (cid, side)
            n += 1
    assert n and t["params"]["carry_arms"]["mode"] == "carried (seated)"
    assert t["validity"]["limits_new"] == 0 and JK.hand_contact(b, "chair")["ok"], cid
    # a brisk shrug is not carried, a slower one is (open-body port, wave 3, D26: on SMPL+H the grade-5 intensity's
    # shrug rose at 339 deg/s, past the bound; on the anatomical table its collar has jarm2022's 20 deg of elevation and
    # it rises at 167 deg/s, under the bound, so it is carried: the rule is read both ways on it)
    b5, t5 = built["jerk_tics_E1_intensity_i5"]
    brisk = t5["params"]["sites"]["shrug_left"]["peak_speed"] > JK.CARRY_SHRUG_MAX_DEG_S
    carried = t5["params"]["carry_arms"].get("shrug_carried_share", {}).get("left", 0.0)
    assert brisk == (carried == 0.0), (brisk, carried)


SIZED = ("jerk_tics_E1_number_n1", "jerk_myoclonus_E1_propriospinal_f3_a3", "jerk_myoclonus_E1_generalized_f3_a2",
         "jerk_negative_E7_leg_lapses_f4_d2", "jerk_myoclonus_E7_propriospinal_f3_a2")


@pytest.mark.parametrize("cid", SIZED)
def test_jerks_of_one_kind_vary_in_size(cid, JK, specs, built):
    """Review M2 (wave SH1d part B): every jerk of a kind repeated at the same size (CV 0.4 to 2 percent); each event
    now draws its own size (lognormal of median 1, SIZE_SIGMA): the largest displacement of any joint against the
    level 0 over each event varies with a CV of at least 0.15, the factors recorded in the truth."""
    b, t = built[cid]
    b0 = built[specs[cid].level0][0]
    D = np.linalg.norm(b.joints().numpy() - b0.joints().numpy(), axis=-1).max(1)
    ev = t["params"]["events"]
    sizes = []
    for e in ev:
        a, c = JK.event_span(e)
        sizes.append(D[int(round(a * b.rate)):int(round(c * b.rate)) + 1].max())
    sizes = np.asarray(sizes)
    if "leg_lapses" in cid:              # a lapse's size is its pelvis drop (the hip strategy moves the rest with it)
        sizes = np.asarray(JK.measure_dips(b, b0)["depth_cm"])
    assert len(sizes) >= 4 and sizes.std() / sizes.mean() >= 0.15, (cid, sizes)
    sf = [e["size_factor"] for e in ev]
    assert all(JK.SIZE_CLIP[0] - 1e-12 <= f <= JK.SIZE_CLIP[1] + 1e-12 for f in sf) and len(set(sf)) > 1
    assert t["params"]["size_variation"]["size_factors"] == pytest.approx(sf)


def test_the_size_draw_and_its_bounds(JK, specs, built):
    """The segmental (rhythmic) jerks vary less (SIZE_SIGMA_RHYTHMIC), the non-rhythmic rises vary by up to
    RISE_JITTER; every rotation peak stays under ROOM_CAP_SHARE of its room and under the speed cap."""
    seg = built["jerk_myoclonus_E1_segmental_right_arm_a3"][1]["params"]["size_variation"]
    assert seg["sigma"] == JK.SIZE_SIGMA_RHYTHMIC and seg["rise_jitter"] == 0.0
    f = np.log(seg["size_factors"])
    assert 0.03 < f.std() < 0.2
    tic = built["jerk_tics_E1_frequency_f3"][1]["params"]
    rf = np.asarray(tic["size_variation"]["rise_factors"])
    assert rf.min() >= 1 - JK.RISE_JITTER - 1e-12 and rf.max() <= 1 + JK.RISE_JITTER + 1e-12 and np.ptp(rf) > 0.2
    for e in tic["events"]:
        for q in e["parts"]:
            assert JK.MJ_PEAK_SPEED * abs(math.degrees(q["peak"])) / q["rise_s"] <= JK.PEAK_SPEED_CAP_DEG_S + 1e-6
