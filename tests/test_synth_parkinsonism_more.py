"""Tests of the clips added to mdfm/synth/parkinsonism.py by task park_gaps (catalogue rows AP01, AP03, G10, P33):
every clip's values read back by the module's measures match what it was built to show (the applause count, the
sit-down's speed at the seat contact and the part of its descent behind the heels, the abrupt rise's time and lean,
the motor state's grade over time, the hands' path up the thighs in Gowers' manoeuvre and the trunk they push up,
the hands relaxed at rest in the applause), the joints inside the SP1 table and the hand
limits, continuity, the feet planted, balance where it applies, the posed contacts (the palms never cross the
midline, the hands kept off the thighs' skin), the level-0 partner on the same time base and the same background
(task sh1b), the applause's approaches of the hands as many as its claps, the 3.9 level 0 named a healthy rise, the
new specs complete.
Procedural bodies in the neutral shape only; no lockbox, EVAL, CODY or care_pd record is read. The arising template
takes about 15 s once (cached); Gowers' manoeuvre about 95 s."""
import os

import numpy as np
import pytest
import torch

from mdfm import paths

HAVE_MA = os.path.exists(paths.smplh_model("neutral"))
pytestmark = pytest.mark.skipif(not HAVE_MA, reason="MoveAtlas SMPL+H model not found")
torch.set_num_threads(2)

NEW_IDS = {"park_E7_arise_abrupt", "park_E22_sitdown_level0", "park_E22_sitdown_enbloc", "park_E21_applause_g0",
           "park_E21_applause_g1", "park_E21_applause_g2", "park_E21_applause_g3", "park_E1_motorstate_level0",
           "park_E1_motorstate_offon", "park_E1_motorstate_onoff", "park_E7_gowers"}


@pytest.fixture(scope="module")
def P():
    from mdfm.synth import parkinsonism
    return parkinsonism


@pytest.fixture(scope="module")
def g0(P):
    return P.arise_grade("g0")


def _finite(b):
    return bool(torch.isfinite(b.rot6d).all() and torch.isfinite(b.transl).all()
                and (getattr(b, "hand6d", None) is None or torch.isfinite(b.hand6d).all()))


def _valid(tr):
    v = tr["validity"]
    assert v["limits_outside"] == 0 and v["hand_limits"]["ok"] and v["continuity"]["ok"]
    assert v["skate_p95_m_s"] < 1e-3


# ---------------------------------------------------------------- AP03
def test_abrupt_arising_is_faster_and_leans_less(P, g0):
    b, tr = P.abrupt_arise()
    b0, t0 = g0
    assert b.T == b0.T and b.rate == b0.rate and _finite(b)
    assert torch.equal(b.rot6d[:100], b0.rot6d[:100])               # the same background before the rise
    _valid(tr)
    assert tr["validity"]["balance"]["ok"] and tr["validity"]["balance"]["anterior_margin_min_m"] > 0.0
    from mdfm.synth.motions import balance
    rise = [a for a in tr["params"]["attempts"] if a["kind"] == "rises"][0]
    lift, end = rise["liftoff_knot"], int(round(rise["end_s"] * b.rate))
    assert balance(b, lift)["anterior"][lift:end].min() == pytest.approx(
        tr["validity"]["balance"]["anterior_margin_min_m"])                     # re-measured on the clip
    m, m0 = P.measure_arise(b), P.measure_arise(b0)
    assert (m["seat_offs"], m["rises"], m["falls_back"]) == (1, 1, 0)
    assert m["rise_time_s"] / m0["rise_time_s"] == pytest.approx(P.ABRUPT_RISE_S / 1.89, rel=0.1)
    # the trunk leans less (D26, open-body port wave 6: the 10 deg margin read the SMPL+H bout of the AMASS table, 50
    # to 60 deg; on the anatomical table the open body's bout is about 22 deg, and the abrupt rise keeps a share of it)
    assert m["trunk_lean_peak_deg"] < m0["trunk_lean_peak_deg"] and rise["abrupt"]["trunk_share"] < 1.0
    vy = np.gradient(b.joints().numpy()[:, 0, 1], 0.01).max()
    vy0 = np.gradient(b0.joints().numpy()[:, 0, 1], 0.01).max()
    assert vy > vy0 + 0.2
    assert tr["params"]["remeasured"]["pelvis_up_speed_max_m_s"] == pytest.approx(vy)
    J = b.joints().numpy()
    assert np.abs(J[:, [7, 8]] - J[:1, [7, 8]]).max() < 2e-3


@pytest.fixture(scope="module")
def sitdowns(P):
    return {k: P.sitdown(k) for k in ("controlled", "en_bloc")}


def test_the_sitdown_fold_stays_off_the_thighs(P, sitdowns):
    """Plan phase E: the controlled descent's added hip lean brought the right hand of the fold 13.2 mm into the thighs
    on the mesh; its collars turn the fold up by SITDOWN_FOLD_TURN_DEG on the lean's profile (one rigid block: the
    hands keep their places on the opposite arms), and the truth's 'fold' reads the descent: nothing enters."""
    (bc, tc), (be, te) = sitdowns["controlled"], sitdowns["en_bloc"]
    for tr in (tc, te):
        f = tr["validity"]["fold"]
        assert f["ok"] and f["entry_ok"] and f["thigh_left"]["min_mm"] >= 0.0 and f["thigh_right"]["min_mm"] >= 0.0
    assert tc["params"]["descent"]["fold_turn_deg"] == P.SITDOWN_FOLD_TURN_DEG
    assert "fold_turn_deg" not in te["params"]["descent"]                     # en bloc: no added lean, no turn


def test_sitdown_en_bloc_drops_onto_the_seat(P, sitdowns):
    (bc, tc), (be, te) = sitdowns["controlled"], sitdowns["en_bloc"]
    assert bc.T == be.T and _finite(bc) and _finite(be)
    assert tc["params"]["descent_knots"][0] == te["params"]["descent_knots"][0]       # the descents start together
    for b, tr in ((bc, tc), (be, te)):
        _valid(tr)
        assert tr["validity"]["seat_ok"] and tr["validity"]["ankle_move_max_m"] < 2e-3
        m = P.measure_sitdown(b, k_from=tr["params"]["descent_knots"][0])
        for key, val in m.items():
            ref = tr["params"]["remeasured"][key]
            assert (val is None and ref is None) or val == pytest.approx(ref), key
        assert m["sat_down"] and m["seat_clearance_min_m"] > -1e-3
        assert m["onset_s"] < m["contact_s"] < m["seated_s"]
    mc = P.measure_sitdown(bc, k_from=tc["params"]["descent_knots"][0])
    me = P.measure_sitdown(be, k_from=te["params"]["descent_knots"][0])
    assert me["impact_speed_m_s"] == pytest.approx(P.BLOC_IMPACT_M_S, rel=0.1)     # the speed at the seat contact
    assert mc["impact_speed_m_s"] < 0.5 * me["impact_speed_m_s"]
    assert me["descent_s"] < 0.7 * mc["descent_s"]                                    # onset to seated: faster
    assert mc["xcom_behind_heels_s"] is None and mc["behind_heels_pelvis_above_seated_m"] is None   # controlled
    d = tc["params"]["descent"]                          # its clock solved so (controlled_descent_u)
    assert 1.0 <= d["stretch"] <= P.SITDOWN_STRETCH_MAX and d["xcom_margin_min_m"] >= d["mos_min_m"] - 1e-9
    assert d["descent_s"] == pytest.approx((tc["params"]["descent_knots"][1] - tc["params"]["descent_knots"][0]) / 100)
    assert me["behind_heels_pelvis_above_seated_m"] > 0.2                             # en bloc: falls back from high
    assert me["behind_heels_descent_share_left"] > 0.6 and me["behind_heels_after_onset_s"] < 0.2
    assert mc["trunk_lean_peak_deg"] > me["trunk_lean_peak_deg"] + 20.0               # en bloc: the trunk upright
    assert me["seated_s"] < mc["seated_s"]
    J = be.joints().numpy()
    assert J[-1, 0, 1] == pytest.approx(bc.joints().numpy()[-1, 0, 1], abs=2e-3)     # both end seated


def test_sitdown_rejects_an_unknown_kind(P):
    with pytest.raises(ValueError):
        P.sitdown_body("plop")


# ---------------------------------------------------------------- AP01
@pytest.fixture(scope="module")
def applause(P):
    return {s: P.applause(s) for s in P.CLAP_COUNTS}


def test_applause_count_is_read_back(P, applause):
    assert len({b.T for b, _ in applause.values()}) == 1
    for s, (b, tr) in applause.items():
        assert _finite(b)
        _valid(tr)
        m = P.measure_claps(b)
        assert m["claps"] == P.CLAP_COUNTS[s] == tr["params"]["n_claps"]
        assert m["approaches"] == m["claps"] and tr["validity"]["approaches"]["ok"]     # no extra beat of the hands
        p = tr["params"]
        k0, k1 = (int(round(x * b.rate)) for x in p["raise_s"])
        g = tr["arrays"]["gap_m"][k0:k1 + 1]                     # the raise: the hands come in once, to the opening
        assert g.min() >= P.CLAP_OPEN_M - 1e-3 and np.diff(g).max() < 1e-3, s
        assert p["raise_clearance"]["medial_min_m"] >= P.CLAP_OPEN_M / 2.0 - 1e-3
        k0, k1 = (int(round(x * b.rate)) for x in p["lower_s"])
        gl = tr["arrays"]["gap_m"][k0:k1 + 1]                    # the lowering: the hands only part
        assert np.diff(gl).min() > -1e-3, s
        # the arms' outward turn does not jolt the wrists (task sh1b review: 9.9 m/s2 on the raise and 10.6 m/s2 on
        # the lowering before the solved turn was smoothed; the base version without the turn, 5.0 and 1.1 m/s2)
        acc = np.linalg.norm(np.diff(b.joints().numpy()[:, 16:22], 2, axis=0), axis=-1) * b.rate ** 2
        r0, r1 = (int(round(x * b.rate)) for x in p["raise_s"])
        assert acc[r0 - 1:r1].max() < 6.0 and acc[k0 - 1:k1].max() < 3.5, s
        assert p["raise_clearance"]["smoothed_miss_max_m"] < 2e-3 and p["lower_clearance"]["smoothed_miss_max_m"] < 2e-3
        assert tr["params"]["positive"] == (s > 0) and tr["grade"] == s
        assert m["gap_min_m"] > 0.0 and tr["validity"]["hands_apart"]["ok"]           # the palms never cross
        assert m["contact_gap_max_m"] < P.CLAP_TOUCH_M
        if m["claps"] > 1:
            assert m["rate_hz"] == pytest.approx(P.CLAP_HZ, rel=0.1)
            assert 0.8 * P.CLAP_OPEN_M < m["opening_max_m"] < 1.3 * P.CLAP_OPEN_M
        assert tr["validity"]["hand_rotation_error_deg"] < 0.1 and tr["validity"]["reach_max"] < 0.97
        assert np.abs(m["clap_times_s"][0] - tr["params"]["claps"][0]["contact_s"]) < 0.05


def test_applause_hands_relax_at_rest(P, applause):
    """The hands are relaxed (with the fingers' drift) at rest before the raise and after the lowering, shaped to clap
    only from the raise to the lowering."""
    from mdfm.synth.body import rotation_angle_deg
    clap = P.clap_hand()
    for s, (b, tr) in applause.items():
        hR = b.hand_R()
        p = tr["params"]
        k_rest = int(round(0.5 * P.CLAP_REST_S * b.rate))
        k_end = b.T - 10
        k_ik = int(round(p["claps"][0]["contact_s"] * b.rate))
        assert rotation_angle_deg(hR[k_ik].transpose(-1, -2) @ clap).max() < 1e-3, s     # clapping hands (rot6d)
        for k in (k_rest, k_end):
            assert rotation_angle_deg(hR[k].transpose(-1, -2) @ clap).max() > 5.0, (s, k)   # relaxed at rest
        assert rotation_angle_deg(hR[0].transpose(-1, -2) @ hR[k_end]).max() > 0.2       # the fingers drift
        assert p["lower_s"][1] < k_end / b.rate


def test_applause_palms_meet_flat(P, applause):
    """At a clap the palms meet along their length: the heel and the fingers both near the midline (the version
    before the fit of the palmar plane met at one corner)."""
    from mdfm.synth import hands as H
    b, tr = applause[2]
    k = int(round(tr["params"]["claps"][1]["contact_s"] * b.rate)) + 1
    J = b.joints().numpy()
    a = np.radians(P.CLAP_UP_DEG)
    f = np.array([0.0, np.sin(a), np.cos(a)])
    for side in ("left", "right"):
        Pv = H.body_points(b, P._hand_ids(side), knots=[k])[0]
        along = (Pv - J[k, H.WRIST[side]]) @ f
        x = np.abs(Pv[:, 0])
        heel = x[(along > 0.0) & (along < 0.05)].min()
        tips = x[(along > 0.13) & (along < 0.19)].min()
        assert heel < 0.006 and tips < 0.006, (side, heel, tips)


# ---------------------------------------------------------------- P33
@pytest.fixture(scope="module")
def motor(P):
    return {d: P.motor_state(d) for d in ("level0",) + P.MS_DIRECTIONS}


def test_motor_state_grade_over_time(P, motor):
    from mdfm.synth import grades as G
    b0, t0 = motor["level0"]
    assert len({b.T for b, _ in motor.values()}) == 1 and t0["grade"] == 0
    off, on = G.tremor_level_cm(P.MS_GRADES["off"]), G.tremor_level_cm(P.MS_GRADES["on"], "below_floor")
    for d in P.MS_DIRECTIONS:
        b, tr = motor[d]
        assert _finite(b) and tr["validity"]["limits_new"] == 0 and tr["validity"]["hand_limits"]["ok"]
        assert tr["validity"]["continuity"]["ok"]
        m = P.measure_motor_state(b, b0)
        first, last = (3, 1) if d == "off_to_on" else (1, 3)
        assert (m["grade_first"], m["grade_last"]) == (first, last)
        assert m["pp_cm_first"] == pytest.approx(off if first == 3 else on, rel=0.1)
        assert m["pp_cm_last"] == pytest.approx(on if last == 1 else off, rel=0.1)
        assert m["frequency_hz"] == pytest.approx(P.MS_HZ, abs=0.2)
        design = P.ms_crossings(d)
        for edge in (3.0, 1.0):
            assert m["crossings_s"][edge] == pytest.approx(design[edge], abs=0.6), (d, edge)
        g = np.asarray(m["grades"])
        steps = np.diff(g)
        assert (steps <= 0).all() if d == "off_to_on" else (steps >= 0).all()          # monotonic, no flicker
        assert tr["grade"] == f"{first} to {last}"
    still = P.measure_motor_state(b0, b0)
    assert max(still["pp_cm"]) == 0.0


# ---------------------------------------------------------------- G10
@pytest.fixture(scope="module")
def gowers(P):
    return P.gowers()


def test_gowers_hands_climb_the_thighs(P, gowers, g0):
    b, tr = gowers
    p, v = tr["params"], tr["validity"]
    assert b.T == g0[0].T and _finite(b)
    _valid(tr)
    assert v["ankle_move_max_m"] < 2e-3
    assert v["balance"]["ok"] and v["balance"]["posterior_margin_min_m"] >= 0.004            # over the feet
    assert v["hand_thigh"]["gap_min_m"] > 0.002 and v["hand_thigh"]["reach_max"] < 0.97
    k_lift = p["liftoff_knot"]
    k_rel = int(round(p["hands_released_s"] * b.rate))
    k_fin = int(round(p["phases"]["final"][1] * b.rate))
    m = P.measure_gowers(b, k_lift, k_rel, k_fin)
    # D26 (open-body port, wave 6 follow-up): the 50 deg read SMPL+H's lean at lift-off, from the slumped seat with 70
    # deg of spine flexion, past jarm2022's 45 of the thoracolumbar spine; on the open body the spine bends at most by
    # its table room (D91) from the arising's upright seat, and a deeper pelvis pitch turns the elbows of the hands
    # flat on the thighs past the table, so the lean at lift-off reads 41.16 deg (scripts/openbody_wave6_checks.py,
    # gowers): the test asks for the deep bow above 40
    assert m["trunk_lean_liftoff_deg"] > 40.0 and m["trunk_lean_peak_deg"] > g0[1]["params"]["remeasured"][
        "trunk_lean_peak_deg"] + 5.0
    assert m["pelvis_rise_m"] > 0.25
    # the hands push the trunk up: the lean held while a hand moves, falling at every step, released from 30 deg or
    # less, the hands kept off the thighs as they leave them
    J0 = b.joints().numpy()
    tv = J0[:, 12] - J0[:, 0]
    lean = np.degrees(np.arctan2(tv[:, 2], tv[:, 1]))
    lean = lean - lean[0]
    for st in p["steps"]:
        a, mv, e = (int(round(st[k] * b.rate)) for k in ("start_s", "moved_s", "end_s"))
        assert abs(lean[mv] - lean[a]) < 1.0 and lean[e] < lean[a] - 4.0, st
    assert m["trunk_lean_release_deg"] == pytest.approx(lean[k_rel]) == pytest.approx(v["release"]["trunk_lean_deg"])
    assert m["trunk_lean_release_deg"] <= P.GOWERS_RELEASE_MAX_DEG and v["release"]["ok"]
    k_off = int(round(p["hands_off_s"] * b.rate))
    assert abs(lean[k_off] - lean[k_rel]) < 1.0                                       # held while the hands lift
    assert m["final_gap_min_m"] > 0.0 and v["hand_thigh"]["ok"]
    assert m["final_gap_min_m"] == pytest.approx(v["hand_thigh"]["final_gap_min_m"])
    ext = -np.gradient(lean, 1.0 / b.rate)[k_off:].min()
    assert ext < 60.0                                                                 # deg/s, unhurried
    knots = np.asarray(m["gap_knots"])
    for side in ("left", "right"):
        al = np.asarray(m[f"{side}_along"])
        steps = [s for s in p["steps"] if s["side"] == side]
        ends = [int(round(s["end_s"] * b.rate)) for s in steps]
        assert all(al[e] < al[int(round(s["start_s"] * b.rate))] - 0.08 for s, e in zip(steps, ends))   # up
        c0, c1 = (int(round(x * b.rate)) for x in p["phases"]["climb"])
        assert (np.diff(al[c0:c1]) < 5e-3).all()                                        # never back down
        gap = np.asarray(m[f"{side}_gap_m"])
        lifted = gap > 0.01
        runs = np.diff(np.r_[0, lifted.astype(int), 0])
        assert (runs == 1).sum() == len(steps)                                           # one lift per step
        t = knots / b.rate
        stepping = np.zeros(len(knots), bool)
        for st in steps:
            stepping |= (t >= st["start_s"]) & (t <= st["moved_s"])
        planted = ~stepping & np.isfinite(gap)
        # posed on the skin: the per-knot correction smoothed upward (never nearer the thigh) lifts a hand by a few
        # mm where the lowest vertex changes (task sh1b), instead of the one-knot steps the wrists made before
        clear = P.THIGH_REST["clear_m"]
        assert gap[planted].min() > clear - 5e-4 and gap[planted].max() < clear + 5e-3
        # plan phase E: the declared touch as the skeleton page reads it (render_skeleton_catalogue.MESH_TOUCHES)
        assert v["hand_thigh"]["touch_mm"][side] == pytest.approx(1000.0 * gap[np.isfinite(gap)].min())
    J = b.joints().numpy()
    assert J[-1, 0, 1] == pytest.approx(g0[0].joints().numpy()[-1, 0, 1], abs=0.01)     # standing at the end
    # no one-knot jolt (task sh1b review: the wrists 36 to 38 m/s2 from the per-knot correction, the knees 9 m/s2 at
    # lift-off from the legs' reference, the wrists 20 m/s2 where a sine step lift starts and ends)
    acc = np.linalg.norm(np.diff(J, 2, axis=0), axis=-1) * b.rate ** 2
    assert acc[:, 16:22].max() < 6.0 and acc[:, [4, 5]].max() < 1.0


def test_gowers_hands_stay_apart_and_out_of_the_trunk(P, gowers):
    # review of task sh1c: with the palm centre on the thigh's axis the fingertips of the two hands met at the
    # midline and the hands and forearms entered the hip crease, and no check refused the clip
    b, tr = gowers
    p, v = tr["params"], tr["validity"]
    f = v["fold"]
    assert f["ok"] and all(f[c]["min_mm"] >= 0.0 for c in P.GOWERS_SKIN_CHECKS)
    assert f["n_knots"] >= int(round(p["phases"]["final"][1] * b.rate)) // 5
    lat = p["hand_lateral"]
    tried = lat["tried"]
    assert p["hand_lateral_m"] == lat["lateral_m"] == tried[-1]["lateral_m"] <= P.GOWERS_LATERAL_MAX_M
    assert [t["lateral_m"] for t in tried] == pytest.approx([i * P.GOWERS_LATERAL_STEP_M for i in range(len(tried))])
    clear = 1000.0 * P.GOWERS_SKIN_CLEAR_M

    def least(t):
        if "least_mm_first_read" in t:          # a try that failed at the knots where an earlier try failed
            return t["least_mm_first_read"]
        return min(t[c]["min_mm"] for c in P.GOWERS_SKIN_CHECKS)
    assert all(least(t) < clear for t in tried[:-1]) and least(tried[-1]) >= clear     # the smallest on the grid
    assert tried[-1]["knots_read"] > 100


# ---------------------------------------------------------------- specs
def test_new_specs_are_complete_and_partnered(P):
    from mdfm.synth import exam as EX
    s = {x.clip_id: x for x in P.specs()}
    assert NEW_IDS <= set(s) and len(s) == 28 + len(NEW_IDS)
    for cid in NEW_IDS:
        x = s[cid]
        assert x.since == "catalogue_v1"
        assert x.level0 is None or x.level0 in s
        assert x.segment in EX.SEGMENTS or x.segment_title
        cap = x.caption_spec()
        assert cap["segment_text"].startswith(x.segment)
        for txt in (x.operator, x.grade_text, x.params_text, x.note, cap["segment_text"]):
            assert chr(0x2013) not in txt and chr(0x2014) not in txt
    assert s["park_E7_gowers"].level0 == "park_E7_arise_g0" == s["park_E7_arise_abrupt"].level0
    assert s["park_E7_arise_g0"].operator == P.ARISE_LEVEL0_OPERATOR and "healthy" in P.ARISE_LEVEL0_OPERATOR
    assert "parkinsonian" not in s["park_E7_arise_g0"].operator                    # the partner of PSP and Gowers
    assert all("parkinsonian" in x.operator for x in s.values() if x.item == "3.9" and x.grade != 0)
    assert s["park_E22_sitdown_enbloc"].level0 == "park_E22_sitdown_level0"
    assert {s[f"park_E21_applause_g{g}"].grade for g in range(4)} == {0, 1, 2, 3}
    assert {s["park_E1_motorstate_offon"].grade, s["park_E1_motorstate_onoff"].grade} == {"3 to 1", "1 to 3"}
    for cid in ("park_E1_motorstate_offon", "park_E1_motorstate_onoff"):
        assert ("OFF to ON" in s[cid].grade_text) == cid.endswith("offon")
        assert ("ON to OFF" in s[cid].grade_text) == cid.endswith("onoff")


def test_new_clips_share_their_level0_background(P, g0, gowers, sitdowns, applause, motor):
    """One background per series (task sh1b): the abrupt arising and Gowers' manoeuvre lay the layer of their level
    0 (park_E7_arise_g0), the two sit-downs one layer, the four applause clips one, the motor state clips one."""
    def rec(tr):
        return tr["params"]["background"]
    groups = [("arise", [g0[1], P.abrupt_arise()[1], gowers[1]]), ("sitdown", [t for _, t in sitdowns.values()]),
              ("applause", [t for _, t in applause.values()]), ("motor_state", [t for _, t in motor.values()])]
    for name, trs in groups:
        r0 = rec(trs[0])
        assert r0["series"] == P.SERIES[name], name
        for tr in trs[1:]:
            assert rec(tr)["series"] == r0["series"] and rec(tr)["layer_sha256"] == r0["layer_sha256"], name
    b0, bg = g0[0], gowers[0]
    assert b0.T == bg.T
    k = int(round(0.6 * b0.rate))                 # seated, before either clip moves: the head drift is one signal
    R0, Rg = b0.local_R()[:k, 15], bg.local_R()[:k, 15]
    D0, Dg = R0[:1].transpose(-1, -2) @ R0, Rg[:1].transpose(-1, -2) @ Rg     # the drift in the head's own frame
    assert float((D0 - torch.eye(3, dtype=D0.dtype)).abs().max()) > 1e-4 and torch.allclose(D0, Dg, atol=1e-9)


def test_new_clips_are_deterministic(P, applause, sitdowns):
    b, _ = P.applause(1)
    assert torch.equal(b.rot6d, applause[1][0].rot6d)
    b2, _ = P.sitdown("en_bloc")
    assert torch.equal(b2.rot6d, sitdowns["en_bloc"][0].rot6d)
    assert torch.equal(b2.transl, sitdowns["en_bloc"][0].transl)
