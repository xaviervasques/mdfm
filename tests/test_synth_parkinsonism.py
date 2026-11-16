"""Tests of the parkinsonian items of the symptom catalogue, mdfm/synth/parkinsonism.py (task CAT-PDATAX, revised by
task FIX-parkinsonism): every clip's realized values read back by the module's measures match its grade (3.9
attempts, falls back, arm use, rise time and trunk lean; 3.14 movement counts, a nested ladder whose mean peak falls
with the grade, a background layer that moves, physio.py's values and one layer per series (task sh1b); the
sequence effect's slope, loss and cycle-to-cycle variability;
upper-limb freezing's episodes, trembling frequency and hastening), joints inside the SP1 table and the hand limits,
no foot skate where the feet are planted, no NaN, the level-0 partner on the same time base, determinism. Task
sh1c: the move from the folded grip to the chair arms keeps the forearms, hands, trunk and thighs apart (one arm
waits, motions.fold_open_lag; on one clock the right hand entered the left forearm), the push-off hands leave the
chair arms before the rising thighs reach them and a failed push-off's peak keeps them off, the folded arms of every
3.9 clip enter nothing (the validity's 'fold'); the standing 3.14 shrug keeps the arms hanging (limits_ext's table
'shrug'); the 3.14 ladder scales the head and finger drift only (the session decision kept in the docstring).
Procedural bodies in the neutral shape only; no lockbox, EVAL, CODY or care_pd record is read. The arising templates
take about 25 s once (cached)."""
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
    from mdfm.synth import parkinsonism
    return parkinsonism


@pytest.fixture(scope="module")
def arising(P):
    return {v: P.arise_grade(v) for v in P.ARISE_PLANS}


def _finite(b):
    return bool(torch.isfinite(b.rot6d).all() and torch.isfinite(b.transl).all()
                and (getattr(b, "hand6d", None) is None or torch.isfinite(b.hand6d).all()))


EXPECT_ARISE = {   # seat-offs, rises, falls back, arms used
    "g0": (1, 1, 0, False), "g1_slow": (1, 1, 0, False), "g1_attempts": (2, 1, 1, False),
    "g2": (2, 1, 1, True), "g3": (3, 1, 2, True), "g4": (4, 0, 4, True)}


def test_arise_ladder_is_read_back_as_its_grade(P, arising):
    T = {b.T for b, _ in arising.values()}
    assert len(T) == 1 and {b.rate for b, _ in arising.values()} == {100.0}       # one time base
    for v, (b, tr) in arising.items():
        m = P.measure_arise(b)
        assert (m["seat_offs"], m["rises"], m["falls_back"], m["arms_used"]) == EXPECT_ARISE[v], v
        assert tr["grade"] == P.ARISE_GRADES[v] and tr["scale"] == "MDS-UPDRS 3.9"
        assert _finite(b)
        val = tr["validity"]
        assert val["limits_outside"] == 0 and val["hand_limits"]["ok"] and val["continuity"]["ok"], v
        assert val["balance"]["ok"] and val["balance"]["folded_fails_behind_heels"], v       # render refusal key
        assert val["skate_p95_m_s"] < 1e-3                                            # the feet stay planted
        J = b.joints().numpy()
        assert np.abs(J[:, [7, 8]] - J[:1, [7, 8]]).max() < 2e-3
        fails = [a for a in tr["params"]["attempts"] if a["kind"] == "fails"]
        assert len(fails) == m["falls_back"]
        for a in m["attempts"]:
            if not a["rises"]:
                assert 0.02 < a["peak_lift_m"] < 0.15                                   # a few centimetres up
        for a in fails:
            lo, hi = P.FAIL_U if a["arms"] == "folded" else (0.5, P.PUSH_FAIL_U[1])
            assert lo - 1e-9 <= a["u_peak"] <= hi + 1e-9
            # a push-off's drop is braked by the arms so that the trunk keeps TRUNK_RATE_MAX (wave SH1d part B)
            top = P.FALL_S[1] if a["arms"] == "folded" else max(P.FALL_S[1], 1.875 * P.trunk_turn_deg(a) /
                                                                P.TRUNK_RATE_MAX / P.TRUNK_LEAD)
            assert P.FALL_S[0] - 0.01 <= a["fall_s"] <= top + 0.01
    g0, slow = P.measure_arise(arising["g0"][0]), P.measure_arise(arising["g1_slow"][0])
    assert slow["rise_time_s"] - g0["rise_time_s"] == pytest.approx(P.SLOW_RISE_S - 1.89, abs=0.15)
    assert slow["trunk_lean_peak_deg"] <= g0["trunk_lean_peak_deg"] + 1.0                # no deeper bow
    assert P.measure_arise(arising["g4"][0])["final_height_m"] < 0.01                     # still seated


def test_failed_attempts_vary(P, arising):
    fails = [a for a in arising["g4"][1]["params"]["attempts"] if a["kind"] == "fails"]
    assert len(fails) == 4
    for key in ("u_peak", "up_s", "trunk_share"):
        assert len({round(a[key], 3) for a in fails}) == 4, key
    assert len({a["fall_s"] for a in fails}) > 1                 # drawn in FALL_S, rounded to the 100 Hz knots
    push = [a["pelvis_peak_m"] for a in fails if a["arms"] == "push"]
    folded = [a["pelvis_peak_m"] for a in fails if a["arms"] == "folded"]
    assert min(push) > max(folded)                           # the push-offs rise higher, the elbows extending
    starts = [a["start_s"] for a in fails]
    gaps = np.diff(starts) - np.array([a["up_s"] + a["fall_s"] for a in fails[:-1]])
    assert len({round(g, 2) for g in gaps}) > 1                                   # the seated pauses vary


def test_a_failed_attempt_falls_back_and_the_rise_is_balanced(P):
    margin, _ = P.fail_margin_m()
    assert margin < 0.0                                       # the failed attempt's XcoM behind the heels
    _, info = P.arise_template("normal")
    assert info["balance"]["ok"]
    u, si = P.slow_rise_u()
    assert len(u) == int(round(P.SLOW_RISE_S * 100)) and u[-1] == pytest.approx(1.0) and (np.diff(u) >= 0).all()
    assert si["rise_s"] == P.SLOW_RISE_S and si["hesitation_s"] > 0.3 and si["after_liftoff_s"] > 1.05


def test_push_off_hands_rest_on_the_chair_arms_then_release(P, arising):
    from mdfm.synth import hands as H
    tg = P.armrest_targets()
    _, _, dom = H._rest()
    for v in ("g2", "g3", "g4"):
        b, tr = arising[v]
        arm = tr["arrays"]["arm_code"]
        J = b.joints().numpy()
        push = np.flatnonzero(arm == 2)
        rel = tr["params"]["release"][0]["ease_knot"] if tr["params"]["release"] else b.T   # the hands ease off
        lift = tr["params"]["push_wrist_lift_max_m"]
        assert len(push) and rel > push[0] and lift < 0.03
        for s, w in (("left", 20), ("right", 21)):
            d = J[push[push < rel], w] - tg["wrist"][s].numpy()
            assert np.hypot(d[:, 0], d[:, 2]).max() < 1e-6                         # the wrists over their targets
            assert -1e-6 <= d[:, 1].min() and d[:, 1].max() <= lift + 1e-6         # raised as they extend
        for s, w in (("left", 20), ("right", 21)):               # no hand inside its chair arm, before or after
            ids = np.flatnonzero(np.isin(dom, (w,) + tuple(range(H.FIRST[s], H.FIRST[s] + 15))))
            a = tg["armrest"][s]
            lo, hi = np.asarray(a["box_lo_m"]), np.asarray(a["box_hi_m"])
            knots = list(range(int(push[0]), min(b.T, rel + 80), 2))
            Pv = H.body_points(b, ids, knots=knots)
            depth = np.minimum(Pv - lo, hi - Pv).min(-1).max()
            assert depth < 0.002, (v, s, depth)
        if v == "g2":
            low = H.body_points(b, np.flatnonzero(np.isin(dom, (20,) + tuple(range(22, 37)))),
                                knots=[int(push[0])])[0][:, 1].min()
            assert 0.0 <= low - tg["armrest"]["left"]["top_y_m"] < 0.01             # the palm on the armrest
            assert tr["params"]["release"][0]["reach_share"] <= P.RELEASE_EXT + 0.02   # one knot past the share
            assert (arm[rel:] == 3).all()


def test_the_wrists_speed_has_no_step_at_the_release(P, arising):
    """Around the release the speed of each wrist changes by less than 0.25 m/s between two knots (task sh1c: with the
    arm rows held from the release knot it stepped up in one knot)."""
    n = 0
    for v in ("g2", "g3", "g4"):
        b, tr = arising[v]
        J = b.joints().numpy()
        for rel in tr["params"]["release"]:
            ks = np.arange(rel["ease_knot"] - 10, min(b.T - 2, rel["ease_knot"] + int(P.RELEASE_S * b.rate)))
            for w in (20, 21):
                speed = np.linalg.norm(np.diff(J[:, w], axis=0), axis=1) * b.rate
                assert np.abs(np.diff(speed)[ks]).max() < 0.25, (v, w)
            n += 1
    assert n >= 2


def test_arise_is_deterministic(P, arising):
    b, tr = P.arise_grade("g3")
    b0, _ = arising["g3"]
    assert torch.equal(b.rot6d, b0.rot6d) and torch.equal(b.transl, b0.transl) and torch.equal(b.hand6d, b0.hand6d)


def test_arise_standing_is_not_a_statue(P, arising):
    b, tr = arising["g0"]
    J = b.joints().numpy()
    k = int(round(tr["params"]["attempts"][0]["end_s"] * b.rate)) + 100        # the sway fully weighted in
    pel = J[k:, 0] - J[k:, 0].mean(0)
    _, Lt = P._layers(P.SERIES["arise"])
    dz = np.asarray(Lt["sway_ap_m"][k:b.T])
    dz = dz - dz.mean()
    rms, rms_d = np.sqrt((pel[:, 2] ** 2).mean()), np.sqrt((dz ** 2).mean())
    assert rms > 0.001 and rms == pytest.approx(rms_d, rel=0.15)              # the series' sway (physio's)
    R = b.local_R()
    from mdfm.synth.body import rotation_angle_deg
    sp = rotation_angle_deg(R[k:, 9].transpose(-1, -2) @ R[k:k + 1, 9]).numpy()
    assert sp.max() > 0.5                                                        # breathing


def test_one_background_per_series(P, arising):
    """Every clip of a series lays the background of its root (physio's layer of the root's seed): the records name
    the same series and layer digests, and the arising clips are the same body until the first one moves."""
    from mdfm.synth import physio as PH
    b0, t0 = arising["g0"]
    ref = t0["params"]["background"]
    assert ref["series"] == "park_E7_arise_g0" == P.SERIES["arise"] and ref["seed"] == PH.series_seed(ref["series"])
    assert ref["version"] == P.BG_VERSION and ref["physio_version"] == PH.VERSION
    for v, (b, tr) in arising.items():
        rec = tr["params"]["background"]
        assert rec["layer_sha256"] == ref["layer_sha256"] and rec["knots"] == b0.T, v
        assert torch.equal(b.rot6d[:50], b0.rot6d[:50]) and torch.equal(b.transl[:50], b0.transl[:50]), v
    for base in ("sit", "stand"):
        recs = [P.spontaneity_parts(base, g)[3]["background"] for g in (0, 2, 4)]
        assert len({json_key(r["layer_sha256"]) for r in recs}) == 1
        assert [r["scale"] for r in recs] == [1.0, P.BG_RATIO ** 2, P.BG_RATIO ** 4]
        assert recs[0]["series"] == P.SPONT_SERIES[base]
        for r in recs:                     # the postural sway never scaled; a part the clip does not lay records 0
            assert r["head_drift_rms_deg"] == [r["scale"] * x for x in P.HEAD_DRIFT_DEG]
            if base == "sit":
                assert r["sway_rms_m"] == [0.0, 0.0] and r["trunk_lean_rms_deg"] == recs[0]["trunk_lean_rms_deg"]
                assert min(r["trunk_lean_rms_deg"]) > 0.0 and r["trunk_forward_offset_mean_deg"] > 0.0
            else:
                assert r["sway_rms_m"] == list(P.SWAY_RMS_M) and r["trunk_lean_rms_deg"] == [0.0, 0.0]
                assert r["trunk_forward_offset_mean_deg"] == 0.0
    g4 = arising["g4"][1]["params"]["background"]                    # grade 4 stays seated: no sway laid
    assert g4["parts"]["sway"].startswith("none") and g4["sway_rms_m"] == [0.0, 0.0]
    assert ref["sway_rms_m"] == list(P.SWAY_RMS_M) and ref["finger_drift_rms_deg"] == 0.0   # the folded arms


def json_key(d):
    return tuple(sorted(d.items()))


def test_background_takes_physio_values(P):
    """The values are physio's (its sources), the signals its layers: breathing and the postural sway (the seated
    trunk's forward-only lean, the standing sway) identical on every grade, the head and finger drifts scaled by the
    grade."""
    from mdfm.synth import physio as PH
    assert (P.BREATH_PER_MIN, P.BREATH_TI_TE, P.SWAY_RMS_M, P.SWAY_BAND_HZ, P.HEAD_DRIFT_DEG, P.HEAD_BAND_HZ,
            P.FINGER_DRIFT_DEG) == (PH.BREATH_PER_MIN, PH.BREATH_TI_TE, PH.SWAY_RMS_M, PH.SWAY_BAND_HZ,
                                    PH.HEAD_DRIFT_DEG, PH.HEAD_BAND_HZ, PH.FINGER_DRIFT_DEG)
    root, T = P.SERIES["spont_sit"], 4501
    Ls, Lt = P._layers(root)
    for k in ("breath", "head_yaw_deg", "head_pitch_deg", "fingers_deg"):
        assert np.array_equal(Ls[k], Lt[k]), k                     # one draw: the seated and standing layers agree
    b1 = P.background_signals(root, T, 1.0, np.ones(T))
    bk = P.background_signals(root, T, 0.5, np.ones(T))
    assert np.array_equal(b1["breath"], Ls["breath"][:T]) and np.array_equal(bk["breath"], b1["breath"])
    assert np.allclose(bk["yaw"], 0.5 * b1["yaw"]) and np.allclose(bk["pitch"], 0.5 * b1["pitch"])
    assert np.allclose(bk["fingers"], 0.5 * b1["fingers"]) and np.abs(b1["fingers"]).max() > 0.0
    assert np.array_equal(b1["sway_z"], Lt["sway_ap_m"][:T]) and np.array_equal(b1["sway_x"], Lt["sway_ml_m"][:T])
    assert np.array_equal(b1["lean"], Ls["lean_ap_deg"][:T]) and b1["lean"].min() >= 0.0
    for k in ("sway_z", "sway_x", "lean", "lean_ml"):
        assert np.array_equal(bk[k], b1[k]), k                     # postural sway: not graded
    ups = np.flatnonzero(np.diff((np.asarray(Ls["breath"]) > 0.5).astype(int)) == 1)       # over physio's horizon
    assert 60.0 * 100.0 / np.diff(ups).mean() == pytest.approx(PH.BREATH_PER_MIN, rel=0.08)  # breaths per minute


@pytest.mark.parametrize("base", ["sit", "stand"])
def test_spontaneity_counts_and_sizes_follow_the_grade(P, base):
    out = [P.spontaneity(base, g) for g in range(5)]
    assert len({b.T for b, _ in out}) == 1
    counts, peaks, kept = [], [], []
    for g, (b, tr) in enumerate(out):
        p, val = tr["params"], tr["validity"]
        bgb, ob, _, _, _ = P.spontaneity_parts(base, g)
        assert torch.equal(ob.rot6d, b.rot6d)
        m = P.measure_spontaneity(bgb, b)
        assert m["count"] == p["n_movements"] == P.spontaneity_count(g, p["duration_s"])
        assert _finite(b) and val["limits_new"] == 0 and val["hand_limits"]["ok"] and val["continuity"]["ok"]
        assert val["skate_p95_m_s"] < 1e-3 and val["planted_foot_speed_max_m_s"] < 1e-3
        if base == "stand":     # the legs under the sway: no one-knot kink (task sh1b review: 1.6 m/s2 at grade 0)
            J = b.joints().numpy()[:, [0, 1, 2, 4, 5, 7, 8]]
            assert np.linalg.norm(np.diff(J, 2, axis=0), axis=-1).max() * b.rate ** 2 < 1.0, g
        for e, mv in zip(p["events"], m["movements"]):
            assert e["start_s"] - 0.02 <= mv["start_s"] <= e["start_s"] + e["move_s"]       # the min-jerk start
            if e["graded"]:
                assert e["amplitude"] <= e["level0_amplitude"] * e["factor"] * P.spont_amp_ratio(base) ** g + 1e-9
            assert e["move_s"] == pytest.approx(P.SPONT_KINDS[e["kind"]][3] * P.SPONT_SLOW_RATIO ** g)
        counts.append(m["count"])
        # standing (wave SH1d part B, review B2), the kept movements are chosen by visibility, so their mean joint
        # angle no longer falls (the watch outlives the small shifts): the summed reach of the movements falls
        peaks.append(m["mean_peak_deg"] if base == "sit" else sum(p["movement_reach_mm"]))
        kept.append({e["level0_index"] for e in p["events"]})
    assert counts == sorted(counts, reverse=True) and len(set(counts)) == 5
    assert all(peaks[g + 1] <= peaks[g] for g in range(4)), peaks              # the ladder is monotonic
    assert all(kept[g + 1] <= kept[g] for g in range(4))                       # nested thinning
    b2, _ = P.spontaneity(base, 2)
    assert torch.equal(b2.rot6d, out[2][0].rot6d)


@pytest.mark.parametrize("base", ["sit", "stand"])
def test_spontaneity_background_moves(P, base):
    from mdfm.synth.body import rotation_angle_deg
    bgb, _, _, _, _ = P.spontaneity_parts(base, 0)
    R = bgb.local_R()
    dev = rotation_angle_deg(R[:1].transpose(-1, -2) @ R).numpy()
    still = (dev.max(1) < 1e-3).mean()
    assert still < 0.01                                                          # no statue between movements
    assert rotation_angle_deg(R[:, 9].transpose(-1, -2) @ R[:1, 9]).max() > 0.8 * P.BREATH_SPINE_DEG   # breathing
    J = bgb.joints().numpy()
    pel = J[:, 0] - J[:, 0].mean(0)
    if base == "stand":
        rms = np.sqrt((pel[:, [0, 2]] ** 2).mean(0))
        _, Lt = P._layers(P.SPONT_SERIES[base])
        d = np.stack([np.asarray(Lt["sway_ml_m"][:bgb.T]), np.asarray(Lt["sway_ap_m"][:bgb.T])], -1)
        rms_d = np.sqrt(((d - d.mean(0)) ** 2).mean(0))
        assert rms == pytest.approx(rms_d, rel=0.15) and rms.min() > 0.001       # physio's sway, sideways and AP
        assert np.abs(J[:, [7, 8]] - J[:1, [7, 8]]).max() < 1e-6                  # the feet planted
    else:
        sr = P.seated_rest()
        assert all(0.0 < c < 0.005 for c in sr["clear_m"].values())               # the hands on the thighs
        assert sr["together_inside_table"]


def test_seated_kinds_are_valid(P):
    """Every seated and standing kind once, at level 0 and at the grade-4 amplitude: inside the table (standing, the
    table 'shrug' of limits_ext: the standing shrug turns the shoulders back under the raised collars, task sh1c),
    the hands above the thighs, the feet still where planted."""
    from mdfm.synth import limits_ext as LX
    for base in ("sit", "stand"):
        for amp in (1.0, P.SPONT_AMP_RATIO ** 4):
            ev, t0 = [], 0.8
            for i, k in enumerate(P.spont_kinds(base)):
                _, a0, unit, mv, hold, graded = P.SPONT_KINDS[k]
                ev.append({"kind": k, "level0_amplitude": a0, "unit": unit, "move_s": mv, "hold_s": 1.2,
                           "factor": 1.0, "drift": 0.15, "drift_seed": i, "graded": graded,
                           "amplitude": a0 * (amp if graded else 1.0), "start_s": t0})
                t0 += 2 * mv + 1.2 + 0.8
            T = int(round((t0 + 0.5) * 100)) + 1
            t = np.arange(T) / 100.0
            P._limit_head(base, ev)
            bg = P.background_signals(P.SPONT_SERIES[base], T, 1.0, np.ones(T) if base == "stand" else None)
            out, planted = P._realize(base, P._channels(base, t, bg, ev, None), T)
            table = LX.SHRUG if base == "stand" else LX.HEALTHY
            assert _finite(out) and LX.violations(table, out.rot6d)["outside"] == 0, (base, amp)
            J = out.joints().numpy()
            v = np.linalg.norm(np.gradient(J[:, [7, 8]], axis=0), axis=-1) * 100.0
            assert np.where(planted, v, 0.0).max() < 1e-3


@pytest.mark.parametrize("task", ["finger_tapping", "pronation_supination"])
def test_sequence_effect_slope_is_read_back(P, task):
    out = [P.sequence_effect(task, lv) for lv in range(4)]
    assert len({b.T for b, _ in out}) == 1
    n = P.SEQ_FT_CYCLES if task == "finger_tapping" else P.SEQ_PS_CYCLES
    losses = []
    for lv, (b, tr) in enumerate(out):
        p = tr["params"]
        m = P.measure_decrement(tr["arrays"]["signal"], b.rate)
        assert m["n_cycles"] == n
        assert m["relative_slope"] == pytest.approx(P.SEQUENCE_SLOPES[lv], rel=0.03, abs=2e-4)
        assert _finite(b) and tr["validity"]["limits_new"] == 0
        assert m["mean_rate_hz"] == pytest.approx(p["realized_rate_hz"], rel=0.01)
        pk = np.asarray(m["peak_times_s"])
        per = np.diff(pk)
        if lv < 3:
            assert 0.03 < per.std() / per.mean() < 0.07                          # not a metronome
        A = np.asarray(m["amplitudes"])
        c = np.arange(len(A))
        res = A - np.polyval(np.polyfit(c, A, 2), c)
        assert res.std() / A[0] > 0.04                                           # the amplitude varies too
        assert len(p["pauses"]) == P.SEQ_HESITATIONS.get(lv, 0)
        if task == "finger_tapping":
            assert tr["validity"]["hand_limits"]["ok"]
        else:                                                                   # both ends shrink
            sig = np.asarray(tr["arrays"]["signal"])
            tr_ = np.array([cy["trough_deg"] for cy in p["cycles"]])
            if lv >= 1:
                assert tr_[-5:].mean() > tr_[1:6].mean() + 2.0
            assert sig.min() > -20.0
        losses.append(m["loss_last5_vs_first5"])
    assert losses == sorted(losses) and losses[1] > 1.5 * losses[0]


@pytest.mark.parametrize("plan", ["3.4_g3", "long"])
def test_upper_limb_freezing_episodes_are_read_back(P, plan):
    from mdfm.synth.handtasks import hand_task
    from mdfm.synth.motions import posture_body
    b, tr = P.upper_limb_freezing(plan)
    p = tr["params"]
    m = P.measure_freezing(tr["arrays"]["signal"], b.rate)
    assert m["n_episodes"] == len(p["episodes"]) == len(P.FREEZE_PLANS[plan]["episodes_after"])
    for ep, got in zip(p["episodes"], m["episodes"]):
        assert got["start_s"] <= ep["onset_s"] + 0.02 and got["end_s"] >= ep["offset_s"] - 0.3
        assert got["tremble_hz"] == pytest.approx(P.TREMBLE_HZ, abs=0.2)             # inside the freeze band
        assert got["pre_rate_over_median"] > 1.1                                       # hastening before it
    assert _finite(b) and tr["validity"]["limits_new"] == 0 and tr["validity"]["hand_limits"]["ok"]
    base = posture_body("sit_finger_tapping", P.freeze_duration_s(plan))
    l0, t0 = hand_task(base, "finger_tapping", "left", n_cycles=P.FREEZE_PLANS[plan]["cycles"],
                       rate_hz=P.freeze_rate_hz(plan), decrement_onset=None, seed=P.SEED)
    assert l0.T == b.T and P.measure_freezing(t0["arrays"]["signal"], l0.rate)["n_episodes"] == 0
    b2, _ = P.upper_limb_freezing(plan)
    assert torch.equal(b2.hand6d, b.hand6d)


def test_specs_are_complete_and_partnered(P):
    from mdfm.synth import exam as EX
    s = P.specs()
    ids = {x.clip_id for x in s}
    assert len(ids) == len(s) == 39
    for x in s:
        assert x.since == "catalogue_v1"
        assert x.level0 is None or x.level0 in ids
        assert x.segment in EX.SEGMENTS or x.segment_title
        assert x.caption_spec()["segment_text"].startswith(x.segment)
        if "_ft_" in x.clip_id or "ulfreeze" in x.clip_id:
            assert "hand_left_medial" in x.views
    assert {x.grade for x in s if x.item == "3.9"} == {0, 1, 2, 3, 4}
    assert {x.grade for x in s if x.item == "3.14"} == {0, 1, 2, 3, 4}


def test_level0_partners_share_the_time_base(P):
    s = {x.clip_id: x for x in P.specs()}
    for cid in ("park_E1_spontaneity_g3", "park_E4L_sequence_ps_level2", "park_E4a_ulfreeze_g3"):
        a, _ = s[cid].build()
        b, _ = s[s[cid].level0].build()
        assert a.T == b.T and a.rate == b.rate, cid


# ---------------------------------------------------------------- task sh1c: the fold's contacts, the standing shrug
def test_the_hands_leave_the_fold_one_after_the_other_without_entering_anything(P, arising):
    from mdfm.synth import motions as MO
    lag, side, rec = P.opening_lag()
    # D26 (open-body port, wave 6 last pass): on SMPL+H's fold the one-clock opening entered the skins and a lag was
    # needed, which this test pinned; the open body's fold, solved again on the soft-kneed 'stand' (base.STAND_REACH),
    # opens on one clock with the skins apart (as the PKD opening's, motions D85), so the test reads the solve's rule:
    # the smallest lag of FOLD_OPEN_LAGS whose rehearsal keeps them apart, every smaller one entering
    assert lag in MO.FOLD_OPEN_LAGS and (lag == 0.0 or side in ("left", "right"))
    first = rec["tried"][0]
    assert first["lag"] == 0.0
    assert all(t["min_mm"] < rec["clear_mm"] for t in rec["tried"] if t["lag"] < lag)
    assert min(t["min_mm"] for t in rec["tried"] if t["lag"] == lag and t["lag_side"] == side) >= rec["clear_mm"]
    for v, (b, tr) in arising.items():
        f = tr["validity"]["fold"]
        assert f["ok"], (v, f)
        assert all(x["min_mm"] >= 0.0 for part in ("folded", "opening", "push") if f[part] for x in f[part].values())
        op = tr["params"]["opening"]
        if v in ("g2", "g3", "g4"):
            assert op["lag"] == lag and op["lag_side"] == side and f["opening"] is not None
            assert min(x["min_mm"] for x in f["opening"].values()) >= 0.0
        else:
            assert op is None and f["opening"] is None


def test_the_push_off_hands_stay_out_of_the_drawn_chair_arms(P, arising):
    """Phase D of docs/PLAN_2026-11-12.md: the hands rest on the chair arms the scene draws (the clip's own
    motions.arise_chair, clip_chair_arms) without entering them; park_E7_arise_g4's left fingers dipped 1.23 mm into
    its arm past the normal template's arm end (arise_fold_check 'chair_entry_mm', reported)."""
    for v in ("g2", "g3", "g4"):
        f = arising[v][1]["validity"]["fold"]
        assert f["chair_entry_mm"] == 0.0, (v, f["chair_entry_mm"])


def test_the_push_off_hands_turn_out_and_keep_off_the_rising_thighs(P, arising):
    yaw, rec = P.push_hand_yaw()
    # D26 (open-body port, wave 6 follow-up): on SMPL+H, with the wrists 8 cm behind the armrests' front ends, the
    # straight right hand's thumb side entered the rising thigh, and this test asked for an entry at no turn; with the
    # hands at the front ends (WRIST_BACK_M 0) the open body's straight hands keep off the thighs (push_hand_yaw's
    # record, scripts/openbody_wave6_checks.py push_off), so the test reads the solve's rule: the smallest turn of the
    # grid whose rehearsal keeps PUSH_THIGH_CLEAR_M, every smaller one under it
    tried, clear = rec["tried"], 1000.0 * P.PUSH_THIGH_CLEAR_M
    assert [t["yaw_deg"] for t in tried] == pytest.approx(P.PUSH_YAW_STEP_DEG * np.arange(len(tried)))
    assert all(t["thigh_gap_min_mm"] < clear for t in tried[:-1]) and tried[-1]["thigh_gap_min_mm"] >= clear
    assert 0.0 <= yaw <= P.PUSH_YAW_MAX_DEG and yaw == tried[-1]["yaw_deg"]
    assert P.armrest_targets()["hand_yaw_deg"] == yaw
    for v in ("g2", "g3"):
        b, tr = arising[v]
        rel = tr["params"]["release"][0]
        assert rel["by"] in ("reach", "table") and rel["thigh_gap_mm"] >= 1000.0 * P.PUSH_THIGH_GUARD_M, (v, rel)
        assert tr["params"]["push_hand_yaw"]["yaw_deg"] == yaw
    for v in ("g2", "g3", "g4"):
        f = arising[v][1]["validity"]["fold"]["push"]
        assert f["thigh_left"]["min_mm"] >= 0.0 and f["thigh_right"]["min_mm"] >= 0.0, (v, f)
    for v in ("g3", "g4"):
        pushes = [a for a in arising[v][1]["params"]["attempts"] if a["kind"] == "fails" and a["arms"] == "push"]
        assert pushes and all(a["thigh_gap_min_m"] >= P.PUSH_THIGH_GUARD_M for a in pushes), v


def test_the_standing_shrug_keeps_the_arms_hanging(P):
    from mdfm.body.rotations import geodesic
    from mdfm.synth import limits_ext as LX
    assert "stand" in P.SPONT_KINDS["shrug"][0] and "3.14 standing shrug" not in P.NOT_RENDERED
    bgb, out, ev, p, _ = P.spontaneity_parts("stand", 0)
    shrugs = [e for e in ev if e["kind"] == "shrug"]
    assert shrugs, [e["kind"] for e in ev]
    b, tr = P.spontaneity("stand", 0)
    assert tr["params"]["limits_table"]["table"] == LX.SHRUG and tr["validity"]["limits_new"] == 0
    assert tr["validity"]["coupling"]["ok"]
    G0, G = bgb.fk()[1], out.fk()[1]
    J0, J = bgb.joints(), out.joints()
    for e in shrugs:
        k = np.arange(int(round(e["start_s"] * 100)), int(round((e["start_s"] + e["len_s"]) * 100)) + 1)
        for s in (16, 17):
            assert float(torch.rad2deg(geodesic(G[k, s], G0[k, s])).max()) < 1e-6      # the upper arms hang
            assert float((J[k, s, 1] - J0[k, s, 1]).max()) > 0.005                  # the shoulders rise


# ---------------------------------------------------------------- wave SH1d part B (lane parkinsonism)
@pytest.fixture(scope="module")
def stand_ladder(P):
    return {g: P.spontaneity("stand", g) for g in range(5)}


def test_the_standing_spontaneity_ladder_stays_visible(P, stand_ladder):
    """Review B2: standing, grades 3 and 4 differed by 11 mm and grades 1 and 2 hung on the wrist-watch look alone.
    Every kept movement now moves the body by SPONT_VISIBLE_M or more against its background, a head turn is kept up
    to grade 3, the watch never carries a grade alone, and adjacent grades differ by at least one movement."""
    kinds = {}
    for g in range(1, 5):
        b, tr = stand_ladder[g]
        p, v = tr["params"], tr["validity"]
        assert v["visible"]["ok"] and min(p["movement_reach_mm"]) >= 1000 * P.SPONT_VISIBLE_M - 0.1, (g, p)
        kinds[g] = [e["kind"] for e in p["events"]]
        assert len(kinds[g]) == P.spontaneity_count(g, P.SPONT_DUR["stand"]), g
        if g <= 3:
            assert any(k.startswith("head_turn") for k in kinds[g]), (g, kinds[g])
        assert any(k != "watch" for k in kinds[g]), g
    for g in range(1, 4):
        assert len(kinds[g]) > len(kinds[g + 1]) and set(stand_ladder[g + 1][1]["params"]["level0_kept"]) < set(
            stand_ladder[g][1]["params"]["level0_kept"]), g
    from mdfm.synth import reach as RC                                       # grades 3 and 4 read apart (the nose)
    N3, N4 = RC.nose_point(stand_ladder[3][0]), RC.nose_point(stand_ladder[4][0])
    assert np.linalg.norm(N3 - N4, axis=-1).max() > P.SPONT_VISIBLE_M
    assert P.spont_amp_ratio("stand") == P.SPONT_AMP_RATIO_BY_BASE["stand"] and P.spont_amp_ratio("sit") == \
        P.SPONT_AMP_RATIO


def test_the_one_movement_caption_is_singular(P):
    spec = {s.clip_id: s for s in P.specs()}
    for cid in ("park_E1_spontaneity_g4", "park_E7_spontaneity_g4"):
        assert spec[cid].params_text.startswith("1 spontaneous movement in "), spec[cid].params_text


@pytest.mark.parametrize("v", ["g1_attempts", "g2", "g3", "g4"])
def test_a_failed_attempt_falls_back_without_a_whiplash(P, v):
    """Review M10: the fall back after a failed rise turned the trunk at 150 to 233 deg/s and moved the head at 17 to
    30 m/s2 (healthy rise: 108 deg/s, 11 m/s2). The trunk now returns over the whole drop and on past the seat
    contact (folded arms) or over a drop the arms brake (push-off): over each drop the trunk turns at most
    TRUNK_RATE_MAX (10 percent for the legs' re-solve) and the head accelerates at most HEAD_ACCEL_MAX."""
    b, tr = P.arise_grade(v)
    c = tr["validity"]["continuity"]
    assert c["ok"] and c["head_accel_bound_m_s2"] == P.HEAD_ACCEL_MAX and c["head_accel_max_m_s2"] <= P.HEAD_ACCEL_MAX
    J = b.joints().numpy()
    u = J[:, 12] - J[:, 0]
    w = np.abs(np.gradient(np.degrees(np.arctan2(u[:, 2], u[:, 1])), 1.0 / b.rate))
    fails = [a for a in tr["params"]["attempts"] if a["kind"] == "fails"]
    assert fails
    for a in fails:
        k0, k1 = int(round(a["peak_s"] * b.rate)), int(round(a["end_s"] * b.rate))
        assert w[k0:k1].max() <= 1.1 * P.TRUNK_RATE_MAX, (v, a, w[k0:k1].max())
        if a["arms"] == "folded":
            assert P.AFTER_SEAT_S[0] - 1e-9 <= a["trunk_after_seat_s"] <= P.AFTER_SEAT_S[1] + 1e-9


@pytest.mark.parametrize("v", ["g3", "g4"])
def test_a_failed_push_off_rises_without_a_whiplash(P, arising, v):
    """Wave SH1d fix B: on the minimum-jerk profile of the folded attempts the push-offs of grades 3 and 4 turned the
    pelvis-to-neck line at 145 to 191 deg/s at 100 Hz (the healthy rise: 130), over the plan's 120 and the healthy
    rise on the skeleton page. A push-off rises at a steady pace eased in and out (PUSH_EASE), stretched until its
    trunk keeps TRUNK_RATE_MAX: over each push-off's rise the trunk turns at most that (5 percent for the arms' solve
    and the background layer), the stretch recorded on the drawn timing factor."""
    b, tr = arising[v]
    w = P.trunk_line_rate(b.local_R(), b.transl, b.rate)
    push = [a for a in tr["params"]["attempts"] if a["kind"] == "fails" and a["arms"] == "push"]
    assert push
    for a in push:
        k0, k1 = int(round(a["start_s"] * b.rate)), int(round(a["peak_s"] * b.rate))
        assert w[k0:k1].max() <= 1.05 * P.TRUNK_RATE_MAX, (v, a, w[k0:k1].max())
        assert a["push_rise_stretch"] >= 1.0 and a["rise_trunk_rate_max_deg_s"] <= P.TRUNK_RATE_MAX + 1e-9
        assert a["timing_factor"] == pytest.approx(a["drawn_timing_factor"] * a["push_rise_stretch"])


@pytest.mark.parametrize("amp", [1.0, 0.75 ** 4])
def test_a_seated_foot_slide_travels_only_while_lifted(P, amp):
    """Wave SH1d fix B: with its way and its lift on one profile the seated foot slid 4 to 5 mm along the floor at
    each end of a slide (park_E1_spontaneity_g0 on the skeleton page, a skate). The ankle now moves horizontally only
    on the knots where it is lifted at least FOOT_CLEAR_M (1 mm for the legs' solve), its whole way and at least
    FOOT_LIFT_MIN_M of lift, at level 0 and at the grade-4 amplitude."""
    a0 = P.SPONT_KINDS["foot_slide_left"][1]
    ev = [{"kind": "foot_slide_left", "level0_amplitude": a0, "unit": "m", "move_s": 0.9, "hold_s": 1.2,
           "factor": 1.0, "drift": 0.15, "drift_seed": 0, "graded": True, "amplitude": a0 * amp, "start_s": 0.8}]
    T = 501
    t = np.arange(T) / 100.0
    bg = P.background_signals(P.SPONT_SERIES["sit"], T, 1.0, None)
    out, _ = P._realize("sit", P._channels("sit", t, bg, ev, None), T)
    A = out.joints().numpy()[:, 7]
    lift = A[:, 1] - A[0, 1]
    way = np.linalg.norm(A[:, [0, 2]] - A[0, [0, 2]], axis=-1)
    v = np.linalg.norm(np.gradient(A[:, [0, 2]], axis=0), axis=-1) * 100.0
    moving = v > 1e-3
    assert moving.any() and lift[moving].min() >= P.FOOT_CLEAR_M - 1e-3, lift[moving].min()
    assert way.max() == pytest.approx(a0 * amp, rel=0.02)
    assert lift.max() >= max(P.FOOT_LIFT_MIN_M, P.FOOT_LIFT_SHARE * a0 * amp) - 1e-3
