"""Tests of the gap wave of the gait generator, mdfm/synth/gait.py (gap_specs, 2026-11-02): seven catalogue rows,
nine clips, each read back from the body alone against its plan:

- AP02 PSP gait: speed and cadence at the PSP group's values (dale2023: 0.83 m/s, 100.5 steps/min), the step width
  at the healthy walk's times the Takamatsu ratio, the swing knee stiffer and the start slower than the level 0's;
- D19 dromedary gait: the trunk flexes once per stride at the planned phase of the left cycle, part of it at the
  hips (the pelvis pitching), the arms hanging plumb against it, the caption's values those read back;
- G14 vestibular gait: the heading drifts to the lesion side at the planned rate per metre, more with the eyes
  closed (frame_text), the steps more irregular in length, time and width, the planted feet not sliding with the
  drift; shown in the side view only (no shared camera looks along the start heading: NOT_RENDERED);
- F08 functional dragging leg: the dragged forefoot slides on the floor through its swings (a deliberate skate in
  the truth) and ends beside the sound foot while every stance contact stays exact, without a hip hike on that side;
- F09 functional fluctuations: the near-normal bouts read back near-normal steps between abnormal ones;
- A13 sensory ataxic gait: high steps, the head flexed with the eyes open and level with them closed, twice the
  variability with them closed;
- P17 freezing by shuffling forward: small fast steps inside three episodes (start, turn, straight), each leg at
  about 2.25 Hz against about 1 Hz in walking, the body still progressing in the start and straight episodes, the
  turn stalled inside its episode, the pelvis slowing and relaunching no harder than at the level 0's gait
  initiation, grade 3 of the program's map.
Every clip passes the generator's validity (joint-limit table, stance slip, swing penetration) and the balance check
(gait_balance, with a positive control), and its caption fits the frame; the new parameters at their defaults and as
constant functions of time change nothing, and three earlier clips keep their motion (pinned fingerprints).
Procedural bodies in the neutral shape only; no store, lockbox, EVAL, CODY or care_pd record is read.
"""
import os
from dataclasses import replace

import numpy as np
import pytest
import torch

from mdfm import paths

HAVE_MA = os.path.exists(paths.smplh_model("neutral"))
pytestmark = pytest.mark.skipif(not HAVE_MA, reason="MoveAtlas SMPL+H model not found")

NEW_IDS = ("gait_E8_psp", "gait_E8_dromedary", "gait_E8_vestibular_left_eyes_open",
           "gait_E19_vestibular_left_eyes_closed", "gait_E8_functional_dragging_left",
           "gait_E8_functional_fluctuating", "gait_E8_sensory_ataxic_eyes_open", "gait_E19_sensory_ataxic_eyes_closed",
           "gait_E8_fog_shuffling_g3")


@pytest.fixture(scope="module")
def GA():
    torch.set_num_threads(2)
    from mdfm.synth import gait
    return gait


@pytest.fixture(scope="module")
def built(GA):
    """{clip id: (body, truth)} of every gap-wave clip and of the level 0."""
    out = {}
    for s in GA.specs():
        if s.clip_id in NEW_IDS or s.clip_id == GA.LEVEL0:
            out[s.clip_id] = s.build()
    return out


@pytest.fixture(scope="module")
def level0(GA, built):
    b, tr = built[GA.LEVEL0]
    return b, tr, GA.gap_read("level0", b, None)


def test_the_gap_specs(GA):
    sp = GA.specs()
    ids = [s.clip_id for s in sp]
    assert len(set(ids)) == len(ids)
    gap = [s for s in sp if s.clip_id in NEW_IDS]
    assert [s.clip_id for s in gap] == list(NEW_IDS)
    for s in gap:
        # the freezing by shuffling is built on the 3.10 grade 1 walk and paired with it (2026-11-03 review)
        l0 = GA.PD_WALK_LEVEL0 if s.clip_id == "gait_E8_fog_shuffling_g3" else GA.LEVEL0
        assert s.since == "catalogue_v1" and s.level0 == l0
        assert set(s.views) <= {"track_left", "examiner", "left", "back", "front"}
        if "vestibular" in s.clip_id:              # the side view only: every other shared camera reads the veer
            assert s.views == ("track_left",)      # to the left as one to the right (NOT_RENDERED)
        assert s.segment in ("E8", "E19") and (s.segment == "E8" or s.segment_title)
        for text in (s.operator, s.grade_text, s.params_text, s.segment_title):
            assert chr(0x2013) not in text and chr(0x2014) not in text      # no en or em dash
    for cid in ("gait_E8_functional_dragging_left", "gait_E8_functional_fluctuating"):   # the S-FMDRS anchors
        s = [x for x in gap if x.clip_id == cid][0]
        assert f"S-FMDRS gait duration: {GA.S_FMDRS_GAIT[cid]}" in s.grade_text


def test_every_gap_caption_fits_the_frame(GA):
    """The viewer cuts a caption line that runs past the frame with an ellipsis: every line of every gap-wave clip
    fits the 960 px frame at the caption's 14 px font, its 8 px margins kept."""
    from PIL import Image, ImageDraw
    from mdfm.synth import viewer as V
    d = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    f = V._font(14)
    for s in GA.specs():
        if s.clip_id in NEW_IDS:
            for line in V.caption_lines(s.caption_spec()):
                assert d.textlength(line, font=f) <= V.SIZE[0] - 16, (s.clip_id, line)


def test_every_gap_clip_is_valid_and_balanced(GA, built):
    # open-body kind (wave 5 second follow-up, D26): (a) the 1 mm/s stance slip asked here was read on SMPL+H's soles;
    # the clips' own acceptance is the contact point under SLIP_TOL_M_S (D76); after the push-off pivot's fix (D102)
    # the gap clips read under it (at most 0.189 mm/s in catalogue_limits_wave5c), the 1 mm/s kept; passes
    for cid in NEW_IDS:
        b, tr = built[cid]
        v = tr["validity"]
        assert v["gait"]["ok"] and v["limits_outside"] == 0 and v["finite"], cid
        assert v["stance_slip_p95_m_s"] < 1e-3, cid
        assert v["balance"]["ok"] and v["gait"]["balance_ok"], cid
        assert v["balance"]["standing_inside_share"] == 1.0, cid
        assert tr["params"]["unsourced"] and "module" in tr["sources"], cid
        for k in tr["sources"]:
            assert k in GA.SOURCES, (cid, k)


def test_the_balance_check_sees_an_extrapolated_centre_of_mass_outside_the_foot(GA):
    """Positive control: the whole walking body slid sideways at 1 m/s carries its extrapolated centre of mass about
    0.3 m (v / omega0) beyond the lateral border of every stance foot on that side."""
    b, info = GA.generate(GA.NORMAL, duration_s=7.0, start_at_s=1.0, stop_at_s=5.5, seed=6)
    ok = GA.gait_balance(b, info)
    assert ok["ok"] and ok["ml_mos_p2_m"] > 0.03 and ok["standing_inside_share"] == 1.0
    tr = b.transl.clone()
    tr[:, 0] += 1.0 * b.times()                       # +x: the body's left on the start heading
    bad = GA.gait_balance(replace(b, transl=tr), info)
    assert not bad["ok"] and bad["ml_mos_p2_m"] < -0.1 and bad["standing_inside_share"] < 1.0


def test_new_parameters_at_their_defaults_or_constant_in_time_change_nothing(GA):
    p = replace(GA.NORMAL, step_m=0.6)
    a, _ = GA.generate(p, duration_s=5.0, start_at_s=1.0, stop_at_s=4.0, seed=4)
    q = replace(p, clear_at=GA.CLEAR_AT, clear_width=0.2, pump_deg=0.0, drag_sides=(), drag_height_m=0.0,
                swing_min_jerk_m=None, arm_swing_step_ref_m=None)
    b, _ = GA.generate(q, duration_s=5.0, start_at_s=1.0, stop_at_s=4.0, seed=4)
    assert torch.equal(a.rot6d, b.rot6d) and torch.equal(a.transl, b.transl)
    const = {k: (lambda t, v=GA.pair(getattr(p, k))[0]: np.full(np.shape(t), v)) for k in
             ("contact_pitch_deg", "liftoff_pitch_deg", "hip_overshoot_deg", "swing_ankle_deg", "arm_abduction_deg")}
    c, _ = GA.generate(replace(p, **const), duration_s=5.0, start_at_s=1.0, stop_at_s=4.0, seed=4)
    assert torch.allclose(a.rot6d, c.rot6d, atol=1e-12) and torch.allclose(a.transl, c.transl, atol=1e-12)


def test_the_psp_gait_reads_back_its_sources(GA, built, level0):
    # open-body kind (wave 5 second follow-up, D26): (b) a clinical property, the PSP's stiff knees 10 deg under the
    # level 0's: kept; on Anny the clearance lift bent the asked 38 deg back to 58.45 deg against the level 0's
    # 65.99. Wave 8 second part (D163): the motion fixed, the ankle held up further in early swing
    # (PSP_SWING_ANKLE_DEG), the knee near 52 deg (design section 20)
    _, _, r0 = level0
    b, tr = built["gait_E8_psp"]
    r = tr["params"]["read_back"]
    P = GA.PSP_GAIT
    assert abs(r["speed_steps_m_s"] / P["speed_m_s"] - 1) < 0.04            # Hatanaka 0.83 m/s
    assert abs(r["cadence_spm"] / P["cadence_spm"] - 1) < 0.03               # Hatanaka 100.5 steps/min
    want_w = GA.NORMAL.step_width_m * P["width_psp"] / P["width_control"]
    assert abs(r["step_width_m"] - want_w) < 0.015                           # Takamatsu's ratio
    assert r["step_width_m"] > r0["step_width_m"] + 0.02
    for side in ("left", "right"):                                          # stiff legs
        assert r[f"knee_swing_{side}_deg"] < r0[f"knee_swing_{side}_deg"] - 10.0
    assert r["first_step_s"] > r0["first_step_s"] + 0.3                     # slow initiation
    assert r["trunk_lean_deg"] < r0["trunk_lean_deg"] - 2.0                 # upright, not stooped
    assert r["head_pitch_down_deg"] < 0.0                                   # the neck extended


def test_the_dromedary_trunk_pumps_once_per_stride(GA, built, level0):
    b, tr = built["gait_E8_dromedary"]
    tc = tr["params"]["read_back"]["trunk_cycle"]
    D = GA.DROMEDARY
    assert tc["n"] >= 8
    assert tc["range_deg"] > 0.6 * D["pump_deg"]
    dphi = (tc["peak_share"] - D["pump_peak"] + 0.5) % 1.0 - 0.5
    assert abs(dphi) < 0.08                                                 # locked to the left stance
    assert abs(tc["stride_hz"] / (104.0 / 120.0) - 1) < 0.05
    assert tc["pelvis_pitch_range_deg"] > 0.25 * D["pump_deg"]             # part of it at the hips
    t0 = GA.trunk_cycle(level0[0])
    assert t0["range_deg"] < 0.2 * tc["range_deg"]
    seen = GA.DROMEDARY_SEEN_DEG                                            # the caption states the read-back
    assert abs(tc["range_deg"] - seen["trunk"]) <= 1.0 and abs(tc["pelvis_pitch_range_deg"] - seen["pelvis"]) <= 1.0
    assert tr["sources"] == ["module"] or tuple(tr["sources"]) == ("module",)
    # the arms hang: without arm swing the upper arm stays near plumb while the trunk pumps
    p, _ = GA.dromedary()
    a, info = GA.generate(replace(p, arm_swing_deg=(0.0, 0.0)), duration_s=8.0, start_at_s=1.0, seed=3)
    J = a.joints().numpy()
    psi = info["heading_rad"]
    ua = J[:, 18] - J[:, 16]
    arm = np.degrees(np.arctan2(ua[:, 0] * np.sin(psi) + ua[:, 2] * np.cos(psi), -ua[:, 1]))
    assert np.ptp(arm[300:750]) < 4.0


def test_the_vestibular_walk_veers_more_with_the_eyes_closed(GA, built):
    rows = {}
    for eyes, cid in (("open", "gait_E8_vestibular_left_eyes_open"),
                      ("closed", "gait_E19_vestibular_left_eyes_closed")):
        b, tr = built[cid]
        p = tr["params"]
        assert p["eyes"] == eyes and len(p["frame_text"]) == 1
        ft = p["frame_text"][0]
        assert ft["text"] == ("EYES OPEN" if eyes == "open" else "EYES CLOSED")
        assert ft["from_s"] == 0.0 and abs(ft["to_s"] - b.duration) < 1e-9
        rows[eyes] = p["read_back"]
        want = GA.VESTIBULAR[eyes]["veer_deg_per_m"]
        assert abs(rows[eyes]["path"]["deg_per_m"] - want) < 0.15 * want + 0.1, eyes      # to the left: positive
        assert rows[eyes]["path"]["lateral_m"] > 0.0
    o, c = rows["open"], rows["closed"]
    assert c["path"]["lateral_m"] > 3.0 * o["path"]["lateral_m"] and c["path"]["heading_change_deg"] > 15.0
    assert c["speed_steps_m_s"] < o["speed_steps_m_s"] - 0.15 and c["step_width_m"] > o["step_width_m"] + 0.02
    # more irregular with the eyes closed: step length, step time and step width (the caption's claim)
    assert c["step_length_cv"] > o["step_length_cv"] and c["step_time_cv"] > o["step_time_cv"]
    assert c["step_width_sd_m"] > o["step_width_sd_m"]
    assert abs(c["step_width_sd_m"] / GA.VESTIBULAR["closed"]["step_width_sd_m"] - 1) < 0.2


def test_veer_path_reads_back(GA):
    for rate in (0.0, 2.0):
        b, info = GA.generate(replace(GA.NORMAL, pivot_share=0.0), duration_s=11.0, start_at_s=1.0, stop_at_s=9.0,
                              seed=5, path_fn=GA.veer_path(rate))
        d = GA.path_drift(b)
        assert abs(d["deg_per_m"] - rate) < 0.2, (rate, d)
        assert GA.validity(b, info)["stance_slip_p95_m_s"] < 1e-3


def test_the_dragged_foot_slides_and_the_stances_stay_planted(GA, built):
    # open-body kind (wave 5 second follow-up, D26): (b) a physical property, a stride's slide per dragged swing:
    # kept; the read slide on Anny's longer foot ran 15.8 percent over the stride (0.648 m against 0.56). Wave 8
    # second part (D165): the reader fixed: the slide is the touching sole's net travel per swing, the path its
    # velocities trace (the toe pad sweeping about the ankle) reported beside it (design section 20)
    b, tr = built["gait_E8_functional_dragging_left"]
    sk = tr["params"]["deliberate_skate"]
    assert sk["deliberate"] and sk["side"] == "left" and sk["phase"] == "swing"
    assert sk["n_swings"] >= 8 and sk["contact_share"] > 0.9                # the forefoot on the floor in swing
    assert sk["slide_speed_median_m_s"] > 0.3
    p, _ = GA.functional_drag("left")
    assert abs(sk["slide_m_per_swing_median"] / (2.0 * p.step_m) - 1) < 0.15    # a stride per swing
    assert sk["sound_strides"] >= 8 and abs(sk["cadence_spm"] / p.cadence_spm - 1) < 0.04
    assert abs(sk["end_ahead_of_sound_m_median"]) < 0.10                    # brought up beside the sound foot
    assert abs(sk["speed_m_s"] / (p.step_m * p.cadence_spm / 60.0) - 1) < 0.1
    assert tr["validity"]["stance_slip_p95_m_s"] < 1e-4                       # the stance contacts exact
    gi = tr["params"]["gait_params"]
    assert gi["drag_sides"] == ("left",) or list(gi["drag_sides"]) == ["left"]
    # no hike of the dragged side in its swing, no circumduction asked
    st = np.asarray(tr["arrays"]["stance"])
    J = b.joints().numpy()
    sw = ~st[:, 0] & st[:, 1]
    assert (J[sw, 1, 1] - J[sw, 2, 1]).mean() < 0.005
    assert GA.pair(p.swing_abduction_deg)[0] == 0.0


def test_the_fluctuating_gait_has_near_normal_bouts(GA, built):
    b, tr = built["gait_E8_functional_fluctuating"]
    rows = tr["params"]["bouts"]
    nn = [r for r in rows if r["kind"] == "near_normal"]
    ab = [r for r in rows if r["kind"] == "abnormal"]
    assert len(nn) == 2 and len(ab) >= 2
    for r in nn:
        assert abs(r["step_length_m"] / GA.FLUCT_NORMAL["step_m"] - 1) < 0.08
        assert r["step_width_m"] < GA.FLUCT_NORMAL["step_width_m"] + 0.03       # the near-normal base
    for r in ab:
        assert abs(r["step_length_m"] / GA.FLUCT_ABNORMAL["step_m"] - 1) < 0.08
        assert r["step_width_m"] > GA.FLUCT_ABNORMAL["step_width_m"] - 0.04     # the wide base on ice
    assert min(r["pelvis_speed_m_s"] for r in nn) > 1.8 * max(r["pelvis_speed_m_s"] for r in ab)


def test_the_sensory_ataxic_gait(GA, built, level0):
    # open-body kind (wave 5 second follow-up, D26): (b) a clinical property, the high steps' clearance over 5 cm:
    # kept; on Anny the early swing's toe keeps the read minimum at 4.64 cm, the motion not fixed: carried to wave 8's
    # gait pass (design 16.9, D109), left open there with its readings and tries (design 20.7, D170)
    _, _, r0 = level0
    r = {e: built[c][1]["params"]["read_back"] for e, c in (("open", "gait_E8_sensory_ataxic_eyes_open"),
                                                              ("closed", "gait_E19_sensory_ataxic_eyes_closed"))}
    for e in r:
        assert r[e]["knee_swing_left_deg"] > r0["knee_swing_left_deg"] + 15.0          # high steps
        assert r[e]["clearance_left_m"] > 0.05 and r[e]["contact_pitch_left_deg"] > 14.0
        assert r[e]["step_width_m"] > 0.17
        assert built[f"gait_{'E8' if e == 'open' else 'E19'}_sensory_ataxic_eyes_{e}"][1]["params"]["frame_text"][0][
            "text"] == f"EYES {e.upper()}"
    assert r["open"]["head_pitch_down_deg"] > r0["head_pitch_down_deg"] + 20.0          # looking at the feet
    assert r["closed"]["head_pitch_down_deg"] < r["open"]["head_pitch_down_deg"] - 15.0
    assert abs(r["closed"]["head_pitch_down_deg"] - r0["head_pitch_down_deg"]) < 3.0   # head up: as the level 0's
    # eyes closed: about the Romberg ratio of anagnostou2024 on the step variability
    assert r["closed"]["step_length_cv"] > 1.6 * r["open"]["step_length_cv"]
    assert r["closed"]["step_width_sd_m"] > 1.5 * r["open"]["step_width_sd_m"]
    assert r["closed"]["step_width_m"] > r["open"]["step_width_m"]


def _forward_speed(GA, b):
    """m/s: the pelvis's speed along its heading (the heading smoothed over 0.3 s, the speed over 0.1 s)."""
    psi = GA._smooth(np.unwrap(GA._root_yaw(b)), 30)
    fwd = np.stack([np.sin(psi), np.cos(psi)], -1)
    P = b.joints().numpy()[:, 0, [0, 2]]
    return GA._smooth((np.gradient(P, 1.0 / b.rate, axis=0) * fwd).sum(1), 10)


def test_freezing_by_shuffling_forward(GA, built, level0):
    from mdfm.synth.grades import FOG_GRADES
    b, tr = built["gait_E8_fog_shuffling_g3"]
    p = tr["params"]
    assert tr["grade"] == 3 and tr["scale"] == "MDS-UPDRS 3.11"
    assert p["fog_grade_map"] == FOG_GRADES[3] and "moore2008" not in tr["sources"]
    eps = {e["context"]: e for e in p["episodes"]}
    assert set(eps) == {"start", "turn", "straight"}
    assert abs(eps["straight"]["duration_s"] - GA.FOG_STRAIGHT_S) < 1e-6
    sh = GA.FOG_SHUFFLE
    rate = sh["cadence_spm"] / 60.0
    for ctx in ("start", "turn", "straight"):
        r = eps[ctx]["read"]
        assert r["n_steps"] >= 5 and r["to_s"] - r["from_s"] >= 1.0, ctx
        assert abs(r["step_rate_hz"] / rate - 1) < 0.05, ctx
        for side in GA.SIDES:                  # each leg at half the step rate: 2.25 Hz, below the 3 to 8 Hz band
            assert abs(r["leg_hz"][side] / (0.5 * rate) - 1) < 0.1, (ctx, side)
    for ctx in ("start", "straight"):
        r = eps[ctx]["read"]
        assert abs(r["step_length_m"] / sh["step_m"] - 1) < 0.25, ctx
        assert 0.1 < r["pelvis_speed_m_s"] < 0.3, ctx                          # shuffling forward, not in place
    # the turn stalls inside its episode: the heading barely moves while the feet shuffle, against the turn's
    # parts before and after the stall
    st = eps["turn"]["read"]
    assert abs(st["turn_rate_deg_s"]) < 15.0 and st["pelvis_speed_m_s"] < 0.1
    a, c = eps["turn"]["turn_stall_s"]
    assert abs(c - a - GA.FOG_TURN_STALL_S) < 1e-6 and st["from_s"] >= a and st["to_s"] <= c
    for part in p["turn_parts"].values():
        assert abs(part["turn_rate_deg_s"]) > 30.0
    walk = p["walking"]
    for w in walk:
        assert abs(w["step_length_m"] / 0.60 - 1) < 0.08 and w["pelvis_speed_m_s"] > 0.9
        assert max(w["leg_hz"].values()) < 1.2                                   # the walk's stride frequency
    # the episode edges: the pelvis slows and relaunches no harder than at the level 0's gait initiation (at
    # FOG_RAMP_S 0.3 it stopped at 7.5 m/s2, as against a wall)
    acc = np.abs(np.gradient(_forward_speed(GA, b), 1.0 / b.rate))
    b0 = level0[0]
    acc0 = np.abs(np.gradient(_forward_speed(GA, b0), 1.0 / b0.rate))
    peak0 = float(acc0[int(b0.rate):int(3.5 * b0.rate)].max())              # the level 0's initiation
    k = int(1.5 * b.rate)
    assert float(acc[k:-k].max()) <= 1.2 * peak0, (float(acc[k:-k].max()), peak0)


def _fingerprint(b):
    """T, a fixed-weight sum and the absolute sum of rot6d, the translation at four knots."""
    r = b.rot6d.numpy().astype(np.float64)
    t = b.transl.numpy().astype(np.float64)
    w = np.cos(0.7 * np.arange(r.size, dtype=np.float64)).reshape(r.shape)
    k = [b.T // 4, b.T // 2, 3 * b.T // 4, b.T - 1]
    return [float(b.T), float((r * w).sum()), float(np.abs(r).sum())] + t[k].ravel().tolist()


# the fingerprints of three earlier clips: computed from the code of commit eeea6f8 (before the gap wave) and equal
# on the gap wave's; recomputed on 2026-11-03 for the leg-to-leg calibration (the healthy base STEP_WIDTH_M, the
# turn's widening TURN_BASE_M, the swing, double-support and post-overlay guards), which changes them on purpose;
# recomputed again in wave SH1d (2026-11-05) for the clinical review's fixes: the planted foot pivoting knot to knot
# in the turns (_pivot_offsets, TURN_ROCKER_DROP: the level 0 and the 3.10 grade 2 walk), the landing foot set
# down wider in place of a guard bump at heel strike (LEG_GUARD_END_SHARE), and for the choreic walk
# the soft-limited sway with the head righting and the choreic hands' contact guard (involuntary.py's guard, which
# wave SH1d's involuntary lane also changes: recompute this one after it lands); recomputed once more after the
# review of lane gait (2026-11-05): a turning stance foot pivots on the roll's contact point while it rolls and on
# its touching sole point nearest the ball while it stands flat; the generator's other changes leave them as they are;
# recomputed on 2026-11-11 for the page's skate check: a turning stance foot pivots on the heel while the heel rocker
# rolls and on the push-off's pivot point from the forefoot's landing to toe-off (_push_off_pivot), and the closing
# step lands without gliding (LAND_CLEAR_M); computed on commit b126b4a's tree with these changes only; recomputed
# on the open body (open-body port, wave 5 follow-up, D100): the program's rotations on Anny's rest and sole (the
# body model changes every clip), computed on commit ae7944b's worktree checkout; recomputed in the wave 5 second
# follow-up (D107) for the swing's bump widening on the knee rate and the push-off pivot's 1 mm touch (D101, D102,
# D106), computed on commit 197d2e7's worktree checkout; the 3.10 grade 2 walk's recomputed in wave 8's second part
# (D169) for its swing ankle at LOW_FLAT_ANKLE_DEG and its dip lift by the ankle first (D162), the other two unchanged
EARLIER = {"gait_E8_level0": [2307.0, 0.14120218241284288, 109451.58595952237, -0.021919558932288242,
                              0.7783130995849898, 5.399319314327336, 0.5643588856952717, 0.7816103656934675,
                              8.419220508010193, 0.6072524164102061, 0.780200809803876, 0.9715229271584844,
                              0.5860509746857044, 0.7971944732930589, -2.35097473466183],
           "gait_E8_parkinsonian_g2": [3486.0, -1.6545566842434418, 166484.12995001738, -0.00285331537477697,
                                       0.7802162209607043, 5.761199215751797, 0.1965308020918796, 0.7833014351170255,
                                       9.043187056618054, 0.1679400736890821, 0.7850693114995918, 2.4190284258313595,
                                       0.17862812091770308, 0.7971944732930589, -2.181917659600683],
           "gait_E8_choreic": [2561.0, -0.8588400109769707, 123289.22558101194, -0.015197488173933543,
                               0.7806332847124229, 4.263927795298341, 0.3975211798185532, 0.7841400272215813,
                               6.460116003200949, 0.6152697617064182, 0.7944925548477508, 0.28997101800221337,
                               0.5272300333400776, 0.7971314651941865, -2.2319120674956627]}


def test_earlier_clips_keep_their_motion(GA, built):
    specs = {s.clip_id: s for s in GA.specs()}
    for cid, want in EARLIER.items():
        b = built[cid][0] if cid in built else specs[cid].build()[0]
        assert np.allclose(_fingerprint(b), want, rtol=1e-9, atol=1e-8), cid


def test_the_drag_lift_solver(GA):
    s = np.linspace(-1.0, 1.0, 7)

    def low(d):                                  # a sole whose height rises with the lift, offset per knot
        return 0.002 * (np.asarray(d) - 5.0 * s)
    d = GA._drag_lift(low, len(s), 0.0)
    assert np.allclose(low(d), 0.0, atol=1e-6)
    d = GA._drag_lift(lambda x: 0.002 * (np.asarray(x) + 100.0), 3, 0.0)    # floats even at the lowest lift
    assert np.all(d == GA.DRAG_LIFT_RANGE[0])


def test_episodes_fn_switches_inside_its_intervals(GA):
    f = GA.episodes_fn(1.0, 3.0, [(2.0, 4.0)], 0.5)
    t = np.array([0.0, 1.74, 2.0, 2.26, 3.0, 4.0, 4.26, 6.0])
    v = f(t)
    assert v[0] == 1.0 and v[1] == 1.0 and abs(v[2] - 2.0) < 1e-9 and v[3] == 3.0 and v[4] == 3.0
    assert abs(v[5] - 2.0) < 1e-9 and v[6] == 1.0 and v[7] == 1.0
