"""The open-body port's wave 8 second-part readings that the design's section 20 cites (docs/OPEN_BODY_PORT_DESIGN.md):
the gait clips and walks whose tests failed on the open body (section 16.9), each read at the values this part
commits and at the values before it (the same generator with the former parameters put back by
dataclasses.replace), the layered clips the background layer refused (section 18.6) and the plantigrade walk's turns.

Sections (all by default, or those named with --only; each section's commit and time in 'measured'):
- steppage: gait.steppage(('left',)) on the tests' 8 s walk (seed 2): the dropped side's swing knee, the other side's,
  the mid-swing ankles and the swing hips, at the committed asks (STEPPAGE_KNEE_DEG, STEPPAGE_HIP_DEG) and at the
  former ones (60 and 12 deg) and 55 and 12 deg;
- low_flat: the 3.10 grade 2 walk (11 s, stop at 8.5 s, seed 13) and the walking on ice (14 s, stop at 10 s, seed 4)
  of the tests: the swing knees per side (gait.summary of gait.measure_gait) and every swing's knee peak, at the
  committed LOW_FLAT_ANKLE_DEG and LOW_FLAT_DIP_SYNERGY and at the former values (16 and 14 deg, no dip synergy);
- antalgic: the antalgic gait's painful swing at the committed and the former ankle, and the healthy walk's swing
  knee peaks;
- gap_clips: the read-backs of the gap-wave clips whose tests failed (the PSP gait, the sensory ataxic gait with the
  eyes open and closed, the dragged foot, the start hesitation's steps in place) and their level 0's;
- scissor: the scissoring clips' records (gait.scissor_record) against the healthy walk's, the tests' walks;
- layer: the clips the background layer refused on 5816ade (section 18.6) and the PD posture grades, each built raw
  and with the layer (spec.build.raw, with_background): the layer's verdict, the checks it fails and their readings;
- sensory: the sensory ataxic gait's read clearance, as committed and under the tries of this part;
- scenes: the chair readings the scene tests pinned on SMPL+H, on the open body;
- test_readings: the readings the other re-derived tests write beside their assertions (the standing inversion
  term, the supine spine's held row, the standing tics' read-back, the video driver's tapping offset on the
  synthetic body's SMPL+H reference);
- plantigrade_turns: the catalogue's and the exam's clips walked on base.walk's plantigrade walk (kind 'plantigrade')
  and their turns.

Output: reports/openbody/wave8b_checks.json (numbers only; no mesh, shape or model array). CPU; procedural bodies on
the catalogue's one body; no store, lockbox, EVAL, CODY or care_pd record is read.

    PYTHONDONTWRITEBYTECODE=1 <conda>/envs/mdfm/python.exe scripts/openbody_wave8b_checks.py [--only s]
"""
import argparse
import datetime
import json
import os
import subprocess
import sys
import time
from dataclasses import replace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "reports", "openbody", "wave8b_checks.json")

import numpy as np  # noqa: E402
import torch  # noqa: E402

LAYER_CLIPS = ("tremor_E2_raise_long_rest_g0", "tremor_E2_raise_postural_g2", "tremor_E2_raise_reemergent_g2",
               "jerk_functional_E7_axial_sev2_dur2", "invol_chorea_E2_fingers_g3", "invol_chorea_E2_fingers_g4",
               "invol_td_E1_rocking_s4", "post_E7_pd_posture_g1", "post_E7_pd_posture_g2", "invol_chorea_E1_g3",
               "post_E1_pnkd_excerpt", "post_E7_pkd_dystonic")


def _head():
    r = subprocess.run(["git", "-c", f"safe.directory={ROOT.replace(os.sep, '/')}", "-C", ROOT, "rev-parse", "--short",
                        "HEAD"], capture_output=True, text=True)
    return r.stdout.strip() or None


def _r(x, n=3):
    return None if x is None else round(float(x), n)


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items() if k not in ("arrays", "_bad_knots")}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist()) if x.size <= 64 else f"array {x.shape}"
    if isinstance(x, torch.Tensor):
        return _jsonable(x.detach().cpu().numpy())
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def _failed(validity):
    return sorted(k for k, v in (validity or {}).items() if isinstance(v, dict) and v.get("ok") is False)


# ---------------------------------------------------------------- steppage
def steppage():
    from mdfm.synth import gait as GA
    from mdfm.synth.body import components_deg
    out = {"committed": {"knee_deg": GA.STEPPAGE_KNEE_DEG, "hip_overshoot_deg": GA.STEPPAGE_HIP_DEG}, "asks": {}}
    for knee, hip in ((GA.STEPPAGE_KNEE_DEG, GA.STEPPAGE_HIP_DEG), (60.0, 12.0), (55.0, 12.0)):
        p = GA.steppage(("left",))
        p = replace(p, knee_swing_deg=(knee, GA.pair(p.knee_swing_deg)[1]),
                    hip_overshoot_deg=(hip, GA.pair(p.hip_overshoot_deg)[1]))
        b, info = GA.generate(p, duration_s=8.0, start_at_s=1.0, seed=2)
        sw = info["swing_weight"]
        comp = components_deg(b.rot6d)
        s = GA.summary(GA.measure_gait(b))
        out["asks"][f"knee {knee:g}, hip {hip:g}"] = {
            "knee_swing_left_deg": _r(s["knee_swing_left_deg"]), "knee_swing_right_deg": _r(s["knee_swing_right_deg"]),
            "ankle_mid_swing_left_deg": _r(comp[sw[:, 0] > 0.9, 7, 0].mean()),
            "ankle_mid_swing_right_deg": _r(comp[sw[:, 1] > 0.9, 8, 0].mean()),
            "hip_flexion_swing_max_left_deg": _r((-comp[sw[:, 0] > 0, 1, 0]).max()),
            "hip_flexion_swing_max_right_deg": _r((-comp[sw[:, 1] > 0, 2, 0]).max()),
            "knee_peaks_left_deg": [_r(w["knee_peak_deg"], 2) for w in info["swings"]["left"]],
            "limits_outside": GA.validity(b, info).get("limits_outside")}
    out["test_band_deg"] = [64.0, 78.0]
    out["source"] = "park2023's 70.5 deg of moderate CMT1A (the module docstring)"
    return out


# ---------------------------------------------------------------- the low flat-footed swings
def low_flat():
    from mdfm.synth import gait as GA
    out = {"committed": {"ankle_deg": GA.LOW_FLAT_ANKLE_DEG, "dip_synergy": list(GA.LOW_FLAT_DIP_SYNERGY)}}

    def read(p, **kw):
        b, info = GA.generate(p, **kw)
        s = GA.summary(GA.measure_gait(b))
        return {"knee_swing_left_deg": _r(s["knee_swing_left_deg"]),
                "knee_swing_right_deg": _r(s["knee_swing_right_deg"]),
                "difference_deg": _r(s["knee_swing_left_deg"] - s["knee_swing_right_deg"]),
                "knee_peaks_left_deg": [_r(w["knee_peak_deg"], 1) for w in info["swings"]["left"]],
                "knee_peaks_right_deg": [_r(w["knee_peak_deg"], 1) for w in info["swings"]["right"]],
                "gait_ok": GA.validity(b, info)["gait"]["ok"],
                "limits_outside": GA.validity(b, info).get("limits_outside")}
    pd2, _ = GA.parkinsonian(2)
    ice = GA.functional("walking_on_ice")
    kw_pd = dict(duration_s=11.0, start_at_s=1.0, stop_at_s=8.5, seed=13)
    kw_ice = dict(duration_s=14.0, start_at_s=1.0, stop_at_s=10.0, seed=4)
    out["pd_grade2"] = {"committed": read(pd2, **kw_pd),
                        "before (ankle 16, no dip synergy)": read(replace(pd2, swing_ankle_deg=(16.0, 16.0),
                                                                          dip_synergy=None), **kw_pd)}
    out["walking_on_ice"] = {"committed": read(ice, **kw_ice),
                             "before (ankle 14)": read(replace(ice, swing_ankle_deg=(14.0, 14.0)), **kw_ice)}
    out["test_bounds"] = {"symmetry_band_deg": 4.0, "ice_right_knee_below_deg": 45.0}
    return out


def antalgic():
    """The antalgic gait's painful swing (the steppage test's third walk, 10 s, seed 2) at the committed
    ANTALGIC_ANKLE_DEG and at the former 12 deg, and the healthy walk's swing knee peaks (the low flat-footed test's
    third walk, 11 s, stop at 8.5 s, seed 7)."""
    from mdfm.synth import gait as GA
    out = {"committed_ankle_deg": GA.ANTALGIC_ANKLE_DEG}
    for name, ank in (("committed", GA.ANTALGIC_ANKLE_DEG), ("before (12 deg)", 12.0)):
        p = GA.antalgic("left")
        p = replace(p, swing_ankle_deg=(ank, GA.pair(p.swing_ankle_deg)[1]))
        b, info = GA.generate(p, duration_s=10.0, start_at_s=1.0, seed=2)
        s = GA.summary(GA.measure_gait(b))
        out[name] = {k: _r(s[k], 4) for k in ("knee_swing_left_deg", "knee_swing_right_deg", "stance_share_left",
                                               "stance_share_right")}
    b, info = GA.generate(GA.NORMAL, duration_s=11.0, start_at_s=1.0, stop_at_s=8.5, seed=7)
    out["healthy_swing_knee_peaks_deg"] = {side: [_r(w["knee_peak_deg"], 2) for w in info["swings"][side]
                                                  if w["clearance_m"] is not None] for side in GA.SIDES}
    out["test_bounds"] = {"antalgic_knee_below_deg": 48.0, "healthy_peak_below_deg": 72.0}
    return out


# ---------------------------------------------------------------- the gap clips
GAP_KEYS = ("knee_swing_left_deg", "knee_swing_right_deg", "clearance_left_m", "contact_pitch_left_deg", "step_width_m",
            "step_width_sd_m", "step_length_cv", "speed_steps_m_s", "cadence_spm", "first_step_s", "trunk_lean_deg",
            "head_pitch_down_deg")


def gap_clips():
    from mdfm.synth import gait as GA
    want = ("gait_E8_psp", "gait_E8_sensory_ataxic_eyes_open", "gait_E19_sensory_ataxic_eyes_closed",
            "gait_E8_functional_dragging_left", GA.LEVEL0)
    out = {}
    for s in GA.specs():
        if s.clip_id not in want:
            continue
        b, tr = s.build()
        p = tr["params"]
        rb = p.get("read_back") or GA.gap_read("level0", b, None)
        rec = {k: _r(rb.get(k), 4) for k in GAP_KEYS if rb.get(k) is not None}
        if "deliberate_skate" in p:
            rec["deliberate_skate"] = {k: v for k, v in p["deliberate_skate"].items()
                                       if not isinstance(v, (list, dict))}
        rec["failed"] = _failed(tr.get("validity"))
        out[s.clip_id] = rec
    return out


def _hesitation_read(over=None, consts=None):
    from mdfm.synth import gait as GA
    old = {k: getattr(GA, k) for k in (consts or {})}
    for k, v in (consts or {}).items():
        setattr(GA, k, v)
    try:
        p, _ = GA.start_hesitation(1)
        p = replace(p, **(over or {}))
        b, info = GA.generate(p, duration_s=10.0, start_at_s=1.0, stop_at_s=7.0, seed=GA.SEED + 6)
    finally:
        for k, v in old.items():
            setattr(GA, k, v)
    J = b.joints().numpy()
    psi = info["heading_rad"]
    fwd = np.stack([np.sin(psi), np.cos(psi)], -1)
    rng = {}
    for name, (a, c) in (("in_place", (1.8, 3.2)), ("walking", (6.5, 8.5))):
        k = slice(int(a * b.rate), int(c * b.rate))
        w = [((J[k, wr][:, [0, 2]] - J[k, 0][:, [0, 2]]) * fwd[k]).sum(-1) for wr in (20, 21)]
        rng[name] = max(float(np.ptp(x)) for x in w)
    heel = J[int(1.8 * b.rate):int(3.2 * b.rate), [7, 8], 1]
    out = {"arm_range_in_place_over_walking": _r(rng["in_place"] / rng["walking"]),
           "ankle_range_in_place_m": _r(heel.max() - heel.min(), 4), "steps_in_place": []}
    for s in GA.SIDES:
        h, kn, an = GA.LEG[s][:3]
        u, v = J[:, h] - J[:, kn], J[:, an] - J[:, kn]
        cos = (u * v).sum(-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1))
        knee = 180.0 - np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
        k_st = int(0.5 * b.rate)
        sk, sa = float(knee[:k_st].mean()), float(J[:k_st, an, 1].mean())
        for w in info["swings"][s]:
            if w["k_to"] / b.rate >= 3.2:
                continue
            k0, k1 = w["k_to"], w["k_hs"]
            out["steps_in_place"].append({"side": s, "toe_off_s": _r(k0 / b.rate, 2),
                                          "knee_over_standing_deg": _r(knee[k0:k1 + 1].max() - sk, 2),
                                          "ankle_rise_mm": _r(1000 * (J[k0:k1 + 1, an, 1].max() - sa), 1),
                                          "lift_gain_deg": _r(w["gain_deg"], 2),
                                          "early_gain_deg": _r(w["early_gain_deg"], 2)})
    out["gait_ok"] = GA.validity(b, info)["gait"]["ok"]
    return out


def hesitation():
    """The start hesitation's steps in place as committed, and the tries of this part (none kept): the dip lift and
    the clearance lift by the ankle first, the ankle further dorsiflexed, the toe-off roll and the knee smaller."""
    ankle = (0.5, 0.5, 2.0)
    out = {"committed": _hesitation_read(),
           "tries": {"dip lift by the ankle (0.5, 0.5, 2)": _hesitation_read({"dip_synergy": ankle}),
                     "dip and clearance lifts by the ankle, ankle 18, roll 5, knee 6": _hesitation_read(
                         {"dip_synergy": ankle, "lift_synergy": ankle},
                         {"HESITATION_ANKLE_DEG": 18.0, "HESITATION_LIFTOFF_DEG": 5.0, "HESITATION_KNEE_DEG": 6.0})},
           "test_bounds": {"knee_over_standing_deg": 20.0, "ankle_rise_mm": 25.0, "ankle_range_in_place_m": 0.05}}
    return out


def _excursion(GA, b, info):
    """The scissoring test's outward kick: the largest swing hip abduction beyond its toe-off and heel-strike values."""
    R = b.local_R().numpy()
    out = []
    for side, h, sg in (("left", 1, 1.0), ("right", 2, -1.0)):
        _, d, _ = GA._xzy_angles(R[:, h])
        for s_ in info["swings"][side]:
            k0, k1 = s_["k_to"], s_["k_hs"]
            if k0 < 150 or k1 > b.T - 150:
                continue
            out.append(float((sg * d[k0:k1 + 1]).max() - max(sg * d[k0], sg * d[k1])))
    return max(out)


def scissor():
    from mdfm.synth import gait as GA
    hb, hinfo = GA.generate(GA.NORMAL, duration_s=11.0, start_at_s=1.0, stop_at_s=8.5, seed=7)
    h = GA.scissor_record(hb, hinfo)
    out = {"healthy": {k: _r(h[k], 4) for k in ("knee_distance_mid_swing_median_m", "knee_distance_mid_swing_min_m",
                                                 "hard_gap_min_m")}}
    out["healthy"]["kick_deg"] = _r(_excursion(GA, hb, hinfo), 3)

    def read(lv, table=None):
        old = GA.SCISSOR
        if table is not None:
            GA.SCISSOR = table
        try:
            b, info = GA.generate(GA.spastic_diplegic(lv), duration_s=10.0, start_at_s=1.0, stop_at_s=7.5, seed=3)
        finally:
            GA.SCISSOR = old
        r = GA.scissor_record(b, info)
        o = {k: _jsonable(r[k]) for k in ("knee_distance_mid_swing_median_m", "knee_distance_mid_swing_min_m",
                                          "hard_gap_min_m", "touch_share", "guard_added_abduction_deg")}
        o["kick_deg"] = _r(_excursion(GA, b, info), 3)
        o["guard_thigh_gap_min_m"] = _r(min(x["thigh_gap_m"] for x in info["leg_guard"]), 4)
        o["guard_swings_bumped"] = int(sum(x["added_deg"] > 0 for x in info["leg_guard"]))
        return o
    for lv in (2, 1):
        out[f"level_{lv}"] = read(lv)
    t = dict(GA.SCISSOR)
    t[2] = (t[2][0], 14.0) + tuple(t[2][2:])
    out["tries"] = {"level 2, swing adduction 14 deg": read(2, t)}
    out["test_bounds"] = {"knee_distance_below_healthy_by_m": 0.03, "kick_over_healthy_deg": 2.0,
                          "guard_median_below_deg": 2.0}
    return out


def sensory():
    """The sensory ataxic gait's read clearance as committed and under the tries of this part (none kept)."""
    from mdfm.synth import gait as GA
    orig = GA.sensory_ataxic
    out = {}
    for name, over in (("committed", {}), ("clearance 8.5 cm", {"clearance_m": (0.085, 0.085)}),
                       ("knee 92 deg", {"knee_swing_deg": (92.0, 92.0)}),
                       ("clearance bump width 0.30", {"clear_width": 0.30}),
                       ("bump at 0.45, width 0.24", {"clear_at": 0.45, "clear_width": 0.24})):
        def patched(eyes, over=over):
            p, kw = orig(eyes)
            return replace(p, **over), kw
        GA.sensory_ataxic = patched
        try:
            spec = next(s for s in GA.specs() if s.clip_id == "gait_E8_sensory_ataxic_eyes_open")
            b, tr = spec.build()
        finally:
            GA.sensory_ataxic = orig
        rb = tr["params"]["read_back"]
        out[name] = {k: _r(rb.get(k), 4) for k in ("clearance_left_m", "knee_swing_left_deg", "contact_pitch_left_deg")}
        out[name]["failed"] = _failed(tr.get("validity"))
    out["test_bound_m"] = 0.05
    return out


def scenes():
    """The chair readings the scene tests pinned on SMPL+H (tests/test_synth_scenes.py), read on the open body: the
    'wrist' armrests against the SH5 ones, the chair before SH5 against the sit, the forearm fit's profile and the SH5
    armrest's share of the contact patch, the 'wrist' fit's profile, and the arising chair's old backrest."""
    from mdfm.synth import hands as H
    from mdfm.synth import motions as MO
    from mdfm.synth import viewer as V
    from mdfm.synth.base import static_body
    out = {}
    fl = V.mesh_floor_y()
    b5 = V.chair_mesh(fl, fit="sh5")[0].reshape(-1, 8, 3)
    bw = V.chair_mesh(fl, fit="wrist")[0].reshape(-1, 8, 3)
    bx = V.chair_mesh(fl, fit="forearm")[0].reshape(-1, 8, 3)
    sit = static_body("sit", 0.01)
    P, J = V.knot_vertices(sit, [0])[0], sit.joints().numpy()[0]
    dom = V.dominant_joint("neutral")

    def contact(B, e, end=None):
        lo, hi = B.min(0), B.max(0)
        ov = (P[:, 0] >= lo[0]) & (P[:, 0] <= hi[0]) & (P[:, 2] >= lo[2]) & (P[:, 2] <= hi[2])
        along = (dom == e) & (P[:, 2] >= J[e, 2]) & (P[:, 2] <= J[e, 2] + 0.10)
        patch = along & (P[:, 1] <= P[along, 1].min() + V.CHAIR_ARM_PATCH_M)
        stop = hi[2] if end is None else end
        prof = [float(P[s_, 1].min() - hi[1]) for z in np.arange(J[e, 2], stop - 1e-9, 0.02)
                for s_ in [(dom == e) & ov & (P[:, 2] >= z) & (P[:, 2] < z + 0.02)] if s_.any()]
        return ov, float(ov[patch].mean()), float(P[V._arm_ids(e) & ov, 1].min() - hi[1]), prof
    for k, e in ((2, 18), (3, 19)):
        w = e + 2
        side = "left" if e == 18 else "right"
        ov, _, _, prof = contact(bw[k], e, J[w, 2])
        hand = np.isin(dom, (w,) + tuple(range(22, 52))) & ov
        along = (dom == e) & (P[:, 2] >= J[e, 2]) & (P[:, 2] <= J[w, 2])
        _, sh_f, gap_f, prof_f = contact(bx[k], e)
        _, sh_5, gap_5, prof_5 = contact(b5[k], e)
        out[side] = {"wrist_fit_longer_than_sh5_m": _r(bw[k][:, 2].max() - b5[k][:, 2].max(), 4),
                     "wrist_fit_top_over_sh5_top_m": _r(bw[k][:, 1].max() - b5[k][:, 1].max(), 4),
                     "forearm_length_z_m": _r(J[w, 2] - J[e, 2], 4),
                     "wrist_fit_profile_readings": len(prof),
                     "wrist_fit_profile_max_m": _r(max(prof), 4), "wrist_fit_hand_gap_m": _r(P[hand, 1].min()
                                                                                            - bw[k][:, 1].max(), 4),
                     "wrist_fit_forearm_share_over": _r(ov[along].mean(), 3),
                     "forearm_fit_patch_share": _r(sh_f, 3), "forearm_fit_profile_max_m": _r(max(prof_f), 5),
                     "sh5_patch_share": _r(sh_5, 3), "sh5_gap_m": _r(gap_5, 5)}
    keep, seatv, trunk = dom < 18, np.isin(dom, (0, 1, 2)), np.isin(dom, V.CHAIR_BACK_JOINTS)
    seat_top, x_half = float(bx[0][:, 1].max()), float(bx[0][:, 0].max())
    old_front = float(P[seatv, 2].min()) - 0.02
    ins = trunk & (P[:, 1] > seat_top) & (P[:, 1] < seat_top + 0.55) & (np.abs(P[:, 0]) < x_half)
    out["chair_before_sh5"] = {"backrest_front_minus_rearmost_spine_m": _r(old_front - float(P[ins, 2].min()), 5)}
    for k, e in ((2, 18), (3, 19)):
        old = np.array([[J[e, 0] - 0.04, seat_top, old_front - 0.05], [J[e, 0] + 0.04, b5[k][:, 1].max(),
                                                                        J[e, 2] + 0.10]])
        Q = P[np.isin(dom, (0,))]
        out["chair_before_sh5"][f"pelvis_depth_in_armrest_{k}_m"] = _r(np.minimum(Q - old.min(0), old.max(0)
                                                                                  - Q).min(-1).max(), 5)
    b, _ = MO.arise(standing_s=0.5)
    cv, _ = MO.arise_chair(b, V.mesh_floor_y(0.0))
    back = cv.reshape(-1, 8, 3)[1]
    seat = cv.reshape(-1, 8, 3)[0]
    ids = MO._seat_vertices()
    V0 = H.body_points(b, ids, knots=[0])[0]
    st = float(V0[:, 1].min()) - 0.005
    low = H.body_points(b, ids)[..., 1].min(1)
    seated = np.flatnonzero(low <= low[0] + 0.01)
    tr_ids = np.flatnonzero(np.isin(H._rest()[2], (0, 3, 6, 9)))
    Pa = H.body_points(b, tr_ids, knots=seated)
    xh = float(seat[:, 0].max())
    inside = (Pa[..., 1] > st) & (Pa[..., 1] < st + 0.55) & (np.abs(Pa[..., 0]) < xh)
    zmin = float(np.where(inside, Pa[..., 2], np.inf).min())
    out["arising_chair"] = {"old_backrest_front_minus_trunk_m": _r(float(V0[:, 2].min()) - 0.02 - zmin, 5),
                            "backrest_clear_m": _r(zmin - float(back[:, 2].max()), 5)}
    return out


# ---------------------------------------------------------------- the layer's refusals
def layer():
    from mdfm.synth import catalogue as C
    cat = {s.clip_id: s for s in C.catalogue_set()}
    out = {}
    for cid in LAYER_CLIPS:
        s = cat[cid]
        raw, truth = s.build.raw()
        b, tr = s.build.with_background(raw, truth)
        g = tr.get("background") or {}
        v = g.get("validity") or {}
        out[cid] = {"layer_ok": v.get("ok"), "failed": v.get("failed"),
                    "readings": {k: _jsonable(v.get(k)) for k in ("accel_excess_m_s2", "anchor_error_m",
                                                                    "limits_new", "body_added_m", "scene_added_m")
                                 if k in v},
                    "deviations": sorted((g.get("deviations") or {}).keys()),
                    "series_deviations": sorted((g.get("series_deviations") or {}).keys()),
                    "module_failed": _failed(tr.get("validity"))}
        if cid.startswith("post_E7_pd_posture"):
            from mdfm.synth.body import components_deg
            c = components_deg(raw.rot6d)
            out[cid]["knee_flexion_min_deg"] = _r(min(c[:, 4, 0].min(), c[:, 5, 0].min()), 4)
    return out


def test_readings():
    """The readings the re-derived tests write beside their assertions: the standing foot dystonia's inversion term
    on the soft stand, the supine spine's held row and the head it carries (atax_SARA8_heelshin_g2 against its level
    0), the standing tics' onsets and durations (jerk_tics_E7_complexity_c4), and the video driver's finger tapping
    offset on the synthetic body's SMPL+H reference (behind the flag, set here and removed after)."""
    from mdfm.synth import catalogue as C
    from mdfm.synth import jerks as JK
    from mdfm.synth import physio as PH
    from mdfm.synth import postures as PO
    from mdfm.synth.body import joint_limits
    out = {}
    _, tr = PO.foot_dystonia(3, standing=True, levels=PO.FOOT_STAND, duration_s=3.0, seed=2)
    inv = [t for t in tr["params"]["terms"] if "inversion" in t["label"]][0]
    out["foot_dystonia_standing"] = {"inversion_term_deg": _r(inv["deg"], 5),
                                     "straight_stand_value_deg": _r(-joint_limits().lo[7, 2] * PO.DIS_SHARE[3], 5),
                                     "room_deg": _r(inv["deg"] / PO.DIS_SHARE[3], 5)}
    cat = {s.clip_id: s for s in C.catalogue_set()}
    cid = "atax_SARA8_heelshin_g2"
    raw, t0 = cat[cid].build.raw()
    b, _ = cat[cid].build.with_background(raw, t0)
    raw0, t1 = cat[cat[cid].level0].build.raw()
    T = min(raw.T, raw0.T)
    rows = {}
    for j in (0, 3, 6, 9, 12, 15):
        held = PH.rows_equal(raw.rot6d[:, j:j + 1], raw0.rot6d[:, j:j + 1], T)[:, 0]
        d = (b.rot6d[:T, j] - raw.rot6d[:T, j]).abs().amax(-1).numpy()
        rows[str(j)] = {"held_knots": int(held.sum()), "largest_change": float(d.max()),
                        "largest_change_off_held": float(d[~held].max()) if (~held).any() else 0.0}
    out["supine_spine"] = {"rows": rows, "head_moved_m": float((b.joints() - raw.joints())[:, 15].norm(dim=-1).max()),
                           "equal_row_tol": PH.EQUAL_ROW_TOL}
    specs = {s.clip_id: s for s in JK.specs()}
    c = "jerk_tics_E7_complexity_c4"
    bj, tj = specs[c].build()
    base = specs[specs[c].level0].build()[0]
    ev = tj["params"]["events"]
    sites = sorted({p["site"] for e in ev for p in e["parts"]})
    m = JK.measure_jerks(bj, base, JK.site_joints(sites), pelvis=any(JK.SITES[x][1] == "dip" for x in sites),
                         rel=1e-4 if tj["params"].get("touch") else 1e-3)
    span = np.array([JK.event_span(e) for e in ev])
    out["standing_tics"] = [{"label": e["label"], "onset_late_s": _r(o - s_[0], 4),
                             "duration_short_s": _r((s_[1] - s_[0]) - d, 4)}
                            for e, o, s_, d in zip(ev, m["onset_s"], span, m["duration_s"])]
    from mdfm.synth import openbody as OB
    old = {k: os.environ.get(k) for k in (OB.REFERENCE_ENV, OB.BACKEND_ENV)}
    os.environ[OB.REFERENCE_ENV], os.environ[OB.BACKEND_ENV] = "1", OB.SMPLH_REFERENCE
    try:
        from mdfm.synth.handtasks import TASKS, hand_task
        from mdfm.synth.motions import posture_body
        from mdfm.video import exam_driver as X
        from mdfm.video import render as R
        hb, htr = hand_task(posture_body(TASKS["finger_tapping"]["posture"], 4.72), "finger_tapping", "left",
                            grade=2, criterion="decrement", seed=3)
        clip = X.body_clip(hb, htr)
        dc = R.render(clip, 60.0, clip.span_s, R.Framing(rule="torso", view="oblique_right60"), 0, 128, 96)
        rb = X.readback(dc.truth, 60.0, clip.source["operator_truth"], clip.source["events_clip_clock"])
        ops = np.array([cy["amplitude"] for cy in htr["params"]["cycles"]])
        off = np.array([cy["drawn_max_mm"] for cy in rb["tapping"]["cycles"]]) - ops
        out["video_driver_tapping"] = {"offset_median_mm": _r(np.median(off), 2),
                                       "spread_mm": _r(np.abs(off - np.median(off)).max(), 3),
                                       "drawn_min_mm": _r(rb["tapping"]["drawn_min_mm"], 2)}
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return out


def plantigrade_turns():
    """The plantigrade walk's users and two turning walks on it (the exam's FOG_WALK and the faster turns of
    tests/test_synth_pathology.py's walk): the components outside the anatomical table (deg, and the time of the
    largest) and the largest lag of a foot's heading behind the walk's (D164)."""
    from mdfm.synth import base as B
    from mdfm.synth import body as BD
    from mdfm.synth import exam as EX
    lim = BD.joint_limits()
    out = {"fog_walk": _jsonable(EX.FOG_WALK),
           "rule": "the shortest of 1 s steps from 4 s at which the walk's legs keep legcontact's hard tolerance "
                   "(design D135, reports/openbody/wave7_checks.json exam_refs.fog_walk)"}
    W = EX.FOG_WALK
    for name, kw in (("fog_walk", dict(duration_s=W["duration_s"], turns=W["turns"], stop_at_s=W["stop_at_s"])),
                     ("pathology_test_walk", dict(duration_s=40.0, turns=[(14.0, 180.0, 4.0), (26.0, -90.0, 2.5)],
                                                  stop_at_s=36.0))):
        b, info = B.walk(kw["duration_s"], turns=kw["turns"], stop_at_s=kw["stop_at_s"], kind="plantigrade")
        c = BD.components_deg(b.rot6d)
        ex = np.maximum(lim.lo[None] - c, c - lim.hi[None])[:, 1:]
        m = ex.max(0)
        out[name + "_reading"] = {
            "outside_deg": {f"{BD.BODY_JOINTS[j + 1]}.{'xyz'[k]}": [_r(m[j, k], 4),
                                                                      _r(np.argmax(ex[:, j, k]) / b.rate, 2)]
                            for j, k in zip(*np.where(m > 1e-6))},
            "foot_heading_lag_max_deg": [_r(x, 3) for x in np.degrees(np.abs(info["foot_heading_rad"]
                                                                              - info["heading_rad"][:, None])).max(0)]}
    return out


SECTIONS = {"steppage": steppage, "low_flat": low_flat, "antalgic": antalgic, "gap_clips": gap_clips,
            "hesitation": hesitation, "scissor": scissor, "sensory": sensory, "scenes": scenes,
            "test_readings": test_readings, "layer": layer, "plantigrade_turns": plantigrade_turns}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", nargs="*", default=None, choices=sorted(SECTIONS))
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    torch.set_num_threads(a.threads)
    rec = {}
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as f:
            rec = json.load(f)
    rec.setdefault("measured", {})
    for name in (a.only or list(SECTIONS)):
        t0 = time.time()
        rec[name] = _jsonable(SECTIONS[name]())
        rec["measured"][name] = {"mdfm_commit": _head(), "at": datetime.datetime.now().isoformat(timespec="seconds"),
                                 "seconds": round(time.time() - t0, 1)}
        with open(OUT, "w", encoding="utf-8", newline="\n") as f:
            json.dump(rec, f, indent=1)
        print("wrote", name, rec["measured"][name], flush=True)
    rec["script"] = "scripts/openbody_wave8b_checks.py"
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        json.dump(rec, f, indent=1)


if __name__ == "__main__":
    main()
