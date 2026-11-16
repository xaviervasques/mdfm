"""The open-body port's wave 3 readings that the design's section 14 cites (docs/OPEN_BODY_PORT_DESIGN.md): the bases
and operators of the involuntary-movement and jerk modules read on the selected body model (the open body by
default), each beside the value the module held before wave 3 where wave 3 changed it.

- the face touch (reach.touch_path, the right hand of the touching tics, every target on 'sit' and 'stand'): played as
  jerks.touch_timing plays it (a minimum-jerk share over the rise, on the 100 Hz clock), per arm joint the peak
  angular speed and the largest change of that speed from one knot to the next (the kink), with TOUCH_SMOOTH_WEIGHT 0
  (the sample-by-sample path, before) and the module's weight (after); the path's check (touch gap, head, trunk and
  thigh clearance, the elbow's drop, the joint peaks on the path's timing);
- the seated rest of the hands on the thighs (involuntary.solve_hands_on_thighs): the static sit's arm components
  against the table, the former rest (the SMPL+H numbers) read on the open body's mesh, the solve's record, the
  forearm's slope, the wrist's height over the hip-to-knee line, the hands' skin distance, the contact guard's
  reading of the rest, and the scan of the palm's lateral place (PALM_LATERAL_M: the centre patch's distance to the
  skin when the nearest palm vertex lies PALM_MIN_GAP_M off it, the guard's depth, and the centre patch's largest
  distance under the trunk's carriage of the test, a grade-2 trunk chorea); the rest thumb against the palm's plane;
- the open body's seated geometry the forearm slope's test cites (the shoulder over the thigh's top, the upper arm and
  forearm lengths);
- the seated knees (involuntary.closing_ceiling, legcontact): the ceiling and the hard and soft gaps the knee tests
  cite;
- moving toes: the toe block's ROM and extension room on the seated base, the share involuntary.toes_level gives;
- the standing jerks (jerks, every E7 clip of the catalogue): the balance block's margins of each built clip (the
  standing base is not centred over the feet, section 13.7), the hip strategy's backward shift and the pelvis's drop on
  the straight legs' arc outside the dips;
- the squat of the standing complexity clip at 15, 13 and 12 cm: the planted ankles' dorsiflexion and the knees;
- the generalized jerk's peaks with the arms' weight before and now, the neck myoclonus-dystonia peaks, the grade-5
  shrug's speed against the carry bound.
Writes reports/openbody/wave3_checks.json. CPU only; reads no dataset.
"""
import datetime
import json
import math
import os
import subprocess
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "reports", "openbody", "wave3_checks.json")
JOINTS = ("collar", "shoulder", "elbow", "wrist")
FORMER_REST = {"left": {"sh_flex": 6.84, "sh_abd": 7.77, "sh_rot": -38.2, "elbow": -36.91, "forearm": 13.41,
                        "wrist_flex": -23.92, "wrist_dev": -9.48},
               "right": {"sh_flex": 7.29, "sh_abd": 8.2, "sh_rot": -39.21, "elbow": -39.28, "forearm": 15.35,
                         "wrist_flex": -28.7, "wrist_dev": -6.6}}      # the SMPL+H rest of HANDS_ON_THIGHS
LATERALS = (0.008, 0.0, -0.005)                  # PALM_LATERAL_M before wave 3 (+8 mm), on the top, the module's


def _head():
    try:
        return subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def _r(x, n=2):
    return round(float(x), n)


def touch_play(RC, path, rate=100.0):
    """Per arm joint: the peak angular speed and the largest one-knot change of it (deg/s) on the clip's timing."""
    from mdfm.body.rotations import axis_angle_to_matrix, matrix_to_axis_angle
    from mdfm.synth.jerks import touch_timing
    from mdfm.synth.motions import minimum_jerk
    rise, _ = touch_timing(path, rate)
    n = int(round(rise * rate)) + 1
    q = RC.touch_q(path, minimum_jerk(np.linspace(0.0, 1.0, n)))
    R = axis_angle_to_matrix(torch.deg2rad(torch.as_tensor(q.reshape(-1, 4, 3))))
    out = {}
    for i, name in enumerate(JOINTS):
        D = R[1:, i] @ R[:-1, i].transpose(-1, -2)
        w = torch.rad2deg(matrix_to_axis_angle(D)).numpy() * rate
        dw = np.linalg.norm(np.diff(w, axis=0), axis=1)
        out[name] = {"peak_speed_deg_s": _r(np.linalg.norm(w, axis=1).max(), 1),
                     "largest_speed_change_deg_s": _r(dw.max() if len(dw) else 0.0, 1)}
    return {"rise_s": _r(rise, 2), "joints": out}


def touches():
    from mdfm.synth import reach as RC
    weight = RC.TOUCH_SMOOTH_WEIGHT
    rec = {"weight": weight, "paths": {}}
    for posture in RC.TOUCH_POSTURES:
        for target in RC.FACE_TARGETS:
            row = {}
            for label, w in (("before", 0.0), ("after", weight)):
                RC.TOUCH_SMOOTH_WEIGHT = w
                RC.touch_path.cache_clear()
                p = RC.touch_path("right", target, posture)
                c = p["check"]
                row[label] = dict(touch_play(RC, p), check={
                    "ok": c["ok"], "touch_gap_mm": _r(c["touch_gap_mm"]), "head_min_mm": _r(c["head_min_mm"]),
                    "trunk_min_mm": _r(c["trunk_min_mm"], 1), "thighs_min_mm": _r(c["thighs_min_mm"], 1),
                    "elbow_drop_m": _r(c["elbow_drop_m"], 4),
                    "joint_peak_deg_s": {k: _r(v, 1) for k, v in c["joint_peak_deg_s"].items()}},
                    q_touch_deg=[_r(x, 3) for x in p["q_touch_deg"]])
            row["q_touch_change_deg"] = _r(np.abs(np.subtract(row["after"]["q_touch_deg"],
                                                              row["before"]["q_touch_deg"])).max(), 4)
            rec["paths"][f"{posture}_{target}"] = row
            print(posture, target, {j: (row["before"]["joints"][j]["largest_speed_change_deg_s"],
                                        row["after"]["joints"][j]["largest_speed_change_deg_s"]) for j in JOINTS[1:]},
                  flush=True)
    RC.TOUCH_SMOOTH_WEIGHT = weight
    RC.touch_path.cache_clear()
    return rec


def _arm_read(I, b):
    from mdfm.synth import hands as H
    from mdfm.synth import viewer as V
    J = b.joints().numpy()[0]
    out = {}
    v = V.knot_vertices(b, [0])[0]
    dom = H._rest()[2]
    for s, (w, el, h, k) in (("left", (20, 18, 1, 4)), ("right", (21, 19, 2, 5))):
        fa = J[w] - J[el]
        u = (J[k] - J[h]) / np.linalg.norm(J[k] - J[h])
        d = J[w] - J[h]
        out[s] = {"forearm_slope_deg": _r(np.degrees(np.arctan2(-fa[1], np.linalg.norm(fa[[0, 2]]))), 2),
                  "wrist_over_hip_knee_line_mm": _r(1000 * np.linalg.norm(d - (d @ u) * u), 1)}
    hand = {s: np.flatnonzero(np.isin(dom, [20 + i] + list(range(22 + 15 * i, 37 + 15 * i)))) for i, s in
            enumerate(("left", "right"))}
    from scipy.spatial import cKDTree
    out["hands_skin_apart_mm"] = _r(1000 * cKDTree(v[hand["right"]]).query(v[hand["left"]])[0].min(), 1)
    g = I.palm_rest_report(b, [0])
    out["palm_rest"] = {s: {k: _r(x, 2) for k, x in g[s].items()} for s in ("left", "right")}
    out["palm_rest"]["ok"] = g["ok"]
    return out


def seated_rest():
    from mdfm.synth import involuntary as I
    from mdfm.synth.body import components_deg, joint_limits
    lim = joint_limits()
    rec = {"palm_pitch_deg": I.PALM_PITCH_DEG, "palm_contact_m": I.PALM_CONTACT_M, "rest_room_deg": I.REST_ROOM_DEG,
           "palm_lateral_m": I.PALM_LATERAL_M}
    s0 = I._sit_forward(0.02)
    c0 = components_deg(s0.rot6d)[0]
    rec["static_sit_arms"] = {str(j): {"deg": [_r(x) for x in c0[j]], "lo": [_r(x) for x in lim.lo[j]],
                                       "hi": [_r(x) for x in lim.hi[j]]} for j in (16, 17, 18, 19, 20, 21)}

    def with_rest(x):
        sig = {f"{k}_{s}": np.full(s0.T, v) for s in x for k, v in x[s].items()}
        out, _ = I.apply_offsets(s0, I.offsets_of(s0, sig))
        return out
    rec["former_rest_on_this_body"] = _arm_read(I, with_rest(FORMER_REST))
    x, solve = I.solve_hands_on_thighs()
    rec["solve"] = {s: {k: _r(v) for k, v in x[s].items()} for s in x}
    rec["solve_record"] = {s: {k: (_r(v, 6) if not isinstance(v, list) else [_r(a, 4) for a in v])
                               for k, v in solve[s].items()} for s in solve}
    rec["module_rest"] = {s: dict(I.HANDS_ON_THIGHS[s]) for s in I.HANDS_ON_THIGHS}
    b = I.base_body("sit", 0.02)
    rec["module_rest_read"] = _arm_read(I, b)
    from mdfm.synth import viewer as V
    Vk = V.knot_vertices(b, [0])
    rec["module_rest_guard_depth_mm"] = {s: _r(1000 * I._penetration(Vk, s, I._boxes(I.SCENE["sit"], b),
                                                                    I._target_ctx(Vk, heads=True))[0]) for s in ("left", "right")}
    cb = components_deg(b.rot6d)[0]
    rec["module_rest_arms_deg"] = {str(j): [_r(v) for v in cb[j]] for j in (16, 17, 18, 19, 20, 21)}
    J = s0.joints().numpy()[0]
    from mdfm.synth.body import rest_joints
    r = rest_joints().numpy()
    rec["geometry_m"] = {"upper_arm": _r(np.linalg.norm(r[18] - r[16]), 3), "forearm": _r(np.linalg.norm(r[20] - r[18]), 3),
                         "shoulder_over_thigh_top": _r(J[16, 1] - solve["left"]["thigh_top_m"][1], 3)}
    scan, lat = {}, I.PALM_LATERAL_M
    keep = {k: dict(v) for k, v in I.HANDS_ON_THIGHS.items()}
    for la in LATERALS:
        I.PALM_LATERAL_M = la
        x2, sr = I.solve_hands_on_thighs()
        I.HANDS_ON_THIGHS = x2
        bl = I.base_body("sit", 0.02)
        from mdfm.synth import viewer as V
        Vk = V.knot_vertices(bl, [0])
        pen = I._penetration(Vk, "left", I._boxes(I.SCENE["sit"], bl), I._target_ctx(Vk, heads=True))[0]
        scan[str(la)] = {"patch_median_mm": _r(1000 * sr["left"]["patch_median_m"]),
                         "palm_min_mm": _r(1000 * sr["left"]["palm_min_m"]),
                         "guard_depth_mm": _r(1000 * pen), "target_error_mm": _r(sr["left"]["target_error_mm"], 3)}
        out, tr = I.chorea("sit", 2, regions=("trunk",), seconds=8.0, seed=5)     # the test's trunk carriage
        rep = I.palm_rest_report(out, range(0, out.T, 20))
        scan[str(la)]["carriage_patch_median_mm_max"] = _r(max(rep[s]["patch_median_mm_max"] for s in ("left", "right")))
        print("lateral", la, scan[str(la)], flush=True)
    I.PALM_LATERAL_M, I.HANDS_ON_THIGHS = lat, keep
    rec["lateral_scan_left"] = scan
    return rec


def thumb():
    """The rest hand's thumb against the palm's plane (hands.hand_frame 'palmar'): the most palmar vertex of the palm
    (dominant wrist), of the thumb and of the four fingers, on the rest mesh, the thumb at rest and at
    involuntary.SEATED_THUMB_CMC."""
    from mdfm.synth import hands as H
    from mdfm.synth import involuntary as I
    J52, _, dom = H._rest()
    hf = H.hand_frame()
    s, w = "left", H.WRIST["left"]
    palm = np.flatnonzero(dom == w)
    th = np.flatnonzero(np.isin(dom, [H.hand_joint(s, "thumb", lv) for lv in range(3)]))
    ids = np.concatenate([palm, th])
    R = np.eye(3)[None, None].repeat(22, 1)
    rec = {}
    for label, cmc in (("rest", (0.0, 0.0)), ("seated", I.SEATED_THUMB_CMC)):
        Rh = H.hand_rotations({(s, "thumb", 0): cmc})[None].numpy()
        X = H.skin_ids(R, np.zeros((1, 3)), np.zeros(16), 2, ids, Rh)[0]
        hgt = (X - J52[w]) @ hf[s]["palmar"]
        rec[label] = {"cmc_deg": list(cmc), "palm_most_palmar_mm": _r(1000 * hgt[:len(palm)].max()),
                      "thumb_most_palmar_mm": _r(1000 * hgt[len(palm):].max())}
    print("thumb", rec, flush=True)
    return rec


def toes():
    from mdfm.synth import involuntary as I
    b = I.base_body("sit", 0.02)
    room = I.channel_room(b, "toes_left")
    return {"rom_deg": _r(I.channel_rom("toes_left")), "room_extension_deg": _r(room[0]), "room_flexion_deg": _r(room[1]),
            "soft_margin": I.SOFT_MARGIN, "levels": {str(g): _r(I.toes_level(g), 4) for g in (2, 3)},
            "dis_levels": {str(g): I.DIS_LEVELS[g] for g in (2, 3)}}


def knees():
    """The seated knees' closing on this body: the closing ceiling of the seated base, the hard and soft leg-to-leg
    gaps (legcontact, exact to 0.2 m) at closings of 12 and 20 deg split 7 to 5 (the knee guard test's bump), and at
    the ceiling plus 1 deg on each of CLOSING_SPLITS."""
    from mdfm.synth import involuntary as I
    from mdfm.synth.legcontact import gap_from_pose
    base = I.base_body("sit", 0.01)
    T = base.T

    def gap(d):
        b = base
        for side, m in d.items():
            R, _ = I.seated_legs(b, side, swivel_deg=-I.SWIVEL_OUT[side] * np.full(T, m))
            b = b.with_rotations(R, list(I.LEG[side][:3]))
        g = gap_from_pose(b.local_R().numpy(), b.transl.numpy(), b.betas.numpy(), int(b.gender), cull_m=0.2)
        return {"hard_mm": _r(1000 * g["hard_m"].min(), 1), "soft_mm": _r(1000 * g["soft_m"].min(), 1)}
    c = I.closing_ceiling(base, 0)
    rec = {"ceiling_deg": _r(c, 3), "keep_mm": _r(1000 * (I.KNEE_CLEARANCE_M + I.KNEE_GUARD_MARGIN_M), 1),
           "rest": gap({"left": 0.0, "right": 0.0}),
           "closing_7_5": {str(S): gap({"left": 7.0 / 12.0 * S, "right": 5.0 / 12.0 * S}) for S in (12.0, 20.0)},
           "ceiling_plus_1": {f"{a},{o}": gap({"left": a * (c + 1.0) + o, "right": (c + 1.0) - (a * (c + 1.0) + o)})
                              for a, o in I.CLOSING_SPLITS}}
    print("knees", rec, flush=True)
    return rec


def standing_jerks():
    """Every standing (E7) jerk clip: the balance block of its truth, and, replaying its events on its level 0
    (jerks.apply_events, standing), the hip strategy's largest backward pelvis shift and the pelvis's largest drop
    outside the dip sites (the straight legs' arc, jerks._move_pelvis)."""
    from mdfm.synth import jerks as JK
    specs = {s.clip_id: s for s in __import__("mdfm.synth.catalogue", fromlist=["x"]).module_specs("jerks")}
    built = {}

    def get(c):
        if c not in built:
            built[c] = specs[c].build()
        return built[c]
    out = {}
    for cid, s in specs.items():
        if "_E7" not in cid:
            continue
        b, t = get(cid)
        bb = (t.get("validity") or {}).get("balance")
        if not bb:
            out[cid] = None
            continue
        row = {k: (_r(bb[k], 4) if isinstance(bb[k], float) else bb[k]) for k in
               ("ok", "xcom_margin_posterior_m", "xcom_margin_anterior_m", "base_margin_posterior_m",
                "base_margin_anterior_m", "friction_max", "vertical_grf_min_bw")}
        base = get(s.level0)[0]
        o, prof, back = JK.apply_events(base, t["params"]["events"], True)
        dy = 100.0 * (base.joints()[:, 0, 1] - o.joints()[:, 0, 1]).numpy()
        free = np.ones(len(dy), bool)
        for k in prof:
            if JK.SITES[k][1] == "dip":
                free &= np.abs(prof[k]) == 0
        row["hip_strategy_back_cm_max"] = _r(100.0 * np.abs(back).max(), 2)
        row["pelvis_drop_outside_dips_cm_max"] = _r(dy[free].max() if free.any() else 0.0, 4)
        out[cid] = row
        print(cid, row, flush=True)
    vals = [v for v in out.values() if v]
    return {"clips": out, "n": len(vals), "all_ok": all(v["ok"] for v in vals),
            "least_posterior_m": min(v["xcom_margin_posterior_m"] for v in vals) if vals else None,
            "least_anterior_m": min(v["xcom_margin_anterior_m"] for v in vals) if vals else None}


def squat(depths=(15.0, 13.0, 12.0)):
    """The standing complexity clip's squat at several depths (jerks.SQUAT_CM): the planted ankles' largest
    dorsiflexion (deg, the x component's most negative value), the squat's knee angle against the level 0 and the
    clip's new limit violations."""
    from mdfm.synth import catalogue as CAT
    from mdfm.synth import jerks as JK
    from mdfm.synth.body import components_deg
    specs = {s.clip_id: s for s in CAT.module_specs("jerks")}
    base = specs["jerk_E7_level0"].build()[0]
    keep, rec = JK.SQUAT_CM, {"module_cm": JK.SQUAT_CM, "depths": {}}
    for cm in depths:
        JK.SQUAT_CM = cm
        b, t = specs["jerk_tics_E7_complexity_c4"].build()
        knees = []
        for e in t["params"]["events"]:
            if e["label"] == "complex:squat":
                dip = next(q for q in e["parts"] if q["site"] == "knee_dip")
                k = int(round((e["onset_s"] + dip["rise_s"] + 0.5 * dip["hold_s"]) * 100))
                knees.append(float(JK.deviation_deg(b, base, [4, 5])[k].max()))
        rec["depths"][str(cm)] = {"ankle_dorsiflexion_deg_max": _r(-components_deg(b.rot6d)[:, [7, 8], 0].min(), 2),
                                  "squat_knee_deg": _r(max(knees), 1), "limits_new": t["validity"]["limits_new"]}
        print("squat", cm, rec["depths"][str(cm)], flush=True)
    JK.SQUAT_CM = keep
    return rec


def generalized():
    """The generalized reticular jerk (jerk_myoclonus_E1_generalized_f3_a2) and the neck myoclonus-dystonia clip: their
    nominal peaks (deg), with the arms' weight of GENERALIZED_WEIGHTS before wave 3 (0.5) and now."""
    from mdfm.synth import catalogue as CAT
    from mdfm.synth import jerks as JK
    specs = {s.clip_id: s for s in CAT.module_specs("jerks")}
    rec = {}
    w = dict(JK.GENERALIZED_WEIGHTS)
    for label, sh in (("before", 0.5), ("module", w["shoulder_abduction_left"])):
        JK.GENERALIZED_WEIGHTS.update(shoulder_abduction_left=sh, shoulder_abduction_right=sh)
        rec[label] = {k: _r(v, 2) for k, v in specs["jerk_myoclonus_E1_generalized_f3_a2"].build()[1]["params"]
                      ["peaks_deg"].items()}
    JK.GENERALIZED_WEIGHTS.update(w)
    rec["mdyst_neck"] = {k: _r(v, 2) for k, v in specs["jerk_mdyst_E7_neck_f3_a2"].build()[1]["params"]["peaks_deg"].items()}
    t5 = specs["jerk_tics_E1_intensity_i5"].build()[1]["params"]
    rec["intensity_i5_shrug"] = {"peak_speed_deg_s": _r(t5["sites"]["shrug_left"]["peak_speed"], 1),
                                 "carried_share_left": _r(t5["carry_arms"].get("shrug_carried_share", {}).get("left", 0.0), 3),
                                 "bound_deg_s": JK.CARRY_SHRUG_MAX_DEG_S}
    print("generalized", rec, flush=True)
    return rec


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    torch.set_num_threads(4)
    from mdfm.synth import body as B
    rec = {"script": "scripts/openbody_wave3_checks.py", "run_at": datetime.datetime.now().isoformat(timespec="seconds"),
           "mdfm_commit": _head(), "body_model": B.model_name()}
    rec["touch"] = touches()
    rec["seated_rest"] = seated_rest()
    rec["thumb"] = thumb()
    rec["moving_toes"] = toes()
    rec["knees"] = knees()
    rec["standing_jerks"] = standing_jerks()
    rec["squat"] = squat()
    rec["generalized_and_neck"] = generalized()
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        json.dump(rec, f, indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
