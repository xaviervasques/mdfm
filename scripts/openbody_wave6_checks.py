"""The open-body port's wave 6 readings that the design's section 17 cites (docs/OPEN_BODY_PORT_DESIGN.md): the
arising (motions.arise) and the folded arms on the selected body model (the open body by default), the hands of the
3.15 raise and of the 3.14 seated rest, the folded grip's thumb and the paroxysmal kinesigenic clips, each beside the
rule the module held before wave 6 where wave 6 changed it.

- arise_balance: the XcoM margin ahead of the heels from lift-off (motions.balance) at the whole room of the trunk
  bout, on the fold of the record, for the arising of wave 4 (the slumped seat of 'sit', the bout peaking and
  returning at the shank's peak), the upright seat alone, the upright seat with the later return, the upright seat
  with the bout peaking with the hip but returning at the shank's peak; the module's solve (its amplitude, realized
  lean and lean speed, the margins); the folded forearms' and hands' least skin gap to the thighs over the rise
  (motions.fold_gaps) for the bout of the shank's peak at the whole room and for the module's; the standing shank
  tilt and the peak ankle dorsiflexion it gives (the tilt plus ANKLE_PLANTAR_DEG) against the table's bound;
- reference_arise (a child process on the SMPL+H reference, MDFM_BODY_MODEL=smplh_reference with its flag, the SH9
  fold of the record before wave 6, read from git): the same margins of the wave 4 arising on the anatomical table,
  so that the refusal is read on both bodies; numbers only;
- fold_targets: the SH9 joint targets (the SMPL+H numbers, from spine3) carried to the open body by the chest they
  lie on (each target's height below the shoulder line as a share of the hips-to-shoulders height, its depth in front
  of the chest's front skin and its x kept):
  the SMPL+H frame read in a child process on the reference, the open body's here; the result beside FOLD_SOLVE;
- grip_thumb: the folded grip's thumb against its own palm and index and middle fingers (motions.skin_gap) for the
  CMC flexion of SH9 (20 deg) and the 10 deg steps below it, MCP and IP kept;
- outstretched: the 3.15 hand's finger gaps at rest and the spread's steps (motions.outstretched_spread);
- thigh_hands: the 3.14 seated rest's left hand (parkinsonism.seated_rest) for RELAXED_HAND's thumb abduction shifted
  by 0 to -50 deg: each hand part's lowest point over the thigh (parkinsonism._gap_above);
- fold: the record's checks (reports/synth/fold/fold_posture.json);
- ataxia: the stance's farther knee at REACH_ST on the open body's straight rest legs (0.997 before, the module's
  value now); the supine rest's solve (its spinal extension and each part's lowest point over the couch); the search
  of SPINE_BACK_TURN on the anatomical table (the shares of joints 3 and 6 on a grid, joint 9 the rest, the pelvis
  turned by PELVIS_TURN_DEG: joints 3 and 9 at least 1 deg inside the table and joint 6 inside it, the least slide of
  the upper body's resting vertices); the heel-shin's placement margins (at the knee, past it by HS_PAST_MAX_M and by
  the largest grade-3 error);
- pkd: the two PKD clips' builds (postures.paroxysmal_kinesigenic, the catalogue's seeds): their validity entries.
Wave 6 follow-up (--follow-up [section ...] writes these sections, or the ones named, into the same record, the rest
kept; 'measured' gives each section's commit and time):
- push_off: a failed push-off of 3.9 at the largest trunk share and peak (PUSH_TRUNK_RANGE, PUSH_FAIL_U) with the hands
  at WRIST_BACK_M 0.08, 0.04 and 0 behind the armrests' front ends and the elbows' POLE_OUT 0.3, 0.4 and 0.5: the knots
  and arm components outside the table (parkinsonism.push_arms), the forearms' and hands' least gap to the thighs; the
  push rise of grade 2 as built (its release, the arms read as used) with its arms solved again at POLE_OUT 0.4 and 0.5
  up to its over-reach: the first knot outside the table and the components outside (with the first knot of each);
  push_hand_yaw's record; the 'normal' template's pelvis lift over its seated knot at shares 0.54 to 0.74;
- sitdown: the controlled sit-down's XcoM margin to the seat contact on the slowest clock without the hip lean, and the
  solve's lean and stretch (parkinsonism.controlled_descent_u);
- pkd_opening: the opening of the fold at the PKD unfold (motions.unfold_to_stand) over 0.6, 0.7 and 0.8 s: each arm
  joint's largest step in one knot beyond the arising's; the two PKD clips as built (the catalogue's seed): the arm
  joints' largest step beyond the arising's, the opening read on the final body (motions.unfold_check), the attack's
  onset, the guards' records, the realized deviations and the validity entries that fail; the dystonic clip again
  without the guard of the arms against each other over the opening (_arms_need) and with the trunk guard on the
  shoulder alone (PKD_TRUNK_GUARD_ROWS before), in worker processes;
- stand_kink: the standing 3.14 ladder's level 0 (parkinsonism.spontaneity_parts) with 'stand''s legs at full length
  (base.STAND_REACH 1) and at the module's STAND_REACH (in worker processes): the legs' largest joint acceleration
  (one-knot second difference), where, and 'stand''s hip, knee and ankle angles;
- applause: the applause of score 0: the arm joints' largest acceleration over the raise (as its test reads it, to the
  knot where the claps begin) and at that knot; claps, approaches, the raise's least palms' distance, the raise's path
  record;
- gowers_jolts: the module's Gowers build and those of GOWERS_JOLT_SCAN (the spine's cap of before or softened, the
  step's move share): the arm joints' and knees' largest one-knot acceleration, where, in which phase, the knots past
  its test's bounds (6 and 1 m/s2), the hip midpoint's acceleration at the knees' knot, the planted hands' gaps, the
  balance, the lean and the validity entries that fail (in worker processes);
- gowers: Gowers' manoeuvre at the forward lean's pelvis pitch (GOWERS_LEAN) and the slide forward on the seat
  (GOWERS_SCOOT_M) of GOWERS_SCAN, the module's last: the XcoM margins, the trunk lean at lift-off and its peak, the
  hands' lateral shift, the joint components outside the table, the failing validity entries, or the build's error
  (in worker processes).
Writes reports/openbody/wave6_checks.json. CPU only; reads no dataset.
"""
import datetime
import json
import os
import subprocess
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "reports", "openbody", "wave6_checks.json")
SH9_COMMIT = "c633f2f"          # the last commit whose fold record holds the SH9 fold solved on SMPL+H
PARTS = ("wrist", "index2", "middle2", "ring2", "pinky2", "thumb1", "thumb2")


def _head():
    try:
        return subprocess.run(["git", "-c", f"safe.directory={ROOT.replace(os.sep, '/')}", "-C", ROOT, "rev-parse",
                               "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def _r(x, n=2):
    return round(float(x), n)


def _sh9_fold():
    root = ROOT.replace(os.sep, "/")      # a worktree outside the tree meets git's ownership check without this
    txt = subprocess.run(["git", "-c", f"safe.directory={root}", "-C", ROOT, "show",
                          f"{SH9_COMMIT}:reports/synth/fold/fold_posture.json"],
                         capture_output=True, text=True, check=True, encoding="utf-8").stdout
    rec = json.loads(txt)
    return {s: tuple(tuple(float(v) for v in a) for a in rec["fold"][s]) for s in ("left", "right")}


def _legs(b, info):
    """The seated knot's thigh slope (deg, the hip-to-knee line below the horizontal toward the knee, mean of both
    sides) and the rest knee joint's height over the rest sole (m)."""
    from mdfm.synth import hands as H
    from mdfm.synth.contact import sole_vertices
    J = b.joints().numpy()[int(round(info["onset_s"] * b.rate))]
    sl = [float(np.degrees(np.arctan2(J[h, 1] - J[k, 1], abs(J[k, 2] - J[h, 2])))) for h, k in ((1, 4), (2, 5))]
    J52, V, _ = H._rest()
    sv = sole_vertices(2)
    return {"seated_thigh_slope_deg": _r(np.mean(sl)),
            "knee_height_m": _r(float(J52[4, 1] - V[sv["left"], 1].min()), 4)}


def _thigh_gap(M, b, info):
    k_on, k_end = int(round(info["onset_s"] * b.rate)), int(round(info["end_s"] * b.rate))
    g, _ = M.fold_gaps(b, np.arange(k_on, k_end + 1), None, ("thigh_left", "thigh_right"))
    return {c: _r(1000 * float(v.min())) for c, v in g.items()}


def _margins(M, fold, trunk_deg, tilt=None, thighs=False):
    b, info = M.arise(standing_s=0.5, fold=fold, trunk_deg=trunk_deg, stand_tilt_deg=tilt)
    bl = info["balance"]
    out = {"trunk_deg": _r(trunk_deg, 3), "peak_share": M.TRUNK_PEAK_SHARE, "start_share": _r(M.TRUNK_START_SHARE, 4),
           "return_share": _r(M.TRUNK_RETURN_SHARE, 4), "liftoff_margin_mm": _r(1000 * bl["liftoff_margin_m"]),
            "min_margin_mm": _r(1000 * bl["min_margin_from_liftoff_m"]), "min_at_s": bl["min_margin_s"],
            "standing_shank_tilt_deg": _r(info["standing_shank_tilt_deg"], 3),
            "lean_peak_deg": _r(info["trunk"]["realized_lean_peak_deg"]),
            "lean_speed_peak_deg_s": _r(info["trunk"]["realized_lean_speed_peak_deg_s"], 1),
            "limits_outside": info["limits"]["outside"]}
    if thighs:
        out["fold_thigh_min_mm"] = _thigh_gap(M, b, info)
    return out, info


def _shank_peak(M):
    """The bout's timing before wave 6: its flexion peaking and returning at the shank's peak forward tilt, its speed
    at lift-off."""
    return {"TRUNK_PEAK_SHARE": M.ANKLE_PEAK_SHARE, "TRUNK_START_SHARE": 2 * M.LIFTOFF_SHARE - M.ANKLE_PEAK_SHARE,
            "TRUNK_RETURN_SHARE": M.ANKLE_PEAK_SHARE}


def _wave4_rules(M):
    """The arising's rules before wave 6, set on the module: the slumped seat of 'sit' and the shank's peak timing."""
    from mdfm.body.rotations import matrix_to_axis_angle
    from mdfm.synth.base import posture_rotations
    sit = np.degrees(matrix_to_axis_angle(posture_rotations("sit")).numpy())
    return dict(_shank_peak(M), SEAT_TRUNK_DEG=(float(sit[0][0]), float(sit[3][0])))


def _with(M, values, fn):
    keep = {k: getattr(M, k) for k in values}
    try:
        for k, v in values.items():
            setattr(M, k, v)
        return fn()
    finally:
        for k, v in keep.items():
            setattr(M, k, v)


def arise_balance():
    from mdfm.synth import motions as M
    from mdfm.synth.body import joint_limits
    fold = M.ARM_POSTURES["fold"]
    old = _wave4_rules(M)
    out = {"rules_before": {k: list(v) if isinstance(v, tuple) else v for k, v in old.items()}}
    full = lambda: float(M.trunk_split()[1].sum())      # noqa: E731
    room = lambda: [_r(x) for x in M.trunk_split()[1]]  # noqa: E731
    out["wave4"] = _with(M, old, lambda: dict(_margins(M, fold, full())[0], room_deg=room()))
    out["upright_seat"] = _with(M, _shank_peak(M), lambda: dict(_margins(M, fold, full(), thighs=True)[0],
                                                               room_deg=room()))
    late = {k: v for k, v in _shank_peak(M).items() if k != "TRUNK_RETURN_SHARE"}
    out["upright_seat_later_return"] = _with(M, late, lambda: _margins(M, fold, full())[0])
    out["upright_seat_hip_peak_shank_return"] = _with(M, {"TRUNK_RETURN_SHARE": M.ANKLE_PEAK_SHARE},
                                                      lambda: _margins(M, fold, full())[0])
    b, info = M.arise(standing_s=0.5)
    tk, bl = info["trunk"], info["balance"]
    lim = joint_limits()
    out["solved"] = {"amplitude_deg": _r(tk["amplitude_deg"], 3), "peak_share": tk["peak_share"],
                     "start_share": _r(tk["start_share"], 4), "return_share": _r(tk["return_share"], 4),
                     "room_deg": [_r(x) for x in tk["room_deg"]],
                     "lean_seated_deg": _r(tk["realized_lean_seated_deg"]), "lean_peak_deg": _r(tk["realized_lean_peak_deg"]),
                     "lean_speed_peak_deg_s": _r(tk["realized_lean_speed_peak_deg_s"], 1),
                     "liftoff_margin_mm": _r(1000 * bl["liftoff_margin_m"]),
                     "min_margin_mm": _r(1000 * bl["min_margin_from_liftoff_m"]), "min_at_s": bl["min_margin_s"],
                     "liftoff_com_vz_m_s": _r(bl["liftoff_com_vz_m_s"], 3),
                     "standing_com_ahead_of_ankles_mm": _r(1000 * bl["standing_com_ahead_of_ankles_m"]),
                     "standing_shank_tilt_deg": _r(info["standing_shank_tilt_deg"], 3),
                     "peak_dorsiflexion_deg": _r(info["standing_shank_tilt_deg"] + M.ANKLE_PLANTAR_DEG, 3),
                     "dorsiflexion_bound_deg": _r(-float(lim.lo[7][0]), 3),
                     "seated_knee_flexion_deg": _r(info["seated_knee_flexion_deg"]),
                     "realized_hip_flexion_displacement_deg": _r(info["realized_hip_flexion_displacement_deg"], 3),
                     "fold_thigh_min_mm": _thigh_gap(M, b, info), "limits_outside": info["limits"]["outside"],
                     "seat_contact_hip_drop_mm": [_r(1000 * x) for x in info.get("seat_contact_hip_drop_m", [0, 0])],
                     **_legs(b, info)}
    return out


def reference_arise():
    env = dict(os.environ, MDFM_BODY_MODEL="smplh_reference", MDFM_SMPLH_REFERENCE="1")
    r = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), "--reference"], capture_output=True,
                       text=True, env=env, encoding="utf-8")
    if r.returncode:
        return {"error": r.stderr[-2000:]}
    return json.loads(r.stdout.strip().splitlines()[-1])


def _chest_frame():
    """(the shoulder line above spine3 m, the hips-to-shoulders height m, front(y): the chest's front skin's depth
    from spine3 at a height y from spine3) of the rest mesh."""
    from mdfm.synth import hands as H
    J, V, dom = H._rest()
    tr = np.isin(dom, (0, 3, 6, 9, 12, 13, 14))
    sh_y = 0.5 * (J[16, 1] + J[17, 1])
    sh, th = float(sh_y - J[9, 1]), float(sh_y - 0.5 * (J[1, 1] + J[2, 1]))

    def front(y):
        sel = tr & (np.abs(V[:, 1] - (J[9, 1] + y)) < 0.01) & (np.abs(V[:, 0]) < 0.12)
        return float(V[sel, 2].max() - J[9, 2])
    return sh, th, front


SH9_TARGETS = {"left": {"E": (0.157, 0.018, 0.180), "W": (-0.083, 0.063, 0.245), "K": (-0.173, 0.063, 0.190)},
               "right": {"E": (-0.193, -0.047, 0.130), "W": (0.057, -0.022, 0.185), "K": (0.157, -0.032, 0.150)}}
#   the SH9 joint targets as motions.FOLD_SOLVE held them before wave 6 (SMPL+H, m from spine3)


def _reference_main():
    from mdfm.synth import motions as M
    torch.set_num_threads(4)
    sh, th, front = _chest_frame()
    rel = {s: {k: [v[0], (v[1] - sh) / th, v[2] - front(v[1])] for k, v in d.items()} for s, d in SH9_TARGETS.items()}
    fold = _sh9_fold()
    old = _wave4_rules(M)
    full = lambda: float(M.trunk_split()[1].sum())      # noqa: E731
    wave4 = _with(M, old, lambda: dict(_margins(M, fold, full())[0], room_deg=[_r(x) for x in M.trunk_split()[1]]))
    wave4_0 = _with(M, old, lambda: _margins(M, fold, 0.0)[0])
    b, info = _with(M, old, lambda: M.arise(standing_s=0.5, fold=fold, trunk_deg=0.0))
    legs = _legs(b, info)
    legs["stand_ref_com_ahead_of_ankles_mm"] = _r(1000 * info["balance"]["stand_ref_com_ahead_of_ankles_m"])
    print(json.dumps({"body_model": "smplh_reference", "fold": f"SH9, {SH9_COMMIT}:reports/synth/fold/fold_posture.json",
                      "shoulder_line_above_spine3_m": _r(sh, 4), "hips_to_shoulders_m": _r(th, 4),
                      "chest_front_at_spine3_m": _r(front(0.0), 4), "targets_rel": rel,
                      "wave4_whole_room": wave4, "wave4_no_bout": wave4_0, "legs": legs}))


def fold_targets(ref):
    from mdfm.synth import motions as M
    sh, th, front = _chest_frame()
    rel = ref.get("targets_rel")
    if rel is None:
        return {"error": "no reference frame"}
    tg = {}
    for s, d in rel.items():
        tg[s] = {}
        for k, (x, ys, zf) in d.items():
            y = ys * th + sh
            tg[s][k] = [round(x, 3), round(y, 3), round(front(y) + zf, 3)]
    same = all(tuple(tg[s][k]) == tuple(M.FOLD_SOLVE["targets"][s][k]) for s in tg for k in tg[s])
    return {"open_shoulder_line_above_spine3_m": _r(sh, 4), "open_hips_to_shoulders_m": _r(th, 4),
            "reference_hips_to_shoulders_m": ref.get("hips_to_shoulders_m"),
            "open_chest_front_at_spine3_m": _r(front(0.0), 4),
            "reference_shoulder_line_above_spine3_m": ref.get("shoulder_line_above_spine3_m"),
            "reference_chest_front_at_spine3_m": ref.get("chest_front_at_spine3_m"),
            "targets": tg, "equal_to_FOLD_SOLVE": bool(same)}


def grip_thumb():
    from mdfm.body.rotations import matrix_to_rot6d
    from mdfm.synth import hands as H
    from mdfm.synth import motions as M
    from mdfm.synth.base import static_body
    from mdfm.synth.viewer import faces
    _, _, dom = H._rest()
    F = faces()

    def part(joints):
        ids = np.flatnonzero(np.isin(dom, joints))
        loc = -np.ones(len(dom), np.int64)
        loc[ids] = np.arange(len(ids))
        return ids, loc[F[(loc[F] >= 0).all(1)]]
    side = "left"
    th = part([H.hand_joint(side, "thumb", k) for k in (1, 2)])
    fi = part([H.hand_joint(side, f, k) for f in ("index", "middle") for k in (0, 1, 2)])
    pa = part([H.WRIST[side]])
    b0 = H.hands_of(static_body("stand", 0.02))
    _, mcp, ip = M.GRIP["thumb"]
    out = {"module_cmc_flexion_deg": M.GRIP["thumb"][0], "mcp_ip_deg": [mcp, ip], "scan": []}
    for cmc in (20.0, 10.0, 0.0, -10.0):
        pose = M.closed_pose("grip", side)
        pose[(side, "thumb", 0)] = cmc
        Rh = H.hand_rotations(pose)
        b = H.HandBody(b0.rot6d, b0.transl, b0.contact, rate=b0.rate,
                       hand6d=matrix_to_rot6d(Rh)[None].expand(b0.T, -1, 6).clone())
        X = H.body_points(b, np.r_[th[0], fi[0], pa[0]], knots=[0])[0]
        n1, n2 = len(th[0]), len(fi[0])
        out["scan"].append({"cmc_flexion_deg": cmc,
                            "thumb_to_index_middle_mm": _r(1000 * M.skin_gap(X[:n1], th[1], X[n1:n1 + n2], fi[1])),
                            "thumb_to_palm_mm": _r(1000 * M.skin_gap(X[:n1], th[1], X[n1 + n2:], pa[1]))})
    return out


def outstretched():
    from mdfm.synth import motions as M
    deg, rec = M.outstretched_spread()
    b, info = M.outstretched(3.0)
    return {"spread_deg": deg, "tried": [{k: _r(v, 3) for k, v in t.items()} for t in rec["tried"]],
            "apart_mm": rec["apart_mm"],
            "hold_gaps_mm": {s: {k: _r(v, 3) for k, v in g.items()} for s, g in info["finger_gaps_mm"].items()},
            "limits_outside": info["limits"]["outside"]}


def thigh_hands():
    from mdfm.synth import hands as H
    from mdfm.synth import parkinsonism as PK
    _, _, dom = H._rest()
    names = {H.hand_joint("left", f, lv): f"{f}{lv}" for f in H.FINGERS for lv in range(3)}
    names[H.WRIST["left"]] = "wrist"
    R0, s0 = dict(PK.RELAXED_HAND), PK.THUMB_ABD_SHIFT_DEG
    out = {"module_shift_deg": s0, "hand_weight": PK.THIGH_REST["hand_weight"], "scan": []}
    try:
        for sh in (0.0, -10.0, -20.0, -30.0, -40.0, -45.0, -50.0):
            (fl, ab), m, i = R0["thumb"]
            PK.RELAXED_HAND = dict(R0, thumb=((fl, ab - s0 + sh), m, i))
            PK.seated_rest.cache_clear()
            sr = PK.seated_rest()
            b = PK._body2(sr["R"][None], sr["tr"][None], sr["Rh"][None])
            ids = PK._hand_ids("left")
            P = H.body_points(b, ids, knots=[0])[0]
            S = H.body_points(b, np.flatnonzero(np.isin(dom, (1, 4))), knots=[0])[0]
            low = {names[int(j)]: _r(1000 * PK._gap_above(P[dom[ids] == j], S), 1) for j in np.unique(dom[ids])}
            out["scan"].append({"shift_deg": sh, "over_thigh_mm": {k: low[k] for k in PARTS},
                                "lowest": min(PARTS, key=lambda k: low[k])})
    finally:
        PK.RELAXED_HAND = R0
        PK.seated_rest.cache_clear()
    return out


def fold():
    from mdfm.synth import motions as M
    with open(M.fold_record_path(), encoding="utf-8") as f:
        rec = json.load(f)
    v = rec["verify"]
    keys = ("forearms", "hand_left_forearm_right", "hand_right_forearm_left", "hands", "hand_left_upper_right",
            "hand_right_upper_left", "trunk_left", "trunk_right", "thigh_left", "thigh_right")
    return {"state": M._FOLD_STATE.get("state"), "fold": rec["fold"], "solve_s": rec["solve_s"],
            "stand_min_mm": {k: v["stand"][k]["min_mm"] for k in keys},
            "rise_min_mm": {k: v["rise"][k]["min_mm"] for k in keys}, "ok": [v["stand"]["ok"], v["rise"]["ok"]],
            "entry_ok": [v["stand"]["entry_ok"], v["rise"]["entry_ok"]], "limits_outside": v["limits_outside"],
            "arise": v["arise"], "first_stage_gaps_mm": {k: _r(x) for k, x in rec["solve"]["gaps_mm"]["ik"].items()},
            "result_gaps_mm": {k: _r(x) for k, x in rec["solve"]["gaps_mm"]["result"].items()}}


def _knee_at(reach):
    import math
    from mdfm.synth.body import rest_joints
    r = rest_joints().numpy()
    L1, L2 = float(np.linalg.norm(r[4] - r[1])), float(np.linalg.norm(r[7] - r[4]))
    d = reach * (L1 + L2)
    return 180.0 - math.degrees(math.acos((L1 ** 2 + L2 ** 2 - d ** 2) / (2.0 * L1 * L2)))


def ataxia():
    import itertools
    import math
    from mdfm.body.rotations import matrix_to_rot6d
    from mdfm.synth import ataxia as A
    from mdfm.synth import body as B
    from mdfm.synth.viewer import dominant_joint, knot_vertices
    out = {"knee_at_reach_deg": {"0.997": _r(_knee_at(0.997)), str(A.REACH_ST): _r(_knee_at(A.REACH_ST))}}
    _, info = A.supine_posture()
    out["supine"] = {k: (v if not isinstance(v, float) else _r(v, 3)) for k, v in info.items() if k != "solved_on"}
    out["supine"]["lowest_above_couch_mm"] = {k: _r(1000 * v) for k, v in info["lowest_above_couch_m"].items()}
    out["supine"].pop("lowest_above_couch_m", None)
    lim = B.joint_limits()
    base = A.supine_body(0.01)
    top = A.couch_top_y()
    V0 = knot_vertices(base, [0])[0]
    dom = dominant_joint("neutral")
    on0 = V0[:, 1] - top < A.HS_REST_ON_M
    ub = np.flatnonzero(on0 & ~np.isin(dom, (0, 1, 2, 3) + A.HS_LEGS["left"][1:] + A.HS_LEGS["right"][1:]))
    keep = dict(A.SPINE_BACK_TURN)
    rows = []
    try:
        for s3, s6 in itertools.product(np.round(np.arange(1.0, 1.70001, 0.05), 3),
                                        np.round(np.arange(-0.5, 0.50001, 0.1), 3)):
            A.SPINE_BACK_TURN = {3: float(s3), 6: float(s6), 9: float(round(1.0 - s3 - s6, 3))}
            m39, m6, slide = math.inf, math.inf, 0.0
            for side in ("left", "right"):
                R, tr, _, _, _ = A._pose_frame(base, side, A.PELVIS_TURN_DEG * np.ones(2))
                c = B.components_deg(matrix_to_rot6d(R))[0]
                m39 = min(m39, min(float(np.minimum(c[j] - lim.lo[j], lim.hi[j] - c[j]).min()) for j in (3, 9)))
                m6 = min(m6, float(np.minimum(c[6] - lim.lo[6], lim.hi[6] - c[6]).min()))
                b = B.Body(matrix_to_rot6d(R), tr, torch.zeros(2, 2, dtype=torch.bool), rate=100.0)
                V = knot_vertices(b, [0])[0]
                slide = max(slide, float(np.linalg.norm(V[ub][:, [0, 2]] - V0[ub][:, [0, 2]], axis=1).max()))
            rows.append({"shares": A.SPINE_BACK_TURN, "margin_3_9_deg": _r(m39, 3), "margin_6_deg": _r(m6, 3),
                         "slide_mm": _r(1000 * slide)})
    finally:
        A.SPINE_BACK_TURN = keep
    ok = [r for r in rows if r["margin_3_9_deg"] >= 1.0 and r["margin_6_deg"] >= 0.0]
    best = min(ok, key=lambda r: r["slide_mm"]) if ok else None
    before = [r for r in rows if r["shares"] == {3: 1.7, 6: 0.0, 9: -0.7}]
    out["back_turn"] = {"module": {str(k): v for k, v in keep.items()}, "best": best, "n_tried": len(rows),
                        "n_inside": len(ok), "before_1.7_0_-0.7": before[0] if before else None}
    r = math.radians(A.HS_ERROR_DIR_DEG)
    s0 = A.HS_KNEE_BELOW_M
    e3 = (1 + A.HS_ERROR_JITTER) * A.HS_GRADES[3]["error_m"]
    out["placement_margin_deg"] = {
        "knee": _r(A.placement_margin_deg(s0, 0.0)),
        "past_max": _r(A.placement_margin_deg(s0 - A.HS_PAST_MAX_M * math.cos(r), -A.HS_PAST_MAX_M * math.sin(r))),
        "grade3_largest": _r(A.placement_margin_deg(s0 - e3 * math.cos(r), -e3 * math.sin(r)))}
    return out


def pkd():
    from mdfm.synth import postures as P
    out = {}
    for kind in ("dystonic", "choreic"):
        try:
            _, tr = P.paroxysmal_kinesigenic(kind, seed=P.SEED + 90)
            v, p = tr["validity"], tr["params"]
            out[kind] = {"built": True, "limits_new": v.get("limits_new"),
                         "checks": {k: bool(v[k]["ok"]) for k in ("armrest", "hand_thigh", "hand_trunk", "fold")},
                         "fold_min_mm": {k: _r(x) for k, x in v["fold"].items() if isinstance(x, float)},
                         "unfold_lag": {k: p["arms_unfold_lag"][k] for k in ("lag", "lag_side")},
                         "trunk_flexion_at_liftoff_deg": _r(p["trunk_flexion_at_liftoff_deg"])}
        except Exception as e:  # noqa: BLE001
            out[kind] = {"built": False, "error": f"{type(e).__name__}: {e}"[:300]}
    return out


def push_off():
    from mdfm.synth import parkinsonism as P
    from mdfm.synth import body as B
    from mdfm.body.rotations import matrix_to_rot6d
    lim = B.joint_limits()
    tg0 = P.armrest_targets()
    par = {"arms": "push", "trunk_share": P.PUSH_TRUNK_RANGE[1], "u_peak": P.PUSH_FAIL_U[1], "timing_factor": 1.0,
           "fall_s": 0.6}
    R, tr, _, _, nu, _ = P._attempt(par, 100.0)
    keep = (P.POLE_OUT, P.POLE_BACK)
    out = {"module": {"WRIST_BACK_M": P.WRIST_BACK_M, "POLE_OUT": P.POLE_OUT, "POLE_BACK": P.POLE_BACK},
           "attempt": {k: par[k] for k in ("trunk_share", "u_peak")}, "hand_yaw": P.push_hand_yaw()[1], "scan": []}
    uu = np.round(np.arange(0.54, 0.7401, 0.02), 2)                  # the 'normal' template's hips over its seat
    Rt, trt, _ = P._template_pose("normal", np.r_[0.0, uu])
    Jt, _ = P.fk(Rt, trt, P.rest_joints())
    out["template_pelvis_lift_cm"] = {str(x): _r(100 * float(Jt[i + 1, 0, 1] - Jt[0, 0, 1])) for i, x in enumerate(uu)}
    try:
        for back in (0.08, 0.04, 0.0):
            tg = dict(tg0, wrist={sd: tg0["wrist"][sd] + torch.tensor([0.0, 0.0, P.WRIST_BACK_M - back],
                                                                       dtype=torch.float64) for sd in ("left", "right")})
            for po in (0.3, 0.4, 0.5):
                P.POLE_OUT = po
                Rp, reach, inside, _ = P.push_arms(R, tr, tg=tg)
                c = B.components_deg(matrix_to_rot6d(Rp))
                comps = []
                for j in (13, 14, 16, 17, 18, 19, 20, 21):
                    for a in range(3):
                        ex = np.maximum(lim.lo[j][a] - c[:, j, a], c[:, j, a] - lim.hi[j][a])
                        if ex.max() > 1e-6:
                            comps.append([j, "xyz"[a], _r(ex.max(), 2)])
                out["scan"].append({"wrist_back_m": back, "pole_out": po, "knots_outside": int((~inside).sum()),
                                    "components": comps, "reach_max": _r(reach.max(), 3),
                                    "thigh_gap_min_mm": _r(1000 * float(P.push_thigh_gap(R, tr, Rp).min()))})
        # the push rise of grade 2 (the clip as built): its arms solved again at each pole from the push's start to the
        # first knot whose collar-free reach passes RELEASE_EXT
        b, trg = P.arise_grade("g2")
        Rg, tg_ = b.local_R(), b.transl
        tgc = P.clip_chair_arms(tg0, Rg, tg_)
        reach_all = np.maximum(P.reach_share(Rg, tg_, "left", tgc["wrist"]["left"]),
                               P.reach_share(Rg, tg_, "right", tgc["wrist"]["right"]))
        k0 = int(np.flatnonzero(trg["arrays"]["arm_code"] == 2)[0])
        k1 = k0 + int(np.argmax(reach_all[k0:] > P.RELEASE_EXT))
        ks = np.arange(k0, k1 + 1)
        out["g2_rise"] = {"push_knot": k0, "over_reach_knot": k1, "release": trg["params"]["release"],
                          "arms_used": P.measure_arise(b)["arms_used"], "scan": []}
        for po in (0.4, 0.5):
            P.POLE_OUT = po
            Rp, reach, inside, _ = P.push_arms(Rg[ks], tg_[ks], tg=tgc)
            c = B.components_deg(matrix_to_rot6d(Rp))
            comps = []
            for j in (13, 14, 16, 17, 18, 19, 20, 21):
                for a in range(3):
                    ex = np.maximum(lim.lo[j][a] - c[:, j, a], c[:, j, a] - lim.hi[j][a])
                    if ex.max() > 1e-6:
                        comps.append([j, "xyz"[a], _r(ex.max(), 3), int(ks[int(np.argmax(ex > 1e-6))])])
            bad = np.flatnonzero(~inside)
            out["g2_rise"]["scan"].append({"pole_out": po, "first_outside_knot": int(ks[bad[0]]) if len(bad) else None,
                                           "components": comps})
    finally:
        P.POLE_OUT, P.POLE_BACK = keep
    return out


def sitdown():
    from mdfm.synth import hands as H
    from mdfm.synth import motions as M
    from mdfm.synth import parkinsonism as P
    from mdfm.body.rotations import matrix_to_rot6d
    _, ni = P.arise_template("normal")
    nr = int(round(ni["rise_s"] * 100.0))
    seat_v = M._seat_vertices()

    def margin(f, a):
        n = int(round(f * nr))
        ud = 1.0 - (np.arange(n) + 1) / n
        uu = np.r_[np.ones(50), ud, np.zeros(50)]
        R, tr, _ = P._template_pose("normal", uu)
        R, tr = P.sitdown_lean(R, tr, uu, a)
        b = H.HandBody(matrix_to_rot6d(R), tr, torch.zeros(len(uu), 2, dtype=torch.bool), rate=100.0)
        hs = H.body_points(b, seat_v)[..., 1].min(1)
        hs = hs - hs[-1]
        k_c = 50 + int(np.argmax(hs[50:] <= P.SEAT_CONTACT_M))
        post = M.balance(b, 50)["posterior"]
        k = 50 + int(np.argmin(post[50:k_c + 1]))
        return {"min_margin_mm": _r(1000 * float(post[50:k_c + 1].min())), "at_share": _r(uu[k], 3),
                "contact_share": _r(uu[k_c], 3)}
    _, info = P.controlled_descent_u()
    return {"no_lean": {str(f): margin(f, 0.0) for f in (1.0, P.SITDOWN_STRETCH_MAX)},
            "solved": {k: (_r(v, 4) if isinstance(v, float) else v) for k, v in info.items() if k != "rule"}}


def pkd_opening():
    from mdfm.synth import motions as MO
    from mdfm.synth import postures as PO
    from mdfm.synth.body import rotation_angle_deg
    b0, ainfo = MO.arise(standing_s=PO.DUR_S)
    t = np.arange(b0.T) / b0.rate
    t_lift = ainfo["onset_s"] + MO.LIFTOFF_SHARE * ainfo["rise_s"]
    k0 = int(np.flatnonzero((t >= t_lift) & (PO.trunk_flexion_deg(b0) <= PO.PKD["unfold_trunk_deg"]))[0])

    def steps(L):
        return rotation_angle_deg(L[1:].transpose(-1, -2) @ L[:-1]).numpy()
    sb = steps(b0.local_R())
    out = {"module_unfold_dur_s": PO.PKD["unfold_dur_s"], "scan": []}
    for dur in (0.6, 0.7, 0.8):
        R, _, _, rec = MO.unfold_to_stand(b0, k0, int(round(dur * b0.rate)))
        so = steps(R)
        out["scan"].append({"unfold_s": dur, "lag": rec["lag"],
                            "step_max_deg": {str(j): _r((so[:, j] - sb[:, j]).max(), 3) for j in range(13, 22)}})
    from concurrent.futures import ProcessPoolExecutor
    variants = (("dystonic", "dystonic", None, None), ("choreic", "choreic", None, None),
                ("dystonic_no_arms_guard", "dystonic", None, "no_arms"),
                ("dystonic_shoulder_guard", "dystonic", None, "shoulder"))
    with ProcessPoolExecutor(max_workers=4, max_tasks_per_child=1) as ex:   # a fresh process per clip (the module
        futs = {name: ex.submit(_pkd_one, kind, on_s, rows) for name, kind, on_s, rows in variants}   # values set)
        out["clips"] = {name: f.result() for name, f in futs.items()}
    return out


def _pkd_one(kind, on_s=None, rows=None):
    """One PKD clip (the catalogue's seed) with PKD's on_s as given (None: the module's) and the guards as built, or
    (rows 'shoulder') the trunk guard on the shoulder alone, or (rows 'no_arms') no guard of the arms against each
    other over the opening: its reading for pkd_opening (in a worker process)."""
    torch.set_num_threads(2)
    from mdfm.synth import motions as MO
    from mdfm.synth import postures as PO
    from mdfm.synth.body import rotation_angle_deg

    def steps(L):
        return rotation_angle_deg(L[1:].transpose(-1, -2) @ L[:-1]).numpy()
    if on_s is not None:
        PO.PKD["on_s"] = on_s
    if rows == "shoulder":
        PO.PKD_TRUNK_GUARD_ROWS = {sd: PO.ARM[sd][1:2] for sd in PO.ARM}
    if rows == "no_arms":
        PO._arms_need = lambda k0, k1: (lambda sub, kk: np.full(len(kk), -1.0))
    b0, _ = MO.arise(standing_s=PO.DUR_S)
    b, tr = PO.paroxysmal_kinesigenic(kind, seed=PO.SEED + 90)
    d = steps(b.local_R()) - steps(b0.local_R())
    p, v = tr["params"], tr["validity"]
    return {"on_s": PO.PKD["on_s"], "trunk_guard_rows": {sd: list(r) for sd, r in PO.PKD_TRUNK_GUARD_ROWS.items()},
            "trunk_guard": p["trunk_guard"], "arms_guard": p["arms_guard"],
            "realized_deviation_deg": {k: _r(x, 2) for k, x in p["realized_deviation_deg"].items()},
            "arm_step_max_deg": _r(d[:, 13:22].max(), 3), "arm_step_max_joint": int(13 + np.argmax(d[:, 13:22].max(0))),
            "fold": {k: (_r(x, 2) if isinstance(x, float) else x) for k, x in v["fold"].items()},
            "attack_on_s": _r(p["attack_on_s"], 3), "arms_unfold_s": [_r(x, 3) for x in p["arms_unfold_s"]],
            "lag": p["arms_unfold_lag"]["lag"],
            "not_ok": [k for k, x in v.items() if isinstance(x, dict) and x.get("ok") is False]}


def _stand_one(reach):
    """The standing 3.14 ladder's level 0 with base.STAND_REACH at reach (None: the module's), in a worker process:
    the legs' largest one-knot joint acceleration, where, the events then, and 'stand''s leg angles."""
    torch.set_num_threads(2)
    from mdfm.synth import base as BA
    from mdfm.synth import parkinsonism as P
    if reach is not None:
        BA.STAND_REACH = reach
        BA._stand_legs.cache_clear()
    names = (0, 1, 2, 4, 5, 7, 8)
    _, b, ev, _, _ = P.spontaneity_parts("stand", 0)
    J = b.joints().numpy()[:, list(names)]
    a = np.linalg.norm(np.diff(J, 2, axis=0), axis=-1) * b.rate ** 2
    k, j = np.unravel_index(np.argmax(a), a.shape)
    active = [e["kind"] for e in ev if e["start_s"] <= (k + 1) / b.rate <= e["start_s"] + e["len_s"]]
    return {"stand_reach": BA.STAND_REACH, "stand_legs_deg": [_r(x, 3) for x in BA._stand_legs()],
            "leg_accel_max_m_s2": _r(a.max(), 3), "at_s": (k + 1) / b.rate, "joint": int(names[j]), "during": active}


def stand_kink():
    from concurrent.futures import ProcessPoolExecutor
    from mdfm.synth import base as BA
    with ProcessPoolExecutor(max_workers=2, max_tasks_per_child=1) as ex:
        rows = [f.result() for f in [ex.submit(_stand_one, 1.0), ex.submit(_stand_one, None)]]
    return {"module_stand_reach": BA.STAND_REACH, "scan": rows}


def applause():
    from mdfm.synth import parkinsonism as P
    b, tr = P.applause(0)
    m = P.measure_claps(b)
    p = tr["params"]
    k0, k1 = (int(round(x * b.rate)) for x in p["raise_s"])
    g = tr["arrays"]["gap_m"][k0:k1 + 1]
    acc = np.linalg.norm(np.diff(b.joints().numpy()[:, 16:22], 2, axis=0), axis=-1) * b.rate ** 2
    return {"raise_arm_accel_max_m_s2": _r(acc[k0 - 1:k1].max(), 3), "raise_end_arm_accel_m_s2": _r(acc[k1 - 1].max(), 3),
            "claps": m["claps"], "approaches": m["approaches"], "approach_times_s": [_r(x) for x in m["approach_times_s"]],
            "raise_gap_min_m": _r(float(g.min()), 5), "opening_m": P.CLAP_OPEN_M,
            "raise_clearance": {k: (_r(v, 5) if isinstance(v, float) else v) for k, v in p["raise_clearance"].items()
                                if not isinstance(v, dict)}}


GOWERS_SCAN = ((20.0, 0.0), (20.0, 0.02), (24.0, 0.0), (30.0, 0.01))   # (pelvis deg, slide m); the module's values
#                                                                        are read last


def _outside_components(b):
    """{joint+axis: [knots outside, largest excess deg]} of a body against the anatomical table."""
    from mdfm.synth import body as B
    from mdfm.body.rotations import matrix_to_rot6d
    lim = B.joint_limits()
    c = B.components_deg(matrix_to_rot6d(b.local_R()))
    out = {}
    for j in range(1, 22):
        for a in range(3):
            ex = np.maximum(lim.lo[j][a] - c[:, j, a], c[:, j, a] - lim.hi[j][a])
            if (ex > 1e-6).any():
                out[f"{j}{'xyz'[a]}"] = [int((ex > 1e-6).sum()), _r(ex.max(), 3)]
    return out


def _planted_gaps(b, tr):
    """{side: [least, largest] mm}: the hand's lowest vertex off its thigh (measure_gowers, every 5th knot) where the
    hand is planted (not moving up in a step), as the Gowers test reads it."""
    from mdfm.synth import parkinsonism as P
    p = tr["params"]
    m = P.measure_gowers(b, p["liftoff_knot"], int(round(p["hands_released_s"] * b.rate)),
                         int(round(p["phases"]["final"][1] * b.rate)))
    t = np.asarray(m["gap_knots"]) / b.rate
    out = {}
    for side in ("left", "right"):
        gap = np.asarray(m[f"{side}_gap_m"])
        stepping = np.zeros(len(t), bool)
        for st in (s for s in p["steps"] if s["side"] == side):
            stepping |= (t >= st["start_s"]) & (t <= st["moved_s"])
        g = gap[~stepping & np.isfinite(gap)]
        out[side] = [_r(1000 * g.min(), 3), _r(1000 * g.max(), 3)]
    return out


def _gowers_one(pel, scoot):
    torch.set_num_threads(3)
    from mdfm.synth import parkinsonism as P
    if pel is not None:
        P.GOWERS_LEAN = dict(P.GOWERS_LEAN, pelvis_deg=pel)
        P.GOWERS_SCOOT_M = scoot
    P.gowers_lateral.cache_clear()                    # its cache is not keyed on the module values
    rec = {"pelvis_deg": P.GOWERS_LEAN["pelvis_deg"], "scoot_m": P.GOWERS_SCOOT_M}
    try:
        tr_body, tr = P.gowers()
    except RuntimeError as e:
        return dict(rec, error=str(e)[:300])
    v, m = tr["validity"], tr["params"]["remeasured"]
    return dict(rec, posterior_margin_min_mm=_r(1000 * m["posterior_margin_min_m"]),
                anterior_margin_min_mm=_r(1000 * m["anterior_margin_min_m"]),
                trunk_lean_liftoff_deg=_r(m["trunk_lean_liftoff_deg"]), trunk_lean_peak_deg=_r(m["trunk_lean_peak_deg"]),
                trunk_lean_release_deg=_r(m["trunk_lean_release_deg"]), lateral_m=tr["params"]["hand_lateral"]["lateral_m"],
                ankle_move_max_mm=_r(1000 * v["ankle_move_max_m"], 3), balance_ok=v["balance"]["ok"],
                limits_outside=v["limits_outside"], limits_worst_deg=_r(v["limits_worst_deg"], 3),
                limits_components=_outside_components(tr_body), planted_gap_mm=_planted_gaps(tr_body, tr),
                not_ok=[k for k, x in v.items() if isinstance(x, dict) and x.get("ok") is False])


def gowers():
    from concurrent.futures import ProcessPoolExecutor
    from mdfm.synth import parkinsonism as P
    from mdfm.synth.motions import MOS_MIN_M
    with ProcessPoolExecutor(max_workers=3, max_tasks_per_child=1) as ex:   # a fresh process per build
        futs = [ex.submit(_gowers_one, pel, sc) for pel, sc in GOWERS_SCAN] + [ex.submit(_gowers_one, None, None)]
        rows = [f.result() for f in futs]
    return {"module": {"GOWERS_LEAN_pelvis_deg": P.GOWERS_LEAN["pelvis_deg"], "GOWERS_SCOOT_M": P.GOWERS_SCOOT_M},
            "mos_min_m": MOS_MIN_M, "scan": rows[:-1], "module_build": rows[-1]}


GOWERS_JOLT_SCAN = ((None, 0.5), (4.0, 0.5), (8.0, 0.5), (2.0, 0.5), (2.0, 0.6))   # (cap width deg or None: the hard
#                                  cap of before, step move share); the module's values are read last


def _jolt_one(w, move, module=False):
    """One Gowers build (in a worker process) read as its test reads its one-knot jolts: the arm joints' and the knees'
    largest acceleration (one-knot second difference of the joint positions), where, in which phase, the knots past the
    test's bounds, the hip midpoint's acceleration at the knees' knot; the planted hands' gaps. w None: the spine's
    hard cap of before (D91), a cap width of 1e-9 deg."""
    torch.set_num_threads(3)
    from mdfm.synth import parkinsonism as P
    if not module:
        P.GOWERS_CAP_SOFT_DEG = 1e-9 if w is None else w
        P.GOWERS_STEP_MOVE = move
    P.gowers_lateral.cache_clear()
    b, tr = P.gowers()
    p, m = tr["params"], tr["params"]["remeasured"]
    J = b.joints().numpy()
    acc = np.linalg.norm(np.diff(J, 2, axis=0), axis=-1) * b.rate ** 2
    M = 0.5 * (J[:, 1] + J[:, 2])
    am = np.diff(M, 2, axis=0) * b.rate ** 2

    def phase(t):
        return next((k for k, (a, e) in p["phases"].items() if a <= t <= e), None)
    out = {"cap_soft_deg": "hard" if (w is None and not module) else P.GOWERS_CAP_SOFT_DEG,
           "step_move": P.GOWERS_STEP_MOVE}
    for name, cols, bound in (("arms", list(range(16, 22)), 6.0), ("knees", [4, 5], 1.0)):
        a = acc[:, cols]
        k, j = np.unravel_index(np.argmax(a), a.shape)
        over = np.flatnonzero(a.max(1) > bound)
        out[name] = {"max_m_s2": _r(a.max(), 3), "knot": int(k + 1), "t_s": _r((k + 1) / b.rate, 2), "joint": int(cols[j]),
                     "phase": phase((k + 1) / b.rate), "test_bound_m_s2": bound,
                     "knots_over_bound": [int(x + 1) for x in over]}
    kk = out["knees"]["knot"] - 1
    out["hip_mid_accel_at_knee_knot_m_s2"] = [_r(x, 3) for x in am[kk]]
    out["planted_gap_mm"] = _planted_gaps(b, tr)
    out.update(posterior_margin_min_mm=_r(1000 * m["posterior_margin_min_m"]),
               trunk_lean_liftoff_deg=_r(m["trunk_lean_liftoff_deg"]), trunk_lean_peak_deg=_r(m["trunk_lean_peak_deg"]),
               limits_outside=tr["validity"]["limits_outside"],
               not_ok=[k for k, x in tr["validity"].items() if isinstance(x, dict) and x.get("ok") is False])
    return out


def gowers_jolts():
    """Gowers' one-knot jolts as its test reads them (_jolt_one) on the module's build and on GOWERS_JOLT_SCAN."""
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=3, max_tasks_per_child=1) as ex:
        futs = [ex.submit(_jolt_one, w, mv) for w, mv in GOWERS_JOLT_SCAN] + [ex.submit(_jolt_one, None, None, True)]
        rows = [f.result() for f in futs]
    return dict(rows[-1], scan=rows[:-1])


def follow_up():
    sys.stdout.reconfigure(encoding="utf-8")
    torch.set_num_threads(4)
    with open(OUT, encoding="utf-8") as f:
        rec = json.load(f)
    sections = (("push_off", push_off), ("sitdown", sitdown), ("pkd_opening", pkd_opening),
                ("stand_kink", stand_kink), ("applause", applause), ("gowers", gowers), ("gowers_jolts", gowers_jolts))
    only = sys.argv[sys.argv.index("--follow-up") + 1:]          # the sections named after --follow-up, or all
    if not only:
        rec["follow_up"] = {}
    fu = rec.setdefault("follow_up", {})
    if "mdfm_commit" in fu:                                       # a record of one stamp for every section
        old = {"run_at": fu.pop("run_at", None), "mdfm_commit": fu.pop("mdfm_commit")}
        for name, _ in sections:
            if name in fu:
                fu.setdefault("measured", {}).setdefault(name, old)
    stamp = {"run_at": datetime.datetime.now().isoformat(timespec="seconds"), "mdfm_commit": _head()}
    for name, fn in sections:
        if only and name not in only:
            continue
        fu[name] = fn()
        fu.setdefault("measured", {})[name] = stamp
        print(name, "done", flush=True)
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        json.dump(rec, f, indent=1, default=str)
    print("wrote", OUT)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    torch.set_num_threads(4)
    from mdfm.synth import body as B
    rec = {"script": "scripts/openbody_wave6_checks.py",
           "run_at": datetime.datetime.now().isoformat(timespec="seconds"), "mdfm_commit": _head(),
           "body_model": B.model_name()}
    ref = reference_arise()
    rec["reference_arise"] = ref
    print("reference_arise done", flush=True)
    rec["fold_targets"] = fold_targets(ref)
    for name, fn in (("fold", fold), ("arise_balance", arise_balance), ("grip_thumb", grip_thumb),
                     ("outstretched", outstretched), ("thigh_hands", thigh_hands), ("ataxia", ataxia),
                     ("pkd", pkd)):
        rec[name] = fn()
        print(name, "done", flush=True)
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        json.dump(rec, f, indent=1, default=str)
    print("wrote", OUT)


if __name__ == "__main__":
    if "--reference" in sys.argv:
        sys.path.insert(0, ROOT)
        _reference_main()
    elif "--follow-up" in sys.argv:
        follow_up()
    else:
        main()
