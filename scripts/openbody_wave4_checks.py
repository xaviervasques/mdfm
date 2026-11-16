"""The open-body port's wave 4 readings that the design's section 15 cites (docs/OPEN_BODY_PORT_DESIGN.md): the bases
and operators of the postures and dystonia modules read on the selected body model (the open body by default), each
beside the value the module held before wave 4 where wave 4 changed it.

- table_hands: the hands of the writing scene (postures.writing_base) at the first knot for the thumb's palmar
  abduction shifted by 0 to -20 deg (THUMB_ABD_SHIFT_DEG, D50): each hand part's lowest skinned point over the table
  top, the writing thumb's and index's tips, the pen's length; the writing level 0's table check before and after;
- axial: the photograph angles of the level-0 stand; the lateral trunk flexion (the mild band's centre, 7.5 deg, and
  the Pisa syndrome's 15 deg) with the far arm's girdle dropped by dystonia.girdle_drop (D52), and before it (the
  collar lowered by the whole bend, which capped the bend at the collar's depression room); the far forearm's and
  hand's gaps to the trunk and thighs (dystonia.rest_gap); the antecollis at the whole cervical room and at
  NECK_OPEN_DEG (D53);
- trunk_ladder: dystonia's lateral trunk flexion to the left at DIS grades 1 to 4, the far arm's response (the
  shoulder's share, the girdle's drop and its bound, the rest gap, the upper arm's entry), the level 0's own entry,
  the one-way and two-way gap readings at grade 4's drop;
- ramp_step: the largest joint step at the first knot of a trunk bout over the base's (dystonia trunk, grade 3,
  duration 2), with plant_stand's hip margin before (1e-7 m) and now (HIP_MARGIN_M, D55);
- planted_legs: the planted leg postures the pathology tests read (the seated and standing knee flexions);
- foot: the equinovarus foot of grade 3, seated and standing: its terms and the measured components, the pin's rule,
  the ankle's forward travel and, seated, how far ahead of the ankle the pinned contact lies;
- torsion: the generalized torsion dystonia of severity 3 (seed 5): each upper arm's largest angle from the vertical
  with the arms' abduction at 0.15 of its room (before), at none, and at TORSION_ARMS (D57);
- shoulder_elevation: the TWSTRS D grades' collar levels and shoulder rises, and the rise at the collar's cap;
- geste: the geste antagoniste of the tests (rotation to the left, grade 2, trick score 1, seed 3, 15 s): the hand's
  frame and arm, the wrist's bend, the pads' gap, the reduction, the hand's place; at trick score 2 the arm's forward
  excess against GESTE_FORWARD_EXTRA_MAX_M (D59);
- writing: the writing level 0 and the writer's cramp of grade 2: their table checks;
- pkd: the dystonic PKD clip's build as the catalogue runs it, and with the stale fold record's numbers loaded in
  memory (the arising's own solve read past the record; nothing written).
Writes reports/openbody/wave4_checks.json. CPU only; reads no dataset.
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
OUT = os.path.join(ROOT, "reports", "openbody", "wave4_checks.json")
THUMB_SHIFTS = (0.0, -5.0, -10.0, -15.0, -20.0)
PARTS = ("wrist", "index2", "middle2", "ring2", "pinky2", "thumb1", "thumb2")


def _head():
    try:
        return subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def _r(x, n=2):
    return round(float(x), n)


def _parts_low(H, P, b, side, top):
    """{part: mm} the lowest skinned point of each hand part of side over the height top at the first knot."""
    dom = H._rest()[2]
    ids = P.hand_vertex_ids(side)
    y = H.body_points(b, ids, knots=[0])[0][:, 1] - top
    names = {H.hand_joint(side, f, lv): f"{f}{lv}" for f in H.FINGERS for lv in range(3)}
    names[H.WRIST[side]] = "wrist"
    low = {names[int(j)]: _r(1000 * float(y[dom[ids] == j].min()), 1) for j in np.unique(dom[ids])}
    return {k: low[k] for k in PARTS}


def table_hands():
    from mdfm.synth import hands as H
    from mdfm.synth import postures as P
    W0, R0, C0, s0 = dict(P.WRITING_HAND), dict(P.RELAXED_HAND), dict(P.CRAMP_HAND), P.THUMB_ABD_SHIFT_DEG

    def shifted(pose, sh):
        (f, a), m, i = pose["thumb"]
        return dict(pose, thumb=((f, a - s0 + sh), m, i))
    out = {"module_shift_deg": s0, "scan": []}
    try:
        for sh in THUMB_SHIFTS:
            P.WRITING_HAND, P.RELAXED_HAND, P.CRAMP_HAND = shifted(W0, sh), shifted(R0, sh), shifted(C0, sh)
            P.writing_surface.cache_clear()
            b, info = P.writing_base(0.02, ("left", "right"), P.SEED, "right")
            top, paper = info["surface"]["table_top_y"], info["surface"]["paper_top_y"]
            tips = H.tip_vertices()["right"]
            X = H.body_points(b, [tips["thumb"], tips["index"]], knots=[0])[0]
            rec = {"shift_deg": sh, "resting_left_over_table_mm": _parts_low(H, P, b, "left", top),
                   "writing_right_over_paper_mm": _parts_low(H, P, b, "right", paper),
                   "writing_thumb_index_tips_mm": _r(1000 * np.linalg.norm(X[0] - X[1]), 1),
                   "pen_m": _r(info["surface"]["pen_m"], 4)}
            if sh in (0.0, s0):
                _, _, val = P.writing_level0()
                t = val["table"]
                rec["writing_level0_table"] = {"ok": bool(t["ok"]), "hand_gap_mm": [_r(x, 3) for x in t["hand_gap_mm"]],
                                               "edge_gap_mm": [_r(x, 3) for x in t["edge_gap_mm"]],
                                               "depth_mm": _r(t["depth_mm"], 3)}
            out["scan"].append(rec)
    finally:
        P.WRITING_HAND, P.RELAXED_HAND, P.CRAMP_HAND = W0, R0, C0
        P.writing_surface.cache_clear()
    return out


def _gap(D, b, side="right"):
    g = D.rest_gap(b, side, knots=[b.T - 1], upper=True)
    return {p: {a: _r(1000 * float(v[0]), 1) for a, v in c.items()} for p, c in g.items()}


def axial():
    from mdfm.synth import dystonia as D
    from mdfm.synth import postures as P
    from mdfm.synth.base import static_body
    st = static_body("stand", 1.0)
    out = {"level0_deg": {k: _r(v[0]) for k, v in P.axial_angles(st, knots=[0]).items()}, "lateral": {}}
    one = np.ones(st.T)
    room = P.region_room(st, "trunk", "lateral_left", "left")
    j = P.ARM_COLLAR["right"]
    from mdfm.synth.body import room_deg
    croom = room_deg(st.local_R()[0][j], j, "z", 1.0, "parent")
    amp_before = min(room, P.region_room(st, "proximal_arm", "abduction", "left"), croom)
    before, _, _, _ = P.axial_body(st, "lateral", amp_before, one, 5.0 * one,
                                   (P.joint_term(j, "z", amp_before, one, "parent", "collar lowered (right)"),),
                                   "left")
    out["lateral"]["before"] = {"rule": "the far collar lowered by the whole bend, the bend capped by its room",
                                "amplitude_deg": _r(amp_before),
                                "held_deg": _r(P.axial_angles(before, knots=[0])["lateral"][0], 3),
                                "far_arm_gap_mm": _gap(D, before)}
    for deg in (7.5, 15.0):
        b, tr = P.axial_posture(st, "lateral", deg, 5.0)
        p = tr["params"]
        far = p["stand"].get("arms_hang", {}).get("far_arm", {})
        out["lateral"][str(deg)] = {"reachable": p["reachable"], "amplitude_deg": _r(p["amplitude_deg"]),
                                    "held_deg": _r(p["held_deg"], 3), "band": p["band_held"],
                                    "girdle_drop_deg": far.get("girdle_drop_deg"), "drop_bound": far.get("bound"),
                                    "far_arm_gap_mm": _gap(D, b), "limits_new": tr["validity"]["limits_new"]}
    full = P.region_room(st, "neck", "anterocollis", "left")
    nb, _, _, _ = P.axial_body(st, "neck", full, one)
    nk, tn = P.axial_posture(st, "neck", P.NECK_OPEN_DEG, 0.0)
    out["neck"] = {"cervical_flexion_room_deg": _r(full),
                   "held_at_whole_room_deg": _r(P.axial_angles(nb, knots=[0])["neck"][0], 3),
                   "open_band_factor_target_deg": P.OPEN_BAND_FACTOR * P.TF_BANDS["neck"]["cut"],
                   "neck_open_deg": P.NECK_OPEN_DEG, "reachable": tn["params"]["reachable"],
                   "held_deg": _r(tn["params"]["held_deg"], 3), "band": tn["params"]["band_held"]}
    return out


def trunk_ladder():
    from mdfm.synth import dystonia as D
    from mdfm.synth import reach as RC
    from mdfm.synth.base import static_body
    st = static_body("stand", 1.0)
    k = np.array([st.T - 1])
    out = {"level0_upper_arm_in_trunk_mm": _r(-1000 * D.rest_gap(st, "right", k, parts=("trunk",),
                                                                 upper=True)["trunk"]["upper"][0], 2),
           "grades": {}}
    for g in (1, 2, 3, 4):
        b, t = D.dystonic_posture(st, "trunk", "lateral_left", "left", amplitude_grade=g, seed=g)
        arm = t["params"]["response"]["arms"]["right"]
        v = t["validity"]
        out["grades"][str(g)] = {"level_deg": _r(t["params"]["level_deg"]),
                                 "shoulder_share_min": _r(arm["shoulder_share_min"], 3),
                                 "girdle_drop_deg": arm["girdle_drop_deg"], "bound": arm["bound"],
                                 "rest_gap_mm": _r(arm["rest_gap_mm"], 2) if "rest_gap_mm" in arm else None,
                                 "upper_arm_in_trunk_mm": _r(arm["upper_arm_in_trunk_mm"], 2)
                                 if "upper_arm_in_trunk_mm" in arm else None,
                                 "hand_trunk_right_min_mm": _r(v["hand_trunk"]["right"]["min_mm"], 2),
                                 "checks_ok": bool(v["hand_trunk"]["ok"] and v["hand_thigh"]["ok"]
                                                   and v[D.UPPER_ENTRY]["ok"])}
        if g == 4:
            out["grades"]["4"]["one_way_mm"] = {}
            for extra in (0.0, 0.5):
                R = b.local_R().clone()
                R[:, 14] = D._lowered(R[:, 14], "right", extra)
                low = b.with_rotations(R, [14])
                kk = np.array([b.T - 1])
                out["grades"]["4"]["one_way_mm"][f"drop+{extra}"] = {
                    "two_way": _r(1000 * RC.clearance(low, "right", kk, parts=("trunk",))["trunk"]["fore"][0]),
                    "one_way": _r(1000 * D.rest_gap(low, "right", kk, parts=("trunk",))["trunk"]["fore"][0])}
    return out


def ramp_step():
    from mdfm.synth import dystonia as D
    from mdfm.synth import postures as P
    from mdfm.synth.base import static_body
    base = static_body("stand", 8.0)
    out = {}
    m0 = P.HIP_MARGIN_M
    try:
        for name, m in (("before_1e-7", 1e-7), ("now", m0)):
            P.HIP_MARGIN_M = m
            b, tr = D.dystonic_posture(base, "trunk", "lateral_left", "left", amplitude_grade=3, duration_grade=2,
                                       seed=7)
            env = tr["arrays"]["envelope"]
            first = int(np.flatnonzero(env > 0)[0])
            J, Jb = b.joints(), base.joints()
            steps = []
            for i in range(3):
                kk = first + i
                st = ((J[kk] - J[kk - 1]) - (Jb[kk] - Jb[kk - 1])).norm(dim=-1)
                steps.append({"knot": kk, "envelope": float(env[kk]), "step_mm": _r(1000 * float(st.max()), 4),
                              "joint": int(st.argmax())})
            out[name] = {"hip_margin_m": m, "steps": steps}
    finally:
        P.HIP_MARGIN_M = m0
    return out


def planted_legs():
    from mdfm.synth import dystonia as D
    from mdfm.synth.base import static_body
    sit, st = static_body("sit", 3.0), static_body("stand", 3.0)
    out = {}
    for name, b, g in (("sit_knee_flexion_g4", sit, 4), ("stand_knee_flexion_g2", st, 2),
                       ("stand_knee_flexion_g4", st, 4)):
        _, tr = D.dystonic_posture(b, "distal_leg", "knee_flexion", "left", amplitude_grade=g, seed=1)
        p = tr["params"]
        fp = p["foot_pose"]
        out[name] = {"level_deg": _r(p["level_deg"]), "plant_scale_min": _r(p["plant_scale_min"], 3),
                     "reachable": p["reachable"], "toe_stance": p["toe_stance"],
                     "heel_lift_max_m": _r(fp["heel_lift_max_m"], 3), "knee_change_deg": _r(fp["knee_change_deg"]),
                     "pitch_change_deg": _r(fp["pitch_change_deg"]), "ankle_change_deg": _r(fp["ankle_change_deg"])}
    return out


def foot():
    from mdfm.synth import contact as CT
    from mdfm.synth import postures as P
    from mdfm.synth import viewer as V
    out = {}
    for standing in (False, True):
        b, tr = P.foot_dystonia(3, standing=standing, levels=P.FOOT_STAND if standing else None, duration_s=3.0,
                                seed=2)
        p = tr["params"]
        f = p["foot"]
        base = P.static_body("stand" if standing else "sit", 3.0)
        rec = {"terms_deg": {t["label"]: _r(t["deg"]) for t in p["terms"]},
               "limit_scale_min": p["limit_scale_min"],
               "measured_deg": {k: _r(v) for k, v in p["foot_measured"].items() if k != "knot"},
               "pin_rule": f["pin"]["rule"], "heel_rise_cm": _r(f["heel_rise_cm"]),
               "pelvis_hike_deg_max": _r(f["pelvis_hike_deg_max"]), "knee_lift_deg_max": _r(f["knee_lift_deg_max"]),
               "ankle_forward_cm": _r(100 * float((b.joints()[:, 7, 2] - base.joints()[:, 7, 2]).max()))}
        if not standing:
            vid = CT.sole_vertices()["left"][f["pin"]["vertex_in_cloud"]]
            rec["contact_ahead_of_ankle_cm"] = _r(100 * float(V.knot_vertices(base, [0])[0][vid, 2]
                                                              - base.joints()[0, 7, 2]))
        out["standing" if standing else "seated"] = rec
    return out


def torsion():
    from mdfm.synth import postures as P
    from mdfm.synth.base import static_body
    st = static_body("stand", 3.0)

    def from_vertical(b):
        J = b.joints().numpy()
        res = {}
        for side, (s, e) in (("left", (16, 18)), ("right", (17, 19))):
            u = J[:, e] - J[:, s]
            res[side] = _r(np.degrees(np.arccos(np.clip(-u[:, 1] / np.linalg.norm(u, axis=-1), -1.0, 1.0))).max(), 1)
        return res
    A0 = dict(P.TORSION_ARMS)
    out = {"stand_from_vertical_deg": from_vertical(st)}
    try:
        for name, ab in (("before_share_0.15", {sd: 0.15 * P.region_room(st, "proximal_arm", "abduction", sd)
                                                for sd in ("left", "right")}),
                         ("none", {"left": 0.0, "right": 0.0}), ("now", A0["abduction_deg"])):
            P.TORSION_ARMS["abduction_deg"] = ab
            b, tr = P.torsion_dystonia(st, 3, seed=5)
            out[name] = {"abduction_deg": {k: _r(v) for k, v in ab.items()}, "arm_from_vertical_deg": from_vertical(b),
                         "trunk_terms_deg": {t["label"]: _r(t["deg"]) for t in tr["params"]["terms"]
                                             if t["label"].startswith("trunk")}}
    finally:
        P.TORSION_ARMS.clear()
        P.TORSION_ARMS.update(A0)
    return out


def shoulder_elevation():
    from mdfm.synth import dystonia as D
    from mdfm.synth import limits_ext as LX
    from mdfm.synth.base import static_body
    st = static_body("stand", 3.0)
    out = {"grades": {}}
    for g in (1, 2, 3):
        _, tr = D.dystonic_posture(st, "shoulder_elevation", "elevation", "left", amplitude_grade=g, seed=g)
        p = tr["params"]
        out["grades"][str(g)] = {"level_deg": _r(p["level_deg"], 3), "possible_deg": _r(p["possible_deg"], 3),
                                 "shoulder_rise_cm_max": _r(p["shoulder_rise_cm_max"]),
                                 "limits_new": tr["validity"]["limits_new"]}
    _, tr = D.dystonic_posture(st, "shoulder_elevation", "elevation", "left", level=LX.collar_cap("left") - 1e-6,
                               seed=1)
    out["at_collar_cap"] = {"cap_deg": LX.collar_cap("left"),
                            "shoulder_rise_cm_max": _r(tr["params"]["shoulder_rise_cm_max"])}
    return out


def geste():
    from mdfm.synth import postures as P
    from mdfm.synth.base import static_body
    st = static_body("stand", 15.0)
    out = {"trick_1": _geste_read(P, *P.geste_antagoniste(st, "rotation_left", 2, 1, seed=3))}
    _, tr = P.geste_antagoniste(st, "rotation_left", 2, 2, seed=3)
    p, arm = tr["params"], tr["params"]["arm"]
    out["trick_2"] = {"forward_extra_m": _r(arm["forward_extra_m"], 4), "forward_relaxed": arm["forward_relaxed"],
                      "frames_tried": p["frame"]["tried"], "reduction": _r(p["reduction"], 3),
                      "limits_new": tr["validity"]["limits_new"], "bound_m": P.GESTE_FORWARD_EXTRA_MAX_M}
    return out


def _geste_read(P, b, tr):
    p, v = tr["params"], tr["validity"]
    hp = p["hand_place"]
    return {"level_deg": _r(p["level_deg"]), "reduction": _r(p["reduction"], 3), "held_deg": _r(p["held_deg"]),
            "touch_deg": _r(p["touch_deg"]), "frame": {"finger_deg": p["finger_deg"], "tilt_deg": p["tilt_deg"],
                                                        "roll_deg": p["roll_deg"], "tried": p["frame"]["tried"]},
            "arm": {k: p["arm"][k] for k in ("bend_deg", "collar_deg", "twist_deg", "built", "forward_relaxed",
                                             "forward_extra_m")},
            "wrist_deg": {k: ([_r(x) for x in val] if isinstance(val, list) else _r(val, 3))
                          for k, val in p["wrist_deg"].items() if k != "range_deg"},
            "pad_gap_m": _r(p["pad_gap_m"], 4), "skin_m": p["skin_m"],
            "hand_place": {k: (_r(val, 4) if isinstance(val, float) else val) for k, val in hp.items()},
            "limits_new": v["limits_new"], "wrist_range_ok": v["wrist_range"]["ok"]}


def writing():
    from mdfm.synth import postures as P
    _, _, v0 = P.writing_level0()
    _, tr = P.writers_cramp(2)
    keys = ("ok", "edge_gap_mm", "hand_gap_mm", "pen_gap_word_mm", "forearm_gap_mm", "depth_mm", "hands_gap_mm")

    def pick(t):
        return {k: (t[k] if not isinstance(t[k], (float, list)) else
                    ([_r(x, 3) for x in t[k]] if isinstance(t[k], list) else _r(t[k], 3))) for k in keys if k in t}
    return {"level0": pick(v0["table"]), "writers_cramp_g2": pick(tr["validity"]["table"]),
            "writers_cramp_g2_limits_new": tr["validity"]["limits_new"]}


def pkd():
    """The PKD clips' builds: as the catalogue builds them, and with the fold record's numbers loaded in memory past its
    digest (the record stale since wave 1, design 12.5; nothing written), to read whether the arising itself builds."""
    from mdfm.synth import motions as M
    from mdfm.synth import postures as P
    out = {"fold_state": M._FOLD_STATE.get("state")}

    def tryit(name):
        try:
            P.paroxysmal_kinesigenic("dystonic", seed=0)
            out[name] = "built"
        except Exception as e:  # noqa: BLE001
            out[name] = f"{type(e).__name__}: {e}"[:300]
    tryit("as_built")
    with open(M.fold_record_path(), encoding="utf-8") as f:
        rec = json.load(f)
    keep = (M.ARM_POSTURES.get("fold"), M.ARM_TARGETS.get("fold"))
    try:
        M.ARM_POSTURES["fold"] = {s: tuple(tuple(float(v) for v in a) for a in rec["fold"][s])
                                  for s in ("left", "right")}
        M.ARM_TARGETS["fold"] = {"base": "stand", **{s: {k: tuple(v) for k, v in rec["arm_targets"][s].items()}
                                                     for s in ("left", "right")}}
        tryit("with_the_stale_fold")
    finally:
        for d, v in zip((M.ARM_POSTURES, M.ARM_TARGETS), keep):
            if v is None:
                d.pop("fold", None)
            else:
                d["fold"] = v
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    torch.set_num_threads(4)
    from mdfm.synth import body as B
    rec = {"script": "scripts/openbody_wave4_checks.py",
           "run_at": datetime.datetime.now().isoformat(timespec="seconds"), "mdfm_commit": _head(),
           "body_model": B.model_name()}
    for name, fn in (("table_hands", table_hands), ("axial", axial), ("trunk_ladder", trunk_ladder),
                     ("ramp_step", ramp_step), ("planted_legs", planted_legs), ("foot", foot), ("torsion", torsion),
                     ("shoulder_elevation", shoulder_elevation), ("geste", geste), ("writing", writing),
                     ("pkd", pkd)):
        rec[name] = fn()
        print(name, "done", flush=True)
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        json.dump(rec, f, indent=1, default=str)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
