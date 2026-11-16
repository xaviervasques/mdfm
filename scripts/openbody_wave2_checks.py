"""The open-body port's wave 2 readings that the design's section 13 cites (docs/OPEN_BODY_PORT_DESIGN.md): the
bases and operators of the tremor, bradykinesia, reach and hand-task modules read on the selected body model
(the open body by default), each beside the value the module held before wave 2 where wave 2 changed it.

- sit_arms_forward: the collar and shoulder components against the anatomical table, with the version before
  (collars 25 deg, shoulders 65) and the module's;
- the straight-elbow tremor (tremor.tremor on sit_arms_forward, grades 1 to 4, both sides): the rooms and the centre
  shifts;
- toe tapping (bradykinesia, grade 3 decrement, seed 2, as the test): the level-0 amplitude, the room, the rise of
  the foot joint and the bound the test asks (0.9 times the foot joint's horizontal reach from the ankle times the
  sine of the amplitude), the planted and the free foot's sole movement;
- the TETRAS standing posture (tremors.standing_centre_m): the forward carry and the posture's margins without and
  with it; grade 3's extrapolated centre of mass's margins;
- the pill-rolling hand (tremors): the pose's index angles against the hand limits and the contact report of grade 1;
- the simple kinetic tremor on pronation-supination (TETRAS 2 and 3): the blind forearm-frame reading's share of the
  level;
- the seated heel gap of the leg rest level 0 and the outstretched raise's finger gaps (the rest hand);
- the face touch (reach.touch_path, right hand): the elbow's drop below the shoulder at each target on 'sit' and
  'stand', the forehead also at its elevation before wave 2 (45 deg).
Writes reports/openbody/wave2_checks.json. CPU only; reads no dataset.
"""
import datetime
import json
import math
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "reports", "openbody", "wave2_checks.json")


def _head():
    try:
        return subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def _outside(B, rot6d):
    lim = B.joint_limits()
    c = B.components_deg(rot6d)
    ex = np.maximum(lim.lo[None] - c, c - lim.hi[None])[:, 1:].max(0)
    return {f"{B.BODY_JOINTS[j + 1]}.{'xyz'[k]}": round(float(ex[j, k]), 3) for j, k in zip(*np.where(ex > 1e-6))}


def arms_forward(B, base):
    out = {"module": {str(j): spec for j, spec in base.POSTURES["sit_arms_forward"].items() if j in (13, 14, 16, 17)},
           "outside_now": _outside(B, base.static_body("sit_arms_forward", 0.01).rot6d)}
    saved = {j: base.POSTURES["sit_arms_forward"][j] for j in (13, 14, 16, 17)}
    try:
        base.POSTURES["sit_arms_forward"].update({13: [("y", -25.0)], 14: [("y", 25.0)], 16: [("y", -65.0)],
                                                   17: [("y", 65.0)]})
        out["outside_before"] = _outside(B, base.static_body("sit_arms_forward", 0.01).rot6d)
    finally:
        base.POSTURES["sit_arms_forward"].update(saved)
    return out


def straight_elbow(base, T):
    b = base.static_body("sit_arms_forward", 10.0)
    rows = []
    for side in ("left", "right"):
        for g in (1, 2, 3, 4):
            _, tr = T.tremor(b, side, grade=g, frequency_hz=5.0, seed=g)
            p = tr["params"]
            rows.append({"side": side, "grade": g, "peak_deg": round(p["peak_deg"], 3),
                         "room_toward_extension_deg": round(p["room_plus_deg"] if side == "left" else p["room_minus_deg"], 3),
                         "centre_shift_deg": round(p["centre_shift_deg"], 3), "limits_new": tr["validity"]["limits_new"]})
    return {"rows": rows, "shifted": sum(r["centre_shift_deg"] != 0 for r in rows)}


def toe_tapping(base, BR):
    sit = base.static_body("sit", 10.0)
    out, tr = BR.bradykinesia(sit, "toe_tapping", "left", grade=3, criterion="decrement", seed=2)
    J, J0 = out.joints().numpy(), sit.joints().numpy()[0]
    a0 = tr["params"]["amplitude0_deg"]
    reach = float(np.linalg.norm((J0[10] - J0[7])[[0, 2]]))
    _, trf = BR.bradykinesia(sit, "toe_tapping", "left", grade=3, criterion="decrement", seed=2, foot="free")
    cp, cf = tr["validity"]["contact"], trf["validity"]["contact"]
    return {"amplitude0_deg": round(a0, 3), "room_deg": round(tr["params"]["room_deg"], 3),
            "rise_cm": round(100.0 * float(J[:, 10, 1].max() - J[0, 10, 1]), 3),
            "foot_reach_cm": round(100.0 * reach, 3),
            "test_bound_cm": round(100.0 * 0.9 * reach * math.sin(math.radians(a0)), 3),
            "contact_ok": bool(cp["ok"]),
            "planted_sole_moved_mm": [round(1000.0 * cp["moved_low_min_m"], 3), round(1000.0 * cp["moved_low_max_m"], 3)],
            "free_sole_moved_mm": [round(1000.0 * cf["moved_low_min_m"], 3), round(1000.0 * cf["moved_low_max_m"], 3)]}


def seated_hands_and_feet(TR, MO):
    """The leg rest level 0's seated heel gap and the outstretched raise's finger gaps (rest hand), as the tests read
    them."""
    specs = {s.clip_id: s for s in TR.specs()}
    _, t0 = specs["tremor_E1_leg_rest_g0"].build()
    _, info = MO.outstretched(6.0)
    return {"leg_rest_heel_gap_sit_mm": round(t0["params"]["heel_gap_sit_mm"], 3),
            "leg_rest_heel_down_deg": round(t0["params"]["heel_down_deg"], 3),
            "outstretched_finger_gaps_mm": {s: {k: round(v, 3) for k, v in g.items()}
                                            for s, g in info["finger_gaps_mm"].items()}}


def standing(TR):
    from mdfm.synth.motions import balance
    c = TR.standing_centre_m()
    widen = TR._standing_widen()
    out = {"forward_carry_cm": round(100.0 * c, 3)}
    for name, cc in (("without_carry", 0.0), ("with_carry", c)):
        dy, k = TR._knee_table(int(round(widen * 1e6)), dz_key=int(round(cc * 1e6)))
        y = float(np.interp(TR.STANDING["knee_flex_deg"], k, dy))
        b, _ = TR.stand_legs(np.full(2, y), None, widen, 0.01, dz=np.full(2, cc) if cc else None)
        bl = balance(b, 0)
        out[name] = {"posterior_margin_cm": round(100.0 * float(bl["posterior"][0]), 3),
                     "anterior_margin_cm": round(100.0 * float(bl["anterior"][0]), 3)}
    specs = {s.clip_id: s for s in TR.specs()}
    for g in (2, 3):
        _, tr = specs[f"standing_tremor_E10_tetras9_g{g}"].build()
        bl = tr["validity"]["balance"]
        out[f"grade{g}"] = {"xcom_margin_posterior_cm": round(100.0 * bl["xcom_margin_posterior_m"], 3),
                            "xcom_margin_anterior_cm": round(100.0 * bl["xcom_margin_anterior_m"], 3), "ok": bl["ok"]}
    return out


def pill(TR, H):
    pose = TR._pill_pose("left")
    idx = {H.LEVELS[lv]: round(pose[H.hand_joint("left", "index", lv)][0], 3) for lv in range(3)}
    specs = {s.clip_id: s for s in TR.specs()}
    _, tr = specs["tremor_E1_pillrolling_g1"].build()
    c = tr["params"]["contact"]
    return {"index_close": TR.PILL["index_close"], "limit_margin_deg": TR.PILL_LIMIT_MARGIN_DEG,
            "index_flexion_deg": idx, "index_limits_deg": {k: H.FLEX_LIMITS_DEG["index"][k][1] for k in H.LEVELS},
            "gap_cm_range": [round(x, 4) for x in c["gap_cm_range"]], "contact_slide_cm": round(c["contact_slide_cm"], 3),
            "hand_limits_ok": bool(tr["validity"]["hand_limits"]["ok"])}


def kinetic_ps(TR):
    specs = {s.clip_id: s for s in TR.specs()}
    out = {}
    for cid in ("tremor_E4_kinetic_ps_tetras2", "tremor_E4_kinetic_ps_tetras3"):
        body, tr = specs[cid].build()
        p = tr["params"]
        lb = p["limbs"]["left"]
        a, z = p["task"]["window_s"]
        blind = TR.task_removed_readback(body, hz=lb["frequency_hz"], window_s=(a + 0.6, z - 0.6))
        out[cid] = {"level_cm": round(lb["cm_pp"], 3), "blind_pp_cm_median": round(blind["pp_cm_median"], 3),
                    "share": round(blind["pp_cm_median"] / lb["cm_pp"], 3),
                    "shoulder_weight": TR.postural_shoulder_weight(lb["cm_pp"])}
    return out


def touch(RC):
    rows = {}
    for post in RC.TOUCH_POSTURES:
        for tg in ("nose", "chin", "forehead", "cheek"):
            c = RC.touch_path.__wrapped__("right", tg, post)["check"]
            rows[f"{post}/{tg}"] = {"elbow_drop_cm": round(100.0 * c["elbow_drop_m"], 2), "ok": c["ok"]}
    el = RC.TOUCH_DIR_DEG["forehead"]["elevation"]
    try:
        RC.TOUCH_DIR_DEG["forehead"]["elevation"] = 45.0
        for post in RC.TOUCH_POSTURES:
            c = RC.touch_path.__wrapped__("right", "forehead", post)["check"]
            rows[f"{post}/forehead at 45 deg"] = {"elbow_drop_cm": round(100.0 * c["elbow_drop_m"], 2), "ok": c["ok"]}
    finally:
        RC.TOUCH_DIR_DEG["forehead"]["elevation"] = el
    return {"forehead_elevation_deg": el, "elbow_drop_min_cm": 100.0 * RC.TOUCH_ELBOW_DROP_M, "paths": rows}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    from mdfm.synth import base, body as B, bradykinesia as BR, hands as H, motions as MO, reach as RC, tremor as T
    from mdfm.synth import tremors as TR
    rec = {"script": "scripts/openbody_wave2_checks.py",
           "run_at": datetime.datetime.now().isoformat(timespec="seconds"), "mdfm_commit": _head(),
           "body_model": B.model_name(), "table": "reports/openbody/joint_limits.json"}
    for name, fn in (("sit_arms_forward", lambda: arms_forward(B, base)), ("straight_elbow_tremor", lambda: straight_elbow(base, T)),
                     ("toe_tapping", lambda: toe_tapping(base, BR)), ("standing_posture", lambda: standing(TR)),
                     ("pill_rolling", lambda: pill(TR, H)), ("kinetic_pronation_supination", lambda: kinetic_ps(TR)),
                     ("seated_hands_and_feet", lambda: seated_hands_and_feet(TR, MO)),
                     ("face_touch", lambda: touch(RC))):
        rec[name] = fn()
        print(name, json.dumps(rec[name])[:600], flush=True)
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        json.dump(rec, f, indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
