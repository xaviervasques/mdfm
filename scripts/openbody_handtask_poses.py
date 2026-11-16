"""The hand-task poses of MDS-UPDRS 3.4 and 3.5 solved again on the open body (plan phase C, wave 2;
docs/OPEN_BODY_PORT_DESIGN.md section 13): the two arm postures inside the anatomical joint-limit table, the closed
pinch of finger tapping and the fist of hand movements on the Anny surface. The numbers it prints are typed into the
modules (motions.ARM_POSTURES 'finger_tapping' and 'hand_movements', handtasks.PINCH and FIST, reach.POINT_POSE),
and tests/test_synth_pathology.py checks them; the report is reports/openbody/handtask_poses.json.

Rules (the modules' own, unchanged):
- arm postures (motions.py, Arm postures): the left arm's collar, shoulder, elbow and wrist components (deg) by a
  bounded least squares toward the segment directions of motions.ARM_TARGETS (upper arm, forearm, hand, palmar
  normal; 1 - cos of each angle), inside the table of reports/openbody/joint_limits.json less MARGIN_DEG, from the
  numbers solved on SMPL+H, rounded to 0.01 deg; the right arm is the mirror image;
- the pinch (handtasks.py, PINCH): the index MCP, PIP and DIP and the thumb CMC (flexion, palmar abduction), MCP and
  IP, inside the hand limits of hands.py, that bring the index fingertip vertex nearest the thumb's while the two
  distal segments' skins keep PINCH_GAP_MM apart (a penalty below it), from the SMPL+H numbers and PINCH_STARTS seeded starts (Powell), rounded to
  0.01 deg, on the finger-tapping posture;
- the fist (handtasks.py, FIST): every knuckle at its cited largest flexion, the thumb MCP at its cited largest, the
  thumb CMC at handtasks.FIST_THUMB_CMC (the open body's module angles); each digit's distal joints (PIP and DIP, the thumb's IP) at the largest common
  share of their cited largest flexions for which its distal skin keeps handtasks.FIST_CLEARANCE_MM from the palm
  skin (handtasks.palm_clearance_mm), by bisection per digit, rounded down to 0.01 deg, on the hand-movements
  posture.
CPU only; reads no dataset; writes the report only.
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
OUT = os.path.join(ROOT, "reports", "openbody", "handtask_poses.json")
MARGIN_DEG = 0.5            # [script] the arm postures this far inside the table
PINCH_GAP_MM = 2.0          # handtasks.py docstring: the distal skins at least 2 mm apart at closure
PINCH_SEED, PINCH_STARTS = 0, 12   # [script] the pinch search's restarts (uniform inside the hand limits)
F64 = torch.float64


def _head():
    try:
        return subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def solve_arm(name):
    from scipy.optimize import minimize
    from mdfm.body.rotations import axis_angle_to_matrix
    from mdfm.body.smplh import fk
    from mdfm.synth import body as B
    from mdfm.synth import hands as H
    from mdfm.synth import motions as MO
    from mdfm.synth.base import posture_rotations
    side = "left"
    tg = MO.ARM_TARGETS[name]
    joints = MO.ARM_JOINTS[side]
    lim = B.joint_limits()
    lo = np.concatenate([lim.lo[j] for j in joints]) + MARGIN_DEG
    hi = np.concatenate([lim.hi[j] for j in joints]) - MARGIN_DEG
    q_old = np.asarray(MO.ARM_POSTURES[name]["left"], dtype=np.float64).reshape(-1)
    R0 = posture_rotations(tg["base"])
    rest = B.rest_joints()
    J52, _, _ = H._rest()
    mid = H.hand_joint(side, "middle", 0)
    w = joints[3]
    hand_off = torch.as_tensor(J52[mid] - J52[w], dtype=F64)
    palmar = torch.as_tensor(H.hand_frame()[side]["palmar"], dtype=F64)
    keys = [k for k in ("upper", "fore", "hand", "palm") if k in tg]
    T = {k: torch.as_tensor(np.asarray(tg[k], dtype=np.float64) / np.linalg.norm(tg[k])) for k in keys}

    def feats(q):
        R = R0.clone()
        Rq = axis_angle_to_matrix(torch.deg2rad(q.reshape(4, 3)))
        R = torch.cat([R[:joints[0]], Rq[0:1], R[joints[0] + 1:]], 0)
        for i, j in enumerate(joints[1:], 1):
            R = torch.cat([R[:j], Rq[i:i + 1], R[j + 1:]], 0)
        J, Gl = fk(R[None], torch.zeros(1, 3, dtype=F64), rest)
        J, Gl = J[0], Gl[0]
        c, s, e, wj = joints
        return {"upper": J[e] - J[s], "fore": J[wj] - J[e], "hand": Gl[wj] @ hand_off, "palm": Gl[wj] @ palmar}

    def f(x):
        q = torch.tensor(x, dtype=F64, requires_grad=True)
        fe = feats(q)
        loss = sum(1.0 - (fe[k] / fe[k].norm()) @ T[k] for k in keys) + 1e-6 * ((q - torch.as_tensor(q_old)) ** 2).sum()
        loss.backward()
        return float(loss.detach()), q.grad.numpy().copy()
    x0 = np.clip(q_old, lo, hi)
    r = minimize(f, x0, jac=True, method="L-BFGS-B", bounds=list(zip(lo, hi)),
                 options={"maxiter": 2000, "ftol": 1e-15, "gtol": 1e-12})
    q = np.round(r.x, 2)
    q = np.clip(q, lo, hi)
    return [tuple(float(v) for v in q[3 * i:3 * i + 3]) for i in range(4)], q_old


def _pinch_body(MO, HT, pinch):
    """The finger-tapping posture body with the closed hand of the pinch angles."""
    HT.PINCH.clear()
    HT.PINCH.update(pinch)
    return MO.posture_body("sit_finger_tapping", 0.01)


def solve_pinch():
    from scipy.optimize import minimize
    from mdfm.synth import hands as H
    from mdfm.synth import handtasks as HT
    from mdfm.synth import motions as MO
    old = {k: v for k, v in HT.PINCH.items()}
    x_old = np.array([*old["index"], *old["thumb"][0], old["thumb"][1], old["thumb"][2]], dtype=np.float64)
    L = H.FLEX_LIMITS_DEG
    lo = np.array([L["index"]["mcp"][0], L["index"]["pip"][0], L["index"]["dip"][0], -H.CMC_LIMIT_DEG,
                   -H.CMC_LIMIT_DEG, L["thumb"]["mcp"][0], L["thumb"]["ip"][0]])
    hi = np.array([L["index"]["mcp"][1], L["index"]["pip"][1], L["index"]["dip"][1], H.CMC_LIMIT_DEG,
                   H.CMC_LIMIT_DEG, L["thumb"]["mcp"][1], L["thumb"]["ip"][1]])

    def pinch_of(x):
        x = np.round(np.clip(x, lo, hi), 2)
        return {"index": (float(x[0]), float(x[1]), float(x[2])),
                "thumb": ((float(x[3]), float(x[4])), float(x[5]), float(x[6]))}

    def cost(x):
        x = np.clip(x, lo, hi)
        b = _pinch_body(MO, HT, {"index": tuple(x[:3]), "thumb": ((x[3], x[4]), x[5], x[6])})
        d = 1000.0 * HT.tip_distance_m(b, "left", knots=[0])[0]
        g = HT.pinch_gap_mm(b, "left")
        return d + 50.0 * max(PINCH_GAP_MM - g, 0.0) ** 2
    best, rng, starts = None, np.random.default_rng(PINCH_SEED), []
    for _ in range(PINCH_STARTS):            # the SMPL+H numbers stall at 18 mm on Anny: seeded restarts
        x0 = lo + rng.uniform(0.15, 0.85, 7) * (hi - lo)
        x0[[0, 1, 2]] = rng.uniform([10, 20, 10], [60, 80, 70])
        starts.append(x0)
    for x0 in [x_old] + starts:
        r = minimize(cost, x0, method="Powell", bounds=list(zip(lo, hi)),
                     options={"maxiter": 3000, "xtol": 1e-2, "ftol": 1e-5})
        if best is None or r.fun < best.fun:
            best = r
        print("pinch start", round(float(r.fun), 3), flush=True)
    pinch = pinch_of(best.x)
    b = _pinch_body(MO, HT, pinch)
    rec = {"closed_tip_distance_mm": float(1000.0 * HT.tip_distance_m(b, "left", knots=[0])[0]),
           "pinch_gap_mm": float(HT.pinch_gap_mm(b, "left"))}
    a = HT.amplitude_of_share(b, "finger_tapping", "left", np.linspace(0.0, 1.0, 101))
    rec.update(amplitude_rises=bool((np.diff(a) > 0).all()), open_amplitude_mm=float(a[-1]))
    return pinch, old, rec


def solve_fist():
    from mdfm.synth import hands as H
    from mdfm.synth import handtasks as HT
    from mdfm.synth import motions as MO
    old = {k: v for k, v in HT.FIST.items()}
    cited = HT.FIST_CITED
    b = MO.posture_body("sit_hand_movements", 0.02)

    def with_fist(fist):
        pose = {("left", f, lv): v for f, vals in fist.items() for lv, v in enumerate(vals)}
        return b.with_hands(H.hand_rotations(pose)[None].expand(b.T, -1, -1, -1))

    cited = dict(cited)
    cited["thumb"] = (tuple(HT.FIST_THUMB_CMC), cited["thumb"][1], cited["thumb"][2])

    def digit(d, k):
        c = cited[d]
        return (c[0], c[1], round(k * c[2], 9)) if d == "thumb" else (c[0], round(k * c[1], 9), round(k * c[2], 9))

    out, shares = {}, {}
    for d in ("index", "middle", "ring", "pinky", "thumb"):
        def clear(k):
            f = dict(cited)
            f[d] = digit(d, k)
            return HT.palm_clearance_mm(with_fist(f), "left", digits=(d,))[d]
        lo_k, hi_k = 0.0, 1.0
        if clear(lo_k) < HT.FIST_CLEARANCE_MM:
            raise RuntimeError(f"fist: the {d}'s distal skin is within the clearance with its distal joints straight")
        if clear(hi_k) >= HT.FIST_CLEARANCE_MM:
            lo_k = 1.0
        for _ in range(40):
            m = 0.5 * (lo_k + hi_k)
            if clear(m) >= HT.FIST_CLEARANCE_MM:
                lo_k = m
            else:
                hi_k = m
        c = cited[d]
        if d == "thumb":
            v = (c[0], c[1], math.floor(100.0 * lo_k * c[2]) / 100.0)
        else:
            v = (c[0], math.floor(100.0 * lo_k * c[1]) / 100.0, math.floor(100.0 * lo_k * c[2]) / 100.0)
        out[d], shares[d] = v, lo_k
    fist = {d: out[d] for d in ("index", "middle", "ring", "pinky", "thumb")}
    clearance = HT.palm_clearance_mm(with_fist(fist), "left")
    _, _, dom = H._rest()
    th = np.flatnonzero(np.isin(dom, [H.hand_joint("left", "thumb", 1), H.hand_joint("left", "thumb", 2)]))
    fi = np.flatnonzero(np.isin(dom, [H.hand_joint("left", f, lv) for f in ("index", "middle") for lv in (0, 1, 2)]))
    P = H.body_points(with_fist(fist), np.r_[th, fi], knots=[0])[0]
    smplh_thumb = {k: round(HT.palm_clearance_mm(with_fist(dict(HT.FIST_CITED, thumb=(old["thumb"][0], old["thumb"][1],
                                                                                         ip))), "left",
                                                 digits=("thumb",))["thumb"], 3)
                   for k, ip in (("ip_straight", 0.0), ("smplh_fist_ip", old["thumb"][2]))}
    thumb_fingers_mm = float(1000.0 * np.sqrt(((P[:len(th), None] - P[None, len(th):]) ** 2).sum(-1)).min())
    cited_clear = HT.palm_clearance_mm(with_fist(HT.FIST_CITED), "left")
    return fist, old, {"shares_of_cited_distal_flexion": {d: round(s, 4) for d, s in shares.items()},
                       "clearance_mm": {d: round(v, 4) for d, v in clearance.items()},
                       "cited_clearance_mm": {d: round(v, 3) for d, v in cited_clear.items()},
                       "thumb_to_index_middle_skin_mm": round(thumb_fingers_mm, 3),
                       "thumb_cmc_deg": list(HT.FIST_THUMB_CMC),
                       "smplh_thumb_cmc_clearance_mm": smplh_thumb}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    from mdfm.synth import body as B
    from mdfm.synth import hands as H
    from mdfm.synth import handtasks as HT
    from mdfm.synth import motions as MO
    rec = {"script": "scripts/openbody_handtask_poses.py",
           "run_at": datetime.datetime.now().isoformat(timespec="seconds"), "mdfm_commit": _head(),
           "body_model": B.model_name(), "table": "reports/openbody/joint_limits.json", "margin_deg": MARGIN_DEG,
           "arm_postures": {}}
    for name in ("finger_tapping", "hand_movements"):
        before = {s: MO.arm_errors(name, s) for s in ("left", "right")}
        new, old = solve_arm(name)
        MO.ARM_POSTURES[name] = {"left": tuple(new)}
        after = {s: MO.arm_errors(name, s) for s in ("left", "right")}
        post = {"finger_tapping": "sit_finger_tapping", "hand_movements": "sit_hand_movements"}[name]
        viol = {s: int(B.limit_violations(MO.posture_body(post, 0.05, side=s).rot6d)["outside"]) for s in ("left", "right")}
        rec["arm_postures"][name] = {"left": new, "smplh_numbers": [list(old[3 * i:3 * i + 3]) for i in range(4)],
                                     "errors_deg_before": before, "errors_deg_after": after,
                                     "knots_outside_after": viol}
        print(name, new, {k: round(v, 2) for k, v in after["left"].items()}, viol)
    pinch, old_p, rp = solve_pinch()
    rec["pinch"] = {"pinch": pinch, "smplh_numbers": old_p, **rp}
    print("pinch", pinch, rp)
    fist, old_f, rf = solve_fist()
    rec["fist"] = {"fist": fist, "smplh_numbers": old_f, **rf}
    print("fist", fist, rf)
    b = MO.posture_body("sit_hand_movements", 0.02)
    rec["fist"]["hand_limits_ok"] = bool(H.hand_limit_report(H.hand_rotations(
        {("left", f, lv): v for f, vals in fist.items() for lv, v in enumerate(vals)}))["ok"])
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        json.dump(rec, f, indent=1)
    print("wrote", OUT, b.T)


if __name__ == "__main__":
    main()
