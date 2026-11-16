"""The level-0 reach's joint kinks before and after the path smoothing of the open-body port's wave 2
(docs/OPEN_BODY_PORT_DESIGN.md section 13; mdfm/synth/reach.py, _smooth_path).

For each side, the level-0 reach of reach.reach (the base of every E3 clip) is built twice on the selected body
model: with PATH_SMOOTH_WEIGHT 0 (the postures of the sample-by-sample continuation, the version before) and with
the module's weight. Per arm joint (collar, shoulder, elbow, wrist): the angular velocity of its local rotation
between successive knots (deg/s, the axis-angle of R(k+1) R(k)^T times the rate), its peak, and the largest change
of that velocity from one knot to the next (deg/s, the kink); per IK component the knots it sits on its IK bound;
the fingertip's distance from its line at the IK samples and the nose posture's change. Writes
reports/openbody/reach_kink.json. CPU only; no record of any dataset is read.
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
OUT = os.path.join(ROOT, "reports", "openbody", "reach_kink.json")
JOINTS = ("collar", "shoulder", "elbow", "wrist")


def _head():
    try:
        return subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def measure(RC, side, weight):
    from mdfm.body.rotations import matrix_to_axis_angle
    RC.PATH_SMOOTH_WEIGHT = weight
    RC._path.cache_clear()
    body, info = RC.reach(side)
    ch = RC.chain(side)
    R = body.local_R()
    moving = np.zeros(body.T - 2, bool)
    t = np.arange(body.T) / body.rate
    for p in info["phases"]:
        if p["kind"] in ("to_nose", "to_target"):
            moving |= (t[1:-1] >= p["t0_s"] - 1e-9) & (t[1:-1] <= p["t1_s"] + 1e-9)
    out = {"joints": {}}
    for name, j in zip(JOINTS, RC.ARM[side]):
        D = R[1:, j] @ R[:-1, j].transpose(-1, -2)
        w = torch.rad2deg(matrix_to_axis_angle(D)).numpy() * body.rate
        dw = np.linalg.norm(np.diff(w, axis=0), axis=1)
        k = int(np.argmax(np.where(moving, dw, -1.0)))
        out["joints"][name] = {"peak_speed_deg_s": round(float(np.linalg.norm(w, axis=1).max()), 2),
                               "largest_speed_change_deg_s": round(float(dw[moving].max()), 2),
                               "at_s": round(float(t[k + 1]), 2)}
    q = np.degrees(matrix_to_axis_angle(R[:, list(RC.ARM[side])]).numpy()).reshape(body.T, 12)
    at = np.isclose(q, ch.lo[None], atol=1e-6) | np.isclose(q, ch.hi[None], atol=1e-6)
    comp = [f"{n}.{c}" for n in JOINTS for c in "xyz"]
    out["knots_on_an_ik_bound"] = {comp[i]: int(at[:, i].sum()) for i in range(12) if at[:, i].any()}
    tg = RC.targets(side)
    s, qp = RC.path(side, tg["target"], tg["nose_target"], RC.target_posture(side), pointed=True)
    tip = ch.points(qp)["tip"].detach().numpy()
    line = tg["target"][None] + s[:, None] * (tg["nose_target"] - tg["target"])[None]
    out["tip_off_line_mm"] = round(float(1000.0 * np.linalg.norm(tip - line, axis=1).max()), 3)
    out["q_nose_deg"] = [round(float(x), 3) for x in qp[-1]]
    out["ik_residual_mm"] = {k: round(v, 3) for k, v in info["ik_residual_mm"].items()}
    out["index_from_vertical_deg"] = round(info["point_at_nose"]["index_from_vertical_deg"], 2)
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    from mdfm.synth import body as B
    from mdfm.synth import reach as RC
    weight = RC.PATH_SMOOTH_WEIGHT
    rec = {"script": "scripts/openbody_reach_kink.py", "run_at": datetime.datetime.now().isoformat(timespec="seconds"),
           "mdfm_commit": _head(), "body_model": B.model_name(), "path_smooth_weight": weight, "sides": {}}
    for side in RC.SIDES:
        before, after = measure(RC, side, 0.0), measure(RC, side, weight)
        rec["sides"][side] = {"before": before, "after": after,
                              "q_nose_change_deg": round(float(np.abs(np.subtract(after["q_nose_deg"],
                                                                                 before["q_nose_deg"])).max()), 3)}
        for name in JOINTS:
            print(side, name, before["joints"][name]["largest_speed_change_deg_s"], "->",
                  after["joints"][name]["largest_speed_change_deg_s"], "deg/s")
    RC.PATH_SMOOTH_WEIGHT = weight
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        json.dump(rec, f, indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
