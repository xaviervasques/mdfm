"""Solve the folded arms of the arising (motions.ARM_POSTURES['fold'], task sh1c) and write their record.

The folded arms of MDS-UPDRS 3.9 (shared change SH9 of docs/proposals/SYNTH_SHARED_CHANGES_DRAFT.md) were found by a
bounded search kept outside the tree (lane fold, task sh1b). The search is now mdfm.synth.motions.solve_fold: a
first stage toward the joint targets of FOLD_SOLVE (the elbows, wrists and middle knuckles from the spine3 joint,
the palms, the collars), then a second stage under the skin clearances of the six arm parts on the folded stand, of
each forearm and hand against the trunk over the rise of motions.arise and against the chair's arms. This script runs
it and writes:
- reports/synth/fold/fold_posture.json: the fold (0.01 deg), its segment directions and wrists (ARM_TARGETS['fold'],
  motions.fold_directions), the solver's digest (motions.fold_solver_digest: motions reads the record only when the
  digest matches the code), the stages, and the checks of the result on the skinned mesh: fold_report on the folded
  stand and over the rise of motions.arise built with it (with the chair), the joint-limit and hand-limit checks;
- reports/synth/fold/fold_posture.md: the same in a table.
motions.py reads ARM_POSTURES['fold'] and ARM_TARGETS['fold'] from the JSON at import. tests/test_synth_motions_fold.py
solves again and compares with the record.

Usage (CPU, about five minutes on two threads):
    CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 python scripts/solve_fold_posture.py
Nothing outside reports/synth/fold/ is written; no lockbox, EVAL, CODY or care_pd record is read.
"""
import json
import os
import sys
import time

import numpy as np

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if TREE not in sys.path:
    sys.path.insert(0, TREE)

from mdfm.synth import hands as H  # noqa: E402
from mdfm.synth import motions as MO  # noqa: E402
from mdfm.synth.body import limit_violations  # noqa: E402

OUT_DIR = os.path.join(TREE, "reports", "synth", "fold")


def verify(fold):
    """The checks of a fold on the skinned mesh: fold_report on the folded stand and over the rise of motions.arise
    built with it (every knot of the rise, with the chair), the joint and hand limits, the balance."""
    st = MO.posture_body(MO.STAND_REF, 0.02, fold=fold)
    fr_st = MO.fold_report(st, [0])
    b, info = MO.arise(standing_s=0.5, fold=fold)
    k_on, k_end = int(round(info["onset_s"] * b.rate)), int(round(info["end_s"] * b.rate))
    fr = MO.fold_report(b, np.arange(k_on, k_end + 1), MO.arise_chair(b, 0.0))

    def brief(r):
        out = {k: {"min_mm": round(v["min_mm"], 3), "at_s": v["at_s"], "max_mm": round(v["max_mm"], 3)}
               for k, v in r.items() if isinstance(v, dict) and "min_mm" in v}
        out.update({k: {"depth_mm_max": round(v["depth_mm_max"], 3), "at_s": v["at_s"]}
                    for k, v in r.items() if isinstance(v, dict) and "depth_mm_max" in v})
        out.update(ok=r["ok"], entry_ok=r["entry_ok"], n_knots=r["n_knots"])
        return out
    sit = MO.posture_body("sit_arms_folded", 0.02, fold=fold)
    return {"stand": brief(fr_st), "rise": brief(fr),
            "limits_outside": {"stand": limit_violations(st.rot6d)["outside"],
                               "sit": limit_violations(sit.rot6d)["outside"],
                               "rise": limit_violations(b.rot6d)["outside"]},
            "hand_limits_ok": bool(H.hand_limit_report(st.hand_R())["ok"]),
            "arise": {"balance_ok": info["balance"]["ok"], "trunk_deg": info["trunk"]["amplitude_deg"],
                      "standing_shank_tilt_deg": info["standing_shank_tilt_deg"],
                      "realized_ankle_plantarflexion_deg": info["realized_ankle_plantarflexion_deg"],
                      "standing_hip_drop_m": info["standing_hip_drop_m"]}}


def main():
    digest = MO.fold_solver_digest()      # before the solve: the source read is the imported code's (inspect reads
    t0 = time.time()                      # the file again, so an edit during the solve would mix two versions)
    res = MO.solve_fold()
    t_solve = time.time() - t0
    if MO.fold_solver_digest() != digest:
        raise RuntimeError("mdfm/synth/motions.py changed during the solve: run again, no record written")
    fold = res["fold"]
    rec = {"record": "the folded arms of the arising (motions.ARM_POSTURES['fold']), solved by motions.solve_fold",
           "script": "scripts/solve_fold_posture.py", "solver_digest": digest,
           "fold": {s: [list(a) for a in fold[s]] for s in ("left", "right")},
           "arm_targets": {s: {k: list(v) for k, v in d.items()} for s, d in MO.fold_directions(fold).items()},
           "joint_order": [list(MO.ARM_JOINTS["left"]), list(MO.ARM_JOINTS["right"])],
           "units": "axis-angle components (deg) of the local rotations of the collar, shoulder, elbow and wrist",
           "solve": {k: res[k] for k in ("ik", "stages", "gaps_mm", "rise")},
           "solve_settings": MO.FOLD_SOLVE, "verify": verify(fold), "solve_s": round(t_solve, 1)}
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "fold_posture.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(rec, f, indent=1, sort_keys=True, default=float)
        f.write("\n")
    v = rec["verify"]
    lines = ["# The folded arms of the arising (motions.ARM_POSTURES['fold'])", "",
             "Written by scripts/solve_fold_posture.py from mdfm.synth.motions.solve_fold; solver digest "
             f"{rec['solver_digest']}.", "",
             "| side | collar | shoulder | elbow | wrist |", "|---|---|---|---|---|"]
    for s in ("left", "right"):
        lines.append(f"| {s} | " + " | ".join(", ".join(f"{x:.2f}" for x in a) for a in fold[s]) + " |")
    lines += ["", "Least skin gaps (mm; negative inside) on the folded stand and over the rise of motions.arise "
              "(every knot, with the chair):", "", "| check | stand | rise |", "|---|---|---|"]
    for k in MO.FOLD_CHECKS:
        lines.append(f"| {k} | {v['stand'][k]['min_mm']:.2f} | {v['rise'][k]['min_mm']:.2f} |")
    for k in ("armrest_left", "armrest_right", "chair_left", "chair_right"):
        lines.append(f"| {k} (deepest entry) | | {v['rise'][k]['depth_mm_max']:.2f} |")
    lines += ["", f"fold_report ok: stand {v['stand']['ok']}, rise {v['rise']['ok']}; entry_ok: stand "
              f"{v['stand']['entry_ok']}, rise {v['rise']['entry_ok']}; joint-limit components outside: "
              f"{v['limits_outside']}; hand limits ok: {v['hand_limits_ok']}; arising balance ok: "
              f"{v['arise']['balance_ok']}.", "",
              f"Stages: first {res['stages']['ik']['nfev']} evaluations (cost {res['stages']['ik']['cost']:.6f}), "
              f"second {res['stages']['polish']['nfev']} evaluations (cost {res['stages']['polish']['cost']:.6f}, "
              f"status {res['stages']['polish']['status']}); {t_solve:.0f} s.", ""]
    with open(os.path.join(OUT_DIR, "fold_posture.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))
    print(json.dumps({"fold": rec["fold"], "verify_ok": [v["stand"]["ok"], v["rise"]["ok"]],
                      "solve_s": rec["solve_s"]}))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
