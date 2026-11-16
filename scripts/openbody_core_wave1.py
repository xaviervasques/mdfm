"""The core of the synthetic patient on the open body (plan phase C, wave 1; docs/OPEN_BODY_PORT_DESIGN.md section
12): the readings of the core's mesh rules on the catalogue body (openbody.catalogue_phenotype) beside the SMPL+H
reference, written to reports/openbody/core_wave1.json:
- the meshes: vertices, faces, the signed volume of the rest mesh (positive: the faces wound outward, the culling of
  raster.py), the rest stature, the sole offset (viewer.sole_offset_m: the lowest foot joint over the lowest vertex);
- the sets by rule: the sole clouds and their heel and toe ends (contact.py), the fingertips (hands.tip_vertices),
  the leg surfaces (legcontact.leg_mesh: vertices, faces, the crotch height), the head landmarks
  (skeleton.head_landmarks) and the head vertices' mean and extent midpoint (the rule of the head centre), the
  vertices of the inner parts (dominant joint -1 on the open body);
- the leg-to-leg contact readings that the SMPL+H calibration rests on: the rest gap past the inner thigh band, with
  the band halved, and the healthy walk of gait.py (its smallest soft and hard gaps, the knots beyond the tolerances);
- the guard: viewer.mesh_model refuses on the open body.
The reference is read behind its flag (MDFM_BODY_MODEL=smplh_reference, MDFM_SMPLH_REFERENCE=1, set here for its
half only). Joint positions, counts and distances only: no vertex array is written.
"""
import datetime
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "reports", "openbody", "core_wave1.json")


def readings():
    from mdfm.synth import contact as C
    from mdfm.synth import gait, hands as H, legcontact as LC, skeleton as SK, viewer as V
    from mdfm.synth.body import model_name, rest_joints
    v, f = V.rest_vertices(), V.faces()
    vol = float(np.einsum("ij,ij->i", v[f[:, 0]], np.cross(v[f[:, 1]], v[f[:, 2]])).sum() / 6.0)
    sole = C.sole_vertices()
    ht = C.heel_toe()
    m = LC.leg_mesh()
    r = rest_joints().numpy()
    from mdfm.synth.body import rest_joints52
    J52 = rest_joints52().numpy()
    out = {"body_model": model_name(), "n_vertices": int(len(v)), "n_faces": int(len(f)),
           "signed_volume_m3": round(vol, 5), "rest_stature_m": round(float(v[:, 1].max() - v[:, 1].min()), 4),
           "sole_offset_m": round(V.sole_offset_m(), 5),
           "rest_pelvis_over_lowest_foot_joint_m": round(float(r[0, 1] - min(r[10, 1], r[11, 1])), 4),
           "sole_cloud": {s: int(len(ids)) for s, ids in sole.items()},
           "heel_toe": {s: [int(len(a)), int(len(b))] for s, (a, b) in ht.items()},
           "fingertips": {s: len(d) for s, d in H.tip_vertices().items()},
           "leg_mesh": {s: {"vertices": int(len(m[s]["ids"])), "faces": int(len(m[s]["F"]))} for s in LC.SIDES},
           "crotch_y_m": round(float(m["crotch_y"]), 4), "head_landmarks_m": SK.head_landmarks()}
    dom = V.dominant_joint()
    hv = v[dom == SK.HEAD] - J52[SK.HEAD]
    out["head_vertices"] = {"n": int(len(hv)), "mean_m": [round(float(x), 4) for x in hv.mean(0)],
                            "extent_midpoint_m": [round(float(x), 4) for x in 0.5 * (hv.min(0) + hv.max(0))],
                            "inner_part_vertices": int((dom < 0).sum())}
    gap = LC.rest_gap()["gap_m"]
    # the band of the model read (legcontact.crotch_band_m: since the wave 3 follow-up, D36, the open body's is 0 and
    # the reference keeps CROTCH_BAND_REFERENCE_M; this script ran on 8 cm for both, core_wave1.json)
    from mdfm.synth import openbody as OB
    attr = "CROTCH_BAND_REFERENCE_M" if model_name() == OB.SMPLH_REFERENCE else "CROTCH_BAND_M"
    b0 = getattr(LC, attr)
    try:
        setattr(LC, attr, 0.5 * b0)
        LC._mesh.cache_clear()
        half = LC.rest_gap()["gap_m"]
    finally:
        setattr(LC, attr, b0)
        LC._mesh.cache_clear()
    b, _ = gait.generate(gait.NORMAL, duration_s=7.0, start_at_s=1.0, stop_at_s=5.0, seed=7)
    g = LC.leg_gap(b)
    rep = LC.report(g, b.rate)
    out["legcontact"] = {"rest_gap_m": round(float(gap), 5), "rest_gap_half_band_m": round(float(half), 5),
                         "crotch_band_m": b0, "healthy_walk_soft_min_m": round(float(g["soft_m"].min()), 5),
                         "healthy_walk_hard_min_m": round(float(g["hard_m"].min()), 5),
                         "soft_knots_beyond_tol": int(rep["soft_knots_beyond_tol"]),
                         "hard_knots_beyond_tol": int(rep["hard_knots_beyond_tol"]),
                         "soft_tol_m": LC.SOFT_TOL_M, "tol_m": LC.TOL_M}
    return out


def main():
    from mdfm.synth import openbody as OB, viewer as V
    for k in (OB.BACKEND_ENV, OB.REFERENCE_ENV):
        os.environ.pop(k, None)
    res = {"open": readings()}
    try:
        V.mesh_model("neutral")
        refused = False
    except OB.OpenBodyRefused:
        refused = True
    res["open"]["mesh_model_refused"] = refused
    os.environ[OB.BACKEND_ENV], os.environ[OB.REFERENCE_ENV] = OB.SMPLH_REFERENCE, "1"
    try:
        res["smplh_reference"] = readings()
    finally:
        os.environ.pop(OB.BACKEND_ENV), os.environ.pop(OB.REFERENCE_ENV)
    head = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout
    ma = subprocess.run(["git", "-c", "safe.directory=<moveatlas>", "-C", "<moveatlas>",
                         "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout
    out = {"script": "scripts/openbody_core_wave1.py", "run_at": datetime.datetime.now().isoformat(timespec="seconds"),
           "mdfm_commit": head.strip(), "moveatlas_commit": ma.strip(),
           "note": "the reference's SMPL+H model file is read from MoveAtlas, in memory", **res}
    with open(OUT, "w", encoding="utf-8", newline="\n") as fo:
        json.dump(out, fo, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
