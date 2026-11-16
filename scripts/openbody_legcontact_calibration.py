"""The leg-to-leg contact check's calibration read again on the open body (open-body port, wave 3 follow-up;
docs/OPEN_BODY_PORT_DESIGN.md section 14.8, decisions D36 to D38): each value of mdfm/synth/legcontact.py that was a
reading of the SMPL+H surface, by its own rule, on the selected body model (the open body by default), with the
reference's scalar readings beside it behind its flag (MDFM_BODY_MODEL=smplh_reference, MDFM_SMPLH_REFERENCE=1, for its
half only):
- the inner thigh band (CROTCH_BAND_M, its rule 'about the narrowest band whose rest gap is positive on the neutral
  mesh'): the rest gap past bands of 0 to 8 cm, and the narrowest of them whose gap is positive;
- the knee zone (KNEE_ZONE_M, anatomical: the condyles' height over the knee joint): how high above the knee joint the
  knee's own skinning reaches on each mesh, for the record;
- the healthy walk of gait.py (the test's: gait.NORMAL, 7 s, seed 7), the former healthy walk (the 0.09 m base without
  the swing guard nor the turn's widening, the basis of the 2026-11-03 calibration) and the healthy clips of the gait
  catalogue (the level-0 course, the tandem of SARA 0, the pull test), with the band derived here and with the
  SMPL+H band: the smallest hard, soft and distal gaps and the check (SOFT_TOL_M's rule: the healthy walks read their
  deepest inner-thigh contact inside it);
- the seated knees (involuntary.base_body('sit'), the knees swivelled inward about their hip-to-ankle lines, split 7 to
  5 as the knee guard's test): the hard and soft gaps along the closing, the closing at which the inner thighs first
  touch, where along the thigh that contact lies, the closing ceiling of the hard gap (involuntary.closing_ceiling) and
  the soft gap there on each of involuntary.CLOSING_SPLITS: the derivation of the knee guard tests' soft reading.
Writes reports/openbody/legcontact_calibration.json. Distances and counts only: no vertex array is written. CPU only;
reads no dataset.
"""
import datetime
import json
import os
import subprocess
import sys
from dataclasses import replace

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "reports", "openbody", "legcontact_calibration.json")
BANDS = (0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08)
CLOSINGS = (0.0, 4.0, 8.0, 10.0, 11.0, 12.0, 14.0, 16.0, 20.0)
HEALTHY_IDS = ("gait_E8_level0", "gait_E8t_tandem_sara0", "gait_E7p_retropulsion_g0")


def _git(path, *args):
    try:
        return subprocess.run(["git", *args, "-C", path, "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def _mm(x):
    return round(1000.0 * float(x), 2)


def _with_band(LC, band):
    """legcontact with its band for the selected model set to band (the caches cleared)."""
    from mdfm.synth import openbody as OB
    from mdfm.synth.body import model_name
    name = "CROTCH_BAND_REFERENCE_M" if model_name() == OB.SMPLH_REFERENCE else "CROTCH_BAND_M"
    setattr(LC, name, float(band))
    LC._mesh.cache_clear()


def band_scan():
    """The rest gap (mm, exact to 0.2 m) past each band of BANDS, the band the module holds, and the narrowest band of
    the scan whose rest gap is positive (the band's rule)."""
    from mdfm.synth import legcontact as LC
    held = LC.crotch_band_m()
    rows = {}
    try:
        for b in BANDS:
            _with_band(LC, b)
            r = LC.rest_gap()
            rows[f"{b:.2f}"] = {"rest_gap_mm": _mm(r["gap_m"]), "where": r["where"]}
    finally:
        _with_band(LC, held)
    narrow = next((b for b in BANDS if rows[f"{b:.2f}"]["rest_gap_mm"] > 0.0), None)
    return {"held_band_m": held, "rest_gap_by_band": rows, "narrowest_positive_band_m": narrow}


def knee_skinning():
    """How high above the knee joint (cm, along the femur, rest mesh) the vertices whose largest skinning weight lies
    on the knee reach, per side (the highest and the 95th percentile)."""
    from mdfm.synth import viewer as V
    from mdfm.synth.body import body_model
    v, dom = V.rest_vertices(), V.dominant_joint()
    J = body_model().rest_joints52().numpy()
    out = {}
    for side, (h, k) in (("left", (1, 4)), ("right", (2, 5))):
        u = (J[h] - J[k]) / np.linalg.norm(J[h] - J[k])
        up = (v[dom == k] - J[k]) @ u
        out[side] = {"knee_vertices_above_joint_cm_max": round(100.0 * float(up.max()), 2),
                     "knee_vertices_above_joint_cm_p95": round(100.0 * float(np.percentile(up, 95)), 2)}
    return out


def _walk_row(LC, b):
    g = LC.leg_gap(b)
    v = LC.validity(b, g)
    return {"hard_min_mm": _mm(v["hard_min_m"]), "hard_where": v["hard_where"], "soft_min_mm": _mm(v["soft_min_m"]),
            "soft_t_s": round(v["soft_t_s"], 2), "distal_min_mm": _mm(v["distal_min_m"]),
            "hard_knots_beyond_tol": v["hard_knots_beyond_tol"], "soft_knots_beyond_tol": v["soft_knots_beyond_tol"],
            "ok": bool(v["ok"])}


def walks(bands, catalogue=True):
    """The walks' readings (module docstring) with each band of bands {label: m}."""
    from mdfm.synth import gait
    from mdfm.synth import legcontact as LC
    bodies = {"healthy_walk": gait.generate(gait.NORMAL, duration_s=7.0, start_at_s=1.0, stop_at_s=5.0, seed=7)[0],
              "former_healthy_walk": gait.generate(replace(gait.NORMAL, step_width_m=0.09, leg_gap_m=None,
                                                           turn_base_m=0.0), duration_s=7.0, start_at_s=1.0,
                                                   stop_at_s=5.0, seed=7)[0]}
    if catalogue:
        sp = {s.clip_id: s for s in gait.specs()}
        for cid in HEALTHY_IDS:
            bodies[cid] = sp[cid].build()[0]
    held = LC.crotch_band_m()
    out = {}
    try:
        for label, band in bands.items():
            _with_band(LC, band)
            out[label] = {"band_m": band, **{k: _walk_row(LC, b) for k, b in bodies.items()}}
            print("walks", label, out[label], flush=True)
    finally:
        _with_band(LC, held)
    return out


def seated_knees():
    """The seated knees' closing on the held band (module docstring)."""
    from mdfm.synth import involuntary as I
    from mdfm.synth import legcontact as LC
    from scipy.spatial import cKDTree
    base = I.base_body("sit", 0.01)
    T = base.T
    mesh = LC.leg_mesh()

    def swivelled(left, right):
        b = base
        for side, m in (("left", left), ("right", right)):
            R, _ = I.seated_legs(b, side, swivel_deg=-I.SWIVEL_OUT[side] * np.full(T, m))
            b = b.with_rotations(R, list(I.LEG[side][:3]))
        return b

    def gaps(b):
        g = LC.gap_from_pose(b.local_R().numpy(), b.transl.numpy(), cull_m=0.2)
        return float(g["hard_m"][0]), float(g["soft_m"][0])

    def at(S):
        return gaps(swivelled(7.0 / 12.0 * S, 5.0 / 12.0 * S))
    rows = {}
    for S in CLOSINGS:
        h, s = at(S)
        rows[f"{S:g}"] = {"hard_mm": _mm(h), "soft_mm": _mm(s)}
    lo, hi = 0.0, 20.0                                   # the closing at which the inner thighs first touch
    for _ in range(30):
        m = 0.5 * (lo + hi)
        lo, hi = (m, hi) if at(m)[1] > 0.0 else (lo, m)
    onset = 0.5 * (lo + hi)
    b = swivelled(7.0 / 12.0 * onset, 5.0 / 12.0 * onset)
    sk = LC.skin_legs(b.local_R().numpy()[:1], b.transl.numpy()[:1], mesh)
    d, j = cKDTree(sk["right"][0]).query(sk["left"][0])
    o = np.argsort(d)[:10]                               # the ten closest vertex pairs at the onset
    along = 0.5 * (mesh["left"]["along_v"][o] + mesh["right"]["along_v"][j[o]])
    c = I.closing_ceiling(base, 0)
    ceil = {}
    for a, off in I.CLOSING_SPLITS:
        left = a * c + off
        h, s = gaps(swivelled(left, c - left))
        ceil[f"{a},{off}"] = {"hard_mm": _mm(h), "soft_mm": _mm(s)}
    rec = {"band_m": LC.crotch_band_m(), "split": [7, 5], "closing_deg": rows,
           "soft_onset_closing_deg": round(onset, 2),
           "soft_onset_contact_above_knee_cm": [round(100.0 * float(along.min()), 1),
                                                round(100.0 * float(along.max()), 1)],
           "knee_zone_cm": 100.0 * LC.KNEE_ZONE_M, "ceiling_deg": round(c, 3),
           "keep_mm": _mm(I.KNEE_CLEARANCE_M + I.KNEE_GUARD_MARGIN_M), "at_ceiling": ceil,
           "soft_mm_at_ceiling_min": min(v["soft_mm"] for v in ceil.values()),
           "soft_tol_mm": _mm(LC.SOFT_TOL_M)}
    print("seated", rec, flush=True)
    return rec


def readings(reference=False):
    from mdfm.synth import legcontact as LC
    from mdfm.synth.body import model_name
    rec = {"body_model": model_name(), "band_scan": band_scan(), "knee_skinning": knee_skinning()}
    print(rec["body_model"], rec["band_scan"], rec["knee_skinning"], flush=True)
    if reference:
        return rec
    derived = rec["band_scan"]["narrowest_positive_band_m"]
    rec["walks"] = walks({"derived_band": derived, "smplh_band": LC.CROTCH_BAND_REFERENCE_M})
    rec["seated_knees"] = seated_knees()
    rec["derived"] = {
        "crotch_band_m": derived,
        "crotch_band_rule": "the narrowest band of the scan whose rest gap is positive on the neutral mesh",
        "module_crotch_band_m": LC.CROTCH_BAND_M,
        "knee_zone_m": LC.KNEE_ZONE_M, "distal_band_m": LC.DISTAL_BAND_M,
        "knee_zone_rule": "anatomical (the femoral condyles over the knee joint), not a reading of the mesh: kept",
        "soft_tol_m": LC.SOFT_TOL_M,
        "soft_tol_rule": "the healthy walks read their deepest inner-thigh contact inside it; on the open body they "
                         "press none (walks), so the module value stays, read again by wave 5 on the gait clips",
        "tol_m": LC.TOL_M}
    return rec


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    torch.set_num_threads(4)
    from mdfm.synth import openbody as OB
    for k in (OB.BACKEND_ENV, OB.REFERENCE_ENV):
        os.environ.pop(k, None)
    res = {"open": readings()}
    os.environ[OB.BACKEND_ENV], os.environ[OB.REFERENCE_ENV] = OB.SMPLH_REFERENCE, "1"
    try:
        res["smplh_reference"] = readings(reference=True)
    finally:
        os.environ.pop(OB.BACKEND_ENV), os.environ.pop(OB.REFERENCE_ENV)
    out = {"script": "scripts/openbody_legcontact_calibration.py",
           "run_at": datetime.datetime.now().isoformat(timespec="seconds"), "mdfm_commit": _git(ROOT),
           "moveatlas_commit": _git("<moveatlas>", "-c", "safe.directory=<moveatlas>"),
           "note": "the reference's SMPL+H model file is read from MoveAtlas, in memory; scalars only", **res}
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
