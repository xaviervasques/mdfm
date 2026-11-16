"""The forearm's twist split on the open body (plan phase C, wave 1; docs/OPEN_BODY_PORT_DESIGN.md decision D4):
measures, on the catalogue body (openbody.catalogue_phenotype), the rigid forearm of wave 0 against the split of
wave 1 (openbody.OpenBody.twist_split: the elbow's axial rotation spread over lowerarm01 and lowerarm02 by the share
of each bone's segment midpoint along the forearm), and writes reports/openbody/forearm_twist.json:
- the twist bones: each forearm bone head's distance from the elbow-to-wrist axis and its fraction along it, the
  shares (the split rotates about the axis line through the elbow, so an off-axis head moves no vertex apart from its
  neighbours, and the program joints come from the rigid chain);
- for supination and pronation turns of the left elbow (the T pose, and the elbow flexed 90 deg): the largest
  difference of the 52 program joints between the two (0 by construction), and per station along the forearm (the
  vertices whose dominant program joint is the elbow, by their rest fraction along the axis) the mean turn of the
  skin about the axis and its mean distance from the axis over the rest one (below 1: the linear blend's collapse);
  the same distance ratio in the elbow zone and the wrist zone (the vertices of the upper arm, the forearm or the
  hand within ZONE_M of the elbow or the wrist along the axis).
Nothing but this report is written; the model data stay in memory.
"""
import datetime
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "reports", "openbody", "forearm_twist.json")
TURNS_DEG = (-90.0, -45.0, 45.0, 90.0)      # supination negative about the left forearm's +x (body.SUPINATION_SIGN)
STATIONS = np.round(np.arange(0.05, 1.0, 0.1), 2)
STATION_HALF = 0.05
ZONE_M = 0.03


def main():
    import torch
    from mdfm.body.rotations import axis_angle_to_matrix
    from mdfm.synth import openbody as OB
    m = OB.model()
    ph = OB.catalogue_phenotype()
    split, rigid = OB.OpenBody(ph, m, twist_split=True), OB.OpenBody(ph, m, twist_split=False)
    idx = OB.bone_index(m)
    hc = split.canonical_heads
    bones = {}
    for s in ("L", "R"):
        e, w = hc[idx[f"lowerarm01.{s}"]], hc[idx[f"wrist.{s}"]]
        u = (w - e) / (w - e).norm()
        for b in ("lowerarm02",):
            d = hc[idx[f"{b}.{s}"]] - e
            bones[f"{b}.{s}"] = {"off_axis_m": round(float((d - (d @ u) * u).norm()), 6),
                                 "fraction": round(float(d @ u / (w - e).norm()), 4)}
    shares = {m.bone_labels[b]: round(v, 4) for b, v in split.twist_shares().items()}
    J = split.rest_joints52().numpy()
    v0 = split.rest_vertices_np()
    dom = split.dominant_joint()
    e, w = J[18], J[20]
    L = float(np.linalg.norm(w - e))
    u = (w - e) / L
    frac = (v0 - e) @ u / L
    fore = np.flatnonzero(dom == 18)
    elbow_zone = np.flatnonzero(np.isin(dom, (16, 18)) & (np.abs(frac * L) <= ZONE_M))
    wrist_zone = np.flatnonzero(np.isin(dom, (18, 20)) & (np.abs((frac - 1.0) * L) <= ZONE_M))

    def radial(P, Je, Jw):
        a = (Jw - Je) / np.linalg.norm(Jw - Je)
        d = P - Je
        return d - (d @ a)[..., None] * a, a

    r0, _ = radial(v0, e, w)
    cases = []
    for flex in (0.0, 90.0):
        for turn in TURNS_DEG:
            R = torch.eye(3, dtype=OB.F64).repeat(1, 22, 1, 1)
            # the turn about the forearm (x in the T pose) first, then the elbow's flexion about y (left: -y)
            R[0, 18] = axis_angle_to_matrix(torch.tensor([0.0, -np.radians(flex), 0.0], dtype=OB.F64)) @ \
                axis_angle_to_matrix(torch.tensor([np.radians(turn), 0.0, 0.0], dtype=OB.F64))
            tr = torch.zeros(1, 3, dtype=OB.F64)
            Js, Vs = split.pose(R, tr)
            Jr, Vr = rigid.pose(R, tr)
            Jp = Js[0].numpy()
            row = {"elbow_flexion_deg": flex, "turn_deg": turn,
                   "joints_max_diff_m": float((Js - Jr).abs().max()), "stations": []}
            for name, V in (("split", Vs[0].numpy()), ("rigid", Vr[0].numpy())):
                rp, a = radial(V, Jp[18], Jp[20])
                # the rest radial vectors carried by the elbow's swing (the rigid forearm's rotation without its own
                # twist is not separable here: the turn is read against the rigid forearm's own skin at share 1)
                st = []
                for f in STATIONS:
                    sel = fore[np.abs(frac[fore] - f) <= STATION_HALF]
                    ratio = float(np.linalg.norm(rp[sel], axis=1).mean() / np.linalg.norm(r0[sel], axis=1).mean())
                    st.append({"fraction": float(f), "n": int(len(sel)), "radius_ratio": round(ratio, 4)})
                row[name] = {"stations": st,
                             "elbow_zone_radius_ratio": round(float(np.linalg.norm(rp[elbow_zone], axis=1).mean() /
                                                                    np.linalg.norm(r0[elbow_zone], axis=1).mean()), 4),
                             "wrist_zone_radius_ratio": round(float(np.linalg.norm(rp[wrist_zone], axis=1).mean() /
                                                                    np.linalg.norm(r0[wrist_zone], axis=1).mean()), 4)}
            # the skin's turn about the axis: split against rigid at each station (rigid carries the full turn)
            rs, a = radial(Vs[0].numpy(), Jp[18], Jp[20])
            rr, _ = radial(Vr[0].numpy(), Jp[18], Jp[20])
            for f, st in zip(STATIONS, row["split"]["stations"]):
                sel = fore[np.abs(frac[fore] - f) <= STATION_HALF]
                cr = np.cross(rr[sel], rs[sel]) @ a
                dt = (rr[sel] * rs[sel]).sum(1)
                lag = np.degrees(np.arctan2(cr, dt)).mean()
                st["turn_share"] = round(float(1.0 + lag / turn), 4)    # share of the turn the skin carries there
            del row["stations"]
            cases.append(row)
    head = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout
    out = {"script": "scripts/openbody_forearm_twist.py", "run_at": datetime.datetime.now().isoformat(timespec="seconds"),
           "mdfm_commit": head.strip(), "body": "openbody.catalogue_phenotype()", "twist_bones": bones,
           "shares": shares, "zone_m": ZONE_M, "station_half_width": STATION_HALF,
           "n_forearm_vertices": int(len(fore)), "n_elbow_zone": int(len(elbow_zone)),
           "n_wrist_zone": int(len(wrist_zone)), "cases": cases}
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: out[k] for k in ("twist_bones", "shares")}, indent=1))
    for c in cases:
        print(c["elbow_flexion_deg"], c["turn_deg"], "joints", c["joints_max_diff_m"],
              "split elbow/wrist", c["split"]["elbow_zone_radius_ratio"], c["split"]["wrist_zone_radius_ratio"],
              "rigid", c["rigid"]["elbow_zone_radius_ratio"], c["rigid"]["wrist_zone_radius_ratio"],
              "turn shares", [s["turn_share"] for s in c["split"]["stations"]],
              "min radius split/rigid", min(s["radius_ratio"] for s in c["split"]["stations"]),
              min(s["radius_ratio"] for s in c["rigid"]["stations"]))


if __name__ == "__main__":
    main()
