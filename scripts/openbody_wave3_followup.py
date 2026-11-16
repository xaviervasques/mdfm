"""The open-body port's wave 3 follow-up readings that the design's section 14.8 cites (docs/OPEN_BODY_PORT_DESIGN.md,
decisions D39 to D42), on the selected body model (the open body by default):
- functional: the seated functional axial jerks of the catalogue (jerks, F03): per jerk its size factor, the head's
  largest forward and downward excursions against the level 0 and the drop a rigid bow about the hip line gives that
  forward excursion (f^2 / 2r, r the base's hip-to-head distance), the bound the test reads;
- rocking: the seated rocking (involuntary.td_rocking at AIMS steps 3 and 4, stereotypy 'rocking' at grade 3, 8 s as
  the tests build them): the declared trunk pitch, the pitch read at the upper trunk, the trunk_flex channel's forward
  room by its ROM split (channel_room) against the chain's room split by room (room_split) and its shares, the trunk
  region's share of time moving;
- rest: the seated rest of the hands solved again (involuntary.solve_hands_on_thighs) along the thigh (THIGH_SHARE)
  and at several palm pitches (PALM_PITCH_DEG): the palm's centre patch, its nearest vertex, the hand's and the
  elbow's skin distance to the trunk and the forearm's slope;
- carriage: the seated hands carried on the thighs while the trunk moves (the catalogue's seated grade-3 and grade-4
  trunk chorea; the seated grade-3 chorea of the test, 8 s), on wave 3's rest with the wrist's twist kept at the wrist,
  on the same rest with the twist handed to the elbow (involuntary._twist_to_elbow), and on the module's rest: the
  depth the carriage without the elbow's swing takes the hand and the elbow into the trunk, the knots where the table
  scaled a wrist, the palm's centre patch over the clip (palm_rest_report), the twist handed, the elbow's swing and
  lift runs, the arms' contact entry, the arms' own movement against their nominal share and the guard's depth left
  (the grade-3 chorea), and for the trunk clips the share of the clip the catalogue test reads with every carriage run
  left out and with the runs that lift a hand left out.
Writes reports/openbody/wave3_followup.json. CPU only; reads no dataset.
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
OUT = os.path.join(ROOT, "reports", "openbody", "wave3_followup.json")


def _head():
    try:
        return subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def _r(x, n=2):
    return round(float(x), n)


def functional():
    from mdfm.synth import jerks as JK
    sp = {s.clip_id: s for s in JK.specs()}
    L = sp["jerk_E1_level0"].build()[0]
    Jb = L.joints().numpy()
    r = float(np.linalg.norm(Jb[0, 15] - 0.5 * (Jb[0, 1] + Jb[0, 2])))
    rec = {"hip_to_head_m": _r(r, 3), "clips": {}}
    for cid in [c for c in sp if c.startswith("jerk_functional_E1")]:
        b, t = sp[cid].build()
        Jo = b.joints().numpy()
        rows = []
        for e in t["params"]["events"]:
            k0, k1 = (np.array(JK.event_span(e)) * 100).round().astype(int)
            d = Jo[k0:k1, 15] - Jb[k0:k1, 15]
            f = float(d[:, 2].max())
            rows.append({"factor": _r(e["factor"], 3), "forward_cm": _r(100 * f), "down_cm": _r(-100 * d[:, 1].min()),
                         "bow_drop_cm": _r(100 * f * f / (2 * r), 3),
                         "trunk_deg": _r(JK.deviation_deg(b, L, [3, 6, 9]).sum(1)[k0:k1].max(), 1)})
        rec["clips"][cid] = {"severity_share": t["params"]["severity_share"], "jerks": rows,
                             "down_cm_min": min(x["down_cm"] for x in rows),
                             "under_1cm": sum(x["down_cm"] < 1.0 for x in rows),
                             "under_bow_drop": sum(x["down_cm"] < x["bow_drop_cm"] for x in rows)}
        print("functional", cid, rec["clips"][cid]["down_cm_min"], flush=True)
    return rec


def _pitch_pp(out, base, j=9):
    (_, G), (_, Gb) = (x.numpy() for x in out.fk()), (x.numpy() for x in base.fk())
    v, vb = G[:, j] @ np.array([0.0, 1.0, 0.0]), Gb[:, j] @ np.array([0.0, 1.0, 0.0])
    p = np.degrees(np.arctan2(v[:, 2], v[:, 1])) - np.degrees(np.arctan2(vb[:, 2], vb[:, 1]))
    return float(p.max() - p.min()), float(p.min())


def rocking(seconds=8.0):
    from mdfm.synth import involuntary as I
    b = I.base_body("sit", 0.1)
    split, room = I.room_split(b, "trunk_flex")
    rec = {"trunk_flex_room_by_rom_split_deg": [_r(x) for x in I.channel_room(b, "trunk_flex")],
           "rom_split": {int(j): _r(s, 3) for j, _, s in I.channel_axes(b, "trunk_flex")},
           "trunk_flex_room_by_room_split_deg": _r(room), "room_split": {int(j): _r(s, 3) for j, _, s in split},
           "pelvis_share": I.PELVIS_TILT_SHARE, "clips": {}}
    base = I.base_body("sit", seconds)
    for step in (3, 4):
        out, tr = I.td_rocking(step, seconds=seconds)
        pp, mn = _pitch_pp(out, base)
        rec["clips"][f"td_rocking_{step}"] = {"declared_deg_pp": I.TD_ROCK_DEG[step], "pitch_deg_pp": _r(pp),
                                              "pitch_min_deg": _r(mn), "limits_new": tr["validity"]["limits_new"],
                                              "trunk_share": _r(tr["params"]["measured"]["regions"]["trunk"]["share"], 3)}
        print("rocking", step, rec["clips"][f"td_rocking_{step}"], flush=True)
    out, tr = I.stereotypy("rocking", 3, seconds=seconds)
    pp, mn = _pitch_pp(out, base)
    m = tr["params"]["measured"]["regions"]["trunk"]
    rec["clips"]["stereotypy_rocking_3"] = {"declared_deg_pp": I.STEREO["rocking"]["deg_pp"], "pitch_deg_pp": _r(pp),
                                            "trunk_share": _r(m["share"], 3), "dominant_hz": _r(m["dominant_hz"], 3),
                                            "limits_new": tr["validity"]["limits_new"]}
    print("rocking stereotypy", rec["clips"]["stereotypy_rocking_3"], flush=True)
    orig = I._entry                                   # before: the spine's part split by ROM (no split 'room')

    def by_rom(*a, **k):
        return orig(*a, **{x: y for x, y in k.items() if x != "split"})
    I._entry = by_rom
    try:
        out, tr = I.td_rocking(3, seconds=seconds)
        pp, mn = _pitch_pp(out, base)
        rec["before_rom_split"] = {"td_rocking_3": {"pitch_deg_pp": _r(pp), "pitch_min_deg": _r(mn)}}
        out, tr = I.stereotypy("rocking", 3, seconds=seconds)
        pp, _ = _pitch_pp(out, base)
        rec["before_rom_split"]["stereotypy_rocking_3"] = {
            "pitch_deg_pp": _r(pp), "trunk_share": _r(tr["params"]["measured"]["regions"]["trunk"]["share"], 3)}
    finally:
        I._entry = orig
    print("rocking before", rec["before_rom_split"], flush=True)
    return rec


def _carriage_row(I, out, tr, trunk):
    c = tr["params"]["construction"]
    rep = I.palm_rest_report(out, list(range(0, out.T, 20)))
    row = {"wrist_scaled_knots": {str(k): v["scaled_knots"] for k, v in (c["arms"].get("table") or {}).items()
                                  if k in ("20", "21", 20, 21)},
           "twist_to_elbow_deg_max": {s: _r(v.get("twist_to_elbow_deg_max", 0.0)) for s, v in c["arms"]["sides"].items()},
           "palm_patch_median_mm_max": {s: _r(rep[s]["patch_median_mm_max"]) for s in ("left", "right")},
           "palm_min_mm": {s: _r(rep[s]["palm_min_mm"]) for s in ("left", "right")},
           "arm_contact": {k: tr["validity"]["arm_contact"][k] for k in ("ok", "after_cm_max", "read_after")},
           "elbow_swivel_runs": sum(len(v["runs"]) for v in c["arms"]["elbow_swivel"].values()),
           "lift_runs": sum(len(v["runs"]) for v in (c.get("lift") or {}).values())}
    if trunk:
        pad = int(round(0.4 * out.rate))
        for name, every in (("read_share_all_runs_out", True), ("read_share_lifting_runs_out", False)):
            keep = np.ones(out.T, bool)
            runs = [r["knots"] for v in c["arms"]["elbow_swivel"].values() for r in v["runs"]
                    if every or r["raise_m"] > 0.0]
            runs += [r["knots"] for v in (c.get("lift") or {}).values() for r in v["runs"]]
            for k0, k1 in runs:
                keep[max(k0 - pad, 0):k1 + pad + 1] = False
            row[name] = _r(keep.mean(), 3)
    else:
        own = c.get("arms_own") or {}
        nom = {r: c["regions"][r]["nominal_share"] for r in own}
        row["arms_own"] = {r: {"share": _r(v["share"], 3), "nominal": _r(nom[r], 3),
                               "amplitude_share": _r(v["amplitude_share"], 3)} for r, v in own.items()}
        row["guard_after_cm"] = {s: _r(v["after_cm_max"]) for s, v in c["guard"]["sides"].items()}
    return row


REST_SCAN = ((0.45, 5.0, -0.005), (0.55, 15.0, -0.005), (0.6, 15.0, -0.005), (0.65, 20.0, -0.005),
             (0.75, 20.0, -0.005),                                                # along the thigh
             (0.65, 20.0, 0.0), (0.65, 20.0, 0.01), (0.65, 20.0, 0.015), (0.65, 20.0, 0.02), (0.65, 20.0, 0.025),
             (0.65, 15.0, 0.02), (0.65, 25.0, 0.02))                              # across it, and the pitch
CONFIGS = (("wave3", (0.45, 5.0, -0.005), False), ("wave3_twist_to_elbow", (0.45, 5.0, -0.005), True),
           ("module", None, True))
HAND_JOINTS = {"left": (20,) + tuple(range(22, 37)), "right": (21,) + tuple(range(37, 52))}
ELBOW_JOINT = {"left": 18, "right": 19}


def _set_rest(I, share, pitch, lateral):
    """involuntary's seated rest at THIGH_SHARE share, PALM_PITCH_DEG pitch and PALM_LATERAL_M lateral,
    HANDS_ON_THIGHS solved again (rounded to 0.01 deg as the module types it); the solve's record."""
    I.THIGH_SHARE, I.PALM_PITCH_DEG, I.PALM_LATERAL_M = share, pitch, lateral
    x, rec = I.solve_hands_on_thighs()
    I.HANDS_ON_THIGHS = {s: {k: round(v, 2) for k, v in x[s].items()} for s in x}
    return rec


def _rest_gaps(I, b):
    """mm: the smallest distance from the left hand's skin (the wrist's and the fingers' vertices) and from the left
    elbow's skin to the trunk's (the lumbar and lower thoracic vertices, joints 3 and 6) on the base."""
    from scipy.spatial import cKDTree
    from mdfm.synth import hands as H
    from mdfm.synth import viewer as V
    v = V.knot_vertices(b, [0])[0]
    dom = H._rest()[2]
    tree = cKDTree(v[np.flatnonzero(np.isin(dom, (3, 6)))])
    hl = v[np.flatnonzero(np.isin(dom, HAND_JOINTS["left"]))]
    hand = tree.query(hl)[0].min()
    elbow = tree.query(v[np.flatnonzero(dom == ELBOW_JOINT["left"])])[0].min()
    hands = cKDTree(v[np.flatnonzero(np.isin(dom, HAND_JOINTS["right"]))]).query(hl)[0].min()
    boxes = I._boxes(I.SCENE["sit"], b)
    Vk = V.knot_vertices(b, [0])
    guard = I._penetration(Vk, "left", boxes, I._target_ctx(Vk, heads=True))[0]
    J = b.joints().numpy()[0]
    fa = J[20] - J[18]
    return {"hand_to_trunk_mm": _r(1000 * hand, 1), "elbow_to_trunk_mm": _r(1000 * elbow, 1),
            "hands_skin_mm": _r(1000 * hands, 1), "guard_reading_mm": _r(1000 * guard, 1),
            "forearm_slope_deg": _r(np.degrees(np.arctan2(-fa[1], np.linalg.norm(fa[[0, 2]]))), 1)}


def rest():
    """The scan of the seated rest's place along the thigh and palm pitch (REST_SCAN): the solve's record and the
    rest's gaps to the trunk."""
    from mdfm.synth import involuntary as I
    keep = (I.THIGH_SHARE, I.PALM_PITCH_DEG, I.PALM_LATERAL_M, I.HANDS_ON_THIGHS)
    rows = {}
    try:
        for share, pitch, lateral in REST_SCAN:
            rec = _set_rest(I, share, pitch, lateral)
            b = I.base_body("sit", 0.01)
            rep = I.palm_rest_report(b, [0])
            rows[f"{share},{pitch},{lateral}"] = dict(_rest_gaps(I, b), patch_median_mm=_r(1000 * rec["left"]["patch_median_m"]),
                                            palm_min_mm=_r(1000 * rec["left"]["palm_min_m"]),
                                            target_error_mm=_r(rec["left"]["target_error_mm"], 3), palm_rest_ok=rep["ok"],
                                            rest_left={k: v for k, v in I.HANDS_ON_THIGHS["left"].items()})
            print("rest", share, pitch, lateral, rows[f"{share},{pitch},{lateral}"], flush=True)
    finally:
        I.THIGH_SHARE, I.PALM_PITCH_DEG, I.PALM_LATERAL_M, I.HANDS_ON_THIGHS = keep
    b = I.base_body("sit", 0.01)
    return {"module": {"thigh_share": I.THIGH_SHARE, "palm_pitch_deg": I.PALM_PITCH_DEG,
                       "palm_lateral_m": I.PALM_LATERAL_M, **_rest_gaps(I, b)},
            "scan": rows}


def _carriage_depths(I, cap):
    """cm: the deepest a left or right hand's and elbow's skin lies inside the trunk (joints 3, 6) under the carriage
    without the elbow's swing (_support_solve on the captured axial body), every 5 knots."""
    from mdfm.synth import hands as H
    from mdfm.synth import viewer as V
    body, base = cap["support"]
    sol, _ = I._support_solve(body, base)
    dom = H._rest()[2]
    _, self_, tgt, _, _ = I._guard_geometry()
    ks = np.arange(0, sol.T, 5)
    Vk = V.knot_vertices(sol, ks)
    ctx = I._target_ctx(Vk)
    trunk_t = np.isin(dom[tgt], (3, 6))
    out = {}
    for s in ("left", "right"):
        hand = np.isin(dom[self_[s]], HAND_JOINTS[s])
        elbow = dom[self_[s]] == ELBOW_JOINT[s]
        d = {"hand": 0.0, "elbow": 0.0}
        for i in range(len(ks)):
            tree, pts, nrm = ctx[i][:3]
            q = Vk[i][self_[s]]
            dist, j = tree.query(q, distance_upper_bound=0.06)
            ok = (dist < 0.06) & trunk_t[np.minimum(j, len(tgt) - 1)]
            inner = np.where(ok, -np.einsum("ij,ij->i", q - pts[np.minimum(j, len(pts) - 1)],
                                            nrm[np.minimum(j, len(pts) - 1)]), 0.0)
            d["hand"] = max(d["hand"], float(inner[hand].max()))
            d["elbow"] = max(d["elbow"], float(inner[elbow].max()))
        out[s] = {k: _r(100 * v) for k, v in d.items()}
    return out


def carriage():
    """The seated clips of the module docstring under CONFIGS: wave 3's rest with the wrist's twist kept at the wrist,
    the same rest with the twist handed to the elbow, and the module's rest."""
    from mdfm.synth import involuntary as I
    keep = (I.THIGH_SHARE, I.PALM_PITCH_DEG, I.PALM_LATERAL_M, I.HANDS_ON_THIGHS, I._twist_to_elbow,
            I.support_arms)
    module = (I.THIGH_SHARE, I.PALM_PITCH_DEG, I.PALM_LATERAL_M)

    def off(R18, R20, *a, **k):
        return R18, R20, np.zeros(R20.shape[0])
    cap = {}

    def capture(body, base, *a, **k):
        cap["support"] = (body, base)
        return keep[5](body, base, *a, **k)
    sp = {s.clip_id: s for s in I.specs()}
    rec = {}
    try:
        I.support_arms = capture
        for label, place, twist in CONFIGS:
            share, pitch, lateral = place or module
            _set_rest(I, share, pitch, lateral)
            I._twist_to_elbow = keep[4] if twist else off
            rows = {"thigh_share": share, "palm_pitch_deg": pitch, "palm_lateral_m": lateral, "twist_to_elbow": twist,
                    "rest_gaps": _rest_gaps(I, I.base_body("sit", 0.01))}
            for cid in ("invol_chorea_E1_trunk_g3", "invol_chorea_E1_trunk_g4"):
                out, tr = sp[cid].build()
                rows[cid] = dict(_carriage_row(I, out, tr, True), carriage_depth_cm=_carriage_depths(I, cap))
                print("carriage", label, cid, rows[cid], flush=True)
            out, tr = I.chorea("sit", 3, seconds=8.0)
            rows["chorea_sit_3_8s"] = dict(_carriage_row(I, out, tr, False), carriage_depth_cm=_carriage_depths(I, cap))
            print("carriage", label, "chorea_sit_3_8s", rows["chorea_sit_3_8s"], flush=True)
            rec[label] = rows
    finally:
        I.THIGH_SHARE, I.PALM_PITCH_DEG, I.PALM_LATERAL_M, I.HANDS_ON_THIGHS, I._twist_to_elbow, I.support_arms = keep
    return rec


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    torch.set_num_threads(4)
    from mdfm.synth import body as B
    rec = {"script": "scripts/openbody_wave3_followup.py", "run_at": datetime.datetime.now().isoformat(timespec="seconds"),
           "mdfm_commit": _head(), "body_model": B.model_name()}
    parts = sys.argv[1:] or ["functional", "rocking", "rest", "carriage"]
    if os.path.exists(OUT) and sys.argv[1:]:
        with open(OUT, encoding="utf-8") as f:
            rec = dict(json.load(f), **{k: rec[k] for k in ("run_at", "mdfm_commit", "body_model")})
    for p in parts:
        rec[p] = {"functional": functional, "rocking": rocking, "rest": rest, "carriage": carriage}[p]()
        rec.setdefault("ran_at", {})[p] = {"run_at": datetime.datetime.now().isoformat(timespec="seconds"),
                                           "mdfm_commit": _head()}
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        json.dump(rec, f, indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
