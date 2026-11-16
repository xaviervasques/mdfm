"""The open-body port's wave 5 readings that the design's section 16 cites (docs/OPEN_BODY_PORT_DESIGN.md): the
procedural walk of base.py re-solved on the open rest, and the freezing operator's holds on it, read on the selected
body model (the open body by default).

- walk_steps: base.walk's straight 20 s walk at steps of 0.55 to 0.30 m (the 0.05 m grid of STEP_M's rule): the
  components outside the anatomical table, the largest pitch the rocker rule gave a stance foot before the other
  foot's heel strike (0: the stance stays plantigrade up to it), the deepest dorsiflexion there;
- walk_place: the same walk at STEP_M for the ankle placed under its hip at 0.30, 0.25 and 0.20 of the cycle (PLACE):
  the stance knee's largest flexion, the pelvis's lowest and highest height, the hip's largest extension, the early
  stance pitch;
- walks: the straight 20 s walk, the 40 s walk of the pathology tests (two turns and a stop) and the 60 s walk of the
  grade-map tests: outside components, the pitch (largest, its largest step per knot, its largest value before the
  other foot's heel strike and where), the rule's passes, the largest knot-to-knot speed of the hips, knees and ankles,
  the foot joint's stance speed, the walking speed, the skinned soles (the lowest sole point's deepest dip below its
  flat stance height while the heel rises, the heels' lift over the lower sole at each double support's first knot),
  the leg-to-leg check (legcontact.validity);
- before: wave 1's reading of the former walk (reports/openbody/joint_limits.json base_bodies, walk_8s), and the
  former walk (base.py of FORMER_COMMIT, read from git into memory) read again: its components outside, its stance
  knee, its ankles' dorsiflexion in stance and in swing, its speed, and the leg-to-leg check of its 40 s test walk;
- freezing: the episode of the pathology tests (onset requested at 8.0 s, 3.0 s, trembling) on the 40 s walk and a
  grade-4 plan of support 'mesh' on it (seed 21, a doorway at 25 s): each onset's support, its heel lifts and sole
  gap on the skinned mesh, the hold's largest heel lift; the grade-map tests' plans on the 60 s walk (grades 1 to 4,
  seed 21, a doorway at 25 s): each episode's trembling asked and used (freezing's scaling into the table) and the
  new limit violations of the output.
- followup (the wave 5 follow-up): base.walk's default, the gait model's walk (FOLLOWUP_WALK), read back
  (gait.measure_gait's summary, the generated stance and double-support shares, the joint peaks, gait.validity with
  its continuity readings); the catalogue's level 0 and the three clips the continuity check refused (their
  continuity readings, CONTINUITY_CLIPS); the healthy walk of the gait tests: the lowest sole point over the first
  knots after each toe-off, beside gait.FLOOR_TOL_M; the double-support guard's positive control (short steps on a
  narrow base without the guards) at DS_CONTROL_WIDTHS_M, its double support's smallest hard and soft gaps.
- second (the wave 5 second follow-up): the three clips the continuity check refused, built again; the stance
  read-back on the healthy walk at FLOOR_TOL_M and STANCE_READ_TOL_M, the toe's rest clearance on both bodies (the
  reference read in a child process); the gait tests' turning course with the push-off pivot's former touch and the
  current one, its stance slip read at three touch tolerances.
Writes reports/openbody/wave5_checks.json; --only freezing (or other sections) reruns those sections and updates them
in the existing report, its other sections kept (run_at_rerun, mdfm_commit_rerun). CPU only; reads no dataset.
"""
import datetime
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "reports", "openbody", "wave5_checks.json")
STEPS = (0.55, 0.5, 0.45, 0.4, 0.35, 0.3)
PLACES = (0.3, 0.25, 0.2)
TEST_WALK = {"duration_s": 40.0, "turns": [(14.0, 180.0, 4.0), (26.0, -90.0, 2.5)], "stop_at_s": 36.0}
GRADE_WALK = {"duration_s": 60.0, "turns": [(15.0, 180.0, 4.0), (35.0, 180.0, 4.0)], "stop_at_s": 55.0}
FORMER_COMMIT = "c633f2f"     # the last commit before wave 5 (base.py's flat-footed walk)
_GIT = ["git", "-c", "safe.directory=" + ROOT.replace(chr(92), "/"), "-C", ROOT]   # a worktree checkout too


def _head():
    try:
        return subprocess.run(_GIT + ["rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def _r(x, n=2):
    return round(float(x), n)


def _outside(b):
    from mdfm.synth import body as BD
    lim = BD.joint_limits()
    c = BD.components_deg(b.rot6d)
    ex = np.maximum(lim.lo[None] - c, c - lim.hi[None])[:, 1:].max(0)
    return {f"{BD.BODY_JOINTS[j + 1]}.{'xyz'[k]}": _r(ex[j, k], 3) for j, k in zip(*np.where(ex > 1e-6))}


def _u(info):
    return np.stack([(info["phase"] - s) % 1.0 for s in (0.0, 0.5)], -1)


def _early(info):
    from mdfm.synth import base as B
    return info["stance"] & (_u(info) < B.HEEL_OFF)


def walk_steps():
    from mdfm.synth import base as B
    from mdfm.synth import body as BD
    out = {"module_step_m": B.STEP_M, "place": B.PLACE, "rows": []}
    for s in STEPS:
        b, info = B.walk(20.0, step_m=s, kind="plantigrade")
        c = BD.components_deg(b.rot6d)
        e = _early(info)
        e[:300] = False                                        # the walk under way (gait initiation done)
        out["rows"].append({"step_m": s, "outside": _outside(b),
                            "early_stance_pitch_max_deg": _r(info["pitch_deg"][e].max(), 3),
                            "early_stance_dorsiflexion_deg": _r(-min(c[e[:, 0], 7, 0].min(), c[e[:, 1], 8, 0].min())),
                            "plantigrade_inside": bool(not _outside(b) and info["pitch_deg"][e].max() < 1e-9)})
    ok = [r["step_m"] for r in out["rows"] if r["plantigrade_inside"]]
    out["longest_plantigrade_inside_m"] = max(ok) if ok else None
    return out


def walk_place():
    from mdfm.synth import base as B
    from mdfm.synth import body as BD
    keep = B.PLACE
    out = {"module_place": keep, "step_m": B.STEP_M, "rows": []}
    try:
        for p in PLACES:
            B.PLACE = p
            b, info = B.walk(20.0, kind="plantigrade")
            c = BD.components_deg(b.rot6d)
            st = info["stance"].copy()
            st[:300] = False
            J = b.joints().numpy()[300:]
            e = _early(info)
            e[:300] = False
            out["rows"].append({"place": p, "outside": _outside(b),
                                "stance_knee_max_deg": _r(max(c[st[:, 0], 4, 0].max(), c[st[:, 1], 5, 0].max()), 1),
                                "hip_extension_max_deg": _r(c[300:, [1, 2], 0].max()),
                                "pelvis_low_high_m": [_r(J[:, 0, 1].min(), 4), _r(J[:, 0, 1].max(), 4)],
                                "early_stance_pitch_max_deg": _r(info["pitch_deg"][e].max(), 3)})
    finally:
        B.PLACE = keep
    return out


def _walk_read(b, info):
    from mdfm.synth import base as B
    from mdfm.synth import body as BD
    from mdfm.synth import contact as CT
    from mdfm.synth import freezing as F
    from mdfm.synth import legcontact as LC
    c = BD.components_deg(b.rot6d)
    p = info["pitch_deg"]
    st = info["stance"]
    J = b.joints()
    fj = torch.gradient(J[:, [10, 11]], spacing=1.0 / b.rate, dim=0)[0].numpy()
    inner = st & np.roll(st, 1, 0) & np.roll(st, -1, 0)
    e = _early(info)
    dp = np.abs(np.diff(p, axis=0))
    rate = {nm: _r((np.abs(np.diff(c[:, j, 0])) * b.rate).max(), 0)
            for nm, j in (("left_hip", 1), ("right_hip", 2), ("left_knee", 4), ("right_knee", 5), ("left_ankle", 7),
                          ("right_ankle", 8))}
    low = CT.mesh_sole_low(b)
    flat = float(np.median(low[e]))                            # a flat stance foot's lowest skinned point
    rising = st & ~e & (p > 1e-6)
    ds = st[:, 0] & st[:, 1]
    first = np.flatnonzero(ds & ~np.roll(ds, 1))
    first = first[first > 300]
    heel, gap = F.heel_lift_mesh(b, first)
    v, _ = B.gait_kinematics(b)
    lg = LC.validity(b)
    ke = np.flatnonzero((p * e).max(1) > 1e-6)
    return {"outside": _outside(b), "rocker_passes": len(info["rocker_passes"]),
            "rocker_last_excess_deg": _r(info["rocker_passes"][-1], 3),
            "pitch_max_deg": _r(p.max(), 1), "pitch_step_max_deg": _r(dp.max()),
            "early_stance_pitch_max_deg": _r((p * e).max(), 2),
            "early_stance_pitch_t_s": [_r(ke.min() / b.rate), _r(ke.max() / b.rate)] if len(ke) else None,
            "joint_rate_max_deg_s": rate,
            "foot_joint_stance_speed_max_m_s": float(max(np.hypot(*fj[inner[:, i], i][:, [0, 2]].T).max()
                                                       for i in range(2))),
            "speed_median_m_s": _r(np.median(v[300:1300]), 3),
            "sole_flat_mm": _r(1000 * flat), "sole_dip_in_heel_rise_mm": _r(1000 * (flat - low[rising].min())),
            "double_support_first_knots": int(len(first)),
            "heel_lift_at_double_support_start_mm": {"max": _r(1000 * heel.max()),
                                                     "median": _r(1000 * np.median(heel))},
            "sole_gap_at_double_support_start_mm": _r(1000 * gap.max()),
            "legs": {"ok": bool(lg["ok"]), "hard_min_mm": _r(1000 * lg["hard_min_m"]),
                     "soft_min_mm": _r(1000 * lg["soft_min_m"]), "hard_where": lg.get("hard_where")}}


def walks():
    from mdfm.synth import base as B
    out = {}
    for name, kw in (("straight_20s", {"duration_s": 20.0}), ("test_40s", TEST_WALK), ("grade_map_60s", GRADE_WALK)):
        kw = dict(kw)
        t0 = time.time()
        b, info = B.walk(kw.pop("duration_s"), kind="plantigrade", **kw)
        out[name] = _walk_read(b, info)
        out[name]["build_s"] = _r(time.time() - t0, 1)
    return out


def _former_base():
    """The former base.py (FORMER_COMMIT, before wave 5) as a module, read from git into memory."""
    import importlib.util
    import tempfile
    import mdfm.synth  # noqa: F401
    src = subprocess.run(_GIT + ["show", f"{FORMER_COMMIT}:mdfm/synth/base.py"], capture_output=True, text=True,
                         check=True, encoding="utf-8").stdout
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "base_former.py")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(src)
        spec = importlib.util.spec_from_file_location("mdfm.synth.base_former", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["mdfm.synth.base_former"] = mod
        spec.loader.exec_module(mod)
    return mod


def before():
    """Wave 1's reading of the former walk, and the former walk read again here on the open body: the components
    outside, the stance knee, the ankle's dorsiflexion in stance and in swing, the speed, the leg-to-leg check on the
    40 s walk of the pathology tests."""
    from mdfm.synth import base as B
    from mdfm.synth import body as BD
    from mdfm.synth import legcontact as LC
    with open(os.path.join(ROOT, "reports", "openbody", "joint_limits.json"), encoding="utf-8") as f:
        d = json.load(f)
    out = {"source": "reports/openbody/joint_limits.json base_bodies walk_8s (wave 1, the former walk)",
           "walk_8s": d["base_bodies"]["walk_8s"], "former_commit": FORMER_COMMIT}
    old = _former_base()
    b, info = old.walk(20.0)
    c = BD.components_deg(b.rot6d)
    st = info["stance"].copy()
    st[:300] = False
    sw = ~info["stance"]
    sw[:300] = False
    v, _ = B.gait_kinematics(b)
    J = b.joints().numpy()[300:]
    out["former_straight_20s"] = {
        "step_m": old.STEP_M, "outside": _outside(b),
        "stance_knee_max_deg": _r(max(c[st[:, 0], 4, 0].max(), c[st[:, 1], 5, 0].max()), 1),
        "stance_dorsiflexion_max_deg": _r(-min(c[st[:, 0], 7, 0].min(), c[st[:, 1], 8, 0].min())),
        "swing_dorsiflexion_max_deg": _r(-min(c[sw[:, 0], 7, 0].min(), c[sw[:, 1], 8, 0].min())),
        "hip_extension_max_deg": _r(c[300:, [1, 2], 0].max()), "speed_median_m_s": _r(np.median(v[300:1300]), 3),
        "pelvis_low_high_m": [_r(J[:, 0, 1].min(), 4), _r(J[:, 0, 1].max(), 4)]}
    kw = dict(TEST_WALK)
    b, _ = old.walk(kw.pop("duration_s"), **kw)
    lg = LC.validity(b)
    out["former_test_40s_legs"] = {"ok": bool(lg["ok"]), "hard_min_mm": _r(1000 * lg["hard_min_m"]),
                                   "soft_min_mm": _r(1000 * lg["soft_min_m"]), "hard_where": lg.get("hard_where"),
                                   "hard_t_s": lg.get("hard_t_s")}
    return out


def freezing():
    from mdfm.synth import base as B
    from mdfm.synth import freezing as F
    kw = dict(TEST_WALK)
    b, _ = B.walk(kw.pop("duration_s"), kind="plantigrade", **kw)
    on = F.snap_onset(b, 8.0)
    h, g = F.heel_lift_mesh(b, [int(round(on * b.rate))])
    out = {"episode_8s": {"onset_s": on, "heel_lift_mm": [_r(1000 * x) for x in h[0]], "sole_gap_mm": _r(1000 * g[0])}}
    eps = F.plan_episodes(b, 4, seed=21, doorway_s=(25.0,), support="mesh", pattern="alternating")
    frz, tr = F.freezing(b, eps, arms="settle")
    rows = []
    for e in tr["params"]["episodes"]:
        hk = np.arange(int(round(e["hold_s"][0] * b.rate)), int(round(e["hold_s"][1] * b.rate)))
        hh, gg = F.heel_lift_mesh(frz, hk)
        rows.append({"context": e["context"], "onset_base_s": e["onset_base_s"], "support": e["support"],
                     "heel_lift_mesh_mm": [_r(1000 * x) for x in e["heel_lift_mesh_m"]],
                     "foot_gap_mesh_mm": _r(1000 * e["foot_gap_mesh_m"]), "hold_heel_lift_max_mm": _r(1000 * hh.max()),
                     "hold_sole_gap_max_mm": _r(1000 * gg.max())})
    out["grade4_mesh"] = rows
    kw = dict(GRADE_WALK)
    b60, _ = B.walk(kw.pop("duration_s"), kind="plantigrade", **kw)
    grades = {}
    for g in (1, 2, 3, 4):                     # the grade-map tests' plans (seed 21, a doorway at 25 s), in phase
        _, trg = F.freezing_grade(b60, g, seed=21, doorway_s=(25.0,))
        grades[str(g)] = {"limits_new": trg["validity"]["limits_new"], "episodes": [
                     {"context": e["context"], "onset_base_s": e["onset_base_s"], "manifestation": e["manifestation"],
                      "tremble_mm": None if e["tremble_m"] is None else _r(1000 * e["tremble_m"]),
                      "tremble_used_mm": None if e["tremble_m_used"] is None else _r(1000 * e["tremble_m_used"])}
                     for e in trg["params"]["episodes"]]}
    out["grade_map_trembling"] = grades
    out["heel_tol_mm"] = 1000 * F.MESH_HEEL_TOL_M
    out["support_tol_mm"] = 1000 * F.MESH_SUPPORT_TOL_M
    return out


FOLLOWUP_WALK = {"duration_s": 20.0, "turns": [(10.0, 180.0, 4.0)], "stop_at_s": 17.0}
DS_CONTROL_WIDTHS_M = (0.04, 0.02)
CONTINUITY_CLIPS = ("gait_E8_level0", "gait_E8_functional_walking_on_ice", "gait_E8_hemiparetic_left",
                    "gait_E8_parkinsonian_festination")


def _gait_read(b, info):
    """A gait model body's read-back (gait.measure_gait's summary), its generated stance and double-support shares
    over the steady knots, its joint peaks, and its checks (gait.validity)."""
    from mdfm.synth import body as BD
    from mdfm.synth import gait as GA
    s = GA.summary(GA.measure_gait(b))
    v = GA.validity(b, info)
    c = BD.components_deg(b.rot6d)
    st = np.asarray(info["stance"], bool)
    amp = np.asarray(info["amplitude"]) > 0.99
    keys = ("speed_m_s", "cadence_spm", "step_length_m", "step_width_m", "stance_share_left", "stance_share_right",
            "double_support_share", "knee_loading_left_deg", "knee_loading_right_deg", "knee_swing_left_deg",
            "knee_swing_right_deg", "contact_pitch_left_deg", "clearance_left_m", "pelvis_vertical_range_m")
    cont = v["continuity"]
    return {"summary": {k: (_r(s[k], 3) if s.get(k) is not None else None) for k in keys},
            "generated_stance_share": [_r(st[amp, i].mean(), 3) for i in range(2)],
            "generated_double_support_share": _r((st[amp].all(1)).mean(), 3),
            "peaks_deg": {"hip_extension": _r(c[:, [1, 2], 0].max(), 1), "hip_flexion": _r(-c[:, [1, 2], 0].min(), 1),
                          "knee_flexion": _r(c[:, [4, 5], 0].max(), 1),
                          "ankle_dorsiflexion": _r(-c[:, [7, 8], 0].min(), 1),
                          "ankle_plantarflexion": _r(c[:, [7, 8], 0].max(), 1)},
            "outside": _outside(b), "gait_ok": bool(v["gait"]["ok"]),
            "stance_slip_p95_mm_s": _r(1000 * v["stance_slip_p95_m_s"], 3), "legs_ok": bool(v["legs"]["ok"]),
            "continuity": {k: (_r(cont[k], 4) if isinstance(cont[k], float) else cont[k])
                           for k in ("ok", "knee_rate_worst_deg_s", "knee_rate_left_t_s", "knee_rate_right_t_s",
                                     "floor_skim_worst_m", "contact_jump_worst_m", "peak_acc_m_s2", "peak_acc_joint")
                           if k in cont},
            "stance_lift": info.get("stance_lift")}


def followup():
    """Wave 5 follow-up: base.walk's default (the gait model) read back beside the plantigrade stand-in, the
    catalogue's level 0 and the three clips the continuity check refused, and the sole's lowest point over the first
    knots after a toe-off of the healthy walk (the stance read-back's floor tolerance)."""
    import torch as _t
    from mdfm.synth import base as B
    from mdfm.synth import catalogue as CAT
    from mdfm.synth import gait as GA
    out = {}
    kw = dict(FOLLOWUP_WALK)
    b, info = B.walk(kw.pop("duration_s"), **kw)
    out["default_walk"] = dict(_gait_read(b, info), kind=info["kind"], walk=FOLLOWUP_WALK)
    specs = {s.clip_id: s for s in CAT.catalogue_set(background=False)}
    out["clips"] = {}
    for cid in CONTINUITY_CLIPS:
        cb, tr = specs[cid].build()
        v = tr["validity"]
        cont = v["continuity"]
        out["clips"][cid] = {"gait_ok": bool(v["gait"]["ok"]), "continuity": {
            k: (_r(cont[k], 4) if isinstance(cont[k], float) else cont[k])
            for k in ("ok", "knee_rate_worst_deg_s", "knee_rate_left_t_s", "knee_rate_right_t_s", "floor_skim_worst_m",
                      "contact_jump_worst_m", "peak_acc_m_s2", "peak_acc_joint") if k in cont},
            "measured": {k: tr["params"]["measured"].get(k) for k in ("speed_m_s", "cadence_spm", "step_length_m",
                                                                      "stance_share_left", "knee_loading_left_deg",
                                                                      "knee_swing_left_deg")}
            if isinstance(tr["params"].get("measured"), dict) else None}
        print(cid, "done", flush=True)
    b, info = GA.generate(GA.NORMAL, duration_s=11.0, start_at_s=1.0, stop_at_s=8.5, seed=7)   # the tests' walk
    J, G = (x.numpy() for x in b.fk())
    fk = (_t.from_numpy(J), _t.from_numpy(G))
    lows = np.stack([GA.sole_cloud_points(b, s, fk)[..., 1].min(-1) for s in GA.SIDES], -1)
    floor = float(np.median(lows.min(1)))
    after = []
    for a, e in GA._runs(np.asarray(info["stance"], bool)[:, 0]):
        if a > 150 and e < b.T - 150:
            after.append([_r(1000 * (lows[e + j, 0] - floor), 2) for j in range(8)])
    out["healthy_after_toe_off_mm"] = {"floor_tol_mm": 1000 * GA.FLOOR_TOL_M, "knots_from_toe_off": after,
                                       "healthy": _gait_read(b, info)}
    from dataclasses import replace as _rep
    from mdfm.synth import legcontact as LC
    ds = {}
    for w in DS_CONTROL_WIDTHS_M:          # the double-support guard's positive control: short steps, a narrow base
        p = _rep(GA.NORMAL, step_m=0.30, step_width_m=w, step_width_sd_m=0.0, leg_gap_m=None)
        b0, i0 = GA.generate(p, duration_s=7.0, start_at_s=1.0, stop_at_s=4.5, seed=3)
        g0 = LC.leg_gap(b0)
        d0 = np.asarray(i0["stance"], bool).all(1)
        ds[f"{w:.2f}"] = {"hard_min_mm": _r(1000 * g0["hard_m"][d0].min()),
                          "soft_min_mm": _r(1000 * g0["soft_m"][d0].min())}
    out["ds_control_unguarded"] = {"tol_mm": 1000 * LC.TOL_M, "by_width_m": ds}
    return out


TOE_SNIPPET = (
    "import os, sys, json\n"
    "sys.path.insert(0, sys.argv[1])\n"
    "from mdfm.synth import openbody as OB\n"
    "if sys.argv[2] == 'reference':\n"
    "    os.environ[OB.BACKEND_ENV] = OB.SMPLH_REFERENCE\n"
    "    os.environ[OB.REFERENCE_ENV] = '1'\n"
    "from mdfm.synth import contact as CT\n"
    "from mdfm.synth.viewer import rest_vertices\n"
    "v = rest_vertices()\n"
    "ids, ht = CT.sole_vertices(2), CT.heel_toe(2)\n"
    "out = {}\n"
    "for side in ('left', 'right'):\n"
    "    cl = v[ids[side]]\n"
    "    out[side] = round(1000 * float(cl[ht[side][1], 1].min() - cl[:, 1].min()), 2)\n"
    "print(json.dumps(out))\n")


def _toe_rest_clearance(model):
    """{side: mm} the toe set's lowest rest point over the sole cloud's lowest (contact.heel_toe), on the open body or
    the SMPL+H reference (read in a child process, so that the reference's flag never reaches this one)."""
    r = subprocess.run([sys.executable, "-I", "-c", TOE_SNIPPET, ROOT, model], capture_output=True, text=True,
                       check=True, encoding="utf-8")
    return json.loads(r.stdout.strip().splitlines()[-1])


def _slip_read(GA, b, info, touch):
    """stance_slip's 95th percentile (mm/s) read with the touch tolerance touch (m)."""
    keep = GA.SLIP_TOUCH_M
    try:
        GA.SLIP_TOUCH_M = touch
        return _r(1000 * GA.stance_slip(b, info)["p95_m_s"], 3)
    finally:
        GA.SLIP_TOUCH_M = keep


def second():
    """Wave 5 second follow-up: the three clips the continuity check refused, built again (their knee peaks and
    read-backs; before: the followup section); the stance read-back's tolerance on the healthy walk and the toe's rest
    clearance on both bodies; the turning stance's slip on the gait tests' course with the push-off pivot's former
    touch (PIVOT_TOE_NEAR_M 2 mm, set in memory) and the current one, read at three touch tolerances."""
    from mdfm.synth import catalogue as CAT
    from mdfm.synth import gait as GA
    out = {"clips": {}}
    specs = {s.clip_id: s for s in CAT.catalogue_set(background=False)}
    for cid in CONTINUITY_CLIPS[1:]:
        cb, tr = specs[cid].build()
        v = tr["validity"]
        cont = v["continuity"]
        m = tr["params"].get("measured") or {}
        out["clips"][cid] = {"gait_ok": bool(v["gait"]["ok"]), "limits_outside": v["limits_outside"],
                             "knee_rate_worst_deg_s": _r(cont["knee_rate_worst_deg_s"], 1),
                             "continuity_ok": bool(cont["ok"]),
                             "stance_slip_p95_mm_s": _r(1000 * v["stance_slip_p95_m_s"], 3),
                             "measured": {k: (_r(m[k], 3) if isinstance(m.get(k), float) else m.get(k)) for k in
                                          ("speed_m_s", "cadence_spm", "step_length_m", "step_width_m",
                                           "stance_share_left", "stance_share_right", "knee_swing_left_deg",
                                           "knee_swing_right_deg", "knee_loading_left_deg", "clearance_left_m")}}
        print(cid, "done", flush=True)
    return out


def second_read():
    """Wave 5 second follow-up: the stance read-back on the healthy walk of the gait tests at FLOOR_TOL_M and at
    STANCE_READ_TOL_M, and the toe's rest clearance on both bodies."""
    from mdfm.synth import gait as GA
    out = {}
    b, info = GA.generate(GA.NORMAL, duration_s=11.0, start_at_s=1.0, stop_at_s=8.5, seed=7)
    out["stance_read"] = {f"{1000 * tol:g}_mm": {k: _r(s[k], 3) for k in ("stance_share_left", "stance_share_right",
                                                                         "double_support_share", "step_length_m")}
                          for tol in (GA.FLOOR_TOL_M, GA.STANCE_READ_TOL_M)
                          for s in [GA.summary(GA.measure_gait(b, touch_tol_m=tol))]}
    out["toe_rest_clearance_mm"] = {"open": _toe_rest_clearance("open"),
                                    "smplh_reference": _toe_rest_clearance("reference")}
    return out


def second_slip():
    """Wave 5 second follow-up: the gait tests' turning course with the push-off pivot's former touch
    (PIVOT_TOE_NEAR_M 2 mm, set in memory) and the current one, its stance slip read at three touch tolerances."""
    from mdfm.synth import gait as GA
    out = {}
    slip = {}
    keep = GA.PIVOT_TOE_NEAR_M
    for name, val in (("pivot_toe_2mm", 0.002), ("pivot_toe_now", keep)):
        try:
            GA.PIVOT_TOE_NEAR_M = val
            cb, ci = GA.course(GA.NORMAL, distance_m=3.0, seed=GA.SEED)
        finally:
            GA.PIVOT_TOE_NEAR_M = keep
        slip[name] = {f"touch_{1000 * t:g}_mm": _slip_read(GA, cb, ci, t) for t in (0.001, 0.0015, 0.002)}
    out["slip_turning_course_p95_mm_s"] = slip
    out["slip_tol_mm_s"] = 1000 * GA.SLIP_TOL_M_S
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", nargs="*", default=None,
                    help="run these sections only and update them in the existing report (its other sections kept)")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    torch.set_num_threads(4)
    from mdfm.synth import body as B
    rec = {"script": "scripts/openbody_wave5_checks.py",
           "run_at": datetime.datetime.now().isoformat(timespec="seconds"), "mdfm_commit": _head(),
           "body_model": B.model_name()}
    if a.only:
        with open(OUT, encoding="utf-8") as f:
            old = json.load(f)
        rec = dict(old, **{f"{k}_rerun": rec[k] for k in ("run_at", "mdfm_commit")}, rerun_sections=a.only)
    for name, fn in (("walk_steps", walk_steps), ("walk_place", walk_place), ("walks", walks), ("before", before),
                     ("freezing", freezing), ("followup", followup), ("second", second),
                     ("second_read", second_read), ("second_slip", second_slip)):
        if a.only and name not in a.only:
            continue
        rec[name] = fn()
        print(name, "done", flush=True)
    with open(OUT, "w", encoding="utf-8", newline=chr(10)) as f:
        json.dump(rec, f, indent=1, default=str)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
