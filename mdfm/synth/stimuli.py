"""The probe stimulus set and the read-back of each family, for scripts/sp3_decoder_prior_probe.py and the
tests; levels() prints every grade map of grades.py in physical units with the exam length.

probe_set: one 10 s window (WINDOW_KNOTS knots, the model's window, data/body_store.py WIN_S) per stimulus,
in the shared neutral shape, each with its level-0 body:
- tremor: rest tremor of the left forearm on the seated body (E1), MDS-UPDRS 3.17 grades 0 to 4 (grade 1 at
  its above-floor level), 5 Hz, present throughout; a frequency sweep at grade 3 at 4, 6, 8 and 10 Hz in the
  arms-forward posture (E2, postural).
- freezing: a procedural straight walk, base.walk's plantigrade footstep walk (WALK_KIND; the freezing stand-in of
  design D75, the walk of the exam's freezing clips since exam_v10), one episode of 3.0 s (the published median,
  grades.DURATION_MEDIAN_S) about 2.5 s into the window, frozen at a labelled double-support knot (support
  'double'), as akinesia, in-phase trembling and alternating trembling at 5.5 Hz, and the walk without episode; the
  window shows walking before and after the episode. Until the open-body port's wave 7 base.walk's default; since
  its wave 5 follow-up that default is the gait generator's healthy walk (D74), which holds no labelled double-support
  knot within freezing.SEARCH_S of the episode's 2.5 s (its contact labels read at its speed), so the stimulus names
  the plantigrade walk.
- bradykinesia: 10 cycles, pronation-supination (3.6, arms forward) with decrement grades 1 to 3,
  interruptions grade 2 and the arrest level; toe tapping (3.7) and leg agility (3.8, seated) with decrement
  grade 3; level 0 of each task (grade 0 returns the base body, so the task's level 0 is the 10 cycles without
  any criterion).
- dystonic posture on the standing body, held throughout: neck rotation to the left at TWSTRS grades 1 to 4,
  laterocollis to the right and anterocollis at grade 2, shoulder elevation at grade 2, trunk lateral flexion
  to the left at DIS grade 3, elbow flexion at DIS grade 3; the standing body is level 0. The postures are built
  with the operator of version sp5-v6 (V6_DYSTONIA: the healthy joint-limit table and, for the shoulder elevation
  and the trunk, the arms riding with the collar or the trunk, V6_ARM); the SH2 operator of the catalogue
  (limits_ext.py) is not used here.
Open-body port, wave 7 (design D54 and section 18): the stimuli are built on the open body (Anny's catalogue body, the
anatomical joint-limit table); reports/synth/stimuli/stimuli.csv holds their body_sha256 since then. The SP3 runs of
reports/synth/sp3/ were measured on the SMPL+H bodies of the earlier record (git history of stimuli.csv), which no body
of the port rebuilds; a model trained on the open body is probed on these.

read_back(stim, body, ref): the family's measures on any body on the stimulus's clock (the truth, or a body
decoded by the model), timing taken from the truth:
- tremor: dominant frequency (3 to 12 Hz) and peak-to-peak band amplitude (cm, DR-11.5 (b) form) of the left
  wrist relative to the left shoulder in the pelvis heading frame;
- freezing: mean horizontal pelvis speed over the hold and over the 2 s of walking before the episode's ramp,
  their ratio, the horizontal pelvis travel over the hold, each knee's vertical peak to peak over the hold,
  the dominant frequency (0.5 to 12 Hz) and freeze ratio (DR-10.1 (b)) of the vertical acceleration of the
  knee that moves more;
- bradykinesia: per truth cycle, the largest excursion of the task angle (bradykinesia.task_angle, relative
  to the task's start knot) from the straight line between the cycle's end knots; the least-squares slope of
  the post-onset amplitudes; DEC of DR-12.3;
- dystonic posture: the median over the window of the deviation angle of the region's end segment, in the
  pelvis heading frame, from the same segment of a reference body (the level-0 truth for a truth body; the
  level-0 decode for a decoded one).
"""
import math
from dataclasses import dataclass, field

import numpy as np
import torch

from . import grades as G
from .base import POSTURES, static_body, walk
from .body import (ELBOW_FLEXION_SIGN, Body, crop, joint_limits, limit_violations, rest_joints, rotation_angle_deg,
                   window_frame)
from .bradykinesia import TASKS, amplitudes, bradykinesia, task_angle, task_rooms
from .dystonia import REGIONS, dystonic_posture, level_deg, rom, _resolve
from .freezing import Episode, freezing, snap_onset
from .measure import (band_amplitude_pp, decrement_dec, dominant_frequency, freeze_ratio, slope,
                      vertical_acceleration)
from .tremor import ELBOW, WRIST, centre_shift, driver_rooms, hinge_radius, tremor

WINDOW_KNOTS = 1000
FAMILIES = ("tremor", "freezing", "bradykinesia", "dystonia")
SHOULDER = {"left": 16, "right": 17}
EXAM = (("E1", "seated rest", 45.0, "draft"), ("E2", "arms outstretched", 30.0, "draft"),
        ("E3", "reach to nose, 3 per side", None, "reach"), ("E4", "pronation-supination, 10 per side", None,
                                                               "pronation_supination"),
        ("E5", "toe tapping, 10 per side", None, "toe_tapping"),
        ("E6", "leg agility, 10 per side", None, "leg_agility"),
        ("E7", "arising, then 15 s standing", 15.0, "arise"), ("E8", "gait and turn", 60.0, "draft"),
        ("E9", "turning in place", 60.0, "draft"))
REACH_PERIOD_S = 3.0        # [module] one slow reach to the nose and back
# The probe stimuli keep the version-2 leg tasks (no foot planting, operator version sp1-v2), the stimuli of the SP3
# runs (reports/synth/sp3/); since the open-body port's wave 7 on the open body (module docstring).
PROBE_FOOT = {"toe_tapping": "free", "leg_agility": "free"}
ARISE_S = 3.0               # [module] sit to stand


@dataclass
class Stimulus:
    """name, family, variant, grade; body (window frame), base (its level-0 body, window frame); truth (the
    operator's record); info: what read_back needs (side, joint, timing)."""
    name: str
    family: str
    variant: str
    grade: object
    body: Body
    base: Body
    truth: dict
    info: dict = field(default_factory=dict)


def _framed(b):
    return window_frame(b)


def _tremor_set(n):
    out = []
    sit = static_body("sit", (n - 1) / 100.0)
    for g in range(5):
        b, tr = tremor(sit, "left", grade=g, frequency_hz=G.TREMOR_DEFAULT_HZ, condition="rest", seed=100 + g)
        out.append(Stimulus(f"tremor:rest:g{g}", "tremor", "rest", g, _framed(b), _framed(sit), tr,
                            {"side": "left"}))
    arms = static_body("sit_arms_forward", (n - 1) / 100.0)
    for f in (4.0, 6.0, 8.0, 10.0):
        b, tr = tremor(arms, "left", grade=3, frequency_hz=f, condition="postural", tremor_type="essential",
                       seed=200 + int(f))
        out.append(Stimulus(f"tremor:postural_{int(f)}hz:g3", "tremor", f"postural_{int(f)}hz", 3, _framed(b),
                            _framed(arms), tr, {"side": "left"}))
    return out


WALK_KIND = "plantigrade"    # the stimulus walk's kind (module docstring; open-body port, wave 7)


def walk_window(n, lead_s=3.0, seed=0):
    """A straight procedural walk (base.walk of kind WALK_KIND) cropped to start lead_s after gait initiation, long
    enough for an episode of up to 4 s inside a window of n knots: (body, crop start)."""
    total = (n - 1) / 100.0 + lead_s + 6.0
    w, _ = walk(total, start_at_s=0.5, kind=WALK_KIND)
    a = int(round(lead_s * 100))
    return crop(w, a, w.T), a


def _freezing_set(n):
    base, _ = walk_window(n)
    level0 = crop(base, 0, n)
    out = [Stimulus("freezing:none:g0", "freezing", "none", 0, _framed(level0), _framed(level0),
                    freezing(base, [])[1], {"episode": None})]
    on = snap_onset(base, 2.5)
    for var, man, pat in (("akinesia", "akinesia", "in_phase"), ("trembling_in_phase", "trembling", "in_phase"),
                          ("trembling_alternating", "trembling", "alternating")):
        b, tr = freezing(base, [Episode(on, G.DURATION_MEDIAN_S, man, "straight", pattern=pat)], seed=0)
        ep = tr["params"]["episodes"][0]
        out.append(Stimulus(f"freezing:{var}:ep", "freezing", var, "episode", _framed(crop(b, 0, n)),
                            _framed(level0), tr, {"episode": ep}))
    return out


def _brady_set(n):
    out = []
    bases = {name: static_body(name, (n - 1) / 100.0) for name in ("sit", "sit_arms_forward")}
    plan = [("pronation_supination", "decrement", g) for g in (1, 2, 3)] + \
        [("pronation_supination", "interruptions", 2), ("pronation_supination", "arrest", 3),
         ("toe_tapping", "decrement", 3), ("leg_agility", "decrement", 3)]
    for task in TASKS:
        base = bases[TASKS[task]["posture"]]
        b0, tr0 = bradykinesia(base, task, "left", decrement_onset=None, seed=1, foot=PROBE_FOOT.get(task))
        out.append(Stimulus(f"bradykinesia:{task}:level0", "bradykinesia", f"{task}:level0", 0, _framed(b0),
                            _framed(base), tr0, {"task": task, "side": "left"}))
    for task, crit, g in plan:
        base = bases[TASKS[task]["posture"]]
        b, tr = bradykinesia(base, task, "left", grade=g, criterion=crit, seed=10 + g, foot=PROBE_FOOT.get(task))
        out.append(Stimulus(f"bradykinesia:{task}:{crit}:g{g}", "bradykinesia", f"{task}:{crit}", g, _framed(b),
                            _framed(base), tr, {"task": task, "side": "left"}))
    return out


V6_DYSTONIA = {"table": "healthy"}           # the dystonia operator of sp5-v6 (module docstring)
V6_ARM = {"arm": "ride"}                    # its shoulder elevation and trunk: the arms turn with the collar, the trunk


def _v6(region):
    return dict(V6_DYSTONIA, **(V6_ARM if region in ("shoulder_elevation", "trunk") else {}))


def _dystonia_set(n):
    out = []
    stand = static_body("stand", (n - 1) / 100.0)
    out.append(Stimulus("dystonia:none:g0", "dystonia", "none", 0, _framed(stand), _framed(stand),
                        dystonic_posture(stand, "neck", "rotation_left", amplitude_grade=0, **_v6("neck"))[1],
                        {"joint": 15}))
    plan = [("neck", "rotation_left", g) for g in (1, 2, 3, 4)] + \
        [("neck", "tilt_right", 2), ("neck", "anterocollis", 2), ("shoulder_elevation", "elevation", 2),
         ("trunk", "lateral_left", 3), ("distal_arm", "elbow_flexion", 3)]
    for region, direction, g in plan:
        b, tr = dystonic_posture(stand, region, direction, "left", amplitude_grade=g, seed=3, **_v6(region))
        chain = tr["params"]["chain"]
        out.append(Stimulus(f"dystonia:{region}:{direction}:g{g}", "dystonia", f"{region}:{direction}", g,
                            _framed(b), _framed(stand), tr, {"joint": int(chain[-1])}))
    return out


def probe_set(families=FAMILIES, n=WINDOW_KNOTS):
    """[Stimulus] of the probe (module docstring)."""
    make = {"tremor": _tremor_set, "freezing": _freezing_set, "bradykinesia": _brady_set, "dystonia": _dystonia_set}
    out = []
    for f in families:
        out += make[f](n)
    return out


# ---------------------------------------------------------------- read-back
def _heading(Gl):
    from ..body.rotations import rot_y, yaw_angle
    return rot_y(-yaw_angle(Gl[:, 0]))


def read_back(stim, body, ref=None):
    """{measure: value} of the stimulus's family on body (module docstring); ref: the reference body of the
    dystonia family (default the stimulus's level-0 truth)."""
    rate = body.rate
    J, Gl = body.fk()
    H = _heading(Gl)
    if stim.family == "tremor":
        side = stim.info["side"]
        v = torch.einsum("tij,tj->ti", H, J[:, WRIST[side]] - J[:, SHOULDER[side]]).numpy()
        f = dominant_frequency(v, rate, (3.0, 12.0))
        return {"frequency_hz": f, "wrist_pp_cm": 100.0 * band_amplitude_pp(v, rate, f)}
    if stim.family == "freezing":
        ep = stim.info["episode"]
        p = J[:, 0, [0, 2]].numpy()
        sp = np.hypot(*np.gradient(p, 1.0 / rate, axis=0).T)
        if ep is None:
            return {"walk_speed_m_s": float(sp.mean())}
        h0, h1 = int(round(ep["hold_s"][0] * rate)), int(round(ep["hold_s"][1] * rate))
        w1 = int(round((ep["onset_s"] - 0.25) * rate))
        w0 = max(0, w1 - int(round(2.0 * rate)))
        out = {"hold_speed_m_s": float(sp[h0:h1].mean()), "walk_speed_m_s": float(sp[w0:w1].mean()),
               "hold_travel_m": float(np.hypot(*(p[h1 - 1] - p[h0])))}
        out["speed_ratio"] = out["hold_speed_m_s"] / out["walk_speed_m_s"] if out["walk_speed_m_s"] > 0 else math.nan
        pp = {kn: float(J[h0:h1, kn, 1].max() - J[h0:h1, kn, 1].min()) for kn in (4, 5)}
        kk = max(pp, key=pp.get)
        acc = vertical_acceleration(J, kk, rate)[h0:h1]
        out.update(knee_pp_m=pp[kk], knee=kk, knee_frequency_hz=dominant_frequency(acc, rate, (0.5, 12.0)),
                   freeze_ratio=freeze_ratio(acc, rate))
        return out
    if stim.family == "bradykinesia":
        p = stim.truth["params"]
        if not p.get("cycles"):
            return {}
        k0 = int(round(p["start_s"] * rate))
        o = task_angle(body, stim.info["task"], stim.info["side"], k0)
        amps = []
        for c in p["cycles"]:
            a = int(round(c["start_s"] * rate))
            b = a + int(round(c["period_s"] * rate))
            line = np.linspace(o[a], o[b], b - a + 1)
            amps.append(float(np.abs(o[a:b + 1] - line).max()))
        out = {"amplitudes_deg": amps, "dec_pct": decrement_dec(amps)}
        on = p.get("decrement_onset")
        if on:
            out["post_onset_slope_deg"] = slope(amps[on - 1:])
        return out
    if stim.family == "dystonia":
        ref = stim.base if ref is None else ref
        j = stim.info["joint"]
        _, Gr = ref.fk()
        Hr = _heading(Gr)
        A = H @ Gl[:, j]
        B = Hr @ Gr[:, j]
        dev = rotation_angle_deg(B.transpose(-1, -2) @ A).numpy()
        return {"deviation_deg": float(np.median(dev))}
    raise ValueError(stim.family)


def truth_value(stim):
    """{measure: value} the operator injected, in the read_back units."""
    p = stim.truth["params"]
    if stim.family == "tremor":
        return {"frequency_hz": p["frequency_hz"], "wrist_pp_cm": p.get("cm_pp") or 0.0}
    if stim.family == "freezing":
        ep = stim.info["episode"]
        if ep is None:
            return {}
        return {"hold_travel_m": 0.0, "knee_frequency_hz": ep["tremble_hz"]}
    if stim.family == "bradykinesia":
        if not p.get("cycles"):
            return {}
        amps = [c["amplitude_deg"] for c in p["cycles"]]
        out = {"amplitudes_deg": amps, "dec_pct": p["dec_dr12_3_pct"]}
        if p.get("decrement_onset"):
            out["post_onset_slope_deg"] = -p["decrement_slope"] * p["amplitude0_deg"]
        return out
    if stim.family == "dystonia":
        return {"deviation_deg": p.get("peak_deg", 0.0)}
    raise ValueError(stim.family)


# ---------------------------------------------------------------- levels in physical units
def levels():
    """Every grade map of grades.py in physical units (JSON-ready), the posture checks and the exam length."""
    lim = joint_limits()
    out = {"joint_limit_table": {"path": lim.path, "percentiles": list(lim.percentiles), "n_sampled": lim.n_sampled}}
    sit = static_body("sit", 12.0)
    arms = static_body("sit_arms_forward", 1.0)
    stand = static_body("stand", 1.0)
    out["postures"] = {p: limit_violations(static_body(p, 0.1).rot6d) for p in POSTURES}
    rest = rest_joints()
    tr = {"forearm_m": {s: float((rest[WRIST[s]] - rest[ELBOW[s]]).norm()) for s in ("left", "right")},
          "hinge_radius_m": {}, "levels": [], "constancy_share": {g: G.constancy_share(g) for g in range(5)},
          "frequency_bands_hz": {k: list(v) for k, v in G.TREMOR_TYPES.items()},
          "floor_peak_mm": dict(G.FLOOR_PEAK_MM),
          "floor_pp_cm": {k: 2 * v / 10.0 for k, v in G.FLOOR_PEAK_MM.items()},
          "bands_cm": [[lo, None if math.isinf(hi) else hi] for lo, hi in G.TREMOR_BANDS_CM]}
    for posture, body in (("sit", sit), ("sit_arms_forward", arms)):
        tr["hinge_radius_m"][posture] = {s: float(hinge_radius(body, s).median()) for s in ("left", "right")}
    tr["elbow_rooms"] = []                     # the bound of tremor.tremor: room and centre shift per posture
    for posture in POSTURES:
        b = static_body(posture, 0.1)
        for s in ("left", "right"):
            rad = float(hinge_radius(b, s).median())
            rp, rm = driver_rooms(b, ELBOW[s], "y", "parent", torch.ones(b.T, dtype=torch.bool), 180.0)
            flex, ext = (rp, rm) if ELBOW_FLEXION_SIGN[s] > 0 else (rm, rp)
            row = {"posture": posture, "side": s, "radius_m": rad, "flexion_room_deg": flex,
                   "extension_room_deg": ext}
            for g in (1, 2, 3, 4):
                pk = G.elbow_angle_deg(G.tremor_level_cm(g), rad)
                row[f"g{g}_peak_deg"] = pk
                try:
                    row[f"g{g}_shift_toward_flexion_deg"] = ELBOW_FLEXION_SIGN[s] * centre_shift(pk, rp, rm) + 0.0
                except ValueError:
                    row[f"g{g}_shift_toward_flexion_deg"] = None
            tr["elbow_rooms"].append(row)
    for g, g1 in ((1, "below_floor"), (1, "above_floor"), (2, None), (3, None), (4, None)):
        cm = G.tremor_level_cm(g, g1 or "above_floor")
        row = {"grade": g, "grade1_level": g1, "cm_pp": cm}
        for s in ("left", "right"):
            r = tr["hinge_radius_m"]["sit"][s]
            pk = G.elbow_angle_deg(cm, r)
            row[f"{s}_peak_deg"] = pk
            row[f"{s}_pp_deg"] = 2 * pk
        tr["levels"].append(row)
    out["tremor"] = tr
    mu, sig = G.fog_lognormal()
    from scipy.stats import norm
    lo_d, hi_d = G.DURATION_RANGE_S
    ca, cb = norm.cdf((math.log(lo_d) - mu) / sig), norm.cdf((math.log(hi_d) - mu) / sig)
    qs = {f"q{int(q * 100)}": float(math.exp(mu + sig * norm.ppf(ca + q * (cb - ca))))
          for q in (0.1, 0.25, 0.5, 0.75, 0.9)}
    tot = sum(G.TRIGGER_SHARES[k] for k in G.GRADE_TRIGGERS)
    out["freezing"] = {"grades": {str(k): v for k, v in G.FOG_GRADES.items()},
                       "duration": {"median_s": G.DURATION_MEDIAN_S, "p90_s": G.DURATION_P90_S, "mu_log": mu,
                                    "sigma_log": sig,
                                    "range_s": list(G.DURATION_RANGE_S), "quantiles_s": qs},
                       "trigger_weights": {k: G.TRIGGER_SHARES[k] / tot for k in G.GRADE_TRIGGERS},
                       "published_contexts_pct": dict(G.TRIGGER_SHARES),
                       "freeze_band_hz": list(G.FREEZE_BAND_HZ), "locomotor_band_hz": list(G.LOCOMOTOR_BAND_HZ),
                       "tremble_default_hz": G.TREMBLE_DEFAULT_HZ}
    br = {"ladders": {"interruptions": {str(k): v for k, v in G.BRADY_INTERRUPTIONS.items()},
                      "decrement_onset": {str(k): v for k, v in G.BRADY_DECREMENT_ONSET.items()},
                      "decrement_slope": G.BRADY_DECREMENT_SLOPE}, "tasks": {}}
    from .bradykinesia import LEVEL0_SHARE
    for task, tk in TASKS.items():
        base = sit if tk["posture"] == "sit" else static_body(tk["posture"], 12.0)
        room = float(sum(task_rooms(base, task, "left")))
        a0 = LEVEL0_SHARE * room
        b, t = bradykinesia(base, task, "left", grade=3, criterion="decrement", start_s=0.1)
        lift = {}
        if task != "pronation_supination":
            J0 = base.joints()
            b1, _ = bradykinesia(base, task, "left", amplitude0_deg=a0, n_cycles=1, start_s=0.1)
            J1 = b1.joints()
            jj = 10 if task == "toe_tapping" else 7
            lift = {"level0_lift_cm": float(100 * (J1[:, jj, 1] - J0[:, jj, 1]).max()), "lift_joint": jj}
        dec = {}
        for g in (1, 2, 3):
            A = amplitudes(G.N_CYCLES, 1.0, G.BRADY_DECREMENT_ONSET[g], G.BRADY_DECREMENT_SLOPE)
            dec[str(g)] = {"amplitude_share": [float(x) for x in A], "dec_dr12_3_pct": decrement_dec(A)}
        br["tasks"][task] = {"item": tk["item"], "posture": tk["posture"], "chain": list(tk["chain"]["left"]),
                             "room_deg": room, "level0_deg": a0, "level0_hz": tk["level0_hz"],
                             "period_s": t["params"]["period_s"], "duration_10_cycles_s": 10 * t["params"]["period_s"],
                             **lift}
        br["decrement"] = dec
    out["bradykinesia"] = br
    dy = []
    for region, reg in REGIONS.items():
        for direction in reg["dirs"]:
            for posture, body in (("stand", stand), ("sit", sit)):
                chain, axis, sign, kind, frame, scale = _resolve(region, direction, "left")
                row = {"region": region, "direction": direction, "posture": posture, "scale": scale,
                       "chain": list(chain), "axis": axis, "rom_deg": rom(chain, axis)}
                n_g = 3 if (scale == "TWSTRS" and kind in G.TWSTRS_THIRDS) else \
                    len(G.TWSTRS_BANDS_DEG[kind]) if scale == "TWSTRS" else 4
                for g in range(1, n_g + 1):
                    lv, possible = level_deg(region, direction, "left", g, body)
                    row["possible_deg"] = possible
                    row[f"g{g}_deg"] = lv
                    row[f"g{g}_reachable"] = lv is not None and lv <= possible
                dy.append(row)
    out["dystonia"] = {"rows": dy, "twstrs_duration_share": {g: list(G.twstrs_duration(g)) for g in range(1, 6)},
                       "dis_share": {g: G.dis_share(g) for g in range(1, 5)}, "submaximal": G.SUBMAXIMAL}
    ex, total = [], 0.0
    for seg, what, dur, kind in EXAM:
        if kind == "reach":
            d = 2 * 3 * REACH_PERIOD_S
        elif kind in TASKS:
            d = 2 * 10 * 2 * int(round(100.0 / TASKS[kind]["level0_hz"] / 2)) / 100.0
        elif kind == "arise":
            d = ARISE_S + dur
        else:
            d = dur
        total += d
        ex.append({"segment": seg, "task": what, "duration_s": d, "source": "draft" if kind == "draft" else "module"})
    out["exam"] = {"segments": ex, "total_s": total}
    return out
