"""The repeated hand tasks of MDS-UPDRS 3.4 (finger tapping) and 3.5 (hand movements) with the hand joints of
the program's body (task SP5), so that the bradykinesia operator's decrement, interruptions, arrests and slowing act on them
(docs/proposals/SYNTH_PATIENT_DRAFT.md, section 5: version 1 refused both items for want of fingers).

Base postures (motions.POSTURES3): 'sit_finger_tapping' (the arm raised in front at about shoulder height, the
forearm semi-flexed, the palm toward the midline) and 'sit_hand_movements' (the elbow bent, the palm toward the
examiner), the task hand held closed: for finger tapping the index tip on the thumb tip (PINCH), for hand movements
the fist (FIST).

Hand poses (hands.hand_rotations: flexion and abduction about each joint's anatomical axes):
- PINCH: the index MCP, PIP and DIP and the thumb CMC (flexion, palmar abduction), MCP and IP angles that bring the
  index fingertip vertex onto the thumb's, found once on the finger-tapping posture by a bounded search (inside
  the hand limits of hands.py) that keeps the two distal segments' skins at least 2 mm apart at closure and asks the
  fingertip distance to grow as the share opens, and kept as numbers; tests/test_synth_pathology.py checks the
  closure, the gap and the growth on the task posture;
- FIST: every knuckle (MCP) at its cited largest flexion (hands.FLEX_LIMITS_DEG, B.K. et al. 2024), the thumb MCP
  at its cited largest (Boike et al. 2025), the thumb CMC at module angles (FIST_THUMB_CMC: on the open body Anny's
  rest thumb already lies turned toward the palm, the palmar abduction keeps it over the fingers); the distal
  joints (PIP and DIP of a finger, IP of the thumb) at the largest common share of their cited largest flexions for
  which the digit's distal skin keeps FIST_CLEARANCE_MM from the palm skin (palm_clearance_mm), found once on the
  hand-movements posture by bisection per digit and kept as numbers (review of 2026-10-05: at the cited largest
  flexions, FIST_CITED, the middle, ring and little fingertips lay several millimetres inside the palm);
  tests/test_synth_pathology.py checks the clearance, and that FIST_CITED fails it.
The open hand is the rest hand of the selected body model (fingers straight). A task share s in [0, 1] sets every joint of
the closed pose to (1 - s) times its closed angles: s = 0 closed, s = 1 open.

Task measure and amplitude (physical units):
- finger tapping: the distance between the index and thumb fingertip vertices of the skinned mesh; the amplitude of
  a cycle is its peak minus the closed distance (the definition of Matejicka et al. 2024: the maximal against the
  minimal separation); the share of each cycle is solved on the task posture so that the amplitude is the asked
  one (bisection on the skinned tips);
- hand movements: the mean flexion of the four finger knuckles (MCP, deg) from the fist; the amplitude of a cycle
  is the share times the fist's mean MCP flexion (the full fist-to-open range at share 1, the form's 'open the
  hand fully').
Level 0 (grades.BRADY_NORMAL): the cited normal rate and amplitude; finger tapping both from the 12 controls of
Matejicka et al. 2024 (3.67 Hz, 68.09 mm; review of 2026-10-05), hand movements the rate per side of Maremmani et
al. 2022 (movements in 10 s of 50 healthy controls) and the full range.

Cycles (as bradykinesia.py, on a continuous clock): n_cycles cycles of period 1 / rate, s(t) = s_c (1 - cos(2 pi
(t - t_c) / P)) / 2 inside cycle c, the hand closed between cycles and during pauses; the cycle starts are exact
times (not rounded to knots), so the rate is the asked one exactly and the knots sample it. Criteria: decrement
(onset cycles 9, 5, 2 at 10 percent of A0 per cycle), interruptions (2, 4, 7 hesitations of half a level-0 period,
at least bradykinesia.HESITATION_MIN_S since wave SH1d: bradykinesia.hesitation_length_s, the review's M14, half a
period being 0.14 s at the cited finger-tapping rate; a hesitation is then longer than one cycle at these rates and
shorter than an arrest), arrest (one, two level-0 periods, at least bradykinesia.ARREST_MIN_S since wave SH1d:
bradykinesia.arrest_length_s, three hesitation floors, above two periods at the cited finger-tapping and
hand-movement rates), slowing (grades.slowing_rate_hz: the cited normal rate times the ladder ratio to the power of
the grade; a proposal for the neurologist). A grade of 0 returns a copy of the input.
"""
import math

import numpy as np
import torch

from ..body.rotations import axis_angle_to_matrix, matrix_to_rot6d
from . import grades as G
from . import hands as H
from .bradykinesia import ARREST_MIN_S, HESITATION_MIN_S, amplitudes, arrest_length_s, hesitation_length_s
from .body import validity
from .common import as_rng, truth_record

# index (MCP, PIP, DIP) and thumb ((CMC flexion, CMC palmar abduction), MCP, IP), deg (module docstring)
# solved again on the open body (port wave 2, scripts/openbody_handtask_poses.py; SMPL+H: index (27.61, 53.65, 55.85),
# thumb ((18.20, 32.93), 6.18, 33.12))
PINCH = {"index": (15.89, 71.37, 75.23), "thumb": ((-6.52, 2.35), 41.86, 25.18)}
FIST_CITED = {f: tuple(H.FLEX_LIMITS_DEG[f][lv][1] for lv in H.LEVELS) for f in ("index", "middle", "ring", "pinky")}
FIST_CITED["thumb"] = ((30.0, 20.0), H.FLEX_LIMITS_DEG["thumb"]["mcp"][1], H.FLEX_LIMITS_DEG["thumb"]["ip"][1])
FIST_CLEARANCE_MM = 1.0         # [module] the distal skin of each digit this far from the palm skin in the fist
# [module] the fist's thumb CMC (flexion, palmar abduction) on the open body (wave 2 of the port): Anny's rest thumb
# already lies turned toward the palm, and the SMPL+H fist's 30 deg of CMC flexion put the thumb's distal skin 3 to 8
# mm into the palm at its cited MCP flexion (scripts/openbody_handtask_poses.py); the palmar abduction 20 kept
FIST_THUMB_CMC = (0.0, 20.0)
# the fist (module docstring) on the open body (port wave 2, scripts/openbody_handtask_poses.py): shares of the cited
# largest distal flexions index 0.688, middle 0.800, ring 0.729, little 0.666, thumb IP 0.938 with FIST_THUMB_CMC
# (on SMPL+H: 0.916, 0.861, 0.868, 0.811 and 0.373 with the CMC at (30, 20))
FIST = {"index": (86.0, 66.84, 56.11), "middle": (86.6, 76.97, 65.29), "ring": (84.2, 69.94, 59.45),
        "pinky": (85.0, 61.11, 56.32), "thumb": ((0.0, 20.0), 56.0, 68.62)}
TASKS = {
    "finger_tapping": {"item": "3.4", "posture": "sit_finger_tapping", "closed": PINCH, "unit": "mm",
                       "measure": "index-to-thumb fingertip distance minus its closed value"},
    "hand_movements": {"item": "3.5", "posture": "sit_hand_movements", "closed": FIST, "unit": "deg",
                       "measure": "mean flexion of the four finger knuckles (MCP) below the fist's"},
}
CRITERIA = ("decrement", "interruptions", "arrest", "slowing")


def closed_angles(task, side):
    """[(joint, flex_deg, abd_deg)] of the task's closed hand on one side."""
    out = []
    for f, vals in TASKS[task]["closed"].items():
        for lv, v in enumerate(vals):
            fl, ab = (v, 0.0) if np.isscalar(v) else v
            out.append((H.hand_joint(side, f, lv), float(fl), float(ab)))
    return out


def hand_at(task, side, share):
    """(N, 30, 3, 3) hand rotations of the task at shares (N,) (module docstring); the other side's rows at rest."""
    fr = H.hand_frame()[side]["joints"]
    s = torch.as_tensor(np.asarray(share, dtype=np.float64).reshape(-1))
    R = torch.eye(3, dtype=torch.float64).repeat(len(s), H.N_HAND, 1, 1)
    for j, fl, ab in closed_angles(task, side):
        k = (1.0 - s)
        M = torch.eye(3, dtype=torch.float64).expand(len(s), 3, 3)
        if ab:
            M = axis_angle_to_matrix(torch.as_tensor(fr[j]["abd"])[None] * torch.deg2rad(k * ab)[:, None]) @ M
        if fl:
            M = axis_angle_to_matrix(torch.as_tensor(fr[j]["flex"])[None] * torch.deg2rad(k * fl)[:, None]) @ M
        R[:, H.hand_row(j)] = M
    return R


def task_rows(side):
    return list(range(0, 15)) if side == "left" else list(range(15, 30))


def tip_distance_m(body, side, knots=None):
    """(N,) m: the distance between the index and thumb fingertip vertices of the skinned mesh."""
    tips = H.tip_vertices()[side]
    P = H.body_points(body, [tips["index"], tips["thumb"]], knots=knots)
    return np.linalg.norm(P[:, 0] - P[:, 1], axis=-1)


def mcp_flexion_deg(R_hand, side):
    """(N,) deg: the mean flexion of the four finger knuckles of hand rotations (N, 30, 3, 3)."""
    comp = H.hand_components(R_hand)
    return np.mean([comp[H.hand_joint(side, f, 0)][0] for f in ("index", "middle", "ring", "pinky")], axis=0)


def _with_shares(body, task, side, share, knot):
    """A body of len(share) knots (at least two: a single share is repeated): the body joints of one knot, the task
    hand at the shares."""
    from dataclasses import replace
    share = np.asarray(share, dtype=np.float64).reshape(-1)
    if len(share) < 2:
        share = np.r_[share, share]
    n = len(share)
    Rh = body.hand_R()[knot][None].repeat(n, 1, 1, 1)
    Rh[:, task_rows(side)] = hand_at(task, side, share)[:, task_rows(side)]
    b = H.HandBody(body.rot6d[knot][None].repeat(n, 1, 1), body.transl[knot][None].repeat(n, 1),
                   body.contact[knot][None].repeat(n, 1), body.betas, int(body.gender), body.rate)
    return replace(b, hand6d=matrix_to_rot6d(Rh))


def amplitude_of_share(body, task, side, share, knot=0):
    """(N,) the task amplitude (TASKS unit) at shares (N,) on the body's posture at a knot (module docstring)."""
    share = np.asarray(share, dtype=np.float64).reshape(-1)
    if task == "hand_movements":
        return mcp_flexion_deg(hand_at(task, side, np.zeros(1)), side)[0] - \
            mcp_flexion_deg(hand_at(task, side, share), side)
    d = tip_distance_m(_with_shares(body, task, side, np.r_[0.0, share], knot), side)
    return 1000.0 * (d[1:1 + len(share)] - d[0])


def share_of_amplitude(body, task, side, amp, knot=0, iters=48):
    """(N,) shares whose amplitude (amplitude_of_share) is amp (N,), by bisection (the amplitude rises with the
    share; checked on a grid). Raises ValueError beyond the open hand's amplitude."""
    amp = np.asarray(amp, dtype=np.float64).reshape(-1)
    grid = np.linspace(0.0, 1.0, 41)
    a = amplitude_of_share(body, task, side, grid, knot)
    if not (np.diff(a) > 0).all():
        raise RuntimeError(f"{task}: the amplitude does not rise with the share on this posture")
    if (amp > a[-1] + 1e-9).any():
        raise ValueError(f"{task}: {amp.max():.2f} {TASKS[task]['unit']} asked, the open hand gives {a[-1]:.2f}")
    lo = np.interp(amp, a, grid) - 0.05
    hi = lo + 0.1
    lo, hi = np.clip(lo, 0.0, 1.0), np.clip(hi, 0.0, 1.0)
    for _ in range(iters):
        m = 0.5 * (lo + hi)
        below = amplitude_of_share(body, task, side, m, knot) < amp
        lo, hi = np.where(below, m, lo), np.where(below, hi, m)
    return 0.5 * (lo + hi)


def level0(task, side="left"):
    """(amplitude0 in the task unit, rate Hz, info) of the cited normal performance (grades.BRADY_NORMAL); the
    hand-movement amplitude, given as a share of the fist-to-open range, is turned into degrees of mean knuckle
    flexion (the fist's mean MCP flexion times the share)."""
    a = G.BRADY_NORMAL[task]
    rate, sd = G.normal_rate_hz(task, side)
    a0 = a["amplitude"]
    if task == "hand_movements":
        a0 = a["amplitude"] * float(mcp_flexion_deg(hand_at(task, side, np.zeros(1)), side)[0])
    return a0, rate, {"rate_hz": rate, "rate_sd_hz": sd, "rate_source": a["rate_source"],
                      "amplitude": a["amplitude"], "amplitude_unit": a["amplitude_unit"],
                      "amplitude_source": a["amplitude_source"], "amplitude0": a0, "unit": TASKS[task]["unit"]}


def hand_task(body, task="finger_tapping", side="left", grade=None, criterion="decrement", n_cycles=G.N_CYCLES,
              amplitude0=None, rate_hz=None, speed_gain=1.0, decrement_onset=None,
              decrement_slope=G.BRADY_DECREMENT_SLOPE, n_hesitations=0, n_arrests=0, hesitation_s=None,
              arrest_s=None, start_s=0.5, seed=0, check=True, hesitation_floor_s=HESITATION_MIN_S,
              arrest_floor_s=ARREST_MIN_S):
    """(HandBody, truth) of a repeated hand task with bradykinesia on one side (module docstring).

    grade with criterion ('decrement', 'interruptions', 'arrest' by grades.brady_parameters; 'slowing' by
    grades.slowing_rate_hz) sets the parameters; otherwise give them directly. amplitude0 (TASKS unit) and rate_hz
    default to the cited level 0 (level0); speed_gain scales the rate. hesitation_s (default
    bradykinesia.hesitation_length_s of the level-0 period with hesitation_floor_s; None: half a period, the
    version before wave SH1d); arrest_s likewise (bradykinesia.arrest_length_s with arrest_floor_s). The task
    starts at start_s and must end inside the body. seed: int or numpy Generator (pause positions)."""
    if task not in TASKS:
        raise ValueError(f"task {task!r}, expected one of {sorted(TASKS)}")
    if criterion not in CRITERIA:
        raise ValueError(f"criterion {criterion!r}, expected one of {CRITERIA}")
    tk = TASKS[task]
    seed_val = seed if isinstance(seed, (int, np.integer)) else None
    a0_cited, f0, anchor = level0(task, side)
    f0 = f0 if rate_hz is None else float(rate_hz)
    slowing_grade = None
    if grade is not None:
        g = int(grade)
        if criterion == "slowing":
            slowing_grade = g
            speed_gain = G.slowing_rate_hz(task, g, side) / G.normal_rate_hz(task, side)[0]
            n_hesitations = n_arrests = 0
            decrement_onset, decrement_slope = None, 0.0
        else:
            p = G.brady_parameters(g, criterion, decrement_slope)
            n_hesitations, n_arrests = p["n_hesitations"], p["n_arrests"]
            decrement_onset, decrement_slope = p["decrement_onset"], p["decrement_slope"]
        if g == 0:
            params = {"task": task, "item": tk["item"], "side": side, "criterion": criterion, "n_cycles": 0}
            return body.copy(), truth_record("bradykinesia", seed_val, 0, f"MDS-UPDRS {tk['item']}", params,
                                             ("mds_updrs",), None)
    if not H.has_hands(body):
        raise ValueError("the hand tasks need a HandBody (hands.hands_of, motions.posture_body)")
    rng = as_rng(seed)
    rate = f0 * float(speed_gain)
    P, P0 = 1.0 / rate, 1.0 / f0
    hes = hesitation_length_s(P0, hesitation_floor_s) if hesitation_s is None else float(hesitation_s)
    arr = arrest_length_s(P0, arrest_floor_s) if arrest_s is None else float(arrest_s)
    if n_hesitations and hes >= arr:
        raise ValueError("a hesitation must be shorter than an arrest")
    if arr <= P and n_arrests:
        raise ValueError("an arrest must be longer than one cycle period")
    k0 = int(round(start_s * body.rate))
    a0 = a0_cited if amplitude0 is None else float(amplitude0)
    A = amplitudes(n_cycles, a0, decrement_onset, decrement_slope)
    shares = share_of_amplitude(body, task, side, A, k0) if task == "finger_tapping" else \
        A / amplitude_of_share(body, task, side, np.ones(1), k0)[0]
    if (shares > 1.0 + 1e-12).any():
        raise ValueError(f"{task}: amplitude {A.max():.2f} beyond the open hand")
    boundaries = np.arange(1, n_cycles)
    n_p = n_hesitations + n_arrests
    if n_p > len(boundaries):
        raise ValueError(f"{n_p} pauses for {len(boundaries)} cycle boundaries")
    where = rng.choice(boundaries, size=n_p, replace=False) if n_p else np.zeros(0, np.int64)
    kinds = rng.permutation(["hesitation"] * n_hesitations + ["arrest"] * n_arrests) if n_p else []
    pause_after = {int(b): (str(kd), hes if kd == "hesitation" else arr) for b, kd in zip(where, kinds)}
    t = body.times().numpy()
    s = np.zeros(body.T)
    cycles, pauses, tc = [], [], k0 / body.rate
    for c in range(1, n_cycles + 1):
        if tc + P > body.duration:
            raise ValueError("the task does not end inside the body: lengthen it or start earlier")
        inside = (t >= tc) & (t <= tc + P)
        s[inside] = shares[c - 1] * 0.5 * (1.0 - np.cos(2.0 * math.pi * (t[inside] - tc) / P))
        cycles.append({"cycle": c, "start_s": tc, "period_s": P, "peak_s": tc + P / 2, "amplitude": float(A[c - 1]),
                       "share": float(shares[c - 1])})
        tc += P
        if c in pause_after:
            kd, dur = pause_after[c]
            pauses.append({"kind": kd, "after_cycle": c, "start_s": tc, "duration_s": dur})
            tc += dur
    Rh = body.hand_R().clone()
    Rh[:, task_rows(side)] = hand_at(task, side, s)[:, task_rows(side)]
    out = body.with_hands(Rh, task_rows(side))
    if task == "finger_tapping":
        d = tip_distance_m(out, side)
        sig = 1000.0 * (d - float(tip_distance_m(_with_shares(body, task, side, np.zeros(1), k0), side)[0]))
    else:
        sig = mcp_flexion_deg(hand_at(task, side, np.zeros(1)), side)[0] - mcp_flexion_deg(out.hand_R(), side)
    from .measure import decrement_dec
    open_amp = float(amplitude_of_share(body, task, side, np.ones(1), k0)[0])
    params = {"task": task, "item": tk["item"], "side": side, "criterion": criterion if grade is not None else None,
              "measure": tk["measure"], "unit": tk["unit"], "posture": tk["posture"],
              "joints": [int(j) for j, _, _ in closed_angles(task, side)], "n_cycles": int(n_cycles),
              "amplitude0": float(a0), "amplitude0_cited": a0_cited, "open_amplitude": open_amp,
              "level0_hz": float(f0), "rate_hz": float(rate), "speed_gain": float(speed_gain), "period_s": P,
              "anchor": anchor, "slowing_grade": slowing_grade,
              "slowing_ratio": G.slowing_ratio(task, side) if slowing_grade is not None else None,
              "decrement_onset": decrement_onset, "decrement_slope": float(decrement_slope),
              "dec_dr12_3_pct": decrement_dec(A), "n_hesitations": int(n_hesitations), "n_arrests": int(n_arrests),
              "hesitation_floor_s": None if hesitation_floor_s is None else float(hesitation_floor_s),
              "hesitation_s": float(hes),
              "arrest_floor_s": None if arrest_floor_s is None else float(arrest_floor_s), "arrest_s": float(arr),
              "cycles": cycles, "pauses": sorted(pauses, key=lambda p: p["start_s"]), "start_s": k0 / body.rate,
              "end_s": tc, "grade_any_criterion": max(G.brady_grade_of(n_hesitations, n_arrests, decrement_onset),
                                                      slowing_grade or 0)}
    if task == "finger_tapping":
        params["closed_tip_distance_mm"] = float(1000.0 * tip_distance_m(
            _with_shares(body, task, side, np.zeros(1), k0), side)[0])
        params["pinch_gap_mm"] = pinch_gap_mm(_with_shares(body, task, side, np.zeros(1), k0), side)
    val = None
    if check:
        val = validity(body, out)
        val["hand_limits"] = H.hand_limit_report(out.hand_R())
    srcs = ("mds_updrs", anchor["rate_source"], anchor["amplitude_source"], "module")
    return out, truth_record("bradykinesia", seed_val, grade, f"MDS-UPDRS {tk['item']}", params, srcs, val,
                             arrays={"share": s, "signal": np.asarray(sig, dtype=np.float64)})


def pinch_gap_mm(body, side, knot=0):
    """The smallest distance (mm) between the skins of the index and thumb distal segments at a knot: above zero,
    the closed pinch touches without the fingers passing through each other."""
    _, _, dom = H._rest()
    ia = np.flatnonzero(dom == H.hand_joint(side, "index", 2))
    it = np.flatnonzero(dom == H.hand_joint(side, "thumb", 2))
    P = H.body_points(body, np.r_[ia, it], knots=[knot])[0]
    A, C = P[:len(ia)], P[len(ia):]
    return float(1000.0 * np.sqrt(((A[:, None] - C[None]) ** 2).sum(-1)).min())


def _closest_on_triangles(p, a, b, c):
    """(P, F, 3) the closest points of the triangles (a, b, c) (F, 3) to the points p (P, 3) (the region tests of
    Ericson, Real-Time Collision Detection, 2005, section 5.1.5, vectorized)."""
    p, a, b, c = p[:, None], a[None], b[None], c[None]
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = (ab * ap).sum(-1), (ac * ap).sum(-1)
    bp, cp = p - b, p - c
    d3, d4 = (ab * bp).sum(-1), (ac * bp).sum(-1)
    d5, d6 = (ab * cp).sum(-1), (ac * cp).sum(-1)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2

    def safe(x):
        return np.where(x == 0, 1.0, x)
    den = safe(va + vb + vc)
    q = a + ab * (vb / den)[..., None] + ac * (vc / den)[..., None]          # inside the face
    edges = (((vc <= 0) & (d1 >= 0) & (d3 <= 0), a, ab, d1 / safe(d1 - d3)),
             ((vb <= 0) & (d2 >= 0) & (d6 <= 0), a, ac, d2 / safe(d2 - d6)),
             ((va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0), b, c - b, (d4 - d3) / safe((d4 - d3) + (d5 - d6))))
    for m, o, e, t in edges:
        q = np.where(m[..., None], o + e * t[..., None], q)
    for m, o in (((d1 <= 0) & (d2 <= 0), a), ((d3 >= 0) & (d4 <= d3), b), ((d6 >= 0) & (d5 <= d6), c)):
        q = np.where(m[..., None], o + 0.0 * p, q)
    return q


def palm_clearance_mm(body, side, knot=0, digits=("index", "middle", "ring", "pinky", "thumb")):
    """{digit: mm}: the smallest signed distance of the skin of a digit's distal segment (the vertices skinned
    mostly by its DIP joint, IP for the thumb) to the palm skin (the faces whose three vertices are skinned mostly
    by the wrist) at a knot: positive outside the palm, negative inside, signed by the outward normal of the nearest
    face (the faces of both body models wind outward). The distal segments do not border the palm, so the sign is that of the
    side of the palm surface they lie on."""
    from .viewer import faces as mesh_faces
    faces = mesh_faces()
    _, _, dom = H._rest()
    tri = faces[(dom[faces] == H.WRIST[side]).all(1)]
    seg = {d: np.flatnonzero(dom == H.hand_joint(side, d, 2)) for d in digits}
    ids = np.unique(np.concatenate([tri.ravel()] + list(seg.values())))
    P = H.body_points(body, ids, knots=[knot])[0]
    pos = np.zeros((len(dom), 3))
    pos[ids] = P
    a, b, c = pos[tri[:, 0]], pos[tri[:, 1]], pos[tri[:, 2]]
    n = np.cross(b - a, c - a)
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    out = {}
    for d, v in seg.items():
        x = pos[v]
        q = _closest_on_triangles(x, a, b, c)
        dist = np.linalg.norm(x[:, None] - q, axis=-1)
        k = dist.argmin(1)
        r = np.arange(len(x))
        s = np.sign(((x - q[r, k]) * n[k]).sum(-1)) * dist[r, k]
        out[d] = float(1000.0 * s.min())
    return out


def read_cycles(signal, rate, rel_floor=0.02):
    """[(start_knot, peak_knot, peak_value)] of the cycles of a task signal (T,): runs above rel_floor of its
    maximum, each with its largest sample (the read-back of the tests)."""
    x = np.asarray(signal, dtype=np.float64)
    on = x > rel_floor * x.max()
    m = np.concatenate([[False], on, [False]])
    d = np.diff(m.astype(np.int8))
    out = []
    for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        k = a + int(np.argmax(x[a:b]))
        out.append((int(a), int(k), float(x[k])))
    return out
