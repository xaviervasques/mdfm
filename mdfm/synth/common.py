"""Shared pieces of the operators: on and off schedules with raised-cosine ramps, joint offsets bounded by the
SP1 joint-limit table, and the truth record.

Schedule: on intervals [a, b] (s) between ramp midpoints. The envelope is 1 inside [a + r/2, b - r/2], rises
over [a - r/2, a + r/2] and falls over [b - r/2, b + r/2] as raised cosines (r = ramp_s), with no ramp at an
interval edge that touches the start or the end of the body, so that the time spent above half the level is
exactly b - a. schedule() places n bouts of a requested total share at random from a seed, boundaries on
knots (the realized share is the requested one rounded to a knot), gaps and bouts at least one ramp long.

Truth record: a dict with the operator, its version, the seed, the grade and the scale it was mapped from, the
parameters actually applied in physical units, the sources of each map (grades.SOURCES keys), the kinematic
checks of body.validity, and 'arrays' (numpy arrays, left out of truth_json).
"""
import math

import numpy as np
import torch

from .body import F64

VERSION = "sp5-v12"         # v2 (review of 2026-10-05): tremor bounded by the table, right forearm supinates,
#                             freezing grade triggers start, turn and doorway only; v3 (SP4, 2026-10-05): leg
#                             postures and toe tapping keep the foot on the floor (contact.py), foot 'free' is v2;
#                             v4 (SP5, 2026-10-05): the hand joints (hands.py), the hand tasks of 3.4 and 3.5
#                             (handtasks.py), the raise to the 3.15 posture and the arising (motions.py); the v3
#                             operators unchanged; v5 (review of SP5, 2026-10-05): the arising balanced (motions.py),
#                             the fist clear of the palm (handtasks.FIST), finger tapping on one cohort
#                             (grades.BRADY_NORMAL); the v3 operators and the raise unchanged; v6 (shared wave of
#                             the clinical review, 2026-11-02, docs/proposals/SYNTH_SHARED_CHANGES_DRAFT.md):
#                             tremor.tremor driver 'postural' and scene clearance 'clear', parkinsonian rest band 4
#                             to 7 Hz (bhatia2018), exam E1 g4 and every E2 tremor body changed (exam_v5); the exam
#                             chair of SH5 (viewer.chair_mesh) and the arising backrest (motions.arise_chair); the
#                             SARA 3 seated clips moved by ataxia.seat_forward on that chair; the SARA 8 heel-shin
#                             slide (ataxia.py); scissoring in the spastic diplegic gait (gait.py, legcontact.py);
#                             touching tics (reach.touch_path, jerks.py); v7 (SH2 of that note, owner's go of
#                             2026-11-02): the dystonia operators read the SH2 joint-limit table (limits_ext.py),
#                             the shoulder elevation keeps the arm hanging (dystonia.py, arm 'hang'), exam
#                             dystonia_E7_shoulder_elevation_elevation_g1 to g3 changed; the other exam bodies and
#                             the SP3 probe stimuli (stimuli.py, pinned to the v6 operator) unchanged; v8 (wave
#                             SH1b, 2026-11-03): the folded arms with the forearms apart (motions.ARM_POSTURES
#                             ['fold'], SH9) and the arising's standing hip drop, exam ref_arise_E7 changed
#                             (exam_v6); the geste antagoniste on the right cheek (postures.py); the seated knees
#                             kept apart (involuntary.knee_guard); the gaits' leg guards on a wider healthy base
#                             (gait.py, legcontact.py); the hyperekplexias paired with a normal startle (jerks.py);
#                             the controlled sit-down on a solved clock and parkinsonism's background from physio's
#                             values (parkinsonism.py); the other exam bodies unchanged; v9 (wave SH1d, the clinical
#                             review of the English page, 2026-11-03): the 3.17 rest tremor on the 'rest' driver of
#                             tremor.py, levelled at the hand, the tremor raise clips on a new level 0 with a longer
#                             rest, the alternating freezing with the arms settling (freezing.py), the hesitation
#                             and arrest floors (bradykinesia.py, handtasks.py) and the trunk lateral flexion with the
#                             arms hanging (dystonia.py): the exam bodies of exam.BODY_CHANGED_V7 (exam_v7), the SP3
#                             probe stimuli unchanged; the
#                             seated rest with the palms on the thighs, the speed caps and the arm guards of the
#                             involuntary clips, the weight shift of a step in place (involuntary.py); the generated
#                             walks (gait.py: continuity, turns, tandem, sway, retropulsion, the dystonic foot); the
#                             titubation levelled at the head and the stance tremor (ataxia.py);
#                             v10 (wave SH1c, from v8;
#                             v9 is wave SH1d's): the fold solved in the tree (motions.solve_fold), its opening and
#                             push-off on their skin gaps, exam ref_arise_E7 changed (exam_v6); the shrugs of the
#                             jerks and of 3.14 keep the arm hanging (limits_ext 'shrug'); the dysmetric reach over
#                             the target hold, joined to the level 0's path (reach.py); the writing hand and the pen
#                             on the paper, the PNKD arms out of the backrest and armrests, the PKD hand out of the
#                             trunk and thigh (postures.py); the other exam bodies unchanged; v11 (the settling of
#                             waves SH1c and SH1d in main, 2026-11-10): the tremor raise clips rest
#                             exam.REST_BEFORE_RAISE_S again before the raise, their exam bodies changed (exam_v8);
#                             involuntary.lift_off swings a seated elbow outward about the shoulder-to-wrist line where
#                             its lift and turn do not clear a run, and keeps a run's least deep candidate only where
#                             it is shallower than the run (invol_choreoathetosis_E1_a3_d3 changed); the other exam
#                             bodies unchanged; v12 (wave SH1d part C, 2026-11-11): the freezing clips take the
#                             standing background of physio.py on their holds (physio.with_hold_background, exam_v9);
#                             the other exam bodies unchanged


def envelope(times, intervals, ramp_s, t_end):
    """(T,) float64 envelope of on intervals (module docstring) at times (T,) s; t_end the body's last time."""
    t = np.asarray(times, dtype=np.float64)
    e = np.zeros_like(t)
    h = ramp_s / 2.0
    for a, b in intervals:
        up = np.ones_like(t) if a <= 0 or ramp_s <= 0 else np.clip((t - (a - h)) / ramp_s, 0.0, 1.0)
        dn = np.ones_like(t) if b >= t_end or ramp_s <= 0 else np.clip(((b + h) - t) / ramp_s, 0.0, 1.0)
        rise = 0.5 - 0.5 * np.cos(np.pi * up)
        fall = 0.5 - 0.5 * np.cos(np.pi * dn)
        e = np.maximum(e, np.minimum(rise, fall))
    return e


def schedule(duration_s, share, n_bouts, ramp_s, rng, rate=100.0, start_s=0.0):
    """n_bouts on intervals inside [start_s, duration_s] whose lengths sum to share x (duration_s - start_s)
    (boundaries on knots), each at least ramp_s long, inner gaps at least ramp_s (module docstring). share 1
    gives the whole span, share 0 none. Raises ValueError when the bouts cannot fit."""
    if share <= 0:
        return []
    if share >= 1:
        return [(float(start_s), float(duration_s))]
    k0 = int(round(start_s * rate))
    N = int(round(duration_s * rate)) - k0                 # knot intervals in the span
    on = int(round(share * N))
    n = int(n_bouts)
    r = int(math.ceil(ramp_s * rate - 1e-9))
    off = N - on
    if n < 1 or on < n * r or off < (n - 1) * r + 1:
        raise ValueError(f"{n} bouts of {on / rate:.2f} s in {N / rate:.2f} s do not fit with ramps of {ramp_s} s")
    parts = r + rng.multinomial(on - n * r, rng.dirichlet(np.ones(n)))
    gaps = rng.multinomial(off - (n - 1) * r, rng.dirichlet(np.ones(n + 1)))
    gaps[1:-1] += r
    out, k = [], k0
    for i in range(n):
        k += int(gaps[i])
        out.append((k / rate, (k + int(parts[i])) / rate))
        k += int(parts[i])
    return out


def realized_share(intervals, duration_s, start_s=0.0):
    return float(sum(b - a for a, b in intervals) / (duration_s - start_s)) if duration_s > start_s else 0.0


def truth_record(operator, seed, grade, scale, params, sources, validity=None, arrays=None, **extra):
    """The truth dict (module docstring)."""
    out = {"operator": operator, "version": VERSION, "seed": None if seed is None else int(seed), "grade": grade,
           "scale": scale, "params": params, "sources": list(sources), "validity": validity,
           "arrays": arrays or {}}
    out.update(extra)
    return out


def truth_json(truth):
    """The truth without its arrays, numpy scalars as Python numbers (JSON-ready)."""
    def conv(x):
        if isinstance(x, dict):
            return {k: conv(v) for k, v in x.items() if k != "arrays"}
        if isinstance(x, (list, tuple)):
            return [conv(v) for v in x]
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, float) and not math.isfinite(x):
            return None
        return x
    return conv(truth)


def as_rng(seed):
    return seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)


def unit_axis(axis):
    u = torch.zeros(3, dtype=F64)
    u[{"x": 0, "y": 1, "z": 2}[axis]] = 1.0
    return u
