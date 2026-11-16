"""Bradykinesia with decrement on the repeated body tasks of MDS-UPDRS 3.6 to 3.8
(docs/proposals/SYNTH_PATIENT_DRAFT.md, section 5).

Inverse recipe (draft section 5; DR-12.1 to DR-12.5 of docs/rules/DESCRIPTOR_RULES_DRAFT.md): n_cycles cycles
(10, the scale's count) of one task movement about one axis, generated from a level-0 cycle shape with a
per-cycle amplitude sequence (a constant run, then a linear fall from the onset cycle at a fixed slope), a
speed gain, and pauses inserted between cycles (hesitations shorter than one cycle period, arrests longer).

Tasks (TASKS; the base posture each expects in parentheses, base.POSTURES):
- pronation_supination (3.6; sit_arms_forward, the arm extended in front, palm down): axial rotation about the
  forearm's long axis (the rest bone from the elbow to the wrist), shared between the forearm segment (elbow
  joint 18 or 19, which carries the wrist IMU of the factory, factory/imu_render.py) and the hand (wrist joint 20
  or 21) in proportion to the room the SP1 table leaves each, toward supination (body.SUPINATION_SIGN: -1 about
  the distal bone axis on the left, +1 on the right, the mirror image); the task angle is the hand's rotation
  relative to the upper arm, their sum.
- toe_tapping (3.7; sit): ankle dorsiflexion (-x about the shank's axis, a hinge in the parent frame), the
  toes leaving the floor and returning to it with the heel on the ground (foot 'plant', the default since
  version sp4-v3): the foot rolls on its sole (contact.plant_ankle), first about the ball until the heel touches,
  then about the heel, and the leg above it follows with the hip fixed, so that the ankle's flexion is the task
  angle exactly and the lowest sole point stays on the floor; foot 'free' is the version-2 task (the ankle joint
  fixed, the heel of the SMPL+H foot then sinks below the floor as the toes rise). The truth of both leg tasks
  carries params 'foot_pose' (heel and toe lift, hip and knee turned from the base) and the validity 'continuity'
  check (contact.step_report).
- finger_tapping (3.4) and hand_movements (3.5): the hand tasks of handtasks.py (task SP5, the SMPL+H hand joints,
  a cited level 0 and the 'slowing' criterion), reached through bradykinesia() by their task name.
- leg_agility (3.8; sit): hip flexion (-x about the pelvis's axis), the thigh and the foot raised and lowered
  with the knee held (the foot raised by design, foot 'raise'; the truth checks that it never goes below the
  floor and that the other foot stays planted, contact.sole_report).

Arms of the leg agility (plan phase E, the gallery on the open body): on the 'sit' the forearms lie forward as on the
arms of a chair and the hands over the thighs, 46 mm above the open body's thigh skin (the chair's armrests end 10 cm in
front of the elbow, viewer.chair_mesh), so the thigh raised by the level 0's 44 deg of hip flexion ran 20.9 mm into the
left hand on the mesh (the page's rigid thigh read the left thumb tip 22.2 mm inside). The exam's ladder (exam.py)
seats the task on arms_out: both shoulders turned outward about the humerus by LEG_AGILITY_ARMS_OUT_DEG, the forearms
angled out over the chair's arms and the hands beside the thighs, as the test is done seated in a chair with arms; the
body given to bradykinesia() is that base, so the task's rows alone differ from it.

Cycle shape: one cycle of period P leaves the rest posture and returns to it, o(t) = s A_c (1 - cos(2 pi t /
P)) / 2 with s the direction; the rest posture is the constant extreme, A_c = max |o| within the cycle, the
mean speed 2 A_c / P (DR-12.1). Periods are even numbers of knots, so that every cycle starts and ends on a
rest knot and peaks on a knot. Level 0: amplitude A0 = LEVEL0_SHARE (0.8) of the room the SP1 joint-limit
table leaves from the base posture in the movement's direction (body.room_deg, summed over the task's joints)
and rate level0_hz (1.5 to 2 Hz): module values, since normal amplitudes and rates of these tasks were not
sourced (draft section 11); slowing and amplitude levels are therefore gains on the level-0 performance of the
same base body (draft section 5).

Grade map (grades.brady_parameters, draft section 5): single-criterion ladders, interruptions (0, 2, 4, 7
hesitations; grade 3 also as one arrest), decrement (onset cycles 9, 5, 2 with a slope of 10 percent of A0
per cycle by default), slowing (a speed gain set by the neurologist after SP7, refused until then). The
reading of near the end, midway and after the first as onset cycles 9, 5 and 2 is the draft's.
Hesitation and arrest durations: half a level-0 period and two periods by default (module values). Where pauses
go: distinct cycle boundaries drawn from the seed. A grade of 0 returns a copy of the input.

Hesitation floor (wave SH1d, the review of the English page, M14): at the cited normal rates of finger tapping and
hand movements (3.67 and 3.36 Hz) half a period is 0.14 and 0.15 s, about four frames at 30 frames per second, and
a closure with a hesitation read as an irregular rhythm rather than a countable interruption. hesitation_length_s
gives max(HESITATION_PERIODS times the level-0 period, HESITATION_MIN_S); a hesitation must then be shorter than an
arrest (ARREST_PERIODS periods), no longer shorter than one cycle. handtasks.hand_task takes the floor by default;
bradykinesia() takes it when hesitation_floor_s is given (the exam set gives HESITATION_MIN_S since exam_v7) and
keeps half a period otherwise, bit for bit, for the SP3 probe stimuli (stimuli.py builds the 3.6 interruptions).
Arrest floor (wave SH1d, the review of lane exam): with the hesitations at 0.3 s, the finger-tapping arrest of two
level-0 periods (0.545 s at 3.67 Hz) was 1.8 hesitations long, and the longer arrest of MDS-UPDRS 3.4 to 3.8 grade 3
barely stood apart from a hesitation. arrest_length_s gives max(ARREST_PERIODS times the level-0 period,
ARREST_MIN_S), ARREST_MIN_S being ARREST_HESITATIONS hesitation floors; handtasks.hand_task takes it by default,
bradykinesia() when arrest_floor_s is given (the exam set gives ARREST_MIN_S since exam_v7: at the 3.6 to 3.8 rates,
2 and 1.5 Hz, two periods already exceed it and those arrests are unchanged).
"""
from dataclasses import replace

import numpy as np
import torch

from . import contact as CT
from . import grades as G
from .body import SUPINATION_SIGN, axis_rotation, bone_axis, compose, contact_labels, room_deg, validity
from .common import as_rng, truth_record

TASKS = {
    "pronation_supination": {"chain": {"left": (18, 20), "right": (19, 21)}, "frames": ("child", "parent"),
                             "axis": "forearm", "sign": SUPINATION_SIGN, "item": "3.6", "level0_hz": 2.0,
                             "posture": "sit_arms_forward", "leg": False},
    "toe_tapping": {"chain": {"left": (7,), "right": (8,)}, "frames": ("parent",), "axis": "x", "sign": -1.0,
                    "item": "3.7", "level0_hz": 2.0, "posture": "sit", "leg": True},
    "leg_agility": {"chain": {"left": (1,), "right": (2,)}, "frames": ("parent",), "axis": "x", "sign": -1.0,
                    "item": "3.8", "level0_hz": 1.5, "posture": "sit", "leg": True},
}
FOREARM = {"left": (18, 20), "right": (19, 21)}
FOOT_DEFAULT = {"pronation_supination": None, "toe_tapping": "plant", "leg_agility": "raise"}
FOOT_MODES = {"toe_tapping": ("plant", "free"), "leg_agility": ("raise", "free")}
LEVEL0_SHARE = 0.8          # [module] level-0 amplitude, share of the room to the table limit
HESITATION_PERIODS = 0.5    # [module] hesitation length in level-0 periods (above the floor below)
HESITATION_MIN_S = 0.3      # [module] the least hesitation (module docstring: 9 frames at 30 frames per second;
#                             the review's floor), shorter than an arrest at every level-0 rate of TASKS and handtasks
ARREST_PERIODS = 2.0        # [module] arrest length in level-0 periods (longer than one period)
ARREST_HESITATIONS = 3.0    # [module] the least arrest, in hesitation floors (the review of lane exam: at least 3)
ARREST_MIN_S = round(ARREST_HESITATIONS * HESITATION_MIN_S, 9)   # the least arrest (module docstring, Arrest floor)
LEG_AGILITY_ARMS_OUT_DEG = 10.0   # [module] the shoulders' outward turn of arms_out (module docstring, Arms of the leg
#   agility): the least on a 5 deg grid at which the left hand and forearm keep reach.CLEAR_M (the module's distance of
#   a hand from the thighs) from the raised thigh on the mesh over the level 0's lift (postures.arm_part_gap;
#   scripts/openbody_phaseE_checks.py seated_hands)


def arms_out(body, deg=LEG_AGILITY_ARMS_OUT_DEG):
    """A copy of a seated body with both shoulders (16, 17) turned outward about the humerus (the rest bone from the
    shoulder to the elbow, the child frame) by deg, the forearms swung out and the hands carried beside the thighs
    (module docstring, Arms of the leg agility); every other row bit for bit."""
    R = body.local_R().clone()
    for j, s in ((16, 1.0), (17, -1.0)):           # outward: the opposite of postures.arm_internal_term's sign
        ax = bone_axis(j, j + 2, body.betas, body.gender)
        R[:, j] = compose(R[:, j], axis_rotation(ax, torch.tensor(-s * np.radians(float(deg)), dtype=torch.float64)),
                          "child")
    return body.with_rotations(R, [16, 17])


def hesitation_length_s(period0_s, floor_s=HESITATION_MIN_S):
    """s: a hesitation's length for a level-0 period of period0_s (module docstring, Hesitation floor):
    HESITATION_PERIODS periods, at least floor_s (None or 0: no floor, the version before wave SH1d)."""
    return max(HESITATION_PERIODS * float(period0_s), float(floor_s or 0.0))


def arrest_length_s(period0_s, floor_s=ARREST_MIN_S):
    """s: an arrest's length for a level-0 period of period0_s (module docstring, Arrest floor): ARREST_PERIODS
    periods, at least floor_s (None or 0: no floor, the version before wave SH1d)."""
    return max(ARREST_PERIODS * float(period0_s), float(floor_s or 0.0))


def task_sign(task, side):
    """The task direction's sign about its axis on that side (a per-side table for the forearm twist)."""
    s = TASKS[task]["sign"]
    return float(s[side] if isinstance(s, dict) else s)


def task_axis(task, side, body=None):
    """The task's rotation axis: 'x', or the unit forearm axis (3,) of the side (body's shape, default neutral)."""
    tk = TASKS[task]
    if tk["axis"] != "forearm":
        return tk["axis"]
    a, b = FOREARM[side]
    return bone_axis(a, b, None if body is None else body.betas, 2 if body is None else body.gender)


def task_rooms(body, task, side, k=0):
    """Degrees of offset each joint of the task's chain can take from knot k of the base in the task's
    direction before a component leaves the table (body.room_deg)."""
    tk = TASKS[task]
    R = body.local_R()[k]
    ax = task_axis(task, side, body)
    return [room_deg(R[j], j, ax, task_sign(task, side), f) for j, f in zip(tk["chain"][side], tk["frames"])]


def task_angle(body, task, side, k0):
    """(T,) deg: the task movement read from a body, relative to knot k0: the rotation of the chain's product
    of local rotations M(t) (the hand relative to the upper arm, or the hinge joint) relative to M(k0), as the
    component of its axis-angle vector along the task axis (in the frame the first joint's offset acts in),
    signed so that the task's direction is positive."""
    from ..body.rotations import matrix_to_axis_angle
    tk = TASKS[task]
    chain = tk["chain"][side]
    R = body.local_R()
    M = R[:, chain[0]]
    for j in chain[1:]:
        M = M @ R[:, j]
    D = M @ M[k0].transpose(-1, -2)
    ax = task_axis(task, side, body)
    u = torch.zeros(3, dtype=torch.float64)
    if isinstance(ax, str):
        u[{"x": 0, "y": 1, "z": 2}[ax]] = 1.0
    else:
        u = torch.as_tensor(ax, dtype=torch.float64)
    w = u if tk["frames"][0] == "parent" else R[k0, chain[0]] @ u
    return (task_sign(task, side) * torch.rad2deg(matrix_to_axis_angle(D)) @ w).numpy()


def amplitudes(n_cycles, a0, onset=None, slope=0.0):
    """Per-cycle amplitudes (deg): a0 before the 1-based onset cycle, then a0 (1 - slope (c - onset + 1))."""
    c = np.arange(1, n_cycles + 1, dtype=np.float64)
    if onset is None or slope == 0:
        return np.full(n_cycles, float(a0))
    out = a0 * (1.0 - slope * np.clip(c - onset + 1, 0, None))
    if (out <= 0).any():
        raise ValueError("the decrement reaches zero amplitude within the cycles")
    return out


def bradykinesia(body, task="pronation_supination", side="left", grade=None, criterion="decrement",
                 n_cycles=G.N_CYCLES, amplitude0_deg=None, level0_hz=None, speed_gain=1.0, decrement_onset=None,
                 decrement_slope=G.BRADY_DECREMENT_SLOPE, n_hesitations=0, n_arrests=0, hesitation_s=None,
                 arrest_s=None, start_s=0.5, seed=0, check=True, foot=None, hesitation_floor_s=None,
                 arrest_floor_s=None):
    """(Body, truth) of a repeated task with bradykinesia on one side (module docstring).

    grade with criterion ('interruptions', 'arrest', 'decrement') sets the pause counts or the decrement
    onset by grades.brady_parameters; otherwise give decrement_onset, n_hesitations, n_arrests directly.
    speed_gain (1 the level-0 rate) scales the cycle rate. The task starts at start_s and must end inside
    the body. seed: int or numpy Generator (pause positions). foot (leg tasks): 'plant' (toe tapping, default),
    'raise' (leg agility, default) or 'free' (the version-2 task). hesitation_floor_s: the least hesitation (s,
    module docstring, Hesitation floor; None: half a level-0 period, the version before, for 3.6 to 3.8; a hand
    task then takes handtasks' default floor); arrest_floor_s likewise for the arrest (Arrest floor). The hand
    tasks of MDS-UPDRS 3.4 and 3.5 ('finger_tapping', 'hand_movements', task SP5) go to handtasks.hand_task (its
    cited level 0 and the 'slowing' criterion); amplitude0_deg is then in that task's unit."""
    from .handtasks import TASKS as HAND_TASKS
    if task in HAND_TASKS:
        if foot is not None:
            raise ValueError(f"foot {foot!r} for the hand task {task}")
        from .handtasks import hand_task
        return hand_task(body, task, side, grade, criterion, n_cycles, amplitude0_deg, level0_hz, speed_gain,
                         decrement_onset, decrement_slope, n_hesitations, n_arrests, hesitation_s, arrest_s, start_s,
                         seed, check, **({} if hesitation_floor_s is None else
                                         {"hesitation_floor_s": hesitation_floor_s}),
                         **({} if arrest_floor_s is None else {"arrest_floor_s": arrest_floor_s}))
    if task not in TASKS:
        raise ValueError(f"task {task!r}, expected one of {sorted(TASKS)}")
    tk = TASKS[task]
    foot = FOOT_DEFAULT[task] if foot is None else foot
    if foot != FOOT_DEFAULT[task] and foot not in FOOT_MODES.get(task, ()):
        raise ValueError(f"foot {foot!r} for {task}, expected one of {FOOT_MODES.get(task, (None,))}")
    seed_val = seed if isinstance(seed, (int, np.integer)) else None
    if grade is not None:
        p = G.brady_parameters(grade, criterion, decrement_slope)
        n_hesitations, n_arrests = p["n_hesitations"], p["n_arrests"]
        decrement_onset, decrement_slope = p["decrement_onset"], p["decrement_slope"]
        if int(grade) == 0 and speed_gain == 1.0:
            params = {"task": task, "item": tk["item"], "side": side, "criterion": criterion, "n_cycles": 0}
            return body.copy(), truth_record("bradykinesia", seed_val, 0, f"MDS-UPDRS {tk['item']}", params,
                                             ("mds_updrs",), None)
    rng = as_rng(seed)
    chain = tk["chain"][side]
    k_start = int(round(start_s * body.rate))
    rooms = task_rooms(body, task, side, k_start)
    room = float(sum(rooms))
    a0 = LEVEL0_SHARE * room if amplitude0_deg is None else float(amplitude0_deg)
    if a0 > room + 1e-9:
        raise ValueError(f"amplitude {a0:.1f} deg exceeds the {room:.1f} deg left to the joint-limit table")
    split = [r / room for r in rooms] if room > 0 else [1.0 / len(chain)] * len(chain)
    f0 = tk["level0_hz"] if level0_hz is None else float(level0_hz)
    rate = body.rate
    period_k = 2 * int(round(rate / (f0 * speed_gain) / 2))     # even: the cycle's peak on a knot
    if period_k < 4:
        raise ValueError("cycle period shorter than 4 knots")
    P0_k = int(round(rate / f0))
    if hesitation_s is None:
        hes_k = int(round(HESITATION_PERIODS * P0_k if hesitation_floor_s is None else
                          rate * hesitation_length_s(P0_k / rate, hesitation_floor_s)))
    else:
        hes_k = int(round(hesitation_s * rate))
    if arrest_s is None:
        arr_k = int(round(ARREST_PERIODS * P0_k if arrest_floor_s is None else
                          rate * arrest_length_s(P0_k / rate, arrest_floor_s)))
    else:
        arr_k = int(round(arrest_s * rate))
    if n_hesitations and (hes_k >= arr_k or (hesitation_floor_s is None and hes_k >= period_k)):
        raise ValueError("a hesitation must be shorter than an arrest (and, without a floor, than one cycle period)")
    if arr_k <= period_k and n_arrests:
        raise ValueError("an arrest must be longer than one cycle period")
    A = amplitudes(n_cycles, a0, decrement_onset, decrement_slope)
    boundaries = np.arange(1, n_cycles)                    # pause after cycle b (1-based), between b and b + 1
    n_p = n_hesitations + n_arrests
    if n_p > len(boundaries):
        raise ValueError(f"{n_p} pauses for {len(boundaries)} cycle boundaries")
    where = rng.choice(boundaries, size=n_p, replace=False) if n_p else np.zeros(0, np.int64)
    kinds = rng.permutation(["hesitation"] * n_hesitations + ["arrest"] * n_arrests) if n_p else []
    pause_after = {int(b): (str(kd), hes_k if kd == "hesitation" else arr_k) for b, kd in zip(where, kinds)}
    k0 = int(round(start_s * rate))
    o = np.zeros(body.T)
    cycles, pauses, k = [], [], k0
    for c in range(1, n_cycles + 1):
        u = np.arange(period_k + 1) / period_k
        seg = task_sign(task, side) * A[c - 1] * 0.5 * (1.0 - np.cos(2 * np.pi * u))
        if k + period_k >= body.T:
            raise ValueError("the task does not end inside the body: lengthen it or start earlier")
        o[k:k + period_k + 1] = np.radians(seg)             # rad, the whole movement before the split
        cycles.append({"cycle": c, "start_s": k / rate, "period_s": period_k / rate, "amplitude_deg": float(A[c - 1]),
                       "speed_deg_s": float(2 * A[c - 1] * rate / period_k)})
        k += period_k
        if c in pause_after:
            kd, n = pause_after[c]
            pauses.append({"kind": kd, "after_cycle": c, "start_s": k / rate, "duration_s": n / rate})
            k += n
    R = body.local_R()
    Rall = R.clone()
    ax = task_axis(task, side, body)
    for j, f, sh in zip(chain, tk["frames"], split):
        Rall[:, j] = compose(R[:, j], axis_rotation(ax, torch.from_numpy(o * sh)), f)
    out = body.with_rotations(Rall, list(chain))
    pinfo = None
    if foot == "plant":                                 # toe tapping: the heel stays on the floor (docstring)
        knots = CT.stance_knots(body, side) & (o != 0)
        Rp, ok, pinfo = CT.plant_ankle(body, Rall.numpy(), side, knots)
        if not ok.all():
            raise ValueError(f"toe tapping: the foot cannot stay on the floor at {int((~ok).sum())} knots")
        out = body.with_rotations(torch.from_numpy(Rp), list(CT.LEG[side][:3]))
    feet = (0, 1)
    if tk["leg"]:                                       # the task foot leaves the floor: relabel, check the other
        out = replace(out, contact=contact_labels(out))
        feet = (1,) if side == "left" else (0,)
    from .measure import decrement_dec
    params = {"task": task, "item": tk["item"], "side": side, "criterion": criterion if grade is not None else None,
              "chain": [int(j) for j in chain], "frames": list(tk["frames"]),
              "axis": tk["axis"] if isinstance(ax, str) else [float(x) for x in ax], "direction": task_sign(task, side),
              "n_cycles": int(n_cycles), "amplitude0_deg": float(a0), "room_deg": room,
              "rooms_deg": [float(r) for r in rooms], "split": [float(x) for x in split],
              "level0_hz": float(f0), "speed_gain": float(speed_gain), "period_s": period_k / rate,
              "decrement_onset": decrement_onset, "decrement_slope": float(decrement_slope),
              "dec_dr12_3_pct": decrement_dec(A), "n_hesitations": int(n_hesitations), "n_arrests": int(n_arrests),
              "hesitation_floor_s": None if hesitation_floor_s is None else float(hesitation_floor_s),
              "hesitation_s": hes_k / rate,
              "arrest_floor_s": None if arrest_floor_s is None else float(arrest_floor_s), "arrest_s": arr_k / rate,
              "cycles": cycles, "pauses": sorted(pauses, key=lambda p: p["start_s"]), "start_s": k0 / rate,
              "end_s": k / rate, "grade_any_criterion": G.brady_grade_of(n_hesitations, n_arrests, decrement_onset),
              "expected_posture": tk["posture"], "foot": foot, "plant": pinfo}
    if tk["leg"]:                                       # what planting costs (contact.foot_pose_report)
        params["foot_pose"] = CT.foot_pose_report(body, out, side)
    val = validity(body, out, feet=feet) if check else None
    if check and tk["leg"]:
        val["contact"] = CT.sole_report(body, out, side, planted=foot == "plant")
        val["continuity"] = CT.step_report(body, out, side)
    return out, truth_record("bradykinesia", seed_val, grade, f"MDS-UPDRS {tk['item']}", params,
                             ("mds_updrs", "table", "module"), val, arrays={"offset_rad": o})
