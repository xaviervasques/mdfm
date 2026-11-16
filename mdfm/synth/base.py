"""Procedural base bodies of the synthetic patient: held postures and a kinematic walk, in the shared neutral
shape (betas 0, gender 2; plan principle 9; on the open body the catalogue's one body, body.body_model), on the
100 Hz clock, floor at y = 0.

Postures (POSTURES) are local joint rotations built from elementary rotations about the rest axes of the canonical
T pose (x the subject's left, y up, z forward; the same axes on both body models), each list applied in order about the parent's fixed axes (the first
rotation acts first). Anatomical signs in this skeleton: hip flexion -x, knee flexion +x, ankle dorsiflexion
-x, trunk and neck flexion +x, lateral bending to the right +z, axial rotation to the left +y; with the arm
in the T-pose of the rest skeleton, elbow flexion is -y for the left arm and +y for the right. The postures were
placed inside the AMASS-percentile table of SP1; against the anatomical table of the open-body port (wave 1,
reports/openbody/joint_limits.json base_bodies) stand and sit lie inside, sit_arms_forward's collars protracted 25 deg
against jarm2022's girdle flexion of 20 until wave 2 moved 5 deg of the forward swing from the collars to the
shoulders (20 and 70 deg, the arms still straight forward); the walk's flat feet dorsiflexed the ankles 21.6 deg past
jarm2022's 20 deg and extended the hips 3.5 deg past its 15 until wave 5 re-solved it on the open rest (below).
tests/test_synth_pathology.py checks those readings. Postures:
- stand: upright, arms hanging 10 degrees from the trunk, elbows flexed 10 degrees; the knees off full extension so
  that each hip lies STAND_REACH of its leg's length from its ankle, the ankle under the hip and the foot flat (the
  hips flexed and the ankles dorsiflexed by what keeps them so, _stand_legs; open-body port, wave 6 last pass: the
  open body's straight rest legs held at full length lie on the singularity where the legs' inverse kinematics bends a
  knee by the square root of a shortening, and a pelvis moved over 'stand' kinked a knee in one knot).
- sit (exam segments E1 and E4 to E6 of docs/proposals/SYNTH_PATIENT_DRAFT.md, section 2): pelvis tilted
  back 20 degrees with the lower spine flexed 20 degrees so that the trunk is upright, hips flexed 70
  degrees relative to the pelvis (thighs horizontal), knees 90, feet flat; upper arms along the trunk, elbows
  flexed 90 degrees with the forearms forward, forearms turned 80 degrees (40 at the elbow and 40 at the
  wrist) so that the palms face down, as on the arms of a chair. The backward pelvic tilt keeps hip flexion
  inside the table, whose lower percentile of hip flexion lies a few degrees past 90.
- sit_arms_forward (E2): sit with the arms held forward at shoulder height, elbows straight, palms down.
  Under the table, arms forward are reached by bringing the abducted arm forward in the horizontal plane (20
  degrees at the collar, jarm2022's girdle flexion, and 70 at the shoulder; 25 and 65 before the open-body port's
  wave 2); forward flexion from the hanging arm leaves the shoulder's x
  range. In this posture the elbow hinge axis is vertical, so elbow-flexion tremor moves the wrist in the
  horizontal plane.
The chair is scene geometry: the body has no contact channel for the hands or the seat (draft section 2).

Walk (walk): two kinds since the follow-up of the open-body port's wave 5 (design section 16.8). The default, kind
'gait', is the program's gait model: gait.generate with its healthy adult parameters (gait.NORMAL: heel strike with
the toes up, a loading response rolling on the heel, the foot flat in mid stance, the heel rise and the push-off,
the stance knee's loading flexion, the stance guard of gait.py), the walk of the catalogue's level 0. Kind
'plantigrade' (plantigrade_walk) is the footstep walker below, the stand-in of the freezing operator's holds, which
stand on both heels: its double support is plantigrade, which inside jarm2022's 20 deg of dorsiflexion admits short
steps only, a freezer's shortened steps rather than a healthy walk.

Plantigrade walk (plantigrade_walk): a footstep walker for the gait-based operators (freezing of gait), built so that
contacts are exact: the gait phase advances at the cadence (scaled by an envelope that starts from and returns to
standing); each stance (60 percent of the cycle, double support 10 percent at each end) stands on its footstep, the foot
joint (the MTP head) still throughout; the foot is flat from its heel strike to the other foot's (HEEL_OFF), so that the
double support starts plantigrade (both heels down: the holds of freezing.py stand on it), then turns about the foot
joint, the toes held level on the floor, to TOE_OFF_PITCH_DEG at toe-off by a raised cosine (the pre-swing heel rise);
the swing carries the foot joint to the next footstep on a raised-cosine path lifted by CLEARANCE_M, the pitch falling
back to flat over SWING_FLAT of the swing; where a pitch so built leaves an ankle past the anatomical table's
dorsiflexion less ROCKER_MARGIN_DEG (a swing knee flexing over a level foot, a turn), the heel rises further by the
excess, smoothed, the legs solved again (up to ROCKER_ITERS passes; info 'pitch_deg' and 'rocker_passes'); a stance foot
turns with the heading about its foot joint as far as it lies flat (its pitch under TURN_FLAT_PITCH_DEG), its heading
held while its heel is up and carried back to the walk's over SWING_FLAT of the swing (info 'foot_heading_rad'). Each
ankle lies under its hip at PLACE of its cycle; the pelvis advances one stride per cycle along the heading (turns as
raised-cosine yaw-rate pulses), at the highest height from which both legs reach their ankles (REACH times the leg
length), smoothed over 0.15 s by a moving minimum then a moving mean so that it never rises above that height (the bob
of a stiff-legged gait); the legs are solved by two-bone inverse kinematics with the knees bent forward; the arms swing
in counterphase. On the open rest (wave 5, design section 16) the walk lies inside the anatomical table; its steps are
short (STEP_M 0.35 m, 0.64 m/s at the cadence), the longest the plantigrade double support admits inside jarm2022's 20
deg of dorsiflexion. No heel rocker at contact and no pelvic rotation: a test base, not a gait model. gait_contexts
reads the start, turn, straight and stop events of any walking body from its pelvis speed and heading rate.
"""
import functools
import math
from dataclasses import replace

import numpy as np
import torch

from ..body.rotations import axis_angle_to_matrix, matrix_to_axis_angle, matrix_to_rot6d, yaw_angle
from ..body.joints import N_BODY, fk
from .body import AXES, F64, NEUTRAL, RATE, Body, contact_labels, floor_to_zero, joint_limits, rest_joints

POSTURES = {
    "stand": {16: [("z", -80.0)], 17: [("z", 80.0)], 18: [("y", -10.0)], 19: [("y", 10.0)]},
    "sit": {0: [("x", -20.0)], 3: [("x", 20.0)], 1: [("x", -70.0)], 2: [("x", -70.0)], 4: [("x", 90.0)],
            5: [("x", 90.0)], 16: [("z", -80.0)], 17: [("z", 80.0)], 18: [("x", 40.0), ("y", -90.0)],
            19: [("x", 40.0), ("y", 90.0)], 20: [("x", 40.0)], 21: [("x", 40.0)]},
    "sit_arms_forward": {0: [("x", -20.0)], 3: [("x", 20.0)], 1: [("x", -70.0)], 2: [("x", -70.0)],
                         4: [("x", 90.0)], 5: [("x", 90.0)], 13: [("y", -20.0)], 14: [("y", 20.0)],
                         16: [("y", -70.0)], 17: [("y", 70.0)]},
}
STAND_REACH = 0.9995         # [module] 'stand' holds each hip this share of its leg's length (hip to knee to ankle of
#                              the rest) from its ankle (_stand_legs; open-body port, wave 6 last pass: the largest of
#                              0.0005 steps below 1 at which the standing 3.14 ladder's legs keep under its test's
#                              1 m/s2, scripts/openbody_wave6_checks.py stand_kink; the knees flexed about 3.6 deg)
CADENCE_SPM = 110.0          # [cfg] steps per minute (the cadence of the gait fixture DR-21.3)
STEP_M = 0.35                # [cfg] the plantigrade walk's step length: the longest of the 0.05 m grid at which its
#                              plantigrade stance (both feet flat from a heel strike to the other foot's) keeps the
#                              ankle's dorsiflexion inside the anatomical table less ROCKER_MARGIN_DEG (open-body
#                              port wave 5, scripts/openbody_wave5_checks.py 'walk_steps': 0.55 on the AMASS table,
#                              where the hip's extension bound it; on jarm2022's 20 deg the ankle binds first)
START_S = 1.0                # [cfg] rise of gait initiation and fall of the stop, s
STANCE = 0.6                 # [cfg] stance share of the gait cycle (double support 0.1 at each end)
PLACE = 0.25                 # [cfg] the ankle under its hip at this share of the cycle after its heel strike: the
#                              leading ankle at heel strike as far ahead of its hip as the trailing ankle is behind
#                              its own at the other foot's heel strike (0.5), so that the double support's legs
#                              stretch alike (wave 5; 0.3, the footstep under the pelvis at mid-stance, before: the
#                              leading leg alone set the pelvis's lowest height and the trailing knee bent)
HEEL_OFF = 0.5               # [cfg] the heel rises from the other foot's heel strike (the double support's start)
TOE_OFF_PITCH_DEG = 20.0     # [module] the foot's pitch about the foot joint (the MTP head) at toe-off, reached by a
#                              raised cosine from HEEL_OFF, the toes held level on the floor; back to flat over
#                              SWING_FLAT of the swing
SWING_FLAT = 0.5             # [module] share of the swing over which the toe-off pitch falls back to flat
TURN_FLAT_PITCH_DEG = 0.1    # [module] a stance foot turns with the walk's heading fully at zero pitch, not at all
#                              from this pitch on (a smoothstep between; the foot's heading, D164): at 1 deg a stance at
#                              the turn's end still turned while its heel rose, past the page's skate tolerance
ROCKER_MARGIN_DEG = 1.0      # [module] the pitch rule keeps the ankle this far inside the table's dorsiflexion
ROCKER_ITERS = 30            # the pitch rule's passes at most (each one solves the legs again)
ROCKER_TOL_DEG = 0.05        # a pass stops the rule once no ankle lies more than this past its target
ROCKER_SMOOTH_S = 0.09       # the added pitch smoothed by a moving maximum then a moving mean over this
CLEARANCE_M = 0.08           # [cfg] peak lift of the foot joint in swing (the ankle's when the foot is flat)
REACH = 0.995                # [cfg] largest hip-to-ankle distance over the leg length, pelvis height rule
ARM_SWING_DEG = 15.0         # [cfg] arm swing amplitude
LEGS = ((1, 4, 7, 10), (2, 5, 8, 11))          # hip, knee, ankle, foot joints of the left and right leg
CONTEXT = {"smooth_s": 1.0, "start_v": 0.2, "rest_v": 0.1, "rest_s": 0.5, "turn_deg_s": 30.0,
           "straight_deg_s": 10.0, "straight_v": 0.4, "straight_s": 3.0}   # [cfg] gait_contexts thresholds


def rotation(spec):
    """(3, 3) float64 of a list of (axis, degrees), applied in order about fixed parent axes."""
    M = torch.eye(3, dtype=F64)
    for ax, deg in spec:
        v = torch.zeros(3, dtype=F64)
        v[AXES[ax]] = math.radians(deg)
        M = axis_angle_to_matrix(v) @ M
    return M


@functools.lru_cache(maxsize=1)
def _stand_legs():
    """(hip, knee, ankle) deg about x of 'stand' (both legs, flexion of the hip -x, of the knee +x, the ankle's -x
    dorsiflexion): the knee flexion at which the hip lies STAND_REACH of the leg's length (hip to knee to ankle of the
    rest) from the ankle, the hip flexed so that the ankle keeps its rest offset from the hip along z (under it), the
    ankle turned back by the hip's and the knee's sum (the foot as at rest, flat); bisections on the rest skeleton."""
    rest = rest_joints()
    tr = torch.zeros(1, 3, dtype=F64)
    L = float((rest[4] - rest[1]).norm() + (rest[7] - rest[4]).norm())
    dz0 = float(rest[7, 2] - rest[1, 2])

    def joints(h, k):
        R = torch.eye(3, dtype=F64).repeat(N_BODY, 1, 1)
        for hip, knee, ank in ((1, 4, 7), (2, 5, 8)):
            R[hip], R[knee], R[ank] = rotation([("x", h)]), rotation([("x", k)]), rotation([("x", -(h + k))])
        return fk(R[None], tr, rest)[0][0]

    def hip_for(k):
        lo, hi = -k, 0.0
        for _ in range(50):
            m = 0.5 * (lo + hi)
            J = joints(m, k)
            lo, hi = (m, hi) if float(J[7, 2] - J[1, 2]) - dz0 > 0.0 else (lo, m)
        return 0.5 * (lo + hi)
    lo, hi = 0.0, 20.0
    for _ in range(50):
        k = 0.5 * (lo + hi)
        J = joints(hip_for(k), k)
        lo, hi = (k, hi) if float((J[7] - J[1]).norm()) / L > STAND_REACH else (lo, k)
    k = 0.5 * (lo + hi)
    h = hip_for(k)
    return h, k, -(h + k)


def posture_rotations(name):
    """(22, 3, 3) local rotations of a posture of POSTURES (row 0 the root orientation, heading +z); 'stand' with its
    legs off full extension (_stand_legs)."""
    R = torch.eye(3, dtype=F64).repeat(N_BODY, 1, 1)
    for j, spec in POSTURES[name].items():
        R[j] = rotation(spec)
    if name == "stand":
        h, k, a = _stand_legs()
        for hip, knee, ank in ((1, 4, 7), (2, 5, 8)):
            R[hip], R[knee], R[ank] = rotation([("x", h)]), rotation([("x", k)]), rotation([("x", a)])
    return R


def static_body(posture, duration_s, rate=RATE, betas=None, gender=NEUTRAL):
    """A body holding a posture for duration_s (round(duration_s rate) + 1 knots): pelvis x and z at 0, the
    lowest foot joint on the floor y = 0, contacts by the body-store rule."""
    T = int(round(duration_s * rate)) + 1
    R = posture_rotations(posture)
    r6 = matrix_to_rot6d(R)[None].expand(T, N_BODY, 6).clone()
    rest = rest_joints(betas, gender)
    tr = torch.zeros(T, 3, dtype=F64)
    tr[:, [0, 2]] = -rest[0, [0, 2]]
    b = Body(r6, tr, torch.zeros(T, 2, dtype=torch.bool), torch.zeros(16, dtype=F64) if betas is None else betas,
             gender, rate)
    b = floor_to_zero(b)
    return replace(b, contact=contact_labels(b))


# ---------------------------------------------------------------- walker
def _raised(t, a, d):
    """0 before a, 1 after a + d, raised cosine between."""
    if d <= 0:
        return (t >= a).astype(np.float64)
    u = np.clip((t - a) / d, 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * u)


def yaw_matrix(psi):
    """(..., 3, 3) rotations about +y by psi (...) rad (+z turns toward +x)."""
    psi = torch.as_tensor(psi, dtype=F64)
    c, s = torch.cos(psi), torch.sin(psi)
    z, o = torch.zeros_like(psi), torch.ones_like(psi)
    return torch.stack([torch.stack([c, z, s], -1), torch.stack([z, o, z], -1), torch.stack([-s, z, c], -1)], -2)


def _phase(t, env, f, rate, stop_end):
    """Gait phase (strides) from the toe-off of the left foot; after stop_end (s, or None) the phase completes
    the current step at the nominal rate up to the next double support and holds."""
    phi = STANCE + np.concatenate([[0.0], np.cumsum(0.5 * (env[1:] + env[:-1]) * f / rate)])
    if stop_end is not None:
        k = int(np.searchsorted(t, stop_end))
        if k < len(t):
            p = phi[k]
            cand = np.floor(p) + np.array([0.05, 0.55, 1.05])
            target = cand[cand >= p][0]
            more = np.minimum(p + np.arange(len(t) - k) * f / rate, target)
            phi[k:] = np.maximum(phi[k:], more)
    return phi


def _rx(deg):
    """(..., 3, 3) rotations about the x axis by deg (...) degrees (positive lowers the toes of a foot)."""
    a = torch.as_tensor(np.radians(deg), dtype=F64)
    return axis_angle_to_matrix(torch.stack([a, 0 * a, 0 * a], -1))


def _movmax_mean(x, n):
    """A moving maximum then a moving mean of width n (never below x where x is locally flat)."""
    p = (n // 2, n - 1 - n // 2)
    hi = np.lib.stride_tricks.sliding_window_view(np.pad(x, p, mode="edge"), n).max(1)
    return np.convolve(np.pad(hi, p, mode="edge"), np.ones(n) / n, mode="valid")


WALK_KINDS = ("gait", "plantigrade")


def walk(duration_s, cadence_spm=None, step_m=None, start_at_s=1.0, start_s=None, stop_at_s=None, turns=(),
         rate=RATE, betas=None, gender=NEUTRAL, kind="gait", seed=0):
    """A walk of duration_s (module docstring): standing until start_at_s, gait initiation, walking, turns (t_s,
    angle_deg, dur_s), a stop from stop_at_s (None: no stop) ending with the feet side by side. kind 'gait' (the
    default, since the follow-up of the open-body port's wave 5): the program's gait model, gait.generate with its
    healthy adult parameters (gait.NORMAL, cadence_spm and step_m overriding them when given, start_s its gait
    initiation, seed its step-to-step variability), whose heel and forefoot rockers keep a normal step inside the
    anatomical table; kind 'plantigrade': the footstep walker (plantigrade_walk), flat feet from a heel strike to the
    other foot's, on short steps (STEP_M), the stand-in of freezing.py's holds, which stand on both heels. Returns
    (Body, info): info holds the phase, heading, stance (T, 2), the foot pitch (T, 2) deg, the envelope, the kind and
    the construction parameters."""
    if kind not in WALK_KINDS:
        raise ValueError(f"kind {kind!r}, expected one of {WALK_KINDS}")
    if kind == "plantigrade":
        return plantigrade_walk(duration_s, CADENCE_SPM if cadence_spm is None else cadence_spm,
                                STEP_M if step_m is None else step_m, start_at_s, START_S if start_s is None else start_s,
                                stop_at_s, turns, rate, betas, gender)
    from . import gait as GA                       # here, not at the top: gait imports jerks, which imports this module
    over = {k: v for k, v in (("cadence_spm", cadence_spm), ("step_m", step_m), ("start_s", start_s)) if v is not None}
    prm = replace(GA.NORMAL, **over)
    b, info = GA.generate(prm, duration_s=duration_s, start_at_s=start_at_s, stop_at_s=stop_at_s, turns=turns,
                          seed=seed, rate=rate, betas=betas, gender=gender)
    info = dict(info, envelope=info["amplitude"], kind=kind, cadence_spm=float(GA.pair(prm.cadence_spm)[0]),
                step_m=float(GA.pair(prm.step_m)[0]), stop_at_s=None if stop_at_s is None else float(stop_at_s))
    return b, info


def plantigrade_walk(duration_s, cadence_spm=CADENCE_SPM, step_m=STEP_M, start_at_s=1.0, start_s=START_S,
                     stop_at_s=None, turns=(), rate=RATE, betas=None, gender=NEUTRAL):
    """A footstep walk (module docstring) of duration_s: standing until start_at_s, gait initiation over
    start_s, walking at cadence_spm with steps of step_m, a stop beginning at stop_at_s (None: no stop) that
    ends in double support with the feet side by side; turns: (t_s, angle_deg, dur_s), heading changes of
    angle_deg (positive to the left) over dur_s from t_s (in place when the body stands). Returns (Body, info):
    info holds the phase, heading, stance (T, 2), the foot pitch (T, 2) deg, the pitch rule's passes and the
    construction parameters."""
    from .ik import hinge_leg
    T = int(round(duration_s * rate)) + 1
    t = np.arange(T) / rate
    env = _raised(t, start_at_s, start_s)
    stop_end = None
    if stop_at_s is not None:
        env = env * (1.0 - _raised(t, stop_at_s, start_s))
        stop_end = stop_at_s + start_s
    f = cadence_spm / 120.0
    phi = _phase(t, env, f, rate, stop_end)
    yaw_rate = np.zeros(T)
    for (ts, ang, dur) in turns:
        u = (t - ts) / dur
        inside = (u >= 0) & (u <= 1)
        yaw_rate[inside] += math.radians(ang) / dur * (1 - np.cos(2 * np.pi * u[inside]))
    psi = np.concatenate([[0.0], np.cumsum(0.5 * (yaw_rate[1:] + yaw_rate[:-1]) / rate)])
    # pelvis travel: one stride per cycle, ramped over the first half cycle (the first step starts beside the
    # standing foot) and halted when the stop completes its last step
    g = np.clip(0.2 + 1.6 * (phi - STANCE), 0.0, 1.0)
    if stop_end is not None:
        g = np.where(t >= stop_end, 0.0, g)
    step = 2.0 * step_m * np.diff(phi, prepend=phi[0]) * g
    path = np.cumsum(np.stack([step * np.sin(psi), step * np.cos(psi)], -1), 0)      # pelvis x, z
    k_halt = T if stop_end is None else int(min(T, np.searchsorted(t, stop_end)))
    rest = rest_joints(betas, gender)
    half = [((rest[ft] - rest[a]) / 2).numpy() for _, _, a, ft in LEGS]           # ankle to foot point, flat
    off = [((rest[a] + rest[ft]) / 2 - rest[0]).numpy()[[0, 2]] for _, _, a, ft in LEGS]
    a_of_f = [rest[a] - rest[ft] for _, _, a, ft in LEGS]                         # foot joint to ankle, rest

    def place(i, k):
        """Foot joint (x, z) of leg i whose flat foot lies under the pelvis at knot k, along the heading there (the
        ankle under its hip: the canonical rest's legs are straight and vertical)."""
        c, s = math.cos(psi[k]), math.sin(psi[k])
        dx, dz = off[i][0] + half[i][0], off[i][1] + half[i][2]
        return np.array([path[k, 0] + c * dx + s * dz, path[k, 1] - s * dx + c * dz])

    feet = np.zeros((T, 2, 3))                                # the foot joint's x, z and its lift
    stance = np.zeros((T, 2), bool)
    u_leg = np.zeros((T, 2))                                  # each leg's share of its cycle (heel strike 0)
    for i, shift in ((0, 0.0), (1, 0.5)):
        lp = phi - shift                                       # this leg's cycle: heel strike at integers
        steps = {}
        first = int(np.floor(lp[0]))
        for n in range(first - 1, int(np.floor(lp[-1])) + 2):
            if n <= first:                                     # stances under way or done at the start
                steps[n] = place(i, 0)
            else:
                k = int(np.searchsorted(lp, n + PLACE))
                steps[n] = place(i, min(k, k_halt, T - 1) if k < k_halt else min(k_halt, T - 1))
        for k in range(T):
            n = int(np.floor(lp[k]))
            u = lp[k] - n
            u_leg[k, i] = u
            if u < STANCE:
                feet[k, i] = [*steps[n], 0.0]
                stance[k, i] = True
            else:
                a, b = steps[n], steps[n + 1]
                s = (u - STANCE) / (1.0 - STANCE)
                w = 0.5 - 0.5 * math.cos(math.pi * s)
                feet[k, i] = [*(a + (b - a) * w), CLEARANCE_M * math.sin(math.pi * s)]
    # the foot's own pitch (positive lowers the toes, the heel up), turned about the foot joint with the toes level:
    # the pre-swing heel rise and its return in swing, scaled by the gait envelope (module docstring)
    sw_share = (u_leg - STANCE) / (1.0 - STANCE)
    p0 = np.where(stance, TOE_OFF_PITCH_DEG * _raised(u_leg, HEEL_OFF, STANCE - HEEL_OFF),
                  TOE_OFF_PITCH_DEG * (1.0 - _raised(sw_share, 0.0, SWING_FLAT))) * env[:, None]
    G0 = yaw_matrix(torch.from_numpy(psi))
    # each foot's own heading (open-body port, wave 8 second part, D164): a stance foot turns with the walk only as far
    # as it lies flat (its pitch under TURN_FLAT_PITCH_DEG: it turns about its foot joint, over the sole point under
    # it), so that a foot whose heel is up (the heel rise, the pitch rule's lift in a turn) stands still on its toes in
    # front of that joint; a swing carries the heading back to the walk's over SWING_FLAT of the swing, as the pitch.
    # Turning with the heading on its toes, the touching sole circled the joint by 6.1 to 9.9 mm in the freezing walk's
    # turn against the page's skate tolerance of 5.5 to 6.2 mm
    dpsi = np.diff(psi, prepend=psi[0])
    sw_flat = _raised((u_leg - STANCE) / (1.0 - STANCE), 0.0, SWING_FLAT)

    def headings(pitch):
        flat = np.clip(1.0 - np.abs(pitch) / TURN_FLAT_PITCH_DEG, 0.0, 1.0)
        flat = flat * flat * (3.0 - 2.0 * flat)
        out = np.repeat(psi[:, None], 2, 1)
        for i in range(2):
            held = psi[0]
            for k in range(T):
                if stance[k, i]:
                    held = held + dpsi[k] * flat[k, i] if k > 0 and stance[k - 1, i] else out[k, i]
                    out[k, i] = held
                else:
                    out[k, i] = held + sw_flat[k, i] * (psi[k] - held)
        return out

    hips = [torch.einsum("tij,j->ti", G0, rest[h] - rest[0]) for h, _, _, _ in LEGS]
    Ls = [float((rest[a] - rest[h]).norm()) for h, _, a, _ in LEGS]               # straight-leg reach
    fwd = torch.einsum("tij,j->ti", G0, torch.tensor([0.0, 0.0, 1.0], dtype=F64))
    n_win = max(1, int(round(0.15 * rate)))
    lim = joint_limits()
    target = [lim.lo[a, 0] + ROCKER_MARGIN_DEG for _, _, a, _ in LEGS]          # dorsiflexion is -x
    cap = [lim.lo[ft, 0] + ROCKER_MARGIN_DEG for _, _, _, ft in LEGS]           # toe extension is -x

    def solve(pitch):
        Gf, A = [], []
        psi_f = headings(pitch)
        Gy = [yaw_matrix(torch.from_numpy(np.ascontiguousarray(psi_f[:, i]))) for i in range(2)]
        for i in range(2):
            F = torch.from_numpy(feet[:, i][:, [0, 2, 1]].copy())   # the foot joint: the stance foot pivots about it
            G = Gy[i] @ _rx(pitch[:, i])
            Gf.append(G)
            A.append(F + torch.einsum("tij,j->ti", G, a_of_f[i]))
        A = torch.stack(A, 1)
        P = torch.zeros(T, 3, dtype=F64)
        P[:, 0], P[:, 2] = torch.from_numpy(path[:, 0]), torch.from_numpy(path[:, 1])
        hs = []
        for i in range(2):
            d2 = ((P[:, [0, 2]] + hips[i][:, [0, 2]] - A[:, i, [0, 2]]) ** 2).sum(-1)
            hs.append(A[:, i, 1] + torch.sqrt(((REACH * Ls[i]) ** 2 - d2).clamp_min(0.0)) - hips[i][:, 1])
        raw = torch.minimum(hs[0], hs[1]).numpy()
        pad = (n_win // 2, n_win - 1 - n_win // 2)
        lo = np.lib.stride_tricks.sliding_window_view(np.pad(raw, pad, mode="edge"), n_win).min(1)
        P[:, 1] = torch.from_numpy(np.convolve(np.pad(lo, pad, mode="edge"), np.ones(n_win) / n_win, mode="valid"))
        R = posture_rotations("stand").repeat(T, 1, 1, 1)
        R[:, 0] = G0
        for i, (h, kn, a, ft) in enumerate(LEGS):
            H = P + hips[i]
            # the knee bends in the plane of the foot's own heading (D164): where a held foot's heading lags the walk's,
            # the hip turns with it, not the ankle (bent in the walk's heading, the ankle's adduction left the table in
            # the turns)
            fwd_i = torch.einsum("tij,j->ti", Gy[i], torch.tensor([0.0, 0.0, 1.0], dtype=F64))
            pole = H + torch.einsum("tij,j->ti", G0, rest[kn] - rest[h]) + 0.3 * fwd_i   # rest knee, bent forward
            hinge = torch.cross(fwd_i, A[:, i] - H, dim=-1)
            G_th, kappa = hinge_leg(H, A[:, i], pole, rest[kn] - rest[h], rest[a] - rest[kn], G0, hinge=hinge)
            Rk = axis_angle_to_matrix(torch.stack([kappa, 0 * kappa, 0 * kappa], -1))
            R[:, h] = G0.transpose(-1, -2) @ G_th
            R[:, kn] = Rk
            R[:, a] = (G_th @ Rk).transpose(-1, -2) @ Gf[i]    # the foot along the heading, pitched
            R[:, ft] = _rx(-pitch[:, i])                        # the toes kept level
        return R, P, psi_f

    # the pitch rule: where the ankle would pass the table's dorsiflexion less ROCKER_MARGIN_DEG, the heel rises
    # further by the excess (solved again until no ankle lies past its target), smoothed, within the toes' room
    extra = np.zeros((T, 2))
    passes = []
    for it in range(ROCKER_ITERS):
        pitch = p0 + extra
        R, P, psi_f = solve(pitch)
        ax = np.degrees(matrix_to_axis_angle(R[:, [LEGS[0][2], LEGS[1][2]]]).numpy()[..., 0])
        excess = np.stack([target[0] - ax[:, 0], target[1] - ax[:, 1]], -1)
        passes.append(float(excess.max()))
        if excess.max() <= ROCKER_TOL_DEG:
            break
        add = np.maximum(extra + excess, 0.0)
        n = max(1, int(round(ROCKER_SMOOTH_S * rate)))
        extra = np.stack([np.minimum(_movmax_mean(add[:, i], n), cap[i] * -1.0 - p0[:, i]) for i in range(2)], -1)
        extra = np.maximum(extra, 0.0)
    for sj, sgn, shift in ((16, 1.0, 0.0), (17, -1.0, 0.5)):
        arm = -ARM_SWING_DEG * np.cos(2 * np.pi * (phi - shift)) * env
        swing = axis_angle_to_matrix(torch.from_numpy(np.stack([-np.radians(arm), 0 * arm, 0 * arm], -1)))
        R[:, sj] = swing @ rotation([("z", -80.0 * sgn)])
    b = Body(matrix_to_rot6d(R), P - rest[0], torch.zeros(T, 2, dtype=torch.bool),
             torch.zeros(16, dtype=F64) if betas is None else betas, gender, rate)
    b = replace(b, contact=contact_labels(b))
    info = {"t": t, "envelope": env, "phase": phi, "heading_rad": psi, "stance": stance, "pitch_deg": pitch,
            "rocker_passes": passes, "turns": [(float(a), float(g), float(d)) for a, g, d in turns],
            "foot_heading_rad": psi_f,
            "start_at_s": float(start_at_s), "stop_at_s": None if stop_at_s is None else float(stop_at_s),
            "cadence_spm": float(cadence_spm), "step_m": float(step_m), "kind": "plantigrade"}
    return b, info


def _runs(mask):
    """[(start, stop)) index runs of a boolean array."""
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.diff(m.astype(np.int8))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def gait_kinematics(body, smooth_s=CONTEXT["smooth_s"]):
    """(pelvis horizontal speed (T,) m/s, heading rate (T,) deg/s), both averaged over a centred window of
    smooth_s (the gait-cycle sway removed), from the pelvis path and the root heading."""
    J = body.joints()
    p = J[:, 0, [0, 2]].numpy()
    psi = np.unwrap(yaw_angle(body.local_R()[:, 0]).numpy())
    n = max(1, int(round(smooth_s * body.rate)))
    k = np.ones(n) / n

    def smooth(x):
        pad = np.pad(x, (n // 2, n - 1 - n // 2), mode="edge")
        return np.convolve(pad, k, mode="valid")
    v = np.gradient(p, 1.0 / body.rate, axis=0)
    speed = smooth(np.hypot(v[:, 0], v[:, 1]))
    rate = smooth(np.degrees(np.gradient(psi, 1.0 / body.rate)))
    return speed, rate


def gait_contexts(body, cfg=None):
    """Events of a walking body [cfg CONTEXT]: {'start': [(t0, t1)], 'turn': [...], 'straight': [...], 'stop':
    [...]} in s. start: a rise of the smoothed pelvis speed above start_v after at least rest_s below rest_v
    (from the last rest knot to the crossing); turn: runs of |heading rate| above turn_deg_s; straight: runs of
    at least straight_s with speed above straight_v and |heading rate| below straight_deg_s; stop: a fall below
    rest_v after walking (from the crossing to the end of the rest run)."""
    c = dict(CONTEXT, **(cfg or {}))
    speed, rate = gait_kinematics(body, c["smooth_s"])
    r = body.rate
    rest = speed < c["rest_v"]
    out = {"start": [], "turn": [], "straight": [], "stop": []}
    for a, b in _runs(rest):
        if (b - a) / r >= c["rest_s"]:
            after = np.flatnonzero(speed[b:] > c["start_v"])
            if b < len(speed) and len(after):
                out["start"].append((float((b - 1) / r), float((b + after[0]) / r)))
            if a > 0 and speed[:a].max() > c["start_v"]:
                out["stop"].append((float(a / r), float((b - 1) / r)))
    for a, b in _runs(np.abs(rate) > c["turn_deg_s"]):
        out["turn"].append((float(a / r), float((b - 1) / r)))
    for a, b in _runs((speed > c["straight_v"]) & (np.abs(rate) < c["straight_deg_s"])):
        if (b - a) / r >= c["straight_s"]:
            out["straight"].append((float(a / r), float((b - 1) / r)))
    return out
