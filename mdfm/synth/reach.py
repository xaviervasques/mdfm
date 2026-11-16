"""Reach to the nose, the base motion of exam segment E3 (docs/proposals/SYNTH_PATIENT_DRAFT.md, section 2: from the
arm outstretched to the nose region and back, slowly, at least three reaches per side; MDS-UPDRS 3.16, SARA item 6,
TETRAS item 4c, ICARS finger-to-nose items): a procedural SMPL+H body with hands (hands.HandBody), seated, on the
100 Hz clock, in the shared neutral shape, joint floor at y = 0.

Public API (shared with the other symptom modules; keep it small):
- reach(side, n_reaches, ...) -> (HandBody, info): the base motion; info['phases'] is the timeline (kind, t0_s,
  t1_s, fingertip start and end in m), info['targets'] the nose and the examiner's-finger targets;
- fingertip(body, side) -> (T, 3) m: the index fingertip vertex of the skinned mesh (hands.tip_vertices);
- nose_point(body) -> (T, 3) m: the nose tip vertex of the skinned mesh (nose_vertex);
- speed_runs(points, rate, t0_s, t1_s) and endpoint_error_cm(...): the read-back of submovements and of the
  endpoint error of a reach (tests/test_synth_tremors.py);
- examiner_boxes(side), box_sdf(points, boxes) and scene_mesh('chair_target', body, floor_y): the examiner's finger
  at the target, for the viewer's scene and for the room of a tremor on the reach;
- decomposed_pace(side): the slower pace of a decomposed reach on the level 0's clock;
- path_between(side, p0, p1, qa, qb, via), miss_posture(side, miss), correction_via(side, error),
  fingertip_path(p0, p1, s, via) and movement_axis(side): the dysmetric movement's postures and the curve of its
  correction (below);
- joint_step_excess(body): each joint's step between successive knots beyond the median step of its neighbourhood
  (STEP_NEIGHBOURS knots on either side), the read-back of a snap (bounded by STEP_EXCESS_MAX in the tests);
- the face touch (below): face_vertex, face_point, touch_path(side, target, posture), touch_rows(side, target,
  posture, s), touch_rise_s(path), touch_joint_peaks(path), hand_blend(side, w), hand_share(s) and
  clearance(body, side, knots), the paths of the touching tics
  of jerks.py (shared change SH15 of docs/proposals/SYNTH_SHARED_CHANGES_DRAFT.md).

Scene (module docstring of base.py for the 'sit' posture): seated in a chair with arms, the forearms on its arms;
the reaching arm rises to the examiner's finger, the other arm stays on the chair arm. Timeline: REST_S seated, the
raise to the target posture over RAISE_S (a minimum-jerk blend of the arm's joint components, as
motions.outstretched), HOLD_OUT_S held at the target, then n_reaches cycles of [target to nose, hold at the nose,
nose to target, hold at the target], then the lowering back to the chair arm over RAISE_S and REST_S seated.

Geometry (published set-up, module readings where none is published; SOURCES of this module):
- the examiner's finger at TARGET_SHARE (90 percent) of the subject's reach, as SARA item 6 places it ('in front of
  the proband at about 90 percent of the proband's reach', SOURCES['sara2006']) and as Rodrigues et al. 2017 place
  the target of their finger-to-nose task (90 percent of arm length, at nose height, SOURCES['rodrigues2017']); the
  reach is the distance from the shoulder joint to the fingertip of the straight arm with the pointing hand; the
  target at the height of the nose, halfway between the midline and the shoulder (TARGET_X_SHARE, module);
- the nose target: the nose tip vertex (nose_vertex(), the head vertex in front on the midline) plus TOUCH_GAP_M
  toward the examiner's finger, so that the fingertip touches the nose without entering it (module).
Timing: the cycle time of SOURCES['ohberg2019'] (20 healthy adults, finger-to-nose at self-selected speed, 2.35 s
per cycle with the dominant arm and 2.31 s with the non-dominant; the right arm is taken as dominant), split into
two movements and two holds of HOLD_S (module: no hold duration was sourced). Each movement follows the
minimum-jerk profile of Flash and Hogan 1985 (SOURCES['flash1985']): the fingertip on the straight line between its
end points with s(t) = 10 u^3 - 15 u^4 + 6 u^5.

Arm postures (inverse kinematics, solved once per side and cached): the axis-angle components (deg) of the collar,
shoulder, elbow and wrist are the variables of a bounded least-squares problem (scipy L-BFGS-B) whose bounds are the
SP1 joint-limit table less LIMIT_MARGIN_DEG, so that every posture of the reach, and every blend of two of them
(componentwise), lies inside the table: the fingertip on its target (the fingertip as a point rigid on the index
distal joint, within a millimetre or two of the skinned vertex, reported), the wrist near neutral, the elbow without
its valgus component, the elbow below the shoulder, and continuity with the previous posture. The path between the
two targets is solved at N_WAY points along the straight fingertip line by continuation from the target posture,
the index turned toward the face as it nears the nose (POINT_AT_NOSE: a term on the index axis, so that the finger
points at the nose rather than standing upright in front of the lips), then solved again all together with a
term on the second differences of every component along the samples (_smooth_path, PATH_SMOOTH_WEIGHT; open-body
port, wave 2: sample by sample, the free components turned sharply where another reached its bound, and the arm's
speed changed by 90 to 170 deg/s in one knot there, scripts/openbody_reach_kink.py), and the joint components are
interpolated along it (cubic spline in the path parameter, clipped to the bounds); the movement back to the target
runs the same postures in reverse, so that it ends on the target posture.
The pointing hand (POINT_POSE): the index straight, the middle, ring and little fingers and the thumb as the fist of
handtasks.FIST (inside the hand limits of hands.py). The other hand stays at rest.

Movement variants for the cerebellar signs (tremors.py), on the same timeline as level 0 so that a clip and its
level 0 share their clock:
- endpoint_errors {cycle: (3,) m}: the movement to the examiner's finger of that cycle (0-based) ends at target plus
  the error (dysmetria: past or short of the target) over PRIMARY_SHARE of the movement time, then a corrective
  movement to the target over the rest; the movement time of a dysmetric movement is the level 0's movement and the
  hold at the target that follows it (the phase's 'movement_end_s', the end of that hold; the hold's item marked
  'taken_by_correction'), so that the longer path runs at a speed nearer the level 0's: the primary over
  PRIMARY_SHARE of it (to 'primary_end_s'), the correction over the rest ('correction_s'). Both run between given
  postures (path_between): the primary on the straight fingertip line from the nose posture to the miss posture
  (miss_posture: the arm at the end point solved from the target posture, with it as the reference), the correction
  from the miss posture to the level 0's target posture, which it reaches exactly with no joint speed at the end of
  the hold, so that the movement back to the nose follows with no step and the clip joins the level 0's rows there.
  The correction of a miss off to the side runs on a quadratic fingertip curve (correction_via, recorded as the
  phase's 'correction_via_m'): it draws the overshoot back along the movement first and closes the side offset last,
  so that the index passes beside the examiner's fingertip and not under it (on the straight line both offsets
  shrank together and the index grazed the underside of the examiner's finger); the correction of an undershoot
  stays on the line. (The first build ran each line by continuation from its start posture: the correction ended on
  an arm posture of its own, with the fingertip on the target and the elbow elsewhere, and the arm snapped to the
  target posture in one knot at the hold; the primary, solved without the pointing term, turned the pointing hand
  upright in front of the face in a flick at its start; the second build ran the movement within the level 0's
  movement time, the hold kept, and the grade 3 overshoot read as a flung arm.) The primary covers the longer path
  to the miss in PRIMARY_SHARE of that time, so its fingertip still runs faster than the level 0's: a known property
  of the shared clock, no source of this module gives the speed of a dysmetric movement (tests/test_synth_tremors.py
  bounds its peak at twice the level 0's peak on its own movement to the target of the same cycle, and each arm
  joint's turn rate at 1.5 times the same joint's there);
- decomposition n (2 or more): every movement is made joint group by joint group (the shoulder and collar, the
  elbow and wrist; for 3 phases the elbow and wrist in halves around the shoulder), each phase a minimum-jerk blend
  of its group's components lasting in proportion to its fingertip travel to the power PHASE_TIME_EXP,
  PHASE_PAUSE_S apart, inside the movement time (ICARS: movement segmented in 2 phases, or in more than 2); the
  decomposed clips move at a slower pace (decomposed_pace: DECOMP_REACHES reaches in the time of N_REACHES cited
  cycles, the level 0's clock kept).

Face touch (shared change SH15; every value a module reading, no source read gives the geometry of a touching tic).
A hand leaves the rest of a base posture (TOUCH_POSTURES: 'sit', the forearms on the chair arms, as in the seated tic
clips; 'stand', the arms hanging) and touches a point of the face with the index (FACE_TARGETS):
- the targets are vertices of the rest mesh among those skinned mostly by the head in front of the head joint
  (face_vertex): the nose tip (nose_vertex); the chin, the most forward midline vertex in the lower CHIN_SHARE of
  the span from the lowest midline face vertex to the nose tip; the forehead, the most forward midline vertex near
  the height FOREHEAD_SHARE of the way from the nose tip to the top of the head; the cheek, the most forward vertex
  at the nose tip's height and CHEEK_X_SHARE of the head's half width from the midline, on the hand's side; each
  read on the skinned base with its outward vertex normal (face_point);
- the pointing hand is POINT_POSE (inside the hand limits of hands.py); hand_blend forms it to a share w (each
  joint turned by w times its pose about its axes), so that the hand closes as it rises (hand_share: minimum jerk
  from the path share HAND_FORM_START to HAND_FORM_END, the hand still open as the arm leaves the rest, no raised
  index in front of the chest at mid-path) and opens on the way back;
- the pointing direction at the touch (touch_dir, TOUCH_DIR_DEG): into the skin along the normal's horizontal part,
  turned about the vertical toward the hand's side (the hand beside the face, not in front of the mouth, which read
  as a 'hush' gesture on the first sheet) and tilted upward, the finger coming from below with the forearm rising to
  it (a nearly level finger holds the forearm level and lifts the elbow toward shoulder height, a salute);
- the path (touch_path): a joint-space blend of the arm's components (BLEND_LEAD: at 1 every component on one
  share) from the rest to the posture whose fingertip lies TOUCH_APPROACH_M back along the pointing direction,
  then the straight fingertip line into the touch, solved by
  the IK of this module (the collar held at rest, touch_bounds), parameterized by the share s of the fingertip's
  path length; the samples of both parts are then solved again all together with a term on the bending energy of
  every component along s (_smooth_touch, TOUCH_SMOOTH_WEIGHT; open-body port, wave 3: sample by
  sample, the junction of the two parts and the components reaching their bounds on the line changed a joint's speed
  by up to 200 deg/s in one knot of the clip, scripts/openbody_wave3_checks.py), the rest and the touch kept; the
  joint components are a cubic spline in s through the samples. A clip moves s on a minimum-jerk
  profile in time whose fingertip peaks at TOUCH_TIP_SPEED_M_S of the posture (touch_rise_s; jerks.touch_timing), so
  that the fingertip's speed along the path is bell-shaped;
- the touch neither floats nor enters the skin: the fingertip target is moved along the pointing direction until the
  least distance between the hand and the head surfaces of the skinned mesh (clearance: exact point-to-triangle
  distances both ways, signed by the outward faces) is SKIN_GAP_M, within SKIN_GAP_TOL_M (secant steps; a search
  that does not converge raises); the path's check (_path_check, at N_CHECK shares) asks the touch in
  [0, SKIN_GAP_MAX_M], no hand or forearm vertex inside the head anywhere on the way, the hand and forearm at least
  CLEAR_M from the trunk and the thighs (or as far as at rest, when the rest is closer), the elbow at least
  TOUCH_ELBOW_DROP_M below the shoulder joint at the touch, and every arm joint under TOUCH_JOINT_SPEED_CAP_DEG_S on
  the path's timing (touch_joint_peaks).
"""
import functools
import math

import numpy as np
import torch

from ..body.rotations import axis_angle_to_matrix, matrix_to_axis_angle, matrix_to_rot6d
from . import hands as H
from .base import posture_rotations, static_body
from .body import F64, joint_limits
from .motions import minimum_jerk

SOURCES = {
    "ohberg2019": "Ohberg F et al. 2019, Sensors 19(5): 1241, doi 10.3390/s19051241 (full text): 20 healthy adults "
                  "aged 39.8 (SD 11.6), finger-to-nose at self-selected speed, 10 repetitions, cycle time 2.35 s "
                  "(dominant arm) and 2.31 s (non-dominant), mixed-model intercepts",
    "rodrigues2017": "Rodrigues MRM et al. 2017, J Neuroeng Rehabil 14: 6, doi 10.1186/s12984-016-0213-y (full text): "
                     "finger-to-nose target at nose height at 90 percent of arm length, self-paced",
    "sara2006": "Schmitz-Hubsch T et al. 2006, Neurology 66: 1717-1720 (SARA; the scale form, sralab.org copy): item 5 "
                "finger chase, dysmetria under or overshooting the target under 5 cm (1), under 15 cm (2), over 15 cm "
                "(3); item 6 nose-finger test, the examiner's finger at about 90 percent of the proband's reach, "
                "moderate speed, kinetic tremor amplitude under 2 cm (1), under 5 cm (2), over 5 cm (3); 4 unable to "
                "perform 5 pointing movements",
    "flash1985": "Flash T, Hogan N 1985, J Neurosci 5(7): 1688-1703: the minimum-jerk model of point-to-point arm "
                 "movements (its standard form, not read here)",
}
SIDES = ("left", "right")
ARM = {"left": (13, 16, 18, 20), "right": (14, 17, 19, 21)}         # collar, shoulder, elbow, wrist
SPINE3 = 9
HEAD = 15
INDEX = {"left": (22, 23, 24), "right": (37, 38, 39)}                 # index MCP, PIP, DIP
CYCLE_S = {"right": 2.35, "left": 2.31}       # SOURCES['ohberg2019'], dominant (right) and non-dominant (left)
HOLD_S = 0.20                                 # [module] hold at the nose and at the examiner's finger
REST_S = 1.0                                  # [module] seated before and after
RAISE_S = 1.0                                 # [module] the raise to the target posture (motions.RAISE_S)
HOLD_OUT_S = 0.5                              # [module] held at the target before the first reach
N_REACHES = 3                                 # MDS-UPDRS 3.16: at least three per side (draft section 2)
TARGET_SHARE = 0.90                           # SOURCES['sara2006'], SOURCES['rodrigues2017']
TARGET_X_SHARE = 0.5                          # [module] the target halfway between the midline and the shoulder
TOUCH_GAP_M = 0.01                            # [module] the fingertip this far from the nose tip at the touch
LIMIT_MARGIN_DEG = 1.0                        # [module] the IK bounds inside the SP1 table
N_WAY = 25                                    # [module] IK points along a fingertip line
PATH_SMOOTH_WEIGHT = 1.0                      # [module] the weight of the squared second differences (deg^2) of the
#                                               components along a line's samples (_smooth_path, open-body wave 2)
PRIMARY_SHARE = 0.6                           # [module] the dysmetric primary movement's share of the movement
PHASE_PAUSE_S = 0.20                          # [module] pause between the phases of a decomposed movement
DECOMP_REACHES = 2                            # [module] a decomposed reach is slower: 2 reaches in the time of 3
DECOMP_LEADS = tuple(round(0.05 * i, 2) for i in range(11))   # [module] the leads of a two-phase movement tried, 0 to
#                                               0.5 (_decompose)
DECOMP_CLEAR_TOL_M = 1e-6                     # [module] its fingertip's allowance under its clearance at the target
PHASE_TIME_EXP = 0.75                         # [module] a decomposed phase lasts in proportion to its travel ** this
# [module] at the nose the index points at the face, POINT_AT_NOSE['elevation_deg'] above the horizontal (the IK term
# w |u - d|^2 on the index axis u, its weight rising as s^2 along the path to the nose, the fingertip held on the
# line by POINT_AT_NOSE['tip_weight'], the collar held near the target posture's by POINT_AT_NOSE['collar_weight']
# so that the shoulder girdle stays off its extremes): a vertical finger in front of the lips read as a 'hush' gesture
POINT_AT_NOSE = {"weight": 100.0, "elevation_deg": 30.0, "tip_weight": 100.0, "collar_weight": 0.05}
# [module] the dysmetric movement between two given postures (path_between): each inner sample solved on the
# fingertip path (the line, or the correction's curve) with the joint blend as its start and reference (weight per
# deg^2), the fingertip held to the path by the tip weight (per cm^2), the collar held to its blend by the collar
# weight (per deg^2; the miss posture's collar to the target posture's by the same weight, miss_posture): with the
# joint reference alone the collar took up the difference between the curved fingertip path and the linear joint
# blend, and the shoulder girdle dipped mid-correction (the review of wave SH1c); with lighter weights the grade 3
# fingertip left the curve by several millimetres mid-correction and the collar still rose again
# (tests/test_synth_tremors.py bounds both)
BETWEEN = {"ref_weight": 1.0, "tip_weight": 10000.0, "collar_weight": 500.0}
CURVE_SAMPLES = 2001                          # [module] samples of a curved fingertip path read for its arc length
STEP_NEIGHBOURS = 3                           # [module] knots on either side of a step read as its neighbourhood
# [module] the largest step of a joint beyond the median of its neighbourhood (joint_step_excess) the reach clips may
# take: a snap of a few degrees over up to STEP_NEIGHBOURS knots lies above it, the smooth movements and the tremors
# of the E3 clips below it (tests/test_synth_tremors.py)
STEP_EXCESS_MAX = {"mm": 4.0, "deg": 1.5}
# [module] the examiner's finger (scene_mesh 'chair_target'): an index finger pointing at the patient, its tip
# EXAMINER_GAP_M beyond the target along the horizontal of the movement, at the target's height (a box, cm: across,
# height, along the finger); the examiner's hand and forearm are left out, so that they neither hide the face in the
# front view nor stand in the way of the patient's arm
EXAMINER = {"finger_cm": (1.6, 1.6, 7.0)}
EXAMINER_GAP_M = 0.006
POINT_POSE = {"middle": (86.6, 76.97, 65.29), "ring": (84.2, 69.94, 59.45), "pinky": (85.0, 61.11, 56.32),
              "thumb": ((0.0, 20.0), 56.0, 68.62)}                     # handtasks.FIST; the index straight


@functools.lru_cache(maxsize=1)
def nose_vertex():
    """The nose tip vertex: among the vertices skinned mostly by the head (joint 15) within 1.5 cm of the midline,
    the most forward one in the rest mesh."""
    _, v, dom = H._rest()
    ids = np.flatnonzero((dom == HEAD) & (np.abs(v[:, 0]) < 0.015))
    return int(ids[np.argmax(v[ids, 2])])


def point_rotations(side):
    """(30, 3, 3) hand rotations: the pointing hand (POINT_POSE) on side, the other hand at rest."""
    pose = {}
    for f, vals in POINT_POSE.items():
        for lv, v in enumerate(vals):
            pose[(side, f, lv)] = v
    return H.hand_rotations(pose)


def fingertip(body, side, knots=None):
    """(T, 3) m: the index fingertip vertex of the skinned mesh (hands.tip_vertices)."""
    return H.body_points(body, [H.tip_vertices()[side]["index"]], knots=knots)[:, 0]


def nose_point(body, knots=None):
    """(T, 3) m: the nose tip vertex of the skinned mesh (nose_vertex)."""
    return H.body_points(body, [nose_vertex()], knots=knots)[:, 0]


# ---------------------------------------------------------------- the arm chain
class _Chain:
    """Forward kinematics of one arm from spine3 to the index fingertip on the seated base (torch, float64): the
    variables are the axis-angle components (deg) of the collar, shoulder, elbow and wrist."""

    def __init__(self, side, posture="sit"):
        self.side = side
        b = static_body(posture, 0.01)
        J, G = b.fk()
        self.base = b
        self.G9 = G[0, SPINE3]
        self.P = {j: J[0, j] for j in range(22)}
        J52, v, _ = H._rest()
        J52 = torch.as_tensor(J52, dtype=F64)
        c, s, e, w = ARM[side]
        i1, i2, i3 = INDEX[side]
        self.off = [J52[s] - J52[c], J52[e] - J52[s], J52[w] - J52[e], J52[i1] - J52[w], J52[i2] - J52[i1],
                    J52[i3] - J52[i2], torch.as_tensor(v[H.tip_vertices()[side]["index"]], dtype=F64) - J52[i3]]
        Rh = point_rotations(side)
        self.Rh = [Rh[H.hand_row(j)] for j in (i1, i2, i3)]
        lim = joint_limits()
        self.lo = np.concatenate([lim.lo[j] for j in ARM[side]]) + LIMIT_MARGIN_DEG
        self.hi = np.concatenate([lim.hi[j] for j in ARM[side]]) - LIMIT_MARGIN_DEG
        self.q_sit = np.degrees(matrix_to_axis_angle(posture_rotations(posture)[list(ARM[side])]).numpy()).reshape(-1)

    def points(self, q_deg):
        """{'shoulder', 'elbow', 'wrist', 'tip'}: (N, 3) positions for q (N, 12) deg (torch, differentiable)."""
        q = torch.as_tensor(q_deg, dtype=F64)
        q = q.reshape(-1, 4, 3)
        R = axis_angle_to_matrix(torch.deg2rad(q))
        c = ARM[self.side][0]
        Gc = self.G9 @ R[:, 0]
        Ps = self.P[c] + Gc @ self.off[0]
        Gs = Gc @ R[:, 1]
        Pe = Ps + Gs @ self.off[1]
        Ge = Gs @ R[:, 2]
        Pw = Pe + Ge @ self.off[2]
        Gw = Ge @ R[:, 3]
        P1 = Pw + Gw @ self.off[3]
        G1 = Gw @ self.Rh[0]
        P2 = P1 + G1 @ self.off[4]
        G2 = G1 @ self.Rh[1]
        P3 = P2 + G2 @ self.off[5]
        G3 = G2 @ self.Rh[2]
        tip = P3 + G3 @ self.off[6]
        return {"shoulder": Ps, "elbow": Pe, "wrist": Pw, "index_mcp": P1, "tip": tip}

    def reach_m(self):
        """The distance from the shoulder joint to the fingertip with the arm straight and the pointing hand (every
        arm component at 0: the rest arm)."""
        p = self.points(np.zeros((1, 12)))
        return float((p["tip"] - p["shoulder"]).norm())

    def solve(self, target, q0, q_ref, w_ref=1e-3, point_dir=None, w_dir=0.0, w_tip=1.0, collar_ref=None,
              w_collar=0.0, bounds=None):
        """q (12,) deg: the bounded IK of the module docstring for a fingertip target (3,) m; point_dir (3,) unit and
        w_dir: a term w_dir |u - point_dir|^2 on the index axis u (index knuckle to fingertip); collar_ref (3,) deg
        and w_collar: a term holding the collar near collar_ref (the shoulder girdle away from its extremes);
        bounds (lo, hi) (12,) deg in place of the chain's (the face touch, touch_bounds)."""
        from scipy.optimize import minimize
        tgt = torch.as_tensor(np.asarray(target, dtype=np.float64))
        qr = torch.as_tensor(np.asarray(q_ref, dtype=np.float64))
        pd = None if point_dir is None or not w_dir else torch.as_tensor(np.asarray(point_dir, dtype=np.float64))
        cr = None if collar_ref is None or not w_collar else torch.as_tensor(np.asarray(collar_ref, dtype=np.float64))

        def f(x):
            q = torch.tensor(x, dtype=F64, requires_grad=True)
            p = self.points(q[None])
            e = 100.0 * (p["tip"][0] - tgt)                                      # cm
            up = torch.relu(100.0 * (p["elbow"][0, 1] - p["shoulder"][0, 1] + 0.05))
            loss = w_tip * (e ** 2).sum() + w_ref * ((q - qr) ** 2).sum() + 1e-2 * (q[9:12] ** 2).sum() \
                + 1e-2 * q[8] ** 2 + 10.0 * up ** 2
            if pd is not None:
                u = p["tip"][0] - p["index_mcp"][0]
                loss = loss + w_dir * ((u / u.norm() - pd) ** 2).sum()
            if cr is not None:
                loss = loss + w_collar * ((q[0:3] - cr) ** 2).sum()
            loss.backward()
            return float(loss.detach()), q.grad.numpy().copy()
        lo, hi = (self.lo, self.hi) if bounds is None else bounds
        x0 = np.clip(np.asarray(q0, dtype=np.float64), lo, hi)
        r = minimize(f, x0, jac=True, method="L-BFGS-B", bounds=list(zip(lo, hi)),
                     options={"maxiter": 500, "ftol": 1e-14, "gtol": 1e-10})
        return r.x


@functools.lru_cache(maxsize=4)
def chain(side):
    return _Chain(side)


@functools.lru_cache(maxsize=4)
def targets(side):
    """{'target', 'nose_target', 'nose', 'reach_m', 'shoulder'} (numpy, m) of the module docstring, on the seated
    base."""
    ch = chain(side)
    J52, v, _ = H._rest()
    G15 = ch.base.fk()[1][0, HEAD].numpy()
    nose = G15 @ (v[nose_vertex()] - J52[HEAD]) + ch.P[HEAD].numpy()
    sh = ch.points(ch.q_sit[None])["shoulder"][0].numpy()        # the collar is moved by the IK; first estimate
    reach_m = ch.reach_m()
    x = TARGET_X_SHARE * sh[0]
    dy = nose[1] - sh[1]
    dz = math.sqrt(max((TARGET_SHARE * reach_m) ** 2 - (x - sh[0]) ** 2 - dy ** 2, 0.0))
    tgt = np.array([x, nose[1], sh[2] + dz])
    u = (tgt - nose) / np.linalg.norm(tgt - nose)
    return {"target": tgt, "nose_target": nose + TOUCH_GAP_M * u, "nose": nose, "reach_m": reach_m, "shoulder": sh}


def nose_point_dir(side):
    """(3,) unit: the index axis wanted at the nose (POINT_AT_NOSE): toward the face along the horizontal of the
    movement, POINT_AT_NOSE['elevation_deg'] upward."""
    tg = targets(side)
    u = tg["target"] - tg["nose"]
    u[1] = 0.0
    u = u / np.linalg.norm(u)
    e = math.radians(POINT_AT_NOSE["elevation_deg"])
    return -math.cos(e) * u + math.sin(e) * np.array([0.0, 1.0, 0.0])


@functools.lru_cache(maxsize=64)
def _path(side, p0, p1, q0, pointed=False):
    """(s (N_WAY,), q (N_WAY, 12) deg): the postures along the straight fingertip line p0 -> p1 (tuples) by
    continuation from the posture q0 (tuple); pointed: the index turned toward the face as the line nears its end
    (POINT_AT_NOSE)."""
    ch = chain(side)
    s = np.linspace(0.0, 1.0, N_WAY)
    a, b = np.asarray(p0), np.asarray(p1)
    q = [np.asarray(q0, dtype=np.float64)]
    P = POINT_AT_NOSE
    d = nose_point_dir(side) if pointed else None
    for si in s[1:]:
        if pointed:
            q.append(ch.solve(a + si * (b - a), q[-1], q[-1], point_dir=d, w_dir=P["weight"] * si ** 2,
                              w_tip=P["tip_weight"], collar_ref=q[0][0:3], w_collar=P["collar_weight"]))
        else:
            q.append(ch.solve(a + si * (b - a), q[-1], q[-1]))
    return s, _smooth_path(ch, s, np.stack(q), a[None] + s[:, None] * (b - a)[None], d)


def _smooth_path(ch, s, q, pts, point_dir=None, weight=None, ref=None, keep_last=False):
    """(N, 12) deg: the postures q (N, 12) along the fingertip points pts (N, 3) at the shares s, solved again all
    together (open-body port, wave 2; module docstring, Arm postures): the terms of each sample's IK (the fingertip
    on its point, the wrist near neutral, the elbow without its valgus, the elbow below the shoulder; with point_dir
    the pointing term and the collar held near the first posture's, POINT_AT_NOSE; with ref (N, 12) the joint blend
    of path_between as the reference and its collar held to it, BETWEEN's weights) plus PATH_SMOOTH_WEIGHT times the
    squared second differences of every component along the samples, inside the IK bounds, the first posture kept
    (keep_last: the last one too). Sample by sample, the free components turned sharply where another reached its
    bound (the elbow's pronation share and the wrist's radial deviation on the open body, the shoulder's axial
    rotation on the old table): the arm's speed changed by 90 to 170 deg/s in one knot there."""
    from scipy.optimize import minimize
    w_s = PATH_SMOOTH_WEIGHT if weight is None else float(weight)
    n = len(s)
    if w_s <= 0 or n < 3 + int(keep_last):
        return q
    P = POINT_AT_NOSE
    line = torch.as_tensor(np.asarray(pts, dtype=np.float64))
    q0, qn = torch.as_tensor(q[0], dtype=F64), torch.as_tensor(q[-1], dtype=F64)
    pd = None if point_dir is None else torch.as_tensor(np.asarray(point_dir, dtype=np.float64))
    w_dir = torch.as_tensor(P["weight"] * np.asarray(s) ** 2)
    w_tip = BETWEEN["tip_weight"] if ref is not None else P["tip_weight"] if pd is not None else 1.0
    qr = None if ref is None else torch.as_tensor(np.asarray(ref, dtype=np.float64))
    m = n - 1 - int(keep_last)

    def f(x):
        X = torch.tensor(x, dtype=F64, requires_grad=True)
        Q = torch.cat([q0[None], X.reshape(m, 12)] + ([qn[None]] if keep_last else []), 0)
        p = ch.points(Q)
        e = 100.0 * (p["tip"] - line)                                                    # cm
        up = torch.relu(100.0 * (p["elbow"][:, 1] - p["shoulder"][:, 1] + 0.05))
        loss = w_tip * (e ** 2).sum() + 1e-2 * (Q[:, 9:12] ** 2).sum() + 1e-2 * (Q[:, 8] ** 2).sum() \
            + 10.0 * (up ** 2).sum()
        if pd is not None:
            u = p["tip"] - p["index_mcp"]
            u = u / u.norm(dim=1, keepdim=True)
            loss = loss + (w_dir * ((u - pd) ** 2).sum(1)).sum() \
                + P["collar_weight"] * ((Q[:, 0:3] - q0[None, 0:3]) ** 2).sum()
        if qr is not None:
            loss = loss + BETWEEN["ref_weight"] * ((Q - qr) ** 2).sum() \
                + BETWEEN["collar_weight"] * ((Q[:, 0:3] - qr[:, 0:3]) ** 2).sum()
        loss = loss + w_s * ((Q[2:] - 2.0 * Q[1:-1] + Q[:-2]) ** 2).sum()
        loss.backward()
        return float(loss.detach()), X.grad.numpy().copy()
    r = minimize(f, q[1:1 + m].reshape(-1), jac=True, method="L-BFGS-B",
                 bounds=list(zip(np.tile(ch.lo, m), np.tile(ch.hi, m))),
                 options={"maxiter": 5000, "ftol": 1e-15, "gtol": 1e-10})
    return np.concatenate([q[:1], r.x.reshape(m, 12)] + ([q[-1:]] if keep_last else []), 0)


def _key(x):
    return tuple(float(round(v, 9)) for v in np.asarray(x, dtype=np.float64).reshape(-1))


def fingertip_path(p0, p1, s, via=None):
    """(len(s), 3) m: the fingertip path from p0 (3,) to p1 (3,) at the shares s of its length: the straight line,
    or with via (3,) the quadratic curve (1 - r)^2 p0 + 2 r (1 - r) via + r^2 p1 (leaving p0 toward via, reaching p1
    from via), read at equal shares of its arc length (CURVE_SAMPLES)."""
    a, b = np.asarray(p0, dtype=np.float64), np.asarray(p1, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    if via is None:
        return a + s[:, None] * (b - a)
    w = np.asarray(via, dtype=np.float64)

    def curve(r):
        r = r[:, None]
        return (1.0 - r) ** 2 * a + 2.0 * r * (1.0 - r) * w + r ** 2 * b
    r = np.linspace(0.0, 1.0, CURVE_SAMPLES)
    L = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(curve(r), axis=0), axis=1))]
    return curve(np.interp(s * L[-1], L, r))


@functools.lru_cache(maxsize=64)
def _path_between(side, p0, p1, qa, qb, via=None):
    """(s (N_WAY,), q (N_WAY, 12) deg): the postures along the fingertip path p0 -> p1 (tuples; fingertip_path: the
    straight line, or the curve through the control point via) from the posture qa at p0 to the posture qb at p1
    (tuples), both kept exactly: each inner sample solved on the path with the joint blend qa + s (qb - qa) as its
    start and reference and the collar held to the blend's collar (BETWEEN), so that the arm's redundancy follows the
    blend, the shoulder girdle moves steadily from qa to qb and the joint components reach qb with no step (path() by
    continuation ends on a posture of its own); then the samples solved again together, both ends kept
    (_smooth_path, open-body port wave 2)."""
    ch = chain(side)
    s = np.linspace(0.0, 1.0, N_WAY)
    P = fingertip_path(p0, p1, s, via)
    qa, qb = np.asarray(qa, dtype=np.float64), np.asarray(qb, dtype=np.float64)
    q = [qa]
    for i in range(1, N_WAY - 1):
        ref = qa + s[i] * (qb - qa)
        q.append(ch.solve(P[i], ref, ref, w_ref=BETWEEN["ref_weight"], w_tip=BETWEEN["tip_weight"],
                          collar_ref=ref[0:3], w_collar=BETWEEN["collar_weight"]))
    q.append(qb)
    ref = qa[None] + s[:, None] * (qb - qa)[None]
    return s, _smooth_path(ch, s, np.stack(q), P, ref=ref, keep_last=True)


def path_between(side, p0, p1, qa, qb, via=None):
    """(s, q) of _path_between (cached on rounded inputs)."""
    return _path_between(side, _key(p0), _key(p1), _key(qa), _key(qb), None if via is None else _key(via))


def movement_axis(side):
    """(3,) unit: the horizontal of the movement from the nose to the examiner's finger (examiner_boxes' axis)."""
    tg = targets(side)
    a = tg["target"] - tg["nose"]
    a[1] = 0.0
    return a / np.linalg.norm(a)


def correction_via(side, error):
    """(3,) m or None: the control point of the curve of a dysmetric correction (fingertip_path), from the miss
    target + error (3,) m back to the target: the target plus the error's part across movement_axis, so that the
    curve draws the overshoot back along the movement first and closes the side offset last (module docstring);
    None for an error along the movement within a micrometre (an undershoot), whose correction stays on the line."""
    e = np.asarray(error, dtype=np.float64)
    a = movement_axis(side)
    across = e - (e @ a) * a
    if np.linalg.norm(across) < 1e-6:
        return None
    return targets(side)["target"] + across


@functools.lru_cache(maxsize=16)
def _miss_posture(side, miss):
    ch = chain(side)
    q_t = target_posture(side)
    return ch.solve(np.asarray(miss), q_t, q_t, collar_ref=q_t[0:3], w_collar=BETWEEN["collar_weight"])


def miss_posture(side, miss):
    """q (12,) deg: the arm with the fingertip at a dysmetric end point miss (3,) m, solved from the target posture
    (target_posture) with it as the reference and its collar held by BETWEEN['collar_weight'], so that the overshoot
    or undershoot keeps the target's arm posture family and its shoulder girdle; the correction returns to the target
    posture (its largest joint component change is the phase's 'correction_deg' in reach's info)."""
    return _miss_posture(side, _key(miss))


@functools.lru_cache(maxsize=4)
def target_posture(side):
    """q (12,) deg: the arm at the examiner's finger, from the arms-forward posture of base.POSTURES."""
    ch = chain(side)
    qf = np.degrees(matrix_to_axis_angle(posture_rotations("sit_arms_forward")[list(ARM[side])]).numpy()).reshape(-1)
    tg = targets(side)["target"]
    q = ch.solve(tg, qf, qf, w_ref=1e-2)
    return ch.solve(tg, q, q)


def path(side, p0, p1, q0, pointed=False):
    """(s, q) of _path (cached on rounded inputs)."""
    return _path(side, _key(p0), _key(p1), _key(q0), bool(pointed))


def decomposed_pace(side):
    """(n_reaches, cycle_s) of a decomposed reach: DECOMP_REACHES reaches in the time of N_REACHES cited cycles, on
    the level 0's clock."""
    return DECOMP_REACHES, N_REACHES * CYCLE_S[side] / DECOMP_REACHES


def _interp(s_grid, q_grid, s):
    from scipy.interpolate import CubicSpline
    cs = CubicSpline(s_grid, q_grid, axis=0)
    return cs(np.clip(s, 0.0, 1.0))


# ---------------------------------------------------------------- the timeline
def timeline(side="left", n_reaches=N_REACHES, cycle_s=None, rest_s=REST_S, raise_s=RAISE_S, hold_out_s=HOLD_OUT_S,
             hold_s=HOLD_S):
    """[(kind, t0_s, t1_s, cycle)] of the base motion (module docstring) and its duration (s)."""
    c = CYCLE_S[side] if cycle_s is None else float(cycle_s)
    m = c / 2.0 - hold_s
    if m <= 0:
        raise ValueError("the holds leave no movement time in the cycle")
    out, t = [("rest", 0.0, rest_s, None), ("raise", rest_s, rest_s + raise_s, None)], rest_s + raise_s
    out.append(("hold_target", t, t + hold_out_s, None))
    t += hold_out_s
    for i in range(int(n_reaches)):
        for kind, d in (("to_nose", m), ("hold_nose", hold_s), ("to_target", m), ("hold_target", hold_s)):
            out.append((kind, t, t + d, i))
            t += d
    out.append(("lower", t, t + raise_s, None))
    out.append(("rest", t + raise_s, t + raise_s + rest_s, None))
    return out, t + raise_s + rest_s


def _groups(side):
    """Joint groups of a decomposed movement: proximal (collar, shoulder) and distal (elbow, wrist), as indices of
    the 12 components."""
    return {"proximal": np.arange(0, 6), "distal": np.arange(6, 12)}


def reach(side="left", n_reaches=N_REACHES, cycle_s=None, rest_s=REST_S, raise_s=RAISE_S, hold_out_s=HOLD_OUT_S,
          hold_s=HOLD_S, endpoint_errors=None, decomposition=1, rate=100.0):
    """(HandBody, info): the base motion of the module docstring on one side. endpoint_errors {cycle (0-based):
    (3,) m} and decomposition (phases per movement, 1 normal) are the cerebellar movement variants; the timeline is
    that of level 0 in every case. info: side, timeline phases (with the fingertip ends), targets, cycle time and its
    source, the arm postures (deg), the IK residuals (mm) and the decomposition steps."""
    if side not in SIDES:
        raise ValueError(f"side {side!r}")
    errs = {int(k): np.asarray(v, dtype=np.float64) for k, v in (endpoint_errors or {}).items()}
    decomposition = int(decomposition)
    if decomposition < 1:
        raise ValueError("decomposition counts the phases of a movement, at least 1")
    ch = chain(side)
    tg = targets(side)
    q_t = target_posture(side)
    s_n, q_n = path(side, tg["target"], tg["nose_target"], q_t, pointed=True)
    s_b, q_b = 1.0 - s_n[::-1], q_n[::-1].copy()          # back to the target on the same postures
    q_nose = q_n[-1]
    phases, dur = timeline(side, n_reaches, cycle_s, rest_s, raise_s, hold_out_s, hold_s)
    hold_end = {cyc: b for kind, a, b, cyc in phases if kind == "hold_target" and cyc is not None}
    dysmetric = set(errs) if decomposition == 1 else set()
    T = int(round(dur * rate)) + 1
    t = np.arange(T) / rate
    q = np.repeat(ch.q_sit[None], T, 0).astype(np.float64)
    rec = []
    for kind, a, b, cyc in phases:
        sel = (t >= a - 1e-9) & (t <= b + 1e-9)
        u = (t[sel] - a) / (b - a)
        item = {"kind": kind, "t0_s": a, "t1_s": b, "cycle": cyc}
        if kind == "hold_target" and cyc in dysmetric:
            # the dysmetric movement of this cycle runs through the hold (its rows set with the movement)
            item.update(tip_m=tg["target"].tolist(), taken_by_correction=True)
        elif kind == "rest":
            q[sel] = ch.q_sit
        elif kind == "raise":
            q[sel] = ch.q_sit + minimum_jerk(u)[:, None] * (q_t - ch.q_sit)
        elif kind == "lower":
            q[sel] = q_t + minimum_jerk(u)[:, None] * (ch.q_sit - q_t)
        elif kind == "hold_target":
            q[sel] = q_t
            item.update(tip_m=tg["target"].tolist())
        elif kind == "hold_nose":
            q[sel] = q_nose
            item.update(tip_m=tg["nose_target"].tolist())
        elif kind in ("to_nose", "to_target"):
            toward = kind == "to_nose"
            p0, p1 = (tg["target"], tg["nose_target"]) if toward else (tg["nose_target"], tg["target"])
            qa, qb = (q_t, q_nose) if toward else (q_nose, q_t)
            if decomposition > 1:
                q[sel], steps = _decompose(qa, qb, u, (b - a), decomposition, toward, side)
                item.update(decomposition=steps)
            elif not toward and cyc in dysmetric:
                e = errs[cyc]
                miss = p1 + e
                q_miss = miss_posture(side, miss)
                via = correction_via(side, e)
                s1, q1 = path_between(side, p0, miss, qa, q_miss)
                s2, q2 = path_between(side, miss, p1, q_miss, qb, via)
                z = hold_end[cyc]                              # the movement takes the hold at the target
                sel = (t >= a - 1e-9) & (t <= z + 1e-9)
                u = (t[sel] - a) / (z - a)
                k = u <= PRIMARY_SHARE
                qq = np.empty((len(u), 12))
                qq[k] = _interp(s1, q1, minimum_jerk(u[k] / PRIMARY_SHARE))
                qq[~k] = _interp(s2, q2, minimum_jerk((u[~k] - PRIMARY_SHARE) / (1.0 - PRIMARY_SHARE)))
                q[sel] = qq
                item.update(error_m=e.tolist(), miss_m=miss.tolist(), movement_end_s=z,
                            primary_end_s=a + PRIMARY_SHARE * (z - a), correction_s=(1.0 - PRIMARY_SHARE) * (z - a),
                            correction_deg=float(np.abs(q_miss - qb).max()),
                            correction_via_m=None if via is None else via.tolist())
            else:
                sg, qg = (s_n, q_n) if toward else (s_b, q_b)
                q[sel] = _interp(sg, qg, minimum_jerk(u))
            item.update(tip0_m=list(map(float, p0)), tip1_m=list(map(float, p1)))
        rec.append(item)
    q = np.clip(q, ch.lo, ch.hi)
    R = posture_rotations("sit").repeat(T, 1, 1, 1)
    Rq = axis_angle_to_matrix(torch.deg2rad(torch.as_tensor(q.reshape(T, 4, 3))))
    for i, j in enumerate(ARM[side]):
        R[:, j] = Rq[:, i]
    base = ch.base
    tr = base.transl[:1].repeat(T, 1)
    Rh = point_rotations(side)
    b = H.HandBody(matrix_to_rot6d(R), tr, base.contact[:1].repeat(T, 1), rate=rate,
                   hand6d=matrix_to_rot6d(Rh)[None].expand(T, -1, 6).clone())
    res = {}
    for name, qq, p in (("target", q_t, tg["target"]), ("nose", q_nose, tg["nose_target"])):
        res[name] = float(1000.0 * np.linalg.norm(ch.points(qq[None])["tip"][0].detach().numpy() - p))
    info = {"side": side, "posture": "sit", "n_reaches": int(n_reaches), "duration_s": dur, "phases": rec,
            "cycle_s": CYCLE_S[side] if cycle_s is None else float(cycle_s), "cycle_source": "ohberg2019",
            "hold_s": hold_s, "movement_s": (CYCLE_S[side] if cycle_s is None else float(cycle_s)) / 2.0 - hold_s,
            "profile": "minimum jerk (flash1985)", "targets": {k: (v.tolist() if hasattr(v, "tolist") else v)
                                                               for k, v in tg.items()},
            "target_share": TARGET_SHARE, "target_x_share": TARGET_X_SHARE, "touch_gap_m": TOUCH_GAP_M,
            "q_target_deg": q_t.tolist(), "q_nose_deg": q_nose.tolist(), "ik_residual_mm": res,
            "point_at_nose": dict(POINT_AT_NOSE, index_from_vertical_deg=_index_from_vertical(side, q_nose)),
            "decomposition": decomposition, "endpoint_errors_m": {k: v.tolist() for k, v in errs.items()}}
    return b, info


def _index_from_vertical(side, q):
    p = chain(side).points(np.asarray(q)[None])
    u = (p["tip"][0] - p["index_mcp"][0]).detach().numpy()
    return float(np.degrees(np.arccos(np.clip(u[1] / np.linalg.norm(u), -1.0, 1.0))))


# ---------------------------------------------------------------- the examiner's finger (scene)
def examiner_boxes(side):
    """[(centre (3,), axes (3, 3) columns: across, up, away from the patient, half sizes (3,))] m of EXAMINER: the
    finger pointing at the patient along the horizontal of the movement from the nose, its tip EXAMINER_GAP_M beyond
    the target, at its height."""
    tg = targets(side)
    a = movement_axis(side)
    up = np.array([0.0, 1.0, 0.0])
    x = np.cross(up, a)
    Rm = np.stack([x, up, a], 1)
    fw, fh, fl = np.asarray(EXAMINER["finger_cm"]) / 100.0
    c_f = tg["target"] + (EXAMINER_GAP_M + 0.5 * fl) * a
    return [(c_f, Rm, 0.5 * np.array([fw, fh, fl]))]


def box_sdf(P, boxes):
    """(...,) m: the signed distance of points P (..., 3) to the union of oriented boxes (negative inside)."""
    P = np.asarray(P, dtype=np.float64)
    out = np.full(P.shape[:-1], np.inf)
    for c, Rm, h in boxes:
        q = np.abs((P - c) @ Rm) - h
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
        inside = np.minimum(q.max(-1), 0.0)
        out = np.minimum(out, outside + inside)
    return out


def scene_mesh(kind, body, floor_y):
    """(vertices, faces) of the 'chair_target' scene: the chair (viewer.chair_mesh) and the examiner's finger
    (examiner_boxes) of the reaching side (the side whose wrist moves most in the body)."""
    from .viewer import box, chair_mesh
    if kind != "chair_target":
        raise ValueError(f"scene {kind!r}")
    J = body.joints().numpy()
    side = "left" if np.ptp(J[:, ARM["left"][3]], 0).sum() >= np.ptp(J[:, ARM["right"][3]], 0).sum() else "right"
    v0, f0 = chair_mesh(floor_y)
    vs, fs, off = [v0], [f0], len(v0)
    for c, Rm, h in examiner_boxes(side):
        v, f = box(-h, h)
        vs.append(c + v @ Rm.T)
        fs.append(f + off)
        off += len(v)
    return np.concatenate(vs), np.concatenate(fs)


def _decompose(qa, qb, u, dur, n_phases, toward_nose, side="left"):
    """((len(u), 12) deg, steps): the movement qa -> qb group by group (module docstring) at the shares u of a
    movement of dur s: n_phases minimum-jerk phases separated by PHASE_PAUSE_S, each lasting in proportion to the
    fingertip's travel in it to the power PHASE_TIME_EXP (a long segment is not rushed). Two phases: toward the nose
    the proximal group leads, away from it the distal group. Three or more: the distal group (the elbow carries most
    of the fingertip's travel) in halves around the proximal group, each distal part halved again until n_phases.
    Two phases: where the first group's arc leaving the target (toward the nose) or the last group's arc reaching it
    (away from the nose) brings the fingertip (the chain's rigid point) nearer the examiner's finger (examiner_boxes)
    than it stands at the target (the shoulder's arc of reach_E3_decomposition_icars2 came within 0.87 mm of the
    finger, 6.14 at the target, and took the skinned index 0.9 mm into it: Phase D of docs/PLAN_2026-11-12.md), the
    other group moves with that phase by the least lead of DECOMP_LEADS that keeps the fingertip as far as at the
    target (DECOMP_CLEAR_TOL_M), on the same profile (toward the nose the elbow and wrist begin to flex with the
    shoulder; away from it they end their extension with it), recorded in the steps ('lead', 'with')."""
    g = _groups(None)
    if n_phases == 2:
        steps = [("proximal", 0.0, 1.0), ("distal", 0.0, 1.0)] if toward_nose else \
            [("distal", 0.0, 1.0), ("proximal", 0.0, 1.0)]
    else:
        steps = [("distal", 0.0, 0.5), ("proximal", 0.0, 1.0), ("distal", 0.5, 1.0)]
        while len(steps) < n_phases:
            i = max((k for k, st in enumerate(steps) if st[0] == "distal"), key=lambda k: steps[k][2] - steps[k][1])
            grp, f0, f1 = steps.pop(i)
            mid = 0.5 * (f0 + f1)
            steps[i:i] = [(grp, f0, mid), (grp, mid, f1)]
    n = len(steps)
    pause = PHASE_PAUSE_S / dur
    free = 1.0 - pause * (n - 1)
    if free <= 0:
        raise ValueError("the pauses leave no time for the phases")

    def plan(lead):
        """[(group, f0, f1, with)]: the steps with the other group's lead ('with': (group, f0, f1) or None)."""
        if n != 2 or lead <= 0.0:
            return [(grp, f0, f1, None) for grp, f0, f1 in steps]
        (g0, _, _), (g1, _, _) = steps
        if toward_nose:                                   # the second group starts with the first phase
            return [(g0, 0.0, 1.0, (g1, 0.0, lead)), (g1, lead, 1.0, None)]
        return [(g0, 0.0, 1.0 - lead, None), (g1, 0.0, 1.0, (g0, 1.0 - lead, 1.0))]   # the first ends with the last

    def build(pl):
        cur = {k: 0.0 for k in g}                          # the fingertip's travel in each step
        pts = []
        for grp, f0, f1, w_ in [(None, 0.0, 0.0, None)] + pl:
            if grp is not None:
                cur[grp] = f1
            if w_ is not None:
                cur[w_[0]] = w_[2]
            qq = np.concatenate([qa[g[k]] + cur[k] * (qb[g[k]] - qa[g[k]]) for k in ("proximal", "distal")])
            pts.append(chain(side).points(qq[None])["tip"][0].detach().numpy())
        d = np.linalg.norm(np.diff(np.stack(pts), axis=0), axis=1)
        wts = np.maximum(d, 1e-3) ** PHASE_TIME_EXP
        phs = free * wts / wts.sum()
        share = {k: np.zeros(len(u)) for k in g}
        out_steps = []
        a = 0.0
        for i, (grp, f0, f1, w_) in enumerate(pl):
            w = minimum_jerk((u - a) / phs[i])
            share[grp] = np.where(u >= a, f0 + (f1 - f0) * w, share[grp])
            if w_ is not None:
                share[w_[0]] = np.where(u >= a, w_[1] + (w_[2] - w_[1]) * w, share[w_[0]])
            out_steps.append({"group": grp, "from": f0, "to": f1, "u0": a, "u1": a + phs[i],
                              "tip_travel_cm": float(100 * d[i])})
            if w_ is not None:
                out_steps[-1]["with"] = {"group": w_[0], "from": w_[1], "to": w_[2]}
            a += phs[i] + pause
        q = np.empty((len(u), 12))
        for grp, idx in g.items():
            q[:, idx] = qa[idx] + share[grp][:, None] * (qb[idx] - qa[idx])
        return q, out_steps
    if n != 2:
        return build(plan(0.0))
    tip = lambda q: chain(side).points(q)["tip"].detach().numpy()          # noqa: E731
    boxes = examiner_boxes(side)
    for lead in DECOMP_LEADS:
        q, out_steps = build(plan(lead))
        if box_sdf(tip(q), boxes).min() >= box_sdf(tip(np.stack([qa, qb])), boxes).min() - DECOMP_CLEAR_TOL_M:
            break
    for st in out_steps:
        st["lead"] = float(lead)
    return q, out_steps


# ---------------------------------------------------------------- read-back
def speed_runs(points, rate, t0_s, t1_s, rel=0.10):
    """[(start_s, end_s, peak speed m/s)] of the runs of a point's speed above rel of its largest value inside
    [t0_s, t1_s] (the submovements of one movement)."""
    p = np.asarray(points, dtype=np.float64)
    v = np.linalg.norm(np.gradient(p, 1.0 / rate, axis=0), axis=-1)
    k0, k1 = int(round(t0_s * rate)), int(round(t1_s * rate)) + 1
    w = v[k0:k1]
    on = w > rel * w.max()
    m = np.concatenate([[False], on, [False]])
    d = np.diff(m.astype(np.int8))
    return [((k0 + a) / rate, (k0 + b - 1) / rate, float(w[a:b].max()))
            for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))]


def _excess(d, n):
    """d (K, J) less, at each row, the median of the n rows before and the n rows after it (the edges read the rows
    that exist). A step spread over up to n knots leaves the median of its 2 n neighbours on the steps around it (the
    largest neighbour, the first build's read, hid a step spread over two knots: each half matched the other)."""
    K = len(d)
    pad = np.full((n,) + d.shape[1:], np.nan)
    dp = np.concatenate([pad, d, pad])
    nb = np.stack([dp[n - s:n - s + K] for s in range(1, n + 1)] + [dp[n + s:n + s + K] for s in range(1, n + 1)])
    return d - np.nanmedian(nb, axis=0)


def joint_step_excess(body, n=STEP_NEIGHBOURS):
    """{'mm': (T-1, 22), 'deg': (T-1, 22)[, 'hand_deg': (T-1, 30)]}: at each pair of successive knots, the step of
    each body joint (its position, mm; the angle of its local rotation, deg; the hand rows' local rotations when the
    body has hands) less the median step of the same joint within n knots on either side (_excess). A joint that
    snaps over one to n knots steps far beyond its neighbourhood; a smooth movement hardly at all, a tremor by the
    change of its speed over a few knots (a fast tremor of the E3 clips reads the most, STEP_EXCESS_MAX)."""
    def angles(R):
        rel = R[:-1].transpose(-1, -2) @ R[1:]
        c = ((rel.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0).clamp(-1.0, 1.0)
        return np.degrees(torch.arccos(c).numpy())
    J = body.joints().numpy()
    out = {"mm": _excess(1000.0 * np.linalg.norm(np.diff(J, axis=0), axis=-1), n),
           "deg": _excess(angles(body.local_R()), n)}
    if H.has_hands(body):
        out["hand_deg"] = _excess(angles(body.hand_R()), n)
    return out


def endpoint_error_cm(points, rate, t0_s, t1_s, target):
    """cm: the distance from target (3,) of the point where a movement inside [t0_s, t1_s] first comes to rest:
    the slowest knot between the first and the second speed run (speed_runs above 5 percent of the largest
    speed), the primary end point of the
    movement; the last knot when the movement has a single run (it goes straight to the target)."""
    p = np.asarray(points, dtype=np.float64)
    v = np.linalg.norm(np.gradient(p, 1.0 / rate, axis=0), axis=-1)
    runs = speed_runs(p, rate, t0_s, t1_s, rel=0.05)
    k1 = int(round(t1_s * rate))
    if len(runs) < 2:
        k = k1
    else:
        a, b = int(round(runs[0][1] * rate)), int(round(runs[1][0] * rate))
        k = a + int(np.argmin(v[a:b + 1]))
    return float(100.0 * np.linalg.norm(p[k] - np.asarray(target)))


# ---------------------------------------------------------------- the face touch (shared change SH15)
FACE_TARGETS = ("nose", "cheek", "chin", "forehead")
TOUCH_POSTURES = ("sit", "stand")
SKIN_GAP_M = 0.0015          # [module] the finger's least distance to the skin aimed at the touch
SKIN_GAP_MAX_M = 0.003       # [module, task SH15] the touch floats beyond 3 mm; below 0 it enters the skin
SKIN_GAP_TOL_M = 0.0002      # [module] the gap iteration stops within 0.2 mm of SKIN_GAP_M
TOUCH_APPROACH_M = 0.05       # [module] the last 5 cm run on a straight fingertip line along the pointing direction
# [module] the pointing direction at the touch: into the skin along the normal's horizontal part, turned about the
# vertical by 'azimuth' toward the side of the hand (the hand beside the face, not in front of the mouth; a smaller
# azimuth brings back the upright finger across the lips) and 'elevation' upward (the finger comes from below, the
# forearm rising to it; at 10 to 25 deg, the first build, the level forearm lifted the elbow to the IK's floor 5 cm
# below the shoulder, a salute; the forehead's 45 deg left the elbow 5.0 cm below the shoulder on the open body, its 55
# deg 8.3 cm: open-body port, wave 2)
TOUCH_DIR_DEG = {"nose": {"azimuth": 45.0, "elevation": 45.0}, "cheek": {"azimuth": 0.0, "elevation": 45.0},
                 "chin": {"azimuth": 35.0, "elevation": 45.0}, "forehead": {"azimuth": 55.0, "elevation": 55.0}}
TOUCH_POINT_WEIGHT = 100.0    # [module] the IK weight of the pointing direction at the face (POINT_AT_NOSE's)
TOUCH_ELBOW_DROP_M = 0.07     # [module] at the touch the elbow lies at least 7 cm below the shoulder joint
#                               (_path_check; the IK's own floor is 5 cm, the salute of the first build)
HAND_FORM_START = 0.3         # [module] the hand forms its point between these shares of the path (minimum jerk): open
HAND_FORM_END = 0.85          # while the arm leaves the rest, formed before the approach line; it opens on the way back
TOUCH_TIP_SPEED_M_S = {"sit": 2.0, "stand": 3.0}   # [module, unsourced] the fingertip's peak speed along a touch path
#                               (touch_rise_s): the longer reach from the hanging arm runs faster, so that a standing
#                               touch stays a brief tic and not a slow deliberate gesture
TOUCH_JOINT_SPEED_CAP_DEG_S = 400.0   # [module, unsourced] on that profile no arm joint turns faster
#                                       (touch_joint_peaks)
MJ_PEAK = 1.875               # the peak speed of a minimum-jerk profile is 1.875 times its mean (motions.minimum_jerk)
CHIN_SHARE = 0.4              # [module] the chin: the most forward midline vertex in the lower 40 percent of the span
#                               from the lowest midline face vertex to the nose tip
FOREHEAD_SHARE = 0.5          # [module] the forehead: the midline, halfway between the nose tip and the top of the head
CHEEK_X_SHARE = 0.5           # [module] the cheek: at the nose tip's height, half the head's half width off the midline
FACE_TOL_M = 0.006            # [module] the search band of the cheek and forehead vertices
CLEAR_M = 0.01                # [module] on the way the hand and forearm keep 1 cm from the trunk and the thighs (or
#                               their distance at rest, when the rest is closer)
N_BLEND = 16                  # [module] samples of the joint-space part of a touch path
BLEND_LEAD = 1.0              # [module] the joint-space part: the shoulder at the share u^lead of the blend parameter
#                               u, the elbow and wrist at 1 - (1 - u)^lead; at 1 every component on one share (a lead
#                               of 2, the first build, left most of the shoulder's turn to the end of that part, a
#                               flick near the face above the cap below; touch_joint_peaks reads it)
N_TOUCH_WAY = 9               # [module] IK steps to a first posture near the face (continuation, IK only)
N_LINE = 11                   # [module] IK samples of the approach line
N_CHECK = 81                  # [module] the path's clearance read at this many shares of its length
TOUCH_SMOOTH_WEIGHT = 1.0     # [module] the weight of the bending energy of a touch path's components along s
#                               (_smooth_touch, open-body wave 3; deg^2, PATH_SMOOTH_WEIGHT's value); 0 keeps the
#                               sample-by-sample path
TOUCH_REF_WEIGHT = 1.0        # [module] the joint-space part's samples held to their blend by this (per deg^2,
#                               BETWEEN's ref_weight), so that the smoothing moves the junction, not the blend
CLEAR_SEARCH_M = 0.03         # [module] exact point-to-triangle distances below this nearest-vertex distance
PARTS = {"head": (12, 15), "trunk": (0, 3, 6, 9), "thighs": (1, 2)}      # dominant joints of the checked surfaces
ELBOW = {"left": 18, "right": 19}


@functools.lru_cache(maxsize=16)
def face_vertex(name, side="right"):
    """The vertex of a face target on the rest mesh (module docstring, face touch): 'nose' (nose_vertex), 'chin',
    'forehead', 'cheek' (on side), among the vertices skinned mostly by the head in front of the head joint."""
    if name == "nose":
        return nose_vertex()
    if name not in FACE_TARGETS:
        raise ValueError(f"face target {name!r}")
    J, v, dom = H._rest()
    head = np.flatnonzero(dom == HEAD)
    front = head[v[head, 2] > J[HEAD, 2]]
    mid = front[np.abs(v[front, 0]) < 0.015]
    yn = v[nose_vertex(), 1]
    if name == "chin":
        y0 = v[mid, 1].min()
        sel = mid[v[mid, 1] < y0 + CHIN_SHARE * (yn - y0)]
    elif name == "forehead":
        y = yn + FOREHEAD_SHARE * (v[head, 1].max() - yn)
        sel = mid[np.abs(v[mid, 1] - y) < FACE_TOL_M]
    else:
        x = (1.0 if side == "left" else -1.0) * CHEEK_X_SHARE * np.abs(v[head, 0]).max()
        sel = front[(np.abs(v[front, 0] - x) < FACE_TOL_M) & (np.abs(v[front, 1] - yn) < FACE_TOL_M)]
    return int(sel[np.argmax(v[sel, 2])])


def _vertex_normals(V, faces):
    fn = np.cross(V[faces[:, 1]] - V[faces[:, 0]], V[faces[:, 2]] - V[faces[:, 0]])
    n = np.zeros_like(V)
    for k in range(3):
        np.add.at(n, faces[:, k], fn)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def face_point(body, name, side="right", knot=0):
    """(point (3,), outward normal (3,)) m: the face target's vertex on the skinned mesh of a body at a knot."""
    from .viewer import faces, knot_vertices
    V = knot_vertices(body, [knot])[0]
    i = face_vertex(name, side)
    return V[i].copy(), _vertex_normals(V, faces())[i]


def touch_dir(normal, target, side):
    """(3,) unit: the pointing direction at the touch (TOUCH_DIR_DEG): into the skin along the normal's horizontal
    part, turned about the vertical toward the side of the hand and tilted upward."""
    h = -np.array([normal[0], 0.0, normal[2]])
    h = h / np.linalg.norm(h)
    a = math.radians(TOUCH_DIR_DEG[target]["azimuth"]) * (1.0 if side == "left" else -1.0)
    h = np.array([h[0] * math.cos(a) + h[2] * math.sin(a), 0.0, -h[0] * math.sin(a) + h[2] * math.cos(a)])
    e = math.radians(TOUCH_DIR_DEG[target]["elevation"])
    return math.cos(e) * h + math.sin(e) * np.array([0.0, 1.0, 0.0])


@functools.lru_cache(maxsize=4)
def chain_on(side, posture):
    """The arm chain (_Chain) of one side on a base posture of TOUCH_POSTURES."""
    if posture not in TOUCH_POSTURES:
        raise ValueError(f"posture {posture!r}")
    return _Chain(side, posture)


def touch_bounds(side, posture):
    """(lo, hi) (12,) deg: the IK bounds of a touch: the SP1 table less LIMIT_MARGIN_DEG, widened to hold the rest
    posture (the right shoulder of 'sit' and 'stand' lies 0.4 deg inside the table), the collar held at rest."""
    ch = chain_on(side, posture)
    lo, hi = np.minimum(ch.lo, ch.q_sit), np.maximum(ch.hi, ch.q_sit)
    lo[0:3] = hi[0:3] = ch.q_sit[0:3]
    return lo, hi


def hand_blend(side, w):
    """(N, 30, 3, 3): the hand of side formed to the share w (N,) in [0, 1] of the pointing hand (POINT_POSE: each
    joint turned by w times its pose about the axes of hands.hand_rotations), the other hand at rest."""
    w = np.clip(np.asarray(w, dtype=np.float64).reshape(-1), 0.0, 1.0)
    fr = H.hand_frame()[side]["joints"]
    R = torch.eye(3, dtype=F64).repeat(len(w), H.N_HAND, 1, 1)
    wt = torch.as_tensor(w, dtype=F64)[:, None]
    for f, vals in POINT_POSE.items():
        for lv, val in enumerate(vals):
            flex, abd = (val, 0.0) if np.isscalar(val) else val
            j = H.hand_joint(side, f, lv)
            ax = fr[j]
            M = torch.eye(3, dtype=F64).repeat(len(w), 1, 1)
            if abd:
                M = axis_angle_to_matrix(wt * math.radians(abd) * torch.as_tensor(ax["abd"], dtype=F64)) @ M
            if flex:
                M = axis_angle_to_matrix(wt * math.radians(flex) * torch.as_tensor(ax["flex"], dtype=F64)) @ M
            R[:, H.hand_row(j)] = M
    return R


def hand_share(s):
    """(N,) the hand's share of the pointing pose at a path share s (minimum jerk from HAND_FORM_START to
    HAND_FORM_END, 0 before, 1 after)."""
    s = np.asarray(s, dtype=np.float64)
    return minimum_jerk((s - HAND_FORM_START) / (HAND_FORM_END - HAND_FORM_START))


def _arm_body(side, posture, q, w):
    """HandBody of N knots: the posture with the arm components q (N, 12) deg and the hand formed to w (N,)."""
    ch = chain_on(side, posture)
    q = np.asarray(q, dtype=np.float64).reshape(-1, 12)
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    if len(q) == 1:                                          # a Body has at least two knots
        q, w = np.repeat(q, 2, 0), np.repeat(w, 2)
    N = len(q)
    R = posture_rotations(posture).repeat(N, 1, 1, 1)
    Rq = axis_angle_to_matrix(torch.deg2rad(torch.as_tensor(q.reshape(N, 4, 3))))
    for i, j in enumerate(ARM[side]):
        R[:, j] = Rq[:, i]
    return H.HandBody(matrix_to_rot6d(R), ch.base.transl[:1].repeat(N, 1), ch.base.contact[:1].repeat(N, 1),
                      rate=ch.base.rate, hand6d=matrix_to_rot6d(hand_blend(side, w)))


# ---------------------------------------------------------------- surface distances
@functools.lru_cache(maxsize=2)
def _surface_sets(side):
    """{part: (vertex ids, faces (local indices into the ids))} of the moving arm ('hand': the wrist and the
    fingers; 'fore': the forearm) and of the checked surfaces (PARTS), from the dominant joints of the mesh."""
    from .viewer import dominant_joint, faces as mesh_faces
    faces = mesh_faces()
    dom = dominant_joint("neutral")
    groups = dict(PARTS, hand=(H.WRIST[side],) + tuple(range(H.FIRST[side], H.FIRST[side] + 15)),
                  fore=(ELBOW[side],))
    out = {}
    for name, joints in groups.items():
        ids = np.flatnonzero(np.isin(dom, joints))
        loc = -np.ones(len(dom), np.int64)
        loc[ids] = np.arange(len(ids))
        f = faces[(loc[faces] >= 0).all(1)]
        out[name] = (ids, loc[f])
    return out


def _closest_on_triangles(p, a, b, c):
    """(K, 3): the closest point to p on the triangles (a, b, c), all (K, 3) (Ericson 2005, section 5.1.5)."""
    ab, ac = b - a, c - a
    ap, bp, cp = p - a, p - b, p - c
    d1, d2 = (ab * ap).sum(-1), (ac * ap).sum(-1)
    d3, d4 = (ab * bp).sum(-1), (ac * bp).sum(-1)
    d5, d6 = (ab * cp).sum(-1), (ac * cp).sum(-1)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    with np.errstate(divide="ignore", invalid="ignore"):
        den = va + vb + vc
        out = a + ab * (vb / den)[:, None] + ac * (vc / den)[:, None]
        e = (d4 - d3) + (d5 - d6)
        m = (va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0)
        out = np.where(m[:, None], b + (c - b) * ((d4 - d3) / e)[:, None], out)
        m = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
        out = np.where(m[:, None], a + ac * (d2 / (d2 - d6))[:, None], out)
        m = (d6 >= 0) & (d5 <= d6)
        out = np.where(m[:, None], c, out)
        m = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
        out = np.where(m[:, None], a + ab * (d1 / (d1 - d3))[:, None], out)
        m = (d3 >= 0) & (d4 <= d3)
        out = np.where(m[:, None], b, out)
        m = (d1 <= 0) & (d2 <= 0)
        out = np.where(m[:, None], a, out)
    return out


def _signed_to_mesh(P, V, F):
    """(N,) m: the signed distance of points P (N, 3) to the triangle mesh (V, F), negative behind the closest
    triangle's outward face; exact below CLEAR_SEARCH_M (nearest vertex), the nearest-vertex distance above."""
    from scipy.spatial import cKDTree
    tree = cKDTree(V)
    d, _ = tree.query(P)
    out = d.copy()
    near = np.flatnonzero(d < CLEAR_SEARCH_M)
    if not len(near):
        return out
    cand = np.unique(np.concatenate(tree.query_ball_point(P[near], CLEAR_SEARCH_M + 0.01)))
    keep = np.zeros(len(V), bool)
    keep[cand] = True
    Fc = F[keep[F].any(1)]
    if not len(Fc):
        return out
    A, B, C = V[Fc[:, 0]], V[Fc[:, 1]], V[Fc[:, 2]]
    nrm = np.cross(B - A, C - A)
    n, m = len(near), len(Fc)
    p = np.repeat(P[near], m, 0)
    q = _closest_on_triangles(p, np.tile(A, (n, 1)), np.tile(B, (n, 1)), np.tile(C, (n, 1)))
    dist = np.linalg.norm(p - q, axis=1).reshape(n, m)
    k = np.argmin(dist, 1)
    i = np.arange(n) * m + k
    sgn = np.where(((p[i] - q[i]) * np.tile(nrm, (n, 1))[i]).sum(1) < 0.0, -1.0, 1.0)
    out[near] = sgn * dist[np.arange(n), k]
    return out


def clearance(body, side, knots=None, parts=("head", "trunk", "thighs"), chunk=16):
    """{part: {'hand': (K,), 'fore': (K,)}} m: at the knots (default all), the least signed distance between the
    hand (wrist and fingers) or the forearm of side and each part's surface (PARTS) of the skinned mesh, both ways
    (the arm's vertices to the part's triangles and the part's vertices to the arm's triangles); negative: inside."""
    sets = _surface_sets(side)
    k = np.arange(body.T) if knots is None else np.asarray(knots, dtype=np.int64).reshape(-1)
    names = ("hand", "fore") + tuple(parts)
    ids = np.concatenate([sets[n][0] for n in names])
    off = np.cumsum([0] + [len(sets[n][0]) for n in names])
    out = {p: {"hand": np.empty(len(k)), "fore": np.empty(len(k))} for p in parts}
    for a in range(0, len(k), chunk):
        X = H.body_points(body, ids, knots=k[a:a + chunk])
        for i in range(len(X)):
            S = {n: X[i, off[j]:off[j + 1]] for j, n in enumerate(names)}
            for p in parts:
                for arm in ("hand", "fore"):
                    d1 = _signed_to_mesh(S[arm], S[p], sets[p][1]).min()
                    d2 = _signed_to_mesh(S[p], S[arm], sets[arm][1]).min()
                    out[p][arm][a + i] = min(d1, d2)
    return out


# ---------------------------------------------------------------- the touch path
def _blend_shares(u):
    """(N, 12): the share of each arm component at the blend parameter u (N,) of the joint-space part (BLEND_LEAD:
    the shoulder u^lead, the elbow and wrist 1 - (1 - u)^lead; the collar is held by touch_bounds)."""
    u = np.asarray(u, dtype=np.float64)[:, None]
    prox, dist = u ** BLEND_LEAD, 1.0 - (1.0 - u) ** BLEND_LEAD
    return np.concatenate([np.repeat(prox, 6, 1), np.repeat(dist, 6, 1)], 1)


def _solve_touch(side, posture, target, q0, d, w_dir):
    lo, hi = touch_bounds(side, posture)
    return chain_on(side, posture).solve(target, q0, q0, point_dir=d, w_dir=w_dir, w_tip=TOUCH_POINT_WEIGHT,
                                         bounds=(lo, hi))


@functools.lru_cache(maxsize=16)
def touch_path(side, target, posture):
    """dict of the touch path of side to a face target from the rest of a posture (module docstring, face touch):
    's' (N,) path shares and 'q' (N, 12) deg arm components (the joint-space part then the approach line), the
    touch ('point_m', 'normal', 'dir', 'tip_m': the IK fingertip at the touch, 'gap_m' measured, 'dir_error_deg':
    the index's angle from 'dir', a soft term of the IK under the wrist's limits), the lengths, and the clearance read
    along the path ('check')."""
    if side not in SIDES or target not in FACE_TARGETS:
        raise ValueError(f"touch {side!r} {target!r}")
    ch = chain_on(side, posture)
    q0 = ch.q_sit.copy()
    p, n = face_point(ch.base, target, side)
    d = touch_dir(n, target, side)
    tip0 = ch.points(q0[None])["tip"][0].detach().numpy()
    goal = p - (SKIN_GAP_M + TOUCH_APPROACH_M) * d
    q = q0
    for si in np.linspace(0.0, 1.0, N_TOUCH_WAY)[1:]:        # a converged posture near the face (IK only)
        q = _solve_touch(side, posture, tip0 + si * (goal - tip0), q, d, TOUCH_POINT_WEIGHT * si ** 2)
    g, hist = SKIN_GAP_M, []
    for _ in range(8):                                         # the gap read on the skinned mesh (secant on g)
        q = _solve_touch(side, posture, p - g * d, q, d, TOUCH_POINT_WEIGHT)
        c = clearance(_arm_body(side, posture, q[None], [1.0]), side, [0], parts=("head",))["head"]["hand"][0]
        hist.append((g, c))
        if abs(c - SKIN_GAP_M) < SKIN_GAP_TOL_M:
            break
        if len(hist) > 1 and abs(hist[-1][1] - hist[-2][1]) > 1e-6:
            slope = (hist[-1][1] - hist[-2][1]) / (hist[-1][0] - hist[-2][0])
        else:
            slope = 1.0
        g = g + (SKIN_GAP_M - c) / max(slope, 0.2)
    if abs(hist[-1][1] - SKIN_GAP_M) >= SKIN_GAP_TOL_M:        # g moved after the last solve: q no longer matches
        raise ValueError(f"touch {side} {target} {posture}: the gap on the skin did not converge {hist}")
    q_touch = q
    p_touch = p - g * d
    line = [q_touch]
    for si in np.linspace(0.0, 1.0, N_LINE)[1:]:               # the approach line, solved outward from the touch
        line.append(_solve_touch(side, posture, p_touch - si * TOUCH_APPROACH_M * d, line[-1], d, TOUCH_POINT_WEIGHT))
    qb = np.stack(line[::-1])
    qa = q0 + _blend_shares(np.linspace(0.0, 1.0, N_BLEND)) * (qb[0] - q0)
    ta = ch.points(qa)["tip"].detach().numpy()
    la = float(np.linalg.norm(np.diff(ta, axis=0), axis=1).sum())
    lb = TOUCH_APPROACH_M
    sa = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(ta, axis=0), axis=1))]) / (la + lb)
    sb = (la + np.linspace(0.0, 1.0, N_LINE) * lb) / (la + lb)
    s = np.concatenate([sa, sb[1:]])
    qq = np.concatenate([qa, qb[1:]])
    line_pts = p_touch[None] - (1.0 - np.linspace(0.0, 1.0, N_LINE))[:, None] * TOUCH_APPROACH_M * d[None]
    qq = _smooth_touch(side, posture, s, qq, N_BLEND, line_pts, d)
    pt = ch.points(q_touch[None])
    res = float(1000.0 * np.linalg.norm(pt["tip"][0].detach().numpy() - p_touch))
    ui = (pt["tip"][0] - pt["index_mcp"][0]).detach().numpy()
    dir_err = float(np.degrees(np.arccos(np.clip(ui @ d / np.linalg.norm(ui), -1.0, 1.0))))
    out = {"side": side, "target": target, "posture": posture, "s": s, "q": qq, "point_m": p, "normal": n, "dir": d,
           "tip_m": p_touch, "back_m": float(g), "gap_m": float(hist[-1][1]), "gap_iterations": len(hist),
           "ik_residual_mm": res, "dir_error_deg": dir_err, "length_m": la + lb, "blend_length_m": la, "approach_m": lb,
           "q_rest_deg": q0, "q_touch_deg": q_touch, "vertex": face_vertex(target, side)}
    out["check"] = _path_check(out)
    return out


def _smooth_touch(side, posture, s, q, n_blend, line_pts, d, weight=None):
    """(N, 12) deg: the samples q (N, 12) of a touch path at the shares s (N,), solved again all together (open-body
    port, wave 3; module docstring, face touch): the joint-space part's inner samples held to their blend
    (TOUCH_REF_WEIGHT), the approach line's samples (from index n_blend - 1, the junction) on their fingertip points
    line_pts (N_LINE, 3) and pointing along d with the touch's weights (TOUCH_POINT_WEIGHT; the wrist near neutral, the
    elbow without its valgus, the elbow below the shoulder, as _Chain.solve), plus TOUCH_SMOOTH_WEIGHT times the
    bending energy of every component along s (the sum of (v_i+1 - v_i)^2 h^3 / m_i, v the divided first differences,
    m_i the mean of the two spacings, h their mean over the path: on equal spacings the plain squared second difference
    of _smooth_path; on the path's unequal spacings, 15 joint-space samples over most of s and 10 on the 5 cm line,
    the integral of the squared second derivative, so that the dense line is not under-weighted), inside touch_bounds,
    the rest (first) and the touch (last) kept."""
    from scipy.optimize import minimize
    w_s = TOUCH_SMOOTH_WEIGHT if weight is None else float(weight)
    n = len(s)
    if w_s <= 0 or n < 4:
        return q
    ch = chain_on(side, posture)
    lo, hi = touch_bounds(side, posture)
    s = np.asarray(s, dtype=np.float64)
    h = torch.as_tensor(np.diff(s))
    hm = 0.5 * (h[1:] + h[:-1])
    hbar = float(s[-1] - s[0]) / (n - 1)
    ref = torch.as_tensor(np.asarray(q, dtype=np.float64))
    q0, qn = ref[:1], ref[-1:]
    j0 = n_blend - 1                                       # the junction: the first sample of the approach line
    pts = torch.as_tensor(np.asarray(line_pts, dtype=np.float64))
    pd = torch.as_tensor(np.asarray(d, dtype=np.float64))
    m = n - 2

    def f(x):
        X = torch.tensor(x, dtype=F64, requires_grad=True)
        Q = torch.cat([q0, X.reshape(m, 12), qn], 0)
        loss = TOUCH_REF_WEIGHT * ((Q[1:j0] - ref[1:j0]) ** 2).sum()
        p = ch.points(Q[j0:])
        e = 100.0 * (p["tip"] - pts)                                                  # cm
        u = p["tip"] - p["index_mcp"]
        u = u / u.norm(dim=1, keepdim=True)
        up = torch.relu(100.0 * (p["elbow"][:, 1] - p["shoulder"][:, 1] + 0.05))
        loss = loss + TOUCH_POINT_WEIGHT * (e ** 2).sum() + TOUCH_POINT_WEIGHT * ((u - pd) ** 2).sum()             + 1e-2 * (Q[j0:, 9:12] ** 2).sum() + 1e-2 * (Q[j0:, 8] ** 2).sum() + 10.0 * (up ** 2).sum()
        v = (Q[1:] - Q[:-1]) / h[:, None]
        loss = loss + w_s * (((v[1:] - v[:-1]) ** 2) * (hbar ** 3 / hm)[:, None]).sum()
        loss.backward()
        return float(loss.detach()), X.grad.numpy().copy()
    x0 = np.clip(np.asarray(q, dtype=np.float64)[1:-1], lo, hi).reshape(-1)
    r = minimize(f, x0, jac=True, method="L-BFGS-B", bounds=list(zip(np.tile(lo, m), np.tile(hi, m))),
                 options={"maxiter": 5000, "ftol": 1e-15, "gtol": 1e-10})
    return np.concatenate([q[:1], r.x.reshape(m, 12), q[-1:]], 0)


def touch_q(path, s):
    """(N, 12) deg: the arm components at path shares s (N,) (a cubic spline through the path's samples, clipped to
    its bounds)."""
    lo, hi = touch_bounds(path["side"], path["posture"])
    return np.clip(_interp(path["s"], path["q"], np.clip(np.asarray(s, dtype=np.float64), 0.0, 1.0)), lo, hi)


def touch_rows(side, target, posture, s):
    """((N, 4, 3, 3) local rotations of the collar, shoulder, elbow and wrist of side, (N, 30, 3, 3) hand
    rotations) at path shares s (N,) of touch_path."""
    path = touch_path(side, target, posture)
    q = touch_q(path, s)
    R = axis_angle_to_matrix(torch.deg2rad(torch.as_tensor(q.reshape(-1, 4, 3))))
    return R, hand_blend(side, hand_share(s))


def touch_rise_s(path):
    """s: the reach of a touch path on a minimum-jerk profile whose fingertip peaks at TOUCH_TIP_SPEED_M_S of its
    posture (MJ_PEAK x length / speed; jerks.touch_timing puts it on the knot above)."""
    return MJ_PEAK * path["length_m"] / TOUCH_TIP_SPEED_M_S[path["posture"]]


def touch_joint_peaks(path, rise_s=None, n=401):
    """{'collar', 'shoulder', 'elbow', 'wrist'} deg/s: each arm joint's peak angular speed (the angle of its local
    rotation between successive samples) along the path on a minimum-jerk profile over rise_s (default
    touch_rise_s), n samples."""
    T = touch_rise_s(path) if rise_s is None else float(rise_s)
    q = touch_q(path, minimum_jerk(np.linspace(0.0, 1.0, n)))
    R = axis_angle_to_matrix(torch.deg2rad(torch.as_tensor(q.reshape(-1, 4, 3))))
    rel = R[:-1].transpose(-1, -2) @ R[1:]
    cos = ((rel.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0).clamp(-1.0, 1.0)
    w = np.degrees(torch.arccos(cos).numpy()).max(0) * (n - 1) / T
    return dict(zip(("collar", "shoulder", "elbow", "wrist"), (float(x) for x in w)))


def _path_check(path):
    """The clearance of a touch path read on the skinned mesh at N_CHECK shares (the hand formed by hand_share):
    the least distance to the head at the touch and along the way, to the trunk and the thighs along the way against
    the rest; the elbow's drop below the shoulder joint at the touch; the arm joints' peak angular speed on the
    path's timing (touch_joint_peaks). 'ok' when the touch lies in [0, SKIN_GAP_MAX_M], nothing enters the head, the
    trunk and thighs keep CLEAR_M (or their distance at rest), the elbow lies TOUCH_ELBOW_DROP_M or more below the
    shoulder (no salute) and no joint turns faster than TOUCH_JOINT_SPEED_CAP_DEG_S."""
    s = np.linspace(0.0, 1.0, N_CHECK)
    b = _arm_body(path["side"], path["posture"], touch_q(path, s), hand_share(s))
    c = clearance(b, path["side"])
    rest = {p: min(c[p]["hand"][0], c[p]["fore"][0]) for p in ("trunk", "thighs")}
    way = {p: np.minimum(c[p]["hand"], c[p]["fore"]) for p in c}
    touch = float(c["head"]["hand"][-1])
    ok_body = all(bool((way[p] >= min(CLEAR_M, rest[p]) - 1e-4).all()) for p in ("trunk", "thighs"))
    pt = chain_on(path["side"], path["posture"]).points(np.asarray(path["q_touch_deg"])[None])
    drop = float((pt["shoulder"][0, 1] - pt["elbow"][0, 1]).detach())
    peaks = touch_joint_peaks(path)
    return {"touch_gap_mm": 1000.0 * touch, "head_min_mm": 1000.0 * float(way["head"].min()),
            "head_min_at_s": float(s[int(np.argmin(way["head"]))]),
            "trunk_min_mm": 1000.0 * float(way["trunk"].min()), "trunk_rest_mm": 1000.0 * float(rest["trunk"]),
            "thighs_min_mm": 1000.0 * float(way["thighs"].min()), "thighs_rest_mm": 1000.0 * float(rest["thighs"]),
            "n_shares": N_CHECK, "clear_m": CLEAR_M, "elbow_drop_m": drop, "elbow_drop_min_m": TOUCH_ELBOW_DROP_M,
            "rise_s": touch_rise_s(path), "joint_peak_deg_s": peaks, "joint_cap_deg_s": TOUCH_JOINT_SPEED_CAP_DEG_S,
            "ok": bool(0.0 <= touch <= SKIN_GAP_MAX_M and way["head"].min() >= 0.0 and ok_body
                       and drop >= TOUCH_ELBOW_DROP_M and max(peaks.values()) <= TOUCH_JOINT_SPEED_CAP_DEG_S)}
