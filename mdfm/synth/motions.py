"""Base motions of the exam segments that the first exam set lacked (task SP5; docs/proposals/SYNTH_PATIENT_DRAFT.md,
section 2; reports/weekend_2026-10-01/tasks/SP4.md, open issues): procedural bodies with hands (hands.py) on the
selected body model (the open body by default; open-body port, wave 6: the mesh reads take the body model's
triangles and vertices), on the 100 Hz clock, in the shared neutral shape, joint floor at y = 0, every body joint
inside the joint-limit table (the anatomical table since wave 1) and every hand joint inside the hand limits of
hands.py (tests/test_synth_pathology.py).

Arm postures (ARM_POSTURES): local rotations of the collar, shoulder, elbow and wrist (axis-angle components in
degrees, left arm; the right arm is the mirror image, (x, -y, -z)), found once by a bounded search inside the SP1
table toward the segment directions of ARM_TARGETS (upper arm from shoulder to elbow, forearm from elbow to wrist,
hand from wrist to the middle knuckle, palmar normal; world directions with the body facing +z, x its left) and
kept as numbers; tests/test_synth_pathology.py checks that each still points where its targets say (arm_errors)
and lies inside the table:
- 'finger_tapping' (MDS-UPDRS 3.4): seated, the upper arm forward and down, the forearm semi-flexed forward and up,
  the wrist in line with the forearm, the hand at about shoulder height with the palm turned toward the midline
  (the posture of Paparella et al. 2023: arm at roughly shoulder height, forearm semi-flexed, wrist in line with
  the forearm; grades.SOURCES['paparella2023']);
- 'hand_movements' (3.5): seated, the elbow bent, the forearm up and the palm toward the examiner in front (the
  form's instruction, paraphrased in the draft);
- 'fold' (3.9): both arms folded across the chest (the form's instruction; the posture of Li et al. 2021), rebuilt
  by SH9 of docs/proposals/SYNTH_SHARED_CHANGES_DRAFT.md (lane fold, 2026-11-03; the version before passed the
  forearms through each other, its right hand entered the left upper arm, and at the forward lean of the arising
  its right forearm and hand entered the trunk and the right armrest of the arising chair, all measured by
  fold_report): the forearms one above the other in front of the lower chest, the left above
  and in front of the right, their skins at least FOLD_FOREARM_CLEAR_M apart, each hand (GRIP, a module hand pose,
  the thumb laid along the index) resting on or near the opposite upper arm (within FOLD_HAND_NEAR_M, outside it).
  Not mirrored: each side has its own numbers and its own ARM_TARGETS. Open-body port, wave 6: the joint targets
  carried from the SMPL+H rest by the chest they lie on (FOLD_SOLVE's comment), the thighs added to the rise's
  obstacles, GRIP's thumb CMC flexion lowered so that the thumb stays out of its own palm. Solved in the tree since task sh1c
  (solve_fold; the SH9 search had been kept outside it): a first stage (fold_ik), a bounded least squares inside
  the SP1 table (every component FOLD_SOLVE['margin_deg'], 1 deg, inside its bounds) from the hanging arms of
  'stand' toward the joint targets of FOLD_SOLVE (the elbows, wrists and middle knuckles from the spine3 joint, the
  palms, the collars: the targets of the SH9 search's last stage); then a second stage (_FoldProblem) from it under
  the clearances of FOLD_SOLVE on the skinned arms of the folded stand: the per-vertex distances of fold_report's
  rule (_one_way) between the two forearms, each hand and the other forearm, the two hands, each hand and the
  opposite upper arm (near it), the forearms and hands against the trunk of motions.arise built with the first
  stage's arms, at every second knot of the rise carried into the chest's frame (the arms ride on spine3; at the
  forward lean the lumbar flexion of the trunk bout lifts the abdomen toward the forearms, which is why the lower
  forearm sits at the lower end of the chest and not over the abdomen; the plane through the nearest trunk point),
  and against the boxes of the chair's arms over the rise; the result rounded to 0.01 deg. scripts/
  solve_fold_posture.py runs it and writes FOLD_RECORD (reports/synth/fold/fold_posture.json: the numbers, their
  directions, the solver's digest fold_solver_digest, the checks of the result on the mesh); this module reads
  ARM_POSTURES['fold'] and ARM_TARGETS['fold'] from it at import when the digest matches its code (else 'fold' is
  refused, _FOLD_STATE says why), and tests/test_synth_motions_fold.py solves again and compares. ARM_TARGETS['fold']
  records where the result points (fold_directions: a regression record, not the intent); the intent, the
  clearances, is what fold_report measures on the mesh (on the stand and over the rise) and what arise_validity
  carries for the arising.
  Opening the fold (task sh1c): an arm that leaves the fold passes by the other; both on one clock, the right hand
  (under the left forearm) passed through the left forearm on the way to the chair arms (parkinsonism.py, MDS-UPDRS
  3.9 grades 2 to 4) and in the unfolding of the paroxysmal kinesigenic clips (the try at lag 0 in the record of
  fold_open_lag gives the depth on the fold of the tree). Each arm now takes its own clock
  (fold_open_shares: one arm waits by a lag, a share of the opening, and each takes the rest), the lag the smallest of
  FOLD_OPEN_LAGS, with its side, whose rehearsal keeps the skins of FOLD_OPEN_CHECKS FOLD_OPEN_CLEAR_M apart
  (fold_open_lag: the fold's skin-gap check, fold_gaps, read over the opening at every knot, FOLD_OPEN_STEP);
  unfold_to_stand opens the fold to the hanging arms of 'stand' that way, and unfold_check reads the opening on the
  final body (whatever moves the arms afterwards included) at every knot and UNFOLD_CHECK_PAD_S after it: the validity
  'fold' of a clip that opens the fold.
Postures (POSTURES3): a posture of base.POSTURES with arm rows replaced and a hand pose per side ('rest', the
tasks' 'closed' hand, 'grip').

Outstretched arms (outstretched, the base motion of E2 and MDS-UPDRS 3.15): seated with the forearms on the arms of
the chair ('sit'), the arms raised to 'sit_arms_forward' (arms forward at shoulder height, palms down, wrists
straight) over RAISE_S by a minimum-jerk blend of each joint's rotation (geodesic), then held. The fingers apart, as
the form asks: the rest hand of SMPL+H kept them so; the open body's rest fingers lie 3.6 to 6.6 mm apart, and the
hand spreads with the raise (spread_pose: the index and ring knuckles abducted by the spread, the little finger's by
twice it) by the least of SPREAD_STEP_DEG steps at which every pair stands FINGERS_APART_M apart (outstretched_spread;
0 on SMPL+H); the smallest gap between adjacent fingers is reported (finger_gaps). The posture onset (end of the
raise) is in the info: a re-emergent tremor's latency counts from it.

Arising from a chair (arise, MDS-UPDRS 3.9, segment E7): seated on a seat of SEAT_HEIGHT_M (43 cm, the seat of Li
et al. 2021, who had their subjects rise with the arms folded on the chest) with the arms folded, sitting back upright
in the straight-backed chair of 3.9 (SEAT_TRUNK_DEG: the pelvis and the lumbar joint at their standing angles; the
slumped 'sit', pelvis 20 deg back and the lumbar joint 20 deg flexed, left the lumbar 2.5 deg of jarm2022's share,
open-body port, wave 6), rising in ARISE_S
(1.89 s, the controls of Inkster and Eng 2004 at a comfortable pace), then standing still. The phases and joint
excursions are those controls' (grades.SOURCES['inkster2004']): the buttocks leave the seat at LIFTOFF_SHARE (44.4
percent of the rise); before it the pelvis turns forward on the seat about the hips by HIP_FLEXION_DEG (17.0 deg,
their hip flexion displacement between pelvis and thigh), its own turn peaking at HIP_PEAK_SHARE (49.9 percent),
then the pelvis and the lumbar spine return to upright; after lift-off the shank tilts forward by ANKLE_DORSI_DEG
(7.5 deg) to its peak at ANKLE_PEAK_SHARE (58.7 percent) and back by ANKLE_PLANTAR_DEG (15.0 deg), each by a
minimum-jerk profile, while the hips rise from their seated height to the standing one (the knee then at REACH of
the leg length, base.py) by a minimum-jerk profile from lift-off, so they never sink back after it (review of
2026-10-05; the knee flexion follows from the hip height and the shank tilt in the sagittal plane, _leg_plane); the
legs are solved by two-bone inverse kinematics (ik.hinge_leg, the knee bending forward) and the feet stay flat and
still. The seated knee flexion is solved so that the lowest of the seat vertices (_seat_vertices: the buttocks and
the thighs behind the seat's front edge, SEAT_FRONT_M behind the knees) lies on the seat (and stays on it up to
lift-off: the hips are lowered per knot by what the forward turn of the pelvis lifts the buttocks, the contact rolling
onto the thighs, SEAT_ROLL_PASSES, and the pelvis turns with the thigh the lowered hips tilt, so that the cited hip
flexion displacement holds; open-body port, wave 6: from the upright seat the buttocks rose 8 mm off the seat before
lift-off), and the ankle heights are
corrected per knot by the skinned soles (contact.mesh_sole_low; pose blend shapes move the heel by millimetres), so
that the soles stay on the floor of a flat neutral sole. The sagittal leg plane (_leg_plane) does not see these
corrections (each side its own, ankle_height_correction_m in the info), and at the nearly straight standing knee
(REACH) a small change of the hip-to-ankle distance bends the knee and turns the shank by a large angle (the knee
angle's derivative with the distance grows without bound as the knee straightens), so the realized ankle
plantarflexion drifts from the cited one
whenever the soles' pose blend shapes change (SH9 fix of 2026-11-03: the folded arms of SH9 moved the soles and
the realized plantarflexion went out of the test's tolerance). The hips' standing height is therefore lowered by a
drop (positive down; the hips' rise, from lift-off by the same minimum-jerk profile, ends lower by it; their
forward position is the plane's) solved after each sole correction by the secant method inside DROP_RANGE_M so
that the realized ankle plantarflexion lies within DROP_TOL_DEG of ANKLE_PLANTAR_DEG (_solve_drop; the knee bends
more, the standing shank tilts further forward); the drop is a module rule, reported in the info
(standing_hip_drop_m). The realized seat height, phases and excursions, set beside the cited ones, are in the
info. The thigh turns from lift-off with the shank's cited
forward tilt, so the realized pelvis-to-thigh flexion peaks within a few hundredths of a second of lift-off, before
the cited 49.9 percent (realized_hip_peak_share; a pelvis turning as fast as the thigh at 49.9 percent would need
a displacement far above the cited 17.0 deg).

Balance (review of 2026-10-05; the version before it left the extrapolated centre of mass about 13 cm behind the
heels at lift-off, a body that falls back onto the seat). The centre of mass is the de Leva (1996) segment table on
the 22 joints (mdfm/kernels/force.py, com; sex code 2), and the extrapolated centre of mass of Hof et al. (2005;
grades.SOURCES['hof2005']) is XcoM = z + v / sqrt(g / l), z and v its forward position and velocity, l its height
above the lowest sole point (balance). Two module rules, no source read gives them:
- the standing shank tilt (the end of the rise) is solved so that the standing centre of mass lies as far ahead of
  the ankles as in the base stand with the arms folded (STAND_REF, the procedural stand of the standing clips); the
  seated shank then leans forward by that tilt plus ANKLE_PLANTAR_DEG - ANKLE_DORSI_DEG (the cited excursions
  unchanged): the feet are drawn back under the knees;
- a lumbar and thoracic flexion bout (TRUNK_JOINTS, shared in proportion to each joint's room from the seated
  trunk up to the table's bound, trunk_split) whose speed peaks at lift-off (it starts at TRUNK_START_SHARE) and whose
  flexion peaks with the hip's (TRUNK_PEAK_SHARE, the cited 49.9 percent), held through the momentum transfer and
  the first half of the hips' rise (to TRUNK_RETURN_SHARE), then returning to upright by the end of the rise; its
  amplitude is the smallest (1 deg grid, then bisection) that keeps the XcoM at least MOS_MIN_M ahead of the heels at
  every knot from lift-off (Hof's condition for dynamic stability: the XcoM inside the base of support). No thoracic
  or lumbar flexion value was sourced: the amplitude is this solve's, reported with the realized trunk lean and its
  angular speed (the pelvis-to-neck line against the vertical). Open-body port, wave 6: on the anatomical table the
  bout that peaked and returned at the shank's peak forward tilt needed the whole thoracolumbar room (45 deg) and
  still left the XcoM behind the heels at lift-off, and its return with the shank's fall drew the XcoM back to the
  heels in the hips' rise; peaking with the hip and held, it needs about half the room, and the folded forearms keep
  off the thighs (scripts/openbody_wave6_checks.py arise_balance).
The chair (arise_chair) is scene geometry.

The folded arms on the mesh (fold_report, SH9): the skin parts are the vertices of the neutral mesh by their dominant
joint (FOLD_PARTS: each forearm, each upper arm, each hand with the wrist and fingers, the trunk as pelvis and spine,
the thighs); two parts are measured by skin_gap, each vertex's exact distance to the other part's triangles (both
ways), a vertex inside the other part (winding number above one half: the parts are open where they meet the next
one, so the sign of the closest triangle's normal, reach._signed_to_mesh, would call some outside points inside)
counted negative; the chair by the deepest entry of the forearm and hand vertices into its boxes. arise_validity
reads it over the rise of the arising (the knots before and after the rise are still): 'armrest' and 'hand_thigh'
(the arms of the chair, the thighs), entries render_mesh_video.failed_checks reads, and 'fold' (the forearms, the
hands, the upper arms, the trunk), whose 'ok' is the hard rule alone (no skin part inside another: fold_report's
'entry_ok') and whose 'margins_ok' reports the margins (FOLD_FOREARM_CLEAR_M, FOLD_HAND_NEAR_M).
"""
import ast
import functools
import hashlib
import inspect
import json
import math
import os
import textwrap

import numpy as np
import torch

from .. import paths
from ..body.rotations import axis_angle_to_matrix, matrix_to_axis_angle, matrix_to_rot6d
from ..body.joints import fk
from . import hands as H
from .base import POSTURES, REACH, posture_rotations
from .body import F64, Body, contact_labels, floor_to_zero, limit_violations, rest_joints, skate

ARM_JOINTS = {"left": (13, 16, 18, 20), "right": (14, 17, 19, 21)}
# left arm (collar, shoulder, elbow, wrist): axis-angle components (deg); module docstring. 'fold' is read from its
# record (FOLD_RECORD, written by scripts/solve_fold_posture.py from solve_fold) at the end of this module
ARM_POSTURES = {      # finger_tapping and hand_movements solved again on the open body inside the anatomical table
    #                    (open-body port, wave 2: scripts/openbody_handtask_poses.py, reports/openbody/handtask_poses.json)
    "finger_tapping": {"left": ((-5.41, -19.5, -9.5), (-7.02, -37.81, -39.0), (-8.37, -57.18, 28.75),
                                (-7.41, 0.1, 2.37))},
    "hand_movements": {"left": ((-9.5, -13.91, -9.5), (-23.51, -9.61, -53.62), (4.68, -108.5, 55.7),
                                (32.07, -0.41, 1.5))},
}
ARM_TARGETS = {
    # world segment directions (body facing +z): the left arm's, mirrored for the right, the targets the postures
    # were solved toward; 'fold' (each side its own, the directions and wrists its solve realized) is read with it
    "finger_tapping": {"base": "sit", "upper": (0.25, -0.55, 0.8), "fore": (-0.1, 0.45, 0.89),
                       "hand": (-0.1, 0.45, 0.89), "palm": (-0.8, -0.5, 0.2)},
    "hand_movements": {"base": "sit", "upper": (0.35, -0.75, 0.55), "fore": (0.05, 0.95, 0.3),
                       "hand": (0.05, 0.95, 0.3), "palm": (0.0, -0.3, 0.95)},
}
FOLD_RECORD = ("reports", "synth", "fold", "fold_posture.json")   # the fold solve's record (module docstring)
_FOLD_STATE = {"state": "not read"}
# [module] the fold solve (solve_fold, module docstring), carried from the bounded search of lane fold (SH9, task
# sh1b), whose last stage aimed at these targets with these weights:
# - targets: per side the elbow (E), wrist (W) and middle knuckle (K) joints in m from the spine3 joint of 'stand'
#   (x the body's left, y up, z forward; that stage's targets carried to this frame and rounded to the millimetre),
#   the palmar normal's direction and the collar's local rotation (deg) the solve is drawn toward;
# - ik_weights: the first stage (joint targets only, from the hanging arms of 'stand'); polish_weights: the same
#   targets under the clearances in the second; elbow_z and wrist_x draw the elbow's varus and the wrist's twist to 0;
# - clear_m: the least skin gap of each pair (fold_report's pairs; 'trunk_*' a side's forearm and hand against the
#   trunk over the rise, 'armrest' against the chair's arms over the rise), near_m: the largest least gap of a pair
#   that must stay near (the forearms, each hand on the opposite upper arm); clear_weight: the hinges' weight;
# - margin_deg: every component this far inside its SP1 bounds; rise_step: knots between two reads of the rise;
#   crop_m: the trunk points kept within this of the first stage's arms; polish_nfev: the second stage's budget.
#   open-body port, wave 6: the joint targets carried from the SMPL+H rest to the open one by the chest they lie on
#   (scripts/openbody_wave6_checks.py fold_targets: each target's height below the shoulder line as a share of the
#   hips-to-shoulders height, its depth in front of the chest's front skin and its x kept; Anny's spine3 lies 18 cm
#   behind the chest's front skin, SMPL+H's 11 to 13, so the SMPL+H numbers lay inside the open body's chest, and its
#   shoulders stand 46.2 cm above the hips, SMPL+H's 54.0); the palms and collars kept
FOLD_SOLVE = {
    "targets": {"left": {"E": (0.157, -0.027, 0.254), "W": (-0.083, 0.011, 0.321), "K": (-0.173, 0.011, 0.266),
                         "palm": (-0.5, 0.0, -0.85), "collar": (0.0, -5.0, 0.0)},
                "right": {"E": (-0.193, -0.083, 0.202), "W": (0.057, -0.062, 0.261), "K": (0.157, -0.07, 0.222),
                          "palm": (0.5, 0.0, -0.85), "collar": (0.0, 5.0, 0.0)}},
    "ik_weights": {"E": 30.0, "W": 30.0, "K": 15.0, "palm": 0.5, "collar": 0.02, "elbow_z": 0.02, "wrist_x": 0.01},
    "polish_weights": {"E": 3.0, "W": 3.0, "K": 2.0, "palm": 0.1, "collar": 0.01, "elbow_z": 0.01,
                       "wrist_x": 0.005},
    "clear_m": {"forearms": 0.007, "hand_left_forearm_right": 0.004, "hand_right_forearm_left": 0.004,
                "hands": 0.004, "hand_left_upper_right": 0.002, "hand_right_upper_left": 0.002,
                "trunk_left": 0.004, "trunk_right": 0.004, "thigh_left": 0.004, "thigh_right": 0.004,
                "armrest": 0.02},
    "near_m": {"forearms": 0.012, "hand_left_upper_right": 0.010, "hand_right_upper_left": 0.010},
    "clear_weight": 30.0, "margin_deg": 1.0, "rise_step": 2, "crop_m": 0.06, "polish_nfev": 60,
}
GRIP = {"index": (45.0, 45.0, 20.0), "middle": (45.0, 45.0, 20.0), "ring": (45.0, 45.0, 20.0),
        "pinky": (45.0, 45.0, 20.0), "thumb": (-10.0, 30.0, 20.0)}
#   [module] fingers curled on the opposite arm (folded arms); the thumb (CMC, MCP, IP flexion) laid along the index
#   since SH9 (at rest it stood off the hand: on the SH9 arms it entered the chest); open-body port, wave 6: Anny's rest
#   thumb lies turned toward the palm (design 14.2), and the CMC flexion of SH9 (20 deg) put the thumb 8.1 mm into its
#   own palm; the CMC flexion the least of 10 deg steps below it at which the thumb's skin clears the palm and the
#   index and middle fingers (-10: 1.76 and 14.97 mm; scripts/openbody_wave6_checks.py, grip_thumb)
RAISE_S = 1.0                                    # [module] the raise of the arms to the 3.15 posture
FINGERS_APART_M = 0.005                          # [module] the 3.15 hand's adjacent fingers' skins at least this apart
SPREAD_STEP_DEG = 2.5                            # [module] the grid of the 3.15 hand's spread (outstretched_spread)
SEAT_HEIGHT_M = 0.43                             # SOURCES['li2021']
ARISE_S = 1.89                                   # SOURCES['inkster2004'], controls, comfortable pace
LIFTOFF_SHARE = 0.444                            # SOURCES['inkster2004'], preparation phase 44.4 percent
HIP_PEAK_SHARE = 0.499                           # SOURCES['inkster2004'], peak hip flexion at 49.9 percent
HIP_FLEXION_DEG = 17.0                           # SOURCES['inkster2004'], hip flexion displacement
ANKLE_DORSI_DEG = 7.5                            # SOURCES['inkster2004'], ankle dorsiflexion displacement
ANKLE_PEAK_SHARE = 0.587                         # SOURCES['inkster2004'], peak ankle dorsiflexion at 58.7 percent
TILT_MARGIN_DEG = 0.01                           # [module] the rise's dorsiflexion peak kept this far inside the table
#   (_standing_tilt; open-body port, wave 6 last pass: at the bound itself the controlled sit-down, reading the rise
#   between its knots, passed it by 1.4e-4 deg at one knot)
ANKLE_PLANTAR_DEG = 15.0                         # SOURCES['inkster2004'], ankle plantarflexion displacement
SEAT_ITERS = 20                                  # passes of the seated knee solve (buttocks on the seat)
SOLE_PASSES = 3                                  # passes of the skinned-sole correction of the ankle heights
SEAT_ROLL_PASSES = 2                             # passes of the seat contact's correction of the hip height (each
#                                                  sole pass), before lift-off
DROP_TOL_DEG = 0.01                              # [module] realized ankle plantarflexion within this of the cited one
DROP_RANGE_M = (-0.005, 0.01)                    # [module] the standing hip drop's search interval (positive down)
STAND_REF = "stand_arms_folded"                  # [module] the standing centre of mass ahead of the ankles as here
TRUNK_JOINTS = (3, 6, 9)                         # spine1, spine2, spine3: the lumbar and thoracic flexion bout
TRUNK_PEAK_SHARE = HIP_PEAK_SHARE                # [module] the bout's flexion peaks with the hip's (the cited 49.9
#                                                  percent; open-body port, wave 6: at the shank's peak forward tilt
#                                                  before, which on the anatomical table needed the whole thoracolumbar
#                                                  room and laid the folded forearms on the thighs; module docstring)
TRUNK_START_SHARE = 2 * LIFTOFF_SHARE - TRUNK_PEAK_SHARE   # [module] the bout's speed peaks at lift-off
TRUNK_RETURN_SHARE = LIFTOFF_SHARE + 0.5 * (1.0 - LIFTOFF_SHARE)   # [module] the bout returns to upright over the
#                                                  second half of the hips' rise from lift-off (open-body port, wave
#                                                  6: returning from the shank's peak, with the shank's fall, carried
#                                                  the open body's XcoM to 1.6 mm of the heels; module docstring)
SEAT_TRUNK_DEG = (0.0, 0.0)                      # [module] the arising's seated pelvis and lumbar (spine1) about x:
#                                                  upright, sitting back in the straight-backed chair of 3.9 (open-body
#                                                  port, wave 6: the slumped 'sit', pelvis -20 and lumbar 20, left the
#                                                  lumbar 2.5 deg of jarm2022's 22.5, module docstring)
MOS_MIN_M = 0.005                                # [module] the XcoM at least this far ahead of the heels after lift-off
GRAVITY_M_S2 = 9.81
SEAT_FRONT_M = 0.12                              # [module] the seat's front edge behind the knees (arise_chair)
FOLD_FOREARM_CLEAR_M = 0.005                     # [module] the least gap between the two folded forearms' skins
FOLD_HAND_NEAR_M = 0.015                         # [module] each folded hand's skin this near the opposite upper arm
FOLD_CHECK_STEP = 5                              # [module] knots between two reads of the fold checks (0.05 s)
FOLD_PARTS = {"fore_l": (18,), "fore_r": (19,), "up_l": (16,), "up_r": (17,),
              "hand_l": (20,) + tuple(range(22, 37)), "hand_r": (21,) + tuple(range(37, 52)),
              "trunk": (0, 3, 6, 9), "thighs": (1, 2)}   # dominant joints of the skin parts of fold_report


def minimum_jerk(x):
    """10 x^3 - 15 x^4 + 6 x^5 on [0, 1], clipped outside."""
    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)
    return x ** 3 * (10.0 - 15.0 * x + 6.0 * x ** 2)


def _mirror(aa):
    x, y, z = aa
    return (x, -y, -z)


def arm_rotations(name, side, fold=None):
    """{joint: (3, 3)} local rotations of an arm posture of ARM_POSTURES on one side (mirrored when only the left
    arm is stored); fold: the folded arms' numbers to use for 'fold' in place of ARM_POSTURES['fold'] (the solve,
    solve_fold)."""
    if name == "fold" and fold is not None:
        spec = fold
    elif name not in ARM_POSTURES:
        if name == "fold":
            raise RuntimeError(f"ARM_POSTURES['fold'] is not loaded: its record {'/'.join(FOLD_RECORD)} is "
                               f"{_FOLD_STATE['state']} (run scripts/solve_fold_posture.py)")
        raise KeyError(name)
    else:
        spec = ARM_POSTURES[name]
    vals = spec[side] if side in spec else [_mirror(a) for a in spec["left"]]
    return {j: axis_angle_to_matrix(torch.tensor(np.radians(a), dtype=F64)) for j, a in zip(ARM_JOINTS[side], vals)}


# ---------------------------------------------------------------- hand poses
def closed_pose(task, side):
    """{(side, finger, level): (flex, abd)} of the closed hand of a task: 'finger_tapping' the index tip on the
    thumb tip (handtasks.PINCH), 'hand_movements' the fist (handtasks.FIST), 'grip' the curled fingers (GRIP)."""
    from .handtasks import FIST, PINCH
    src = {"finger_tapping": PINCH, "hand_movements": FIST, "grip": {f: v for f, v in GRIP.items()}}[task]
    out = {}
    for f, vals in src.items():
        for lv, v in enumerate(vals):
            out[(side, f, lv)] = v
    return out


POSTURES3 = {
    "sit_finger_tapping": {"base": "sit", "arms": {"left": "finger_tapping"}, "hands": {"left": "finger_tapping"}},
    "sit_hand_movements": {"base": "sit", "arms": {"left": "hand_movements"}, "hands": {"left": "hand_movements"}},
    "sit_arms_folded": {"base": "sit", "arms": {"left": "fold", "right": "fold"},
                        "hands": {"left": "grip", "right": "grip"}},
    "stand_arms_folded": {"base": "stand", "arms": {"left": "fold", "right": "fold"},
                          "hands": {"left": "grip", "right": "grip"}},
}


def posture3(name, side="left", fold=None):
    """(R (22, 3, 3), hand rotations (30, 3, 3)) of a posture of POSTURES3; side: the side of the one-arm task
    postures (the arm and the hand pose move to the other side, the base keeps its own); fold: arm_rotations'."""
    p = POSTURES3[name]
    R = posture_rotations(p["base"])
    pose = {}
    for s0, arm in p["arms"].items():
        s = s0 if len(p["arms"]) == 2 else side
        for j, M in arm_rotations(arm, s, fold).items():
            R[j] = M
    for s0, hand in p["hands"].items():
        s = s0 if len(p["hands"]) == 2 else side
        pose.update(closed_pose(hand, s))
    return R, H.hand_rotations(pose)


def posture_body(name, duration_s, side="left", rate=100.0, fold=None):
    """A HandBody holding a posture of POSTURES3 or base.POSTURES (hands at rest) for duration_s (round(duration_s
    rate) + 1 knots): pelvis x and z at 0, the lowest foot joint on the floor y = 0, contacts by the body-store
    rule (as base.static_body); fold: arm_rotations'."""
    T = int(round(duration_s * rate)) + 1
    if name in POSTURES3:
        R, Rh = posture3(name, side, fold)
    elif name in POSTURES:
        R, Rh = posture_rotations(name), torch.eye(3, dtype=F64).repeat(H.N_HAND, 1, 1)
    else:
        raise ValueError(f"posture {name!r}")
    rest = rest_joints()
    tr = torch.zeros(T, 3, dtype=F64)
    tr[:, [0, 2]] = -rest[0, [0, 2]]
    b = H.HandBody(matrix_to_rot6d(R)[None].expand(T, -1, 6).clone(), tr, torch.zeros(T, 2, dtype=torch.bool),
                   rate=rate, hand6d=matrix_to_rot6d(Rh)[None].expand(T, -1, 6).clone())
    b = floor_to_zero(b)
    return H.HandBody(b.rot6d, b.transl, contact_labels(b), b.betas, b.gender, b.rate, b.hand6d)


def arm_errors(name, side="left"):
    """{feature: angle (deg) between the posture's segment direction and its target} of an arm posture (left
    targets mirrored for the right, or a side's own targets where the posture is not symmetric: the folded arms),
    and 'wrist_mm', the wrist joint's distance from its target where one is given: the check of ARM_TARGETS."""
    tg = ARM_TARGETS[name]
    base = tg["base"]
    own = side in tg
    if own:
        tg = tg[side]
    R = posture_rotations(base)
    for j, M in arm_rotations(name, side).items():
        R[j] = M
    rest = rest_joints()
    J, Gl = fk(R[None], torch.zeros(1, 3, dtype=F64), rest)
    J, Gl = J[0].numpy(), Gl[0].numpy()
    c, s, e, w = ARM_JOINTS[side]
    J52, _, _ = H._rest()
    mid = H.hand_joint(side, "middle", 0)
    feats = {"upper": J[e] - J[s], "fore": J[w] - J[e], "hand": Gl[w] @ (J52[mid] - J52[w]),
             "palm": Gl[w] @ H.hand_frame()[side]["palmar"]}
    out = {}
    if "wrist" in tg:
        wt = tg["wrist"] if own else tg["wrist"][side]
        out["wrist_mm"] = float(1000.0 * np.linalg.norm(J[w] - np.asarray(wt)))
    for k in ("upper", "fore", "hand", "palm"):
        if k not in tg:
            continue
        t = np.asarray(tg[k], dtype=np.float64)
        if side == "right" and not own:
            t = t * np.array([-1.0, 1.0, 1.0])
        a, b = feats[k] / np.linalg.norm(feats[k]), t / np.linalg.norm(t)
        out[k] = float(np.degrees(np.arccos(np.clip(a @ b, -1.0, 1.0))))
    return out


# ---------------------------------------------------------------- finger gaps
def finger_gaps(body, side="left", knot=0):
    """{pair: mm}: the smallest distance between the skin of adjacent fingers (vertices skinned mostly by their
    middle and distal joints) at a knot: the fingers apart of 3.15."""
    _, _, dom = H._rest()
    out = {}
    for a, c in (("index", "middle"), ("middle", "ring"), ("ring", "pinky")):
        ia = np.flatnonzero(np.isin(dom, [H.hand_joint(side, a, k) for k in (1, 2)]))
        ic = np.flatnonzero(np.isin(dom, [H.hand_joint(side, c, k) for k in (1, 2)]))
        P = H.body_points(body, np.r_[ia, ic], knots=[knot])[0]
        A, C = P[:len(ia)], P[len(ia):]
        out[f"{a}_{c}"] = float(1000.0 * np.sqrt(((A[:, None] - C[None]) ** 2).sum(-1)).min())
    return out


# ---------------------------------------------------------------- outstretched arms (3.15)
def _blend(R0, R1, w):
    """(T, n, 3, 3): R0 (n, 3, 3) turned toward R1 (n, 3, 3) along the geodesic by shares w (T,)."""
    D = matrix_to_axis_angle(R0.transpose(-1, -2) @ R1)                      # (n, 3)
    W = torch.as_tensor(np.asarray(w), dtype=F64)
    return R0[None] @ axis_angle_to_matrix(W[:, None, None] * D[None])


def spread_pose(deg):
    """{(side, finger, 0): (0, abduction)} of the 3.15 hand's spread by deg: the index and ring knuckles abducted by
    deg, the little finger's by twice it (each away from the middle finger, a fan), on both sides."""
    out = {}
    for side in H.SIDES:
        for f, k in (("index", 1.0), ("ring", 1.0), ("pinky", 2.0)):
            out[(side, f, 0)] = (0.0, k * float(deg))
    return out


@functools.lru_cache(maxsize=2)
def _outstretched_spread(name):
    rb = posture_body("sit_arms_forward", 0.01)
    tried = []
    for deg in np.arange(0.0, 0.5 * H.ABDUCTION_LIMIT_DEG + 1e-9, SPREAD_STEP_DEG):
        Rh = H.hand_rotations(spread_pose(deg))
        b = H.HandBody(rb.rot6d, rb.transl, rb.contact, rb.betas, rb.gender, rb.rate,
                       matrix_to_rot6d(Rh)[None].expand(rb.T, -1, 6).clone())
        g = {sd: finger_gaps(b, sd, 0) for sd in H.SIDES}
        low = min(min(v.values()) for v in g.values())
        tried.append({"spread_deg": float(deg), "least_gap_mm": low})
        if low >= 1000.0 * FINGERS_APART_M:
            return float(deg), {"tried": tried, "apart_mm": 1000.0 * FINGERS_APART_M}
    raise RuntimeError(f"no spread of the 3.15 hand keeps its fingers {1000.0 * FINGERS_APART_M} mm apart: {tried}")


def outstretched_spread():
    """(deg, record): the 3.15 hand's spread (spread_pose), the least of SPREAD_STEP_DEG steps up to half the
    knuckles' abduction bound at which every pair of adjacent fingers of both hands on 'sit_arms_forward' stands
    FINGERS_APART_M apart (finger_gaps); 0 where the rest hand holds them so (the SMPL+H rest hand). Open-body port,
    wave 6: Anny's rest fingers, laid straight along the hand's axis (design 3.2), stand 3.6 to 6.6 mm apart."""
    from .body import model_name
    return _outstretched_spread(model_name())


def outstretched(duration_s=30.0, rest_s=1.0, raise_s=RAISE_S, rate=100.0):
    """(HandBody, info): the base motion of E2 (module docstring): 'sit' for rest_s, the raise to
    'sit_arms_forward' over raise_s, the fingers spreading with it (outstretched_spread), then the hold to
    duration_s. info: the posture onset (s), the raise, the spread, the finger gaps (mm) of both hands in the hold,
    the joint-limit check."""
    T = int(round(duration_s * rate)) + 1
    t = np.arange(T) / rate
    w = minimum_jerk((t - rest_s) / raise_s)
    R0, R1 = posture_rotations("sit"), posture_rotations("sit_arms_forward")
    if not torch.equal(R0[:13], R1[:13]) or not torch.equal(R0[15], R1[15]):
        raise RuntimeError("sit and sit_arms_forward differ outside the arms")
    R = _blend(R0, R1, w)
    rest = rest_joints()
    tr = torch.zeros(T, 3, dtype=F64)
    tr[:, [0, 2]] = -rest[0, [0, 2]]
    b = Body(matrix_to_rot6d(R), tr, torch.zeros(T, 2, dtype=torch.bool), rate=rate)
    b = floor_to_zero(b)
    spread, srec = outstretched_spread()
    D = matrix_to_axis_angle(H.hand_rotations(spread_pose(spread)))                     # (30, 3), from rest
    Rh = axis_angle_to_matrix(torch.as_tensor(w, dtype=F64)[:, None, None] * D[None])
    b = H.HandBody(b.rot6d, b.transl, contact_labels(b), b.betas, b.gender, b.rate, matrix_to_rot6d(Rh))
    k_hold = int(round((rest_s + raise_s) * rate))
    info = {"posture": "sit_arms_forward", "from": "sit", "rest_s": float(rest_s), "raise_s": float(raise_s),
            "onset_s": k_hold / rate, "hold_s": (T - 1 - k_hold) / rate, "raise_source": "module",
            "finger_spread_deg": spread, "finger_spread": srec,
            "finger_gaps_mm": {s: finger_gaps(b, s, k_hold) for s in H.SIDES},
            "limits": limit_violations(b.rot6d)}
    return b, info


# ---------------------------------------------------------------- arising from a chair (3.9)
def _seat_vertices():
    """The vertices the seat carries: those skinned mostly by the pelvis or a hip (the buttocks and thighs) that lie,
    in the rest pose (thighs vertical), more than SEAT_FRONT_M above the higher knee joint, so behind the seat's
    front edge when seated (arise_chair); the thigh in front of that edge is not on the seat (review of
    2026-10-05: with the feet drawn back the thighs slope down to the knees, and the lowest thigh vertex near the
    knee had become the seat contact, ahead of the seat's front edge)."""
    _, v, dom = H._rest()
    r = rest_joints().numpy()
    return np.flatnonzero(np.isin(dom, (0, 1, 2)) & (v[:, 1] > max(r[4, 1], r[5, 1]) + SEAT_FRONT_M))


def _leg_plane(phi, kappa, L1, L2, ankle_h):
    """(y, z) of the hip midpoint for shank tilts phi (rad, forward positive) and knee flexions kappa (rad), the
    ankle at (ankle_h, 0) in the sagittal plane: knee = ankle + L2 (cos phi, sin phi), hip = knee + L1 (cos (phi -
    kappa), sin (phi - kappa)) as (y, z)."""
    ky, kz = ankle_h + L2 * np.cos(phi), L2 * np.sin(phi)
    return ky + L1 * np.cos(phi - kappa), kz + L1 * np.sin(phi - kappa)


def seat_trunk_deg():
    """{joint: deg about x} of the arising's seated trunk (SEAT_TRUNK_DEG: the pelvis 0, the lumbar spine1 3, the
    thoracic joints 6 and 9 at 'sit''s values)."""
    sit = np.degrees(matrix_to_axis_angle(posture_rotations("sit")).numpy())
    out = {j: float(sit[j][0]) for j in TRUNK_JOINTS}
    out[0], out[3] = float(SEAT_TRUNK_DEG[0]), float(SEAT_TRUNK_DEG[1])
    return out


def trunk_split():
    """((3,) shares, (3,) rooms in deg) of the trunk flexion bout over TRUNK_JOINTS: each joint's room about x from
    its seated value (seat_trunk_deg) up to the table's upper bound, over their sum (module docstring)."""
    from .body import joint_limits
    lim = joint_limits()
    seat = seat_trunk_deg()
    room = np.array([lim.hi[j][0] - seat[j] for j in TRUNK_JOINTS])
    return room / room.sum(), room


def trunk_profile(u):
    """(T,) the share of the trunk flexion bout at the shares u of the rise: a minimum-jerk rise from
    TRUNK_START_SHARE to TRUNK_PEAK_SHARE (its speed peaking at lift-off), held to TRUNK_RETURN_SHARE, then a
    minimum-jerk return to 0 at the end of the rise (module docstring)."""
    up = minimum_jerk((u - TRUNK_START_SHARE) / (TRUNK_PEAK_SHARE - TRUNK_START_SHARE))
    return up - minimum_jerk((u - TRUNK_RETURN_SHARE) / (1.0 - TRUNK_RETURN_SHARE))


def _arise_body(u, kappa0, Rarm, Rh, rate, dy=None, stand_tilt_deg=0.0, trunk_deg=0.0, hip_drop_m=0.0,
                seat_drop=None):
    """The body of the rise (module docstring) for the shares u (T,) of the rise, the seated knee flexion kappa0
    (rad), per-knot ankle height corrections dy (T, 2) m (the skinned sole, arise), the standing shank tilt
    stand_tilt_deg, the amplitude trunk_deg of the trunk flexion bout, the standing hip drop hip_drop_m (m,
    positive down, taken from lift-off by the hips' rise profile; the hips' forward position unchanged) and the
    seat contact's hip drop seat_drop (T,) m (positive down) at the knots up to lift-off (the hips' rise then starts
    from the lift-off knot's)."""
    from .ik import hinge_leg
    T = len(u)
    rest = rest_joints()
    r = rest.numpy()
    mid_off = (r[1] + r[2]) / 2 - r[0]
    L1, L2 = float(np.linalg.norm(r[4] - r[1])), float(np.linalg.norm(r[7] - r[4]))
    ankle_h = float(r[7, 1] - min(r[10, 1], r[11, 1]))
    kap_end = math.acos(((REACH * (L1 + L2)) ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2))
    phi0 = math.radians(stand_tilt_deg + ANKLE_PLANTAR_DEG - ANKLE_DORSI_DEG)      # the seated shank
    v = np.clip((u - LIFTOFF_SHARE) / (1.0 - LIFTOFF_SHARE), 0.0, 1.0)
    rise = minimum_jerk((u - LIFTOFF_SHARE) / (ANKLE_PEAK_SHARE - LIFTOFF_SHARE))
    fall = minimum_jerk((u - ANKLE_PEAK_SHARE) / (1.0 - ANKLE_PEAK_SHARE))
    phi = phi0 + math.radians(ANKLE_DORSI_DEG) * rise - math.radians(ANKLE_PLANTAR_DEG) * fall
    y_seat, _ = _leg_plane(phi0, kappa0, L1, L2, ankle_h)
    y_end, _ = _leg_plane(math.radians(stand_tilt_deg), kap_end, L1, L2, ankle_h)
    y = y_seat + (y_end - y_seat) * minimum_jerk(v)          # the hips rise from lift-off, never sink back
    roll = np.zeros(T)                                       # the knee's added flexion of the seat contact (rad)
    if seat_drop is not None:                                # the pelvis rolling on the seat before lift-off
        sd = np.asarray(seat_drop, dtype=np.float64)
        pre = u <= LIFTOFF_SHARE
        y0 = y_seat - (float(sd[pre][np.argmax(u[pre])]) if pre.any() else 0.0)
        y = np.where(pre, y_seat - sd, y0 + (y_end - y0) * minimum_jerk(v))
    cos_t = (y - ankle_h - L2 * np.cos(phi)) / L1
    if (np.abs(cos_t) > 1.0).any():
        raise ValueError("the hip height cannot be reached with this shank tilt")
    kappa = phi + np.arccos(cos_t)                         # the thigh leans back from the knee: phi - kappa < 0
    My, Mz = _leg_plane(phi, kappa, L1, L2, ankle_h)
    if seat_drop is not None and pre.any():                 # the lowered hips turn the thigh (the knee flexes):
        dk = kappa - kappa0                                  # the pelvis turns with it, so that the hip flexion
        dL = float(dk[pre][np.argmax(u[pre])])               # keeps the cited displacement, faded out after lift-off
        roll = np.where(pre, dk, dL * (1.0 - minimum_jerk(v)))
    seat = seat_trunk_deg()
    pel0, sp0 = seat[0], seat[3]
    pk = minimum_jerk(u / HIP_PEAK_SHARE)
    back = minimum_jerk((u - HIP_PEAK_SHARE) / (1.0 - HIP_PEAK_SHARE))
    pitch = pel0 + HIP_FLEXION_DEG * pk - (pel0 + HIP_FLEXION_DEG) * back      # the pelvis about x, deg
    pitch = pitch - np.degrees(roll)
    split, _ = trunk_split()
    bout = float(trunk_deg) * trunk_profile(u)
    spine = {3: sp0 * (1.0 - back) + split[0] * bout, 6: seat[6] * (1.0 - back) + split[1] * bout,
             9: seat[9] * (1.0 - back) + split[2] * bout}
    M = torch.as_tensor(np.stack([np.zeros(T), My - float(hip_drop_m) * minimum_jerk(v), Mz], -1))
    R = Rarm[None].repeat(T, 1, 1, 1)
    z = np.zeros(T)
    G0 = axis_angle_to_matrix(torch.as_tensor(np.stack([np.radians(pitch), z, z], -1)))
    R[:, 0] = G0
    for j, deg in spine.items():
        R[:, j] = axis_angle_to_matrix(torch.as_tensor(np.stack([np.radians(deg), z, z], -1)))
    P = M - torch.einsum("tij,j->ti", G0, torch.as_tensor(mid_off))
    fwd = torch.tensor([0.0, 0.0, 1.0], dtype=F64).expand(T, 3)
    dy = np.zeros((T, 2)) if dy is None else dy
    for i, (h, kn, a) in enumerate(((1, 4, 7), (2, 5, 8))):
        Hh = P + torch.einsum("tij,j->ti", G0, rest[h] - rest[0])
        Aa = torch.as_tensor(np.stack([np.full(T, r[a, 0]), ankle_h + dy[:, i], np.zeros(T)], -1))
        G_th, kap = hinge_leg(Hh, Aa, Hh + fwd, rest[kn] - rest[h], rest[a] - rest[kn], G0,
                              hinge=torch.cross(fwd, Aa - Hh, dim=-1))
        Rk = axis_angle_to_matrix(torch.stack([kap, 0 * kap, 0 * kap], -1))
        R[:, h] = G0.transpose(-1, -2) @ G_th
        R[:, kn] = Rk
        R[:, a] = (G_th @ Rk).transpose(-1, -2)                              # the foot flat, along +z
    b = H.HandBody(matrix_to_rot6d(R), P - rest[0], torch.zeros(T, 2, dtype=torch.bool), rate=rate,
                   hand6d=matrix_to_rot6d(Rh)[None].expand(T, -1, 6).clone())
    return b, {"pitch_deg": pitch, "spine_deg": spine[3], "phi_deg": np.degrees(phi), "kappa_deg": np.degrees(kappa),
               "ankle_h": ankle_h, "L": (L1, L2), "stand_tilt_deg": float(stand_tilt_deg),
               "seated_tilt_deg": float(math.degrees(phi0)), "trunk_deg": float(trunk_deg),
               "hip_drop_m": float(hip_drop_m)}


def balance(body, k_ref):
    """{com (T, 3), com_v (T, 3), xcom_z (T,), heel_z_m, toe_z_m, floor_y_m, posterior (T,), anterior (T,)} of a body
    whose feet stay still (module docstring): the centre of mass (de Leva table on the 22 joints, mdfm/kernels/
    force.py) and its velocity (central differences on the knots), the extrapolated centre of mass of Hof et al. 2005
    (XcoM = z + v / sqrt(g / l), l the height of the centre of mass above the lowest sole point), the base of support
    from the heel to the toe of the sole clouds (contact.sole_vertices) at knot k_ref, and the XcoM's margins behind
    the toe and ahead of the heel (m; negative outside)."""
    from ..kernels.force import com
    from .contact import sole_vertices
    J = body.joints().numpy()
    c = com(J, int(body.gender))
    v = np.gradient(c, 1.0 / body.rate, axis=0)
    sv = sole_vertices(int(body.gender))
    P = H.body_points(body, np.r_[sv["left"], sv["right"]], knots=[k_ref])[0]
    heel, toe, floor = float(P[:, 2].min()), float(P[:, 2].max()), float(P[:, 1].min())
    x = c[:, 2] + v[:, 2] / np.sqrt(GRAVITY_M_S2 / (c[:, 1] - floor))
    return {"com": c, "com_v": v, "xcom_z": x, "heel_z_m": heel, "toe_z_m": toe, "floor_y_m": floor,
            "posterior": x - heel, "anterior": toe - x}


def com_ahead_of_ankles(body, knot):
    """m: the centre of mass's forward position (de Leva table) minus the mean of the two ankle joints' at a knot."""
    from ..kernels.force import com
    J = body.joints().numpy()[knot]
    return float(com(J, int(body.gender))[2] - 0.5 * (J[7, 2] + J[8, 2]))


def _standing_tilt(Rarm, Rh, rate, fold=None):
    """(tilt deg, reference m): the standing shank tilt whose standing centre of mass lies as far ahead of the
    ankles as in STAND_REF (module docstring), by bisection (the centre of mass moves forward with the tilt), at most
    the tilt whose rise peaks TILT_MARGIN_DEG inside the anatomical table's dorsiflexion (the peak's tilt is the
    standing one plus ANKLE_PLANTAR_DEG; open-body port, wave 6 last pass: with 'stand''s soft knees,
    base.STAND_REACH, the reference tilt became 5.0069 deg and the rise's peak 0.0069 deg past jarm2022's 20); fold:
    arm_rotations'."""
    ref = com_ahead_of_ankles(posture_body(STAND_REF, 1.0 / rate, rate=rate, fold=fold), 0)

    def ahead(tilt):
        b, _ = _arise_body(np.ones(2), math.radians(90.0), Rarm, Rh, rate, None, tilt, 0.0)
        return com_ahead_of_ankles(b, 0)
    lo, hi = -10.0, 30.0
    if not ahead(lo) < ref < ahead(hi):
        raise RuntimeError("the standing shank tilt of the reference stand lies outside -10 to 30 deg")
    for _ in range(50):
        m = 0.5 * (lo + hi)
        lo, hi = (m, hi) if ahead(m) < ref else (lo, m)
    from .body import joint_limits
    top = -float(joint_limits().lo[7][0]) - ANKLE_PLANTAR_DEG - TILT_MARGIN_DEG
    return min(0.5 * (lo + hi), top), ref


def _solve_trunk(u, kappa0, Rarm, Rh, rate, dy, tilt, k_lift, drop=0.0, hint=None, seat_drop=None):
    """The smallest trunk bout amplitude (deg; a 1 deg grid up to the sum of the joints' rooms, then bisection)
    whose XcoM stays at least MOS_MIN_M ahead of the heels at every knot of u from k_lift (module docstring), with
    the standing hip drop drop (m). hint: the amplitude of an earlier pass of arise, whose grid neighbours are read
    first (the margin grows with the amplitude, so the result is the full search's when they bracket it)."""
    _, room = trunk_split()
    grid = [0.0] + [float(f) for f in np.arange(1.0, float(room.sum()) + 1e-9, 1.0)]

    def margin(f):
        b, _ = _arise_body(u, kappa0, Rarm, Rh, rate, dy, tilt, f, drop, seat_drop)
        return float(balance(b, k_lift)["posterior"][k_lift:].min())

    def smallest(i):                         # grid[i - 1] fails and grid[i] holds: bisection between them
        lo, hi = grid[i - 1], grid[i]
        for _ in range(30):
            m = 0.5 * (lo + hi)
            lo, hi = (lo, m) if margin(m) >= MOS_MIN_M else (m, hi)
        return hi
    if hint is not None and hint > 0.0:
        i = int(np.searchsorted(grid, float(hint)))          # grid[i - 1] < hint <= grid[i]
        if 0 < i < len(grid) and margin(grid[i - 1]) < MOS_MIN_M and margin(grid[i]) >= MOS_MIN_M:
            return smallest(i)
    if margin(0.0) >= MOS_MIN_M:
        return 0.0
    for i in range(1, len(grid)):
        if margin(grid[i]) >= MOS_MIN_M:
            return smallest(i)
    raise RuntimeError("no trunk flexion inside the joint-limit table keeps the XcoM inside the base of support")


def _dorsiflexion(c):
    """(T,) deg: the mean of the two ankles' dorsiflexion (-x, base.py) from axis-angle components c (T, 22, 3)."""
    return -0.5 * (c[:, 7, 0] + c[:, 8, 0])


def _solve_drop(u, kappa0, Rarm, Rh, rate, dy, tilt, trunk, k_on, k_end, d0=0.0, seat_drop=None):
    """m: the standing hip drop (module docstring) whose realized ankle plantarflexion (the dorsiflexion's peak over
    the rise minus its value at the end, as in the info) lies within DROP_TOL_DEG of ANKLE_PLANTAR_DEG: the secant
    method from d0 inside DROP_RANGE_M (the excursion shrinks as the drop grows); the closest drop found when none
    is within."""
    from .body import components_deg
    lo, hi = DROP_RANGE_M

    def err(d):
        b, _ = _arise_body(u, kappa0, Rarm, Rh, rate, dy, tilt, trunk, d, seat_drop)
        dors = _dorsiflexion(components_deg(b.rot6d))
        return float(dors[k_on:k_end + 1].max() - dors[k_end]) - ANKLE_PLANTAR_DEG
    x0 = float(np.clip(d0, lo, hi))
    f0 = err(x0)
    if abs(f0) <= DROP_TOL_DEG:
        return x0
    x1 = x0 + 5e-4 if x0 + 5e-4 <= hi else x0 - 5e-4          # a first step of half a millimetre
    f1 = err(x1)
    best = min((abs(f0), x0), (abs(f1), x1))
    for _ in range(12):
        if best[0] <= DROP_TOL_DEG or f1 == f0:
            break
        x0, f0, x1 = x1, f1, float(np.clip(x1 - f1 * (x1 - x0) / (f1 - f0), lo, hi))
        f1 = err(x1)
        best = min(best, (abs(f1), x1))
    return best[1]


def arise(seated_s=1.0, standing_s=15.0, rise_s=ARISE_S, seat_height_m=SEAT_HEIGHT_M, rate=100.0,
          stand_tilt_deg=None, trunk_deg=None, fold=None):
    """(HandBody, info): the base motion of the arising of E7 (module docstring): seated_s seated, the rise over
    rise_s, standing_s standing still, arms folded throughout. stand_tilt_deg and trunk_deg replace the solved
    standing shank tilt and trunk bout amplitude when given (0 and 0: no trunk bout and an upright standing shank,
    the balance of the version before the review of 2026-10-05; the standing hip drop is solved in every case).
    fold: the folded arms' numbers in place of ARM_POSTURES['fold'] (arm_rotations; the fold solve reads the rise
    of its first stage). info: the cited inputs and the realized seat height, phases and joint excursions
    (realized_*), the balance, the trunk bout, the standing hip drop, the limits and the feet."""
    from .contact import mesh_sole_low
    from .viewer import mesh_floor_y
    T = int(round((seated_s + rise_s + standing_s) * rate)) + 1
    t = np.arange(T) / rate
    u = np.clip((t - seated_s) / rise_s, 0.0, 1.0)
    k_on = int(round(seated_s * rate))
    k_end = int(round((seated_s + rise_s) * rate))
    k_lift = k_on + int(round(LIFTOFF_SHARE * rise_s * rate))
    Rarm, Rh = posture3("sit_arms_folded", fold=fold)
    r = rest_joints().numpy()
    L1, L2 = float(np.linalg.norm(r[4] - r[1])), float(np.linalg.norm(r[7] - r[4]))
    ankle_h = float(r[7, 1] - min(r[10, 1], r[11, 1]))
    if stand_tilt_deg is None:
        tilt, ref = _standing_tilt(Rarm, Rh, rate, fold)
    else:
        tilt, ref = float(stand_tilt_deg), None
    phi_seat = math.radians(tilt + ANKLE_PLANTAR_DEG - ANKLE_DORSI_DEG)
    target = mesh_floor_y(0.0) + seat_height_m
    seat_ids = _seat_vertices()
    sole_target = mesh_floor_y(0.0)
    dy = np.zeros((T, 2))
    kappa0 = math.radians(80.0)
    trunk = 0.0 if trunk_deg is None else float(trunk_deg)
    drop = 0.0
    w = slice(0, k_end + 2)                 # the knots of the trunk solve: the standing knots after it are still
    sd = np.zeros(T)                        # the seat contact's hip drop before lift-off
    pre = np.flatnonzero(u <= LIFTOFF_SHARE)
    for _ in range(SOLE_PASSES):            # the seated knee (buttocks on the seat), the trunk, the soles on the floor
        for _ in range(SEAT_ITERS):
            b0, _ = _arise_body(np.zeros(2), kappa0, Rarm, Rh, rate, dy[:2], tilt, trunk)
            low = float(H.body_points(b0, seat_ids, knots=[0])[0][:, 1].min())
            if abs(target - low) < 1e-7:
                break
            y0, _ = _leg_plane(phi_seat, kappa0, L1, L2, ankle_h)
            y1, _ = _leg_plane(phi_seat, kappa0 + 1e-4, L1, L2, ankle_h)
            kappa0 += (target - low) / ((y1 - y0) / 1e-4)
        for _ in range(SEAT_ROLL_PASSES):    # the lowest seat vertex kept on the seat up to lift-off
            bs, _ = _arise_body(u[pre], kappa0, Rarm, Rh, rate, dy[pre], tilt, trunk, drop, sd[pre])
            sd[pre] += H.body_points(bs, seat_ids)[..., 1].min(1) - target
        drop = _solve_drop(u[w], kappa0, Rarm, Rh, rate, dy[w], tilt, trunk, k_on, k_end, drop, sd[w])
        if trunk_deg is None:
            trunk = _solve_trunk(u[w], kappa0, Rarm, Rh, rate, dy[w], tilt, k_lift, drop, trunk, sd[w])
        b, geo = _arise_body(u, kappa0, Rarm, Rh, rate, dy, tilt, trunk, drop, sd)
        dy = dy + (sole_target - mesh_sole_low(b))
    # the last sole correction moved the ankles by micrometres: the drop and the trunk solved again
    drop = _solve_drop(u[w], kappa0, Rarm, Rh, rate, dy[w], tilt, trunk, k_on, k_end, drop, sd[w])
    if trunk_deg is None:
        trunk = _solve_trunk(u[w], kappa0, Rarm, Rh, rate, dy[w], tilt, k_lift, drop, trunk, sd[w])
    b, geo = _arise_body(u, kappa0, Rarm, Rh, rate, dy, tilt, trunk, drop, sd)
    b = H.HandBody(b.rot6d, b.transl, contact_labels(b), b.betas, b.gender, b.rate, b.hand6d)
    geo["ankle_dy_m"] = [float(dy.min()), float(dy.max())]
    geo["seat_drop_m"] = [float(sd.min()), float(sd.max())]
    geo["stand_ref_ahead_m"] = ref
    geo["solved"] = {"stand_tilt": stand_tilt_deg is None, "trunk": trunk_deg is None}
    return b, _arise_info(b, t, u, geo, seated_s, rise_s, standing_s, seat_height_m, target, seat_ids)


def _arise_info(b, t, u, geo, seated_s, rise_s, standing_s, seat_height_m, target, seat_ids):
    from .body import components_deg
    c = components_deg(b.rot6d)
    J = b.joints().numpy()
    k_on = int(round(seated_s * b.rate))
    k_end = int(round((seated_s + rise_s) * b.rate))
    k_lift = k_on + int(round(LIFTOFF_SHARE * rise_s * b.rate))
    hip = 0.5 * (c[:, 1, 0] + c[:, 2, 0])                          # hip flexion is -x (base.py): flexion = -hip
    knee = 0.5 * (c[:, 4, 0] + c[:, 5, 0])
    flex = -hip
    seg = slice(k_on, k_end + 1)
    k_hpk = k_on + int(np.argmax(flex[seg]))
    dors = _dorsiflexion(c)
    k_apk = k_on + int(np.argmax(dors[seg]))
    seat_low = H.body_points(b, seat_ids, knots=list(range(0, k_lift + 1)))[..., 1].min(1)
    from .contact import mesh_sole_low
    sole = mesh_sole_low(b)
    tv = J[:, 12] - J[:, 0]                                        # the pelvis-to-neck line, forward lean positive
    lean = np.degrees(np.arctan2(tv[:, 2], tv[:, 1]))
    k_tpk = k_on + int(np.argmax(lean[seg]))
    bal = balance(b, k_lift)
    post, ant = bal["posterior"], bal["anterior"]
    k_min = k_lift + int(np.argmin(post[k_lift:]))
    split, room = trunk_split()
    ok = bool(post[k_lift:].min() >= MOS_MIN_M - 1e-9 and ant[k_lift:].min() > 0.0)
    return {"sources": {"seat_height": "li2021", "duration_and_phases": "inkster2004",
                        "balance": ["hof2005", "deleva1996"], "trunk_bout": "module", "standing_shank_tilt": "module",
                        "standing_hip_drop": "module"},
            "seat_height_m": float(seat_height_m), "rise_s": float(rise_s), "seated_s": float(seated_s),
            "standing_s": float(standing_s), "onset_s": k_on / b.rate, "end_s": k_end / b.rate,
            "liftoff_share": LIFTOFF_SHARE, "hip_peak_share": HIP_PEAK_SHARE, "hip_flexion_deg": HIP_FLEXION_DEG,
            "seated_shank_tilt_deg": geo["seated_tilt_deg"], "standing_shank_tilt_deg": geo["stand_tilt_deg"],
            "ankle_dorsiflexion_deg": ANKLE_DORSI_DEG,
            "ankle_peak_share": ANKLE_PEAK_SHARE, "ankle_plantarflexion_deg": ANKLE_PLANTAR_DEG,
            "ankle_height_correction_m": geo["ankle_dy_m"], "standing_hip_drop_m": geo["hip_drop_m"],
            "seat_contact_hip_drop_m": geo["seat_drop_m"],
            "arms": "folded across the chest (motions.ARM_POSTURES['fold'], hands GRIP)",
            "realized_seat_height_m": float(seat_low[:k_on + 1].min() - (target - seat_height_m)),
            "seat_residual_m": [float(seat_low.min() - target), float(seat_low.max() - target)],
            "realized_liftoff_s": k_lift / b.rate,
            "realized_hip_flexion_displacement_deg": float(flex[seg].max() - flex[k_on]),
            "realized_hip_peak_share": float((k_hpk - k_on) / (k_end - k_on)),
            "realized_hip_extension_deg": float(flex[seg].max() - flex[k_end]),
            "realized_knee_extension_deg": float(knee[k_on] - knee[k_end]),
            "realized_ankle_dorsiflexion_deg": float(dors[seg].max() - dors[k_on]),
            "realized_ankle_peak_share": float((k_apk - k_on) / (k_end - k_on)),
            "realized_ankle_plantarflexion_deg": float(dors[seg].max() - dors[k_end]),
            "realized_trunk_lean_peak_deg": float(lean[seg].max() - lean[k_on]),
            "trunk": {"amplitude_deg": geo["trunk_deg"], "solved": geo["solved"]["trunk"],
                      "joints": list(TRUNK_JOINTS), "split": [float(x) for x in split],
                      "room_deg": [float(x) for x in room], "start_share": TRUNK_START_SHARE,
                      "peak_share": TRUNK_PEAK_SHARE, "return_share": TRUNK_RETURN_SHARE,
                      "seat_trunk_deg": {str(j): v for j, v in seat_trunk_deg().items()},
                      "realized_lean_seated_deg": float(lean[k_on]), "realized_lean_peak_deg": float(lean[seg].max()),
                      "realized_lean_peak_share": float((k_tpk - k_on) / (k_end - k_on)),
                      "realized_lean_speed_peak_deg_s": float(np.abs(np.gradient(lean, 1.0 / b.rate)).max())},
            "balance": {"com": "de Leva 1996 table on the 22 joints (mdfm/kernels/force.py com, sex code 2)",
                        "xcom": "Hof et al. 2005: z + v / sqrt(g / l), l the centre of mass above the lowest sole point",
                        "mos_min_m": MOS_MIN_M, "heel_z_m": bal["heel_z_m"], "toe_z_m": bal["toe_z_m"],
                        "seated_com_behind_heel_m": float(bal["heel_z_m"] - bal["com"][k_on, 2]),
                        "liftoff_com_z_m": float(bal["com"][k_lift, 2]),
                        "liftoff_com_vz_m_s": float(bal["com_v"][k_lift, 2]),
                        "liftoff_xcom_z_m": float(bal["xcom_z"][k_lift]),
                        "liftoff_margin_m": float(post[k_lift]),
                        "min_margin_from_liftoff_m": float(post[k_lift:].min()), "min_margin_s": k_min / b.rate,
                        "min_anterior_margin_from_liftoff_m": float(ant[k_lift:].min()),
                        "com_vz_peak_m_s": float(bal["com_v"][:, 2].max()),
                        "standing_com_ahead_of_ankles_m": com_ahead_of_ankles(b, b.T - 1),
                        "stand_ref": STAND_REF, "stand_ref_com_ahead_of_ankles_m": geo["stand_ref_ahead_m"],
                        "standing_tilt_solved": geo["solved"]["stand_tilt"], "ok": ok},
            "seated_knee_flexion_deg": float(knee[k_on]), "standing_knee_flexion_deg": float(knee[-1]),
            "pelvis_rise_m": float(J[-1, 0, 1] - J[0, 0, 1]), "pelvis_forward_m": float(J[-1, 0, 2] - J[0, 0, 2]),
            "feet": {"ankle_max_move_m": float(np.abs(J[:, [7, 8]] - J[:1, [7, 8]]).max()),
                     "sole_low_range_m": [float(sole.min()), float(sole.max())],
                     "contact_share": float(b.contact.float().mean())},
            "limits": limit_violations(b.rot6d), "hand_limits_ok": H.hand_limit_report(b.hand_R()[:1])["ok"]}


def arise_validity(b, info=None):
    """The checks of the arising body for the clip refusal (render_mesh_video.failed_checks): limits against the
    table (no base: every component outside it counts as new), skate of both feet in stance, lowest foot joint, and
    with the arise info (the body of arise itself, the arms folded at every knot) the balance (the XcoM inside the
    base of support from lift-off, module docstring) and the folded arms (SH9): 'fold' (fold_report over the rise,
    every FOLD_CHECK_STEP-th knot from the onset to the end of the rise and that end; the knots before and after are
    still; its 'ok' no skin part inside another, fold_report's 'entry_ok', and 'margins_ok' fold_report's 'ok', the
    margins, a report), 'armrest' (the deepest entry of each forearm and hand into the arms of arise_chair) and
    'hand_thigh' (the least distance of each forearm and hand to the thighs' skin), each with its 'ok'."""
    lim = limit_violations(b.rot6d)
    sk = skate(b.joints(), b.contact, b.rate)
    J = b.joints()
    out = {"limits_outside": lim["outside"], "limits_new": lim["outside"], "limits_worst_deg": lim["worst_deg"],
           "skate_feet": [0, 1], "skate_p95_m_s": sk["p95"], "skate_base_p95_m_s": None, "skate_added_m_s": None,
           "contact_lost_share": float(1.0 - b.contact.float().mean()), "moved_foot_contact_lost_share": None,
           "lowest_foot_m": float(torch.minimum(J[:, 10, 1], J[:, 11, 1]).min()), "lowest_foot_base_m": None}
    if info is not None:
        bl = info["balance"]
        out["balance"] = {"ok": bl["ok"], "min_margin_from_liftoff_m": bl["min_margin_from_liftoff_m"],
                          "min_anterior_margin_from_liftoff_m": bl["min_anterior_margin_from_liftoff_m"],
                          "mos_min_m": bl["mos_min_m"]}
        out.update(arise_checks_of(b, arise_checks_record(info)))
    return out


def arise_checks_record(info):
    """{'kind': 'arise', 'onset_s', 'end_s'}: the arguments of the arising's fold, armrest and thigh checks
    (arise_checks_of), kept in the truth (params['layer_checks'] of the arising level 0, arise) so that they run again
    on its layered body (physio.LayeredBuild, Phase D of docs/PLAN_2026-11-12.md: the layered truth copied them from
    the raw body)."""
    return {"kind": "arise", "onset_s": float(info["onset_s"]), "end_s": float(info["end_s"])}


def arise_checks_of(b, rec):
    """{'fold', 'armrest', 'hand_thigh'} of arise_validity on the body b with the arguments of arise_checks_record
    rec (the chair arise_chair of b's first knot, as the scene draws it)."""
    k_on, k_end = int(round(rec["onset_s"] * b.rate)), int(round(rec["end_s"] * b.rate))
    fr = fold_report(b, np.unique(np.r_[np.arange(k_on, k_end + 1, FOLD_CHECK_STEP), k_end]), arise_chair(b, 0.0))
    out = {"fold": dict(fr, ok=fr["entry_ok"], margins_ok=fr["ok"])}
    out["armrest"] = {sd: fr[f"armrest_{sd}"] for sd in ("left", "right")}
    out["armrest"]["ok"] = all(fr[f"armrest_{sd}"]["depth_mm_max"] <= 0.0 for sd in ("left", "right"))
    out["hand_thigh"] = {sd: fr[f"thigh_{sd}"] for sd in ("left", "right")}
    out["hand_thigh"]["ok"] = all(fr[f"thigh_{sd}"]["min_mm"] >= 0.0 for sd in ("left", "right"))
    return out


# ---------------------------------------------------------------- the folded arms: mesh checks (SH9)
@functools.lru_cache(maxsize=1)
def _fold_sets():
    """{part: (vertex ids, faces as local indices)} of FOLD_PARTS (the dominant joint of each vertex of the selected
    body model's mesh; the faces whose three vertices belong to the part; open-body port wave 4: postures' table and
    skin checks read it)."""
    from .viewer import dominant_joint, faces as body_faces
    faces = body_faces()
    dom = dominant_joint("neutral")
    out = {}
    for name, joints in FOLD_PARTS.items():
        ids = np.flatnonzero(np.isin(dom, joints))
        loc = -np.ones(len(dom), np.int64)
        loc[ids] = np.arange(len(ids))
        out[name] = (ids, loc[faces[(loc[faces] >= 0).all(1)]])
    return out


def _winding(P, V, F, chunk=128):
    """(N,) the generalized winding numbers of points P (N, 3) about the triangles (V, F): the sum of the triangles'
    signed solid angles over 4 pi (about 1 inside a closed outward-facing surface, 0 outside; a part mesh open at its
    joints still gives more than one half inside it)."""
    out = np.zeros(len(P))
    A, B, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    for s in range(0, len(P), chunk):
        p = P[s:s + chunk, None]
        a, b, c = A[None] - p, B[None] - p, C[None] - p
        la, lb, lc = (np.linalg.norm(x, axis=-1) for x in (a, b, c))
        num = np.einsum("nmi,nmi->nm", a, np.cross(b, c))
        den = (la * lb * lc + np.einsum("nmi,nmi->nm", a, b) * lc + np.einsum("nmi,nmi->nm", b, c) * la
               + np.einsum("nmi,nmi->nm", c, a) * lb)
        out[s:s + chunk] = (2.0 * np.arctan2(num, den)).sum(1) / (4.0 * np.pi)
    return out


def skin_gap(P, A, Q, B):
    """m: the least signed distance between two skin parts, points P (N, 3) with faces A and points Q (M, 3) with
    faces B, both ways: each vertex's distance to the other part's triangles (exact point-to-triangle below
    reach.CLEAR_SEARCH_M, reach._signed_to_mesh), negative for a vertex inside the other part (winding number above
    one half, read for the vertices nearer than reach.CLEAR_SEARCH_M: a part entering another has vertices near its
    skin; the part meshes are open at their joints, so the sign of the closest triangle's normal is not used)."""
    from .reach import CLEAR_SEARCH_M, _signed_to_mesh
    out = math.inf
    for X, Y, F in ((P, Q, B), (Q, P, A)):
        d = np.abs(_signed_to_mesh(X, Y, F))
        near = np.flatnonzero(d < CLEAR_SEARCH_M)
        inside = np.zeros(len(X), dtype=bool)
        if len(near):
            inside[near] = _winding(X[near], Y, F) > 0.5
        out = min(out, float(np.where(inside, -d, d).min()))
    return out


def _box_entry(P, lo, hi):
    """m: the deepest entry (>= 0) of points P (N, 3) into the box [lo, hi]."""
    d = np.minimum(P - lo, hi - P).min(-1)
    return float(max(d.max(initial=0.0), 0.0))


FOLD_CHECKS = ("forearms", "hand_left_forearm_right", "hand_right_forearm_left", "hands",
               "hand_left_upper_right", "hand_right_upper_left", "trunk_left", "trunk_right",
               "thigh_left", "thigh_right")


FOLD_PART_PAIRS = {"forearms": (("fore_l",), ("fore_r",)), "hand_left_forearm_right": (("hand_l",), ("fore_r",)),
                   "hand_right_forearm_left": (("hand_r",), ("fore_l",)), "hands": (("hand_l",), ("hand_r",)),
                   "hand_left_upper_right": (("hand_l",), ("up_r",)),
                   "hand_right_upper_left": (("hand_r",), ("up_l",)),
                   "trunk_left": (("fore_l", "hand_l"), ("trunk",)), "trunk_right": (("fore_r", "hand_r"), ("trunk",)),
                   "thigh_left": (("fore_l", "hand_l"), ("thighs",)),
                   "thigh_right": (("fore_r", "hand_r"), ("thighs",))}     # the skin parts of each check


def fold_gaps(body, knots, chair=None, checks=FOLD_CHECKS):
    """({check: (K,) m}, {entry: (K,) m}): the least skin gap of each check of FOLD_CHECKS at each of the knots (K,)
    (skin_gap of the parts of FOLD_PART_PAIRS; negative inside) and, with chair ((vertices, faces) of arise_chair),
    the deepest entry (>= 0) of a side's forearm and hand vertices into any of its boxes ('chair_left',
    'chair_right') or into its two arms, boxes 6 and 7 ('armrest_left', 'armrest_right'); fold_report's reading."""
    S = _fold_sets()
    k = np.asarray(knots, dtype=np.int64).reshape(-1)
    names = list(S)
    ids = np.concatenate([S[n][0] for n in names])
    off = dict(zip(names, np.cumsum([0] + [len(S[n][0]) for n in names])[:-1]))
    vals = {c: np.empty(len(k)) for c in checks}
    boxes = None if chair is None else chair[0].reshape(-1, 8, 3)
    entry = {f"{w}_{sd}": np.zeros(len(k)) for w in ("chair", "armrest") for sd in ("left", "right")} if chair is not \
        None else {}
    for i in range(len(k)):
        X = H.body_points(body, ids, knots=[int(k[i])])[0]
        P = {n: X[off[n]:off[n] + len(S[n][0])] for n in names}
        for c in checks:
            a, b = FOLD_PART_PAIRS[c]
            vals[c][i] = min(skin_gap(P[x], S[x][1], P[y], S[y][1]) for x in a for y in b)
        if boxes is not None:
            for sd, parts in (("left", ("fore_l", "hand_l")), ("right", ("fore_r", "hand_r"))):
                arm = np.concatenate([P[p] for p in parts])
                e = [_box_entry(arm, bx.min(0), bx.max(0)) for bx in boxes]
                entry[f"chair_{sd}"][i], entry[f"armrest_{sd}"][i] = max(e), max(e[6], e[7])
    return vals, entry


def fold_report(body, knots=None, chair=None, checks=FOLD_CHECKS):
    """{check: {'min_mm', 'at_s', 'max_mm'}, 'chair_left', 'chair_right', 'armrest_left', 'armrest_right' (with chair),
    'ok' (every rule below), 'entry_ok' (no part inside another, no chair entry), 'rules', 'n_knots'}: the folded
    arms' clearances on the skinned mesh (fold_gaps, skin_gap; mm, negative inside; the
    least over the knots, when, and the largest of the knots' least for the hands' nearness) at the knots
    (default every FOLD_CHECK_STEP-th and the last): 'forearms' the two forearms (at least FOLD_FOREARM_CLEAR_M
    apart), 'hand_left_forearm_right' and 'hand_right_forearm_left' a hand and the other forearm, 'hands' the two
    hands (apart), 'hand_left_upper_right' and 'hand_right_upper_left' a hand and the opposite upper arm (resting on
    or near it: from 0 to FOLD_HAND_NEAR_M), 'trunk_*' and 'thigh_*' a side's forearm and hand and the trunk
    (pelvis and spine) or the thighs (outside them); with chair ((vertices, faces) of arise_chair), 'chair_*' and
    'armrest_*': the deepest entry (mm, 0 none) of a side's forearm and hand vertices into any of its boxes or into
    its two arms (boxes 6 and 7). checks: the subset of FOLD_CHECKS to read."""
    k = (np.unique(np.r_[np.arange(0, body.T, FOLD_CHECK_STEP), body.T - 1]) if knots is None
         else np.asarray(knots, dtype=np.int64).reshape(-1))
    vals, entry = fold_gaps(body, k, chair, checks)
    out = {}
    for c, v in vals.items():
        j = int(np.argmin(v))
        out[c] = {"min_mm": float(1000.0 * v[j]), "at_s": float(k[j] / body.rate), "max_mm": float(1000.0 * v.max())}
    for c, v in entry.items():
        j = int(np.argmax(v))
        out[c] = {"depth_mm_max": float(1000.0 * v[j]), "at_s": float(k[j] / body.rate)}
    clear = {"forearms": FOLD_FOREARM_CLEAR_M}
    ok = all(out[c]["min_mm"] >= 1000.0 * clear.get(c, 0.0) - 1e-9 for c in checks)
    ok = ok and all(out[c]["max_mm"] <= 1000.0 * FOLD_HAND_NEAR_M for c in checks if "_upper_" in c)
    ok = ok and all(v["depth_mm_max"] <= 0.0 for c, v in out.items() if c in entry)
    no_entry = all(out[c]["min_mm"] >= 0.0 for c in checks) and all(out[c]["depth_mm_max"] <= 0.0 for c in entry)
    out.update(ok=bool(ok), entry_ok=bool(no_entry), n_knots=int(len(k)),
               rules={"forearms_min_mm": 1000.0 * FOLD_FOREARM_CLEAR_M, "hand_to_upper_arm_mm": [0.0,
                      1000.0 * FOLD_HAND_NEAR_M], "others_min_mm": 0.0, "chair_entry_mm": 0.0})
    return out


# ---------------------------------------------------------------- the folded arms: opening them (task sh1c)
FOLD_OPEN_CHECKS = ("forearms", "hand_left_forearm_right", "hand_right_forearm_left", "hands", "trunk_left",
                    "trunk_right", "thigh_left", "thigh_right")   # the skin pairs an opening of the fold keeps apart
FOLD_OPEN_CLEAR_M = 0.0005       # [module] the least skin gap of an opening on its rehearsal (fold_open_lag: under
#                                  the fold's own gaps of the hands to the other forearm, FOLD_RECORD's 'gaps_mm')
FOLD_OPEN_LAGS = tuple(round(0.05 * i, 2) for i in range(11))   # [module] the lags tried: 0 to 0.5 of the opening
FOLD_OPEN_STEP = 1               # [module] knots between two reads of an opening: every knot (a read every second
#                                  knot reported the 3.9 openings' least gaps too high: an entry between two reads
#                                  would have passed; review of task sh1c)
FOLD_OPEN_COARSE = 4             # [module] fold_open_lag reads a try every this many reads first: a try whose
#                                  coarse least gap is already under the clearance fails without the full read


def fold_open_shares(s, lag=0.0, lag_side=None):
    """{side: (N,) share in [0, 1]}: each arm's own time share of an opening of the fold at the opening's time shares
    s (N,): both at s without a lag; with a lag, the lagging side (lag_side) starts at lag and the other ends at
    1 - lag, each taking 1 - lag of the opening (module docstring: one arm waits for the other)."""
    s = np.asarray(s, dtype=np.float64)
    if lag <= 0.0 or lag_side is None:
        return {sd: np.clip(s, 0.0, 1.0) for sd in ("left", "right")}
    lead, late = np.clip(s / (1.0 - lag), 0.0, 1.0), np.clip((s - lag) / (1.0 - lag), 0.0, 1.0)
    return {sd: late if sd == lag_side else lead for sd in ("left", "right")}


def fold_open_lag(realize, n, checks=FOLD_OPEN_CHECKS, clear_m=FOLD_OPEN_CLEAR_M, step=FOLD_OPEN_STEP):
    """(lag, lag_side, record): the smallest lag of FOLD_OPEN_LAGS, and the side that lags (both tried at each lag,
    the one with the larger least gap kept), whose rehearsal realize(lag, side) (a HandBody whose knots 0 to n - 1
    hold the opening) keeps every check at least clear_m at every step-th knot and the last (fold_gaps: the fold's
    skin-gap check read over the opening); a try is read at every FOLD_OPEN_COARSE-th of those knots first, and in
    full only when that coarse read keeps clear_m (the coarse least gap is never below the full one, so the result is
    the full read's); the record lists each try's least gap (mm), its check and the knots read ('read': 'coarse' or
    'full'). Raises when no lag does."""
    ks = np.unique(np.r_[np.arange(0, n, step), n - 1])
    coarse = np.unique(np.r_[ks[::FOLD_OPEN_COARSE], ks[-1]])
    tried = []
    for lag in FOLD_OPEN_LAGS:
        best = None
        for side in ((None,) if lag == 0.0 else ("left", "right")):
            body = realize(lag, side)
            read = "coarse"
            vals, _ = fold_gaps(body, coarse, checks=checks)
            if min(float(v.min()) for v in vals.values()) >= clear_m:
                vals, _ = fold_gaps(body, ks, checks=checks)
                read = "full"
            worst = min(checks, key=lambda c: float(vals[c].min()))
            m = float(vals[worst].min())
            tried.append({"lag": lag, "lag_side": side, "min_mm": 1000.0 * m, "check": worst, "read": read})
            if m >= clear_m and (best is None or m > best[1]):
                best = (side, m)
        if best is not None:
            return lag, best[0], {"tried": tried, "clear_mm": 1000.0 * clear_m, "checks": list(checks),
                                  "step_knots": int(step), "knots": int(n)}
    raise RuntimeError(f"no lag of the opening keeps the skins {1000.0 * clear_m} mm apart: {tried}")


def side_rows(side):
    """(arm joint rows, hand rows) of a side: ARM_JOINTS and the 15 rows of hands.HandBody.hand6d."""
    return list(ARM_JOINTS[side]), [H.hand_row(j) for j in range(H.FIRST[side], H.FIRST[side] + 15)]


def unfold_to_stand(body, k0, n, target="stand"):
    """(R (T, 22, 3, 3), Rh (T, 30, 3, 3), shares {side: (T,)}, record): the folded arms of a body opened over n knots
    from knot k0 to the arms and the open hands of target (base.POSTURES): each arm row's axis-angle components on a
    straight line from the body's to target's (so that they stay inside the table's box), each hand row's rotation
    toward rest by the same share (its axis-angle scaled), each side on its own minimum-jerk clock
    (fold_open_shares), the lag and its side solved on the body itself by fold_open_lag over the opening's knots (the
    knots before k0 keep the body's rows, those after the opening the target's arms and open hands). It is the
    opening of postures.paroxysmal_kinesigenic (PKD), whose arms took one clock before task sh1c and whose right hand
    then passed through the left forearm (fold_open_lag's try at lag 0 gives the depth); unfold_check reads the clip's
    final body."""
    T = body.T
    R0, Rh0 = body.local_R(), body.hand_R()
    c0 = matrix_to_axis_angle(R0)
    c1 = matrix_to_axis_angle(posture_rotations(target))
    D0 = matrix_to_axis_angle(Rh0)
    t = np.arange(T, dtype=np.float64)

    def build(lag, side, sl=slice(None)):
        sh = fold_open_shares((t[sl] - k0) / n, lag, side)
        R, Rh = R0[sl].clone(), Rh0[sl].clone()
        w = {}
        for sd in ("left", "right"):
            js, hr = side_rows(sd)
            w[sd] = minimum_jerk(sh[sd])
            ww = torch.as_tensor(w[sd])
            on = torch.as_tensor(w[sd] > 0.0)                  # before its opening a side keeps the body's rows
            Ro = axis_angle_to_matrix(c0[sl][:, js] + ww[:, None, None] * (c1[js][None] - c0[sl][:, js]))
            Rho = axis_angle_to_matrix((1.0 - ww)[:, None, None] * D0[sl][:, hr])
            R[:, js] = torch.where(on[:, None, None, None], Ro, R[:, js])
            Rh[:, hr] = torch.where(on[:, None, None, None], Rho, Rh[:, hr])
        return R, Rh, w
    span = slice(int(k0), min(T, int(k0) + int(n)))

    def realize(lag, side):
        R, Rh, _ = build(lag, side, span)
        return H.HandBody(matrix_to_rot6d(R), body.transl[span], body.contact[span], body.betas, body.gender,
                          body.rate, matrix_to_rot6d(Rh))
    lag, side, rec = fold_open_lag(realize, span.stop - span.start)
    R, Rh, w = build(lag, side)
    rec.update(lag=lag, lag_side=side, target=target, k0=int(k0), n=int(n))
    return R, Rh, w, rec


UNFOLD_CHECK_PAD_S = 0.3         # [module] unfold_check reads this long after the opening too (the arms come to hang)


def unfold_check(body, k0, n, pad_s=UNFOLD_CHECK_PAD_S, checks=FOLD_OPEN_CHECKS):
    """{check: least gap (mm), 'at_s': {check: s}, 'knots': [first, last], 'ok'}: the skin gaps of checks (fold_gaps,
    negative inside) at every knot of an opening of the fold (knots k0 to k0 + n - 1, as unfold_to_stand's) and
    pad_s after it, read on the final body, so that whatever moves the arms after unfold_to_stand (the attack of a
    paroxysmal kinesigenic clip) is read too (review of task sh1c: unfold_to_stand's rehearsal reads the opening
    alone); 'ok' when no skin part enters another: the validity entry 'fold' of a clip that opens the fold, which
    render_mesh_video.failed_checks refuses on."""
    ks = np.arange(int(k0), min(body.T, int(k0) + int(n) + int(round(pad_s * body.rate))))
    g, _ = fold_gaps(body, ks, None, checks)
    out = {c: float(1000.0 * x.min()) for c, x in g.items()}
    out.update(at_s={c: float(ks[int(np.argmin(x))] / body.rate) for c, x in g.items()},
               knots=[int(ks[0]), int(ks[-1])], ok=bool(all(float(x.min()) >= 0.0 for x in g.values())))
    return out


def arise_chair(body, floor_y, knot=0, back_clear_m=0.005, seated_tol_m=0.01, back_joints=(0, 3, 6, 9)):
    """(vertices, faces) of a straight-backed chair with arms under the seated knot of an arising body: a seat slab
    (5 cm thick) whose top lies at the lowest of the seat vertices (_seat_vertices) at that knot (minus 5 mm), from
    the backrest to SEAT_FRONT_M behind the knees, or 2 cm behind the shanks where they pass the slab's height at any
    knot (the standing calves), on four legs (4 cm square) at its corners, so that the heels drawn under the seat
    stay clear of it; a backrest (5 cm thick, 55 cm high) whose front face lies back_clear_m behind the rearmost trunk
    vertex (dominant joint in back_joints: pelvis, spine 1 to 3) inside the backrest's height and width over every
    seated knot (the lowest seat vertex within seated_tol_m of its height at the chair's knot: the failed attempts
    sit back too); two arms beside the hips (8 cm wide, 22 cm above the seat) from 7 cm behind the rearmost seat
    vertex at the knot (or from the backrest's front face when it lies farther back) to the seat's front. Boxes in
    that order: seat, backrest, four legs, the arm on the body's left (x > 0), the right one. Until SH5 of
    docs/proposals/SYNTH_SHARED_CHANGES_DRAFT.md (clinical review of 2026-11-02) the backrest's front face stood 2 cm
    behind the rearmost seat vertex at the knot, inside the seated spines (tests/test_synth_scenes.py measures it);
    the arms, whose tops, front ends and footprints parkinsonism.armrest_targets reads, did not move. The sizes are
    module values (back_clear_m: the review's 5 mm; seated_tol_m 1 cm); the chair stands on the floor plane floor_y
    (viewer.box)."""
    from .viewer import box
    V = H.body_points(body, np.arange(len(H._rest()[1])), knots=[knot])[0]
    J = body.joints().numpy()[knot]
    ids = _seat_vertices()
    seat_top = float(V[ids, 1].min()) - 0.005
    arm_back_z = float(V[ids, 2].min()) - 0.02 - 0.05
    x_half = float(np.abs(V[ids, 0]).max()) + 0.03
    low = H.body_points(body, ids)[..., 1].min(1)
    seated = np.flatnonzero(low <= low[knot] + seated_tol_m)
    trunk = np.flatnonzero(np.isin(H._rest()[2], tuple(back_joints)))
    P = H.body_points(body, trunk, knots=seated)
    inside = (P[..., 1] > seat_top) & (P[..., 1] < seat_top + 0.55) & (np.abs(P[..., 0]) < x_half)
    back_z = float(np.where(inside, P[..., 2], np.inf).min()) - back_clear_m
    arm_back_z = min(arm_back_z, back_z)
    front = float(min(J[4, 2], J[5, 2])) - SEAT_FRONT_M
    legs = np.flatnonzero(np.isin(H._rest()[2], (4, 5, 7, 8, 10, 11)))   # shanks and feet: clear of the slab
    P = H.body_points(body, legs, knots=range(0, body.T, 10))
    band = (P[..., 1] > seat_top - 0.06) & (P[..., 1] < seat_top + 0.01) & (np.abs(P[..., 0]) < x_half)
    if band.any():
        front = min(front, float(P[..., 2][band].min()) - 0.02)
    floor = float(floor_y)
    parts = [box((-x_half, seat_top - 0.05, back_z), (x_half, seat_top, front)),
             box((-x_half, seat_top, back_z - 0.05), (x_half, seat_top + 0.55, back_z))]
    for x0 in (-x_half, x_half - 0.04):
        for z0 in (back_z - 0.05, front - 0.04):
            parts.append(box((x0, floor, z0), (x0 + 0.04, seat_top - 0.05, z0 + 0.04)))
    for sgn in (1.0, -1.0):
        x0 = sgn * x_half
        parts.append(box((min(x0, x0 + sgn * 0.08), seat_top, arm_back_z),
                         (max(x0, x0 + sgn * 0.08), seat_top + 0.22, front)))
    vs, fs, off = [], [], 0
    for v, f in parts:
        vs.append(v)
        fs.append(f + off)
        off += len(v)
    return np.concatenate(vs), np.concatenate(fs)


# ---------------------------------------------------------------- the folded arms: the solve (task sh1c)
FOLD_ORDER = ARM_JOINTS["left"] + ARM_JOINTS["right"]
FOLD_ARMS = ("fore_l", "fore_r", "up_l", "up_r", "hand_l", "hand_r")
FOLD_PAIRS = {"forearms": ("fore_l", "fore_r"), "hand_left_forearm_right": ("hand_l", "fore_r"),
              "hand_right_forearm_left": ("hand_r", "fore_l"), "hands": ("hand_l", "hand_r"),
              "hand_left_upper_right": ("hand_l", "up_r"), "hand_right_upper_left": ("hand_r", "up_l")}


def fold_bounds():
    """(lo, hi) (24,) deg: the SP1 bounds of the eight arm joints in FOLD_ORDER (the left collar, shoulder, elbow
    and wrist, then the right), FOLD_SOLVE['margin_deg'] inside."""
    from .body import joint_limits
    m, lim = FOLD_SOLVE["margin_deg"], joint_limits()
    return (np.concatenate([lim.lo[j] + m for j in FOLD_ORDER]),
            np.concatenate([lim.hi[j] - m for j in FOLD_ORDER]))


def fold_q(fold):
    """(24,) deg: the numbers of a fold ({'left', 'right'}: four (x, y, z) each) in FOLD_ORDER."""
    return np.concatenate([np.asarray(fold[s][i], dtype=np.float64) for s in ("left", "right") for i in range(4)])


def fold_dict(q, ndigits=2):
    """{'left', 'right'}: four (x, y, z) deg each from q (24,) in FOLD_ORDER, rounded to ndigits (None: not)."""
    qq = np.asarray(q, dtype=np.float64).reshape(2, 4, 3)
    qq = qq if ndigits is None else np.round(qq, ndigits)
    return {s: tuple(tuple(float(v) for v in qq[i, k]) for k in range(4)) for i, s in enumerate(("left", "right"))}


def _fold_R(q):
    """(22, 3, 3): 'stand' with the eight arm rows of q (24,) deg."""
    R = posture_rotations("stand").clone()
    a = torch.as_tensor(np.radians(np.asarray(q, dtype=np.float64)).reshape(8, 3))
    R[list(FOLD_ORDER)] = axis_angle_to_matrix(a)
    return R


def fold_features(q):
    """{side: {'E', 'W', 'K' (m, from the spine3 joint of 'stand'), 'palm' (unit)}, 'J' (22, 3)}: the elbow, the
    wrist, the middle knuckle and the palmar normal of the folded arms q (24,) on 'stand' (rest frame)."""
    J, Gl = fk(_fold_R(q)[None], torch.zeros(1, 3, dtype=F64), rest_joints())
    J, Gl = J[0].numpy(), Gl[0].numpy()
    J52, _, _ = H._rest()
    out = {"J": J}
    for side in ("left", "right"):
        _, _, e, w = ARM_JOINTS[side]
        mid = H.hand_joint(side, "middle", 0)
        out[side] = {"E": J[e] - J[9], "W": J[w] - J[9], "K": J[w] + Gl[w] @ (J52[mid] - J52[w]) - J[9],
                     "palm": Gl[w] @ H.hand_frame()[side]["palmar"]}
    return out


def fold_directions(fold):
    """{side: {'upper', 'fore', 'hand', 'palm' (unit, world, the body facing +z), 'wrist' (m, rest frame)}} of a
    fold on 'stand', as arm_errors reads them: ARM_TARGETS['fold'], the regression record of the solve's result."""
    q = fold_q(fold)
    f = fold_features(q)
    J = f["J"]
    J52, _, _ = H._rest()
    _, Gl = fk(_fold_R(q)[None], torch.zeros(1, 3, dtype=F64), rest_joints())
    Gl = Gl[0].numpy()
    out = {}
    for side in ("left", "right"):
        _, s, e, w = ARM_JOINTS[side]
        mid = H.hand_joint(side, "middle", 0)
        d = {"upper": J[e] - J[s], "fore": J[w] - J[e], "hand": Gl[w] @ (J52[mid] - J52[w]),
             "palm": f[side]["palm"]}
        out[side] = {k: tuple(float(x) for x in np.round(v / np.linalg.norm(v), 4)) for k, v in d.items()}
        out[side]["wrist"] = tuple(float(x) for x in np.round(J[w], 4))
    return out


def _fold_target_residuals(q, w):
    """The target residuals of the folded arms q (fold_features against FOLD_SOLVE['targets'], weights w)."""
    f, tg = fold_features(q), FOLD_SOLVE["targets"]
    r = []
    for side in ("left", "right"):
        for k in ("E", "W", "K"):
            r.append(w[k] * (f[side][k] - np.asarray(tg[side][k])))
        p = np.asarray(tg[side]["palm"], dtype=np.float64)
        r.append(w["palm"] * (f[side]["palm"] - p / np.linalg.norm(p)))
    qq = np.asarray(q, dtype=np.float64).reshape(8, 3)
    r.append(w["collar"] * (qq[[0, 4]] - np.asarray([tg["left"]["collar"], tg["right"]["collar"]])).ravel())
    r.append(w["elbow_z"] * qq[[2, 6], 2])
    r.append(w["wrist_x"] * qq[[3, 7], 0])
    return np.concatenate(r)


def fold_ik():
    """(q (24,), scipy result): the first stage of solve_fold, a bounded least squares (trust region reflective) of
    the target residuals with FOLD_SOLVE['ik_weights'], from the arms of 'stand' clipped inside fold_bounds."""
    from scipy.optimize import least_squares
    lo, hi = fold_bounds()
    st = posture_rotations("stand")
    q0 = np.clip(np.concatenate([np.degrees(matrix_to_axis_angle(st[j]).numpy()) for j in FOLD_ORDER]),
                 lo + 1e-6, hi - 1e-6)
    sol = least_squares(_fold_target_residuals, q0, bounds=(lo, hi), args=(FOLD_SOLVE["ik_weights"],),
                        x_scale=10.0, diff_step=1e-4, max_nfev=4000)
    return sol.x, sol


def _vertex_normals(V, F):
    """(..., N, 3) unit vertex normals of meshes V (..., N, 3) with faces F (area-weighted face normals)."""
    a, b, c = V[..., F[:, 0], :], V[..., F[:, 1], :], V[..., F[:, 2], :]
    fn = np.cross(b - a, c - a)
    n = np.zeros_like(V)
    for i in range(3):
        np.add.at(n, (Ellipsis, F[:, i], slice(None)), fn)
    return n / np.linalg.norm(n, axis=-1, keepdims=True).clip(1e-12)


def _fold_rise_obstacles(q):
    """The rise's obstacles of the second stage of solve_fold for the first stage's arms q (24,): motions.arise with
    those arms (standing 0.5 s); at every FOLD_SOLVE['rise_step']-th knot of the rise and its last, the trunk's and
    the thighs' vertices (FOLD_PARTS['trunk'], ['thighs']; the thighs since wave 6 of the open-body port) with their
    normals (the whole skinned mesh's) carried into the frame of the folded stand through spine3 (the arms ride on
    spine3), kept within FOLD_SOLVE['crop_m'] of q's forearms and hands there; the boxes of the chair's arms
    (arise_chair: 6 and 7) with each knot's map from the stand's frame to the world. {'trunk' (N, 3), 'trunk_n' (N, 3),
    'thighs' (N', 3), 'thighs_n' (N', 3), 'M' (K, 3, 3), 't' (K, 3), 'boxes' [(lo, hi)], 'knots' (K,), 'arise'}."""
    from scipy.spatial import cKDTree
    from .viewer import faces as body_faces
    fold = fold_dict(q, None)
    b, info = arise(standing_s=0.5, fold=fold)
    st = posture_body(STAND_REF, 0.02, fold=fold)
    k_on, k_end = int(round(info["onset_s"] * b.rate)), int(round(info["end_s"] * b.rate))
    ks = np.unique(np.r_[np.arange(k_on, k_end + 1, FOLD_SOLVE["rise_step"]), k_end])
    J, G = b.fk()
    Jk, Gk = J[ks, 9].numpy(), G[ks, 9].numpy()
    Js, Gs = st.fk()
    Js, Gs = Js[0, 9].numpy(), Gs[0, 9].numpy()
    faces = body_faces()
    V = H.body_points(b, np.arange(len(H._rest()[1])), knots=ks)
    Nv = _vertex_normals(V, faces)
    S = _fold_sets()
    M = np.einsum("kij,lj->kil", Gk, Gs)                               # stand frame -> world at each knot
    t = Jk - np.einsum("kij,j->ki", M, Js)
    arm = np.concatenate([S[p][0] for p in ("fore_l", "hand_l", "fore_r", "hand_r")])
    tree = cKDTree(H.body_points(st, arm, knots=[0])[0])
    out = {}
    for part in ("trunk", "thighs"):            # the thighs since wave 6 (the open body's deeper lumbar bend)
        ids = S[part][0]
        P = np.einsum("kji,knj->kni", M, V[:, ids] - t[:, None])       # world -> stand frame
        Pn = np.einsum("kji,knj->kni", M, Nv[:, ids])
        keep = tree.query(P.reshape(-1, 3))[0] < FOLD_SOLVE["crop_m"]
        out[part], out[part + "_n"] = P.reshape(-1, 3)[keep], Pn.reshape(-1, 3)[keep]
    Vc, _ = arise_chair(b, 0.0)
    bx = Vc.reshape(-1, 8, 3)
    return {**out, "M": M, "t": t,
            "boxes": [(bx[i].min(0), bx[i].max(0)) for i in (6, 7)], "knots": ks,
            "arise": {"trunk_deg": info["trunk"]["amplitude_deg"],
                      "standing_shank_tilt_deg": info["standing_shank_tilt_deg"],
                      "realized_trunk_lean_peak_deg": info["realized_trunk_lean_peak_deg"]}}


def _one_way(X, Y, F):
    """(N,) m: each point of X (N, 3) to the part (Y, F): the distance of reach._signed_to_mesh (exact below
    reach.CLEAR_SEARCH_M), negative where the winding number about the part exceeds one half (skin_gap's rule)."""
    from .reach import CLEAR_SEARCH_M, _signed_to_mesh
    d = np.abs(_signed_to_mesh(X, Y, F))
    near = np.flatnonzero(d < CLEAR_SEARCH_M)
    inside = np.zeros(len(X), dtype=bool)
    if len(near):
        inside[near] = _winding(X[near], Y, F) > 0.5
    return np.where(inside, -d, d)


class _FoldProblem:
    """The second stage of solve_fold (module docstring): the residuals of the arm rows q (24,) on the folded stand,
    the hinges of FOLD_SOLVE['clear_m'] and ['near_m'] on the six arm parts' skins (per vertex, both ways, skin_gap's
    rule, _one_way), each side's forearm and hand against the trunk over the rise (the plane through the nearest
    trunk point of the rise, _fold_rise_obstacles) and against the chair's arms over the rise (the box depth), and
    the target residuals with FOLD_SOLVE['polish_weights']."""

    def __init__(self, rise):
        from scipy.spatial import cKDTree
        S = _fold_sets()
        self.S = S
        self.ids = np.concatenate([S[p][0] for p in FOLD_ARMS])
        n = [len(S[p][0]) for p in FOLD_ARMS]
        self.sl = {p: slice(int(a), int(a) + m) for p, a, m in zip(FOLD_ARMS, np.cumsum([0] + n)[:-1], n)}
        pose = {}
        for sd in ("left", "right"):
            pose.update(closed_pose("grip", sd))
        self.Rh = H.hand_rotations(pose).numpy()[None]
        st = posture_body("stand", 0.02)
        self.tr, self.betas, self.gender = st.transl[:1].numpy(), st.betas.numpy(), int(st.gender)
        self.rise = rise
        self.tree = {p: cKDTree(rise[p]) for p in ("trunk", "thighs") if len(rise[p])}

    def gaps(self, q):
        """{check: tuple of (n,) m arrays}: the per-vertex gaps (module docstring of _FoldProblem)."""
        X = H.skin_ids(_fold_R(q).numpy()[None], self.tr, self.betas, self.gender, self.ids, self.Rh)[0]
        P = {p: X[self.sl[p]] for p in FOLD_ARMS}
        S = self.S
        g = {k: (_one_way(P[a], P[b], S[b][1]), _one_way(P[b], P[a], S[a][1])) for k, (a, b) in FOLD_PAIRS.items()}
        r = self.rise
        for sd, parts in (("left", ("fore_l", "hand_l")), ("right", ("fore_r", "hand_r"))):
            A = np.concatenate([P[p] for p in parts])
            for part, name in (("trunk", "trunk"), ("thighs", "thigh")):
                if part in self.tree:
                    _, i = self.tree[part].query(A)
                    g[f"{name}_{sd}"] = (np.einsum("ij,ij->i", A - r[part][i], r[part + "_n"][i]),)
        A = np.concatenate([P[p] for p in ("fore_l", "hand_l", "fore_r", "hand_r")])
        W = np.einsum("kij,nj->kni", r["M"], A) + r["t"][:, None]
        depth = np.max([np.minimum(W - lo, hi - W).min(-1) for lo, hi in r["boxes"]], axis=0)   # > 0 inside
        g["armrest"] = (-depth.max(0),)
        return g

    def residuals(self, q):
        g = self.gaps(q)
        wt, c, near = FOLD_SOLVE["clear_weight"], FOLD_SOLVE["clear_m"], FOLD_SOLVE["near_m"]
        res = []
        for k, v in g.items():
            res += [wt * np.maximum(0.0, c[k] - x) for x in v]
            if k in near:
                res.append(np.atleast_1d(wt * max(0.0, min(float(x.min()) for x in v) - near[k])))
        res.append(_fold_target_residuals(q, FOLD_SOLVE["polish_weights"]))
        return np.concatenate(res)

    def report(self, q):
        """{check: least gap (mm)} of q."""
        return {k: float(1000.0 * min(float(x.min()) for x in v)) for k, v in self.gaps(q).items()}


def solve_fold():
    """The folded arms of ARM_POSTURES['fold'] (module docstring): the first stage (fold_ik) toward the joint
    targets, the rise's obstacles of its arms (_fold_rise_obstacles), then the second stage (_FoldProblem, a bounded
    least squares from the first stage's result, at most FOLD_SOLVE['polish_nfev'] evaluations); the result rounded
    to 0.01 deg. {'fold', 'ik', 'stages', 'gaps_mm', 'rise'}: deterministic (no draw), several minutes of CPU."""
    from scipy.optimize import least_squares
    q_ik, s_ik = fold_ik()
    rise = _fold_rise_obstacles(q_ik)
    pr = _FoldProblem(rise)
    lo, hi = fold_bounds()
    sol = least_squares(pr.residuals, np.clip(q_ik, lo + 1e-6, hi - 1e-6), bounds=(lo, hi), x_scale=5.0,
                        diff_step=1e-3, max_nfev=FOLD_SOLVE["polish_nfev"])
    fold = fold_dict(sol.x)
    return {"fold": fold, "ik": fold_dict(q_ik),
            "stages": {"ik": {"nfev": int(s_ik.nfev), "cost": float(s_ik.cost), "status": int(s_ik.status)},
                       "polish": {"nfev": int(sol.nfev), "cost": float(sol.cost), "status": int(sol.status),
                                  "q": [float(x) for x in sol.x]}},
            "gaps_mm": {"ik": pr.report(q_ik), "result": pr.report(fold_q(fold))},
            "rise": {"knots": [int(k) for k in rise["knots"]], "trunk_points": int(len(rise["trunk"])),
                     "thigh_points": int(len(rise["thighs"])),
                     **rise["arise"]}}


FOLD_SOLVER_CODE = ("fold_bounds", "fold_q", "fold_dict", "_fold_R", "fold_features", "fold_directions",
                    "_fold_target_residuals", "fold_ik", "_vertex_normals", "_fold_rise_obstacles", "_one_way",
                    "_winding", "_FoldProblem", "solve_fold", "_fold_sets", "closed_pose", "arm_rotations",
                    "posture3", "posture_body", "arise", "_arise_body", "_standing_tilt", "_solve_trunk",
                    "_solve_drop", "_dorsiflexion", "trunk_split", "trunk_profile", "seat_trunk_deg",
                    "balance",
                    "com_ahead_of_ankles", "arise_chair", "_seat_vertices", "_leg_plane", "minimum_jerk",
                    "_arise_info", "_mirror")
FOLD_SOLVER_VALUES = ("FOLD_SOLVE", "GRIP", "FOLD_PARTS", "ARM_JOINTS", "FOLD_ORDER", "FOLD_ARMS", "FOLD_PAIRS",
                      "SEAT_HEIGHT_M", "ARISE_S", "LIFTOFF_SHARE", "HIP_PEAK_SHARE", "HIP_FLEXION_DEG",
                      "ANKLE_DORSI_DEG", "ANKLE_PEAK_SHARE", "ANKLE_PLANTAR_DEG", "SEAT_ITERS", "SOLE_PASSES", "SEAT_ROLL_PASSES",
                      "DROP_TOL_DEG", "DROP_RANGE_M", "STAND_REF", "TRUNK_JOINTS", "TRUNK_START_SHARE",
                      "TRUNK_PEAK_SHARE", "TRUNK_RETURN_SHARE", "SEAT_TRUNK_DEG", "MOS_MIN_M", "GRAVITY_M_S2",
                      "SEAT_FRONT_M", "POSTURES3", "REACH", "TILT_MARGIN_DEG")
FOLD_SOLVER_BASES = ("sit", "stand")     # the base.POSTURES the solve reads (posture3's bases, posture_rotations)


def _code_digest_text(fn):
    """The syntax tree of a function's or class's source written back as source, its docstrings dropped: a comment or
    a line break does not change it, and this text is the same on Python 3.11 and 3.12, where ast.dump's is not."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and ast.get_docstring(node, clean=False) is not None:
            node.body = node.body[1:] or [ast.Pass()]
    return ast.unparse(tree)


def fold_solver_digest():
    """sha256 of what solve_fold reads in this module: the code of FOLD_SOLVER_CODE (every function of this module
    that solve_fold reaches, and fold_directions, which writes the record's directions; its syntax tree without the
    docstrings, so that a docstring, a comment or a line break does not make the record stale), the values of
    FOLD_SOLVER_VALUES, the local rotations of the base postures FOLD_SOLVER_BASES (base.POSTURES through
    posture_rotations) and the SP1 bounds of the arm joints (the record FOLD_RECORD carries it; a record whose digest
    differs is stale). The rest of the tree it reads (the body model, the scene, the reach and hands modules) is
    checked by tests/test_synth_motions_fold.py, which solves again and compares."""
    from .body import joint_limits
    g = globals()
    src = "\n".join(_code_digest_text(g[n]) for n in FOLD_SOLVER_CODE)
    lim = joint_limits()
    vals = {n: g[n] for n in FOLD_SOLVER_VALUES}
    vals["bounds"] = [[[round(float(x), 9) for x in lim.lo[j]], [round(float(x), 9) for x in lim.hi[j]]]
                      for j in FOLD_ORDER]
    vals["bases"] = {n: [[round(float(x), 9) for x in r] for r in matrix_to_axis_angle(posture_rotations(n))]
                     for n in FOLD_SOLVER_BASES}
    txt = src + "\n" + json.dumps(vals, sort_keys=True, default=list)
    return hashlib.sha256(txt.encode("utf-8")).hexdigest()


def fold_record_path():
    return os.path.join(paths.TREE, *FOLD_RECORD)


def _load_fold():
    """ARM_POSTURES['fold'] and ARM_TARGETS['fold'] from FOLD_RECORD when its digest is fold_solver_digest(), else
    left out (arm_rotations then refuses 'fold' and names the record's state in _FOLD_STATE)."""
    path = fold_record_path()
    if not os.path.exists(path):
        _FOLD_STATE.update(state="missing", path=path)
        return
    with open(path, encoding="utf-8") as f:
        rec = json.load(f)
    d = fold_solver_digest()
    if rec.get("solver_digest") != d:
        _FOLD_STATE.update(state="stale", path=path, record_digest=rec.get("solver_digest"), code_digest=d)
        return
    ARM_POSTURES["fold"] = {s: tuple(tuple(float(v) for v in a) for a in rec["fold"][s]) for s in ("left", "right")}
    ARM_TARGETS["fold"] = {"base": "stand", **{s: {k: tuple(v) for k, v in rec["arm_targets"][s].items()}
                                               for s in ("left", "right")}}
    _FOLD_STATE.update(state="read", path=path, digest=d)


_load_fold()
