"""The physiological background of the synthetic patient (shared change SH1 of
docs/proposals/SYNTH_SHARED_CHANGES_DRAFT.md, approved by the owner on 2026-11-02): breathing, quiet postural sway, a
slow drift of the head, of the seated trunk and of the resting fingers, laid IDENTICALLY (one seed, one layer) on a
clip and on its level 0, so that the difference between a symptom and its level 0 stays the symptom alone. A level 0
without it is a statue to a clinician's eye.

Layer (make_layer): signals of time on a fixed horizon (HORIZON_S at the body's rate), drawn from the series seed
(series_seed: the root of the clip's level-0 chain), in a fixed order whatever the posture, so that a clip takes the
prefix of the layer of its series. Every signal is an analytic function of time (Band: sums of sinusoids with a
long-time RMS normalisation; the breathing phase integrated in closed form), so that the prefix of a long layer is the
layer of a short clip bit for bit.
- breathing (b (T,) 0 exhaled to 1 inhaled): BREATH_PER_MIN, the median resting rate of the healthy seated adults of
  Motamedi-Fakhr et al. 2017 (REST1, pneumotachograph), inspiration over the share BREATH_TI_TE / (1 + BREATH_TI_TE)
  of the cycle (their median tI/tE), a raised cosine up and down; the rate varies slowly by BREATH_RATE_CV (module).
  Kinematics (module values: no source read gives the angles): spine3 extends by BREATH_SPINE_DEG at full
  inhalation, the neck and both collars take it back exactly (their global orientations, hence the gaze and a free
  arm's, unchanged by it: only the translation of the neck and the shoulders remains), then both collars rise by
  BREATH_COLLAR_DEG (the arms ride on them unless the hand rests on a support); supine the spine does not breathe
  (the back and the head lie on the couch), only the collars;
- head drift (yaw and pitch, HEAD_DRIFT_DEG RMS in HEAD_BAND_HZ, shared half and half by neck and head about their
  own axes): module values, no source read gives a seated or standing head drift;
- seated trunk (lean about spine1: forward-only sagittal and two-sided lateral, SIT_BAND_HZ module): the drift's
  expected range over SIT_WINDOW_S at the neck joint (the SMPL joint nearest the T2 vertebra) equals the maximum
  amplitudes Melo et al. 2009 measured at T2 in young adults seated without back or foot support, the hands on the
  thighs, eyes open (read as peak-to-peak ranges; the gain is solved on the procedural 'sit' by sit_gains). The
  sagittal lean is the drift minus its minimum over the horizon: it only leans forward from the clip's posture, so
  the drift never drives the back into the chair's backrest (Melo's seat had none; the catalogue chair's stands
  viewer.CHAIR_BACK_CLEAR_M behind the trunk), at the cost of a constant forward offset of the series, its mean
  recorded in the truth (amplitudes.trunk.forward_offset_mean_deg); breathing adds its own few millimetres;
- standing sway: the pelvis displaced by SWAY_RMS_M RMS (Foulger et al. 2025, centre of mass of hip-width quiet
  stance; the pelvis joint stands for the centre of mass, a module approximation) in SWAY_BAND_HZ (the low-frequency
  band of Yamamoto et al. 2015, whose centre-of-pressure power halves near 0.33 Hz): front-back an ankle
  strategy (the whole body turns about the ankles' axis), sideways a hip strategy (the pelvis shifts, the trunk stays
  upright), as Winter et al. 1998 describe quiet-stance control (ankle plantarflexors sagittally, hip abductors and
  adductors frontally); the pelvis is then raised or lowered and rolled so that both legs keep their hip-to-ankle
  lengths (the spine takes the roll back), and the legs are re-solved so that the ankles keep the clip's positions and
  the feet their global rotations (planted feet stay planted, a moving foot keeps its own path, every knee keeps its
  angle);
- fingers (FINGER_DRIFT_DEG RMS of MCP and PIP flexion of the four fingers, module; the thumb, lying against the
  thigh or the body, is left still: physio-v3): on the hands the series' level 0 holds still and
  supported (resting_hands, and a support at every knot), never on a hand that performs a task or is held
  outstretched, nor on a hanging hand (its fingers lie beside the thigh, which the clips of the series bring closer
  or turn against them: physio-v3); a body without hand joints is given them at rest first (hands.hands_of: its
  other rows kept bit for bit). The fingers only lift (extend) from the posture, never curl into the support,
  whatever a clip of the series does with that hand (the record's fingers_lift_only): the drift x enters as the
  smooth one-sided lift 0.5 (x - sqrt(x^2 + e^2)) (finger_lift, e FINGER_LIFT_SOFT_DEG, module: physio-v4), the
  fingers resting on their posture while the drift would flex them and lifting while it extends them, with no
  constant offset beyond e/2 (physio-v3's drift minus its horizon maximum held every resting hand open). The scenes
  of FINGER_TABLE_SCENES take no finger drift: the hand resting on the writing table bears on a curled fingertip, so
  a lift would raise its lowest point off the table (postures.table_report, read again on the layered body:
  LayeredBuild.with_background).
- still parts of a series (SERIES_QUIET, make_layer quiet, physio-v6): a hand on which a member of the series turns
  the forearm, the wrist or the fingers takes no finger drift, a head whose rows carry a member's sign no head drift,
  in the whole series (the level 0 and its clips stay identical): the drift turned with such a distal sign and the
  difference between the clip and its level 0 took that share of the sign (the finger lift turned by the rest
  tremor's pronation-supination: 0.125 of the sign).
Posture kinds (KINDS): 'seated' (breathing, head, trunk, fingers), 'standing' (breathing, head, sway, fingers),
'supine' (breathing only), 'transfer' (an arising: breathing and head, and since physio-v7 the sway in its standing
phase), 'walking' (a walking segment, WALK_SEGMENTS: breathing, head and sway on its still spans, physio-v7).
Spans (physio-v7, wave SH1d part C, the review's M3: a walk standing still before its first step and after its last,
a freezing hold and the arising's standing phase read as statues): apply's span weighs the layer's signals over the
clip. A walking clip (E8, E8t, E19 and E7p, the pull test) takes the standing layer on its still runs only (still_runs:
the pelvis, the ankles and the feet slower than STILL_M_S for STILL_MIN_S at least), crossfaded over STILL_FADE_S
inside each run at an edge the body moves across (a run shorter than its fades takes none), every other knot the
clip's own bit for bit: the walk moves every segment at every step and the gait signs are trunk and head signs, so a
layer there would add nothing but blur (the owner's 'no layer for E8, E8t, E19 and E7p' option, kept for the walk
itself, revisited for its stance). Each walking clip takes its own solve on its own still spans (a level 0's deltas
would lay its stance's sway on a member's steps), the rows a member holds equal to its level 0's where the two spans
weigh alike taken from the level 0's output (_held_from); the hands hang or swing: no finger drift; the sway's guard
reads the pelvis in the clip's own heading (a walk ends turned). A freezing clip (on the AMASS walk, whose series
takes no layer) lays the same layer on its holds in its own build (with_hold_background, hold_weight: the exam
set's freezing clips since exam_v9; since exam_v10 their walk is procedural, freezing_E8_level0, and the clips and that
level 0 also lay it on the walk's own still runs, still_weight). A transfer sways where it stands (stand_weight: its
still runs within STAND_HIGH_M of its pelvis's highest), the seated phase and the rise keeping the pelvis and the legs
bit for bit.

The solve (apply, solve: the series root's raw body, once per series): the layer's rotations multiply the root's
local rotations (spine1, spine3, neck, head, collars: parent-frame offsets for the trunk, the neck's exact take-back,
child-frame head drift); standing, the pelvis is then displaced and the legs re-solved (stand_sway, two_bone_fix). A
hand that rests on something keeps its place relative to it (the arm re-solved, the hand's global rotation carried
with the support): a scene box, the head or neck, and seated, supine or arising also the thighs and knees
(support_joints), within SUPPORT_M of a forearm or hand vertex at a knot (support_labels); a hand within
HEAD_APPROACH_M of the head follows the head; a free hand rides on the trunk (a standing hand hanging beside the hip
too); the anchoring ramps in and out over ANCHOR_RAMP_S outside the contact (dilate_labels, support_weights). The
trunk, the breathing and the head drift never yield to the hands; only the finger drift of a hand touching or
approaching the face fades (touch_weight). A joint the layer would push outside the healthy joint-limit table beyond
the body's own excursion is blended back toward it at those knots (guard_joints: geodesic blend, the blend factor
smoothed by a moving minimum then two moving means so that it never exceeds the allowed one and shows no kink; the
standing sway is halved, four times at most, then held where a pelvis, spine1 or leg joint would leave it); finger
offsets likewise against the hand limits of hands.py. Then the checks (validity): body.validity against the raw body
(new joint-limit violations, read against the clip's own table, skate added, contact lost), the depth added inside
the scene's boxes, the depth a forearm or hand outside the body takes inside it, the leg-to-leg depth added and the
distance a foot point moves at the clip's stance knots (standing), the acceleration the layer adds beyond a share of
the body's own (a jump of the layer reads there), the largest distance of an anchored wrist from its target; 'ok' and
'failed' sum them up. Where the full layer fails a
contact check, solve() fades the layer locally around the failing knots (local_fade): the breathing and the drifts
first (FADED_FIRST, FADE_TRIES rounds at most), then the sway too; where the anchoring adds acceleration to a hand
flung off and back onto its support, solve() anchors it only where it rests (rest_gate, anchor_rest_gate, physio-v5);
where the acceleration check still fails, the anchoring's elbow takes the damped flexion step ARM_FIX_DAMP_M
(anchor_damped: the undamped step flipped an arm near straight for one knot); where neither passes alone, both
together (physio-v10: the raise after the longer rest on the open body).
The root's guards and fades are the series' own: every clip of the series takes them through the transfer, so they
keep the identity.

The transfer (transfer: every other clip, from its direct level 0, itself layered first; physio-v3 to v5): the level 0's
layer as per-knot deltas (deltas: D_j(t) = R_out_j(t) R_in_j(t)^T of every joint's local rotation in its parent frame,
the world for the pelvis; the pelvis translation's difference; the hand joints' deltas). A row (joint, knot) the clip
holds equal to its level 0's raw row (EQUAL_ROW_TOL) IS the level 0's output row, copied; a row the layer leaves
unchanged in the level 0 keeps the clip's own; every other row takes out_j(t) = D_j(t) R_in_j(t), computed elementwise
in one fixed order (_mul), so that two clips sharing a row compute it alike (a level-0 chain: a clip and its direct
level 0 share their rows whatever the root). Since the level 0 is solved once, a clip and its level 0 carry one layer by
construction, and the difference between a symptom and its level 0 is the symptom turned by the layer's small rotations.
The clip's own corrections touch its own rows only (never a row it holds equal), each named in the record's 'deviations'
and counted in 'guard': standing, where its pelvis row, its translation and every leg row are its own, the pelvis is
raised or lowered so that the legs keep the clip's hip-to-ankle distances, as the root's sway keeps them
(own_leg_lengths, physio-v10: a straight leg shortened bends by the square root of the shortening); a leg whose hip and
knee rows are the clip's own is re-solved so that the ankle keeps the clip's position and the foot its global rotation
(own_legs, legs_own_fix: a stepping or bent leg stays planted);
where only one of its hip and knee rows is the clip's own, the other's held copy cannot follow the sway and the ankle
goes to the point of the vertical through the clip's at the reachable distance (vertical_goal, physio-v5: the planted
foot keeps its horizontal place and takes the residual in height, reported as foot_height_moved_m; a foot on its toes,
toe_stance, keeps its ball instead and turns about it, ball_goal, its foot checks read on the ball, stance_points);
where the leg the clip owns would then leave the joint-limit table it is judged on and its pelvis row, its translation
and every leg row are its own, the level 0's sway is scaled down on those rows, the largest scale found by bisection and
smoothed (own_sway_scaled, _sway_scaled: as the series root's solve scales its own); a hand the clip itself rests on a
support or on any part of its trunk (its own touches, support_labels without the approach, on support_joints and, but
standing, TRUNK_TOUCH, read after the lap and the scene) is kept there on its own shoulder and elbow rows while it rests
(own_arms: the goal ramped in before the touch and cross-faded between two supports as in the solve, the level
0's layered wrist and hand where the clip's stand where its level 0's do, HELD_WRIST_TOL, its rest gate open there,
physio-v7: the arms the seated trunk jerks carry over their supports; the correction
blended in by the transferred wrist's distance from it, ANCHOR_FIX_M, and by rest_weight, physio-v5: none while the
clip's wrist moves faster than ARM_REST_M_S relative to its support, ramped in over ARM_REST_S once it rests, so that a
hand flung off or onto its support takes no correction turning with it (a fallback of transfer where the anchoring
everywhere, physio-v4's, fails the acceleration or anchor check, or passes but turned where the hand moves, and the gate
passes too, its anchor check reading the knots where the hand stands still: a gate by default switched the anchoring on
and off where a reaching hand paused); smoothed in time and
ramped inside the clip's own rows over ARM_FIX_SMOOTH_S,
inside_ramp, or over GUARD_SMOOTH_S where the short ramp fails the acceleration check, or both over ARM_FIX_FAST_S where
it fails the anchor check; none where its shoulder or elbow row is held; a damped elbow step, ARM_FIX_DAMP_M, so that an
arm near straight never jumps); a joint pushed outside the table the clip is judged on (the healthy one, or limits_ext's
dystonia table at each knot, physio-v5) beyond the clip's own excursion is guarded on its own rows (own_limit_guard: the
factor rising to 1 at a held row through a smoothstep, a component still out within that ramp clamped; on the trunk
and legs before the hands are kept on their supports, on the arms after, physio-v10; where every variant of the arms'
correction fails, the guard smoothed longer, GUARD_SMOOTH_LONGER_S). The hand rows
take the level 0's finger deltas like every other row (physio-v4: the lift turns each finger joint about its flexion
axis in its parent frame, so it extends whatever the clip's hand posture), but a member's own finger rows held still
in a posture far from the level 0's (FINGER_RULE, FINGER_HELD_DEG_S, FINGER_HELD_FROM_DEG, within the level 0's length:
a clenched fist does not drift) take none, the drift faded out inside each run of them over FINGER_STILL_RAMP_S
(own_fingers_still, physio-v5), reduced on the clip's own rows where a finger joint would
leave the hand limits beyond the clip's own range (own_finger_guard, _finger_guard), and faded on them around the knots
where it would press a finger the clip holds otherwise into the body or a scene box (own_finger_fade, kept only where it
lowers that depth); a forearm or hand the layer presses into the body (the trunk moving into an arm the clip holds
against it) is turned out about its shoulder on the clip's own shoulder rows (own_arm_contact, contact_turn, physio-v5);
an arm the layer presses into a scene box (a forearm on an armrest's edge) is turned off it on the clip's own rows,
about the line from its shoulder to its wrist, or on a spine row where the shoulder's is held (own_scene_turn,
scene_turn). No other local fade of a member: the layer's fade would change the rows it shares with its level 0; a
member that still fails a check is refused by it. Knots beyond the level 0's length (a clip longer than its level 0)
take the clip's own
solve there, the level 0's last delta carried on at its last rate and blended into the own one over BEYOND_RAMP_S
(_extend; beyond_reference; the hand deltas into the identity where the own solve has no finger layer); a level 0 cannot
be compared there. Then the checks of the solve (a member's skate against SKATE_TOL_MEMBER_M_S), the anchor error read
at every knot of the clip's own touches where its shoulder and elbow rows are its own and its hand rests or stands
still (_anchor_reading: inside the ramp from a held arm, less what the held arm hands over; where either is held, the
level 0's row copied, which no own correction can turn, reported apart: anchor_error_held_rows_m; where the hand moves
on its support or its anchoring ramps in, reported apart: anchor_error_moving_m; where it stands still, ARM_DWELL_M_S
over ARM_DWELL_S, shown apart: anchor_error_dwell_m, read like every rest for a rest gate tried on a passing default,
which, blind to a pause shorter than its ramp, fails where it would leave a pausing hand off its support (reported
only for the gate a refused default falls back on); where the own limit guard holds
its shoulder or elbow, the goal unchanged, reported apart: anchor_error_guarded_m). The verdict is copied
into truth['validity']['background'] so that an export refuses a clip the layer still spoils
(render_skeleton_catalogue.module_failures).

Series (wrap_specs, used by catalogue.catalogue_set(background=True)): the clips whose level-0 chains end at the same
root share the seed and the layer; the posture kind comes from the root spec (its scene and segment), the resting
hands from the root body (built once per catalogue_set call); SeriesCache solves the root once and keeps the layered
level 0s (REF_CACHE). SKIP lists the series left without a layer and why. exam.exam_set() is not wrapped: the frozen
exam page keeps its bodies.
"""
import functools
import hashlib
import math
from collections import OrderedDict
from dataclasses import replace

import numpy as np
import torch

from ..body.rotations import axis_angle_to_matrix, matrix_to_axis_angle
from ..body.joints import N_BODY, fk
from .body import ELBOW_FLEXION_SIGN, F64, LIMIT_TOL_DEG, joint_limits, validity as body_validity

VERSION = "physio-v10"    # v8 (Phase D of docs/PLAN_2026-11-12.md): the longer rest gates of transfer, the
#                           scene and fold checks read again on the layered body (layer_checks_of); v9 (open-body port,
#                           wave 7): the root's anchoring blended in rotation by its weight, its check on the knots
#                           of the whole anchoring, the guard leaving a row the layer did not move, the mesh caches
#                           keyed on the body model, the freezing walk's still runs (with_hold_background still); v10
#                           (plan phase E): the joint-limit guard holds at GUARD_TOL_DEG, half the checks' tolerance;
#                           a root's rest gate and damped anchoring tried together where neither passes alone; a
#                           member's own pelvis keeping its legs' lengths under the level 0's sway (LEG_LENGTHS);
#                           a member's own limit guard on its trunk and legs before its arms (TRUNK_GUARD_FIRST);
#                           a refused member's variants tried again with its guard smoothed longer
SEED = 20261102                 # salt of the series seeds (series_seed)
HORIZON_S = 120.0               # [module] the layer's length; every catalogue clip lasts at most 60 s (longer: refused)
LAYER_RATE = 100.0              # the clock of every catalogue body

SOURCES = {
    "motamedifakhr2017": "Motamedi-Fakhr S, Iles R, Barney A, de Boer W, Conlon J, Khalid A, Wilson RC. Evaluation of "
                         "the agreement of tidal breathing parameters measured simultaneously using pneumotachography "
                         "and structured light plethysmography. Physiol Rep 2017; 5(3): e13124, doi "
                         "10.14814/phy2.13124 (repeatability cohort: 21 adults with no respiratory diagnosis, 25 to "
                         "72 years, seated upright in a high-backed chair, 45 s of tidal breathing; pneumotachograph "
                         "at REST1: respiratory rate median 14.13 breaths per minute, range 10.02 to 20.27; tI/tE "
                         "median 0.75, range 0.52 to 1.18)",
    "melo2009": "Melo PS, Ferreira TP, Santos-Pontelli TEG, Carneiro JAO, Carneiro AAO, Colafemina JF. Comparing "
                "static sitting postural sway of healthy young and older adults. Rev Bras Fisioter 2009; 13(6): "
                "549-554 (17 young adults, 19 to 28 years; seated without back or foot support, the upper limbs "
                "resting on the thighs, eyes open, stable surface, 90 s; magnetic sensor on the spinous process of "
                "T2: maximum amplitude 1.25 +- 0.49 cm anterior-posterior, 0.66 +- 0.32 cm latero-lateral, Table 1; "
                "the paper says the amplitude analyses the extreme points of the sway, read here as a peak-to-peak "
                "range). "
                "Their seat had no backrest; the catalogue chair's backrest stands viewer.CHAIR_BACK_CLEAR_M behind "
                "the trunk, so the layer's sagittal lean only leans forward from the clip's posture (a constant "
                "forward offset of the series, recorded in its truth as forward_offset_mean_deg) and the back never "
                "meets the backrest",
    "foulger2025": "Foulger LH, Reiter ER, Kuo C, Carpenter MG, Blouin J-S. Estimating whole-body centre of mass sway "
                   "during quiet standing with inertial measurement units. PLoS ONE 2025; 20(1): e0315851, doi "
                   "10.1371/journal.pone.0315851 (16 adults, 25 +- 3 years, three 120 s trials per stance; RMS of "
                   "the force-platform centre of mass at hip-width stance: 4.33 +- 1.51 mm anterior-posterior, "
                   "2.04 +- 1.20 mm medio-lateral)",
    "yamamoto2015": "Yamamoto T, Smith CE, Suzuki Y, Kiyono K, Tanahashi T, Sakoda S, Morasso P, Nomura T. Universal "
                    "and individual characteristics of postural sway during quiet standing in healthy young adults. "
                    "Physiol Rep 2015; 3(3): e12329, doi 10.14814/phy2.12329 (15 young men, 70 s eyes open: the "
                    "frequency halving the centre-of-pressure power, PF50, 0.329 +- 0.071 Hz anterior-posterior and "
                    "0.324 +- 0.066 Hz medio-lateral, Table 2; the low-frequency band of the sway, between about 0.04 "
                    "and 0.5 Hz, in their Results and Discussion; PF50 is a centre-of-pressure measure, the centre of "
                    "mass sways more slowly)",
    "winter1998": "Winter DA, Patla AE, Prince F, Ishac M, Gielo-Perczak K. Stiffness control of balance in quiet "
                  "standing. J Neurophysiol 1998; 80(3): 1211-1221, doi 10.1152/jn.1998.80.3.1211 (abstract: the "
                  "sagittal stiffness control at the ankle plantarflexors, the frontal one at the hip abductors and "
                  "adductors; the centre of pressure in phase with the centre of mass)",
    "module": "module values of mdfm/synth/physio.py where no source read gives the number (marked [module])",
}

# ---------------------------------------------------------------- layer parameters
BREATH_PER_MIN = 14.13          # SOURCES['motamedifakhr2017'], REST1 median rate
BREATH_TI_TE = 0.75             # SOURCES['motamedifakhr2017'], REST1 median tI/tE
BREATH_RATE_CV = 0.08           # [module] slow variation of the breathing rate (share of the rate, RMS)
BREATH_RATE_BAND_HZ = (0.01, 0.05)   # [module] band of that variation
BREATH_SPINE_DEG = 1.0          # [module] spine3 extension at full inhalation (the neck takes it back)
BREATH_COLLAR_DEG = 0.5         # [module] collar elevation at full inhalation
HEAD_DRIFT_DEG = (1.0, 0.7)     # [module] RMS yaw and pitch of the head drift
HEAD_BAND_HZ = (0.03, 0.25)     # [module] band of the head drift
SIT_T2_RANGE_M = (0.0125, 0.0066)   # SOURCES['melo2009'], young adults EOSS: AP and latero-lateral maximum amplitude
SIT_WINDOW_S = 90.0             # SOURCES['melo2009'], the trial over which the range is read
SIT_BAND_HZ = (0.05, 0.3)       # [module] band of the seated trunk drift
SWAY_RMS_M = (0.00433, 0.00204)  # SOURCES['foulger2025'], hip-width stance, AP and ML RMS of the centre of mass
SWAY_BAND_HZ = (0.04, 0.5)      # SOURCES['yamamoto2015'], the low-frequency band of the sway (Results, Discussion)
FINGER_DRIFT_DEG = 2.0          # [module] RMS of the MCP and PIP flexion drift of a resting hand
FINGER_LIFT_SOFT_DEG = 1.0      # [module] softness e of the one-sided finger lift 0.5 (x - sqrt(x^2 + e^2))
#                                 (finger_lift): about x where the drift x extends by more than e, about 0 where it
#                                 would flex by more than e, -e/2 at x = 0
FINGER_TABLE_SCENES = ("chair_table", "chair_table_both")   # [module] the writing scenes: no finger drift (module
#                                 docstring), since the resting hand's lowest point there is a fingertip
QUIET_PARTS = ("head", "fingers_left", "fingers_right")   # the parts of a series' layer SERIES_QUIET may leave still
# The series whose layer leaves a part still (physio-v6, the session's decision of 2026-11-10 under the owner's
# delegation, reports/gpu_queue/decisions.md): a hand on which a member's sign turns the forearm, the wrist or the
# fingers takes no finger drift, and a head whose rows carry a member's sign takes no head drift, at the series level,
# so that the level 0 and every clip of the series stay identical (extends the rule of 2026-11-03, a hand whose finger
# rows carry the sign takes no finger drift on them, FINGER_RULE): the drift's rotation turned with the member's distal
# sign, and the pair's change read as that share of the sign. Read on the pairs above the owner's aim (0.05 of the
# sign, scripts/synth_background_sweep.py SHARE_AIM) in the sweep on the code digest 75e36d6f, whose share with the
# finger rows raw lay under it (the finger layer's own share); the sweep lists every pair still above the aim
SERIES_QUIET = {
    "tremor_E1_rest_g0": ("fingers_left", "fingers_right"),     # the 'rest' driver turns the left forearm about its
    #   axis and flexes the wrist, the essential rest tremor flexes both wrists (0.125 and 0.084 of the sign, 0.014 and
    #   0.001 with the finger rows raw)
    "tremor_E1_pillrolling_g0": ("fingers_left",),               # the left fingers rub, the forearm turns (0.120 and
    #   0.089, 0.004 and 0.019 raw)
    "invol_E1_level0": ("fingers_left", "fingers_right"),        # the athetotic, choreic, dyskinetic and tardive hands
    #   and fingers (0.050 to 0.069, 0.010 to 0.049 raw)
    "jerk_E1_level0": ("fingers_left", "fingers_right"),         # the tics of the arms and hands (0.052, 0.050 raw)
    "dystonia_E1_level0": ("head",),                            # the head tremors and the cervical dystonias: the
    #   head drift turned the pitch tremor (0.052 of the sign at every grade; 0.031 without the head drift, measured on
    #   the working tree of 2026-11-10, reports/gpu_queue/decisions.md)
}
FINGER_BAND_HZ = (0.03, 0.25)   # [module]
FINGER_COMMON = 0.6             # [module] share of the drift common to the five fingers of a hand (0.4 their own)
N_SINES = 12                    # [module] sinusoids per band signal
CAL_SEEDS = 64                  # [module] seeds over which the expected 90 s range of the unit trunk signal is averaged

# ---------------------------------------------------------------- application parameters
SUPPORT_M = 0.015               # [module] a hand vertex this close to a scene box or to the body rests on it
HEAD_APPROACH_M = 0.05          # [module] a wrist or finger vertex this close to the head's vertices follows the head
#                                 (anchored to its frame: a hand reaching the nose, the chin or the cheek follows the
#                                 head drift before it touches, so that a clip stopping short of its level 0's touch
#                                 takes the same layer)
ANCHOR_RAMP_S = 1.0             # [module] the anchoring ramps in and out over this where a hand meets or leaves
SUPPORT_EVERY = 10              # [module] knots between two support readings (10 Hz)
GUARD_SMOOTH_S = 0.2            # [module] window of the blend factor's moving minimum and moving mean
GUARD_SMOOTH_LONGER_S = (0.3, 0.4, 0.5)   # [module] physio-v10: a member's own limit guard smoothed over these, in
#                                 turn, where every variant of transfer still fails (transfer's last fallback)
GUARD_TOL_DEG = 0.5 * LIMIT_TOL_DEG  # [module] the joint-limit guard holds a row within half the checks'
#   tolerance (body.LIMIT_TOL_DEG) of the bound, so that the exp-log round trip of a guarded row cannot carry it past
#   the check (plan phase E, D145: held at the full tolerance, the choreic fingers of E2 grade 3 read one component
#   1.0000000046e-6 deg past the table, limits_new 1)
REST_STILL_M = 0.002            # [module] a level-0 wrist moving less than this (and turning less than
REST_STILL_DEG = 0.5            # [module] this) holds its hand still
HANG_SHARE = 0.7                # [module] a hand hangs when its wrist lies this share of the forearm below the elbow
CHECK_EVERY = 25                # [module] knots between two scene and body checks (4 Hz at 100 Hz)
CONTACT_M = 0.01                # [module] the hand-to-body depth (signed_distance) is read on the forearm and hand
#                                 vertices of the layered body lying within this of a body vertex
SCENE_TOL_M = 0.001             # [module] depth the layer may add inside a scene box
BODY_TOL_M = 0.001              # [module] approach to the body or leg depth the layer may add
SKATE_TOL_M_S = 1e-6            # [module] the foot speed in stance the layer may add to a series root or an own solve
#                                 (p95, body.validity: its legs re-solved in full, as physio-v2 held every clip)
SKATE_TOL_MEMBER_M_S = 1e-3     # [module] the same for a member (transfer): one holding its knee or hip row equal to
#                                 its level 0's while the other differs keeps a residual under the sway that no own
#                                 row can take (legs_own_fix); a pooled p95 does not bound a slide, FOOT_TOL_M does
FOOT_TOL_M = 0.001              # [module] largest horizontal distance the layer may move a foot point
#                                 (body.foot_points) from the clip's at the clip's stance knots (validity
#                                 'foot_moved_m', standing)
ANCHOR_TOL_M = 0.001            # [module] largest distance of an anchored wrist from its target
ACCEL_TOL_M_S2 = 1.0            # [module] largest acceleration the layer may add to a joint beyond ACCEL_REL of the
ACCEL_REL = 0.1                 # [module] clip's own there (a rotation of a few degrees laid on a fast jerk changes
#                                 the jerk's acceleration by about the sine of that angle, and an anchored arm's
#                                 correction turns with a limb that jumps in the clip); a jump of 0.1 mm at one knot
#                                 of a smooth clip already reaches it at 100 Hz
EQUAL_ROW_TOL = 1e-9            # [module] a clip's raw row (rot6d values; the pelvis translation in m) this close to
#                                 its level 0's is held equal: its output row is the level 0's, copied (transfer); far
#                                 above the round-off a builder leaves on rows it does not change, far below a sign (a
#                                 rotation of 1e-9 rad moves a point 1 m away by 1 nm)
ANCHOR_FIX_M = (0.00025, 0.00075)   # [module] a member's own correction of a hand on its support blends in from the
#                                 first to the second distance of the transferred wrist from its target (smoothstep)
ARM_FIX_DAMP_M = 0.05           # [module] damping of the elbow flexion step of a member's own arm correction (m per
#                                 rad of the shoulder-to-wrist distance; two_bone_fix damp)
ARM_FIX_SMOOTH_S = 0.1          # [module] the moving means smoothing a member's own arm correction in time
ARM_FIX_FAST_S = 0.05           # [module] the same, and its ramp inside the own rows, where ARM_FIX_SMOOTH_S leaves a
#                                 hand off its support beyond ANCHOR_TOL_M (a jerk lifting or landing the hand;
#                                 transfer)
ARM_REST_M_S = 0.25             # [module] a member's own hand rests on its support where the clip's wrist moves slower
#                                 than this relative to the support (rest_weight); a hand flung off or onto it (the
#                                 hemiballism's fling, 500 to 900 deg/s at the shoulder, involuntary.BALLISM_FLING)
#                                 takes no own anchoring while it moves faster
ARM_REST_S = 0.1                # [module] the own anchoring ramps in over about twice this once the hand rests, and
#                                 out as long before it moves again (rest_weight)
ARM_REST_LONGER_S = (0.2, 0.3, 0.4, 0.5)   # [module] physio-v8: the rest gate's longer ramps tried, in turn, where the
#                                 gated anchoring of a passing default leaves the pair's change above PAIR_SHARE_AIM of
#                                 its sign (transfer): the first that passes and keeps under it is taken
PAIR_SHARE_AIM = 0.05           # [module] the change the layer brings to a clip minus its level 0, as a share of the
#                                 sign (pair_change; scripts/synth_background_sweep.py lists the pairs above it)
ARM_DWELL_M_S = 0.05            # [module] a member's own hand stands still on its support or target where the clip's
#                                 wrist moves slower than this relative to it at every knot within ARM_DWELL_S
ARM_DWELL_S = 0.05              # [module] (_anchor_reading): a pause too short for rest_weight to reach 1 still holds
#                                 the anchor check; the window leaves out the reversal of a tremor or a fling
SWAY_BISECT = 10                # [module] bisection rounds of a member's own sway scale (transfer, step 1)
CONTACT_PUSH_EVERY = 5          # [module] knots between two readings of contact_turn (20 Hz at 100 Hz)
CONTACT_PUSH_MARGIN_M = 0.0002  # [module] a pressed forearm or hand vertex is turned out this far beyond the surface
SCENE_AIM = 0.9                 # [module] a member's arm pressed into a scene box is turned back to this share of
#                                 SCENE_TOL_M (scene_turn: no more than the check needs)
FINGER_RULE = "held"            # which of a member's own finger rows take no finger drift (transfer): 'held', the rows
#                                 held still in a posture far from the level 0's (a clenched fist: no spontaneous
#                                 movement there); 'all', every own row (the session's decision of 2026-11-03: it leaves
#                                 the level 0's drift in the difference with the level 0 and raises the change of a
#                                 moving finger sign, reports/synth/background/finger_rule.json, so it is kept for that
#                                 measure only); 'none', no row (physio-v4)
FINGER_HELD_DEG_S = 1.0         # [module] a finger row held still turns slower than this, and
FINGER_HELD_FROM_DEG = 15.0     # [module] lies farther than this from the level 0's row (several times the drift's
#                                 RMS, FINGER_DRIFT_DEG: a posture, not a drift)
FINGER_STILL_RAMP_S = 0.5       # [module] the finger drift fades out over about twice this inside a run of a member's
#                                 own finger rows (transfer; physio-v5)
LEG_LENGTHS = True              # physio-v10: a member's own pelvis keeps its legs' lengths under the level 0's sway
#                                 (own_leg_lengths; False: physio-v9's transfer, read by
#                                 scripts/openbody_phaseE_checks.py)
TRUNK_GUARD_FIRST = True        # physio-v10: a member's own limit guard on its trunk and legs before its arms'
#                                 anchoring (transfer; False: physio-v9's single guard after the arms)
LEG_LENGTHS_ACCEL_M_S2 = 0.1 * ACCEL_TOL_M_S2   # [module] the acceleration the ramp of own_leg_lengths may add
#                                 at a held row's edge (its length set by the shift there)
LEG_LENGTHS_FIXED_RAMP_S = None  # a fixed ramp instead (s; None: the ramp above)
LEG_FIX_M = 1e-9                # [module] a transferred ankle this far from the clip's own is re-solved (own rows)
HELD_WRIST_TOL = 1e-9           # [module] a member's wrist this close to its level 0's (m, and the hand's global
#                                 rotation within this per entry) is held where the level 0 holds it (transfer,
#                                 physio-v7)
BEYOND_RAMP_S = 1.0             # [module] beyond its level 0's length a clip's own solve is blended in over this
REF_CACHE = 12                  # [module] layered level 0s kept per catalogue_set call (SeriesCache)
IK_ITERS = 8                    # [module] iterations of two_bone_fix
MAX_STEP = math.radians(3.0)     # [module] largest flexion step of two_bone_fix per iteration
STRAIGHT_STEP = math.radians(0.5)  # [module] flexion step of a straight limb whose goal lies closer

FADED = ("breath", "head_yaw_deg", "head_pitch_deg", "lean_ap_deg", "lean_ml_deg", "sway_ap_m", "sway_ml_m",
         "fingers_deg")
FADED_FIRST = tuple(k for k in FADED if not k.startswith("sway"))   # the sway moves the body as one: faded last
FADE_TRIES = 3                  # [module] rounds of the local fade (LayeredBuild)
FADE_HOLD_S = 0.5               # [module] the local fade holds the clip this long around a failing knot
KINDS = {"seated": ("breath", "head", "trunk", "fingers"), "standing": ("breath", "head", "sway", "fingers"),
         "supine": ("breath",), "transfer": ("breath", "head", "sway"), "walking": ("breath", "head", "sway")}
SEATED_SCENES = ("chair", "chair_armless", "chair_table", "chair_table_both", "chair_target")
WALK_SEGMENTS = ("E8", "E8t", "E19", "E7p")
STILL_M_S = 0.005               # [module] a knot is still where the pelvis, both ankles and both feet (STILL_JOINTS)
STILL_JOINTS = (0, 7, 8, 10, 11)    # all move slower than STILL_M_S (still_weight: the generated walks stand exactly
#                                 still before their first step and after their last, a 0.005 m/s pelvis moves 1 mm in
#                                 0.2 s)
STILL_MIN_S = 0.5               # [module] a still run shorter than this takes no layer (a pause between two steps)
STILL_FADE_S = 0.5              # [module] the layer crossfades over this inside a still run or a freezing hold, from 0
#                                 at an edge the body moves across (raised cosine; the review's 0.5 s of the arms'
#                                 settling, freezing.ARM_SETTLE_S)
STAND_HIGH_M = 0.05             # [module] a transfer's standing phase: its still runs whose pelvis lies within this of
#                                 its highest (stand_weight)
SPAN_KEYS = {"walking": FADED, "transfer": ("sway_ap_m", "sway_ml_m")}   # the layer's signals a kind's span weighs
ARM = {"left": (13, 16, 18, 20), "right": (14, 17, 19, 21)}       # collar, shoulder, elbow, wrist
LEG = {"left": (1, 4, 7), "right": (2, 5, 8)}                     # hip, knee, ankle
TRUNK_JOINTS = (3, 9, 12, 13, 14, 15)
SUPPORTS = {"head": (12, 15), "lap": (1, 2, 4, 5)}                # body joints a hand may rest on (support_joints)
TRUNK_TOUCH = (3, 6, 9)                                           # and, for a member's own touches seated, supine
#                                                                   or arising, the trunk (standing, an arm hanging
#                                                                   beside the hip reads as touching it)
HEAD = 15                                                         # the head joint a hand approaching the face follows
LEG_JOINTS = (1, 2, 4, 5, 7, 8)
ARM_JOINTS = (16, 17, 18, 19, 20, 21)

_OWN = "carries its own physiological background layer"
SKIP = {
    # open-body port, wave 7 (exam_v10): the AMASS references left the exam set; the procedural stand (ref_stand_E7)
    # and the healthy walk (ref_walk_E8) take the layer as any standing or walking series; the freezing clips' walk
    # lays it in its own build, on the walk's still runs and the clips' holds
    "freezing_E8_level0": ("the freezing clips lay the layer on their holds and on the walk's still runs in their own "
                           "build, as their level 0 does (physio.with_hold_background, still)"),
    "park_E7_arise_g0": f"{_OWN} (parkinsonism.background_signals: the 3.9 arising, the abrupt arising, Gowers)",
    "park_E1_spontaneity_g0": f"{_OWN} (parkinsonism.background_signals, graded with 3.14)",
    "park_E7_spontaneity_g0": f"{_OWN} (parkinsonism.background_signals, graded with 3.14)",
    "park_E22_sitdown_level0": f"{_OWN} (parkinsonism.sitdown_body)",
    "park_E21_applause_g0": f"{_OWN} (parkinsonism.applause_parts)",
    "park_E1_motorstate_level0": f"{_OWN} (parkinsonism.ms_base)",
    "atax_SARA2_natural_g0": "the stance sway is the sign (SARA 2)",
    "atax_SARA2_together_g0": "the stance sway is the sign (SARA 2)",
    "atax_SARA2_tandem_g0": "the stance sway is the sign (SARA 2)",
    "atax_ROMB_level0": "the stance sway is the sign (Romberg)",
    "atax_SARA3_sitting_g0": "the seated sway is the sign (SARA 3)",
    "atax_TITUB_g0": f"{_OWN} (ataxia.seated_background) and the titubation is the sign",
}


# ---------------------------------------------------------------- signals
def series_seed(root_id):
    """The seed of a series: the first 32 bits of SHA-256 of 'SEED:root_id'."""
    return int(hashlib.sha256(f"{SEED}:{root_id}".encode("utf-8")).hexdigest()[:8], 16)


class Band:
    """A unit-RMS band signal: N_SINES sinusoids at frequencies drawn log-uniformly in band (Hz: as many per octave),
    weights in [0.5, 1] times sqrt(band[0] / f) (the power per octave falls as 1/f: the signal wanders slowly with
    smaller quicker ripples instead of rocking at the band's top; a module choice), random phases; normalised by the
    long-time RMS sqrt(sum a^2 / 2), so that its value at t does not depend on the span evaluated. integral(t) is its
    closed-form integral from 0."""

    def __init__(self, rng, band, n=N_SINES):
        self.f = np.exp(rng.uniform(math.log(band[0]), math.log(band[1]), size=n))
        self.ph = rng.uniform(0.0, 2.0 * math.pi, size=n)
        self.a = rng.uniform(0.5, 1.0, size=n) * np.sqrt(band[0] / self.f)
        self.norm = math.sqrt(float((self.a ** 2).sum()) / 2.0)

    def __call__(self, t):
        t = np.asarray(t, dtype=np.float64)
        return (self.a[:, None] * np.sin(2.0 * math.pi * self.f[:, None] * t[None] + self.ph[:, None])).sum(0) \
            / self.norm

    def integral(self, t):
        t = np.asarray(t, dtype=np.float64)
        w = 2.0 * math.pi * self.f[:, None]
        return ((self.a[:, None] / w) * (np.cos(self.ph[:, None]) - np.cos(w * t[None] + self.ph[:, None]))).sum(0) \
            / self.norm


def breath_wave(phase_cycles):
    """(T,) 0 exhaled to 1 inhaled at phase (cycles): a raised cosine up over the inspiratory share BREATH_TI_TE /
    (1 + BREATH_TI_TE) of each cycle and down over the rest."""
    u = np.mod(np.asarray(phase_cycles, dtype=np.float64), 1.0)
    si = BREATH_TI_TE / (1.0 + BREATH_TI_TE)
    up = 0.5 - 0.5 * np.cos(math.pi * np.clip(u / si, 0.0, 1.0))
    dn = 0.5 + 0.5 * np.cos(math.pi * np.clip((u - si) / (1.0 - si), 0.0, 1.0))
    return np.where(u < si, up, dn)


def sit_gains():
    """sit_gains_of the selected body model (its rest skeleton sets the levers)."""
    return sit_gains_of(_model())


@functools.lru_cache(maxsize=2)
def sit_gains_of(model):
    """{'lever_ap_m', 'lever_ml_m' (m per rad of spine1 lean at the neck joint, on the procedural 'sit'),
    'range_per_rms' (the mean over CAL_SEEDS seeds of the SIT_WINDOW_S range of a unit SIT_BAND_HZ signal),
    'lean_rms_deg' (AP, ML): the lean RMS whose expected range at the neck joint is SIT_T2_RANGE_M}."""
    from .base import static_body
    from .body import axis_rotation
    b = static_body("sit", 0.01)
    R, tr, rest = b.local_R()[:1], b.transl[:1], b.rest()
    J0 = fk(R, tr, rest)[0][0, 12].numpy()
    out = {}
    eps = 1e-4
    for name, ax, comp in (("lever_ap_m", "x", 2), ("lever_ml_m", "z", 0)):
        R2 = R.clone()
        R2[:, 3] = axis_rotation(ax, torch.tensor(eps, dtype=F64)) @ R2[:, 3]
        out[name] = float(abs(fk(R2, tr, rest)[0][0, 12].numpy()[comp] - J0[comp]) / eps)
    t = np.arange(int(round(SIT_WINDOW_S * LAYER_RATE)) + 1) / LAYER_RATE
    ranges = []
    for s in range(CAL_SEEDS):
        x = Band(np.random.default_rng([SEED, s]), SIT_BAND_HZ)(t)
        ranges.append(float(np.ptp(x)))
    r = float(np.mean(ranges))
    out["range_per_rms"] = r
    out["lean_rms_deg"] = (math.degrees(SIT_T2_RANGE_M[0] / (out["lever_ap_m"] * r)),
                           math.degrees(SIT_T2_RANGE_M[1] / (out["lever_ml_m"] * r)))
    return out


@functools.lru_cache(maxsize=256)
def make_layer(seed, kind, rate=LAYER_RATE, quiet=()):
    """The layer of a series (module docstring) on the horizon: {'t' (N,), 'breath', 'head_yaw_deg',
    'head_pitch_deg', 'lean_ap_deg', 'lean_ml_deg', 'sway_ap_m', 'sway_ml_m' (N,), 'fingers_deg' (N, 2, 5)}, the
    components of kind (KINDS) non-zero, the others zero; arrays read-only. Every draw happens whatever the kind, in
    one order. quiet: parts of QUIET_PARTS left at zero (the series' SERIES_QUIET: 'head' the head drift,
    'fingers_left' and 'fingers_right' a hand's finger drift), the other signals bit for bit those of the full
    layer."""
    if kind not in KINDS:
        raise ValueError(f"posture kind {kind!r}")
    bad = sorted(set(quiet) - set(QUIET_PARTS))
    if bad:
        raise ValueError(f"quiet parts {bad}")
    rng = np.random.default_rng(int(seed))
    t = np.arange(int(round(HORIZON_S * rate)) + 1, dtype=np.float64) / rate
    f0 = BREATH_PER_MIN / 60.0
    rate_var = Band(rng, BREATH_RATE_BAND_HZ, 3)
    phase0 = float(rng.uniform(0.0, 1.0))
    breath = breath_wave(phase0 + f0 * (t + BREATH_RATE_CV * rate_var.integral(t)))
    yaw, pitch = Band(rng, HEAD_BAND_HZ)(t), Band(rng, HEAD_BAND_HZ)(t)
    lean_ap, lean_ml = Band(rng, SIT_BAND_HZ)(t), Band(rng, SIT_BAND_HZ)(t)
    sway_ap, sway_ml = Band(rng, SWAY_BAND_HZ)(t), Band(rng, SWAY_BAND_HZ)(t)
    common = [Band(rng, FINGER_BAND_HZ)(t) for _ in range(2)]
    fing = np.zeros((len(t), 2, 5))
    k = math.sqrt(FINGER_COMMON ** 2 + (1.0 - FINGER_COMMON) ** 2)
    for si in range(2):
        for fi in range(5):
            fing[:, si, fi] = (FINGER_COMMON * common[si] + (1.0 - FINGER_COMMON) * Band(rng, FINGER_BAND_HZ)(t)) / k
    comps = KINDS[kind]
    z = np.zeros_like(t)
    out = {"t": t, "breath": breath if "breath" in comps else z.copy()}
    on = "head" in comps
    out["head_yaw_deg"] = HEAD_DRIFT_DEG[0] * yaw if on else z.copy()
    out["head_pitch_deg"] = HEAD_DRIFT_DEG[1] * pitch if on else z.copy()
    if "trunk" in comps:
        g = sit_gains()["lean_rms_deg"]
        ap = g[0] * lean_ap
        out["lean_ap_deg"] = ap - ap.min()                  # forward only (module docstring)
        out["lean_ml_deg"] = g[1] * lean_ml
    else:
        out["lean_ap_deg"], out["lean_ml_deg"] = z.copy(), z.copy()
    on = "sway" in comps
    out["sway_ap_m"] = SWAY_RMS_M[0] * sway_ap if on else z.copy()
    out["sway_ml_m"] = SWAY_RMS_M[1] * sway_ml if on else z.copy()
    out["fingers_deg"] = FINGER_DRIFT_DEG * fing if "fingers" in comps else np.zeros_like(fing)
    if "head" in quiet:
        out["head_yaw_deg"], out["head_pitch_deg"] = z.copy(), z.copy()
    for si, side in enumerate(ARM):
        if f"fingers_{side}" in quiet:
            out["fingers_deg"][:, si] = 0.0
    for v in out.values():
        v.setflags(write=False)
    return out


def finger_lift(x, e=FINGER_LIFT_SOFT_DEG):
    """(...) deg: the one-sided lift of a finger drift x (deg, flexion positive): 0.5 (x - sqrt(x^2 + e^2)), never
    positive (the fingers only extend from the posture, never curl into a support), about x where x < -e, about 0
    where x > e, smooth; no constant offset beyond -e/2 (physio-v4: v3 took the drift minus its horizon maximum, a
    permanent extension by that maximum)."""
    x = np.asarray(x, dtype=np.float64)
    return 0.5 * (x - np.sqrt(x * x + e * e))


def layer_prefix(layer, T):
    """The first T knots of every signal of a layer (a clip longer than the horizon is refused)."""
    if T > len(layer["t"]):
        raise ValueError(f"a clip of {T} knots is longer than the layer's horizon ({len(layer['t'])} knots)")
    return {k: v[:T] for k, v in layer.items()}


def layer_digest(layer, T):
    """SHA-256 of the first T knots of every signal of a layer (the identity of a clip's background)."""
    h = hashlib.sha256()
    for k in sorted(layer):
        h.update(k.encode("utf-8") + np.ascontiguousarray(layer[k][:T]).tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------- spans of the layer (physio-v7)
def run_ramp(n, fade_in, fade_out, m):
    """(n,) 1, with a raised cosine from 0 over m knots at each faded end (0 on the end knot itself; m at most n // 2
    when both ends fade, n when one does)."""
    w = np.ones(n)
    m = int(min(m, n // 2 if fade_in and fade_out else n))
    if m > 0:
        r = 0.5 - 0.5 * np.cos(np.pi * np.arange(m) / m)
        if fade_in:
            w[:m] = np.minimum(w[:m], r)
        if fade_out:
            w[n - m:] = np.minimum(w[n - m:], r[::-1])
    return w


def _runs(mask):
    m = np.r_[False, np.asarray(mask, dtype=bool), False]
    e = np.flatnonzero(np.diff(m.astype(np.int8)))
    return list(zip(e[::2].tolist(), e[1::2].tolist()))


def still_runs(body, joints=STILL_JOINTS, speed=STILL_M_S, min_s=STILL_MIN_S):
    """[(a, b)]: the knot runs [a, b) in which every joint of `joints` moves slower than `speed` (central
    differences), min_s long at least: where a walking clip stands."""
    rate = float(body.rate)
    J = body.joints().numpy()[:, list(joints)]
    if len(J) < 2:
        return []
    v = np.linalg.norm(np.gradient(J, 1.0 / rate, axis=0), axis=-1).max(-1)
    n = int(round(min_s * rate))
    return [(a, b) for a, b in _runs(v < speed) if b - a >= n]


def weight_of_runs(runs, T, rate, fade_s=STILL_FADE_S):
    """(T,) 0 to 1: 1 on each run, fading in and out over fade_s inside it (run_ramp) at an edge the body moves
    across, none at the clip's first or last knot (the background went on before and goes on after), 0 elsewhere; a
    run shorter than its fades (fade_s per faded edge) takes none: a layer faded in and out over a shorter pause would
    rock the head (gait_E8_ataxic_sara1's 0.5 s pause: 1.5 m/s2 at the head over two 0.25 s fades)."""
    w = np.zeros(T)
    m = int(round(fade_s * rate))
    for a, b in runs:
        if b - a < m * (int(a > 0) + int(b < T)):
            continue
        w[a:b] = np.maximum(w[a:b], run_ramp(b - a, a > 0, b < T, m))
    return w


def still_weight(body, fade_s=STILL_FADE_S):
    """(T,) the span of a walking clip's layer (posture kind 'walking'): its still runs (still_runs), crossfaded."""
    return weight_of_runs(still_runs(body), body.T, float(body.rate), fade_s)


def stand_weight(body, fade_s=STILL_FADE_S):
    """(T,) the span of a transfer's sway (posture kind 'transfer'): its still runs whose pelvis lies within
    STAND_HIGH_M of its highest over the body (the standing phase after the rise), crossfaded."""
    y = body.joints().numpy()[:, 0, 1]
    top = float(y.max())
    runs = [(a, b) for a, b in still_runs(body) if float(np.median(y[a:b])) >= top - STAND_HIGH_M]
    return weight_of_runs(runs, body.T, float(body.rate), fade_s)


def hold_weight(truth, T, rate, fade_s=STILL_FADE_S):
    """(T,) the span of a freezing clip's layer: each hold of freezing.standing_spans (the body stopped on both feet,
    trembling or akinetic), crossfaded inside it; the ramps in and out, where the walk slows and resumes, unlayered."""
    from .freezing import standing_spans
    runs = []
    for sp in standing_spans(truth):
        a, b = int(round(sp["hold_s"][0] * rate)), int(round(sp["hold_s"][1] * rate))
        if b > a:
            runs.append((max(0, a), min(T, b)))
    return weight_of_runs(runs, T, rate, fade_s)


def span_record(span, rate, keys=FADED):
    """{'spans_s', 'share', 'keys'} of a span weight (JSON-ready): the runs where it is above 0, in s."""
    sp = np.asarray(span, dtype=np.float64)
    return {"spans_s": [[a / rate, b / rate] for a, b in _runs(sp > 0.0)], "share": float((sp > 0.0).mean()),
            "full_share": float((sp >= 1.0 - 1e-12).mean()), "keys": list(keys), "fade_s": STILL_FADE_S}


FREEZING_ROOT = "freezing_E8_level0"   # the exam's freezing clips' walk (exam.py, exam_v10; until then ref_walk_E8,
#                                         the AMASS walk)


def with_hold_background(body, truth, seed=None, still=False):
    """(Body, truth) of a freezing clip with the standing layer on its holds (M3b, wave SH1d part C, physio-v7): the
    series seed of the walk it is built on (FREEZING_ROOT) unless given, posture kind 'walking' (breathing, head
    drift, sway; the hands hang: no finger drift), hold_weight its span, solve() with its local fades; every knot
    outside the holds the clip's own bit for bit. still (open-body port, wave 7): the walk's own still runs too
    (still_weight: its standing before the first step and after the stop), as a walking series takes the layer, for
    the procedural walk of exam_v10 whose series takes none (SKIP) and which stands still at both ends (the AMASS walk
    never stood still). The record goes to truth['params']['hold_background'], its verdict to
    truth['validity']['hold_background'] (an export refuses a clip whose layer fails its checks)."""
    seed = series_seed(FREEZING_ROOT) if seed is None else int(seed)
    rate = float(body.rate)
    w = hold_weight(truth, body.T, rate)
    if still:
        w = np.maximum(w, still_weight(body))
    layer = make_layer(seed, "walking", rate)
    out, rec = solve(body, layer, kind="walking", resting={"left": False, "right": False}, span=w)
    truth = dict(truth)
    v = rec["validity"]
    truth["params"] = dict(truth.get("params") or {})
    truth["params"]["hold_background"] = {
        "version": VERSION, "seed": seed, "kind": "walking", "still": bool(still), "components": rec["components"],
        "span": rec["span"],
        "deviations": sorted(rec["deviations"]), "ok": bool(v["ok"]), "failed": list(v["failed"]),
        "foot_moved_m": v.get("foot_moved_m", 0.0), "accel_excess_m_s2": v["accel_excess_m_s2"],
        "amplitudes": amplitudes("walking")}
    truth["validity"] = dict(truth.get("validity") or {})
    truth["validity"]["hold_background"] = {"ok": bool(v["ok"]), "failed": list(v["failed"]), "version": VERSION}
    return out, truth


# ---------------------------------------------------------------- rotations
def _rot(axis, deg):
    from .body import axis_rotation
    return axis_rotation(axis, torch.from_numpy(np.radians(np.asarray(deg, dtype=np.float64))))


def _aa(R):
    return matrix_to_axis_angle(R)


def _exp(aa):
    return axis_angle_to_matrix(aa)


def _rmin(a, b):
    from .ik import rmin
    return rmin(a, b)


def _unit(v):
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _smooth_min(lam, n):
    """A moving minimum of width 2n + 1 then two moving means of width n (never above lam: the two means span 2n - 1
    knots inside the minimum's window; continuous with a continuous slope, so that a blend shows no kink)."""
    lam = np.asarray(lam, dtype=np.float64)
    if n <= 1:
        return lam
    m = 2 * n + 1
    lo = np.lib.stride_tricks.sliding_window_view(np.pad(lam, (n, n), mode="edge"), m).min(1)
    return np.minimum(_moving_mean(_moving_mean(lo, n), n), lam)


def _moving_mean(x, n, axis=0):
    """Centred moving mean of width n along axis 0 (edges held), x (T, ...)."""
    if n <= 1:
        return x
    pad = [(n // 2, n - 1 - n // 2)] + [(0, 0)] * (x.ndim - 1)
    c = np.cumsum(np.pad(x, pad, mode="edge"), axis=0)
    c = np.concatenate([np.zeros((1,) + c.shape[1:]), c], 0)
    return (c[n:] - c[:-n]) / n


# ---------------------------------------------------------------- the joint-limit guard
def _distance_to(mask):
    """(T,) float: knots from each knot to the nearest True of mask (inf without one)."""
    m = np.asarray(mask, bool)
    T = len(m)
    d = np.full(T, np.inf)
    idx = np.flatnonzero(m)
    if not len(idx):
        return d
    k = np.arange(T)
    i = np.clip(np.searchsorted(idx, k), 1, len(idx)) - 1
    j = np.minimum(i + 1, len(idx) - 1)
    return np.minimum(np.abs(k - idx[i]), np.abs(idx[j] - k)).astype(np.float64)


def guard_joints(R_out, R_in, joints, limits, rate, tol=GUARD_TOL_DEG, allowed=None, smooth_s=GUARD_SMOOTH_S,
                 table="healthy"):
    """R_out (T, 22, 3, 3) with each joint of joints blended back toward R_in along the geodesic at the knots where
    an axis-angle component would leave [min(lo, in), max(hi, in)] of the table limits (module docstring); returns
    (R, {joint: knots blended}). The allowed factor is found by bisection, then smoothed (_smooth_min). allowed: (T,
    22) bool, the rows the guard may change (transfer: a member clip's own rows; default every row); the others keep
    R_out bit for bit, and the factor ramps from 1 at a row held over smooth_s (no jump where the clip's rows
    leave its level 0's; a limit reached within that ramp is reported by the checks, not hidden). table other than
    'healthy' (limits_ext's 'dystonia', or its 'shrug' where that module has it): the bounds of that table at each
    knot instead (knot_bounds against R_in, the shoulder's adduction widened by its collar's rise over R_in in R as
    guarded so far: the table such a clip is judged on, physio-v5)."""
    R = R_out.clone()
    n = max(1, int(round(smooth_s * rate)))
    stats = {}
    c_in_all = np.degrees(_aa(R_in).numpy()) if table != "healthy" else None
    for j in joints:
        ok_rows = np.ones(R.shape[0], bool) if allowed is None else np.asarray(allowed[:, j], bool)
        if not ok_rows.any():
            continue
        c_in = np.degrees(_aa(R_in[:, j]).numpy())
        if table != "healthy":
            from . import limits_ext as LX
            lo_b, hi_b = LX.knot_bounds(table, np.degrees(_aa(R).numpy()), c_in_all)
            lo, hi = lo_b[:, j] - tol, hi_b[:, j] + tol
        else:
            lo = np.minimum(limits.lo[j], c_in) - tol
            hi = np.maximum(limits.hi[j], c_in) + tol
        D = _aa(R_in[:, j].transpose(-1, -2) @ R[:, j])

        def inside(lam, idx):
            ti = torch.from_numpy(np.asarray(idx, dtype=np.int64))
            Rl = R_in[ti, j] @ _exp(D[ti] * torch.from_numpy(np.asarray(lam, dtype=np.float64))[:, None])
            c = np.degrees(_aa(Rl).numpy())
            return ((c >= lo[idx]) & (c <= hi[idx])).all(-1)
        T = R.shape[0]
        lam = np.ones(T)
        # a knot whose row the layer left as it came in is inside its own bounds ([min(lo, in), max(hi, in)]) by
        # their definition: not read again (open-body port, wave 7: the axis-angle round trip of a row held at a bound
        # of the table, the supine spine of D93, read up to 1e-6 deg past it and the guard rewrote the row by 1e-9)
        moved = (R[:, j] != R_in[:, j]).flatten(1).any(1).numpy()
        bad = np.flatnonzero(~inside(lam, np.arange(T)) & ok_rows & moved)
        if not len(bad):
            continue
        a, b = np.zeros(len(bad)), np.ones(len(bad))
        for _ in range(30):
            m = 0.5 * (a + b)
            ok = inside(m, bad)
            a, b = np.where(ok, m, a), np.where(ok, b, m)
        lam[bad] = a
        # the factor rises to 1 at a held row through a smoothstep of the distance to it, composed with the smoothed
        # factor as floor + (1 - floor) factor: no kink where the clip's rows leave the held ones (physio-v4 took
        # the larger of the factor and a linear floor, whose kinks jumped a fast own arm's acceleration at the start
        # of a fling, invol_hemiballism_E1_left_g2)
        x = np.zeros(T) if allowed is None else np.clip(1.0 - _distance_to(~ok_rows) / n, 0.0, 1.0)
        floor = x * x * (3.0 - 2.0 * x)
        for _ in range(4):
            lam_s = np.where(ok_rows, floor + (1.0 - floor) * _smooth_min(lam, n), 1.0)
            still = np.flatnonzero(~inside(lam_s, np.arange(T)) & ok_rows)
            if not len(still):
                break
            lam[still] = 0.0
        Rj = R_in[:, j] @ _exp(D * torch.from_numpy(lam_s)[:, None])
        # the rows held, and those left whole, keep R_out bit for bit (wave 7: every row left whole, also where the
        # guard may change every row, so that a knot it does not blend keeps its value, not the exp-log round trip)
        keep = ~ok_rows | (lam_s >= 1.0 - 1e-12)
        Rj = torch.where(torch.from_numpy(keep)[:, None, None], R_out[:, j], Rj)
        if allowed is not None:
            # within the ramp from a held row the factor may stay above the allowed one: the components still out
            # are clamped to the bounds there (a small excess, so a small kink)
            c = np.degrees(_aa(Rj).numpy())
            out_k = ok_rows & ~((c >= lo) & (c <= hi)).all(-1)
            if out_k.any():
                cc = np.clip(c[out_k], lo[out_k] + 0.5 * tol, hi[out_k] - 0.5 * tol)
                Rj[torch.from_numpy(out_k)] = _exp(torch.from_numpy(np.radians(cc)))
                lam_s = np.where(out_k, np.minimum(lam_s, 1.0 - 1e-9), lam_s)
        R[:, j] = Rj
        stats[int(j)] = int((lam_s < 1.0 - 1e-12).sum())
    return R, stats


# ---------------------------------------------------------------- two-bone correction (legs and arms)
def two_bone_fix(R, tr, rest, chain, goal, axis_of, end_global=None, knots=None, iters=IK_ITERS, damp=0.0,
                 flex=True, swing=True):
    """R (T, 22, 3, 3) with the proximal and middle joints of chain = (parent, proximal, middle, end) re-solved at
    the knots (default all) so that the end joint reaches goal (T, 3): the middle joint bends about axis_of(G_prox)
    (its anatomical flexion axis in the world, from the proximal segment's global rotation) by a Newton step on the
    proximal-to-end distance (at most MAX_STEP; a straight limb whose goal lies closer flexes by STRAIGHT_STEP), then
    the whole limb swings about the proximal joint by the minimal rotation onto the goal, IK_ITERS times; the
    end segment takes the global rotation end_global (T, 3, 3) (default: kept). Other components of the joints
    (twist, carrying angle) are kept. damp (m per rad, > 0): a damped flexion step, (dg - d) deriv / (deriv^2 +
    damp^2), so that a limb near straight leaves the distance it cannot take without a large flexion instead of
    jumping (a member's own arm correction, ARM_FIX_DAMP_M). flex False keeps the middle joint, swing False the
    proximal one (a member's own leg correction where only one of them is the clip's own: legs_own_fix). Returns (R,
    the reached end positions (T, 3))."""
    par, p, m, e = chain
    R = R.clone()
    J, G = fk(R, tr, rest)
    k = torch.arange(R.shape[0]) if knots is None else torch.as_tensor(knots, dtype=torch.int64)
    if not len(k):
        return R, J[:, e]
    S, E, W = J[k, p], J[k, m], J[k, e]
    Gp, Gm = G[k, p], G[k, m]
    Ge = G[k, e] if end_global is None else end_global[k]
    tgt = goal[k]
    L1, L2 = (E - S).norm(dim=-1), (W - E).norm(dim=-1)
    for _ in range(iters):
        d = (W - S).norm(dim=-1)
        dg = (tgt - S).norm(dim=-1).clamp(min=(L1 - L2).abs() + 1e-9, max=L1 + L2)
        ax = _unit(axis_of(Gp))
        deriv = ((E - S) * torch.cross(ax, W - E, dim=-1)).sum(-1) / d          # d(distance) / d(flexion)
        safe = deriv.abs() > 1e-5
        if damp > 0.0:
            inc = (dg - d) * deriv / (deriv * deriv + damp * damp)
        else:
            inc = torch.where(safe, (dg - d) / torch.where(safe, deriv, torch.ones_like(deriv)),
                              torch.where(dg < d, torch.full_like(d, STRAIGHT_STEP), torch.zeros_like(d)))
        inc = inc.clamp(-MAX_STEP, MAX_STEP)
        inc = torch.where((dg - d).abs() < 1e-11, torch.zeros_like(inc), inc)
        if not flex:
            inc = torch.zeros_like(inc)
        Q = _exp(ax * inc[:, None])                                  # flexion increment about the anatomical axis
        Gm = Q @ Gm
        W = E + torch.einsum("nij,nj->ni", Q, W - E)
        P = _rmin(_unit(W - S), _unit(tgt - S)) if swing else torch.eye(3, dtype=F64).expand(len(k), 3, 3)
        Gp, Gm = P @ Gp, P @ Gm
        E = S + torch.einsum("nij,nj->ni", P, E - S)
        W = S + torch.einsum("nij,nj->ni", P, W - S)
    R[k, p] = G[k, par].transpose(-1, -2) @ Gp
    R[k, m] = Gp.transpose(-1, -2) @ Gm
    R[k, e] = Gm.transpose(-1, -2) @ Ge
    reached = J[:, e].clone()
    reached[k] = W
    return R, reached


def _knee_axis(Gp):
    return Gp[..., :, 0]                                    # knee flexion +x of the thigh (base.py)


def _elbow_axis(side):
    s = ELBOW_FLEXION_SIGN[side]
    return lambda Gp: s * Gp[..., :, 1]                     # elbow flexion about the upper arm's y axis (body.py)


# ---------------------------------------------------------------- standing sway
def stand_sway(R, tr, rest, dz, dx, R_ref, tr_ref, iters=8):
    """(R, tr, reached ankles (T, 2, 3)) of a standing body with the pelvis displaced dz along the heading (the whole
    body turns about the lateral axis through the ankles' midpoint: an ankle strategy) and dx sideways (the pelvis
    shifts with the trunk upright: a hip strategy), then raised or lowered and rolled about the heading so that both
    hip-to-ankle distances keep their values in the reference (R_ref, tr_ref: the clip), the spine taking the roll
    back (the trunk keeps its global orientation), and the legs re-solved (two_bone_fix) so that the ankles keep the
    reference's positions and the feet its global rotations. Keeping the leg lengths keeps every knee angle: a straight
    knee never has to bend (a straight leg made shorter bends by the square root of the shortening, a kink)."""
    J, G = fk(R, tr, rest)
    Jr, Gr = fk(R_ref, tr_ref, rest)
    up = torch.tensor([0.0, 1.0, 0.0], dtype=F64)
    A = 0.5 * (Jr[:, 7] + Jr[:, 8])
    lat = G[:, 0, :, 0].clone()
    lat[:, 1] = 0.0
    lat = _unit(lat)                                         # the subject's left, horizontal
    fwd = torch.cross(lat, up.expand_as(lat), dim=-1)       # the heading
    h = ((J[:, 0] - A) @ up).numpy()
    th = torch.from_numpy(np.arctan2(np.asarray(dz, dtype=np.float64), h))
    Rap = _exp(lat * th[:, None])
    J0 = A + torch.einsum("tij,tj->ti", Rap, J[:, 0] - A) + lat * torch.from_numpy(np.asarray(dx, np.float64))[:, None]
    G0 = Rap @ G[:, 0]
    hips = [LEG[s][0] for s in LEG]
    anks = [LEG[s][2] for s in LEG]
    d0 = torch.stack([(Jr[:, a] - Jr[:, hp]).norm(dim=-1) for hp, a in zip(hips, anks)], -1)

    def lengths(u):
        P = J0 + u[:, :1] * up
        Rr = _exp(fwd * u[:, 1:2])
        Gp = Rr @ G0
        return torch.stack([(P + torch.einsum("tij,j->ti", Gp, rest[hp] - rest[0]) - Jr[:, a]).norm(dim=-1)
                            for hp, a in zip(hips, anks)], -1)
    u = torch.zeros(J0.shape[0], 2, dtype=F64)
    eps = 1e-7
    for _ in range(iters):
        f = lengths(u) - d0
        Jm = torch.stack([(lengths(u + eps * torch.eye(2, dtype=F64)[c]) - lengths(u)) / eps for c in range(2)], -1)
        u = u - torch.linalg.solve(Jm, f[..., None])[..., 0]
    P = J0 + u[:, :1] * up
    G0p = _exp(fwd * u[:, 1:2]) @ G0
    R = R.clone()
    R[:, 3] = G0p.transpose(-1, -2) @ G0 @ R[:, 3]          # the roll taken back by the spine
    R[:, 0] = G0p
    tr2 = P - rest[0]
    reached = []
    for side, (hip, kn, ank) in LEG.items():
        R, w = two_bone_fix(R, tr2, rest, (0, hip, kn, ank), Jr[:, ank], _knee_axis, end_global=Gr[:, ank])
        reached.append(w)
    return R, tr2, torch.stack(reached, 1)


# ---------------------------------------------------------------- supports of the hands
def _model():
    """The selected body model's name (body.model_name): the key of every mesh cache of this module (open-body port,
    wave 7: keyed on nothing, an open-body read followed in one process by a read on the SMPL+H reference took the
    open body's vertex ids, an IndexError, design section 16.7)."""
    from .body import model_name
    return model_name()


def _probe_ids():
    """_probe_ids_of the selected body model."""
    return _probe_ids_of(_model())


@functools.lru_cache(maxsize=2)
def _probe_ids_of(model):
    """{'hand': {side: vertex ids}, 'fingers_mask': {side: (n,) bool}, 'body': ids, 'body_dom': their dominant
    joints, 'arm': {side: ids}} of the neutral mesh: the forearm and hand of each side (dominant joint the elbow, the
    wrist or a hand joint), every 2nd, the probes of the supports ('hand', the wrist and finger ones marked) and of
    the approach check ('arm'); every vertex of the body less both arms
    (dominant joint not 16 to 21 nor a hand joint: the mesh's spacing on the thigh is about the support distance)."""
    from .hands import FIRST
    from .viewer import dominant_joint
    dom = dominant_joint("neutral")
    out = {"hand": {}, "arm": {}, "fingers_mask": {}}
    for side, (_, s, e, w) in ARM.items():
        hj = list(range(FIRST[side], FIRST[side] + 15))
        out["hand"][side] = np.flatnonzero(np.isin(dom, [e, w] + hj))[::2]
        out["arm"][side] = out["hand"][side]
        out["fingers_mask"][side] = np.isin(dom[out["hand"][side]], [w] + hj)
    body = np.flatnonzero(~np.isin(dom, list(ARM_JOINTS) + list(range(22, 52))) & (dom >= 0))   # not the open
    #                                                     body's inner parts (dominant joint -1, design section 12.3)
    out["body"], out["body_dom"] = body, dom[body]
    return out


def scene_boxes(scene, body, floor_y):
    """[(origin (3,), axes (3, 3) unit rows, extents (3,))] of the boxes of a clip's scene (exam.scene_mesh, made of
    viewer.box parts of 8 vertices each), [] without a scene."""
    if scene is None:
        return []
    from .exam import scene_mesh
    v, _ = scene_mesh(scene, body, floor_y)
    v = np.asarray(v, dtype=np.float64)
    if len(v) % 8:
        raise ValueError(f"scene {scene!r}: {len(v)} vertices, not boxes of 8")
    out = []
    for b in v.reshape(-1, 8, 3):
        o = b[0]
        ax = np.stack([b[1] - o, b[3] - o, b[4] - o])
        ext = np.linalg.norm(ax, axis=1)
        out.append((o, ax / ext[:, None], ext))
    return out


def box_depth(P, boxes):
    """(..., ) m: the largest depth of points P (..., 3) inside any box (positive inside, the distance to the nearest
    face), and (..., ) m the distance to the nearest box (0 inside); (-inf, inf) without boxes."""
    if not boxes:
        return np.full(P.shape[:-1], -np.inf), np.full(P.shape[:-1], np.inf)
    depth, dist = [], []
    for o, ax, ext in boxes:
        q = (P - o) @ ax.T
        inner = np.minimum(q, ext - q).min(-1)
        out = np.linalg.norm(np.maximum(0.0, np.maximum(-q, q - ext)), axis=-1)
        depth.append(inner)
        dist.append(out)
    return np.max(depth, 0), np.min(dist, 0)


def _points(body, ids, knots, chunk=128):
    """(len(knots), len(ids), 3) the mesh vertices ids at the knots (viewer.knot_vertices, in chunks)."""
    from .viewer import knot_vertices
    knots = np.asarray(knots, dtype=np.int64)
    return np.concatenate([knot_vertices(body, knots[a:a + chunk])[:, ids] for a in range(0, len(knots), chunk)], 0)


def support_labels(body, boxes, joints=None, every=SUPPORT_EVERY, approach=True):
    """{side: (T,) int}: per knot the support of each hand in the body (module docstring): -1 none, -2 a scene box,
    else a body joint of joints (default every joint but the arms'), read every `every` knots (each knot takes the
    label of the nearest reading). In order: the head or neck when a vertex of the hand (wrist and fingers) lies within
    SUPPORT_M of their vertices (a hand touching the face, even with the forearm still near an armrest); a scene box
    within SUPPORT_M of a vertex of the forearm or hand; the other allowed joints (the dominant joint of the nearest
    body vertex to the forearm or hand within SUPPORT_M), the trunk joints of TRUNK_TOUCH only after them (a hand
    on the thigh whose forearm brushes the belly rests on the thigh); the head (HEAD) when a vertex of the hand lies
    within HEAD_APPROACH_M of the head's vertices (a hand approaching the face follows the head before it touches, so
    that a finger stopping short of its level 0's touch takes the same anchoring; approach False: touches only, the
    labels a member clip's own anchor check reads, transfer)."""
    from scipy.spatial import cKDTree
    ids = _probe_ids()
    T = body.T
    ks = np.unique(np.r_[np.arange(0, T, every), T - 1])
    allp = _points(body, np.concatenate([ids["hand"]["left"], ids["hand"]["right"], ids["body"]]), ks)
    nl, nr = len(ids["hand"]["left"]), len(ids["hand"]["right"])
    hands = {"left": allp[:, :nl], "right": allp[:, nl:nl + nr]}
    bodyp = allp[:, nl + nr:]
    bdom = ids["body_dom"]
    allowed = np.ones(len(bdom), bool) if joints is None else np.isin(bdom, list(joints))
    head = allowed & np.isin(bdom, SUPPORTS["head"])
    face = allowed & (bdom == HEAD)
    trunk = allowed & ~head & np.isin(bdom, TRUNK_TOUCH)
    rest_ = allowed & ~head & ~trunk
    labs = {side: np.full(len(ks), -1, dtype=np.int64) for side in hands}
    sdist = {}
    for side, P in hands.items():
        _, sd = box_depth(P, boxes)
        sdist[side] = sd.min(-1)
    for i in range(len(ks)):
        t_head = cKDTree(bodyp[i][head]) if head.any() else None
        t_rest = cKDTree(bodyp[i][rest_]) if rest_.any() else None
        t_trunk = cKDTree(bodyp[i][trunk]) if trunk.any() else None
        t_face = None
        for side, P in hands.items():
            lab = labs[side]
            fing = ids["fingers_mask"][side]
            if t_head is not None:
                d, j = t_head.query(P[i][fing], k=1)
                if d.min() <= SUPPORT_M:
                    lab[i] = int(bdom[head][j[int(np.argmin(d))]])
                    continue
            if sdist[side][i] <= SUPPORT_M:
                lab[i] = -2
                continue
            if t_rest is not None:
                d, j = t_rest.query(P[i], k=1)
                if d.min() <= SUPPORT_M:
                    lab[i] = int(bdom[rest_][j[int(np.argmin(d))]])
                    continue
            if t_trunk is not None:                    # the trunk last: a hand on the thigh whose forearm brushes
                d, j = t_trunk.query(P[i], k=1)        # the belly rests on the thigh
                if d.min() <= SUPPORT_M:
                    lab[i] = int(bdom[trunk][j[int(np.argmin(d))]])
                    continue
            if approach and face.any():
                if t_face is None:
                    t_face = cKDTree(bodyp[i][face])
                if float(t_face.query(P[i][fing], k=1, distance_upper_bound=HEAD_APPROACH_M + 1e-9)[0].min()) \
                        <= HEAD_APPROACH_M:
                    lab[i] = HEAD
    nearest = np.clip(np.searchsorted(ks, np.arange(T) - every / 2.0), 0, len(ks) - 1)
    return {side: lab[nearest] for side, lab in labs.items()}


def _hanging(J, side):
    _, _, e, w = ARM[side]
    fore = (J[:, w] - J[:, e]).norm(dim=-1)
    return bool(((J[:, e, 1] - J[:, w, 1]) >= HANG_SHARE * fore).all())


def resting_hands(body, boxes, kind="standing", labels=None):
    """{side: bool}: the hands a level-0 body holds still (the wrist moving less than REST_STILL_M and turning less
    than REST_STILL_DEG, the hand joints fixed) and supported (support_labels at every knot; labels: those of the
    body, read here by default) or hanging (the wrist HANG_SHARE of the forearm below the elbow at every knot)."""
    from .body import rotation_angle_deg
    from .hands import has_hands
    J, G = body.joints(), body.fk()[1]
    sup = support_labels(body, boxes, support_joints(kind)) if labels is None else labels
    out = {}
    for i, (side, (_, _, _, w)) in enumerate(ARM.items()):
        still = float((J[:, w] - J[:1, w]).norm(dim=-1).max()) <= REST_STILL_M and \
            float(rotation_angle_deg(G[:1, w].transpose(-1, -2) @ G[:, w]).max()) <= REST_STILL_DEG
        if has_hands(body):
            h = body.hand6d[:, 15 * i:15 * (i + 1)]
            still = still and float((h - h[:1]).abs().max()) <= 1e-9
        out[side] = bool(still and ((sup[side] != -1).all() or _hanging(J, side)))
    return out


# ---------------------------------------------------------------- composition
def _compose_trunk(R, L, kind=None):
    """R (T, 22, 3, 3) in place: the seated lean (spine1, parent frame), breathing (spine3 extension; the neck and
    both collars take it back exactly, so that the gaze and the arms keep their global orientations and only the
    shoulders' translation reaches a free arm; then the collars' elevation), the head drift (neck and head, child
    frame). Supine the spine does not breathe (the back and the head lie on the couch: an extension would press the
    upper back into it, a flexion would lift the chest and the head off it; a skeleton cannot show the chest wall's
    rise), only the collars do."""
    R[:, 3] = _rot("z", L["lean_ml_deg"]) @ _rot("x", L["lean_ap_deg"]) @ R[:, 3]
    if kind != "supine":
        R9 = R[:, 9].clone()
        R[:, 9] = _rot("x", -BREATH_SPINE_DEG * L["breath"]) @ R9
        back = R[:, 9].transpose(-1, -2) @ R9                       # the extension taken back by the children
        for j in (12, 13, 14):
            R[:, j] = back @ R[:, j]                                # their global rotations unchanged
    c = BREATH_COLLAR_DEG * L["breath"]
    R[:, 13] = _rot("z", c) @ R[:, 13]
    R[:, 14] = _rot("z", -c) @ R[:, 14]
    for j in (12, 15):
        R[:, j] = R[:, j] @ _rot("y", 0.5 * L["head_yaw_deg"]) @ _rot("x", 0.5 * L["head_pitch_deg"])


def dilate_labels(lab, m):
    """(T,) the support labels with each supported run widened by m knots on both sides (a knot takes the label of
    the nearest supported knot within m), so that the anchoring's ramp is complete when the hand meets its support
    and starts only after it leaves."""
    lab = np.asarray(lab)
    on = np.flatnonzero(lab != -1)
    if not len(on) or m <= 0:
        return lab.copy()
    k = np.arange(len(lab))
    i = np.clip(np.searchsorted(on, k), 1, len(on)) - 1
    j = np.minimum(i + 1, len(on) - 1)
    near = np.where(np.abs(k - on[i]) <= np.abs(on[j] - k), on[i], on[j])
    return np.where((lab == -1) & (np.abs(near - k) <= m), lab[near], lab)


def _moving_max(x, m):
    """(T,) the moving maximum of x over 2m + 1 knots (edges held)."""
    x = np.asarray(x, dtype=np.float64)
    if m <= 0 or len(x) == 0:
        return x.copy()
    return np.lib.stride_tricks.sliding_window_view(np.pad(x, (m, m), mode="edge"), 2 * m + 1).max(1)


def support_weights(labels, side, rate):
    """({support: (T,) weight 0 to 1}, anchored (T,) bool) of a hand: each support's indicator on the supported runs
    widened by the ramp (dilate_labels, by 2n knots, n = ANCHOR_RAMP_S / 2: a knot takes the nearest run's label, so
    that two runs split the gap between them) and smoothed by two moving means of n, so that the ramp lies outside
    the contact and a switch between two supports cross-fades; the weights sum to 1 on the widened runs."""
    n = max(1, int(round(ANCHOR_RAMP_S * rate / 2)))
    lab = dilate_labels(labels[side], 2 * n)
    out = {int(s): np.clip(_moving_mean(_moving_mean((lab == s).astype(np.float64), n), n), 0.0, 1.0)
           for s in np.unique(lab[lab != -1])}
    return out, (lab != -1) | (sum(out.values(), np.zeros(len(lab))) > 0.0)


def _anchor_goals(J_in, G_in, J_free, G_free, labels, side, rate):
    """(goal wrist (T, 3), goal hand rotation (T, 3, 3), anchored (T,) bool, solved knots (T,) bool): a supported hand
    keeps its place relative to its support (the scene: the world; a body joint: that joint's frame), with the weights
    of support_weights; each support's target is defined at every knot and never smoothed."""
    w = ARM[side][3]
    weights, on = support_weights(labels, side, rate)
    T = J_in.shape[0]
    off = torch.zeros(T, 3, dtype=F64)
    rot = torch.zeros(T, 3, dtype=F64)
    for s, a in weights.items():
        if s == -2:
            Wt, Ht = J_in[:, w], G_in[:, w]
        else:
            M = G_free[:, s] @ G_in[:, s].transpose(-1, -2)
            Wt = J_free[:, s] + torch.einsum("nij,nj->ni", M, J_in[:, w] - J_in[:, s])
            Ht = M @ G_in[:, w]
        a = torch.from_numpy(a)[:, None]
        off = off + a * (Wt - J_free[:, w])
        rot = rot + a * _aa(Ht @ G_free[:, w].transpose(-1, -2))
    goal = J_free[:, w] + off
    H = _exp(rot) @ G_free[:, w]
    moved = (off.norm(dim=-1) > 1e-12) | (rot.norm(dim=-1) > 1e-12)
    return goal, H, torch.from_numpy(on), moved


def touch_weight(labels, side, rate):
    """(T,) 0 to 1: the share of the anchoring on the head or neck of a hand (support_weights). The finger drift of
    a hand touching or approaching the face fades to zero there, so that the clip keeps its own contact; the hand
    itself follows the head (_anchor_goals). The trunk, the breathing and the head drift never yield to it: they are
    one layer for the whole series, whatever a clip's hands do."""
    weights, _ = support_weights(labels, side, rate)
    return np.clip(sum((weights[s] for s in SUPPORTS["head"] if s in weights), np.zeros(len(labels[side]))),
                   0.0, 1.0)


def support_joints(kind):
    """The body joints a hand may rest on in a posture kind (SUPPORTS): the head and neck always (a hand touching
    the face), the thighs and knees seated, supine or arising (hands on the thighs or in the lap); standing, a hand
    hanging beside the hip rides with the trunk."""
    return SUPPORTS["head"] + (SUPPORTS["lap"] if kind in ("seated", "supine", "transfer") else ())


def apply(body, layer, scene=None, table="healthy", resting=None, labels=None, checks=True, kind=None, fade=None,
          fade_keys=FADED, rest_gate=False, arm_damp=0.0, span=None, span_keys=FADED):
    """(Body, record): the layer (make_layer) composed onto a built body (module docstring); scene: the clip's scene
    kind (exam.scene_mesh) for the hands' supports and the scene check; table: the joint-limit table the clip is
    judged on ('healthy', or a table of limits_ext.py, 'dystonia' for the SH2 extension: the re-check reads it; the
    guard of the solve always holds the layer inside the healthy table or the clip's own excursion, which lies inside
    any);
    resting: {side: bool} of the series (the hands that take the finger drift; a body without hand joints is given
    them at rest when a side is True); labels: support_labels of the body (default read here, on the joints
    support_joints(kind) allows; kind: the series' posture kind); fade: (T,) 0 to 1, a local fade of the breathing,
    the head, trunk and finger drifts and the sway (fade_keys, of FADED; LayeredBuild: where the full layer fails a
    contact check, local_fade); rest_gate: a hand anchored only where it rests on its support (rest_weight, the
    series root's form of a member's rest gate: solve); arm_damp: the damped elbow flexion step of the anchoring
    (two_bone_fix damp, m per rad; 0 the undamped Newton step; solve's fallback ARM_FIX_DAMP_M); span: (T,) 0 to 1,
    the weight of the signals span_keys over the clip (physio-v7: a walking clip's still spans, still_weight, and a
    freezing clip's holds, hold_weight, every signal; a transfer's standing phase, stand_weight, the sway only),
    where the sway is zero the body's pelvis and legs kept, and where the span weighs every signal and is zero every
    row the body's own, bit for bit. record: the layer's digest, the components applied, the guard, the anchors and
    (checks) the validity of the output against the body (validity)."""
    from .hands import hands_of, has_hands
    lim = joint_limits()
    if abs(float(body.rate) - LAYER_RATE) > 1e-9:
        raise ValueError(f"the layer runs at {LAYER_RATE} Hz, the body at {body.rate}")
    T = body.T
    L = layer_prefix(layer, T)
    if fade is not None:
        L = dict(L)
        f = np.asarray(fade, dtype=np.float64)
        for key in fade_keys:
            v = np.asarray(L[key])
            L[key] = v * f.reshape((-1,) + (1,) * (v.ndim - 1))
    if span is not None:
        L = dict(L)
        sp = np.asarray(span, dtype=np.float64)
        if sp.shape != (T,) or sp.min() < 0.0 or sp.max() > 1.0:
            raise ValueError(f"span: ({T},) weights in [0, 1] expected")
        for key in span_keys:
            v = np.asarray(L[key])
            L[key] = v * sp.reshape((-1,) + (1,) * (v.ndim - 1))
    rest = body.rest()
    R_in, tr_in = body.local_R(), body.transl
    J_in, G_in = fk(R_in, tr_in, rest)
    boxes = []
    if scene is not None:
        from .viewer import scene_floor
        boxes = scene_boxes(scene, body, scene_floor(body, 5.0)[0])
    if labels is None:
        labels = support_labels(body, boxes, support_joints(kind or "standing"))
    touch = np.maximum(touch_weight(labels, "left", body.rate), touch_weight(labels, "right", body.rate))
    R = R_in.clone()
    _compose_trunk(R, L, kind)
    R, g_trunk = guard_joints(R, R_in, TRUNK_JOINTS, lim, body.rate)
    tr = tr_in.clone()
    rec = {"version": VERSION, "layer_sha256": layer_digest(layer, T), "knots": T, "guard": {"trunk": g_trunk},
           "touch_share": float((touch > 0).mean()), "deviations": {}}
    if g_trunk:
        rec["deviations"]["trunk_guard"] = ("the trunk layer blended back toward the clip at the knots of "
                                            "guard.trunk (a joint at its limit): the clip's layer differs there")
    sway = bool(np.any(L["sway_ap_m"] != 0) or np.any(L["sway_ml_m"] != 0))
    if sway:
        sj = [0, 3] + list(LEG_JOINTS)                          # the sway turns the pelvis, the spine takes the roll
        head_of = None
        if kind == "walking":
            # physio-v7: a walking clip stands in any heading; its pelvis is read in its own heading's frame (the
            # yaw of its forward axis taken off), not against the table's pelvis range, which a pelvis turned 83 deg
            # leaves by its heading alone and where any sway then read as a new excursion (the freezing holds' sway
            # scaled to nothing)
            f = R_in[:, 0, :, 2]
            psi = torch.atan2(f[:, 0], f[:, 2])
            head_of = _exp(torch.stack([torch.zeros_like(psi), psi, torch.zeros_like(psi)], -1)).transpose(-1, -2)

        def sway_deg(Rx):
            a = Rx[:, sj]
            if head_of is not None:
                a = torch.cat([(head_of @ a[:, 0])[:, None], a[:, 1:]], 1)
            return np.degrees(_aa(a).numpy())
        ci = sway_deg(R_in)
        lo = np.minimum(lim.lo[sj][None], ci) - LIMIT_TOL_DEG
        hi = np.maximum(lim.hi[sj][None], ci) + LIMIT_TOL_DEG
        n_s = max(1, int(round(GUARD_SMOOTH_S * body.rate)))
        scale = np.ones(T)
        for it in range(6):
            R2, tr2, ank = stand_sway(R, tr, rest, scale * L["sway_ap_m"], scale * L["sway_ml_m"], R_in, tr_in)
            c = sway_deg(R2)
            bad = ((c < lo) | (c > hi)).any(-1).any(-1)
            if not bad.any() or it == 5:
                break
            if it < 4:
                scale[bad] *= 0.5                               # halved four times, then held at the clip
            else:
                scale[bad] = 0.0
            scale = _smooth_min(scale, n_s)
        if span is not None:                                    # no sway at a knot: the pelvis and the legs kept
            z = torch.from_numpy((scale * L["sway_ap_m"] == 0.0) & (scale * L["sway_ml_m"] == 0.0))
            R2 = torch.where(z[:, None, None, None], R, R2)
            tr2 = torch.where(z[:, None], tr, tr2)
            no_sway = z
        R, tr = R2, tr2
        rec["guard"]["sway_scaled_knots"] = int((scale < 1.0 - 1e-12).sum())
        rec["guard"]["sway_bad_knots"] = int(bad.sum())
        rec["ankle_error_m"] = float((ank - J_in[:, [7, 8]]).norm(dim=-1).max())
        if rec["guard"]["sway_scaled_knots"]:
            rec["deviations"]["sway_scaled"] = (
                f"the sway scaled down on {rec['guard']['sway_scaled_knots']} knots (a pelvis, spine or leg joint "
                "at its limit): the clip's sway, hence its whole body, differs there from its level 0's")
    # the hands: anchored to their supports, or riding
    J_free, G_free = fk(R, tr, rest)
    R_pre_arm = R.clone()
    anchors = {}
    for side, chain in ARM.items():
        goal, H, on, moved = _anchor_goals(J_in, G_in, J_free, G_free, labels, side, body.rate)
        w = ARM[side][3]
        # the anchoring's weight at each knot (the supports' weights summed): the arm is solved toward the whole
        # target and its rows blended from the free arm by that weight (open-body port, wave 7). Solved toward the
        # blended goal, a straight arm's elbow flexion goes as the square root of the distance it must take back, so
        # that it fell to the free arm's with an infinite slope where the ramp ends (the raise's straight elbows on
        # Anny: 7.4 m/s2 in one knot at the release, tremor_E2_raise_g0); blended in rotation it falls with the ramp
        wsum = torch.from_numpy(sum(support_weights(labels, side, body.rate)[0].values(),
                                    np.zeros(J_in.shape[0])))
        if rest_gate:
            g = torch.from_numpy(rest_weight(wrist_speed(J_in, G_in, labels, side, body.rate), body.rate))
            goal = J_free[:, w] + g[:, None] * (goal - J_free[:, w])
            H = _exp(_aa(H @ G_free[:, w].transpose(-1, -2)) * g[:, None]) @ G_free[:, w]
            moved = moved & (g > 0.0)
            on = on & (g >= 1.0 - 1e-12)
            wsum = wsum * g
        k = torch.nonzero(moved).flatten()
        anchors[side] = {"anchored_share": float(on.float().mean()), "solved_knots": int(len(k)),
                         "supports": sorted(int(x) for x in np.unique(labels[side]) if x != -1)}
        if len(k):
            ws = wsum.clamp(min=1e-12)[:, None]
            whole = (wsum > 0.0)[:, None]
            goal_w = torch.where(whole, J_free[:, w] + (goal - J_free[:, w]) / ws, goal)
            H_w = torch.where(whole[..., None], _exp(_aa(H @ G_free[:, w].transpose(-1, -2)) / ws) @ G_free[:, w], H)
            R_s, reached = two_bone_fix(R, tr, rest, chain, goal_w, _elbow_axis(side), end_global=H_w, knots=k,
                                        damp=arm_damp)
            wk = wsum[k].clamp(max=1.0).numpy()
            for j in chain[1:]:
                R[k, j] = torch.where(torch.from_numpy(wk >= 1.0 - 1e-12)[:, None, None], R_s[k, j],
                                      _slerp(R[k, j], R_s[k, j], wk))
            anchors[side]["goal"] = goal
            # the check reads the knots of the whole anchoring (the ramp lies outside the contact, support_weights:
            # there the arm is a blend, not a hand held on its support)
            anchors[side]["on"] = on & (wsum >= 1.0 - 1e-12)
    R, g_arm = guard_joints(R, R_pre_arm, ARM_JOINTS, lim, body.rate)
    rec["guard"]["arms"] = g_arm
    J_out, G_out = fk(R, tr, rest)
    anchor_bad = []
    for side, a in anchors.items():
        if "goal" in a:
            w = ARM[side][3]
            err = (J_out[:, w] - a.pop("goal")).norm(dim=-1)
            on = a.pop("on")
            a["anchor_error_m"] = float(err[on].max()) if bool(on.any()) else 0.0
            anchor_bad.append(np.flatnonzero((on & (err > ANCHOR_TOL_M)).numpy()))
        else:
            a["anchor_error_m"] = 0.0
    rec["anchors"] = anchors
    changed = [j for j in range(N_BODY) if not torch.equal(R[:, j], R_in[:, j])]
    out = body.with_rotations(R, changed) if changed else body.copy()
    out = replace(out, transl=tr.clone(), contact=body.contact.clone())
    if span is not None and sway:                           # the pelvis's and the legs' own rows where no sway is
        m = no_sway[:, None] & torch.isin(torch.arange(N_BODY), torch.tensor([0] + list(LEG_JOINTS)))[None]
        r6 = out.rot6d.clone()
        r6[m] = body.rot6d[m]
        out = replace(out, rot6d=r6, transl=torch.where(no_sway[:, None], body.transl, out.transl))
    if span is not None and set(FADED) <= set(span_keys):    # outside the spans the body's own rows, bit for bit
        z = torch.from_numpy(np.asarray(span) <= 0.0)
        out = replace(out, rot6d=torch.where(z[:, None, None], body.rot6d, out.rot6d),
                      transl=torch.where(z[:, None], body.transl, out.transl))
    # the fingers of the resting hands
    rest_h = resting or {"left": False, "right": False}
    fing = np.asarray(L["fingers_deg"])
    lifted = [side for si, side in enumerate(ARM) if rest_h.get(side) and len(labels[side])
              and scene not in FINGER_TABLE_SCENES and bool((np.asarray(labels[side]) != -1).all())
              and bool(np.any(fing[:, si] != 0))]       # the resting hands lying on a support (and not quiet)
    if lifted and np.any(fing != 0):
        from .hands import FINGERS
        fing = fing.copy()
        ff = np.ones(T) if fade is None or "fingers_deg" not in fade_keys else np.asarray(fade, dtype=np.float64)
        if span is not None and "fingers_deg" in span_keys:
            ff = ff * np.asarray(span, dtype=np.float64)
        for si, side in enumerate(ARM):
            if side not in lifted:
                fing[:, si] = 0.0                                 # a hanging hand keeps its fingers
                continue
            # the fingers only lift (extend) from the posture (finger_lift of the unfaded drift, then the fade), so
            # that a clip of the series whose hand lies closer to the support never presses them into it
            fing[:, si] = finger_lift(np.asarray(layer["fingers_deg"])[:T, si]) * ff[:, None]
            fing[:, si] *= (1.0 - touch_weight(labels, side, body.rate))[:, None]
        fing[:, :, FINGERS.index("thumb")] = 0.0                     # the thumb lies against the thigh or the body
        out = hands_of(out) if not has_hands(out) else out
        out, rec["guard"]["fingers"] = _fingers(out, fing, {side: side in lifted for side in ARM})
    rec["components"] = [c for c in ("breath", "head", "trunk", "sway") if _on(L, c)]
    if lifted and _on(L, "fingers"):
        rec["components"].append("fingers")
    rec["resting_hands"] = dict(rest_h)
    rec["fingers_lift_only"] = lifted
    if span is not None:
        rec["span"] = span_record(span, float(body.rate), span_keys)
    if g_arm:
        rec["deviations"]["arm_guard"] = ("an anchored arm blended back toward the clip at the knots of guard.arms (a "
                                          "shoulder, elbow or wrist at its limit): the hand may leave its support "
                                          "there (anchor_error_m)")
    if checks:
        v = validity(body, out, lim, boxes, sway, table)
        v["anchor_error_m"] = max(a["anchor_error_m"] for a in anchors.values())
        v["ok"] = bool(v["ok"] and v["anchor_error_m"] <= ANCHOR_TOL_M)
        v["_bad_knots"] = np.unique(np.concatenate([v["_bad_knots"]] + anchor_bad))
        v["failed"] = failed_checks(v)
        rec["validity"] = v
    return out, rec


def failed_checks(v):
    """[names] of the checks of validity() a record fails (with its anchor error; the skate against the record's
    skate_tol_m_s, SKATE_TOL_M_S by default)."""
    bad = []
    if v.get("limits_new"):
        bad.append("limits_new")
    if v.get("skate_added_m_s", 0.0) > v.get("skate_tol_m_s", SKATE_TOL_M_S):
        bad.append("skate")
    if (v.get("contact_lost_share") or 0.0) != 0.0:
        bad.append("contact_lost")
    for k, tol in (("scene_added_m", SCENE_TOL_M), ("body_added_m", BODY_TOL_M), ("legs_added_m", BODY_TOL_M),
                   ("foot_moved_m", FOOT_TOL_M), ("accel_excess_m_s2", ACCEL_TOL_M_S2),
                   ("anchor_error_m", ANCHOR_TOL_M)):
        if v.get(k, 0.0) > tol:
            bad.append(k)
    return bad


def _on(L, comp):
    keys = {"breath": ("breath",), "head": ("head_yaw_deg", "head_pitch_deg"), "trunk": ("lean_ap_deg", "lean_ml_deg"),
            "sway": ("sway_ap_m", "sway_ml_m"), "fingers": ("fingers_deg",)}[comp]
    return any(bool(np.any(np.asarray(L[k]) != 0)) for k in keys)


def _fingers(body, fing, resting):
    """(HandBody, {side: knots reduced}): the MCP and PIP (thumb MCP) flexion of each resting hand offset by the
    layer's fingers_deg (T, 2, 5) about the joint's flexion axis (parent frame), reduced where a joint would leave the
    hand limits of hands.py beyond the clip's own range."""
    from . import hands as H
    fr = H.hand_frame()
    Rh_in = body.hand_R()
    Rh = Rh_in.clone()
    comp_in = H.hand_components(Rh_in.numpy())
    stats = {}
    for si, side in enumerate(H.SIDES):
        if not resting.get(side):
            continue
        n_red = 0
        for fi, f in enumerate(H.FINGERS):
            for lv in ((0, 1) if f != "thumb" else (1,)):
                j = H.hand_joint(side, f, lv)
                row = H.hand_row(j)
                name = (H.THUMB_LEVELS if f == "thumb" else H.LEVELS)[lv]
                lo, hi = H.FLEX_LIMITS_DEG[f][name]
                fl_in = comp_in[j][0]
                lo_k, hi_k = np.minimum(lo, fl_in), np.maximum(hi, fl_in)
                ax = torch.as_tensor(fr[side]["joints"][j]["flex"], dtype=F64)
                d = fing[:, si, fi].copy()
                for _ in range(3):
                    Rj = _exp(ax[None] * torch.from_numpy(np.radians(d))[:, None]) @ Rh_in[:, row]
                    fl = H.hand_components(_with_row(Rh_in, row, Rj).numpy())[j][0]
                    over = np.maximum(fl - hi_k, 0.0) - np.maximum(lo_k - fl, 0.0)
                    if not np.any(np.abs(over) > 1e-9):
                        break
                    n_red += int((np.abs(over) > 1e-9).sum())
                    d = d - over
                Rh[:, row] = Rj
        stats[side] = n_red
    return body.with_hands(Rh), stats


def _finger_guard(Hn, Hc, D, own):
    """(Hn, {side: knots reduced}), numpy (T, 30, 3, 3): a member's hand rows Hn = D Hc (D: the level 0's hand
    deltas) with, on its own rows (own (T, 30) bool), the delta scaled down along its geodesic where it would take a
    hand joint's flexion component outside [min(lo, own), max(hi, own)] of the hand limits of hands.py (own: the
    clip's own component at that knot), three rounds as _fingers; every other row and knot of Hn kept bit for bit."""
    from . import hands as HD
    fr = HD.hand_frame()
    T = Hn.shape[0]
    Hn = Hn.copy()
    stats = {}

    def flex(Rr, axis):
        return np.degrees(_aa(torch.from_numpy(np.ascontiguousarray(Rr))).numpy()) @ axis
    for side in HD.SIDES:
        reduced = np.zeros(T, bool)
        for j, ax in fr[side]["joints"].items():
            r = HD.hand_row(j)
            d_aa = _aa(torch.from_numpy(np.ascontiguousarray(D[:, r]))).numpy()
            act = own[:, r] & (np.abs(d_aa).max(-1) > 1e-12)
            if not act.any():
                continue
            f, name = ax["finger"], (HD.THUMB_LEVELS if ax["finger"] == "thumb" else HD.LEVELS)[ax["level"]]
            lo, hi = (-HD.CMC_LIMIT_DEG, HD.CMC_LIMIT_DEG) if (f, name) == ("thumb", "cmc") \
                else HD.FLEX_LIMITS_DEG[f][name]
            fx = np.asarray(ax["flex"], dtype=np.float64)
            fl_c = flex(Hc[:, r], fx)
            lo_k, hi_k = np.minimum(lo, fl_c), np.maximum(hi, fl_c)
            lam = np.ones(T)
            Rr = Hn[:, r].copy()
            for _ in range(3):
                fl = flex(Rr, fx)
                over = np.maximum(fl - hi_k, 0.0) - np.maximum(lo_k - fl, 0.0)
                bad = act & (np.abs(over) > 1e-9)
                if not bad.any():
                    break
                reduced |= bad
                d = fl - fl_c                                   # the flexion the scaled delta adds
                ok_d = np.abs(d) > 1e-12
                share = np.clip(np.where(ok_d, (d - over) / np.where(ok_d, d, 1.0), 0.0), 0.0, 1.0)
                lam = np.where(bad, lam * share, lam)
                low = lam < 1.0
                Rr = Hn[:, r].copy()
                Rr[low] = _mul(_exp(torch.from_numpy(d_aa[low] * lam[low][:, None])).numpy(), Hc[low, r])
            Hn[:, r] = Rr
        stats[side] = int(reduced.sum())
    return Hn, stats


def _with_row(Rh, row, Rj):
    out = Rh.clone()
    out[:, row] = Rj
    return out


# ---------------------------------------------------------------- checks
def _faces():
    """_faces_of the selected body model."""
    return _faces_of(_model())


@functools.lru_cache(maxsize=2)
def _faces_of(model):
    """(F, 3): the selected body model's triangles (open-body port, wave 3: the SMPL+H mesh read here refused on the
    open body, and involuntary's palm gaps read signed_distance); keyed on the model's name (wave 7)."""
    from .viewer import faces
    return faces()


def _body_surface():
    """_body_surface_of the selected body model."""
    return _body_surface_of(_model())


@functools.lru_cache(maxsize=2)
def _body_surface_of(model):
    """(faces (nF, 3), open-edge flags (nF, 3) of the edges 0-1, 1-2, 2-0, vertex-to-face incidence (6890, deg)
    padded with -1) of the body without its arms: the faces of the neutral mesh whose three vertices belong to
    _probe_ids()['body']; an edge held by one such face only (where the arms were cut off) is open."""
    F = _faces()
    inb = np.zeros(int(F.max()) + 1, bool)
    inb[_probe_ids()["body"]] = True
    Fb = F[inb[F].all(1)]
    e = np.stack([np.sort(Fb[:, [u, w]], 1) for u, w in ((0, 1), (1, 2), (2, 0))], 1)      # (nF, 3, 2)
    key = e[..., 0].astype(np.int64) * 100000 + e[..., 1]
    _, inv, cnt = np.unique(key.ravel(), return_inverse=True, return_counts=True)
    bnd = (cnt[inv] == 1).reshape(key.shape)
    deg = np.bincount(Fb.ravel(), minlength=len(inb))
    inc = -np.ones((len(inb), int(deg.max())), np.int64)
    fill = np.zeros(len(inb), np.int64)
    for f, tri in enumerate(Fb):
        for v in tri:
            inc[v, fill[v]] = f
            fill[v] += 1
    return Fb, bnd, inc


def signed_distance(V, P, k=6, return_point=False):
    """(M,) m: the signed distance of points P (M, 3) to the surface of the body without its arms (_body_surface)
    of a mesh V (6890, 3), negative inside: the closest point on the faces around the k nearest body vertices, signed
    by the pseudo-normal of the faces sharing it (legcontact._signed_closest; a closest point on an open edge reads
    outside); with return_point also (M, 3) that closest point."""
    from scipy.spatial import cKDTree
    from .legcontact import _signed_closest
    Fb, bnd, inc = _body_surface()
    nb = _probe_ids()["body"]
    _, j = cKDTree(V[nb]).query(P, k=k)
    faces = inc[nb[j]].reshape(len(P), -1)                              # (M, k * deg), -1 padded
    ok = faces >= 0
    fsafe = np.where(ok, faces, 0)
    tri = np.where(ok[..., None, None], V[Fb[fsafe]], np.nan)
    if return_point:
        sd, _, q = _signed_closest(P, tri, bnd[fsafe] & ok[..., None], return_point=True)
        return sd, q
    sd, _ = _signed_closest(P, tri, bnd[fsafe] & ok[..., None])
    return sd


def validity(base, out, limits, boxes, legs=False, table="healthy", every=CHECK_EVERY, skate_tol=SKATE_TOL_M_S,
             toe=None):
    """The checks of a layered body against the clip it was laid on (module docstring), JSON-ready: body.validity
    (joint limits, skate, contacts; with a table other than 'healthy' the joint-limit counts read against that table
    of limits_ext at each knot, 'limits_table' naming it), 'scene_added_m' (the largest depth a vertex gains inside a
    scene box, at every `every`-th knot), 'body_added_m' (the largest depth inside the body without its arms,
    signed_distance, that a forearm or hand vertex lying outside it in the clip takes in the layered body; read where
    both lie within
    CONTACT_M of its surface), 'body_deepened_m' (the same for a vertex the clip already holds inside the body: a
    defect of the clip, reported, not refused), 'legs_added_m' (with legs: the depth the layer adds between the
    legs, legcontact) and 'foot_moved_m' (with legs: the largest horizontal distance of a foot point,
    body.foot_points, from the clip's at the knots where the clip's contact label of that foot holds, against
    FOOT_TOL_M; 'foot_height_moved_m' the largest change of its height there, reported; a foot on its toes, toe
    {side: (T,) bool} of toe_stance, read on its ball there, stance_points, and so is the skate, 'toe_knots' counting
    them), 'accel_added_m_s2' (the
    largest acceleration the layer adds to a joint) and 'accel_excess_m_s2' (the largest beyond ACCEL_REL of the
    clip's own acceleration there), 'skate_tol_m_s' (the skate's tolerance:
    SKATE_TOL_M_S, a member's SKATE_TOL_MEMBER_M_S), 'ok'; '_bad_knots' (not JSON: the knots where a contact check,
    scene, body or legs, fails, for the local fade of LayeredBuild)."""
    from scipy.spatial import cKDTree
    out_v = body_validity(base, out, limits)
    out_v["limits_table"] = "healthy"
    if table != "healthy":
        from . import limits_ext as LX
        lv = LX.violations(table, out.rot6d, base.rot6d)
        out_v.update(limits_outside=lv["outside"], limits_new=lv["new"], limits_worst_deg=lv["worst_deg"],
                     limits_table=table)
    T = base.T
    ks = np.unique(np.r_[np.arange(0, T, every), T - 1])
    from .viewer import knot_vertices
    Vb, Vo = knot_vertices(base, ks), knot_vertices(out, ks)
    bad = []
    if boxes:
        db, _ = box_depth(Vb, boxes)
        do, _ = box_depth(Vo, boxes)
        sa = np.maximum(0.0, do - np.maximum(db, 0.0)).max(-1)
        scene_added = float(sa.max())
        bad.append(ks[sa > SCENE_TOL_M])
    else:
        scene_added = 0.0
    ids = _probe_ids()
    body_added = body_deepened = 0.0
    nb = ids["body"]
    for i in range(len(ks)):
        to = cKDTree(Vo[i, nb])
        for side in ARM:
            a = ids["arm"][side]
            near = to.query(Vo[i, a], k=1)[0] < CONTACT_M
            if not near.any():
                continue
            sel = a[near]
            so, sb = signed_distance(Vo[i], Vo[i, sel]), signed_distance(Vb[i], Vb[i, sel])
            both = (np.abs(so) < CONTACT_M) & (np.abs(sb) < CONTACT_M)
            gain = np.where(both, np.maximum(0.0, -so) - np.maximum(0.0, -sb), 0.0)
            new = sb >= 0.0                                   # outside the body in the clip
            ba = float(np.max(np.where(new, gain, 0.0)))
            body_added = max(body_added, ba)
            body_deepened = max(body_deepened, float(np.max(np.where(new, 0.0, gain))))
            if ba > BODY_TOL_M:
                bad.append(ks[i:i + 1])
    legs_added = 0.0
    if legs:
        from .legcontact import gap_from_pose
        gb = gap_from_pose(base.local_R()[ks].numpy(), base.transl[ks].numpy(), base.betas.numpy(), int(base.gender))
        go = gap_from_pose(out.local_R()[ks].numpy(), out.transl[ks].numpy(), out.betas.numpy(), int(out.gender))
        hb, ho = np.asarray(gb["gap_m"]), np.asarray(go["gap_m"])
        la = np.maximum(0.0, np.maximum(-ho, 0.0) - np.maximum(-hb, 0.0))
        legs_added = float(np.max(la))
        bad.append(ks[la.reshape(len(ks), -1).max(-1) > BODY_TOL_M])
    foot_moved = foot_height = 0.0
    n_toe = 0
    if legs:
        Jb_t, Jo_t = base.joints(), out.joints()
        dp = (stance_points(Jo_t, toe) - stance_points(Jb_t, toe)).numpy()
        n_toe = 0 if toe is None else int(sum(np.sum(v) for v in toe.values()))
        if n_toe:
            # the skate on the same points (body.validity reads body.foot_points): the ball of a foot on its toes
            def sk(J, c):
                v = torch.gradient(stance_points(J, toe), spacing=1.0 / float(base.rate), dim=0)[0]
                sp = torch.sqrt(v[..., 0] ** 2 + v[..., 2] ** 2)[torch.as_tensor(c, dtype=torch.bool)[:, :2]]
                return float(torch.quantile(sp, 0.95)) if sp.numel() else 0.0
            so, sb = sk(Jo_t, out.contact), sk(Jb_t, base.contact)
            out_v.update(skate_p95_m_s=so, skate_base_p95_m_s=sb, skate_added_m_s=max(0.0, so - sb))
        dh = np.hypot(dp[..., 0], dp[..., 2])                          # (T, 2), horizontal
        st = base.contact[:, :2].numpy().astype(bool)
        foot_moved = float(dh[st].max()) if st.any() else 0.0
        foot_height = float(np.abs(dp[..., 1])[st].max()) if st.any() else 0.0
    Jb, Jo = base.joints().numpy(), out.joints().numpy()
    if T > 2:
        r2 = float(base.rate) ** 2
        acc = np.linalg.norm(np.diff(Jo - Jb, 2, axis=0), axis=-1) * r2
        own = np.linalg.norm(np.diff(Jb, 2, axis=0), axis=-1) * r2
    else:
        acc = own = np.zeros((1, N_BODY))
    rec = dict(out_v, scene_added_m=scene_added, body_added_m=body_added, body_deepened_m=body_deepened,
               legs_added_m=legs_added, foot_moved_m=foot_moved, foot_height_moved_m=foot_height, toe_knots=n_toe,
               accel_added_m_s2=float(acc.max()), accel_excess_m_s2=float((acc - ACCEL_REL * own).max()),
               checked_knots=int(len(ks)), skate_tol_m_s=float(skate_tol))
    rec["ok"] = bool(out_v["limits_new"] == 0 and out_v["skate_added_m_s"] <= skate_tol
                     and (out_v["contact_lost_share"] or 0.0) == 0.0 and scene_added <= SCENE_TOL_M
                     and body_added <= BODY_TOL_M and legs_added <= BODY_TOL_M and foot_moved <= FOOT_TOL_M
                     and rec["accel_excess_m_s2"] <= ACCEL_TOL_M_S2)
    rec["_bad_knots"] = (np.unique(np.concatenate([np.asarray(x, dtype=np.int64) for x in bad])) if bad
                         else np.zeros(0, np.int64))
    return rec


# ---------------------------------------------------------------- transfer (physio-v3)
def _mul(A, B):
    """A @ B of (..., 3, 3) float64 arrays as three elementwise products summed in one fixed order, so that equal
    inputs give equal outputs bit for bit wherever they sit in a batch (a clip and its level 0 compute a shared row
    alike)."""
    return (A[..., :, 0:1] * B[..., 0:1, :] + A[..., :, 1:2] * B[..., 1:2, :]) + A[..., :, 2:3] * B[..., 2:3, :]


def _tp(A):
    return np.swapaxes(A, -1, -2)


def _slerp(Ra, Rb, w):
    """(T, 3, 3) torch: the geodesic blend Ra exp(w log(Ra^T Rb)), w (T,) 0 to 1."""
    w = torch.as_tensor(np.asarray(w, dtype=np.float64))
    return Ra @ _exp(_aa(Ra.transpose(-1, -2) @ Rb) * w[:, None])


def rows_equal(a, b, T, tol=EQUAL_ROW_TOL):
    """(T, n) bool numpy: the rows (T, n, k) of a and b within tol of each other (every value) at the first T knots."""
    return ((a[:T] - b[:T]).abs().amax(-1) <= tol).numpy()


def deltas(ref_raw, ref_out):
    """{'R' (T, 22, 3, 3), 'tr' (T, 3), 'H' (T, 30, 3, 3) or None} numpy: the layer of a reference as per-knot
    deltas: D_j(t) = R_out_j(t) R_in_j(t)^T of every joint's local rotation (the parent frame; the world for the
    pelvis), the pelvis translation's difference, and the hand joints' deltas when the layered reference carries
    hands (a reference without hand joints in its raw body reads them at rest)."""
    from .hands import _eye6, has_hands
    from ..body.rotations import rot6d_to_matrix
    Ri, Ro = ref_raw.local_R().numpy(), ref_out.local_R().numpy()
    out = {"R": _mul(Ro, _tp(Ri)), "tr": (ref_out.transl - ref_raw.transl).numpy(), "H": None}
    if has_hands(ref_out):
        hi = ref_raw.hand6d if has_hands(ref_raw) else _eye6(ref_raw.T)
        out["H"] = _mul(ref_out.hand_R().numpy(), _tp(rot6d_to_matrix(hi).numpy()))
    return out


def _extend(dl, T, own_d, rate):
    """The deltas dl (deltas) of a reference of Tr knots extended to T knots with own_d (the deltas of the clip's own
    solve): the reference's last delta carried on at its last rate (a constant angular velocity, the translation's
    last velocity), blended into the own one over BEYOND_RAMP_S (a raised cosine), so that neither the delta nor its
    rate jumps at the reference's end; the reference's knots unchanged. An own solve without hand joints (no finger
    layer of its own) blends the hand deltas into the identity."""
    Tr = dl["R"].shape[0]
    k = np.arange(T - Tr)
    nb = max(1, int(round(BEYOND_RAMP_S * rate)))
    w = 0.5 - 0.5 * np.cos(math.pi * np.clip((k + 1) / nb, 0.0, 1.0))
    steps = torch.from_numpy((k + 1).astype(np.float64))[:, None]
    out = {}
    for key in ("R", "H"):
        if dl[key] is None:
            out[key] = None
            continue
        last = torch.from_numpy(np.ascontiguousarray(dl[key][-1]))
        prev = torch.from_numpy(np.ascontiguousarray(dl[key][-2] if Tr > 1 else dl[key][-1]))
        n = last.shape[0]
        own_k = own_d[key] if own_d[key] is not None else np.broadcast_to(np.eye(3), (T,) + dl[key].shape[1:])
        own = torch.from_numpy(np.ascontiguousarray(own_k[Tr:T]))
        cols = []
        for j in range(n):
            om = _aa(prev[j].transpose(-1, -2) @ last[j])               # the last step, in the delta's own frame
            lin = last[j] @ _exp(om[None] * steps)
            cols.append(_slerp(lin, own[:, j], w))
        out[key] = np.concatenate([dl[key], torch.stack(cols, 1).numpy()], 0)
    v = dl["tr"][-1] - (dl["tr"][-2] if Tr > 1 else dl["tr"][-1])
    lin = dl["tr"][-1] + (k + 1)[:, None] * v
    out["tr"] = np.concatenate([dl["tr"], lin + w[:, None] * (own_d["tr"][Tr:T] - lin)], 0)
    return out


def vertical_goal(P0, A, L):
    """(n, 3): the point of the vertical through A (n, 3) nearest A at the distance L (n,) from P0 (n, 3): a planted
    foot whose leg cannot reach A keeps its horizontal place and takes the residual in height (legs_own_fix); where
    no point of that vertical lies at L, the point of the sphere about P0 nearest A."""
    d = A - P0
    b = d[:, 1]
    disc = b * b - ((d * d).sum(-1) - L * L)
    r = disc.clamp_min(0.0).sqrt()
    up_s = torch.where((-b + r).abs() <= (-b - r).abs(), -b + r, -b - r)
    G = A.clone()
    G[:, 1] = G[:, 1] + up_s
    return torch.where((disc >= 0.0)[:, None], G, P0 + L[:, None] * _unit(d))


def ball_goal(P0, L, B, Lf, A):
    """(n, 3): the point nearest A (n, 3) at the distance L (n,) from P0 (n, 3) and Lf (n,) from the ball B (n, 3): a
    foot on its toes whose leg cannot reach A pivots about its planted ball (legs_own_fix, physio-v5); where the two
    spheres do not meet, vertical_goal."""
    d = B - P0
    dn = d.norm(dim=-1).clamp_min(1e-12)
    u = d / dn[:, None]
    a = (L * L - Lf * Lf + dn * dn) / (2.0 * dn)
    r2 = L * L - a * a
    C = P0 + a[:, None] * u
    w = A - C
    w = w - (w * u).sum(-1, keepdim=True) * u
    G = C + r2.clamp_min(0.0).sqrt()[:, None] * _unit(w)
    return torch.where((r2 > 0.0)[:, None], G, vertical_goal(P0, A, L))


def toe_stance(J_in, J_ref):
    """{side: (T,) bool}: the knots where a clip's foot stands on its toes, its ankle risen over its ball (the foot
    joint, 10 or 11) by more than contact.TOL_M beyond its level 0's at the same knot (the level 0's last knot beyond
    its length): a knee or hip posture planted on a standing leg is held on the toes (dystonia.py, contact.py)."""
    from .contact import TOL_M
    T = J_in.shape[0]
    k = np.minimum(np.arange(T), J_ref.shape[0] - 1)
    out = {}
    for side, (_, _, ank) in LEG.items():
        rise = (J_in[:, ank, 1] - J_in[:, ank + 3, 1]).numpy() - (J_ref[k, ank, 1] - J_ref[k, ank + 3, 1]).numpy()
        out[side] = rise > TOL_M
    return out


def stance_points(J, toe=None):
    """(T, 2, 3): the point of each foot that stands on the floor: body.foot_points (the mean of the ankle and foot
    joints), the ball (the foot joint) at the knots of toe[side] (toe_stance)."""
    from .body import foot_points
    P = foot_points(J).clone()
    for i, (side, (_, _, ank)) in enumerate(LEG.items()):
        if toe is not None and np.any(toe[side]):
            k = torch.from_numpy(np.flatnonzero(toe[side]))
            P[k, i] = J[k, ank + 3]
    return P


def _sway_scaled(R, tr, Rc, trc, dl, scale, own_rows, eq_tr):
    """(R, tr) of a member's transfer with the level 0's sway scaled by scale (T,) on the clip's own rows: the
    deltas of the pelvis (0, the sway's turn), of spine1 (3, the roll it takes back) and of the legs (the level 0's
    legs re-solved under it: at scale 0 the clip's own legs) along their geodesics, the translation's difference;
    every other row and knot as given."""
    R, tr = R.clone(), tr.clone()
    s = torch.from_numpy(np.asarray(scale, dtype=np.float64))
    low = np.asarray(scale) < 1.0 - 1e-12
    for j in (0, 3) + LEG_JOINTS:
        k = torch.from_numpy(np.flatnonzero(own_rows[:, j] & low))
        if len(k):
            D = torch.from_numpy(np.ascontiguousarray(dl["R"][k.numpy(), j]))
            R[k, j] = _exp(_aa(D) * s[k][:, None]) @ Rc[k, j]
    k = torch.from_numpy(np.flatnonzero(~eq_tr & low))
    if len(k):
        tr[k] = trc[k] + s[k][:, None] * torch.from_numpy(np.ascontiguousarray(dl["tr"][k.numpy()]))
    return R, tr


def own_leg_lengths(R, tr, rest, J_in, mask, rate):
    """(tr, shift (T,) m): a member's pelvis translation raised or lowered, at the knots of mask (its pelvis row, its
    translation and every leg row its own), by the vertical shift that keeps its hip-to-ankle distances at the clip's
    (J_in), the mean of the two legs' exact shifts (the hip on the vertical through it at the clip's distance from
    the clip's ankle; where none lies at it, the nearest height), as stand_sway raises or lowers a root's pelvis
    (physio-v10, plan phase E): the level 0's sway delta laid on the clip's own posture changes its legs' lengths, and
    a leg the clip holds straight then bends by the square root of the shortening, its knee kinking where the clip's
    straightens (the functional axial jerk standing, sev2 dur2: 18.70 m/s2 added at the knee at 7.89 s, the left leg
    0.24 mm shorter than the clip's, scripts/openbody_phaseE_checks.py section legs). The legs are then re-solved as
    before (legs_own_fix) for what the mean leaves (the two legs' difference). The shift ramps in from each edge of
    mask beside a held row through a smoothstep over the time that keeps the ramp's own acceleration within
    LEG_LENGTHS_ACCEL_M_S2 for the shift at that edge (6 A / T^2; at least one knot): where the clip stands as its
    level 0 the shift starts near 0 and the ramp is short (over a fixed 0.2 s the jerk, straightening its knee inside
    the ramp, still added 15.58 m/s2 at 26.55 s), where its legs differ from the level 0's it is longer (the standing
    chorea g2b: over a fixed 0.05 s, 1.42 m/s2 at the knee at 14.75 s; section legs). LEG_LENGTHS_FIXED_RAMP_S, a
    number: that ramp at every edge instead (the tries of the checks script). A shift within LEG_FIX_M is none."""
    m = np.asarray(mask, bool)
    T = len(m)
    shift = np.zeros(T)
    k = np.flatnonzero(m)
    if not len(k):
        return tr, shift
    kk = torch.from_numpy(k)
    J = fk(R[kk], tr[kk], rest)[0]
    s = []
    for hip, _, ank in LEG.values():
        d = J[:, hip] - J_in[kk, ank]
        d0 = (J_in[kk, hip] - J_in[kk, ank]).norm(dim=-1)
        b = d[:, 1]
        disc = b * b - ((d * d).sum(-1) - d0 * d0)
        r = disc.clamp_min(0.0).sqrt()
        s.append(torch.where(disc >= 0.0, torch.where((-b + r).abs() <= (-b - r).abs(), -b + r, -b - r), -b))
    sk = (0.5 * (s[0] + s[1])).numpy()
    full = np.zeros(T)
    full[k] = np.where(np.abs(sk) > LEG_FIX_M, sk, 0.0)          # legs already at the clip's lengths: none (the
    #                                                                round-off of the lengths left untouched)
    if not np.any(full != 0.0):
        return tr, shift
    w = np.zeros(T)
    for a, b_ in _runs(m):                                         # each run of mask [a, b_)
        u = np.ones(b_ - a)
        for edge, dist in ((a, np.arange(1, b_ - a + 1)), (b_ - 1, np.arange(b_ - a, 0, -1))):
            if (edge == a and a == 0) or (edge == b_ - 1 and b_ == T):
                continue                                           # the clip's own end: no held row beyond
            if LEG_LENGTHS_FIXED_RAMP_S is not None:
                n = max(1, int(round(LEG_LENGTHS_FIXED_RAMP_S * rate)))
            else:
                n = max(1, int(math.ceil(math.sqrt(6.0 * abs(full[edge]) / LEG_LENGTHS_ACCEL_M_S2) * rate)))
            u = np.minimum(u, np.clip(dist / n, 0.0, 1.0))
        w[a:b_] = u * u * (3.0 - 2.0 * u)
    shift = full * w
    tr = tr.clone()
    tr[kk, 1] = tr[kk, 1] + torch.from_numpy(shift[k])
    return tr, shift


def legs_own_fix(R, tr, rest, leg, J_in, G_in, own_rows, toe=None):
    """(R, knots corrected) with one leg (hip, knee, ankle) re-solved on the clip's own rows so that the ankle keeps
    the clip's position J_in and the foot its global rotation G_in where the transferred one misses it (LEG_FIX_M):
    the hip and the knee where both rows are the clip's own (two_bone_fix), the hip alone where only it is (the
    swing of the leg about the hip), the knee alone where only it is (the minimal turn of the shank about the knee),
    the ankle row for the foot wherever it is; every row held keeps its copy. With one of the hip and knee held the
    hip-to-ankle distance (knee held) or the knee's place (hip held) cannot follow the sway, and the ankle cannot reach
    the clip's position: it goes to the point of the vertical through it at the reachable distance (vertical_goal,
    physio-v5), so that the planted foot keeps its horizontal place (no slide, the foot check FOOT_TOL_M) and takes
    the residual in height (the record's foot_height_moved_m); physio-v4's nearest point of the knee's flexion arc
    slid the foot (dystonia_E7_start_distal_leg_knee_flexion_30deg). A foot on its toes (toe (T,)
    bool, toe_stance) is planted on its ball, not under its ankle: there the ankle goes to the point nearest the clip's
    at the reachable distance and at the foot's length from the clip's ball (ball_goal), and the foot turns by the
    minimal rotation that keeps its ball in place (its heel's height changes a little, the ball stays: physio-v5; the
    vertical goal lifted or sank the whole foot on its toes on that clip)."""
    hip, kn, ank = leg
    toe = np.zeros(R.shape[0], bool) if toe is None else np.asarray(toe, bool)
    G_end = G_in[:, ank].clone()

    def goal_at(kk, P0, L):                                      # the ankle's goal and the foot's rotation there
        g = vertical_goal(P0, J_in[kk, ank], L)
        t = torch.from_numpy(toe[kk.numpy()] & oa[kk.numpy()])         # the foot can turn: its ankle row its own
        if t.any():
            B, A = J_in[kk, ank + 3], J_in[kk, ank]
            gb = ball_goal(P0, L, B, (B - A).norm(dim=-1), A)
            g = torch.where(t[:, None], gb, g)
            Ge = _rmin(_unit(B - A), _unit(B - g)) @ G_in[kk, ank]
            G_end[kk] = torch.where(t[:, None, None], Ge, G_in[kk, ank])
        return g
    J, G = fk(R, tr, rest)
    err = (J[:, ank] - J_in[:, ank]).norm(dim=-1).numpy()
    rot = rotation_err(G[:, ank], G_in[:, ank])
    miss = (err > LEG_FIX_M) | (rot > LEG_FIX_M)
    oh, ok_, oa = own_rows[:, hip], own_rows[:, kn], own_rows[:, ank]
    out = R
    n = 0
    k = np.flatnonzero(oh & ok_ & miss)                          # the hip and the knee
    if len(k):
        out, _ = two_bone_fix(out, tr, rest, (0, hip, kn, ank), J_in[:, ank], _knee_axis, end_global=G_in[:, ank],
                              knots=k)
        n += len(k)
    k = np.flatnonzero(oh & ~ok_ & miss)                         # the hip alone: the leg swung onto the vertical goal
    if len(k):
        kk = torch.from_numpy(k)
        goal = J_in[:, ank].clone()
        goal[kk] = goal_at(kk, J[kk, hip], (J[kk, ank] - J[kk, hip]).norm(dim=-1))
        out, _ = two_bone_fix(out, tr, rest, (0, hip, kn, ank), goal, _knee_axis, end_global=G_end,
                              knots=k, flex=False)
        n += len(k)
    k = np.flatnonzero(~oh & ok_ & miss)                         # the knee alone: the shank onto the vertical goal
    if len(k):
        Jn, Gn = fk(out, tr, rest)
        kk = torch.from_numpy(k)
        goal = goal_at(kk, Jn[kk, kn], (Jn[kk, ank] - Jn[kk, kn]).norm(dim=-1))
        Gk = _rmin(_unit(Jn[kk, ank] - Jn[kk, kn]), _unit(goal - Jn[kk, kn])) @ Gn[kk, kn]
        out = out.clone()
        out[kk, kn] = Gn[kk, hip].transpose(-1, -2) @ Gk
        out[kk, ank] = Gk.transpose(-1, -2) @ G_end[kk]
        n += len(k)
    k = np.flatnonzero(~oh & ~ok_ & oa & miss)                   # the foot alone
    if len(k):
        _, Gn = fk(out, tr, rest)
        kk = torch.from_numpy(k)
        out = out.clone()
        out[kk, ank] = Gn[kk, kn].transpose(-1, -2) @ G_in[kk, ank]
        n += len(k)
    for j, own_j in ((hip, oh), (kn, ok_), (ank, oa)):             # the rows held keep their copies
        out[:, j] = torch.where(torch.from_numpy(own_j)[:, None, None], out[:, j], R[:, j])
    return out, int(n)


def _anchor_reading(err, touch, own_arm, full, n, rest=None, guarded=None, dwell=None, binds=True):
    """{'anchor_error_m', 'anchor_error_ramp_m', 'anchor_error_held_rows_m', 'anchor_error_moving_m',
    'anchor_error_guarded_m', 'anchor_error_dwell_m'}: a member's
    wrist distance from its support goal err (T,) read at every knot of its own touches (touch) where its shoulder and
    elbow rows are its own (own_arm) and its hand rests (rest (T,) 1, rest_weight; default everywhere) or stands still
    (dwell (T,) bool, ARM_DWELL_M_S over ARM_DWELL_S; default nowhere: a pause too short for rest_weight to reach 1,
    read as a rest where binds, its largest error shown apart in any case, anchor_error_dwell_m): where its
    correction is whole (full), as it is; inside the ramp from a held arm (own_arm, not full: ARM_FIX_SMOOTH_S), less
    the distance the nearest knot of the held arm within 2n + 1 knots hands over (the level 0's arm, copied, on a
    support the level 0 does not hold: the own rows cannot take it at once without a jump), the ramp's own reading
    reported apart; where either row is held (the level 0's row, copied), reported, not refused; where the hand moves
    on or off its support or its anchoring ramps in at rest (rest below 1 and not dwell: no own anchoring there,
    physio-v5), reported, not refused; where the own limit guard holds its shoulder or elbow (guarded (T,) bool: those
    knots are read apart, the goal unchanged, the arm at the bound of the table the clip is judged on, physio-v5),
    reported, not refused."""
    def mx(m, x=err):
        return float(x[m].max()) if m.any() else 0.0
    still = np.zeros(len(err), bool) if dwell is None else np.asarray(dwell, bool)
    at_rest = np.ones(len(err), bool) if rest is None else (np.asarray(rest) >= 1.0 - 1e-12) | (still & bool(binds))
    free = np.ones(len(err), bool) if guarded is None else ~np.asarray(guarded, bool)
    held = ~own_arm
    base = np.zeros_like(err)
    hk = np.flatnonzero(held)
    if len(hk):
        k = np.arange(len(err))
        i = np.clip(np.searchsorted(hk, k), 1, len(hk)) - 1
        j = np.minimum(i + 1, len(hk) - 1)
        near = np.where(np.abs(k - hk[i]) <= np.abs(hk[j] - k), hk[i], hk[j])
        base = np.where(np.abs(near - k) <= 2 * n + 1, err[near], 0.0)
    ramp = touch & own_arm & ~full & at_rest & free
    return {"anchor_error_m": max(mx(touch & own_arm & full & at_rest & free), mx(ramp, np.maximum(err - base, 0.0))),
            "anchor_error_ramp_m": mx(ramp), "anchor_error_held_rows_m": mx(touch & held),
            "anchor_error_moving_m": mx(touch & own_arm & ~at_rest),
            "anchor_error_guarded_m": mx(touch & own_arm & at_rest & ~free),
            "anchor_error_dwell_m": mx(touch & own_arm & still & free)}


def rotation_err(Ga, Gb):
    """(T,) numpy: the angle (rad) between two series of rotations."""
    return _aa(Ga.transpose(-1, -2) @ Gb).norm(dim=-1).numpy()


def contact_turn(base, out, own_rows, rate, every=CONTACT_PUSH_EVERY):
    """{side: (T, 3)} world axis-angle (rad) or None: the turn of each arm about its shoulder that takes a forearm or
    hand vertex of out back out of the body without its arms where the layer pressed it in (the depth body_added
    reads: outside the body in base, inside in out, both within CONTACT_M of its surface), read every `every` knots
    on the deepest such vertex (its closest surface point gives the direction, the depth plus CONTACT_PUSH_MARGIN_M
    the length), on the knots whose shoulder row is the clip's own; spread to the knots around by the largest turn
    within ARM_FIX_SMOOTH_S, then two moving means of it (no jump), and ramped inside the own shoulder rows
    (inside_ramp). None where no vertex is pressed in."""
    from scipy.spatial import cKDTree
    from .viewer import knot_vertices
    ids = _probe_ids()
    nb = ids["body"]
    T = base.T
    ks = np.unique(np.r_[np.arange(0, T, every), T - 1])
    Vb, Vo = knot_vertices(base, ks), knot_vertices(out, ks)
    J = fk(out.local_R()[ks], out.transl[ks], out.rest())[0].numpy()
    om = {side: np.zeros((len(ks), 3)) for side in ARM}
    hit = False
    for i in range(len(ks)):
        to = None
        for side, (_, sh, _, _) in ARM.items():
            if not own_rows[ks[i], sh]:
                continue
            to = cKDTree(Vo[i, nb]) if to is None else to
            a = ids["arm"][side]
            near = to.query(Vo[i, a], k=1)[0] < CONTACT_M
            if not near.any():
                continue
            sel = a[near]
            so, q = signed_distance(Vo[i], Vo[i, sel], return_point=True)
            sb = signed_distance(Vb[i], Vb[i, sel])
            depth = np.where((sb >= 0.0) & (np.abs(sb) < CONTACT_M) & (np.abs(so) < CONTACT_M), np.maximum(0.0, -so),
                             0.0)
            m = int(np.argmax(depth))
            if depth[m] <= 0.0:
                continue
            p = Vo[i, sel[m]]
            u = q[m] - p
            push = u / max(float(np.linalg.norm(u)), 1e-12) * (depth[m] + CONTACT_PUSH_MARGIN_M)
            r = p - J[i, sh]
            om[side][i] = np.cross(r, push) / float(r @ r)
            hit = True
    if not hit:
        return None
    n = max(1, int(round(ARM_FIX_SMOOTH_S * rate)))
    out_om = {}
    for side, (_, sh, _, _) in ARM.items():
        w = np.stack([np.interp(np.arange(T), ks, om[side][:, c]) for c in range(3)], 1)
        mag = np.linalg.norm(w, axis=-1)
        idx = np.lib.stride_tricks.sliding_window_view(np.pad(np.arange(T), (n, n), mode="edge"), 2 * n + 1)
        w = w[idx[np.arange(T), np.argmax(mag[idx], 1)]]          # the largest turn within n knots
        w = _moving_mean(_moving_mean(w, n), n) * inside_ramp(own_rows[:, sh], n)[:, None]
        out_om[side] = w
    return out_om


def with_arm_turn(body, om, own_rows):
    """body with each arm turned about its shoulder by om[side] (T, 3) world axis-angle (contact_turn): the shoulder
    row only (the arm below rides with it), on the clip's own shoulder rows; every other row bit for bit."""
    from ..body.rotations import matrix_to_rot6d
    R = body.local_R()
    _, G = fk(R, body.transl, body.rest())
    r6 = body.rot6d.clone()
    for side, (col, sh, _, _) in ARM.items():
        w = torch.from_numpy(np.asarray(om[side], dtype=np.float64))
        k = torch.from_numpy(np.flatnonzero((w.norm(dim=-1) > 0.0).numpy() & own_rows[:, sh]))
        if len(k):
            Rk = G[k, col].transpose(-1, -2) @ _exp(w[k]) @ G[k, sh]
            r6[k, sh] = matrix_to_rot6d(Rk)
    return replace(body, rot6d=r6)


def box_normal(p, boxes):
    """(3,): the outward normal of the face nearest the point p (3,) of the box it lies deepest in (box_depth)."""
    best, n = -np.inf, np.zeros(3)
    for o, ax, ext in boxes:
        q = (p - o) @ ax.T
        lo, hi = q, ext - q
        depth = min(lo.min(), hi.min())
        if depth > best:
            best = depth
            i = int(np.argmin(np.minimum(lo, hi)))
            n = ax[i] * (-1.0 if lo[i] < hi[i] else 1.0)
    return n


def scene_turn(base, out, own_rows, rate, boxes, every=CONTACT_PUSH_EVERY):
    """{joint: (T, 3)} world axis-angle (rad) or None: the turn that takes an arm's vertex back out of a scene box
    where the layer pressed it in beyond SCENE_AIM of SCENE_TOL_M (the depth scene_added reads: the part beyond the
    clip's own), read every `every` knots on the arm's deepest such vertex, its box face giving the direction. Where
    the shoulder row is the clip's own, the arm turns about the line from its shoulder to its wrist (the hand keeps its
    place on its support, the elbow swings); where it is held, the first of the spine rows 9, 6, 3 that is the clip's
    own turns the trunk above it, the least rotation that moves the vertex that far. Spread, smoothed and ramped
    inside the own rows as contact_turn. None where no vertex is pressed in."""
    from .hands import FIRST
    from .viewer import dominant_joint, knot_vertices
    dom = dominant_joint("neutral")
    T = base.T
    ks = np.unique(np.r_[np.arange(0, T, every), T - 1])
    Vb, Vo = knot_vertices(base, ks), knot_vertices(out, ks)
    J = fk(out.local_R()[ks], out.transl[ks], out.rest())[0].numpy()
    db, do = box_depth(Vb, boxes)[0], box_depth(Vo, boxes)[0]
    added = np.maximum(0.0, do - np.maximum(db, 0.0))
    aim = SCENE_AIM * SCENE_TOL_M
    arm = {side: np.flatnonzero(np.isin(dom, [sh, el, w] + list(range(FIRST[side], FIRST[side] + 15))))
           for side, (_, sh, el, w) in ARM.items()}
    om = {}
    for i in range(len(ks)):
        for side, (_, sh, _, w) in ARM.items():
            a = arm[side]
            m = int(np.argmax(added[i, a]))
            if added[i, a[m]] <= aim:
                continue
            p = Vo[i, a[m]]
            push = box_normal(p, boxes) * (added[i, a[m]] - aim + CONTACT_PUSH_MARGIN_M)
            spine = [j for j in (9, 6, 3) if own_rows[ks[i], j]]
            if own_rows[ks[i], sh]:
                pivot = sh
                u = J[i, w] - J[i, sh]
                u = u / max(float(np.linalg.norm(u)), 1e-12)
                lever = float(np.cross(u, p - J[i, sh]) @ push) / max(float(np.linalg.norm(push)), 1e-12)
                if abs(lever) < 1e-3:
                    continue
                turn = u * float(np.linalg.norm(push)) / lever
            elif spine:
                pivot = spine[0]
                r = p - J[i, pivot]
                turn = np.cross(r, push) / float(r @ r)
            else:
                continue
            om.setdefault(pivot, np.zeros((len(ks), 3)))[i] += turn
    if not om:
        return None
    n = max(1, int(round(ARM_FIX_SMOOTH_S * rate)))
    idx = np.lib.stride_tricks.sliding_window_view(np.pad(np.arange(T), (n, n), mode="edge"), 2 * n + 1)
    res = {}
    for j, x in om.items():
        y = np.stack([np.interp(np.arange(T), ks, x[:, c]) for c in range(3)], 1)
        y = y[idx[np.arange(T), np.argmax(np.linalg.norm(y, axis=-1)[idx], 1)]]
        res[j] = _moving_mean(_moving_mean(y, n), n) * inside_ramp(own_rows[:, j], n)[:, None]
    return res


def with_row_turn(body, om, own_rows):
    """body with the rows of each joint of om turned by om[joint] (T, 3) world axis-angle about that joint, on the
    clip's own rows of it, the joints taken in chain order; every other row bit for bit (scene_turn)."""
    from ..body.rotations import matrix_to_rot6d
    from ..body.joints import PARENTS
    o = body
    for j in sorted(om):
        w = torch.from_numpy(np.asarray(om[j], dtype=np.float64))
        k = torch.from_numpy(np.flatnonzero((w.norm(dim=-1) > 0.0).numpy() & own_rows[:, j]))
        if len(k):
            _, G = fk(o.local_R(), o.transl, o.rest())
            r6 = o.rot6d.clone()
            r6[k, j] = matrix_to_rot6d(G[k, PARENTS[j]].transpose(-1, -2) @ _exp(w[k]) @ G[k, j])
            o = replace(o, rot6d=r6)
    return o


def inside_ramp(mask, n):
    """(T,) 0 to 1: 0 off mask, ramping to 1 within each of its runs (a moving minimum over 2n + 1 knots, then two
    moving means of width n: it never leaves the mask and shows no kink); a run shorter than 2n + 1 knots stays 0."""
    m = np.asarray(mask, dtype=np.float64)
    if n <= 1 or not len(m):
        return m
    lo = np.lib.stride_tricks.sliding_window_view(np.pad(m, (n, n), mode="edge"), 2 * n + 1).min(1)
    return np.minimum(_moving_mean(_moving_mean(lo, n), n), m)


def wrist_speed(J_in, G_in, labels, side, rate):
    """(T,) m/s: the speed of a clip's wrist relative to the support its hand is anchored to at each knot (the
    supported runs widened by the anchoring's ramp, support_weights: in the frame of that body joint, the world for a
    scene box or no support), read between each knot and the one before in the same frame (0 at the first)."""
    w = ARM[side][3]
    n_a = max(1, int(round(ANCHOR_RAMP_S * rate / 2)))
    lab = dilate_labels(labels[side], 2 * n_a)
    T = J_in.shape[0]
    k = torch.arange(T)
    km = (k - 1).clamp(min=0)
    p, pp = J_in[k, w].clone(), J_in[km, w].clone()
    for s in np.unique(lab[lab >= 0]):
        sel = torch.from_numpy(np.flatnonzero(lab == s))
        sm = km[sel]
        p[sel] = torch.einsum("nji,nj->ni", G_in[sel, s], J_in[sel, w] - J_in[sel, s])
        pp[sel] = torch.einsum("nji,nj->ni", G_in[sm, s], J_in[sm, w] - J_in[sm, s])
    return ((p - pp).norm(dim=-1) * rate).numpy()


def rest_weight(speed, rate, n_edge=0, rest_s=ARM_REST_S):
    """(T,) 0 to 1: where a member's own hand rests on its support, so that its own anchoring applies (transfer): 1
    where the wrist (wrist_speed) has moved slower than ARM_REST_M_S at every knot within n_edge + rest_s * rate
    knots, 0 within n_edge knots of a faster one, a ramp between (a moving minimum, then two moving means: no kink; the
    edge keeps a correction smoothed over n_edge knots off the moving knots)."""
    still = (np.asarray(speed) <= ARM_REST_M_S).astype(np.float64)
    if n_edge > 0:
        still = np.lib.stride_tricks.sliding_window_view(np.pad(still, (n_edge, n_edge), mode="edge"),
                                                         2 * n_edge + 1).min(1)
    return _smooth_min(still, max(1, int(round(rest_s * rate))))


def transfer(body, ref_raw, ref_out, layer, scene=None, table="healthy", kind=None, resting=None, own=None,
             reference=None, checks=True, _trace=None):
    """(Body, record): _transfer with the own arm correction smoothed and ramped in over ARM_FIX_SMOOTH_S inside the
    clip's own rows; where that fails the acceleration check (a large correction ramped in fast), the ramp over
    GUARD_SMOOTH_S, and where it fails the anchor check (a jerk lifting or landing the hand faster than the
    correction follows), both over ARM_FIX_FAST_S (physio-v4's variants); then the own anchoring only where the
    clip's hand rests (rest_weight over ARM_REST_S, physio-v5: a hand flung off or onto its support at metres per
    second), alone, then with the failing check's ramp: the first variant passing every check, the anchor check read
    where the hand stands still too (_anchor_reading, ARM_DWELL_M_S), is kept, else the
    default (a refused clip shows the default's failures), named in the record's guard (arm_ramp_s, arm_smooth_s,
    arm_rest_s, 0 for no rest gate). The anchor check of the gates a refused default falls back on leaves the pauses
    out (reported, anchor_error_dwell_m): such a clip moves its hand throughout (a jerk on the support) and the gate
    is its last resort. A clip passing with the default whose own anchoring turned only where its hand
    rests (guard arms_fixed_moving_knots 0) never runs another, so its body is the default's bit for bit; one whose
    anchoring also turned where the hand moves on its support takes the rest gate when that passes too (the smaller
    own correction, physio-v5), and keeps the default where the gate would leave a pausing hand off its support (the
    fingertip below the target of reach_E3_decomposition_icars3). Where every variant fails, the default on the
    acceleration check with the own limit guard holding rows, the same variants with the guard smoothed over
    GUARD_SMOOTH_LONGER_S in turn (physio-v10). The clip's own solve beyond its level 0 (own) is computed once."""
    memo = []

    def own_once():
        if not memo:
            memo.append(own())
        return memo[0]
    kw = dict(scene=scene, table=table, kind=kind, resting=resting, own=None if own is None else own_once,
              reference=reference, checks=checks, _trace=_trace)
    best = _transfer(body, ref_raw, ref_out, layer, arm_ramp_s=ARM_FIX_SMOOTH_S, arm_smooth_s=ARM_FIX_SMOOTH_S,
                     rest_s=0.0, **kw)
    failed = best[1].get("validity", {}).get("failed", [])
    if not checks or not any(best[1]["guard"]["arms_fixed_knots"].values()):
        return best
    if not failed:
        # the default passes; where its own anchoring turned with a hand moving on its support (a tremor of large
        # amplitude shaking the forearm on and off an armrest), the rest gate, kept when it passes too: the own
        # correction is what a clip carries beyond its level 0's layer, and on the moving knots it turned the arm
        # against its level 0's and the gate lowers that (tremor_E1_holmes_g3, tests/test_synth_physio.py); the gate
        # fails where a hand pausing shorter than its ramp would stand off its support (_anchor_reading, dwell)
        if not any(best[1]["guard"].get("arms_fixed_moving_knots", {}).values()):
            return best
        res = _transfer(body, ref_raw, ref_out, layer, arm_ramp_s=ARM_FIX_SMOOTH_S, arm_smooth_s=ARM_FIX_SMOOTH_S,
                        rest_s=ARM_REST_S, **kw)
        if res[1]["validity"]["failed"]:
            return best
        # physio-v8: where the gate's anchoring still turns with the moving hand enough to leave the pair's change
        # above PAIR_SHARE_AIM of its sign (tremor_E1_holmes_g3: its large rest tremor lifts the hand on and off the
        # armrest, the gate's 0.1 s ramp anchored 815 moving knots, 0.125 of the sign), the longer ramps of
        # ARM_REST_LONGER_S, in turn: the first that passes and keeps the change under the aim is taken (a hand that
        # never rests that long takes no own anchoring)
        share = pair_change(body, res[0], ref_raw, ref_out)["share"]
        res[1]["guard"]["pair_share_gate"] = float(share)
        if share > PAIR_SHARE_AIM:
            for rest_s in ARM_REST_LONGER_S:
                r2 = _transfer(body, ref_raw, ref_out, layer, arm_ramp_s=ARM_FIX_SMOOTH_S,
                               arm_smooth_s=ARM_FIX_SMOOTH_S, rest_s=rest_s, **kw)
                if r2[1]["validity"]["failed"]:
                    continue
                s2 = pair_change(body, r2[0], ref_raw, ref_out)["share"]
                if s2 <= PAIR_SHARE_AIM:
                    r2[1]["guard"].update(pair_share_gate=float(s2), pair_share_gate_short=float(share))
                    return r2
        return res
    accel, anchor = "accel_excess_m_s2" in failed, "anchor_error_m" in failed
    tries = [(GUARD_SMOOTH_S, ARM_FIX_SMOOTH_S, 0.0)] * accel + [(ARM_FIX_FAST_S, ARM_FIX_FAST_S, 0.0)] * anchor
    tries += [(ARM_FIX_SMOOTH_S, ARM_FIX_SMOOTH_S, ARM_REST_S)]
    tries += [(GUARD_SMOOTH_S, ARM_FIX_SMOOTH_S, ARM_REST_S)] * accel
    tries += [(ARM_FIX_FAST_S, ARM_FIX_FAST_S, ARM_REST_S)] * anchor
    for ramp_s, smooth_s, rest_s in tries:
        res = _transfer(body, ref_raw, ref_out, layer, arm_ramp_s=ramp_s, arm_smooth_s=smooth_s, rest_s=rest_s,
                        dwell_binds=False, **kw)
        if not res[1]["validity"]["failed"]:
            return res
    if accel and best[1]["guard"].get("limits"):
        # physio-v10 (plan phase E): where every variant still fails and the default failed the acceleration check
        # with the own limit guard holding rows, the guard's factor smoothed over the longer windows of
        # GUARD_SMOOTH_LONGER_S, in turn, on the same variants: the first that passes is kept (named in the record,
        # guard.limits_smooth_s). The choreic fingers of E2 grade 4: the rest gate leaves the hand brushing its thigh
        # unanchored and passes the anchor check, but the left collar's guard, its factor smoothed over
        # GUARD_SMOOTH_S, adds 1.041 m/s2 at the left wrist at 4.35 s (scripts/openbody_phaseE_checks.py section
        # chorea_g4). Only a clip every variant refuses runs them: no other clip changes
        for gs in GUARD_SMOOTH_LONGER_S:
            for ramp_s, smooth_s, rest_s in tries:
                res = _transfer(body, ref_raw, ref_out, layer, arm_ramp_s=ramp_s, arm_smooth_s=smooth_s,
                                rest_s=rest_s, dwell_binds=False, guard_smooth_s=gs, **kw)
                if not res[1]["validity"]["failed"]:
                    res[1]["guard"]["limits_smooth_s"] = float(gs)
                    res[1]["deviations"]["own_limit_guard_smoothed"] = (
                        f"the own limit guard's factor smoothed over {gs:g} s (GUARD_SMOOTH_LONGER_S): over "
                        f"GUARD_SMOOTH_S every variant failed")
                    return res
    return best


def _transfer(body, ref_raw, ref_out, layer, scene=None, table="healthy", kind=None, resting=None, own=None,
              reference=None, checks=True, _trace=None, arm_ramp_s=ARM_FIX_SMOOTH_S, arm_smooth_s=ARM_FIX_SMOOTH_S,
              rest_s=0.0, dwell_binds=True, guard_smooth_s=GUARD_SMOOTH_S):
    """(Body, record): a member clip's background, transferred from its level 0 (ref_raw, ref_out: the level 0's
    raw and layered bodies; module docstring, Transfer). A row (joint, knot) the clip holds equal to its level 0's
    raw row (EQUAL_ROW_TOL) takes the level 0's output row, copied; a row the layer leaves unchanged in the level 0
    keeps the clip's own; every other row takes the level 0's delta, out_j(t) = D_j(t) R_in_j(t), the pelvis its
    translation delta, the hand joints theirs. Then the clip's own corrections, each on its own rows only:
    standing the legs re-solved so that the ankles keep the clip's positions and the feet its global rotations
    (legs_own_fix; the vertical goal where one of the hip and knee rows is held) and the sway scaled down on the
    clip's own pelvis rows where its own leg would leave the table to follow it (_sway_scaled), a hand resting on a
    support kept there (the clip's own touches, approach excluded; the shoulder and elbow rows its own; blended in by
    the transferred wrist's distance from its target, ANCHOR_FIX_M, and by rest_weight; smoothed in time,
    ARM_FIX_SMOOTH_S), the joint-limit guard (the clip's own excursion or the table it is judged on), the finger
    drift faded out on the clip's own finger rows held still (FINGER_RULE) and reduced where a finger joint would
    leave the hand limits beyond the clip's own range (_finger_guard); where the checks then read a finger or an arm
    pressed into
    the body, the finger lift faded and the arm turned out about its shoulder (contact_turn). Knots beyond the level
    0's length take the clip's own solve (own: a callable returning it), blended in (_extend). The record: the clip's
    own corrections in 'guard' and 'deviations', 'transfer' (the reference, the shares of rows held, unchanged and
    transferred, the rows corrected, the knots beyond), the anchors (anchor_error_m on the own arm rows of a resting
    hand, anchor_error_held_rows_m where either is held, anchor_error_moving_m where the hand moves,
    anchor_error_dwell_m where it stands still, read with the rest) and the
    validity as apply()'s."""
    from .hands import N_HAND, _eye6, hands_of, has_hands
    from ..body.rotations import matrix_to_rot6d, rot6d_to_matrix
    lim = joint_limits()
    if abs(float(body.rate) - LAYER_RATE) > 1e-9 or abs(float(ref_raw.rate) - float(body.rate)) > 1e-9:
        raise ValueError(f"the layer runs at {LAYER_RATE} Hz, the bodies at {body.rate} and {ref_raw.rate}")
    T, Tr = body.T, ref_raw.T
    Tm = min(T, Tr)
    rate = float(body.rate)
    rest = body.rest()
    dl = deltas(ref_raw, ref_out)
    rec_beyond = None
    if T > Tr:
        if own is None:
            raise ValueError(f"a clip of {T} knots beyond its level 0's {Tr} needs its own solve")
        o_out, rec_beyond = own()
        dl = _extend(dl, T, deltas(body, o_out), rate)
    else:
        dl = {k: (None if v is None else v[:T]) for k, v in dl.items()}
    # body rows: held (copied), unchanged by the layer (the clip's own), or transferred
    Rc_t = body.local_R()
    Rc = Rc_t.numpy()
    R = _mul(dl["R"], Rc)
    eq = np.zeros((T, N_BODY), bool)
    eq[:Tm] = rows_equal(body.rot6d, ref_raw.rot6d, Tm)
    untouched = np.zeros((T, N_BODY), bool)
    untouched[:Tm] = (ref_out.rot6d[:Tm] == ref_raw.rot6d[:Tm]).all(-1).numpy()
    Ro_ref = ref_out.local_R().numpy()
    R[untouched] = Rc[untouched]
    R[:Tm][eq[:Tm]] = Ro_ref[:Tm][eq[:Tm]]
    own_rows = ~eq
    eq_tr = np.zeros(T, bool)
    eq_tr[:Tm] = ((body.transl[:Tm] - ref_raw.transl[:Tm]).abs().amax(-1) <= EQUAL_ROW_TOL).numpy()
    tr = body.transl.numpy() + dl["tr"]
    tr[eq_tr] = ref_out.transl.numpy()[:Tm][eq_tr[:Tm]]
    R = torch.from_numpy(np.ascontiguousarray(R))
    tr = torch.from_numpy(np.ascontiguousarray(tr))
    R_moved = R.clone()                                         # before the clip's own corrections
    J_in, G_in = fk(Rc_t, body.transl, rest)
    toe = toe_stance(J_in, fk(ref_raw.local_R(), ref_raw.transl, ref_raw.rest())[0])   # feet on their toes
    guard, dev = {}, {}
    # 1. standing: the legs re-solved on their own rows so that the ankles and feet keep the clip's; where a leg
    #    joint the clip owns would then leave the joint-limit table it is judged on (a hip held at the clip's own
    #    extreme), the sway is scaled down on the clip's own pelvis and leg rows, to the largest scale that keeps the
    #    leg inside (bisection, SWAY_BISECT rounds), smoothed (_smooth_min) and back to 1 before a held row
    #    (physio-v5: the limit guard of step 3 undid the leg's solve and slid the foot,
    #    post_E7_torsion_dystonia_sev3)
    sway = bool(np.any(dl["tr"] != 0.0))
    legs_fixed = {}
    if sway:
        R_t, tr_t = R, tr                                      # the transfer, before the legs
        # the pelvis row, its translation and every leg row the clip's own (a held leg row is the level 0's answer to
        # the whole sway: under a scaled one its foot would slide)
        can = own_rows[:, 0] & ~eq_tr & own_rows[:, list(LEG_JOINTS)].all(-1)
        c_in_all = np.degrees(_aa(Rc_t).numpy())
        # physio-v10: the pelvis raised or lowered on those knots so that the legs keep the clip's lengths
        # (own_leg_lengths, ramped in from a held row)
        m_len = can if LEG_LENGTHS else np.zeros(T, bool)
        shifts = []

        def legs_at(scale):
            Rs, trs = (R_t, tr_t) if scale is None else _sway_scaled(R_t, tr_t, Rc_t, body.transl, dl, scale,
                                                                     own_rows, eq_tr)
            trs, sh = own_leg_lengths(Rs, trs, rest, J_in, m_len, rate)
            shifts.append(sh)
            fixed = {}
            for side_, leg in LEG.items():
                Rs, fixed[side_] = legs_own_fix(Rs, trs, rest, leg, J_in, G_in, own_rows, toe=toe[side_])
            return Rs, trs, fixed

        def outside(Rs):                                       # a leg joint the clip owns outside the table
            from . import limits_ext as LX
            c = np.degrees(_aa(Rs).numpy())
            lo, hi = LX.knot_bounds(table, c, c_in_all)
            j = list(LEG_JOINTS)
            out_ = ((c[:, j] < lo[:, j] - LIMIT_TOL_DEG) | (c[:, j] > hi[:, j] + LIMIT_TOL_DEG)).any(-1)
            return (out_ & own_rows[:, j]).any(-1)
        R, tr, legs_fixed = legs_at(None)
        bad = outside(R) & can
        scale = np.ones(T)
        if bad.any():
            a, b = np.zeros(T), np.ones(T)
            for _ in range(SWAY_BISECT):
                m = np.where(bad, 0.5 * (a + b), 1.0)
                ok = ~outside(legs_at(m)[0])
                a, b = np.where(bad & ok, m, a), np.where(bad & ~ok, m, b)
            n_g = max(1, int(round(GUARD_SMOOTH_S * rate)))
            x = np.clip(1.0 - _distance_to(~can) / n_g, 0.0, 1.0)
            floor = x * x * (3.0 - 2.0 * x)                    # back to the level 0's sway before a held row
            scale = np.where(can, floor + (1.0 - floor) * _smooth_min(np.where(bad, a, 1.0), n_g), 1.0)
            R, tr, legs_fixed = legs_at(scale)
        guard["sway_scaled_knots"] = int((scale < 1.0 - 1e-12).sum())
        sh = shifts[-1]                                        # the last legs_at is the one kept
        if np.any(sh != 0.0):                                  # (named only where it acts: other records unchanged)
            guard["leg_lengths_knots"] = int(np.count_nonzero(sh))
            guard["leg_lengths_shift_m"] = float(np.abs(sh).max())
            dev["own_leg_lengths"] = ("the pelvis raised or lowered on the clip's own pelvis and leg rows so that the "
                                      "legs keep the clip's hip-to-ankle distances under the level 0's sway "
                                      "(guard.leg_lengths_knots, leg_lengths_shift_m)")
        if any(legs_fixed.values()):
            dev["own_legs"] = ("the legs re-solved on their own rows (guard.legs_fixed_knots: the hip and knee "
                               "where both are the clip's, the one of them that is elsewhere, the ankle for the foot) "
                               "so that the ankles keep the clip's positions and the feet its global rotations")
        if guard["sway_scaled_knots"]:
            dev["own_sway_scaled"] = ("the level 0's sway scaled down on the clip's own pelvis rows at the knots of "
                                      "guard.sway_scaled_knots, where its own leg would leave the joint-limit table "
                                      "to follow it")
    guard["legs_fixed_knots"] = legs_fixed
    # physio-v10 (plan phase E): the joint-limit guard of step 3 on every joint but the arms' first, as apply() guards
    # the trunk before it anchors the hands, so that the hands are kept on their supports on the trunk and legs as
    # guarded: guarded after the arms, the spine1 of the tardive rocking s4 held at its flexion bound carried both
    # hands resting on the thighs off their places (the left wrist 1.92 mm from its goal at 11.38 s, 2.16 mm before
    # the arms' correction; scripts/openbody_phaseE_checks.py section trunk_guard). Each
    # joint's guard reads its own rows (the shoulders' dystonia bound the collars as guarded, which this pass keeps
    # before them), so a clip whose trunk and legs the guard leaves alone is unchanged bit for bit
    g_first = {}
    if TRUNK_GUARD_FIRST:
        R, g_first = guard_joints(R, Rc_t, [j for j in range(1, N_BODY) if j not in ARM_JOINTS], lim, rate,
                                  allowed=own_rows, table=table, smooth_s=guard_smooth_s)
    # 2. the hands on their supports: the clip's own touches, on its own arm rows
    boxes = []
    if scene is not None:
        from .viewer import scene_floor
        boxes = scene_boxes(scene, body, scene_floor(body, 5.0)[0])
    # a member's touches: the supports of the solve and the trunk (a hand on the belly or the chest)
    kind_ = kind or "standing"
    labels = support_labels(body, boxes, support_joints(kind_) + (TRUNK_TOUCH if kind_ != "standing" else ()),
                            approach=False)
    J_tr, G_tr = fk(R, tr, rest)
    # physio-v7: a wrist the clip holds where its level 0 holds it (its position and its hand's global rotation equal
    # to the level 0's raw ones, HELD_WRIST_TOL: the arms the seated trunk jerks carry over their supports, jerks.py)
    # goes where the level 0's layer took it, whatever its support does under the clip's own trunk (the propriospinal
    # jerk's carried wrist moved 3.76 mm against its level 0's under physio-v6, its gate off while the thigh moved)
    Jr_in, Gr_in = fk(ref_raw.local_R(), ref_raw.transl, ref_raw.rest())
    Jr_out, Gr_out = fk(ref_out.local_R(), ref_out.transl, ref_out.rest())
    held_wrists = {}
    anchors, arms_fixed, arms_moving = {}, {}, {}
    n_s = max(1, int(round(arm_smooth_s * rate)))               # the smoothing of the correction (transfer)
    n_r = max(1, int(round(arm_ramp_s * rate)))                 # the ramp inside the own rows (transfer)
    for side, chain in ARM.items():
        _, sh, el, wr = chain
        touch = np.asarray(labels[side]) != -1
        # the place on the support, ramped in before the touch and cross-faded between two supports (the anchor
        # check reads the same goal)
        goal, H, on, _ = _anchor_goals(J_in, G_in, J_tr, G_tr, labels, side, rate)
        hp = np.zeros(T, bool)
        if Tm:
            hp[:Tm] = (((J_in[:Tm, wr] - Jr_in[:Tm, wr]).norm(dim=-1) <= HELD_WRIST_TOL)
                       & ((G_in[:Tm, wr] - Gr_in[:Tm, wr]).abs().amax((-1, -2)) <= HELD_WRIST_TOL)).numpy()
        held_wrists[side] = int(hp.sum())
        hw = np.zeros(T)
        if hp.any():
            # ramped in and out inside each held run (inside_ramp over ARM_REST_S): a goal switched at the run's
            # edge jumped where the hand leaves its rest for the face (the touching tics, whose fallback then pressed
            # the index 5 mm into the head on the page's check)
            hw = inside_ramp(hp, max(1, int(round(ARM_REST_S * rate))))
            g_ref, h_ref = goal.clone(), H.clone()
            g_ref[:Tm], h_ref[:Tm] = Jr_out[:Tm, wr], Gr_out[:Tm, wr]
            hw_t = torch.from_numpy(hw)
            goal = goal + hw_t[:, None] * (g_ref - goal)
            H = _slerp(H, h_ref, hw)
        err = (J_tr[:, wr] - goal).norm(dim=-1).numpy()
        # 0 where the shoulder or the elbow row is held: an elbow alone under the level 0's shoulder cannot reach the
        # goal's direction, and bending it toward the distance moved a kinetic tremor's wrist on the reach by up to
        # 15 mm against its level 0's (tremor_E3_kinetic_g4, physio-v4 trial); there the arm reads as held
        mine = inside_ramp(own_rows[:, sh] & own_rows[:, el], n_r)
        # with rest_s (transfer's fallback, physio-v5) only where the clip's own hand rests on its support
        # (rest_weight): a hand flung off or onto it moves at metres per second, and a correction turning with it adds
        # acceleration of its own; rest_s 0 (the default): everywhere, as physio-v4
        rest_w = np.ones(T)
        # the knots where the hand moves on its support (the rest gate's weight below 1), read whatever rest_s, so
        # that transfer knows whether a passing default anchoring turned with a moving hand
        speed = wrist_speed(J_in, G_in, labels, side, rate)
        rest_probe = rest_weight(speed, rate, n_s, ARM_REST_S)
        if rest_s > 0.0:
            rest_w = rest_probe if rest_s == ARM_REST_S else rest_weight(speed, rate, n_s, rest_s)
            rest_w = np.maximum(rest_w, hw)                    # a held wrist stands where its level 0's stands
        a = np.clip((err - ANCHOR_FIX_M[0]) / (ANCHOR_FIX_M[1] - ANCHOR_FIX_M[0]), 0.0, 1.0)
        w = mine * rest_w * a * a * (3.0 - 2.0 * a)            # smoothstep of the distance: no threshold jump
        k = np.flatnonzero(w > 0.0)
        arms_fixed[side] = int(len(k))
        arms_moving[side] = int(np.count_nonzero((w > 0.0) & (rest_probe < 1.0 - 1e-12)))
        if _trace is not None:                                 # diagnostics only
            _trace[side] = {"err": err, "w": w, "mine": mine, "touch": touch, "rest": rest_w}
        if len(k):
            R2, _ = two_bone_fix(R, tr, rest, chain, goal, _elbow_axis(side), end_global=H, knots=k,
                                 damp=ARM_FIX_DAMP_M)
            for j in (sh, el, wr):
                # the correction (parent frame, weighted) smoothed by two moving means, then held to the own rows
                c = _aa(R2[:, j] @ R[:, j].transpose(-1, -2)).numpy() * w[:, None]
                c = _moving_mean(_moving_mean(c, n_s), n_s) * (mine if j != wr else
                                                               mine * inside_ramp(own_rows[:, wr], n_r))[:, None]
                if np.any(c != 0.0):
                    R[:, j] = torch.where(torch.from_numpy(np.any(c != 0.0, -1))[:, None, None],
                                          _exp(torch.from_numpy(c)) @ R[:, j], R[:, j])
        anchors[side] = {"anchored_share": float(touch.mean()), "solved_knots": int(len(k)),
                         "supports": sorted(int(x) for x in np.unique(labels[side]) if x != -1),
                         "_goal": goal, "_touch": touch, "_own_arm": own_rows[:, sh] & own_rows[:, el],
                         "_full": mine >= 1.0 - 1e-12, "_rest": rest_w,
                         "_dwell": _moving_max(speed, int(round(ARM_DWELL_S * rate))) <= ARM_DWELL_M_S}
    guard["arms_fixed_knots"] = arms_fixed
    guard["arms_fixed_moving_knots"] = arms_moving
    guard["wrists_held_knots"] = held_wrists
    guard["arm_ramp_s"], guard["arm_smooth_s"] = float(arm_ramp_s), float(arm_smooth_s)
    guard["arm_rest_s"] = float(rest_s)
    if any(arms_fixed.values()):
        dev["own_arms"] = ("a hand kept on its own support on the clip's own arm rows (guard.arms_fixed_knots), "
                           "blended in by the transferred wrist's distance from it")
    # 3. the joint-limit guard on the clip's own rows (the clip's own excursion or the table it is judged on); the
    #    knots where it holds an arm's shoulder or elbow are kept for the anchor reading (physio-v5)
    R_pre_guard = R
    R, g = guard_joints(R, Rc_t, ARM_JOINTS if TRUNK_GUARD_FIRST else range(1, N_BODY), lim, rate, allowed=own_rows,
                        table=table, smooth_s=guard_smooth_s)
    held_by_guard = ~(R == R_pre_guard).flatten(-2).all(-1).numpy()
    g = {j: g_first[j] if j in g_first else g[j] for j in sorted(set(g_first) | set(g))}
    guard["limits"] = g
    if g:
        dev["own_limit_guard"] = ("the transferred layer blended back toward the clip on its own rows at the knots of "
                                  "guard.limits (a joint at its limit)")
    write = own_rows & ~(R == Rc_t).flatten(-2).all(-1).numpy()
    r6 = torch.where(torch.from_numpy(write)[..., None], matrix_to_rot6d(R), body.rot6d)
    r6[:Tm] = torch.where(torch.from_numpy(eq[:Tm])[..., None], ref_out.rot6d[:Tm], r6[:Tm])
    out = replace(body.copy(), rot6d=r6, transl=tr.clone(), contact=body.contact.clone())
    # 4. the hand joints: every own row takes the level 0's delta (the finger lift about each joint's flexion axis
    #    in its parent frame, so that it lifts whatever the clip's hand posture), guarded against the hand limits
    finger_guarded = {}
    hand_parts = None

    def with_hands(o, fade=None):
        # the hand rows: held ones copied, unchanged ones the clip's, own ones Hn (faded toward the clip's by fade)
        hb, h_in, Hc, Hn, heq, hun = hand_parts
        H = Hn
        if fade is not None:
            ti, ri = np.nonzero(~heq & ~hun & (fade[:, None] < 1.0 - 1e-12))
            if len(ti):
                H = Hn.copy()
                H[ti, ri] = _slerp(torch.from_numpy(Hc[ti, ri]), torch.from_numpy(Hn[ti, ri]), fade[ti]).numpy()
        h6 = matrix_to_rot6d(torch.from_numpy(np.ascontiguousarray(H)))
        h6 = torch.where(torch.from_numpy(hun)[..., None], h_in, h6)
        h6[:Tm] = torch.where(torch.from_numpy(heq[:Tm])[..., None], ref_out.hand6d[:Tm], h6[:Tm])
        if h6.shape[1] != N_HAND:
            raise RuntimeError("hand rows")
        return replace(hb, rot6d=o.rot6d, transl=o.transl, contact=o.contact, hand6d=h6)
    if dl["H"] is not None:
        hb = hands_of(out) if not has_hands(out) else out
        h_in = body.hand6d if has_hands(body) else _eye6(T)
        Hc = rot6d_to_matrix(h_in).numpy()
        Hn = _mul(dl["H"], Hc)
        ref_h_in = ref_raw.hand6d if has_hands(ref_raw) else _eye6(Tr)
        heq = np.zeros(Hn.shape[:2], bool)
        heq[:Tm] = rows_equal(h_in, ref_h_in, Tm)
        hun = np.zeros(Hn.shape[:2], bool)
        hun[:Tm] = (ref_out.hand6d[:Tm] == ref_h_in[:Tm]).all(-1).numpy()
        # a finger row held still in a posture far from the level 0's takes no finger drift (FINGER_RULE, physio-v5,
        # from the session's decision of 2026-11-03): the drift fades out over FINGER_STILL_RAMP_S inside each run of
        # such rows (inside_ramp: whole at the run's ends, so that nothing jumps), the clip's own row kept beyond;
        # knots beyond the level 0 keep the extended layer
        sign_rows = np.zeros(Hn.shape[:2], bool)
        if FINGER_RULE != "none":
            sign_rows[:Tm] = ~heq[:Tm] & ~hun[:Tm]
        if FINGER_RULE == "held" and sign_rows.any():
            Hc_t = torch.from_numpy(Hc[:Tm])
            turn = _aa(Hc_t[1:].transpose(-1, -2) @ Hc_t[:-1]).norm(dim=-1).numpy()
            speed = np.degrees(np.maximum(np.r_[turn[:1], turn], np.r_[turn, turn[-1:]])) * rate
            far = np.degrees(_aa(rot6d_to_matrix(ref_h_in[:Tm]).transpose(-1, -2) @ Hc_t).norm(dim=-1).numpy())
            sign_rows[:Tm] &= (speed < FINGER_HELD_DEG_S) & (far > FINGER_HELD_FROM_DEG)
        n_f = max(1, int(round(FINGER_STILL_RAMP_S * rate)))
        keep = np.stack([1.0 - inside_ramp(sign_rows[:, r], n_f) for r in range(Hn.shape[1])], 1)
        ti, ri = np.nonzero(keep < 1.0 - 1e-12)
        D_h = dl["H"]
        if len(ti):
            Hn[ti, ri] = _slerp(torch.from_numpy(Hc[ti, ri]), torch.from_numpy(Hn[ti, ri]), keep[ti, ri]).numpy()
            D_h = D_h.copy()
            D_h[ti, ri] = _mul(Hn[ti, ri], _tp(Hc[ti, ri]))
        guard["fingers_still_knots"] = int((keep < 1.0 - 1e-12).any(1).sum())
        if guard["fingers_still_knots"]:
            dev["own_fingers_still"] = ("the finger drift faded out on the clip's own finger rows held still "
                                        "(guard.fingers_still_knots, FINGER_RULE, FINGER_STILL_RAMP_S)")
        Hn, finger_guarded = _finger_guard(np.ascontiguousarray(Hn), Hc, D_h, ~heq & ~hun)
        hand_parts = (hb, h_in, Hc, Hn, heq, hun)
        out = with_hands(out)
    elif has_hands(out):                                        # no finger layer: the held hand rows copied too
        ref_h_in = ref_raw.hand6d if has_hands(ref_raw) else _eye6(Tr)
        ref_h_out = ref_out.hand6d if has_hands(ref_out) else _eye6(Tr)
        heq = torch.from_numpy(rows_equal(out.hand6d, ref_h_in, Tm))
        h6 = out.hand6d.clone()
        h6[:Tm] = torch.where(heq[..., None], ref_h_out[:Tm], h6[:Tm])
        out = replace(out, hand6d=h6)
    guard["fingers_guarded_knots"] = finger_guarded
    if any(finger_guarded.values()):
        dev["own_finger_guard"] = ("the transferred finger lift reduced on the clip's own finger rows where a joint "
                                   "would leave the hand limits of hands.py beyond the clip's own range "
                                   "(guard.fingers_guarded_knots)")
    if rec_beyond is not None:
        dev["beyond_reference"] = (f"the {T - Tr} knots beyond the level 0's {Tr} take the clip's own solve, blended "
                                   f"in over {BEYOND_RAMP_S} s")
    tw_all = np.maximum(touch_weight(labels, "left", rate), touch_weight(labels, "right", rate))
    rec = {"version": VERSION, "layer_sha256": layer_digest(layer, T), "knots": T, "guard": guard,
           "touch_share": float((tw_all > 0).mean()), "deviations": dev}
    corrected = ~((R == R_moved).flatten(-2).all(-1)).numpy()
    rec["transfer"] = {"reference": reference, "reference_knots": int(Tr), "beyond_knots": int(max(0, T - Tr)),
                       "rows_held_share": float(eq[:Tm].mean()) if Tm else 0.0,
                       "rows_unchanged_share": float((untouched[:Tm] & ~eq[:Tm]).mean()) if Tm else 0.0,
                       "rows_transferred_share": float((~eq[:Tm] & ~untouched[:Tm]).mean()) if Tm else 0.0,
                       "rows_corrected": int(corrected.sum()),
                       "rows_corrected_held": int((corrected & eq).sum()),
                       "beyond_record": None if rec_beyond is None else
                       {"deviations": sorted(rec_beyond.get("deviations", {})),
                        "failed": list(rec_beyond.get("validity", {}).get("failed", []))}}
    a_parts = {side: tuple(a.pop(x) for x in ("_goal", "_touch", "_own_arm", "_full", "_rest", "_dwell"))
               + (held_by_guard[:, [ARM[side][1], ARM[side][2]]].any(-1),) for side, a in anchors.items()}

    def read_anchors(o):
        Jo = fk(o.local_R(), o.transl, rest)[0]
        for side_, a_ in anchors.items():
            goal_, touch_, own_arm_, full_, rest_w_, dwell_, guarded_ = a_parts[side_]
            err_ = (Jo[:, ARM[side_][3]] - goal_).norm(dim=-1).numpy()
            a_.update(_anchor_reading(err_, touch_, own_arm_, full_, n_r, rest_w_, guarded_, dwell_, dwell_binds))
    read_anchors(out)
    rec["anchors"] = anchors
    J_out, _ = fk(out.local_R(), out.transl, rest)
    if sway:
        rec["ankle_error_m"] = float((J_out[:, [7, 8]] - J_in[:, [7, 8]]).norm(dim=-1).max())
    if checks:
        v = validity(body, out, lim, boxes, sway, table, skate_tol=SKATE_TOL_MEMBER_M_S, toe=toe)

        def excess(x):
            return max(x["body_added_m"] - BODY_TOL_M, x["scene_added_m"] - SCENE_TOL_M)
        if hand_parts is not None and excess(v) > 0.0:
            # the transferred finger lift may press a finger the clip holds otherwise (a palm turned up, a finger
            # against the belly) into the body or a scene box: faded on the clip's own finger rows around the failing
            # knots (local_fade, FADE_TRIES rounds), kept only where that lowers the depth added
            fade, o2, v2 = None, out, v
            for _ in range(FADE_TRIES):
                if not len(v2["_bad_knots"]):
                    break
                fade = local_fade(v2["_bad_knots"], T, rate, fade)
                o2 = with_hands(out, fade)
                v2 = validity(body, o2, lim, boxes, sway, table, skate_tol=SKATE_TOL_MEMBER_M_S, toe=toe)
                if excess(v2) <= 0.0:
                    break
            if fade is not None and excess(v2) < excess(v):
                out, v = o2, v2
                guard["fingers_contact_faded_knots"] = int((fade < 1.0 - 1e-12).sum())
                dev["own_finger_fade"] = ("the transferred finger lift faded on the clip's own finger rows around the "
                                          "knots where it added depth inside the body or a scene box "
                                          "(guard.fingers_contact_faded_knots)")
        if v["body_added_m"] > BODY_TOL_M:
            # the layer moves the trunk into a forearm or hand the clip holds against it (a choreic hand on the
            # trunk, invol_chorea_E1_g4): the arm turned out about its shoulder on the clip's own shoulder rows
            # (contact_turn, FADE_TRIES rounds), kept only where it lowers that depth and fails no other check
            # (physio-v5)
            om, o2, v2 = None, out, v
            for _ in range(FADE_TRIES):
                add = contact_turn(body, o2, own_rows, rate)
                if add is None:
                    break
                om = add if om is None else {s: om[s] + add[s] for s in om}
                o2 = with_arm_turn(out, om, own_rows)
                v2 = validity(body, o2, lim, boxes, sway, table, skate_tol=SKATE_TOL_MEMBER_M_S, toe=toe)
                if v2["body_added_m"] <= BODY_TOL_M:
                    break
            if om is not None and v2["body_added_m"] < v["body_added_m"] and \
                    not set(failed_checks(v2)) - set(failed_checks(v)) - {"anchor_error_m"}:
                out, v = o2, v2
                read_anchors(out)
                guard["arm_contact_turned_knots"] = {s: int((np.linalg.norm(om[s], axis=-1) > 0.0).sum()) for s in om}
                dev["own_arm_contact"] = ("an arm turned out about its shoulder on the clip's own shoulder rows where "
                                          "the layer pressed its forearm or hand into the body "
                                          "(guard.arm_contact_turned_knots)")
        if v["scene_added_m"] > SCENE_TOL_M and boxes:
            # the layer turns a forearm the clip holds on an armrest's edge into that edge (scene_turn, FADE_TRIES
            # rounds), kept only where the depth falls within SCENE_TOL_M and no other check fails
            om, o2, v2 = {}, out, v
            for _ in range(FADE_TRIES):
                add = scene_turn(body, o2, own_rows, rate, boxes)
                if add is None:
                    break
                for j, x in add.items():
                    om[j] = om[j] + x if j in om else x
                o2 = with_row_turn(out, om, own_rows)
                v2 = validity(body, o2, lim, boxes, sway, table, skate_tol=SKATE_TOL_MEMBER_M_S, toe=toe)
                if v2["scene_added_m"] <= SCENE_TOL_M:
                    break
            if om and v2["scene_added_m"] <= SCENE_TOL_M and not set(failed_checks(v2)) - set(failed_checks(v)):
                out, v = o2, v2
                read_anchors(out)
                guard["scene_turned_knots"] = {int(j): int((np.linalg.norm(x, axis=-1) > 0.0).sum())
                                               for j, x in om.items()}
                dev["own_scene_turn"] = ("an arm turned off a scene box on the clip's own rows where the layer pressed "
                                         "it in (guard.scene_turned_knots: the shoulder row, about the line to the "
                                         "wrist, or a spine row)")
        v["anchor_error_m"] = max(a["anchor_error_m"] for a in anchors.values())
        for key in ("anchor_error_ramp_m", "anchor_error_held_rows_m", "anchor_error_moving_m",
                    "anchor_error_guarded_m", "anchor_error_dwell_m"):
            v[key] = max(a[key] for a in anchors.values())
        v["ok"] = bool(v["ok"] and v["anchor_error_m"] <= ANCHOR_TOL_M)
        v["failed"] = failed_checks(v)
        v.pop("_bad_knots", None)
        rec["validity"] = v
    return out, rec


def solve(body, layer, **kw):
    """(Body, record): apply() with the local fade of a contact failure (the series root's own solve, and a clip's
    own solve beyond its level 0's length): where the full layer fails a contact check, the layer fades locally
    around the failing knots (local_fade), the breathing and the drifts first (FADED_FIRST, FADE_TRIES rounds), then
    the sway too; the fade is named in the record's deviations. Where the anchoring adds acceleration to a hand flung
    off and back onto its support (two supported runs meeting inside the anchoring's ramp, the wrist pinned to the
    clip's and the elbow of an arm near straight swinging: the right hand's ballistic movement of the functional
    tremor on the armrest), the anchoring only where the hand rests (rest_gate, kept where that fails fewer checks;
    a root passing without it never runs it, so its body and its series keep their bits). Where the acceleration
    check still fails, the anchoring's elbow takes the damped flexion step of a member's own arm correction
    (arm_damp ARM_FIX_DAMP_M, kept where that fails fewer checks: 'anchor_damped'): the undamped Newton step can take
    the other elbow solution of an arm near straight for one knot (the right hand lifting off the armrest in the
    raise after a rest, the elbow near straight). Where neither passes alone (the gate not kept), both together, kept
    likewise (physio-v10)."""
    out, rec = apply(body, layer, **kw)
    gated = damped = False
    if "accel_excess_m_s2" in rec["validity"]["failed"]:
        o2, r2 = apply(body, layer, rest_gate=True, **kw)
        if len(r2["validity"]["failed"]) < len(rec["validity"]["failed"]):
            out, rec, gated = o2, r2, True
            kw = dict(kw, rest_gate=True)
    if "accel_excess_m_s2" in rec["validity"]["failed"]:
        o2, r2 = apply(body, layer, arm_damp=ARM_FIX_DAMP_M, **kw)
        if len(r2["validity"]["failed"]) < len(rec["validity"]["failed"]):
            out, rec, damped = o2, r2, True
            kw = dict(kw, arm_damp=ARM_FIX_DAMP_M)
    if "accel_excess_m_s2" in rec["validity"]["failed"] and not gated:
        # physio-v10 (plan phase E): neither fallback passing alone, both together, kept likewise. On Anny the raise
        # after the longer rest lifts both straightening arms off the armrests while the anchoring still holds them
        # at full weight: undamped 80.71 m/s2 added at the elbow at 4.81 s, the gate alone 6.65 (5.25 s), the
        # damped step alone 3.49 (4.79 s), both 0.28 (tremor_E2_raise_long_rest_g0); the gate failed as many checks
        # as the default, so the damped step was tried on the default only
        o2, r2 = apply(body, layer, **dict(kw, rest_gate=True, arm_damp=ARM_FIX_DAMP_M))
        if len(r2["validity"]["failed"]) < len(rec["validity"]["failed"]):
            out, rec, gated, damped = o2, r2, True, True
            kw = dict(kw, rest_gate=True, arm_damp=ARM_FIX_DAMP_M)
    fade, trigger, keys = None, [], FADED_FIRST
    for i in range(2 * FADE_TRIES):
        bad = rec["validity"]["_bad_knots"]
        if rec["validity"]["ok"] or not len(bad):
            break
        if i == FADE_TRIES:                                # the drifts and the breathing faded did not do:
            keys = FADED                                   # the sway too
        trigger = sorted(set(trigger) | set(rec["validity"]["failed"]))
        fade = local_fade(bad, body.T, float(body.rate), fade)
        out, rec = apply(body, layer, fade=fade, fade_keys=keys, **kw)
    rec["validity"].pop("_bad_knots", None)
    if fade is not None:
        low = np.r_[False, fade < 1.0 - 1e-12, False]
        edges = np.flatnonzero(np.diff(low.astype(np.int8)))
        rec["guard"]["faded_knots"] = int((fade < 1.0 - 1e-12).sum())
        rec["guard"]["faded_spans_s"] = [[float(a) / body.rate, float(b - 1) / body.rate]
                                         for a, b in zip(edges[::2], edges[1::2])]
        rec["guard"]["fade_trigger"] = trigger
        rec["guard"]["faded_components"] = [k for k in keys if np.any(np.asarray(layer[k]) != 0)]
        rec["deviations"]["faded"] = (
            f"the layer ({', '.join(rec['guard']['faded_components'])}) faded locally on "
            f"{rec['guard']['faded_knots']} knots, where the full layer failed {', '.join(trigger)}")
    if gated:
        rec["guard"]["anchor_rest_gate"] = True
        rec["deviations"]["anchor_rest_gate"] = ("a hand anchored only where it rests on its support: the anchoring "
                                                 "on a hand flung off it failed the acceleration check")
    if damped:
        rec["guard"]["anchor_damped"] = True
        rec["deviations"]["anchor_damped"] = (f"the anchoring's elbow solved with the damped flexion step "
                                              f"(ARM_FIX_DAMP_M {ARM_FIX_DAMP_M:g} m per rad): the undamped step "
                                              "failed the acceleration check")
    return out, rec


PAIR_EVERY = 5                  # [module] knots between two mesh readings of pair_change (20 Hz at 100 Hz)


def pair_change(c_raw, c_out, l_raw, l_out, every=PAIR_EVERY, chunk=128):
    """{'joints': {...}, 'vertices': {...}, 'sign_m', 'change_m', 'share'} of a clip and its level 0 (raw and
    layered bodies, the first min(T) knots): the sign (the largest |clip - level 0| without the layer) and the change
    the layer brings to it (the largest |(clip - level 0) with the layer - the same without|), on the 22 body joints
    at every knot and on every vertex of the mesh at every `every`-th knot (a finger, a toe or a head turn moves
    vertices, not joints; viewer.knot_vertices with the bodies' hand joints), each with where the change peaks
    ('at_s'; the joint, or the vertex's dominant joint); 'share' the largest change over the largest sign, both read
    on joints and vertices."""
    from .viewer import dominant_joint, knot_vertices
    T = min(c_raw.T, l_raw.T)
    rate = float(c_raw.rate)
    wo = c_raw.joints().numpy()[:T] - l_raw.joints().numpy()[:T]
    wb = c_out.joints().numpy()[:T] - l_out.joints().numpy()[:T]
    sj, cj = np.linalg.norm(wo, axis=-1), np.linalg.norm(wb - wo, axis=-1)
    k, j = np.unravel_index(int(np.argmax(cj)), cj.shape)
    out = {"joints": {"sign_m": float(sj.max()), "change_m": float(cj.max()), "at_s": float(k) / rate,
                      "joint": int(j)}}
    ks = np.arange(0, T, every)
    sv = cv = 0.0
    at = (0, 0)
    for a in range(0, len(ks), chunk):
        kk = ks[a:a + chunk]
        V = [knot_vertices(b, kk) for b in (c_raw, l_raw, c_out, l_out)]
        so = V[0] - V[1]
        d = np.linalg.norm((V[2] - V[3]) - so, axis=-1)
        sv = max(sv, float(np.linalg.norm(so, axis=-1).max()))
        i, v = np.unravel_index(int(np.argmax(d)), d.shape)
        if float(d[i, v]) > cv:
            cv, at = float(d[i, v]), (int(kk[i]), int(v))
    out["vertices"] = {"sign_m": sv, "change_m": cv, "at_s": float(at[0]) / rate,
                       "joint": int(dominant_joint("neutral")[at[1]]), "every": int(every)}
    out["sign_m"] = max(out["joints"]["sign_m"], sv)
    out["change_m"] = max(out["joints"]["change_m"], cv)
    out["share"] = out["change_m"] / max(out["sign_m"], 1e-12)
    return out


def _held_from(body, out, rec, span, ref_raw, ref_out, ref_span, scene=None, table="healthy"):
    """(Body, record) of a walking member's own solve (SeriesCache.layered, physio-v7) with every knot it holds equal
    to its level 0's raw one (all its body rows and the pelvis translation within EQUAL_ROW_TOL) where the two spans
    weigh alike and above 0 taken from the level 0's output, so that the two carry one layer there bit for bit; a knot
    holding only some rows keeps the own solve (a held pelvis row under an own translation: the level 0's row copied
    beside the own sway jumped by 3.4e-4 where the copy stopped, 1.74 m/s2 at the wrist of gait_E8_dystonic_left_l2);
    the validity read again when the copy changed the output (record '_held': the shares, the largest change the copy
    made, the largest difference left on a row held at a knot held in part)."""
    T, Tm = body.T, min(body.T, ref_raw.T)
    same = np.zeros(T, bool)
    same[:Tm] = (np.asarray(span[:Tm]) == np.asarray(ref_span[:Tm])) & (np.asarray(span[:Tm]) > 0.0)   # outside
    #                                                           the span the rows are the clip's own
    rows = np.zeros((T, N_BODY), bool)
    rows[:Tm] = rows_equal(body.rot6d, ref_raw.rot6d, Tm)
    eq_tr = np.zeros(T, bool)
    eq_tr[:Tm] = ((body.transl[:Tm] - ref_raw.transl[:Tm]).abs().amax(-1) <= EQUAL_ROW_TOL).numpy()
    whole = same & rows.all(-1) & eq_tr
    r6, tr = out.rot6d.clone(), out.transl.clone()
    k = torch.from_numpy(whole[:Tm])
    r6[:Tm] = torch.where(k[:, None, None], ref_out.rot6d[:Tm], r6[:Tm])
    tr[:Tm] = torch.where(k[:, None], ref_out.transl[:Tm], tr[:Tm])
    diff = max(float((r6 - out.rot6d).abs().max()), float((tr - out.transl).abs().max()))
    part = rows[:Tm] & same[:Tm, None] & ~whole[:Tm, None]
    left = float((out.rot6d[:Tm] - ref_out.rot6d[:Tm]).abs().amax(-1).numpy()[part].max()) if part.any() else 0.0
    rec = dict(rec, _held={"knots_held_share": float(whole[:Tm].mean()) if Tm else 0.0,
                           "rows_held_share": float((rows[:Tm] & same[:Tm, None]).mean()) if Tm else 0.0,
                           "copied_max_diff": diff, "part_held_rows_maxdiff": left})
    if diff == 0.0:
        return out, rec
    out = replace(out, rot6d=r6, transl=tr)
    if "validity" in rec:
        boxes = []
        if scene is not None:
            from .viewer import scene_floor
            boxes = scene_boxes(scene, body, scene_floor(body, 5.0)[0])
        old = rec["validity"]
        v = validity(body, out, joint_limits(), boxes, True, table)
        v.pop("_bad_knots", None)
        v["anchor_error_m"] = old.get("anchor_error_m", 0.0)
        v["ok"] = bool(v["ok"] and v["anchor_error_m"] <= ANCHOR_TOL_M)
        v["failed"] = failed_checks(v)
        rec["validity"] = v
    return out, rec


# ---------------------------------------------------------------- series
def series_root(spec_id, by_id):
    """The root of a clip's level-0 chain (its own id for a level 0)."""
    seen = []
    s = by_id[spec_id]
    while s.level0 is not None:
        if s.clip_id in seen:
            raise RuntimeError(f"level-0 cycle through {s.clip_id}")
        seen.append(s.clip_id)
        s = by_id[s.level0]
    return s.clip_id


def posture_kind(spec):
    """'walking', 'seated', 'supine', 'transfer' or 'standing' of a spec, from its segment and scene (module
    docstring)."""
    if spec.segment in WALK_SEGMENTS:
        return "walking"
    if spec.scene in SEATED_SCENES:
        return "seated"
    if spec.scene == "exam_bed":
        return "supine"
    if spec.scene == "chair_arise":
        return "transfer"
    return "standing"


def skip_reason(root_spec):
    """Why a series takes no layer (SKIP), None when it takes one (a walking segment's series takes it on its still
    spans since physio-v7)."""
    return SKIP.get(root_spec.clip_id)


def amplitudes(kind):
    """The physical amplitudes of a kind's components and their sources (JSON-ready)."""
    comps = KINDS[kind]
    out = {}
    if "breath" in comps:
        out["breath"] = {"rate_per_min": BREATH_PER_MIN, "ti_te": BREATH_TI_TE, "rate_cv": BREATH_RATE_CV,
                         "spine3_extension_deg": BREATH_SPINE_DEG, "collar_elevation_deg": BREATH_COLLAR_DEG,
                         "sources": ["motamedifakhr2017", "module"]}
    if "head" in comps:
        out["head"] = {"yaw_pitch_rms_deg": list(HEAD_DRIFT_DEG), "band_hz": list(HEAD_BAND_HZ), "sources": ["module"]}
    if "trunk" in comps:
        g = sit_gains()
        out["trunk"] = {"t2_range_m": list(SIT_T2_RANGE_M), "window_s": SIT_WINDOW_S, "band_hz": list(SIT_BAND_HZ),
                        "lean_rms_deg": list(g["lean_rms_deg"]), "range_per_rms": g["range_per_rms"],
                        "lever_m": [g["lever_ap_m"], g["lever_ml_m"]], "forward_only": True,
                        "sources": ["melo2009", "module"]}
    if "sway" in comps:
        out["sway"] = {"pelvis_rms_m": list(SWAY_RMS_M), "band_hz": list(SWAY_BAND_HZ),
                       "strategy": "ankle front-back, hip sideways",
                       "sources": ["foulger2025", "yamamoto2015", "winter1998", "module"]}
    if "fingers" in comps:
        out["fingers"] = {"flexion_rms_deg": FINGER_DRIFT_DEG, "band_hz": list(FINGER_BAND_HZ), "sources": ["module"]}
    return out


def _raw_build(spec):
    return spec.build.raw if isinstance(spec.build, LayeredBuild) else spec.build


def _same_body(a, b):
    from .hands import has_hands
    if a.T != b.T or not (torch.equal(a.rot6d, b.rot6d) and torch.equal(a.transl, b.transl)):
        return False
    if has_hands(a) != has_hands(b):
        return False
    return not has_hands(a) or torch.equal(a.hand6d, b.hand6d)


class SeriesCache:
    """The series of one catalogue_set call. context(root): {'root', 'kind', 'skip', 'seed', 'quiet' (SERIES_QUIET:
    the parts its layer leaves still, make_layer), and for a layered series 'resting', 'labels' (support_labels of the
    root body)}, the root body built once (unwrapped) when a member
    of its series is first built. layered(clip): (raw body, layered body, record) of a clip, solved once: the root by
    solve() (its own fades and guards), a member by transfer() from its direct level 0's, itself layered here first;
    the clips that are some clip's level 0 are kept (REF_CACHE, the least recently used dropped and rebuilt on
    demand)."""

    def __init__(self, by_id):
        self.by_id = by_id
        self.ctx = {}
        self.refs = OrderedDict()
        self._pending = {}

    def context(self, root_id, root_body=None, root_truth=None):
        if root_id not in self.ctx:
            spec = self.by_id[root_id]
            c = {"root": root_id, "kind": posture_kind(spec), "skip": skip_reason(spec), "seed": series_seed(root_id),
                 "quiet": tuple(SERIES_QUIET.get(root_id, ()))}
            if c["skip"] is None:
                if root_body is None:
                    root_body, root_truth = _raw_build(spec)()
                if c["kind"] == "walking":                    # every clip its own solve on its still spans; the
                    c["labels"] = None                        # hands hang or swing: no finger drift
                    c["resting"] = {"left": False, "right": False}
                else:
                    boxes = self._boxes(spec, root_body)
                    c["labels"] = support_labels(root_body, boxes, support_joints(c["kind"]))
                    c["resting"] = resting_hands(root_body, boxes, c["kind"], labels=c["labels"])
                self._pending[root_id] = (root_body, root_truth)
            self.ctx[root_id] = c
        return self.ctx[root_id]

    def layered(self, clip_id, body=None, truth=None):
        hit = self.refs.get(clip_id)
        if hit is not None and (body is None or _same_body(body, hit[0])):
            self.refs.move_to_end(clip_id)
            return hit
        spec = self.by_id[clip_id]
        root = series_root(clip_id, self.by_id)
        if body is None and clip_id in self._pending:
            body, truth = self._pending[clip_id]
        if body is None:
            body, truth = _raw_build(spec)()
        c = self.context(root, body if root == clip_id else None, truth if root == clip_id else None)
        self._pending.pop(clip_id, None)
        if c["skip"] is not None:
            raise ValueError(f"{clip_id}: its series {root} takes no layer ({c['skip']})")
        layer = make_layer(c["seed"], c["kind"], float(body.rate), quiet=c["quiet"])
        kw = dict(scene=spec.scene, table=table_of(truth), resting=c["resting"], kind=c["kind"])
        if c["kind"] == "walking":
            # physio-v7 (M3a): the layer on the clip's own still spans only, crossfaded (still_weight), the walk
            # unlayered bit for bit; each clip its own solve (a level 0's deltas would lay its stance's sway on a
            # member's steps), the rows a member holds equal to its level 0 where both weigh alike taken from the
            # level 0's output (_held_from)
            w = still_weight(body)
            out, rec = solve(body, layer, span=w, **kw)
            ref = None
            if clip_id != root:
                ref_raw, ref_out, _ = self.layered(spec.level0)
                out, rec = _held_from(body, out, rec, w, ref_raw, ref_out, still_weight(ref_raw), scene=spec.scene,
                                      table=kw["table"])
                ref = spec.level0
            rec["transfer"] = {"reference": None, "reference_knots": int(body.T), "beyond_knots": 0,
                               "own_solve": "walking", "level0": ref, **rec.pop("_held", {})}
        elif clip_id == root:
            if c["kind"] == "transfer":                       # the sway in the standing phase only (physio-v7)
                kw = dict(kw, span=stand_weight(body), span_keys=SPAN_KEYS["transfer"])
            out, rec = solve(body, layer, labels=c["labels"], **kw)
            rec["transfer"] = {"reference": None, "reference_knots": int(body.T), "beyond_knots": 0}
        else:
            ref_raw, ref_out, _ = self.layered(spec.level0)
            out, rec = transfer(body, ref_raw, ref_out, layer, own=lambda: solve(body, layer, **kw),
                                reference=spec.level0, **kw)
        res = (body, out, rec)
        if clip_id == root or any(s.level0 == clip_id for s in self.by_id.values()):
            self.refs[clip_id] = res
            while len(self.refs) > REF_CACHE:
                self.refs.popitem(last=False)
        return res

    @staticmethod
    def _boxes(spec, body):
        if spec.scene is None:
            return []
        from .viewer import scene_floor
        return scene_boxes(spec.scene, body, scene_floor(body, 5.0)[0])


def local_fade(bad, T, rate, prev=None):
    """(T,) 0 to 1: 0 at the knots bad and within FADE_HOLD_S of them, back to 1 over ANCHOR_RAMP_S on each side (a
    moving maximum then two moving means, a slow ramp: the fade itself adds no jump), the minimum with prev."""
    ind = np.zeros(T)
    ind[np.asarray(bad, dtype=np.int64)] = 1.0
    n = max(1, int(round(ANCHOR_RAMP_S * rate / 2)))
    m = int(round(FADE_HOLD_S * rate)) + n
    f = 1.0 - np.clip(_moving_mean(_moving_mean(_moving_max(ind, m), n), n), 0.0, 1.0)
    return f if prev is None else np.minimum(prev, f)


def table_of(truth):
    """The joint-limit table a clip's truth names (limits_ext.record in its params, or limits_ext.check in its
    validity): 'dystonia' (the SH2 extension, the dystonia operators), any other table of limits_ext.TABLES ('shrug',
    the healthy shrug with the coupling, where that module has it: physio-v5), else 'healthy'."""
    from . import limits_ext as LX
    known = tuple(getattr(LX, "TABLES", ("healthy", "dystonia")))
    t = truth or {}
    for d in (t.get("params") or {}, t.get("validity") or {}, t):
        if not isinstance(d, dict):
            continue
        v = d.get("limits_table")
        name = v.get("table") if isinstance(v, dict) else v
        if name in known and name != "healthy":
            return name
    return "healthy"


class LayeredBuild:
    """The build of a ClipSpec with the series' background (wrap_specs): calls the clip's own build, then the
    series' layer (SeriesCache.layered: the root's solve, a member's transfer from its level 0); .func is the clip's
    own build function (the module that builds the clip, for the code records of the exports), .raw the unwrapped
    build, .last_times {'clip_s', 'layer_s'} the wall times of the last call (kept out of the truth, which stays
    deterministic; the layer's time includes a level 0 built and layered for the first time)."""

    def __init__(self, raw, clip_id, root_id, cache, scene):
        self.raw, self.clip_id, self.root_id, self.cache, self.scene = raw, clip_id, root_id, cache, scene
        self.func = getattr(raw, "func", raw)
        self.last_times = None

    def __call__(self):
        import time
        t0 = time.perf_counter()
        body, truth = self.raw()
        return self.with_background(body, truth, t0)

    def with_background(self, body, truth, t0=None):
        """(Body, truth): the series' layer on a body the clip's own build gave (body, truth = self.raw()), so that
        a caller holding the raw body does not build it twice."""
        import copy
        import time
        t1 = time.perf_counter()
        t0 = t1 if t0 is None else t0
        is_root = self.root_id == self.clip_id
        c = self.cache.context(self.root_id, body if is_root else None, truth if is_root else None)
        truth = dict(truth) if truth is not None else {}
        kind_own = posture_kind(self.cache.by_id[self.clip_id])
        if c["skip"] is not None:
            truth["background"] = {"version": VERSION, "series": self.root_id, "skipped": True, "reason": c["skip"]}
            self.last_times = {"clip_s": t1 - t0, "layer_s": 0.0}
            return body, truth
        _, out, rec = self.cache.layered(self.clip_id, body, truth)
        rec = copy.deepcopy(rec)
        root_rec = rec if is_root else self.cache.layered(self.root_id)[2]
        layer = make_layer(c["seed"], c["kind"], float(body.rate), quiet=c["quiet"])
        amp = amplitudes(c["kind"])
        if "trunk" in amp:
            amp["trunk"]["forward_offset_mean_deg"] = float(np.mean(layer["lean_ap_deg"]))
        rec.update({"series": self.root_id, "seed": c["seed"], "kind": c["kind"], "skipped": False,
                    "quiet": list(c["quiet"]),
                    "kind_of_clip": kind_own, "amplitudes": amp,
                    "components": list(root_rec.get("components", [])),
                    "resting_hands": dict(c["resting"]),
                    "series_deviations": sorted(root_rec.get("deviations", {})),
                    "fingers_lift_only": list(root_rec.get("fingers_lift_only", [])),
                    "sources": sorted({s for a in amp.values() for s in a["sources"]})})
        truth["background"] = rec
        v = rec["validity"]
        truth["validity"] = dict(truth.get("validity") or {})
        truth["validity"]["background"] = {"ok": v["ok"], "failed": list(v["failed"]), "version": VERSION,
                                           "limits_table": v["limits_table"]}
        check = (truth.get("params") or {}).get("table_check")
        if check is not None:                 # a writing clip: its table check read again on the layered body
            from .postures import table_report_of
            truth["validity"]["table"] = table_report_of(out, check)
        relayered = layer_checks_of(out, truth)
        if relayered:                         # the scene and fold checks read again on the layered body (Phase D)
            truth["validity"].update(relayered)
            rec["checks_read_on_layer"] = sorted(relayered)
        self.last_times = {"clip_s": t1 - t0, "layer_s": time.perf_counter() - t1}
        return out, truth


def layer_checks_of(body, truth):
    """{validity entry: report}: the checks of a clip's own build that read the body against its scene or itself,
    read again on its layered body (Phase D of docs/PLAN_2026-11-12.md: the layered truth copied 'fold',
    'hand_thigh', 'hand_trunk' and 'backrest' from the raw body, and only the writing clips' 'table' was read
    again): the paroxysmal excerpts, the paroxysmal kinesigenic clips and the geste antagoniste through their
    params['layer_checks'] (postures.scene_checks_record, postures.scene_checks_of), the arising level 0 through its
    phases (operator 'arise' with onset_s and end_s: motions.arise_checks_of); {} for any other clip."""
    p = truth.get("params") or {}
    rec = p.get("layer_checks")
    if rec is not None:
        from .postures import scene_checks_of
        return scene_checks_of(body, rec)
    if truth.get("operator") == "arise" and "onset_s" in p and "end_s" in p:
        from .motions import arise_checks_of, arise_checks_record
        return arise_checks_of(body, arise_checks_record(p))
    return {}


def wrap_specs(specs):
    """[ClipSpec]: every spec with its build wrapped (LayeredBuild); one SeriesCache for the list."""
    by_id = {s.clip_id: s for s in specs}
    cache = SeriesCache(by_id)
    out = []
    for s in specs:
        root = series_root(s.clip_id, by_id)
        out.append(replace(s, build=LayeredBuild(s.build, s.clip_id, root, cache, s.scene)))
    cache.by_id = {s.clip_id: s for s in out}
    return out
