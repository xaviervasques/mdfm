"""The tremor family and the reach-based limb signs of the symptom catalogue (catalogue_v1): specs() lists the clips
(exam.ClipSpec, since 'catalogue_v1'), each tied to a scale item with its grade, the parameters of the grade printed
in params_text, and a level-0 partner on the same clock (a clip of this list or an exam clip id).

Symptoms (sources in SOURCES, a superset of grades.SOURCES and reach.SOURCES; amplitudes are peak-to-peak
displacements in cm, decision D4 of the draft, read here for every amplitude scale):
- kinetic tremor, MDS-UPDRS 3.16 grades 1 (below and above the estimator floor) to 4, on the reach (reach.py);
- essential tremor: postural (E2, both arms, TETRAS item 4a, 1.5 to 3.5: the hand flexing and extending at the
  wrist, the shoulder joining from 3 cm, read at the middle fingertip) and kinetic on the reach (TETRAS 4c); head
  tremor yes-yes and no-no over the neck and the head (TETRAS item 1, grades 1 to 4 seated, grades 2 and 3
  standing);
- cerebellar signs on the reach: intention tremor rising near the target (SARA item 6, grades 1 to 3, 3.5 Hz, the
  shoulder girdle swinging the hand across the path, read at the index fingertip), dysmetria (SARA item 5 bands read on the reach end point,
  past and short of the target), decomposition of the movement (ICARS finger-to-nose: 2 and more than 2 phases, at
  a slower pace), and one combined clip;
- Holmes tremor: slow (3 Hz), proximal and distal, at rest (E1), in posture (E2) and in action (E3), MDS-UPDRS
  3.17, 3.15 and 3.16;
- dystonic tremor: jerky, irregular in period and amplitude, of the arm (E2, MDS-UPDRS 3.15: on a dystonic wrist
  posture, the fast phases toward the pull, nearly gone while the wrist is let into the pull, the null point) and
  of the head with a dystonic rotation (TETRAS item 1, its level 0 the rotation alone; the fast phases away from
  the pull);
- wing-beating tremor: proximal, the arms abducted with the elbows flexed and the hands near each other at nose
  level (TETRAS item 4b posture, draft segment E2b);
- orthostatic tremor: 15.7 Hz in the legs while standing, palpable rather than visible: what the video shows is the
  wide base and the weight shifts (catalogue T17);
- enhanced physiological tremor: 10 Hz, small, both hands at the wrist (TETRAS 1 and 1.5);
- functional tremor: variable frequency and amplitude, entrained by the other hand's tapping at 3 Hz (with a
  residual phase jitter), paused just after the onset of the other arm's ballistic movement (MDS-UPDRS 3.15
  amplitude);
- parkinsonian pill-rolling of the fingers at rest (hand joints): the thumb pad on the index pad, rubbing along it,
  alone and with forearm pronation-supination (MDS-UPDRS 3.17 amplitude at the index fingertip).
Re-emergent tremor is not repeated: exam.py has it (tremor_E2_reemergent_g2, tremor_E2_raise_reemergent_g2).

Operator (oscillate): an angle a_i(t) = w_i theta(t) e(t) y(t) + c_i(t) composed with the base rotation of each
drive joint about its axis (frame 'parent': a hinge fixed in the proximal segment; 'child': about the segment's own
axis), y(t) the waveform in [-1, 1] (a sine, or the jerky waveform of dystonic tremor), e(t) the envelope in
[0, 1]. theta(t) is calibrated at every knot so that the measured point (a wrist joint, the middle fingertip or the
nose, rigid on their joint) moves by the requested peak-to-peak chord between the two extremes of the oscillation
at full envelope (forward kinematics at both extremes, a few fixed-point passes). The joint-limit table bounds it as
tremor.py does: c_i(t), the centre shift, moves the oscillation toward the side with room where a peak would leave
the table (rooms on a 0.5 degree grid at every distinct base rotation, smoothed by a moving maximum then a moving
mean so that the shift covers the need), and a peak that does not fit the two rooms together raises. On the reach
the face is a room too (FaceRoom): near the nose the centre moves the hand away from the head until the inner
extreme of the oscillation keeps FACE_GAP_CM from the head surface, so that the finger taps the face without
entering it. A grade 0 is the level-0 partner.

Waveforms: sine (frequency possibly varying in time: the phase integrates it); jerky: per cycle a fast rise over
RISE_SHARE of the cycle and a slow return (raised cosines, C1), periods drawn lognormal (PERIOD_CV), cycle
amplitudes A_MIN + (1 - A_MIN) u with u drawn Beta(2, 2) and interpolated smoothly, and one cycle at 1 in every
window of PEAK_EVERY_S, so that the maximal amplitude, the level, recurs (the scales rate the maximal amplitude:
TETRAS, MDS-UPDRS 3.15 to 3.17); fast_sign sets the direction of the fast phase. No variability value was sourced:
the jitter and modulation are module values (SOURCES['module']).

Levels inside a band (band_level): the geometric centre of a closed band; an open lower band (0, hi) at hi divided
by the ratio of the next band's centre to its lower edge; an open upper band (lo, inf) at lo times the ratio of the
previous band's centre to its lower edge (the log placement of grades.tremor_level_cm, Elble et al. 2006); TETRAS
item 4 grade 1 ('barely visible') one more such ratio below grade 1.5. MDS-UPDRS levels are grades.tremor_level_cm,
except the open top band of a limb tremor (grade 4), placed just above its lower edge (TOP_LEVEL_CM, module: 17.3 cm
of forearm swing read as a caricature in the clinical review). Frequencies at the top grades of a limb tremor fall
to TOP_HZ (module values, limb_hz).

Re-measurement (readback): the displacement of the measured point against the clip's own centre (the level-0
partner, or the rebuilt base of a reach or of the dystonic arm, with the drives' centre shifts applied: the clip
without its oscillation; high-passed when there is none), its dominant frequency (measure.dominant_frequency, the
spectral peak) and its peak-to-peak chord along the first principal axis in sliding windows of 1.5 periods (largest
and median); the dysmetria as the distance of the movement's first rest point from the target
(reach.endpoint_error_cm, on the centre when a tremor rides on it), the decomposition as the count of fingertip speed
runs per movement (reach.speed_runs); the fingertip, the middle fingertip and the nose read on the skinned mesh.

Gap wave (2026-11-02, catalogue rows not shown before; constants TERMINAL to ORTHO_MYO, sources GAP_SOURCES; every
level, envelope, phase and amplitude without a source is a module value, said where it is set):
- T04, essential intention tremor with a terminal crescendo on the finger-nose-finger (TETRAS 4c grades 2 and 3): the
  kinetic tremor of T03 (elbow) whose envelope rises with the level 0's fingertip nearing the nose or the examiner's
  finger (terminal_envelope), the level read at the index fingertip;
- T05, simple kinetic tremor during a non-target-directed movement: on the exam's level-0 pronation-supination
  (brady_E4_ps_level0), the hand flexing and extending at the wrist (the 'postural' drives of T01), read at the middle
  fingertip, and blind with the task band removed (task_removed_readback); the TETRAS upper-limb bands (grades 2 and
  3; no item of its own, the FTM action item waits for its chapter, catalogue SC6);
- T08, mixed head tremor (TETRAS 1 grades 2 and 3): no-no at the level and yes-yes at MIXED_HEAD['pitch_share'] of
  it, MIXED_HEAD['phase_deg'] behind (the nose runs an ellipse);
- T10, standing tremor of the legs and trunk (TETRAS 9 grades 1 to 3, words only): the TETRAS standing posture, the
  knees flexing about it with the feet planted, the trunk in phase; at grade 3 ('impairs stance') weight shifts and a
  slow forward and backward sway of the body over the planted feet that narrows the extrapolated centre of mass's
  margins (STANDING_LEVEL, STANDING_SWAY, module levels; clip ids standing_tremor_*);
- T13, essential-type rest tremor (essential tremor plus; MDS-UPDRS 3.17 grades 1 and 2): both hands at rest flexing
  and extending at the wrist at the essential frequencies, read at the middle fingertips; the hands of 'sit' are held
  level beyond the chair arms, not hanging or on the thighs (a supported rest posture is a base.py change);
- T16, isometric tremor of the clenched fist (presence; ids isometric_tremor_*): seated, the left forearm on the chair
  arm ('sit', not held against gravity), the hand open, then closing into the fist of handtasks.FIST, the tremor
  ramping in only once the fist is clenched (the wrist flexing and extending), read at the front of the fist; the
  squeeze and its force are not shown;
- P29, rest tremor of the leg and foot seated (MDS-UPDRS 3.17 grades 1 to 3, lower extremity): on its own level 0
  (tremor_E1_leg_rest_g0: 'sit' rests the foot on its ball, the heel above the floor; the level 0 lowers the heel onto
  the floor, heel_down_deg), the forefoot rising and falling about the planted heel (contact.plant_ankle), read at
  the big-toe vertex, the heel and the knee nearly still;
- P30, rest tremor of the arm while walking (MDS-UPDRS 3.17 grades 2 and 3): the elbow tremor on the walk of
  gait_E8_parkinsonian_g1 (its own level 0 here, tremor_E8_walk_g0, the same body), read at the wrist in the upper
  spine's frame, and blind with the swing band removed (swing_removed_readback);
- M07, orthostatic myoclonus (presence; id orthostatic_myoclonus_*): brief knee-extension twitches at irregular
  intervals (whole knots inside 3 to 7 Hz) on the knees-unlocked stance of the orthostatic level 0, the feet planted.
  Its family is 'tremor':
  tests/test_synth_tremors.py holds this module to the families tremor and reach.
The clips graded on a scale without cm bands (TETRAS 9) or on presence (T16, M07) carry ids outside 'tremor_*', which
tests/test_synth_tremors.py reads as cm-band oscillations; tests/test_synth_tremors_more.py reads them back.
"""
import functools
import math
from typing import NamedTuple

import numpy as np
import torch

from ..body.rotations import axis_angle_to_matrix, matrix_to_axis_angle, matrix_to_rot6d
from ..body.joints import fk
from . import bradykinesia as BK
from . import grades as G
from . import hands as H
from . import measure as M
from . import reach as RC
from .base import posture_rotations, static_body
from .body import F64, Body, compose, axis_rotation, contact_labels, joint_limits, validity, LIMIT_TOL_DEG
from .common import as_rng, envelope, truth_record
from .exam import DUR, SEED, TREMOR_HZ, ClipSpec

SINCE = "catalogue_v1"
WRIST = {"left": 20, "right": 21}
ELBOW = {"left": 18, "right": 19}
SHOULDER = {"left": 16, "right": 17}
COLLAR = {"left": 13, "right": 14}
NECK = 12
HEAD = 15
INF = math.inf
SOURCES = dict(G.SOURCES, **RC.SOURCES, **{
    "tetras2012": "Elble R et al. 2012, Mov Disord 27(12): 1567-1569, doi 10.1002/mds.25162 (PMC4157921, Table 1): "
                  "TETRAS head tremor 1 under 0.5 cm, 2 0.5 to under 2.5, 3 2.5 to 5, 4 above 5; upper limb 1 barely "
                  "visible, 1.5 visible under 1 cm, 2 1 to under 3, 2.5 3 to under 5, 3 5 to under 10, 3.5 10 to "
                  "under 20, 4 above 20 cm; the rater estimates the maximum amplitude; item 4 postures: arms "
                  "forward, lateral 'wing beating' with the elbows flexed, finger-nose-finger (also Hernandez AB et "
                  "al. 2024, Tremor Other Hyperkinet Mov 14: 14, doi 10.5334/tohm.874, Table 2)",
    "icars1997": "Trouillas P et al. 1997, J Neurol Sci 145: 205-211 (ICARS), finger-to-nose items read in a secondary "
                 "table (medicalalgorithms.com), not in the paper: decomposition and dysmetria 1 oscillating without "
                 "decomposition, 2 segmented in 2 phases and/or moderate dysmetria, 3 segmented in more than 2 phases "
                 "and/or considerable dysmetria, 4 dysmetria prevents reaching the nose; intention tremor 2 under 10 "
                 "cm, 3 10 to 40 cm, 4 above 40 cm",
    "buijink2012": "Buijink AWG et al. 2012, Front Neurol 3: 146, doi 10.3389/fneur.2012.00146 (full text; Table 2 "
                   "adapted from Deuschl et al. 1998): enhanced physiological 5 to 12 Hz (text: 8 to 12 Hz, low "
                   "amplitude, postural, bilateral); essential 4 to 12; Parkinson rest 4 to 9; dystonic 4 to 10, "
                   "irregular amplitude and frequency; cerebellar 2 to 5, intention tremor below 5 Hz; Holmes 2 to 5, "
                   "rest below 4.5 Hz increasing with posture and intention, often not as rhythmic; orthostatic above "
                   "13 Hz (13 to 18 Hz EMG); psychogenic 4 to 12, entrainment, inconsistent over time; wing-beating: "
                   "a proximal tremor of high amplitude, best seen with the arms stretched",
    "deuschl1998": "Deuschl G, Bain P, Brin M 1998, Mov Disord 13 Suppl 3: 2-23, doi 10.1002/mds.870131303 "
                   "(consensus statement; read through buijink2012)",
    "hassan2016": "Hassan A et al. 2016, Neurology 86(5): 458-464, doi 10.1212/WNL.0000000000002328 (abstract): 184 "
                  "patients with orthostatic tremor, mean frequency 15.7 Hz, range 12.5 to 20 Hz",
    "schwingenschuh2011": "Schwingenschuh P et al. 2011, Mov Disord 26(14): 2509-2515, doi 10.1002/mds.23922 "
                          "(abstract and the author's EAN 2019 teaching slides): entrainment or frequency change "
                          "while tapping at 1, 3 and 5 Hz with the other hand, pause of the tremor during a "
                          "contralateral ballistic movement",
    "elble2000": "Elble RJ 2000, Neurology 55(10): 1547-1551, doi 10.1212/wnl.55.10.1547 (abstract, as read in "
                 "docs/proposals/SYMPTOM_CATALOGUE_DRAFT.md, T01): essential tremor frequency f = 9.94 - 0.061 x age "
                 "Hz, falling 0.06 to 0.08 Hz per year",
    "elble2017": "Elble RJ et al. 2017, Mov Disord Clin Pract 4(2): 205-211, doi 10.1002/mdc3.12379 (as read in "
                 "docs/proposals/SYMPTOM_CATALOGUE_DRAFT.md, T06): head tremor 2.3 to 6.4 Hz, mean 4.35",
    "benitoleon2016": "Benito-Leon J, Domingo-Santos A 2016, Tremor Other Hyperkinet Mov 6: 411, doi "
                      "10.5334/tohm.324 (as read in docs/proposals/SYMPTOM_CATALOGUE_DRAFT.md, T17): orthostatic "
                      "tremor palpable more than visible; wide base, toe clawing, weight shifting, walking on the "
                      "spot",
    "bhatia2018": "Bhatia KP et al. 2018, Mov Disord 33(1): 75-87, doi 10.1002/mds.27121 (consensus statement on the "
                  "classification of tremors, as read in docs/proposals/SYMPTOM_CATALOGUE_DRAFT.md, P27 and T18): "
                  "dystonic tremor relieved at a null point, worse in some postures; the classes read for P27, T05, "
                  "T13 and T16: simple kinetic tremor during a non-target-directed movement; essential tremor with a "
                  "rest tremor counts as essential tremor plus; isometric tremor while making a fist or holding "
                  "against resistance; classic parkinsonian rest tremor typically 4 to 7 Hz (grades.SOURCES)",
})

# ---------------------------------------------------------------- scales: bands in cm, peak to peak (D4)
SCALES = {
    "MDS-UPDRS": {1: (0.0, 1.0), 2: (1.0, 3.0), 3: (3.0, 10.0), 4: (10.0, INF)},          # grades.TREMOR_BANDS_CM
    "TETRAS head": {1: (0.0, 0.5), 2: (0.5, 2.5), 3: (2.5, 5.0), 4: (5.0, INF)},           # SOURCES['tetras2012']
    "TETRAS upper limb": {1.0: (0.0, 1.0), 1.5: (0.0, 1.0), 2.0: (1.0, 3.0), 2.5: (3.0, 5.0), 3.0: (5.0, 10.0),
                          3.5: (10.0, 20.0), 4.0: (20.0, INF)},                            # SOURCES['tetras2012']
    "SARA 6": {1: (0.0, 2.0), 2: (2.0, 5.0), 3: (5.0, INF)},                                # SOURCES['sara2006']
    "SARA 5": {1: (0.0, 5.0), 2: (5.0, 15.0), 3: (15.0, INF)},                              # SOURCES['sara2006']
}
SCALE_SOURCE = {"MDS-UPDRS": "mds_updrs", "TETRAS head": "tetras2012", "TETRAS upper limb": "tetras2012",
                "SARA 6": "sara2006", "SARA 5": "sara2006"}
HZ = {"kinetic": 6.0,          # [module] the postural tremor frequency of exam.TREMOR_HZ, applied to 3.16
      "essential": 6.0,        # inside 4 to 12 Hz (buijink2012), the exam's postural frequency
      "essential_right": 6.3,  # [module] the right arm's own frequency (per limb, design section 10)
      "essential_head": 4.5,   # [module] inside 4 to 12 Hz (buijink2012), toward its low end for the head
      "essential_head_g4": 3.5,  # [module] inside 2.3 to 6.4 Hz (elble2017), slower for the largest head tremor
      "intention": 3.5,        # inside 2 to 5 Hz, below 5 (buijink2012)
      "holmes": 3.0,           # inside 2 to 5 Hz, below 4.5 at rest (buijink2012)
      "dystonic": 5.0,         # [module] mean of the jittered periods, inside 4 to 10 Hz (buijink2012)
      "wing": 3.0,             # [module] low frequency (buijink2012 gives no number)
      "orthostatic": 15.7,     # hassan2016 mean
      "physiological": 10.0,   # inside 8 to 12 Hz (buijink2012 text)
      "functional": 5.5,       # [module] inside 4 to 12 Hz (buijink2012)
      "tapping": 3.0,          # schwingenschuh2011 (one of 1, 3 and 5 Hz)
      "pill_rolling": 5.0}     # inside 4 to 9 Hz (buijink2012), the exam's rest frequency
# [module] the limb tremor frequency at the top grades (MDS-UPDRS 3, 4; TETRAS upper limb 3, 3.5 and 4): large
# tremors are slower (the essential tremor frequency falls with age and duration, elble2000: 5.1 Hz at 80 years);
# no source gives a frequency per grade. 6 Hz at 17 cm (12.5 g at the wrist) read as a caricature in the review.
TOP_HZ = {3: 5.0, 4: 4.5}
TOP_LEVEL_CM = {"MDS-UPDRS": 12.0}   # [module] the open top band of a limb tremor, just above its lower edge
RAMP_S = 0.5                   # [module] envelope ramps (tremor.RAMP_S)
INTENTION_FLOOR = 0.15         # [module] the intention tremor's share of its level in mid-movement
INTENTION_DEPART = 0.25        # [module] the intention tremor fades over this share of the path leaving a target
HOLMES_FLOOR = 0.7             # [module] Holmes tremor away from the targets (rest and posture components)
RISE_SHARE = 0.3               # [module] the jerky waveform's fast phase (share of a cycle)
PERIOD_CV = {"dystonic": 0.25, "holmes": 0.18, "wing": 0.12}    # [module] lognormal sigma of the periods
A_MIN = {"dystonic": 0.55, "holmes": 0.6, "wing": 0.75}          # [module] smallest cycle amplitude share
PEAK_EVERY_S = (4.0, 5.0)      # [module] one cycle at the full level in every window of this length (uniform)
FUNCTIONAL_CV = 0.12           # [module] slow drift of the functional tremor's own frequency (lognormal sigma)
FUNCTIONAL_AMP_MIN = 0.65      # [module] slow amplitude variation of the functional tremor, share of the level
FUNCTIONAL_JITTER_RAD = 0.5    # [module] residual phase jitter of the entrained tremor (sigma, knots 1 s apart)
OT_DEPTH_CM = 1.5              # [module] the knees unlocked: the pelvis this much below the straight stance
OT_PP_CM = 0.25                # [module] the pelvis's vertical peak-to-peak oscillation (barely visible)
OT_STANCE_CM = 28.0            # [module] ankle-to-ankle width of the wide base (benitoleon2016: wide base)
OT_SHIFT_CM = 2.5              # [module] lateral pelvis excursion of a weight shift (benitoleon2016: weight shifting)
OT_SHIFT_S = 0.8               # [module] duration of one weight shift
OT_SHIFT_FIRST_S = 4.0         # [module] the first weight shift after this much standing
OT_SHIFT_EVERY_S = (2.5, 3.5)  # [module] interval between the weight shifts (uniform)
TAP_DEG = 25.0                 # [module] wrist extension of a tap of the other hand
# [module] the 4b posture (draft segment E2b: shoulders abducted, elbows flexed, hands near each other at nose
# level), solved by wing_rotations: the middle fingertips WING['tip_gap_cm'] apart, WING['tip_forward_cm'] in front
# of the nose and at its height, the upper arms WING['elevation_deg'] above the horizontal
WING = {"tip_gap_cm": 10.0, "tip_forward_cm": 22.0, "elevation_deg": 0.0}
# [module] postural tremor of the outstretched hand (catalogue T01): wrist flexion-extension, the shoulder's flexion
# joining from 3 cm (TETRAS 2.5) with these angle weights (about a quarter of the fingertip's chord from 3 cm, half
# from 5 cm)
POSTURAL_SHOULDER = ((3.0, 0.1), (5.0, 0.25))
FACE_GAP_CM = 0.3              # [module] the hand's least distance from the head surface at the tremor's inner extreme
SHIFT_STEP_DEG = 0.5


# ---------------------------------------------------------------- levels
def band_level(scale, grade):
    """The level (cm, peak to peak) of a grade of SCALES[scale] (module docstring)."""
    if scale == "MDS-UPDRS":
        raise ValueError("MDS-UPDRS levels come from grades.tremor_level_cm")
    bands = SCALES[scale]
    keys = sorted(bands)
    if scale == "TETRAS upper limb" and float(grade) == 1.0:
        lo2, hi2 = bands[2.0]
        return band_level(scale, 1.5) / (math.sqrt(lo2 * hi2) / lo2)
    lo, hi = bands[grade]
    i = keys.index(grade)
    if lo > 0 and hi < INF:
        return math.sqrt(lo * hi)
    if lo == 0:
        nxt = [k for k in keys[i + 1:] if bands[k][0] > 0][0]
        a, b = bands[nxt]
        return hi / (math.sqrt(a * b) / a)
    a, b = bands[keys[i - 1]]
    return lo * (math.sqrt(a * b) / a)


def grade_of(scale, cm):
    """The grade of SCALES[scale] whose band holds cm (TETRAS upper limb below 1 cm: 1.5, 'visible')."""
    if cm <= 0:
        return 0
    if scale == "MDS-UPDRS":
        return G.tremor_grade_of_cm(cm)
    for g in sorted(SCALES[scale]):
        if scale == "TETRAS upper limb" and g == 1.0:
            continue
        lo, hi = SCALES[scale][g]
        if lo <= cm < hi:
            return g
    return max(SCALES[scale])


def in_band(scale, grade, cm):
    lo, hi = SCALES[scale][grade]
    return lo < cm < hi


def level_cm(scale, grade, grade1="above_floor"):
    """The level (cm) of a grade: band_level, grades.tremor_level_cm for MDS-UPDRS, TOP_LEVEL_CM for its open top
    band (module docstring)."""
    if scale == "MDS-UPDRS":
        return TOP_LEVEL_CM["MDS-UPDRS"] if int(grade) == 4 else G.tremor_level_cm(grade, grade1)
    return band_level(scale, grade)


def limb_hz(scale, grade, hz):
    """The frequency (Hz) of a limb tremor at a grade: hz, lowered to TOP_HZ at MDS-UPDRS 3 and 4 and at TETRAS upper
    limb 3 and above (module values, see TOP_HZ)."""
    g = float(grade or 0)
    top = None
    if scale == "MDS-UPDRS" and g >= 3:
        top = TOP_HZ[3] if g < 4 else TOP_HZ[4]
    if scale == "TETRAS upper limb" and g >= 3.0:
        top = TOP_HZ[3] if g < 3.5 else TOP_HZ[4]
    return float(hz) if top is None else float(min(hz, top))


# ---------------------------------------------------------------- the oscillator
class Drive(NamedTuple):
    joint: int
    axis: object          # 'x', 'y', 'z' or a unit 3-vector, in the frame below
    frame: str            # 'parent' or 'child' (body.compose)
    weight: float = 1.0


def drives_of(name, side="left", cm=None):
    """The drives of a named tremor pattern: 'elbow' (the hinge fixed in the humerus, tremor.py), 'shoulder_elbow'
    (proximal and distal: the shoulder about the lateral axis of its parent and the elbow hinge, equal angles),
    'shoulder_vertical' (the shoulder about the vertical axis of its parent: the arm swings sideways),
    'collar_vertical' (the shoulder girdle about the vertical axis of the spine: the whole arm swings sideways, with
    room on both sides at the nose and at the examiner's finger), 'shoulder_abduction' (about the forward axis of
    its parent), 'wing' (the shoulder about the horizontal axis normal to the line from the shoulder to the middle
    of the elbow and the wrist in the wing posture, wing_axis: the elbow and the hand flap up and down together),
    'wrist' (flexion-extension of the hand about its own z axis, positive lifting the fingers), 'postural' (the
    outstretched hand: 'wrist' with the shoulder's flexion joining by POSTURAL_SHOULDER for a chord of cm, both
    lifting the fingers when positive), 'head_pitch' (yes-yes, the neck and the head about their lateral axes, equal
    angles), 'head_yaw' (no-no, about their vertical axes)."""
    up = 1.0 if side == "left" else -1.0          # the sign of a wrist z rotation that lifts the fingers
    if name == "elbow":
        return [Drive(ELBOW[side], "y", "parent")]
    if name == "shoulder_elbow":
        return [Drive(SHOULDER[side], "x", "parent", 0.5), Drive(ELBOW[side], "y", "parent", 0.5)]
    if name == "shoulder_vertical":
        return [Drive(SHOULDER[side], "y", "parent")]
    if name == "collar_vertical":
        return [Drive(COLLAR[side], "y", "parent")]
    if name == "shoulder_abduction":
        return [Drive(SHOULDER[side], "z", "parent")]
    if name == "wing":
        return [Drive(SHOULDER[side], tuple(wing_axis(side)), "parent")]
    if name == "wrist":
        return [Drive(WRIST[side], "z", "child", up)]
    if name == "postural":
        sh = postural_shoulder_weight(cm)
        out = [Drive(WRIST[side], "z", "child", up * (1.0 - sh))]
        return out + ([Drive(SHOULDER[side], "x", "parent", -sh)] if sh else [])
    if name == "head_pitch":
        return [Drive(NECK, "x", "child", 0.5), Drive(HEAD, "x", "child", 0.5)]
    if name == "head_yaw":
        return [Drive(NECK, "y", "child", 0.5), Drive(HEAD, "y", "child", 0.5)]
    raise ValueError(f"drive {name!r}")


def postural_shoulder_weight(cm):
    """The shoulder's angle weight in the 'postural' drives for a chord of cm (POSTURAL_SHOULDER)."""
    w = 0.0
    for lo, wt in POSTURAL_SHOULDER:
        if cm is not None and cm >= lo:
            w = wt
    return w


@functools.lru_cache(maxsize=1)
def _nose_offset():
    J52, v, _ = H._rest()
    return torch.as_tensor(v[RC.nose_vertex()] - J52[HEAD], dtype=F64)


@functools.lru_cache(maxsize=2)
def _middle_tip_offset(side):
    J52, v, _ = H._rest()
    return torch.as_tensor(v[H.tip_vertices()[side]["middle"]] - J52[WRIST[side]], dtype=F64)


@functools.lru_cache(maxsize=2)
def _index_tip_offset(side):
    b, _ = reach_base(side)
    J, Gl = b.fk()
    tip = RC.fingertip(b, side, knots=[0])[0]
    return torch.as_tensor((tip - J[0, WRIST[side]].numpy()) @ Gl[0, WRIST[side]].numpy(), dtype=F64)


def point_spec(name, side="left"):
    """('joint', j) for 'wrist', ('rigid', 15, offset) for 'nose' (the nose tip vertex rigid on the head),
    ('rigid', wrist, offset) for 'middle_tip' (the middle fingertip vertex of the hand at rest, rigid on the wrist:
    the point TETRAS item 4 reads, catalogue T01) and for 'index_tip' (the index fingertip of the reach's pointing
    hand, rigid on the wrist: the finger the nose-finger test watches); 'fist_knuckle' (the front of the held fist,
    _fist_knuckle, rigid on the wrist); 'wrist_trunk' the wrist joint (read in the upper spine's frame);
    'middle_tip_trunk' the middle fingertip rigid on the wrist (read in the upper spine's frame: the walking rest
    tremor, levelled at the hand, wave SH1d part B)."""
    if name == "wrist":
        return ("joint", WRIST[side])
    if name == "nose":
        return ("rigid", HEAD, _nose_offset())
    if name == "middle_tip":
        return ("rigid", WRIST[side], _middle_tip_offset(side))
    if name == "index_tip":
        return ("rigid", WRIST[side], _index_tip_offset(side))
    if name == "fist_knuckle":
        return ("rigid", WRIST[side], _fist_knuckle(side)[1])
    if name == "wrist_trunk":
        return ("joint", WRIST[side])
    if name == "middle_tip_trunk":
        return ("rigid", WRIST[side], _middle_tip_offset(side))
    raise ValueError(f"point {name!r}")


def point_path(body, name, side="left"):
    """(T, 3) m: the measured point of a clip on the skinned mesh ('nose', 'middle_tip', 'index_tip', 'toe' the
    big-toe vertex, 'fist_knuckle'), the wrist joint or the middle fingertip in the frame of the upper spine
    ('wrist_trunk', 'middle_tip_trunk', relative to joint 9) or the wrist joint."""
    if name == "nose":
        return RC.nose_point(body)
    if name == "middle_tip":
        return H.body_points(body, [H.tip_vertices()[side]["middle"]])[:, 0]
    if name == "index_tip":
        return RC.fingertip(body, side)
    if name == "toe":
        return H.body_points(body, [toe_vertex(side)])[:, 0]
    if name == "fist_knuckle":
        return H.body_points(body, [_fist_knuckle(side)[0]])[:, 0]
    if name in ("wrist_trunk", "middle_tip_trunk"):  # the point in the frame of the upper spine (joint 9)
        J, Gl = body.fk()
        P = J[:, WRIST[side]].numpy() if name == "wrist_trunk" else             H.body_points(body, [H.tip_vertices()[side]["middle"]])[:, 0]
        return np.einsum("tji,tj->ti", Gl[:, 9].numpy(), P - J[:, 9].numpy())
    return body.joints()[:, WRIST[side]].numpy()


def _point(R, transl, rest, spec):
    J, Gl = fk(R, transl, rest)
    if spec[0] == "joint":
        return J[:, spec[1]]
    return J[:, spec[1]] + Gl[:, spec[1]] @ spec[2]


def _apply(R, drives, angles):
    out = R.clone()
    for d, a in zip(drives, angles):
        out[:, d.joint] = compose(out[:, d.joint], axis_rotation(d.axis, torch.as_tensor(a, dtype=F64)), d.frame)
    return out


def _rooms(Rj, joint, axis, frame, max_deg):
    """(plus (K,), minus (K,)) deg: the room the table (or the base's own excursion) leaves about axis for each base
    rotation Rj (K, 3, 3) of joint, on a SHIFT_STEP_DEG grid (floored)."""
    lim = joint_limits()
    c0 = torch.rad2deg(matrix_to_axis_angle(Rj)).numpy()
    lo = np.minimum(lim.lo[joint][None], c0) - LIMIT_TOL_DEG
    hi = np.maximum(lim.hi[joint][None], c0) + LIMIT_TOL_DEG
    grid = np.arange(SHIFT_STEP_DEG, max_deg + SHIFT_STEP_DEG / 2, SHIFT_STEP_DEG)
    out = []
    for sign in (1.0, -1.0):
        Roff = axis_rotation(axis, torch.as_tensor(sign * np.radians(grid)))
        Rn = compose(Rj[:, None], Roff[None], frame)
        c = torch.rad2deg(matrix_to_axis_angle(Rn)).numpy()
        ok = ((c >= lo[:, None]) & (c <= hi[:, None])).all(-1)
        first = np.where(ok.all(1), len(grid), np.argmin(ok, axis=1))
        out.append(np.where(first == 0, 0.0, grid[np.maximum(first - 1, 0)]) if len(grid) else np.zeros(len(Rj)))
    return out[0], out[1]


def _cover(x, n):
    """A smooth curve at least x (T,) >= 0 everywhere: a moving maximum then a moving mean, both of n samples."""
    if n <= 1 or not x.any():
        return x.copy()
    pad = np.pad(x, (n // 2, n - 1 - n // 2), mode="edge")
    mx = np.lib.stride_tricks.sliding_window_view(pad, n).max(1)
    pad2 = np.pad(mx, (n // 2, n - 1 - n // 2), mode="edge")
    return np.convolve(pad2, np.ones(n) / n, mode="valid")


def _shifts(R0, drives, peak_deg, rate, window_s):
    """[(T,) deg]: the centre shift of each drive (module docstring); raises when a peak does not fit."""
    out = []
    n = max(1, int(round(window_s * rate)))
    for d in drives:
        need = np.abs(d.weight) * peak_deg
        sel = need > 0
        c = np.zeros(len(need))
        if sel.any():
            r6 = matrix_to_rot6d(R0[:, d.joint])
            uniq, inv = torch.unique(r6[torch.as_tensor(sel)], dim=0, return_inverse=True)
            from ..body.rotations import rot6d_to_matrix
            mx = float(need.max()) * 2.0 + 1.0
            rp, rm = _rooms(rot6d_to_matrix(uniq), d.joint, d.axis, d.frame, min(180.0, mx))
            rp, rm = rp[inv.numpy()], rm[inv.numpy()]
            nd = need[sel]
            if (2 * nd > rp + rm + 1e-9).any():
                k = int(np.argmax(2 * nd - rp - rm))
                raise ValueError(f"joint {d.joint}: a tremor of {2 * nd[k]:.2f} deg peak to peak does not fit the "
                                 f"{rp[k] + rm[k]:.2f} deg the table leaves")
            req = np.where(nd > rp, rp - nd, 0.0) + np.where(nd > rm, nd - rm, 0.0)
            c[sel] = req
        c = _cover(np.maximum(c, 0.0), n) - _cover(np.maximum(-c, 0.0), n)
        out.append(c)
    return out


def oscillate(body, drives, cm_pp, y, env, point, rate_hz_for_window=5.0, iters=4, room=None):
    """(Body, info): the oscillator of the module docstring. cm_pp: the chord (cm) at full envelope, scalar or (T,);
    y (T,) waveform in [-1, 1]; env (T,) envelope in [0, 1]; point: point_spec; room: None or a callable
    (R0, drives, shifts_deg, peak_deg, rate, window_s) -> [(T,) deg] adding to the centre shifts (FaceRoom). info:
    theta_deg (T,) at full envelope, the shifts (deg), the realized chord (cm) at the extremes over the knots with
    env = 1."""
    R0 = body.local_R()
    rest, tr = body.rest(), body.transl
    T = body.T
    env = np.asarray(env, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    target = np.broadcast_to(np.asarray(cm_pp, dtype=np.float64) / 100.0, (T,)).copy()
    on = env > 0
    d = math.radians(0.5)
    w = [dr.weight for dr in drives]
    pp = _point(_apply(R0, drives, [wi * d * np.ones(T) for wi in w]), tr, rest, point)
    pm = _point(_apply(R0, drives, [-wi * d * np.ones(T) for wi in w]), tr, rest, point)
    lever = ((pp - pm).norm(dim=-1) / (2 * math.sin(d))).numpy()
    theta = np.where(on, np.arcsin(np.clip(target / (2 * np.maximum(lever, 1e-9)), 0.0, 0.999)), 0.0)
    window = max(0.3, 1.5 / rate_hz_for_window)
    extra = [np.zeros(T) for _ in drives]

    def all_shifts(theta, with_room=True):
        peak = np.degrees(theta * env)
        s = _shifts(R0, drives, peak, body.rate, window)
        if room is None or not with_room:
            return s, [np.zeros(T) for _ in drives]
        e = room(R0, drives, s, peak, body.rate, window)
        return [a + b for a, b in zip(s, e)], e
    for i in range(iters):
        shifts, extra = all_shifts(theta, i == iters - 1)      # the room once theta has settled
        cs = [np.radians(s) for s in shifts]
        pp = _point(_apply(R0, drives, [c + wi * theta for c, wi in zip(cs, w)]), tr, rest, point)
        pm = _point(_apply(R0, drives, [c - wi * theta for c, wi in zip(cs, w)]), tr, rest, point)
        chord = (pp - pm).norm(dim=-1).numpy()
        ratio = np.where(on & (chord > 1e-12), target / np.maximum(chord, 1e-12), 1.0)
        theta = np.clip(theta * ratio, 0.0, math.radians(89.0))
    shifts, extra = all_shifts(theta)
    cs = [np.radians(s) for s in shifts]
    for _ in range(3 if room is not None else 0):        # the chord again on the final centres
        pp = _point(_apply(R0, drives, [c + wi * theta for c, wi in zip(cs, w)]), tr, rest, point)
        pm = _point(_apply(R0, drives, [c - wi * theta for c, wi in zip(cs, w)]), tr, rest, point)
        chord = (pp - pm).norm(dim=-1).numpy()
        theta = np.clip(theta * np.where(on & (chord > 1e-12), target / np.maximum(chord, 1e-12), 1.0), 0.0,
                        math.radians(89.0))
    ang = [c + wi * theta * env * y for c, wi in zip(cs, w)]
    Rn = _apply(R0, drives, ang)
    joints = sorted({dr.joint for dr in drives})
    out = body.with_rotations(Rn, joints)
    pp = _point(_apply(R0, drives, [c + wi * theta for c, wi in zip(cs, w)]), tr, rest, point)
    pm = _point(_apply(R0, drives, [c - wi * theta for c, wi in zip(cs, w)]), tr, rest, point)
    chord = 100.0 * (pp - pm).norm(dim=-1).numpy()
    full = env >= 1.0 - 1e-9
    pname = {"joint": "joint"}.get(point[0], "nose (rigid on the head)" if point[1] == HEAD else
                                   "fingertip (rigid on the wrist)")
    info = {"drives": [{"joint": int(dr.joint), "axis": dr.axis if isinstance(dr.axis, str) else list(dr.axis),
                        "frame": dr.frame, "weight": float(dr.weight)} for dr in drives],
            "point": pname,
            "point_joint": int(point[1]), "peak_deg_range": [float(np.degrees(theta[on]).min()) if on.any() else 0.0,
                                                             float(np.degrees(theta[on]).max()) if on.any() else 0.0],
            "centre_shift_deg_max": [float(np.abs(s).max()) for s in shifts],
            "room_shift_deg_max": [float(np.abs(s).max()) for s in extra],
            "chord_cm_at_full_envelope": [float(chord[full].min()), float(chord[full].max())] if full.any() else None}
    return out, info, {"angle_rad": np.stack(ang), "centre_rad": np.stack(cs), "envelope": env, "waveform": y}


# ---------------------------------------------------------------- the face as a room (reach)
FACE_GRID_M = 0.004            # the head's signed-distance grid step
FACE_SEARCH_M = 0.15           # the knots searched: the hand within this distance of the head at the base
FACE_MAX_DEG = 30.0            # the largest face shift searched, either way
FACE_STEP_DEG = 1.0            # the scan step of the face shift


def _vertex_normals(v, faces):
    fn = np.cross(v[faces[:, 1]] - v[faces[:, 0]], v[faces[:, 2]] - v[faces[:, 0]])
    n = np.zeros_like(v)
    for k in range(3):
        np.add.at(n, faces[:, k], fn)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


@functools.lru_cache(maxsize=2)
def _face_model(side):
    """The head of the seated reach (the signed distance to its surface on a FACE_GRID_M grid: the distance to the
    nearest head or neck vertex, negative behind that vertex's normal) and the vertices of the pointing hand and of
    the forearm in the frames of the wrist and of the elbow (rigid on them), taken at the first hold at the nose of
    the level-0 reach (the head never moves on the reach)."""
    from scipy.spatial import cKDTree
    from .viewer import dominant_joint, faces as mesh_faces, vertices_at
    body, info = reach_base(side)
    hold = [p for p in info["phases"] if p["kind"] == "hold_nose"][0]
    k = int(round(0.5 * (hold["t0_s"] + hold["t1_s"]) * body.rate))
    Vk = vertices_at(body, body.times().numpy()[[k]])[0]
    faces = mesh_faces()
    nrm = _vertex_normals(Vk, faces)
    dom = dominant_joint("neutral")
    head = np.flatnonzero(np.isin(dom, (NECK, HEAD)))
    hand = np.flatnonzero(np.isin(dom, (WRIST[side],) + tuple(range(H.FIRST[side], H.FIRST[side] + 15))))
    fore = np.flatnonzero(dom == ELBOW[side])
    J, Gl = body.fk()
    J, Gl = J[k].numpy(), Gl[k].numpy()
    loc = {WRIST[side]: (Vk[hand] - J[WRIST[side]]) @ Gl[WRIST[side]],
           ELBOW[side]: (Vk[fore] - J[ELBOW[side]]) @ Gl[ELBOW[side]]}
    hv, hn = Vk[head], nrm[head]
    lo, hi = hv.min(0) - 0.06, hv.max(0) + 0.06
    axes = [np.arange(lo[i], hi[i] + FACE_GRID_M, FACE_GRID_M) for i in range(3)]
    P = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    d, j = cKDTree(hv).query(P)
    sdf = (np.where(((P - hv[j]) * hn[j]).sum(1) < 0.0, -d, d)).reshape([len(a) for a in axes])
    return {"lo": lo, "sdf": sdf, "loc": loc, "loc_search": {j: (L if j == WRIST[side] else L[::3])
                                                             for j, L in loc.items()},
            "ids": np.concatenate([hand, fore]), "head_joints": J[[NECK, HEAD]].copy(), "rest": body.rest(),
            "boxes": RC.examiner_boxes(side)}


def _sdf_at(fm, P, boxes=True):
    """The signed distance (m) of points P (..., 3) to the head and, with boxes, to the examiner's finger
    (reach.examiner_boxes)."""
    from scipy.ndimage import map_coordinates
    idx = (np.asarray(P, dtype=np.float64).reshape(-1, 3) - fm["lo"]) / FACE_GRID_M
    d = map_coordinates(fm["sdf"], idx.T, order=1, mode="constant", cval=1.0).reshape(np.shape(P)[:-1])
    return np.minimum(d, RC.box_sdf(P, fm["boxes"])) if boxes else d


class FaceRoom:
    """The face (and the examiner's finger) as a room of a tremor on the reach (module docstring; oscillate's room):
    at the knots where the hand comes within FACE_SEARCH_M of the head or of the examiner's finger
    (reach.examiner_boxes), every third knot, the least shift s (deg, a scan of FACE_STEP_DEG then a bisection
    inside the step; one direction for each run of such knots, the one asking the smaller shift) along the drives'
    weights, after which every point of the oscillation
    (seven waveform values from -1 to 1) keeps the hand and forearm (vertices rigid on the wrist and the elbow)
    FACE_GAP_CM from their surfaces; spread to the knots between and covered as the joint-limit shifts are (a moving
    maximum then a moving mean over the window, each direction apart). Oscillate asks it once theta has settled.
    check(body) reads the skinned mesh."""

    def __init__(self, body, side, gap_cm=FACE_GAP_CM):
        self.side, self.gap = side, gap_cm / 100.0
        self.fm = _face_model(side)
        self.transl = body.transl
        J = body.joints()[:, [NECK, HEAD]].numpy()
        if np.abs(J - self.fm["head_joints"][None]).max() > 1e-6:
            raise ValueError("FaceRoom: the head moves on this body; the face model holds a still head")

    def clearance(self, R, transl, search=True):
        """(N,) m: the least signed distance of the hand and forearm points to the head surface."""
        from ..body.joints import fk as _fk
        J, Gl = _fk(torch.as_tensor(R, dtype=F64), torch.as_tensor(transl, dtype=F64), self.fm["rest"])
        pts = [J[:, j, None, :] + torch.einsum("nab,mb->nma", Gl[:, j], torch.as_tensor(L, dtype=F64))
               for j, L in self.fm["loc_search" if search else "loc"].items()]
        return _sdf_at(self.fm, torch.cat(pts, 1).numpy()).min(1)

    def __call__(self, R0, drives, shifts, peak, rate, window_s):
        T = len(peak)
        zero = [np.zeros(T) for _ in drives]
        on = peak > 0
        if not on.any():
            return zero
        base = self.clearance(R0, self.transl)
        K = np.flatnonzero(on & (base < FACE_SEARCH_M))
        if not len(K):
            return zero
        K = np.unique(np.r_[K[::3], K[-1]])
        Y = np.linspace(-1.0, 1.0, 7)

        def margin(idx, s):
            """least margin (m) over the waveform values at the knots K[idx] with the signed shifts s (deg)."""
            k = K[idx]
            Rk = R0[torch.as_tensor(k)].repeat_interleave(len(Y), 0)
            trk = self.transl[torch.as_tensor(k)].repeat_interleave(len(Y), 0)
            ang = [np.radians(shifts[i][k][:, None] + d.weight * (np.asarray(s)[:, None] + peak[k][:, None] * Y[None]))
                   .reshape(-1) for i, d in enumerate(drives)]
            return self.clearance(_apply(Rk, drives, ang), trk).reshape(len(k), len(Y)).min(1) - self.gap
        need = margin(np.arange(len(K)), np.zeros(len(K))) < 0
        if not need.any():
            return zero
        best = np.full(len(K), np.nan)
        n = max(1, int(round(window_s * rate)))
        grid = np.arange(FACE_STEP_DEG, FACE_MAX_DEG + 1e-9, FACE_STEP_DEG)
        idx = np.flatnonzero(need)
        runs = np.split(idx, np.flatnonzero(np.diff(K[idx]) > n) + 1)
        for run in runs:                       # one direction per run, so that the covered shifts never cancel
            cands = []
            for sg in (1.0, -1.0):
                vals = np.full(len(run), np.nan)
                for sv in grid:
                    todo = np.flatnonzero(np.isnan(vals))
                    if not len(todo):
                        break
                    ok = margin(run[todo], np.full(len(todo), sg * sv)) >= 0
                    vals[todo[ok]] = sv
                if not np.isnan(vals).any():
                    cands.append((float(vals.max()), sg, vals))
            if not cands:
                raise ValueError(f"the {self.side} hand cannot keep {100 * self.gap:g} cm from the face and the "
                                 f"examiner's finger with a shift of {FACE_MAX_DEG:g} deg either way")
            _, sg, hi_ = min(cands, key=lambda c: c[0])
            lo_ = hi_ - FACE_STEP_DEG
            for _ in range(5):                                   # refined inside the step
                mid = 0.5 * (lo_ + hi_)
                ok = margin(run, sg * mid) >= 0
                hi_, lo_ = np.where(ok, mid, hi_), np.where(ok, lo_, mid)
            best[run] = sg * hi_
        req = np.zeros(T)
        req[K] = np.where(need, best, 0.0)
        out = []
        for part in (np.maximum(req, 0.0), np.maximum(-req, 0.0)):
            part = np.maximum.reduce([np.r_[np.zeros(i), part[:len(part) - i]] for i in range(3)]
                                     + [np.r_[part[i:], np.zeros(i)] for i in range(1, 3)])
            out.append(_cover(part, n))
        cov = out[0] - out[1]
        return [d.weight * cov for d in drives]

    def check(self, body):
        """{'min_clearance_cm', 'max_depth_cm', 't_s', 'knots_checked'}: the hand and forearm vertices of the skinned
        mesh against the head surface at the knots where the rigid points come within 10 cm of it (the validity
        check of the clinical review: a depth of 3 mm or more fails)."""
        c = self.clearance(body.local_R(), body.transl)
        K = np.flatnonzero(c < 0.10)
        if not len(K):
            return {"min_clearance_cm": float(100 * c.min()), "max_depth_cm": 0.0, "t_s": None, "knots_checked": 0,
                    "examiner_min_clearance_cm": None, "examiner_max_depth_cm": 0.0}
        P = H.body_points(body, self.fm["ids"], knots=K)
        m = _sdf_at(self.fm, P, boxes=False).min(1)
        x = RC.box_sdf(P, self.fm["boxes"]).min(1)
        i = int(np.argmin(m))
        return {"min_clearance_cm": float(100 * m[i]), "max_depth_cm": float(max(0.0, -100 * m[i])),
                "t_s": float(K[i] / body.rate), "knots_checked": int(len(K)), "gap_cm": 100 * self.gap,
                "examiner_min_clearance_cm": float(100 * x.min()),
                "examiner_max_depth_cm": float(max(0.0, -100 * x.min()))}


# ---------------------------------------------------------------- waveforms and envelopes
def phase_of(t, f_hz, phase0=0.0):
    """(T,) rad: phase0 + 2 pi times the running integral of f (Hz, scalar or (T,)) over t (trapezoid)."""
    t = np.asarray(t, dtype=np.float64)
    f = np.broadcast_to(np.asarray(f_hz, dtype=np.float64), t.shape)
    return phase0 + 2 * math.pi * np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(t))])


def jerky_wave(t, f0_hz, rng, period_cv, a_min, rise_share=RISE_SHARE, fast_sign=1.0):
    """((T,) waveform in [-1, 1], info): the jerky waveform of the module docstring; the fast phase runs toward
    fast_sign (+1 or -1)."""
    t = np.asarray(t, dtype=np.float64)
    P0 = 1.0 / f0_hz
    starts = [-float(rng.uniform(0.0, P0))]
    while starts[-1] < t[-1] + P0:
        starts.append(starts[-1] + P0 * float(np.exp(period_cv * rng.standard_normal() - 0.5 * period_cv ** 2)))
    starts = np.asarray(starts)
    k = np.clip(np.searchsorted(starts, t, side="right") - 1, 0, len(starts) - 2)
    u = (t - starts[k]) / (starts[k + 1] - starts[k])
    r = rise_share
    w = np.where(u < r, 0.5 - 0.5 * np.cos(math.pi * u / r), 0.5 + 0.5 * np.cos(math.pi * (u - r) / (1 - r)))
    centres = 0.5 * (starts[1:] + starts[:-1])
    amps = a_min + (1.0 - a_min) * rng.beta(2.0, 2.0, len(centres))
    full, a = [], float(t[0])
    while a < t[-1]:                                     # one cycle at 1 in every window of PEAK_EVERY_S
        z = min(a + float(rng.uniform(*PEAK_EVERY_S)), float(t[-1]))
        inside = np.flatnonzero((centres >= a) & (centres < z))
        if len(inside):
            i = int(inside[int(rng.integers(len(inside)))])
            amps[i] = 1.0
            full.append(i)
        a = z
    from scipy.interpolate import PchipInterpolator
    A = np.clip(PchipInterpolator(centres, amps)(np.clip(t, centres[0], centres[-1])), a_min, 1.0)
    periods = np.diff(starts)
    y = fast_sign * A * (2 * w - 1)
    inside = (centres >= t[0]) & (centres <= t[-1])
    return y, {"y_pp_max": wave_pp_max(y, t, P0), "waveform": "jerky", "rise_share": r, "period_cv": period_cv,
               "a_min": a_min, "amplitude_draw": "a_min + (1 - a_min) Beta(2, 2) per cycle, one cycle at 1 in every "
               f"{PEAK_EVERY_S[0]:g} to {PEAK_EVERY_S[1]:g} s", "full_cycles": len(full),
               "cycle_amplitude_median": float(np.median(amps[inside])), "fast_sign": float(fast_sign),
               "mean_period_s": float(periods.mean()), "period_range_s": [float(periods.min()), float(periods.max())],
               "n_cycles": int(len(periods))}


def wave_pp_max(y, t, period_s):
    """The largest peak-to-peak of a waveform y (T,) in windows of 1.5 periods (the read-back's windows), at most 2
    (y in [-1, 1])."""
    rate = 1.0 / float(t[1] - t[0])
    n = int(round(1.5 * period_s * rate)) + 1
    if n >= len(y):
        return float(y.max() - y.min())
    W = np.lib.stride_tricks.sliding_window_view(np.asarray(y), n)
    return float((W.max(1) - W.min(1)).max())


def intention_profile(t, phases, floor):
    """(T,) in [floor, 1]: floor at mid-movement, rising as the fingertip nears the target of each reach movement
    (((s - 1/2) / (1/2))^2 over the second half of the minimum-jerk share s), 1 in the holds at the targets, and
    fading faster as it leaves ((1 - s / INTENTION_DEPART)^2 over the first INTENTION_DEPART of the path); 0 outside
    the reach cycles."""
    t = np.asarray(t, dtype=np.float64)
    e = np.zeros_like(t)
    for ph in phases:
        a, b = ph["t0_s"], ph["t1_s"]
        sel = (t >= a) & (t <= b)
        if ph["kind"] in ("to_nose", "to_target"):
            from .motions import minimum_jerk
            s = minimum_jerk((t[sel] - a) / (b - a))
            arrive = np.clip((s - 0.5) / 0.5, 0.0, 1.0) ** 2
            leave = np.clip(1.0 - s / INTENTION_DEPART, 0.0, 1.0) ** 2
            e[sel] = np.maximum(e[sel], floor + (1 - floor) * np.maximum(arrive, leave))
        elif ph["kind"] in ("hold_nose", "hold_target") and ph["cycle"] is not None:
            e[sel] = 1.0
    return e


def reach_window(info):
    """(t0, t1) s: from the first movement to the nose to the end of the last hold at the target."""
    ph = [p for p in info["phases"] if p["cycle"] is not None]
    return ph[0]["t0_s"], ph[-1]["t1_s"]


# ---------------------------------------------------------------- bases (cached, copied out)
@functools.lru_cache(maxsize=16)
def _reach_cached(side, errors_key, decomposition):
    errs = {int(k): np.asarray(v) for k, v in errors_key}
    if decomposition > 1:
        n, cyc = RC.decomposed_pace(side)
        return RC.reach(side, n_reaches=n, cycle_s=cyc, endpoint_errors=errs or None, decomposition=decomposition)
    return RC.reach(side, endpoint_errors=errs or None, decomposition=decomposition)


def reach_base(side="left", errors=None, decomposition=1):
    key = tuple(sorted((int(k), tuple(float(x) for x in v)) for k, v in (errors or {}).items()))
    b, info = _reach_cached(side, key, int(decomposition))
    return b.copy(), info


def wing_rotations():
    """(22, 3, 3): 'sit' with both arms in the TETRAS 4b posture (WING, draft segment E2b): the upper arms abducted
    (WING['elevation_deg'] above the horizontal), the elbows flexed, the hands near each other at nose level, palms
    down and fingers pointing toward each other; a copy of the cached solution (_wing_solution)."""
    return _wing_solution()[0].clone()


@functools.lru_cache(maxsize=1)
def _wing_solution():
    """(R (22, 3, 3), report): the bounded least squares (scipy L-BFGS-B, the SP1 table less 1 deg) on the shoulder,
    elbow and wrist components of each arm: the middle fingertip at nose height, WING['tip_forward_cm'] in front of
    the nose and WING['tip_gap_cm'] from its mirror image; the elbow at the shoulder's height plus the elevation; the
    palm facing the floor; the fingers pointing toward the midline; small off-hinge elbow and wrist components."""
    from scipy.optimize import minimize
    from ..body.joints import fk as _fk
    from .body import rest_joints
    R = posture_rotations("sit")
    rest = rest_joints()
    zero = torch.zeros(1, 3, dtype=F64)
    J0, G0 = _fk(R[None], zero, rest)
    nose = (J0[0, HEAD] + G0[0, HEAD] @ _nose_offset()).numpy()
    lim = joint_limits()
    palmar = {s: torch.as_tensor(H.hand_frame()[s]["palmar"], dtype=F64) for s in ("left", "right")}
    report = {}
    for side, sg in (("left", 1.0), ("right", -1.0)):
        R[COLLAR[side]] = torch.eye(3, dtype=F64)
        js = (SHOULDER[side], ELBOW[side], WRIST[side])
        lo = np.concatenate([lim.lo[j] for j in js]) + 1.0
        hi = np.concatenate([lim.hi[j] for j in js]) - 1.0
        tgt = torch.as_tensor(nose + np.array([sg * WING["tip_gap_cm"] / 200.0, 0.0, WING["tip_forward_cm"] / 100.0]))
        off = _middle_tip_offset(side)
        el_rise = math.tan(math.radians(WING["elevation_deg"]))
        R_fixed = R.clone()

        def pose(x):
            q = torch.deg2rad(x.reshape(3, 3))
            RR = R_fixed.clone()
            RR[list(js)] = axis_angle_to_matrix(q)
            J, Gl = _fk(RR[None], zero, rest)
            return RR, J[0], Gl[0]

        def f(x):
            xt = torch.tensor(x, dtype=F64, requires_grad=True)
            _, J, Gl = pose(xt)
            s, e, w = js
            tip = J[w] + Gl[w] @ off
            arm = J[e] - J[s]
            rise = 100.0 * (J[e][1] - J[s][1] - el_rise * torch.sqrt(arm[0] ** 2 + arm[2] ** 2))
            n = Gl[w] @ palmar[side]
            fing = (tip - J[w]) / (tip - J[w]).norm()
            loss = ((100.0 * (tip - tgt)) ** 2).sum() + rise ** 2 + 20.0 * ((n[1] + 1.0) ** 2) \
                + 5.0 * (fing[0] * sg + 0.7).clamp_min(0.0) ** 2 \
                + 1e-3 * (xt ** 2).sum() + 1e-2 * (xt[3] ** 2 + xt[5] ** 2) + 1e-2 * (xt[6:] ** 2).sum()
            loss.backward()
            return float(loss.detach()), xt.grad.numpy().copy()
        x0 = np.zeros(9)
        x0[4] = -sg * 100.0                               # the elbow flexed
        best = None
        for start in (x0, np.r_[0.0, -sg * 30.0, 0.0, 0.0, -sg * 110.0, 0.0, 0.0, 0.0, 0.0]):
            r = minimize(f, np.clip(start, lo, hi), jac=True, method="L-BFGS-B", bounds=list(zip(lo, hi)),
                         options={"maxiter": 2000, "ftol": 1e-15, "gtol": 1e-10})
            if best is None or r.fun < best.fun:
                best = r
        RR, J, Gl = pose(torch.as_tensor(best.x))
        R = RR.detach()
        s, e, w = js
        tip = (J[w] + Gl[w] @ off).detach().numpy()
        report[side] = {"q_deg": best.x.reshape(3, 3).round(2).tolist(), "loss": float(best.fun),
                        "tip_error_cm": float(100 * np.linalg.norm(tip - tgt.numpy())),
                        "elbow_above_shoulder_cm": float(100 * (J[e][1] - J[s][1])),
                        "palm_down_cos": float(-(Gl[w] @ palmar[side])[1])}
    return R, report


@functools.lru_cache(maxsize=2)
def wing_axis(side):
    """(3,) unit axis in the shoulder's parent frame (the collar's): horizontal and normal to the line from the
    shoulder to the middle of the elbow and the wrist in the wing posture (drives_of 'wing')."""
    from .body import rest_joints
    R = wing_rotations()
    J, Gl = fk(R[None], torch.zeros(1, 3, dtype=F64), rest_joints())
    J, Gl = J[0], Gl[0]
    m = 0.5 * (J[ELBOW[side]] + J[WRIST[side]]) - J[SHOULDER[side]]
    a = torch.linalg.cross(torch.tensor([0.0, 1.0, 0.0], dtype=F64), torch.stack([m[0], 0 * m[1], m[2]]))
    a = a / a.norm()
    parent = 13 if side == "left" else 14
    return (Gl[parent].T @ a).numpy()


def posture_from(R, duration_s, rate=100.0):
    """A HandBody holding the rotations R (22, 3, 3) (hands at rest), placed as base.static_body."""
    from .body import floor_to_zero, rest_joints
    T = int(round(duration_s * rate)) + 1
    rest = rest_joints()
    tr = torch.zeros(T, 3, dtype=F64)
    tr[:, [0, 2]] = -rest[0, [0, 2]]
    b = Body(matrix_to_rot6d(R)[None].expand(T, -1, 6).clone(), tr, torch.zeros(T, 2, dtype=torch.bool), rate=rate)
    b = floor_to_zero(b)
    return H.hands_of(Body(b.rot6d, b.transl, contact_labels(b), b.betas, b.gender, b.rate))


# [module] the dystonic arm (catalogue T18, bhatia2018: relieved at a null point, worse in some postures): the wrist
# pulled into flexion and ulnar deviation; between null_window_s (ramps of null_ramp_s) the wrist let further into
# the pull by null_extra_deg, where the tremor falls to null_gain of its level (the null point)
DYSTONIC_ARM = {"flexion_deg": 20.0, "ulnar_deg": 10.0, "null_extra_deg": 25.0, "null_gain": 0.12,
                "null_window_s": (13.0, 20.0), "null_ramp_s": 1.2}


def dystonic_arm_base(side="left", duration_s=DUR["E2"]):
    """(Body, info, gain (T,)): 'sit_arms_forward' with the dystonic wrist posture of DYSTONIC_ARM held, and let
    into the pull around the null window; gain: the tremor's share of its level against that posture (1, falling to
    null_gain at the null point, with the posture)."""
    from .motions import minimum_jerk
    D = DYSTONIC_ARM
    b = static_body("sit_arms_forward", duration_s)
    t = b.times().numpy()
    a, z = D["null_window_s"]
    r = D["null_ramp_s"]
    p = minimum_jerk((t - a) / r) - minimum_jerk((t - z + r) / r)
    up = 1.0 if side == "left" else -1.0                 # the wrist z sign that lifts the fingers (drives_of)
    flex = -up * np.radians(D["flexion_deg"] + p * D["null_extra_deg"])
    R = b.local_R().clone()
    w = WRIST[side]
    Rw = compose(R[:, w], axis_rotation("z", torch.as_tensor(flex)), "child")
    R[:, w] = compose(Rw, axis_rotation("y", torch.as_tensor(np.full(b.T, up * np.radians(D["ulnar_deg"])))),
                      "child")
    gain = 1.0 - (1.0 - D["null_gain"]) * p
    return b.with_rotations(R, [w]), dict(D, side=side), gain


def functional_rotations():
    """(22, 3, 3): 'sit' with the left arm forward as in 'sit_arms_forward' (the tremor arm), the right forearm on
    the chair arm (the tapping hand)."""
    R = posture_rotations("sit")
    F = posture_rotations("sit_arms_forward")
    for j in (13, 16, 18, 20):
        R[j] = F[j]
    return R


# ---------------------------------------------------------------- the operators
def _truth(op, seed, grade, scale, params, sources, base, out, arrays):
    val = validity(base, out)
    if H.has_hands(out):
        val["hand_limits"] = H.hand_limit_report(out.hand_R())
    return truth_record(op, seed, grade, scale, params, sources, val, arrays=arrays)


def limb_tremor(body, sides, cm_pp, hz, drive="elbow", point="wrist", env=None, wave="sine", seed=0, grade=None,
                scale="MDS-UPDRS", item="", tremor_type="", condition="", extra=None, operator="tremor_catalogue",
                fast_sign=1.0, room=None):
    """(Body, truth): a tremor on each of sides (one dict of overrides per side allowed through hz and cm_pp as
    {side: value}), each limb its own phase (and, for 'jerky', its own periods and amplitudes) from the seed;
    fast_sign: the direction of the jerky waveform's fast phase along the drives; room: {side: FaceRoom} or None."""
    rng = as_rng(seed)
    t = body.times().numpy()
    env = np.ones(body.T) if env is None else np.asarray(env, dtype=np.float64)
    out, limbs, arrays = body, {}, {}
    for side in sides:
        f = hz[side] if isinstance(hz, dict) else hz
        cm = cm_pp[side] if isinstance(cm_pp, dict) else cm_pp
        winfo = {"waveform": wave}
        if wave == "sine":
            phase0 = float(rng.uniform(0.0, 2 * math.pi))
            y = np.sin(phase_of(t, f, phase0))
            winfo["phase_rad"] = phase0
        else:
            a_min = A_MIN[wave]
            if grade is not None and scale in SCALES and SCALES[scale].get(grade, (0.0,))[0] > 0:
                a_min = max(a_min, min(0.95, 1.05 * SCALES[scale][grade][0] / cm))
            y, winfo = jerky_wave(t, f, rng, PERIOD_CV[wave], a_min, RISE_SHARE if wave == "dystonic" else 0.5,
                                  fast_sign)
        dr = drives_of(drive, side, cm=cm) if isinstance(drive, str) else drive
        ps = point_spec(point, side)
        # the level is the largest cycle's chord: the chord at y = +-1 is the level over the waveform's largest
        # windowed peak-to-peak (2 for a sine)
        cm_extremes = cm * 2.0 / winfo["y_pp_max"] if "y_pp_max" in winfo else cm
        out, info, arr = oscillate(out, dr, cm_extremes, y, env, ps, f, room=(room or {}).get(side))
        limbs[side] = dict(info, frequency_hz=float(f), cm_pp=float(cm), cm_pp_at_unit_waveform=float(cm_extremes),
                           **winfo)
        arrays.update({f"{side}_{k}": v for k, v in arr.items()})
    params = {"symptom": operator, "item": item, "scale": scale, "grade": grade, "tremor_type": tremor_type,
              "condition": condition, "sides": list(sides), "drive": drive if isinstance(drive, str) else "custom",
              "point": point, "limbs": limbs, "envelope_on_share": float((env > 0).mean()),
              "readback": "oscillation"}
    params.update(extra or {})
    src = ["mds_updrs" if scale == "MDS-UPDRS" else SCALE_SOURCE[scale], "elble2006", "module"]
    truth = _truth(operator, seed if isinstance(seed, (int, np.integer)) else None, grade, f"{scale} {item}",
                   params, src + list((extra or {}).get("sources", ())), body, out, arrays)
    for side, fr in (room or {}).items():
        truth["validity"]["hand_head"] = fr.check(out)
    return out, truth


def weight_shifts(t, rng, first_s=OT_SHIFT_FIRST_S, every_s=OT_SHIFT_EVERY_S, dur_s=OT_SHIFT_S, cm=OT_SHIFT_CM):
    """((T,) m lateral pelvis offset, [shift onsets s]): from first_s on, the pelvis carried to one side then the
    other by cm (minimum jerk over dur_s), every every_s (uniform), the first toward the left foot."""
    from .motions import minimum_jerk
    x = np.zeros_like(t)
    onsets, s, side, cur = [], float(first_s), 1.0, 0.0
    while s + dur_s <= t[-1]:
        goal = side * cm / 100.0
        x = np.where(t >= s, cur + (goal - cur) * minimum_jerk((t - s) / dur_s), x)
        onsets.append(s)
        cur, side = goal, -side
        s += float(rng.uniform(*every_s))
    return x, onsets


def orthostatic(duration_s=DUR["E7"], hz=HZ["orthostatic"], pp_cm=OT_PP_CM, depth_cm=OT_DEPTH_CM, seed=0,
                rate=100.0, signs=True):
    """(Body, truth): standing with the knees unlocked (the pelvis depth_cm below the straight stance, both feet
    planted, the legs re-solved by ik.hinge_leg with the feet's orientation kept) and the pelvis oscillating
    vertically by pp_cm peak to peak at hz (0 for the level 0). signs: the visible signs of orthostatic tremor
    (benitoleon2016 via the catalogue, T17): the feet set OT_STANCE_CM apart (ankle to ankle, the feet moved
    outward, flat and planted) and weight shifts (weight_shifts: the pelvis carried sideways over the planted feet);
    the level 0 stands on the ordinary stance, still."""
    from .ik import hinge_leg
    base = static_body("stand", duration_s, rate)
    J, Gl = base.fk()
    R = base.local_R().clone()
    rest = base.rest()
    t = base.times().numpy()
    rng = as_rng(seed)
    phase0 = float(rng.uniform(0.0, 2 * math.pi))
    dy = -depth_cm / 100.0 + (pp_cm / 200.0) * np.sin(phase_of(t, hz, phase0)) * envelope(t, [(0.0, t[-1])], RAMP_S,
                                                                                           t[-1])
    width0 = float(J[0, 7, 0] - J[0, 8, 0])
    widen = (OT_STANCE_CM / 100.0 - width0) / 2.0 if signs else 0.0
    dx, onsets = weight_shifts(t, rng) if signs else (np.zeros_like(t), [])
    dyt, dxt = torch.as_tensor(dy), torch.as_tensor(dx)
    G0 = Gl[:, 0]
    fwd = torch.tensor([0.0, 0.0, 1.0], dtype=F64).expand(base.T, 3)
    for h, kn, a, sg in ((1, 4, 7, 1.0), (2, 5, 8, -1.0)):
        Hp = J[:, h].clone()
        Hp[:, 1] += dyt
        Hp[:, 0] += dxt
        A = J[:, a].clone()
        A[:, 0] += sg * widen
        G_th, k = hinge_leg(Hp, A, Hp + fwd, rest[kn] - rest[h], rest[a] - rest[kn], G0, hinge=Gl[:, h][..., :, 0])
        Rk = axis_angle_to_matrix(torch.stack([k, 0 * k, 0 * k], -1))
        R[:, h] = G0.transpose(-1, -2) @ G_th
        R[:, kn] = Rk
        R[:, a] = (G_th @ Rk).transpose(-1, -2) @ Gl[:, a]
    tr = base.transl.clone()
    tr[:, 1] += dyt
    tr[:, 0] += dxt
    out = Body(matrix_to_rot6d(R), tr, base.contact.clone(), rate=rate)
    out = Body(out.rot6d, out.transl, contact_labels(out), rate=rate)
    J2 = out.joints()
    params = {"symptom": "orthostatic", "item": "presence", "scale": "none (presence; no amplitude scale)",
              "frequency_hz": float(hz), "pelvis_pp_cm": float(pp_cm), "depth_cm": float(depth_cm),
              "phase_rad": phase0, "stance_width_cm": float(100 * (J2[0, 7, 0] - J2[0, 8, 0])),
              "stance_width_base_cm": 100 * width0,
              "ankle_drift_mm": float(1000 * (J2[:, [7, 8]] - J2[:1, [7, 8]]).norm(dim=-1).max()),
              "weight_shift_cm": OT_SHIFT_CM if signs else 0.0, "weight_shift_onsets_s": onsets,
              "weight_shift_s": OT_SHIFT_S, "readback": "orthostatic",
              "note": ("palpable rather than visible: 15.7 Hz lies above the 15 Hz Nyquist rate of a 30 fps video, a "
                       "30 fps clip shows it aliased (about 14.3 Hz) and at a few millimetres; the 100 Hz body "
                       "carries it, and scripts/render_mesh_video.py renders this clip at 60 fps (fps_rule, SH14), "
                       "which samples it without aliasing but does not make a few millimetres visible; the visible "
                       "signs are the wide base and the weight shifts")}
    val = validity(base, out)
    return out, truth_record("tremor_catalogue", seed, 1 if pp_cm else 0, "presence", params,
                             ("hassan2016", "buijink2012", "benitoleon2016", "module"), val,
                             arrays={"pelvis_dy_m": dy, "pelvis_dx_m": dx})


def tapping_body(duration_s, tap_window, ballistic_window, hz_tap=HZ["tapping"], rate=100.0):
    """(HandBody, info): the functional-tremor scene (functional_rotations) with the right hand tapping by wrist
    extension (TAP_DEG, one-sided taps from the chair arm) at hz_tap inside tap_window, and a ballistic movement of
    the right arm to the arms-forward posture and back (minimum jerk, 0.3 s each way) inside ballistic_window."""
    from .motions import minimum_jerk
    b = posture_from(functional_rotations(), duration_s, rate)
    t = b.times().numpy()
    R = b.local_R().clone()
    a, z = tap_window
    on = (t >= a) & (t <= z)
    tap = np.where(on, 0.5 - 0.5 * np.cos(2 * math.pi * hz_tap * (t - a)), 0.0)
    # right wrist extension: about the hand's own z axis, negative for the right hand (fingers up)
    R[:, 21] = compose(R[:, 21], axis_rotation("z", torch.as_tensor(-np.radians(TAP_DEG) * tap)), "child")
    s0, s1 = ballistic_window
    half = 0.5 * (s1 - s0)
    w = minimum_jerk((t - s0) / half) - minimum_jerk((t - s0 - half) / half)
    Rs, Rf = posture_rotations("sit"), posture_rotations("sit_arms_forward")
    for j in (14, 17, 19, 21):
        D = matrix_to_axis_angle(Rs[j].T @ Rf[j])
        Rb = Rs[j][None] @ axis_angle_to_matrix(torch.as_tensor(w)[:, None] * D[None])
        sel = torch.as_tensor(w > 0)
        R[sel, j] = Rb[sel] if j != 21 else Rb[sel]
    out = b.with_rotations(R, [14, 17, 19, 21])
    return out, {"tap_window_s": list(tap_window), "tap_hz": float(hz_tap), "tap_deg": TAP_DEG,
                 "ballistic_window_s": list(ballistic_window), "ballistic": "right arm to the arms-forward posture "
                 "and back, minimum jerk"}


# [module] the functional-tremor timeline (s): the taps, the entrainment's transitions, the ballistic movement, the
# pause starting pause_delay_s after its onset and falling over pause_fall_s, the tremor back from resume_s
FUNCTIONAL = {"duration_s": DUR["E2"], "tap": (10.0, 18.0), "entrain_s": 2.0, "ballistic": (22.0, 22.6),
              "pause_delay_s": 0.08, "pause_fall_s": 0.12, "resume_s": 23.4}


def functional(cm_pp, seed=0, hz=HZ["functional"], level0=False):
    """(Body, truth): left postural tremor of the outstretched hand ('postural' drives, read at the middle
    fingertip) whose own frequency drifts slowly (FUNCTIONAL_CV) and whose amplitude varies slowly (between
    FUNCTIONAL_AMP_MIN and 1 of the level), entrained to the right hand's tapping frequency during the taps (smooth
    transitions of FUNCTIONAL['entrain_s'], a residual phase jitter of FUNCTIONAL_JITTER_RAD) and paused just after
    the onset of the right arm's ballistic movement (SOURCES['schwingenschuh2011'])."""
    F = FUNCTIONAL
    base, sinfo = tapping_body(F["duration_s"], F["tap"], F["ballistic"])
    t = base.times().numpy()
    if level0:
        return base, truth_record("none", None, 0, "level 0", {"base": "functional scene", "scene": sinfo,
                                                                 "readback": "none"}, (), validity(base, base))
    rng = as_rng(seed)
    from scipy.interpolate import PchipInterpolator
    from .motions import minimum_jerk
    knots = np.arange(0.0, t[-1] + 2.0, 2.0)
    drift = PchipInterpolator(knots, FUNCTIONAL_CV * rng.standard_normal(len(knots)))(t)
    f_own = hz * np.exp(drift)
    a, z = F["tap"]
    r = F["entrain_s"]
    mix = minimum_jerk((t - a) / r) - minimum_jerk((t - z) / r)
    f = (1 - mix) * f_own + mix * HZ["tapping"]
    k1 = np.arange(0.0, t[-1] + 1.0, 1.0)
    jitter = mix * PchipInterpolator(k1, FUNCTIONAL_JITTER_RAD * rng.standard_normal(len(k1)))(t)
    am = PchipInterpolator(knots, rng.uniform(0.0, 1.0, len(knots)))(t)
    am = FUNCTIONAL_AMP_MIN + (1.0 - FUNCTIONAL_AMP_MIN) * (am - am.min()) / max(am.max() - am.min(), 1e-12)
    b0 = F["ballistic"][0] + F["pause_delay_s"]
    p0, p1 = b0 + 0.5 * F["pause_fall_s"], F["resume_s"]
    env = np.maximum(envelope(t, [(0.0, p0)], F["pause_fall_s"], t[-1]),
                     envelope(t, [(p1, t[-1])], RAMP_S * 0.6, t[-1]))
    phase0 = float(rng.uniform(0.0, 2 * math.pi))
    y = am * np.sin(phase_of(t, f, phase0) + jitter)
    out, info, arr = oscillate(base, drives_of("postural", "left", cm=cm_pp), cm_pp, y, env,
                               point_spec("middle_tip", "left"), hz)
    g = G.tremor_grade_of_cm(cm_pp)
    p = {"symptom": "tremor_catalogue", "item": "3.15", "scale": "MDS-UPDRS", "grade": g, "tremor_type": "functional",
         "condition": "postural", "sides": ["left"], "drive": "postural", "point": "middle_tip",
         "limbs": {"left": dict(info, frequency_hz=float(hz), cm_pp=float(cm_pp), waveform="sine (entrained, "
                                "amplitude varying)", phase_rad=phase0)},
         "envelope_on_share": float((env > 0).mean()), "scene": sinfo, "readback": "functional",
         "functional": {"own_hz": float(hz), "own_drift_sigma": FUNCTIONAL_CV,
                        "own_hz_range": [float(f_own.min()), float(f_own.max())], "entrained_hz": HZ["tapping"],
                        "entrain_window_s": [a, z], "entrain_transition_s": r,
                        "phase_jitter_sigma_rad": FUNCTIONAL_JITTER_RAD,
                        "amplitude_share_range": [float(am.min()), 1.0],
                        "ballistic_onset_s": F["ballistic"][0], "pause_onset_s": b0,
                        "pause_window_s": [p0, p1]}}
    val = validity(base, out)
    val["hand_limits"] = H.hand_limit_report(out.hand_R())
    return out, truth_record("tremor_catalogue", seed, g, "MDS-UPDRS 3.15", p,
                             ("mds_updrs", "schwingenschuh2011", "buijink2012", "module"), val,
                             arrays={"left_angle_rad": arr["angle_rad"], "envelope": env, "frequency_hz": f,
                                     "amplitude_share": am})


# [module] the pill-rolling hand (the clinical review: the thumb pad on the index pad, rubbing along it): the index at
# index_close of the finger-tapping pinch (handtasks.PINCH), the thumb at thumb_close of it with its CMC flexion set
# so that the thumb's distal pad lies gap_cm from the index (the contact pose); the rub: the index MCP and PIP
# flexing (PILL_SHARE of the rub angle) and the thumb CMC flexing across the palm by thumb_rub_deg, phase_lag_deg
# apart, the thumb MCP flexion corrected at every phase (n_phase phases, scan then bisection) so that the gap stays
# gap_cm: the contact point slides along the index pad
PILL = {"index_close": 1.4, "thumb_close": 1.0, "gap_cm": 0.1, "thumb_rub_deg": 8.0, "phase_lag_deg": 90.0,
        "n_phase": 24}
REST_CURL = (20.0, 30.0, 15.0)         # [module] MCP, PIP, DIP of the other fingers at rest
PILL_SHARE = {"index_mcp": -0.8, "index_pip": -0.4}   # [module] the index's part of the rub (share of the angle)
PILL_LIMIT_MARGIN_DEG = 6.0            # [module] a closed index joint of the pill pose this far inside its limit (room
#                                        for the rub: at 1 deg the rub took the PIP past it at grade 1; open-body port, wave 2)


def _pill_pose(side):
    """{joint: (flex, abd)} deg of the resting hand of pill-rolling before the contact solve (PILL), the other
    fingers loosely curled (module); an index joint whose index_close share passes its flexion limit
    (hands.FLEX_LIMITS_DEG) stays PILL_LIMIT_MARGIN_DEG inside it (open-body port, wave 2: the open body's pinch
    flexes the index's distal joints more than the SMPL+H pinch did)."""
    from .handtasks import PINCH
    out = {}
    for f, k in (("index", PILL["index_close"]), ("thumb", PILL["thumb_close"])):
        for lv, v in enumerate(PINCH[f]):
            fl, ab = (v, 0.0) if np.isscalar(v) else v
            fl = k * fl
            if f == "index":
                fl = min(fl, H.FLEX_LIMITS_DEG["index"][H.LEVELS[lv]][1] - PILL_LIMIT_MARGIN_DEG)
            out[H.hand_joint(side, f, lv)] = (fl, k * ab)
    for f in ("middle", "ring", "pinky"):
        for lv, v in enumerate(REST_CURL):
            out[H.hand_joint(side, f, lv)] = (v, 0.0)
    return out


def _hand_series(side, pose, add=None, T=None):
    """(T, 30, 3, 3): the hand pose {joint: (flex, abd)} with add {joint: (flex (T,), abd (T,))} deg added, each
    joint turned about its abduction axis then its flexion axis (hands.hand_rotations)."""
    fr = H.hand_frame()[side]["joints"]
    add = add or {}
    T = T or len(next(iter(add.values()))[0])
    R = torch.eye(3, dtype=F64).repeat(T, H.N_HAND, 1, 1)
    for j, (fl, ab) in pose.items():
        dfl, dab = add.get(j, (0.0, 0.0))
        flex = torch.as_tensor(np.broadcast_to(fl + np.asarray(dfl, dtype=np.float64), (T,)).copy())
        abd = np.broadcast_to(ab + np.asarray(dab, dtype=np.float64), (T,)).copy()
        M = torch.eye(3, dtype=F64).expand(T, 3, 3)
        if np.any(abd != 0.0):
            M = axis_angle_to_matrix(torch.as_tensor(fr[j]["abd"])[None] * torch.deg2rad(torch.as_tensor(abd))[:, None])
        M = axis_angle_to_matrix(torch.as_tensor(fr[j]["flex"])[None] * torch.deg2rad(flex)[:, None]) @ M
        R[:, H.hand_row(j)] = M
    return R


def _rub(side, osc_deg, phi, thumb_scale=1.0):
    """{joint: (flex (n,), abd (n,))} deg of the rub at phases phi (n,) rad: the index MCP and PIP by osc_deg sin phi
    (PILL_SHARE), the thumb CMC flexion (across the palm, along the index pad) by thumb_rub_deg sin(phi +
    phase_lag_deg) times thumb_scale (osc_deg and thumb_scale scalars or (n,))."""
    lag = math.radians(PILL["phase_lag_deg"])
    phi = np.asarray(phi, dtype=np.float64)
    s = np.sin(phi)
    c = np.sin(phi + lag)
    z = np.zeros_like(phi)
    return {H.hand_joint(side, "index", 0): (PILL_SHARE["index_mcp"] * osc_deg * s, z),
            H.hand_joint(side, "index", 1): (PILL_SHARE["index_pip"] * osc_deg * s, z),
            H.hand_joint(side, "thumb", 0): (thumb_scale * PILL["thumb_rub_deg"] * c, z)}


@functools.lru_cache(maxsize=4)
def _pill_geometry(side):
    """The vertices of the thumb's distal segment and of the index's middle and distal segments, the mesh faces,
    and the seated rest HandBody (the pill-rolling contact measure)."""
    from .viewer import dominant_joint, faces as mesh_faces
    dom = dominant_joint("neutral")
    th = np.flatnonzero(dom == H.hand_joint(side, "thumb", 2))
    ix = np.flatnonzero(np.isin(dom, [H.hand_joint(side, "index", 1), H.hand_joint(side, "index", 2)]))
    faces = mesh_faces()
    return th, ix, faces, H.hands_of(static_body("sit", 0.01))


def _thumb_index_gap(side, Rh):
    """((n,) least signed distance m of the thumb's distal pad to the index surface, (n, 3) contact points on the
    index) for hand rotations Rh (n, 30, 3, 3) on the seated rest."""
    from scipy.spatial import cKDTree
    from .viewer import vertices_at
    th, ix, faces, rest = _pill_geometry(side)
    n = Rh.shape[0]
    m = max(n, 2)
    Rm = torch.cat([Rh, Rh[-1:].expand(m - n, -1, -1, -1)], 0) if m > n else Rh
    b = H.HandBody(rest.rot6d[:1].expand(m, -1, -1).clone(), rest.transl[:1].expand(m, -1).clone(),
                   rest.contact[:1].expand(m, -1).clone(), rate=rest.rate, hand6d=matrix_to_rot6d(Rm))
    V = vertices_at(b, b.times().numpy())
    gaps, pts = np.empty(n), np.empty((n, 3))
    for k in range(n):
        nrm = _vertex_normals(V[k], faces)
        _, j = cKDTree(V[k, ix]).query(V[k, th], k=4)
        na = nrm[ix[j]].mean(1)
        na /= np.linalg.norm(na, axis=1, keepdims=True)
        s = ((V[k, th] - V[k, ix[j[:, 0]]]) * na).sum(1)       # to the tangent plane of the nearest index vertices
        i = int(np.argmin(s))
        gaps[k], pts[k] = s[i], V[k, ix[j[i, 0]]]
    return gaps, pts


def _bisect_thumb(side, pose, add, goal_m, open_deg, closed_deg, level=0, comp=0, step=1.0, iters=10):
    """(n,) deg: the offsets of a thumb joint's component (level 0 CMC, 1 MCP; comp 0 flexion, 1 palmar abduction)
    that set the thumb-index gap to goal_m under the rub angles add ({joint: (flex (n,), abd (n,))}): the first
    closing of the gap to goal_m on a grid from open_deg toward closed_deg, then a bisection inside that step (the gap
    is not monotone past the contact: the thumb slides under the index); raises when the grid holds no closing."""
    j = H.hand_joint(side, "thumb", level)
    n = len(next(iter(add.values()))[0])

    def gap(c):
        a = dict(add)
        fl, ab = a.get(j, (np.zeros(n), np.zeros(n)))
        a[j] = (fl + c, ab) if comp == 0 else (fl, ab + c)
        return _thumb_index_gap(side, _hand_series(side, pose, a))[0]
    grid = np.arange(0.0, abs(closed_deg - open_deg) + 1e-9, step) * np.sign(closed_deg - open_deg) + open_deg
    Gs = np.stack([gap(np.full(n, c)) for c in grid], 0)
    shut = Gs <= goal_m
    if shut[0].any() or not shut.any(0).all():
        raise ValueError("pill-rolling: the thumb CMC cannot set the thumb-index gap at every phase")
    k = np.argmax(shut, axis=0)
    lo_, hi_ = grid[k - 1], grid[k]
    for _ in range(iters):
        m = 0.5 * (lo_ + hi_)
        far = gap(m) > goal_m
        lo_, hi_ = np.where(far, m, lo_), np.where(far, hi_, m)
    return 0.5 * (lo_ + hi_)


@functools.lru_cache(maxsize=8)
def _pill_contact(side, osc_key):
    """(c0 deg, phases (n,), corr (n,) deg, report): the thumb CMC flexion of the contact pose, and the correction at
    each phase of the rub of osc_key/1e4 deg that keeps the gap (PILL)."""
    osc = osc_key / 1e4
    pose = _pill_pose(side)
    j = H.hand_joint(side, "thumb", 0)
    goal = PILL["gap_cm"] / 100.0
    z1 = np.zeros(1)
    c0 = float(_bisect_thumb(side, pose, {j: (z1, z1)}, goal, -20.0, 8.0)[0])
    pose = dict(pose)
    pose[j] = (pose[j][0] + c0, pose[j][1])
    phi = np.linspace(0.0, 2 * math.pi, PILL["n_phase"], endpoint=False)
    if not osc_key:
        return c0, phi, np.zeros_like(phi), {"thumb_cmc_flexion_offset_deg": c0}
    add = _rub(side, osc, phi)
    corr = _bisect_thumb(side, pose, add, goal, -13.0, 24.0, level=1)
    a = dict(add)
    m = H.hand_joint(side, "thumb", 1)
    a[m] = (corr, np.zeros_like(corr))
    g, pts = _thumb_index_gap(side, _hand_series(side, pose, a))
    slide = float(np.max(np.linalg.norm(pts[:, None] - pts[None], axis=-1)))
    rep = {"thumb_cmc_flexion_offset_deg": c0, "thumb_mcp_correction_deg_range": [float(corr.min()),
                                                                                  float(corr.max())],
           "gap_cm_range": [float(100 * g.min()), float(100 * g.max())], "contact_slide_cm": 100 * slide}
    return c0, phi, corr, rep


def pill_rolling(cm_pp=None, pronation_cm_pp=None, side="left", hz=HZ["pill_rolling"], duration_s=20.0, seed=0):
    """(HandBody, truth): seated rest (E1) with the pill-rolling hand (PILL: the thumb pad on the index pad); cm_pp:
    the index fingertip's peak-to-peak chord of the rub alone (the index MCP and PIP flexing, the thumb CMC
    adducting and abducting a quarter period apart, the thumb kept on the index pad), solved by bisection on the
    skinned fingertip; pronation_cm_pp: the fingertip chord with a forearm pronation-supination tremor added in phase
    (tremor.tremor, driver 'pronation_supination'), its degrees solved likewise. None and None: the level 0 (the
    resting hand in the contact pose, no tremor)."""
    from .tremor import tremor
    base = H.hands_of(static_body("sit", duration_s))
    t = base.times().numpy()
    j = H.hand_joint(side, "thumb", 0)
    rows = list(range(0, 15)) if side == "left" else list(range(15, 30))
    c0 = _pill_contact(side, 0)[0]
    pose = dict(_pill_pose(side))
    pose[j] = (pose[j][0] + c0, pose[j][1])
    Rh0 = _hand_series(side, pose, T=base.T)
    b0 = base.with_hands(Rh0, rows)
    if cm_pp is None and pronation_cm_pp is None:
        g, _ = _thumb_index_gap(side, Rh0[:1])
        return b0, truth_record("none", None, 0, "level 0", {"base": "seated rest, resting hand", "readback": "none",
                                                              "pose": {str(k): v for k, v in pose.items()},
                                                              "thumb_index_gap_cm": float(100 * g[0])}, (),
                                validity(base, b0))
    rng = as_rng(seed)
    phase0 = float(rng.uniform(0.0, 2 * math.pi))
    tip = H.tip_vertices()[side]["index"]
    env = envelope(t, [(0.0, t[-1])], RAMP_S, t[-1])

    def chord(osc_deg, pron_deg=0.0):
        """index tip chord (cm) between the extremes of the rub (and of the pronation, in phase)."""
        pts = []
        for sgn in (1.0, -1.0):
            Rh = _hand_series(side, pose, _rub(side, osc_deg, np.full(2, sgn * math.pi / 2), 0.0))
            bb = H.HandBody(b0.rot6d[:2].clone(), b0.transl[:2].clone(), b0.contact[:2].clone(), rate=b0.rate,
                            hand6d=matrix_to_rot6d(Rh))
            if pron_deg:
                from .body import bone_axis
                ax = bone_axis(ELBOW[side], WRIST[side])
                R = bb.local_R().clone()
                R[:, ELBOW[side]] = compose(R[:, ELBOW[side]], axis_rotation(ax, torch.full((2,), sgn * math.radians(
                    pron_deg / 2.0), dtype=F64)), "child")
                bb = bb.with_rotations(R, [ELBOW[side]])
            pts.append(H.body_points(bb, [tip], knots=[0])[0, 0])
        return 100.0 * float(np.linalg.norm(pts[0] - pts[1]))

    def solve(fun, target, hi):
        lo_, hi_ = 0.0, hi
        if fun(hi_) < target:
            raise ValueError(f"the fingertip chord reaches {fun(hi_):.2f} cm at most, {target:.2f} asked")
        for _ in range(40):
            m = 0.5 * (lo_ + hi_)
            lo_, hi_ = (m, hi_) if fun(m) < target else (lo_, m)
        return 0.5 * (lo_ + hi_)
    rub_cm = cm_pp if cm_pp is not None else level_cm("MDS-UPDRS", 1, "below_floor")
    osc = solve(lambda a: chord(a), rub_cm, 25.0)
    key = int(round(osc * 1e4))

    def series(ph):
        """the hand rotations of the rub at the phases ph (T,) rad under the envelope, the contact kept."""
        _, phis, corr, _ = _pill_contact(side, key)
        a = _rub(side, osc * env, ph, env)
        cc = np.interp(np.mod(ph, 2 * math.pi), np.r_[phis, 2 * math.pi], np.r_[corr, corr[0]])
        a[H.hand_joint(side, "thumb", 1)] = (env * cc, np.zeros_like(cc))
        return _hand_series(side, pose, a)
    ph = phase_of(t, hz, phase0)
    out = b0.with_hands(series(ph), rows)
    params = {"symptom": "pill_rolling", "item": "3.17", "scale": "MDS-UPDRS", "side": side, "frequency_hz": hz,
              "phase_rad": phase0, "rub_peak_deg": osc, "rub_cm_pp": rub_cm, "rub_joints": PILL_SHARE,
              "pill": PILL, "contact": _pill_contact(side, key)[3], "pose": {str(k): v for k, v in pose.items()},
              "readback": "fingertip"}
    srcs = ["mds_updrs", "buijink2012", "module"]
    total = rub_cm
    if pronation_cm_pp is not None:
        pron = solve(lambda d: chord(osc, d), pronation_cm_pp, 120.0)
        out2, tt = tremor(out, side, deg_pp=pron, driver="pronation_supination", frequency_hz=hz, condition="rest",
                          tremor_type="parkinsonian_rest", seed=seed)
        # align the pronation phase with the rub: rebuild the rub at the pronation tremor's phase
        ph = phase_of(t, hz, tt["params"]["phase_rad"])
        out = out2.with_hands(series(ph), rows)
        params.update(pronation_deg_pp=pron, pronation_truth=tt["params"], phase_rad=tt["params"]["phase_rad"])
        total = pronation_cm_pp
    params.update(fingertip_cm_pp=total, grade_of_cm=G.tremor_grade_of_cm(total))
    val = validity(base, out)
    val["hand_limits"] = H.hand_limit_report(out.hand_R())
    return out, truth_record("tremor_catalogue", seed, G.tremor_grade_of_cm(total), "MDS-UPDRS 3.17", params, srcs,
                             val, arrays={"rub_deg": osc * env * np.sin(ph)})


# ---------------------------------------------------------------- dysmetria geometry
def dysmetria_error(side, magnitude_cm, kind="past"):
    """(3,) m: the end-point error of a reach movement to the examiner's finger: 'past' (hypermetria) beyond the
    target along the movement up to the room left before 96 percent of the reach, the rest of the magnitude
    sideways (away from the midline) and down by a third (past-pointing); 'short' (hypometria) short of the
    target along the movement (module geometry)."""
    tg = RC.targets(side)
    d = tg["target"] - tg["nose_target"]
    d = d / np.linalg.norm(d)
    m = magnitude_cm / 100.0
    if kind == "short":
        return -m * d
    sh, reach_m = tg["shoulder"], tg["reach_m"]
    lat = np.array([1.0 if side == "left" else -1.0, -1.0 / 3.0, 0.0])
    lat = lat - (lat @ d) * d
    lat = lat / np.linalg.norm(lat)
    along = 0.0
    for a in np.linspace(0.0, 0.6 * m, 61):
        rest = math.sqrt(max(m * m - a * a, 0.0))
        if np.linalg.norm(tg["target"] + a * d + rest * lat - sh) <= 0.96 * reach_m:
            along = a
    return along * d + math.sqrt(max(m * m - along * along, 0.0)) * lat


# ---------------------------------------------------------------- read-back
def highpass(x, rate, fc):
    from scipy.signal import butter, sosfiltfilt
    sos = butter(4, fc, "highpass", fs=rate, output="sos")
    return sosfiltfilt(sos, np.asarray(x, dtype=np.float64), axis=0)


def oscillation_readback(path, base_path, rate, band=(1.5, 25.0), mask=None, f_hint=None, fmask=None):
    """{'frequency_hz', 'pp_cm_max', 'pp_cm_median'} of a point path (T, 3) m against its level-0 path (or
    high-passed at 0.6 f_hint without it): the spectral peak inside band over the knots of fmask (default mask),
    and the chord along the first principal axis in windows of 1.5 periods, a sixth of a window apart, lying
    inside mask; the median over the windows above a fifth of the largest."""
    p = np.asarray(path, dtype=np.float64)
    D = p - np.asarray(base_path) if base_path is not None else highpass(p, rate, 0.6 * f_hint)
    sel = np.ones(len(D), bool) if mask is None else np.asarray(mask, bool)
    fsel = sel if fmask is None else np.asarray(fmask, bool)
    f = M.dominant_frequency(D[fsel], rate, band)
    n = int(round(1.5 * rate / f)) + 1
    idx = np.flatnonzero(sel)
    pps = []
    for k in range(idx[0], idx[-1] - n + 2, max(1, n // 6)):
        if sel[k:k + n].all():
            pps.append(M.chord_pp(D[k:k + n]))
    pps = np.asarray(pps) if pps else np.zeros(1)
    big = pps[pps > 0.2 * pps.max()] if pps.max() > 0 else pps
    return {"frequency_hz": float(f), "pp_cm_max": float(100 * pps.max()), "pp_cm_median": float(100 * np.median(big))}


def centred_body(base, limb, centre_rad):
    """The base body with the centre shifts of a limb's drives (oscillate's centre_rad (n_drives, T)) applied: the
    clip without its oscillation, the reference of the read-back."""
    if centre_rad is None or not np.any(centre_rad):
        return base
    drives = [Drive(d["joint"], d["axis"] if isinstance(d["axis"], str) else tuple(d["axis"]), d["frame"],
                    d["weight"]) for d in limb["drives"]]
    R = _apply(base.local_R(), drives, list(np.asarray(centre_rad)))
    return base.with_rotations(R, sorted({d.joint for d in drives}))


def readback(spec, body, truth, level0_body=None):
    """The re-measured quantities of a clip (module docstring): per kind of truth['params']['readback']."""
    p = truth["params"]
    kind = p.get("readback")
    rate = body.rate
    if isinstance(kind, list):
        return {k: readback(spec, body, dict(truth, params=dict(p, readback=k)), level0_body) for k in kind}
    if kind == "oscillation_highpass":
        return readback(spec, body, dict(truth, params=dict(p, readback="oscillation")), None)
    if kind == "oscillation":
        out = {}
        for side, lb in p["limbs"].items():
            path = point_path(body, p["point"], side)
            base = level0_body
            if p.get("base_rebuild") == "dystonic_arm":           # the level 0 is the moving dystonic posture
                base = dystonic_arm_base(side)[0]
            elif p.get("base_rebuild") == "reach":                # the reach base of the clip (errors, pace)
                r = p["reach"]
                base = reach_base(side, {int(k): v for k, v in r["endpoint_errors_m"].items()},
                                  r["decomposition"])[0]
            if base is not None:                                  # the clip's own centre (joint-limit and room shifts)
                base = centred_body(base, lb, truth["arrays"].get(f"{side}_centre_rad"))
            bpath = point_path(base, p["point"], side) if base is not None else None
            env = truth["arrays"].get(f"{side}_envelope")
            out[side] = oscillation_readback(path, bpath, rate, mask=None if env is None else env > 0.5,
                                             f_hint=lb["frequency_hz"], fmask=None if env is None else env > 0)
        return out
    if kind == "functional":
        F = p["functional"]
        path = point_path(body, p["point"], "left")
        bpath = point_path(level0_body, p["point"], "left")
        t = body.times().numpy()
        a, z = F["entrain_window_s"]
        own = (t > 1.0) & (t < a - 0.5)
        ent = (t > a + 2.0) & (t < z - 0.5)
        p0, p1 = F["pause_window_s"]
        pause = (t > p0 + 0.4) & (t < p1 - 0.4)
        r_own = oscillation_readback(path, bpath, rate, mask=own)
        r_ent = oscillation_readback(path, bpath, rate, mask=ent)
        D = path - bpath
        return {"own": r_own, "entrained": r_ent,
                "pause_pp_cm": float(100 * M.chord_pp(D[pause])) if pause.any() else None}
    if kind == "standing":
        return standing_readback(body, truth)
    if kind == "orthostatic_myoclonus":
        return ortho_myoclonus_readback(body, level0_body)
    if kind == "orthostatic":
        y = body.joints()[:, 0, 1].numpy()
        y0 = level0_body.joints()[:, 0, 1].numpy() if level0_body is not None else None
        D = (y - y0) if y0 is not None else highpass(y, rate, 5.0)
        f = M.dominant_frequency(D, rate, (8.0, 25.0))
        return {"frequency_hz": f, "pp_cm": float(100 * (np.percentile(D, 99.5) - np.percentile(D, 0.5)))}
    if kind == "fingertip":
        side = p["side"]
        tip = RC.fingertip(body, side)
        tip0 = RC.fingertip(level0_body, side) if level0_body is not None else None
        t = body.times().numpy()
        mask = (t > 1.0) & (t < t[-1] - 1.0)
        return oscillation_readback(tip, tip0, rate, mask=mask, f_hint=p["frequency_hz"])
    if kind == "dysmetria":
        side = p["side"]
        src = body
        if p.get("limbs") and p.get("base_rebuild") == "reach":    # a tremor on top: the movement under it
            r = p["reach"]
            base = reach_base(side, {int(k): v for k, v in r["endpoint_errors_m"].items()}, r["decomposition"])[0]
            src = centred_body(base, p["limbs"][side], truth["arrays"].get(f"{side}_centre_rad"))
        tip = RC.fingertip(src, side)
        errs = []
        for ph in p["reach_phases"]:
            if ph["kind"] == "to_target" and ph.get("error_m") is not None:
                nxt = [q for q in p["reach_phases"] if q["kind"] == "hold_target" and q["cycle"] == ph["cycle"]][0]
                k_end = int(round(nxt["t1_s"] * rate))
                errs.append(RC.endpoint_error_cm(tip, rate, ph["t0_s"], nxt["t1_s"], tip[k_end]))
        return {"endpoint_error_cm": errs}
    if kind == "decomposition":
        side = p["side"]
        tip = RC.fingertip(body, side)
        counts = []
        for ph in p["reach_phases"]:
            if ph["kind"] in ("to_nose", "to_target"):
                counts.append(len(RC.speed_runs(tip, rate, ph["t0_s"], ph["t1_s"])))
        return {"phases_per_movement": counts}
    return {}


# ---------------------------------------------------------------- reach-based operators
def reach_envelope(t, info, profile):
    """(T,) envelope of a tremor on the reach: 'constant' (on from mid-raise to mid-lowering, raised-cosine
    ramps), 'intention' (intention_profile, rising during the raise and fading during the lowering), 'holmes'
    (at least HOLMES_FLOOR over the whole clip: rest, posture and action, the intention profile above it)."""
    from .motions import minimum_jerk
    ph = {p["kind"]: p for p in info["phases"] if p["kind"] in ("raise", "lower")}
    r, lw = ph["raise"], ph["lower"]
    if profile == "constant":
        a = 0.5 * (r["t0_s"] + r["t1_s"])
        z = 0.5 * (lw["t0_s"] + lw["t1_s"])
        return envelope(t, [(a, z)], RAMP_S, t[-1])
    e = intention_profile(t, info["phases"], INTENTION_FLOOR)
    for p in info["phases"]:
        sel = (t >= p["t0_s"]) & (t <= p["t1_s"])
        s = minimum_jerk((t[sel] - p["t0_s"]) / (p["t1_s"] - p["t0_s"]))
        if p["kind"] == "raise":
            e[sel] = s ** 2
        elif p["kind"] == "lower":
            e[sel] = (1 - s) ** 2
        elif p["kind"] == "hold_target" and p["cycle"] is None:
            e[sel] = 1.0
    if profile == "intention":
        return e
    if profile == "holmes":
        return np.maximum(HOLMES_FLOOR, e)
    raise ValueError(f"profile {profile!r}")


def reach_tremor(side, cm_pp, hz, drive, profile, scale, grade, item, tremor_type, seed, wave="sine", errors=None,
                 decomposition=1, point="wrist"):
    """(Body, truth): a tremor of the reaching arm on the reach base (reach_base), its envelope by profile, the face
    a room (FaceRoom: the finger taps the face without entering it), the level read at point ('wrist' or
    'index_tip')."""
    base, info = reach_base(side, errors, decomposition)
    t = base.times().numpy()
    env = reach_envelope(t, info, profile)
    extra = {"reach": {"cycle_s": info["cycle_s"], "n_reaches": info["n_reaches"], "profile": profile,
                       "decomposition": decomposition, "endpoint_errors_m": info["endpoint_errors_m"]},
             "reach_phases": info["phases"], "side": side, "face_gap_cm": FACE_GAP_CM, "base_rebuild": "reach",
             "sources": ["ohberg2019", "sara2006", "rodrigues2017", "flash1985"]}
    return limb_tremor(base, [side], cm_pp, hz, drive, point, env, wave, seed, grade, scale, item, tremor_type,
                       "kinetic", extra, room={side: FaceRoom(base, side)})


def reach_sign(side, kind, grade, seed=0, errors=None, decomposition=1, scale="", item=""):
    """(Body, truth) of a reach with dysmetria (kind 'dysmetria', errors {cycle: (3,) m}) or decomposition (kind
    'decomposition', phases per movement), no tremor."""
    body, info = reach_base(side, errors, decomposition)
    params = {"symptom": kind, "item": item, "scale": scale, "grade": grade, "side": side,
              "reach_phases": info["phases"], "decomposition": decomposition,
              "endpoint_errors_m": info["endpoint_errors_m"], "readback": kind,
              "error_cm": {k: float(100 * np.linalg.norm(v)) for k, v in (errors or {}).items()}}
    base, _ = reach_base(side)
    val = validity(base, body)
    val["hand_limits"] = H.hand_limit_report(body.hand_R())
    val["hand_head"] = FaceRoom(body, side).check(body)
    src = ["sara2006", "icars1997", "ohberg2019", "flash1985", "module"]
    return body, truth_record("reach_sign", seed, grade, f"{scale} {item}".strip(), params, src, val)


def _level0(body, kind, extra=None):
    return body, truth_record("none", None, 0, "level 0", dict({"base": kind, "readback": "none"}, **(extra or {})),
                              (), validity(body, body))


def dystonic_head_base():
    """The seated body with a dystonic neck rotation to the left, TETRAS rotation grade 1 (dystonia.py), held."""
    from .dystonia import dystonic_posture
    return dystonic_posture(static_body("sit", DUR["E7"]), "neck", "rotation_left", "left", amplitude_grade=1,
                            seed=SEED)


# ---------------------------------------------------------------- the gap wave (catalogue rows T04, T05, T08,
# T10, T13, T16, P29, P30, M07; module docstring, 'Gap wave')
GAP_SOURCES = {
    "louis2009": "Louis ED, Frucht SJ, Rios E 2009, Mov Disord 24(4): 626-627, doi 10.1002/mds.22370 (as read in "
                 "docs/proposals/SYMPTOM_CATALOGUE_DRAFT.md, T04): intention tremor in 45 of 117 essential tremor "
                 "cases (38.5 percent)",
    "deuschl2000": "Deuschl G et al. 2000, Brain 123(8): 1568-1580, doi 10.1093/brain/123.8.1568 (abstract, as read in "
                   "docs/proposals/SYMPTOM_CATALOGUE_DRAFT.md, T04): about 25 percent of essential tremor with a "
                   "moderate or severe intention component",
    "robakis2016": "Robakis D, Louis ED 2016, Parkinsonism Relat Disord 22: 98-101, doi "
                   "10.1016/j.parkreldis.2015.11.002 (abstract, as read in docs/proposals/SYMPTOM_CATALOGUE_DRAFT.md, "
                   "T06 and T08): head tremor of 51 essential tremor patients no-no 27.5, yes-yes 17.6, mixed 54.9 "
                   "percent ('round and round' in its title); no phase between the axes",
    "tetras9": "TETRAS item 9, standing tremor (Elble R et al. 2012, Mov Disord 27(12): 1567-1569, doi "
               "10.1002/mds.25162, form v3.1, as read in docs/proposals/SYMPTOM_CATALOGUE_DRAFT.md, E10 and T10): "
               "standing with the knees 10 to 20 cm apart and flexed 10 to 20 deg; 1 barely perceptible, 2 obvious, "
               "mild, no instability, 3 moderate, impairs stance, 4 cannot stand unassisted; no amplitude in cm, no "
               "frequency",
    "hassanvangerpen2016": "Hassan A, van Gerpen JA 2016, Tremor Other Hyperkinet Mov 6: 417, doi 10.7916/d84x584k "
                           "(as read in docs/proposals/SYMPTOM_CATALOGUE_DRAFT.md, M07 and T17): orthostatic "
                           "myoclonus of the legs on standing, 3 to 7 Hz, bursts under 50 ms (weight-bearing "
                           "hyperkinetic disorders)",
    "jerks_twitch": "the twitch of a brief burst of jerks.py (jerks.timing('cortical'): the burst at the geometric "
                    "centre of 25 to 50 ms, jerks.SOURCES['vanderveen2022']; the rise lasting the burst plus "
                    "jerks.TWITCH_S, the return jerks.RETURN_RATIO times the rise: module values of jerks.py)",
    "brady_base": "the level-0 pronation-supination of the exam set (exam clip brady_E4_ps_level0, bradykinesia.py: "
                  f"{G.N_CYCLES} cycles at its module rate of "
                  f"{BK.TASKS['pronation_supination']['level0_hz']:g} Hz, the left arm forward)",
    "gait_base": "the parkinsonian walk of gait.py, MDS-UPDRS 3.10 grade 1, left more affected (catalogue clip "
                 "gait_E8_parkinsonian_g1: zanardi2021, lewek2010, pirker2017 in gait.SOURCES), its reduced left arm "
                 "swing",
}
SOURCES.update(GAP_SOURCES)
# [module] T04, the terminal crescendo of essential intention tremor on the finger-nose-finger (no envelope was
# sourced, catalogue T04): the level's share mid-path (floor), rising as the level 0's index fingertip comes within
# near_cm of the nose target or of the examiner's finger, to the full level within plateau_cm of it (terminal_envelope)
TERMINAL = {"floor": 0.35, "near_cm": 10.0, "plateau_cm": 1.5}
# [module] T08, the mixed head tremor (robakis2016 gives the share of mixed tremor, no phase): the yes-yes component
# at pitch_share of the no-no chord at the nose, phase_deg behind it (an ellipse: 'round and round')
MIXED_HEAD = {"pitch_share": 0.6, "phase_deg": 90.0}
# [module] T16, the isometric tremor of the clenched fist (bhatia2018: while making a fist; no scale, no amplitude or
# frequency sourced): seated, the left forearm on the chair arm ('sit': the arm not held against gravity, unlike the
# postural tremor of TETRAS 4a), the hand open (the rest hand) until fist_at_s, closing into the fist of
# handtasks.FIST over close_s (handtasks.hand_at 'hand_movements', share 1 to 0 on a half cosine), then clenched; the
# tremor ramping in over RAMP_S once the fist is closed (none before: the tremor depends on the contraction), the
# wrist flexing and extending at the essential tremor frequency, its chord at the front of the fist the TETRAS
# upper-limb grade-2 band level (a visible, mild tremor)
ISOMETRIC = {"duration_s": 20.0, "hz": HZ["essential"], "level_grade": 2.0, "fist_at_s": 5.0, "close_s": 0.5}
# [module] P29, the seated rest tremor of the leg (catalogue P29: excursion module, unsourced; frequency as P27, the
# exam's rest frequency exam.TREMOR_HZ['rest']): 'sit' rests the foot on its ball, the heel above the floor; the level
# 0 lowers the heel onto it (the planted dorsiflexion heel_down_deg: the lowest skinned heel vertex within
# contact.MESH_TOL_M of the floor), and the forefoot rises and falls about the planted heel (ankle dorsiflexion from
# that foot, one-sided: the toes back on the floor at every cycle), read at the toe vertex; n_grid: the planted
# solutions tabulated over the angle and interpolated; the mesh checks (contact.sole_report, foot_pose_report) run on
# the first mesh_check_s (at 5 Hz on the 100 Hz clock the phase steps by a fixed angle and every pose of the clip
# recurs every 20 knots)
LEG_REST = {"hz": TREMOR_HZ["rest"], "n_grid": 49, "mesh_check_s": 2.0}
LEG_ROWS = {"left": (1, 4, 7), "right": (2, 5, 8)}    # hip, knee, ankle
# [module] P30: the rest tremor of the left forearm (elbow, read at the wrist) on the walk of gait_base, at the exam's
# rest frequency (exam.TREMOR_HZ['rest']); the blind read (swing_removed_readback) takes the swing band out with a
# high-pass at swing_cut_hz
WALK_REST = {"hz": TREMOR_HZ["rest"], "swing_cut_hz": 3.0}
WALK_SOURCE = "gait_E8_parkinsonian_g1"
PS_LEVEL0 = "brady_E4_ps_level0"
# [module] T10, TETRAS item 9 (tetras9: no amplitude or frequency published): the TETRAS standing posture (the knees
# knee_flex_deg flexed, the ankles ankle_width_cm apart, the feet flat), the knees flexing and extending about it by
# knee_pp_deg peak to peak with the feet planted (the pelvis moved vertically, the legs solved by ik.hinge_leg), the
# trunk flexing and extending in phase (the spine, STANDING_SPINE, read at the neck joint, trunk_cm peak to peak);
# grade 3 'impairs stance' shown by weight shifts between the feet (weight_shifts, those of the orthostatic clip) and
# by STANDING_SWAY; the trunk's level at grade 3 bounded by the friction the floor must supply (jerks.balance_block:
# 2 cm at 5 Hz asked 0.61, above jerks.FRICTION_MAX); ankle_width_cm sets the knees near the middle of TETRAS's 10 to
# 20 cm; the pelvis carried forward (standing_centre_m) so that the posture's centre of mass lies midway between the
# heels and the toes (open-body port, wave 2: on the open body it stood 6.4 cm ahead of the heels and 19.6 behind the
# toes, and grade 3's sway took the extrapolated centre of mass behind the heels; scripts/openbody_wave2_checks.py)
STANDING = {"knee_flex_deg": 15.0, "ankle_width_cm": 12.0}
# [module] T10 grade 3, the unsteadiness (no amplitude sourced): the body swaying forward and backward over the planted
# feet as an inverted pendulum about the ankles (physio.stand_sway: the whole body turns about the line through the
# ankles, the legs re-solved with every knee angle kept), the sum of two sines at hz with phases from the seed, scaled
# so that the head joint travels head_cm_pp peak to peak over the clip, ramped in over ramp_s; it narrows the
# extrapolated centre of mass's margins inside the feet (jerks.balance_block) against the posture's. head_cm_pp: at
# least twice the head's forward and backward travel under the standing background layer of its level 0 (physio,
# 4.24 cm over the 15 s of standing_tremor_E10_posture_g0 with its series seed, read by
# tests/test_synth_tremors_more.py), so that grade 3 stays beyond the quiet sway once the level 0 takes the layer
# (wave SH1d part B, review M3c: the sway of 3 cm at the pelvis with the trunk upright slid the body as a block,
# inside the physiological sway). The arms guard the balance: both shoulders abduct by arm_guard_deg times
# arm_base_share plus (1 - arm_base_share) times the squared sway share, ramped in with the sway (module values).
STANDING_SWAY = {"head_cm_pp": 9.0, "hz": (0.21, 0.34), "ramp_s": 2.0, "arm_guard_deg": 15.0, "arm_base_share": 0.6}
STANDING_LEVEL = {1: {"knee_pp_deg": 1.5, "trunk_cm": 0.3, "hz": 6.0, "shifts": False},
                  2: {"knee_pp_deg": 4.0, "trunk_cm": 1.0, "hz": 6.0, "shifts": False},
                  3: {"knee_pp_deg": 8.0, "trunk_cm": 1.5, "hz": 5.0, "shifts": True}}
STANDING_WORDS = {1: "barely perceptible", 2: "obvious, mild, no instability", 3: "moderate, impairs stance"}
STANDING_SPINE = ((3, 0.40), (6, 0.35), (9, 0.25))  # [module] the trunk's angle shares (gait.SPINE_SHARE)
NECK_POINT = ("joint", NECK)
# [module] M07, orthostatic myoclonus (hassanvangerpen2016: 3 to 7 Hz, bursts under 50 ms): on the knees-unlocked
# stance of the orthostatic level 0, brief knee-extension twitches (the pelvis raised by rise_cm times a share drawn
# uniform in amp_share, the legs solved with the feet planted), the twitch of jerks_twitch, the intervals between
# twitch onsets whole knots drawn uniform from ceil(rate / rate_hz[1]) to floor(rate / rate_hz[0]) (every realized
# rate inside rate_hz), from onset_s on
ORTHO_MYO = {"onset_s": 2.0, "rise_cm": 0.5, "amp_share": (0.6, 1.0), "rate_hz": (3.0, 7.0)}


def _gap_level0(body, kind, extra=None):
    """_level0 with the hand limits of a HandBody."""
    body, truth = _level0(body, kind, extra)
    if H.has_hands(body):
        truth["validity"]["hand_limits"] = H.hand_limit_report(body.hand_R())
    return body, truth


def windowed_pp(x, rate, f_hz, mask=None):
    """{'pp_max', 'pp_median'}: the peak-to-peak (same unit) of a (T,) or (T, 3) signal (the chord along its first
    principal axis for a path) in windows of 1.5 periods of f_hz, a sixth of a window apart, lying inside mask; the
    median over the windows above a fifth of the largest (oscillation_readback's windows)."""
    x = np.asarray(x, dtype=np.float64)
    x = x[:, None] if x.ndim == 1 else x
    n = int(round(1.5 * rate / f_hz)) + 1
    sel = np.ones(len(x), bool) if mask is None else np.asarray(mask, bool)
    idx = np.flatnonzero(sel)
    pps = [M.chord_pp(x[k:k + n]) for k in range(idx[0], idx[-1] - n + 2, max(1, n // 6)) if sel[k:k + n].all()]
    pps = np.asarray(pps) if pps else np.zeros(1)
    big = pps[pps > 0.2 * pps.max()] if pps.max() > 0 else pps
    return {"pp_max": float(pps.max()), "pp_median": float(np.median(big))}


# -- T04: essential tremor with a terminal crescendo on the finger-nose-finger
def terminal_envelope(base, info, side="left", floor=TERMINAL["floor"], near_cm=TERMINAL["near_cm"],
                      plateau_cm=TERMINAL["plateau_cm"]):
    """((T,) envelope, (T,) m distance) of TERMINAL on a reach base (reach_base): d the distance of the base's index
    fingertip (skinned) to the nearer of the nose target and the examiner's finger (reach.targets), the envelope
    floor + (1 - floor) (1 - max(d - plateau, 0) / near)^2 (the square clipped to [0, 1]: smooth at near, 1 within
    the plateau), times the 'constant' gate of reach_envelope (on from mid-raise to mid-lowering)."""
    t = base.times().numpy()
    gate = reach_envelope(t, info, "constant")
    tip = RC.fingertip(base, side)
    tg = RC.targets(side)
    d = np.minimum(np.linalg.norm(tip - tg["nose_target"], axis=1), np.linalg.norm(tip - tg["target"], axis=1))
    near = np.clip(1.0 - np.maximum(d - plateau_cm / 100.0, 0.0) / (near_cm / 100.0), 0.0, 1.0) ** 2
    return gate * (floor + (1.0 - floor) * near), d


def terminal_reach_tremor(side, cm_pp, hz, grade, seed, drive="elbow", point="index_tip"):
    """(Body, truth): T04, the kinetic tremor of essential tremor on the reach (as tremor_E3_essential_*) whose
    envelope is terminal_envelope: the level, read at the index fingertip, near the nose and the examiner's finger,
    TERMINAL['floor'] of it mid-path; the face a room (FaceRoom)."""
    base, info = reach_base(side)
    env, dist = terminal_envelope(base, info, side)
    extra = {"reach": {"cycle_s": info["cycle_s"], "n_reaches": info["n_reaches"], "profile": "terminal",
                       "decomposition": 1, "endpoint_errors_m": info["endpoint_errors_m"]},
             "reach_phases": info["phases"], "side": side, "face_gap_cm": FACE_GAP_CM, "base_rebuild": "reach",
             "terminal": dict(TERMINAL, rule="floor + (1 - floor) (1 - max(d - plateau, 0) / near)^2, d the level 0's "
                                             "fingertip distance to the nearer target"),
             "sources": ["ohberg2019", "sara2006", "rodrigues2017", "flash1985", "louis2009", "deuschl2000",
                         "buijink2012", "deuschl1998"]}
    body, truth = limb_tremor(base, [side], cm_pp, hz, drive, point, env, "sine", seed, grade, "TETRAS upper limb",
                              "4c", "essential", "kinetic with a terminal crescendo (intention)", extra,
                              room={side: FaceRoom(base, side)})
    truth["arrays"]["target_distance_m"] = dist
    return body, truth


# -- T05: simple kinetic tremor during pronation-supination
@functools.lru_cache(maxsize=1)
def _ps_cached():
    from .exam import exam_set
    return {s.clip_id: s for s in exam_set()}[PS_LEVEL0].build()


def ps_base():
    """(Body, truth): a copy of the exam's level-0 pronation-supination (PS_LEVEL0, SOURCES['brady_base'])."""
    b, t = _ps_cached()
    return b.copy(), t


def ps_kinetic(cm_pp, hz, grade, seed, side="left"):
    """(Body, truth): T05, a simple kinetic tremor (bhatia2018: during a non-target-directed movement) of the
    left hand on the pronation-supination of PS_LEVEL0: the hand flexing and extending at the wrist with the
    shoulder joining from 3 cm (drives_of 'postural', as the essential postural tremor of T01: the elbow hinge of the
    outstretched arm moves the hand sideways, review B2), so that the tremor's direction turns with the palm, read
    at the middle fingertip, on from the first cycle to the last (RAMP_S ramps); task_removed_readback reads it
    without the level 0. Amplitude on the TETRAS upper-limb bands (no item of its own; the FTM action item waits for
    its 1988 chapter, catalogue SC6)."""
    if side != "left":
        raise ValueError("the level-0 pronation-supination moves the left arm")
    base, bt = ps_base()
    p = bt["params"]
    t = base.times().numpy()
    env = envelope(t, [(p["start_s"], p["end_s"])], RAMP_S, t[-1])
    extra = {"sources": ["bhatia2018", "brady_base", "buijink2012", "deuschl1998"],
             "task": {"base": PS_LEVEL0, "task": "pronation_supination", "task_hz": 1.0 / p["period_s"],
                      "window_s": [p["start_s"], p["end_s"]], "amplitude0_deg": p["amplitude0_deg"]},
             "scale_note": "TETRAS upper-limb bands read on a non-target movement (no item of its own; the FTM action "
                           "item is not used before its 1988 chapter is read, catalogue SC6)"}
    return limb_tremor(base, [side], cm_pp, hz, "postural", "middle_tip", env, "sine", seed, grade,
                       "TETRAS upper limb", "bands on pronation-supination", "essential",
                       "kinetic (simple, non-target-directed)", extra)


def task_removed_readback(body, side="left", hz=HZ["essential"], cut_hz=3.0, window_s=None):
    """{'frequency_hz', 'pp_cm_max', 'pp_cm_median'}: the blind read of a tremor of the hand during
    pronation-supination: the middle fingertip (skinned) against the wrist joint in the frame of the forearm (joint
    18 or 19: the forearm's share of the turn out; the hand's share moves the fingertip, near the forearm's axis,
    by millimetres), high-passed at cut_hz (the task band at 2 Hz out), read as oscillation_readback inside
    window_s (s, default the whole clip less a second at each end)."""
    J, Gl = body.fk()
    tip = H.body_points(body, [H.tip_vertices()[side]["middle"]])[:, 0]
    e = ELBOW[side]
    loc = np.einsum("tji,tj->ti", Gl[:, e].numpy(), tip - J[:, WRIST[side]].numpy())
    D = highpass(loc, body.rate, cut_hz)
    t = body.times().numpy()
    a, z = window_s or (1.0, t[-1] - 1.0)
    return oscillation_readback(D, np.zeros_like(D), body.rate, mask=(t > a) & (t < z), f_hint=hz)


# -- T08: mixed head tremor
def mixed_head_tremor(cm_pp, hz, grade, seed, posture="sit", duration_s=DUR["E7"]):
    """(Body, truth): T08, a head tremor with a no-no component of cm_pp at the nose (the level: the largest chord
    in one plane, TETRAS item 1) and a yes-yes component of MIXED_HEAD['pitch_share'] of it, MIXED_HEAD['phase_deg']
    behind, at one frequency: the nose runs an ellipse. Two oscillate passes (head_yaw, then head_pitch on its
    output), neck and head in equal shares; the readback reads the first pass's limb."""
    base = static_body(posture, duration_s)
    t = base.times().numpy()
    rng = as_rng(seed)
    phase0 = float(rng.uniform(0.0, 2 * math.pi))
    ph = phase_of(t, hz, phase0)
    lag = math.radians(MIXED_HEAD["phase_deg"])
    env = np.ones(base.T)
    nose = point_spec("nose")
    b1, i1, a1 = oscillate(base, drives_of("head_yaw"), cm_pp, np.sin(ph), env, nose, hz)
    pcm = MIXED_HEAD["pitch_share"] * cm_pp
    out, i2, a2 = oscillate(b1, drives_of("head_pitch"), pcm, np.sin(ph - lag), env, nose, hz)
    params = {"symptom": "tremor_catalogue", "item": "1", "scale": "TETRAS head", "grade": grade,
              "tremor_type": "essential", "condition": "postural", "sides": ["left"], "drive": "head_yaw, head_pitch",
              "point": "nose", "direction": "mixed (no-no and yes-yes)",
              "limbs": {"left": dict(i1, frequency_hz=float(hz), cm_pp=float(cm_pp),
                                     cm_pp_at_unit_waveform=float(cm_pp), waveform="sine", phase_rad=phase0)},
              "mixed": {"pitch": dict(i2, cm_pp=float(pcm)), "pitch_share": MIXED_HEAD["pitch_share"],
                        "phase_deg": MIXED_HEAD["phase_deg"], "rule": "pitch = sin(phase - phase_deg), yaw = "
                        "sin(phase), the yaw pass first"},
              "envelope_on_share": 1.0, "readback": "oscillation", "sources": ["robakis2016", "buijink2012"]}
    arrays = {"left_angle_rad": a1["angle_rad"], "left_centre_rad": a1["centre_rad"], "left_envelope": env,
              "left_waveform": a1["waveform"], "pitch_angle_rad": a2["angle_rad"],
              "pitch_centre_rad": a2["centre_rad"]}
    return out, truth_record("tremor_catalogue", seed, grade, "TETRAS head 1", params,
                             ("tetras2012", "elble2006", "robakis2016", "buijink2012", "module"), validity(base, out),
                             arrays=arrays)


# -- T13: essential-type rest tremor
REST_HANDS_NOTE = ("the hands rest as in 'sit' (base.py): the elbows and the proximal third of the forearms on the "
                   "chair arms, the hands held level beyond them, not hanging or on the thighs; a supported rest "
                   "posture is a shared change of base.py, not made here")


def essential_rest(cm_pp, grade, seed):
    """(Body, truth): T13, a rest tremor of essential type (essential tremor plus, bhatia2018): both hands
    seated at rest, flexing and extending at the wrist, each at its own essential frequency (HZ 'essential' and
    'essential_right'), read at the middle fingertips on the MDS-UPDRS 3.17 bands; the rest posture of 'sit'
    (REST_HANDS_NOTE, in the truth)."""
    return limb_tremor(static_body("sit", DUR["E1"]), ["left", "right"], cm_pp,
                       {"left": HZ["essential"], "right": HZ["essential_right"]}, "wrist", "middle_tip", None, "sine",
                       seed, grade, "MDS-UPDRS", "3.17", "essential", "rest",
                       {"sources": ["bhatia2018", "buijink2012", "deuschl1998"],
                        "et_plus": "essential tremor with a rest tremor: essential tremor plus (bhatia2018)",
                        "rest_note": REST_HANDS_NOTE})


# -- T16: isometric tremor of the clenched fist
def fist_share(t, fist_at_s=ISOMETRIC["fist_at_s"], close_s=ISOMETRIC["close_s"]):
    """(T,): the hand-movements share (handtasks: 1 the open hand, 0 the fist) of the T16 scene at times t (s): 1
    until fist_at_s, a half cosine down to 0 over close_s, 0 after."""
    u = np.clip((np.asarray(t, dtype=np.float64) - fist_at_s) / close_s, 0.0, 1.0)
    return 0.5 + 0.5 * np.cos(np.pi * u)


def fist_base(side="left", duration_s=ISOMETRIC["duration_s"], held=False):
    """HandBody: 'sit' (the left forearm on the chair arm, the elbow flexed; the right side at rest) with the left
    hand open (the rest hand), closing into the fist of handtasks.FIST (fist_share; handtasks.hand_at
    'hand_movements', whose closed hand is FIST); held: the fist from the first knot."""
    from .handtasks import FIST, TASKS, hand_at, task_rows
    if side != "left":
        raise ValueError("the fist scene closes the left hand")
    if TASKS["hand_movements"]["closed"] is not FIST:
        raise ValueError("the hand-movements closed hand is not handtasks.FIST")
    b = H.hands_of(static_body("sit", duration_s))
    share = np.zeros(b.T) if held else fist_share(b.times().numpy())
    R = b.hand_R().clone()
    rows = task_rows(side)
    R[:, rows] = hand_at("hand_movements", side, share)[:, rows]
    return b.with_hands(R)


@functools.lru_cache(maxsize=2)
def _fist_knuckle(side="left"):
    """(vertex id, (3,) offset in the wrist frame): the vertex of the middle finger's proximal segment farthest from
    the wrist joint on the clenched fist (fist_base, held), at the front of the fist; rigid on the wrist while the
    fist is clenched (point_spec 'fist_knuckle')."""
    from .viewer import dominant_joint
    b = fist_base(side, 0.01, held=True)
    ids = np.flatnonzero(dominant_joint("neutral") == H.hand_joint(side, "middle", 0))
    V = H.body_points(b, ids, knots=[0])[0]
    J, Gl = b.fk()
    w = WRIST[side]
    i = int(np.argmax(np.linalg.norm(V - J[0, w].numpy(), axis=1)))
    return int(ids[i]), torch.as_tensor((V[i] - J[0, w].numpy()) @ Gl[0, w].numpy(), dtype=F64)


def fist_envelope(t, I=ISOMETRIC):
    """(T,) the T16 tremor envelope: 0 until the fist is closed (fist_at_s + close_s), then a RAMP_S ramp to 1."""
    closed = I["fist_at_s"] + I["close_s"]
    return envelope(t, [(closed + RAMP_S / 2.0, float(t[-1]))], RAMP_S, float(t[-1]))


def isometric_fist(seed, side="left", level0=False):
    """(HandBody, truth): T16, the isometric tremor of the clenched fist (ISOMETRIC; the squeeze and its force are
    not shown): seated, the forearm on the chair arm, the hand closing into a fist (fist_base), the tremor ramping in
    once the fist is closed (fist_envelope), the fist flexing and extending at the wrist ('wrist' drives), its chord
    read at the front of the fist (point 'fist_knuckle'); a presence clip (no scale). level0: the same hand closing,
    no tremor."""
    base = fist_base(side)
    t = base.times().numpy()
    scene = {"posture": "sit: the left forearm on the chair arm, the elbow flexed (not held against gravity)",
             "fist_share": "1 open until fist_at_s, a half cosine to 0 (the fist) over close_s",
             "fist_closed_s": ISOMETRIC["fist_at_s"] + ISOMETRIC["close_s"]}
    if level0:
        out = _gap_level0(base, "seated, the left hand closing into a fist, the forearm on the chair arm",
                          {"isometric": dict(ISOMETRIC), "scene": scene})
        out[1]["arrays"]["fist_share"] = fist_share(t)
        return out
    cm = band_level("TETRAS upper limb", ISOMETRIC["level_grade"])
    env = fist_envelope(t)
    body, truth = limb_tremor(base, [side], cm, ISOMETRIC["hz"], "wrist", "fist_knuckle", env, "sine", seed, None,
                              "TETRAS upper limb", "none", "isometric", "isometric (the fist clenched, the forearm "
                              "supported)",
                              {"sources": ["bhatia2018", "buijink2012"],
                               "isometric": dict(ISOMETRIC, level_cm=cm, level_rule="the TETRAS upper-limb grade-2 "
                                                 "band level (module choice: no isometric amplitude was read)"),
                               "scene": scene,
                               "not_shown": "the squeeze and its force (no examiner's hand, no object)"})
    p = truth["params"]
    p.update(scale="none (presence; isometric tremor, bhatia2018)", grade=1, item="presence")
    truth.update(grade=1, scale="presence")
    truth["arrays"]["fist_share"] = fist_share(t)
    return body, truth


# -- P29: rest tremor of the leg and foot, seated
@functools.lru_cache(maxsize=2)
def toe_vertex(side="left"):
    """The toe vertex: the most anterior rest-mesh vertex of those whose dominant joint is the foot joint of side
    (10 left, 11 right), the tip of the big toe (point 'toe')."""
    _, v, dom = H._rest()
    ids = np.flatnonzero(dom == (10 if side == "left" else 11))
    return int(ids[np.argmax(v[ids, 2])])


def planted_ankle(side, degs):
    """((n, 22, 3, 3) numpy rotations, plant info): the seated body ('sit') at n knots whose ankle of side is
    dorsiflexed by degs (n,) deg about its parent-frame x axis (the toe-tapping axis of bradykinesia.py, -x), the
    foot planted (contact.plant_ankle: the foot rolls on its heel, the hip fixed, the knee and the hip following)."""
    from . import contact as CT
    degs = np.asarray(degs, dtype=np.float64)
    n = max(len(degs), 2)
    d = np.r_[degs, np.zeros(n - len(degs))]
    base = static_body("sit", (n - 1) / 100.0)
    a = LEG_ROWS[side][2]
    R = base.local_R().numpy().copy()
    R[:, a] = axis_rotation("x", torch.as_tensor(-np.radians(d))).numpy() @ R[:, a]
    knots = CT.stance_knots(base, side) & (d != 0)
    Rp, ok, info = CT.plant_ankle(base, R, side, knots)
    if not ok.all():
        raise ValueError(f"the {side} foot cannot stay planted at {int((~ok).sum())} of the angles")
    return Rp[:len(degs)], info


def _seated_leg(side, R):
    """Body: 'sit' at len(R) knots (at least two) with the leg rows of side (LEG_ROWS) from R (n, 22, 3, 3)."""
    R = torch.as_tensor(np.asarray(R, dtype=np.float64))
    return static_body("sit", (len(R) - 1) / 100.0).with_rotations(R, list(LEG_ROWS[side]))


def heel_gap_m(side, degs):
    """(n,) m: the height above the seated floor (the lowest skinned sole point of 'sit', contact.skin_points) of the
    lowest skinned vertex of the heel set (contact.heel_toe) of side at the planted dorsiflexions degs (n,) deg."""
    from . import contact as CT
    d = np.r_[np.asarray(degs, dtype=np.float64).reshape(-1), 0.0]       # the resting foot last: the floor
    b = _seated_leg(side, planted_ankle(side, d)[0])
    x = CT.skin_points(b.local_R().numpy(), b.transl.numpy(), b.betas.numpy(), b.gender, side)
    heel = CT.heel_toe(int(b.gender))[side][0]
    return x[:-1, heel, 1].min(1) - x[-1, :, 1].min()


@functools.lru_cache(maxsize=2)
def heel_down_deg(side="left", hi_deg=10.0, iters=24):
    """deg: the smallest planted dorsiflexion (planted_ankle) at which the heel of side is down, its lowest skinned
    vertex within contact.MESH_TOL_M of the seated floor (heel_gap_m), by bisection on [0, hi_deg]: 'sit' rests the
    foot on its ball, the heel above the floor; dorsiflexed and planted, the foot rolls on its ball (the knee
    lowering) until the heel lands, then about the heel (the knee nearly still)."""
    from . import contact as CT
    if heel_gap_m(side, [hi_deg])[0] > CT.MESH_TOL_M:
        raise ValueError(f"the {side} heel is not down at {hi_deg:g} deg")
    lo, hi = 0.0, float(hi_deg)
    for _ in range(iters):
        m = 0.5 * (lo + hi)
        if heel_gap_m(side, [m])[0] > CT.MESH_TOL_M:
            lo = m
        else:
            hi = m
    return hi


def leg_rest_level0(side="left", duration_s=DUR["E1"]):
    """(Body, truth): the level 0 of P29 (tremor_E1_leg_rest_g0): 'sit' with the ankle of side dorsiflexed by
    heel_down_deg and planted (the heel lowered onto the floor, the foot flat), held; the other leg as in 'sit'.
    Validity against 'sit'; the moved foot's contact on the mesh (contact.sole_report, planted)."""
    from ..body.rotations import rot6d_to_matrix
    from . import contact as CT
    from .body import crop
    base = static_body("sit", duration_s)
    dh = heel_down_deg(side)
    Rg, pinfo = planted_ankle(side, [dh, dh])
    rows = list(LEG_ROWS[side])
    R = base.local_R().clone()
    R[:, rows] = rot6d_to_matrix(matrix_to_rot6d(torch.as_tensor(Rg[0, rows])))[None]
    out = base.with_rotations(R, rows)
    out = Body(out.rot6d, out.transl, contact_labels(out), out.betas, out.gender, out.rate)
    val = validity(base, out)
    cb, co = crop(base, 0, 2), crop(out, 0, 2)
    val["contact"] = dict(CT.sole_report(cb, co, side, planted=True), knots="0 to 1 (the pose is held)")
    params = {"base": "sit, the heel down", "readback": "none", "side": side, "heel_down_deg": float(dh),
              "heel_gap_sit_mm": 1000.0 * float(heel_gap_m(side, [0.0])[0]),
              "heel_gap_mm": 1000.0 * float(heel_gap_m(side, [dh])[0]),
              "plant": pinfo, "foot_pose": CT.foot_pose_report(cb, co, side),
              "foot": "'sit' rests the foot on its ball, the heel above the floor; here the ankle is dorsiflexed and "
                      "planted until the heel is down (heel_down_deg), the foot flat, the knee lowered with it"}
    return out, truth_record("none", None, 0, "level 0", params, ("module",), val)


def _leg_table(side, degs):
    """(n, 3) m: the toe vertex (skinned) at the planted dorsiflexions degs (n,) deg."""
    return H.body_points(_seated_leg(side, planted_ankle(side, degs)[0]), [toe_vertex(side)])[:, 0]


@functools.lru_cache(maxsize=16)
def _toe_angle(side, cm_key, max_deg=40.0, n=161):
    """(deg, chord cm): the planted dorsiflexion beyond heel_down_deg (the heel down: the level 0 of P29) whose toe
    vertex (skinned) lies cm_key / 1e9 cm from its place at heel_down_deg: read on a table of n angles up to max_deg
    beyond it (the chord rising with the angle), then secant steps on the exact chord to 1e-10 cm."""
    cm = cm_key / 1e9
    dh = heel_down_deg(side)
    grid = np.linspace(0.0, max_deg, n)
    P = _leg_table(side, dh + grid)
    ch = 100.0 * np.linalg.norm(P - P[0], axis=1)
    if not (np.diff(ch) > 0).all() or ch[-1] < cm:
        raise ValueError(f"the planted forefoot's toe chord is not monotone or reaches {ch[-1]:.2f} cm at most, "
                         f"{cm:.2f} asked")

    def f(deg):
        Q = _leg_table(side, [dh, dh + deg])
        return 100.0 * float(np.linalg.norm(Q[1] - Q[0])) - cm
    x0 = float(np.interp(cm, ch, grid))
    x1 = x0 + 1e-3
    f0, f1 = f(x0), f(x1)
    for _ in range(12):
        if abs(f1) < 1e-10 or f1 == f0:
            break
        x0, x1 = x1, x1 - f1 * (x1 - x0) / (f1 - f0)
        f0, f1 = f1, f(x1)
    if abs(f1) > 1e-8:
        raise ValueError(f"the toe chord did not converge ({f1:.2e} cm off)")
    return x1, f1 + cm


def leg_motion_pp_cm(body, level0_body, side="left", knots=None):
    """{'toe', 'heel', 'knee'}: cm, the peak-to-peak chords (measure.chord_pp) of the toe vertex, of the mean of the
    heel set (contact.heel_toe, skinned) and of the knee joint against the level 0, over knots (default all)."""
    from . import contact as CT
    k = np.arange(body.T) if knots is None else np.asarray(knots)
    out = {"toe": 100.0 * M.chord_pp((point_path(body, "toe", side) - point_path(level0_body, "toe", side))[k])}
    heel = CT.heel_toe(int(body.gender))[side][0]
    hs = []
    for b in (body, level0_body):
        x = CT.skin_points(b.local_R().numpy()[k], b.transl.numpy()[k], b.betas.numpy(), b.gender, side)
        hs.append(x[:, heel].mean(1))
    out["heel"] = 100.0 * M.chord_pp(hs[0] - hs[1])
    kn = LEG_ROWS[side][1]
    out["knee"] = 100.0 * M.chord_pp((body.joints()[:, kn] - level0_body.joints()[:, kn]).numpy()[k])
    return {key: float(v) for key, v in out.items()}


def leg_rest_tremor(cm_pp, grade, seed, side="left", hz=LEG_REST["hz"], duration_s=DUR["E1"]):
    """(Body, truth): P29, the seated rest tremor of the leg (LEG_REST) on its level 0 (leg_rest_level0, the heel
    down): the ankle dorsiflexed by heel_down_deg + deg (1 - cos phase) / 2 (one-sided: the forefoot leaves the floor
    and comes back to it, about the planted heel), deg solved so that the toe vertex moves cm_pp between the extremes
    (_toe_angle), the planted leg (planted_ankle) tabulated on LEG_REST['n_grid'] angles and interpolated (rot6d,
    re-orthonormalized) at every knot. Validity against the level 0; the skate check covers the other foot; the
    moved foot's contact is validity['contact'] (contact.sole_report, planted, on the mesh, the floor of 'sit'); the
    toe, heel and knee chords against the level 0 (leg_motion_pp_cm) in params['leg_motion_pp_cm']."""
    from ..body.rotations import rot6d_to_matrix
    from . import contact as CT
    from .body import crop
    sit = static_body("sit", duration_s)
    l0, _ = leg_rest_level0(side, duration_s)
    t = sit.times().numpy()
    rng = as_rng(seed)
    phase0 = float(rng.uniform(0.0, 2 * math.pi))
    env = np.ones(sit.T)
    dh = heel_down_deg(side)
    deg, chord = _toe_angle(side, int(round(cm_pp * 1e9)))
    u = 0.5 * (1.0 - np.cos(phase_of(t, hz, phase0)))
    o = dh + deg * env * u
    grid = np.linspace(dh, dh + deg, LEG_REST["n_grid"])
    Rg, pinfo = planted_ankle(side, grid)
    rows = list(LEG_ROWS[side])
    r6 = matrix_to_rot6d(torch.as_tensor(Rg[:, rows]))
    x = (o - dh) / deg * (len(grid) - 1)
    i0 = np.clip(np.floor(x).astype(np.int64), 0, len(grid) - 2)
    w = torch.as_tensor(x - i0)[:, None, None]
    R = sit.local_R().clone()
    R[:, rows] = rot6d_to_matrix((1.0 - w) * r6[i0] + w * r6[i0 + 1])
    out = sit.with_rotations(R, rows)
    out = Body(out.rot6d, out.transl, contact_labels(out), out.betas, out.gender, out.rate)
    other = (1,) if side == "left" else (0,)
    val = validity(l0, out, feet=other)
    kc = min(sit.T, int(round(LEG_REST["mesh_check_s"] * sit.rate)) + 1)
    cs, c0, co = crop(sit, 0, kc), crop(l0, 0, kc), crop(out, 0, kc)
    val["contact"] = dict(CT.sole_report(cs, co, side, planted=True), knots=f"0 to {kc - 1} (LEG_REST mesh_check_s)")
    val["continuity"] = CT.step_report(l0, out, side)
    gap = heel_gap_m(side, grid)
    a = LEG_ROWS[side][2]
    limb = {"drives": [{"joint": int(a), "axis": "x", "frame": "parent", "weight": -1.0}],
            "point": "toe vertex (skinned)", "point_joint": 10 if side == "left" else 11, "frequency_hz": float(hz),
            "cm_pp": float(cm_pp), "cm_pp_at_unit_waveform": float(cm_pp), "chord_cm_at_full_envelope": [chord, chord],
            "waveform": "sine", "one_sided": "heel_down_deg + deg (1 - cos phase) / 2: from the flat foot up and back",
            "peak_deg": float(deg), "heel_down_deg": float(dh), "phase_rad": phase0}
    params = {"symptom": "tremor_catalogue", "item": "3.17", "scale": "MDS-UPDRS", "grade": grade,
              "tremor_type": "parkinsonian_rest", "condition": "rest", "sides": [side], "drive": "ankle (planted)",
              "point": "toe", "limbs": {side: limb}, "envelope_on_share": 1.0, "readback": "oscillation",
              "plant": pinfo, "plant_grid": {"n": LEG_REST["n_grid"], "deg_range": [float(dh), float(dh + deg)],
                                             "interpolation": "rot6d linear, re-orthonormalized"},
              "foot_pose": dict(CT.foot_pose_report(c0, co, side), knots=f"0 to {kc - 1}", against="the level 0"),
              "heel_gap_mm_range": [float(1000 * gap.min()), float(1000 * gap.max())],
              "leg_motion_pp_cm": leg_motion_pp_cm(co, c0, side),
              "foot": "on the level 0 (the heel down), the forefoot rising and falling about the planted heel "
                      "(contact.plant_ankle); the skate check covers the other foot, the moved foot's contact is "
                      "validity['contact']",
              "level0": "tremor_E1_leg_rest_g0", "sources": ["bhatia2018", "buijink2012"]}
    return out, truth_record("tremor_catalogue", seed, grade, "MDS-UPDRS 3.17", params,
                             ("mds_updrs", "elble2006", "bhatia2018", "buijink2012", "module"), val,
                             arrays={f"{side}_envelope": env, "ankle_offset_deg": o - dh, "ankle_deg": o})


# -- P30: rest tremor of the arm while walking
@functools.lru_cache(maxsize=1)
def _walk_cached():
    from . import gait as GT
    return {s.clip_id: s for s in GT.specs()}[WALK_SOURCE].build()


def walk_base():
    """(Body, gait truth): a copy of the walk of WALK_SOURCE (SOURCES['gait_base'])."""
    b, t = _walk_cached()
    return b.copy(), t


def walk_level0():
    """(Body, truth): the walk of WALK_SOURCE as a level 0 of this module (the same body, validity against
    itself)."""
    b, gt = walk_base()
    return _gap_level0(b, "walk", {"walk": {"source": WALK_SOURCE, "gait_operator": gt["operator"],
                                            "gait_grade": gt["grade"], "gait_scale": gt["scale"]}})


def walk_rest_tremor(cm_pp, grade, seed, side="left", hz=WALK_REST["hz"]):
    """(Body, truth): P30, a rest tremor of the left forearm (the elbow hinge) on the walk of WALK_SOURCE, constant
    over the clip: it rides on the reduced arm swing. The level is the middle fingertip's chord (rule D4, wave SH1d
    part B: every hand tremor levelled at the hand, as raters score the largest amplitude they see; before, the
    wrist's chord, which read one band lower than the hand); the read-back reads the middle fingertip in the frame of
    the upper spine (point 'middle_tip_trunk', which the tremor does not move: the same chord, and the walk's turn
    does not smear the spectrum)."""
    base, gt = walk_base()
    gp = gt["params"]["gait_params"]
    return limb_tremor(base, [side], cm_pp, hz, "elbow", "middle_tip_trunk", None, "sine", seed, grade, "MDS-UPDRS", "3.17",
                       "parkinsonian_rest", "rest, while walking",
                       {"sources": ["gait_base", "bhatia2018", "buijink2012"],
                        "walk": {"source": WALK_SOURCE, "arm_swing_deg": gp.get("arm_swing_deg"),
                                 "elbow_deg": gp.get("elbow_deg")}})


def swing_removed_readback(body, side="left", hz=WALK_REST["hz"], cut_hz=WALK_REST["swing_cut_hz"], margin_s=1.0,
                           point="middle_tip"):
    """{'frequency_hz', 'pp_cm_max', 'pp_cm_median'}: the blind read of a tremor on a walking arm: the middle
    fingertip (point 'middle_tip', the skinned vertex; or the 'wrist' joint) against the shoulder joint in the frame
    of the upper spine (joint 9: the walk's progression and trunk motion out), high-passed at cut_hz (the arm swing
    band out), read as oscillation_readback, the first and last margin_s left out (the filter's edges)."""
    J, Gl = body.fk()
    P = J[:, WRIST[side]].numpy() if point == "wrist" else H.body_points(body, [H.tip_vertices()[side]["middle"]])[:, 0]
    d = P - J[:, SHOULDER[side]].numpy()
    loc = np.einsum("tji,tj->ti", Gl[:, 9].numpy(), d)
    D = highpass(loc, body.rate, cut_hz)
    t = body.times().numpy()
    mask = (t > margin_s) & (t < t[-1] - margin_s)
    return oscillation_readback(D, np.zeros_like(D), body.rate, mask=mask)


# -- the standing body with the pelvis moved and the feet planted (T10, M07)
def stand_legs(dy, dx=None, widen_m=0.0, duration_s=DUR["E7"], rate=100.0, dz=None):
    """(Body, stand base): the procedural stand with the pelvis moved by dy (T,) m vertically, dx (T,) m sideways and
    dz (T,) m forward (along +z, the stand's heading; the trunk kept upright) and the ankles moved outward by widen_m
    each (the feet flat, their global rotations kept), the legs solved by ik.hinge_leg; the construction of
    orthostatic(), repeated so that its clips stay as they are (the same dy and no widening give its level 0 bit for
    bit)."""
    from .ik import hinge_leg
    base = static_body("stand", duration_s, rate)
    J, Gl = base.fk()
    R = base.local_R().clone()
    rest = base.rest()
    dyt = torch.as_tensor(np.asarray(dy, dtype=np.float64))
    dxt = torch.as_tensor(np.zeros(base.T) if dx is None else np.asarray(dx, dtype=np.float64))
    dzt = None if dz is None else torch.as_tensor(np.asarray(dz, dtype=np.float64))
    G0 = Gl[:, 0]
    fwd = torch.tensor([0.0, 0.0, 1.0], dtype=F64).expand(base.T, 3)
    for h, kn, a, sg in ((1, 4, 7, 1.0), (2, 5, 8, -1.0)):
        Hp = J[:, h].clone()
        Hp[:, 1] += dyt
        Hp[:, 0] += dxt
        if dzt is not None:
            Hp[:, 2] += dzt
        A = J[:, a].clone()
        A[:, 0] += sg * widen_m
        G_th, k = hinge_leg(Hp, A, Hp + fwd, rest[kn] - rest[h], rest[a] - rest[kn], G0, hinge=Gl[:, h][..., :, 0])
        Rk = axis_angle_to_matrix(torch.stack([k, 0 * k, 0 * k], -1))
        R[:, h] = G0.transpose(-1, -2) @ G_th
        R[:, kn] = Rk
        R[:, a] = (G_th @ Rk).transpose(-1, -2) @ Gl[:, a]
    tr = base.transl.clone()
    tr[:, 1] += dyt
    tr[:, 0] += dxt
    if dzt is not None:
        tr[:, 2] += dzt
    out = Body(matrix_to_rot6d(R), tr, base.contact.clone(), rate=rate)
    return Body(out.rot6d, out.transl, contact_labels(out), rate=rate), base


def knee_flexion_deg(body):
    """(T, 2) deg: the flexion of the left and right knees (the x component of their local rotations)."""
    R = body.local_R()
    return torch.rad2deg(matrix_to_axis_angle(R[:, [4, 5]]))[..., 0].numpy()


# -- T10: standing tremor of the legs and trunk (TETRAS 9)
@functools.lru_cache(maxsize=64)
def _knee_table(widen_key, n=601, depth_m=0.12, dz_key=0):
    """((n,) pelvis offsets m, (n,) left knee flexion deg): the knee of stand_legs over pelvis offsets from 2 mm
    below the straight leg's reach down by depth_m with the ankles widened by widen_key / 1e6 m and the pelvis carried
    forward by dz_key / 1e6 m."""
    top = -0.002
    if dz_key:                                  # the straight leg reaches the forward hip only this much lower
        J = static_body("stand", 0.01).joints().numpy()[0]
        h = float(np.linalg.norm(J[1] - J[7]))
        top -= h - math.sqrt(max(h * h - (dz_key / 1e6) ** 2, 0.0))
    dy = np.linspace(top, top - depth_m, n)
    dz = None if not dz_key else np.full(n, dz_key / 1e6)
    b, _ = stand_legs(dy, widen_m=widen_key / 1e6, duration_s=(n - 1) / 100.0, dz=dz)
    return dy, knee_flexion_deg(b)[:, 0]


@functools.lru_cache(maxsize=2)
def _standing_centre_um(model, iters=24):
    """um: the forward carry of the pelvis that centres the TETRAS standing posture's centre of mass between the heel
    and the toe of the sole clouds (motions.balance: the posterior and anterior margins equal), by bisection on 0 to
    15 cm (open-body port, wave 2: STANDING's comment). model: the body model's name (the cache key)."""
    from .motions import balance
    widen = _standing_widen()
    wk = int(round(widen * 1e6))

    def f(c_um):
        dy, k = _knee_table(wk, dz_key=c_um)
        y = float(np.interp(STANDING["knee_flex_deg"], k, dy))
        b, _ = stand_legs(np.full(2, y), None, widen, 0.01, dz=np.full(2, c_um / 1e6))
        bl = balance(b, 0)
        return float(bl["posterior"][0] - bl["anterior"][0])
    lo, hi = 0, 150000
    if f(lo) >= 0:
        return 0
    for _ in range(iters):
        m = (lo + hi) // 2
        if f(m) < 0:
            lo = m
        else:
            hi = m
    return int(lo)


def standing_centre_m():
    """m: the forward carry of the TETRAS standing posture's pelvis (_standing_centre_um) on the selected body."""
    from .body import model_name
    return _standing_centre_um(model_name()) / 1e6


def _standing_widen():
    """m: the outward move of each ankle that sets them STANDING['ankle_width_cm'] apart."""
    J = static_body("stand", 0.01).joints().numpy()
    return (STANDING["ankle_width_cm"] / 100.0 - float(J[0, 7, 0] - J[0, 8, 0])) / 2.0


def standing_posture(knee_deg, dx=None, duration_s=DUR["E7"], dz=None):
    """(Body, stand base, dy (T,) m): the TETRAS standing posture (STANDING) with the left knee at knee_deg (T,) or
    scalar deg (the pelvis offset read from _knee_table, without dz), the pelvis carried sideways by dx (T,) m and
    forward by dz (T,) m beyond its centring carry (standing_centre_m: the centre of mass between the heels and the
    toes, open-body port wave 2)."""
    T = int(round(duration_s * 100.0)) + 1
    widen = _standing_widen()
    c = standing_centre_m()
    tab_dy, tab_k = _knee_table(int(round(widen * 1e6)), dz_key=int(round(c * 1e6)))
    k = np.broadcast_to(np.asarray(knee_deg, dtype=np.float64), (T,))
    dy = np.interp(k, tab_k, tab_dy)
    dzc = np.full(T, c) + (0.0 if dz is None else np.asarray(dz, dtype=np.float64))
    body, base = stand_legs(dy, dx, widen, duration_s, dz=dzc if c or dz is not None else None)
    return body, base, dy


def standing_sway(t, rng, S=STANDING_SWAY, pp_m=None):
    """(T,) m: the forward and backward sway of T10 grade 3 (STANDING_SWAY): the sum of two sines at S['hz'] with
    phases drawn from rng, scaled to pp_m peak to peak over the clip (default S['head_cm_pp']), ramped in over
    S['ramp_s'] from 0."""
    ph = rng.uniform(0.0, 2 * math.pi, 2)
    y = sum(np.sin(2 * math.pi * f * t + p) for f, p in zip(S["hz"], ph))
    y = y * (0.5 - 0.5 * np.cos(np.pi * np.clip(t / S["ramp_s"], 0.0, 1.0)))
    pp = S["head_cm_pp"] / 100.0 if pp_m is None else pp_m
    return pp * (y - 0.5 * (y.max() + y.min())) / np.ptp(y)


def pendulum_ratio(body):
    """The head joint's height over the pelvis joint's, both above the ankles' midpoint at knot 0: the head's travel
    over the pelvis's under an ankle sway (physio.stand_sway turns the body about the ankles)."""
    J = body.joints().numpy()[0]
    a = 0.5 * (J[7, 1] + J[8, 1])
    return float((J[HEAD, 1] - a) / (J[0, 1] - a))


def ankle_sway(body, dz):
    """The body turned forward and backward about the line through its ankles by the pelvis travel dz (T,) m along
    the heading (physio.stand_sway: an ankle strategy, the legs re-solved to the body's own ankles and feet, every
    knee angle kept)."""
    from . import physio as PH
    R, tr = body.local_R(), body.transl
    R2, tr2, _ = PH.stand_sway(R, tr, body.rest(), np.asarray(dz, dtype=np.float64), np.zeros(body.T), R, tr)
    out = Body(matrix_to_rot6d(R2), tr2, body.contact.clone(), rate=body.rate)
    return Body(out.rot6d, out.transl, contact_labels(out), rate=body.rate)


def guard_arms(body, share, S=STANDING_SWAY):
    """The body with both shoulders abducted by S['arm_guard_deg'] times share (T,) (about the forward axis of their
    parent frames: the arms out from the sides)."""
    R = body.local_R()
    a = torch.as_tensor(np.radians(S["arm_guard_deg"] * np.asarray(share, dtype=np.float64)), dtype=F64)
    for j, sg in ((SHOULDER["left"], 1.0), (SHOULDER["right"], -1.0)):
        R[:, j] = compose(R[:, j], axis_rotation("z", sg * a), "parent")
    return body.with_rotations(R, [SHOULDER["left"], SHOULDER["right"]])


def standing_tremor(grade, seed, duration_s=DUR["E7"], level0=False):
    """(Body, truth): T10, the standing tremor of the legs and trunk of TETRAS item 9 (STANDING, STANDING_LEVEL;
    module levels, no amplitude published): the left knee at STANDING['knee_flex_deg'] + knee_pp_deg / 2 sin(phase)
    (the pelvis moved vertically, both legs solved, the feet planted), the trunk flexing in phase with the knees by
    trunk_cm at the neck joint (oscillate on STANDING_SPINE); at grade 3 weight shifts and the forward and backward
    sway of standing_sway (the extrapolated centre of mass's margins narrowed, reported against the posture's).
    level0: the posture held still."""
    from .jerks import balance_block
    t = np.arange(int(round(duration_s * 100.0)) + 1) / 100.0
    k0 = STANDING["knee_flex_deg"]
    if level0:
        body, base, dy = standing_posture(k0, None, duration_s)
        J = body.joints().numpy()
        return _gap_level0(body, "TETRAS standing posture", {
            "standing": dict(STANDING, knee_width_cm=float(100 * (J[0, 4, 0] - J[0, 5, 0])),
                             ankle_width_cm_realized=float(100 * (J[0, 7, 0] - J[0, 8, 0])),
                             knee_deg=knee_flexion_deg(body)[0].tolist(), pelvis_offset_cm=float(100 * dy[0])),
            "balance": balance_block(body, body), "sources": ["tetras9"]})
    L = STANDING_LEVEL[grade]
    rng = as_rng(seed)
    phase0 = float(rng.uniform(0.0, 2 * math.pi))
    hz = L["hz"]
    env = envelope(t, [(0.0, t[-1])], RAMP_S, t[-1])
    y = np.sin(phase_of(t, hz, phase0))
    dx, onsets = weight_shifts(t, rng) if L["shifts"] else (np.zeros_like(t), [])
    legs, stand, dy = standing_posture(k0 + 0.5 * L["knee_pp_deg"] * env * y, dx, duration_s)
    post, _, dy0 = standing_posture(k0, dx, duration_s)              # the posture with the shifts, no tremor
    still, _, _ = standing_posture(k0, None, duration_s)              # the posture held still (the level 0)
    dz, guard = None, None
    if L["shifts"]:                                                   # the ankle sway, its head travel set
        ratio = pendulum_ratio(post)
        dz = standing_sway(t, rng, pp_m=STANDING_SWAY["head_cm_pp"] / 100.0 / ratio)
        legs, post = ankle_sway(legs, dz), ankle_sway(post, dz)
        S = STANDING_SWAY
        ramp = 0.5 - 0.5 * np.cos(np.pi * np.clip(t / S["ramp_s"], 0.0, 1.0))
        guard = ramp * (S["arm_base_share"] + (1.0 - S["arm_base_share"]) * (dz / np.abs(dz).max()) ** 2)
    drives = [Drive(j, "x", "child", w) for j, w in STANDING_SPINE]
    out, info, arr = oscillate(legs, drives, L["trunk_cm"], y, env, NECK_POINT, hz)
    if guard is not None:
        out, post = guard_arms(out, guard), guard_arms(post, guard)
    val = validity(post, out)
    val["balance"] = balance_block(post, out)
    if not val["balance"]["ok"]:
        raise ValueError(f"standing tremor grade {grade}: the balance block fails {val['balance']}")
    bs = balance_block(still, out)
    J = out.joints().numpy()
    kd = knee_flexion_deg(out)
    Jp, Js = post.joints().numpy(), still.joints().numpy()
    unsteady = {"sway": dict(STANDING_SWAY, pelvis_cm_pp=float(100 * np.ptp(dz)),
                             realized_head_cm_pp=float(100 * np.ptp(Jp[:, HEAD, 2] - Js[:, HEAD, 2])),
                             arm_guard_deg_range=[float(STANDING_SWAY["arm_guard_deg"] * guard.min()),
                                                  float(STANDING_SWAY["arm_guard_deg"] * guard.max())],
                             rule="an ankle sway (physio.stand_sway) of two sines, phases from the seed, scaled to "
                                  "head_cm_pp at the head joint, ramped in; the arms abducting with it (module "
                                  "values)"),
                "xcom_margin_min_m": min(bs["xcom_margin_posterior_m"], bs["xcom_margin_anterior_m"]),
                "posture_xcom_margin_min_m": min(bs["base_margin_posterior_m"], bs["base_margin_anterior_m"])} \
        if L["shifts"] else None
    params = {"symptom": "standing_tremor", "item": "9", "scale": "TETRAS 9 (words; module levels)", "grade": grade,
              "words": STANDING_WORDS[grade], "tremor_type": "essential", "condition": "standing",
              "frequency_hz": float(hz), "phase_rad": phase0, "level": dict(L),
              "standing": dict(STANDING, knee_width_cm=float(100 * (J[0, 4, 0] - J[0, 5, 0])),
                               ankle_width_cm_realized=float(100 * (J[0, 7, 0] - J[0, 8, 0]))),
              "knee_deg_range": [[float(kd[:, i].min()), float(kd[:, i].max())] for i in (0, 1)],
              "pelvis_dy_cm_range": [float(100 * dy.min()), float(100 * dy.max())],
              "trunk": dict(info, drives_rule="the spine about its x axes in STANDING_SPINE shares, in phase with the "
                                              "knee flexion"),
              "weight_shift_cm": OT_SHIFT_CM if L["shifts"] else 0.0, "weight_shift_onsets_s": onsets,
              "unsteadiness": unsteady,
              "stance_note": "grade 3 'impairs stance' is shown by weight shifts between the planted feet, a slow "
                             "forward and backward sway of the whole body about the ankles that narrows the "
                             "extrapolated centre of mass's margins, and the arms held out from the sides (module "
                             "reading of the words)" if L["shifts"] else "no instability shown",
              "readback": "standing", "sources": ["tetras9", "buijink2012"]}
    return out, truth_record("tremor_catalogue", seed, grade, "TETRAS 9", params, ("tetras9", "buijink2012", "module"),
                             val, arrays={"pelvis_dy_m": dy, "pelvis_dy0_m": dy0, "pelvis_dx_m": dx,
                                          "pelvis_dz_m": np.zeros_like(t) if dz is None else dz,
                                          "arm_guard_share": np.zeros_like(t) if guard is None else guard,
                                          "trunk_angle_rad": arr["angle_rad"], "envelope": env, "waveform": y})


def standing_readback(body, truth):
    """{'knee_left', 'knee_right' (deg), 'pelvis', 'neck' (cm)}: frequency (spectral peak) and windowed peak-to-peak
    (windowed_pp) of the knee flexion, of the pelvis's height and of the neck joint's path against the rebuilt
    posture (standing_posture with the clip's weight shifts, then its ankle sway and its guarding arms, no tremor),
    the first and last second left out."""
    p, a = truth["params"], truth["arrays"]
    dz = a.get("pelvis_dz_m")
    post, _, _ = standing_posture(STANDING["knee_flex_deg"], a["pelvis_dx_m"], body.duration)
    if dz is not None and np.any(dz):
        post = guard_arms(ankle_sway(post, dz), a["arm_guard_share"])
    rate, f = body.rate, p["frequency_hz"]
    t = body.times().numpy()
    mask = (t > 1.0) & (t < t[-1] - 1.0)
    out = {}
    kd, k0 = knee_flexion_deg(body), knee_flexion_deg(post)
    for i, name in enumerate(("knee_left", "knee_right")):
        x = kd[:, i] - k0[:, i]
        out[name] = dict(windowed_pp(x, rate, f, mask), frequency_hz=M.dominant_frequency(x[mask], rate, (2.0, 15.0)))
    J, J0 = body.joints().numpy(), post.joints().numpy()
    for name, j in (("pelvis", 0), ("neck", NECK)):
        D = 100.0 * (J[:, j] - J0[:, j]) if name == "neck" else 100.0 * (J[:, 0, 1] - J0[:, 0, 1])
        out[name] = dict(windowed_pp(D, rate, f, mask), frequency_hz=M.dominant_frequency(D[mask], rate, (2.0, 15.0)))
    return out


# -- M07: orthostatic myoclonus
def ortho_myo_interval_knots(rate=100.0):
    """(lo, hi) knots: the whole-knot intervals between M07 twitch onsets, ceil(rate / rate_hz[1]) to
    floor(rate / rate_hz[0]) (ORTHO_MYO), so that every realized rate lies inside rate_hz."""
    lo_hz, hi_hz = ORTHO_MYO["rate_hz"]
    return int(math.ceil(rate / hi_hz - 1e-9)), int(math.floor(rate / lo_hz + 1e-9))


def ortho_myoclonus(seed, duration_s=DUR["E7"], depth_cm=OT_DEPTH_CM):
    """(Body, truth): M07, orthostatic myoclonus (ORTHO_MYO) on the knees-unlocked stance of tremor_E7_orthostatic_g0
    (the pelvis depth_cm below the straight stance): twitches of the pelvis upward (knee extension; jerks.pulse with
    jerks.timing('cortical')), the onsets on knots, ortho_myo_interval_knots apart, the feet planted; the level 0's
    knots where no twitch acts are kept bit for bit."""
    from .jerks import balance_block, burst_s, pulse, timing
    t = np.arange(int(round(duration_s * 100.0)) + 1) / 100.0
    rng = as_rng(seed)
    rise, fall = timing("cortical")
    klo, khi = ortho_myo_interval_knots()
    onsets, k = [], int(round(ORTHO_MYO["onset_s"] * 100.0))
    while k / 100.0 + rise + fall <= t[-1]:
        onsets.append(k / 100.0)
        k += int(rng.integers(klo, khi + 1))
    amps = rng.uniform(*ORTHO_MYO["amp_share"], len(onsets))
    y = np.zeros_like(t)
    for on, a in zip(onsets, amps):
        y = y + a * pulse(t, on, rise, fall)
    dy = -depth_cm / 100.0 + (ORTHO_MYO["rise_cm"] / 100.0) * y
    out, _ = stand_legs(dy, None, 0.0, duration_s)
    lvl0, _ = orthostatic(pp_cm=0.0, seed=SEED, signs=False, duration_s=duration_s)
    val = validity(lvl0, out)
    val["balance"] = balance_block(lvl0, out)
    if not val["balance"]["ok"]:
        raise ValueError(f"orthostatic myoclonus: the balance block fails {val['balance']}")
    iv = np.diff(onsets)
    params = {"symptom": "orthostatic_myoclonus", "item": "presence", "scale": "none (presence; no scale)",
              "rule": dict(ORTHO_MYO, interval_knots=[klo, khi]), "rise_s": rise, "fall_s": fall,
              "burst_ms": 1000.0 * burst_s("cortical"),
              "n_twitches": len(onsets), "onsets_s": onsets, "amplitude_share": [float(a) for a in amps],
              "interval_s_range": [float(iv.min()), float(iv.max())], "rate_hz_mean": float(1.0 / iv.mean()),
              "pelvis_rise_cm_max": float(100 * (dy.max() + depth_cm / 100.0)), "depth_cm": float(depth_cm),
              "level0": "tremor_E7_orthostatic_g0", "readback": "orthostatic_myoclonus",
              "sources": ["hassanvangerpen2016", "jerks_twitch"]}
    return out, truth_record("tremor_catalogue", seed, 1, "presence", params,
                             ("hassanvangerpen2016", "jerks_twitch", "module"), val,
                             arrays={"twitch_profile": y, "pelvis_dy_m": dy})


def ortho_myoclonus_readback(body, level0_body, rel=0.3, merge_s=0.03):
    """{'n_twitches', 'onsets_s', 'rate_hz_median', 'rate_hz_range', 'pp_cm'}: the twitches read on the pelvis's
    height against the level 0: the runs where its upward speed exceeds rel times its largest (gaps under merge_s
    joined), their starts the onsets."""
    rate = body.rate
    D = body.joints()[:, 0, 1].numpy() - level0_body.joints()[:, 0, 1].numpy()
    v = np.gradient(D, 1.0 / rate)
    on = v > rel * v.max()
    m = np.concatenate([[False], on, [False]])
    starts, ends = np.flatnonzero(m[1:] & ~m[:-1]), np.flatnonzero(~m[1:] & m[:-1])
    keep_s, keep_e = [], []
    for a, b in zip(starts, ends):
        if keep_e and (a - keep_e[-1]) / rate < merge_s:
            keep_e[-1] = b
        else:
            keep_s.append(a)
            keep_e.append(b)
    ons = np.asarray(keep_s) / rate
    r = 1.0 / np.diff(ons) if len(ons) > 1 else np.zeros(1)
    return {"n_twitches": int(len(ons)), "onsets_s": ons.tolist(), "rate_hz_median": float(np.median(r)),
            "rate_hz_range": [float(r.min()), float(r.max())], "pp_cm": float(100 * (D.max() - D.min()))}


def _gap_specs(add):
    """The clips of the gap wave (module docstring), appended to specs() (their own seeds, SEED + 700 on)."""
    seeds = iter(range(SEED + 700, SEED + 800))
    L0R = "tremor_E3_reach_left_g0"
    E3V = ("front", "left", "hands")
    face = f"the finger kept {FACE_GAP_CM:g} cm off the face"
    TM = TERMINAL

    # T04: essential tremor with a terminal crescendo (TETRAS 4c)
    for g in (2.0, 3.0):
        cm = band_level("TETRAS upper limb", g)
        hz = limb_hz("TETRAS upper limb", g, HZ["essential"])
        s = next(seeds)

        def b(cm=cm, g=g, s=s, hz=hz):
            return terminal_reach_tremor("left", cm, hz, g, s)
        add(f"tremor_E3_essential_terminal_tetras{_tag(g)}", "tremor", "E3", "TETRAS 4c",
            "essential tremor, kinetic with a terminal crescendo, left arm, on the reach", g,
            f"TETRAS 4c grade {_fmt(g)} (intention component)",
            f"{hz:g} Hz, {cm:.2f} cm peak to peak at the fingertip within {TM['plateau_cm']:g} cm of the nose and of "
            f"the examiner's finger (band {_band_text('TETRAS upper limb', g)}), {100 * TM['floor']:g} percent of it "
            f"mid-path, rising over the last {TM['near_cm']:g} cm (module envelope), elbow, {face}", E3V, L0R,
            E3_SCENE, b)

    # T05: simple kinetic tremor during pronation-supination
    for g in (2.0, 3.0):
        cm = band_level("TETRAS upper limb", g)
        hz = limb_hz("TETRAS upper limb", g, HZ["essential"])
        s = next(seeds)

        def b(cm=cm, g=g, s=s, hz=hz):
            return ps_kinetic(cm, hz, g, s)
        add(f"tremor_E4_kinetic_ps_tetras{_tag(g)}", "tremor", "E4", "TETRAS bands (simple kinetic)",
            "simple kinetic tremor, left arm, during pronation-supination", g,
            f"TETRAS upper-limb band of grade {_fmt(g)} (no item of its own)",
            f"{hz:g} Hz, {cm:.2f} cm peak to peak at the middle fingertip (band {_band_text('TETRAS upper limb', g)}), "
            f"{_postural_text(cm)}, during the {G.N_CYCLES} cycles of the level-0 pronation-supination at "
            f"{BK.TASKS['pronation_supination']['level0_hz']:g} Hz (not target-directed)",
            ("front", "left", "hands"), PS_LEVEL0, "chair", b)

    # T08: mixed head tremor (TETRAS 1)
    for g in (2, 3):
        cm = band_level("TETRAS head", g)
        hz = HZ["essential_head"]
        s = next(seeds)

        def b(cm=cm, g=g, s=s, hz=hz):
            return mixed_head_tremor(cm, hz, g, s)
        add(f"tremor_E1_head_mixed_g{g}", "tremor", "E1", "TETRAS 1", "head tremor, mixed (no-no and yes-yes), "
            "seated", g, f"TETRAS 1 grade {g}", f"{hz:g} Hz, {cm:.2f} cm peak to peak at the nose tip side to side "
            f"(band {_band_text('TETRAS head', g)}), the yes-yes component {100 * MIXED_HEAD['pitch_share']:g} percent "
            f"of it {MIXED_HEAD['phase_deg']:g} deg of phase behind (module), neck and head in equal shares",
            ("front", "left", "top", "head"),
            "dystonia_E1_level0", "chair", b)

    # T13: essential-type rest tremor (MDS-UPDRS 3.17)
    for g, g1, tag in ((1, "above_floor", "1a"), (2, None, "2")):
        cm = G.tremor_level_cm(g, g1 or "above_floor")
        s = next(seeds)

        def b(cm=cm, g=g, s=s):
            return essential_rest(cm, g, s)
        add(f"tremor_E1_essential_rest_g{tag}", "tremor", "E1", "3.17", "essential tremor at rest (essential tremor "
            "plus), both hands", g, f"MDS-UPDRS 3.17 grade {g}" + (" (above the estimator floor)" if tag == "1a"
                                                                    else ""),
            f"{HZ['essential']:g} Hz left, {HZ['essential_right']:g} Hz right, {cm:.2f} cm peak to peak at each middle "
            f"fingertip (band {_band_text('MDS-UPDRS', g)}), the hands flexing and extending at the wrist",
            ("front", "left", "hands"), "tremor_E1_rest_g0", "chair", b)

    # T16: isometric tremor of the clenched fist (presence)
    iso_cm = band_level("TETRAS upper limb", ISOMETRIC["level_grade"])
    I = ISOMETRIC
    closed = I["fist_at_s"] + I["close_s"]
    fist_t = "fist clenched, forearm supported"
    add("isometric_tremor_E4b_fist_g0", "tremor", "E4b", "level 0", "seated, the left hand clenching into a fist, no "
        "tremor (level 0)", 0, "level 0", f"the left forearm on the chair arm, the hand open, closing into a fist "
        f"(handtasks.FIST) from {I['fist_at_s']:g} to {closed:g} s, then clenched", ("front", "left", "hand_left"),
        None, "chair", lambda: isometric_fist(None, level0=True), fist_t)
    s_iso = next(seeds)
    add("isometric_tremor_E4b_fist_present", "tremor", "E4b", "presence", "isometric tremor, left fist clenched, "
        "forearm supported", 1, "isometric tremor present (no scale)", f"the left forearm on the chair arm, the hand "
        f"open, clenched into a fist from {I['fist_at_s']:g} to {closed:g} s; then {I['hz']:g} Hz, {iso_cm:.2f} cm "
        f"peak to peak at the front of the fist, flexing and extending at the wrist, ramped in over {RAMP_S:g} s once "
        "the fist is clenched, none before (module level and frequency); the squeeze and its force are not shown",
        ("front", "left", "hand_left"), "isometric_tremor_E4b_fist_g0", "chair", lambda s=s_iso: isometric_fist(s),
        fist_t)

    # P29: rest tremor of the leg and foot, seated (MDS-UPDRS 3.17), on its own level 0 (the heel down)
    add("tremor_E1_leg_rest_g0", "tremor", "E1", "level 0", "seated rest, the left foot flat on the floor (level 0)", 0,
        "level 0", "seated, the left ankle dorsiflexed until the heel is on the floor (the seated base rests the foot "
        "on its ball), still", ("left", "foot_left", "front"), None, "chair", leg_rest_level0)
    for g, g1, tag in ((1, "above_floor", "1a"), (2, None, "2"), (3, None, "3")):
        cm = G.tremor_level_cm(g, g1 or "above_floor")
        s = next(seeds)

        def b(cm=cm, g=g, s=s):
            return leg_rest_tremor(cm, g, s)
        add(f"tremor_E1_leg_rest_g{tag}", "tremor", "E1", "3.17", "rest tremor, left leg and foot, seated", g,
            f"MDS-UPDRS 3.17 grade {g}, lower extremity" + (" (above the estimator floor)" if tag == "1a" else ""),
            f"{LEG_REST['hz']:g} Hz, {cm:.2f} cm peak to peak at the big toe (band {_band_text('MDS-UPDRS', g)}), "
            "the forefoot rising and falling about the heel, which stays on the floor (module excursion)",
            ("left", "foot_left", "front"), "tremor_E1_leg_rest_g0", "chair", b)

    # P30: rest tremor of the arm while walking (MDS-UPDRS 3.17)
    add("tremor_E8_walk_g0", "tremor", "E8", "level 0", "parkinsonian walk, left more affected, no tremor (level 0)",
        0, "level 0 (MDS-UPDRS 3.10 grade 1 walk)", "the walk of the parkinsonian gait clip, grade 1: shorter steps, less left "
        "arm swing, slight stoop, slower turn", ("track_left", "examiner"), None, None, walk_level0)
    for g in (2, 3):
        cm = G.tremor_level_cm(g)
        s = next(seeds)

        def b(cm=cm, g=g, s=s):
            return walk_rest_tremor(cm, g, s)
        add(f"tremor_E8_walk_rest_g{g}", "tremor", "E8", "3.17", "rest tremor, left forearm, while walking", g,
            f"MDS-UPDRS 3.17 grade {g}", f"{WALK_REST['hz']:g} Hz, {cm:.2f} cm peak to peak at the middle "
            "fingertip (band "
            f"{_band_text('MDS-UPDRS', g)}), elbow, on the reduced left arm swing of the 3.10 grade 1 walk",
            ("track_left", "examiner"), "tremor_E8_walk_g0", None, b)

    # T10: standing tremor of the legs and trunk (TETRAS 9)
    st = "TETRAS standing posture"
    add("standing_tremor_E10_posture_g0", "tremor", "E10", "level 0", "standing in the TETRAS posture (level 0)", 0,
        "level 0", f"the knees {STANDING['knee_flex_deg']:g} deg flexed, the ankles {STANDING['ankle_width_cm']:g} cm "
        "apart, still", ("front", "left", "foot_left"), None, None, lambda: standing_tremor(0, None, level0=True), st)
    SW = STANDING_SWAY
    for g in (1, 2, 3):
        L = STANDING_LEVEL[g]
        s = next(seeds)
        shift = (f"; unsteady: weight shifts of {OT_SHIFT_CM:g} cm to each side, a slow forward and backward sway "
                 f"of the whole body about the ankles, {SW['head_cm_pp']:g} cm peak to peak at the head ("
                 f"{SW['hz'][0]:g} and {SW['hz'][1]:g} Hz), and the arms held out from the sides up to "
                 f"{SW['arm_guard_deg']:g} deg" if L["shifts"] else "")
        add(f"standing_tremor_E10_tetras9_g{g}", "tremor", "E10", "TETRAS 9", "standing tremor, legs and trunk", g,
            f"TETRAS 9 grade {g} ({STANDING_WORDS[g]})", f"{L['hz']:g} Hz, the knees flexing {L['knee_pp_deg']:g} deg "
            f"peak to peak about {STANDING['knee_flex_deg']:g} deg, the trunk {L['trunk_cm']:g} cm at the neck, in "
            f"phase (module levels), the feet planted{shift}", ("front", "left", "foot_left"),
            "standing_tremor_E10_posture_g0", None, lambda g=g, s=s: standing_tremor(g, s), st)

    # M07: orthostatic myoclonus (presence)
    from .jerks import timing
    rise, fall = timing("cortical")
    klo, khi = ortho_myo_interval_knots()
    lo_hz, hi_hz = ORTHO_MYO["rate_hz"]
    s_om = next(seeds)
    add("orthostatic_myoclonus_E7_present", "tremor", "E7", "presence", "orthostatic myoclonus, legs, standing", 1,
        "orthostatic myoclonus present (no scale)", f"brief knee-extension twitches (rise {1000 * rise:.0f} ms, return "
        f"{1000 * fall:.0f} ms) at irregular intervals of {klo / 100:.2f} to {khi / 100:.2f} s ({100 / khi:.1f} to "
        f"{100 / klo:.1f} Hz, inside {lo_hz:g} to {hi_hz:g} Hz) from {ORTHO_MYO['onset_s']:g} s, "
        f"the pelvis raised by up to {ORTHO_MYO['rise_cm']:g} cm a twitch, the feet planted", ("front", "left",
                                                                                                "foot_left"),
        "tremor_E7_orthostatic_g0", None, lambda s=s_om: ortho_myoclonus(s))


# ---------------------------------------------------------------- the clip list
def _fmt(x):
    return f"{x:.2f}".rstrip("0").rstrip(".")


def _tag(g):
    return _fmt(g).replace(".", "p")


# The scene of the E3 clips: the chair and the examiner's finger at the reach target (exam.scene_mesh routes
# 'chair_target' to reach.scene_mesh; shared wave of 2026-11-02, SH6).
E3_SCENE = "chair_target"


def _postural_text(cm):
    sh = postural_shoulder_weight(cm)
    w = "the hand flexing and extending at the wrist"
    return w if not sh else f"{w}, the shoulder joining (angle weight {sh:g})"


def _band_text(scale, grade):
    lo, hi = SCALES[scale][grade]
    if scale == "TETRAS upper limb" and float(grade) == 1.0:
        return "barely visible"
    if lo == 0:
        return f"under {_fmt(hi)} cm"
    if hi == INF:
        return f"above {_fmt(lo)} cm"
    return f"{_fmt(lo)} to {_fmt(hi)} cm"


def specs():
    """[ClipSpec] of the tremor family and the reach-based limb signs (module docstring), since 'catalogue_v1'."""
    out = []
    seeds = iter(range(SEED + 300, SEED + 600))

    def add(cid, family, seg, item, operator, grade, grade_text, params_text, views, level0, scene, build, title=""):
        out.append(ClipSpec(cid, family, seg, item, operator, grade, grade_text, params_text, tuple(views), level0,
                            scene, build, since=SINCE, segment_title=title))

    # level 0 of the reach, both sides
    for side in ("left", "right"):
        def b0(side=side):
            body, info = reach_base(side)
            out = _level0(body, "reach", {"reach": {"cycle_s": info["cycle_s"], "n_reaches": info["n_reaches"]},
                                          "reach_phases": info["phases"], "side": side})
            out[1]["validity"]["hand_head"] = FaceRoom(body, side).check(body)
            return out
        add(f"tremor_E3_reach_{side}_g0", "tremor", "E3", "level 0", f"reach to the nose, {side} arm (level 0)", 0,
            "level 0", f"{RC.N_REACHES} reaches, {RC.CYCLE_S[side]:g} s per cycle (cited healthy), examiner's finger "
            f"at {100 * RC.TARGET_SHARE:g} percent of reach", ("front", side, "hands"), None, E3_SCENE, b0)
    L0R = "tremor_E3_reach_left_g0"
    E3V = ("front", "left", "hands")
    face = f"the finger kept {FACE_GAP_CM:g} cm off the face"

    # kinetic tremor 3.16 on the reach
    for g, g1, tag in ((1, "below_floor", "1b"), (1, "above_floor", "1a"), (2, None, "2"), (3, None, "3"),
                       (4, None, "4")):
        cm = level_cm("MDS-UPDRS", g, g1 or "above_floor")
        hz = limb_hz("MDS-UPDRS", g, HZ["kinetic"])
        s = next(seeds)

        def b(cm=cm, g=g, s=s, hz=hz):
            return reach_tremor("left", cm, hz, "elbow", "constant", "MDS-UPDRS", g, "3.16", "kinetic", s,
                                point="index_tip")
        lab = {"1b": " (below the estimator floor)", "1a": " (above the estimator floor)"}.get(tag, "")
        mod = " (module level, just above the band's lower edge)" if g == 4 else ""
        add(f"tremor_E3_kinetic_g{tag}", "tremor", "E3", "3.16", "kinetic tremor, left arm, on the reach", g,
            f"MDS-UPDRS 3.16 grade {g}{lab}", f"{hz:g} Hz, {cm:.2f} cm peak to peak at the index fingertip{mod} "
            "(band "
            f"{_band_text('MDS-UPDRS', g)}), elbow, constant through the reaches, {face}", E3V, L0R, E3_SCENE, b)

    # essential tremor: postural (E2, both arms), kinetic (E3), head (E1, E7)
    for g in (1.5, 2.0, 2.5, 3.0, 3.5):
        cm = band_level("TETRAS upper limb", g)
        hz = limb_hz("TETRAS upper limb", g, HZ["essential"])
        hz_r = hz + HZ["essential_right"] - HZ["essential"]
        s = next(seeds)

        def b(cm=cm, g=g, s=s, hz=hz, hz_r=hz_r):
            return limb_tremor(static_body("sit_arms_forward", DUR["E2"]), ["left", "right"], cm,
                               {"left": hz, "right": hz_r}, "postural", "middle_tip", None,
                               "sine", s, g, "TETRAS upper limb", "4a", "essential", "postural",
                               {"sources": ["buijink2012", "deuschl1998", "elble2000"]})
        add(f"tremor_E2_essential_tetras{_tag(g)}", "tremor", "E2", "TETRAS 4a",
            "essential tremor, postural, both arms", g, f"TETRAS 4a grade {_fmt(g)} (MDS-UPDRS 3.15 grade "
            f"{G.tremor_grade_of_cm(cm)})", f"{hz:g} Hz left, {hz_r:g} Hz right, {cm:.2f} cm peak to peak at each "
            f"middle fingertip (band {_band_text('TETRAS upper limb', g)}), {_postural_text(cm)}",
            ("front", "left", "top", "hands"), "tremor_E2_postural_g0", "chair", b)
    for g in (2.0, 2.5, 3.0):
        cm = band_level("TETRAS upper limb", g)
        hz = limb_hz("TETRAS upper limb", g, HZ["essential"])
        s = next(seeds)

        def b(cm=cm, g=g, s=s, hz=hz):
            return reach_tremor("left", cm, hz, "elbow", "constant", "TETRAS upper limb", g, "4c",
                                "essential", s, point="index_tip")
        add(f"tremor_E3_essential_tetras{_tag(g)}", "tremor", "E3", "TETRAS 4c",
            "essential tremor, kinetic, left arm, on the reach", g, f"TETRAS 4c grade {_fmt(g)}",
            f"{hz:g} Hz, {cm:.2f} cm peak to peak at the index fingertip (band "
            f"{_band_text('TETRAS upper limb', g)}), constant through the reaches, {face}", E3V, L0R, E3_SCENE, b)
    head_plan = [("sit", "E1", "pitch", g) for g in (1, 2, 3, 4)] + [("sit", "E1", "yaw", g) for g in (1, 2, 3, 4)]
    head_plan += [("stand", "E7", "pitch", 2), ("stand", "E7", "yaw", 2), ("stand", "E7", "yaw", 3)]
    for posture, seg, dirn, g in head_plan:
        cm = band_level("TETRAS head", g)
        hz = HZ["essential_head_g4"] if g == 4 else HZ["essential_head"]
        s = next(seeds)
        name = {"pitch": "yes-yes", "yaw": "no-no"}[dirn]

        def b(cm=cm, g=g, s=s, posture=posture, dirn=dirn, name=name, hz=hz):
            return limb_tremor(static_body(posture, DUR["E7"]), ["left"], cm, hz, f"head_{dirn}",
                               "nose", None, "sine", s, g, "TETRAS head", "1", "essential", "postural",
                               {"sources": ["buijink2012", "deuschl1998"] + (["elble2017"] if g == 4 else []),
                                "direction": name})
        add(f"tremor_{seg}_head_{dirn}_g{g}", "tremor", seg, "TETRAS 1", f"head tremor, {name}, "
            f"{'seated' if posture == 'sit' else 'standing'}", g, f"TETRAS 1 grade {g}",
            f"{hz:g} Hz, {cm:.2f} cm peak to peak at the nose tip (band "
            f"{_band_text('TETRAS head', g)}), neck and head in equal shares", ("front", "left", "head")
            if dirn == "pitch" else ("front", "top", "head"), f"dystonia_{seg}_level0",
            "chair" if posture == "sit" else None, b)

    # cerebellar: intention tremor (SARA 6), dysmetria (SARA 5), decomposition (ICARS)
    for g in (1, 2, 3):
        cm = band_level("SARA 6", g)
        s = next(seeds)

        def b(cm=cm, g=g, s=s):
            return reach_tremor("left", cm, HZ["intention"], "collar_vertical", "intention", "SARA 6", g, "6",
                                "cerebellar intention", s, point="index_tip")
        add(f"tremor_E3_intention_sara{g}", "tremor", "E3", "SARA 6", "intention tremor, left arm, rising near the "
            "target", g, f"SARA 6 grade {g}", f"{HZ['intention']:g} Hz, {cm:.2f} cm peak to peak at the fingertip near "
            f"the target (band {_band_text('SARA 6', g)}), the mid-movement amplitude {100 * INTENTION_FLOOR:g} percent "
            "of the terminal amplitude, "
            f"the shoulder girdle swinging the hand across the path, {face}", E3V, L0R, E3_SCENE, b)
    for g, kind in ((1, "past"), (2, "past"), (3, "past"), (2, "short")):
        cm = band_level("SARA 5", g)

        def b(cm=cm, g=g, kind=kind):
            e = dysmetria_error("left", cm, kind)
            return reach_sign("left", "dysmetria", g, SEED, {i: e for i in range(RC.N_REACHES)}, 1, "SARA 5",
                              "5 (read on the reach)")
        word = "past" if kind == "past" else "short of"
        add(f"reach_E3_dysmetria_{kind}_sara{g}", "reach", "E3", "SARA 5", f"dysmetria, left arm, {word} the "
            "examiner's finger", g, f"SARA 5 grade {g} (dysmetria band, read on the reach)",
            f"end point {cm:.2f} cm {word} the target (band {_band_text('SARA 5', g)}), then a correction, every "
            "reach", E3V, L0R, E3_SCENE, b)
    for g, n in ((2, 2), (3, 3)):
        def b(g=g, n=n):
            return reach_sign("left", "decomposition", g, SEED, None, n, "ICARS", "finger-to-nose decomposition")
        nr, cyc = RC.decomposed_pace("left")
        add(f"reach_E3_decomposition_icars{g}", "reach", "E3", "ICARS finger-to-nose", "decomposition of movement, "
            "left arm", g, f"ICARS finger-to-nose decomposition grade {g}", f"each movement in {n} phases (joint "
            f"groups one after the other, {1000 * RC.PHASE_PAUSE_S:g} ms apart), {nr} slower reaches of "
            f"{cyc:.2f} s (module pace, on the level 0's clock)", E3V, L0R, E3_SCENE, b)
    s_comb = next(seeds)

    def b_comb(s=s_comb):
        e = dysmetria_error("left", band_level("SARA 5", 2), "past")
        body, truth = reach_tremor("left", band_level("SARA 6", 2), HZ["intention"], "collar_vertical",
                                   "intention", "SARA 6", 2, "6", "cerebellar intention", s,
                                   errors={i: e for i in range(RC.N_REACHES)}, point="index_tip")
        truth["params"]["readback"] = ["oscillation_highpass", "dysmetria"]
        return body, truth
    add("tremor_E3_cerebellar_combined", "tremor", "E3", "SARA 5 and 6", "cerebellar limb signs, left arm: intention "
        "tremor with dysmetria", 2, "SARA 6 grade 2 and SARA 5 grade 2",
        f"{HZ['intention']:g} Hz, {band_level('SARA 6', 2):.2f} cm at the fingertip near the target; end point "
        f"{band_level('SARA 5', 2):.2f} cm past it", E3V, L0R, E3_SCENE, b_comb)

    # Holmes tremor: rest (E1), posture (E2), action (E3)
    for seg, g, posture, item in (("E1", 2, "sit", "3.17"), ("E1", 3, "sit", "3.17"),
                                  ("E2", 3, "sit_arms_forward", "3.15")):
        cm = G.tremor_level_cm(g)
        s = next(seeds)

        def b(cm=cm, g=g, s=s, posture=posture, seg=seg, item=item):
            return limb_tremor(static_body(posture, DUR[seg]), ["left"], cm, HZ["holmes"], "shoulder_elbow",
                               "middle_tip",
                               None, "holmes", s, g, "MDS-UPDRS", item, "holmes", "rest" if seg == "E1" else
                               "postural", {"sources": ["buijink2012", "deuschl1998"]})
        add(f"tremor_{seg}_holmes_g{g}", "tremor", seg, item, f"Holmes tremor, left arm, "
            f"{'at rest' if seg == 'E1' else 'in posture'}", g, f"MDS-UPDRS {item} grade {g}",
            f"{HZ['holmes']:g} Hz (periods jittered {100 * PERIOD_CV['holmes']:g} percent, cycle amplitudes "
            f"{100 * A_MIN['holmes']:g} to 100 percent of the maximum), {cm:.2f} cm peak to peak at the middle "
            f"fingertip, shoulder and elbow",
            ("front", "hands") if seg == "E1" else ("front", "left", "top", "hands"),
            "tremor_E1_rest_g0" if seg == "E1" else "tremor_E2_postural_g0", "chair", b)
    s_h3 = next(seeds)

    def b_h3(s=s_h3):
        return reach_tremor("left", G.tremor_level_cm(3), HZ["holmes"], "shoulder_elbow", "holmes", "MDS-UPDRS", 3,
                            "3.16", "holmes", s, wave="holmes", point="index_tip")
    add("tremor_E3_holmes_g3", "tremor", "E3", "3.16", "Holmes tremor, left arm, at rest, in posture and in action", 3,
        "MDS-UPDRS 3.16 grade 3", f"{HZ['holmes']:g} Hz, {G.tremor_level_cm(3):.2f} cm peak to peak at the index "
        f"fingertip near "
        f"the targets, {100 * HOLMES_FLOOR:g} percent of it at rest and in posture, shoulder and elbow, {face}", E3V,
        L0R, E3_SCENE, b_h3)

    # dystonic tremor: arm (E2) and head with a dystonic rotation (E1)
    D = DYSTONIC_ARM
    for g in (2, 3):
        cm = G.tremor_level_cm(g)
        s = next(seeds)

        def b(cm=cm, g=g, s=s):
            base, dinfo, gain = dystonic_arm_base("left")
            body, truth = limb_tremor(base, ["left"], cm, HZ["dystonic"], "postural", "middle_tip", gain, "dystonic",
                                      s, g, "MDS-UPDRS", "3.15", "dystonic", "postural",
                                      {"sources": ["buijink2012", "deuschl1998", "bhatia2018"],
                                       "dystonic_posture": dinfo, "base_rebuild": "dystonic_arm",
                                       "fast_phase": "toward the dystonic pull (wrist flexion, module choice)"},
                                      fast_sign=-1.0)
            return body, truth
        add(f"tremor_E2_dystonic_g{g}", "tremor", "E2", "3.15", "dystonic tremor, left arm, jerky and irregular", g,
            f"MDS-UPDRS 3.15 grade {g}", f"mean {HZ['dystonic']:g} Hz, periods jittered "
            f"{100 * PERIOD_CV['dystonic']:g} percent, cycles {100 * A_MIN['dystonic']:g} to 100 percent of "
            f"{cm:.2f} cm peak to peak at the middle fingertip, fast phases toward the pull; the wrist held in "
            f"{D['flexion_deg']:g} deg flexion and {D['ulnar_deg']:g} deg ulnar deviation, let "
            f"{D['null_extra_deg']:g} deg further into flexion from {D['null_window_s'][0]:g} to "
            f"{D['null_window_s'][1]:g} s, where the tremor falls to {100 * D['null_gain']:g} percent (null point)",
            ("front", "left", "top", "hands"), "tremor_E2_postural_g0", "chair", b)

    def b_dp():
        body, truth = dystonic_head_base()
        truth["params"]["readback"] = "none"
        return body, truth
    add("tremor_E1_dystonic_head_posture", "tremor", "E1", "TWSTRS rotation", "dystonic neck rotation to the left, "
        "no tremor (level 0 of the dystonic head tremor)", 0, "TWSTRS rotation grade 1, no tremor",
        "the head turned to the left (TWSTRS rotation grade 1), held", ("front", "top", "head"), "dystonia_E1_level0",
        "chair", b_dp)
    for g in (2, 3):
        cm = band_level("TETRAS head", g)
        s = next(seeds)

        def b(cm=cm, g=g, s=s):
            base, pt = dystonic_head_base()
            return limb_tremor(base, ["left"], cm, HZ["dystonic"], "head_yaw", "nose", None, "dystonic", s, g,
                               "TETRAS head", "1", "dystonic", "postural",
                               {"sources": ["buijink2012", "deuschl1998", "twstrs"], "direction": "no-no",
                                "posture": pt["params"],
                                "fast_phase": "away from the dystonic pull, to the right (a corrective jerk, module "
                                              "choice)"},
                               fast_sign=-1.0)
        add(f"tremor_E1_dystonic_head_g{g}", "tremor", "E1", "TETRAS 1", "dystonic head tremor, no-no, jerky, on a "
            "neck rotation to the left", g, f"TETRAS 1 grade {g}", f"mean {HZ['dystonic']:g} Hz jittered, up to "
            f"{cm:.2f} cm peak to peak at the nose tip, the fast phases to the right, away from the pull, the head "
            "turned left (TWSTRS rotation grade 1)", ("front", "top", "head"), "tremor_E1_dystonic_head_posture",
            "chair", b)

    # wing-beating tremor (TETRAS 4b posture)
    wt = "arms abducted, elbows flexed, hands at nose level (TETRAS 4b posture)"

    def b_w0():
        return _level0(posture_from(wing_rotations(), 20.0), "wing posture", {"wing": WING})
    add("tremor_E2w_wing_g0", "tremor", "E2w", "level 0", "arms in the wing-beating posture (level 0)", 0, "level 0",
        f"both arms abducted, elbows flexed, the fingertips {WING['tip_gap_cm']:g} cm apart at nose height, "
        f"{WING['tip_forward_cm']:g} cm in front of the face", ("front", "left", "hands"), None, "chair", b_w0, wt)
    for g in (2.5, 3.0, 3.5):
        cm = band_level("TETRAS upper limb", g)
        s = next(seeds)

        def b(cm=cm, g=g, s=s):
            return limb_tremor(posture_from(wing_rotations(), 20.0), ["left", "right"], cm, HZ["wing"],
                               "wing", "middle_tip", None, "wing", s, g, "TETRAS upper limb", "4b",
                               "wing-beating", "postural", {"sources": ["buijink2012"], "wing": WING})
        add(f"tremor_E2w_wing_tetras{_tag(g)}", "tremor", "E2w", "TETRAS 4b", "wing-beating tremor, both arms, "
            "proximal", g, f"TETRAS 4b grade {_fmt(g)}", f"{HZ['wing']:g} Hz, {cm:.2f} cm peak to peak at each middle "
            "fingertip "
            f"(band {_band_text('TETRAS upper limb', g)}), elbow and hand flapping at the shoulder",
            ("front", "left", "hands"),
            "tremor_E2w_wing_g0", "chair", b, wt)

    # orthostatic tremor
    add("tremor_E7_orthostatic_g0", "tremor", "E7", "level 0", "standing, knees unlocked (level 0)", 0, "level 0",
        f"pelvis {OT_DEPTH_CM:g} cm below the straight stance, ordinary stance, still", ("front", "left", "foot_left"),
        None, None, lambda: orthostatic(pp_cm=0.0, seed=SEED, signs=False))
    add("tremor_E7_orthostatic_present", "tremor", "E7", "presence", "orthostatic tremor, legs, standing", 1,
        "orthostatic tremor present (no amplitude scale)", f"{HZ['orthostatic']:g} Hz, pelvis {OT_PP_CM:g} cm peak "
        "to peak: palpable rather than visible (aliased at 30 fps); seen: the feet "
        f"{OT_STANCE_CM:g} cm apart and weight shifts of {OT_SHIFT_CM:g} cm to each side from {OT_SHIFT_FIRST_S:g} s",
        ("front", "left", "foot_left"), "tremor_E7_orthostatic_g0", None, lambda: orthostatic(seed=SEED))

    # enhanced physiological tremor
    for g in (1.0, 1.5):
        cm = band_level("TETRAS upper limb", g)
        s = next(seeds)

        def b(cm=cm, g=g, s=s):
            return limb_tremor(static_body("sit_arms_forward", DUR["E2"]), ["left", "right"], cm,
                               {"left": HZ["physiological"], "right": HZ["physiological"] + 0.4}, "wrist",
                               "middle_tip", None, "sine", s, g, "TETRAS upper limb", "4a", "enhanced physiological",
                               "postural", {"sources": ["buijink2012"]})
        add(f"tremor_E2_physiological_tetras{_tag(g)}", "tremor", "E2", "TETRAS 4a",
            "enhanced physiological tremor, both arms", g, f"TETRAS 4a grade {_fmt(g)}",
            f"{HZ['physiological']:g} Hz left, {HZ['physiological'] + 0.4:g} Hz right, {cm:.2f} cm peak to peak at "
            f"each middle fingertip ({_band_text('TETRAS upper limb', g)}), the hand flexing and extending at the wrist",
            ("front", "left", "top", "hands"), "tremor_E2_postural_g0", "chair", b)

    # functional tremor
    ft = "left arm outstretched, the right hand tapping, then a ballistic movement"
    F = FUNCTIONAL
    add("tremor_E2f_functional_g0", "tremor", "E2f", "level 0", "the right hand taps and moves, no tremor (level 0)",
        0, "level 0", f"taps at {HZ['tapping']:g} Hz {F['tap'][0]:g} to {F['tap'][1]:g} s, "
        f"ballistic movement at {F['ballistic'][0]:g} s", ("front", "hands", "top"), None, "chair",
        lambda: functional(0.0, level0=True), ft)
    for g in (2, 3):
        cm = G.tremor_level_cm(g)
        s = next(seeds)

        def b(cm=cm, s=s):
            return functional(cm, seed=s)
        add(f"tremor_E2f_functional_g{g}", "tremor", "E2f", "3.15", "functional tremor, left arm", g,
            f"MDS-UPDRS 3.15 grade {g}", f"own {HZ['functional']:g} Hz drifting, amplitude varying from "
            f"{100 * FUNCTIONAL_AMP_MIN:g} percent; entrained to the {HZ['tapping']:g} Hz taps over "
            f"{F['entrain_s']:g} s with a residual phase jitter; paused {1000 * F['pause_delay_s']:g} ms after the "
            f"onset of the ballistic movement; {cm:.2f} cm peak to peak at the middle fingertip, "
            f"{_postural_text(cm)}", ("front", "hands", "top"), "tremor_E2f_functional_g0", "chair", b, ft)

    # pill-rolling (hand joints)
    hv = ("hand_left", "hand_left_medial", "hands")
    add("tremor_E1_pillrolling_g0", "tremor", "E1", "level 0", "seated rest, resting left hand (level 0)", 0,
        "level 0", f"the thumb pad on the index pad ({PILL['gap_cm']:g} cm), fingers loosely curled", hv, None,
        "chair", lambda: pill_rolling())
    c1 = G.tremor_level_cm(1, "below_floor")
    add("tremor_E1_pillrolling_g1", "tremor", "E1", "3.17", "pill-rolling rest tremor, left fingers", 1,
        "MDS-UPDRS 3.17 grade 1", f"{HZ['pill_rolling']:g} Hz, the thumb pad rubbing along the index pad (thumb "
        f"{PILL['thumb_rub_deg']:g} deg at the base, a quarter period from the index), {c1:.2f} cm peak to peak at "
        "the index fingertip", hv, "tremor_E1_pillrolling_g0", "chair", lambda: pill_rolling(cm_pp=c1, seed=SEED + 1))
    c2 = G.tremor_level_cm(2)
    add("tremor_E1_pillrolling_supination_g2", "tremor", "E1", "3.17", "pill-rolling with pronation-supination, left "
        "hand", 2, "MDS-UPDRS 3.17 grade 2", f"{HZ['pill_rolling']:g} Hz, the rub of grade 1 and the forearm turning "
        f"in phase, {c2:.2f} cm peak to peak at the index fingertip", hv, "tremor_E1_pillrolling_g0", "chair",
        lambda: pill_rolling(cm_pp=c1, pronation_cm_pp=c2, seed=SEED + 2))
    _gap_specs(add)
    ids = [s.clip_id for s in out]
    if len(set(ids)) != len(ids):
        raise RuntimeError("duplicate clip ids in the tremor catalogue")
    return out
