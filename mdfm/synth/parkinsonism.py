"""Parkinsonian exam items of the synthetic-patient symptom catalogue (task CAT-PDATAX, since 'catalogue_v1'): the
MDS-UPDRS items and signs the exam set of exam.py did not grade, each as a ladder of ClipSpecs with its level-0
partner on the same time base (specs()), a re-measurement per operator (measure_*) and the sources of every
parameter (SOURCES, merged with grades.SOURCES). Revised after the clinical review of the family (task
FIX-parkinsonism): the slow rise no longer bows deeper, the failed attempts vary and fall back, every clip carries a
background layer (no statue), the 3.14 movements are composite actions on a seated rest with the hands on the
thighs, the 3.14 ladder is nested, and the repeated tasks vary from cycle to cycle. Revised by task sh1b: the
background takes physio.py's sourced values and one layer per series (section 0), the applause raise no longer
brings the hands closer than the first opening (section 6), the 3.9 level 0 is named a healthy rise (section 1).

0. Background layer (background_signals; task sh1b, version BG_VERSION): the six series physio.SKIP leaves to this
   module (SERIES) take the signals and the values of physio.py, the catalogue's one physiological background (SH1):
   for each series the layers physio.make_layer draws from physio.series_seed of its root (the seated and the
   standing layer share every draw), so that every clip of a series, its level 0 included, lays the same signals on
   the same time base. Breathing at BREATH_PER_MIN with the inspiratory share BREATH_TI_TE (physio
   SOURCES['motamedifakhr2017']; the rate varying by BREATH_RATE_CV), spine3 extending by BREATH_SPINE_DEG at full
   inhalation, the neck taking back BREATH_NECK_SHARE of it, the collars rising by BREATH_COLLAR_DEG under hanging
   arms; a slow drift of the head (HEAD_DRIFT_DEG RMS in HEAD_BAND_HZ, yaw and pitch shared by neck and head); the
   seated trunk's lean (physio.sit_gains, after physio SOURCES['melo2009']: forward only sagittally, so that the
   back never meets the chair's backrest, two-sided laterally; sagittally over spine1 to spine3 by each joint's room
   at the clip's base, sagittal_shares, laterally at spine1: open-body port, wave 6, the 'sit' seat's lumbar joint
   stands 2.5 deg from jarm2022's flexion bound); the fingers' flexion drifting by
   FINGER_DRIFT_DEG; standing, a sway of the pelvis of SWAY_RMS_M (physio SOURCES['foulger2025']) in SWAY_BAND_HZ
   (physio SOURCES['yamamoto2015']): front-back an ankle strategy (the whole body turns about the ankles' axis),
   sideways a hip strategy (the pelvis shifts, the trunk stays upright), the legs re-solved to the planted ankles
   (place_pelvis, ik.hinge_leg). Each truth records it (background_record: the series, physio's seed, version and
   layer digests, the parts laid and where).
   Left to this module, and why: the 3.14 ladder scales the head drift and the fingers by BG_RATIO^grade (the sign
   is a poverty of small movements); breathing and the postural sway (the seated trunk's lean, the standing sway)
   are the level 0's on every grade (postural control, not a spontaneous movement; no source was read for a
   smaller sway with the grade; kept by the session decision of 2026-11-03 under the owner's delegation,
   reports/gpu_queue/decisions.md: breathing and postural sway are not spontaneous movement); an arising is a
   transfer, so the trunk lean is laid on the seated knots only (ramped over 0.3 s inside each seated run) and the
   sway on the standing knots after a rise (physio's 'transfer' kind lays breathing and the head only, its
   series-wide parts cannot follow the phases of a clip); the composition is this module's (the neck takes back
   BREATH_NECK_SHARE of the breathing extension, physio all of it: the head pitches by about 0.3 deg at full
   inhalation; the collars do not take it back, so the arms ride on the chest: the folded arms of the arisings stay
   on it, the forearms of the seated rests over the chair arms move with each breath; no support anchoring, no
   joint-limit guard and no fade, the module's own checks judge the clips; the sway keeps the legs' reach
   (place_pelvis) where physio keeps their lengths); the fingers drift only on the hands the module holds relaxed
   (3.14, applause rest, motor-state rest).
1. MDS-UPDRS 3.9, arising from a chair (segment E7), grades 0 to 4, on the balanced procedural arising of
   motions.arise (seat 43 cm, arms folded, the trunk solved for balance), re-timed: a template rise is read at the
   share u of the rise that a clip's timeline asks for (viewer.pose_at, the body's Hermite interpolant), so that a
   clip strings seated holds, failed attempts, an arm transition and the successful rise on one 100 Hz clock.
   - grade 0: one rise at the cited control pace (1.89 s, Inkster and Eng 2004), arms folded; the level 0 of the
     ladder, of the abrupt arising and of Gowers' manoeuvre (sections 5 and 8), named a healthy rise
     (ARISE_LEVEL0_OPERATOR): beside a PSP or a myopathic sign, the ladder's 'parkinsonian' misled;
   - grade 1 (slow): one rise of SLOW_RISE_S (2.8 s, about the control mean plus 2.4 SD, a module value) that
     reads the normal template on a slower clock (slow_rise_u): its trunk lean is the normal one (the version
     before the review solved a deeper bow, 59 against 45 deg from vertical, the opposite of the parkinsonian
     pattern); the pace keeps the template's from 0.34 to 0.56 of the rise (the momentum transfer around lift-off,
     so that the extrapolated centre of mass stays ahead of the heels after lift-off as in the template) and is
     slower before it, with a hesitation of the forward lean (SLOW_HESITATE), and after it (the extension);
   - grade 1 (attempts): one failed attempt with the arms folded, then a rise at the control pace;
   - grade 2: one failed attempt with the arms folded (the form's protocol allows the arms only after the
     folded attempts fail), the hands moved to the chair arms, one push-off rise without difficulty;
   - grade 3: as grade 2, but the first push-off falls back onto the seat, the second succeeds;
   - grade 4: two failed folded attempts and two failed push-offs, the patient stays seated; the help the form
     then gives is not shown (no examiner in the scene): the caption says so.
   A failed attempt is drawn from the clip's seed (_attempt_params): it rises over UP_STRETCH times its share of
   the rise time, varied by TIMING_VAR, on the 'fail' template (FAIL_TRUNK_SHARE of the solved trunk bout) with its
   own trunk share (FAIL_TRUNK_RANGE folded, lowered until its extrapolated centre of mass lies FALL_MARGIN_M behind
   the heels after lift-off: the body falls back; PUSH_TRUNK_RANGE for a push-off), to a share FAIL_U of the rise
   folded (the buttocks 2 to 9 cm up) or PUSH_FAIL_U pushing (capped where an arm reaches PUSH_PEAK_EXT of its
   length: 8 to 15 cm, the elbows extending; a push-off at a steady pace eased in and out, PUSH_EASE, and stretched
   until the pelvis-to-neck line turns at most TRUNK_RATE_MAX), then drops in FALL_S: the hip midpoint back to the seat
   on a straight line (a minimum-jerk profile of x^FALL_SKEW, accelerating), the pelvis and the trunk back to their
   seated lean
   ahead of it (TRUNK_LEAD), the legs re-solved to the planted ankles; the seated pauses after an attempt are drawn
   in PAUSE_S (shrunk to fit the clip). The push-off: the hands rest on the chair arms of motions.arise_chair (wrist
   joint above the armrest top by the palm's thickness, over the armrests' front ends, WRIST_BACK_M), the arms solved per knot by two-bone inverse kinematics
   (ik.hinge_leg on the shoulder, elbow flexion about the upper arm's y axis, the forearm twist shared half and
   half between elbow and wrist, the hand held palm down, the elbows pointing back: POLE_OUT, POLE_BACK), the wrist
   raised per knot by what keeps the lowest hand vertex on the armrest's top (push_arms), until the arm reaches
   RELEASE_EXT of its length, a component would leave the SP1 table or, a guard, a forearm's or hand's skin would
   come within PUSH_THIGH_GUARD_M of a thigh (push_thigh_gap); the hands ease off the chair arms over
   RELEASE_EASE_S before that knot (the push-off rows cross-faded into the release's, so that the wrists' speed
   rises from the push-off's without a step: with the arm rows held from the release knot the wrists' speed stepped
   up in one knot, review of task sh1c), the elbows flex by RELEASE_ELBOW_DEG over RELEASE_LIFT_S (the
   hands lift off) while the arms blend to hanging (the 'stand' arms) over RELEASE_S and the hands relax
   (RELAXED_HAND). At the end of the rise the thighs rise toward the hands: with the fingers straight
   forward, the thumb side of the right hand entered the thigh (task sh1c; push_hand_yaw's try at 0 deg gives the
   depth); the hands are now
   turned outward about the vertical (the fingers forward and out) by the smallest yaw of a grid (push_hand_yaw,
   PUSH_YAW_STEP_DEG up to PUSH_YAW_MAX_DEG) whose rehearsal on the 'normal' template keeps them PUSH_THIGH_CLEAR_M
   off the thighs until the release. A failed push-off's peak is lowered until its hands also stay
   PUSH_THIGH_GUARD_M off the thighs (_attempt_params). The move from the folded grip to the chair arms
   (_open_rows; task sh1c): on one clock the right hand, under the left forearm, passed through it; one arm now
   waits (opening_lag: motions.fold_open_lag on a rehearsal of the move, the seated template,
   the smallest lag of motions.FOLD_OPEN_LAGS whose skins of motions.FOLD_OPEN_CHECKS stay motions.FOLD_OPEN_CLEAR_M
   apart, the right arm leaving first). The validity's 'fold' (arise_fold_check) reads the skin gaps over the folded
   phases, the move and the push-offs: no skin part enters another. The balance of a push-off is not checked by the
   feet-only rule (the hands bear weight).
2. MDS-UPDRS 3.14, global spontaneity of movement (body bradykinesia), grades 0 to 4, seated (E1) and standing
   (E7), 45 s each. Seated rest (seated_rest): 'sit' with the trunk upright and the collars protracted
   (THIGH_REST), the hands resting on the thighs (arm_ik with the collar kept, the lowest hand vertex just above
   the thigh's skin, the fingers relaxed), the elbows over the chair arms; standing rest: 'stand' (its knees soft,
   base.STAND_REACH: the open body's straight rest legs held at full length kink under the legs' inverse kinematics)
   and relaxed hands. Composite actions (SPONT_KINDS): head turns with the trunk turning along (TRUNK_TURN_SHARE), a glance
   down, looking at a wrist watch (WATCH: the left forearm across the front, the head down; standing, a geodesic
   blend of the arm's joints, stand_watch), the hands together in the lap (LAP), a hand sliding along its thigh, a
   shrug of both shoulders (seated with the hands kept on the thighs; standing with the arms hanging under the
   raised collars, the shoulders turned back by the collars' rise and read against the table 'shrug' of
   limits_ext, SHRUG_TABLE: the SH2 coupling extended to the healthy shrug by the session decision of 2026-11-03,
   limits_ext.SHRUG_DECISION; built since task sh1c, the standing ladder's level-0 draws taking it), a seat
   adjustment (the pelvis forward and unweighted, the trunk leaning midway, the legs re-solved), a foot slid forward
   and left there for a while (the foot lifts midway, FOOT_LIFT_SHARE of its way and FOOT_LIFT_MIN_M at least, and
   travels only while FOOT_CLEAR_M off the floor, then planted), and standing a weight shift (the pelvis
   sideways, dropping on the unloaded side, the trunk upright, both feet planted). Each is a minimum-jerk move, a
   hold that drifts by SPONT_HOLD_DRIFT of the excursion (not for a planted foot or the hands in the lap) and a
   return. Level 0 (_spont_events):
   SPONT_RATE_PER_MIN movements per minute, kinds drawn without repeating one in a row nor beyond
   SPONT_MAX_PER_CLIP, holds, amplitude factors (SPONT_AMP_JITTER) and anchors drawn from one seed per base. Grade
   g keeps a nested subset of the level-0 movements (_level0_order: spontaneity_count(g) of them, the subset whose
   predicted mean peak lies nearest the level-0 mean times SPONT_AMP_RATIO^g and below SPONT_LADDER_STEP of the grade
   below's), scales the amplitude within each kind by SPONT_AMP_RATIO^g (not the hands together in the lap, a
   target, nor beyond its target) and the movement time by SPONT_SLOW_RATIO^g (a log ladder, as Elble et al. 2006
   found for tremor ratings; for 3.14 a module proposal: no published rate of spontaneous movements was found,
   SOURCES['spont_unsourced']).
3. The sequence effect (Bologna et al. 2023: a progressive reduction of amplitude or speed during a repetitive
   movement): 15 s of finger tapping (50 cycles at the 3.33 Hz of the controls of Ling et al. 2012, the cited
   level-0 amplitude, the fingers closing fully every cycle) and 30 cycles of pronation-supination (the version-2
   rate and amplitude, turning about the excursion's midpoint so that the pronation and the supination ends both
   shrink), the amplitudes' least-squares relative slope at SEQUENCE_SLOPES: level 0 the controls' of Ling et al.
   2012 (0.12 deg per cycle of 45.9 deg), level 1 their PD OFF patients' (0.20 of 37.8), levels 2 and 3 twice and
   three times that (module), linear at levels 0 and 1, a fast fall to a plateau at levels 2 and 3
   (SEQ_PLATEAU_TAU); the cycles vary (sequence_cycles: AR(1) jitter of SEQ_PERIOD_CV in the period and SEQ_AMP_CV
   in the amplitude, the latter orthogonal to the trend's fit and to the first and last five cycles' means) and
   level 3 hesitates twice (SEQ_HESITATIONS, SEQ_HESITATION_PERIODS of a period). No scale grade of its own
   (MDS-UPDRS 3.4 and 3.6 rate 10 cycles).
4. Upper-limb freezing in finger tapping (Vercruysse et al. 2012: hastening and an amplitude breakdown before the
   episode, power in the 3 to 8 Hz freeze band during it): HASTEN_CYCLES cycles whose rate rises by HASTEN_GAIN
   and whose amplitude falls to BREAKDOWN_SHARE of A0, an episode of FREEZE_S (the uncued mean of Delval et al.
   2017) in which the fingers tremble near closure at TREMBLE_HZ, then the tapping resumes at RESUME_SHARE of A0.
   Two clips: 10 cycles with one episode (MDS-UPDRS 3.4 grade 3 by its arrest criterion, a freeze in ongoing
   movement) and a 15 s run with two episodes. The finger-tapping clips carry the hand_left_medial view, where the
   decrement and the trembling are seen (the front close-up foreshortens the index-thumb gap).
5. AP03 (task park_gaps), the abrupt arising and sitting down en bloc, no grade (the PSPRS wording was not read):
   - the abrupt arising (abrupt_arise, park_E7_arise_abrupt): the 'normal' template on the faster clock of
     ABRUPT_RISE_S (1.2 s against 1.89), its trunk bout lowered to the smallest share that keeps the extrapolated
     centre of mass MOS_MIN_M ahead of the heels from lift-off (abrupt_rise_u: the speed carries the centre of mass
     forward, the trunk leans less); the seed of park_E7_arise_g0, so that the two differ by the rise only;
   - sitting down (sitdown, E22, park_E22_sitdown_level0 and park_E22_sitdown_enbloc): seated, a rise at the control
     pace, SITDOWN_STAND_S standing, then the sit-down: level 0 the 'normal' template read backward (the trunk leans
     forward as the hips go back, the descent slows to the seat: a module reading, no stand-to-sit source), on a clock
     slowed by the smallest stretch that keeps the extrapolated centre of mass MOS_MIN_M ahead of the heels to the seat
     contact (controlled_descent_u: read at the rise's own pace, the reversed velocities carry it behind the heels), the
     hips leaning further forward by the smallest amplitude the slowest clock needs (sitdown_lean: on the anatomical
     table the template's bout alone leaves the centre of mass behind the heels at the seat contact), the folded arms
     turned up about the collars' line on the same profile so that the lean does not bring them into the thighs
     (sitdown_fold_turn, SITDOWN_FOLD_TURN_DEG; plan phase E);
     en bloc (_bloc_descent) the reversed template with BLOC_TRUNK_SHARE of its trunk bout down to BLOC_U, then a drop:
     the hip midpoint falls with a constant acceleration to BLOC_IMPACT_M_S at SEAT_CONTACT_M above the seat and stops
     within it (the cushion's compression is not drawn), the trunk stays upright, the pelvis and the trunk reach their
     seated lean by cubic Hermites (the velocities kept at the join), the legs re-solved to the planted ankles. The
     read-back (measure_sitdown): the descent time (onset to seated), the pelvis's downward speed at the seat contact,
     where the extrapolated centre of mass leaves the heels (its time after the onset, the pelvis's height above its
     seated height and the share of the descent left there: en bloc, nearly the whole descent is a backward fall into
     the seat), the trunk lean, the seat clearance. The abrupt arising is a brisk rise with less lean only: the
     backward instability after the rise is not shown (NOT_RENDERED).
6. AP01 (applause, E21): after the examiner's three claps (not shown), the patient claps CLAP_COUNTS times: 3 (level 0,
   negative), 4, 7 and 14 (the graded scoring 4, 5 to 10, more than 10 of the catalogue row, not traced to its
   source; positive above 3, Dubois 2005). Seated on 'sit' with the background layer, the hands relaxed at rest
   (CLAP_REST_HAND with the fingers' drift) and raised by a joint blend that also shapes them to clap (back to relaxed
   over the lowering), then each hand by arm_ik toward a wrist target in spine3's frame: the clapping hand
   (clap_hand: the fingers together and slightly flexed, the thumb drawn into the palm's plane) oriented by the plane
   fitted to its palmar surface (_hand_rest_axes), the fingers forward and CLAP_UP_DEG up, the palms facing; the
   palms' distance
   (clap_gap: a close with the speed largest at contact, a dwell at CLAP_GAP_M, an opening; AR(1) jitter of the period
   and the opening) is set per knot by moving each wrist so that its hand's most medial vertex lies at half the
   distance from the midsagittal plane (the hands never cross it: the contact is posed, 2 mm). The raise only closes
   the hands and the lowering only opens them (_clear_path: the joint blend alone turned the palms to face while the
   relaxed thumbs still stood out; the hands came to 16.3 cm and opened again to the first opening's 20, one more
   approach than the claps, and came back in by 7 mm while lowered; each upper arm is now turned outward about the
   vertical by what a monotone path needs, the solved turn smoothed so that the wrists do not jolt).
   Read-back (measure_claps): the claps (runs of the palms closer than CLAP_TOUCH_M), their rate, the least
   distance, the approaches (every minimum of the palms' distance at least CLAP_APPROACH_M deep: as many as the
   claps stated).
7. P33 (motor state transition, E1): the rest tremor of the left hand at the wrist (tremors.oscillate with the 'wrist'
   drive, the chord calibrated per knot at the middle fingertip of the resting hand) at MS_HZ on the seated rest
   (ms_base: 'sit' with the background layer), its level ms_level_cm(t): MDS-UPDRS 3.17 grade 3 (OFF) and grade 1
   (ON, 0.58 cm) held MS_HOLD_S, a minimum-jerk ramp in log amplitude over MS_RAMP_S (a clinical transition takes
   minutes: compressed, unsourced), the other state to the end; OFF to ON and ON to OFF (wearing-off). Read-back
   (measure_motor_state): the chord in windows of MS_WINDOW_S, its grade over time and the times it crosses 3 and
   1 cm. Only the rest tremor changes (no dyskinesia in ON, no bradykinesia ramp).
8. G10 (Gowers' manoeuvre when arising, E7): from the seated pose of the 'normal' template (seat 43 cm, the feet where
   the template plants them), the trunk upright and the hands on the proximal thighs; a deep forward lean with the
   hands sliding toward the knees (GOWERS_LEAN), the buttocks rising first with the trunk bent (GOWERS_PUSH), six
   hand steps up the thighs, hand over hand (GOWERS_STEPS: each hand passes the other, so that the lower hand, which
   bounds how far the trunk can extend, moves up at every step); in each step the hand moves up, lifted
   GOWERS_STEP_LIFT_M, while the trunk is held, then both hands push the trunk up by its share (GOWERS_TRUNK_SHARES,
   the lean falling about 6 deg a step) while the knees straighten (GOWERS_KNEES_STEPS); the last step leaves the
   hands on the proximal thighs with the trunk at most GOWERS_RELEASE_MAX_DEG from its seated start (GOWERS_CLIMB),
   the hands lift off the thighs (GOWERS_RELEASE_LIFT_M), then the final extension with the arms let down to hang.
   The spine bends at most by its table room (motions.trunk_split's, from the arising's upright seat); past it the
   bow is the hips' (the pelvis's pitch; open-body port, wave 6: the lean's 50 and the push's 55 deg passed
   jarm2022's thoracolumbar 45). Over the forward lean the hips slide GOWERS_SCOOT_M forward on the seat (wave 6
   follow-up: from the upright seat the lean alone left the centre of mass behind the heels at lift-off).
   The hip midpoint's forward position is solved per knot after lift-off so that the centre of mass follows a ramp
   to the standing value (a slow rise balanced over the feet: the hands push on the body itself, no external
   support); the legs re-solved to the planted ankles; each hand flat on its thigh's front by arm_ik (the thumb spread
   clear of the palm's plane, GOWERS_THUMB_CMC_DEG; the fingers along the forearm's projection on the thigh, the
   palm's distance corrected per knot to THIGH_REST['clear_m'] off
   the skin; the palm centre shifted toward the outside of the thigh by the smallest step of GOWERS_LATERAL_STEP_M
   whose rehearsal keeps the two hands apart and the forearms and hands out of the trunk and the thighs,
   gowers_lateral: on the thigh's axis the fingertips met at the midline and the hands entered the hip crease, review
   of task sh1c): the contact posed, not simulated. The validity's 'fold' (gowers_skin_check) reads those skin gaps
   up to the end of the final extension: no skin part enters another. Read-back (measure_gowers): the wrist's share
   of the way along each thigh, the hand's distance off it (also as the hands leave), the trunk lean (at lift-off,
   its peak, at the release), the extrapolated centre of mass's margins from lift-off.
Micrographia is not built: writing needs a pen, a table and the trace on paper (the size reduction is read from
the trace), none of which the scene or the body carries; a pen-less hand moving millimetres above no table is
not a credible stimulus (task CAT-PDATAX, reported).

Re-measurement (the tests compare them with the truth): measure_arise (attempts, seat-offs, falls back, rise
time, arm use, trunk lean), measure_spontaneity (movement count, amplitude, duration, against the background
body), measure_decrement (per-cycle amplitudes, relative slope, loss), measure_freezing (episodes by the amplitude
criterion of Vercruysse et al. 2012 as quoted by Williams et al. 2013, the trembling frequency, the hastening).
Every body is procedural, in the neutral shape, CPU only; the SMPL+H arrays stay in memory.
"""
import functools
import itertools
import math

import numpy as np
import torch

from ..body.rotations import axis_angle_to_matrix, matrix_to_axis_angle, matrix_to_rot6d
from ..body.joints import fk
from . import grades as G
from . import hands as H
from . import limits_ext as LX
from . import physio as PH
from .base import posture_rotations, static_body
from .body import (ELBOW_FLEXION_SIGN, F64, axis_rotation, compose, contact_labels, limit_violations, rest_joints,
                   room_deg, skate, validity)
from .common import as_rng, truth_record
from .exam import ClipSpec

SINCE = "catalogue_v1"
SEED = 20261102
SOURCES = dict(G.SOURCES)
SOURCES.update({
    "duncan2011": "Duncan RP, Leddy AL, Earhart GM 2011, Arch Phys Med Rehabil 92: 1431-1436, doi "
                  "10.1016/j.apmr.2011.04.008 (full text): five-times sit-to-stand 20.25 (SD 14.12) s in 80 PD, 43 cm "
                  "chair, arms crossed; reported, not applied",
    "ling2012": "Ling H et al. 2012, Brain 135: 1141-1153, doi 10.1093/brain/aws038 (full text, Table 2): 15 s finger "
                "tapping, controls 50.0 taps, 45.9 deg, slope -0.12 deg per cycle; PD OFF 41.5 taps, 37.8 deg, slope "
                "-0.20 deg per cycle",
    "bologna2023": "Bologna M et al. 2023, Mov Disord 38: 551-557, doi 10.1002/mds.29362 (full text): the sequence "
                   "effect, a progressive reduction in amplitude and/or velocity during repetitive movement",
    "vercruysse2012": "Vercruysse S et al. 2012, Mov Disord 27: 254-263, doi 10.1002/mds.24015 (abstract): upper-limb "
                      "freezing preceded by hastened movement and a strong amplitude breakdown, broadband spectra "
                      "with more energy in the 3 to 8 Hz freeze band",
    "williams2013": "Williams AJ et al. 2013, Parkinsons Dis 2013: 595378 (full text), quoting Vercruysse et al.: an "
                    "upper-limb freeze is a halt of at least 75 percent of the mean cycle duration, or an amplitude at "
                    "or below 50 percent with irregular frequency for at least two cycle durations",
    "delval2017": "Delval A et al. 2017, PLoS One 12: e0181973 (full text): upper-limb freezing episodes 0.84 (SD 0.45) "
                  "s uncued, 1.10 (SD 0.60) s cued",
    "spont_unsourced": "no published rate, amplitude or duration of spontaneous movements during seated rest or quiet "
                       "standing in healthy or parkinsonian adults was found for this task (2026-11-02): module "
                       "values, a proposal for the neurologist",
    "dubois2005": "Dubois B, Slachevsky A, Pillon B et al. 2005, Neurology 64(12): 2132-2133, doi "
                  "10.1212/01.wnl.0000165977.38272.15 (abstract, as read in docs/proposals/SYMPTOM_CATALOGUE_DRAFT.md, "
                  "AP01): after the examiner claps three times, more than three claps is positive; positive in 30 of "
                  "42 PSP, 0 of 17 PD, 0 of 39 controls; the graded scoring 3, 4, 5 to 10, more than 10 claps of the "
                  "catalogue row was not traced to its source",
})
SOURCES.update({k: PH.SOURCES[k] for k in ("motamedifakhr2017", "melo2009", "foulger2025", "yamamoto2015",
                                           "winter1998")})    # the background's sources, as physio.py read them
BG_SOURCES = ("motamedifakhr2017", "melo2009", "foulger2025", "yamamoto2015")

# ---------------------------------------------------------------- background layer (breathing, drift, sway)
# The values and the signals of physio.py (shared change SH1; their sources in physio.SOURCES, read there), one layer
# per series (background_signals); the composition is this module's (module docstring, section 0).
BREATH_PER_MIN = PH.BREATH_PER_MIN       # 14.13: physio SOURCES['motamedifakhr2017'], REST1 median rate
BREATH_TI_TE = PH.BREATH_TI_TE           # 0.75: physio SOURCES['motamedifakhr2017'], REST1 median tI/tE
BREATH_RATE_CV = PH.BREATH_RATE_CV       # [module] physio's slow variation of the rate
BREATH_SPINE_DEG = PH.BREATH_SPINE_DEG   # [module] physio's spine3 extension at full inhalation
BREATH_NECK_SHARE = 0.7        # [module] the neck takes back this share of it, the gaze stays nearly level (physio:
#                                all of it)
BREATH_COLLAR_DEG = PH.BREATH_COLLAR_DEG   # [module] physio's collar elevation at full inhalation
HEAD_DRIFT_DEG = PH.HEAD_DRIFT_DEG       # [module] physio's RMS yaw and pitch of the head drift
HEAD_BAND_HZ = PH.HEAD_BAND_HZ           # [module] physio's band of the head drift
SIT_BAND_HZ = PH.SIT_BAND_HZ             # [module] physio's band of the seated trunk drift (its amplitude:
#                                physio.sit_gains, from physio SOURCES['melo2009'])
SWAY_RMS_M = PH.SWAY_RMS_M               # physio SOURCES['foulger2025']: hip-width stance, AP and ML RMS 4.33, 2.04 mm
SWAY_BAND_HZ = PH.SWAY_BAND_HZ           # physio SOURCES['yamamoto2015']: the low-frequency band of the sway
FINGER_DRIFT_DEG = PH.FINGER_DRIFT_DEG   # [module] physio's RMS of the MCP and PIP flexion drift of a resting hand
BG_RATIO = 0.75                # [module] the head and finger drift per 3.14 grade (breathing, sway not graded)
BG_VERSION = "park-bg-v2"      # v1 (catalogue_v1): this module's own signals and values, one draw per clip
TRANSFER_PARTS = {"breath": "throughout (spine3; the neck takes back BREATH_NECK_SHARE)",
                  "head": "throughout (neck and head)",
                  "trunk": "seated knots only, ramped over 0.3 s inside each seated run (spine1)",
                  "sway": "standing after a rise, weighted in over 0.5 s (the feet planted)",
                  "collars": "under the hanging arms only (after a release or the final extension)",
                  "fingers": "none (the arms folded, on the chair arms or on the thighs)"}   # an arising (E7, E22)
REST_PARTS = {"breath": "throughout (spine3; the neck takes back BREATH_NECK_SHARE, the arms ride on the chest)",
              "head": "throughout", "trunk": "throughout (spine1)", "sway": "none (seated)",
              "collars": "none (the forearms over the chair arms ride on the trunk)",
              "fingers": "both relaxed hands"}                              # the seated rest of 'sit' (motor state)
APPLAUSE_PARTS = dict(REST_PARTS, fingers="the relaxed hands at rest, before the raise and after the lowering (none "
                      "on the clapping hands)", collars="none (the clapping arms solved in spine3's frame)")
SERIES = {"arise": "park_E7_arise_g0", "spont_sit": "park_E1_spontaneity_g0",
          "spont_stand": "park_E7_spontaneity_g0", "sitdown": "park_E22_sitdown_level0",
          "applause": "park_E21_applause_g0", "motor_state": "park_E1_motorstate_level0"}   # the roots of physio.SKIP
#                                whose clips carry this module's background (one layer per root, background_signals)
# relaxed hand: (MCP, PIP, DIP) flexion and knuckle abduction deg (thumb: CMC (flexion, palmar abduction), MCP,
# IP); module values (a hand hanging or resting, the fingers slightly curled and closer together than at rest)
THUMB_ABD_SHIFT_DEG = -45.0
#   [module] open-body port, wave 6 (design D50's rule): Anny's rest thumb lies turned toward the palm, past its plane;
#   RELAXED_HAND's thumb takes this much less palmar abduction at the CMC, the least of 5 deg steps at which the 3.14
#   seated rest's hand (THIGH_REST, at its hand_weight) rests on a fingertip and not on its thumb (before: the thumb's
#   tip lowest, 3 mm over the thigh, the fingertips 32.5 to 40.9 mm and the wrist 40.8 mm over it; at -45 the middle
#   fingertip is lowest, the thumb 4.8 mm and the wrist 11.8 mm over it); scripts/openbody_wave6_checks.py, thigh_hands
RELAXED_HAND = {"index": ((14.0, -4.0), 18.0, 8.0), "middle": (16.0, 20.0, 8.0), "ring": ((18.0, -4.0), 22.0, 9.0),
                "pinky": ((20.0, -7.0), 24.0, 10.0), "thumb": ((8.0, 8.0 + THUMB_ABD_SHIFT_DEG), 8.0, 8.0)}


def _band(rng, t, band, n=8):
    """(T,) a unit-RMS sum of n sinusoids at frequencies drawn in band (Hz), with random phases and weights."""
    f = rng.uniform(*band, size=n)
    ph = rng.uniform(0.0, 2.0 * math.pi, size=n)
    a = rng.uniform(0.5, 1.0, size=n)
    x = (a[:, None] * np.sin(2.0 * math.pi * f[:, None] * t[None] + ph[:, None])).sum(0)
    return x / max(float(np.sqrt((x ** 2).mean())), 1e-12)


def _inside_ramp(mask, n):
    """(T,) 0 outside mask (T,) bool, rising by a minimum-jerk ramp to 1 over the first and last n knots inside each
    of its runs (so that a layer fades in and out where it is allowed)."""
    mask = np.asarray(mask, dtype=bool)
    out_k = np.flatnonzero(~mask)
    if not len(out_k):
        return np.ones(len(mask))
    k = np.arange(len(mask))
    i = np.clip(np.searchsorted(out_k, k), 1, len(out_k)) - 1
    d = np.minimum(np.abs(k - out_k[i]), np.abs(k - out_k[np.minimum(i + 1, len(out_k) - 1)]))
    return np.where(mask, _mj(d / max(n, 1)), 0.0)


def _rot(axis, deg):
    """(T, 3, 3) rotations about a body axis by deg (T,)."""
    return axis_rotation(axis, torch.from_numpy(np.radians(np.asarray(deg, dtype=np.float64))))


def relaxed_hand(weight=1.0, flex_offset=None):
    """(30, 3, 3) or (T, 30, 3, 3): the RELAXED_HAND pose on both sides, its angles times weight; flex_offset
    (T, 2, 5) deg adds to the MCP and PIP flexion of each finger (left, right; FINGERS order) per knot."""
    pose = {}
    for side in H.SIDES:
        for f, vals in RELAXED_HAND.items():
            for lv, v in enumerate(vals):
                fl, ab = (v, 0.0) if np.isscalar(v) else v
                pose[(side, f, lv)] = (weight * fl, weight * ab)
    R0 = H.hand_rotations(pose)
    if flex_offset is None:
        return R0
    T = flex_offset.shape[0]
    R = R0[None].repeat(T, 1, 1, 1)
    fr = H.hand_frame()
    for si, side in enumerate(H.SIDES):
        for fi, f in enumerate(H.FINGERS):
            for lv in (0, 1) if f != "thumb" else (1,):
                j = H.hand_joint(side, f, lv)
                ax = torch.as_tensor(fr[side]["joints"][j]["flex"], dtype=F64)
                off = axis_angle_to_matrix(ax[None] * torch.from_numpy(np.radians(flex_offset[:, si, fi]))[:, None])
                R[:, H.hand_row(j)] = off @ R[:, H.hand_row(j)]
    return R


def _collar_breath(R, deg, counter=False):
    """R (T, 22, 3, 3) in place: both collars elevated by deg (T,); with counter the shoulders turn back so that the
    upper arms keep their global direction (a shrug of the shoulder girdle, not of the arm), else the arms ride on
    the collars (breathing: 0.5 deg moves a hanging hand by about 5 mm; the hanging arm of 'stand' lies within
    0.4 deg of the healthy table's adduction limit, which leaves no room to turn it back under that table, so the
    breathing collars carry the arms; the standing 3.14 shrug turns them back, counter=True, under the table
    'shrug' of limits_ext, SHRUG_TABLE, whose adduction bound widens with the collar's rise)."""
    for c, s, sgn in ((13, 16, 1.0), (14, 17, -1.0)):
        E = _rot("z", sgn * np.asarray(deg))
        Rc = R[:, c].clone()
        R[:, c] = E @ Rc
        if counter:
            R[:, s] = Rc.transpose(-1, -2) @ E.transpose(-1, -2) @ Rc @ R[:, s]


SPINE = (3, 6, 9)


@functools.lru_cache(maxsize=64)
def _sagittal_shares(key):
    R0 = torch.as_tensor(np.asarray(key, dtype=np.float64).reshape(22, 3, 3))
    out = {}
    for sign in (1.0, -1.0):
        rooms = np.array([room_deg(R0[j], j, "x", sign, "parent") for j in SPINE])
        out[sign] = rooms / rooms.sum() if rooms.sum() > 0 else np.full(3, 1.0 / 3.0)
    return out


def sagittal_shares(R0):
    """{+1: (3,), -1: (3,)}: the shares of a sagittal bend over spine1 to spine3 (flexion +1, extension -1) by each
    joint's room in that direction at the base rotations R0 (22, 3, 3) (body.room_deg; open-body port, wave 6, the
    rule of design D40: the seated lumbar joint of 'sit' stands 2.5 deg from jarm2022's flexion bound, so the lean laid
    on spine1 alone, or by SPINE_SHARES, left the table)."""
    return _sagittal_shares(tuple(np.round(torch.as_tensor(R0).numpy().reshape(-1), 12)))


def _sagittal(R, deg, R0=None):
    """R (T, 22, 3, 3) in place: a sagittal bend deg (T,) about x over spine1 to spine3 by sagittal_shares of R0
    (default R's first knot), each knot's share by its sign, each joint's part before its rotation (parent frame)."""
    sh = sagittal_shares(R[0] if R0 is None else R0)
    d = np.asarray(deg, dtype=np.float64) * np.ones(R.shape[0])
    for i, j in enumerate(SPINE):
        R[:, j] = _rot("x", np.where(d >= 0.0, sh[1.0][i], sh[-1.0][i]) * d) @ R[:, j]


def _trunk_layer(R, breath, yaw_deg, pitch_deg, lean_deg, lean_ml_deg=None):
    """R (T, 22, 3, 3) in place: breathing (spine3 extension, the neck taking back BREATH_NECK_SHARE), the head
    drift (yaw and pitch shared half and half by neck and head) and the trunk's lean drift (sagittal over spine1 to
    spine3 by each joint's room at R's first knot, _sagittal, then lateral at spine1, in physio's order; open-body
    port, wave 6: the sagittal lean on spine1 alone left the table on the 'sit' seats)."""
    R0 = R[0].clone()
    R[:, 9] = _rot("x", -BREATH_SPINE_DEG * breath) @ R[:, 9]
    _sagittal(R, lean_deg, R0)
    if lean_ml_deg is not None:
        R[:, 3] = _rot("z", lean_ml_deg) @ R[:, 3]
    for j in (12, 15):
        R[:, j] = R[:, j] @ _rot("y", 0.5 * yaw_deg) @ _rot("x", 0.5 * pitch_deg)
    R[:, 12] = _rot("x", BREATH_NECK_SHARE * BREATH_SPINE_DEG * breath) @ R[:, 12]


def _layers(root, rate=100.0):
    """(seated, standing) physio layers (physio.make_layer, read-only, on physio's horizon) of the series root: the
    same seed (physio.series_seed) and the same draws, so that the breathing, the head drift and the fingers are
    one signal in both; the seated layer carries the trunk lean, the standing one the sway."""
    seed = PH.series_seed(root)
    return PH.make_layer(seed, "seated", float(rate)), PH.make_layer(seed, "standing", float(rate))


def background_signals(root, T, scale=1.0, standing=None, rate=100.0):
    """{breath, yaw, pitch, lean, lean_ml, sway_z, sway_x (T,), fingers (T, 2, 5)}: the first T knots of the
    background of the series root (module docstring, section 0: physio.make_layer of physio.series_seed(root), the
    same for every clip of the series); the head drift and the fingers times scale (the 3.14 ladder: fewer small
    movements); the breathing, the seated trunk's lean and the standing sway never scaled (postural sway, seated
    after physio SOURCES['melo2009'] and standing after physio SOURCES['foulger2025']: postural control, not a
    spontaneous movement, and no source read for a smaller sway with the grade); the sway times standing (T,) (0
    seated, 1 standing; None: none)."""
    Ls, Lt = _layers(root, rate)
    s, st = PH.layer_prefix(Ls, T), PH.layer_prefix(Lt, T)
    w = np.zeros(T) if standing is None else np.asarray(standing, dtype=np.float64)
    return {"breath": np.array(s["breath"]), "yaw": scale * s["head_yaw_deg"], "pitch": scale * s["head_pitch_deg"],
            "lean": np.array(s["lean_ap_deg"]), "lean_ml": np.array(s["lean_ml_deg"]),
            "sway_z": w * st["sway_ap_m"], "sway_x": w * st["sway_ml_m"], "fingers": scale * s["fingers_deg"]}


def _laid(parts, part):
    """False when the parts record ({component: where}) says the clip does not lay this component ('none ...')."""
    return not str((parts or {}).get(part, "")).startswith("none")


def background_record(root, T, scale=1.0, rate=100.0, parts=None):
    """JSON-ready record of a clip's background (truth params 'background'): the series, physio's seed, version and
    layer digests (physio.layer_digest of the first T knots of the seated and the standing layer: equal on every clip
    of a series), the scale (3.14: the head drift and the fingers), the amplitudes and their sources, the parts the
    clip lays (parts: {component: where}), the forward offset of the sagittal lean (its series mean, deg); a
    component the parts say the clip does not lay ('none ...') records the amplitude 0."""
    Ls, Lt = _layers(root, rate)
    g = PH.sit_gains()
    trunk, sway = _laid(parts, "trunk"), _laid(parts, "sway")
    return {"version": BG_VERSION, "series": root, "physio_version": PH.VERSION, "seed": PH.series_seed(root),
            "layer_sha256": {"seated": PH.layer_digest(Ls, T), "standing": PH.layer_digest(Lt, T)}, "knots": int(T),
            "scale": float(scale), "breath_per_min": BREATH_PER_MIN, "breath_ti_te": BREATH_TI_TE,
            "breath_spine_deg": BREATH_SPINE_DEG, "breath_neck_share": BREATH_NECK_SHARE,
            "breath_collar_deg": BREATH_COLLAR_DEG if _laid(parts, "collars") else 0.0,
            "head_drift_rms_deg": [scale * x for x in HEAD_DRIFT_DEG], "head_band_hz": list(HEAD_BAND_HZ),
            "trunk_lean_rms_deg": [float(x) if trunk else 0.0 for x in g["lean_rms_deg"]],
            "trunk_band_hz": list(SIT_BAND_HZ),
            "trunk_forward_offset_mean_deg": float(np.mean(Ls["lean_ap_deg"])) if trunk else 0.0,
            "sway_rms_m": [float(x) if sway else 0.0 for x in SWAY_RMS_M], "sway_band_hz": list(SWAY_BAND_HZ),
            "finger_drift_rms_deg": scale * FINGER_DRIFT_DEG if _laid(parts, "fingers") else 0.0,
            "parts": dict(parts or {}), "sources": list(BG_SOURCES) + ["module"],
            "rule": ("one background per series (physio.make_layer of the root's seed), identical on a clip and its "
                     "level 0; the 3.14 ladder scales the head drift and the fingers by BG_RATIO^grade, never the "
                     "breathing nor the postural sway (seated trunk lean, standing sway)")}


def legs_to(R, tr, ankles, G_feet, hinge_x):
    """(T, 22, 3, 3): R (T, 22, 3, 3) with the root already placed (row 0 and tr (T, 3)) and the hips, knees and
    ankles re-solved (ik.hinge_leg) so that the ankle joints lie at ankles (T, 2, 3) and the feet keep their global
    rotations G_feet (T, 2, 3, 3); hinge_x (T, 2, 3): the knee hinge directions (each thigh's x axis in a reference
    pose turned with the pelvis)."""
    from .ik import hinge_leg
    rest = rest_joints()
    R = R.clone()
    J, Gl = fk(R, tr, rest)
    G0 = Gl[:, 0]
    for i, (h, kn, a) in enumerate(((1, 4, 7), (2, 5, 8))):
        Hh, A = J[:, h], ankles[:, i]
        G_th, kap = hinge_leg(Hh, A, Hh + G0[:, :, 2], rest[kn] - rest[h], rest[a] - rest[kn], G0,
                              hinge=hinge_x[:, i])
        Rk = axis_angle_to_matrix(torch.stack([kap, 0 * kap, 0 * kap], -1))
        R[:, h] = G0.transpose(-1, -2) @ G_th
        R[:, kn] = Rk
        R[:, a] = (G_th @ Rk).transpose(-1, -2) @ G_feet[:, i]
    return R


REACH_SMOOTH_S = 0.05          # [module] the pelvis drop of place_pelvis(keep_reach) smoothed over this
REACH_MARGIN_M = 1e-10         # [module] keep_reach holds each leg this much short of its reference reach (open-body
#                                port, wave 6, as postures.HIP_MARGIN_M, design D55: 1e-7 m bends a straight knee by its
#                                square root; the standing 3.14 ladder's one-knot leg kink is not this margin's, it
#                                stays, design 17.9)


def place_pelvis(R, tr, J0, G0, ankles, ref_R, ref_tr, keep_reach=True, rate=100.0):
    """(R, tr): the root of R (T, 22, 3, 3) moved to the joint position J0 (T, 3) and global rotation G0 (T, 3,
    3), the legs re-solved to the ankle positions ankles (T, 2, 3) with the feet's global rotations and the knee
    hinge directions of the reference pose (ref_R (T, 22, 3, 3), ref_tr (T, 3)) carried with the pelvis; with
    keep_reach the pelvis is lowered where a leg would reach further than in the reference (no leg stretches), the
    drop smoothed (a moving minimum over REACH_SMOOTH_S on each side, then two moving means over it: never less
    than the drop needed, no corner where the clamp sets in; the clipped drop kinked a nearly straight knee, task
    sh1b review: 1.6 m/s2 at one knot of park_E7_spontaneity_g0)."""
    rest = rest_joints()
    Jr, Gr = fk(ref_R, ref_tr, rest)
    J0 = J0.clone()
    if keep_reach:
        drop = torch.zeros(J0.shape[0], dtype=F64)
        for i, h in enumerate((1, 2)):
            d0 = (Jr[:, 7 + i] - Jr[:, h]).norm(dim=-1) - REACH_MARGIN_M
            Hh = J0 + torch.einsum("tij,j->ti", G0, rest[h] - rest[0])
            w = ankles[:, i] - Hh
            S = (d0 ** 2 - w[:, 0] ** 2 - w[:, 2] ** 2).clamp_min(0.0)
            drop = torch.minimum(drop, w[:, 1] + torch.sqrt(S))
        n = max(1, int(round(REACH_SMOOTH_S * rate)))
        if len(drop) > 2 * n:
            d = drop.numpy()
            lo = np.lib.stride_tricks.sliding_window_view(np.pad(d, (n, n), mode="edge"), 2 * n + 1).min(1)
            drop = torch.from_numpy(np.minimum(_mean_edge(_mean_edge(lo, n), n), d))
        J0[:, 1] += drop
    R = R.clone()
    R[:, 0] = G0
    tr2 = J0 - rest[0]
    turn = G0 @ Gr[:, 0].transpose(-1, -2)
    hinge = torch.stack([turn @ Gr[:, 1, :, 0][..., None], turn @ Gr[:, 2, :, 0][..., None]], 1)[..., 0]
    R = legs_to(R, tr2, ankles, torch.stack([Gr[:, 7], Gr[:, 8]], 1), hinge)
    return R, tr2


def stand_sway(R, tr, dz, dx, ref_R=None, ref_tr=None):
    """(R, tr) of a standing body (R (T, 22, 3, 3), tr (T, 3)) with the pelvis swaying dz forward (an ankle
    strategy: the whole body turns about the ankles' axis) and dx sideways (a hip strategy: the pelvis shifts,
    the trunk stays upright), the feet planted (place_pelvis)."""
    rest = rest_joints()
    ref_R = R if ref_R is None else ref_R
    ref_tr = tr if ref_tr is None else ref_tr
    J, Gl = fk(R, tr, rest)
    A = 0.5 * (J[:, 7] + J[:, 8])
    h = (J[:, 0, 1] - A[:, 1]).numpy()
    th = np.arctan2(np.asarray(dz, dtype=np.float64), h)
    Rx = axis_rotation("x", torch.from_numpy(th))
    J0 = A + torch.einsum("tij,tj->ti", Rx, J[:, 0] - A)
    J0[:, 0] += torch.from_numpy(np.asarray(dx, dtype=np.float64))
    return place_pelvis(R, tr, J0, Rx @ Gl[:, 0], torch.stack([J[:, 7], J[:, 8]], 1), ref_R, ref_tr)


# ---------------------------------------------------------------- 3.9 arising
SEATED_S = 0.5                 # template: seated before its rise
SLOW_RISE_S = 2.8              # [module] grade 1 (slow): about the control mean + 2.4 SD (SOURCES['inkster2004'])
SLOW_FAST_U = (0.29, 0.34, 0.56, 0.62)  # [module] the slow rise keeps the template's pace from 0.34 to 0.56 of the
#                                rise (the momentum transfer around lift-off), ramps over the outer pairs
SLOW_HESITATE = (0.2, 0.035, 0.85)      # [module] a hesitation of the forward lean: centre and width (share of
#                                the rise) and the depth of the pace's dip
FAIL_U = (0.55, 0.66)          # [module] share of the rise a failed folded attempt reaches (lift-off at 0.444; the
#                                centre of mass then lies at most about 3 cm ahead of the heels: it falls back)
PUSH_FAIL_U = (0.64, 0.70)     # [module] a failed push-off, capped where an arm reaches PUSH_PEAK_EXT; open-body port,
#                                wave 6 follow-up: (0.66, 0.72) before, where on the open body's arising the hips rise
#                                10.01 and 17.41 cm; the width kept, the top the largest share of a 0.02 grid at which
#                                they rise less than the 15 cm its test reads as a few centimetres (14.87 cm at 0.70;
#                                scripts/openbody_wave6_checks.py, push_off)
PUSH_PEAK_EXT = 0.97           # [module] the arms of a failed push-off extend to this share of their length at most
FAIL_TRUNK_SHARE = 0.5         # [module] trunk bout of the 'fail' template, share of the solved one
FAIL_TRUNK_RANGE = (0.4, 0.6)  # [module] trunk bout of a failed folded attempt, share of the solved one
PUSH_TRUNK_RANGE = (0.6, 0.9)  # [module] trunk bout of a failed push-off (the arms push: the trunk leans more)
TIMING_VAR = 0.15              # [module] an attempt's rise time varies within this share
UP_STRETCH = 1.15              # [module] a failed attempt rises over this times its share of the rise time
PUSH_EASE = 0.25               # [module] a failed push-off rises at a steady pace through its middle (the arms push
#   steadily; the template's own pace is steady), easing in and out over this share of its rise time (_eased: the
#   pace's ramps smoothstep, its peak 1 / (1 - PUSH_EASE) times its mean); on the minimum-jerk profile of the folded
#   attempts (peak 1.875 times the mean) the push-offs of park_E7_arise_g3 and g4 (fe89df3) turned the pelvis-to-neck
#   line at 145 to 191 deg/s at 100 Hz, against the healthy rise's 130 (wave SH1d fix B); the rise is then stretched
#   until the trunk keeps TRUNK_RATE_MAX (_attempt_params, push_rise_stretch)
FALL_S = (0.4, 0.5)            # [module] the drop back onto the seat
FALL_SKEW = 1.3                # [module] the drop accelerates: a minimum-jerk profile of x^FALL_SKEW
TRUNK_LEAD = 1.0               # [module] the trunk returns over at least the whole drop and goes on past the seat
#                                contact by AFTER_SEAT_S (wave SH1d part B, review M10: at 0.75 of a 0.4 s drop the
#                                trunk swung back at 150 to 233 deg/s, a whiplash against the healthy rise's 108)
AFTER_SEAT_S = (0.3, 0.5)      # [module] the trunk still returning after the seat contact, at least the first value,
#                                longer when TRUNK_RATE_MAX asks it, at most the second
TRUNK_RATE_MAX = 120.0         # [module] the trunk's return turns at most at this speed (deg/s, minimum-jerk peak),
#                                about the healthy rise's extension (the review's 108 to 110 deg/s)
HEAD_ACCEL_MAX = 12.0          # [module] a failed attempt's head joint accelerates at most this much (m/s2, the
#                                healthy rise's 11 at 30 frames per second; continuity_report reads it)
PAUSE_S = (0.8, 2.0)           # [module] a seated pause after a failed attempt
FALL_MARGIN_M = 0.005          # [module] a failed folded attempt's XcoM at least this far behind the heels at some
#                                knot after lift-off (its trunk share is lowered until it is)
ARM_MOVE_S = 1.2               # [module] the hands from the folded grip to the chair arms
ARM_WAYPOINT_M = (0.0, 0.12, 0.04)   # [module] through a point this far above and ahead of each wrist target
ARM_WAYPOINT_AT = 0.6          # [module] reached at this share of the move (each part minimum-jerk)
RELEASE_EXT = 0.95             # [module] the hands leave the chair arms at this share of the arm's length
RELEASE_S = 0.7                # [module] the arms then blend to hanging
RELEASE_LIFT_S = 0.15          # [module] first the elbows flex by RELEASE_ELBOW_DEG over this: the hands lift off
RELEASE_ELBOW_DEG = 15.0       # [module]
RELEASE_EASE_S = 0.15          # [module] the hands ease off the chair arms over this up to the release knot (the
#                                push-off rows cross-faded into the release's; task sh1c review: with the rows held
#                                from the release knot the wrists' speed stepped up in one knot)
PALM_CLEAR_M = 0.003           # [module] the lowest hand vertex this far above the armrest top
PUSH_THIGH_CLEAR_M = 0.005     # [module] the push-off forearms' and hands' skin this far off the thighs over the
#                                rehearsal of push_hand_yaw (at the end of the rise the thighs rise toward the hands)
PUSH_YAW_STEP_DEG = 2.5        # [module] the outward turn of the push-off hands (push_hand_yaw) on this grid
PUSH_YAW_MAX_DEG = 30.0        # [module] up to this
PUSH_THIGH_GUARD_M = 0.002     # [module] a clip's guards: the hands leave the chair arms before a skin comes this near
#                                a thigh, and a failed push-off's peak is lowered until they stay this far (task sh1c)
WRIST_BACK_M = 0.0             # [module] the wrist this far behind the armrest's front end (the fingers over it);
#                                open-body port, wave 6 follow-up: at the front end the hands grip the armrests where
#                                one pushes up from them (0.08 before: as the trunk leaned the shoulders' y component
#                                passed jarm2022's horizontal extension of 30 deg, and no failed push-off of 3.9
#                                grades 3 and 4 fitted the arms); scripts/openbody_wave6_checks.py, push_off
WRIST_EXT_CAP_DEG = 70.0       # [module] wrist extension cap (the table's upper percentile is 77): beyond it the
#                                hand pitches about the wrist and the fingers leave the armrest
ARISE_DUR_S = 15.0             # every 3.9 clip and its level 0
POLE_OUT, POLE_BACK = 0.5, 1.0  # [module] the elbows point back and a little out while the hands push (arm_ik);
#                                open-body port, wave 6 follow-up: 0.3 before; with the hands at the armrests' front
#                                ends (WRIST_BACK_M) the least of 0.1 steps at which the shoulders stay inside the table
#                                both on the failed push-off of the largest trunk share and peak (0.4) and on the push
#                                rise of 3.9 grade 2 up to its over-reach (at 0.4 its shoulders passed the 30 deg of
#                                horizontal extension as the trunk came over the hands); scripts/openbody_wave6_checks.py,
#                                push_off
EXT_SIGN = {"left": 1.0, "right": -1.0}     # wrist extension about z in the parent frame (dystonia.REGIONS)
ARM = {"left": (13, 16, 18, 20), "right": (14, 17, 19, 21)}
OUT = {"left": 1.0, "right": -1.0}
ARISE_PLANS = {      # ('sit', None): a pause drawn in PAUSE_S
    "g0": [("sit", 1.0), ("rise", "normal", "folded")],
    "g1_slow": [("sit", 1.0), ("rise", "slow", "folded")],
    "g1_attempts": [("sit", 1.0), ("attempt", "fail", "folded"), ("sit", None), ("rise", "normal", "folded")],
    "g2": [("sit", 1.0), ("attempt", "fail", "folded"), ("sit", None), ("arms", ARM_MOVE_S), ("sit", 0.3),
           ("rise", "normal", "push")],
    "g3": [("sit", 1.0), ("attempt", "fail", "folded"), ("sit", None), ("arms", ARM_MOVE_S), ("sit", 0.3),
           ("attempt", "fail", "push"), ("sit", None), ("rise", "normal", "push")],
    "g4": [("sit", 0.8), ("attempt", "fail", "folded"), ("sit", None), ("attempt", "fail", "folded"), ("sit", None),
           ("arms", ARM_MOVE_S), ("sit", 0.3), ("attempt", "fail", "push"), ("sit", None),
           ("attempt", "fail", "push")],
}
ARISE_GRADES = {"g0": 0, "g1_slow": 1, "g1_attempts": 1, "g2": 2, "g3": 3, "g4": 4}


def _mj(x):
    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)
    return x ** 3 * (10.0 - 15.0 * x + 6.0 * x ** 2)


def _eased(x, a=PUSH_EASE):
    """(x,) from 0 to 1 at a steady pace between a and 1 - a, eased in and out over a (the pace a smoothstep ramp,
    3 y^2 - 2 y^3: speed and acceleration zero at both ends, as _mj); a in (0, 0.5]."""
    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)

    def ramp(y):                                         # the integral of the smoothstep from 0 to y
        return y ** 3 - 0.5 * y ** 4
    out = np.where(x < a, a * ramp(x / a), 0.5 * a + (x - a))
    out = np.where(x > 1.0 - a, (1.0 - a) - a * ramp((1.0 - x) / a), out)
    return out / (1.0 - a)


@functools.lru_cache(maxsize=4)
def arise_template(kind):
    """(HandBody, info) of a template rise (module docstring): 'normal' (motions.arise at the cited pace, solved),
    'fail' (the cited pace, the normal standing shank tilt, FAIL_TRUNK_SHARE of the normal trunk bout). The slow
    rise of grade 1 reads the 'normal' template on a slower clock (slow_rise_u). Cached: a solve takes about 12 s
    on two threads."""
    from .motions import ARISE_S, arise
    if kind == "normal":
        return arise(seated_s=SEATED_S, standing_s=0.5, rise_s=ARISE_S)
    if kind == "fail":
        _, ni = arise_template("normal")
        return arise(seated_s=SEATED_S, standing_s=0.5, rise_s=ARISE_S,
                     stand_tilt_deg=ni["standing_shank_tilt_deg"],
                     trunk_deg=FAIL_TRUNK_SHARE * ni["trunk"]["amplitude_deg"])
    raise ValueError(f"template {kind!r}")


def _slow_pace(u, r_slow):
    """(N,) the slow rise's pace (template shares per template second) at shares u: r_slow outside SLOW_FAST_U, the
    template's pace (1) inside, minimum-jerk ramps, the hesitation's dip (SLOW_HESITATE)."""
    a0, a1, b0, b1 = SLOW_FAST_U
    win = _mj((u - a0) / (a1 - a0)) * (1.0 - _mj((u - b0) / (b1 - b0)))
    c, w, d = SLOW_HESITATE
    return (r_slow + (1.0 - r_slow) * win) * (1.0 - d * np.exp(-0.5 * ((u - c) / w) ** 2))


@functools.lru_cache(maxsize=2)
def slow_rise_u(rate=100.0):
    """((N,) shares of the 'normal' template at the knots of the slow rise (N = SLOW_RISE_S x rate, the last at 1),
    info): the template read on a slower clock (module docstring): its pace outside SLOW_FAST_U is solved by
    bisection so that the rise lasts SLOW_RISE_S; the trunk lean is the template's (no deeper bow)."""
    _, ni = arise_template("normal")
    rs = ni["rise_s"]
    ug = np.linspace(0.0, 1.0, 20001)

    def times(r):
        inv = 1.0 / _slow_pace(ug, r)
        return rs * np.r_[0.0, np.cumsum(0.5 * (inv[1:] + inv[:-1]) * np.diff(ug))]
    lo, hi = 0.05, 1.0
    for _ in range(60):
        m = 0.5 * (lo + hi)
        lo, hi = (m, hi) if times(m)[-1] > SLOW_RISE_S else (lo, m)
    r = 0.5 * (lo + hi)
    tg = times(r)
    n = int(round(SLOW_RISE_S * rate))
    tk = (np.arange(n) + 1) / rate
    u = np.interp(tk * tg[-1] / SLOW_RISE_S, tg, ug)
    from .motions import LIFTOFF_SHARE
    c, w, _ = SLOW_HESITATE
    return u, {"rise_s": SLOW_RISE_S, "template_rise_s": rs, "slow_pace": r, "fast_u": list(SLOW_FAST_U),
               "hesitation_u": [c - 2 * w, c + 2 * w],
               "hesitation_s": float(np.interp(c + 2 * w, ug, tg) - np.interp(c - 2 * w, ug, tg)),
               "liftoff_s": float(np.interp(LIFTOFF_SHARE, ug, tg)),
               "after_liftoff_s": float(tg[-1] - np.interp(LIFTOFF_SHARE, ug, tg))}


def _rot_about(axis, ang):
    """(N, 3, 3) rotations about fixed unit axis (3,) by angles (N,) rad."""
    a = torch.as_tensor(np.asarray(ang, dtype=np.float64))
    return axis_angle_to_matrix(a[:, None] * torch.as_tensor(axis, dtype=F64)[None])


def _twist(M, axis):
    """(N,) rad: the twist angle of rotations M (N, 3, 3) about a unit axis (3,) (swing-twist decomposition)."""
    aa = matrix_to_axis_angle(M)
    th = aa.norm(dim=-1)
    n = aa / th.clamp_min(1e-15)[:, None]
    v = n * torch.sin(th / 2)[:, None]
    return 2.0 * torch.atan2(v @ torch.as_tensor(axis, dtype=F64), torch.cos(th / 2))


@functools.lru_cache(maxsize=1)
def armrest_targets():
    """{side: wrist target (3,) m}, the hand rotation (side: (3, 3) global), the chair arms' tops and 'hand_yaw_deg':
    _armrest_base with each hand turned outward about the vertical by push_hand_yaw (the fingers forward and out:
    the thumb side clears the rising thigh; task sh1c)."""
    yaw, rec = push_hand_yaw()
    return dict(_yawed_targets(_armrest_base(), yaw), hand_yaw=rec)


def _yawed_targets(base, yaw_deg):
    """armrest_targets' dict base with each hand rotation turned outward about the vertical by yaw_deg (the left
    hand about +y, the right about -y: OUT)."""
    out = {"wrist": dict(base["wrist"]), "armrest": dict(base["armrest"]), "hand_yaw_deg": float(yaw_deg),
           "hand_R": {s: _rot("y", np.array([OUT[s] * yaw_deg]))[0] @ base["hand_R"][s] for s in ("left", "right")}}
    return out


@functools.lru_cache(maxsize=1)
def push_hand_yaw(rate=100.0):
    """(yaw deg, record): the smallest outward turn of the push-off hands on the grid of PUSH_YAW_STEP_DEG up to
    PUSH_YAW_MAX_DEG whose rehearsal keeps the forearms and hands PUSH_THIGH_CLEAR_M off the thighs
    (push_thigh_gap) until the arms would leave the chair arms: the 'normal' template's rise with the hands on the
    chair arms (push_arms), up to its first knot that over-reaches (RELEASE_EXT) or leaves the table (task sh1c: the
    hands palm down with the fingers forward, the thumb side of the right hand entered the rising thigh). Cached."""
    _, ni = arise_template("normal")
    n = int(round(ni["rise_s"] * rate))
    u = (np.arange(n) + 1) / n
    R, tr, _ = _template_pose("normal", u)
    base = _armrest_base()
    reach = np.maximum(reach_share(R, tr, "left", base["wrist"]["left"]),
                       reach_share(R, tr, "right", base["wrist"]["right"]))
    over = np.flatnonzero(reach > RELEASE_EXT)
    end = int(over[0]) if len(over) else n
    tried = []
    for yaw in np.arange(0.0, PUSH_YAW_MAX_DEG + 1e-9, PUSH_YAW_STEP_DEG):
        tg = _yawed_targets(base, float(yaw))
        Rp, rr, ok, _ = push_arms(R[:end], tr[:end], tg=tg)
        bad = np.flatnonzero(~ok)
        e = int(bad[0]) if len(bad) else end
        g = float(push_thigh_gap(R[:e], tr[:e], Rp[:e]).min())
        tried.append({"yaw_deg": float(yaw), "thigh_gap_min_mm": 1000.0 * g, "knots": e,
                      "reach_end": float(rr[e - 1])})
        if g >= PUSH_THIGH_CLEAR_M:
            return float(yaw), {"yaw_deg": float(yaw), "tried": tried, "clear_mm": 1000.0 * PUSH_THIGH_CLEAR_M}
    raise RuntimeError(f"no outward turn of the push-off hands keeps them off the thighs: {tried}")


@functools.lru_cache(maxsize=1)
def _armrest_base():
    """{side: wrist target (3,) m}, the hand rotation (side: (3, 3) global) and the chair arms' tops: the hands
    palm down on the chair arms of motions.arise_chair under the normal template's seated knot (module
    docstring). The hand's global rotation is the 'sit' posture's (palms down, fingers forward); the wrist lies
    above the armrest top by the wrist joint's height over the lowest hand vertex in that rotation, plus
    PALM_CLEAR_M, WRIST_BACK_M behind the armrest's front end, on its midline."""
    from .motions import arise_chair
    from .viewer import mesh_floor_y
    body, _ = arise_template("normal")
    V, _ = arise_chair(body, mesh_floor_y(0.0))
    boxes = V.reshape(-1, 8, 3)
    arms = {"left": boxes[6], "right": boxes[7]}          # sgn +1 (x > 0, the body's left), then -1
    sit = H.hands_of(static_body("sit", 0.01))
    J, Gl = sit.fk()
    _, _, dom = H._rest()
    out = {"hand_R": {}, "wrist": {}, "armrest": {}}
    for side, w in (("left", 20), ("right", 21)):
        ids = np.flatnonzero(np.isin(dom, (w,) + tuple(range(H.FIRST[side], H.FIRST[side] + 15))))
        P = H.body_points(sit, ids, knots=[0])[0]
        drop = float(J[0, w, 1] - P[:, 1].min())
        b = arms[side]
        top, front = float(b[:, 1].max()), float(b[:, 2].max())
        xc = float(0.5 * (b[:, 0].min() + b[:, 0].max()))
        out["hand_R"][side] = Gl[0, w].clone()
        out["wrist"][side] = torch.tensor([xc, top + drop + PALM_CLEAR_M, front - WRIST_BACK_M], dtype=F64)
        out["armrest"][side] = {"top_y_m": top, "front_z_m": front, "x_centre_m": xc, "palm_drop_m": drop,
                                "box_lo_m": b.min(0).tolist(), "box_hi_m": b.max(0).tolist()}
    return out


def arm_ik(R, tr, side, W, G_hand, keep_collar=False, pole=(None, None)):
    """(R_collar, R_shoulder, R_elbow, R_wrist) (N, 3, 3) local rotations that put the wrist joint at W ((3,) or
    (N, 3)) with the hand's global rotation G_hand ((3, 3) or (N, 3, 3)), the collar at rest or, with keep_collar,
    as in R (module docstring), for body rotations R (N, 22, 3, 3) and translations tr (N, 3); with the arm's
    reach share d / (L1 + L2) (N,). pole: (out, back) weights of the elbow's pole, scalars or (N,) (default
    POLE_OUT, POLE_BACK)."""
    c, s, e, w = ARM[side]
    rest = rest_joints()
    R = R.clone()
    if not keep_collar:
        R[:, c] = torch.eye(3, dtype=F64)
    J, Gl = fk(R, tr, rest)
    S, Gc = J[:, s], Gl[:, c]
    p, q = rest[e] - rest[s], rest[w] - rest[e]
    C = axis_rotation("z", torch.tensor(-math.pi / 2 if side == "left" else math.pi / 2))
    pp, qq = C.T @ p, C.T @ q
    N = R.shape[0]
    Wn = torch.as_tensor(W, dtype=F64)
    Wn = Wn[None].expand(N, 3) if Wn.dim() == 1 else Wn
    Gh = torch.as_tensor(G_hand, dtype=F64)
    Gh = Gh[None].expand(N, 3, 3) if Gh.dim() == 2 else Gh
    po = torch.as_tensor(np.broadcast_to(POLE_OUT if pole[0] is None else pole[0], (N,)).copy(), dtype=F64)
    pb = torch.as_tensor(np.broadcast_to(POLE_BACK if pole[1] is None else pole[1], (N,)).copy(), dtype=F64)
    fwd = torch.tensor([0.0, 0.0, 1.0], dtype=F64)
    out_ = torch.tensor([OUT[side], 0.0, 0.0], dtype=F64)
    pole_pt = 0.5 * (S + Wn) + 0.3 * (po[:, None] * out_[None] - pb[:, None] * fwd[None])
    reach = (Wn - S).norm(dim=-1) / float(p.norm() + q.norm())
    from .ik import hinge_leg
    Gp, k = hinge_leg(S, Wn, pole_pt, pp, qq, Gc @ C)
    Gs = Gp @ C.T
    Rs = Gc.transpose(-1, -2) @ Gs
    Re = C[None] @ axis_angle_to_matrix(torch.stack([k, 0 * k, 0 * k], -1)) @ C.T[None]
    qh = (q / q.norm()).numpy()
    Rw0 = (Gs @ Re).transpose(-1, -2) @ Gh
    phi = _twist(Rw0, qh).numpy()
    Re2 = Re @ _rot_about(qh, phi / 2)
    Rw = _rot_about(qh, -phi / 2) @ Rw0
    for _ in range(3):                                   # the extension cap: pitch the hand about the wrist
        ext = EXT_SIGN[side] * np.degrees(matrix_to_axis_angle(Rw).numpy()[:, 2])
        over = np.maximum(ext - WRIST_EXT_CAP_DEG, 0.0)
        if not over.any():
            break
        Rw = _rot_about(np.array([0.0, 0.0, 1.0]), -EXT_SIGN[side] * np.radians(over)) @ Rw
    Rc = R[:, c].clone() if keep_collar else torch.eye(3, dtype=F64).expand(N, 3, 3).clone()
    return Rc, Rs, Re2, Rw, reach.numpy()


def reach_share(R, tr, side, W):
    """(N,) the shoulder-to-wrist-target distance over the arm's length, the collar at rest (arm_ik)."""
    c, s, e, w = ARM[side]
    rest = rest_joints()
    R = R.clone()
    R[:, c] = torch.eye(3, dtype=F64)
    J, _ = fk(R, tr, rest)
    L = float((rest[e] - rest[s]).norm() + (rest[w] - rest[e]).norm())
    return ((torch.as_tensor(W, dtype=F64)[None] - J[:, s]).norm(dim=-1) / L).numpy()


def _blend(R0, R1, w):
    """(N, n, 3, 3): R0 (N, n, 3, 3) turned toward R1 (N, n, 3, 3) along the geodesic by shares w (N,)."""
    D = matrix_to_axis_angle(R0.transpose(-1, -2) @ R1)
    W = torch.as_tensor(np.asarray(w, dtype=np.float64))
    return R0 @ axis_angle_to_matrix(W[:, None, None] * D)


def _arm_joints():
    return [j for s in ("left", "right") for j in ARM[s]]


def _limits_ok_rows(R, joints):
    """(N,) bool: every component of the given joints' local rotations R (N, 22, 3, 3) inside the SP1 table."""
    from .body import LIMIT_TOL_DEG, joint_limits
    lim = joint_limits()
    a = np.degrees(matrix_to_axis_angle(R[:, joints]).numpy())
    lo, hi = lim.lo[joints][None], lim.hi[joints][None]
    return ((a >= lo - LIMIT_TOL_DEG) & (a <= hi + LIMIT_TOL_DEG)).all(-1).all(-1)


def _x_angle(M):
    """(N,) deg: the angle of rotations M (N, 3, 3) about x (the arising's pelvis and spine turn about x only)."""
    return np.degrees(matrix_to_axis_angle(M).numpy()[:, 0])


def _template_pose(name, u):
    """(R (N, 22, 3, 3), tr (N, 3), Rh (N, 30, 3, 3)) of a template at shares u (N,) of its rise."""
    from .viewer import hands_at, pose_at
    body, info = arise_template(name)
    tau = SEATED_S + np.asarray(u, dtype=np.float64) * info["rise_s"]
    r, x = pose_at(body, tau)
    return r, x, hands_at(body, tau)


def _with_trunk_share(R, u, share):
    """R (N, 22, 3, 3) of the 'fail' template at shares u, its trunk bout turned to share (N,) of the normal one
    (the bout adds about x to TRUNK_JOINTS in the proportions of motions.trunk_split; the legs do not depend on
    it)."""
    from .motions import TRUNK_JOINTS, trunk_profile, trunk_split
    _, ni = arise_template("normal")
    split, _ = trunk_split()
    d = (np.asarray(share) - FAIL_TRUNK_SHARE) * ni["trunk"]["amplitude_deg"] * trunk_profile(np.asarray(u))
    R = R.clone()
    for j, s in zip(TRUNK_JOINTS, split):
        R[:, j] = _rot("x", s * d) @ R[:, j]
    return R


def _attempt_params(rng, arms):
    """{arms, trunk_share, u_peak, timing_factor, fall_s} of a failed attempt drawn from rng (module docstring); a
    push-off's u_peak capped where an arm would reach PUSH_PEAK_EXT."""
    push = arms == "push"
    share = float(rng.uniform(*(PUSH_TRUNK_RANGE if push else FAIL_TRUNK_RANGE)))
    u_pk = float(rng.uniform(*(PUSH_FAIL_U if push else FAIL_U)))
    f = float(rng.uniform(1.0 - TIMING_VAR, 1.0 + TIMING_VAR))
    fall = float(rng.uniform(*FALL_S))
    from .motions import LIFTOFF_SHARE, balance
    _, fi = arise_template("fail")
    par = {"arms": arms, "trunk_share": share, "u_peak": u_pk, "timing_factor": f, "fall_s": fall}
    while not push:                 # a folded attempt falls back: its XcoM behind the heels after lift-off
        R, tr, _, u, nu, _ = _attempt(par, 100.0)
        b = H.HandBody(matrix_to_rot6d(R), tr, torch.zeros(len(u), 2, dtype=torch.bool))
        k_lift = int(np.argmax(u > LIFTOFF_SHARE))
        par["xcom_margin_min_m"] = float(balance(b, k_lift)["posterior"][k_lift:nu].min())
        if par["xcom_margin_min_m"] < -FALL_MARGIN_M or par["trunk_share"] <= 0.25:
            break
        par["trunk_share"] -= 0.02
    while push:      # lower the peak until the arms reach, stay inside the table and the hands above the chair arms
        R, tr, _, _, _, _ = _attempt(par, 100.0)
        tg = armrest_targets()
        reach = np.maximum(reach_share(R, tr, "left", tg["wrist"]["left"]),
                           reach_share(R, tr, "right", tg["wrist"]["right"]))
        ok = reach.max() <= PUSH_PEAK_EXT
        if ok:
            Rp, reach, inside, _ = push_arms(R, tr)
            ok = bool(inside.all()) and reach.max() <= PUSH_PEAK_EXT
        if ok:                          # the thighs rise toward the hands: the peak keeps them PUSH_THIGH_GUARD_M off
            par["thigh_gap_min_m"] = float(push_thigh_gap(R, tr, Rp).min())
            ok = par["thigh_gap_min_m"] >= PUSH_THIGH_GUARD_M
        if ok:
            break
        if par["u_peak"] <= 0.5:
            raise RuntimeError("no failed push-off fits the arms")
        par["u_peak"] -= 0.005
    if push:                        # the arms brake the drop: it lasts long enough for the trunk to keep TRUNK_RATE_MAX
        par["fall_s"] = max(fall, 1.875 * trunk_turn_deg(par) / TRUNK_RATE_MAX / TRUNK_LEAD)
        par["drawn_timing_factor"] = f          # and the rise long enough for the same (wave SH1d fix B)
        par["push_rise_stretch"] = 1.0
        for _ in range(4):
            R, tr, _, _, nu, _ = _attempt(par, 100.0)
            peak = float(trunk_line_rate(R[:nu], tr[:nu], 100.0).max())
            if peak <= TRUNK_RATE_MAX:
                break
            par["push_rise_stretch"] *= 1.002 * peak / TRUNK_RATE_MAX
            par["timing_factor"] = f * par["push_rise_stretch"]
        par["rise_trunk_rate_max_deg_s"] = peak
    par["length_s"] = (UP_STRETCH * par["u_peak"] * fi["rise_s"] * par["timing_factor"] + par["fall_s"]
                       + trunk_after_s(par))
    return par


def trunk_line_rate(R, tr, rate=100.0):
    """(N,) deg/s: how fast the pelvis-to-neck line turns in the room (central differences, 0 at both ends) at the
    knots of R (N, 22, 3, 3) and tr (N, 3): the trunk as the skeleton page reads it."""
    J = fk(R, tr, rest_joints())[0].numpy()
    d = J[:, 12] - J[:, 0]
    u = d / np.linalg.norm(d, axis=-1, keepdims=True)
    w = np.zeros(len(u))
    if len(u) > 2:
        w[1:-1] = np.degrees(np.arccos(np.clip((u[2:] * u[:-2]).sum(-1), -1.0, 1.0))) * rate / 2.0
    return w


def trunk_turn_deg(par):
    """deg: the trunk's return of a failed attempt, from its peak to the seated lean (the pelvis pitch and the
    spine's flexions summed, the 'fail' template at u_peak with the attempt's trunk share)."""
    from .motions import TRUNK_JOINTS
    R, _, _ = _template_pose("fail", np.array([par["u_peak"]]))
    R = _with_trunk_share(R, np.array([par["u_peak"]]), np.array([par["trunk_share"]]))
    Rs, _, _ = _template_pose("normal", np.zeros(1))
    return float(abs(sum(_x_angle(R[:, j])[0] - _x_angle(Rs[:, j])[0] for j in (0,) + tuple(TRUNK_JOINTS))))


def trunk_after_s(par):
    """s: how long the trunk of a failed folded attempt goes on returning after the seat contact (AFTER_SEAT_S): long
    enough that its minimum-jerk return over the drop and this keeps under TRUNK_RATE_MAX; 0 for a push-off, whose
    hands stay on the chair arms (its drop is slowed instead, _attempt_params: the arms brake it)."""
    if par["arms"] == "push":
        return 0.0
    need = 1.875 * trunk_turn_deg(par) / TRUNK_RATE_MAX - TRUNK_LEAD * par["fall_s"]
    return float(np.clip(need, *AFTER_SEAT_S))


def _attempt(par, rate):
    """(R, tr, Rh, u, nu, record) of a failed attempt (module docstring): the rise to u_peak on the 'fail' template
    with its trunk share, then the drop: the hip midpoint goes back to the seat on a straight line (a minimum-jerk
    profile of x^FALL_SKEW), the pelvis pitch and the trunk go back to their seated lean over the drop and on past the
    seat contact (TRUNK_LEAD, trunk_after_s: the hips seated while the trunk ends its return),
    the legs are re-solved to the planted ankles (place_pelvis), the arms folded (they follow the trunk) or on the
    chair arms (solved later). u: the template shares of the rise and, for the drop, the share of the hip
    midpoint's way back (u_peak to 0)."""
    from .motions import TRUNK_JOINTS
    _, fi = arise_template("fail")
    rs = fi["rise_s"]
    arms, share, u_pk, f, fall = par["arms"], par["trunk_share"], par["u_peak"], par["timing_factor"], par["fall_s"]
    nu = int(round(UP_STRETCH * u_pk * rs * f * rate))
    nd = int(round(fall * rate))
    na = int(round(trunk_after_s(par) * rate))          # the trunk still returning, the hips on the seat
    u_up = u_pk * (_eased if arms == "push" else _mj)((np.arange(nu) + 1) / nu)
    x = (np.arange(nd + na) + 1) / nd
    sx = _mj(np.minimum(x, 1.0) ** FALL_SKEW)
    R, tr, Rh = _template_pose("fail", np.r_[u_up, u_pk * (1.0 - sx)])
    R[:nu] = _with_trunk_share(R[:nu], u_up, np.full(nu, share))
    rest = rest_joints()
    Rs, trs, _ = _template_pose("normal", np.zeros(1))
    pk, trpk = R[nu - 1:nu].clone(), tr[nu - 1:nu].clone()
    Jpk, Js = fk(pk, trpk, rest)[0], fk(Rs, trs, rest)[0]
    Mpk, Ms = 0.5 * (Jpk[0, 1] + Jpk[0, 2]), 0.5 * (Js[0, 1] + Js[0, 2])
    M = Mpk[None] + (Ms - Mpk)[None] * torch.from_numpy(sx)[:, None]
    w = _mj(np.minimum(1.0, (np.arange(nd + na) + 1) / (TRUNK_LEAD * nd + na)))
    ang = {j: _x_angle(pk[:, j])[0] + (_x_angle(Rs[:, j])[0] - _x_angle(pk[:, j])[0]) * w
           for j in (0,) + tuple(TRUNK_JOINTS)}
    nd = nd + na
    G0 = _rot("x", ang[0])
    Rd = pk.expand(nd, -1, -1, -1).clone()
    for j in TRUNK_JOINTS:
        Rd[:, j] = _rot("x", ang[j])
    mid = torch.as_tensor(0.5 * (rest[1] + rest[2]) - rest[0])
    J0 = M - torch.einsum("tij,j->ti", G0, mid)
    ank = torch.stack([Js[0, 7], Js[0, 8]], 0)[None].expand(nd, 2, 3).clone()
    R[nu:], tr[nu:] = place_pelvis(Rd, trpk.expand(nd, 3).clone(), J0, G0, ank, R[nu:], tr[nu:], keep_reach=False)
    rec = {"kind": "fails", "arms": arms, "template": "fail", "u_peak": u_pk, "trunk_share": share,
           "up_s": nu / rate, "fall_s": (nd - na) / rate, "trunk_after_seat_s": na / rate, "timing_factor": f}
    return R, tr, Rh, np.r_[u_up, u_pk * (1.0 - sx)], nu, rec


def _hand_clearance(R, tr, tg=None):
    """(N,) m: the lowest vertex of either flat hand (the push-off hands, at rest) above the top of its chair arm,
    over the vertices inside the arm's footprint (negative: inside the chair arm); tg: armrest_targets'."""
    tg = armrest_targets() if tg is None else tg
    _, _, dom = H._rest()
    b = H.HandBody(matrix_to_rot6d(R), tr, torch.zeros(R.shape[0], 2, dtype=torch.bool))
    out = np.full(R.shape[0], np.inf)
    for side, w in (("left", 20), ("right", 21)):
        ids = np.flatnonzero(np.isin(dom, (w,) + tuple(range(H.FIRST[side], H.FIRST[side] + 15))))
        P = H.body_points(b, ids)
        a = tg["armrest"][side]
        lo, hi = np.asarray(a["box_lo_m"]), np.asarray(a["box_hi_m"])
        inside = (P[..., 0] > lo[0]) & (P[..., 0] < hi[0]) & (P[..., 2] > lo[2]) & (P[..., 2] < hi[2])
        y = np.where(inside, P[..., 1] - a["top_y_m"], np.inf)
        out = np.minimum(out, y.min(1))
    return out


def clip_chair_arms(tg, R, tr):
    """armrest_targets' dict tg with the chair arms' boxes and tops of the clip's own chair (motions.arise_chair under
    the clip's knots R (N, 22, 3, 3) and tr (N, 3), the chair the scene draws and arise_fold_check reads), its wrist
    targets and hand rotations kept (Phase D of docs/PLAN_2026-11-12.md): the targets' chair is the normal template's,
    whose arms end at its seat's front; on park_E7_arise_g4 the clip's own seat front lies 5.2 cm further forward, its
    arms with it, and the fingers that pass the template's arm end dipped 1.23 mm into the drawn one (push_arms read
    the hand's clearance over the template's footprint only)."""
    from .motions import arise_chair
    b = H.HandBody(matrix_to_rot6d(R), tr, torch.zeros(R.shape[0], 2, dtype=torch.bool))
    V, _ = arise_chair(b, 0.0)
    boxes = V.reshape(-1, 8, 3)
    out = dict(tg, armrest={})
    for side, bx in (("left", boxes[6]), ("right", boxes[7])):
        out["armrest"][side] = dict(tg["armrest"][side], top_y_m=float(bx[:, 1].max()),
                                    front_z_m=float(bx[:, 2].max()), box_lo_m=bx.min(0).tolist(),
                                    box_hi_m=bx.max(0).tolist())
    return out


def push_arms(R, tr, passes=3, tg=None):
    """(R with the arm rows solved (N, 22, 3, 3), reach share (N,), inside the table (N,), wrist lift (N,) m): the
    hands palm down on the chair arms (arm_ik), the wrist targets raised per knot by what keeps the lowest hand
    vertex on the chair arm's top (the palm's heel bears on the armrest as the wrist extends; linear blend skinning
    lowers the wrist crease), solved again over passes; tg: armrest_targets' (default armrest_targets())."""
    tg = armrest_targets() if tg is None else tg
    lift = np.zeros(R.shape[0])
    for _ in range(passes):
        Rp = R.clone()
        reach = np.zeros(R.shape[0])
        for sd in ("left", "right"):
            W = tg["wrist"][sd][None] + torch.from_numpy(np.c_[0 * lift, lift, 0 * lift])
            sol = arm_ik(R, tr, sd, W, tg["hand_R"][sd])
            for j, M in zip(ARM[sd], sol[:4]):
                Rp[:, j] = M
            reach = np.maximum(reach, sol[4])
        clear = _hand_clearance(Rp, tr, tg)
        if clear.min() >= -1e-4:
            break
        lift = lift + np.maximum(0.0, -clear)
    return Rp, reach, _limits_ok_rows(Rp, _arm_joints()), lift


def push_thigh_gap(R, tr, Rpush, near_m=0.04):
    """(N,) m: the least skin gap of either forearm and hand to the thighs (motions.fold_gaps, 'thigh_left' and
    'thigh_right') at the knots of R (N, 22, 3, 3) and tr (N, 3) with the push-off arm rows of Rpush (N, 22, 3, 3)
    and open hands; read where a forearm or hand vertex lies within near_m of a thigh vertex (inf elsewhere)."""
    from scipy.spatial import cKDTree
    from .motions import _fold_sets, fold_gaps
    N = len(tr)
    Rp = R.clone()
    Rp[:, _arm_joints()] = Rpush[:, _arm_joints()]
    if N < 2:                                                     # a body has two knots
        return push_thigh_gap(R.expand(2, -1, -1, -1).clone(), tr.expand(2, -1).clone(),
                              Rpush.expand(2, -1, -1, -1).clone(), near_m)[:N]
    b = H.HandBody(matrix_to_rot6d(Rp), tr, torch.zeros(N, 2, dtype=torch.bool))
    S = _fold_sets()
    arms = np.concatenate([S[p][0] for p in ("fore_l", "hand_l", "fore_r", "hand_r")])
    P = H.body_points(b, np.r_[arms, S["thighs"][0]])
    na = len(arms)
    near = [i for i in range(len(P)) if cKDTree(P[i, na:]).query(P[i, :na])[0].min() < near_m]
    out = np.full(len(P), np.inf)
    if near:
        g, _ = fold_gaps(b, near, checks=("thigh_left", "thigh_right"))
        out[near] = np.minimum(g["thigh_left"], g["thigh_right"])
    return out


def _waypoint_rows(R1, tr1):
    """(1, 22, 3, 3): R1 (1, 22, 3, 3) with each arm solved (arm_ik) to its wrist target above the chair arm
    (armrest_targets plus ARM_WAYPOINT_M), the hand palm down: the waypoint of the move from the folded grip."""
    tg = armrest_targets()
    Rway = R1.clone()
    for s in ("left", "right"):
        sw = arm_ik(R1, tr1, s, tg["wrist"][s] + torch.tensor(ARM_WAYPOINT_M, dtype=F64), tg["hand_R"][s])
        for j, M in zip(ARM[s], sw[:4]):
            Rway[:, j] = M
    return Rway


def _open_rows(Rfold, Rh_fold, Rway, Rt, s, lag=0.0, lag_side=None):
    """(R (N, 22, 3, 3), Rh (N, 30, 3, 3)): Rfold and Rh_fold (N knots of the folded arms and the grip) with the
    arm and hand rows of the move from the folded grip to the chair arms at its time shares s (N,) (module
    docstring, section 1): each side on its own clock (motions.fold_open_shares, the lag of opening_lag), its arm
    rows along the geodesic to the waypoint Rway (1, 22, 3, 3) until ARM_WAYPOINT_AT of its clock, then to the
    push-off rows Rt (N, 22, 3, 3), each part minimum-jerk; its hand opens from the grip by its share of the way to
    the waypoint."""
    from .motions import fold_open_shares, side_rows
    R, Rh = Rfold.clone(), Rh_fold.clone()
    N = R.shape[0]
    sh = fold_open_shares(s, lag, lag_side)
    for sd in ("left", "right"):
        js, hr = side_rows(sd)
        w = sh[sd]
        Rw = Rway[:, js].expand(N, -1, -1, -1)
        first = _blend(Rfold[:, js], Rw, _mj(w / ARM_WAYPOINT_AT))
        second = _blend(Rw, Rt[:, js], _mj((w - ARM_WAYPOINT_AT) / (1.0 - ARM_WAYPOINT_AT)))
        R[:, js] = torch.where(torch.as_tensor(w < ARM_WAYPOINT_AT)[:, None, None, None], first, second)
        Rh[:, hr] = _blend(Rh_fold[:, hr], torch.eye(3, dtype=F64).expand(N, len(hr), 3, 3),
                           np.minimum(1.0, w / ARM_WAYPOINT_AT))
    return R, Rh


@functools.lru_cache(maxsize=2)
def opening_lag(rate=100.0):
    """(lag, lag_side, record): the lag of one arm in the move from the folded grip to the chair arms
    (motions.fold_open_lag: the smallest lag of motions.FOLD_OPEN_LAGS, the side that lags tried both ways, whose
    rehearsal keeps the forearms, the hands, the trunk and the thighs motions.FOLD_OPEN_CLEAR_M apart), solved once
    on its rehearsal: the seated knot of the 'normal' template (no background), its waypoint (_waypoint_rows) and
    push-off arms (push_arms), the move over ARM_MOVE_S (before task sh1c both arms moved on one clock and the right
    hand passed through the left forearm: the try at lag 0 of the record). Cached."""
    from .motions import fold_open_lag
    Rs0, trs0, Rhs0 = _template_pose("normal", np.zeros(1))
    n = int(round(ARM_MOVE_S * rate))
    Rt = push_arms(Rs0.expand(2, -1, -1, -1).clone(), trs0.expand(2, -1).clone())[0][:1]   # a body has two knots
    Rway = _waypoint_rows(Rs0, trs0)
    s = (np.arange(n) + 1) / n
    Rf, Rhf = Rs0.expand(n, -1, -1, -1).clone(), Rhs0.expand(n, -1, -1, -1).clone()
    tr, Rtn = trs0.expand(n, -1).clone(), Rt.expand(n, -1, -1, -1).clone()

    def realize(lag, side):
        R, Rh = _open_rows(Rf, Rhf, Rway, Rtn, s, lag, side)
        return H.HandBody(matrix_to_rot6d(R), tr, torch.zeros(n, 2, dtype=torch.bool), rate=rate,
                          hand6d=matrix_to_rot6d(Rh))
    return fold_open_lag(realize, n)


def release_ease(Rp, n):
    """(Re (n, m, 3, 3), R_end (m, 3, 3)): the hands easing off the chair arms (module docstring, section 1): the
    push-off rows Rp (n + 1, m, 3, 3) of the n + 1 knots up to the release knot read on a clock that slows from the
    push-off's pace to a stop (its rate 1 - x^3 (10 - 15 x + 6 x^2) over the n knots, so that it stops n / 2 knots
    on; the geodesic between two knots), so that the arms' local rows come to rest and the hands go from resting on
    the chair arms to riding on the rising trunk without a step in their speed; Re the rows of the n knots before the
    release knot, R_end those where the clock stops (the release starts from them)."""
    if n <= 0:
        return Rp[:0], Rp[0]
    s = np.arange(n + 1) / n
    tau = n * (s - 2.5 * s ** 4 + 3.0 * s ** 5 - s ** 6)       # the integral of the clock's rate
    i0 = np.minimum(np.floor(tau).astype(np.int64), n - 1)
    Rw = _blend(Rp[i0], Rp[i0 + 1], tau - i0)
    return Rw[:n], Rw[n]


def arise_plan_body(plan, duration_s=ARISE_DUR_S, rate=100.0, seed=SEED, series=SERIES["arise"]):
    """(HandBody, info) of an arising timeline (module docstring): plan, a list of ('sit', s or None), ('arms', s),
    ('attempt', 'fail', arms) and ('rise', template, arms) with template 'normal', 'slow' (slow_rise_u) or 'abrupt'
    (abrupt_rise_u) and arms 'folded' or 'push'; the clip is filled to duration_s seated, or standing after a rise.
    The attempts and the pauses are drawn from seed; the background layer is the series' (background_signals of
    series: the same on every arising clip). info: the phases with their times, the attempts, the release knots of
    the push-offs, the templates' solved values."""
    from .motions import LIFTOFF_SHARE, balance
    rng = as_rng(seed)
    T = int(round(duration_s * rate)) + 1
    R = torch.zeros(T, 22, 3, 3, dtype=F64)
    tr = torch.zeros(T, 3, dtype=F64)
    Rh = torch.zeros(T, H.N_HAND, 3, 3, dtype=F64)
    arm = np.zeros(T, dtype=np.int64)          # 0 folded, 1 folded->push blend, 2 push, 3 hang (release)
    rising = np.zeros(T, dtype=bool)           # the successful rise and the standing after it
    seated = np.zeros(T, dtype=bool)           # the seated knots of the template
    stand_w = np.zeros(T)                      # the standing knots after a rise (the sway's weight)
    armw = np.zeros(T)
    k, mode, risen, phases, attempts = 0, 0, False, [], []
    Rs0, trs0, Rhs0 = _template_pose("normal", np.zeros(1))
    _, ni = arise_template("normal")

    def seat(n_k, m, w=0.0):
        nonlocal k
        sl = slice(k, min(T, k + n_k))
        R[sl], tr[sl], Rh[sl] = Rs0, trs0, Rhs0
        arm[sl], armw[sl], seated[sl] = m, w, True
        k = sl.stop
    draws = [float(rng.uniform()) if ph[0] == "sit" and ph[1] is None else
             _attempt_params(rng, ph[2]) if ph[0] == "attempt" else None for ph in plan]
    fixed = sum(PAUSE_S[0] if ph[0] == "sit" and ph[1] is None else ph[1] if ph[0] in ("sit", "arms") else
                d["length_s"] if ph[0] == "attempt" else
                (SLOW_RISE_S if ph[1] == "slow" else ABRUPT_RISE_S if ph[1] == "abrupt" else ni["rise_s"]) + 0.5
                for ph, d in zip(plan, draws))
    extra = sum(d * (PAUSE_S[1] - PAUSE_S[0]) for ph, d in zip(plan, draws) if ph[0] == "sit" and ph[1] is None)
    room = duration_s - 0.6 - fixed
    squeeze = 1.0 if extra <= room else max(0.0, room) / extra       # the drawn pauses shrink to fit the clip
    for ph, dr in zip(plan, draws):
        if k >= T:
            raise ValueError("the plan does not fit in the clip")
        if ph[0] == "sit":
            s = PAUSE_S[0] + dr * (PAUSE_S[1] - PAUSE_S[0]) * squeeze if ph[1] is None else float(ph[1])
            n = int(round(s * rate))
            phases.append({"phase": "sit", "start_s": k / rate, "duration_s": n / rate})
            seat(n, mode, 1.0 if mode == 1 else 0.0)
        elif ph[0] == "arms":
            n = int(round(ph[1] * rate))
            phases.append({"phase": "arms to the chair arms", "start_s": k / rate, "duration_s": n / rate})
            k0 = k
            seat(n, 1)
            armw[k0:k] = (np.arange(k - k0) + 1) / n
            mode = 2
        elif ph[0] == "attempt":
            am = ph[2]
            if am == "push" and mode != 2:
                raise ValueError("a push-off needs the hands on the chair arms first ('arms')")
            Ra, tra, Rha, ua, nu, rec = _attempt(dr, rate)
            if "thigh_gap_min_m" in dr:                  # a failed push-off: its peak keeps the thighs off the hands
                rec["thigh_gap_min_m"] = dr["thigh_gap_min_m"]
            for key in ("drawn_timing_factor", "push_rise_stretch", "rise_trunk_rate_max_deg_s"):
                if key in dr:                            # a failed push-off: its rise stretched for the trunk's speed
                    rec[key] = dr[key]
            sl = slice(k, k + len(ua))
            if sl.stop > T:
                raise ValueError("the plan does not fit in the clip")
            R[sl], tr[sl], Rh[sl], arm[sl] = Ra, tra, Rha, 2 if am == "push" else 0
            seated[sl] = ua <= 1e-12
            k_lift = k + int(np.argmax(ua > LIFTOFF_SHARE))
            rec.update({"start_s": k / rate, "peak_s": (k + nu) / rate, "end_s": sl.stop / rate,
                        "liftoff_knot": k_lift, "peak_knot": k + nu - 1})
            attempts.append(rec)
            phases.append({"phase": f"failed attempt ({am})", "start_s": k / rate, "duration_s": len(ua) / rate})
            k = sl.stop
        else:
            _, tn, am = ph
            if am == "push" and mode != 2:
                raise ValueError("a push-off needs the hands on the chair arms first ('arms')")
            if tn == "slow":
                ua, sinfo = slow_rise_u(rate)
            elif tn == "abrupt":
                ua, sinfo = abrupt_rise_u(rate)
            else:
                ua = (np.arange(int(round(ni["rise_s"] * rate))) + 1) / rate / ni["rise_s"]
                sinfo = None
            n = len(ua)
            sl = slice(k, min(T, k + n))
            if sl.stop - sl.start < n:
                raise ValueError("the plan does not fit in the clip")
            R[sl], tr[sl], Rh[sl] = _template_pose("normal", ua)
            if tn == "abrupt":                         # the abrupt rise leans less (abrupt_rise_u)
                R[sl] = _normal_trunk_share(R[sl], ua, sinfo["trunk_share"])
            arm[sl], rising[sl] = (2 if am == "push" else 0), True
            rec = {"kind": "rises", "arms": am, "template": "normal", "pace": tn, "start_s": k / rate,
                   "end_s": sl.stop / rate, "rise_s": n / rate,
                   "liftoff_knot": k + int(np.argmax(ua > LIFTOFF_SHARE))}
            if sinfo is not None:
                rec["slow" if tn == "slow" else tn] = sinfo
            attempts.append(rec)
            phases.append({"phase": f"rise ({am}{', ' + tn if tn in ('slow', 'abrupt') else ''})",
                           "start_s": k / rate, "duration_s": n / rate})
            k = sl.stop
            risen = True
            sl = slice(k, T)
            R[sl], tr[sl], Rh[sl] = _template_pose("normal", np.ones(1))
            arm[sl], rising[sl] = (2 if am == "push" else 0), True
            stand_w[sl] = _mj((np.arange(T - k) + 1) / (0.5 * rate))
            phases.append({"phase": "standing", "start_s": k / rate, "duration_s": (T - k) / rate})
            k = T
            break
    if k < T:
        phases.append({"phase": "sit", "start_s": k / rate, "duration_s": (T - k) / rate})
        seat(T - k, mode, 1.0 if mode == 1 else 0.0)
    # the background layer (module docstring): breathing and the head's drift throughout, the trunk's while seated
    bg = background_signals(series, T, 1.0, stand_w, rate)
    w_sit = _inside_ramp(seated, int(0.3 * rate))
    _trunk_layer(R, bg["breath"], bg["yaw"], bg["pitch"], w_sit * bg["lean"], w_sit * bg["lean_ml"])
    # the arms: push-off IK, the transition from the folded grip, the release to hanging
    tg = armrest_targets()
    stand = posture_rotations("stand")
    release = []
    wrist_lift = None
    opening = None
    push = np.flatnonzero(arm >= 1)
    if len(push):
        # the hands leave at the first knot of the rise that over-reaches (reach_share) or whose solution leaves
        # the table; a failed push-off or a seated knot that does either is refused
        reach_all = np.maximum(reach_share(R[push], tr[push], "left", tg["wrist"]["left"]),
                               reach_share(R[push], tr[push], "right", tg["wrist"]["right"]))
        over = np.flatnonzero(reach_all > RELEASE_EXT)
        over = over[rising[push[over]]] if len(over) else over
        bad = np.flatnonzero((reach_all > PUSH_PEAK_EXT + 1e-9) & ~rising[push])
        if len(bad):
            raise RuntimeError(f"a seated or failed push-off knot over-reaches ({int(push[bad[0]])})")
        if len(over):
            push = push[:over[0] + 1]
        Rpush, reach, ok, lift = push_arms(R[push], tr[push], tg=clip_chair_arms(tg, R, tr))
        wrist_lift = float(lift[:len(push)].max())
        pos = {int(kk): ii for ii, kk in enumerate(push)}
        bad_seated = [int(kk) for kk in push if seated[kk] and not ok[pos[int(kk)]]]
        if bad_seated:
            raise RuntimeError(f"the hands cannot rest on the chair arms seated ({len(bad_seated)} knots)")
        # the thighs rise toward the hands at the end of the rise: the hands leave before they come within
        # PUSH_THIGH_GUARD_M (push_thigh_gap on the push-off arms of the rising knots; the turned hands keep
        # PUSH_THIGH_CLEAR_M on the rehearsal, push_hand_yaw)
        up = [ii for ii, kk in enumerate(push) if rising[kk] and arm[kk] == 2]
        thigh = np.full(len(push), np.inf)
        if up:
            thigh[up] = push_thigh_gap(R[push[up]], tr[push[up]], Rpush[up])
        rel_k = None
        for ii, kk in enumerate(push):
            if arm[kk] == 1:                                             # folded -> push, seated
                continue
            if not rising[kk] and not ok[ii]:
                raise RuntimeError(f"a failed push-off leaves the table at knot {int(kk)}")
            far = max(reach[ii], reach_all[ii]) > RELEASE_EXT        # the solved arm or the collar-free reach
            if rising[kk] and rel_k is None and (far or not ok[ii] or thigh[ii] < PUSH_THIGH_GUARD_M):
                rel_k = int(kk)
                release.append({"knot": rel_k, "t_s": rel_k / rate, "reach_share": float(reach[ii]),
                                "reach_share_collar_rest": float(reach_all[ii]), "inside_table": bool(ok[ii]),
                                "thigh_gap_mm": float(1000.0 * thigh[ii]),
                                "by": "reach" if far else "table" if not ok[ii] else "thigh"})
                break
        n_rel = int(round(RELEASE_S * rate))
        n_lift = int(round(RELEASE_LIFT_S * rate))
        Rfold = R.clone()
        blend_k = [int(kk) for kk in push if arm[kk] == 1]
        if blend_k:            # the move from the folded grip, one arm waiting for the other (opening_lag)
            lag, lag_side, open_rec = opening_lag(rate)
            bi = [pos[kk] for kk in blend_k]
            way = _waypoint_rows(R[blend_k[:1]], tr[blend_k[:1]])
            R[blend_k], Rh[blend_k] = _open_rows(Rfold[blend_k], Rh[blend_k], way, Rpush[bi], armw[blend_k], lag,
                                                 lag_side)
            opening = dict(open_rec, lag=lag, lag_side=lag_side)
        for ii, kk in enumerate(push):
            kk = int(kk)
            if rel_k is not None and kk >= rel_k:
                break
            if arm[kk] != 1:
                R[kk, _arm_joints()] = Rpush[ii, _arm_joints()]
                Rh[kk] = torch.eye(3, dtype=F64)
        if rel_k is not None:
            # the release (module docstring): the hands ease off the chair arms over RELEASE_EASE_S up to the
            # release knot (the push-off rows on a clock slowing to a stop, release_ease), then the elbows flex, so
            # that the hands lift off the chair arms, the arms blend to hanging and the hands relax
            first = min(int(kk) for kk in push if rising[kk] and arm[kk] == 2)
            r0 = min(rel_k, max(first + 1, rel_k - int(round(RELEASE_EASE_S * rate))))
            n_e = rel_k - r0
            Re, R0 = release_ease(Rpush[pos[r0]:pos[rel_k] + 1, _arm_joints()], n_e)
            n_s = T - rel_k
            x1 = _mj((np.arange(n_s) + 1) / n_lift)
            Rl = R0[None].expand(n_s, -1, -1, -1).clone()
            for side, col in (("left", 2), ("right", 6)):
                Rl[:, col] = compose(Rl[:, col], _rot("y", ELBOW_FLEXION_SIGN[side] * RELEASE_ELBOW_DEG * x1),
                                     "parent")
            w2 = _mj((np.arange(n_s) + 1 - 0.5 * n_lift) / n_rel)
            Rt = stand[_arm_joints()][None].expand(n_s, -1, -1, -1).clone()
            R[r0:rel_k, _arm_joints()] = Re
            R[rel_k:, _arm_joints()] = _blend(Rl, Rt, w2)
            Rh[rel_k:] = _blend(torch.eye(3, dtype=F64).expand(n_s, H.N_HAND, 3, 3).clone(),
                                relaxed_hand()[None].expand(n_s, -1, -1, -1).clone(), w2)
            release[-1].update(ease_knot=r0, ease_s=n_e / rate)
            arm[r0:] = 3
    # the collars breathe under the hanging arms only (the folded arms ride on the breathing chest; TRANSFER_PARTS)
    _collar_breath(R, _inside_ramp(arm == 3, int(0.3 * rate)) * BREATH_COLLAR_DEG * bg["breath"])
    if stand_w.any():                                  # the standing sway after a rise, the feet planted
        st = np.flatnonzero(stand_w > 0)
        R[st], tr[st] = stand_sway(R[st], tr[st], bg["sway_z"][st], bg["sway_x"][st])
    b = H.HandBody(matrix_to_rot6d(R), tr, torch.zeros(T, 2, dtype=torch.bool), rate=rate,
                   hand6d=matrix_to_rot6d(Rh))
    b = H.HandBody(b.rot6d, b.transl, contact_labels(b), b.betas, b.gender, b.rate, b.hand6d)
    for a in attempts:                       # the folded attempts' extrapolated centre of mass (motions.balance)
        if a["arms"] != "folded":
            continue
        post = balance(b, a["liftoff_knot"])["posterior"]
        if a["kind"] == "fails":
            seg = post[a["liftoff_knot"]:a["peak_knot"] + 1]
            a["xcom_margin_min_m"], a["xcom_margin_max_m"] = float(seg.min()), float(seg.max())
        else:
            a["xcom_margin_min_m"] = float(post[a["liftoff_knot"]:int(round(a["end_s"] * rate))].min())
    J = b.joints().numpy()
    for a in attempts:
        if a["kind"] == "fails":
            a["pelvis_peak_m"] = float(J[a["peak_knot"], 0, 1] - J[0, 0, 1])
    info = {"phases": phases, "attempts": attempts, "release": release, "risen": risen,
            "push_wrist_lift_max_m": wrist_lift, "opening": opening, "push_hand_yaw": tg["hand_yaw"],
            "push_thigh_guard_m": PUSH_THIGH_GUARD_M,
            "templates": {n: {"rise_s": arise_template(n)[1]["rise_s"],
                              "trunk_deg": arise_template(n)[1]["trunk"]["amplitude_deg"],
                              "trunk_solved": arise_template(n)[1]["trunk"]["solved"],
                              "balance_ok": arise_template(n)[1]["balance"]["ok"],
                              "min_margin_from_liftoff_m":
                                  arise_template(n)[1]["balance"]["min_margin_from_liftoff_m"]}
                          for n in ("normal", "fail")},
            "armrest": tg["armrest"], "wrist_targets_m": {s: tg["wrist"][s].tolist() for s in ("left", "right")},
            "arm_code": arm, "fail_u": list(FAIL_U), "push_fail_u": list(PUSH_FAIL_U),
            "push_peak_ext": PUSH_PEAK_EXT, "fail_trunk_share": list(FAIL_TRUNK_RANGE),
            "push_trunk_share": list(PUSH_TRUNK_RANGE), "fall_s": list(FALL_S), "pause_s": list(PAUSE_S),
            "background": background_record(series, T, 1.0, rate, dict(
                TRANSFER_PARTS, **({} if stand_w.any() else {"sway": "none (the patient stays seated)"}),
                **({} if (arm == 3).any() else {"collars": "none (the arms never hang)"})))}
    return b, info


def fail_margin_m():
    """The extrapolated centre of mass's smallest margin ahead of the heels (m) of the 'fail' template between its
    lift-off and the largest share of FAIL_U (negative: behind the heels, the body falls back; motions.balance)."""
    from .motions import LIFTOFF_SHARE, balance
    b, info = arise_template("fail")
    k_lift = int(round(info["realized_liftoff_s"] * b.rate))
    k_peak = int(round((SEATED_S + FAIL_U[1] * info["rise_s"]) * b.rate))
    post = balance(b, k_lift)["posterior"]
    return float(post[k_lift:k_peak + 1].min()), LIFTOFF_SHARE


def arise_fold_check(b, arm, phases):
    """{'folded', 'opening', 'push' ({check: least gap mm, with its time}), 'chair_entry_mm', 'n_knots', 'ok'}: the
    skin gaps of the arms over a 3.9 clip (motions.fold_gaps, negative inside), by the clip's arm code: the folded
    knots of the moving phases (failed attempts and rises, every motions.FOLD_CHECK_STEP-th knot) and the first and
    last knot of each hold (motions.FOLD_CHECKS; the fold's own margins are the solve's, motions.solve_fold), the move
    to the chair arms (every motions.FOLD_OPEN_STEP-th knot, motions.FOLD_OPEN_CHECKS), the push-offs and the first
    RELEASE_S of a release (every knot, the forearms and hands against the thighs; a read every
    motions.FOLD_CHECK_STEP-th knot reported the least gaps too high, review of task sh1c), and the deepest entry of
    a forearm or hand into the chair of the scene (motions.arise_chair) at all of them, a report; 'ok' when no skin part
    enters another (task sh1c: the move to the chair arms and the push-off entered the other forearm and the thigh)."""
    from .motions import (FOLD_CHECK_STEP, FOLD_CHECKS, FOLD_OPEN_CHECKS, FOLD_OPEN_STEP, arise_chair, fold_gaps)
    arm = np.asarray(arm)
    T = len(arm)
    sel = {"folded": set(), "opening": set(), "push": set()}
    for ph in phases:
        a = int(round(ph["start_s"] * b.rate))
        e = min(T, a + int(round(ph["duration_s"] * b.rate)))
        moving = ph["phase"].startswith(("failed attempt", "rise"))
        ks = range(a, e, FOLD_CHECK_STEP) if moving else (a, e - 1)
        sel["folded"].update(k for k in ks if arm[k] == 0)
    sel["opening"] = set(np.flatnonzero(arm == 1)[::FOLD_OPEN_STEP].tolist())
    rel = np.flatnonzero(arm == 3)
    rel = rel[:int(round(RELEASE_S * b.rate))]
    sel["push"] = set(np.flatnonzero(arm == 2).tolist()) | set(rel.tolist())
    checks = {"folded": FOLD_CHECKS, "opening": FOLD_OPEN_CHECKS, "push": ("thigh_left", "thigh_right")}
    chair = arise_chair(b, 0.0)
    out, ok, entry, n = {}, True, 0.0, 0
    for name, ks in sel.items():
        ks = np.array(sorted(ks), dtype=np.int64)
        n += len(ks)
        if not len(ks):
            out[name] = None
            continue
        g, en = fold_gaps(b, ks, chair, checks[name])
        out[name] = {c: {"min_mm": float(1000.0 * v.min()), "at_s": float(ks[int(np.argmin(v))] / b.rate)}
                     for c, v in g.items()}
        ok = ok and all(v.min() >= 0.0 for v in g.values())
        entry = max(entry, max(float(1000.0 * v.max()) for k, v in en.items() if k.startswith("armrest")))
    return dict(out, chair_entry_mm=entry, n_knots=int(n), ok=bool(ok))


def arise_grade(variant, duration_s=ARISE_DUR_S, seed=SEED):
    """(HandBody, truth) of a 3.9 ladder variant of ARISE_PLANS (module docstring); the plan's draws from seed plus
    the variant's index, the background layer the series' (one for the ladder, its level 0 included)."""
    from .motions import MOS_MIN_M, arise_validity
    plan = ARISE_PLANS[variant]
    b, info = arise_plan_body(plan, duration_s, seed=seed + list(ARISE_PLANS).index(variant))
    g = ARISE_GRADES[variant]
    lim = limit_violations(b.rot6d)
    J = b.joints()
    sk = skate(J, b.contact, b.rate)
    val = arise_validity(b)
    val["hand_limits"] = H.hand_limit_report(b.hand_R())
    drops = [k for a in info["attempts"] if a["kind"] == "fails"          # the failed attempts' drops (review M10)
             for k in range(int(round(a["peak_s"] * b.rate)), min(b.T, int(round(a["end_s"] * b.rate)) + 1))]
    val["continuity"] = continuity_report(b, head_bound_m_s2=HEAD_ACCEL_MAX if drops else None,
                                          head_knots=drops if drops else None)
    val["fold"] = arise_fold_check(b, info["arm_code"], info["phases"])
    folded_fails = [a for a in info["attempts"] if a["kind"] == "fails" and a["arms"] == "folded"]
    rises = [a for a in info["attempts"] if a["kind"] == "rises" and a["arms"] == "folded"]
    val["balance"] = {"folded_fails_behind_heels": all(a["xcom_margin_min_m"] < 0.0 for a in folded_fails),
                      "folded_rises_ok": all(a["xcom_margin_min_m"] >= MOS_MIN_M - 1e-3 for a in rises),
                      "mos_min_m": MOS_MIN_M}
    val["balance"]["ok"] = val["balance"]["folded_fails_behind_heels"] and val["balance"]["folded_rises_ok"]
    fm, _ = fail_margin_m()
    arm = info.pop("arm_code")
    p = {"variant": variant, "plan": [list(x) for x in plan], "duration_s": float(duration_s),
         "seat_height_m": 0.43, "control_rise_s": 1.89, "slow_rise_s": SLOW_RISE_S,
         "fail_margin_m": fm, "skate_p95_m_s": sk["p95"], "limits_outside": lim["outside"],
         "remeasured": measure_arise(b, info), **info}
    srcs = ["mds_updrs", "inkster2004", "li2021", "hof2005", "deleva1996", "module"]
    return b, truth_record("arise_parkinsonian", seed, g, "MDS-UPDRS 3.9", p, srcs, val,
                           arrays={"arm_code": arm})


def continuity_report(b, bound_deg_s=700.0, transl_bound_m_s=2.5, accel_bound_m_s2=30.0, head_bound_m_s2=None,
                      head_knots=None):
    """{ok, max joint step deg per knot, max translation speed and acceleration}: no joint's local rotation turns
    faster than bound_deg_s, the pelvis moves slower than transl_bound_m_s (a drop back onto the seat reaches about
    the 1.5 m/s of a free fall from 12 cm) and accelerates less than accel_bound_m_s2 (a join of the timeline that
    jumped by 3 mm would show 30 m/s2 at 100 Hz)."""
    from .body import rotation_angle_deg
    R = b.local_R()
    st = rotation_angle_deg(R[:-1].transpose(-1, -2) @ R[1:]).numpy()            # (T - 1, 22)
    v = float((b.transl[1:] - b.transl[:-1]).norm(dim=-1).max()) * b.rate
    acc = float((b.transl[2:] - 2 * b.transl[1:-1] + b.transl[:-2]).norm(dim=-1).max()) * b.rate ** 2
    mx = float(st.max())
    Jh = b.joints()[:, 15].numpy()
    ha = np.linalg.norm(Jh[2:] - 2 * Jh[1:-1] + Jh[:-2], axis=-1) * b.rate ** 2       # knots 1 .. T - 2
    if head_knots is not None:
        m = np.zeros(b.T, bool)
        m[np.asarray(head_knots, dtype=np.int64)] = True
        ha = np.where(m[1:-1], ha, 0.0)
    head = float(ha.max()) if len(ha) else 0.0
    ok = mx * b.rate <= bound_deg_s and v <= transl_bound_m_s and acc <= accel_bound_m_s2
    if head_bound_m_s2 is not None:
        ok = ok and head <= head_bound_m_s2
    return {"ok": bool(ok),
            "max_step_deg": mx, "max_step_knot": int(np.unravel_index(st.argmax(), st.shape)[0]),
            "bound_deg_s": bound_deg_s, "pelvis_speed_max_m_s": v, "pelvis_speed_bound_m_s": transl_bound_m_s,
            "pelvis_accel_max_m_s2": acc, "pelvis_accel_bound_m_s2": accel_bound_m_s2,
            "head_accel_max_m_s2": head, "head_accel_bound_m_s2": head_bound_m_s2,
            "head_knots": "the failed attempts' drops" if head_knots is not None else "every knot"}


def measure_arise(body, info=None, lift_m=0.02, stand_share=0.9, onset_deg=1.0):
    """The 3.9 quantities read back from a body: the pelvis height above its first knot; a seat-off is a run of
    knots above lift_m; it rises when it reaches stand_share of the standing height (the 'normal' template's) and
    stays there to the end, else it falls back; the rise time from the movement onset (the trunk lean of the
    pelvis-to-neck line more than onset_deg above its median over the first 0.5 s, the last such crossing before
    the seat-off) to the pelvis at 98 percent of its final height; the arms used when both wrists stay within 1.5
    cm of their chair-arm targets horizontally and within 3 cm above them (push_arms raises the wrists as they extend)
    during a seat-off; the trunk lean peak per seat-off."""
    J = body.joints().numpy()
    h = J[:, 0, 1] - J[0, 0, 1]
    tb, _ = arise_template("normal")
    Jt = tb.joints().numpy()
    h_stand = float(Jt[-1, 0, 1] - Jt[0, 0, 1])
    tv = J[:, 12] - J[:, 0]
    lean = np.degrees(np.arctan2(tv[:, 2], tv[:, 1]))
    lean0 = float(np.median(lean[:int(round(0.5 * body.rate)) + 1]))
    tg = armrest_targets()
    wd = np.zeros(len(J))
    for w, sd in ((20, "left"), (21, "right")):
        d = J[:, w] - tg["wrist"][sd].numpy()
        wd = np.maximum(wd, np.where((d[:, 1] > -0.005) & (d[:, 1] < 0.03), np.hypot(d[:, 0], d[:, 2]), np.inf))
    on = h > lift_m
    m = np.r_[False, on, False].astype(np.int8)
    d = np.diff(m)
    runs = list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))
    r = body.rate
    out = []
    for a, e in runs:
        risen = bool(e == len(h) and h[-1] >= stand_share * h_stand)
        on_k = a
        while on_k > 0 and lean[on_k] > lean0 + onset_deg:
            on_k -= 1
        rec = {"seat_off_s": a / r, "end_s": e / r, "peak_lift_m": float(h[a:e].max()), "rises": risen,
               "onset_s": on_k / r, "trunk_lean_peak_deg": float(lean[a:e].max() - lean0),
               "arms_used": bool((wd[a:min(e, a + 20)] < 0.015).all())}
        if risen:
            k98 = a + int(np.argmax(h[a:] >= 0.98 * h[-1]))
            rec["rise_time_s"] = (k98 - on_k) / r
        out.append(rec)
    return {"seat_offs": len(out), "rises": int(sum(x["rises"] for x in out)),
            "falls_back": int(sum(not x["rises"] for x in out)),
            "arms_used": bool(any(x["arms_used"] for x in out)), "attempts": out, "standing_height_m": h_stand,
            "final_height_m": float(h[-1]),
            "rise_time_s": next((x["rise_time_s"] for x in out if x["rises"]), None),
            "trunk_lean_peak_deg": max((x["trunk_lean_peak_deg"] for x in out), default=None)}


# ---------------------------------------------------------------- 3.14 global spontaneity
SPONT_RATE_PER_MIN = 10.0      # [module] level-0 spontaneous movements per minute (SOURCES['spont_unsourced'])
SPONT_COUNT_RATIO = 0.6        # [module] per grade
SPONT_AMP_RATIO = 0.75         # [module] per grade
SPONT_AMP_RATIO_BY_BASE = {"sit": SPONT_AMP_RATIO, "stand": 0.8}   # [module] wave SH1d part B (review B2): standing,
#                                0.8, so that a level-0 head turn still moves the nose by SPONT_VISIBLE_M at grade 4
SPONT_VISIBLE_M = 0.025        # [module] standing (review B2: grades 3 and 4 differed by 11 mm, their movements inside
#                                the sway): every kept movement moves some body joint or the nose by at least this
#                                against the background body at its grade (its level-0 reading times the amplitude
#                                ratio), several times the background's own travel of a point over a movement's span
SPONT_CHAIN_BASES = ("stand",)  # [module] the bases whose nested subsets are chosen by visibility (_visible_chain); the
#                                seated ladder, judged right by the review, keeps the mean-peak rule
SPONT_SLOW_RATIO = 1.25        # [module] movement time per grade
SPONT_HOLD_S = (0.6, 2.0)      # [module] hold range of a short action
SPONT_LONG_HOLD_S = (2.5, 6.0)  # [module] hold range of a kept position (hands together, a foot moved, weight shift)
SPONT_GAP_S = 0.8              # [module] smallest gap between two movements
SPONT_HOLD_DRIFT = (0.1, 0.2)  # [module] a hold drifts by this share of the excursion
SPONT_AMP_JITTER = (0.8, 1.2)  # [module] per-movement factor on the kind's level-0 amplitude
SPONT_ROOM_SHARE = 0.8         # a head turn takes at most this share of its room in the table
SPONT_LADDER_STEP = 0.97       # [module] a grade's predicted mean peak at most this share of the grade below's
SPONT_DUR = {"sit": 45.0, "stand": 45.0}     # E1 seated rest 45 s (draft section 2); standing 45 s (module)
SPONT_SEED_OFFSET = {"sit": 100, "stand": 101}   # one draw of the level-0 movements per base, thinned per grade
SPONT_SERIES = {"sit": SERIES["spont_sit"], "stand": SERIES["spont_stand"]}   # the background of each base's ladder
SPONT_PARTS = {"sit": {"breath": "throughout", "head": "throughout, under the movements",
                       "trunk": "throughout (spine1), under the movements", "sway": "none (seated)",
                       "collars": "throughout (the hands kept on the thighs by the arm solve)",
                       "fingers": "both hands, lift only (never into the thigh)"},
               "stand": {"breath": "throughout", "head": "throughout, under the movements", "trunk": "none (standing)",
                         "sway": "throughout, the feet planted",
                         "collars": "throughout, the hanging arms riding (a shrug raises them with the arms hanging)",
                         "fingers": "both hands"}}
SHRUG_TABLE = LX.SHRUG         # the joint-limit table of a standing 3.14 clip with a shrug (limits_ext, task sh1c)
THIGH_REST = {"along": 0.62, "yaw_deg": 15.0, "clear_m": 0.003, "hand_weight": 0.5, "upright_deg": 14.0,
              "protract_deg": 8.0}   # [module] the seated hands rest on the thighs: the hand's centre over the
#   thigh's axis at 'along' of the way from hip to knee, the fingers turned 'yaw_deg' toward the midline, the lowest
#   hand vertex 'clear_m' above the thigh's skin, the fingers at 'hand_weight' of RELAXED_HAND; the trunk of 'sit'
#   (reclined: the shoulders 10 cm behind the hips) flexed by 'upright_deg' (spine1 to spine3, the neck taking it
#   back) and the collars protracted by 'protract_deg', so that the arms reach the thighs with the elbows bent
WATCH = {"down_m": 0.32, "front_m": 0.28, "x_m": 0.10, "yaw_deg": -60.0, "pitch_deg": 18.0}  # [module] looking at
#   a wrist watch: the left wrist this far below the neck and in front of spine3, this far left of the midline, the
#   hand palm down turned 'yaw_deg' from forward (the fingers toward the right, the back of the wrist up), the head
#   down by 'pitch_deg' (and 5 deg toward the watch)
LAP_POLE = (0.8, 0.5)          # [module] the elbows' pole with the hands in the lap (clear of the chair arms)
LAP = {"along": 0.50, "x_m": 0.0, "yaw_deg": 45.0}  # [module] hands together in the lap: the hands' centres at the
#   midline, 'along' of the way from hip to knee, the left fingers turned 'yaw_deg' to the right, the right ones to
#   the left (the right hand across the back of the left)
ARC_M = 0.05                   # [module] a hand travelling to a target rises on an arc this high (clear of the thigh)
SEAT_POLE = (0.6, 0.8)         # [module] the 3.14 elbows point out and back (arm_ik's pole): seated, the upper arm
#                                keeps the table's least abduction (about 9 deg from the trunk)
# kind: (bases, level-0 amplitude, unit, movement time s, hold, amplitude graded); module values
SPONT_KINDS = {
    "head_turn_left": (("sit", "stand"), 25.0, "deg", 0.7, "short", True),
    "head_turn_right": (("sit", "stand"), 25.0, "deg", 0.7, "short", True),
    "glance_down": (("sit", "stand"), 12.0, "deg", 0.6, "short", True),
    "watch": (("sit", "stand"), 1.0, "share", 1.0, "short", True),
    "hands_together": (("sit",), 1.0, "share", 1.0, "long", False),
    "rub_thigh_left": (("sit",), 0.07, "m", 0.8, "short", True),
    "rub_thigh_right": (("sit",), 0.07, "m", 0.8, "short", True),
    "shrug": (("sit", "stand"), 7.0, "deg", 0.6, "short", True),
    "seat_adjust": (("sit",), 0.03, "m", 1.0, "short", True),
    "foot_slide_left": (("sit",), 0.07, "m", 0.9, "long", True),
    "foot_slide_right": (("sit",), 0.07, "m", 0.9, "long", True),
    "weight_shift_left": (("stand",), 0.025, "m", 1.0, "long", True),
    "weight_shift_right": (("stand",), 0.025, "m", 1.0, "long", True),
}
SPONT_MAX_PER_CLIP = {"watch": 1, "hands_together": 1, "seat_adjust": 1}   # [module] other kinds at most twice
HEAD_SHARES = (0.4, 0.6)       # neck, head of a head turn
TRUNK_TURN_SHARE = 0.2         # [module] the trunk turns with the head by this share (spine1 to spine3)
SPINE_SHARES = (0.4, 0.3, 0.3)  # spine1, spine2, spine3 of a trunk movement
SHIFT_ROLL_DEG = 2.0           # [module] a 2.5 cm weight shift drops the pelvis on the unloaded side by this
SEAT_LEAN_DEG = 8.0            # [module] a 3 cm seat adjustment leans the trunk forward by this midway
SEAT_LIFT_M = 0.006            # [module] and unweights the buttocks by this
FOOT_LIFT_SHARE = 0.2          # [module] a foot slide lifts the foot by this share of its way, midway
FOOT_LIFT_MIN_M = 0.010        # [module] and by this at least (a short slide at grade 4: 0.2 of 2 cm was 4 mm)
FOOT_CLEAR_M = 0.007           # [module] the foot travels only while lifted at least this (its way a minimum-jerk
#   course inside the lift's span above it): with the way and the lift on one profile the seated foot slid 4 to 5 mm
#   along the floor at each end of a slide, still within the 5 mm contact height the skeleton page reads, a skate
#   (park_E1_spontaneity_g0 of fe89df3, wave SH1d fix B)


def spont_amp_ratio(base_name):
    """The per-grade amplitude ratio of a base (SPONT_AMP_RATIO_BY_BASE)."""
    return SPONT_AMP_RATIO_BY_BASE.get(base_name, SPONT_AMP_RATIO)


def movement_reach_m(bgb, out, ev):
    """[m]: per movement of ev (with 'start_s' and 'len_s'), the largest displacement of a body joint or of the nose
    against the background body over its span."""
    from . import reach as RC
    J, J0 = out.joints().numpy(), bgb.joints().numpy()
    D = np.maximum(np.linalg.norm(J - J0, axis=-1).max(1),
                   np.linalg.norm(RC.nose_point(out) - RC.nose_point(bgb), axis=-1))
    r = out.rate
    return [float(D[int(round(e["start_s"] * r)):int(round((e["start_s"] + e["len_s"]) * r)) + 1].max()) for e in ev]


def _visible_chain(ev, reach0, duration_s, base_name):
    """{grade: kept indices} for grades 1 to 4 (review B2): nested subsets of spontaneity_count(g) movements each, every
    kept movement moving by at least SPONT_VISIBLE_M at its grade (reach0 times the amplitude ratio for a graded
    kind), a head turn kept up to grade 3; among the chains that hold, the one that drops the wrist-watch look
    earliest, then the one whose kept reach falls most evenly (the sum over grades of |log mean reach - log target|,
    the target the level-0 mean times the ratio); RuntimeError when none holds."""
    ratio = spont_amp_ratio(base_name)
    n = len(ev)
    vis = {g: [i for i in range(n) if reach0[i] * (ratio ** g if ev[i]["graded"] else 1.0) >= SPONT_VISIBLE_M]
           for g in range(1, 5)}
    head = {i for i in range(n) if ev[i]["kind"].startswith("head_turn")}
    watch = {i for i in range(n) if ev[i]["kind"] == "watch"}
    m0 = float(np.mean(reach0))
    best = None

    def walk(g, prev, chain):
        nonlocal best
        if g == 5:
            last = max([gg for gg, c in chain.items() if set(c) & watch], default=0)
            cost = sum(abs(math.log(np.mean([reach0[i] * (ratio ** gg if ev[i]["graded"] else 1.0) for i in c]))
                           - math.log(m0 * ratio ** gg)) for gg, c in chain.items())
            key = (last, cost)
            if best is None or key < best[0]:
                best = (key, dict(chain))
            return
        k = spontaneity_count(g, duration_s)
        for c in itertools.combinations([i for i in prev if i in vis[g]], k):
            if g <= 3 and head and not set(c) & head:
                continue
            chain[g] = list(c)
            walk(g + 1, list(c), chain)
            del chain[g]
    walk(1, list(range(n)), {})
    if best is None:
        raise RuntimeError(f"3.14 {base_name}: no nested choice keeps every movement visible")
    return best[1]


def spont_kinds(base_name):
    return tuple(k for k, v in SPONT_KINDS.items() if base_name in v[0])


def spontaneity_count(grade, duration_s, rate_per_min=SPONT_RATE_PER_MIN):
    """Movements in a clip of duration_s at a 3.14 grade: round(rate x minutes x SPONT_COUNT_RATIO^grade), at least
    one (module docstring)."""
    n = int(round(rate_per_min * duration_s / 60.0 * SPONT_COUNT_RATIO ** int(grade)))
    return max(n, 1)


def _hand_ids(side):
    _, _, dom = H._rest()
    w = 20 if side == "left" else 21
    return np.flatnonzero(np.isin(dom, (w,) + tuple(range(H.FIRST[side], H.FIRST[side] + 15))))


def _gap_above(P, S, radius=0.012):
    """The smallest height of points P (n, 3) above the highest of the surface points S (m, 3) within radius in
    the horizontal plane (points with no surface point under them are left out); inf when none."""
    d = np.hypot(P[:, None, 0] - S[None, :, 0], P[:, None, 2] - S[None, :, 2])
    top = np.where(d < radius, S[None, :, 1], -np.inf).max(1)
    ok = np.isfinite(top)
    return float((P[ok, 1] - top[ok]).min()) if ok.any() else math.inf


def _body2(R, tr, Rh):
    """A two-knot HandBody repeating knot 0 of R (1, 22, 3, 3), tr (1, 3), Rh (1, 30, 3, 3) (bodies need two knots)."""
    return H.HandBody(matrix_to_rot6d(R[:1].expand(2, -1, -1, -1)), tr[:1].expand(2, -1),
                      torch.zeros(2, 2, dtype=torch.bool), hand6d=matrix_to_rot6d(Rh[:1].expand(2, -1, -1, -1)))


@functools.lru_cache(maxsize=1)
def seated_rest():
    """The seated rest of the 3.14 clips (module docstring): {'R', 'tr' (22 rows and the translation of 'sit' with
    the arms solved), 'Rh' (the hands), per side 'wrist_local' and 'hand_local' (the wrist target and the hand's
    rotation in the thigh's frame), 'together' (the right wrist and hand on the back of the left hand, in the left
    thigh's frame), 'rub' (the end of a slide toward the knee, in the thigh's frame, m), 'watch' (the left wrist and
    hand in spine3's frame), and the clearances realized}."""
    base = static_body("sit", 0.01)
    R = base.local_R()[:1].clone()
    tr = base.transl[:1].clone()
    rest = rest_joints()
    up = np.array([THIGH_REST["upright_deg"]])
    _sagittal(R, up)                                   # by each joint's forward room (wave 6; SPINE_SHARES before)
    R[:, 12] = _rot("x", -up) @ R[:, 12]
    pr = np.array([THIGH_REST["protract_deg"]])
    R[:, 13] = _rot("y", -pr) @ R[:, 13]
    R[:, 14] = _rot("y", pr) @ R[:, 14]
    Rh = relaxed_hand(THIGH_REST["hand_weight"])[None]
    J, Gl = fk(R, tr, rest)
    sit_hand = {"left": Gl[0, 20].clone(), "right": Gl[0, 21].clone()}
    _, _, dom = H._rest()
    out = {"wrist_local": {}, "hand_local": {}, "clear_m": {}}
    targets = {}
    for side, (h, k), sgn in (("left", (1, 4), -1.0), ("right", (2, 5), 1.0)):
        G = _rot("y", np.array([sgn * THIGH_REST["yaw_deg"]]))[0] @ sit_hand[side]
        P0 = J[0, h] + THIGH_REST["along"] * (J[0, k] - J[0, h])
        W = P0 + torch.tensor([0.0, 0.07, -0.06], dtype=F64)
        thigh = np.flatnonzero(np.isin(dom, (h, k)))
        for it in range(6):
            sol = arm_ik(R, tr, side, W, G, keep_collar=True, pole=SEAT_POLE)
            Rk = R.clone()
            for j, M in zip(ARM[side], sol[:4]):
                Rk[:, j] = M
            b = _body2(Rk, tr, Rh)
            Ph = H.body_points(b, _hand_ids(side), knots=[0])[0]
            St = H.body_points(b, thigh, knots=[0])[0]
            c = Ph.mean(0)
            gap = _gap_above(Ph, St)
            step = np.array([float(P0[0]) - c[0], THIGH_REST["clear_m"] - gap, float(P0[2]) - c[2]])
            if it == 5 or np.abs(step).max() < 1e-4:
                break
            W = W + torch.from_numpy(step)
        R = Rk
        targets[side] = (W, G)
        out["wrist_local"][side] = Gl[0, h].T @ (W - J[0, h])
        out["hand_local"][side] = Gl[0, h].T @ G
        out["clear_m"][side] = float(gap)
        out[f"rub_dir_{side}"] = Gl[0, h].T @ ((J[0, k] - J[0, h]) / (J[0, k] - J[0, h]).norm())
    # hands together in the lap: the left hand at the midline, the right hand across its back
    lap = np.flatnonzero(np.isin(dom, (1, 2, 4, 5)))         # the thighs (the lower belly is not a lap)
    PL0 = J[0, 1] + LAP["along"] * (J[0, 4] - J[0, 1])
    Pm = torch.tensor([LAP["x_m"], float(PL0[1]), float(PL0[2])], dtype=F64)
    tog = {}
    Rt = R.clone()
    for side, yaw, under in (("left", -LAP["yaw_deg"], None), ("right", LAP["yaw_deg"], "left")):
        G = _rot("y", np.array([yaw]))[0] @ sit_hand[side]
        W = Pm + torch.tensor([0.0, 0.07 + (0.03 if under else 0.0), -0.06], dtype=F64)
        for it in range(8):
            sol = arm_ik(Rt, tr, side, W, G, keep_collar=True, pole=LAP_POLE)
            Rk = Rt.clone()
            for j, M in zip(ARM[side], sol[:4]):
                Rk[:, j] = M
            b = _body2(Rk, tr, Rh)
            Ph = H.body_points(b, _hand_ids(side), knots=[0])[0]
            surf = H.body_points(b, lap, knots=[0])[0]
            if under:
                surf = np.r_[surf, H.body_points(b, _hand_ids(under), knots=[0])[0]]
            gap = _gap_above(Ph, surf)
            c = Ph.mean(0)
            step = np.array([float(Pm[0]) - c[0], THIGH_REST["clear_m"] - gap, float(Pm[2]) - c[2]])
            if it == 7 or np.abs(step).max() < 1e-4:
                break
            W = W + torch.from_numpy(step)
        Rt = Rk
        tog[side] = (Gl[0, 0].T @ (W - J[0, 0]), Gl[0, 0].T @ G, float(sol[4][0]), float(gap))
    out["together_inside_table"] = bool(_limits_ok_rows(Rt, _arm_joints())[0])
    out["together"] = tog
    # the watch: the left forearm across the front of the body, the back of the wrist up
    S3, G3 = J[0, 9], Gl[0, 9]
    Ww = torch.tensor([WATCH["x_m"], float(J[0, 12, 1]) - WATCH["down_m"], float(S3[2]) + WATCH["front_m"]],
                      dtype=F64)
    Gw = _rot("y", np.array([WATCH["yaw_deg"]]))[0] @ sit_hand["left"]
    out["watch"] = (G3.T @ (Ww - S3), G3.T @ Gw)
    out["R"], out["tr"], out["Rh"] = R[0], tr[0], Rh[0]
    return out


def _spont_events(base_name, duration_s, seed):
    """The level-0 movements of a clip (module docstring): kinds (no kind twice in a row, none more often than
    SPONT_MAX_PER_CLIP), anchors, holds, hold drifts and amplitude factors drawn from seed, as a list of dicts sorted
    by time."""
    rng = as_rng(seed)
    kinds_all = spont_kinds(base_name)
    n = spontaneity_count(0, duration_s)
    kinds = []
    for _ in range(n):
        c = [k for k in kinds_all if (not kinds or k != kinds[-1]) and kinds.count(k) < SPONT_MAX_PER_CLIP.get(k, 2)]
        kinds.append(str(rng.choice(c)))
    ev = []
    for k in kinds:
        _, a0, unit, mv, hold, graded = SPONT_KINDS[k]
        ev.append({"kind": k, "level0_amplitude": a0, "unit": unit, "move_s0": mv,
                   "hold_s": float(rng.uniform(*(SPONT_LONG_HOLD_S if hold == "long" else SPONT_HOLD_S))),
                   "factor": float(rng.uniform(SPONT_AMP_JITTER[0], SPONT_AMP_JITTER[1] if unit != "share" else 1.0))
                   if graded else 1.0,
                   "drift": float(rng.uniform(*SPONT_HOLD_DRIFT)), "drift_seed": int(rng.integers(0, 2 ** 31)),
                   "graded": graded})
    lens = np.array([2 * e["move_s0"] + e["hold_s"] for e in ev])
    free = duration_s - 1.0 - lens.sum() - SPONT_GAP_S * (n - 1)
    if free < 0:
        raise ValueError("the movements do not fit in the clip")
    gaps = rng.dirichlet(np.ones(n + 1)) * free
    tc = 0.5
    for i, e in enumerate(ev):
        tc += gaps[i] + (SPONT_GAP_S if i else 0.0)
        e["anchor_s"] = float(tc)
        tc += lens[i]
    return ev, rng


def _place(ev, duration_s):
    """Start times of the kept movements (each at its level-0 anchor or after the previous one ends, then pulled back
    from the clip's end; module docstring)."""
    st = []
    for i, e in enumerate(ev):
        s = e["anchor_s"] if not st else max(e["anchor_s"], st[-1] + ev[i - 1]["len_s"] + SPONT_GAP_S)
        st.append(s)
    end = duration_s - 0.5
    for i in range(len(ev) - 1, -1, -1):
        lim = end if i == len(ev) - 1 else st[i + 1] - SPONT_GAP_S
        st[i] = min(st[i], lim - ev[i]["len_s"])
    if st and st[0] < 0.3:
        raise ValueError("the movements do not fit in the clip")
    for e, s in zip(ev, st):
        e["start_s"] = float(s)
    return ev


def _profile(t, e):
    """(T,) the movement's excursion share: a minimum-jerk rise over move_s, the hold (drifting by up to drift of the
    excursion), a minimum-jerk return."""
    up = _mj((t - e["start_s"]) / e["move_s"])
    dn = _mj((t - (e["start_s"] + e["move_s"] + e["hold_s"])) / e["move_s"])
    rng = np.random.default_rng(e["drift_seed"])
    dr = np.clip(0.5 * _band(rng, t, (0.1, 0.4), 4), -1.0, 1.0)
    p = up - dn
    if e["kind"].startswith("foot_slide") or e["kind"] == "hands_together":     # planted, or resting on a hand
        return p, up, dn
    return p * (1.0 + e["drift"] * dr * up * (1.0 - dn)), up, dn


def _channels(base_name, t, bg, ev, rest):
    """The channels of the realization (module docstring) from the background signals bg and the movements ev."""
    T = len(t)
    z = np.zeros(T)
    ch = {"breath": bg["breath"], "head_yaw": bg["yaw"].copy(), "head_pitch": bg["pitch"].copy(),
          "lean": bg["lean"] if base_name == "sit" else z.copy(),
          "lean_ml": bg["lean_ml"] if base_name == "sit" else z.copy(), "trunk_yaw": z.copy(), "trunk_lean": z.copy(),
          "trunk_bend": z.copy(), "pel": np.zeros((T, 3)), "roll": z.copy(), "sway_z": bg["sway_z"],
          "sway_x": bg["sway_x"], "collar": z.copy(), "ankle": np.zeros((T, 2, 3)), "arm": {"left": [], "right": []},
          "fingers": bg["fingers"]}
    for e in ev:
        p, up, dn = _profile(t, e)
        a, k = e["amplitude"], e["kind"]
        bump = np.sin(np.pi * up) + np.sin(np.pi * dn)                # midway through each movement
        if k.startswith("head_turn"):
            sgn = 1.0 if k.endswith("left") else -1.0
            ch["head_yaw"] += sgn * a * p
            ch["trunk_yaw"] += sgn * TRUNK_TURN_SHARE * a * p
        elif k == "glance_down":
            ch["head_pitch"] += a * p
        elif k == "watch":
            ch["arm"]["left"].append(("watch", a * p))
            ch["head_pitch"] += WATCH["pitch_deg"] * a * p
            ch["head_yaw"] += 5.0 * a * p
        elif k == "hands_together":
            ch["arm"]["left"].append(("together", a * p))
            ch["arm"]["right"].append(("together", a * p))
        elif k.startswith("rub_thigh"):
            ch["arm"][k.rsplit("_", 1)[1]].append(("rub", a * p))
        elif k == "shrug":
            ch["collar"] += a * p
        elif k == "seat_adjust":
            ch["pel"][:, 2] += a * p
            ch["pel"][:, 1] += SEAT_LIFT_M * a / 0.03 * bump
            ch["trunk_lean"] += SEAT_LEAN_DEG * a / 0.03 * bump
        elif k.startswith("foot_slide"):
            i = 0 if k.endswith("left") else 1
            lift = max(FOOT_LIFT_SHARE * abs(a), FOOT_LIFT_MIN_M)
            u0 = math.asin(min(1.0, FOOT_CLEAR_M / lift)) / math.pi      # the lift sin(pi u) reaches FOOT_CLEAR_M

            def way(q, u0=u0):
                return _mj((q - u0) / (1.0 - 2.0 * u0))
            ch["ankle"][:, i, 2] += a * (way(up) - way(dn))
            ch["ankle"][:, i, 1] += lift * bump
        elif k.startswith("weight_shift"):
            sgn = 1.0 if k.endswith("left") else -1.0
            ch["pel"][:, 0] += sgn * a * p
            ch["roll"] += sgn * SHIFT_ROLL_DEG * a / 0.025 * p
    return ch


def _arm_targets(side, J, Gl, rest_, kind, q):
    """(W (T, 3), G (T, 3, 3)) of an arm movement of kind at progress q (T,) from the seated rest (module
    docstring), in the current frames J, Gl (T, 22, ...)."""
    h = 1 if side == "left" else 2
    wl = torch.as_tensor(rest_["wrist_local"][side])
    hl = torch.as_tensor(rest_["hand_local"][side])
    W0 = J[:, h] + torch.einsum("tij,j->ti", Gl[:, h], wl)
    G0 = Gl[:, h] @ hl
    qq = torch.from_numpy(np.asarray(q, dtype=np.float64))
    if kind == "rub":
        d = torch.einsum("tij,j->ti", Gl[:, h], torch.as_tensor(rest_[f"rub_dir_{side}"]))
        return W0 + qq[:, None] * d, G0
    if kind == "together":
        wl1, hl1 = rest_["together"][side][:2]
        W1 = J[:, 0] + torch.einsum("tij,j->ti", Gl[:, 0], torch.as_tensor(wl1))
        G1 = Gl[:, 0] @ torch.as_tensor(hl1)
    else:                                                               # watch, in spine3's frame
        W1 = J[:, 9] + torch.einsum("tij,j->ti", Gl[:, 9], torch.as_tensor(rest_["watch"][0]))
        G1 = Gl[:, 9] @ torch.as_tensor(rest_["watch"][1])
    W = W0 + qq[:, None] * (W1 - W0)
    W[:, 1] += ARC_M * 4.0 * qq * (1.0 - qq).clamp_min(0.0)
    G = _blend(G0[:, None], G1[:, None], np.asarray(q))[:, 0]
    return W, G


@functools.lru_cache(maxsize=1)
def stand_watch():
    """(3, 3, 3): the local rotations of the left shoulder, elbow and wrist looking at the watch on 'stand' (arm_ik
    toward seated_rest()['watch'] in spine3's frame, the collar at rest); the standing arm reaches it by a geodesic
    blend of each joint from the hanging arm (a joint-space reach: the hand rises in front of the body)."""
    b = static_body("stand", 0.01)
    R, tr = b.local_R()[:1], b.transl[:1]
    J, Gl = fk(R, tr, rest_joints())
    wl, hl = seated_rest()["watch"]
    W = J[0, 9] + Gl[0, 9] @ torch.as_tensor(wl)
    sol = arm_ik(R, tr, "left", W, Gl[0, 9] @ torch.as_tensor(hl), keep_collar=True, pole=SEAT_POLE)
    return torch.stack([sol[1][0], sol[2][0], sol[3][0]], 0)


def _realize(base_name, ch, T, rate=100.0):
    """(HandBody, planted (T, 2) bool) of the channels ch on the static base (module docstring): trunk and head,
    the pelvis with the legs re-solved to the (moved) ankles, the collars, the arms (seated: on the thighs by arm_ik
    or toward a movement's target; standing: the 'stand' arms, or arm_ik toward a target blended in), the hands."""
    rest = rest_joints()
    stand = base_name == "stand"
    if stand:
        b0 = static_body("stand", (T - 1) / rate)
        R0, tr0 = b0.local_R()[:1], b0.transl[:1]
        hand_w = 1.0
    else:
        sr = seated_rest()
        R0, tr0 = sr["R"][None], sr["tr"][None]
        hand_w = THIGH_REST["hand_weight"]
    R = R0.expand(T, -1, -1, -1).clone()
    tr = tr0.expand(T, -1).clone()
    Jb, Gb = fk(R0, tr0, rest)
    # trunk and head
    R[:, 9] = _rot("x", -BREATH_SPINE_DEG * ch["breath"]) @ R[:, 9]
    sh_x = sagittal_shares(R0[0])                      # the sagittal parts by each joint's room (wave 6)
    lean = np.asarray(ch["trunk_lean"], dtype=np.float64) * np.ones(T)
    for i, (j, sh) in enumerate(zip(SPINE, SPINE_SHARES)):
        ax = np.where(lean >= 0.0, sh_x[1.0][i], sh_x[-1.0][i]) * lean
        R[:, j] = _rot("z", sh * ch["trunk_bend"]) @ _rot("x", ax) @ R[:, j] @ _rot("y", sh * ch["trunk_yaw"])
    _sagittal(R, ch["lean"], R0[0])
    R[:, 3] = _rot("z", ch["lean_ml"]) @ R[:, 3]
    for j, sh in zip((12, 15), HEAD_SHARES):
        R[:, j] = R[:, j] @ _rot("y", sh * ch["head_yaw"]) @ _rot("x", sh * ch["head_pitch"])
    R[:, 12] = _rot("x", BREATH_NECK_SHARE * BREATH_SPINE_DEG * ch["breath"]) @ R[:, 12]
    # the pelvis and the legs
    ank0 = torch.stack([Jb[0, 7], Jb[0, 8]], 0)[None].expand(T, 2, 3)
    ank = ank0 + torch.from_numpy(ch["ankle"])
    pel = torch.from_numpy(ch["pel"])
    if stand:
        A = 0.5 * (Jb[0, 7] + Jb[0, 8])
        h = float(Jb[0, 0, 1] - A[1])
        th = np.arctan2(ch["sway_z"], h)
        Rx = _rot("x", np.degrees(th))
        Rr = _rot("z", ch["roll"])
        J0 = A[None] + torch.einsum("tij,j->ti", Rx, Jb[0, 0] - A) + pel
        J0[:, 0] += torch.from_numpy(ch["sway_x"])
        G0 = Rr @ Rx @ Gb[0, 0][None]
        R[:, 3] = _rot("z", -ch["roll"]) @ R[:, 3]                     # the trunk stays upright over the shift
    else:
        J0 = Jb[0, 0][None] + pel
        G0 = Gb[0, 0][None].expand(T, 3, 3).clone()
    R, tr = place_pelvis(R, tr, J0, G0, ank, R0.expand(T, -1, -1, -1), tr0.expand(T, -1), keep_reach=stand)
    # collars: breathing, and the shrug (seated: the arms are solved after, the hands stay on the thighs; standing:
    # the arms ride on the breathing collars and hang under the shrug, the shoulders turned back by its rise, read
    # against the table 'shrug' of limits_ext, SHRUG_TABLE)
    if stand:
        _collar_breath(R, BREATH_COLLAR_DEG * ch["breath"])
        on = np.flatnonzero(ch["collar"] != 0.0)
        if len(on):
            Ro = R[on]
            _collar_breath(Ro, ch["collar"][on], counter=True)
            R[on] = Ro
    else:
        _collar_breath(R, BREATH_COLLAR_DEG * ch["breath"] + ch["collar"])
    J, Gl = fk(R, tr, rest)
    for side in ("left", "right"):
        moves = ch["arm"][side]
        q = sum((m[1] for m in moves), np.zeros(T))
        kind = moves[0][0] if moves else None
        kinds = {m[0] for m in moves}
        if not stand:
            if len(kinds) > 1:                       # one target kind per call: realize each movement in turn
                W, G = _arm_targets(side, J, Gl, sr, "rub", np.zeros(T))
                for kd, qk in moves:
                    on = qk != 0
                    Wk, Gk = _arm_targets(side, J, Gl, sr, kd, qk)
                    W[on], G[on] = Wk[on], Gk[on]
            else:
                W, G = _arm_targets(side, J, Gl, sr, kind or "rub", q)
            qt = sum((m[1] for m in moves if m[0] == "together"), np.zeros(T))
            pole = tuple(SEAT_POLE[i] + (LAP_POLE[i] - SEAT_POLE[i]) * _mj(qt) for i in (0, 1))
            sol = arm_ik(R, tr, side, W, G, keep_collar=True, pole=pole)
            for jj, M in zip(ARM[side], sol[:4]):
                R[:, jj] = M
        elif kind is not None:                       # the watch, standing (stand_watch)
            on = np.flatnonzero(q > 0)
            rows = list(ARM[side][1:])
            R[on[:, None], rows] = _blend(R[on][:, rows], stand_watch()[None].expand(len(on), -1, -1, -1), q[on])
    fing = ch["fingers"] if stand else ch["fingers"] - ch["fingers"].max(0, keepdims=True)   # seated: lift only
    Rh = relaxed_hand(hand_w, fing)
    planted = (np.abs(ch["ankle"]).sum(-1) < 1e-12) | (np.abs(np.gradient(ch["ankle"], axis=0)).sum(-1) < 1e-12)
    b = H.HandBody(matrix_to_rot6d(R), tr, torch.zeros(T, 2, dtype=torch.bool), rate=rate, hand6d=matrix_to_rot6d(Rh))
    lab = contact_labels(b) & torch.from_numpy(planted)
    return H.HandBody(b.rot6d, b.transl, lab, b.betas, b.gender, b.rate, b.hand6d), planted


@functools.lru_cache(maxsize=4)
def _level0_order(base_name, duration_s, seed):
    """(level-0 movements, {grade: indices kept}, re-measured level-0 peaks deg): the grades keep nested subsets
    (module docstring), each the one of its size whose predicted mean peak (the level-0 peaks times the grade's
    amplitude ratio for the graded kinds) lies nearest the level-0 mean times that ratio and below
    SPONT_LADDER_STEP of the grade below's, ties broken by the seed."""
    ev, rng = _spont_events(base_name, duration_s, seed)
    for e in ev:
        e["amplitude"] = e["level0_amplitude"] * e["factor"]
        if e["unit"] == "share":
            e["amplitude"] = min(1.0, e["amplitude"])
        e["move_s"] = e["move_s0"]
        e["len_s"] = 2 * e["move_s"] + e["hold_s"]
        e["start_s"] = e["anchor_s"]
    _limit_head(base_name, ev)
    T = int(round(duration_s * 100.0)) + 1
    t = np.arange(T) / 100.0
    bg_sig = background_signals(SPONT_SERIES[base_name], T, 1.0, np.ones(T) if base_name == "stand" else None)
    bgb, _ = _realize(base_name, _channels(base_name, t, bg_sig, [], None), T)
    out, _ = _realize(base_name, _channels(base_name, t, bg_sig, ev, None), T)
    m = measure_spontaneity(bgb, out)
    if m["count"] != len(ev):
        raise RuntimeError(f"level 0 of {base_name}: {m['count']} movements read back for {len(ev)}")
    peaks = np.array([mv["peak_deg"] for mv in m["movements"]])
    kept, sets, m_prev = list(range(len(ev))), {0: list(range(len(ev)))}, float(peaks.mean())
    if base_name in SPONT_CHAIN_BASES:      # by visibility (review B2)
        sets.update(_visible_chain(ev, movement_reach_m(bgb, out, ev), duration_s, base_name))
        return [{k: v for k, v in e.items() if k not in ("amplitude", "move_s", "len_s", "start_s")} for e in ev], \
            sets, [mv["peak_deg"] for mv in m["movements"]]
    for g in range(1, 5):              # nested: each grade keeps a subset of the previous grade's movements
        sc = np.array([spont_amp_ratio(base_name) ** g if e["graded"] else 1.0 for e in ev])
        target = float(peaks.mean()) * spont_amp_ratio(base_name) ** g
        cand = [list(c) for c in itertools.combinations(kept, spontaneity_count(g, duration_s))]
        pred = np.array([float(np.mean(peaks[c] * sc[c])) for c in cand])
        cost = np.abs(pred - target) + rng.uniform(0.0, 1e-3, size=len(cand))
        cost[pred > SPONT_LADDER_STEP * m_prev] += 1e6           # the mean peak falls from grade to grade
        i = int(np.argmin(cost))
        kept, m_prev = cand[i], float(pred[i])
        sets[g] = kept
    return [{k: v for k, v in e.items() if k not in ("amplitude", "move_s", "len_s", "start_s")} for e in ev], \
        sets, [mv["peak_deg"] for mv in m["movements"]]


def _limit_head(base_name, ev):
    """Head turns and glances bounded by SPONT_ROOM_SHARE of the room the table leaves the neck and head."""
    R = posture_rotations("stand" if base_name == "stand" else "sit")
    for e in ev:
        if e["kind"].startswith("head_turn") or e["kind"] == "glance_down":
            ax = "y" if e["kind"].startswith("head_turn") else "x"
            sgn = (1.0 if e["kind"].endswith("left") else -1.0) if ax == "y" else 1.0
            room = min(room_deg(R[j], j, ax, sgn, "child") / s for j, s in zip((12, 15), HEAD_SHARES))
            e["amplitude"] = min(e["amplitude"], SPONT_ROOM_SHARE * room)


def spontaneity_parts(base_name="sit", grade=0, duration_s=None, seed=None):
    """(background HandBody, HandBody, events, params) of the 3.14 operator (module docstring)."""
    duration_s = SPONT_DUR[base_name] if duration_s is None else float(duration_s)
    seed = SEED + SPONT_SEED_OFFSET[base_name] if seed is None else int(seed)
    g = int(grade)
    ev0, sets, peaks0 = _level0_order(base_name, duration_s, seed)
    keep = set(sets[g])
    amp_k, slow_k, bg_k = spont_amp_ratio(base_name) ** g, SPONT_SLOW_RATIO ** g, BG_RATIO ** g
    ev = []
    for i, e0 in enumerate(ev0):
        if i not in keep:
            continue
        e = dict(e0, level0_index=i, level0_peak_deg=peaks0[i])
        e["amplitude"] = e["level0_amplitude"] * e["factor"] * (amp_k if e["graded"] else 1.0)
        if e["unit"] == "share":
            e["amplitude"] = min(1.0, e["amplitude"])
        e["move_s"] = e["move_s0"] * slow_k
        e["len_s"] = 2 * e["move_s"] + e["hold_s"]
        ev.append(e)
    _limit_head(base_name, ev)
    ev = _place(ev, duration_s)
    T = int(round(duration_s * 100.0)) + 1
    t = np.arange(T) / 100.0
    bg_sig = background_signals(SPONT_SERIES[base_name], T, bg_k, np.ones(T) if base_name == "stand" else None)
    bgb, _ = _realize(base_name, _channels(base_name, t, bg_sig, [], None), T)
    out, planted = _realize(base_name, _channels(base_name, t, bg_sig, ev, None), T)
    p = {"base": base_name, "duration_s": duration_s, "n_movements": len(ev),
         "rate_per_min": len(ev) / duration_s * 60.0, "level0_rate_per_min": SPONT_RATE_PER_MIN,
         "count_ratio": SPONT_COUNT_RATIO ** g, "amplitude_ratio": amp_k, "movement_time_ratio": slow_k,
         "background_ratio": bg_k, "level0_kept": sorted(keep),
         "events": [{"kind": e["kind"], "start_s": e["start_s"], "move_s": e["move_s"], "hold_s": e["hold_s"],
                     "end_s": e["start_s"] + e["len_s"], "amplitude": e["amplitude"], "unit": e["unit"],
                     "level0_amplitude": e["level0_amplitude"], "factor": e["factor"], "graded": e["graded"],
                     "hold_drift": e["drift"], "level0_index": e["level0_index"],
                     "level0_peak_deg": e["level0_peak_deg"]} for e in ev],
         "background": background_record(SPONT_SERIES[base_name], T, bg_k, 100.0, SPONT_PARTS[base_name]),
         "rest": ("hands on the thighs (seated_rest)" if base_name == "sit" else "'stand', relaxed hands")}
    return bgb, out, ev, p, planted


def spontaneity(base_name="sit", grade=0, duration_s=None, seed=None):
    """(HandBody, truth) of the 3.14 body bradykinesia operator (module docstring) on a seated or standing rest."""
    bgb, out, ev, p, planted = spontaneity_parts(base_name, grade, duration_s, seed)
    base = static_body(base_name, p["duration_s"])
    if base_name == "stand" and any(e["kind"] == "shrug" for e in ev):     # the arms hang under the shrug
        p["limits_table"] = LX.record(SHRUG_TABLE)
        val = LX.check(SHRUG_TABLE, base, out)
    else:
        val = validity(base, out)
    val["hand_limits"] = H.hand_limit_report(out.hand_R())
    val["continuity"] = continuity_report(out)
    J = out.joints()
    fs = np.linalg.norm(np.gradient(J[:, [7, 8]].numpy(), axis=0), axis=-1) * out.rate
    val["planted_foot_speed_max_m_s"] = float(np.where(planted, fs, 0.0).max())
    p["remeasured"] = measure_spontaneity(bgb, out)
    reach = movement_reach_m(bgb, out, ev)
    p["movement_reach_mm"] = [1000.0 * x for x in reach]
    if base_name in SPONT_CHAIN_BASES and int(grade) > 0:   # review B2: a grade whose movements fall under the
        #                                                     floor is refused (the level 0 is not graded)
        val["visible"] = {"ok": bool(min(reach) >= SPONT_VISIBLE_M - 1e-4) if reach else True,
                          "reach_mm_min": 1000.0 * min(reach) if reach else None,
                          "floor_mm": 1000.0 * SPONT_VISIBLE_M}
        if not val["visible"]["ok"]:
            raise RuntimeError(f"3.14 {base_name} grade {grade}: a movement under the visibility floor {reach}")
    seed_v = SEED + SPONT_SEED_OFFSET[base_name] if seed is None else int(seed)
    return out, truth_record("spontaneity", seed_v, int(grade), "MDS-UPDRS 3.14", p,
                             ("mds_updrs", "elble2006", "spont_unsourced", "module"), val)


def measure_spontaneity(base, out, speed_deg_s=1.0, min_gap_s=0.3, on_deg=0.3):
    """Movements read back from a body against its background body (the same rest, breathing, drift and sway
    without the movements): the largest deviation (deg) over the joints of every local rotation from the
    background, its speed; a movement is a run whose deviation rises above on_deg with speed above speed_deg_s at
    some knot (runs closer than min_gap_s merged); per movement its peak deviation and its duration (the run above
    on_deg)."""
    from .body import rotation_angle_deg
    D = rotation_angle_deg(base.local_R()[:, 1:].transpose(-1, -2) @ out.local_R()[:, 1:]).numpy().max(1)
    on = D > on_deg
    m = np.diff(np.r_[False, on, False].astype(np.int8))
    runs = list(zip(np.flatnonzero(m == 1), np.flatnonzero(m == -1)))
    merged = []
    for a, e in runs:
        if merged and (a - merged[-1][1]) / out.rate < min_gap_s:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((a, e))
    v = np.abs(np.gradient(D, 1.0 / out.rate))
    mv = [{"start_s": a / out.rate, "duration_s": (e - a) / out.rate, "peak_deg": float(D[a:e].max())}
          for a, e in merged if v[a:e].max() > speed_deg_s]
    return {"count": len(mv), "movements": mv,
            "mean_peak_deg": float(np.mean([x["peak_deg"] for x in mv])) if mv else 0.0,
            "mean_duration_s": float(np.mean([x["duration_s"] for x in mv])) if mv else 0.0}


# ---------------------------------------------------------------- sequence effect
SEQ_RATE_HZ = 50.0 / 15.0                 # SOURCES['ling2012'] controls, 50.0 taps in 15 s
SEQ_FT_CYCLES = 50
SEQ_PS_CYCLES = 30
SEQ_CONTROL = 0.12 / 45.9                 # SOURCES['ling2012'], share of the first amplitude per cycle
SEQ_PD = 0.20 / 37.8
SEQUENCE_SLOPES = {0: SEQ_CONTROL, 1: SEQ_PD, 2: 2 * SEQ_PD, 3: 3 * SEQ_PD}   # levels 2 and 3: module
SEQ_PLATEAU_TAU = {"finger_tapping": 8.0, "pronation_supination": 5.0}   # [module] levels 2 and 3 fall fast, then
#   plateau: A_c = A0 (1 - L (1 - exp(-(c - 1) / tau))), L such that the least-squares relative slope is the level's
SEQ_PERIOD_CV = 0.05           # [module] cycle-to-cycle variability of the period (4 to 6 percent)
SEQ_AMP_CV = 0.08              # [module] of the amplitude, share of A0 (6 to 10 percent)
SEQ_AR = 0.5                   # [module] the variability's correlation from one cycle to the next
SEQ_HESITATIONS = {3: 2}       # [module] brief hesitations at level 3 (the fingers or the forearm held still)
SEQ_HESITATION_PERIODS = 0.6   # [module] a hesitation lasts this share of a level-0 period (shorter than the 75
#                                percent of a freeze, SOURCES['williams2013'])


def _ar1(rng, n, rho=SEQ_AR):
    x = np.zeros(n)
    e = rng.standard_normal(n)
    x[0] = e[0]
    for i in range(1, n):
        x[i] = rho * x[i - 1] + math.sqrt(1.0 - rho * rho) * e[i]
    return x


def sequence_cycles(task, level, n, P0, seed):
    """(periods (n,) s, amplitudes (n,) share of A0, pauses {cycle: s}, info) of a sequence-effect run (module
    docstring): the level's trend (linear for levels 0 and 1, a fast fall to a plateau for 2 and 3) at the relative
    least-squares slope SEQUENCE_SLOPES[level], an AR(1) amplitude jitter of SEQ_AMP_CV made orthogonal to the
    trend's least-squares fit and to the first and last five cycles' means (the read-back slope and loss are the
    level's) and zero on the first cycle, an AR(1) period
    jitter of SEQ_PERIOD_CV whose cycles keep the level-0 span from the first peak to the last, and the level-3
    hesitations after cycles drawn in the second half."""
    rng = as_rng(seed + 31 * int(level) + (task == "pronation_supination"))
    s = SEQUENCE_SLOPES[int(level)]
    c = np.arange(n, dtype=np.float64)
    if level in (0, 1):
        f, shape = 1.0 - s * c, "linear"
    else:
        g = 1.0 - np.exp(-c / SEQ_PLATEAU_TAU[task])
        bg, ag = np.polyfit(c, g, 1)
        L = s / (bg + s * ag)
        f, shape = 1.0 - L * g, f"fall to a plateau (tau {SEQ_PLATEAU_TAU[task]:g} cycles, L {L:.3f})"
    X = np.c_[np.ones(n), c, np.eye(n)[0], (c < 5).astype(float), (c >= n - 5).astype(float)]
    nz = _ar1(rng, n)
    nz = nz - X @ np.linalg.lstsq(X, nz, rcond=None)[0]
    nz = nz / nz.std()
    A = f + SEQ_AMP_CV * nz
    pm = _ar1(rng, n)
    per = P0 * (1.0 + SEQ_PERIOD_CV * (pm - pm.mean()) / pm.std())
    span = 0.5 * per[0] + per[1:-1].sum() + 0.5 * per[-1]
    per = per * (n - 1) * P0 / span
    pauses = {}
    k = SEQ_HESITATIONS.get(int(level), 0)
    if k:
        for after in sorted(rng.choice(np.arange(n // 2, n - 2), size=k, replace=False)):
            pauses[int(after) + 1] = SEQ_HESITATION_PERIODS * P0
    b, a = np.polyfit(c, A, 1)
    return per, A, pauses, {"shape": shape, "relative_slope_design": s, "relative_slope_realized": float(-b / a),
                            "period_cv": float(per.std() / per.mean()), "amplitude_cv_of_A0": SEQ_AMP_CV,
                            "ar1": SEQ_AR, "hesitations_after_cycles": sorted(pauses)}


def sequence_effect(task, level, seed=SEED):
    """(Body, truth) of the sequence effect (module docstring): task 'finger_tapping' (50 cycles at SEQ_RATE_HZ, the
    cited level-0 amplitude; the fingers close fully every cycle, the opening shrinks) or 'pronation_supination'
    (30 cycles at the version-2 rate and amplitude, turning about the excursion's midpoint so that both ends
    shrink), the cycles of sequence_cycles."""
    from .bradykinesia import LEVEL0_SHARE, TASKS as BT, task_angle, task_axis, task_rooms, task_sign
    from .handtasks import TASKS as HT, hand_at, level0, share_of_amplitude, task_rows, tip_distance_m
    from .motions import posture_body
    side, lv = "left", int(level)
    if task == "finger_tapping":
        n, P0 = SEQ_FT_CYCLES, 1.0 / SEQ_RATE_HZ
        a0, _, anchor = level0(task, side)
        unit = "mm"
    elif task == "pronation_supination":
        n, P0 = SEQ_PS_CYCLES, 2 * int(round(100.0 / BT[task]["level0_hz"] / 2)) / 100.0
        unit = "deg"
    else:
        raise ValueError(f"task {task!r}")
    dur = round(0.5 + n * P0 + 1.5, 2)
    base = posture_body("sit_finger_tapping", dur) if task == "finger_tapping" else \
        static_body("sit_arms_forward", dur)
    k0 = int(round(0.5 * base.rate))
    if task == "pronation_supination":
        rooms = task_rooms(base, task, side, k0)
        a0 = LEVEL0_SHARE * float(sum(rooms))
    per, A, pauses, info = sequence_cycles(task, lv, n, P0, seed)
    amps = a0 * A
    t = base.times().numpy()
    x = np.zeros(base.T)
    tc, cycles = 0.5, []
    if task == "finger_tapping":
        shares = share_of_amplitude(base, task, side, amps, k0)
        for i in range(n):
            u = (t - tc) / per[i]
            ins = (u >= 0) & (u <= 1)
            x[ins] = shares[i] * 0.5 * (1.0 - np.cos(2.0 * math.pi * u[ins]))
            cycles.append({"cycle": i + 1, "start_s": float(tc), "period_s": float(per[i]), "amplitude": float(amps[i]),
                           "share": float(shares[i])})
            tc += per[i] + pauses.get(i + 1, 0.0)
        Rh = base.hand_R().clone()
        Rh[:, task_rows(side)] = hand_at(task, side, x)[:, task_rows(side)]
        out = base.with_hands(Rh, task_rows(side))
        Rc = base.hand_R().clone()
        Rc[:, task_rows(side)] = hand_at(task, side, np.zeros(base.T))[:, task_rows(side)]
        closed = float(tip_distance_m(base.with_hands(Rc, task_rows(side)), side, knots=[k0])[0])
        sig = 1000.0 * (tip_distance_m(out, side) - closed)
        val = validity(base, out)
        val["hand_limits"] = H.hand_limit_report(out.hand_R())
        extra = {"measure": HT[task]["measure"], "closed_tip_distance_mm": 1000.0 * closed, "anchor": anchor}
        srcs = ["mds_updrs", anchor["rate_source"], anchor["amplitude_source"], "module"]
        arrays = {"share": x}
    else:
        mid = 0.5 * a0
        trough = np.r_[0.0, mid - 0.5 * amps[1:], 0.0]          # the first starts and the last ends at the rest
        for i in range(n):
            u = (t - tc) / per[i]
            pk = mid + 0.5 * amps[i]
            up = (u >= 0) & (u < 0.5)
            dn = (u >= 0.5) & (u <= 1)
            x[up] = trough[i] + (pk - trough[i]) * 0.5 * (1.0 - np.cos(2.0 * math.pi * u[up]))
            x[dn] = trough[i + 1] + (pk - trough[i + 1]) * 0.5 * (1.0 - np.cos(2.0 * math.pi * u[dn]))
            cycles.append({"cycle": i + 1, "start_s": float(tc), "period_s": float(per[i]),
                           "amplitude_deg": float(pk - trough[i]), "peak_deg": float(pk),
                           "trough_deg": float(trough[i])})
            tc += per[i]
            if i + 1 in pauses:
                hold = (t > tc) & (t <= tc + pauses[i + 1])
                x[hold] = trough[i + 1]
                tc += pauses[i + 1]
        chain, frames = BT[task]["chain"][side], BT[task]["frames"]
        split = [r / sum(rooms) for r in rooms]
        ax = task_axis(task, side, base)
        R = base.local_R()
        Rall = R.clone()
        o = np.radians(task_sign(task, side) * x)
        for j, fr, sh in zip(chain, frames, split):
            Rall[:, j] = compose(R[:, j], axis_rotation(ax, torch.from_numpy(o * sh)), fr)
        out = base.with_rotations(Rall, list(chain))
        sig = task_angle(out, task, side, 0)
        val = validity(base, out)
        extra = {"chain": [int(j) for j in chain], "frames": list(frames), "split": [float(v) for v in split],
                 "room_deg": float(sum(rooms)), "midpoint_deg": float(mid),
                 "axis": [float(v) for v in ax], "direction": task_sign(task, side)}
        srcs = ["mds_updrs", "table", "module"]
        arrays = {"offset_rad": o}
    p = {"task": task, "item": "3.4" if task == "finger_tapping" else "3.6", "side": side, "n_cycles": n,
         "unit": unit, "amplitude0": float(a0), "level0_hz": 1.0 / P0, "period_s": P0, "start_s": 0.5,
         "end_s": float(tc), "cycles": cycles,
         "pauses": [{"kind": "hesitation", "after_cycle": c, "duration_s": d}
                    for c, d in sorted(pauses.items())],
         "realized_rate_hz": float((n - 1) / (sum(per[1:-1]) + 0.5 * (per[0] + per[-1]) +
                                              sum(d for c, d in pauses.items() if c < n))),
         "sequence_level": lv, "relative_slope_per_cycle": SEQUENCE_SLOPES[lv],
         "slope_source": "ling2012" if lv in (0, 1) else "module", **info, **extra,
         "remeasured": measure_decrement(sig, out.rate)}
    srcs += ["ling2012", "bologna2023"]
    return out, truth_record("sequence_effect", seed, lv, "sequence effect (MDS-UPDRS 3.4, 3.6 decrement)", p, srcs,
                             val, arrays=dict(arrays, signal=np.asarray(sig)))


def _extremes(x):
    """(maxima, minima) knot indices of a smooth signal (sign changes of its first difference)."""
    d = np.diff(x)
    s = np.sign(d)
    for i in range(1, len(s)):                     # flats take the previous sign
        if s[i] == 0:
            s[i] = s[i - 1]
    ch = np.diff(s)
    return np.flatnonzero(ch < 0) + 1, np.flatnonzero(ch > 0) + 1


def measure_decrement(signal, rate, rel=0.05):
    """Per-cycle amplitudes of a task signal that rests at its minimum between cycles (the hand tasks' opening, the
    forearm's turn): every local maximum above rel of the largest, its height above the preceding minimum; the
    relative least-squares slope (share of the fitted first amplitude per cycle) and the loss of the last five
    cycles against the first five."""
    x = np.asarray(signal, dtype=np.float64)
    mx, mn = _extremes(x)
    mx = mx[x[mx] > rel * x.max()]
    amps, times = [], []
    for k in mx:
        prev = mn[mn < k]
        lo = x[prev[-1]] if len(prev) else x[:k].min()
        amps.append(float(x[k] - lo))
        times.append(k / rate)
    A = np.asarray(amps)
    if len(A) < 2:
        return {"n_cycles": int(len(A)), "amplitudes": amps}
    b, a = np.polyfit(np.arange(len(A)), A, 1)
    return {"n_cycles": int(len(A)), "amplitudes": amps, "peak_times_s": times, "relative_slope": float(-b / a),
            "loss_last5_vs_first5": float(1.0 - A[-5:].mean() / A[:5].mean()),
            "mean_rate_hz": float((len(A) - 1) / (times[-1] - times[0]))}


# ---------------------------------------------------------------- upper-limb freezing (finger tapping)
HASTEN_CYCLES = 5              # [module] cycles of hastening and amplitude breakdown before an episode
HASTEN_GAIN = 0.4              # [module] rate rise over them (the size of the hastening was not found)
BREAKDOWN_SHARE = 0.35         # [module] amplitude at the last hastened cycle, share of A0 (at or below 50
#                                percent: SOURCES['williams2013'])
FREEZE_S = 0.84                # SOURCES['delval2017'], uncued mean
TREMBLE_HZ = 5.5               # centre of the freeze band (SOURCES['vercruysse2012'], grades.FREEZE_BAND_HZ)
TREMBLE_LO, TREMBLE_HI = 0.04, 0.14    # [module] the trembling opens the fingers between these shares of A0
RESUME_SHARE = 0.8             # [module] amplitude after an episode, share of A0
FREEZE_PLANS = {
    "3.4_g3": {"cycles": 10, "episodes_after": (7,), "rate_hz": None, "durations_s": (FREEZE_S,)},
    "long": {"cycles": 50, "episodes_after": (18, 38), "rate_hz": SEQ_RATE_HZ, "durations_s": (FREEZE_S, 1.29)},
}


def freeze_rate_hz(plan, side="left"):
    from .handtasks import level0
    pl = FREEZE_PLANS[plan]
    return level0("finger_tapping", side)[1] if pl["rate_hz"] is None else pl["rate_hz"]


def freeze_duration_s(plan):
    """Seconds of a freezing clip and of its level 0 (the same time base): 0.5 s lead, the cycles at the level-0
    rate, the episodes, 2.5 s after (the hastened cycles are shorter: the margin covers them)."""
    pl = FREEZE_PLANS[plan]
    return round(0.5 + pl["cycles"] / freeze_rate_hz(plan) + sum(pl["durations_s"]) + 2.5, 2)


def upper_limb_freezing(plan="3.4_g3", seed=SEED, side="left"):
    """(HandBody, truth) of finger tapping with upper-limb freezing (module docstring), on the handtasks primitives:
    per-cycle rate and amplitude, the episodes' trembling share, the hand rows set from the share signal."""
    from .handtasks import hand_at, level0, share_of_amplitude, task_rows, tip_distance_m
    from .motions import posture_body
    task = "finger_tapping"
    pl = FREEZE_PLANS[plan]
    a0, _, anchor = level0(task, side)
    f0 = freeze_rate_hz(plan, side)
    nC = pl["cycles"]
    per, amp = np.full(nC, 1.0 / f0), np.full(nC, a0)
    hasten = []
    for i, after in enumerate(pl["episodes_after"]):
        idx = np.arange(after - HASTEN_CYCLES, after)                 # 0-based cycles before the episode
        w = (np.arange(1, HASTEN_CYCLES + 1)) / HASTEN_CYCLES
        per[idx] = 1.0 / (f0 * (1.0 + HASTEN_GAIN * w))
        amp[idx] = a0 * (1.0 - (1.0 - BREAKDOWN_SHARE) * w)
        nxt = pl["episodes_after"][i + 1] - HASTEN_CYCLES if i + 1 < len(pl["episodes_after"]) else nC
        amp[after:nxt] = a0 * RESUME_SHARE
        hasten.append({"cycles": [int(c) + 1 for c in idx], "rate_hz": [float(1 / p) for p in per[idx]],
                       "amplitude": [float(a) for a in amp[idx]]})
    base = posture_body("sit_finger_tapping", freeze_duration_s(plan), side)
    k0 = int(round(0.5 * base.rate))
    shares = share_of_amplitude(base, task, side, amp, k0)
    tl = share_of_amplitude(base, task, side, np.array([TREMBLE_LO * a0, TREMBLE_HI * a0]), k0)
    t = base.times().numpy()
    s = np.zeros(base.T)
    tc, cycles, episodes = 0.5, [], []
    ep_after = dict(zip(pl["episodes_after"], pl["durations_s"]))
    for c in range(1, nC + 1):
        P = per[c - 1]
        inside = (t >= tc) & (t <= tc + P)
        s[inside] = shares[c - 1] * 0.5 * (1.0 - np.cos(2.0 * math.pi * (t[inside] - tc) / P))
        cycles.append({"cycle": c, "start_s": float(tc), "period_s": float(P), "amplitude": float(amp[c - 1])})
        tc += P
        if c in ep_after:
            D = ep_after[c]
            u = (t - tc) / D
            ins = (u > 0) & (u < 1)
            env = np.sin(np.pi * np.clip(u, 0, 1)) ** 0.5             # rises and falls inside the episode
            tremble = tl[0] + (tl[1] - tl[0]) * 0.5 * (1.0 - np.cos(2.0 * math.pi * TREMBLE_HZ * (t - tc)))
            s[ins] = (env * tremble)[ins]
            episodes.append({"after_cycle": c, "onset_s": float(tc), "offset_s": float(tc + D), "duration_s": D,
                             "tremble_hz": TREMBLE_HZ,
                             "tremble_amplitude": [TREMBLE_LO * a0, TREMBLE_HI * a0]})
            tc += D
    Rh = base.hand_R().clone()
    Rh[:, task_rows(side)] = hand_at(task, side, s)[:, task_rows(side)]
    out = base.with_hands(Rh, task_rows(side))
    Rc = base.hand_R().clone()
    Rc[:, task_rows(side)] = hand_at(task, side, np.zeros(base.T))[:, task_rows(side)]
    closed = float(tip_distance_m(base.with_hands(Rc, task_rows(side)), side, knots=[k0])[0])
    sig = 1000.0 * (tip_distance_m(out, side) - closed)
    val = validity(base, out)
    val["hand_limits"] = H.hand_limit_report(out.hand_R())
    p = {"task": task, "item": "3.4", "side": side, "plan": plan, "n_cycles": nC, "level0_hz": float(f0),
         "amplitude0": float(a0), "unit": "mm", "hasten_cycles": HASTEN_CYCLES, "hasten_gain": HASTEN_GAIN,
         "breakdown_share": BREAKDOWN_SHARE, "resume_share": RESUME_SHARE, "cycles": cycles, "episodes": episodes,
         "hastening": hasten, "end_s": float(tc), "anchor": anchor,
         "remeasured": measure_freezing(sig, out.rate)}
    g = 3 if plan == "3.4_g3" else None
    return out, truth_record("upper_limb_freezing", seed, g, "MDS-UPDRS 3.4" if g else "upper-limb freezing", p,
                             ("mds_updrs", "vercruysse2012", "williams2013", "delval2017", anchor["rate_source"],
                              "module"), val, arrays={"share": s, "signal": np.asarray(sig)})


def measure_freezing(signal, rate, share=0.5, band=(3.0, 8.0)):
    """Upper-limb freezing read back from a finger-tapping signal (mm): the cycles (local maxima above 20 percent of
    the median cycle amplitude... every local maximum, its height over the preceding minimum); the median amplitude
    and period of the cycles above half the largest; an episode is a run of consecutive cycles at or below share of
    that median amplitude spanning at least two median periods (SOURCES['williams2013']); its trembling frequency
    is the dominant frequency in band of the signal over the run; the hastening is the rate of the three cycles
    before the run over the median rate."""
    from .measure import dominant_frequency
    x = np.asarray(signal, dtype=np.float64)
    mx, mn = _extremes(x)
    amps, pk = [], []
    for k in mx:
        prev = mn[mn < k]
        lo = x[prev[-1]] if len(prev) else x[:k].min()
        amps.append(x[k] - lo)
        pk.append(k)
    amps, pk = np.asarray(amps), np.asarray(pk)
    keep = amps > 0.02 * amps.max()
    amps, pk = amps[keep], pk[keep]
    big = amps > 0.5 * amps.max()
    med_a = float(np.median(amps[big]))
    med_p = float(np.median(np.diff(pk[big]))) / rate
    low = amps <= share * med_a
    m = np.diff(np.r_[False, low, False].astype(np.int8))
    eps = []
    for a, e in zip(np.flatnonzero(m == 1), np.flatnonzero(m == -1)):
        t0 = (pk[a - 1] if a > 0 else pk[a]) / rate
        t1 = (pk[e] if e < len(pk) else pk[e - 1]) / rate
        if t1 - t0 < 2 * med_p:
            continue
        core = np.flatnonzero(amps[a:e] <= 0.25 * med_a) + a          # the trembling, without the breakdown
        f = math.nan
        if len(core) >= 2:
            seg = x[int(pk[core[0]]):int(pk[core[-1]]) + 1]
            f = dominant_frequency(seg, rate, band) if len(seg) > 8 else math.nan
        pre = pk[max(0, a - 4):a]
        hz = (len(pre) - 1) / ((pre[-1] - pre[0]) / rate) if len(pre) >= 2 else math.nan
        eps.append({"start_s": float(t0), "end_s": float(t1), "duration_s": float(t1 - t0),
                    "n_low_cycles": int(e - a), "tremble_hz": float(f),
                    "pre_rate_over_median": float(hz * med_p) if hz == hz else math.nan})
    return {"median_amplitude": med_a, "median_period_s": med_p, "episodes": eps, "n_episodes": len(eps)}


# ---------------------------------------------------------------- AP03: the abrupt arising and sitting down en bloc
ABRUPT_RISE_S = 1.2            # [module] the abrupt arising, against the cited 1.89 s (no PSP rise time was read)
SITDOWN_DUR_S = 12.0           # [module] the E22 clips: seated, a rise at the control pace, standing, the sit-down
SITDOWN_SEATED_S = 1.0         # [module] seated before the rise
SITDOWN_STAND_S = 3.0          # [module] standing between the rise and the sit-down
SITDOWN_SEED = SEED + 60       # one background for the two sit-down clips
SEAT_CONTACT_M = 0.02          # [module] the seat contact: the lowest seat vertex this far above its seated height (a
#                                cushion's uncompressed top; the descent stops inside it, the compression not drawn)
BLOC_U = 0.8                   # [module] the en bloc sit-down follows the reversed rise down to this share, then drops
BLOC_IMPACT_M_S = 1.0          # [module] the hips' downward speed at the seat contact of the en bloc drop (stopped
#                                within SEAT_CONTACT_M: 25 m/s2, under the continuity bound)
BLOC_TRUNK_SHARE = 0.3         # [module] the reversed rise of the en bloc sit-down leans with this share of the
#                                template's trunk bout
BLOC_TRUNK_LEAD = 0.6          # [module] the pelvis and the trunk reach their seated lean at this share of the drop
SITDOWN_STRETCH_MAX = 1.5      # [module] the slowest clock the controlled sit-down may take (controlled_descent_u)
SITDOWN_LEAN_STEP_DEG = 1.0    # [module] the grid of the controlled sit-down's hip lean (controlled_descent_u), then
#                                bisection; open-body port, wave 6 follow-up
SITDOWN_FOLD_TURN_DEG = 9.0    # [module] the folded arms of the controlled sit-down turned forward and up about the line
#   of the collars by this times sitdown_lean_profile (sitdown_fold_turn; plan phase E): the hips' added lean brought
#   the right forearm and hand of the fold 13.2 mm into the thighs on the mesh (motions.fold_report over the descent;
#   the page's rigid thigh read the right palm 11.1 mm inside the left thigh); the least on a 1 deg grid at which the
#   descent's fold enters no skin part (fold_report 'entry_ok', the arising's own rule for its folded arms; 8 deg:
#   -2.0 mm); the fold as in the rise needs 13 deg (8.2 mm), past the collars' table (scripts/openbody_phaseE_checks.py
#   sitdown)


def _normal_trunk_share(R, u, share):
    """R (N, 22, 3, 3) of the 'normal' template at shares u, its trunk bout turned to share (scalar) of the solved one
    (as _with_trunk_share for the 'fail' template)."""
    from .motions import TRUNK_JOINTS, trunk_profile, trunk_split
    _, ni = arise_template("normal")
    split, _ = trunk_split()
    d = (float(share) - 1.0) * ni["trunk"]["amplitude_deg"] * trunk_profile(np.asarray(u))
    R = R.clone()
    for j, s in zip(TRUNK_JOINTS, split):
        R[:, j] = _rot("x", s * d) @ R[:, j]
    return R


@functools.lru_cache(maxsize=2)
def abrupt_rise_u(rate=100.0):
    """((N,) shares of the 'normal' template at the knots of the abrupt rise, info): the template read on the faster
    clock of ABRUPT_RISE_S (uniform), its trunk bout lowered to the smallest share (bisection) whose extrapolated
    centre of mass stays MOS_MIN_M ahead of the heels from lift-off (motions.balance): the faster rise carries the
    centre of mass forward with less lean; the anterior margin (behind the toes) is reported and must stay
    positive."""
    from .motions import LIFTOFF_SHARE, MOS_MIN_M, balance
    _, ni = arise_template("normal")
    n = int(round(ABRUPT_RISE_S * rate))
    u = (np.arange(n) + 1) / n
    uu = np.r_[np.zeros(int(round(0.5 * rate))), u, np.ones(int(round(0.5 * rate)))]
    R0, tr, _ = _template_pose("normal", uu)
    k_lift = int(np.argmax(uu > LIFTOFF_SHARE))

    def margins(share):
        b = H.HandBody(matrix_to_rot6d(_normal_trunk_share(R0, uu, share)), tr,
                       torch.zeros(len(uu), 2, dtype=torch.bool), rate=rate)
        bl = balance(b, k_lift)
        J = b.joints().numpy()
        tv = J[:, 12] - J[:, 0]
        lean = np.degrees(np.arctan2(tv[:, 2], tv[:, 1]))
        return (float(bl["posterior"][k_lift:].min()), float(bl["anterior"][k_lift:].min()),
                float(lean.max() - lean[0]), float(np.gradient(J[:, 0, 1], 1.0 / rate).max()))
    lo, hi = 0.0, 1.0
    if margins(hi)[0] < MOS_MIN_M:
        raise RuntimeError("the abrupt rise falls back even with the full trunk bout")
    for _ in range(30):
        m = 0.5 * (lo + hi)
        lo, hi = (lo, m) if margins(m)[0] >= MOS_MIN_M else (m, hi)
    post, ant, lean, vy = margins(hi)
    if ant <= 0.0:
        raise RuntimeError("the abrupt rise carries the extrapolated centre of mass past the toes")
    return u, {"rise_s": ABRUPT_RISE_S, "template_rise_s": ni["rise_s"], "trunk_share": hi,
               "trunk_bout_deg": hi * ni["trunk"]["amplitude_deg"], "posterior_margin_min_m": post,
               "anterior_margin_min_m": ant, "trunk_lean_peak_deg": lean, "pelvis_up_speed_max_m_s": vy,
               "source": "module (ABRUPT_RISE_S unsourced; the trunk share solved for balance)"}


def abrupt_arise(duration_s=ARISE_DUR_S, seed=SEED):
    """(HandBody, truth) of the abrupt arising of AP03 (module docstring): seated 1 s, the rise of abrupt_rise_u with
    the arms folded, standing; the seed and the background layer of park_E7_arise_g0 (SERIES['arise']), so that the
    clip and its level 0 differ by the rise only."""
    from .motions import MOS_MIN_M, arise_validity, balance
    plan = [("sit", 1.0), ("rise", "abrupt", "folded")]
    b, info = arise_plan_body(plan, duration_s, seed=seed)
    val = arise_validity(b)
    val["hand_limits"] = H.hand_limit_report(b.hand_R())
    val["continuity"] = continuity_report(b)
    rise = [a for a in info["attempts"] if a["kind"] == "rises"][0]
    lift, end = rise["liftoff_knot"], int(round(rise["end_s"] * b.rate))
    ant = float(balance(b, lift)["anterior"][lift:end].min())        # re-measured on the clip (background included)
    val["balance"] = {"rise_ok": rise["xcom_margin_min_m"] >= MOS_MIN_M - 1e-3, "mos_min_m": MOS_MIN_M,
                      "xcom_margin_min_m": rise["xcom_margin_min_m"], "anterior_margin_min_m": ant,
                      "anterior_margin_min_m_solve": rise["abrupt"]["anterior_margin_min_m"]}
    val["balance"]["ok"] = val["balance"]["rise_ok"] and val["balance"]["anterior_margin_min_m"] > 0.0
    J = b.joints().numpy()
    info.pop("arm_code")
    m = measure_arise(b, info)
    m["pelvis_up_speed_max_m_s"] = float(np.gradient(J[:, 0, 1], 1.0 / b.rate).max())
    p = {"variant": "abrupt", "plan": [list(x) for x in plan], "duration_s": float(duration_s), "seat_height_m": 0.43,
         "control_rise_s": 1.89, "abrupt_rise_s": ABRUPT_RISE_S, "skate_p95_m_s": val["skate_p95_m_s"],
         "limits_outside": val["limits_outside"], "remeasured": m, **info}
    srcs = ["inkster2004", "li2021", "hof2005", "deleva1996", "module"]
    return b, truth_record("arise_abrupt", seed, "present", "PSPRS arising (wording not verified; no grade)", p, srcs,
                           val)


def _hermite(s, p0, v0, p1, v1, T):
    """Cubic Hermite from (p0, v0) to (p1, v1) over T at shares s (N,) (velocities per second)."""
    s = np.clip(np.asarray(s, dtype=np.float64), 0.0, 1.0)
    h00, h10 = 2 * s ** 3 - 3 * s ** 2 + 1, s ** 3 - 2 * s ** 2 + s
    h01, h11 = -2 * s ** 3 + 3 * s ** 2, s ** 3 - s ** 2
    return h00 * p0 + h10 * T * v0 + h01 * p1 + h11 * T * v1


def _bloc_descent(rate=100.0):
    """(R, tr, Rh, info) of the en bloc sit-down from standing (module docstring): the 'normal' template read backward
    at its pace from standing down to BLOC_U, then the drop: the hip midpoint falls with a constant acceleration from
    its velocity at BLOC_U to BLOC_IMPACT_M_S downward at SEAT_CONTACT_M above its seated height, then stops at the
    seat with a constant deceleration; it goes back to its seated position by a cubic Hermite (the velocity at BLOC_U
    kept, zero at the seat); the pelvis and the trunk turn about x to their seated lean by a cubic Hermite over
    BLOC_TRUNK_LEAD of the drop (the trunk never leans forward: en bloc); the legs re-solved to the planted ankles
    (place_pelvis), the folded arms riding on the trunk."""
    from .motions import TRUNK_JOINTS
    _, ni = arise_template("normal")
    rs = ni["rise_s"]
    rest = rest_joints()
    nA = int(round((1.0 - BLOC_U) * rs * rate))
    uA = 1.0 - (1.0 - BLOC_U) * (np.arange(nA) + 1) / nA
    RA, trA, RhA = _template_pose("normal", uA)
    RA = _normal_trunk_share(RA, uA, BLOC_TRUNK_SHARE)
    du = 1.0 / (rs * rate)
    uq = np.array([BLOC_U + du, BLOC_U, BLOC_U - du])
    Rq, trq, _ = _template_pose("normal", uq)
    Rq = _normal_trunk_share(Rq, uq, BLOC_TRUNK_SHARE)
    Jq = fk(Rq, trq, rest)[0]
    Mq = (0.5 * (Jq[:, 1] + Jq[:, 2])).numpy()
    M_s, v_s = Mq[1], (Mq[2] - Mq[0]) * rate / 2.0
    joints = (0,) + tuple(TRUNK_JOINTS)
    a_s = {j: float(_x_angle(Rq[1:2, j])[0]) for j in joints}
    w_s = {j: float(_x_angle(Rq[2:3, j])[0] - _x_angle(Rq[0:1, j])[0]) * rate / 2.0 for j in joints}
    Re, tre, Rhe = _template_pose("normal", np.zeros(1))
    Je = fk(Re, tre, rest)[0]
    M_e = (0.5 * (Je[0, 1] + Je[0, 2])).numpy()
    a_e = {j: float(_x_angle(Re[:, j])[0]) for j in joints}
    c, V = SEAT_CONTACT_M, BLOC_IMPACT_M_S
    drop = float(M_s[1] - M_e[1])
    vy0 = max(0.0, float(-v_s[1]))
    T1 = 2.0 * (drop - c) / (vy0 + V)
    T2 = 2.0 * c / V
    Tt = T1 + T2
    nB = int(math.ceil(Tt * rate - 1e-9))
    tB = np.minimum((np.arange(nB) + 1) / rate, Tt)
    a1, a2 = (V - vy0) / T1, V / T2
    tb = tB - T1
    y = np.where(tB <= T1, M_s[1] - vy0 * tB - 0.5 * a1 * tB ** 2, M_e[1] + c - V * tb + 0.5 * a2 * tb ** 2)
    z = _hermite(tB / Tt, M_s[2], v_s[2], M_e[2], 0.0, Tt)
    M = torch.as_tensor(np.stack([np.full(nB, 0.5 * float(Je[0, 1, 0] + Je[0, 2, 0])), y, z], -1))
    Ta = BLOC_TRUNK_LEAD * Tt
    ang = {j: _hermite(tB / Ta, a_s[j], w_s[j], a_e[j], 0.0, Ta) for j in joints}
    G0 = _rot("x", ang[0])
    Rd = Re.expand(nB, -1, -1, -1).clone()
    for j in TRUNK_JOINTS:
        Rd[:, j] = _rot("x", ang[j])
    mid = torch.as_tensor(0.5 * (rest[1] + rest[2]) - rest[0])
    J0 = M - torch.einsum("tij,j->ti", G0, mid)
    ank = torch.stack([Je[0, 7], Je[0, 8]], 0)[None].expand(nB, 2, 3).clone()
    u_ref = BLOC_U * np.clip((y - M_e[1]) / drop, 0.0, 1.0)
    Rr, trr, _ = _template_pose("normal", u_ref)
    RB, trB = place_pelvis(Rd, trr.clone(), J0, G0, ank, Rr, trr, keep_reach=False)
    RhB = Rhe.expand(nB, -1, -1, -1).clone()
    info = {"kind": "en bloc", "controlled_to_u": BLOC_U, "controlled_s": nA / rate,
            "controlled_trunk_share": BLOC_TRUNK_SHARE, "drop_s": float(Tt),
            "drop_height_m": drop, "start_down_speed_m_s": vy0, "impact_speed_m_s": V, "contact_height_m": c,
            "fall_accel_m_s2": float(a1), "stop_decel_m_s2": float(a2), "stop_s": float(T2),
            "trunk_lead": BLOC_TRUNK_LEAD, "descent_s": nA / rate + float(Tt)}
    return torch.cat([RA, RB]), torch.cat([trA, trB]), torch.cat([RhA, RhB]), info


def sitdown_lean_profile(u):
    """(N,) the share of the controlled sit-down's hip lean at the template shares u of the descent (read from 1 to 0):
    a minimum-jerk rise from the standing knot to the template's TRUNK_RETURN_SHARE (where, read backward, its own trunk
    bout comes to its hold), held to its ANKLE_PEAK_SHARE (the hips at the seat-contact height), a minimum-jerk return
    to 0 at the seated knot (module docstring, section 5)."""
    from .motions import ANKLE_PEAK_SHARE, TRUNK_RETURN_SHARE
    u = np.asarray(u, dtype=np.float64)
    return np.where(u >= ANKLE_PEAK_SHARE, _mj((1.0 - u) / (1.0 - TRUNK_RETURN_SHARE)), _mj(u / ANKLE_PEAK_SHARE))


def sitdown_lean(R, tr, u, deg):
    """(R, tr): the poses R (N, 22, 3, 3), tr (N, 3) of the descent at template shares u (N,) with the pelvis turned
    forward by deg times sitdown_lean_profile(u) about the line of the hip joints (the hips stay where they are, the
    trunk rides on the pelvis), the hips' rows turned back so that the thighs and the legs keep their place."""
    if not deg:
        return R, tr
    rest = rest_joints()
    R = R.clone()
    J, _ = fk(R, tr, rest)
    Mh = 0.5 * (J[:, 1] + J[:, 2])
    Rx = _rot("x", float(deg) * sitdown_lean_profile(u))
    G0 = R[:, 0].clone()
    R[:, 0] = Rx @ G0
    tr = Mh + torch.einsum("tij,tj->ti", Rx, J[:, 0] - Mh) - rest[0]
    for h in (1, 2):
        R[:, h] = G0.transpose(-1, -2) @ Rx.transpose(-1, -2) @ G0 @ R[:, h]
    return R, tr


def sitdown_fold_turn(R, u, deg=SITDOWN_FOLD_TURN_DEG):
    """(N, 22, 3, 3): the poses R of the controlled descent at template shares u with both collars (13, 14) turned
    forward and up by deg times sitdown_lean_profile(u) about their parent's x axis: one line through both collar
    joints, so the folded arms turn as one rigid block (the hands keep their places on the opposite arms) and rise off
    the thighs that the hips' added lean brings under them (SITDOWN_FOLD_TURN_DEG)."""
    if not deg:
        return R
    R = R.clone()
    Rx = _rot("x", -float(deg) * sitdown_lean_profile(u))
    for j in (13, 14):
        R[:, j] = Rx @ R[:, j]
    return R


@functools.lru_cache(maxsize=2)
def controlled_descent_u(rate=100.0):
    """((N,) shares of the 'normal' template at the knots of the controlled sit-down, from standing (the first knot
    below 1) to seated (the last at 0), info): the template read backward on a clock slowed by the smallest stretch
    (bisection, 1 to SITDOWN_STRETCH_MAX) whose extrapolated centre of mass stays MOS_MIN_M ahead of the heels from
    the start of the descent to the seat contact (motions.balance; the contact as measure_sitdown reads it, the
    lowest seat vertex within SEAT_CONTACT_M of its seated height). Read backward at its own pace, the rise's
    velocities turn backward: the hips' first move back carries the extrapolated centre of mass toward the heels at
    the start of the descent (the rise is solved for balance forward only, motions.arise). Open-body port, wave 6
    follow-up: on the anatomical table the template's trunk bout (motions D81) leaves the centre of mass behind the
    heels at the seat contact on every clock up to SITDOWN_STRETCH_MAX; the descent leans further at the hips
    (sitdown_lean), by the smallest amplitude (SITDOWN_LEAN_STEP_DEG grid, then bisection) at which the slowest clock
    keeps the XcoM ahead, and the stretch is then the smallest at that lean. info adds 'hip_lean_deg'."""
    from .motions import MOS_MIN_M, _seat_vertices, balance
    _, ni = arise_template("normal")
    nr = int(round(ni["rise_s"] * rate))
    lead, tail = int(round(0.5 * rate)), int(round(0.5 * rate))
    seat_v = _seat_vertices()

    def shares(f):
        n = int(round(f * nr))
        return 1.0 - (np.arange(n) + 1) / n

    def margin(f, a=0.0):
        ud = shares(f)
        uu = np.r_[np.ones(lead), ud, np.zeros(tail)]
        R, tr, _ = _template_pose("normal", uu)
        R, tr = sitdown_lean(R, tr, uu, a)
        b = H.HandBody(matrix_to_rot6d(R), tr, torch.zeros(len(uu), 2, dtype=torch.bool), rate=rate)
        hs = H.body_points(b, seat_v)[..., 1].min(1)
        hs = hs - hs[-1]
        k_c = lead + int(np.argmax(hs[lead:] <= SEAT_CONTACT_M))
        return float(balance(b, lead)["posterior"][lead:k_c + 1].min())
    from .body import joint_limits
    lim = joint_limits()
    a_max = float(-min(lim.lo[1][0], lim.lo[2][0]))      # at most the hips' flexion range (the table checks the rest)
    lean = 0.0
    if margin(SITDOWN_STRETCH_MAX) < MOS_MIN_M:
        prev, found = 0.0, None
        for a in np.arange(SITDOWN_LEAN_STEP_DEG, a_max + 1e-9, SITDOWN_LEAN_STEP_DEG):
            if margin(SITDOWN_STRETCH_MAX, float(a)) >= MOS_MIN_M:
                found = float(a)
                break
            prev = float(a)
        if found is None:
            raise RuntimeError("the controlled sit-down falls back even on the slowest clock")
        lo_a, hi_a = prev, found
        for _ in range(20):
            m = 0.5 * (lo_a + hi_a)
            lo_a, hi_a = (lo_a, m) if margin(SITDOWN_STRETCH_MAX, m) >= MOS_MIN_M else (m, hi_a)
        lean = hi_a
    lo, hi = 1.0, SITDOWN_STRETCH_MAX
    if margin(lo, lean) >= MOS_MIN_M:
        hi = lo
    else:
        for _ in range(20):
            m = 0.5 * (lo + hi)
            lo, hi = (lo, m) if margin(m, lean) >= MOS_MIN_M else (m, hi)
    ud = shares(hi)
    return ud, {"stretch": hi, "descent_s": len(ud) / rate, "template_rise_s": ni["rise_s"], "hip_lean_deg": lean,
                "xcom_margin_min_m": margin(hi, lean), "mos_min_m": MOS_MIN_M,
                "rule": "the 'normal' template read backward on a clock slowed by the smallest stretch that keeps the "
                        "extrapolated centre of mass MOS_MIN_M ahead of the heels to the seat contact (module "
                        "reading: no stand-to-sit source)"}


def sitdown_body(kind, duration_s=SITDOWN_DUR_S, rate=100.0, seed=SITDOWN_SEED):
    """(HandBody, info) of an E22 clip (module docstring): SITDOWN_SEATED_S seated, a rise on the 'normal' template at
    the control pace, SITDOWN_STAND_S standing (the sway weighted in and out), the sit-down, seated to the end; kind
    'controlled' (the 'normal' template read backward on the clock of controlled_descent_u: the trunk leans forward
    as the hips go back, the descent slows to the seat) or 'en_bloc' (_bloc_descent); the arms folded throughout; the
    background layer of the series SERIES['sitdown'] (breathing, the head's drift, the seated trunk's drift, the
    standing sway: the same on both sit-down clips; seed is recorded, nothing is drawn from it)."""
    T = int(round(duration_s * rate)) + 1
    _, ni = arise_template("normal")
    rs = ni["rise_s"]
    n0, nr, ns = int(round(SITDOWN_SEATED_S * rate)), int(round(rs * rate)), int(round(SITDOWN_STAND_S * rate))
    k_rise, k_stand, k_desc = n0, n0 + nr, n0 + nr + ns
    u = np.zeros(k_desc)
    u[k_rise:k_stand] = (np.arange(nr) + 1) / nr
    u[k_stand:] = 1.0
    R = torch.zeros(T, 22, 3, 3, dtype=F64)
    tr = torch.zeros(T, 3, dtype=F64)
    Rh = torch.zeros(T, H.N_HAND, 3, 3, dtype=F64)
    R[:k_desc], tr[:k_desc], Rh[:k_desc] = _template_pose("normal", u)
    if kind == "controlled":
        ud, cinfo = controlled_descent_u(rate)
        Rd, trd, Rhd = _template_pose("normal", ud)
        Rd, trd = sitdown_lean(Rd, trd, ud, cinfo["hip_lean_deg"])
        Rd = sitdown_fold_turn(Rd, ud)
        dinfo = {"kind": "controlled", **cinfo, "fold_turn_deg": SITDOWN_FOLD_TURN_DEG}
    elif kind == "en_bloc":
        Rd, trd, Rhd, dinfo = _bloc_descent(rate)
    else:
        raise ValueError(f"sit-down {kind!r}")
    k_end = k_desc + len(Rd)
    if k_end + int(rate) > T:
        raise ValueError("the sit-down does not fit in the clip")
    R[k_desc:k_end], tr[k_desc:k_end], Rh[k_desc:k_end] = Rd, trd, Rhd
    R[k_end:], tr[k_end:], Rh[k_end:] = _template_pose("normal", np.zeros(1))
    seated = np.zeros(T, dtype=bool)
    seated[:k_rise] = True
    seated[k_end:] = True
    k = np.arange(T)
    nramp = int(round(0.5 * rate))
    stand_w = np.where((k >= k_stand) & (k < k_desc), _mj((k - k_stand + 1) / nramp) *
                       (1.0 - _mj((k - (k_desc - nramp)) / nramp)), 0.0)
    bg = background_signals(SERIES["sitdown"], T, 1.0, stand_w, rate)
    w_sit = _inside_ramp(seated, int(0.3 * rate))
    _trunk_layer(R, bg["breath"], bg["yaw"], bg["pitch"], w_sit * bg["lean"], w_sit * bg["lean_ml"])
    st = np.flatnonzero(stand_w > 0)
    R[st], tr[st] = stand_sway(R[st], tr[st], bg["sway_z"][st], bg["sway_x"][st])
    b = H.HandBody(matrix_to_rot6d(R), tr, torch.zeros(T, 2, dtype=torch.bool), rate=rate, hand6d=matrix_to_rot6d(Rh))
    b = H.HandBody(b.rot6d, b.transl, contact_labels(b), b.betas, b.gender, b.rate, b.hand6d)
    info = {"kind": kind, "phases": [{"phase": "sit", "start_s": 0.0, "duration_s": k_rise / rate},
                                     {"phase": "rise (folded)", "start_s": k_rise / rate, "duration_s": nr / rate},
                                     {"phase": "standing", "start_s": k_stand / rate, "duration_s": ns / rate},
                                     {"phase": f"sit-down ({kind.replace('_', ' ')})", "start_s": k_desc / rate,
                                      "duration_s": (k_end - k_desc) / rate},
                                     {"phase": "sit", "start_s": k_end / rate, "duration_s": (T - k_end) / rate}],
            "descent": dinfo, "descent_knots": [k_desc, k_end], "arms": "folded across the chest",
            "background": background_record(SERIES["sitdown"], T, 1.0, rate,
                                            dict(TRANSFER_PARTS, sway="standing between the rise and the sit-down, "
                                                 "weighted in and out over 0.5 s (the feet planted)",
                                                 collars="none (the arms folded)", fingers="none (the arms folded)"))}
    return b, info


def measure_sitdown(body, k_from=None, contact_m=SEAT_CONTACT_M, onset_m=0.01):
    """The sit-down read back from a body (module docstring): the lowest seat vertex (motions._seat_vertices) above its
    height at the last knot; after the highest pelvis knot (or from k_from, the start of the sit-down), the seat
    contact is the first knot where it comes within contact_m and the seated knot the first after it within 1 mm;
    the descent onset the last knot before the contact with the pelvis within onset_m of its standing height (the
    median over the knots within 2 cm of the highest); the descent time from the onset to the seated knot (and to the
    contact); the pelvis's downward speed at the contact (central differences); the extrapolated centre of mass
    behind the heels (motions.balance, base of support at the onset): the first such knot from the start to the
    contact, its time after the onset (negative: before it), the pelvis's height above its seated height there and
    the share of the pelvis's descent still to come (the uncontrolled part of the descent; None when the centre of
    mass stays over the feet to the contact); the trunk lean peak (the pelvis-to-neck line, forward positive) over the
    descent above its value at the onset; the seat clearance (the lowest seat vertex's least height above its seated
    height over the clip: negative means inside the seat)."""
    from .motions import _seat_vertices, balance
    J = body.joints().numpy()
    y = J[:, 0, 1]
    seat = H.body_points(body, _seat_vertices())[..., 1].min(1)
    hs = seat - seat[-1]
    k_top = int(np.argmax(y)) if k_from is None else int(k_from)
    stand = float(np.median(y[y > y.max() - 0.02]))
    after = np.flatnonzero(hs[k_top:] <= contact_m)
    if not len(after):
        return {"sat_down": False}
    k_c = k_top + int(after[0])
    k_o = k_c
    while k_o > 0 and y[k_o] < stand - onset_m:
        k_o -= 1
    vy = np.gradient(y, 1.0 / body.rate)
    k_s = k_c + int(np.argmax(hs[k_c:] <= 1e-3)) if (hs[k_c:] <= 1e-3).any() else len(y) - 1
    post = balance(body, k_o)["posterior"]
    k0 = min(k_top, k_o)
    behind = np.flatnonzero(post[k0:k_c + 1] < 0.0)
    k_b = k0 + int(behind[0]) if len(behind) else None
    tv = J[:, 12] - J[:, 0]
    lean = np.degrees(np.arctan2(tv[:, 2], tv[:, 1]))
    y_seat = float(y[k_s])
    return {"sat_down": True, "onset_s": k_o / body.rate, "contact_s": k_c / body.rate,
            "seated_s": k_s / body.rate, "descent_s": (k_s - k_o) / body.rate,
            "to_contact_s": (k_c - k_o) / body.rate, "impact_speed_m_s": float(-vy[k_c]),
            "down_speed_max_m_s": float(-vy[k_o:k_s + 1].min()),
            "xcom_behind_heels_s": k_b / body.rate if k_b is not None else None,
            "behind_heels_after_onset_s": (k_b - k_o) / body.rate if k_b is not None else None,
            "behind_heels_pelvis_above_seated_m": float(y[k_b] - y_seat) if k_b is not None else None,
            "behind_heels_descent_share_left": (float((y[k_b] - y_seat) / (stand - y_seat)) if k_b is not None
                                                else None),
            "trunk_lean_peak_deg": float(lean[k_o:k_c + 1].max() - lean[k_o]),
            "seat_clearance_min_m": float(hs.min()), "standing_pelvis_m": stand, "seated_pelvis_m": y_seat}


def sitdown(kind, duration_s=SITDOWN_DUR_S, seed=SITDOWN_SEED):
    """(HandBody, truth) of an E22 sit-down clip (sitdown_body), with its checks: the joint-limit table, the hand
    limits, continuity, the feet planted (skate, ankle displacement), the seat clearance (no seat vertex inside the
    seat), the folded arms over the descent ('fold': no skin part inside another, the thighs among them; plan phase E)
    and the read-back (measure_sitdown)."""
    from .motions import arise_validity
    b, info = sitdown_body(kind, duration_s, seed=seed)
    val = arise_validity(b)
    val["hand_limits"] = H.hand_limit_report(b.hand_R())
    val["continuity"] = continuity_report(b)
    J = b.joints().numpy()
    val["ankle_move_max_m"] = float(np.abs(J[:, [7, 8]] - J[:1, [7, 8]]).max())
    m = measure_sitdown(b, k_from=info["descent_knots"][0])
    val["seat_clearance_min_m"] = m["seat_clearance_min_m"]
    val["seat_ok"] = m["seat_clearance_min_m"] > -1e-3
    from .motions import FOLD_CHECK_STEP, fold_report     # plan phase E: the folded arms over the descent, the
    kd, ke = info["descent_knots"]                        # arising's rule (motions.arise_checks_of 'fold')
    fr = fold_report(b, np.unique(np.r_[np.arange(kd, ke, FOLD_CHECK_STEP), ke - 1]))
    val["fold"] = dict(fr, ok=fr["entry_ok"], margins_ok=fr["ok"], span="the descent")
    p = {"variant": kind, "duration_s": float(duration_s), "seat_height_m": 0.43, "seat_contact_m": SEAT_CONTACT_M,
         "remeasured": m, **info}
    if kind == "en_bloc":
        p["unsourced"] = ("BLOC_U, BLOC_IMPACT_M_S, BLOC_TRUNK_LEAD and SEAT_CONTACT_M are module values; no "
                          "stand-to-sit kinematics of PSP was read; the seat's cushion is not simulated")
    srcs = ["inkster2004", "li2021", "hof2005", "deleva1996", "module"]
    g = 0 if kind == "controlled" else "present"
    return b, truth_record("sitdown", seed, g, "PSPRS sitting down (wording not verified; no grade)", p, srcs, val)


# ---------------------------------------------------------------- AP01: the applause sign
CLAP_COUNTS = {0: 3, 1: 4, 2: 7, 3: 14}   # claps per score: 3 (negative), 4, 5 to 10, more than 10 (the graded
#                                scoring of the catalogue row, not traced to its source; positive above 3, Dubois2005)
CLAP_HZ = 3.0                  # [module] claps per second ('as quickly as possible'; no rate was read)
CLAP_PERIOD_CV = 0.05          # [module] cycle-to-cycle variability of the period (AR(1), SEQ_AR)
CLAP_AMP_CV = 0.08             # [module] of the opening
CLAP_OPEN_M = 0.20             # [module] the palms' distance at the top of a clap
CLAP_GAP_M = 0.002             # [module] the palms' distance at contact (posed: the skins do not meet)
CLAP_DWELL_S = 0.03            # [module] the palms together at each clap
_C5 = 3.0 * (0.5 * math.pi - 2.0)                      # the first close's quintic (clap_gap): 1 + a s^3 + b s^4 + c s^5,
CLAP_FIRST_CLOSE = (-1.0 - (1.0 - 7.0 * _C5 / 3.0) - _C5, 1.0 - 7.0 * _C5 / 3.0, _C5)   # 0 at 1 with the cosine's
#   slope and no curvature there, flat to its second derivative at 0 (open-body port, wave 6 follow-up: the cosine's
#   close began with its full acceleration from the held opening, a step of the open body's elbows at the raise's last
#   knot past the 6 m/s2 of the raise's test; scripts/openbody_wave6_checks.py, applause)
CLAP_UP_DEG = 40.0             # [module] the fingers point forward and this much up, the palms facing each other
CLAP_CENTRE = (0.0, 0.0, 0.30)   # [module] the point between the palms, from spine3, in the seated reference pose
#                                (m: left, up, forward)
CLAP_POLE = (1.0, 1.0)         # [module] the elbows point out and back (arm_ik's pole; the centre, the
#                                finger angle and the pole chosen on a grid for the largest margin inside the table)
CLAP_REST_S, CLAP_RAISE_S = 1.0, 0.8       # [module] seated rest, then the hands raised to the start (joint blend)
CLAP_HOLD_S, CLAP_PART_S, CLAP_LOWER_S = 0.4, 0.3, 0.9   # [module] after the last clap: held together, parted,
#                                lowered back to the chair arms (joint blend)
CLAP_DUR_S = 10.0              # every applause clip (the longest run fits)
CLAP_SEED = SEED + 70
CLAP_TOUCH_M = 0.005           # read-back: a clap is a run of knots with the palms closer than this
CLAP_PATH_PASSES = 8           # [module] _clear_path's passes at most (each stops once the path is met within 0.2
#                                mm; open-body port, wave 6 follow-up: 4 left the open body's raise 1.9 mm under it)
CLAP_SMOOTH_PASSES = 2         # [module] _clear_path's passes on the smoothed turn's shortfall (wave 6 follow-up)
CLAP_RAISE_SMOOTH_S = 0.05     # [module] the outward turn of the raise and the lowering (_clear_path) smoothed
CLAP_APPROACH_M = 0.01         # [module] read-back: an approach of the hands is a minimum of the palms' distance this
#                                prominent
CLAP_REST_HAND = 0.5           # [module] the hands at rest beyond the chair arms, before the raise and after the
#                                lowering: RELAXED_HAND at this share with the fingers' drift (as the motor state rest)


CLAP_HAND = {"mcp_deg": 12.0, "pip_deg": 5.0, "thumb_flex_deg": -15.0, "thumb_palmar_deg": -25.0}   # [module] the
#   clapping hand: the fingers together (RELAXED_HAND's knuckle abduction), slightly flexed at the knuckles and the
#   middle joints, the thumb drawn back into the palm's plane (the rest thumb stands 3 cm out of it, palmar)


def clap_hand():
    """(30, 3, 3) the clapping hands (CLAP_HAND), both sides."""
    pose = {}
    for side in H.SIDES:
        for f, vals in RELAXED_HAND.items():
            if f == "thumb":
                pose[(side, f, 0)] = (CLAP_HAND["thumb_flex_deg"], CLAP_HAND["thumb_palmar_deg"])
                continue
            v = vals[0]
            pose[(side, f, 0)] = (CLAP_HAND["mcp_deg"], 0.0 if np.isscalar(v) else v[1])
            pose[(side, f, 1)] = (CLAP_HAND["pip_deg"], 0.0)
    return H.hand_rotations(pose)


@functools.lru_cache(maxsize=2)
def _hand_rest_axes(side, cell_m=(0.015, 0.012)):
    """(finger direction, palmar normal) (3,) of a clapping hand (clap_hand) in the rest frame: the palmar surface's
    plane, a least-squares plane through the most palmar point of each cell of a grid along and across the hand (in
    the frame of the wrist-to-middle-knuckle direction and hands.hand_frame's palmar normal), its normal toward the
    palm and the finger direction in it; two such hands meet along their palmar surfaces, not at one corner."""
    R = torch.eye(3, dtype=F64).expand(1, 22, 3, 3).clone()
    tr = torch.zeros(1, 3, dtype=F64)
    b = _body2(R, tr, clap_hand()[None])
    J = b.joints().numpy()[0]
    w = J[H.WRIST[side]]
    P = H.body_points(b, _hand_ids(side), knots=[0])[0] - w
    J52, _, _ = H._rest()
    f0 = J52[H.hand_joint(side, "middle", 0)] - J52[H.WRIST[side]]
    f0 = f0 / np.linalg.norm(f0)
    n0 = np.asarray(H.hand_frame()[side]["palmar"], dtype=np.float64)
    n0 = n0 - (n0 @ f0) * f0
    n0 = n0 / np.linalg.norm(n0)
    c0 = np.cross(n0, f0)
    a, c, h = P @ f0, P @ c0, P @ n0
    ia = np.floor((a - a.min()) / cell_m[0]).astype(int)
    ic = np.floor((c - c.min()) / cell_m[1]).astype(int)
    rows = []
    for key in set(zip(ia, ic)):
        sel = (ia == key[0]) & (ic == key[1])
        if sel.sum() >= 3:
            k = np.flatnonzero(sel)[np.argmax(h[sel])]
            rows.append((a[k], c[k], h[k]))
    X = np.asarray(rows)
    coef, *_ = np.linalg.lstsq(np.c_[np.ones(len(X)), X[:, 0], X[:, 1]], X[:, 2], rcond=None)
    n = n0 - coef[1] * f0 - coef[2] * c0
    n = n / np.linalg.norm(n)
    f = f0 - (f0 @ n) * n
    return f / np.linalg.norm(f), n


def clap_hand_rotation(side):
    """(3, 3) the global rotation of a clapping hand (module values): the fingers forward and CLAP_UP_DEG up, the
    palm facing the midline."""
    f0, n0 = _hand_rest_axes(side)
    a = math.radians(CLAP_UP_DEG)
    f1 = np.array([0.0, math.sin(a), math.cos(a)])
    n1 = np.array([-1.0 if side == "left" else 1.0, 0.0, 0.0])

    def frame(f, n):
        n = n - (n @ f) * f
        n = n / np.linalg.norm(n)
        return np.stack([f, n, np.cross(f, n)], 1)
    return torch.as_tensor(frame(f1, n1) @ frame(f0, n0).T, dtype=F64)


def clap_gap(n_claps, duration_s=CLAP_DUR_S, rate=100.0, seed=CLAP_SEED):
    """((T,) the palms' distance m, (T,) bool IK knots, info) of an applause run (module docstring): rest, the raise
    (the distance CLAP_OPEN_M from its end), n_claps cycles at CLAP_HZ (each a close, the first eased in from the
    held opening by CLAP_FIRST_CLOSE, a dwell at CLAP_GAP_M, an
    opening to the next top; the period and the opening jittered by AR(1) noise of CLAP_PERIOD_CV and CLAP_AMP_CV),
    the hold, the parting to half the opening, then the lowering."""
    rng = as_rng(seed + n_claps)
    T = int(round(duration_s * rate)) + 1
    t = np.arange(T) / rate
    P0 = 1.0 / CLAP_HZ
    per = P0 * (1.0 + CLAP_PERIOD_CV * _ar1(rng, n_claps))
    top = CLAP_OPEN_M * (1.0 + CLAP_AMP_CV * _ar1(rng, n_claps + 1))
    top[0] = CLAP_OPEN_M
    g = np.full(T, np.nan)
    t0 = CLAP_REST_S + CLAP_RAISE_S
    g[(t >= t0 - 1e-9) & (t < t0 + 1e-9)] = top[0]
    tc, claps = t0, []
    for i in range(n_claps):
        half = 0.5 * (per[i] - CLAP_DWELL_S)
        a, b, c = tc, tc + half, tc + half + CLAP_DWELL_S
        last = i == n_claps - 1
        e = c + (CLAP_HOLD_S if last else half)
        s = (t - a) / half
        m = (t > a) & (t <= b)
        if i == 0:                                       # from the held opening: eased in (CLAP_FIRST_CLOSE)
            q = s[m]
            fa, fb, fc = CLAP_FIRST_CLOSE
            g[m] = CLAP_GAP_M + (top[i] - CLAP_GAP_M) * (1.0 + q ** 3 * (fa + q * (fb + q * fc)))
        else:
            g[m] = CLAP_GAP_M + (top[i] - CLAP_GAP_M) * np.cos(0.5 * math.pi * s[m])
        g[(t > b) & (t <= c)] = CLAP_GAP_M
        if last:
            g[(t > c) & (t <= e)] = CLAP_GAP_M
        else:
            s2 = (t - c) / half
            m2 = (t > c) & (t <= e)
            g[m2] = CLAP_GAP_M + (top[i + 1] - CLAP_GAP_M) * np.sin(0.5 * math.pi * s2[m2])
        claps.append({"clap": i + 1, "contact_s": float(b), "release_s": float(c), "period_s": float(per[i]),
                      "opening_m": float(top[i])})
        tc = e
    part_end = tc + CLAP_PART_S
    m = (t > tc) & (t <= part_end)
    g[m] = CLAP_GAP_M + (0.5 * CLAP_OPEN_M - CLAP_GAP_M) * _mj((t[m] - tc) / CLAP_PART_S)
    ik = ~np.isnan(g)
    lower_end = part_end + CLAP_LOWER_S
    if lower_end > duration_s - 0.5:
        raise ValueError("the applause run does not fit in the clip")
    info = {"n_claps": n_claps, "claps": claps, "rate_hz": CLAP_HZ, "raise_s": [CLAP_REST_S, t0],
            "hold_s": [claps[-1]["release_s"], tc], "part_s": [tc, part_end], "lower_s": [part_end, lower_end],
            "openings_m": [float(x) for x in top], "periods_s": [float(x) for x in per]}
    return np.where(ik, g, 0.0), ik, info


@functools.lru_cache(maxsize=1)
def _clap_reference():
    """The seated reference of the applause ('sit', no background): {'centre' (3,) and the wrist offsets from the
    palm centre (side: (3,)) in spine3's frame, 'medial' (side: m, the wrist's distance from the hand's most medial
    vertex along x), 'G' (side: hand rotation in spine3's frame)} for the hand rotations of clap_hand_rotation and
    the clapping hands (clap_hand)."""
    b = static_body("sit", 0.01)
    R, tr = b.local_R()[:1], b.transl[:1]
    J, Gl = fk(R, tr, rest_joints())
    Rh = clap_hand()[None]
    out = {"wrist_off": {}, "medial": {}, "G": {}}
    G3 = Gl[0, 9]
    for side in ("left", "right"):
        G = clap_hand_rotation(side)
        W = J[0, 9] + torch.tensor([0.2 if side == "left" else -0.2, 0.0, 0.3], dtype=F64)
        sol = arm_ik(R, tr, side, W, G, keep_collar=True, pole=CLAP_POLE)
        Rk = R.clone()
        for j, M in zip(ARM[side], sol[:4]):
            Rk[:, j] = M
        Jk = fk(Rk, tr, rest_joints())[0]
        Ph = H.body_points(_body2(Rk, tr, Rh), _hand_ids(side), knots=[0])[0]
        w = Jk[0, H.WRIST[side]].numpy()
        out["wrist_off"][side] = G3.T @ torch.as_tensor(w - Ph.mean(0))
        out["medial"][side] = float(w[0] - Ph[:, 0].min()) if side == "left" else float(Ph[:, 0].max() - w[0])
        out["G"][side] = G3.T @ G
    out["centre"] = torch.as_tensor(CLAP_CENTRE, dtype=F64)
    return out


def _clap_targets(J, Gl, g, side, corr):
    """(W (N, 3), G (N, 3, 3)) of one hand for the palms' distances g (N,) in the current spine3 frames (J, Gl (N,
    22, ...)), plus the per-knot x corrections corr (N,)."""
    ref = _clap_reference()
    sgn = 1.0 if side == "left" else -1.0
    G3 = Gl[:, 9]
    C = J[:, 9] + torch.einsum("tij,j->ti", G3, ref["centre"])
    off = torch.einsum("tij,j->ti", G3, ref["wrist_off"][side])
    W = C + off
    W[:, 0] = C[:, 0] + sgn * (torch.as_tensor(g) / 2.0 + ref["medial"][side]) + torch.as_tensor(corr)
    return W, G3 @ ref["G"][side]


def applause_parts(n_claps, duration_s=CLAP_DUR_S, rate=100.0, seed=CLAP_SEED, passes=3):
    """(HandBody, info) of the applause test (module docstring): the seated 'sit' with the background layer, the
    hands raised by a joint blend to the start, the claps by arm_ik toward wrist targets in spine3's frame (the palms'
    distance of clap_gap, corrected per knot over passes so that each hand's most medial vertex lies at half the
    distance from the midline: the hands never cross the midsagittal plane), then parted and lowered by a joint
    blend back to the chair arms."""
    g, ik, cinfo = clap_gap(n_claps, duration_s, rate, seed)
    T = len(g)
    t = np.arange(T) / rate
    base = static_body("sit", (T - 1) / rate)
    R = base.local_R().clone()
    tr = base.transl.clone()
    bg = background_signals(SERIES["applause"], T, 1.0, None, rate)
    _trunk_layer(R, bg["breath"], bg["yaw"], bg["pitch"], bg["lean"], bg["lean_ml"])
    Rsit = R.clone()
    J, Gl = fk(R, tr, rest_joints())
    kk = np.flatnonzero(ik)
    Rh = clap_hand()[None].expand(T, -1, -1, -1).clone()
    corr = {s: np.zeros(len(kk)) for s in ("left", "right")}
    reach = np.zeros(len(kk))
    for _ in range(passes):
        Rk = R[kk].clone()
        for side in ("left", "right"):
            W, G = _clap_targets(J[kk], Gl[kk], g[kk], side, corr[side])
            sol = arm_ik(R[kk], tr[kk], side, W, G, keep_collar=True, pole=CLAP_POLE)
            for j, M in zip(ARM[side], sol[:4]):
                Rk[:, j] = M
            reach = np.maximum(reach, sol[4])
        bk = H.HandBody(matrix_to_rot6d(Rk), tr[kk], torch.zeros(len(kk), 2, dtype=torch.bool),
                        hand6d=matrix_to_rot6d(Rh[kk]))
        err = 0.0
        for side in ("left", "right"):
            Ph = H.body_points(bk, _hand_ids(side))[..., 0]
            x = Ph.min(1) if side == "left" else -Ph.max(1)
            d = g[kk] / 2.0 - x
            corr[side] += d * (1.0 if side == "left" else -1.0)
            err = max(err, float(np.abs(d).max()))
        if err < 2e-4:
            break
    R[kk] = Rk
    rows = _arm_joints()
    k_up0, k_up1 = int(round(CLAP_REST_S * rate)), int(kk[0])
    n_up = k_up1 - k_up0
    R[k_up0:k_up1, rows] = _blend(Rsit[k_up0:k_up1, rows], R[k_up1:k_up1 + 1, rows].expand(n_up, -1, -1, -1),
                                  _mj((np.arange(n_up) + 1) / n_up))
    k_lo0 = int(kk[-1]) + 1
    n_lo = int(round(CLAP_LOWER_S * rate))
    R[k_lo0:k_lo0 + n_lo, rows] = _blend(R[k_lo0 - 1:k_lo0, rows].expand(n_lo, -1, -1, -1).clone(),
                                         Rsit[k_lo0:k_lo0 + n_lo, rows], _mj((np.arange(n_lo) + 1) / n_lo))
    # the hands: relaxed at rest (the fingers' drift), shaped to clap over the raise, relaxed again over the lowering
    Rrel = relaxed_hand(CLAP_REST_HAND, bg["fingers"])
    clap = Rh[:1].clone()
    Rh[:k_up0] = Rrel[:k_up0]
    Rh[k_up0:k_up1] = _blend(Rrel[k_up0:k_up1], clap.expand(n_up, -1, -1, -1).clone(),
                             _mj((np.arange(n_up) + 1) / n_up))
    Rh[k_lo0:k_lo0 + n_lo] = _blend(clap.expand(n_lo, -1, -1, -1).clone(), Rrel[k_lo0:k_lo0 + n_lo],
                                    _mj((np.arange(n_lo) + 1) / n_lo))
    Rh[k_lo0 + n_lo:] = Rrel[k_lo0 + n_lo:]
    raise_fix = _clear_path(R, tr, Rh, np.arange(k_up0, k_up1), rate, True, float(g[k_up1]) / 2.0)
    lower_fix = _clear_path(R, tr, Rh, np.arange(k_lo0, k_lo0 + n_lo), rate, False)
    b = H.HandBody(matrix_to_rot6d(R), tr, torch.zeros(T, 2, dtype=torch.bool), rate=rate, hand6d=matrix_to_rot6d(Rh))
    b = H.HandBody(b.rot6d, b.transl, contact_labels(b), b.betas, b.gender, b.rate, b.hand6d)
    from .body import rotation_angle_deg
    J2, Gl2 = b.fk()
    hand_err = max(float(rotation_angle_deg(Gl2[kk, H.WRIST[s]].transpose(-1, -2) @
                                            _clap_targets(J2[kk], Gl2[kk], g[kk], s, corr[s])[1]).max())
                   for s in ("left", "right"))
    info = dict(cinfo, gap_design_m=g, ik_knots=[int(kk[0]), int(kk[-1])], reach_max=float(reach.max()),
                hand_rotation_error_deg=hand_err, x_correction_max_m=float(max(np.abs(c).max() for c in corr.values())),
                centre_m=list(CLAP_CENTRE), pole=list(CLAP_POLE), hand=dict(CLAP_HAND),
                rest_hand_weight=CLAP_REST_HAND, raise_clearance=raise_fix, lower_clearance=lower_fix,
                background=background_record(SERIES["applause"], T, 1.0, rate, APPLAUSE_PARTS))
    return b, info


def _mean_edge(x, n):
    """(T,) the centred moving mean of x (T,) over n knots, the edges held (as physio's)."""
    if n <= 1:
        return np.asarray(x, dtype=np.float64)
    c = np.cumsum(np.r_[0.0, np.pad(np.asarray(x, dtype=np.float64), (n // 2, n - 1 - n // 2), mode="edge")])
    return (c[n:] - c[:-n]) / n


def _medial_m(R, tr, Rh):
    """{side: (N,) m} the distance of each hand's most medial vertex from the midsagittal plane x = 0 (positive on its
    own side; measure_claps' palms' distance is their sum) for rotations R (N, 22, 3, 3), tr (N, 3), hands Rh (N,
    30, 3, 3)."""
    b = H.HandBody(matrix_to_rot6d(R), tr, torch.zeros(R.shape[0], 2, dtype=torch.bool), hand6d=matrix_to_rot6d(Rh))
    return {"left": H.body_points(b, _hand_ids("left"))[..., 0].min(1),
            "right": -H.body_points(b, _hand_ids("right"))[..., 0].max(1)}


def _clear_path(R, tr, Rh, kr, rate, closing=True, half=None, passes=CLAP_PATH_PASSES):
    """R (T, 22, 3, 3) in place over the knots kr of the raise (closing) or of the lowering (opening): a joint blend
    between the rest and the clapping pose turns the hands palm to palm while the relaxed thumbs still stand out of the
    palms, so the hands' medial vertices came in and went out again on the way (task sh1b, the base version: the raise
    brought them to 16.3 cm against the 20 cm of the first opening, one more approach of the hands than the claps; the
    lowering came back in by 7 mm at 11 cm). Each upper arm is turned outward about the vertical through its shoulder
    (the elbow swinging out) by what brings its hand's medial vertex onto a path that only closes (the suffix maximum
    of its distance from the midline) or only opens (the prefix maximum), never below the blend's, smoothed by a
    moving maximum then two moving means over CLAP_RAISE_SMOOTH_S (still monotone and never below it, no kink), the
    turn tapered to nothing over the first and the last CLAP_RAISE_SMOOTH_S (it meets the rest and the claps' solve);
    solved over passes (the lever: the wrist's forward distance from the shoulder), then the solved turn itself
    smoothed by two moving means over CLAP_RAISE_SMOOTH_S and tapered again before the arms are turned once: the
    medial vertex the solve follows passes from one skin point to another, so the per-knot turn had its corners and
    the wrists jolted at one knot (task sh1b review, score 0: 9.9 m/s2 on the raise and 10.6 m/s2 on the lowering,
    against 5.0 and 1.1 m/s2 in the base version, which had no turn). Returns the record (half: half the first
    opening, the deficit below it; the rise the path needs, the turn applied, the least distance after, the solve's
    residual and the smoothed turn's largest miss of the path)."""
    n = max(1, int(round(CLAP_RAISE_SMOOTH_S * rate)))
    rest = rest_joints()
    up = torch.tensor([0.0, 1.0, 0.0], dtype=F64)
    R0 = R[kr].clone()
    d0 = _medial_m(R0, tr[kr], Rh[kr])
    rec = {"path": "closing" if closing else "opening", "smooth_s": CLAP_RAISE_SMOOTH_S}
    if half is not None:
        rec["half_opening_m"] = half
    phi = {}
    N = len(kr)
    k = np.arange(N)
    taper = (1.0 - _mj((k - (N - 1 - n)) / n)) * _mj(k / n)
    for side, sgn in (("left", 1.0), ("right", -1.0)):
        env = np.maximum.accumulate(d0[side][::-1])[::-1] if closing else np.maximum.accumulate(d0[side])
        lo = np.lib.stride_tricks.sliding_window_view(np.pad(env, (n, n), mode="edge"), 2 * n + 1).max(1)
        need = np.maximum(_mean_edge(_mean_edge(lo, n), n) - d0[side], 0.0) * taper
        rec[side] = {"path_rise_max_m": float((env - d0[side]).max()), "turned_knots": int((need > 1e-9).sum())}
        if half is not None:
            rec[side]["deficit_max_m"] = float(np.maximum(0.0, half - d0[side]).max())
        phi[side] = (sgn, need, np.zeros(N))
    d, err, miss_s = d0, 0.0, 0.0
    turns = {side: np.zeros(N) for side in phi}
    if any(rec[s]["turned_knots"] for s in ("left", "right")):
        J, Gl = fk(R0, tr[kr], rest)

        def turned(turn):
            Rk = R0.clone()
            for side, (sgn, _, _) in phi.items():
                c, s, _, _ = ARM[side]
                Ry = axis_angle_to_matrix(up[None] * torch.from_numpy(sgn * turn[side])[:, None])
                Rk[:, s] = Gl[:, c].transpose(-1, -2) @ Ry @ Gl[:, c] @ R0[:, s]
            return Rk
        for _ in range(passes):
            d = _medial_m(turned({side: p[2] for side, p in phi.items()}), tr[kr], Rh[kr])
            err = 0.0
            for side, (sgn, need, ph) in phi.items():
                c, s, _, _ = ARM[side]
                lever = (J[:, H.WRIST[side], 2] - J[:, s, 2]).abs().clamp_min(0.1).numpy()
                miss = (d0[side] + need) - d[side]
                ph += np.where(need > 0, miss / lever, 0.0)
                err = max(err, float(np.abs(np.where(need > 0, miss, 0.0)).max()))
            if err < 2e-4:
                break
        turns = {side: _mean_edge(_mean_edge(p[2], n), n) * taper for side, p in phi.items()}
        for _ in range(CLAP_SMOOTH_PASSES):          # where the smoothed turn leaves the hand under its path, the
            d = _medial_m(turned(turns), tr[kr], Rh[kr])     # shortfall added, smoothed the same way (open-body
            for side, (sgn, need, ph) in phi.items():        # port, wave 6 follow-up: the smoothing alone left
                c, s, _, _ = ARM[side]                       # the open body's raise 1.1 mm under it)
                lever = (J[:, H.WRIST[side], 2] - J[:, s, 2]).abs().clamp_min(0.1).numpy()
                short = np.where(need > 0, np.maximum((d0[side] + need) - d[side], 0.0), 0.0)
                turns[side] = turns[side] + _mean_edge(_mean_edge(short / lever, n), n) * taper
        Rk = turned(turns)
        d = _medial_m(Rk, tr[kr], Rh[kr])
        miss_s = max(float(np.abs(np.where(need > 0, d0[side] + need - d[side], 0.0)).max())
                     for side, (_, need, _) in phi.items())
        R[kr] = Rk
    rec["turn_max_deg"] = float(np.degrees(max(np.abs(x).max() for x in turns.values())))
    rec["medial_min_m"] = float(min(d["left"].min(), d["right"].min()))
    rec["residual_m"] = err
    rec["smoothed_miss_max_m"] = miss_s
    return rec


def measure_claps(body, touch_m=CLAP_TOUCH_M):
    """The applause read back from a body: the palms' distance per knot (the left hand's most medial vertex's x minus
    the right hand's, hands.body_points: positive while the hands lie on their own sides of the midsagittal plane),
    the claps (runs closer than touch_m), their times and rate, the least distance (negative: the hands crossed),
    the largest opening; the approaches (every local minimum of the distance with a prominence of CLAP_APPROACH_M at
    least, touching or not: what an eye counts as a beat of the hands), their times and distances."""
    from scipy.signal import find_peaks
    xl = H.body_points(body, _hand_ids("left"))[..., 0].min(1)
    xr = H.body_points(body, _hand_ids("right"))[..., 0].max(1)
    gap = xl - xr
    on = gap < touch_m
    m = np.diff(np.r_[False, on, False].astype(np.int8))
    runs = list(zip(np.flatnonzero(m == 1), np.flatnonzero(m == -1)))
    times = [a / body.rate for a, _ in runs]
    pk, _ = find_peaks(-np.round(gap, 6), prominence=CLAP_APPROACH_M)   # to the micrometre: the posed contact's
    #                                   plateau at CLAP_GAP_M is one minimum (open-body port, wave 6 follow-up: the open
    #                                   body's skinning noise split the last clap's held contact into two)
    return {"claps": len(runs), "clap_times_s": times, "approaches": int(len(pk)),
            "approach_times_s": (pk / body.rate).tolist(), "approach_gaps_m": gap[pk].tolist(),
            "rate_hz": float((len(times) - 1) / (times[-1] - times[0])) if len(times) > 1 else None,
            "gap_min_m": float(gap.min()), "left_min_x_m": float(xl.min()), "right_max_x_m": float(xr.max()),
            "contact_gap_max_m": float(max(gap[a:e].min() for a, e in runs)) if runs else None,
            "opening_max_m": float(gap[int(runs[0][0]):int(runs[-1][1])].max()) if len(runs) > 1 else None,
            "gap": gap}


def applause(score, duration_s=CLAP_DUR_S, seed=CLAP_SEED):
    """(HandBody, truth) of the applause test at a score of CLAP_COUNTS (module docstring), with the checks: the
    joint-limit table, the hand limits, continuity, the feet, the hands on their own sides of the midline, the hand
    rotation reached, the arms' reach; the read-back (measure_claps)."""
    n = CLAP_COUNTS[int(score)]
    b, info = applause_parts(n, duration_s, seed=seed)
    base = static_body("sit", b.duration)
    val = validity(base, b)
    val["hand_limits"] = H.hand_limit_report(b.hand_R())
    val["continuity"] = continuity_report(b)
    m = measure_claps(b)
    gap = m.pop("gap")
    val["hands_apart"] = {"ok": m["gap_min_m"] > 0.0, "gap_min_m": m["gap_min_m"],
                          "left_min_x_m": m["left_min_x_m"], "right_max_x_m": m["right_max_x_m"]}
    val["approaches"] = {"ok": m["approaches"] == m["claps"] == n, "approaches": m["approaches"],
                         "claps": m["claps"], "stated": n,
                         "rule": "every approach of the hands is a clap: no other minimum of the palms' distance"}
    val["hand_rotation_error_deg"] = info["hand_rotation_error_deg"]
    val["reach_max"] = info["reach_max"]
    g = info.pop("gap_design_m")
    p = dict(info, score=int(score), demonstration="3 claps by the examiner, not shown (no examiner in the scene)",
             positive=n > 3, remeasured=m)
    srcs = ["dubois2005", "module"]
    return b, truth_record("applause", seed, int(score), "applause test (Dubois2005; graded scoring not traced)", p,
                           srcs, val, arrays={"gap_design_m": g, "gap_m": gap})


# ---------------------------------------------------------------- P33: a motor state transition
MS_DUR_S = 45.0                # the E1 seated rest (exam.DUR['E1'])
MS_HZ = 5.0                    # the exam's rest tremor frequency (exam.TREMOR_HZ['rest'], tremors.TOP_HZ[3])
MS_GRADES = {"off": 3, "on": 1}   # [module] MDS-UPDRS 3.17 in OFF and in ON (a slight residual tremor in ON, at
#                                the lower grade-1 level of grades.tremor_level_cm, inside the band, not at its edge)
MS_HOLD_S = 12.0               # [module] the first state held, then the transition, then the second state
MS_RAMP_S = 20.0               # [module] the transition (minimum jerk in log amplitude); a clinical transition takes
#                                minutes: compressed into the clip, no time course sourced
MS_HAND_WEIGHT = 0.5           # [module] the resting hand beyond the chair arm, RELAXED_HAND at this share
MS_SEED = SEED + 80
MS_WINDOW_S = 1.0              # read-back: the chord in windows of this length (5 cycles), every MS_STEP_S
MS_STEP_S = 0.5
MS_DIRECTIONS = ("off_to_on", "on_to_off")


def ms_level_cm(t, direction):
    """(T,) cm peak to peak at the middle fingertip of a motor state clip (module docstring): the OFF or ON level
    (grades.tremor_level_cm of MS_GRADES, grade 1 at its 'below_floor' level, 0.58 cm) held MS_HOLD_S, a minimum-jerk
    ramp in log amplitude over MS_RAMP_S, the other level to the end; direction 'off_to_on' or 'on_to_off'."""
    off, on = G.tremor_level_cm(MS_GRADES["off"]), G.tremor_level_cm(MS_GRADES["on"], "below_floor")
    a, b = (off, on) if direction == "off_to_on" else (on, off)
    s = _mj((np.asarray(t, dtype=np.float64) - MS_HOLD_S) / MS_RAMP_S)
    return np.exp(np.log(a) + (np.log(b) - np.log(a)) * s)


def ms_crossings(direction):
    """{cm: t s}: the times the designed level crosses the MDS-UPDRS band edges 3 and 1 cm (grades 3 to 2, 2 to 1)."""
    t = np.arange(int(round(MS_DUR_S * 1000)) + 1) / 1000.0
    c = ms_level_cm(t, direction)
    out = {}
    for edge in (3.0, 1.0):
        k = np.flatnonzero(np.diff(np.sign(c - edge)) != 0)
        out[edge] = float(t[k[0]]) if len(k) else None
    return out


@functools.lru_cache(maxsize=1)
def _ms_tip_offset():
    """(3,) the middle fingertip vertex of the left resting hand (relaxed_hand(MS_HAND_WEIGHT)) from the wrist joint,
    in the wrist's frame (rest pose)."""
    R = torch.eye(3, dtype=F64).expand(1, 22, 3, 3).clone()
    b = _body2(R, torch.zeros(1, 3, dtype=F64), relaxed_hand(MS_HAND_WEIGHT)[None])
    tip = H.body_points(b, [H.tip_vertices()["left"]["middle"]], knots=[0])[0, 0]
    return torch.as_tensor(tip - b.joints().numpy()[0, H.WRIST["left"]], dtype=F64)


@functools.lru_cache(maxsize=2)
def ms_base(duration_s=MS_DUR_S, seed=MS_SEED):
    """(HandBody) the seated rest of the motor state clips and their level 0: 'sit' (the forearms on the chair arms,
    the hands beyond them) with the background layer of SERIES['motor_state'] (breathing, the head's and the trunk's
    drift, the fingers' drift on relaxed hands; seed is the cache key only, nothing is drawn from it)."""
    base = static_body("sit", duration_s)
    R = base.local_R().clone()
    bg = background_signals(SERIES["motor_state"], base.T, 1.0, None, base.rate)
    _trunk_layer(R, bg["breath"], bg["yaw"], bg["pitch"], bg["lean"], bg["lean_ml"])
    Rh = relaxed_hand(MS_HAND_WEIGHT, bg["fingers"])
    b = H.HandBody(matrix_to_rot6d(R), base.transl, torch.zeros(base.T, 2, dtype=torch.bool), rate=base.rate,
                   hand6d=matrix_to_rot6d(Rh))
    return H.HandBody(b.rot6d, b.transl, contact_labels(b), b.betas, b.gender, b.rate, b.hand6d)


def measure_motor_state(body, level0, window_s=MS_WINDOW_S, step_s=MS_STEP_S):
    """The tremor's level over time read back from a body against its level 0 (the same rest without the tremor):
    the left middle fingertip vertex's displacement, its chord along the first principal axis (measure.chord_pp) in
    windows of window_s every step_s, in cm, and the MDS-UPDRS 3.17 grade of each (grades.tremor_grade_of_cm); the
    dominant frequency; the median grade over the windows of the first and the last 10 s; the window centres where
    the chord crosses 3 and 1 cm (linear interpolation)."""
    from . import measure as MM
    tid = [H.tip_vertices()["left"]["middle"]]
    D = H.body_points(body, tid)[:, 0] - H.body_points(level0, tid)[:, 0]
    n, s = int(round(window_s * body.rate)), int(round(step_s * body.rate))
    centres, cm = [], []
    for a in range(0, body.T - n + 1, s):
        centres.append((a + n / 2) / body.rate)
        cm.append(100.0 * MM.chord_pp(D[a:a + n]))
    centres, cm = np.asarray(centres), np.asarray(cm)
    grades = np.array([G.tremor_grade_of_cm(c) for c in cm])
    first, last = centres < 10.0, centres > centres[-1] - 10.0
    cross = {}
    for edge in (3.0, 1.0):
        k = np.flatnonzero(np.diff(np.sign(cm - edge)) != 0)
        if len(k):
            i = k[0]
            cross[edge] = float(centres[i] + (edge - cm[i]) / (cm[i + 1] - cm[i]) * (centres[i + 1] - centres[i]))
        else:
            cross[edge] = None
    return {"window_s": window_s, "step_s": step_s, "centres_s": centres.tolist(), "pp_cm": cm.tolist(),
            "grades": grades.tolist(), "grade_first": int(np.median(grades[first])),
            "grade_last": int(np.median(grades[last])), "pp_cm_first": float(np.median(cm[first])),
            "pp_cm_last": float(np.median(cm[last])), "crossings_s": cross,
            "frequency_hz": float(MM.dominant_frequency(D, body.rate, (3.0, 12.0)))}


def motor_state(direction, duration_s=MS_DUR_S, seed=MS_SEED):
    """(HandBody, truth) of a motor state transition (module docstring): the rest tremor of the left hand at the
    wrist (tremors.oscillate, the 'wrist' drive, the chord at the middle fingertip calibrated per knot) at MS_HZ, its
    level ms_level_cm(direction) over the clip (the grade ramped from OFF to ON or back), on ms_base; 'level0': the
    base alone."""
    from . import tremors as TR
    base = ms_base(duration_s, seed)
    if direction == "level0":
        val = validity(base, base)
        val["hand_limits"] = H.hand_limit_report(base.hand_R())
        val["continuity"] = continuity_report(base)
        p = {"direction": "level0", "duration_s": float(duration_s), "rest": "'sit', forearms on the chair arms",
             "hand_weight": MS_HAND_WEIGHT,
             "background": background_record(SERIES["motor_state"], base.T, 1.0, base.rate, REST_PARTS)}
        return base, truth_record("motor_state", seed, 0, "MDS-UPDRS 3.17 level 0", p, ("mds_updrs", "module"), val)
    if direction not in MS_DIRECTIONS:
        raise ValueError(f"direction {direction!r}")
    rng = as_rng(seed + 1 + MS_DIRECTIONS.index(direction))
    t = base.times().numpy()
    cm = ms_level_cm(t, direction)
    phase0 = float(rng.uniform(0.0, 2.0 * math.pi))
    y = np.sin(TR.phase_of(t, MS_HZ, phase0))
    env = np.ones(base.T)
    ps = ("rigid", H.WRIST["left"], _ms_tip_offset())
    out, oinfo, arr = TR.oscillate(base, TR.drives_of("wrist", "left"), cm, y, env, ps, MS_HZ)
    val = validity(base, out)
    val["hand_limits"] = H.hand_limit_report(out.hand_R())
    val["continuity"] = continuity_report(out)
    m = measure_motor_state(out, base)
    off, on = G.tremor_level_cm(MS_GRADES["off"]), G.tremor_level_cm(MS_GRADES["on"], "below_floor")
    p = {"direction": direction, "duration_s": float(duration_s), "frequency_hz": MS_HZ, "drive": "wrist",
         "point": "left middle fingertip (relaxed hand)", "grades": dict(MS_GRADES), "off_cm": off, "on_cm": on,
         "hold_s": MS_HOLD_S, "ramp_s": MS_RAMP_S, "ramp": "minimum jerk in log amplitude",
         "designed_crossings_s": ms_crossings(direction), "phase_rad": phase0, "oscillator": oinfo,
         "time_course": "module value, unsourced: a clinical OFF to ON or wearing-off transition takes minutes",
         "remeasured": m, "background": background_record(SERIES["motor_state"], base.T, 1.0, base.rate, REST_PARTS)}
    g = f"{MS_GRADES['off']} to {MS_GRADES['on']}" if direction == "off_to_on" else \
        f"{MS_GRADES['on']} to {MS_GRADES['off']}"
    return out, truth_record("motor_state", seed, g, "MDS-UPDRS 3.17 (motor state transition)", p,
                             ("mds_updrs", "elble2006", "module"), val,
                             arrays={"level_cm": cm, "angle_rad": arr["angle_rad"]})


# ---------------------------------------------------------------- G10: Gowers' manoeuvre when arising
GOWERS_DUR_S = ARISE_DUR_S     # the clip, on the time base of its level 0 (park_E7_arise_g0)
GOWERS_SEED = SEED + 90
# [module] the phases (s): seated, the forward lean (the hands slide toward the knees), the push (the buttocks rise,
# the trunk stays bent), six hand steps up the thighs (left and right in turn; each hand moves up while the trunk is
# held, then both hands push the trunk up), the final extension from the proximal thighs (the hands leave the thighs,
# the arms hang); then standing
GOWERS_PHASES_S = {"seated": 1.0, "lean": 1.4, "push": 1.6, "step": 0.8, "final": 1.6}
GOWERS_STEPS = (("left", 0.70), ("right", 0.55), ("left", 0.42), ("right", 0.30), ("left", 0.20),
                ("right", 0.15))   # [module] each hand's new place on its thigh, the share of the way from hip to knee:
#   hand over hand, each passing the other, so that the lower hand (which bounds the trunk's extension) moves up at
#   every step; the fifth stops at 0.20 (at 0.18 the left shoulder's extension leaves the table by 1 deg)
GOWERS_TRUNK_SHARES = (0.14, 0.28, 0.42, 0.58, 0.72, 1.0)   # [module] the trunk's way from GOWERS_PUSH to
#   GOWERS_CLIMB after each step: the lean falls 6 deg at each of the first five steps, as far as the lower hand
#   keeps its reach (the arm's reach share at most 0.97), and to the release pose at the last
GOWERS_STEP_MOVE = 0.55        # [module] a step: the hand moves up over this share of it (the trunk, the hips and the
#                                knees held), then both hands push the trunk up over the rest; open-body port, wave 6 last
#                                pass: 0.5 before, at which the open body's right elbow moving up its longer thigh
#                                reached 6.54 m/s2 against its test's 6; the least of 0.05 steps under it
#                                (scripts/openbody_wave6_checks.py, gowers_jolts)
GOWERS_LEAN = {"pelvis_deg": 20.0, "spine_deg": 50.0, "along": 0.85}   # [module] the seated lean at lift-off: the
#   pelvis's forward pitch, the spine's flexion (shared by trunk_split), the hands near the knees; open-body port,
#   wave 6 follow-up: kept, since a deeper pitch turns the forearms of the hands flat on the thighs past the table (at
#   30 deg the elbows' z component, at 24 too: scripts/openbody_wave6_checks.py, gowers)
GOWERS_PUSH = {"pelvis_deg": 40.0, "spine_deg": 55.0, "drop_m": 0.12}  # [module] after the push: the trunk bent,
#   the hips this far below their standing height (the knees bent)
GOWERS_CLIMB = {"pelvis_deg": 17.0, "spine_deg": 28.0, "drop_m": 0.03}  # [module] after the last hand step, the
#   hands on the proximal thighs: the trunk leans GOWERS_RELEASE_MAX_DEG or less, the pose the hands leave the thighs
#   from (at about 27 deg of lean the arms reach the thighs' proximal sixth, the knees nearly straight); each step
#   extends the trunk by its part of GOWERS_TRUNK_SHARES
GOWERS_RELEASE_MAX_DEG = 30.0  # [module] the trunk lean (from the seated start) at most this when the hands leave
GOWERS_KNEES_STEPS = 6         # [module] the knees straighten (to drop_m of GOWERS_CLIMB) over this many hand steps:
#   kept bent while the hands are low on the thighs, so that the lower hand reaches its thigh with less trunk lean
GOWERS_SEATED_ALONG = 0.35     # [module] the seated hands rest on the proximal thighs
GOWERS_SCOOT_M = 0.03          # [module] over the forward lean the hips slide this far forward on the seat (the hip
#                                midpoint's forward position, the seat's height kept), toward the seat's front; open-
#                                body port, wave 6 follow-up: from the arising's upright seat (motions D80) with the
#                                spine at its table room (D91) the centre of mass stayed behind the heels at lift-off;
#                                the smallest of 0.01 m steps at which the extrapolated centre of mass keeps MOS_MIN_M
#                                ahead of the heels from lift-off (scripts/openbody_wave6_checks.py, gowers)
GOWERS_SEATED_FLEX_DEG = 20.0  # [module] the seated spine flexed this much (trunk_split): upright, not reclined
GOWERS_CAP_SOFT_DEG = 2.0      # [module] the width of the smooth cap of the spine's flexion at its table room (a
#                                softened minimum, w log(1 + exp((room - flex) / w)) under the room, the rest to the
#                                pelvis's pitch); open-body port, wave 6 last pass: the hard minimum of D91 stopped the
#                                spine in one knot where the lean crossed the room (1.98 s) and where the climb crossed
#                                it back (6.23 s), and the shoulders' and hips' paths kinked there (the right elbow 7.47
#                                and the knees 1.79 m/s2 in one knot); the least of the widths 2, 4 and 8 deg tried, each
#                                of which removes both (scripts/openbody_wave6_checks.py, gowers_jolts)
GOWERS_STEP_LIFT_M = 0.025     # [module] a hand rises this far off the thigh while it steps (a sine-squared bump:
#                                no jolt where a step starts and ends; the sine it replaced jolted the wrist by about
#                                20 m/s2 at one knot, task sh1b)
GOWERS_RELEASE_LIFT_S = 0.3    # [module] at the end of the climb both hands lift off the thighs over this (the trunk,
GOWERS_RELEASE_LIFT_M = 0.03   # the hips and the knees held) by this much, before the final extension and the arms'
#                                swing to hanging (a joint blend from the thighs' skin dips the hands into them)
GOWERS_POLE = (1.0, 1.0)       # [module] the elbows point out and back while the hands push on the thighs (the pole
#                                with the largest margin inside the joint-limit table on a small grid)
GOWERS_COM_ITERS = 4
GOWERS_HAND_PASSES = 5         # passes of the hand-on-thigh clearance correction
GOWERS_CORR_SMOOTH_S = 0.055   # [module] the correction then smoothed over this before the last solve (_gowers_hands;
#                                0.05 until wave SH1d fix B: GOWERS_THUMB_CMC_DEG)
GOWERS_THUMB_CMC_DEG = (4.0, -12.5)   # [module] the thumb's base (CMC flexion, palmar abduction) while the hands
#   push on the thighs (gowers_hand): the thumb spread toward the back of the hand, out of the palm's way. With the
#   relaxed thumb (4 and 4 deg) the thumb tip was the hand's lowest point at every knot of the climb and held the palm
#   off the skin, 18 to 21 mm on the posed mesh and on the skeleton page's palm reading alike (15.8 and 19.2 mm, its
#   limit 15) while the thumb tip kept THIGH_REST['clear_m'] (wave SH1d fix B). Tried on whole builds: at (10, -20) the
#   palm came within 13.4 and 6.6 mm on the page but a wrist jolted at 6.8 m/s2 where the lowest point changed at the
#   first step; at (4, -16) the hands' rehearsal took 10 cm of lateral shift and the left palm read 17.1 mm; at these
#   values, with GOWERS_CORR_SMOOTH_S 0.055, the page passes, the wrists stay under 5.1 m/s2 and the planted hands'
#   lowest point within 3.0 to 7.1 mm of the skin. Turning the palm onto the skin's own slope under it as well changed
#   the mesh's palm gap by less than 1 mm (not kept)
GOWERS_THIGH_FRONT_M = 0.07    # the first guess of the palm centre's distance in front of the thigh's axis
GOWERS_REACH = 0.97            # the hips kept within this share of the leg's length from the ankles while solving
GOWERS_LATERAL_STEP_M = 0.01   # [module] the grid of the hands' lateral shift on the thighs (gowers_lateral)
GOWERS_LATERAL_MAX_M = 0.12    # [module] its largest value tried
GOWERS_SKIN_CLEAR_M = 0.001    # [module] the least skin gap of GOWERS_SKIN_CHECKS on the rehearsal of gowers_lateral
GOWERS_SKIN_CHECKS = ("hands", "trunk_left", "trunk_right", "thigh_left", "thigh_right")   # the skin pairs the
#   hands on the thighs keep apart (motions.fold_gaps: the two hands, a side's forearm and hand against the trunk,
#   pelvis and spine, and against the thighs); with the palm centre on the thigh's axis and the fingers along the
#   forearm, the fingertips of the two hands met at the midline and the hands and forearms entered the hip crease
#   on the proximal thighs (review of task sh1c; gowers_lateral's try at 0 gives the depths)


def _seg_mj(t, keys):
    """(T,) a track through keys [(time s, value)], minimum-jerk between consecutive keys, held outside."""
    t = np.asarray(t, dtype=np.float64)
    out = np.full(t.shape, float(keys[0][1]))
    for (t0, v0), (t1, v1) in zip(keys[:-1], keys[1:]):
        if t1 > t0:
            out = np.where(t >= t0, v0 + (v1 - v0) * _mj((t - t0) / (t1 - t0)), out)
    return out


def gowers_plan():
    """{phase: (start s, end s)} and the hand steps [(side, start s, end s, from share, to share)] of GOWERS_PHASES_S
    and GOWERS_STEPS."""
    ph = GOWERS_PHASES_S
    t = 0.0
    out = {}
    for name in ("seated", "lean", "push"):
        out[name] = (t, t + ph[name])
        t += ph[name]
    steps, along = [], {"left": GOWERS_LEAN["along"], "right": GOWERS_LEAN["along"]}
    out["climb"] = (t, t + ph["step"] * len(GOWERS_STEPS))
    for side, to in GOWERS_STEPS:
        steps.append((side, t, t + ph["step"], along[side], to))
        along[side] = to
        t += ph["step"]
    out["final"] = (t, t + ph["final"])
    out["lift_off"] = (t, t + GOWERS_RELEASE_LIFT_S)
    out["standing"] = (t + ph["final"], GOWERS_DUR_S)
    return out, steps


def _thigh_gap(b, side, knots):
    """(N,) m: the hand's lowest vertex above the thigh's surface in the thigh's frame (its local +z, the front of the
    thigh; _gap_above on the hand and thigh vertices brought into that frame), at the given knots."""
    h = 1 if side == "left" else 2
    _, _, dom = H._rest()
    thigh = np.flatnonzero(np.isin(dom, (h, h + 3)))
    J, Gl = b.fk()
    Ph = H.body_points(b, _hand_ids(side), knots=knots)
    St = H.body_points(b, thigh, knots=knots)
    out = np.zeros(len(knots))
    for i, k in enumerate(knots):
        G, o = Gl[k, h].numpy(), J[k, h].numpy()
        sw = lambda P: ((P - o) @ G)[:, [0, 2, 1]]  # noqa: E731  local (x, z, y): z becomes the height
        out[i] = _gap_above(sw(Ph[i]), sw(St[i]), radius=0.015)
    return out


def _frame_np(f, n):
    """(3, 3) the orthonormal frame [f, n (made normal to f), f x n] as columns."""
    f = np.asarray(f, dtype=np.float64)
    f = f / np.linalg.norm(f)
    n = np.asarray(n, dtype=np.float64) - (np.asarray(n) @ f) * f
    n = n / np.linalg.norm(n)
    return np.stack([f, n, np.cross(f, n)], 1)


def gowers_hand():
    """(30, 3, 3) both hands as they push on the thighs in Gowers' manoeuvre: relaxed_hand at THIGH_REST['hand_weight']
    with each thumb's CMC at GOWERS_THUMB_CMC_DEG (flexion, palmar abduction)."""
    w = THIGH_REST["hand_weight"]
    pose = {}
    for side in H.SIDES:
        for f, vals in RELAXED_HAND.items():
            for lv, v in enumerate(vals):
                fl, ab = (v, 0.0) if np.isscalar(v) else v
                pose[(side, f, lv)] = (w * fl, w * ab)
        pose[(side, "thumb", 0)] = GOWERS_THUMB_CMC_DEG
    return H.hand_rotations(pose)


@functools.lru_cache(maxsize=2)
def _palm_geometry(side):
    """(palm centre from the wrist joint (3,), finger direction (3,), palmar normal (3,)) of the resting hand
    (gowers_hand()) in the rest frame: the mean of the vertices the wrist dominates, the
    wrist-to-middle-knuckle direction, hands.hand_frame's palmar normal."""
    R = torch.eye(3, dtype=F64).expand(1, 22, 3, 3).clone()
    b = _body2(R, torch.zeros(1, 3, dtype=F64), gowers_hand()[None])
    _, _, dom = H._rest()
    P = H.body_points(b, np.flatnonzero(dom == H.WRIST[side]), knots=[0])[0]
    J = b.joints().numpy()[0]
    J52, _, _ = H._rest()
    f = J52[H.hand_joint(side, "middle", 0)] - J52[H.WRIST[side]]
    return P.mean(0) - J[H.WRIST[side]], f / np.linalg.norm(f), np.asarray(H.hand_frame()[side]["palmar"])


def _gowers_frame(duration_s=GOWERS_DUR_S, rate=100.0, seed=GOWERS_SEED):
    """The body of Gowers' manoeuvre without its arms on the thighs (gowers_body): {'R', 'tr', 't', 'plan', 'steps',
    'alongs', 'lifts', 'k_lift', 'target', 'stand_com', 'bg', 'stand_w'}; the collars protracted from rest while the
    hands are on the thighs."""
    from ..kernels.force import com
    from .motions import trunk_split
    plan, steps = gowers_plan()
    T = int(round(duration_s * rate)) + 1
    t = np.arange(T) / rate
    rest = rest_joints()
    split, rooms = trunk_split()
    Rs, trs, Rhs = _template_pose("normal", np.zeros(1))
    Re, tre, _ = _template_pose("normal", np.ones(1))
    Js, Je = fk(Rs, trs, rest)[0], fk(Re, tre, rest)[0]
    Ms, Me = 0.5 * (Js[0, 1] + Js[0, 2]), 0.5 * (Je[0, 1] + Je[0, 2])
    ys, ye = float(Ms[1]), float(Me[1])
    l0, l1 = plan["lean"]
    p0, p1 = plan["push"]
    c0, c1 = plan["climb"]
    f0, f1 = plan["final"]
    pel0 = float(_x_angle(Rs[:, 0])[0])

    def per_step(v0, v1, shares):
        """keys of a track held while a hand moves (the first GOWERS_STEP_MOVE of each step) and moved while both
        hands push (the rest of it), at the shares (one per step) of the way from v0 to v1 after each step."""
        keys, prev = [], v0
        for sh, st in zip(shares, steps):
            v = v0 + (v1 - v0) * sh
            keys += [(st[1] + GOWERS_STEP_MOVE * (st[2] - st[1]), prev), (st[2], v)]
            prev = v
        return keys
    fl = plan["lift_off"][1]                    # the hands lift off the thighs, then the final extension
    d0, d1 = GOWERS_PUSH["drop_m"], GOWERS_CLIMB["drop_m"]
    knees = [min(1.0, (i + 1) / GOWERS_KNEES_STEPS) for i in range(len(steps))]
    y = _seg_mj(t, [(l1, ys), (p1, ye - d0)] + per_step(ye - d0, ye - d1, knees) + [(fl, ye - d1), (f1, ye)])
    cp, cs = GOWERS_CLIMB["pelvis_deg"], GOWERS_CLIMB["spine_deg"]
    pitch = _seg_mj(t, [(l0, pel0), (l1, GOWERS_LEAN["pelvis_deg"]), (p1, GOWERS_PUSH["pelvis_deg"])] +
                    per_step(GOWERS_PUSH["pelvis_deg"], cp, GOWERS_TRUNK_SHARES) + [(fl, cp), (f1, 0.0)])
    up0 = GOWERS_SEATED_FLEX_DEG
    flex = _seg_mj(t, [(l0, up0), (l1, GOWERS_LEAN["spine_deg"]), (p1, GOWERS_PUSH["spine_deg"])] +
                   per_step(GOWERS_PUSH["spine_deg"], cs, GOWERS_TRUNK_SHARES) + [(fl, cs), (f1, 0.0)])
    spine_max = float(rooms.sum())               # the spine bends at most by its table room (the arising's seat to the
    w = GOWERS_CAP_SOFT_DEG                      # bounds, trunk_split); past it the bow is the hips' (the pelvis's
    capped = spine_max - w * np.logaddexp(0.0, (spine_max - flex) / w)   # pitch): open-body port, wave 6, design D91;
    pitch = pitch + (flex - capped)              # the hand-over smooth (GOWERS_CAP_SOFT_DEG)
    flex = capped
    protract = _seg_mj(t, [(fl, THIGH_REST["protract_deg"]), (f1, 0.0)])
    sp0 = float(_x_angle(Rs[:, 3])[0])
    base1 = _seg_mj(t, [(l1, sp0), (p1, 0.0)])
    G0 = _rot("x", pitch)
    R = Rs.expand(T, -1, -1, -1).clone()
    R[:, 3] = _rot("x", base1 + split[0] * flex)
    R[:, 6] = _rot("x", split[1] * flex)
    R[:, 9] = _rot("x", split[2] * flex)
    k_lift = int(round(l1 * rate))
    mid = torch.as_tensor(0.5 * (rest[1] + rest[2]) - rest[0])
    ank = torch.stack([Js[0, 7], Js[0, 8]], 0)[None].expand(T, 2, 3).clone()
    up = np.clip((y - ys) / (ye - ys), 0.0, 1.0)
    # the legs' reference (knee hinges): the template's seated lean read over the forward lean, then its rise; the
    # version that held the seated start until lift-off and jumped to the template's lift-off share there stepped
    # the knees by about 1 mm at one knot (task sh1b: 9 m/s2 at the knees, 11 m/s2 at the elbows and wrists)
    u_ref = np.where(up > 0, 0.444 + 0.556 * up, 0.444 * _mj((t - l0) / (l1 - l0)))
    Rref, trref, _ = _template_pose("normal", u_ref)

    def place(z):
        M = torch.as_tensor(np.stack([np.full(T, float(Ms[0])), y, z], -1))
        J0 = M - torch.einsum("tij,j->ti", G0, mid)
        return place_pelvis(R, trs.expand(T, 3).clone(), J0, G0, ank, Rref, trref, keep_reach=False)

    def com_z(Rb, trb):
        return com(fk(Rb, trb, rest)[0].numpy(), 2)[:, 2]
    Lleg = float((rest[4] - rest[1]).norm() + (rest[7] - rest[4]).norm())
    hy = float(0.5 * (Js[0, 7, 1] + Js[0, 8, 1]))
    az = float(0.5 * (Js[0, 7, 2] + Js[0, 8, 2]))

    Dmax = max(GOWERS_REACH * Lleg, float(np.hypot(ye - hy, float(Me[2]) - az)) + 1e-3)

    def clampz(z):                 # the hips no further from the ankles than GOWERS_REACH of the leg (or standing)
        lim = az - np.sqrt(np.maximum(Dmax ** 2 - (y - hy) ** 2, 0.0))
        return np.where(t > l1, np.maximum(z, lim), z)
    z_seat = float(Ms[2]) + GOWERS_SCOOT_M * _mj((t - l0) / (l1 - l0))   # the slide forward on the seat
    z = clampz(z_seat + (float(Me[2]) - z_seat) * up)
    Rb, trb = place(z)
    cz = com_z(Rb, trb)
    stand_com = float(com_z(Re, tre)[0])
    target = cz[k_lift] + (stand_com - cz[k_lift]) * _mj((t - l1) / (f1 - l1))
    after = t > l1
    for _ in range(GOWERS_COM_ITERS):             # secant on the hip midpoint's forward position, after lift-off
        dz = 0.01
        cz1 = com_z(*place(clampz(z + dz * after)))
        gain = np.where(after, (cz1 - cz) / dz, 1.0)
        z = clampz(np.where(after, z + (target - cz) / np.maximum(gain, 0.2), z))
        Rb, trb = place(z)
        cz = com_z(Rb, trb)
    R, tr = Rb, trb
    standing = (t >= f1).astype(np.float64)
    stand_w = _mj((t - f1) / 0.5) * standing
    bg = background_signals(SERIES["arise"], T, 1.0, stand_w, rate)     # the background of its level 0's series
    seated = t < l0
    w_sit = _inside_ramp(seated, int(0.3 * rate))
    _trunk_layer(R, bg["breath"], bg["yaw"], bg["pitch"], w_sit * bg["lean"], w_sit * bg["lean_ml"])
    alongs, lifts = {}, {}
    for side in ("left", "right"):
        keys = [(l0, GOWERS_SEATED_ALONG), (l1, GOWERS_LEAN["along"])]
        lift = np.zeros(T)
        for sd, a, b_, fr, to in steps:
            if sd == side:
                mv = GOWERS_STEP_MOVE * (b_ - a)
                keys += [(a, fr), (a + mv, to)]
                lift += GOWERS_STEP_LIFT_M * np.sin(np.pi * np.clip((t - a) / mv, 0.0, 1.0)) ** 2
        lift += GOWERS_RELEASE_LIFT_M * _mj((t - f0) / GOWERS_RELEASE_LIFT_S)
        alongs[side], lifts[side] = _seg_mj(t, keys), lift
    R[:, 13] = _rot("y", -protract)                   # the shoulder girdles protracted from rest, as seated_rest
    R[:, 14] = _rot("y", protract)
    return {"R": R, "tr": tr, "t": t, "plan": plan, "steps": steps, "alongs": alongs, "lifts": lifts,
            "k_lift": k_lift, "target": target, "stand_com": stand_com, "bg": bg, "stand_w": stand_w}


def _gowers_hands(R, tr, kk, alongs, lifts, pole=None, passes=GOWERS_HAND_PASSES, rate=100.0, lateral_m=0.0):
    """(arm and collar rows solved at the knots kk (N, 22, 3, 3), reach share (N,), thigh offsets {side: (N,)}): each
    hand flat on the front of its thigh (the palm centre of _palm_geometry at the share alongs[side] of the way from hip
    to knee, lifts[side] off it, shifted by lateral_m toward the outside of the thigh: the pelvis's lateral axis made
    normal to the thigh's front; gowers_lateral), the palm toward the thigh, the fingers along the forearm's projection
    on the thigh's front (three solves), by arm_ik with the collars kept (the hands of gowers_hand); the palm centre's
    distance from the thigh's axis corrected over passes so that the hand's lowest vertex keeps THIGH_REST['clear_m']
    off the thigh (_thigh_gap),
    then the correction smoothed (on each side a moving maximum over GOWERS_CORR_SMOOTH_S of its excess over its own
    trend, the trend two moving means of it, then two moving means over that maximum, added back to the trend: never
    nearer the thigh than the per-knot correction) and solved once more: the lowest vertex passes from one
    skin point to another, so the per-knot correction had steps of a few mm and the wrists jolted at one knot (task sh1b
    review: 36 to 38 m/s2); the reach share is the last solve's."""
    rest = rest_joints()
    pole = GOWERS_POLE if pole is None else pole
    N = len(kk)
    Rh = gowers_hand()[None].expand(N, -1, -1, -1).clone()
    corr = {s: np.full(N, GOWERS_THIGH_FRONT_M) for s in ("left", "right")}
    J, Gl = fk(R[kk], tr[kk], rest)
    geo = {s: _palm_geometry(s) for s in ("left", "right")}
    A0 = {s: _frame_np(geo[s][1], geo[s][2]) for s in ("left", "right")}
    Rk = R[kk].clone()

    def solve(corr):
        reach = np.zeros(N)
        for side, h, kn, e in (("left", 1, 4, 18), ("right", 2, 5, 19)):
            ant = Gl[:, h][:, :, 2].numpy()
            axis = (J[:, h] + torch.as_tensor(alongs[side][kk])[:, None] * (J[:, kn] - J[:, h])).numpy()
            lat = (1.0 if side == "left" else -1.0) * Gl[:, 0][:, :, 0].numpy()    # the body's left is +x
            lat = lat - (lat * ant).sum(-1, keepdims=True) * ant
            lat = lat / np.linalg.norm(lat, axis=-1, keepdims=True)
            Pc = axis + ant * (corr[side] + lifts[side][kk])[:, None] + lateral_m * lat
            fd = (J[:, kn] - J[:, h]).numpy()
            for _ in range(3):              # the fingers along the forearm's projection on the thigh's front
                fd = fd - (fd * ant).sum(-1, keepdims=True) * ant
                fd = fd / np.linalg.norm(fd, axis=-1, keepdims=True)
                A1 = np.stack([fd, -ant, np.cross(fd, -ant)], -1)
                G = torch.as_tensor(A1 @ A0[side].T)
                W = torch.as_tensor(Pc) - torch.einsum("tij,j->ti", G, torch.as_tensor(geo[side][0]))
                sol = arm_ik(R[kk], tr[kk], side, W, G, keep_collar=True, pole=pole)
                for j, Mm in zip(ARM[side], sol[:4]):
                    Rk[:, j] = Mm
                Jk = fk(Rk, tr[kk], rest)[0]
                fd = (Jk[:, H.WRIST[side]] - Jk[:, e]).numpy()
            reach = np.maximum(reach, sol[4])
        return reach
    for _ in range(passes):
        solve(corr)
        bk = H.HandBody(matrix_to_rot6d(Rk), tr[kk], torch.zeros(N, 2, dtype=torch.bool), hand6d=matrix_to_rot6d(Rh))
        err = 0.0
        for side in ("left", "right"):
            gap = _thigh_gap(bk, side, list(range(N)))
            ok = np.isfinite(gap)
            dlt = np.where(ok, THIGH_REST["clear_m"] + lifts[side][kk] - gap, 0.0)
            corr[side] += dlt
            err = max(err, float(np.abs(dlt[lifts[side][kk] == 0]).max()))
        if err < 5e-4:
            break
    n = max(1, int(round(GOWERS_CORR_SMOOTH_S * rate)))
    if N > 2 * n:
        # the moving maximum taken about the correction's own trend (two moving means): open-body port, wave 6
        # follow-up, on the open body's thighs the correction falls steeply as the hands slide to the knees, and its
        # plain moving maximum lifted the planted hands past the 5 mm over THIGH_REST['clear_m'] their test allows
        for side in ("left", "right"):
            trend = _mean_edge(_mean_edge(corr[side], n), n)
            lo = np.lib.stride_tricks.sliding_window_view(np.pad(corr[side] - trend, (n, n), mode="edge"),
                                                          2 * n + 1).max(1)
            corr[side] = np.maximum(trend + _mean_edge(_mean_edge(lo, n), n), corr[side])
    reach = solve(corr)
    return Rk, reach, corr


@functools.lru_cache(maxsize=2)
def gowers_lateral(duration_s=GOWERS_DUR_S, rate=100.0, seed=GOWERS_SEED):
    """(lateral shift m, record): the smallest shift of the hands toward the outside of the thighs on the grid of
    GOWERS_LATERAL_STEP_M up to GOWERS_LATERAL_MAX_M (the palm centre moved along the thigh's lateral direction,
    _gowers_hands) whose rehearsal keeps the skins of GOWERS_SKIN_CHECKS GOWERS_SKIN_CLEAR_M apart: the hands
    solved at every motions.FOLD_CHECK_STEP-th knot of the hands on the thighs (_gowers_frame; the correction's
    smoothing on the same time base) and read at each of them (motions.fold_gaps); the knots where an earlier try
    failed are read first, so a try that fails there stops (task sh1c). Cached."""
    from .motions import FOLD_CHECK_STEP, fold_gaps
    fr = _gowers_frame(duration_s, rate, seed)
    t = fr["t"]
    ks = np.flatnonzero(t < fr["plan"]["lift_off"][1] + 1e-9)[::FOLD_CHECK_STEP]
    N = len(ks)
    Rh = matrix_to_rot6d(gowers_hand()[None].expand(N, -1, -1, -1).clone())
    tried, first = [], []
    for i in range(int(round(GOWERS_LATERAL_MAX_M / GOWERS_LATERAL_STEP_M)) + 1):
        lat = round(i * GOWERS_LATERAL_STEP_M, 6)
        Rk, reach, _ = _gowers_hands(fr["R"], fr["tr"], ks, fr["alongs"], fr["lifts"], rate=rate / FOLD_CHECK_STEP,
                                     lateral_m=lat)
        b = H.HandBody(matrix_to_rot6d(Rk), fr["tr"][ks], torch.zeros(N, 2, dtype=torch.bool), hand6d=Rh)
        rec = {"lateral_m": lat}
        if first:
            g, _ = fold_gaps(b, first, None, GOWERS_SKIN_CHECKS)
            least = min(float(v.min()) for v in g.values())
            if least < GOWERS_SKIN_CLEAR_M:
                tried.append(dict(rec, least_mm_first_read=1000.0 * least, knots_read=len(first)))
                continue
        g, _ = fold_gaps(b, np.arange(N), None, GOWERS_SKIN_CHECKS)
        rec.update({c: {"min_mm": 1000.0 * float(v.min()), "at_s": float(ks[int(np.argmin(v))] / rate)}
                    for c, v in g.items()}, knots_read=N, reach_max=float(reach.max()))
        tried.append(rec)
        if min(float(v.min()) for v in g.values()) >= GOWERS_SKIN_CLEAR_M:
            return lat, {"lateral_m": lat, "tried": tried, "clear_mm": 1000.0 * GOWERS_SKIN_CLEAR_M,
                         "checks": list(GOWERS_SKIN_CHECKS), "step_knots": FOLD_CHECK_STEP}
        first = sorted({int(np.argmin(v)) for v in g.values() if float(v.min()) < GOWERS_SKIN_CLEAR_M} | set(first))
    raise RuntimeError(f"no lateral shift of the Gowers hands keeps their skins apart: {tried}")


def gowers_skin_check(b, k_end):
    """{check: {'min_mm', 'at_s'}, 'n_knots', 'ok'}: the skin gaps of GOWERS_SKIN_CHECKS on the Gowers body
    (motions.fold_gaps, negative inside) at every motions.FOLD_CHECK_STEP-th knot up to k_end (the end of the final
    extension: the hands on the thighs, their lift-off and the arms' swing to hanging) and k_end; 'ok' when no skin
    part enters another (review of task sh1c: the fingertips of the two hands met at the midline and the hands and
    forearms entered the hip crease, with no check refusing it)."""
    from .motions import FOLD_CHECK_STEP, fold_gaps
    ks = np.unique(np.r_[np.arange(0, k_end + 1, FOLD_CHECK_STEP), k_end])
    g, _ = fold_gaps(b, ks, None, GOWERS_SKIN_CHECKS)
    out = {c: {"min_mm": 1000.0 * float(v.min()), "at_s": float(ks[int(np.argmin(v))] / b.rate)}
           for c, v in g.items()}
    out.update(n_knots=int(len(ks)), ok=bool(all(float(v.min()) >= 0.0 for v in g.values())))
    return out


def gowers_body(duration_s=GOWERS_DUR_S, rate=100.0, seed=GOWERS_SEED):
    """(HandBody, info, arrays) of Gowers' manoeuvre when arising (module docstring): on the seated and standing
    poses of the 'normal' template (the feet planted where it plants them), tracks of the hip midpoint's height, the
    pelvis's pitch and the spine's flexion (minimum-jerk between GOWERS_* keys; in each hand step held while the hand
    moves, then extended while both hands push); the hip midpoint's forward position seated, sliding GOWERS_SCOOT_M
    forward on the seat over the forward lean until lift-off, then solved per knot (GOWERS_COM_ITERS secant passes) so that the centre of mass (de Leva, the
    22 joints) follows a minimum-jerk ramp from its lift-off value to the template's standing value: a slow rise
    balanced over the feet, the hands on the thighs pushing on the body itself (no external support); the legs
    re-solved to the planted ankles (place_pelvis); the hands on the thighs (_gowers_hands; the collars protracted
    by THIGH_REST['protract_deg']), at the shares of the way from hip to knee of the plan, raised by
    GOWERS_STEP_LIFT_M while a hand moves up (the contact posed, not simulated); the hands leave the thighs at the
    end of the last step (the trunk within GOWERS_RELEASE_MAX_DEG of its seated start) and the final extension blends
    the arms to the hanging 'stand' arms; the background layer of its level 0's series (SERIES['arise']); the
    standing sway at the end. arrays:
    {'alongs', 'lifts', 'ik_knots'}."""
    fr = _gowers_frame(duration_s, rate, seed)
    R, tr, t, plan, steps = fr["R"], fr["tr"], fr["t"], fr["plan"], fr["steps"]
    alongs, lifts, bg, stand_w = fr["alongs"], fr["lifts"], fr["bg"], fr["stand_w"]
    T = len(t)
    f0, f1 = plan["final"]
    Rh = gowers_hand()[None].expand(T, -1, -1, -1).clone()
    fl = plan["lift_off"][1]
    kk = np.flatnonzero(t < fl + 1e-9)          # the hands on the thighs, then lifted off them (GOWERS_RELEASE_LIFT_M)
    lat, lat_rec = gowers_lateral(duration_s, rate, seed)
    Rk, reach, _ = _gowers_hands(R, tr, kk, alongs, lifts, lateral_m=lat)
    k_lift, target, stand_com = fr["k_lift"], fr["target"], fr["stand_com"]
    R[kk] = Rk
    # the final extension: the arms blend to the hanging 'stand' arms
    stand = posture_rotations("stand")
    k_off = int(kk[-1])
    rows = _arm_joints()
    n_rel = T - k_off - 1
    w = _mj((np.arange(n_rel) + 1) / ((f1 - fl) * rate))
    R[k_off + 1:, rows] = _blend(R[k_off:k_off + 1, rows].expand(n_rel, -1, -1, -1).clone(),
                                 stand[rows][None].expand(n_rel, -1, -1, -1).clone(), w)
    Rh[k_off + 1:] = _blend(Rh[k_off + 1:], relaxed_hand()[None].expand(n_rel, -1, -1, -1).clone(), w)
    _collar_breath(R, _inside_ramp(t >= f1, int(0.3 * rate)) * BREATH_COLLAR_DEG * bg["breath"])
    st = np.flatnonzero(stand_w > 0)
    R[st], tr[st] = stand_sway(R[st], tr[st], bg["sway_z"][st], bg["sway_x"][st])
    b = H.HandBody(matrix_to_rot6d(R), tr, torch.zeros(T, 2, dtype=torch.bool), rate=rate, hand6d=matrix_to_rot6d(Rh))
    b = H.HandBody(b.rot6d, b.transl, contact_labels(b), b.betas, b.gender, b.rate, b.hand6d)
    info = {"phases": {k: list(v) for k, v in plan.items()},
            "steps": [{"side": s, "start_s": a, "moved_s": a + GOWERS_STEP_MOVE * (e - a), "end_s": e,
                       "from_share": fr, "to_share": to} for s, a, e, fr, to in steps],
            "lean": dict(GOWERS_LEAN), "push": dict(GOWERS_PUSH), "climb": dict(GOWERS_CLIMB),
            "step_lift_m": GOWERS_STEP_LIFT_M, "pole": list(GOWERS_POLE), "liftoff_knot": k_lift,
            "hand_lateral_m": lat, "hand_lateral": lat_rec,
            "com_target_m": [float(target[k_lift]), stand_com], "reach_max": float(reach.max()),
            "hand_clear_m": THIGH_REST["clear_m"], "hands_released_s": float(f0), "hands_off_s": k_off / rate,
            "release_lift_m": GOWERS_RELEASE_LIFT_M, "trunk_shares": list(GOWERS_TRUNK_SHARES),
            "knees_steps": GOWERS_KNEES_STEPS, "step_move_share": GOWERS_STEP_MOVE,
            "contact": "the hands on the thighs posed by arm IK, not simulated (no force)",
            "background": background_record(SERIES["arise"], T, 1.0, rate,
                                            dict(TRANSFER_PARTS, trunk="seated knots before the forward lean only, "
                                                 "ramped over 0.3 s (spine1)", sway="standing after the final "
                                                 "extension, weighted in over 0.5 s (the feet planted)",
                                                 collars="under the hanging arms after the final extension"))}
    return b, info, {"alongs": alongs, "lifts": lifts, "ik_knots": kk}


def measure_gowers(body, k_lift, k_rel, k_final=None):
    """Gowers' manoeuvre read back: per hand, the wrist's share of the way from hip to knee projected on the thigh's
    axis per knot (the hand path up the thigh) and the hand's lowest vertex off the thigh's front (_thigh_gap) at
    every 5th knot while the hands are on the thighs, and from the release k_rel to k_final (the end of the final
    extension, when given: the hands must not pass through the thighs as they leave; inf once a hand is off its
    thigh); the trunk lean (the pelvis-to-neck line, forward positive, from its seated start) at lift-off, its peak
    and at the release; the pelvis's rise; the extrapolated centre of mass's margins (motions.balance, the base of
    support at lift-off) from lift-off."""
    from .motions import balance
    J = body.joints().numpy()
    out = {}
    ks = list(range(0, k_rel + 1, 5))
    kf = list(range(k_rel, int(k_final) + 1, 5)) if k_final is not None else []
    for side, h, kn, w in (("left", 1, 4, 20), ("right", 2, 5, 21)):
        ax = J[:, kn] - J[:, h]
        out[f"{side}_along"] = (((J[:, w] - J[:, h]) * ax).sum(-1) / (ax ** 2).sum(-1)).tolist()
        out[f"{side}_gap_m"] = _thigh_gap(body, side, ks).tolist()
        if kf:
            out[f"{side}_final_gap_m"] = _thigh_gap(body, side, kf).tolist()
    out["gap_knots"] = ks
    if kf:
        out["final_gap_knots"] = kf
        out["final_gap_min_m"] = float(min(min(out["left_final_gap_m"]), min(out["right_final_gap_m"])))
    tv = J[:, 12] - J[:, 0]
    lean = np.degrees(np.arctan2(tv[:, 2], tv[:, 1]))
    bl = balance(body, k_lift)
    out.update({"trunk_lean_liftoff_deg": float(lean[k_lift] - lean[0]),
                "trunk_lean_peak_deg": float(lean.max() - lean[0]),
                "trunk_lean_release_deg": float(lean[k_rel] - lean[0]),
                "pelvis_rise_m": float(J[-1, 0, 1] - J[0, 0, 1]),
                "posterior_margin_min_m": float(bl["posterior"][k_lift:].min()),
                "anterior_margin_min_m": float(bl["anterior"][k_lift:].min())})
    return out


def gowers(duration_s=GOWERS_DUR_S, seed=GOWERS_SEED):
    """(HandBody, truth) of Gowers' manoeuvre when arising (gowers_body) with its checks and read-back."""
    from .motions import MOS_MIN_M, arise_validity
    b, info, arr = gowers_body(duration_s, seed=seed)
    val = arise_validity(b)
    val["hand_limits"] = H.hand_limit_report(b.hand_R())
    val["continuity"] = continuity_report(b)
    J = b.joints().numpy()
    val["ankle_move_max_m"] = float(np.abs(J[:, [7, 8]] - J[:1, [7, 8]]).max())
    k_rel = int(round(info["hands_released_s"] * b.rate))
    k_fin = int(round(info["phases"]["final"][1] * b.rate))
    m = measure_gowers(b, info["liftoff_knot"], k_rel, k_fin)
    val["balance"] = {"ok": m["posterior_margin_min_m"] >= MOS_MIN_M - 1e-3 and m["anterior_margin_min_m"] > 0.0,
                      "posterior_margin_min_m": m["posterior_margin_min_m"],
                      "anterior_margin_min_m": m["anterior_margin_min_m"], "mos_min_m": MOS_MIN_M,
                      "rule": "feet only: the hands push on the body itself"}
    gaps = np.array(m["left_gap_m"] + m["right_gap_m"])
    val["hand_thigh"] = {"gap_min_m": float(gaps[np.isfinite(gaps)].min()), "reach_max": info["reach_max"],
                         "final_gap_min_m": m["final_gap_min_m"], "posed": True,
                         "touch_mm": {sd: float(1000.0 * np.min(np.asarray(m[f"{sd}_gap_m"])[
                             np.isfinite(m[f"{sd}_gap_m"])])) for sd in ("left", "right")}}
    #   touch_mm: each hand's smallest gap above its thigh's skin on the posed mesh while the hands are on the thighs
    #   (measure_gowers), the clip's declared touch as the skeleton page reads it (render_skeleton_catalogue
    #   MESH_TOUCHES: the rigid thigh stand-in reads the planted palms 27 mm off at the deep flexion; plan phase E)
    val["hand_thigh"]["ok"] = val["hand_thigh"]["gap_min_m"] > 0.0 and m["final_gap_min_m"] > 0.0
    val["release"] = {"trunk_lean_deg": m["trunk_lean_release_deg"], "max_deg": GOWERS_RELEASE_MAX_DEG,
                      "ok": m["trunk_lean_release_deg"] <= GOWERS_RELEASE_MAX_DEG}
    val["fold"] = gowers_skin_check(b, k_fin)
    p = dict(info, duration_s=float(duration_s), seat_height_m=0.43, remeasured=m,
             unsourced="every GOWERS_* value is a module value; no kinematics of Gowers' manoeuvre was read")
    return b, truth_record("gowers", seed, "present", "Gowers' manoeuvre (no scale item)", p,
                           ("inkster2004", "li2021", "hof2005", "deleva1996", "module"), val,
                           arrays={"left_along_design": arr["alongs"]["left"],
                                   "right_along_design": arr["alongs"]["right"]})


# ---------------------------------------------------------------- clip specifications
def _desc_arise(variant):
    return {"g0": "one rise at the control pace (1.89 s), arms folded",
            "g1_slow": (f"one rise in {SLOW_RISE_S:g} s (against 1.89 s): a hesitant forward lean and a slow "
                        "extension, the trunk leaning no more than at the control pace, arms folded"),
            "g1_attempts": "one failed attempt that falls back, then a rise at the control pace, arms folded",
            "g2": "one failed folded attempt, the hands to the chair arms, one push-off rise without difficulty",
            "g3": "one failed folded attempt, a push-off that falls back, a second push-off rise",
            "g4": ("two failed folded attempts, two push-offs that fall back; unable to arise without help (the "
                   "examiner's help is not shown)")}[variant]


# the 3.9 level 0 is a healthy rise, and the level 0 of the abrupt arising (AP03) and of Gowers' manoeuvre (G10) too:
# its operator names it so (task sh1b; the ladder's operator, 'parkinsonian', read as a parkinsonian rise beside them)
ARISE_LEVEL0_OPERATOR = "arising from a chair, healthy (procedural, seat 43 cm, arms folded)"


def _arise_specs():
    out = []
    l0 = "park_E7_arise_g0"
    for v, g in ARISE_GRADES.items():
        def build(v=v):
            return arise_grade(v)
        lab = {"g1_slow": "grade 1 (slow)", "g1_attempts": "grade 1 (more than one attempt)"}.get(v, f"grade {g}")
        out.append(ClipSpec(f"park_E7_arise_{v}", "parkinsonism", "E7", "3.9",
                            ARISE_LEVEL0_OPERATOR if v == "g0" else
                            "arising from a chair, parkinsonian (procedural, seat 43 cm)", g,
                            f"MDS-UPDRS 3.9 {lab}" if g else "MDS-UPDRS 3.9 grade 0 (level 0)", _desc_arise(v),
                            ("front", "left"), None if v == "g0" else l0, "chair_arise", build, since=SINCE,
                            note="grade 4 shows the failed attempts only: the help of the form is not shown"
                            if v == "g4" else ""))
    return out


def _spont_specs():
    out = []
    for base_name, seg in (("sit", "E1"), ("stand", "E7")):
        l0 = f"park_{seg}_spontaneity_g0"
        for g in range(5):
            def build(g=g, base_name=base_name):
                return spontaneity(base_name, g)
            n = spontaneity_count(g, SPONT_DUR[base_name])
            out.append(ClipSpec(f"park_{seg}_spontaneity_g{g}", "parkinsonism", seg, "3.14",
                                f"global spontaneity of movement, {'seated' if base_name == 'sit' else 'standing'}",
                                g, f"MDS-UPDRS 3.14 grade {g} (proposed ladder)",
                                f"{n} spontaneous movement{'' if n == 1 else 's'} in {SPONT_DUR[base_name]:g} s, "
                                f"amplitude x {spont_amp_ratio(base_name) ** g:.2f}"
                                + (f" (each still moving the body by {100 * SPONT_VISIBLE_M:g} cm or more)"
                                   if base_name in SPONT_CHAIN_BASES and g else "") + f", movement time x {SPONT_SLOW_RATIO ** g:.2f}, head "
                                f"and finger drift x {BG_RATIO ** g:.2f}, breathing and "
                                f"{'seated' if base_name == 'sit' else 'standing'} sway as at grade 0",
                                ("front", "left"), None if g == 0 else l0, "chair" if base_name == "sit" else None,
                                build, since=SINCE))
    return out


def _sequence_specs():
    out = []
    for task, seg, title, views in (("finger_tapping", "E4aL", "finger tapping, 15 s run",
                                     ("front", "hand_left", "hand_left_medial")),
                                    ("pronation_supination", "E4L", "pronation-supination, 30 cycles",
                                     ("front", "hands_top", "hands"))):
        short = "ft" if task == "finger_tapping" else "ps"
        l0 = f"park_{seg}_sequence_{short}_level0"
        for lv in range(4):
            def build(task=task, lv=lv):
                return sequence_effect(task, lv)
            s = SEQUENCE_SLOPES[lv]
            gt = ("level 0 (the controls' slight decrement, Ling et al. 2012)" if lv == 0 else
                  f"sequence effect level {lv}" + (" (PD OFF mean, Ling et al. 2012)" if lv == 1 else
                                                    " (proposed)"))
            out.append(ClipSpec(l0 if lv == 0 else f"park_{seg}_sequence_{short}_level{lv}", "parkinsonism", seg,
                                "sequence effect", f"sequence effect, {task.replace('_', '-')}, left", lv, gt,
                                (f"amplitude falls {100 * s:.2f} percent of the first per cycle" +
                                 (" (linear)" if lv < 2 else " (least squares; fast, then a plateau)") +
                                 f"; cycles vary (period CV {100 * SEQ_PERIOD_CV:g}, amplitude CV "
                                 f"{100 * SEQ_AMP_CV:g} percent)" +
                                 (f"; {SEQ_HESITATIONS[lv]} brief hesitations" if lv in SEQ_HESITATIONS else "")),
                                views, None if lv == 0 else l0, "chair", build, since=SINCE, segment_title=title))
    return out


FREEZE_VIEWS = ("front", "hand_left", "hand_left_medial")     # the medial view shows the trembling (review)


def _freezing_specs():
    out = []

    def lvl0(plan):
        from .handtasks import hand_task
        from .motions import posture_body
        base = posture_body("sit_finger_tapping", freeze_duration_s(plan))
        return hand_task(base, "finger_tapping", "left", n_cycles=FREEZE_PLANS[plan]["cycles"],
                         rate_hz=freeze_rate_hz(plan), decrement_onset=None, seed=SEED)
    out.append(ClipSpec("park_E4a_ulfreeze_level0", "parkinsonism", "E4a", "3.4", "finger tapping, left", 0,
                        "level 0", "10 cycles at the cited normal rate", FREEZE_VIEWS, None, "chair",
                        functools.partial(lvl0, "3.4_g3"), since=SINCE))
    out.append(ClipSpec("park_E4aL_ulfreeze_level0", "parkinsonism", "E4aL", "upper-limb freezing",
                        "finger tapping, left", 0, "level 0", "50 cycles at the 3.33 Hz of the controls of Ling et "
                        "al. 2012", FREEZE_VIEWS, None, "chair", functools.partial(lvl0, "long"),
                        since=SINCE, segment_title="finger tapping, 15 s run"))
    for plan, title in (("3.4_g3", None), ("long", "finger tapping, 15 s run")):
        def build(plan=plan):
            return upper_limb_freezing(plan)
        if plan == "3.4_g3":
            out.append(ClipSpec("park_E4a_ulfreeze_g3", "parkinsonism", "E4a", "3.4",
                                "upper-limb freezing in finger tapping, left", 3,
                                "MDS-UPDRS 3.4 grade 3 (a freeze in ongoing movement)",
                                f"hastening over {HASTEN_CYCLES} cycles, amplitude to {100 * BREAKDOWN_SHARE:g} "
                                f"percent, a {FREEZE_S:g} s freeze trembling at {TREMBLE_HZ:g} Hz",
                                FREEZE_VIEWS, "park_E4a_ulfreeze_level0", "chair", build, since=SINCE))
        else:
            out.append(ClipSpec("park_E4aL_ulfreeze_long", "parkinsonism", "E4aL", "upper-limb freezing",
                                "upper-limb freezing in finger tapping, left", "level",
                                "two freezes in a 15 s run (no scale grade)",
                                "each after hastening and an amplitude breakdown", FREEZE_VIEWS,
                                "park_E4aL_ulfreeze_level0", "chair", build, since=SINCE, segment_title=title))
    return out


SITDOWN_TITLE = "sitting down onto the chair"
APPLAUSE_TITLE = "applause test (the examiner's 3 claps not shown)"


def _motor_state_specs():
    off, on = G.tremor_level_cm(MS_GRADES["off"]), G.tremor_level_cm(MS_GRADES["on"], "below_floor")
    l0 = "park_E1_motorstate_level0"
    out = [ClipSpec(l0, "parkinsonism", "E1", "3.17", "seated rest, the left hand resting beyond the chair arm", 0,
                    "level 0 (no tremor)", "the seated rest with breathing and drift, no tremor", ("front", "hands"),
                    None, "chair", functools.partial(motor_state, "level0"), since=SINCE)]
    for d, cid, a, b, txt in (("off_to_on", "park_E1_motorstate_offon", off, on, "OFF to ON"),
                              ("on_to_off", "park_E1_motorstate_onoff", on, off, "ON to OFF (wearing-off)")):
        ga, gb = (MS_GRADES["off"], MS_GRADES["on"]) if d == "off_to_on" else (MS_GRADES["on"], MS_GRADES["off"])
        out.append(ClipSpec(cid, "parkinsonism", "E1", "3.17", "rest tremor, left hand, through a motor state change",
                            f"{ga} to {gb}", f"MDS-UPDRS 3.17 grade {ga} to grade {gb}: {txt} (the time course "
                            "compressed, module)",
                            f"{MS_HZ:g} Hz at the wrist: {a:.2f} cm at the middle fingertip for {MS_HOLD_S:g} s, "
                            f"{'falling' if a > b else 'rising'} over {MS_RAMP_S:g} s to {b:.2f} cm",
                            ("front", "hands"), l0, "chair", functools.partial(motor_state, d), since=SINCE))
    return out


def _gowers_specs():
    return [ClipSpec("park_E7_gowers", "gait", "E7", "Gowers' manoeuvre",
                     "arising from a chair with Gowers' manoeuvre (procedural, seat 43 cm)", "present",
                     "Gowers' manoeuvre (catalogue G10; no scale item)",
                     "a deep forward lean, the hands near the knees, the buttocks rising first; the hands climb the "
                     f"thighs hand over hand ({len(GOWERS_STEPS)} steps), each push straightening the trunk, and "
                     f"leave the proximal thighs with the trunk still bent (at most {GOWERS_RELEASE_MAX_DEG:g} deg); "
                     "the hand-thigh contact posed, not simulated", ("front", "left"), "park_E7_arise_g0",
                     "chair_arise", gowers, since=SINCE)]


def _applause_specs():
    out = []
    for sc, n in CLAP_COUNTS.items():
        rng_txt = {0: "3 claps", 1: "4 claps", 2: "5 to 10 claps", 3: "more than 10 claps"}[sc]
        gt = ("level 0: 3 claps, as demonstrated (negative)" if sc == 0 else
              f"applause sign, score {sc}: {rng_txt} (positive above 3, Dubois 2005; graded scoring not traced)")
        out.append(ClipSpec(f"park_E21_applause_g{sc}", "atypical parkinsonism", "E21", "applause test",
                            "applause test, seated (procedural)", sc, gt,
                            f"{n} claps at {CLAP_HZ:g} per second (module), palms {100 * CLAP_OPEN_M:g} cm apart at "
                            "the top, meeting palm to palm (the contact posed, 2 mm)", ("front", "hands"),
                            None if sc == 0 else "park_E21_applause_g0", "chair", functools.partial(applause, sc),
                            since=SINCE, segment_title=APPLAUSE_TITLE))
    return out


def _ap03_specs():
    out = [ClipSpec("park_E7_arise_abrupt", "atypical parkinsonism", "E7", "PSPRS arising",
                    "arising from a chair, abrupt (procedural, seat 43 cm)", "present",
                    "abrupt arising (catalogue AP03; PSPRS wording not verified, no grade)",
                    f"one rise in {ABRUPT_RISE_S:g} s (against 1.89 s), arms folded; the trunk leans less, the speed "
                    "keeps the extrapolated centre of mass inside the feet; a brisk rise only: the backward "
                    "instability after the rise is not shown", ("front", "left"), "park_E7_arise_g0",
                    "chair_arise", abrupt_arise, since=SINCE)]
    out.append(ClipSpec("park_E22_sitdown_level0", "atypical parkinsonism", "E22", "PSPRS sitting down",
                        "sitting down onto a chair (procedural, seat 43 cm, arms folded)", 0,
                        "level 0 (controlled: the rise read backward)",
                        "a rise at the control pace, standing, then the rise read backward on a slower clock (solved: "
                        "the extrapolated centre of mass stays over the feet): the trunk leans forward as the hips go "
                        "back, the descent slows to the seat", ("front", "left"), None,
                        "chair_arise", functools.partial(sitdown, "controlled"), since=SINCE,
                        segment_title=SITDOWN_TITLE))
    out.append(ClipSpec("park_E22_sitdown_enbloc", "atypical parkinsonism", "E22", "PSPRS sitting down",
                        "sitting down en bloc (procedural, seat 43 cm, arms folded)", "present",
                        "sitting down en bloc (catalogue AP03; PSPRS wording not verified, no grade)",
                        f"the rise read backward to {100 * BLOC_U:g} percent, then a drop with the trunk upright: the "
                        f"hips reach the seat at {BLOC_IMPACT_M_S:g} m/s, stopped within "
                        f"{100 * SEAT_CONTACT_M:g} cm (module values); the extrapolated centre of mass is behind the "
                        "heels from the start of the descent: a fall back into the seat", ("front", "left"),
                        "park_E22_sitdown_level0",
                        "chair_arise", functools.partial(sitdown, "en_bloc"), since=SINCE,
                        segment_title=SITDOWN_TITLE))
    return out


NOT_RENDERED = {
    "3.9 moving forward in the chair": ("the grade-1 'moves forward in the chair', and the protocol's move forward "
                                        "before the arms in grades 3 and 4, are not built: the arising templates "
                                        "(motions.arise) are solved for one seat depth and the chair is built at "
                                        "the first knot"),
    "3.9 grade 4 help": "the examiner's help is not shown (no examiner in the scene); grade 4 shows the failures",
    "3.14 leg crossing": ("crossing the legs or the ankles is not built (the contact of one leg on the other is not "
                          "solved); the seated set slides a foot forward and back, the standing set shifts the "
                          "weight"),
    "micrographia": ("not built: no pen, table or trace; a pen-less hand above no table is not a credible stimulus"),
    "upper-limb freezing grades": "no scale grades its own: the 10-cycle clip is MDS-UPDRS 3.4 grade 3 by arrest",
    "AP03 grades": ("the PSPRS wording of arising and sitting down was not read: the abrupt arising and the en bloc "
                    "sit-down carry no grade; their speeds and shares are module values"),
    "AP03 abrupt arising": ("a brisk rise with less forward lean only: the backward instability or retropulsion "
                            "after the rise is not shown (no post-rise balance response is built), so the clip alone "
                            "is close to a healthy quick rise; it reads against its level 0"),
    "AP01 demonstration": "the examiner's three claps are not shown (no examiner in the scene); the caption says so",
    "P33 other operators": ("only the rest tremor changes with the motor state: no peak-dose dyskinesia in ON, no "
                            "bradykinesia or freezing ramp; the time course is compressed into the clip"),
    "G10 from the floor": ("Gowers' manoeuvre is shown arising from a chair (catalogue G10, base E7), not from the "
                           "floor: no floor-rise template exists"),
}


def specs():
    """[ClipSpec] of this module (module docstring); level-0 partners included."""
    s = (_arise_specs() + _spont_specs() + _sequence_specs() + _freezing_specs() + _ap03_specs() +
         _applause_specs() + _motor_state_specs() + _gowers_specs())
    ids = [x.clip_id for x in s]
    if len(set(ids)) != len(ids):
        raise RuntimeError("duplicate clip ids")
    for x in s:
        if x.level0 is not None and x.level0 not in ids:
            raise RuntimeError(f"{x.clip_id}: level 0 {x.level0} missing")
    return s
