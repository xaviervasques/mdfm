"""The brief-jerk family of the synthetic-patient symptom catalogue (task CAT-JERK): myoclonus (cortical focal,
multifocal, generalized, rhythmic segmental, propriospinal, stimulus-sensitive, myoclonus-dystonia), negative
myoclonus (asterixis of the outstretched hands, postural lapses of the standing legs), motor tics (simple, complex,
in waxing and waning bouts), the startle reflex and hyperekplexia, and ankle clonus, as motion on the program's body
(docs/proposals/SYNTH_PATIENT_DRAFT.md: 22 joints, the 100 Hz clock, the shared neutral shape; the open body
Anny since the open-body port's wave 3, docs/OPEN_BODY_PORT_DESIGN.md section 14).

A jerk (the shared engine). One event moves one or more sites; a site is a joint group, an axis and an anatomical
sign (SITES). Each part of an event is a pulse of the site's angle: a minimum-jerk rise of rise_s from 0 to the
peak, an optional hold, a minimum-jerk return of fall_s, after a delay from the event onset (pulse); a site's angle
is the sum of its parts over the events. Every time is on a knot. Kinds of site:
- 'parent': the offset turns each joint of the group about a fixed axis of its parent's frame (a hinge fixed in the
  proximal segment: the elbow about the humerus's y axis, the collar, the knee and the ankle about their x axis), as
  tremor.py and dystonia.py do (a shrug turns the collar and the shoulder turns back under it, so that the arm keeps
  hanging, read against the table 'shrug' of limits_ext: KEEP_CHILD, limits_table);
- 'world': the offset turns each joint about an axis of the heading frame (x the body's left, y up, z its heading)
  through the joint, expressed in the joint's parent frame from the base's forward kinematics, so that a chain (neck,
  trunk) turns its end segment by the total about that axis; the anatomical signs of base.py hold for an upright
  segment (flexion +x, rotation to the left +y, lateral bending to the right +z) and a forward arm or palm-down hand
  rises about -x;
- 'pelvis': the pelvis turns forward about the line through the two hip joints (both hips stay where they are), the
  thighs keep their global rotations, so that the legs and feet do not move and the trunk flexes at the hips;
- 'dip': the pelvis is lowered by the site's value (m) with the trunk upright, the ankles and the feet held still,
  the legs solved by ik.hinge_leg (knees bending forward), the foot's global rotation kept: a knee buckle or a
  crouch of a standing body with both feet planted.
A chain splits its offset in proportion to each joint's room (body.room_deg) in that direction at a reference knot
of the base. Knots outside every event keep their base rows bit for bit.

Standing (both feet planted, task FIX-jerks). After every site, the hip strategy: the pelvis is moved backward along
the heading (the legs solved again by ik.hinge_leg, ankles and feet still) so that the whole-body centre of mass
(de Leva table, kernels/force.py) keeps its base position along the heading at every knot (_hold_com); a trunk
flexion jerk then pushes the hips back instead of pivoting about hips fixed in space. The balance block of the truth
(balance_block) checks the result: the extrapolated centre of mass (motions.balance, Hof et al. 2005) at least
BALANCE_MARGIN_M inside the base of support, the friction the floor must supply (horizontal over vertical ground
force of the centre-of-mass acceleration) at most FRICTION_MAX, and the vertical ground force at least GRF_MIN_BW body
weights; a clip that fails is refused. A dip (minimum-jerk) falls and rises at most at DIP_ACCEL_MAX: its rise and
return last at least sqrt(MJ_PEAK_ACC x depth / DIP_ACCEL_MAX).

Lapses (negative myoclonus, shape 'lapse'). The posture is lost ballistically: a constant acceleration from rest over
the silent period (the drop: support or tone gone, at most the gravity bound of the segment), then a constant
deceleration to rest at the lapse depth (the catch, at the catch acceleration), then a minimum-jerk return (lapse_timing,
lapse_pulse); a lapse of the legs drops at most at LAPSE_ACCEL_MAX and returns no faster than DIP_ACCEL_MAX allows.

Duration (JERK_TYPES, sources in SOURCES). The published quantity is the EMG burst (or, for negative myoclonus, the
silent period). The movement's rise lasts the burst plus TWITCH_S (the force of a twitch outlasts its burst; a
module value, no source read gives the burst-to-movement relation), its return RETURN_RATIO times the rise (module);
a negative jerk (a lapse) falls ballistically over the silent period, is caught, and returns over RETURN_RATIO times
the silent period (Lapses, below). The burst of a type is the geometric centre of its published band (a module reading
on a log scale), rounded to a knot. Every rotation turns at most at PEAK_SPEED_CAP_DEG_S (jerk_part lengthens its
rise); the segmental burst is the lower end of its band (SEGMENTAL_BURST_S, with a passive return SEGMENTAL_RETURN_S).

Onsets (the point processes): 'poisson' (a Poisson process given its count: the onsets of n events of duration d with
gaps of at least min_gap are uniform order statistics in the free time, place_events); 'rhythmic' (a period with a
jitter, segmental myoclonus); 'stimulus' (each stated stimulus time plus a latency); 'action' (Poisson inside the
window of a held posture); 'bouts' (tics placed inside bouts of common.schedule, their amplitude waxing and waning over
each bout). A global refractory min_gap keeps events apart, so that a count read from the body is the count placed.

Grades. Myoclonus: the Unified Myoclonus Rating Scale (UMRS, Frucht et al. 2002) rates frequency and amplitude 0 to 4
per body region at rest and with action, stimulus sensitivity present or absent and negative myoclonus present and
its severity 0 to 3; its verbal anchors were not read (section structure read through a trial protocol), so the
physical levels are declared ladders: the frequency ladder takes the counts per 30 s that the asterixis grading of
Conn (rare 1 to 2, occasional 3 to 4, frequent 5 to 30, almost continuous above 30 flaps per 30 s) gives the same
words (UMRS_COUNT_30S, the centre of each band, geometric for the wide one, above 30 a module level), and the amplitude
ladder is a share of the room the joint-limit table leaves in the jerk's direction (UMRS_AMPLITUDE_SHARE, module,
the D3 denominator of the dystonic postures). Asterixis: the Conn grades by flaps per 30 s; lapse depth by the UMRS
negative-myoclonus severity (declared degrees). Tics: the Yale Global Tic Severity Scale (YGTSS, Leckman et al. 1989)
motor number (distinct tic types; orchestrated patterns at 4 and 5), intensity (amplitude share, declared) and
complexity (share of complex tics, declared); the YGTSS frequency anchors count days and hours, so frequency is the
video-based Modified Rush scale's (MRVS, Goetz et al. 1999, as tabulated by Riechmann et al. 2023: 1 to 20, 21 to 40,
41 to 60, above 60 tics per minute) and the body areas its eleven (MRVS_AREAS; eyes, nose and mouth not shown).
Startle: no scale; normal (habituating within 2 to 5 trials) against hyperekplexia (exaggerated, then a short
generalized stiffness), van der Veen et al. 2022. Each hyperekplexia (standing and seated) has for level 0 the normal
startle of its posture to the same sounds at the same times, which carries the same stimulus marks (the clinical
comparison is a normal startle to the same stimulus, not a body that hears nothing); each normal startle has the
stimulus-free body of its posture for level 0 (STARTLE_PAIRING), so that the series root, and with it the
physiological background of physio.py, does not change. Clonus: the SCATS clonus score (Benz et al. 2005: under 3 s,
3 to 10 s, above 10 s) at the cited frequency and excursion (Wallace et al. 2012).

What the body cannot show (said, not caricatured): the face, eyes and mouth (blinks, grimaces, the facial and
palatal myoclonus, the eye and face tics), the voice (phonic tics), the fingers of a plain Body, the examiner (the
stimulus is stated in the caption and the truth, with stimulus_marks for a viewer that burns them into the frames; the
hand that presses the knee down to load the forefoot of the clonus clips is not drawn), the EMG itself, and falls (no
ballistic body model: a jump tic and the fall of hyperekplexia are not built).

Clonus (catalogue row O01, its seated alternative): seated with the left forefoot loaded, the ball of the foot on the
floor and the heel raised (forefoot_rows: the foot turned about the foot joint, the hip on the seat, the leg solved by
ik.hinge_leg, the toes kept flat); the beats are the ankle's flexion about the loaded angle, solved knot by knot so
that the ankle's own angle follows the cited frequency and excursion, the heel bouncing and the knee bobbing with it.

Touching tics (shared change SH15 of docs/proposals/SYNTH_SHARED_CHANGES_DRAFT.md; the review of 2026-11-02 dropped
them for want of a path). A complex tic in which the right index touches the face (the YGTSS checklist lists touching
and tapping among the complex motor tics, SOURCES['ygtss_checklist']): a site of kind 'touch' (touch_<target>_<side>)
whose value is the share of reach.touch_path (the hand from the rest of 'sit' or 'stand' to a face target, the
pointing hand forming on the way, the touch kept between 0 and 3 mm from the skin); an event moves the share from 0
to 1 on a minimum-jerk profile whose fingertip peaks at reach.TOUCH_TIP_SPEED_M_S of the posture (touch_timing: the
longer reach from the hanging arm runs faster and still lasts longer than from the chair arm), holds it TOUCH_HOLD_S
(a touch) or taps TAP_N times (each lifting the fingertip TAP_DEPTH_M back along the approach line over TAP_OUT_S and
returning over TAP_IN_S, TAP_REST_S on the skin between), then back over TOUCH_RETURN_RATIO times the reach
(touch_parts; every value a module reading, no source read gives the timing of a touching tic). _apply_touch sets
the arm rows and the hand of that side (the body becomes a hands.HandBody) where the share is positive; the trunk,
neck, head and arms must hold the posture's rest there (touch_posture). The clips (build_touch_tics) mix the touches
with simple tics (TOUCH_TIC_TYPES) in the share of the family's complexity ladder (YGTSS_COMPLEX_SHARE), but their
complexity grade stops at TOUCH_COMPLEXITY_MAX: brief touches and taps could be rationalized as normal behaviour
(SOURCES['leckman1989'], anchor 3), not the sustained orchestrated bouts of anchor 4. The YGTSS number counts the
discrete tics shown, simple and touching, with enough simple types (TOUCH_TIC_TYPES) that a rater who counts every
touch as one 'touching' tic and every tap as one 'tapping' tic reads the same anchor. touch_readback reads the gap on
the skin, the clearance to the head, trunk and thighs, the taps and the speeds from the skinned mesh.

Re-measurement (measure_jerks, measure_clonus, measure_dips, tic_areas): read back from the body against its base;
count, onsets and durations from the joint angular velocity (runs of the speed of the deviation from the base above
a threshold, gaps shorter than merge_s joined, a run that starts at the base is a jerk), the peak deviation (deg) and
the end-point displacement (cm), the clonus frequency (measure.dominant_frequency) and beats, the lapse depth.
"""
import functools
import math

import numpy as np
import torch

from ..body.rotations import axis_angle_to_matrix, geodesic, matrix_to_rot6d, vee, yaw_angle
from ..body.joints import PARENTS
from . import limits_ext as LX
from .base import posture_rotations, static_body, yaw_matrix
from .body import (F64, LIMIT_TOL_DEG, Body, components_deg, contact_labels, floor_to_zero, rest_joints, room_deg,
                   rotation_angle_deg, validity)
from .common import as_rng, envelope, schedule, truth_record
from .measure import dominant_frequency
from .motions import minimum_jerk

RATE = 100.0
SINCE = "catalogue_v1"
SEED = 20261102
CLIP_S = 30.0               # [module] clip length: the 30 s window of the counting ladders
TWITCH_S = 0.04             # [module] the movement's rise outlasts the EMG burst by this
RETURN_RATIO = 1.5          # [module] return phase over rise phase
MIN_GAP_S = 0.10            # [module] the global refractory between events
MERGE_S = 0.05              # [module] velocity runs closer than this are one movement (measure_jerks)
SPEED_FLOOR_DEG_S = 0.1     # [module] the angular-speed threshold of measure_jerks (no noise in a synthetic body)

SOURCES = {
    "vanderveen2022": "van der Veen S, Caviness JN, Dreissen YEM, Ganos C, Ibrahim A, Koelman JHTM, Stefani A, Tijssen "
                      "MAJ. Myoclonus and other jerky movement disorders. Clin Neurophysiol Pract 2022; 7: 285-316, doi "
                      "10.1016/j.cnp.2022.09.003 (PMC9619152, full text read in parts): EMG discharges cortical usually "
                      "25-50 ms, subcortical and reticular 25-300 ms, myoclonus-dystonia mean 95 ms (25-256), upper limb "
                      "and trunk or neck, worse on action; spinal segmental 50-500 ms at 1-3 Hz (0.2-8); propriospinal "
                      "50-300 ms; negative myoclonus and asterixis silent period 50-200 ms; simple motor tics bursts about "
                      "200 ms, like voluntary movements; startle: sternocleidomastoid onset 58.3 ms (40.4-136, n 53), "
                      "craniocaudal spread, antagonists ending within about 400 ms of onset, habituation within 2-5 "
                      "trials, eye closure, grimace, neck and trunk flexion, slight arm abduction, elbow flexion, forearm "
                      "pronation; hyperekplexia: abnormal startle followed by a short generalized stiffness",
    "merchant2020": "Merchant SHI, Vial-Undurraga F, Leodori G, van Gerpen JA, Hallett M. Myoclonus: an "
                    "electrophysiological diagnosis. Mov Disord Clin Pract 2020; 7(5): 489-499, doi 10.1002/mdc3.12986 "
                    "(PMC7328418, Table 1 read): cortical under 50 ms, myoclonus-dystonia 50-100 ms, segmental 50-500 ms, "
                    "propriospinal about 150-450 ms, arrhythmic flexor jerks of trunk, hips and knees",
    "termsarasab2015": "Termsarasab P, Thammongkolchai T, Frucht SJ. Spinal-generated movement disorders: a clinical "
                       "review. J Clin Mov Disord 2015; 2: 18, doi 10.1186/s40734-015-0028-1 (as summarized by a web "
                       "search, not read): segmental myoclonus rhythmic or semi-rhythmic at about 1-2 Hz, one or two "
                       "limbs or the trunk",
    "frucht2002": "Frucht SJ, Leurgans SE, Hallett M, Fahn S. The Unified Myoclonus Rating Scale. Adv Neurol 2002; 89: "
                  "361-376 (not read; sections as described by the protocol of trial NCT05873062): rest and action "
                  "myoclonus frequency x amplitude per body region, 0-4 each; stimulus sensitivity present 1 or absent "
                  "0; negative myoclonus present 1 or 0, severity 0-3",
    "conn_asterixis": "asterixis grades 0-4 by flaps in 30 s (1-2 rare, 3-4 occasional, 5-30 frequent, above 30 almost "
                      "continuous) as tabulated in J Clin Diagn Res 2019 (jcdr.net article 13063) and attributed to Conn; "
                      "the original (Conn HO et al., Gastroenterology 1977; 72: 573-583) not read",
    "leckman1989": "Leckman JF, Riddle MA, Hardin MT, Ort SI, Swartz KL, Stevenson J, Cohen DJ. The Yale Global Tic "
                   "Severity Scale. J Am Acad Child Adolesc Psychiatry 1989; 28: 566-573, doi "
                   "10.1097/00004583-198907000-00015 (anchors paraphrased from the scale as widely reproduced, not read "
                   "in this task): motor number 1 single, 2 two to five, 3 above five discrete tics, 4 with at least one "
                   "orchestrated pattern, 5 with several frequent orchestrated paroxysms; intensity 1 not visible, 2 not "
                   "more forceful than a comparable voluntary act, 3 more forceful, 4 exaggerated, 5 extremely forceful; "
                   "complexity 0 to 5 by the share and length of complex and orchestrated tics: 2 complex tics that "
                   "mimic brief automatic behaviours such as grooming and could be camouflaged, 3 more purposive "
                   "tics, hard to camouflage, that could be rationalized as normal behaviour (picking, tapping), 4 "
                   "very complex tics in sustained orchestrated bouts that could not be rationalized as normal "
                   "behaviour, 5 lengthy bouts impossible to camouflage or rationalize",
    "ygtss_checklist": "the YGTSS motor tic symptom checklist (Leckman et al. 1989) as widely reproduced, not read in "
                       "this task: touching and tapping listed among the complex motor tics",
    "riechmann2023": "Riechmann R, Jakubovski E, Essing J, Haas M, Goetz CG, Stebbins GT, Mueller-Vahl KR. The Modified "
                     "Rush Video-Based Tic Rating Scale revised. Mov Disord Clin Pract 2023; 10: 802-810, doi "
                     "10.1002/mdc3.13713 (PMC10187019, full text): MRVS (Goetz et al. 1999, Mov Disord 14: 502-506) "
                     "body areas eleven (eyes, nose, mouth, neck, shoulders, arms, hands, trunk, pelvis, legs, feet), "
                     "0 none, 1 one or two, 2 three or four, 3 five or six, 4 seven or more; motor tic frequency 1-20, "
                     "21-40, 41-60, above 60 per minute; 102 patients: mean 53, median 52, range 7-150 per minute",
    "benz2005": "Benz EN, Hornby TG, Bode RK, Scheidt RA, Schmit BD. A physiologically based clinical measure for "
                "spastic reflexes in spinal cord injury. Arch Phys Med Rehabil 2005; 86: 52-59, doi "
                "10.1016/j.apmr.2004.01.033 (SCATS worksheet as read on the SCIRE project page): clonus after a rapid "
                "ankle dorsiflexion, 0 none, 1 under 3 s, 2 3 to 10 s, 3 above 10 s",
    "wallace2012": "Wallace DM, Ross BH, Thomas CK. Characteristics of lower extremity clonus after human cervical spinal "
                   "cord injury. J Neurotrauma 2012; 29(5): 915-924, doi 10.1089/neu.2010.1549 (PMC3303097): soleus "
                   "clonus 5.4 (SD 0.9) Hz, range 4.5-7.0, 14 subjects; ankle excursion mean 10 (SD 3) deg, range 5-15, "
                   "maximal 16 (SD 5) deg",
    "catalogue_m05": "docs/proposals/SYMPTOM_CATALOGUE_DRAFT.md, row M05 (spinal segmental myoclonus): burst 100 to "
                     "500 ms, rhythmic, usually 1 to 3 Hz, as the catalogue cites Latorre 2025 (not read in this task)",
    "catalogue_o01": "docs/proposals/SYMPTOM_CATALOGUE_DRAFT.md, row O01 (ankle clonus): seated with the forefoot "
                     "loaded, spontaneous clonus shown; elicited clonus needs an examiner",
    "table": "the anatomical joint-limit table (reports/openbody/joint_limits.json, jarm2022; open-body port, "
             "docs/OPEN_BODY_PORT_DESIGN.md section 12.1), the room of each jerk direction (decision D3)",
    "module": "a reading or default of this module, not taken from a source (stated where used)",
}

# EMG burst (or silent period) bands, ms, and their source; the burst of a type is the band's geometric centre
JERK_TYPES = {
    "cortical": {"band_ms": (25.0, 50.0), "negative": False, "source": "vanderveen2022"},
    "reticular": {"band_ms": (25.0, 300.0), "negative": False, "source": "vanderveen2022"},
    "myoclonus_dystonia": {"band_ms": (95.0, 95.0), "negative": False, "source": "vanderveen2022"},   # the mean
    "segmental": {"band_ms": (50.0, 500.0), "negative": False, "source": "vanderveen2022"},
    "propriospinal": {"band_ms": (50.0, 300.0), "negative": False, "source": "vanderveen2022"},
    "negative": {"band_ms": (50.0, 200.0), "negative": True, "source": "vanderveen2022"},
    "tic": {"band_ms": (200.0, 200.0), "negative": False, "source": "vanderveen2022"},
    "startle": {"band_ms": (25.0, 300.0), "negative": False, "source": "vanderveen2022"},   # reticular, as startle
}
SEGMENTAL_HZ = (1.0, 2.0)            # the overlap of SOURCES['vanderveen2022'] (1-3 Hz) and 'termsarasab2015' (1-2 Hz)
STARTLE_SCM_MS = 58.3                # SOURCES['vanderveen2022'], sternocleidomastoid onset, mean
STARTLE_SPREAD_MS = {"neck": 0.0, "shoulders": 10.0, "arms": 20.0, "trunk": 30.0, "legs": 50.0}   # [module] craniocaudal
STARTLE_END_MS = 400.0               # SOURCES['vanderveen2022']: the early response ends within about 400 ms
STIFFNESS_S = 1.0                    # [module] the short generalized stiffness after a hyperekplexic startle
HABITUATION = (1.0, 0.5, 0.25)       # [module] response factor per trial (habituation within 2-5 trials, cited)
HYPER_HABITUATION = (1.0, 1.0, 0.9)  # [module] hyperekplexia: poor habituation, the response kept over the trials
RETICULAR_SPREAD_MS = dict(STARTLE_SPREAD_MS, hands=20.0, pelvis=40.0, feet=50.0)   # [module] a reticular jerk spreads
#                                      from the neck and shoulders down, as the startle (catalogue row M04)
CLONUS_HZ = 5.4                      # SOURCES['wallace2012']
CLONUS_PP_DEG = 10.0                 # SOURCES['wallace2012'], mean excursion read as peak to peak
CLONUS_AMP_DRIFT = 0.10              # [module] the beats' amplitude drifts by up to +-10 percent
CLONUS_HZ_DRIFT = 0.2                # [module] their frequency by up to +-0.2 Hz (zero mean over the beating span)
CLONUS_TAPER_S = 0.2                 # [module] a sustained clonus still ends smoothly at the clip's end
SCATS_S = {1: (0.0, 3.0), 2: (3.0, 10.0), 3: (10.0, math.inf)}       # SOURCES['benz2005']
CONN_30S = {1: (1, 2), 2: (3, 4), 3: (5, 30), 4: (31, None)}       # SOURCES['conn_asterixis']
MRVS_PER_MIN = {1: (1, 20), 2: (21, 40), 3: (41, 60), 4: (61, None)}   # SOURCES['riechmann2023']
MRVS_AREA_BANDS = {1: (1, 2), 2: (3, 4), 3: (5, 6), 4: (7, None)}
MRVS_AREAS = ("eyes", "nose", "mouth", "neck", "shoulders", "arms", "hands", "trunk", "pelvis", "legs", "feet")
OPEN_BAND_FACTOR = 1.2               # [module] an open band (above n) is placed at 1.2 n


def _band_level(band):
    lo, hi = band
    if hi is None:
        return OPEN_BAND_FACTOR * lo
    return math.sqrt(lo * hi) if hi > 2 * lo else 0.5 * (lo + hi)


UMRS_COUNT_30S = {g: int(round(_band_level(b))) for g, b in CONN_30S.items()}    # 2, 4, 12, 37 (module reading)
UMRS_AMPLITUDE_SHARE = {1: 0.05, 2: 0.12, 3: 0.25, 4: 0.50}                     # [module] declared ladder
NEGATIVE_DEPTH_DEG = {1: 10.0, 2: 20.0, 3: 35.0}                              # [module] UMRS negative severity 1-3
LAPSE_DEPTH_CM = {1: 1.5, 2: 3.0, 3: 5.0}                                     # [module] leg lapses (bouncing)
YGTSS_INTENSITY_SHARE = {2: 0.12, 3: 0.25, 4: 0.45, 5: 0.70}                  # [module] declared ladder
YGTSS_NUMBER_TYPES = {1: 1, 2: 3, 3: 6}                                       # [module] inside the cited bands
YGTSS_COMPLEX_SHARE = {0: 0.0, 2: 0.2, 4: 0.5}                                # [module] declared ladder
ORCHESTRATED_STEP_S = 0.15                                                    # [module] delay between the parts
WRIST_EXTENSION_DEG = 40.0          # [module] the extended wrists of the asterixis posture (capped by the room)
CLONUS_STRETCH_DEG = 4.0            # [module] the ankle dorsiflexion of loading the forefoot (the knee pressed down)
#                                     before the beats
HEEL_UP_M = 0.05                    # [module] the loaded forefoot: the heel this far above its flat place (O01: 3-5 cm)
FOREFOOT_BACK_M = 0.04              # [module] the foot joint this far behind its place in 'sit' (knee near 96 deg)
DIP_PER_SHARE_M = 0.10              # [module] a myoclonic knee dip: the amplitude share x 10 cm (amplitude 3: 2.5 cm)
STIMULUS_LATENCY_S = 0.05           # [module] reflex myoclonus latency, of the order of cortical reflex latencies
STIMULUS_MARK_S = 0.2               # [module] how long a viewer shows a stimulus mark from the stimulus time
STIMULUS_TAPS = ((4.0, "left"), (9.0, "right"), (14.0, "left"), (19.0, "right"), (24.0, "left"))   # [module] the
#                                     taps of the stimulus-sensitive clips: time (s) and the tapped hand
AXIAL_WEIGHTS = {"trunk_flexion": 0.5, "hip_flexion": 0.5, "neck_flexion": 0.3}   # [module] an event of several
#                                     axial sites: the grade's share split so that the trunk, hips and neck add up
GENERALIZED_WEIGHTS = {"neck_flexion": 1.0, "trunk_flexion": 0.75, "hip_flexion": 0.5,
                       "shoulder_abduction_left": 0.1, "shoulder_abduction_right": 0.1}   # [module] the reticular jerk
#                                     led by the axial flexion, the arms' abduction slight (task FIX-jerks); the arms'
#                                     weight 0.5 before the open-body port's wave 3: the anatomical table leaves the
#                                     hanging arm 170 deg of abduction and the seated trunk 25 deg of flexion, and
#                                     at 0.5 the arms' 10.2 deg passed the neck's 7.2 and the trunk's 2.25 (grade a2)
FOCAL_WEIGHTS = {"elbow_flexion_left": 0.4}   # [module] the focal elbow jerk: 0.4 of the share (amplitude 4: 27 deg)
TIC_BOUT_COUNT = 24                 # [module] tics of the 60 s bout clip (MRVS 2 over the minute, about 49 per minute
#                                     inside the bouts)
SEGMENTAL_BURST_S = 0.10            # 100 ms: the lower end of the band of SOURCES['catalogue_m05'], inside the
#                                     50 to 500 ms of SOURCES['vanderveen2022'] (a module choice inside both)
SEGMENTAL_RETURN_S = 0.20           # [module] a passive return of 0.2 s (rise 0.14 s): moving 48 percent of the period
SIZE_SIGMA = 0.3                    # [module] wave SH1d part B (review M2): each jerk of a kind draws its own size, a
#                                     lognormal factor of median 1 and this log-sigma on the peaks of all its parts
SIZE_SIGMA_RHYTHMIC = 0.1           # [module] the same for the rhythmic segmental myoclonus and the startle trials
SIZE_CLIP = (0.5, 1.3)              # [module] the factor clipped to this range, then each peak kept under the room cap;
#                                     1.3 and not the plan's 1.6: at 1.43 the largest seated trunk jerk (propriospinal f1
#                                     a4) carried its arms faster than the background layer can anchor them (the sweep
#                                     refused it: acceleration added, anchor error 1.3 mm); at 1.2 and 1.3 it passes
RISE_JITTER = 0.2                   # [module] the rise (and the return with it) of a non-rhythmic jerk times a uniform
#                                     factor in 1 -+ this (before the speed cap of jerk_part)
ROOM_CAP_SHARE = 0.9                # [module] a jerk over a held posture stays under 0.9 of the room the posture leaves

# kinematic bounds (task FIX-jerks)
GRAVITY = 9.81
MJ_PEAK_ACC = 10.0 / math.sqrt(3.0)  # a minimum-jerk step of d over T peaks at 5.77 d / T^2 (acceleration)
MJ_PEAK_SPEED = 1.875                # and at 1.875 d / T (speed)
PEAK_SPEED_CAP_DEG_S = 800.0        # [module] a jerk of a limb joint turns at most at 800 deg/s (its rise lengthened)
DIP_ACCEL_MAX = 0.5 * GRAVITY       # [module] an active crouch or rise of the pelvis: at most half of g
LAPSE_ACCEL_MAX = 0.7 * GRAVITY     # [module] a lapse of the legs falls at most at 0.7 g (support partly kept)
LAPSE_CATCH_ACCEL = 0.5 * GRAVITY   # [module] and is caught at 0.5 g
LAPSE_ACCEL_DEG_S2 = {"wrist": (4500.0, 4500.0), "arm": (1200.0, 1200.0)}   # [module] (drop bound, catch) of a hand
#                                     (a relaxed hand falls under gravity at about 4000 to 5500 deg/s2) and of an arm
FINGER_LAPSE_SHARE = 0.75           # [module] the knuckles lapse by 0.75 of the wrist drop
FINGER_LAPSE_MAX_DEG = 15.0         # [module] at most 15 deg on average, each finger x U(0.67, 1.33) (10 to 20 deg)
FINGER_LAPSE_DELAY_S = (0.0, 0.01, 0.02)   # [module] each finger's lapse starts 0, 10 or 20 ms after the wrist's
BALANCE_MARGIN_M = 0.02             # [module] the extrapolated centre of mass at least 2 cm inside the feet
FRICTION_MAX = 0.6                  # [module] the friction the floor must supply, at most
GRF_MIN_BW = 0.2                    # [module] the vertical ground force, at least (body weights)


def burst_s(kind, rate=RATE):
    """EMG burst (or silent period) of a jerk type (s, on a knot): the geometric centre of its band."""
    lo, hi = JERK_TYPES[kind]["band_ms"]
    return round(math.sqrt(lo * hi) / 1000.0 * rate) / rate


def timing(kind, rate=RATE):
    """(rise_s, fall_s) of a jerk type (module docstring): rise = burst + TWITCH_S (a lapse: the silent period),
    return = RETURN_RATIO x rise, both on knots."""
    b = burst_s(kind, rate)
    rise = b if JERK_TYPES[kind]["negative"] else round((b + TWITCH_S) * rate) / rate
    return rise, round(RETURN_RATIO * rise * rate) / rate


# ---------------------------------------------------------------- sites
# name: (joints, kind, axis, sign, end joint for the displacement or None, MRVS body area)
SITES = {
    "neck_flexion": ((12, 15), "world", "x", 1.0, 15, "neck"),
    "neck_extension": ((12, 15), "world", "x", -1.0, 15, "neck"),
    "neck_rotation_left": ((12, 15), "world", "y", 1.0, 15, "neck"),
    "neck_rotation_right": ((12, 15), "world", "y", -1.0, 15, "neck"),
    "neck_tilt_left": ((12, 15), "world", "z", -1.0, 15, "neck"),
    "neck_tilt_right": ((12, 15), "world", "z", 1.0, 15, "neck"),
    "trunk_flexion": ((3, 6, 9), "world", "x", 1.0, 12, "trunk"),
    "trunk_rotation_left": ((3, 6, 9), "world", "y", 1.0, 12, "trunk"),
    "trunk_tilt_right": ((3, 6, 9), "world", "z", 1.0, 12, "trunk"),
    "hip_flexion": ((0,), "pelvis", "x", 1.0, 12, "pelvis"),
    "knee_dip": ((1, 4, 7, 2, 5, 8), "dip", None, 1.0, 0, "legs"),
    "shrug_left": ((13,), "parent", "z", 1.0, 16, "shoulders"),
    "shrug_right": ((14,), "parent", "z", -1.0, 17, "shoulders"),
    "shoulder_abduction_left": ((16,), "parent", "z", 1.0, 18, "arms"),
    "shoulder_abduction_right": ((17,), "parent", "z", -1.0, 19, "arms"),
    "elbow_flexion_left": ((18,), "parent", "y", -1.0, 20, "arms"),
    "elbow_flexion_right": ((19,), "parent", "y", 1.0, 21, "arms"),
    "elbow_extension_right": ((19,), "parent", "y", -1.0, 21, "arms"),
    "arm_raise_left": ((16,), "world", "x", -1.0, 20, "arms"),        # a forward arm rises about -x
    "arm_raise_right": ((17,), "world", "x", -1.0, 21, "arms"),
    "arm_drop_left": ((16,), "world", "x", 1.0, 20, "arms"),
    "arm_drop_right": ((17,), "world", "x", 1.0, 21, "arms"),
    "wrist_extension_left": ((20,), "world", "x", -1.0, None, "hands"),   # a palm-down hand rises about -x
    "wrist_extension_right": ((21,), "world", "x", -1.0, None, "hands"),
    "wrist_flexion_left": ((20,), "world", "x", 1.0, None, "hands"),
    "wrist_flexion_right": ((21,), "world", "x", 1.0, None, "hands"),
    "knee_extension_left": ((4,), "parent", "x", -1.0, 7, "legs"),      # knee flexion is +x (base.py)
    "knee_extension_right": ((5,), "parent", "x", -1.0, 8, "legs"),
    "ankle_dorsiflexion_left": ((7,), "parent", "x", -1.0, None, "feet"),   # dorsiflexion is -x (base.py)
    "ankle_dorsiflexion_right": ((8,), "parent", "x", -1.0, None, "feet"),
}
TOUCH_ARMS = {"left": (13, 16, 18, 20), "right": (14, 17, 19, 21)}     # collar, shoulder, elbow, wrist (reach.ARM)
for _side, _arm in TOUCH_ARMS.items():          # kind 'touch': the site's value is the share of a reach.touch_path
    for _target in ("nose", "cheek", "chin", "forehead"):                  # reach.FACE_TARGETS
        SITES[f"touch_{_target}_{_side}"] = (_arm, "touch", _target, 1.0, _arm[3], "arms")
LEGS = ((1, 4, 7, 10), (2, 5, 8, 11))
KEEP_CHILD = {"shrug_left": 16, "shrug_right": 17}
#   site: the child joint that keeps its global rotation (task sh1c): a shrug raises the shoulder girdle with the arm
#   hanging under it, the shoulder turned back by the collar's rise, its adduction read against the table 'shrug' of
#   limits_ext (the healthy table with the SH2 coupling: the adduction bound widened by the collar's rise over the
#   base; session decision of 2026-11-03, limits_ext.SHRUG_DECISION). Before it the dict was empty: the healthy
#   table's shoulder adduction bound lies just past the hanging arm of the procedural postures, so the arm moved
#   with the girdle, as the SMPL collar carries it, and swung out


def site_area(name):
    return SITES[name][5]


def _heading_axes(body, axis):
    """(T, 3) unit vector of a heading-frame axis ('x', 'y', 'z') at every knot (the root's yaw)."""
    psi = yaw_angle(body.local_R()[:, 0])
    u = torch.zeros(3, dtype=F64)
    u[{"x": 0, "y": 1, "z": 2}[axis]] = 1.0
    return torch.einsum("tij,j->ti", yaw_matrix(psi), u)


def site_axes(body, name, G=None):
    """{joint: (T, 3) unit axis in the joint's parent frame} of a 'parent' or 'world' site on a body."""
    joints, kind, axis, _, _, _ = SITES[name]
    if kind == "parent":
        u = torch.zeros(3, dtype=F64)
        u[{"x": 0, "y": 1, "z": 2}[axis]] = 1.0
        return {j: u.expand(body.T, 3) for j in joints}
    if kind != "world":
        raise ValueError(f"site {name}: kind {kind} has no joint axes")
    if G is None:
        _, G = body.fk()
    aw = _heading_axes(body, axis)
    return {j: torch.einsum("tji,tj->ti", G[:, PARENTS[j]], aw) for j in joints}


def site_rooms(body, name, knot):
    """[deg] the room of each joint of a rotation site in its direction at a knot of the base (body.room_deg); a
    site of KEEP_CHILD (a shrug) keeps the room of its joint where the child, turned back, stays inside the table
    'shrug' of limits_ext (kept_child_room)."""
    joints, kind, _, sign, _, _ = SITES[name]
    if kind in ("pelvis", "dip", "touch"):
        return None
    ax = site_axes(body, name)
    R = body.local_R()[knot]
    rooms = [room_deg(R[j], j, ax[j][knot], sign, "parent") for j in joints]
    if name in KEEP_CHILD:                  # the compensating child turns the other way about R_j^T a
        rooms[-1] = kept_child_room(R, joints[-1], KEEP_CHILD[name], ax[joints[-1]][knot], sign, rooms[-1])
    return rooms


def kept_child_room(R, j, c, axis, sign, room, iters=40):
    """deg: the largest offset (at most room) of joint j about axis (its parent's frame) in the direction sign, from
    the local rotations R (22, 3, 3) of a base knot, whose child c turned back (its global rotation kept) leaves every
    component of j and c inside the table 'shrug' of limits_ext at that knot (the child's adduction bound widened by
    the collar's rise over the base); room when the whole of it does, else by bisection (iters halvings)."""
    base = components_deg(matrix_to_rot6d(R)[None])

    def ok(deg):
        Rn = R.clone()
        Rn[j] = axis_angle_to_matrix(torch.as_tensor(axis, dtype=F64) * math.radians(sign * deg)) @ R[j]
        Rn[c] = Rn[j].transpose(-1, -2) @ R[j] @ R[c]
        comp = components_deg(matrix_to_rot6d(Rn)[None])
        return bool((LX.excess(LX.SHRUG, comp, base, rows=[j, c]) <= LIMIT_TOL_DEG).all())
    if ok(room):
        return float(room)
    lo, hi = 0.0, float(room)
    for _ in range(iters):
        m = 0.5 * (lo + hi)
        lo, hi = (m, hi) if ok(m) else (lo, m)
    return lo


def limits_table(sites):
    """The joint-limit table a clip's checks read (limits_ext): 'shrug' when a site of KEEP_CHILD moves (a shrug with
    the arm hanging under it), else the healthy table."""
    return LX.SHRUG if any(s in KEEP_CHILD for s in sites) else LX.HEALTHY


def jerk_validity(base, out, sites, feet=(0, 1)):
    """body.validity of out against base read against limits_table(sites) (limits_ext.check: unchanged for the
    healthy table; with a shrug its joint-limit counts at each knot against the table 'shrug' and the coupling
    report)."""
    return LX.check(limits_table(sites), base, out, feet=feet)


def _pelvis_room(body, knot):
    """deg: the forward pelvis turn about the hip line that keeps the hip joints (the thighs held) inside the table."""
    from .body import joint_limits
    lim = joint_limits()
    lo, hi = 0.0, 90.0
    for _ in range(30):
        m = 0.5 * (lo + hi)
        b = _apply_pelvis(_crop1(body, knot), torch.full((2,), math.radians(m), dtype=F64))
        c = components_deg(b.rot6d)[:, [1, 2]]
        base_c = components_deg(_crop1(body, knot).rot6d)[:, [1, 2]]
        ok = ((c >= np.minimum(lim.lo[[1, 2]], base_c) - 1e-6) & (c <= np.maximum(lim.hi[[1, 2]], base_c) + 1e-6)).all()
        lo, hi = (m, hi) if ok else (lo, m)
    return lo


def _crop1(body, k):
    return Body(body.rot6d[k:k + 1].expand(2, -1, -1).clone(), body.transl[k:k + 1].expand(2, -1).clone(),
                body.contact[k:k + 1].expand(2, -1).clone(), body.betas.clone(), int(body.gender), body.rate)


# ---------------------------------------------------------------- pulses and the point processes
def pulse(t, onset, rise_s, fall_s, hold_s=0.0):
    """(T,) a minimum-jerk rise over rise_s from onset to 1, a hold of hold_s, a minimum-jerk return over fall_s."""
    t = np.asarray(t, dtype=np.float64)
    up = minimum_jerk((t - onset) / rise_s) if rise_s > 0 else (t >= onset).astype(np.float64)
    t2 = onset + rise_s + hold_s
    dn = minimum_jerk((t - t2) / fall_s) if fall_s > 0 else (t >= t2).astype(np.float64)
    out = up - dn
    out[(t <= onset) | (t >= t2 + fall_s)] = 0.0
    return out


def _knot(x, rate=RATE):
    return round(float(x) * rate) / rate


def _ceil_knot(x, rate=RATE):
    return math.ceil(float(x) * rate - 1e-9) / rate


def lapse_timing(depth, silent_s, catch_acc, max_acc, rate=RATE):
    """(drop_s, catch_s, drop_acc, catch_acc) of a ballistic lapse of a depth (m or deg; the accelerations in its unit
    per s^2), module docstring: the drop lasts the silent period at the constant acceleration that, with the catch at
    catch_acc, reaches the depth; when that acceleration would pass max_acc (the segment's gravity bound) the drop is
    at max_acc and lasts longer. Both phases on knots (at least one), the accelerations as realized on them."""
    t1 = float(silent_s)
    t2 = 0.5 * (-t1 + math.sqrt(t1 * t1 + 8.0 * depth / catch_acc))
    if 2.0 * depth / (t1 * (t1 + t2)) > max_acc:
        t1 = math.sqrt(2.0 * depth / (max_acc * (1.0 + max_acc / catch_acc)))
        t2 = max_acc * t1 / catch_acc
        t1 = max(_ceil_knot(t1, rate), 1.0 / rate)
    else:
        t1 = max(_knot(t1, rate), 1.0 / rate)
    t2 = max(_ceil_knot(t2, rate), 1.0 / rate)
    return t1, t2, 2.0 * depth / (t1 * (t1 + t2)), 2.0 * depth / (t2 * (t1 + t2))


def lapse_pulse(t, onset, drop_s, catch_s, fall_s, hold_s=0.0):
    """(T,) a ballistic lapse to 1 (module docstring): 0.5 a1 tau^2 over drop_s, a constant deceleration to rest at 1
    over catch_s, a hold of hold_s, a minimum-jerk return over fall_s; 0 outside its span."""
    tau = np.asarray(t, dtype=np.float64) - onset
    t1, t2 = float(drop_s), float(catch_s)
    a1 = 2.0 / (t1 * (t1 + t2))
    a2 = a1 * t1 / t2
    x1, v1 = 0.5 * a1 * t1 * t1, a1 * t1
    s = tau - t1
    x = np.where(tau < t1, 0.5 * a1 * tau * tau, x1 + v1 * s - 0.5 * a2 * s * s)
    t3 = t1 + t2 + hold_s
    x = np.where(tau >= t1 + t2, 1.0, x)
    out = x - minimum_jerk((tau - t3) / fall_s)
    out[(tau <= 0) | (tau >= t3 + fall_s)] = 0.0
    return out


def _part_profile(t, on, p):
    if p.get("shape") == "lapse":
        return lapse_pulse(t, on, p["drop_s"], p.get("catch_s", p["rise_s"] - p["drop_s"]), p["fall_s"],
                           p.get("hold_s", 0.0))
    return pulse(t, on, p["rise_s"], p["fall_s"], p.get("hold_s", 0.0))


def place_events(n, t0, t1, dur_s, rng, min_gap_s=MIN_GAP_S, rate=RATE):
    """[onset s] of n events of dur_s each inside [t0, t1], at least min_gap_s apart (between one's end and the next
    one's onset), on knots: a Poisson process given its count (module docstring). Raises when they cannot fit."""
    n = int(n)
    if n <= 0:
        return []
    K0, K1 = int(round(t0 * rate)), int(round(t1 * rate))
    d, g = int(round(dur_s * rate)), int(round(min_gap_s * rate))
    free = (K1 - K0) - n * d - (n - 1) * g
    if free < 0:
        raise ValueError(f"{n} events of {dur_s} s with gaps of {min_gap_s} s do not fit in {t1 - t0:.2f} s")
    gaps = rng.multinomial(free, rng.dirichlet(np.ones(n + 1)))
    out, k = [], K0
    for i in range(n):
        k += int(gaps[i])
        out.append(k / rate)
        k += d + g
    return out


def rhythmic_events(t0, t1, period_s, dur_s, rng, jitter_s=0.0, min_gap_s=MIN_GAP_S, rate=RATE):
    """[onset s]: one event every period_s from t0 (each moved by a normal jitter of SD jitter_s, on knots), those
    that end before t1 and keep min_gap_s from the previous one."""
    out, k = [], 0
    while True:
        t = _knot(t0 + k * period_s + (rng.normal(0.0, jitter_s) if jitter_s > 0 else 0.0), rate)
        if t + dur_s > t1:
            break
        if not out or t >= out[-1] + dur_s + min_gap_s - 1e-9:
            out.append(max(t, t0))
        k += 1
    return out


def site_profiles(T, events, rate=RATE):
    """{site: (T,) angle (rad, or m for a dip)}: the sum over events of their parts' pulses (module docstring).
    events: [{'onset_s', 'parts': [{'site', 'peak', 'rise_s', 'fall_s', 'hold_s', 'delay_s'}]}]."""
    t = np.arange(T) / rate
    out = {}
    for e in events:
        for p in e["parts"]:
            on = e["onset_s"] + p.get("delay_s", 0.0)
            prof = p["peak"] * _part_profile(t, on, p)
            out[p["site"]] = out.get(p["site"], np.zeros(T)) + prof
    return out


# ---------------------------------------------------------------- applying a site
def _replace_rows(body, R_new, joints, act):
    r6 = body.rot6d.clone()
    a = torch.as_tensor(act, dtype=torch.bool)
    for j in joints:
        r6[a, j] = matrix_to_rot6d(R_new[a, j])
    return r6


def _apply_rotation_site(body, name, theta):
    """The body with a 'parent' or 'world' site turned by theta (T,) rad (split over its joints, module docstring)."""
    joints, kind, axis, sign, _, _ = SITES[name]
    act = np.abs(theta) > 0
    if not act.any():
        return body.copy()
    k_ref = int(np.flatnonzero(act)[0])
    rooms = site_rooms(body, name, k_ref)
    tot = sum(rooms)
    split = [r / tot for r in rooms] if tot > 0 else [1.0 / len(joints)] * len(joints)
    ax = site_axes(body, name)
    R = body.local_R()
    Rn = R.clone()
    th = torch.from_numpy(sign * theta)
    for j, s in zip(joints, split):
        Rn[:, j] = axis_angle_to_matrix((s * th)[:, None] * ax[j]) @ R[:, j]
    rows = list(joints)
    if name in KEEP_CHILD:                            # the child keeps its global rotation
        j, c = joints[-1], KEEP_CHILD[name]
        Rn[:, c] = Rn[:, j].transpose(-1, -2) @ R[:, j] @ R[:, c]
        rows.append(c)
    out = body.copy()
    out.rot6d = _replace_rows(body, Rn, rows, act)
    return out


def _apply_pelvis(body, ang):
    """The body with the pelvis turned forward by ang (T,) rad about the line through the hip joints, the thighs'
    global rotations kept (module docstring, kind 'pelvis')."""
    ang = torch.as_tensor(ang, dtype=F64)
    act = ang.abs() > 0
    out = body.copy()
    if not bool(act.any()):
        return out
    J, G = body.fk()
    u = J[:, 1] - J[:, 2]
    u = u / u.norm(dim=-1, keepdim=True)                       # the hip line, pointing to the body's left (+x)
    Rp = axis_angle_to_matrix(ang[:, None] * u)
    mid = 0.5 * (J[:, 1] + J[:, 2])
    G0n = Rp @ G[:, 0]
    J0n = mid + torch.einsum("tij,tj->ti", Rp, J[:, 0] - mid)
    R = body.local_R()
    Rn = R.clone()
    Rn[:, 0] = G0n
    for h in (1, 2):
        Rn[:, h] = G0n.transpose(-1, -2) @ G[:, h]
    out.rot6d = _replace_rows(body, Rn, (0, 1, 2), act.numpy())
    tr = body.transl.clone()
    tr[act] = (J0n - body.rest()[0])[act]
    out.transl = tr
    return out


def _move_pelvis(body, offset, act):
    """The body with the pelvis root moved by offset (T, 3) m at the knots act (T,) bool, the ankles and the feet
    still: the legs solved by ik.hinge_leg (knees bending forward), the feet's global rotations kept. A hip that the
    offset carries beyond its leg's reach is lowered onto the reach's arc about the ankle (open-body port, wave 3: the
    open body's standing legs are straight, the hip-to-ankle distance the leg's full reach, so a pelvis moved back by b
    drops by about b^2 / 2L, the leg turning about the ankle as one piece; a bent leg straightens first)."""
    from .ik import hinge_leg
    offset = torch.as_tensor(offset, dtype=F64).clone()
    a = torch.as_tensor(np.asarray(act), dtype=torch.bool)
    out = body.copy()
    if not bool(a.any()):
        return out
    J, G = body.fk()
    rest = body.rest()
    drop = torch.zeros(body.T, dtype=F64)
    for h, kn, an, _ in LEGS:                            # the hips kept within their leg's reach (ik.hinge_leg's)
        p, q = rest[kn] - rest[h], rest[an] - rest[kn]
        rho = torch.sqrt((p[1] * q[1] + p[2] * q[2]) ** 2 + (p[2] * q[1] - p[1] * q[2]) ** 2)
        d0 = torch.sqrt((p * p).sum() + (q * q).sum() + 2 * p[0] * q[0] + 2 * rho) * (1.0 - 1e-9)
        v = J[:, h] + offset - J[:, an]
        vh2 = v[:, 0] ** 2 + v[:, 2] ** 2
        need = torch.where((v.norm(dim=-1) > d0) & (vh2 < d0 ** 2), v[:, 1] - torch.sqrt((d0 ** 2 - vh2).clamp_min(0.0)),
                           torch.zeros_like(drop))
        drop = torch.maximum(drop, need)
    offset[:, 1] -= drop
    tr = body.transl.clone()
    tr[a] += offset[a]
    G0 = G[a, 0]
    fwd = torch.einsum("tij,j->ti", G0, torch.tensor([0.0, 0.0, 1.0], dtype=F64))
    R = body.local_R()
    Rn = R.clone()
    for h, kn, an, _ in LEGS:
        H = J[a, h] + offset[a]
        A = J[a, an]
        pole = J[a, kn] + offset[a] + 0.3 * fwd
        G_th, kap = hinge_leg(H, A, pole, rest[kn] - rest[h], rest[an] - rest[kn], G0,
                              hinge=torch.cross(fwd, A - H, dim=-1))
        Rk = axis_angle_to_matrix(torch.stack([kap, 0 * kap, 0 * kap], -1))
        Rh, Rkn, Ra = Rn[:, h].clone(), Rn[:, kn].clone(), Rn[:, an].clone()
        Rh[a] = G0.transpose(-1, -2) @ G_th
        Rkn[a] = Rk
        Ra[a] = (G_th @ Rk).transpose(-1, -2) @ G[a, an]
        Rn[:, h], Rn[:, kn], Rn[:, an] = Rh, Rkn, Ra
    out.rot6d = _replace_rows(body, Rn, (1, 4, 7, 2, 5, 8), a.numpy())
    out.transl = tr
    return out


def _apply_dip(body, depth):
    """The body with the pelvis lowered by depth (T,) m, the ankles and the feet still (module docstring, 'dip')."""
    depth = torch.as_tensor(depth, dtype=F64)
    up = torch.tensor([0.0, 1.0, 0.0], dtype=F64)
    return _move_pelvis(body, -depth[:, None] * up, depth > 0)


TOUCH_ROWS = (0, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21)   # the rows a touch path is solved on


def touch_posture(body, knot):
    """The base posture of reach.TOUCH_POSTURES whose trunk, neck, head and arm rows the body holds at a knot (a
    touch path is solved from that rest); ValueError when none."""
    from . import reach as RC
    R = body.local_R()[knot, list(TOUCH_ROWS)]
    for name in RC.TOUCH_POSTURES:
        if float((R - posture_rotations(name)[list(TOUCH_ROWS)]).abs().max()) < 1e-9:
            return name
    raise ValueError(f"a touch starts from the rest of {RC.TOUCH_POSTURES}; knot {knot} holds neither")


def _apply_touch(body, name, share):
    """The body (a hands.HandBody) with the arm of a 'touch' site on its reach.touch_path at the shares share (T,)
    (module docstring, touching tics): at the knots where the share is positive the collar, shoulder, elbow and wrist
    rows are the path's and the hand of that side forms the pointing hand (reach.hand_share); the hand must be at
    rest there."""
    from . import hands as HD
    from . import reach as RC
    joints, _, target, _, _, _ = SITES[name]
    side = "left" if joints[-1] == 20 else "right"
    share = np.asarray(share, dtype=np.float64)
    act = share > 0
    out = HD.hands_of(body)
    if not act.any():
        return out
    k = np.flatnonzero(act)
    posture = touch_posture(body, int(k[0]))
    R_arm, R_hand = RC.touch_rows(side, target, posture, share[act])
    Rn = out.local_R().clone()
    for i, j in enumerate(joints):
        Rn[torch.as_tensor(k), j] = R_arm[:, i]
    rows = torch.arange(15) + (0 if side == "left" else 15)
    h6 = out.hand6d.clone()
    rest = matrix_to_rot6d(torch.eye(3, dtype=F64))
    if float((h6[torch.as_tensor(k)][:, rows] - rest).abs().max()) > 1e-12:
        raise ValueError(f"{name}: the {side} hand is not at rest where the touch moves it")
    h6[torch.as_tensor(k)[:, None], rows[None]] = matrix_to_rot6d(R_hand[:, rows])
    out.rot6d = _replace_rows(out, Rn, joints, act)
    out.hand6d = h6
    return out


def _com_np(body):
    from ..kernels.force import com
    return com(body.joints().numpy(), int(body.gender))


def _hold_com(base, out, act, iters=10, gain=0.8, tol_m=1e-6):
    """(Body, back (T,) m): the hip strategy of a standing body (module docstring): the pelvis of out moved backward
    along the base's heading at the knots act so that the centre of mass keeps its base position along the heading
    (Newton steps of gain dCoM/dback, the legs solved again each time), and the backward shift per knot."""
    act = np.asarray(act, bool)
    back = np.zeros(base.T)
    if not act.any():
        return out, back
    fwd = _heading_axes(base, "z").numpy()
    c0 = (_com_np(base) * fwd).sum(-1)
    cur = out
    for _ in range(iters):
        err = np.where(act, (_com_np(cur) * fwd).sum(-1) - c0, 0.0)
        if np.abs(err).max() < tol_m:
            break
        back = back + err / gain
        cur = _move_pelvis(out, torch.from_numpy(-back[:, None] * fwd), act)
    return cur, back


SITE_ORDER = ("hip_flexion", "knee_dip")          # the pelvis sites last, after every rotation site


def apply_events(base, events, standing=False, arms=False, info=None):
    """(Body, profiles, back): the base with every site of the events applied (rotation sites first, then the
    pelvis turn, then the dip; with arms, the arms kept by arms_rule, its record put in the dict info; standing, then
    the hip strategy, _hold_com, on the arms as kept), the site profiles (site_profiles) and the hip strategy's
    backward pelvis shift (T,) m (zeros when not standing)."""
    prof = site_profiles(base.T, events, base.rate)
    out = base.copy()
    for name in sorted(prof, key=lambda s: (s in SITE_ORDER, SITE_ORDER.index(s) if s in SITE_ORDER else 0, s)):
        kind = SITES[name][1]
        if kind == "pelvis":
            out = _apply_pelvis(out, SITES[name][3] * prof[name])
        elif kind == "dip":
            out = _apply_dip(out, prof[name])
        elif kind == "touch":
            out = _apply_touch(out, name, prof[name])
        else:
            out = _apply_rotation_site(out, name, prof[name])
    if arms:
        out, rec = arms_rule(base, out, events, standing)
        if info is not None:
            info.update(rec)
    back = np.zeros(base.T)
    if standing and prof:
        act = np.any([np.abs(p) > 0 for p in prof.values()], axis=0)
        out, back = _hold_com(base, out, act)
    return out, prof, back


def peak_of(body, site, share, knot):
    """The peak (rad, or m for a dip) of a jerk at a share of the room in its direction at a knot of the base (the
    room of a chain is the sum over its joints; the pelvis turn's room keeps the hips inside the table; a dip's
    'room' is not a share: share is the depth in m)."""
    kind = SITES[site][1]
    if kind == "dip":
        return float(share) * DIP_PER_SHARE_M
    if kind == "pelvis":
        return math.radians(share * _pelvis_room(body, knot))
    return math.radians(share * sum(site_rooms(body, site, knot)))


# ---------------------------------------------------------------- the operator
def balance_block(base, out, k_ref=0):
    """The balance checks of a standing body (module docstring), JSON-ready: the extrapolated centre of mass's
    margins inside the feet (motions.balance; m, the clip's minimum and the base's), the friction the floor must
    supply (|horizontal| over vertical centre-of-mass force, the clip's maximum), the vertical ground force in body
    weights (minimum and maximum), the centre of mass's largest move along the heading against the base (cm); 'ok'
    when every bound holds (BALANCE_MARGIN_M, FRICTION_MAX, GRF_MIN_BW)."""
    from .motions import balance
    bo, bb = balance(out, k_ref), balance(base, k_ref)
    dt = 1.0 / out.rate
    a = np.gradient(np.gradient(bo["com"], dt, axis=0), dt, axis=0)
    vert = GRAVITY + a[:, 1]
    fr = np.hypot(a[:, 0], a[:, 2]) / np.maximum(vert, 1e-9)
    grf = vert / GRAVITY
    post, ant = float(bo["posterior"].min()), float(bo["anterior"].min())
    moved = float(np.abs(bo["com"][:, 2] - bb["com"][:, 2]).max())
    ok = bool(min(post, ant) >= BALANCE_MARGIN_M and fr.max() <= FRICTION_MAX and grf.min() >= GRF_MIN_BW)
    return {"ok": ok, "xcom_margin_posterior_m": post, "xcom_margin_anterior_m": ant,
            "base_margin_posterior_m": float(bb["posterior"].min()), "base_margin_anterior_m": float(bb["anterior"].min()),
            "friction_max": float(fr.max()), "vertical_grf_min_bw": float(grf.min()),
            "vertical_grf_max_bw": float(grf.max()), "com_moved_forward_cm": 100.0 * moved,
            "bounds": {"margin_m": BALANCE_MARGIN_M, "friction": FRICTION_MAX, "grf_min_bw": GRF_MIN_BW}}


def jerk_operator(base, events, operator, grade, scale, params, sources, seed=None, feet=(0, 1), check=True,
                  extra_arrays=None, standing=False, arms_on=True):
    """(Body, truth): the base with the events applied (apply_events, with the hip strategy when standing), the arms
    kept by arms_rule (arms_on: seated carried over the thighs, standing hanging) where the trunk or the hips move,
    contacts relabelled when a leg site moves a foot, the truth record (common.truth_record) with the events, the
    realized peaks and speeds per site, and the kinematic checks of body.validity (feet: the feet whose stance is
    checked) with, standing, the balance block (balance_block); a standing clip that fails it is refused."""
    arms = {} if arms_on else None
    out, prof, back = apply_events(base, events, standing, arms_on, arms)
    shrink = None
    if standing and not balance_block(base, out)["ok"] and any("parts0" in e for e in events):
        lo, hi, best = 0.0, 1.0, None                      # the drawn sizes and rises toward the median jerk
        for _ in range(SHRINK_STEPS):
            lam = 0.5 * (lo + hi)
            ev2 = shrunk(events, lam)
            a2 = {} if arms_on else None
            o2, p2, b2 = apply_events(base, ev2, standing, arms_on, a2)
            if balance_block(base, o2)["ok"]:
                lo, best = lam, (ev2, o2, p2, b2, a2)
            else:
                hi = lam
        if best is None:
            ev2 = shrunk(events, 0.0)
            a2 = {} if arms_on else None
            o2, p2, b2 = apply_events(base, ev2, standing, arms_on, a2)
            best, lo = (ev2, o2, p2, b2, a2), 0.0
        events, out, prof, back, arms = best
        shrink = {"power": lo, "rule": "standing, the drawn size and rise factors raised to this power (bisection, "
                                       "SHRINK_STEPS) so that the balance block holds (module)"}
    if standing or any(SITES[s][1] in ("dip",) or SITES[s][0][0] in (4, 5, 7, 8) for s in prof):
        out.contact = contact_labels(out)
    sites = {}
    for s, p in prof.items():
        unit = {"dip": "m", "touch": "path share"}.get(SITES[s][1], "deg")
        v = np.abs(np.gradient(p, 1.0 / base.rate))
        sites[s] = {"unit": unit, "peak": float(np.abs(p).max() * (180.0 / math.pi if unit == "deg" else 1.0)),
                    "peak_speed": float(v.max() * (180.0 / math.pi if unit == "deg" else 1.0)),
                    "area": site_area(s), "end_cm": end_displacement_cm(base, out, s, np.abs(p) > 0)}
    ev = [{"onset_s": float(e["onset_s"]), "label": e.get("label", ""), "stimulus_s": e.get("stimulus_s"),
           "factor": e.get("factor"), "size_factor": e.get("size_factor"), "rise_factor": e.get("rise_factor"),
           "parts": [{k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in p.items()}
                     for p in e["parts"]]} for e in events]
    params = dict(params, events=ev, n_events=len(events), sites=sites, clip_s=float(base.duration))
    if shrink is not None:
        sv = dict(params.get("size_variation") or {}, standing_shrink=shrink,
                  size_factors=[float(e.get("size_factor", 1.0)) for e in events],
                  rise_factors=[float(e.get("rise_factor", 1.0)) for e in events])
        params["size_variation"] = sv
    if arms is not None:
        params["carry_arms"] = arms
    if limits_table(prof) != LX.HEALTHY:                       # a shrug: the arm hangs under the raised girdle
        params["limits_table"] = LX.record(limits_table(prof))
    val = jerk_validity(base, out, prof, feet=feet) if check else None
    arrays = {f"profile_{s}": p for s, p in prof.items()}
    if standing:
        bal = balance_block(base, out)
        params["hip_strategy"] = {"rule": "pelvis moved back along the heading so that the centre of mass keeps its "
                                          "base position (module)", "max_back_cm": 100.0 * float(back.max()),
                                  "min_back_cm": 100.0 * float(back.min())}
        arrays["hip_strategy_back_m"] = back
        if val is not None:
            val["balance"] = bal
        if not bal["ok"]:
            raise ValueError(f"{operator}: the standing clip fails its balance block {bal}")
    arrays.update(extra_arrays or {})
    return out, truth_record(operator, seed, grade, scale, params, sources, val, arrays=arrays)


def event_span(e):
    """(start_s, end_s) of an event: its earliest part onset to its latest part end."""
    a = min(e["onset_s"] + p.get("delay_s", 0.0) for p in e["parts"])
    b = max(e["onset_s"] + p.get("delay_s", 0.0) + p["rise_s"] + p.get("hold_s", 0.0) + p["fall_s"] for p in e["parts"])
    return a, b


def end_displacement_cm(base, out, site, knots=None):
    """cm: the largest displacement of the site's end joint against the base over the knots where the site moves
    (knots (T,) bool; other sites moving at the same knots add to it), None for a site without an end joint."""
    end = SITES[site][4]
    if end is None:
        return None
    Jb, Jo = base.joints(), out.joints()
    d = (Jo[:, end] - Jb[:, end]).norm(dim=-1)
    if knots is not None:
        d = d[torch.as_tensor(np.asarray(knots), dtype=torch.bool)]
    return float(100.0 * d.max()) if d.numel() else 0.0


# ---------------------------------------------------------------- re-measurement
def _runs(mask):
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.diff(m.astype(np.int8))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def deviation_deg(out, base, joints):
    """(T, n) deg: the angle of each joint's local rotation against the base's (geodesic; 0 without a base is the
    angle against the first knot)."""
    Ro = out.local_R()[:, list(joints)]
    Rb = base.local_R()[:, list(joints)] if base is not None else Ro[:1].expand_as(Ro)
    return torch.rad2deg(geodesic(Rb, Ro)).numpy()


def angular_speed_deg_s(out, base, joints):
    """(T - 1, n) deg/s: the speed of each joint's deviation from the base between knots (geodesic of D_k^T D_k+1,
    D = R_base^T R_out); without a base, of its own local rotation."""
    Ro = out.local_R()[:, list(joints)]
    D = Ro if base is None else base.local_R()[:, list(joints)].transpose(-1, -2) @ Ro
    return torch.rad2deg(geodesic(D[:-1], D[1:])).numpy() * out.rate


def measure_jerks(out, base, joints, merge_s=MERGE_S, floor_deg_s=SPEED_FLOOR_DEG_S, rel=1e-3, pelvis=False):
    """Jerks read from a body (module docstring): the runs of knots where the angular speed of any of the joints
    (deviation from the base) exceeds max(floor_deg_s, rel x the clip's peak), runs closer than merge_s joined; a run
    that starts at the base (deviation under 10 percent of its peak at its first knot) is a jerk. pelvis: also the
    pelvis height against the base (a dip; its speed in cm/s counted as deg/s). A run that starts away from the base
    is the return of the previous jerk after a hold (its duration runs on). Returns {'n', 'onset_s', 'duration_s',
    'peak_deg', 'peak_speed_deg_s', 'runs'}."""
    w = angular_speed_deg_s(out, base, joints)
    dev = deviation_deg(out, base, joints)
    if pelvis:
        dy = 100.0 * (base.joints()[:, 0, 1] - out.joints()[:, 0, 1]).numpy()
        w = np.concatenate([w, np.abs(np.diff(dy))[:, None] * out.rate], 1)
        dev = np.concatenate([dev, np.abs(dy)[:, None]], 1)
    s = w.max(1)
    thr = max(floor_deg_s, rel * float(s.max())) if s.size else floor_deg_s
    runs = _runs(s > thr)
    g = int(round(merge_s * out.rate))
    merged = []
    for a, b in runs:
        if merged and a - merged[-1][1] <= g:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    on, du, pk, sp = [], [], [], []
    for a, b in merged:                          # steps a .. b-1 move: knots a .. b
        d = dev[a:b + 1].max(1)
        if d[0] > 0.1 * d.max():
            if on:                               # the return after a hold: the same jerk goes on
                du[-1] = b / out.rate - on[-1]
                pk[-1] = max(pk[-1], float(d.max()))
                sp[-1] = max(sp[-1], float(s[a:b].max()))
            continue
        on.append(a / out.rate)
        du.append((b - a) / out.rate)
        pk.append(float(d.max()))
        sp.append(float(s[a:b].max()))
    return {"n": len(on), "onset_s": np.array(on), "duration_s": np.array(du), "peak_deg": np.array(pk),
            "peak_speed_deg_s": np.array(sp), "runs": len(merged), "threshold_deg_s": thr}


def signed_angle_deg(out, base, joint, axis):
    """(T,) deg: the angle of the joint's offset R_out R_base^T about a parent-frame axis ('x', 'y', 'z'), signed
    (atan2 of the axis component of its skew part and its cosine; exact for an offset about that axis)."""
    D = out.local_R()[:, joint] @ base.local_R()[:, joint].transpose(-1, -2)
    c = (D[..., 0, 0] + D[..., 1, 1] + D[..., 2, 2] - 1) / 2
    return torch.rad2deg(torch.atan2(vee(D)[:, {"x": 0, "y": 1, "z": 2}[axis]], c)).numpy()


def beat_peaks(x, rel=0.10):
    """([knot] of the beats, (M,) prominences of every local maximum) of a signal (T,): its local maxima whose
    prominence (the peak over the higher of the two minima between it and its neighbouring maxima) is at least rel of
    the largest prominence; a monotonic stretch makes no beat."""
    x = np.asarray(x, dtype=np.float64)
    mx = [k for k in range(1, len(x) - 1) if x[k] > x[k - 1] and x[k] >= x[k + 1]]
    if not mx:
        return [], np.zeros(0)
    edges = [0] + mx + [len(x) - 1]
    prom = np.asarray([x[k] - max(x[edges[i]:k + 1].min(), x[k:edges[i + 2] + 1].min()) for i, k in enumerate(mx)])
    return [k for k, q in zip(mx, prom) if q >= rel * prom.max() and q > 1e-6], prom


def measure_clonus(out, base, joint=7, axis="x", sign=-1.0, rel=0.10, band=(3.0, 12.0)):
    """Clonus read from a body: the ankle's offset about its flexion axis (dorsiflexion positive); the beats are its
    local maxima whose prominence (the peak over the higher of the two minima between it and its neighbouring maxima)
    is at least rel of the largest prominence (a monotonic stretch makes no beat); their frequency
    (measure.dominant_frequency over the beating span, mean removed); the span from the first to the last beat plus
    one period (the SCATS duration); the peak-to-peak excursion of the first full beats. Returns {'beats',
    'frequency_hz', 'duration_s', 'pp_deg', 'scats', 'first_beat_s', 'last_beat_s'}."""
    x = sign * signed_angle_deg(out, base, joint, axis)
    pk, prom = beat_peaks(x, rel)
    if not len(prom):
        return {"beats": 0, "frequency_hz": None, "duration_s": 0.0, "pp_deg": 0.0, "scats": 0}
    if len(pk) < 2:
        return {"beats": len(pk), "frequency_hz": None, "duration_s": 0.0, "pp_deg": float(prom.max()),
                "scats": 1 if pk else 0}
    seg = x[pk[0]:pk[-1] + 1]
    f = dominant_frequency(seg, out.rate, band)
    dur = (pk[-1] - pk[0]) / out.rate + 1.0 / f
    w = x[pk[0]:pk[min(len(pk) - 1, 2)] + 1]
    return {"beats": len(pk), "frequency_hz": float(f), "duration_s": float(dur), "pp_deg": float(w.max() - w.min()),
            "scats": scats_grade(dur), "first_beat_s": pk[0] / out.rate, "last_beat_s": pk[-1] / out.rate}


def scats_grade(duration_s):
    if duration_s <= 0:
        return 0
    return next(g for g, (lo, hi) in SCATS_S.items() if lo <= duration_s < hi)


def measure_dips(out, base, floor_cm=0.05):
    """Dips of the pelvis read from a body: runs where it lies more than floor_cm below the base. Returns {'n',
    'onset_s', 'depth_cm', 'duration_s'}."""
    dy = 100.0 * (base.joints()[:, 0, 1] - out.joints()[:, 0, 1]).numpy()
    runs = _runs(dy > floor_cm)
    return {"n": len(runs), "onset_s": np.array([a / out.rate for a, _ in runs]),
            "depth_cm": np.array([float(dy[a:b].max()) for a, b in runs]),
            "duration_s": np.array([(b - a) / out.rate for a, b in runs])}


def site_joints(sites):
    """Sorted joints moved by a set of sites (the pelvis root for a pelvis or dip site, with the legs)."""
    js = set()
    for s in sites:
        js.update(SITES[s][0])
    return sorted(js)


DIP_AREA_CM = 0.05                  # [module] tic_areas: a lowering of the pelvis past this is a dip (the legs' area);
#                                     open-body port, wave 3: the open body's standing legs are straight, and the hip
#                                     strategy's pelvis moved back by b drops on the arc about the ankles by b^2 / 2L
#                                     (0.005 cm under the standing touching tics, 0.007 outside the squats of the
#                                     complexity clip; scripts/openbody_wave3_checks.py); before, any lowering (1e-7)


def tic_areas(out, base, sites=None, tol_deg=0.05, standing=False):
    """MRVS body areas moved in a clip: a dip of the pelvis (measure_dips) is the legs' area and owns the leg joints
    at its knots (a crouch flexes hips, knees and ankles); a shoulder row counts for the arms only where the upper
    arm's global rotation moves too (a shrug compensates it, KEEP_CHILD); every other site of SITES (or the given ones) whose joints
    deviate from the base by more than tol_deg at some other knot adds its area. standing (both feet planted): the
    leg joints move only with a dip or with the hip strategy (module docstring), so no leg-joint site adds an area
    (the legs' area is the dip's). Returns (sorted areas, {area: [sites]})."""
    hit = {}
    dy = 100.0 * (base.joints()[:, 0, 1] - out.joints()[:, 0, 1]).numpy()
    dip = dy > DIP_AREA_CM                           # a lowering: the legs bend as the square root of the drop
    if (sites is None or "knee_dip" in sites) and dip.any():
        hit["legs"] = ["knee_dip"]
    legs = set(SITES["knee_dip"][0])
    for s in (sites or SITES):
        kind, j = SITES[s][1], SITES[s][0]
        if kind in ("dip", "touch") or (standing and set(j) <= legs):     # a touch counts by its joints' sites
            continue
        dev = deviation_deg(out, base, j).max(1)
        if set(j) <= set(KEEP_CHILD.values()):        # a shrug turns the shoulder row but not the upper arm
            G0, G1 = base.fk()[1][:, list(j)], out.fk()[1][:, list(j)]
            dev = np.minimum(dev, torch.rad2deg(geodesic(G0, G1)).numpy().max(1))
        if set(j) <= legs:
            dev = np.where(dip, 0.0, dev)
        if dev.max() > tol_deg:
            hit.setdefault(site_area(s), []).append(s)
    return sorted(hit), hit


def grade_of_count(count_30s, bands=CONN_30S):
    """The 0 to 4 grade whose band (per 30 s) holds a count."""
    if count_30s <= 0:
        return 0
    for g, (lo, hi) in bands.items():
        if hi is None or count_30s <= hi:
            return g if count_30s >= lo or g == 1 else g - 1
    return max(bands)


def mrvs_frequency_grade(per_min):
    return grade_of_count(per_min, MRVS_PER_MIN)


def mrvs_area_grade(n_areas):
    return grade_of_count(n_areas, MRVS_AREA_BANDS)


# ---------------------------------------------------------------- base bodies
def _static(name, seconds=CLIP_S):
    return static_body(name, seconds)


def outstretched_base(seconds=CLIP_S):
    """(HandBody, info): motions.outstretched (the base of the exam clip tremor_E2_raise_g0 at 30 s)."""
    from .motions import outstretched
    return outstretched(seconds)


def asterixis_base(seconds=CLIP_S):
    """(HandBody, info): the outstretched arms with both wrists extended, the extension rising with the raise
    (minimum jerk over its span) to WRIST_EXTENSION_DEG or 0.8 of the table's room, whichever is less (module)."""
    b, info = outstretched_base(seconds)
    t = b.times().numpy()
    w = minimum_jerk((t - info["rest_s"]) / info["raise_s"])
    k_hold = int(round(info["onset_s"] * b.rate))
    out = b
    ext = {}
    for side in ("left", "right"):
        s = f"wrist_extension_{side}"
        room = sum(site_rooms(b, s, k_hold))
        deg = min(WRIST_EXTENSION_DEG, 0.8 * room)
        ext[side] = {"deg": deg, "room_deg": room}
        out = _apply_rotation_site(out, s, math.radians(deg) * w)
    info = dict(info, wrist_extension=ext, wrist_rule="min(WRIST_EXTENSION_DEG, 0.8 room), with the raise (module)")
    return out, info


FOOT = {"left": (1, 4, 7, 10), "right": (2, 5, 8, 11)}         # hip, knee, ankle, foot joint


def _sit_body(T):
    from dataclasses import replace
    R = posture_rotations("sit")
    rest = rest_joints()
    tr = torch.zeros(T, 3, dtype=F64)
    tr[:, [0, 2]] = -rest[0, [0, 2]]
    b = Body(matrix_to_rot6d(R)[None].expand(T, -1, 6).clone(), tr, torch.zeros(T, 2, dtype=torch.bool))
    b = floor_to_zero(b)
    return replace(b, contact=contact_labels(b))


def forefoot_rows(sit, side, psi, back_m=FOREFOOT_BACK_M):
    """(T, 4, 3, 3) local rotations of the hip, knee, ankle and foot joints of one leg of a seated body (sit, the
    'sit' posture on its knots) with that foot turned by psi (T,) rad about the body's x axis through the foot joint
    (heel up for psi > 0) and the foot joint moved back by back_m: the hip on the seat, the ankle where the turned foot
    puts it, the thigh and the knee by ik.hinge_leg, the toes kept flat (their global rotation as in 'sit'). Also the
    knee angle (T,) deg."""
    from .ik import hinge_leg
    h, kn, an, ft = FOOT[side]
    J, G = sit.fk()
    rest = sit.rest()
    psi = torch.as_tensor(psi, dtype=F64)
    x = torch.tensor([1.0, 0.0, 0.0], dtype=F64)
    Rp = axis_angle_to_matrix(psi[:, None] * x)
    P = J[:, ft] - torch.tensor([0.0, 0.0, back_m], dtype=F64)
    A = P + torch.einsum("tij,tj->ti", Rp, J[:, an] - J[:, ft])
    Gf = Rp @ G[:, an]
    fwd = torch.tensor([0.0, 0.0, 1.0], dtype=F64).expand_as(A)
    Gt, kap = hinge_leg(J[:, h], A, J[:, kn] + 0.3 * fwd, rest[kn] - rest[h], rest[an] - rest[kn], G[:, 0],
                        hinge=torch.cross(fwd, A - J[:, h], dim=-1))
    Rk = axis_angle_to_matrix(torch.stack([kap, 0 * kap, 0 * kap], -1))
    rows = torch.stack([G[:, 0].transpose(-1, -2) @ Gt, Rk, (Gt @ Rk).transpose(-1, -2) @ Gf,
                        Gf.transpose(-1, -2) @ G[:, ft]], 1)
    return rows, torch.rad2deg(kap).numpy()


def _with_leg(body, side, rows, act=None):
    R = body.local_R()
    T = body.T
    a = np.ones(T, bool) if act is None else np.asarray(act, bool)
    Rn = R.clone()
    for i, j in enumerate(FOOT[side]):
        Rn[:, j] = rows[:, i]
    out = body.copy()
    out.rot6d = _replace_rows(body, Rn, FOOT[side], a)
    return out


def heel_ids(side, gender=2, share=0.3):
    """Vertex ids of the heel: the sole cloud's vertices (contact.sole_vertices) in the back share of its length on
    the selected body model's rest mesh (viewer.rest_vertices; open-body port, wave 3)."""
    from .contact import sole_vertices
    from .viewer import rest_vertices
    ids = sole_vertices(int(gender))[side]
    v = rest_vertices()[ids]
    z0, z1 = v[:, 2].min(), v[:, 2].max()
    return ids[v[:, 2] <= z0 + share * (z1 - z0)]


def heel_height_m(body, side, knots=None):
    """(K,) m: the lowest heel vertex (heel_ids) above the lowest sole vertex of the other foot (the floor), at the
    knots (default all)."""
    from .contact import sole_vertices
    from .hands import body_points
    other = "right" if side == "left" else "left"
    k = np.arange(body.T) if knots is None else np.asarray(knots)
    hp = body_points(body, heel_ids(side, int(body.gender)), knots=k)[..., 1].min(1)
    fl = body_points(body, sole_vertices(int(body.gender))[other], knots=k)[..., 1].min(1)
    return hp - fl


@functools.lru_cache(maxsize=4)
def forefoot_psi0(side="left", heel_up_m=HEEL_UP_M, back_m=FOREFOOT_BACK_M):
    """rad: the turn of the foot about the foot joint that raises the heel by heel_up_m (heel_height_m, bisection)."""
    sit = _sit_body(2)
    base_h = float(heel_height_m(sit, side, [0])[0])
    lo, hi = 0.0, math.radians(40.0)
    for _ in range(40):
        m = 0.5 * (lo + hi)
        rows, _ = forefoot_rows(sit, side, torch.full((sit.T,), m, dtype=F64), back_m)
        h = float(heel_height_m(_with_leg(sit, side, rows), side, [0])[0]) - base_h
        lo, hi = (m, hi) if h < heel_up_m else (lo, m)
    return 0.5 * (lo + hi)


def held_leg_base(seconds=20.0, side="left"):
    """Body: seated (base 'sit') with one forefoot loaded (catalogue row O01): the ball of the foot on the floor, the
    foot joint FOREFOOT_BACK_M behind its 'sit' place, the heel HEEL_UP_M up (forefoot_psi0, forefoot_rows), the knee
    near 96 deg; the other foot flat, contacts by the body-store rule."""
    from dataclasses import replace
    T = int(round(seconds * RATE)) + 1
    sit = _sit_body(T)
    rows, _ = forefoot_rows(sit, side, torch.full((T,), forefoot_psi0(side), dtype=F64))
    b = _with_leg(sit, side, rows)
    return replace(b, contact=contact_labels(b))


# ---------------------------------------------------------------- event builders
def jerk_part(site, peak, kind, delay_s=0.0, hold_s=0.0, rate=RATE, rise_s=None, fall_s=None):
    """One part of an event (module docstring): the type's timing (or rise_s and fall_s given), then the bounds of
    the module: a rotation turns at most at PEAK_SPEED_CAP_DEG_S (the rise lengthened to 1.875 peak / cap, the
    return RETURN_RATIO times it), a dip falls and rises at most at DIP_ACCEL_MAX."""
    rise, fall = timing(kind, rate)
    if rise_s is not None:
        rise = _knot(rise_s, rate)
        fall = _knot(fall_s, rate) if fall_s is not None else round(RETURN_RATIO * rise * rate) / rate
    bound = None
    if SITES[site][1] == "dip":
        need = _ceil_knot(math.sqrt(MJ_PEAK_ACC * abs(peak) / DIP_ACCEL_MAX), rate)
        if need > min(rise, fall):
            rise, fall, bound = max(rise, need), max(fall, need), "dip_accel"
    else:
        need = _ceil_knot(MJ_PEAK_SPEED * math.degrees(abs(peak)) / PEAK_SPEED_CAP_DEG_S, rate)
        if need > rise:
            rise, fall, bound = need, max(fall, round(RETURN_RATIO * need * rate) / rate), "speed_cap"
    out = {"site": site, "peak": float(peak), "rise_s": rise, "fall_s": fall, "hold_s": _knot(hold_s, rate),
           "delay_s": _knot(delay_s, rate), "type": kind}
    if bound:
        out["bound"] = bound
    return out


def vary_sizes(base, events, rng, sigma=SIZE_SIGMA, rise_jitter=RISE_JITTER, knot=0, rate=RATE, skip=()):
    """The events with their sizes varied (module constants SIZE_SIGMA to RISE_JITTER, wave SH1d part B): per event a
    factor f, lognormal of median 1 and log-sigma sigma clipped to SIZE_CLIP, times every part's peak, each rotation
    peak then kept under ROOM_CAP_SHARE of the base's room at knot (a lapse's depth likewise); per event a rise factor
    u, uniform in 1 -+ rise_jitter, times the part's rise and return (jerk_part, whose speed and acceleration bounds
    then apply; a lapse keeps its ballistic timing, a touch path and the events labelled in skip keep their shape and
    size). The factors are drawn after every other draw of the clip's events and recorded per event ('size_factor',
    'rise_factor'); the grade ladders describe the median jerk. An event placed with onsets keeps MIN_GAP_S before
    the next one: where its drawn size or rise would make it run longer, the factors above 1 are brought to 1, and
    if it still would, it keeps its parts (factors 1)."""
    out = []
    caps = {}
    ons = sorted(float(e["onset_s"]) for e in events if "onset_s" in e)
    for e in events:
        f = float(np.clip(math.exp(sigma * rng.standard_normal()), *SIZE_CLIP)) if sigma > 0 else 1.0
        u = float(rng.uniform(1.0 - rise_jitter, 1.0 + rise_jitter)) if rise_jitter > 0 else 1.0
        if e.get("label") in skip or any(SITES[p["site"]][1] == "touch" for p in e["parts"]):
            out.append(dict(e, size_factor=1.0, rise_factor=1.0))
            continue
        if any(p.get("shape") == "lapse" for p in e["parts"]):
            raise ValueError("vary_sizes: a lapse draws its size in build_lapses (its ballistic timing)")
        for p in e["parts"]:
            if SITES[p["site"]][1] in ("parent", "world") and p["site"] not in caps:
                caps[p["site"]] = peak_of(base, p["site"], ROOM_CAP_SHARE, knot)
        parts = sized_parts(e["parts"], f, u, caps, rate)
        if "onset_s" in e:
            nxt = [t for t in ons if t > float(e["onset_s"]) + 1e-9]
            room = (nxt[0] - float(e["onset_s"]) - MIN_GAP_S) if nxt else math.inf
            if _event_dur(parts) > room + 1e-9:
                f, u = min(f, 1.0), min(u, 1.0)
                parts = sized_parts(e["parts"], f, u, caps, rate)
                if _event_dur(parts) > room + 1e-9:
                    f, u, parts = 1.0, 1.0, e["parts"]
        out.append(dict(e, parts=parts, size_factor=f, rise_factor=u, parts0=e["parts"], caps=caps))
    return out


def sized_parts(parts, f, u, caps, rate=RATE):
    """The parts of an event with every peak times f (a rotation peak kept under its cap in caps {site: rad}) and
    every rise and return times u (on knots), rebuilt by jerk_part (its speed and acceleration bounds)."""
    out = []
    for p in parts:
        pk = p["peak"] * f
        if p["site"] in caps:
            pk = math.copysign(min(abs(pk), caps[p["site"]]), pk)
        out.append(jerk_part(p["site"], pk, p["type"], delay_s=p.get("delay_s", 0.0), hold_s=p.get("hold_s", 0.0),
                             rate=rate, rise_s=max(_knot(u * p["rise_s"], rate), 1.0 / rate),
                             fall_s=max(_knot(u * p["fall_s"], rate), 1.0 / rate)))
    return out


SHRINK_STEPS = 8                    # [module] bisection steps of the standing size shrink (jerk_operator)


def shrunk(events, lam, rate=RATE):
    """The events of vary_sizes with each size and rise factor raised to the power lam (0: the median jerk of its
    kind, 1: as drawn): the standing clip keeps its balance block (jerk_operator)."""
    out = []
    for e in events:
        if "parts0" not in e:
            out.append(e)
            continue
        f, u = e["size_factor"] ** lam, e["rise_factor"] ** lam
        out.append(dict(e, parts=sized_parts(e["parts0"], f, u, e["caps"], rate), size_factor=f, rise_factor=u))
    return out


def size_record(events, sigma, rise_jitter):
    """{'rule', 'sigma', 'rise_jitter', 'size_factors', 'rise_factors'} of vary_sizes, JSON-ready."""
    return {"rule": "per event a lognormal size factor of median 1 (clipped, under the room cap) and a uniform rise "
                    "factor (module, wave SH1d part B)", "sigma": sigma, "clip": list(SIZE_CLIP),
            "rise_jitter": rise_jitter, "size_factors": [float(e.get("size_factor", 1.0)) for e in events],
            "rise_factors": [float(e.get("rise_factor", 1.0)) for e in events]}


def lapse_part(site, depth, silent_s, fall_s, rate=RATE):
    """One part of a lapse (shape 'lapse', module docstring) of a depth (m for a dip, else rad): the drop over the
    silent period and the catch from lapse_timing with the bounds of the site (LAPSE_ACCEL_MAX and LAPSE_CATCH_ACCEL
    for a dip, LAPSE_ACCEL_DEG_S2 for a wrist or an arm), the return over fall_s, no faster than DIP_ACCEL_MAX for a
    dip."""
    if SITES[site][1] == "dip":
        d, (mx, ca) = abs(depth), (LAPSE_ACCEL_MAX, LAPSE_CATCH_ACCEL)
        fall_s = max(fall_s, _ceil_knot(math.sqrt(MJ_PEAK_ACC * d / DIP_ACCEL_MAX), rate))
    else:
        d = math.degrees(abs(depth))
        mx, ca = LAPSE_ACCEL_DEG_S2["wrist" if site.startswith("wrist") else "arm"]
    t1, t2, a1, a2 = lapse_timing(d, silent_s, ca, mx, rate)
    return {"site": site, "peak": float(depth), "rise_s": round((t1 + t2) * rate) / rate, "fall_s": _knot(fall_s, rate),
            "hold_s": 0.0, "delay_s": 0.0, "type": "negative", "shape": "lapse", "drop_s": t1, "catch_s": t2,
            "drop_acc": a1, "catch_acc": a2, "acc_unit": "m/s2" if SITES[site][1] == "dip" else "deg/s2"}


def spread_delay_s(site, spread_ms):
    """s: the delay of a site in a spreading jerk (spread_ms by MRVS body area, RETICULAR_SPREAD_MS)."""
    return (spread_ms or {}).get(site_area(site), 0.0) / 1000.0


def _event_dur(parts):
    return max(p.get("delay_s", 0.0) + p["rise_s"] + p.get("hold_s", 0.0) + p["fall_s"] for p in parts)


def myoclonus_events(base, kind, repertoire, count, share, window, rng, process="poisson", period_s=None,
                     jitter_s=0.0, stimuli=None, latency_s=0.05, ref_knot=None, weights=None, room_body=None,
                     spread_ms=None, rise_s=None, fall_s=None, stimulus_entries=None):
    """Events of a myoclonus clip: each event one entry of the repertoire (a list of site lists, drawn at random:
    a focal or multifocal jerk moves one entry; a generalized jerk is one entry of many sites) at share x weight of
    each site's room (weights {site: w}, default 1: an event of several axial sites shares the grade among them;
    room_body: the body whose room sets the peak, default the base, the peak then kept under ROOM_CAP_SHARE of the
    base's own room), each part delayed by spread_ms of its area (a spreading jerk), timed by the type or by rise_s
    and fall_s, onsets by the process ('poisson' or 'action': count in window; 'rhythmic': period_s; 'stimulus': the
    stimuli times plus latency_s, the entry of each stimulus given by stimulus_entries). Returns (events, peaks {site:
    rad, or m for a dip}, peak_rule {site: {'room_deg', 'share', 'capped'}})."""
    k = int(round(window[0] * base.rate)) if ref_knot is None else int(ref_knot)
    w = weights or {}
    peaks, rule = {}, {}
    for s in _uniq([s for entry in repertoire for s in entry]):
        sh = share * w.get(s, 1.0)
        pk = peak_of(base if room_body is None else room_body, s, sh, k)
        capped = False
        if room_body is not None and SITES[s][1] in ("parent", "world"):
            cap = peak_of(base, s, ROOM_CAP_SHARE, k)
            capped, pk = pk > cap, min(pk, cap)
        peaks[s] = pk
        rule[s] = {"share": sh, "capped": bool(capped), "room_deg": None if SITES[s][1] in ("dip", "pelvis") else
                   sum(site_rooms(base if room_body is None else room_body, s, k))}

    def parts_of(entry):
        return [jerk_part(s, peaks[s], kind, delay_s=spread_delay_s(s, spread_ms), rise_s=rise_s, fall_s=fall_s)
                for s in entry]
    rhythmic = process == "rhythmic"
    dur = max(_event_dur(parts_of(entry)) for entry in repertoire)
    if process in ("poisson", "action"):
        onsets = place_events(count, window[0], window[1], dur, rng)
    elif process == "rhythmic":
        onsets = rhythmic_events(window[0], window[1], period_s, dur, rng, jitter_s)
    elif process == "stimulus":
        onsets = [_knot(s + latency_s) for s in stimuli]
    else:
        raise ValueError(f"process {process!r}")
    events = []
    for i, t in enumerate(onsets):
        if process == "stimulus" and stimulus_entries is not None:
            entry = repertoire[int(stimulus_entries[i])]
        else:
            entry = repertoire[int(rng.integers(len(repertoire)))] if len(repertoire) > 1 else repertoire[0]
        e = {"onset_s": float(t), "parts": parts_of(entry), "label": "+".join(entry)}
        if process == "stimulus":
            e["stimulus_s"] = float(stimuli[i])
        events.append(e)
    events = vary_sizes(base, events, rng, SIZE_SIGMA_RHYTHMIC if rhythmic else SIZE_SIGMA,
                        0.0 if rhythmic else RISE_JITTER, k)
    return events, peaks, rule


def _uniq(seq):
    out = []
    for x in seq:
        if x not in out:
            out.append(x)
    return out


# ---------------------------------------------------------------- clip builders
def _params(kind, process, rise_s=None, fall_s=None, burst=None, **kw):
    rise, fall = timing(kind)
    over = rise_s is not None
    return dict({"jerk_type": kind, "burst_ms": 1000.0 * (burst_s(kind) if burst is None else burst),
                 "burst_band_ms": list(JERK_TYPES[kind]["band_ms"]), "rise_s": rise_s if over else rise,
                 "fall_s": fall_s if over else fall, "timing": "module override" if over else "the type's",
                 "process": process, "twitch_s": TWITCH_S, "return_ratio": RETURN_RATIO, "min_gap_s": MIN_GAP_S,
                 "peak_speed_cap_deg_s": PEAK_SPEED_CAP_DEG_S, "dip_accel_max_m_s2": DIP_ACCEL_MAX}, **kw)


def stimulus_marks(stimuli, texts):
    """[{t_s, dur_s, text}]: the marks a viewer burns into the frames from each stimulus time (STIMULUS_MARK_S)."""
    return [{"t_s": float(s), "dur_s": STIMULUS_MARK_S, "text": x} for s, x in zip(stimuli, texts)]


def build_myoclonus(base, kind, repertoire, freq_grade, amp_grade, window, seed, operator, scale, sources,
                    process="poisson", period_s=None, jitter_s=0.0, stimuli=None, latency_s=0.05, count=None,
                    base_info=None, feet=(0, 1), ref_knot=None, extra=None, weights=None, room_body=None,
                    spread_ms=None, rise_s=None, fall_s=None, burst=None, stimulus_entries=None, stimulus_texts=None,
                    standing=False, grade=None):
    """(Body, truth) of a myoclonus clip (myoclonus_events on the base, jerk_operator; standing: the hip strategy and
    the balance block); grade: the truth's grade (default frequency and amplitude)."""
    rng = as_rng(seed)
    n = UMRS_COUNT_30S[freq_grade] if count is None else int(count)
    n = int(round(n * (window[1] - window[0]) / CLIP_S)) if process in ("poisson", "action") and count is None \
        and window[1] - window[0] < CLIP_S - 1e-9 else n
    share = UMRS_AMPLITUDE_SHARE[amp_grade]
    events, peaks, rule = myoclonus_events(base, kind, repertoire, n, share, window, rng, process, period_s,
                                           jitter_s, stimuli, latency_s, ref_knot, weights, room_body, spread_ms,
                                           rise_s, fall_s, stimulus_entries)
    span = window[1] - window[0]
    p = _params(kind, process, rise_s, fall_s, burst, window_s=[float(window[0]), float(window[1])],
                frequency_grade=freq_grade, amplitude_grade=amp_grade, amplitude_share=share, count=len(events),
                count_per_30s=len(events) * CLIP_S / span, repertoire=[list(e) for e in repertoire],
                peaks_deg={s: (v if SITES[s][1] == "dip" else math.degrees(v)) for s, v in peaks.items()},
                peak_rule=rule, room_body="the given body (before the posture)" if room_body is not None else "the base",
                room_cap_share=ROOM_CAP_SHARE if room_body is not None else None, spread_ms=spread_ms,
                period_s=period_s, jitter_s=jitter_s, latency_s=latency_s if stimuli else None, weights=weights,
                stimuli_s=list(stimuli) if stimuli else None, base_motion=base_info,
                stimulus_marks=stimulus_marks(stimuli, stimulus_texts) if stimuli and stimulus_texts else None,
                size_variation=size_record(events, SIZE_SIGMA_RHYTHMIC if process == "rhythmic" else SIZE_SIGMA,
                                           0.0 if process == "rhythmic" else RISE_JITTER))
    p.update(extra or {})
    grade = {"frequency": freq_grade, "amplitude": amp_grade} if grade is None else grade
    return jerk_operator(base, events, operator, grade, scale, p, sources,
                         seed if isinstance(seed, (int, np.integer)) else None, feet=feet, standing=standing)


def _finger_lapses(out, events, rng, depth_deg):
    """The body with the four fingers' knuckles (MCP) of each lapsing hand flexed with its wrist (module docstring:
    FINGER_LAPSE_SHARE of the wrist drop, at most FINGER_LAPSE_MAX_DEG on average, each finger x U(0.67, 1.33) and
    delayed by one of FINGER_LAPSE_DELAY_S), about the knuckle's flexion axis (hands.hand_frame), and the record."""
    from . import hands as HD
    fr = HD.hand_frame()
    t = out.times().numpy()
    Rh = out.hand_R().clone()
    R0 = Rh.clone()
    rec = []
    for e in events:
        for p in e["parts"]:
            if not p["site"].startswith("wrist_flexion_"):
                continue
            side = p["site"].rsplit("_", 1)[1]
            wrist = abs(math.degrees(p["peak"]))
            mean = min(FINGER_LAPSE_MAX_DEG, FINGER_LAPSE_SHARE * wrist)
            for f in ("index", "middle", "ring", "pinky"):
                j = HD.hand_joint(side, f, "mcp")
                deg = mean * float(rng.uniform(0.67, 1.33))
                dl = float(rng.choice(FINGER_LAPSE_DELAY_S))
                prof = deg * _part_profile(t, e["onset_s"] + p.get("delay_s", 0.0) + dl, p)
                act = np.abs(prof) > 0
                ax = torch.as_tensor(fr[side]["joints"][j]["flex"], dtype=F64)
                Rf = axis_angle_to_matrix(torch.from_numpy(np.radians(prof))[:, None] * ax)
                r = HD.hand_row(j)
                Rh[act, r] = Rf[act] @ Rh[act, r]
                rec.append({"onset_s": e["onset_s"], "side": side, "finger": f, "deg": deg, "delay_s": dl})
    rows = sorted({HD.hand_row(HD.hand_joint(s, f, "mcp")) for s in HD.SIDES for f in ("index", "middle", "ring",
                                                                                           "pinky")})
    res = out.with_hands(Rh, rows) if not torch.equal(Rh, R0) else out
    return res, {"knuckles": rec, "share_of_wrist": FINGER_LAPSE_SHARE, "max_mean_deg": FINGER_LAPSE_MAX_DEG,
                 "jitter": [0.67, 1.33], "delays_s": list(FINGER_LAPSE_DELAY_S),
                 "hand_limits": HD.hand_limit_report(res.hand_R())}


def build_lapses(base, sites_by_event, depth, kind, count, window, seed, operator, scale, sources, grade,
                 base_info=None, depth_unit="deg", feet=(0, 1), standing=False, fingers=False):
    """(Body, truth) of negative myoclonus: count lapses (lapse_part: the posture lost ballistically over the silent
    period, caught, then regained), each moving one entry of sites_by_event by depth (deg, or m for a dip), Poisson in
    the window; standing: the hip strategy and the balance block; fingers: the knuckles lapse with the wrists
    (_finger_lapses)."""
    rng = as_rng(seed)
    silent = burst_s(kind)
    fall = timing(kind)[1]

    k = int(round(window[0] * base.rate))

    def parts_of(entry, f=1.0):
        out = []
        for s in entry:
            d = f * (depth if SITES[s][1] == "dip" else math.radians(depth if np.isscalar(depth) else depth[s]))
            if SITES[s][1] in ("parent", "world"):
                d = min(d, peak_of(base, s, ROOM_CAP_SHARE, k))
            out.append(lapse_part(s, d, silent, fall))
        return out
    dur = max(_event_dur(parts_of(entry, SIZE_CLIP[1])) for entry in sites_by_event)
    onsets = place_events(count, window[0], window[1], dur, rng)
    entries = []
    for t in onsets:
        entries.append(sites_by_event[int(rng.integers(len(sites_by_event)))] if len(sites_by_event) > 1
                       else sites_by_event[0])
    events = []
    for t, entry in zip(onsets, entries):            # the sizes drawn after the onsets and entries (vary_sizes)
        f = float(np.clip(math.exp(SIZE_SIGMA * rng.standard_normal()), *SIZE_CLIP))
        events.append({"onset_s": float(t), "parts": parts_of(entry, f), "label": "+".join(entry), "size_factor": f,
                       "rise_factor": 1.0})
    span = window[1] - window[0]
    shapes = {s: {k: v for k, v in parts_of([s])[0].items() if k in ("rise_s", "fall_s", "drop_s", "catch_s",
                                                                       "drop_acc", "catch_acc", "acc_unit")}
              for s in _uniq([s for e in sites_by_event for s in e])}
    p = _params(kind, "poisson", window_s=[float(window[0]), float(window[1])], count=len(events),
                count_per_30s=len(events) * CLIP_S / span, depth=depth, depth_unit=depth_unit,
                lapse_sites=[list(e) for e in sites_by_event], base_motion=base_info, negative=True,
                lapse_shape="ballistic: constant acceleration over the silent period, constant deceleration to the "
                            "depth (the catch), minimum-jerk return (module)", lapse=shapes,
                silent_period_s=silent, size_variation=size_record(events, SIZE_SIGMA, 0.0))
    out, truth = jerk_operator(base, events, operator, grade, scale, p, sources,
                               seed if isinstance(seed, (int, np.integer)) else None, feet=feet, standing=standing)
    if fingers:
        out, rec = _finger_lapses(out, events, rng, depth)
        truth["params"]["finger_lapse"] = rec
        if truth.get("validity") is not None:
            truth["validity"]["hand_limits"] = {"ok": bool(rec["hand_limits"]["ok"]),
                                                "violations": rec["hand_limits"]["violations"]}
    return out, truth


# tic repertoire: type name -> list of sites (one part each, simultaneous)
TIC_TYPES = {
    "head_nod": ["neck_flexion"],
    "head_turn_right": ["neck_rotation_right"],
    "shoulder_shrug_left": ["shrug_left"],
    "arm_jerk_right": ["elbow_flexion_right", "shoulder_abduction_right"],
    "trunk_jerk": ["trunk_flexion"],
    "leg_kick_left": ["knee_extension_left"],
    "head_tilt_left": ["neck_tilt_left"],
    "shoulder_shrug_both": ["shrug_left", "shrug_right"],
    "wrist_flick_right": ["wrist_extension_right"],
}
TIC_ORDER = ("head_nod", "shoulder_shrug_left", "arm_jerk_right", "head_turn_right", "trunk_jerk", "leg_kick_left",
             "head_tilt_left", "shoulder_shrug_both", "wrist_flick_right")
ORCHESTRATED = ("head_turn_right", "shoulder_shrug_left", "arm_jerk_right")    # [module] one complex sequence
SQUAT_CM = 12.0                     # [module] the complex squat tic: the pelvis down 12 cm (knees near 63 deg); 15 cm
#                                     before the open-body port's wave 3: with the heels down the planted ankles
#                                     dorsiflexed 21.0 deg at 15 cm, 19.3 at 13 and 18.3 at 12 against jarm2022's 20
#                                     (scripts/openbody_wave3_checks.py, squat): 12 keeps 1.7 deg of room
SQUAT_RISE_S = 0.45                 # [module] down in 0.45 s (DIP_ACCEL_MAX needs 0.42 s at 15 cm, 0.38 at 12)
SQUAT_HOLD_S = 0.30                 # [module] held 0.3 s
SQUAT_RETURN_S = 0.50               # [module] up in 0.5 s
SQUAT_LEAN_DEG = {"hip_flexion": 15.0, "trunk_flexion": 10.0}   # [module] the squat's forward lean (pelvis, spine)


def ygtss_number_grade(n_types, n_complex=0, frequent_orchestrated=False):
    """The YGTSS motor number anchor (SOURCES['leckman1989']) of the tics shown: 1 a single type, 2 two to five, 3
    above five discrete types, 4 with at least one orchestrated pattern, 5 with several frequent ones."""
    if n_complex:
        return 5 if frequent_orchestrated else 4
    return 1 if n_types <= 1 else (2 if n_types <= 5 else 3)


def tic_counts(freq_grade, clip_s=CLIP_S, complexity=0, count=None):
    """(count, n_complex): the tics of a clip and how many are complex (tic_events)."""
    n = int(round(MRVS_LEVEL_PER_MIN[freq_grade] * clip_s / 60.0)) if count is None else int(count)
    return n, int(round(YGTSS_COMPLEX_SHARE.get(complexity, 0.0) * n))


def tic_events(base, types, count, intensity, rng, window, complex_share=0.0, bouts=None, squat=False,
               standing=False):
    """Events of a tic clip: count tics in the window (or inside the bouts, amplitude factor 0.5 + 0.5 sin over each
    bout: waxing and waning), every type at least once (the YGTSS number counts the types seen) and the others drawn
    with weights 1/rank (one tic dominates, module), at the intensity's
    share of the room; complex_share of them complex: the orchestrated sequence (ORCHESTRATED, parts ORCHESTRATED_STEP_S
    apart) or, standing with squat, a squat (SQUAT_CM down with the lean of SQUAT_LEAN_DEG, SQUAT_RISE_S down,
    SQUAT_HOLD_S held, SQUAT_RETURN_S up). Returns (events, peaks)."""
    share = YGTSS_INTENSITY_SHARE[intensity]
    k = int(round(window[0] * base.rate))
    all_sites = _uniq([s for ty in list(types) + (list(ORCHESTRATED) if complex_share else []) for s in TIC_TYPES[ty]])
    peaks = {s: peak_of(base, s, share, k) for s in all_sites}
    squat_peaks = {"knee_dip": SQUAT_CM / 100.0, **{s: math.radians(d) for s, d in SQUAT_LEAN_DEG.items()}}

    def squat_parts(fac):
        return [jerk_part(s, v * fac, "tic", hold_s=SQUAT_HOLD_S, rise_s=SQUAT_RISE_S, fall_s=SQUAT_RETURN_S)
                for s, v in squat_peaks.items()]
    simple_dur = max(_event_dur([jerk_part(s, peaks[s], "tic") for s in TIC_TYPES[ty]]) for ty in types)
    orch_dur = max(ORCHESTRATED_STEP_S * j + _event_dur([jerk_part(s, peaks[s], "tic") for s in TIC_TYPES[ty]])
                   for j, ty in enumerate(ORCHESTRATED)) if complex_share else 0.0
    dur = max(simple_dur, orch_dur, _event_dur(squat_parts(1.0)) if squat else 0.0)
    n_complex = int(round(complex_share * count))
    if bouts:
        tot = sum(b - a for a, b in bouts)
        per = [int(round(count * (b - a) / tot)) for a, b in bouts]
        per[-1] += count - sum(per)
        onsets = []
        for (a, b), m in zip(bouts, per):
            onsets += place_events(m, a, b, dur, rng)
    else:
        onsets = place_events(count, window[0], window[1], dur, rng)
    w = 1.0 / np.arange(1, len(types) + 1)
    w = w / w.sum()
    is_complex = np.zeros(len(onsets), bool)
    if n_complex:
        is_complex[rng.choice(len(onsets), n_complex, replace=False)] = True
    n_simple = int((~is_complex).sum())
    types = list(types)[:max(1, n_simple)]                   # fewer tics than types: the first types only
    w = 1.0 / np.arange(1, len(types) + 1)
    w = w / w.sum()
    seq = list(range(len(types))) + [int(x) for x in rng.choice(len(types), n_simple - len(types), p=w)]
    seq = [seq[i] for i in rng.permutation(len(seq))]          # every type at least once, then by weight
    events = []
    for i, t in enumerate(onsets):
        fac = 1.0
        if bouts:
            a, b = next((a, b) for a, b in bouts if a - 1e-9 <= t <= b + 1e-9)
            fac = 0.5 + 0.5 * math.sin(math.pi * (t - a) / (b - a))
        if is_complex[i]:
            if squat and rng.uniform() < 0.5:
                parts = squat_parts(fac)
                label = "complex:squat"
            else:
                parts = [jerk_part(s, peaks[s] * fac, "tic", delay_s=j * ORCHESTRATED_STEP_S)
                         for j, ty in enumerate(ORCHESTRATED) for s in TIC_TYPES[ty]]
                label = "complex:orchestrated"
        else:
            ty = types[seq.pop()]
            parts = [jerk_part(s, peaks[s] * fac, "tic") for s in TIC_TYPES[ty]]
            label = ty
        events.append({"onset_s": float(t), "parts": parts, "label": label, "factor": fac})
    events = vary_sizes(base, events, rng, knot=k, skip=("complex:squat",))
    return events, peaks


def build_tics(base, n_types, freq_grade, intensity, seed, window, complexity=0, bouts=None, count=None,
               squat=False, clip_s=CLIP_S, types=None, feet=(0, 1), standing=False):
    """(Body, truth) of a motor tic clip (tic_events, jerk_operator; standing: the hip strategy and the balance
    block); grades YGTSS number (ygtss_number_grade of the simple types shown, or of the orchestrated ones),
    intensity, complexity and MRVS frequency, areas and number recorded with their realized values."""
    rng = as_rng(seed)
    types = list(types or TIC_ORDER[:n_types])
    count, _ = tic_counts(freq_grade, clip_s, complexity, count)
    cs = YGTSS_COMPLEX_SHARE.get(complexity, 0.0)
    events, peaks = tic_events(base, types, count, intensity, rng, window, cs, bouts, squat)
    shown = [ty for ty in types if any(e["label"] == ty for e in events)]
    used = _uniq([s for e in events for p in e["parts"] for s in [p["site"]]])
    areas = sorted({site_area(s) for s in used})
    distinct = sorted({e["label"] for e in events})
    p = _params("tic", "bouts" if bouts else "poisson", window_s=[float(window[0]), float(window[1])],
                tic_types=types, n_types=len(types), tic_types_shown=shown, count=len(events), per_min=len(events) * 60.0 / clip_s,
                mrvs_frequency_grade=mrvs_frequency_grade(len(events) * 60.0 / clip_s), areas=areas,
                mrvs_area_grade=mrvs_area_grade(len(areas)), distinct_labels=distinct,
                n_complex=int(sum(e["label"].startswith("complex") for e in events)), complexity_share=cs,
                intensity=intensity, intensity_share=YGTSS_INTENSITY_SHARE[intensity],
                bouts_s=[list(b) for b in bouts] if bouts else None,
                peaks_deg={s: (v if SITES[s][1] == "dip" else math.degrees(v)) for s, v in peaks.items()},
                orchestrated=list(ORCHESTRATED) if cs else None, orchestrated_step_s=ORCHESTRATED_STEP_S,
                squat={"depth_cm": SQUAT_CM, "lean_deg": dict(SQUAT_LEAN_DEG), "rise_s": SQUAT_RISE_S,
                       "hold_s": SQUAT_HOLD_S, "return_s": SQUAT_RETURN_S} if squat else None,
                size_variation=size_record(events, SIZE_SIGMA, RISE_JITTER))
    n_complex = int(sum(e["label"].startswith("complex") for e in events))
    grade = {"ygtss_number": ygtss_number_grade(len(shown), n_complex, squat), "mrvs_frequency": freq_grade,
             "ygtss_intensity": intensity, "ygtss_complexity": complexity}
    return jerk_operator(base, events, "tics", grade, "YGTSS motor and MRVS", p,
                         ("leckman1989", "riechmann2023", "vanderveen2022", "table", "module"),
                         seed if isinstance(seed, (int, np.integer)) else None, feet=feet, standing=standing)


# touching tics (shared change SH15, module docstring): the hand, the timing and the kinds
TOUCH_SIDE = "right"                # [module] the touching hand (the dominant hand of reach.py)
TOUCH_RETURN_RATIO = 1.2            # [module] the way back lasts 1.2 times the reach
TOUCH_HOLD_S = 0.30                 # [module] a single touch: the finger held on the skin
TAP_N = 3                           # [module] the taps of a repeated tap
TAP_DEPTH_M = 0.015                 # [module] each tap lifts the fingertip 1.5 cm back along the approach line
TAP_OUT_S = 0.10                    # [module] the lift of a tap
TAP_IN_S = 0.10                     # [module] its return to the skin
TAP_REST_S = 0.06                   # [module] on the skin before each tap and after the last one
TOUCH_KINDS = {"nose": ("nose", 0), "chin": ("chin", 0), "cheek": ("cheek", 0), "forehead": ("forehead", 0),
               "nose_tap": ("nose", TAP_N), "chin_tap": ("chin", TAP_N)}   # kind: (reach.FACE_TARGETS, taps)
TOUCH_TIC_TYPES = ("head_nod", "shoulder_shrug_left", "arm_jerk_right", "head_turn_right")   # [module] the simple
#                                     tics of the touching clips: four, so that the number anchor does not hang on
#                                     counting each face target and the tap as separate tics (module docstring)
TOUCH_COMPLEXITY_MAX = 3            # [module reading of SOURCES['leckman1989'], not read in this task] brief touches
#                                     and taps reach complexity 3 at most, whatever their share of the tics
TOUCH_AREA_TOL_DEG = 0.05           # tic_areas' tolerance: a path component that moves more adds its area


def touch_site(target, side=TOUCH_SIDE):
    return f"touch_{target}_{side}"


def touch_label(kind):
    target, taps = TOUCH_KINDS[kind]
    return f"complex:{'tap' if taps else 'touch'}_{target}"


def touch_parts(kind, posture, side=TOUCH_SIDE, rate=RATE):
    """The parts of a touching tic (module docstring): the share of reach.touch_path from 0 to 1 over the rise of
    touch_timing (minimum jerk), held, back to 0; a tap kind holds TAP_REST_S + TAP_N tap cycles and adds TAP_N lifts
    of the share by TAP_DEPTH_M over the path's length (on the approach line, where the share runs with the
    fingertip's path length)."""
    from . import reach as RC
    target, n_taps = TOUCH_KINDS[kind]
    if TAP_DEPTH_M >= RC.TOUCH_APPROACH_M:
        raise ValueError("a tap must stay on the approach line")
    site = touch_site(target, side)
    cycle = TAP_OUT_S + TAP_IN_S + TAP_REST_S
    hold = TAP_REST_S + n_taps * cycle if n_taps else TOUCH_HOLD_S
    path = RC.touch_path(side, target, posture)
    length = path["length_m"]
    rise, fall = touch_timing(path, rate)
    parts = [{"site": site, "peak": 1.0, "rise_s": rise, "fall_s": fall, "hold_s": _knot(hold, rate), "delay_s": 0.0,
              "type": "touch", "path_length_m": length, "posture": posture}]
    for i in range(n_taps):
        parts.append({"site": site, "peak": -TAP_DEPTH_M / length, "rise_s": _knot(TAP_OUT_S, rate),
                      "fall_s": _knot(TAP_IN_S, rate), "hold_s": 0.0,
                      "delay_s": _knot(rise + TAP_REST_S + i * cycle, rate), "type": "tap"})
    return parts


def touch_timing(path, rate=RATE):
    """(rise_s, fall_s) of a touch along a reach.touch_path: the minimum-jerk reach whose fingertip peaks at
    reach.TOUCH_TIP_SPEED_M_S of its posture (reach.touch_rise_s, on the knot above, so that no joint turns faster
    than the path's check read) and the way back TOUCH_RETURN_RATIO times it."""
    from . import reach as RC
    rise = _ceil_knot(RC.touch_rise_s(path), rate)
    return rise, _knot(TOUCH_RETURN_RATIO * rise, rate)


def place_mixed(durs_s, t0, t1, rng, min_gap_s=MIN_GAP_S, rate=RATE):
    """[onset s] of events of durations durs_s (in their order) inside [t0, t1], at least min_gap_s apart, on knots:
    place_events with a duration per event (the free time spread by the same draw)."""
    n = len(durs_s)
    if not n:
        return []
    K0, K1 = int(round(t0 * rate)), int(round(t1 * rate))
    d, g = [int(round(x * rate)) for x in durs_s], int(round(min_gap_s * rate))
    free = (K1 - K0) - sum(d) - (n - 1) * g
    if free < 0:
        raise ValueError(f"{n} events of {sum(durs_s):.2f} s with gaps of {min_gap_s} s do not fit in {t1 - t0:.2f} s")
    gaps = rng.multinomial(free, rng.dirichlet(np.ones(n + 1)))
    out, k = [], K0
    for i in range(n):
        k += int(gaps[i])
        out.append(k / rate)
        k += d[i] + g
    return out


def touch_tic_events(base, posture, types, count, n_touch, intensity, rng, window, touches, side=TOUCH_SIDE):
    """Events of a touching-tic clip: count tics in the window, n_touch of them touching tics (touch_parts, every
    kind of touches at least once, the others drawn uniformly), the rest simple tics of types (every type at least
    once, the others with weights 1/rank, at the intensity's share of the room, as tic_events), in a random order,
    placed by place_mixed. Returns (events, peaks)."""
    share = YGTSS_INTENSITY_SHARE[intensity]
    k = int(round(window[0] * base.rate))
    peaks = {s: peak_of(base, s, share, k) for s in _uniq([s for ty in types for s in TIC_TYPES[ty]])}
    n_simple = count - n_touch
    types = list(types)[:max(1, n_simple)]
    w = 1.0 / np.arange(1, len(types) + 1)
    seq = list(range(len(types))) + [int(x) for x in rng.choice(len(types), n_simple - len(types), p=w / w.sum())]
    extra = rng.choice(len(touches), max(0, n_touch - len(touches)))
    kinds = list(touches)[:n_touch] + [touches[int(i)] for i in extra]
    items = [("simple", types[i]) for i in seq] + [("touch", kd) for kd in kinds]
    items = [items[i] for i in rng.permutation(len(items))]
    parts = [[jerk_part(s, peaks[s], "tic") for s in TIC_TYPES[name]] if what == "simple"
             else touch_parts(name, posture, side) for what, name in items]
    onsets = place_mixed([_event_dur(p) for p in parts], window[0], window[1], rng)
    events = [{"onset_s": float(t), "parts": p, "label": name if what == "simple" else touch_label(name),
               "factor": 1.0} for t, p, (what, name) in zip(onsets, parts, items)]
    return events, peaks


def touch_areas(path):
    """MRVS areas a touch path moves (tic_areas' reading of its joints): 'shoulders' for the collar, 'arms' for the
    shoulder or the elbow, 'hands' for the wrist, each when its components move more than TOUCH_AREA_TOL_DEG."""
    dq = np.abs(path["q"] - path["q_rest_deg"][None]).max(0)
    out = set()
    for area, sl in (("shoulders", slice(0, 3)), ("arms", slice(3, 9)), ("hands", slice(9, 12))):
        if dq[sl].max() > TOUCH_AREA_TOL_DEG:
            out.add(area)
    return out


def touch_readback(base, out, events, side=TOUCH_SIDE, every=3):
    """The touching tics read from the body (module docstring): per touch event, on the skinned mesh
    (reach.clearance), the gap between the hand and the head at every second knot on the skin (the share at 1), the
    least distance to the head, the trunk and the thighs at every `every` knot of the event (the trunk and thighs
    against the base at the onset), the taps (runs of the fingertip more than half TAP_DEPTH_M from its place on the
    skin) and their depth, the elbow's least drop below the shoulder joint on the skin, the fingertip's peak speed and
    the arm joints' peak angular speed; 'ok' when every gap lies in [0, reach.SKIN_GAP_MAX_M], nothing enters the
    head, the trunk and thighs keep reach.CLEAR_M (or their distance at rest), the taps are those planned, the elbow
    lies reach.TOUCH_ELBOW_DROP_M or more below the shoulder and every joint turns under
    reach.TOUCH_JOINT_SPEED_CAP_DEG_S."""
    from . import reach as RC
    rows, ok = [], True
    joints = list(TOUCH_ARMS[side])
    w = angular_speed_deg_s(out, base, joints).max(1)
    J = out.joints().numpy()
    for e in events:
        if not e["label"].startswith("complex:t"):
            continue
        site = e["parts"][0]["site"]
        sh = site_profiles(out.T, [e], out.rate)[site]
        act = np.flatnonzero(sh > 0)
        on = np.flatnonzero(sh == 1.0)
        c_on = RC.clearance(out, side, on[::2], parts=("head",))["head"]["hand"]
        kk = np.unique(np.r_[act[::every], act[-1]])
        c = RC.clearance(out, side, kk)
        rest = RC.clearance(base, side, [max(int(act[0]) - 1, 0)], parts=("trunk", "thighs"))
        body_ok = all(bool((np.minimum(c[p]["hand"], c[p]["fore"]) >= min(RC.CLEAR_M, rest[p]["hand"][0],
                                                                           rest[p]["fore"][0]) - 1e-4).all())
                      for p in ("trunk", "thighs"))
        tip = RC.fingertip(out, side, knots=act)
        place = RC.fingertip(out, side, knots=on).mean(0)
        dist = np.linalg.norm(tip - place, axis=1)
        hold = (act >= on[0]) & (act <= on[-1])
        taps = _runs(hold & (dist > 0.5 * TAP_DEPTH_M))
        planned = sum(1 for p in e["parts"] if p.get("type") == "tap")
        v = np.linalg.norm(np.gradient(tip, 1.0 / out.rate, axis=0), axis=1)
        spd = float(w[max(int(act[0]) - 1, 0):int(act[-1]) + 1].max())
        drop = float((J[on, joints[1], 1] - J[on, joints[2], 1]).min())
        r = {"label": e["label"], "onset_s": e["onset_s"], "target": SITES[site][2],
             "gap_mm_min": 1000.0 * float(c_on.min()), "gap_mm_max": 1000.0 * float(c_on.max()),
             "knots_on_skin": int(len(on)), "head_min_mm": 1000.0 * float(min(c["head"]["hand"].min(),
                                                                                c["head"]["fore"].min())),
             "trunk_min_mm": 1000.0 * float(np.minimum(c["trunk"]["hand"], c["trunk"]["fore"]).min()),
             "thighs_min_mm": 1000.0 * float(np.minimum(c["thighs"]["hand"], c["thighs"]["fore"]).min()),
             "taps_planned": planned, "taps_read": len(taps),
             "tap_depth_cm": [100.0 * float(dist[a:b].max()) for a, b in taps],
             "elbow_drop_cm": 100.0 * drop, "tip_peak_speed_m_s": float(v.max()), "joint_peak_speed_deg_s": spd,
             "knots_checked": int(len(kk))}
        r["ok"] = bool(0.0 <= c_on.min() and c_on.max() <= RC.SKIN_GAP_MAX_M and r["head_min_mm"] >= 0.0 and body_ok
                       and len(taps) == planned and drop >= RC.TOUCH_ELBOW_DROP_M
                       and spd <= RC.TOUCH_JOINT_SPEED_CAP_DEG_S)
        ok &= r["ok"]
        rows.append(r)
    return {"ok": bool(ok), "events": rows, "gap_max_mm": 1000.0 * RC.SKIN_GAP_MAX_M, "clear_mm": 1000.0 * RC.CLEAR_M,
            "elbow_drop_min_cm": 100.0 * RC.TOUCH_ELBOW_DROP_M, "joint_cap_deg_s": RC.TOUCH_JOINT_SPEED_CAP_DEG_S,
            "n_touches": len(rows)}


def build_touch_tics(base, posture, freq_grade, intensity, share_level, touches, seed, window, types=TOUCH_TIC_TYPES,
                     clip_s=CLIP_S, feet=(1,), standing=False, side=TOUCH_SIDE):
    """(HandBody, truth) of a motor tic clip whose complex tics touch the face (module docstring): the count of the
    MRVS frequency grade, YGTSS_COMPLEX_SHARE[share_level] of them touching tics of the kinds touches (TOUCH_KINDS),
    the rest simple tics of types (touch_tic_events, jerk_operator; standing: the hip strategy and the balance
    block); the YGTSS number of the discrete tics shown, simple and touching (a touch is a discrete complex tic, not
    an orchestrated pattern); the YGTSS complexity the ladder's level capped at TOUCH_COMPLEXITY_MAX; the touch
    read-back (touch_readback) and the hand limits in the validity."""
    from . import hands as HD
    from . import reach as RC
    rng = as_rng(seed)
    base = HD.hands_of(base)
    count, n_touch = tic_counts(freq_grade, clip_s, share_level)
    cs = YGTSS_COMPLEX_SHARE[share_level]
    events, peaks = touch_tic_events(base, posture, types, count, n_touch, intensity, rng, window, list(touches), side)
    events = vary_sizes(base, events, rng, rise_jitter=0.0, knot=int(round(window[0] * base.rate)))
    paths = {tg: RC.touch_path(side, tg, posture) for tg in sorted({TOUCH_KINDS[k][0] for k in touches})}
    shown = [ty for ty in types if any(e["label"] == ty for e in events)]
    labels = sorted({e["label"] for e in events})
    complex_shown = [x for x in labels if x.startswith("complex")]
    used = _uniq([p["site"] for e in events for p in e["parts"]])
    areas = sorted({site_area(s) for s in used if SITES[s][1] != "touch"}.union(
        *[touch_areas(paths[SITES[s][2]]) for s in used if SITES[s][1] == "touch"]))
    n_complex = int(sum(e["label"].startswith("complex") for e in events))
    path_rec = {tg: {"vertex": p["vertex"], "length_m": p["length_m"], "blend_length_m": p["blend_length_m"],
                     "approach_m": p["approach_m"], "dir": p["dir"].tolist(), "back_mm": 1000.0 * p["back_m"],
                     "gap_mm": 1000.0 * p["gap_m"], "ik_residual_mm": p["ik_residual_mm"],
                     "q_rest_deg": p["q_rest_deg"].tolist(), "q_touch_deg": p["q_touch_deg"].tolist(),
                     "areas": sorted(touch_areas(p)), "check": p["check"]} for tg, p in paths.items()}
    prm = _params("tic", "poisson", window_s=[float(window[0]), float(window[1])], tic_types=list(types),
                  n_types=len(types), tic_types_shown=shown, count=len(events), per_min=len(events) * 60.0 / clip_s,
                  mrvs_frequency_grade=mrvs_frequency_grade(len(events) * 60.0 / clip_s), areas=areas,
                  mrvs_area_grade=mrvs_area_grade(len(areas)), distinct_labels=labels, n_complex=n_complex,
                  complexity_share=cs, intensity=intensity, intensity_share=YGTSS_INTENSITY_SHARE[intensity],
                  peaks_deg={s: math.degrees(v) for s, v in peaks.items()}, frequent_orchestrated=False,
                  size_variation=size_record(events, SIZE_SIGMA, 0.0),
                  ygtss_number_rule="discrete tic types shown, simple and touching (a touch is a discrete complex "
                                    "tic, not an orchestrated pattern); counting every touch as one 'touching' tic "
                                    "and every tap as one 'tapping' tic gives the same anchor",
                  ygtss_number_lumped=ygtss_number_grade(len(shown) + len({x.split("_")[0] for x in complex_shown})),
                  complexity_rule=f"the share of the family's ladder level {share_level}, the grade capped at "
                                  f"{TOUCH_COMPLEXITY_MAX} (brief touches and taps that could be rationalized as "
                                  f"normal behaviour, SOURCES['leckman1989'])",
                  touch={"side": side, "posture": posture, "kinds": list(touches), "labels_shown": complex_shown,
                         "tip_speed_m_s": RC.TOUCH_TIP_SPEED_M_S[posture], "return_ratio": TOUCH_RETURN_RATIO,
                         "timing_s": {tg: list(touch_timing(p)) for tg, p in paths.items()},
                         "hold_s": TOUCH_HOLD_S, "taps": TAP_N,
                         "tap_depth_m": TAP_DEPTH_M, "tap_out_s": TAP_OUT_S, "tap_in_s": TAP_IN_S,
                         "tap_rest_s": TAP_REST_S, "gap_goal_mm": 1000.0 * RC.SKIN_GAP_M,
                         "gap_max_mm": 1000.0 * RC.SKIN_GAP_MAX_M,
                         "hand_form_shares": [RC.HAND_FORM_START, RC.HAND_FORM_END],
                         "elbow_drop_min_m": RC.TOUCH_ELBOW_DROP_M, "joint_cap_deg_s": RC.TOUCH_JOINT_SPEED_CAP_DEG_S,
                         "blend_lead": RC.BLEND_LEAD,
                         "dir_deg": {tg: RC.TOUCH_DIR_DEG[tg] for tg in paths}, "paths": path_rec,
                         "rule": "the share of reach.touch_path on a minimum-jerk profile (module)"})
    grade = {"ygtss_number": ygtss_number_grade(len(shown) + len(complex_shown)), "mrvs_frequency": freq_grade,
             "ygtss_intensity": intensity, "ygtss_complexity": min(share_level, TOUCH_COMPLEXITY_MAX)}
    out, truth = jerk_operator(base, events, "tics", grade, "YGTSS motor and MRVS", prm,
                               ("leckman1989", "ygtss_checklist", "riechmann2023", "vanderveen2022", "table", "module"),
                               seed if isinstance(seed, (int, np.integer)) else None, feet=feet, standing=standing)
    rb = touch_readback(base, out, events, side)
    truth["params"]["touch_readback"] = rb
    hl = HD.hand_limit_report(out.hand_R())
    truth["validity"]["hand_limits"] = {"ok": bool(hl["ok"]), "violations": hl["violations"]}
    truth["validity"]["face_touch"] = {"ok": rb["ok"], "n_touches": rb["n_touches"],
                                       "gap_mm_range": [min(r["gap_mm_min"] for r in rb["events"]),
                                                        max(r["gap_mm_max"] for r in rb["events"])],
                                       "head_min_mm": min(r["head_min_mm"] for r in rb["events"]),
                                       "trunk_min_mm": min(r["trunk_min_mm"] for r in rb["events"]),
                                       "thighs_min_mm": min(r["thighs_min_mm"] for r in rb["events"])}
    return out, truth


MRVS_LEVEL_PER_MIN = {g: _band_level(b) for g, b in MRVS_PER_MIN.items()}       # 10.5, 30.5, 50.5, 73.2 (module)
STARTLE_SHARES = {"neck_flexion": 0.15, "shrug_left": 0.15, "shrug_right": 0.15, "shoulder_abduction_left": 0.05,
                  "shoulder_abduction_right": 0.05, "elbow_flexion_left": 0.10, "elbow_flexion_right": 0.10,
                  "trunk_flexion": 0.08}                                        # [module] the normal startle
STARTLE_DIP_M = 0.01                 # [module] the knee flexion of a standing startle (pelvis lowered 1 cm)
HYPER_GAIN = 2.5                     # [module] the hyperekplexic response over the normal one (capped at 0.9 room)
HYPER_UNSCALED = ("shoulder_abduction_left", "shoulder_abduction_right", "shrug_left", "shrug_right")
#                                      the cited abduction is slight and the cited pattern names no shoulder elevation:
#                                      kept at the normal level (visual check of 2026-11-02: scaled, the arms flared out)
STIMULI_S = (5.0, 14.0, 23.0)        # [module] three stated stimuli (a sudden loud sound)
STARTLE_CLIPS = (("normal", "stand"), ("hyperekplexia", "stand"), ("normal", "sit"), ("hyperekplexia", "sit"))
STARTLE_SEGMENT = {"stand": "E7", "sit": "E1"}
STARTLE_PAIRING = ("level 0: the normal startle of the same posture to the same sounds (the clinical comparison: "
                   "an exaggerated startle against a normal one); the normal startle's own level 0 is the "
                   "stimulus-free body, so the series root does not change")
#                                      an open defect of handoff 13 (2026-11-02): beside a stimulus-free level 0 the
#                                      viewer showed a body that hears nothing; the seated normal startle
#                                      (jerk_startle_E1_normal, new) partners the seated hyperekplexia, on the same
#                                      stimulus times (STIMULI_S) and the same seed


def startle_id(condition, posture):
    """The clip id of a startle clip: jerk_startle_<E7 or E1>_<condition>."""
    return f"jerk_startle_{STARTLE_SEGMENT[posture]}_{condition}"


def startle_level0(condition, posture, neutral):
    """The level 0 of a startle clip (STARTLE_PAIRING): the normal startle of the same posture for a hyperekplexia,
    the stimulus-free body of the posture (neutral) for a normal startle."""
    return startle_id("normal", posture) if condition == "hyperekplexia" else neutral


def build_startle(base, condition, seed, standing, stimuli=STIMULI_S):
    """(Body, truth) of the startle reflex to stated stimuli: the cited pattern (neck and trunk flexion, shoulder
    elevation, slight arm abduction, elbow flexion; standing, a knee flexion), sternocleidomastoid onset at the cited
    latency and a module craniocaudal spread, habituating over trials (HABITUATION); 'hyperekplexia' at HYPER_GAIN,
    followed by STIFFNESS_S of held posture (the short generalized stiffness) and habituating poorly
    (HYPER_HABITUATION). Standing: the hip strategy and the balance block (jerk_operator)."""
    if condition not in ("normal", "hyperekplexia"):
        raise ValueError(f"condition {condition!r}")
    gain = 1.0 if condition == "normal" else HYPER_GAIN
    hold = 0.0 if condition == "normal" else STIFFNESS_S
    habit = HABITUATION if condition == "normal" else HYPER_HABITUATION
    k = int(round(stimuli[0] * base.rate))
    lat = STARTLE_SCM_MS / 1000.0
    events, peaks = [], {}
    shares = {s: (sh if s in HYPER_UNSCALED else min(0.9, sh * gain)) for s, sh in STARTLE_SHARES.items()}
    for s, sh in shares.items():
        peaks[s] = peak_of(base, s, sh, k)
    if standing:
        peaks["knee_dip"] = STARTLE_DIP_M * gain
    srng = as_rng(seed if isinstance(seed, (int, np.integer)) else SEED)
    sizes = np.clip(np.exp(SIZE_SIGMA_RHYTHMIC * srng.standard_normal(len(stimuli))), *SIZE_CLIP)
    for i, t0 in enumerate(stimuli):
        fac = habit[min(i, len(habit) - 1)] * float(sizes[i])
        parts = []
        for s, pk in peaks.items():
            area = site_area(s)
            d = STARTLE_SPREAD_MS["legs" if area == "legs" else area] / 1000.0
            parts.append(jerk_part(s, pk * fac, "startle", delay_s=d, hold_s=hold))
        events.append({"onset_s": _knot(t0 + lat), "parts": parts, "label": f"startle trial {i + 1}",
                       "stimulus_s": float(t0), "factor": fac, "size_factor": float(sizes[i])})
    p = _params("startle", "stimulus", condition=condition, stimuli_s=list(stimuli), latency_ms=STARTLE_SCM_MS,
                spread_ms=STARTLE_SPREAD_MS, habituation=list(habit),
                habituation_rule="habituating within 2-5 trials (cited), factors module" if condition == "normal"
                else "poor habituation (module): the response kept over the trials", gain=gain, stiffness_s=hold,
                shares=shares, unscaled=list(HYPER_UNSCALED),
                peaks_deg={s: (v if SITES[s][1] == "dip" else math.degrees(v)) for s, v in peaks.items()},
                response_end_ms=1000.0 * max(_event_dur(e["parts"]) for e in events) + STARTLE_SCM_MS,
                cited_end_ms=STARTLE_END_MS, standing=bool(standing),
                size_variation=dict(size_record(events, SIZE_SIGMA_RHYTHMIC, 0.0),
                                    rule="each trial's habituation factor times a lognormal size factor of median 1 "
                                         "(module, wave SH1d part B)"),
                stimulus_marks=stimulus_marks(stimuli, ["SOUND"] * len(stimuli)))
    return jerk_operator(base, events, "startle", condition, "no scale (normal against hyperekplexia)", p,
                         ("vanderveen2022", "module", "table"), seed if isinstance(seed, (int, np.integer)) else None,
                         standing=bool(standing))


def _ankle_dorsi_deg(R_ankle, R_base):
    """(T,) deg: the dorsiflexion (-x) of ankle rotations against the base's (signed_angle_deg's formula)."""
    D = R_ankle @ R_base.transpose(-1, -2)
    c = (D[..., 0, 0] + D[..., 1, 1] + D[..., 2, 2] - 1) / 2
    return -torch.rad2deg(torch.atan2(vee(D)[:, 0], c)).numpy()


def clonus_angle(t, t_on, dur, scats, frequency_hz, pp_deg, rng, stretch_deg, stretch_s, end_s):
    """((T,) deg ankle dorsiflexion, envelope (T,), info): the loading ramp to stretch_deg over 0.15 s from stretch_s,
    then beats about it for dur s from t_on: phase from a frequency drifting by up to +-CLONUS_HZ_DRIFT (a sine of a
    whole number of cycles over the span, zero mean), amplitude pp_deg / 2 drifting by up to +-CLONUS_AMP_DRIFT, the
    envelope constant then tapering to 0 over the second half (raised cosine, SCATS 1 and 2), or constant to end_s with
    a CLONUS_TAPER_S taper (SCATS 3)."""
    ramp = minimum_jerk((t - stretch_s) / 0.15)
    env = np.zeros_like(t)
    osc = np.zeros_like(t)
    info = {}
    if dur > 0:
        u = t - t_on
        env = np.where((u >= 0) & (u <= dur), 1.0, 0.0)
        if scats in (1, 2):
            tail = np.clip((u / dur - 0.5) / 0.5, 0.0, 1.0)
        else:
            tail = np.clip((u - (dur - CLONUS_TAPER_S)) / CLONUS_TAPER_S, 0.0, 1.0)
        env = env * (0.5 + 0.5 * np.cos(np.pi * tail))
        nf, na = max(1, int(round(dur / 6.0))), max(1, int(round(dur / 5.0)))
        sf, sa = float(rng.choice((-1.0, 1.0))), float(rng.choice((-1.0, 1.0)))
        w = 2.0 * math.pi * nf / dur
        phase = 2.0 * math.pi * frequency_hz * u - sf * 2.0 * math.pi * CLONUS_HZ_DRIFT / w * (np.cos(w * u) - 1.0)
        amp = 1.0 + sa * CLONUS_AMP_DRIFT * np.sin(2.0 * math.pi * na * u / dur)
        osc = 0.5 * pp_deg * env * amp * np.sin(phase)
        info = {"frequency_drift_hz": sf * CLONUS_HZ_DRIFT, "frequency_drift_cycles": nf,
                "amplitude_drift": sa * CLONUS_AMP_DRIFT, "amplitude_drift_cycles": na}
    return stretch_deg * ramp + osc, env, info


def build_clonus(base, scats, seed, side="left", stretch_s=2.0, frequency_hz=CLONUS_HZ, pp_deg=CLONUS_PP_DEG):
    """(Body, truth) of ankle clonus with the forefoot loaded (held_leg_base, catalogue row O01): the knee pressed
    down at stretch_s (not drawn) dorsiflexes the ankle by CLONUS_STRETCH_DEG over 0.15 s, then beats at
    frequency_hz of pp_deg peak to peak about the loaded angle for the SCATS duration (1: 2.0 s, 2: the geometric
    centre of 3 and 10 s, 3: to the clip's end, more than 10 s) with the drifts and envelope of clonus_angle. Each
    knot's foot turn about the foot joint is solved (Newton) so that the ankle's own angle against the base is the
    planned one: the heel bounces, the knee bobs, the ball of the foot stays put. scats 0: the loading, no beat."""
    ankle = FOOT[side][2]
    t = base.times().numpy()
    rng = as_rng(seed)
    t_on = _knot(stretch_s + 0.15)
    end = base.duration - 0.5
    dur = {0: 0.0, 1: 2.0, 2: math.sqrt(3.0 * 10.0), 3: end - t_on}[int(scats)]
    target, env, drift = clonus_angle(t, t_on, dur, int(scats), frequency_hz, pp_deg, rng, CLONUS_STRETCH_DEG,
                                      stretch_s, end)
    act = np.abs(target) > 0
    sit = _sit_body(base.T)
    psi0 = forefoot_psi0(side)
    R0 = base.local_R()[:, ankle]
    dpsi = np.zeros(base.T)
    for _ in range(8):                                # Newton on the foot turn, the slope by a central difference
        h = 1e-5
        f = lambda d: _ankle_dorsi_deg(forefoot_rows(sit, side, torch.from_numpy(psi0 + d))[0][:, 2], R0)  # noqa
        e = f(dpsi) - target
        if np.abs(e[act]).max(initial=0.0) < 1e-7:
            break
        g = (f(dpsi + h) - f(dpsi - h)) / (2 * h)
        dpsi = np.where(act, dpsi - e / g, 0.0)
    rows, knee = forefoot_rows(sit, side, torch.from_numpy(psi0 + dpsi))
    out = _with_leg(base, side, rows, act)
    other = (1,) if side == "left" else (0,)
    out.contact = contact_labels(out)
    realized = _ankle_dorsi_deg(out.local_R()[:, ankle], R0)
    beats, _ = beat_peaks(realized)
    hh = heel_height_m(out, side, np.arange(0, base.T, 2))
    ft = FOOT[side][3]
    ball = float((out.joints()[:, ft] - base.joints()[:, ft]).norm(dim=-1).max())
    params = {"side": side, "joint": ankle, "frequency_hz": frequency_hz, "pp_deg": pp_deg,
              "stretch_deg": CLONUS_STRETCH_DEG, "stretch_s": stretch_s, "beats_onset_s": t_on, "duration_s": dur,
              "scats": int(scats), "beats_planted": len(beats),
              "first_beat_s": beats[0] / base.rate if beats else None,
              "last_beat_s": beats[-1] / base.rate if beats else None, "drift": drift,
              "posture": "seated, the forefoot loaded: the ball of the foot on the floor, the heel raised "
                         "(catalogue row O01)", "heel_up_m": HEEL_UP_M, "forefoot_back_m": FOREFOOT_BACK_M,
              "foot_turn_deg": math.degrees(psi0), "knee_flexion_deg_range": [float(knee.min()), float(knee.max())],
              "heel_height_cm_range": [100.0 * float(hh.min()), 100.0 * float(hh.max())],
              "ball_moved_m": ball, "angle_error_deg": float(np.abs(realized - target).max()),
              "envelope": "constant, then a raised-cosine taper over the second half (SCATS 1 and 2); constant to "
                          f"the clip's end with a {CLONUS_TAPER_S:g} s taper (SCATS 3)",
              "trigger": "the knee pressed down (loading the forefoot) at the stretch time, not drawn",
              "grade_note": "SCATS duration bands (benz2005) on beats after loading the forefoot; the SCATS test "
                            "itself elicits them by the examiner's rapid dorsiflexion"}
    val = validity(base, out, feet=other)
    val["forefoot"] = {"ok": bool(ball < 1e-9 and hh.min() > 0.005), "ball_moved_m": ball,
                       "heel_min_cm": 100.0 * float(hh.min())}
    return out, truth_record("clonus", seed if isinstance(seed, (int, np.integer)) else None, int(scats),
                             "SCATS clonus", params, ("benz2005", "wallace2012", "catalogue_o01", "module", "table"),
                             val, arrays={"angle_deg": target, "envelope": env, "foot_turn_deg": np.degrees(psi0 + dpsi)})


# ---------------------------------------------------------------- the clip specifications
def _spec(**kw):
    from .exam import ClipSpec
    kw.setdefault("since", SINCE)
    return ClipSpec(**kw)


def _level0(cid, segment, posture, seconds, views, scene, title=""):
    def build():
        b = _static(posture, seconds)
        return b, truth_record("none", None, 0, "level 0", {"base": f"procedural {posture}", "clip_s": seconds},
                               (), None)
    return _spec(clip_id=cid, family="reference", segment=segment, item="level 0", operator=f"none (procedural "
                 f"{posture})", grade=0, grade_text="level 0", params_text=f"the shared neutral body, {posture}, "
                 f"{seconds:g} s", views=views, level0=None, scene=scene, build=build, segment_title=title)


def _deg(x):
    return f"{x:.1f} deg"


def specs():
    """[ClipSpec] of the brief-jerk family (module docstring), since 'catalogue_v1'."""
    out = []
    L_E1, L_E7, L_E1_60 = "jerk_E1_level0", "jerk_E7_level0", "jerk_E1_level0_60s"
    L_E2 = "tremor_E2_raise_g0"                       # exam clip: motions.outstretched(30 s), the same base
    out += [_level0(L_E1, "E1", "sit", CLIP_S, ("front", "left"), "chair"),
            _level0(L_E7, "E7", "stand", CLIP_S, ("front", "left"), None),
            _level0(L_E1_60, "E1", "sit", 60.0, ("front", "left"), "chair")]
    rise_c, fall_c = timing("cortical")

    # -- cortical focal action myoclonus, left arm, arms outstretched (UMRS action)
    rep_focal = [["wrist_extension_left"], ["wrist_flexion_left"], ["arm_raise_left", "wrist_extension_left"],
                 ["elbow_flexion_left", "arm_raise_left"]]       # the elbow jerk with the arm: the hand up and in

    def focal(fg, ag):
        def build():
            b, info = outstretched_base()
            win = (info["onset_s"] + 0.5, b.duration - 0.5)
            return build_myoclonus(b, "cortical", rep_focal, fg, ag, win, SEED + 10 * fg + ag, "myoclonus",
                                   "UMRS action myoclonus", ("frucht2002", "conn_asterixis", "vanderveen2022",
                                                             "merchant2020", "table", "module"), process="action",
                                   base_info={"onset_s": info["onset_s"], "raise_s": info["raise_s"]},
                                   ref_knot=int(round((info["onset_s"] + 0.5) * b.rate)), weights=FOCAL_WEIGHTS)
        return build
    plan = [(fg, 2) for fg in (1, 2, 3, 4)] + [(3, ag) for ag in (1, 3, 4)]
    for fg, ag in plan:
        out.append(_spec(clip_id=f"jerk_myoclonus_E2_cortical_focal_f{fg}_a{ag}", family="myoclonus", segment="E2",
                         item="UMRS action myoclonus (left arm)", operator="myoclonus, cortical, focal, left arm, "
                         "action-induced", grade={"frequency": fg, "amplitude": ag},
                         grade_text=f"UMRS action: frequency {fg}, amplitude {ag}",
                         params_text=f"{UMRS_COUNT_30S[fg]} jerks per 30 s of posture, {100 * UMRS_AMPLITUDE_SHARE[ag]:g} "
                         f"percent of the joint's range in that direction; burst "
                         f"{1000 * burst_s('cortical'):.0f} ms, movement "
                         f"{1000 * (rise_c + fall_c):.0f} ms", views=("front", "left", "top", "hands"), level0=L_E2,
                         scene="chair", build=focal(fg, ag)))

    # -- multifocal and generalized at rest (seated)
    rep_multi = [["wrist_extension_left"], ["elbow_flexion_right"], ["neck_flexion"], ["shrug_left"],
                 ["knee_extension_right"], ["trunk_rotation_left"], ["shoulder_abduction_left"]]
    rep_gen = [["neck_flexion", "shrug_left", "shrug_right", "elbow_flexion_left", "elbow_flexion_right",
                "trunk_flexion", "hip_flexion", "shoulder_abduction_left", "shoulder_abduction_right"]]
    for name, rep, kind, fgs in (("multifocal", rep_multi, "cortical", (2, 3)),
                                 ("generalized", rep_gen, "reticular", (2, 3))):
        for fg in fgs:
            def build(rep=rep, kind=kind, fg=fg, name=name):
                b = _static("sit")
                feet = (0,) if name == "multifocal" else (0, 1)
                gen = name == "generalized"
                return build_myoclonus(b, kind, rep, fg, 2, (0.5, b.duration - 0.5), SEED + 100 + fg + len(rep),
                                       "myoclonus", "UMRS rest myoclonus", ("frucht2002", "conn_asterixis",
                                                                            "vanderveen2022", "table", "module"),
                                       feet=feet, weights=GENERALIZED_WEIGHTS if gen else None,
                                       spread_ms=RETICULAR_SPREAD_MS if gen else None)
            r, f = timing(kind)
            out.append(_spec(clip_id=f"jerk_myoclonus_E1_{name}_f{fg}_a2", family="myoclonus", segment="E1",
                             item="UMRS rest myoclonus", operator=f"myoclonus, {name}, at rest ({kind})",
                             grade={"frequency": fg, "amplitude": 2},
                             grade_text=f"UMRS rest: frequency {fg}, amplitude 2 (each region involved)",
                             params_text=f"{UMRS_COUNT_30S[fg]} jerks per 30 s, {100 * UMRS_AMPLITUDE_SHARE[2]:g} percent "
                             f"of the room; burst {1000 * burst_s(kind):.0f} ms, movement {1000 * (r + f):.0f} ms",
                             views=("front", "left"), level0=L_E1, scene="chair", build=build))

    # -- rhythmic segmental myoclonus, right arm (C5-C6), seated
    seg_hz = math.sqrt(SEGMENTAL_HZ[0] * SEGMENTAL_HZ[1])
    period = _knot(1.0 / seg_hz)
    for ag in (2, 3):
        def build(ag=ag):
            b = _static("sit")
            return build_myoclonus(b, "segmental", [["shoulder_abduction_right", "elbow_flexion_right"]], 4, ag,
                                   (0.5, b.duration - 0.5), SEED + 200 + ag, "myoclonus", "UMRS rest myoclonus",
                                   ("frucht2002", "vanderveen2022", "termsarasab2015", "catalogue_m05", "table",
                                    "module"), process="rhythmic", period_s=period, jitter_s=0.02,
                                   rise_s=SEGMENTAL_BURST_S + TWITCH_S, fall_s=SEGMENTAL_RETURN_S,
                                   burst=SEGMENTAL_BURST_S)
        r, f = _knot(SEGMENTAL_BURST_S + TWITCH_S), SEGMENTAL_RETURN_S
        out.append(_spec(clip_id=f"jerk_myoclonus_E1_segmental_right_arm_a{ag}", family="myoclonus", segment="E1",
                         item="UMRS rest myoclonus (right arm)", operator="myoclonus, spinal segmental, rhythmic, "
                         "right arm", grade={"frequency": 4, "amplitude": ag},
                         grade_text=f"UMRS rest: frequency 4 (rhythmic), amplitude {ag}",
                         params_text=f"{1 / period:.2f} Hz rhythmic (period {period:.2f} s, jitter 20 ms); burst "
                         f"{1000 * SEGMENTAL_BURST_S:.0f} ms, movement {1000 * (r + f):.0f} ms",
                         views=("front", "right"), level0=L_E1, scene="chair", build=build))

    # -- propriospinal: axial flexion jerks (trunk, hips; standing also the knees)
    for posture, fg, ag in (("stand", 2, 3), ("stand", 3, 2), ("sit", 3, 3), ("sit", 1, 4)):
        def build(posture=posture, fg=fg, ag=ag):
            b = _static(posture)
            rep = [["trunk_flexion", "hip_flexion", "neck_flexion"] + (["knee_dip"] if posture == "stand" else [])]
            return build_myoclonus(b, "propriospinal", rep, fg, ag, (0.5, b.duration - 0.5), SEED + 300 + fg + ag,
                                   "myoclonus", "UMRS rest myoclonus (trunk)", ("frucht2002", "conn_asterixis",
                                                                                  "vanderveen2022", "merchant2020",
                                                                                  "table", "module"),
                                   extra={"dip_m_per_share": DIP_PER_SHARE_M}, weights=AXIAL_WEIGHTS,
                                   standing=posture == "stand")
        seg = "E7" if posture == "stand" else "E1"
        r, f = timing("propriospinal")
        out.append(_spec(clip_id=f"jerk_myoclonus_{seg}_propriospinal_f{fg}_a{ag}", family="myoclonus", segment=seg,
                         item="UMRS rest myoclonus (trunk)", operator=f"myoclonus, propriospinal, axial flexion "
                         f"({'standing' if posture == 'stand' else 'seated'})", grade={"frequency": fg, "amplitude": ag},
                         grade_text=f"UMRS rest: frequency {fg}, amplitude {ag} (trunk)",
                         params_text=f"{UMRS_COUNT_30S[fg]} jerks per 30 s, trunk, hips"
                         f"{', knees' if posture == 'stand' else ''} and neck flexing together; burst "
                         f"{1000 * burst_s('propriospinal'):.0f} ms, movement {1000 * (r + f):.0f} ms",
                         views=("left", "front"), level0=L_E7 if posture == "stand" else L_E1,
                         scene=None if posture == "stand" else "chair", build=build))

    # -- stimulus-sensitive (reflex) myoclonus: jerks locked to stated stimuli, a tap on the hand of the jerking arm
    stim = tuple(s for s, _ in STIMULUS_TAPS)
    sides = [side for _, side in STIMULUS_TAPS]
    for ag in (2, 3):
        def build(ag=ag):
            b, info = outstretched_base()
            return build_myoclonus(b, "cortical", [["arm_raise_left", "wrist_extension_left"],
                                                    ["arm_raise_right", "wrist_extension_right"]], 3, ag,
                                   (info["onset_s"] + 0.5, b.duration - 0.5), SEED + 400 + ag, "myoclonus",
                                   "UMRS stimulus sensitivity", ("frucht2002", "vanderveen2022", "table", "module"),
                                   process="stimulus", stimuli=stim, latency_s=STIMULUS_LATENCY_S,
                                   base_info={"onset_s": info["onset_s"]},
                                   ref_knot=int(round((info["onset_s"] + 0.5) * b.rate)),
                                   stimulus_entries=[0 if s == "left" else 1 for s in sides],
                                   grade={"stimulus_sensitivity": 1, "amplitude": ag},
                                   stimulus_texts=[f"TAP {s} hand" for s in sides],
                                   extra={"stimulus": "a light tap on the back of the outstretched hand of the jerking "
                                                      "arm (module plan STIMULUS_TAPS), stated in the caption and in "
                                                      "stimulus_marks (the examiner not drawn)",
                                          "stimulus_sides": sides, "stimulus_sensitivity": 1})
        taps = "; ".join(f"taps on the {side} hand at " + ", ".join(f"{s:g}" for s, x in STIMULUS_TAPS if x == side)
                         + " s" for side in ("left", "right"))
        out.append(_spec(clip_id=f"jerk_myoclonus_E2_stimulus_a{ag}", family="myoclonus", segment="E2",
                         item="UMRS stimulus sensitivity", operator="myoclonus, stimulus-sensitive (reflex)",
                         grade={"stimulus_sensitivity": 1, "amplitude": ag},
                         grade_text=f"UMRS stimulus sensitivity present (1), amplitude {ag}",
                         params_text=f"{taps}; the tapped arm jerks {1000 * STIMULUS_LATENCY_S:.0f} ms after each",
                         views=("front", "left", "top"), level0=L_E2, scene="chair", build=build))

    # -- negative myoclonus: asterixis (Conn grades x UMRS negative severity)
    L_AST = "jerk_asterixis_E2_level0"

    def ast0():
        b, info = asterixis_base()
        return b, truth_record("none", None, 0, "level 0", {"base": "outstretched, wrists extended",
                                                            "base_motion": {k: v for k, v in info.items()
                                                                            if k != "finger_gaps_mm"}}, (), None)
    out.append(_spec(clip_id=L_AST, family="reference", segment="E2", item="level 0", operator="none (arms "
                     "outstretched, wrists extended)", grade=0, grade_text="level 0", params_text="no lapse",
                     views=("left", "front", "hands"), level0=None, scene="chair", build=ast0,
                     segment_title="arms outstretched, wrists extended"))
    rep_ast = [["wrist_flexion_left"], ["wrist_flexion_right"]]
    rep_ast3 = [["wrist_flexion_left", "arm_drop_left"], ["wrist_flexion_right", "arm_drop_right"]]
    for cg, sev in [(g, 2) for g in (1, 2, 3, 4)] + [(3, 1), (3, 3)]:
        def build(cg=cg, sev=sev):
            b, info = asterixis_base()
            depth = NEGATIVE_DEPTH_DEG[sev]
            rep = rep_ast3 if sev == 3 else rep_ast
            dep = {s: (depth if s.startswith("wrist") else depth / 4.0) for e in rep for s in e}
            return build_lapses(b, rep, dep, "negative", UMRS_COUNT_30S[cg],
                                (info["onset_s"] + 0.5, b.duration - 0.5), SEED + 500 + 10 * cg + sev, "asterixis",
                                "Conn asterixis grade and UMRS negative myoclonus",
                                ("conn_asterixis", "frucht2002", "vanderveen2022", "table", "module"),
                                {"conn": cg, "umrs_negative_severity": sev},
                                base_info={"onset_s": info["onset_s"], "wrist_extension": info["wrist_extension"]},
                                fingers=True)
        lp = lapse_part("wrist_flexion_left", math.radians(NEGATIVE_DEPTH_DEG[sev]), burst_s("negative"),
                        timing("negative")[1])
        out.append(_spec(clip_id=f"jerk_asterixis_E2_conn{cg}_sev{sev}", family="negative_myoclonus", segment="E2",
                         item="asterixis (Conn); UMRS negative myoclonus", operator="negative myoclonus, asterixis, "
                         "both hands", grade={"conn": cg, "umrs_negative_severity": sev},
                         grade_text=f"asterixis grade {cg} (Conn), UMRS negative myoclonus severity {sev}",
                         params_text=f"{UMRS_COUNT_30S[cg]} lapses per 30 s of posture, wrist drop "
                         f"{NEGATIVE_DEPTH_DEG[sev]:g} deg: falling over {1000 * lp['drop_s']:.0f} ms (silent period), "
                         f"caught in {1000 * lp['catch_s']:.0f} ms, back in {1000 * lp['fall_s']:.0f} ms; the "
                         "knuckles lapse with it", views=("left", "front", "hands"), level0=L_AST,
                         scene="chair", build=build, segment_title="arms outstretched, wrists extended"))

    # -- postural lapses of the legs while standing (bouncing)
    for fg, dg in ((3, 2), (4, 2), (3, 3)):
        def build(fg=fg, dg=dg):
            b = _static("stand")
            return build_lapses(b, [["knee_dip"]], LAPSE_DEPTH_CM[dg] / 100.0, "negative", UMRS_COUNT_30S[fg],
                                (0.5, b.duration - 0.5), SEED + 600 + 10 * fg + dg, "negative_myoclonus",
                                "UMRS negative myoclonus (legs, standing)",
                                ("frucht2002", "conn_asterixis", "vanderveen2022", "module"),
                                {"frequency": fg, "umrs_negative_severity": dg}, depth_unit="m", standing=True)
        lp = lapse_part("knee_dip", LAPSE_DEPTH_CM[dg] / 100.0, burst_s("negative"), timing("negative")[1])
        out.append(_spec(clip_id=f"jerk_negative_E7_leg_lapses_f{fg}_d{dg}", family="negative_myoclonus",
                         segment="E7", item="UMRS negative myoclonus (legs)", operator="negative myoclonus, postural "
                         "lapses of the legs (bouncing)", grade={"frequency": fg, "umrs_negative_severity": dg},
                         grade_text=f"UMRS negative myoclonus present, severity {dg}; frequency {fg}",
                         params_text=f"{UMRS_COUNT_30S[fg]} knee buckles per 30 s, pelvis down "
                         f"{LAPSE_DEPTH_CM[dg]:g} cm: falling over {1000 * lp['drop_s']:.0f} ms (silent period), "
                         f"caught in {1000 * lp['catch_s']:.0f} ms, back in {1000 * lp['fall_s']:.0f} ms",
                         views=("left", "front"), level0=L_E7, scene=None, build=build))

    # -- myoclonus-dystonia: jerks over a dystonic posture
    for region, cid, post, seg, views, rep in (
            ("neck", "jerk_mdyst_E7_neck", "stand", "E7", ("front", "top", "head"),
             [["neck_rotation_left"], ["shrug_left", "neck_rotation_left"], ["neck_rotation_right"], ["neck_flexion"],
              ["neck_extension"], ["elbow_flexion_right"]]),
            ("arm", "jerk_mdyst_E1_right_arm", "sit", "E1", ("front", "right"),
             [["shoulder_abduction_right"], ["elbow_flexion_right"], ["elbow_extension_right"], ["neck_flexion"]])):
        def base_fn(region=region, post=post):
            from .dystonia import dystonic_posture
            b = _static(post)
            if region == "neck":
                return dystonic_posture(b, "neck", "rotation_left", "left", amplitude_grade=2, seed=SEED)
            return dystonic_posture(b, "distal_arm", "elbow_flexion", "right", amplitude_grade=2, seed=SEED)

        def lvl(base_fn=base_fn):
            return base_fn()

        def build(base_fn=base_fn, rep=rep, region=region, post=post):
            b, t0 = base_fn()
            o, t = build_myoclonus(b, "myoclonus_dystonia", rep, 3, 2, (1.5, b.duration - 0.5), SEED + 700 + len(rep),
                                   "myoclonus", "UMRS rest myoclonus over a dystonic posture",
                                   ("frucht2002", "vanderveen2022", "twstrs" if region == "neck" else "dis", "table",
                                    "module"),
                                   extra={"posture": {k: t0["params"].get(k) for k in
                                                      ("region", "direction", "side", "scale", "level_deg",
                                                       "realized_peak_deg")}},
                                   room_body=_static(post), standing=post == "stand")
            return o, t
        dname = "TWSTRS rotation grade 2 (left)" if region == "neck" else "DIS distal arm grade 2 (right elbow flexion)"
        out.append(_spec(clip_id=f"{cid}_posture_only", family="myoclonus", segment=seg, item="level 0 (posture only)",
                         operator=f"dystonic posture, {dname}, no jerk", grade=0, grade_text="UMRS 0 over the posture",
                         params_text=dname, views=views, level0=L_E7 if post == "stand" else L_E1,
                         scene=None if post == "stand" else "chair", build=lvl))
        r, f = timing("myoclonus_dystonia")
        out.append(_spec(clip_id=f"{cid}_f3_a2", family="myoclonus", segment=seg, item="UMRS rest myoclonus",
                         operator=f"myoclonus-dystonia: jerks over {dname}", grade={"frequency": 3, "amplitude": 2},
                         grade_text="UMRS rest: frequency 3, amplitude 2, over the dystonic posture",
                         params_text=f"{UMRS_COUNT_30S[3]} jerks per 30 s, burst {1000 * burst_s('myoclonus_dystonia'):.0f}"
                         f" ms (cited mean), movement {1000 * (r + f):.0f} ms", views=views,
                         level0=f"{cid}_posture_only", scene=None if post == "stand" else "chair", build=build))

    # -- motor tics (YGTSS and MRVS), seated
    win = (0.5, CLIP_S - 0.5)
    tic_plan = [("number", 1, 2, 3, 0), ("number", 2, 2, 3, 0), ("number", 3, 2, 3, 0),
                ("frequency", 2, 1, 3, 0), ("frequency", 2, 3, 3, 0), ("frequency", 2, 4, 3, 0),
                ("intensity", 2, 2, 2, 0), ("intensity", 2, 2, 4, 0), ("intensity", 2, 2, 5, 0),
                ("complexity", 2, 2, 3, 2)]
    for ladder, nb, fg, it, cx in tic_plan:
        def build(nb=nb, fg=fg, it=it, cx=cx):
            b = _static("sit")
            return build_tics(b, YGTSS_NUMBER_TYPES[nb], fg, it, SEED + 800 + 100 * nb + 10 * fg + it + cx, win,
                              complexity=cx, feet=(1,))
        tag = {"number": f"n{nb}", "frequency": f"f{fg}", "intensity": f"i{it}", "complexity": f"c{cx}"}[ladder]
        n_all, n_cx = tic_counts(fg, CLIP_S, cx)
        shown = min(YGTSS_NUMBER_TYPES[nb], max(1, n_all - n_cx))      # fewer tics than types: the first types
        gn = ygtss_number_grade(shown, n_cx)
        out.append(_spec(clip_id=f"jerk_tics_E1_{ladder}_{tag}", family="tics", segment="E1",
                         item="YGTSS motor; MRVS", operator="motor tics, seated" + (", with orchestrated sequences"
                                                                                    if cx else ""),
                         grade={"ygtss_number": gn, "mrvs_frequency": fg, "ygtss_intensity": it,
                                "ygtss_complexity": cx},
                         grade_text=f"YGTSS number {gn}, intensity {it}, complexity {cx}; MRVS frequency {fg}",
                         params_text=f"{shown} tic type{'' if shown == 1 else 's'}, {n_all} tics in {CLIP_S:g} s "
                         f"({60.0 * n_all / CLIP_S:g} per minute), "
                         f"{100 * YGTSS_INTENSITY_SHARE[it]:g} percent of the joint's range in that direction"
                         + (f", {100 * YGTSS_COMPLEX_SHARE[cx]:g} percent orchestrated" if cx else ""),
                         views=("front", "left"), level0=L_E1, scene="chair", build=build))

    def build_bouts():
        b = _static("sit", 60.0)
        bouts = schedule(b.duration - 1.0, 0.5, 3, 2.0, as_rng(SEED + 990), b.rate, start_s=0.5)
        return build_tics(b, 3, 2, 3, SEED + 991, (0.5, 59.5), bouts=bouts, clip_s=60.0, count=TIC_BOUT_COUNT,
                          feet=(1,))
    out.append(_spec(clip_id="jerk_tics_E1_bouts_60s", family="tics", segment="E1", item="YGTSS motor; MRVS",
                     operator="motor tics in bouts, waxing and waning", grade={"ygtss_number": 2, "mrvs_frequency": 2,
                                                                                "ygtss_intensity": 3},
                     grade_text="YGTSS number 2, intensity 3; MRVS frequency 2 over the minute, three bouts",
                     params_text=f"{TIC_BOUT_COUNT} tics in 3 bouts over half the minute, amplitude waxing and waning "
                                 "over each bout",
                     views=("front", "left"), level0=L_E1_60, scene="chair", build=build_bouts))

    def build_stand_complex():
        b = _static("stand")
        return build_tics(b, 4, 2, 3, SEED + 995, win, complexity=4, squat=True,
                          types=("head_nod", "shoulder_shrug_left", "arm_jerk_right", "trunk_jerk"), standing=True)
    out.append(_spec(clip_id="jerk_tics_E7_complexity_c4", family="tics", segment="E7", item="YGTSS motor; MRVS",
                     operator="motor tics, standing, complex (orchestrated sequences and squat dips)",
                     grade={"ygtss_number": 5, "mrvs_frequency": 2, "ygtss_intensity": 3, "ygtss_complexity": 4},
                     grade_text="YGTSS number 5, intensity 3, complexity 4; MRVS frequency 2",
                     params_text=f"half of the tics complex: orchestrated head, shoulder and arm, or a squat of "
                     f"{SQUAT_CM:g} cm", views=("front", "left"), level0=L_E7, scene=None, build=build_stand_complex))

    # -- touching tics (shared change SH15): complex tics of the right index on the face, mixed with simple tics
    def touch_text(kinds):
        names = []
        for kd in kinds:
            tg, taps = TOUCH_KINDS[kd]
            names.append(f"a tap of {taps} on the {tg}" if taps else f"the {tg}")
        return ", ".join(names)
    for posture, lv, kinds in (("sit", 2, ("nose", "chin", "nose_tap")),
                               ("sit", 4, ("nose", "chin", "nose_tap", "cheek", "forehead")),
                               ("stand", 2, ("nose", "chin", "nose_tap"))):
        seg = "E7" if posture == "stand" else "E1"
        cx = min(lv, TOUCH_COMPLEXITY_MAX)            # the ladder's share of touching tics, the grade capped

        def build(posture=posture, lv=lv, kinds=kinds):
            stand = posture == "stand"
            return build_touch_tics(_static(posture), posture, 2, 3, lv, kinds, SEED + 1200 + 10 * lv + 5 * stand,
                                    win, feet=(0, 1) if stand else (1,), standing=stand)
        n_all, n_cx = tic_counts(2, CLIP_S, lv)
        shown = min(len(TOUCH_TIC_TYPES), n_all - n_cx)
        gn = ygtss_number_grade(shown + min(len(kinds), n_cx))
        where = "from the chair arm" if posture == "sit" else "from the hanging arm"
        out.append(_spec(clip_id=f"jerk_tics_{seg}_touch_c{cx}", family="tics", segment=seg, item="YGTSS motor; MRVS",
                         operator=f"motor tics, {'seated' if posture == 'sit' else 'standing'}, with complex tics "
                         f"touching the face (the right index, {where})",
                         grade={"ygtss_number": gn, "mrvs_frequency": 2, "ygtss_intensity": 3,
                                "ygtss_complexity": cx},
                         grade_text=f"YGTSS number {gn}, intensity 3, complexity {cx}; MRVS frequency 2",
                         params_text=f"{shown} simple tic types and {len(kinds)} touching tics, "
                         f"{MRVS_LEVEL_PER_MIN[2]:.1f} per minute, {100 * YGTSS_INTENSITY_SHARE[3]:g} percent of the "
                         f"joint's range in that direction, {100 * YGTSS_COMPLEX_SHARE[lv]:g} percent touching: {touch_text(kinds)}",
                         views=("front", "right", "head"), level0=L_E7 if posture == "stand" else L_E1,
                         scene=None if posture == "stand" else "chair", build=build))

    # -- startle and hyperekplexia (STARTLE_PAIRING): a hyperekplexia beside the normal startle to the same sounds,
    #    the normal startle beside the stimulus-free body of its posture
    for cond, posture in STARTLE_CLIPS:
        def build(cond=cond, posture=posture):
            return build_startle(_static(posture), cond, SEED + 1000, standing=posture == "stand")
        seg = STARTLE_SEGMENT[posture]
        out.append(_spec(clip_id=startle_id(cond, posture), family="startle", segment=seg, item="startle reflex",
                         operator=f"startle reflex, {cond}", grade=cond,
                         grade_text="normal startle (habituating)" if cond == "normal" else
                         "hyperekplexia: exaggerated startle, then stiffness, habituating poorly",
                         params_text="a sudden loud sound at " + ", ".join(f"{s:g}" for s in STIMULI_S) +
                         f" s; onset {STARTLE_SCM_MS:g} ms (neck), craniocaudal", views=("front", "left"),
                         level0=startle_level0(cond, posture, L_E7 if posture == "stand" else L_E1),
                         scene=None if posture == "stand" else "chair", build=build,
                         note=STARTLE_PAIRING if cond == "hyperekplexia" else ""))

    # -- ankle clonus (SCATS duration bands), seated, the left forefoot loaded (catalogue row O01)
    L_CL = "jerk_clonus_J2_scats0"
    for g in (0, 1, 2, 3):
        def build(g=g):
            return build_clonus(held_leg_base(20.0), g, SEED + 1100 + g)
        dur = {0: "no beat", 1: "beats for 2.0 s", 2: f"beats for {math.sqrt(30.0):.1f} s", 3: "sustained beats"}[g]
        out.append(_spec(clip_id=f"jerk_clonus_J2_scats{g}", family="clonus" if g else "reference", segment="J2",
                         item="SCATS clonus", operator="ankle clonus, left, the forefoot loaded (the heel bouncing)",
                         grade=g, grade_text=f"SCATS clonus {g}",
                         params_text=f"heel {100 * HEEL_UP_M:g} cm up, the knee pressed down at 2.0 s, {dur} at "
                         f"{CLONUS_HZ:g} Hz, {CLONUS_PP_DEG:g} deg peak to peak"
                         + (" (drifting)" if g else ""), views=("left", "foot_left"),
                         level0=None if g == 0 else L_CL, scene="chair", build=build,
                         segment_title="ankle clonus, seated, the left forefoot loaded"))
    out += hyper_gap_specs(L_E1, L_E7)
    ids = [s.clip_id for s in out]
    if len(set(ids)) != len(ids):
        raise RuntimeError("duplicate clip ids in the jerk family")
    return out


# ================================================================ catalogue gap wave, lane hyper_gaps (2026-11-02)
# Catalogue rows K06 (dystonic or tonic tics) and F03 (functional axial jerks): new builders and clip specifications
# appended to the family (hyper_gap_specs), in a block of their own; no existing clip or function changes.
#
# Tonic tics (K06). A tonic tic tightens like a clonic tic (the rise of the tic type, from its 200 ms burst) and then
# holds the posture (TONIC_HOLD_S, log-uniform) before the return: a held tic posture longer than a clonic tic (the
# catalogue's unverified secondary rule; the hold a module value). Its repertoire (TONIC_TYPES) holds a neck twist
# (rotation with tilt), an abdominal tensing and a shoulder pulled up; a tonic tic is an isometric tensing with little
# excursion, so each site takes TONIC_WEIGHTS of the intensity's share (the abdominal tensing is shown as a small trunk
# flexion, the shoulder mostly shrugged and a little abducted: at the full share the trunk bowed 48 deg and the arm
# rose toward the horizontal at intensity 4, read as voluntary bowing and arm raising); a mixed clip adds the module's
# first three clonic types (TIC_ORDER) at the full share, so that the held postures stand beside brief jerks. Events
# of different lengths are placed by place_varied (the onsets of a Poisson process given its count, every event's own
# length kept); the YGTSS number, intensity and complexity and the MRVS frequency and areas are those of build_tics.
#
# Functional axial jerks (F03). Mostly trunk flexion (flexion in 70 of 76 axial cases, erro2014; the axial pattern in
# 121 of 170, latorre2025), each jerk with its own shape: an active phase (the burst) of FUNCTIONAL['burst_s'], over
# the 1000 ms of latorre2025, a variable rise, a hold to the burst's end and a variable return; the trunk, the hips
# and the neck at weights drawn per jerk with a variable recruitment order (FUNCTIONAL 'lead_s'), sometimes the
# shoulders, sometimes a second flexion during the hold; at least FUNCTIONAL['pause_s'] of rest between two jerks
# (intermittent: two jerks never merge into one double bow). Distractibility (latorre2025): during a dual-task period
# (DISTRACTION: counting backward by sevens aloud, stated in params['frame_text'] for the viewer to burn in, the body
# having no voice) the jerks come at DISTRACTION['rate'] of their rate and DISTRACTION['amp'] of their size; the
# counts per window (functional_windows) are deterministic and stated in the caption. Grades: the S-FMDRS
# (nielsen2017) severity (mild, moderate, severe: a declared share of the room, FUNCTIONAL_SEVERITY_SHARE) and
# duration (once or a few times, frequent but intermittent: declared counts per 30 s outside the dual task,
# FUNCTIONAL_PER_30S; the share of time moving is read back); the continuous duration 3 does not fit jerks of over a
# second in a 30 s clip and is not built. Seated, the hands held just over the thighs stay there while the trunk bows
# (carry_arms: involuntary.support_arms at the knots where the trunk or the hips move; rigidly carried, the fingers
# went 2 cm into the thighs); the arms' contact with the chair and the body is read back (hand_contact).
# Standing: the hip strategy and the balance block (jerk_operator). Entrainment and the Bereitschaftspotential are
# not shown.
SOURCES.update({
    "latorre2025": "Latorre A, Ganos C, Hamada M, et al. Diagnostic utility of clinical neurophysiology in jerky "
                   "movement disorders. Mov Disord Clin Pract 2025; 12(3): 272-284, doi 10.1002/mdc3.14306 (table 4 as "
                   "the catalogue reads it, row F03): functional jerks, axial pattern in 121 of 170 across series, "
                   "bursts over 1000 ms, a Bereitschaftspotential in 25 to 86 percent, distractibility, entrainment",
    "erro2014": "Erro R et al. Psychogenic axial myoclonus: clinical features and long-term outcome. Parkinsonism "
                "Relat Disord 2014; 20(6): 596-599, doi 10.1016/j.parkreldis.2014.02.026 (abstract, as the catalogue "
                "reads it): flexion in 70 of 76 axial cases",
    "zutt2017": "Zutt R et al. Myoclonus subtypes in tertiary referral center. Clin Neurophysiol 2017; 128(1): "
                "253-259, doi 10.1016/j.clinph.2016.10.093 (abstract, as the catalogue reads it): functional jerks "
                "in 47 percent of 85 myoclonus referrals",
    "nielsen2017": "Nielsen G et al. The Simplified Functional Movement Disorders Rating Scale (S-FMDRS). Mov Disord "
                   "Clin Pract 2017; 4(5): 710-716, doi 10.1002/mdc3.12475 (as the catalogue reads it): per region "
                   "severity 0 absent, 1 mild, 2 moderate, 3 severe; duration once or a few times, frequent but "
                   "intermittent, continuous",
    "catalogue_k06": "docs/proposals/SYMPTOM_CATALOGUE_DRAFT.md, row K06 (dystonic or tonic tics): tonic tics longer "
                     "than clonic ones (an unverified secondary rule), the hold a module default",
})

TONIC_TYPES = {
    "tonic_neck_twist_right": ["neck_rotation_right", "neck_tilt_right"],
    "tonic_trunk_tense": ["trunk_flexion"],
    "tonic_shoulder_hold_right": ["shrug_right", "shoulder_abduction_right"],
}
# [module] the held tic postures: the head turned and tilted to the right, the abdomen tensed (seen as trunk
# flexion), the right shoulder pulled up and out
TONIC_WEIGHTS = {"neck_rotation_right": 1.0, "neck_tilt_right": 1.0, "trunk_flexion": 0.35, "shrug_right": 1.0,
                 "shoulder_abduction_right": 0.3}
# [module] a tonic tic's site at this share of the intensity's share of the room (module comment above): at intensity
# 4 the trunk 17 deg, the shoulder abducted 14 deg with the shrug at its full share
TONIC_ORDER = ("tonic_neck_twist_right", "tonic_trunk_tense", "tonic_shoulder_hold_right")
TONIC_HOLD_S = (0.5, 1.5)            # [module] the hold of a tonic tic (log-uniform), after the tic type's rise
TONIC_COUNT_30S = 6                  # [module] the tonic-only clips: 6 tics per 30 s (12 per minute, inside MRVS
#                                      band 1, above its level of 4.5 per minute so that each of the three types
#                                      shows twice)
FUNCTIONAL = {"burst_s": (1.0, 2.0), "rise_s": (0.15, 0.35), "return_s": (0.4, 1.2), "trunk_w": (0.55, 0.85),
              "hip_w": (0.0, 0.5), "neck_w": (0.2, 0.8), "lead_s": (-0.15, 0.15), "amp": (0.6, 1.0),
              "shrug_p": 0.3, "shrug_w": 0.3, "double_p": 0.3, "double_amp": 0.3, "pause_s": 1.0}
# [module] a functional axial jerk: the burst (the active phase: the rise and the hold) 1 to 2 s (log-uniform, over
# the 1000 ms of latorre2025), the rise 0.15 to 0.35 s (a jerk, not a slow bow: at 0.45 to 0.55 s the trunk turned at
# 76 deg/s), the return 0.4 to 1.2 s; the room shares split over the trunk (0.55 to 0.85 of the grade's share), the
# hips (0 to 0.5) and the neck (0.2 to 0.8), the neck and the hips starting up to 0.15 s before or after the trunk;
# the whole jerk at 0.6 to 1 of its size; in 30 percent of the jerks the shoulders shrug (0.3 of the share), in 30
# percent a second trunk flexion of 0.3 of the peak rides on the hold; at least 1 s of rest between two jerks and at
# the edges of the dual task
FUNCTIONAL_SEVERITY_SHARE = {1: 0.12, 2: 0.25, 3: 0.45}    # [module] S-FMDRS severity, declared share of the room
FUNCTIONAL_PER_30S = {1: 3, 2: 7}                          # [module] S-FMDRS duration 1 and 2, jerks per 30 s outside
#                                                            the dual task (2 and 4 in its 18 s): at 9 the jerks of 1.5
#                                                            to 3 s took two thirds of the time, close to continuous
DISTRACTION = {"window_s": (11.0, 21.0), "rate": 0.5, "amp": 0.6,
               "text": "DUAL TASK: COUNTING BACKWARD BY SEVENS ALOUD"}
# [module] the dual-task period (latorre2025: distractibility, no size given): the jerks at half their rate and 0.6 of
# their size while the patient counts backward from 100 by sevens aloud (stated, the body has no voice); with these
# counts the duration 2 clips show 1 jerk in the dual task against 4 outside (half the rate per second), the duration 1
# clip none


def place_varied(durations, t0, t1, rng, min_gap_s=MIN_GAP_S, rate=RATE):
    """[onset s] of events of the given durations (s, in order) inside [t0, t1], at least min_gap_s apart, on knots:
    the free time split at random (multinomial over a Dirichlet, as place_events) so that each event keeps its own
    length. Raises when they cannot fit."""
    n = len(durations)
    if n == 0:
        return []
    K0, K1 = int(round(t0 * rate)), int(round(t1 * rate))
    d = [int(math.ceil(x * rate - 1e-9)) for x in durations]
    g = int(round(min_gap_s * rate))
    free = (K1 - K0) - sum(d) - (n - 1) * g
    if free < 0:
        raise ValueError(f"{n} events of {sum(durations):.2f} s with gaps of {min_gap_s} s do not fit in "
                         f"{t1 - t0:.2f} s")
    gaps = rng.multinomial(free, rng.dirichlet(np.ones(n + 1)))
    out, k = [], K0
    for i in range(n):
        k += int(gaps[i])
        out.append(k / rate)
        k += d[i] + g
    return out


def tonic_tic_events(base, tonic, clonic, count, intensity, rng, window):
    """(events, peaks {type: {site: rad}}): count tics in the window, every type of tonic + clonic at least once, the
    others drawn with weights 1/rank (as tic_events), at the intensity's share of the room, a tonic site at
    TONIC_WEIGHTS of it; a tonic tic (TONIC_TYPES) rises with the tic type's timing, holds TONIC_HOLD_S and returns
    (jerk_part with hold_s); a clonic tic (TIC_TYPES) as tic_events."""
    share = YGTSS_INTENSITY_SHARE[intensity]
    k = int(round(window[0] * base.rate))
    types = list(tonic) + list(clonic)
    sites_of = {**{ty: TONIC_TYPES[ty] for ty in tonic}, **{ty: TIC_TYPES[ty] for ty in clonic}}
    peaks = {ty: {s: peak_of(base, s, share * (TONIC_WEIGHTS[s] if ty in TONIC_TYPES else 1.0), k)
                  for s in sites_of[ty]} for ty in types}
    w = 1.0 / np.arange(1, len(types) + 1)
    seq = list(range(len(types))) + [int(x) for x in rng.choice(len(types), max(count - len(types), 0), p=w / w.sum())]
    seq = [seq[i] for i in rng.permutation(len(seq))][:count]
    events = []
    for i in seq:
        ty = types[i]
        hold = math.exp(rng.uniform(math.log(TONIC_HOLD_S[0]), math.log(TONIC_HOLD_S[1]))) if ty in TONIC_TYPES else 0.0
        parts = [jerk_part(s, peaks[ty][s], "tic", hold_s=hold) for s in sites_of[ty]]
        events.append({"parts": parts, "label": ty, "factor": 1.0, "hold_s": parts[0]["hold_s"]})
    onsets = place_varied([_event_dur(e["parts"]) for e in events], window[0], window[1], rng)
    for e, t in zip(events, onsets):
        e["onset_s"] = float(t)
    events = vary_sizes(base, events, rng, rise_jitter=0.0, knot=k)    # a tonic tic keeps its type's timing
    return events, peaks


def build_tonic_tics(base, n_tonic, freq_grade, intensity, seed, window=(0.5, CLIP_S - 0.5), clonic=(), feet=(0, 1),
                     count=None):
    """(Body, truth) of a tonic tic clip (catalogue row K06): the first n_tonic types of TONIC_ORDER and the clonic
    types (tonic_tic_events), the count of the MRVS frequency grade's level over the clip (or count), at the YGTSS
    intensity's share of the room; jerk_operator; the grades and the read-back keys of build_tics (YGTSS number of
    the types shown, MRVS frequency and areas), with each tic's hold."""
    rng = as_rng(seed)
    tonic = list(TONIC_ORDER[:n_tonic])
    count, _ = tic_counts(freq_grade, base.duration, 0, count)
    events, peaks = tonic_tic_events(base, tonic, clonic, count, intensity, rng, window)
    types = tonic + list(clonic)
    shown = [ty for ty in types if any(e["label"] == ty for e in events)]
    used = _uniq([p["site"] for e in events for p in e["parts"]])
    areas = sorted({site_area(s) for s in used})
    holds = [e["hold_s"] for e in events if e["label"] in TONIC_TYPES]
    p = _params("tic", "poisson", window_s=[float(window[0]), float(window[1])], tic_types=types, n_types=len(types),
                tic_types_shown=shown, tonic_types=tonic, clonic_types=list(clonic), count=len(events),
                per_min=len(events) * 60.0 / base.duration,
                mrvs_frequency_grade=mrvs_frequency_grade(len(events) * 60.0 / base.duration), areas=areas,
                mrvs_area_grade=mrvs_area_grade(len(areas)), distinct_labels=sorted({e["label"] for e in events}),
                n_complex=0, complexity_share=0.0, intensity=intensity,
                intensity_share=YGTSS_INTENSITY_SHARE[intensity],
                bouts_s=None, peaks_deg={ty: {s: math.degrees(v) for s, v in d.items()} for ty, d in peaks.items()},
                orchestrated=None, tonic_hold_s_range=list(TONIC_HOLD_S), holds_s=holds, n_tonic=len(holds),
                tonic_weights=TONIC_WEIGHTS,
                hold_rule="a tonic tic rises with the tic type's timing and holds TONIC_HOLD_S (module; "
                          "the catalogue's secondary rule: tonic tics last longer than clonic ones)",
                not_shown="the eye and face tonic tics (no face); the abdominal tensing is shown as a small trunk "
                          "flexion", size_variation=size_record(events, SIZE_SIGMA, 0.0))
    grade = {"ygtss_number": ygtss_number_grade(len(shown)), "mrvs_frequency": freq_grade, "ygtss_intensity": intensity,
             "ygtss_complexity": 0}
    for e in events:
        e.pop("hold_s", None)
    out, truth = jerk_operator(base, events, "tics", grade, "YGTSS motor and MRVS", p,
                               ("catalogue_k06", "leckman1989", "riechmann2023", "vanderveen2022", "table", "module"),
                               seed if isinstance(seed, (int, np.integer)) else None, feet=feet)
    truth["validity"]["hand_contact"] = hand_contact(out, "chair")
    return out, truth


HAND_CONTACT_HZ = 40.0      # [module] hand_contact: knots read per second (twice the involuntary guard's rate)


def hand_contact(out, scene):
    """{left, right: cm, tol_cm}: the deepest vertex of each arm inside the scene's boxes ('chair' or None) or of
    each forearm and hand behind the trunk and thigh surface, read on the final body at HAND_CONTACT_HZ (the measure
    of the involuntary module's contact guard, involuntary._penetration), against its tolerance GUARD_TOL_M."""
    from . import involuntary as IV
    from . import viewer as V
    ks = np.unique(np.r_[np.arange(0, out.T, max(1, int(round(out.rate / HAND_CONTACT_HZ)))), out.T - 1])
    Vk = V.knot_vertices(out, ks)
    ctx = IV._target_ctx(Vk)
    boxes = IV._boxes(scene, out)
    res = {}
    for s in ("left", "right"):
        d = 100.0 * IV._penetration(Vk, s, boxes, ctx)
        res[s] = float(d.max())
        res[f"{s}_t_s"] = float(ks[int(d.argmax())] / out.rate)
    res["tol_cm"] = 100.0 * IV.GUARD_TOL_M
    res["ok"] = bool(max(res["left"], res["right"]) <= res["tol_cm"] + 1e-9)
    return res


CARRY_SITES = ("trunk_flexion", "hip_flexion")     # the sites that bring the seated shoulders over the hands
CARRY_SHRUG = {"left": "shrug_left", "right": "shrug_right"}
#   seated, a shrug carries its own side's arm too (Phase D of docs/PLAN_2026-11-12.md): the raised girdle lifted the
#   hanging arm as a block, the forearm off its armrest; carried, the wrist keeps its place and the elbow takes the rise
CARRY_SHRUG_MAX_DEG_S = 200.0
#   [module] a shrug whose girdle rises faster than this lifts the forearm off its support and is not carried (a brisk
#   tic flings it; the grade-5 tic intensity's 339 deg/s shrug carried kept its wrist on the armrest, but the
#   background layer then failed its acceleration and anchor checks on that arm, 4.76 m/s2 over its bound; the other
#   seated shrugs rise at 20 to 158 deg/s)
ARM_SITE_JOINTS = {"left": (16, 18, 20), "right": (17, 19, 21)}


def carry_arms(base, out, events):
    """(Body, info): seated, the hands held over the thighs stay there while the trunk or the hips flex (module
    comment above), and each arm while its own side's shrug raises the girdle (CARRY_SHRUG; not a shrug faster than
    CARRY_SHRUG_MAX_DEG_S): at the knots where a
    CARRY_SITES profile or that shrug moves, the shoulder, elbow and wrist rows of the arm from
    involuntary.support_arms (the wrist kept where the base puts it over its thigh, the hand's orientation kept,
    inside the table the clip is judged on, limits_table: with a shrug the shoulder is turned back past the healthy
    table), except at the knots where a site of that arm's own joints moves; every other knot keeps out's rows bit for
    bit."""
    from . import involuntary as IV
    prof = site_profiles(base.T, events, base.rate)
    act_trunk = np.zeros(base.T, bool)
    for s in CARRY_SITES:
        if s in prof:
            act_trunk |= np.abs(prof[s]) > 0
    def carried_shrug(side):                          # a brisk shrug (CARRY_SHRUG_MAX_DEG_S) is not carried
        p = prof.get(CARRY_SHRUG[side])
        if p is None or np.degrees(np.abs(np.gradient(p, 1.0 / base.rate))).max() > CARRY_SHRUG_MAX_DEG_S:
            return False
        return np.abs(p) > 0
    act_side = {side: act_trunk | carried_shrug(side) for side in ARM_SITE_JOINTS}
    act = act_side["left"] | act_side["right"]
    if not act.any():
        return out, {"carried_share": 0.0}
    from dataclasses import replace
    table = limits_table(prof)
    bounds = None if table == LX.HEALTHY else LX.knot_bounds(table, components_deg(out.rot6d),
                                                             components_deg(base.rot6d))
    sup, info = IV.support_arms(out, base, bounds=bounds)
    r6 = out.rot6d.clone()
    for side, js in ARM_SITE_JOINTS.items():
        own = np.zeros(base.T, bool)
        for s, p in prof.items():
            if set(SITES[s][0]) & set(js):
                own |= np.abs(p) > 0
        m = torch.from_numpy(act_side[side] & ~own)
        for j in js:
            r6[m, j] = sup.rot6d[m, j]
    return replace(out.copy(), rot6d=r6), {"carried_share": float(act.mean()),
                                         "shrug_carried_share": {sd: float((act_side[sd] & ~act_trunk).mean())
                                                                 for sd in act_side},
                                         "elbow_change_deg_max": {s: v["elbow_change_deg_max"]
                                                                  for s, v in info["sides"].items()},
                                         "rule": "support_arms at the knots where the trunk or the hips flex, or "
                                                 "where the arm's own side shrugs"}


HANG_SITES = ("trunk_flexion", "hip_flexion", "trunk_tilt_right")   # the sites a standing hanging arm does not ride


def hang_arms_standing(base, out, events):
    """(Body, info): standing, the arms hang (wave SH1d part B, review M6: the arms rode on a trunk jerk as one block,
    the wrists ending behind the buttocks): at the knots where a HANG_SITES profile moves, each shoulder row turned so
    that the upper arm keeps its global rotation in the base (the forearm and the hand follow it on their own rows),
    except at the knots where a site of that arm's own joints moves; the collars ride with the trunk; every other
    knot keeps the rows of out bit for bit. A quick jerk leaves the hanging arm in space (its inertia) and a slow bow
    lets it hang plumb (gravity): both keep its global direction, so no pendulum lag is modelled (module choice)."""
    prof = site_profiles(base.T, events, base.rate)
    act = np.zeros(base.T, bool)
    for s in HANG_SITES:
        if s in prof:
            act |= np.abs(prof[s]) > 0
    if not act.any():
        return out, {"mode": "hanging (standing)", "carried_share": 0.0}
    _, Gb = base.fk()
    _, Go = out.fk()
    R = out.local_R()
    r6 = out.rot6d.clone()
    turned = {}
    for side, js in ARM_SITE_JOINTS.items():
        own = np.zeros(base.T, bool)
        for s, p in prof.items():
            if set(SITES[s][0]) & set(js):
                own |= np.abs(p) > 0
        m = act & ~own
        sh = js[0]
        Rn = Go[:, PARENTS[sh]].transpose(-1, -2) @ Gb[:, sh]
        mt = torch.from_numpy(m)
        r6[mt, sh] = matrix_to_rot6d(Rn[mt])
        d = rotation_angle_deg(R[:, sh].transpose(-1, -2) @ Rn).numpy()
        turned[side] = float(d[m].max()) if m.any() else 0.0
    from dataclasses import replace
    return replace(out.copy(), rot6d=r6), {"mode": "hanging (standing)", "carried_share": float(act.mean()),
                                         "shoulder_turn_deg_max": turned,
                                         "rule": "the upper arms keep the global rotation of the base where the "
                                                 "trunk or the hips move (HANG_SITES)"}


def arms_rule(base, out, events, standing):
    """(Body, info): the arms where the trunk or the hips move and the arm has no movement of its own (wave SH1d part
    B, review M6): seated, carried over the thighs (carry_arms, before only for the functional jerks), standing,
    hanging (hang_arms_standing)."""
    if standing:
        return hang_arms_standing(base, out, events)
    o, info = carry_arms(base, out, events)
    return o, dict(info, mode="carried (seated)")


def functional_windows(n_out, window=(0.5, CLIP_S - 0.5), distraction=DISTRACTION, pause_s=FUNCTIONAL["pause_s"]):
    """[(start_s, end_s, rate factor, size factor, count)]: the three windows of a functional clip (before, inside and
    after the dual task, the inner edges pause_s / 2 inside, so that a jerk never runs across an edge and the pause
    holds there too) and their counts: n_out jerks per 30 s outside the dual task, DISTRACTION['rate'] of that inside
    it, rounded by the largest remainder (the total is the rounded sum). Deterministic."""
    a, b = distraction["window_s"]
    h = 0.5 * pause_s
    wins = [(window[0], a - h, 1.0, 1.0), (a + h, b - h, distraction["rate"], distraction["amp"]),
            (b + h, window[1], 1.0, 1.0)]
    want = np.array([n_out * (w1 - w0) / CLIP_S * rf for w0, w1, rf, _ in wins])
    counts = np.floor(want).astype(int)
    for i in np.argsort(-(want - counts), kind="stable")[:int(round(want.sum())) - int(counts.sum())]:
        counts[i] += 1
    return [(float(w0), float(w1), rf, af, int(n)) for (w0, w1, rf, af), n in zip(wins, counts)]


def _occupancy(events, t0, t1, rate=RATE):
    """Share of [t0, t1] covered by the events' spans (event_span)."""
    k0, k1 = int(round(t0 * rate)), int(round(t1 * rate))
    m = np.zeros(max(k1 - k0, 1), bool)
    for e in events:
        a, b = event_span(e)
        m[max(int(round(a * rate)) - k0, 0):max(int(round(b * rate)) - k0, 0)] = True
    return float(m.mean())


def functional_events(base, severity, n_out, rng, window=(0.5, CLIP_S - 0.5), distraction=DISTRACTION):
    """(events, record): functional axial jerks (module comment above): the counts of functional_windows, each jerk
    with its own shape (FUNCTIONAL), at FUNCTIONAL_SEVERITY_SHARE[severity] of the room split over its sites,
    DISTRACTION['amp'] of its size inside the dual task, placed in each window by place_varied at least
    FUNCTIONAL['pause_s'] apart; the record: per window its count and the share of its time moving."""
    F = FUNCTIONAL
    share = FUNCTIONAL_SEVERITY_SHARE[severity]
    k = int(round(window[0] * base.rate))
    u = lambda lo_hi: float(rng.uniform(*lo_hi))  # noqa: E731
    lu = lambda lo_hi: math.exp(rng.uniform(math.log(lo_hi[0]), math.log(lo_hi[1])))  # noqa: E731
    events, rec = [], []
    for w0, w1, rf, af, n in functional_windows(n_out, window, distraction):
        evs = []
        for _ in range(n):
            burst, rise, ret = lu(F["burst_s"]), u(F["rise_s"]), u(F["return_s"])
            amp = af * u(F["amp"])
            hold = max(burst + TWITCH_S - rise, 0.0)
            wts = {"trunk_flexion": u(F["trunk_w"]), "hip_flexion": u(F["hip_w"]), "neck_flexion": u(F["neck_w"])}
            if rng.uniform() < F["shrug_p"]:
                wts.update(shrug_left=F["shrug_w"], shrug_right=F["shrug_w"])
            lead = {s: (0.0 if s == "trunk_flexion" else u(F["lead_s"])) for s in wts}
            d0 = -min(lead.values())
            parts = []
            for s, wt in wts.items():
                if wt <= 0:
                    continue
                pk = peak_of(base, s, share * wt * amp, k)
                parts.append(jerk_part(s, pk, "propriospinal", delay_s=d0 + lead[s], hold_s=hold, rise_s=rise,
                                       fall_s=ret))
            dbl = rng.uniform() < F["double_p"]
            if dbl:
                main = next(q for q in parts if q["site"] == "trunk_flexion")
                parts.append(jerk_part("trunk_flexion", F["double_amp"] * main["peak"], "propriospinal",
                                       delay_s=main["delay_s"] + main["rise_s"] + 0.3 * main["hold_s"], rise_s=0.2,
                                       fall_s=0.3))
            for q in parts:                      # jerk_part takes the given timing; the part is a functional one
                q["type"] = "functional"
            lab = "+".join(sorted({q["site"] for q in parts})) + (" double" if dbl else "")
            evs.append({"parts": parts, "label": lab, "factor": amp, "burst_s": burst, "dual_task": rf < 1.0})
        onsets = place_varied([_event_dur(e["parts"]) for e in evs], w0, w1, rng, min_gap_s=F["pause_s"])
        for e, t in zip(evs, onsets):
            e["onset_s"] = float(t)
        events += evs
        rec.append({"window_s": [w0, w1], "count": len(evs), "rate_factor": rf, "amp_factor": af,
                    "moving_share": _occupancy(evs, w0, w1)})
    return events, rec


def build_functional(base, severity, duration, seed, standing=False, feet=(0, 1)):
    """(Body, truth) of functional axial jerks (catalogue row F03): functional_events at the S-FMDRS severity's share
    and the duration grade's count (FUNCTIONAL_PER_30S) outside the dual task, the dual-task period in
    params['frame_text']; jerk_operator (standing: the hip strategy and the balance block)."""
    rng = as_rng(seed)
    events, wins = functional_events(base, severity, FUNCTIONAL_PER_30S[duration], rng)
    bursts = [e.pop("burst_s") for e in events]
    dual = [e.pop("dual_task") for e in events]
    a, b = DISTRACTION["window_s"]
    outside = [w for w in wins if w["rate_factor"] == 1.0]
    out_s = sum(w["window_s"][1] - w["window_s"][0] for w in outside)
    p = {"jerk_type": "functional", "severity": severity, "duration": duration,
         "severity_share": FUNCTIONAL_SEVERITY_SHARE[severity], "per_30s_outside": FUNCTIONAL_PER_30S[duration],
         "shape": FUNCTIONAL, "bursts_s": bursts, "dual_task_events": dual, "windows": wins,
         "count_outside": int(sum(w["count"] for w in outside)), "count_dual_task": int(sum(dual)),
         "moving_share_outside": float(sum(w["moving_share"] * (w["window_s"][1] - w["window_s"][0])
                                           for w in outside) / out_s),
         "moving_share_dual_task": float(next(w["moving_share"] for w in wins if w["rate_factor"] < 1.0)),
         "duration_rule": "S-FMDRS duration 1 once or a few times, 2 frequent but intermittent (FUNCTIONAL_PER_30S, "
                          "at least FUNCTIONAL['pause_s'] between two jerks); the moving shares read back",
         "distraction": DISTRACTION,
         "frame_text": [{"from_s": float(a), "to_s": float(b), "text": DISTRACTION["text"]}],
         "labels": [e["label"] for e in events], "standing": bool(standing),
         "not_shown": "entrainment, the Bereitschaftspotential (EEG), the voice of the dual task"}
    out, truth = jerk_operator(base, events, "functional_jerks", {"sfmdrs_severity": severity,
                                                                  "sfmdrs_duration": duration},
                               "S-FMDRS trunk (declared map)", p,
                               ("latorre2025", "erro2014", "zutt2017", "nielsen2017", "table", "module"),
                               seed if isinstance(seed, (int, np.integer)) else None, feet=feet, standing=standing)
    truth["validity"]["hand_contact"] = hand_contact(out, None if standing else "chair")
    return out, truth


def hyper_gap_specs(L_E1, L_E7):
    """[ClipSpec] of the lane hyper_gaps in this family (catalogue rows K06 and F03), since 'catalogue_v1'; L_E1, L_E7
    the module's level-0 clips."""
    out = []
    r, f = timing("tic")
    for nt, fg, it, clonic, tag in ((3, 1, 3, (), "f1_i3"), (3, 1, 4, (), "f1_i4"),
                                    (3, 2, 3, tuple(TIC_ORDER[:3]), "f2_i3_mixed")):
        cnt = None if clonic else TONIC_COUNT_30S

        def build(nt=nt, fg=fg, it=it, clonic=clonic, cnt=cnt):
            return build_tonic_tics(_static("sit"), nt, fg, it, SEED + 1200 + 10 * fg + it + len(clonic), clonic=clonic,
                                    count=cnt)
        per_min = MRVS_LEVEL_PER_MIN[fg] if cnt is None else cnt * 60.0 / CLIP_S
        n_types = nt + len(clonic)
        gn = ygtss_number_grade(n_types)
        out.append(_spec(clip_id=f"jerk_tonic_tics_E1_{tag}", family="tics", segment="E1", item="YGTSS motor; MRVS",
                         operator="tonic tics, seated" + (", with clonic tics" if clonic else ""),
                         grade={"ygtss_number": gn, "mrvs_frequency": fg, "ygtss_intensity": it, "ygtss_complexity": 0},
                         grade_text=f"YGTSS number {gn}, intensity {it}, complexity 0; MRVS frequency {fg}",
                         params_text=f"{n_types} tic types ({nt} held"
                         + (f", {len(clonic)} clonic" if clonic else "") + f"), {per_min:.1f} per minute, "
                         f"{100 * YGTSS_INTENSITY_SHARE[it]:g} percent of the joint's range in that direction "
                         f"(the held trunk at "
                         f"{100 * TONIC_WEIGHTS['trunk_flexion']:g} and the held shoulder's abduction at "
                         f"{100 * TONIC_WEIGHTS['shoulder_abduction_right']:g} percent of that); tightening in "
                         f"{1000 * r:.0f} ms, held {TONIC_HOLD_S[0]:g} to {TONIC_HOLD_S[1]:g} s, back in "
                         f"{1000 * f:.0f} ms",
                         views=("front", "left"), level0=L_E1, scene="chair", build=build,
                         note="the head twist, the abdominal tensing (a small trunk flexion) and the shoulder held up; "
                              "eye and face tics not shown (no face)"))
    for posture, sev, dur in (("sit", 2, 2), ("sit", 3, 2), ("sit", 2, 1), ("stand", 2, 2)):
        def build(posture=posture, sev=sev, dur=dur):
            return build_functional(_static(posture), sev, dur, SEED + 1300 + 10 * sev + dur + (5 if posture == "stand"
                                                                                                  else 0),
                                    standing=posture == "stand")
        seg = "E7" if posture == "stand" else "E1"
        a, b = DISTRACTION["window_s"]
        wins = functional_windows(FUNCTIONAL_PER_30S[dur])
        n_in = wins[1][4]
        n_out_ = wins[0][4] + wins[2][4]
        out_s = (wins[0][1] - wins[0][0]) + (wins[2][1] - wins[2][0])
        dual = (f"{n_in} jerk{'s' if n_in != 1 else ''} at {DISTRACTION['amp']:g} of the size" if n_in else "no jerk")
        out.append(_spec(clip_id=f"jerk_functional_{seg}_axial_sev{sev}_dur{dur}", family="functional", segment=seg,
                         item="S-FMDRS trunk", operator="functional jerks, axial flexion "
                         f"({'standing' if posture == 'stand' else 'seated'})",
                         grade={"sfmdrs_severity": sev, "sfmdrs_duration": dur},
                         grade_text=f"S-FMDRS severity {sev}, duration {dur} (declared map)",
                         params_text=f"{n_out_} jerks in the {out_s:g} s outside the dual task, at least "
                         f"{FUNCTIONAL['pause_s']:g} s apart, trunk flexion with the hips and neck in a varying order, "
                         f"{100 * FUNCTIONAL_SEVERITY_SHARE[sev]:g} percent of the joint's range in that direction, "
                         f"bursts of 1 to 2 s; dual task "
                         f"{a:g} to {b:g} s: {dual}",
                         views=("left", "front"), level0=L_E7 if posture == "stand" else L_E1,
                         scene=None if posture == "stand" else "chair", build=build,
                         note="the dual task (counting backward by sevens aloud) is stated in the truth's frame_text; "
                              "entrainment and the Bereitschaftspotential not shown"))
    return out
