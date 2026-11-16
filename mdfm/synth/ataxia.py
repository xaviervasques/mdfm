"""Cerebellar ataxia items of the synthetic-patient symptom catalogue (task CAT-PDATAX, since 'catalogue_v1'): the
SARA items other than gait, finger chase and nose-finger (other modules own them), each a ladder of ClipSpecs with
its level-0 partner on the same time base (specs()), a re-measurement per operator (measure_*) and the sources
(SOURCES, merged with grades.SOURCES). The SARA anchors are paraphrased (Schmitz-Huebsch et al. 2006), never
copied. Revised after the clinical review of 2026-11-02 (task FIX-ataxia): straighter knees, side-to-side sway at
the hips, a 3 Hz component in antiphase, catching steps that clear the other foot and follow the balance loss, a
bedside Romberg sign, seated sway from the pelvis with the back free, a living titubation background, SARA 7
interruptions and overshoots. SARA 8 (the heel-shin slide, supine on a couch) added by task SH7.

1. SARA 2, stance (segment SARA2): procedural standings in natural stance, feet together (the medial edges of the
   two sole clouds 4 mm apart) and tandem (the feet on one line, the front heel 5 mm ahead of the back toe), legs
   solved by two-bone inverse kinematics to fixed ankles with the feet flat (ik.hinge_leg, as base.walk), the
   farther leg's hip at REACH_ST of its length from its ankle, the tilt included (knees about 5 to 7 deg on the
   SMPL+H rest legs, bent at rest; the farther knee 8.9 deg on the open body's straight rest legs). Sway: a
   centre-of-mass target per axis, 1/f-shaped noise in SWAY_BAND_HZ (mean frequency near the cited 0.37 Hz) at the
   cited RMS of healthy older adults (Quijoux et al. 2021: centre-of-pressure RMS radius 0.59 cm, split equally
   between the axes, read as the centre of mass's: a module reading), times a gain per condition and grade (module
   values). Forward-back it is carried by a tilt of the whole body about the ankles (an inverted pendulum), side to
   side by a lateral shift of the pelvis with the trunk upright and the legs inclined (the hips' load and unload,
   Winter 1995); the angles are corrected twice by the centre-of-mass residual (STANCE_SOLVE), so that the read-back
   equals the target. The cerebellar clips add a narrow-band forward-back component at 3 Hz (the postural tremor of
   anterior-lobe atrophy, Mauritz et al. 1979, Diener et al. 1984), carried by the legs tilting about the ankles
   while the pelvis and trunk turn back about the hips (TREMOR3_TRUNK: the hips move, the head much less; the
   segment pattern is a module reading; the sources give the frequency, the direction and, Baloh et al. 1998, a
   tremor seen at the bedside), the hanging arms keeping their orientation in space (TREMOR3_ARMS_STILL, an
   approximation of a hanging limb driven above its pendulum frequency: a module reading), at TREMOR3_RMS_M of the
   centre of mass (halved after the review of 2026-11-03). A draw whose
   held part lets the extrapolated centre of mass (Hof et al. 2005, of the sway below XCOM_LOWPASS_HZ) leave the
   base of support is drawn again (STANCE_DRAWS). A condition that fails before 10 s grows its sway over GROW_S (by
   GROW_GAIN, side to side the most), leans toward the stepping side by LEAN_DEG and lifts that foot STEP_REACT_S
   after the extrapolated centre of mass leaves the base; both shoulders abduct by ARM_ABD_DEG over the last second;
   the swing moves the foot sideways first, then forward, lifted by STEP_LIFT_M with the knee turned outward, to a
   natural stance, after which the sway settles and the arms come down. Clips (each one condition, 12 s): natural
   stance grades 0 and 3, feet together grades 0, 2 and 3 (grade 3: the feet-together stance fails), tandem grades
   0 and 1 (grade 1: the tandem stance fails). Grades 4 to 6 need support (an arm, an examiner): not shown.
2. Romberg (segment ROMB): feet together for 20 s, eyes open for 10 s then closed; the eye state per time span is
   in the truth (params['frame_text']: the body has no face; the viewer burns it into the frames, viewer.frame_texts,
   SH12, and the caption states it). The posturographic pair: each half scaled to its own sway level so that the
   eyes-closed over eyes-open ratio is the published one: healthy and cerebellar about 1.5 (Baloh et al. 1998: the
   Romberg ratio did not separate cerebellar patients from controls, control mean about 1.5), sensory about 2
   (Anagnostou et al. 2024);
   the cerebellar sway is large with the eyes open and carries the 3 Hz component, the sensory sway is mild with
   the eyes open (gains: module values). Neither shows a bedside Romberg sign (no step, no loss of balance: a ratio a
   platform measures), and their texts say so (review of 2026-11-03). The bedside Romberg sign of a sensory ataxia
   (atax_ROMB_sensory_bedside): a normal sway with the eyes open, growing to ROMBERG['sensory_bedside']['ratio']
   times it over ROMBERG_GROW_S after the closure, then failing as a SARA 2 stance does (a catching step near
   ROMBERG_FAIL_S); its ratio is a module value.
3. SARA 3, sitting (segment SARA3): seated with the arms outstretched to the front ('sit_arms_forward'), the noise
   generator on both axes, SIT_PELVIS_SHARE of each angle on the pelvis rocking on the seat (seat_pivot), the rest
   spread over the lumbar and thoracic joints (3, 6, 9), the legs replanted to the feet (replant_legs), the
   amplitude solved so that the pelvis-to-neck line sways SIT_LADDER's RMS: grade 0 a small sway, grade 1 two bouts
   of SIT_BOUT_S, grade 2 constant sway; grades 3 and 4 need support: not shown. The body sits forward on the seat,
   clear of the backrest at every knot (seat_forward: SARA tests sitting without support). The SARA protocol seats
   the patient on an examination bed with the feet free; the feet rest on the floor of the chair scene (stated in the
   caption; the 'exam_bed' couch of item 6 exists, a seated posture on its edge does not yet).
4. Head and trunk titubation (segment TITUB): seated rest ('sit') on a shared physiological background (its level 0:
   seated sway at SARA 3 grade 0's level and quiet breathing at BREATH_HZ, the chest rising BREATH_CHEST_MM), an
   anteroposterior oscillation of the trunk, neck and head (TITUB_CHAIN: half of it on the lumbar and thoracic
   joints, half on the neck and head, which lag the trunk by TITUB_LAG_DEG; the shoulders move less than half as
   much as the head centre, a few millimetres at the low grades (the truth's remeasured 'shoulders_pp_cm'), so the
   sign reads at the head first, hence the name) at TITUBATION_HZ (3 Hz: Dankova et al. 2020; the head titubation of
   the Human Phenotype Ontology, 3 to 4 Hz) with a slow amplitude modulation (TITUB_AM) and frequency jitter
   (TITUB_FM_HZ). Its amplitude is read where a rater sees it, at the visible head (review M13 of 2026-11-03; the
   skull-base joint moves less than the head centre, the truth's remeasured 'points'): the nose's largest cycle,
   peak to peak against the level 0, is set at TITUB_MAX_SHARE of the Fahn-Tolosa-Marin band's upper edge
   (titubation_max_cm; the open band of grade 4 closed at twice its lower edge, as the bands below span a factor of
   2; the bands as commonly reproduced, the original chapter not read), and the head centre's median cycle then lies
   above the band's lower edge: the visible head's readings, from the head centre's typical cycle to the nose's
   largest, inside the band (tested, also on the skeleton page's 30 fps points). Seen from the left and in a side
   close-up of the head.
5. SARA 7, fast alternating hand movements (segment SARA7): 10 pronation-supination cycles of the left forearm
   (the chain, axis, sign and room of bradykinesia.py's pronation-supination task), seated with the forearm on the
   chair arm (the SARA protocol alternates on the thigh; stated), as a sequence of extremes joined by half-cosines:
   level 0 at the SARA demonstration pace (10 cycles in 7 s), full and regular; grades 1 to 3 draw the half-cycle
   durations, the peak amplitudes and incomplete returns with the coefficients of variation of DDK_GRADES, add
   interruptions (a hold near the rest, drifting slowly) and, in grades 2 and 3, supination overshoots corrected
   back, so that grades 0 to 2 end under 10 s and grade 3 over it (the anchors' time cut); grade 4 stops after
   DDK_G4_CYCLES cycles (unable to complete 10). The variabilities are module values: no published cycle CV of this
   task was found (SOURCES['ddk_unsourced']). On the thigh it is not built: from the upright 'sit' trunk the
   forearm slopes about 30 deg down to a palm resting on the thigh, and the wrist extension that lays the palm flat
   leaves at most about 80 deg of supination inside the joint-limit table (157 deg asked).
6. SARA 8, heel-shin slide (segment SARA8, task SH7 of docs/proposals/SYNTH_SHARED_CHANGES_DRAFT.md): supine on an
   examination couch (scene 'exam_bed', couch_boxes: a padded top COUCH_TOP_M above the room floor on a base, module
   values; the viewer draws these clips by bed_view: the room floor, not the lowest vertex, and the couch's heading).
   The supine rest (supine_posture) is the 'stand' posture turned onto the back (the root SUPINE_TILT_DEG short of
   90 deg, so that its heading is defined), solved on the skinned mesh so that the upper back, the head, the arms and
   hands and each leg rest at the buttocks' level, the couch top; the posterior thighs and calves lie above it (no
   soft tissue: their gaps are in the truth). The left heel (the heel set of its sole cloud, its centre rigid in the
   shank frame) is lifted, put on the right knee (HS_KNEE_BELOW_M below its joint line), slid down the shin line (the
   most anterior vertex of the right leg in bands, fitted) to HS_SLIDE_END of the way to the ankle (the lower shin,
   short of the ankle: NOT_RENDERED), lifted and laid back, HS_CYCLES times (SOURCES['sara8']). The leg is solved by
   two-bone inverse kinematics from the heel centre (ik.hinge_leg; the knee turned out of the vertical plane by the
   swivel with the best SP1 table margin), and the contact height on the skinned mesh: the foot touches the shin
   where the plan says contact and never enters it elsewhere. The left hip at the knee lies near the edge of the SP1
   table, so the pelvis turns in the couch plane by PELVIS_TURN_DEG about the right hip while the heel is on the
   right leg (the lumbar and thoracic joints turn back by SPINE_BACK_TURN: a side bend at the lumbar joint and a
   smaller one back higher up, so that the thorax keeps its orientation and nearly its place on the couch; the right
   leg does not move). Grades (HS_GRADES; module values, SOURCES['hs_unsourced'], except the slide time and the
   anchors): the level 0 is one smooth arc, a slide of about 1 s and a return; the ataxic grades take the family's
   reach parameters (reach.PRIMARY_SHARE: a dysmetric landing past or short of the knee, then the correction;
   reach.PHASE_PAUSE_S and PHASE_TIME_EXP: the approach decomposed into lift, carry and lowering, none of its
   movements faster than the level 0's arc) and the intention frequency of tremors.HZ (the heel's lateral oscillation
   on the approach and the slide). Every arc leaves and lands with zero velocity. Grade 1 keeps contact with the
   shin; grades 2 and 3 leave it the HS_GRADES number of times over the three slides (off-shin excursions of
   HS_EVENT_LAT_M to either side, lifted HS_EVENT_LIFT_M); grade 4 slides beside the shin in all three trials (rated
   4 by the form). On the AMASS table a landing past the knee near or beyond HS_PAST_MAX_M brought the moving hip to
   the table's edge or out of it (placement_margin_deg), so grades 3 and 4 land short of it; on the anatomical table
   (open-body port, wave 6) the hip keeps its room there and the split of the grades is kept. The standing checks do not apply; the
   validity is heel_shin_validity (joint limits, continuity, the couch with its padded top's yield COUCH_YIELD_M, the
   resting upper body's slide on the couch, the heel on the shin, the legs apart, the thighs pressed together within
   HS_THIGH_PRESS_M away from the groin). A right-side level 0 (the right heel on the left shin) is included.

Re-measurement (the tests compare them with the truth): measure_sway (centre-of-mass RMS per axis, mean frequency,
the 2 to 5 Hz over 0 to 2 Hz quotient of Baloh et al. 1998, the dominant frequency in 2 to 5 Hz, the step),
xcom_margin (the extrapolated centre of mass against the base of support), measure_trunk_sway (the pelvis-to-neck
line's angles, RMS and the share of time above a threshold), measure_titubation (dominant frequency and the peak to
peak of each cycle at the head joint, the head centre and the nose, titubation_readings), chest_rise_mm,
measure_alternation (cycles, time, rhythm and amplitude CVs, interruptions) and measure_heel_shin (slides done and
their time, off-shin excursions of the heel-to-shin-line distance above OFF_SHIN_M, the share of each slide on the
shin, the landing's error at the knee).
"""
import functools
import math

import numpy as np
import torch

from ..body.rotations import axis_angle_to_matrix, matrix_to_axis_angle, matrix_to_rot6d
from . import grades as G
from . import tremors as TR
from .base import posture_rotations, static_body
from .body import Body, axis_rotation, compose, contact_labels, joint_limits, rest_joints, validity
from .common import as_rng, truth_record
from .exam import ClipSpec

SINCE = "catalogue_v1"
SEED = 20261103
SOURCES = dict(G.SOURCES)
SOURCES.update({
    "sara": "Schmitz-Huebsch T et al. 2006, Neurology 66: 1717-1720, doi 10.1212/01.wnl.0000219042.60538.92 (PubMed "
            "record); the item anchors read on the SARA form (paraphrased): stance natural, feet together, tandem, "
            "10 s; sitting with the arms outstretched; 10 alternations on the thigh, demonstrated at about 10 in 7 s, "
            "grades 0 to 2 under 10 s, grade 3 over, grade 4 unable to complete",
    "quijoux2021": "Quijoux F et al. 2021, Physiol Rep 9: e15067 (full text), Santos and Duarte data, 76 healthy older "
                   "adults, eyes open, natural stance: centre-of-pressure RMS radius 0.59 (SD 0.26) cm, mean frequency "
                   "0.37 (SD 0.12) Hz",
    "mauritz1979": "Mauritz KH, Dichgans J, Hufschmidt A 1979, Brain 102: 461-482, doi 10.1093/brain/102.3.461 "
                   "(abstract): a 3 Hz anteroposterior postural tremor in anterior-lobe atrophy, a smaller mainly "
                   "lateral sway near 0.5 Hz",
    "diener1984": "Diener HC et al. 1984, EEG Clin Neurophysiol 57: 134-142, doi 10.1016/0013-4694(84)90172-X "
                  "(abstract): anterior-lobe atrophy, mainly anteroposterior sway with a 3 Hz body tremor",
    "baloh1998": "Baloh RW et al. 1998, Arch Neurol 55: 649-654, doi 10.1001/archneur.55.5.649 (abstract and text): "
                 "the Romberg ratio did not separate cerebellar patients from controls; 3 Hz anteroposterior tremor, "
                 "which 'could be seen at the bedside without the need for recording equipment' (Discussion, full "
                 "text read on jamanetwork.com 2026-11-05); the frequency quotient, power 2 to 5 Hz over 0 to 2 Hz "
                 "of the centre of pressure (read here on the centre of mass: a module reading)",
    "dankova2020": "Dankova M et al. 2020, Clin Neurophysiol 131: 2349-2356, doi 10.1016/j.clinph.2020.07.007 "
                   "(abstract): a 3 Hz postural tremor in 90 percent of 30 degenerative ataxia patients",
    "hpo_titubation": "Human Phenotype Ontology HP:0002599, head titubation: a head tremor of moderate speed (3 to 4 "
                      "Hz), anteroposterior (read on MedGen)",
    "ftm": "Fahn S, Tolosa E, Marin C 1988, in Jankovic J, Tolosa E (eds) Parkinson's disease and movement disorders, "
           "pp. 225-234 (not read): tremor amplitude bands under 0.5, 0.5 to 1, 1 to 2 and over 2 cm (grades 1 to 4) "
           "as commonly reproduced; a module reading for titubation",
    "ddk_unsourced": "no published cycle-duration or amplitude coefficient of variation of rapid pronation-supination "
                     "in healthy adults or cerebellar ataxia was found for this task (2026-11-02; Tran et al. 2020, "
                     "doi 10.1186/s12984-020-00790-3, report spectral features only): module values",
    "anagnostou2024": "Anagnostou E et al. 2024, Neurophysiol Clin 54: 102999, doi 10.1016/j.neucli.2024.102999 "
                      "(abstract): Romberg quotients around 2 in sensory neuropathy, modest separation from controls",
    "helmchen2017": "Helmchen C et al. 2017, PLoS One 12: e0168808 (full text): Romberg ratio 2.09 (SE 0.22) in "
                    "cerebellar downbeat nystagmus against 1.78 (0.12) in controls, not significant",
    "winter1995": "Winter DA 1995, Gait Posture 3: 193-214 (not read here, as commonly cited): side-to-side balance "
                  "in stance by the hips' load and unload, forward-back by the ankles",
    "romberg_unsourced": "the eyes-open sway gains are module values; the posturographic eyes-closed over eyes-open "
                         "sway ratios follow the published values (healthy and cerebellar about 1.5, baloh1998; "
                         "sensory about 2, anagnostou2024); the bedside sign's ratio (3.5, then a catching step) "
                         "is a module value: no published ratio of a bedside Romberg sign was found",
})

# ---------------------------------------------------------------- sway noise
SWAY_BAND_HZ = (0.12, 2.0)     # [module] 1/f amplitude band: mean frequency about 0.36 Hz (quijoux2021: 0.37)
SWAY_RMS_M = 0.0059 / math.sqrt(2.0)   # SOURCES['quijoux2021'] RMS radius split between the axes
TREMOR3_HZ = 3.0               # SOURCES['mauritz1979'], ['diener1984'], ['baloh1998']
TREMOR3_HALF_HZ = 0.25         # [module] half width of the 3 Hz band
TREMOR3_RMS_M = 0.0015         # [module] RMS of the 3 Hz centre-of-mass component in the ataxic clips (0.003 until the
#                                review of 2026-11-03 read the pelvis at 10 to 30 mm peak to peak and 0.6 g; a centre of
#                                mass moving x RMS at f needs a horizontal ground force of (2 pi f)^2 x / g of the body
#                                weight RMS, in the truth: 'com_shear_bw_rms'); baloh1998 reports the tremor seen at the
#                                bedside but gives no amplitude, so the pelvis's 5 to 11 mm peak to peak (review of
#                                2026-11-05) is a module reading, and the caption calls the tremor small


def colored_noise(T, rate, rms, band, rng, slope=1.0):
    """(T,) zero-mean noise whose amplitude spectrum is f^-slope inside band (Hz) and zero outside, scaled to rms."""
    f = np.fft.rfftfreq(T, 1.0 / rate)
    amp = np.where((f >= band[0]) & (f <= band[1]), 1.0 / np.maximum(f, 1e-9) ** slope, 0.0)
    ph = rng.uniform(0, 2 * np.pi, size=len(f))
    x = np.fft.irfft(amp * np.exp(1j * ph), n=T)
    x -= x.mean()
    s = x.std()
    return x * (rms / s) if s > 0 else x


def narrow_noise(T, rate, rms, f0, half, rng):
    return colored_noise(T, rate, rms, (f0 - half, f0 + half), rng, slope=0.0)


# ---------------------------------------------------------------- stance bodies
REACH_ST = 0.997               # [module] the farther leg's hip-to-ankle distance over its length, tilt included
#                                (open-body port, wave 6: on the open body's straight rest legs it bends the farther
#                                knee 8.9 deg; 0.999, which bends it 5.0, broke the stance's continuity check at the
#                                straight knee; scripts/openbody_wave6_checks.py, ataxia)
#                                (knees about 7 deg; 0.985 until the review of 2026-11-02 bent them 15 to 17 deg)
STEP_S = 0.5                   # [module] the catching step's swing
STEP_LIFT_M = 0.06             # [module] its ankle lift (the foot kept flat; open-body port, wave 6: where the swing
#                                ankle would pass the table's dorsiflexion, the toes drop by the excess: the tandem
#                                stance's catching step stood 11.9 deg past jarm2022's 20 with the foot flat)
STEP_SIDE_END = 0.6            # [module] the swing foot moves sideways over the first STEP_SIDE_END of the swing and
STEP_FWD_START = 0.35          # [module] forward or back from STEP_FWD_START on: it passes beside the other foot
STEP_KNEE_OUT_DEG = 15.0       # [module] the swing knee points this far outward at mid-swing (away from the other)
STEP_REACT_S = 0.1             # [module] the foot lifts this long after the extrapolated centre of mass leaves the base
STEP_REACT_MAX_S = 0.2         # [module] at most this long, at least -STEP_REACT_S (read on the final body)
STANCE_DRAWS = 20              # [module] noise draws tried until the held stance keeps the extrapolated centre of mass
#                                inside the base (and a failing one leaves it first toward its stepping side)
GROW_S = 2.0                   # [module] the sway grows over this before a failure
GROW_GAIN = (1.2, 2.5)         # [module] by these factors, forward-back and side to side
LEAN_DEG = 4.0                 # [module] the lean toward the stepping side (the centre of mass's shift of a whole-body
#                                tilt of this angle), built over LEAN_S before the failure, LEAN_TILT_SHARE of it as
#                                that tilt, the rest as a shift of the pelvis (a tilt alone bends the near knee 20 deg)
LEAN_TILT_SHARE = 0.3
LEAN_S = 1.5
SETTLE_GAIN = 0.6              # [module] sway after the step, share of the condition's
ARM_ABD_DEG = 20.0             # [module] both shoulders abduct by this over the last ARM_LEAD_S before the step, held
ARM_LEAD_S = 1.0               # to the landing, released over ARM_RELEASE_S
ARM_RELEASE_S = 1.0
TREMOR3_TRUNK = 1.0            # [module] 3 Hz component: the trunk's angle over the legs' angle, in antiphase
TREMOR3_ARMS_STILL = ((16, (3, 6, 9, 13)), (17, (3, 6, 9, 14)))   # [module] the shoulders (and their chains from the
#                                pelvis) whose arms keep their orientation in space against the trunk's 3 Hz turn: an
#                                approximation (a module reading) of a hanging limb driven above its pendulum
#                                frequency, which keeps a point near two thirds of its length still rather than its
#                                orientation (the counter-swing of the wrist, a few mm, is not modelled); rigid on the
#                                trunk, the wrists at hip height followed the pelvis to within 1 mm (review of
#                                2026-11-03)
STANCE_SOLVE = 2               # corrections of the stance angles by the centre-of-mass residual
STANCE_S = 12.0
FAIL_AT_S = {"tandem": 5.0, "together": 4.0}    # [module] the nominal failure (the sway and lean are full here)
LEGS = ((1, 4, 7, 10), (2, 5, 8, 11))



@functools.lru_cache(maxsize=1)
def foot_geometry():
    """{side: {'heel_dz', 'toe_dz', 'medial_dx', 'centre_dx'}} m: the sole cloud's extent relative to the ankle joint
    on the neutral rest mesh (z forward; medial_dx the edge toward the midline, signed in x)."""
    from . import hands as H
    from .contact import sole_vertices
    J, v, _ = H._rest()
    out = {}
    for side, a in (("left", 7), ("right", 8)):
        ids = sole_vertices(2)[side]
        x, z = v[ids, 0] - J[a, 0], v[ids, 2] - J[a, 2]
        out[side] = {"heel_dz": float(z.min()), "toe_dz": float(z.max()),
                     "medial_dx": float(x.min() if side == "left" else x.max()),
                     "centre_dx": float(0.5 * (x.min() + x.max()))}
    return out


def ankle_targets(kind):
    """{'left': (x, z), 'right': (x, z)} m of the ankle joints in a stance kind (module docstring), the pelvis at x =
    z = 0 of the procedural stand."""
    r = rest_joints().numpy()
    fg = foot_geometry()
    z0 = float(r[7, 2])
    if kind == "natural":
        return {"left": (float(r[7, 0]), z0), "right": (float(r[8, 0]), float(r[8, 2]))}
    if kind == "together":
        xl = 0.002 - fg["left"]["medial_dx"]
        xr = -0.002 - fg["right"]["medial_dx"]
        return {"left": (xl, z0), "right": (xr, z0)}
    if kind == "tandem":
        d = fg["right"]["toe_dz"] - fg["left"]["heel_dz"] + 0.005
        return {"left": (-fg["left"]["centre_dx"], z0 + d / 2), "right": (-fg["right"]["centre_dx"], z0 - d / 2)}
    raise ValueError(f"stance {kind!r}")


@functools.lru_cache(maxsize=1)
def stand_gains():
    """{'ankle_h', 'h_com'} of the procedural stand: the ankle height over the joint floor and the centre of mass's
    height over the ankles (m: its forward shift per radian of a rigid tilt about them)."""
    from ..kernels.force import com
    r = rest_joints().numpy()
    floor_j = min(r[10, 1], r[11, 1])
    ankle_h = 0.5 * (float(r[7, 1] - floor_j) + float(r[8, 1] - floor_j))
    J0 = static_body("stand", 0.01).joints().numpy()[0]
    return {"ankle_h": ankle_h, "h_com": float(com(J0, 2)[1] - ankle_h)}


@functools.lru_cache(maxsize=8)
def pattern_gain(kind, pattern, eps=1e-4):
    """The centre of mass's shift per unit of a stance pattern (_stance) in a stance kind: 'tremor3' m forward per
    radian of the 3 Hz leg angle (the pelvis and trunk at -TREMOR3_TRUNK of it about the hips), 'shift' m toward
    the left per m of lateral pelvis shift (the trunk upright), 'tilt' m forward per radian of whole-body tilt about
    the ankles (the legs solved to the fixed ankles)."""
    from ..kernels.force import com
    z = np.zeros(2)
    c0 = com(_stance(kind, z, z, z).joints().numpy(), 2)
    if pattern == "tilt":
        return float((com(_stance(kind, z + eps, z, z).joints().numpy(), 2)[0, 2] - c0[0, 2]) / eps)
    if pattern == "tremor3":
        return float((com(_stance(kind, z, z, z + eps).joints().numpy(), 2)[0, 2] - c0[0, 2]) / eps)
    if pattern == "shift":
        return float((com(_stance(kind, z, z, z, shift_ml=z + eps).joints().numpy(), 2)[0, 0] - c0[0, 0]) / eps)
    raise ValueError(pattern)


def _step_path(A0, to, t, t_s, lift):
    """(T, 3) the swing ankle of a catching step from A0 (3,) to `to` (3,), starting at t_s (module docstring)."""
    from .parkinsonism import _mj
    u = np.clip((t - t_s) / STEP_S, 0.0, 1.0)
    wx = _mj(u / STEP_SIDE_END)
    wz = _mj((u - STEP_FWD_START) / (1.0 - STEP_FWD_START))
    out = np.repeat(np.asarray(A0, dtype=np.float64)[None], len(t), 0)
    out[:, 0] += wx * (to[0] - A0[0])
    out[:, 2] += wz * (to[2] - A0[2])
    out[:, 1] += lift * np.sin(np.pi * u)
    return out, (u > 0) & (u < 1)


def stance_body(kind, ap_m, ml_m, rate=100.0, step=None, ap3_m=None, arms_deg=None, lean_ml_m=None):
    """(Body, info) of a stance (module docstring): ap_m (T,) the forward centre-of-mass sway target (m), carried by
    a tilt of the whole body about the ankle midpoint (an inverted pendulum); ml_m (T,) the target toward the body's
    left, carried by a lateral shift of the pelvis with the trunk upright and the legs inclined (the load and unload
    of the hips: winter1995); lean_ml_m (T,) a lean toward the left carried by a whole-body tilt; ap3_m (T,) a
    forward target carried by the 3 Hz pattern (the legs tilt about the ankles by phi, the pelvis and trunk turn back
    about the hips to -TREMOR3_TRUNK phi: the hips move, the head much less); arms_deg (T,) the abduction of both
    shoulders; step: None or {'side', 't_s', 'to': (x, z)} the catching step of one foot (sideways first, then
    forward or back: _step_path)."""
    from ..kernels.force import com
    h_com = stand_gains()["h_com"]
    T = len(ap_m)
    phi3 = np.zeros(T) if ap3_m is None else np.asarray(ap3_m, dtype=np.float64) / pattern_gain(kind, "tremor3")
    lean = np.zeros(T) if lean_ml_m is None else np.asarray(lean_ml_m, dtype=np.float64)
    want = np.stack([np.asarray(ml_m, dtype=np.float64) + lean,
                     np.asarray(ap_m, dtype=np.float64) + (0.0 if ap3_m is None else np.asarray(ap3_m))], -1)
    z = np.zeros(T)
    ref = com(_stance(kind, z, z, z, rate, step, arms_deg).joints().numpy(), 2)[:, [0, 2]]
    g_tilt, g_shift = pattern_gain(kind, "tilt"), pattern_gain(kind, "shift")
    th_ap, shift = np.asarray(ap_m, dtype=np.float64) / g_tilt, np.asarray(ml_m, dtype=np.float64) / g_shift
    for it in range(STANCE_SOLVE + 1):             # the gains are linear readings: correct by the residual
        b, info = _stance(kind, th_ap, -lean / h_com, phi3, rate, step, arms_deg, shift_ml=shift, with_info=True)
        res = want - (com(b.joints().numpy(), 2)[:, [0, 2]] - ref)
        if it < STANCE_SOLVE:
            th_ap, shift = th_ap + res[:, 1] / g_tilt, shift + res[:, 0] / g_shift
    info["tremor3_pattern"] = None if ap3_m is None else {
        "legs_deg_rms": float(np.degrees(phi3.std())), "trunk_over_legs": -TREMOR3_TRUNK,
        "arms": ("orientation kept in space against the trunk's 3 Hz turn (an approximation of a hanging limb "
                 "driven above its pendulum frequency, whose wrist would counter-swing by a few mm: not modelled)"),
        "com_m_per_rad": pattern_gain(kind, "tremor3"),
        "com_shear_bw_rms": float((2 * math.pi * TREMOR3_HZ) ** 2 * np.std(ap3_m) / 9.81)}
    info["com_residual_mm_max"] = float(1000 * np.abs(res).max())
    return b, info


def _stance(kind, th_ap, th_ml, phi3, rate=100.0, step=None, arms_deg=None, shift_ml=None, with_info=False):
    """The stance body of stance_body from its angles (rad, (T,)): the rigid tilt th_ap (about +x, forward) and
    th_ml (about +z, toward the right for positive angles), the 3 Hz leg angle phi3; shift_ml (T,) m the lateral
    shift of the pelvis toward the left."""
    from .ik import hinge_leg
    T = len(th_ap)
    t = np.arange(T) / rate
    rest = rest_joints()
    r = rest.numpy()
    floor_j = min(r[10, 1], r[11, 1])                     # the procedural stand's floor (base.static_body)
    ah = (float(r[7, 1] - floor_j), float(r[8, 1] - floor_j))   # per side: the rest skeleton is not symmetric
    ankle_h = stand_gains()["ankle_h"]
    tg = ankle_targets(kind)
    A = np.zeros((T, 2, 3))
    for i, s in enumerate(("left", "right")):
        A[:, i] = [tg[s][0], ah[i], tg[s][1]]
    swing = np.zeros(T, bool)
    if step is not None:
        i = 0 if step["side"] == "left" else 1
        A[:, i], swing = _step_path(A[0, i], (step["to"][0], ah[i], step["to"][1]), t, step["t_s"], STEP_LIFT_M)
    # the untilted pelvis follows the midpoint of the ankles horizontally (min-jerk over the step and after it)
    mid = A[:, :, [0, 2]].mean(1)
    if step is not None:
        from .parkinsonism import _mj
        m0, m1 = A[0][:, [0, 2]].mean(0), A[-1][:, [0, 2]].mean(0)
        w = _mj((t - step["t_s"]) / (STEP_S + 0.5))
        mid = m0[None] + w[:, None] * (m1 - m0)[None]
    nat = np.array([0.5 * (r[7, 0] + r[8, 0]), 0.5 * (r[7, 2] + r[8, 2])])
    th_ml = np.asarray(th_ml, dtype=np.float64)

    def tilt(ap):
        return (axis_angle_to_matrix(torch.from_numpy(np.stack([ap, 0 * ap, 0 * ap], -1))) @
                axis_angle_to_matrix(torch.from_numpy(np.stack([0 * th_ml, 0 * th_ml, th_ml], -1))))
    RL = tilt(np.asarray(th_ap, dtype=np.float64) + phi3)                     # the legs (the hips' place)
    RP = tilt(np.asarray(th_ap, dtype=np.float64) - TREMOR3_TRUNK * phi3)     # the pelvis and above
    pivot = np.zeros((T, 3))
    pivot[:, [0, 2]] = mid
    pivot[:, 1] = ankle_h
    # the pelvis height at which the farther tilted hip lies REACH_ST of its leg from its ankle, then a 0.3 s
    # running minimum and mean (both keep it at or below that height)
    hip_off = np.stack([r[1] - r[0], r[2] - r[0]])
    hm = hip_off.mean(0)
    Ls = [float(np.linalg.norm(r[a] - r[h])) for h, _, a, _ in LEGS]
    RLn, RPn = RL.numpy(), RP.numpy()
    c = np.zeros((T, 3))
    c[:, [0, 2]] = mid - nat[None] + hm[[0, 2]][None] - pivot[:, [0, 2]]
    c[:, 1] = hm[1] - ankle_h
    if shift_ml is not None:
        c[:, 0] += shift_ml
    u = RLn[:, :, 1]
    pys = []
    for i in range(2):
        w = pivot + np.einsum("tij,tj->ti", RLn, c) + np.einsum("tij,j->ti", RPn, hip_off[i] - hm) - A[:, i]
        wu = (w * u).sum(-1)
        pys.append(-wu + np.sqrt(np.maximum(wu ** 2 - (w * w).sum(-1) + (REACH_ST * Ls[i]) ** 2, 0.0)))
    py = np.minimum(pys[0], pys[1])
    n = int(round(0.3 * rate))
    pad = np.pad(py, (n // 2, n - 1 - n // 2), mode="edge")
    lo = np.lib.stride_tricks.sliding_window_view(pad, n).min(1)
    py = np.convolve(np.pad(lo, (n // 2, n - 1 - n // 2), mode="edge"), np.ones(n) / n, mode="valid")
    c[:, 1] += py
    hipmid = pivot + np.einsum("tij,tj->ti", RLn, c)
    Pp = torch.from_numpy(hipmid - np.einsum("tij,j->ti", RPn, hm))
    R = posture_rotations("stand").repeat(T, 1, 1, 1)
    R[:, 0] = RP
    if arms_deg is not None:
        a = torch.from_numpy(np.radians(np.asarray(arms_deg, dtype=np.float64)))
        R[:, 16] = compose(R[:, 16], axis_rotation("z", a), "parent")
        R[:, 17] = compose(R[:, 17], axis_rotation("z", -a), "parent")
    if np.any(np.asarray(phi3) != 0.0):
        # the hanging arms keep their orientation in space against the trunk's 3 Hz turn (TREMOR3_ARMS_STILL, an
        # approximation: a module reading)
        RP0 = tilt(np.asarray(th_ap, dtype=np.float64))
        R0 = posture_rotations("stand")
        for sh, chain in TREMOR3_ARMS_STILL:
            Q = R0[chain[0]]
            for j in chain[1:]:
                Q = Q @ R0[j]
            R[:, sh] = (Q.transpose(-1, -2) @ RP.transpose(-1, -2) @ RP0 @ Q) @ R[:, sh]
    At = torch.from_numpy(A)
    fwd = RP[:, :, 2]
    knee = []
    for i, (h, kn, a, _) in enumerate(LEGS):
        Hh = Pp + torch.einsum("tij,j->ti", RP, rest[h] - rest[0])
        f = fwd
        if step is not None and i == (0 if step["side"] == "left" else 1):   # the swing knee turned outward
            e = (np.sin(np.pi * np.clip((t - step["t_s"]) / STEP_S, 0.0, 1.0)) *
                 math.tan(math.radians(STEP_KNEE_OUT_DEG)))
            out = torch.zeros(T, 3, dtype=torch.float64)
            out[:, 0] = torch.from_numpy(e) * (1.0 if i == 0 else -1.0)
            f = fwd + out
            f = f / f.norm(dim=-1, keepdim=True)
        G_th, kap = hinge_leg(Hh, At[:, i], Hh + f, rest[kn] - rest[h], rest[a] - rest[kn], RP,
                              hinge=torch.cross(f, At[:, i] - Hh, dim=-1))
        Rk = axis_angle_to_matrix(torch.stack([kap, 0 * kap, 0 * kap], -1))
        R[:, h] = RP.transpose(-1, -2) @ G_th
        R[:, kn] = Rk
        R[:, a] = (G_th @ Rk).transpose(-1, -2)                           # the foot flat, along +z
        if step is not None and i == (0 if step["side"] == "left" else 1):   # the swing ankle kept in the table:
            lo = float(joint_limits().lo[a][0])                        # past its dorsiflexion the toes drop by the
            for _ in range(4):                                         # excess (open-body port, wave 6), read again
                x = np.degrees(matrix_to_axis_angle(R[:, a]).numpy()[:, 0])     # on the turned foot
                d = np.where(swing, np.maximum(lo - x, 0.0), 0.0)
                if not (d > 1e-9).any():
                    break
                R[:, a] = compose(R[:, a], axis_rotation("x", torch.from_numpy(np.radians(d))), "parent")
        knee.append(np.degrees(kap.numpy()))
    b = Body(matrix_to_rot6d(R), Pp - rest[0], torch.zeros(T, 2, dtype=torch.bool), rate=rate)
    b = Body(b.rot6d, b.transl, contact_labels(b), b.betas, b.gender, b.rate)
    if not with_info:
        return b
    info = {"kind": kind, "ankles_m": {s: list(tg[s]) for s in tg}, "reach": REACH_ST,
            "h_com_m": stand_gains()["h_com"], "tilt_ap_deg_max": float(np.degrees(np.abs(th_ap).max())),
            "tilt_ml_deg_max": float(np.degrees(np.abs(th_ml).max())), "step": step,
            "swing_knots": int(np.asarray(swing).sum()),
            "knee_deg": {"min": float(min(k.min() for k in knee)), "max": float(max(k.max() for k in knee)),
                         "median": float(np.median(np.concatenate(knee)))},
            "arms_abduction_deg_max": 0.0 if arms_deg is None else float(np.max(arms_deg)),
            "shift_ml_cm_max": 0.0 if shift_ml is None else float(100 * np.abs(shift_ml).max())}
    return b, info


@functools.lru_cache(maxsize=4)
def support_hull(kind):
    """(equations (F, 3), sole floor y m) of the convex hull of the two sole clouds (contact.sole_vertices) of a
    stance kind standing still, in the floor plane (x, z): equations . (x, z, 1) <= 0 inside."""
    from scipy.spatial import ConvexHull
    from . import hands as H
    from .contact import sole_vertices
    z = np.zeros(2)
    b, _ = stance_body(kind, z, z)
    sv = sole_vertices(2)
    P = np.concatenate([H.body_points(b, sv[s], knots=[0])[0] for s in ("left", "right")])
    return ConvexHull(P[:, [0, 2]]).equations, float(P[:, 1].min())


XCOM_LOWPASS_HZ = 2.0          # [module] the sway's centre of mass for the extrapolated centre of mass: the 3 Hz
#                                tremor's own velocity is left out (it does not make the patient step)


def xcom_margin(body, kind, lowpass_hz=XCOM_LOWPASS_HZ):
    """(T, 2) m and (T,) m: the extrapolated centre of mass (hof2005: x + v / omega0, omega0 = sqrt(g / l), l the
    centre of mass's height over the soles) of the centre of mass low-passed at lowpass_hz (zero-phase, 4th-order
    Butterworth; None: unfiltered) in (x, z) and its signed distance to the base of support of the kind's planted
    feet (support_hull), positive outside."""
    from scipy.signal import butter, filtfilt
    from ..kernels.force import com
    eq, floor = support_hull(kind)
    c = com(body.joints().numpy(), int(body.gender))
    if lowpass_hz:
        bb, aa = butter(4, lowpass_hz / (0.5 * body.rate))
        c = filtfilt(bb, aa, c, axis=0)
    v = np.gradient(c, 1.0 / body.rate, axis=0)
    w0 = np.sqrt(9.81 / (c[:, 1] - floor))
    X = c[:, [0, 2]] + v[:, [0, 2]] / w0[:, None]
    return X, (X @ eq[:, :2].T + eq[:, 2]).max(1)


def failing_stance(kind, ap_low, ml_low, ap3, tf, rate=100.0, side="right", t_from=None, level=None):
    """(Body, info, (ap, ml) the realized targets) of a stance that fails (module docstring): the targets ap_low,
    ml_low, ap3 (T,) m grow from tf - GROW_S to tf by GROW_GAIN (AP, ML), the body leans toward `side` by LEAN_DEG
    over LEAN_S before tf, and the foot of that side lifts STEP_REACT_S after the extrapolated centre of mass first
    leaves the base of support (xcom_margin, searched from t_from, default tf - GROW_S) for a natural stance; the
    shoulders abduct over the last ARM_LEAD_S before the lift; after it the sway settles to SETTLE_GAIN and the lean
    and the arms are released. level (T,): an extra gain on both axes (the Romberg eye closure)."""
    from .parkinsonism import _mj
    T = len(ap_low)
    t = np.arange(T) / rate
    h_com = stand_gains()["h_com"]
    lv = np.ones(T) if level is None else np.asarray(level)
    grow = _mj((t - (tf - GROW_S)) / GROW_S)
    g_ap = lv * (1.0 + (GROW_GAIN[0] - 1.0) * grow)
    g_ml = lv * (1.0 + (GROW_GAIN[1] - 1.0) * grow)
    sgn = 1.0 if side == "right" else -1.0
    lean_on = math.radians(LEAN_DEG) * _mj((t - (tf - LEAN_S)) / LEAN_S)
    b0, _ = stance_body(kind, g_ap * ap_low, g_ml * ml_low - (1.0 - LEAN_TILT_SHARE) * sgn * lean_on * h_com, rate,
                        ap3_m=g_ap * ap3, lean_ml_m=-LEAN_TILT_SHARE * sgn * lean_on * h_com)
    _, d0 = xcom_margin(b0, kind)
    a = int(round((tf - GROW_S if t_from is None else t_from) * rate))
    out = np.flatnonzero(d0[a:] > 0)
    t_exit = (a + int(out[0])) / rate if len(out) else None
    t_s = (t_exit + STEP_REACT_S) if t_exit is not None else tf
    settle = _mj((t - t_s) / 1.5)
    g_ap = g_ap - (g_ap - SETTLE_GAIN * lv) * settle
    g_ml = g_ml - (g_ml - SETTLE_GAIN * lv) * settle
    lean = lean_on * (1.0 - _mj((t - t_s - 0.3) / 1.5))
    arms = ARM_ABD_DEG * (_mj((t - (t_s - ARM_LEAD_S)) / ARM_LEAD_S) - _mj((t - t_s - STEP_S) / ARM_RELEASE_S))
    nat = ankle_targets("natural")[side]
    step = {"side": side, "t_s": float(t_s), "to": nat}
    ap, ml = g_ap * ap_low, g_ml * ml_low - (1.0 - LEAN_TILT_SHARE) * sgn * lean * h_com
    b, info = stance_body(kind, ap, ml, rate, step, ap3_m=g_ap * ap3, arms_deg=arms,
                          lean_ml_m=-LEAN_TILT_SHARE * sgn * lean * h_com)
    X, d = xcom_margin(b, kind)
    k_s = int(round((t_s + STEP_S) * rate))                # read on the final body, up to the landing
    outside = np.flatnonzero(d[a:k_s + 1] > 0)
    t_out = (a + int(outside[0])) / rate if len(outside) else None
    side_out = None
    if t_out is not None:
        k = int(round(t_out * rate))
        side_out = "right" if X[k, 0] < 0.5 * sum(ankle_targets(kind)[s][0] for s in ("left", "right")) else "left"
    info["failure"] = {"t_nominal_s": tf, "grow_gain": list(GROW_GAIN), "lean_deg": LEAN_DEG,
                       "lean_tilt_share": LEAN_TILT_SHARE, "side": side,
                       "xcom_exit_s": t_out, "xcom_exit_side": side_out, "lift_s": float(t_s),
                       "lift_delay_s": None if t_out is None else float(t_s - t_out),
                       "xcom_outside_before_search_s": float(np.flatnonzero(d[:a] > 0)[0] / rate)
                       if (d[:a] > 0).any() else None,
                       "arms_abduction_deg": ARM_ABD_DEG, "search_from_s": a / rate}
    return b, info, (ap + g_ap * ap3, ml - LEAN_TILT_SHARE * sgn * lean * h_com)


# conditions and grades: (sway gain AP, ML) on SWAY_RMS_M, 3 Hz component, failure (module values)
STANCE_GAIN = {"natural": (1.0, 1.0), "together": (1.0, 1.5), "tandem": (1.2, 2.0)}
STANCE_LADDER = {
    ("natural", 0): {"gain": 1.0, "tremor3": False, "fail": False},
    ("natural", 3): {"gain": 3.0, "tremor3": True, "fail": False},
    ("together", 0): {"gain": 1.0, "tremor3": False, "fail": False},
    ("together", 2): {"gain": 3.0, "tremor3": True, "fail": False},
    ("together", 3): {"gain": 3.0, "tremor3": True, "fail": True},
    ("tandem", 0): {"gain": 1.0, "tremor3": False, "fail": False},
    ("tandem", 1): {"gain": 1.0, "tremor3": False, "fail": True},
}
STANCE_TEXT = {
    ("natural", 0): "natural stance held, normal sway", ("natural", 3): "natural stance held over 10 s with sway",
    ("together", 0): "feet together held without sway", ("together", 2): "feet together held over 10 s, with sway",
    ("together", 3): "feet together not held: a catching step before 10 s",
    ("tandem", 0): "tandem held over 10 s", ("tandem", 1): "tandem not held: a catching step before 10 s",
}
TREMOR3_TEXT = f"a small {TREMOR3_HZ:g} Hz forward and back postural tremor of the legs and trunk"   # the caption's
#                                words (small: the pelvis moves about 1 cm peak to peak at most, review of 2026-11-05)


def stance_text(kind, grade):
    """The caption's parameter text of a SARA 2 clip: STANCE_TEXT, and the 3 Hz component when the grade has it
    (review of 2026-11-04: the hips' 3 Hz oscillation is visible, so the caption names it)."""
    lad = STANCE_LADDER[(kind, int(grade))]
    return STANCE_TEXT[(kind, int(grade))] + (f"; {TREMOR3_TEXT}" if lad["tremor3"] else "")


def sway_targets(T, rate, gain_ap, gain_ml, tremor3, rng, envelope=None):
    """(ap (T,), ml (T,), parts) m: the centre-of-mass sway targets (module docstring)."""
    ap = colored_noise(T, rate, SWAY_RMS_M * gain_ap, SWAY_BAND_HZ, rng)
    ml = colored_noise(T, rate, SWAY_RMS_M * gain_ml, SWAY_BAND_HZ, rng)
    tr3 = narrow_noise(T, rate, TREMOR3_RMS_M, TREMOR3_HZ, TREMOR3_HALF_HZ, rng) if tremor3 else np.zeros(T)
    e = np.ones(T) if envelope is None else envelope
    return e * (ap + tr3), e * ml, {"ap_low": ap, "ml_low": ml, "ap_3hz": tr3}


def stance_grade(kind, grade, seed=SEED, duration_s=STANCE_S, rate=100.0):
    """(Body, truth) of a SARA 2 stance clip (module docstring)."""
    lad = STANCE_LADDER[(kind, int(grade))]
    rng = as_rng(seed)
    T = int(round(duration_s * rate)) + 1
    ga, gm = STANCE_GAIN[kind]
    for draw in range(1, STANCE_DRAWS + 1):
        _, _, parts = sway_targets(T, rate, lad["gain"] * ga, lad["gain"] * gm, lad["tremor3"], rng)
        ap3 = parts["ap_3hz"] if lad["tremor3"] else None
        if lad["fail"]:
            b, info, (ap, ml) = failing_stance(kind, parts["ap_low"], parts["ml_low"], parts["ap_3hz"],
                                               FAIL_AT_S[kind], rate)
            f = info["failure"]
            w_end = min(FAIL_AT_S[kind] - GROW_S, f["lift_s"] - ARM_LEAD_S)
            ok = (f["xcom_outside_before_search_s"] is None and f["xcom_exit_side"] == f["side"] and
                  f["lift_delay_s"] is not None and -STEP_REACT_S <= f["lift_delay_s"] <= STEP_REACT_MAX_S)
        else:
            ap, ml = parts["ap_low"] + parts["ap_3hz"], parts["ml_low"]
            b, info = stance_body(kind, parts["ap_low"], ml, rate, ap3_m=ap3)
            w_end = None
            ok = bool(xcom_margin(b, kind)[1].max() < 0.0)
        if ok:
            break
    else:
        raise RuntimeError(f"{kind} grade {grade}: no draw of {STANCE_DRAWS} keeps the held stance inside the base")
    info["draws"] = draw
    info["xcom_margin_min_cm"] = float(-100 * xcom_margin(b, kind)[1][:None if w_end is None else
                                                                    int(rate * (info["failure"]["xcom_exit_s"]))].max())
    feet = (0, 1) if not lad["fail"] else (0,)
    base = static_body("stand", duration_s)
    val = validity(base, b, feet=feet)
    val["continuity"] = _continuity(b)
    k = None if w_end is None else int(rate * w_end)
    p = {"condition": kind, "grade": int(grade), "duration_s": duration_s, "sway_rms_m": SWAY_RMS_M,
         "gain": lad["gain"], "condition_gain": [ga, gm], "tremor3": lad["tremor3"],
         "tremor3_hz": TREMOR3_HZ if lad["tremor3"] else None,
         "tremor3_pattern": ("legs about the ankles, trunk in antiphase (module)" if lad["tremor3"] else None),
         "fails": lad["fail"], "fail_at_s": info["failure"]["lift_s"] if lad["fail"] else None,
         "text": stance_text(kind, grade),
         "target_window_s": [0.0, w_end if w_end is not None else duration_s],
         "target_rms_cm": {"ap": float(100 * np.std(ap[:k])), "ml": float(100 * np.std(ml[:k]))},
         "stance": info, "remeasured": measure_sway(b, None if w_end is None else (0.0, w_end))}
    srcs = ["sara", "quijoux2021", "winter1995", "module"]
    srcs += ["mauritz1979", "diener1984", "baloh1998"] if lad["tremor3"] else []
    srcs += ["hof2005"] if lad["fail"] else []
    return b, truth_record("ataxic_stance", seed, int(grade), "SARA 2 stance", p, srcs, val,
                           arrays={"ap_m": ap, "ml_m": ml})


def _continuity(b, bound_deg_s=500.0):
    from .body import rotation_angle_deg
    R = b.local_R()
    st = rotation_angle_deg(R[:-1].transpose(-1, -2) @ R[1:]).numpy()
    return {"ok": bool(st.max() * b.rate <= bound_deg_s), "max_step_deg": float(st.max()), "bound_deg_s": bound_deg_s}


def measure_sway(body, window_s=None):
    """Centre-of-mass sway read back (de Leva table, mdfm/kernels/force.py): over window_s (default the clip), the
    horizontal displacement from its mean, RMS per axis (cm; ap forward z, ml x), the mean frequency (power-weighted,
    0.05 to 6 Hz) per axis, the frequency quotient P[2, 5] / P[0.05, 2] of the AP displacement (baloh1998) and its
    dominant frequency in 2 to 5 Hz; the largest horizontal displacement of each ankle (a step)."""
    from ..kernels.force import com
    from .measure import band_power, dominant_frequency, periodogram
    J = body.joints().numpy()
    c = com(J, int(body.gender))
    r = body.rate
    a, e = (0, len(c)) if window_s is None else (int(window_s[0] * r), int(window_s[1] * r) + 1)
    x = c[a:e, [2, 0]] - c[a:e, [2, 0]].mean(0)
    out = {"window_s": [a / r, (e - 1) / r]}
    for i, ax in enumerate(("ap", "ml")):
        f, p = periodogram(x[:, i], r, pad=4)
        sel = (f >= 0.05) & (f <= 6.0)
        out[f"{ax}_rms_cm"] = float(100 * x[:, i].std())
        out[f"{ax}_mean_hz"] = float((f[sel] * p[sel]).sum() / p[sel].sum())
    out["ap_quotient_2_5_over_0_2"] = band_power(x[:, 0], r, (2.0, 5.0)) / max(band_power(x[:, 0], r, (0.05, 2.0)),
                                                                                1e-30)
    out["ap_peak_2_5_hz"] = dominant_frequency(x[:, 0], r, (2.0, 5.0))
    out["ankle_travel_m"] = [float(np.linalg.norm(J[:, j, [0, 2]] - J[0, j, [0, 2]], axis=1).max()) for j in (7, 8)]
    return out


# ---------------------------------------------------------------- Romberg
ROMBERG_S = 20.0
ROMBERG_CLOSE_S = 10.0
ROMBERG = {"healthy": {"gain_open": 1.0, "ratio": 1.5, "tremor3": False},       # baloh1998: control mean about 1.5
           "cerebellar": {"gain_open": 2.2, "ratio": 1.5, "tremor3": True},     # baloh1998: not separated from controls
           "sensory": {"gain_open": 1.5, "ratio": 2.0, "tremor3": False},       # anagnostou2024: about 2
           "sensory_bedside": {"gain_open": 1.1, "ratio": 3.5, "tremor3": False, "fails": True}}
#                       gains open and the bedside ratio: module values (SOURCES['romberg_unsourced'])
ROMBERG_GROW_S = 2.0           # [module] the bedside sign: the sway grows over this after the eye closure
ROMBERG_FAIL_S = 15.5          # [module] its nominal failure (the sway and lean full; the step follows the XcoM)
ROMBERG_EYES = ({"from_s": 0.0, "to_s": ROMBERG_CLOSE_S, "text": "EYES OPEN"},
                {"from_s": ROMBERG_CLOSE_S, "to_s": ROMBERG_S, "text": "EYES CLOSED"})


def romberg(kind, seed=SEED, rate=100.0):
    """(Body, truth) of feet together, eyes open then closed (module docstring). The posturographic kinds (healthy,
    cerebellar, sensory) scale each half to its own level, so that the realized ratio is the asked one (a 1 s
    change at the closure); the bedside sensory sign (sensory_bedside) grows from the open level to ratio times it
    over ROMBERG_GROW_S after the closure, then fails as a SARA 2 stance does (failing_stance: growth, lean, a
    catching step after the extrapolated centre of mass leaves the base), its ratio read over the steady closed
    part before that growth. A draw whose held part lets the extrapolated centre of mass leave the base is drawn
    again (STANCE_DRAWS). The eye state per time span is in params['frame_text'] (the body has no face)."""
    from .parkinsonism import _mj
    rm = ROMBERG[kind]
    fails = rm.get("fails", False)
    rng = as_rng(seed)
    T = int(round(ROMBERG_S * rate)) + 1
    t = np.arange(T) / rate
    ga, gm = STANCE_GAIN["together"]
    k = int(ROMBERG_CLOSE_S * rate)
    open_w = (1.0, ROMBERG_CLOSE_S - 0.5)
    closed_w = ((ROMBERG_CLOSE_S + ROMBERG_GROW_S, ROMBERG_FAIL_S - GROW_S) if fails else
                (ROMBERG_CLOSE_S + 1.0, ROMBERG_S - 0.5))
    kc = (int(closed_w[0] * rate), int(closed_w[1] * rate) + 1)
    level = rm["gain_open"] * math.hypot(ga, gm) * SWAY_RMS_M
    for draw in range(1, STANCE_DRAWS + 1):
        _, _, parts = sway_targets(T, rate, rm["gain_open"] * ga, rm["gain_open"] * gm, rm["tremor3"], rng)
        ap_low, ml_low, ap3 = parts["ap_low"], parts["ml_low"], parts["ap_3hz"]
        lvl = [level / math.hypot(ap_low[:k].std(), ml_low[:k].std()),
               rm["ratio"] * level / math.hypot(ap_low[kc[0]:kc[1]].std(), ml_low[kc[0]:kc[1]].std())]
        if fails:
            env = lvl[0] + (lvl[1] - lvl[0]) * _mj((t - ROMBERG_CLOSE_S) / ROMBERG_GROW_S)
            b, info, (ap, ml) = failing_stance("together", ap_low, ml_low, ap3, ROMBERG_FAIL_S, rate,
                                               t_from=closed_w[0], level=env)
            f = info["failure"]
            ok = (f["xcom_outside_before_search_s"] is None and f["xcom_exit_side"] == f["side"] and
                  f["lift_delay_s"] is not None and -STEP_REACT_S <= f["lift_delay_s"] <= STEP_REACT_MAX_S)
        else:
            env = lvl[0] + (lvl[1] - lvl[0]) * _mj((t - ROMBERG_CLOSE_S + 0.5) / 1.0)
            ap, ml = (ap_low + ap3) * env, ml_low * env
            b, info = stance_body("together", ap_low * env, ml, rate, ap3_m=ap3 * env if rm["tremor3"] else None)
            ok = bool(xcom_margin(b, "together")[1].max() < 0.0)
        if ok:
            break
    else:
        raise RuntimeError(f"Romberg {kind}: no draw of {STANCE_DRAWS} keeps the held stance inside the base")
    info["draws"] = draw
    val = validity(static_body("stand", ROMBERG_S), b, feet=(0,) if fails else (0, 1))
    val["continuity"] = _continuity(b)
    mo = measure_sway(b, open_w)
    mc = measure_sway(b, closed_w)
    p = {"kind": kind, "eyes": f"open 0 to {ROMBERG_CLOSE_S:g} s, closed from {ROMBERG_CLOSE_S:g} s",
         "frame_text": [dict(x) for x in ROMBERG_EYES],
         "gain_open": rm["gain_open"], "ratio_closed_over_open": rm["ratio"], "tremor3": rm["tremor3"],
         "bedside_sign": fails, "fail_at_s": info["failure"]["lift_s"] if fails else None,
         "windows_s": {"open": list(open_w), "closed": list(closed_w)}, "stance": info,
         "remeasured": {"open": mo, "closed": mc,
                        "ratio_rms": float(math.hypot(mc["ap_rms_cm"], mc["ml_rms_cm"]) /
                                           math.hypot(mo["ap_rms_cm"], mo["ml_rms_cm"]))}}
    srcs = ["sara", "quijoux2021", "baloh1998", "anagnostou2024", "helmchen2017", "romberg_unsourced"]
    srcs += (["mauritz1979"] if rm["tremor3"] else []) + (["hof2005", "winter1995"] if fails else ["winter1995"])
    return b, truth_record("romberg", seed, 0 if kind == "healthy" else None, "Romberg", p, srcs, val,
                           arrays={"ap_m": ap, "ml_m": ml})


# ---------------------------------------------------------------- SARA 3 sitting
SIT_S = 15.0
TRUNK = (3, 6, 9)
SIT_LADDER = {0: {"line_deg": 0.2, "bouts": None}, 1: {"line_deg": 1.0, "bouts": 2},
              2: {"line_deg": 2.5, "bouts": 0}}           # [module] the pelvis-to-neck line's RMS per axis (deg)
SIT_BOUT_S = 2.75              # [module] each intermittent bout (grade 1), equal lengths
SIT_BOUT_RAMP_S = 0.6          # [module] its raised-cosine ramps
SIT_BAND_HZ = (0.15, 2.0)      # [module]
SIT_PELVIS_SHARE = 0.3         # [module] share of each sway angle on the pelvis, rocking on the seat
SIT_BACK_CLEAR_M = 0.01        # [module] the trunk kept this far in front of the chair's backrest at every knot
SIT_SOLVE = 2                  # corrections of the sway amplitude by the measured line


@functools.lru_cache(maxsize=4)
def seat_pivot(posture):
    """(3,) m: the point the pelvis rocks about in a seated posture: below the midpoint of the hip joints, at the
    lowest skinned point of the buttocks and thighs (the seat contact), on the rest mesh of the posture."""
    from .viewer import dominant_joint, knot_vertices
    b = static_body(posture, 0.01)
    V0 = knot_vertices(b, [0])[0]
    J = b.joints().numpy()[0]
    seat = np.isin(dominant_joint("neutral"), (0, 1, 2))
    return np.array([0.5 * (J[1, 0] + J[2, 0]), float(V0[seat, 1].min()), 0.5 * (J[1, 2] + J[2, 2])])


def replant_legs(out, base):
    """out with both legs solved again (ik.hinge_leg) from its hips to base's ankles, the knee hinge along base's
    thigh x axis and each foot at base's global orientation: the feet stay where base has them, exactly."""
    from .ik import hinge_leg
    rest = rest_joints(out.betas, out.gender)
    Jo, Go = out.fk()
    Jb, Gb = base.fk()
    R = out.local_R().clone()
    for h, kn, a, _ in LEGS:
        G_th, kap = hinge_leg(Jo[:, h], Jb[:, a], Jo[:, h], rest[kn] - rest[h], rest[a] - rest[kn], Go[:, 0],
                              hinge=Gb[:, h][:, :, 0])
        Rk = axis_angle_to_matrix(torch.stack([kap, 0 * kap, 0 * kap], -1))
        R[:, h] = Go[:, 0].transpose(-1, -2) @ G_th
        R[:, kn] = Rk
        R[:, a] = (G_th @ Rk).transpose(-1, -2) @ Gb[:, a]
    return out.with_rotations(R, [j for leg in LEGS for j in leg[:3]])


def seated_sway(base, ap_rad, ml_rad, pelvis_share=SIT_PELVIS_SHARE, posture="sit_arms_forward"):
    """A seated body swaying (module docstring): the forward (ap_rad, about +x) and lateral (ml_rad, about +z) angles
    (T,), pelvis_share of each on the pelvis rocking about seat_pivot (a global rotation of the root about it), the
    rest spread equally over TRUNK; the legs replanted to base's feet (replant_legs)."""
    ap, ml = np.asarray(ap_rad, dtype=np.float64), np.asarray(ml_rad, dtype=np.float64)
    R = base.local_R()
    Rn = R.clone()
    D = (axis_rotation("x", torch.from_numpy(pelvis_share * ap)) @
         axis_rotation("z", torch.from_numpy(pelvis_share * ml)))
    Rn[:, 0] = D @ R[:, 0]
    s = (1.0 - pelvis_share) / len(TRUNK)
    for j in TRUNK:
        Rn[:, j] = compose(Rn[:, j], axis_rotation("x", torch.from_numpy(s * ap)), "child")
        Rn[:, j] = compose(Rn[:, j], axis_rotation("z", torch.from_numpy(s * ml)), "child")
    out = base.with_rotations(Rn, [0] + list(TRUNK))
    Q = torch.from_numpy(seat_pivot(posture))
    P0 = base.joints()[:, 0]
    tr = out.transl.clone()
    tr += (Q + torch.einsum("tij,tj->ti", D, P0 - Q)) - P0
    out = Body(out.rot6d, tr, out.contact, out.betas, out.gender, out.rate)
    out = replant_legs(out, base)
    return Body(out.rot6d, out.transl, contact_labels(out), out.betas, out.gender, out.rate)


def seat_forward(out, clearance=SIT_BACK_CLEAR_M, every=5):
    """(Body, info): out moved forward on the seat (a constant shift of the translation, feet included) so that every
    vertex above the seat top lies at least clearance in front of the chair's backrest (viewer.chair_mesh: its
    second box) at every knot sampled every `every` (the SARA protocol sits without support)."""
    from .viewer import chair_mesh, knot_vertices
    cv, _ = chair_mesh()
    back = cv[8:16]
    lo, hi = back.min(0), back.max(0)
    ks = list(range(0, out.T, every)) + [out.T - 1]
    zmin = math.inf
    for a in range(0, len(ks), 64):
        Vk = knot_vertices(out, ks[a:a + 64])
        sel = (Vk[..., 1] > lo[1] + 0.02) & (np.abs(Vk[..., 0]) < hi[0])
        zmin = min(zmin, float(np.where(sel, Vk[..., 2], np.inf).min()))
    dz = max(0.0, hi[2] + clearance - zmin)
    tr = out.transl.clone()
    tr[:, 2] += dz
    return (Body(out.rot6d, tr, out.contact, out.betas, out.gender, out.rate),
            {"seat_forward_m": dz, "backrest_front_z_m": float(hi[2]), "trunk_back_z_min_m": zmin + dz,
             "clearance_m": zmin + dz - float(hi[2])})


def sitting_grade(grade, seed=SEED, rate=100.0):
    """(Body, truth) of SARA 3 (module docstring): seated sway on 'sit_arms_forward' (seated_sway: SIT_PELVIS_SHARE
    on the pelvis rocking on the seat, the rest over TRUNK), its amplitude solved so that the pelvis-to-neck line
    sways SIT_LADDER's RMS per axis (grade 1 inside its two bouts of SIT_BOUT_S, the grade-0 level between them),
    then moved forward on the seat clear of the backrest (seat_forward); the feet stay on the floor."""
    from .common import envelope
    lad = SIT_LADDER[int(grade)]
    rng = as_rng(seed)
    base = static_body("sit_arms_forward", SIT_S)
    T = base.T
    t = base.times().numpy()
    ap = colored_noise(T, rate, 1.0, SIT_BAND_HZ, rng)
    ml = colored_noise(T, rate, 1.0, SIT_BAND_HZ, rng)
    bouts, e, mask = [], np.ones(T), np.ones(T, bool)
    if lad["bouts"]:
        s1 = 2.5 + 1.5 * rng.uniform()
        gap = 2.5 + 1.0 * rng.uniform()
        bouts = [(round(s1, 2), round(s1 + SIT_BOUT_S, 2)),
                 (round(s1 + SIT_BOUT_S + gap, 2), round(s1 + 2 * SIT_BOUT_S + gap, 2))]
        small = SIT_LADDER[0]["line_deg"] / lad["line_deg"]
        e = small + (1.0 - small) * envelope(t, bouts, SIT_BOUT_RAMP_S, SIT_S)
        h = SIT_BOUT_RAMP_S / 2
        mask = np.zeros(T, bool)
        for a, b_ in bouts:
            mask |= (t >= a + h) & (t <= b_ - h)
    k = math.radians(lad["line_deg"])
    for it in range(SIT_SOLVE + 1):
        out = seated_sway(base, k * e * ap, k * e * ml)
        m = measure_trunk_sway(base, out, mask=mask)
        if it < SIT_SOLVE:
            k *= lad["line_deg"] / math.hypot(m["ap_rms_in_deg"], m["ml_rms_in_deg"]) * math.sqrt(2.0)
    out, seat = seat_forward(out)
    base_f = Body(base.rot6d, base.transl + torch.tensor([0.0, 0.0, seat["seat_forward_m"]], dtype=torch.float64),
                  base.contact, base.betas, base.gender, base.rate)
    val = validity(base_f, out)
    p = {"grade": int(grade), "line_rms_deg_per_axis": lad["line_deg"], "bouts_s": bouts,
         "pelvis_share": SIT_PELVIS_SHARE, "amplitude_deg_per_axis": math.degrees(k), "seat": seat,
         "feet": "on the floor (SARA: on an examination bed, the feet free; no bed in the scene set)",
         "back": "free: the trunk kept clear of the backrest (seat_forward)",
         "remeasured": measure_trunk_sway(base, out, mask=mask)}
    return out, truth_record("ataxic_sitting", seed, int(grade), "SARA 3 sitting", p, ("sara", "module"), val,
                             arrays={"ap_rad": k * e * ap, "ml_rad": k * e * ml})


def measure_trunk_sway(base, out, threshold_deg=1.0, mask=None):
    """The pelvis-to-neck line of out against base's: its forward and lateral angles (deg), RMS per axis (over the
    clip, and over the knots of mask: '_in'), the share of time with the deviation above threshold_deg."""
    Jb, Jo = base.joints().numpy(), out.joints().numpy()

    def ang(J):
        v = J[:, 12] - J[:, 0]
        return np.degrees(np.arctan2(v[:, 2], v[:, 1])), np.degrees(np.arctan2(v[:, 0], v[:, 1]))
    ab, lb = ang(Jb)
    ao, lo = ang(Jo)
    d_ap, d_ml = ao - ab, lo - lb
    dev = np.hypot(d_ap, d_ml)
    m = np.ones(len(dev), bool) if mask is None else np.asarray(mask, bool)
    return {"ap_rms_deg": float(d_ap.std()), "ml_rms_deg": float(d_ml.std()),
            "ap_rms_in_deg": float(d_ap[m].std()), "ml_rms_in_deg": float(d_ml[m].std()),
            "share_above": float((dev > threshold_deg).mean()), "threshold_deg": threshold_deg,
            "max_deg": float(dev.max())}


# ---------------------------------------------------------------- titubation
TITUBATION_HZ = 3.0            # SOURCES['dankova2020'], ['hpo_titubation']
TITUB_S = 20.0
TITUB_CHAIN = ((3, 0.15), (6, 0.15), (9, 0.2), (12, 0.25), (15, 0.25))   # [module] shares of the oscillation
TITUB_HEAD = (12, 15)          # [module] these lag the trunk joints by TITUB_LAG_DEG
TITUB_LAG_DEG = 25.0
TITUB_AM = 0.25                # [module] amplitude modulation depth (peak), in TITUB_AM_BAND_HZ
TITUB_AM_BAND_HZ = (0.1, 0.3)
TITUB_FM_HZ = 0.2              # [module] peak frequency jitter, in TITUB_FM_BAND_HZ, zero mean
TITUB_FM_BAND_HZ = (0.05, 0.3)
FTM_BANDS_CM = ((0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, math.inf))   # SOURCES['ftm']
TITUB_MAX_SHARE = 0.8          # [module] the nose's largest cycle, peak to peak, at this share of the band's upper edge
#                                (review M13 of 2026-11-03: levelled at the head joint, the visible head read a band
#                                higher at its peaks; the principle of decision D4 of 2026-11-03, raters score the
#                                largest amplitude they see, applied here to the head)
TITUB_POINTS = ("head_joint", "head_centre", "nose")   # the read points: the skull-base joint (15) and the two head
#                                landmarks the skeleton page draws (skeleton.head_landmarks(): the head circle's centre
#                                and the nose, fixed in the head joint's frame)
BREATH_HZ = 0.25               # [module] seated quiet breathing, its rate
BREATH_JITTER = 0.08           # [module] peak relative jitter of the breathing rate
BREATH_CHEST_MM = 3.0          # [module] the chest's rise, peak to peak (sternum, relative to the pelvis)
BACKGROUND_SEED = SEED + 40    # every titubation grade shares one physiological background (its level 0)


def titubation_band_cm(grade):
    """(lo, hi) cm: the FTM band of a grade (1 to 4), the open band of grade 4 closed at its lower edge times the
    span of the band below (a factor of 2), so that every band has a place for its largest cycle."""
    g = int(grade)
    if not 1 <= g <= len(FTM_BANDS_CM):
        raise ValueError("FTM grades run 1 to 4")
    lo, hi = FTM_BANDS_CM[g - 1]
    if math.isinf(hi):
        plo, phi = FTM_BANDS_CM[g - 2]
        hi = lo * phi / plo
    return lo, hi


def titubation_max_cm(grade):
    """The largest cycle of the nose, peak to peak (cm), of an FTM grade (module docstring): TITUB_MAX_SHARE of the
    band's upper edge (titubation_band_cm); 0 for grade 0."""
    g = int(grade)
    if g == 0:
        return 0.0
    return TITUB_MAX_SHARE * titubation_band_cm(g)[1]


def _unit_noise(T, rate, band, rng):
    """(T,) zero-mean band noise scaled to a peak absolute value of 1."""
    x = colored_noise(T, rate, 1.0, band, rng, slope=0.0)
    return x / np.abs(x).max()


@functools.lru_cache(maxsize=1)
def _sternum():
    """Vertex ids of the sternum: the 20 most forward vertices of the upper trunk (dominant joint 9) on the rest
    mesh."""
    from . import hands as H
    from .viewer import dominant_joint
    _, v, _ = H._rest()
    ids = np.flatnonzero(dominant_joint("neutral") == 9)
    return ids[np.argsort(v[ids, 2])[-20:]]


def _breathing_apply(body, beta):
    """body with the thoracic spine (6, 9) extended by beta / 2 each (rad, (T,)) and the neck (12) flexed back by
    beta, so that the head keeps its orientation."""
    R = body.local_R()
    Rn = R.clone()
    for j, s in ((6, -0.5), (9, -0.5), (12, 1.0)):
        Rn[:, j] = compose(Rn[:, j], axis_rotation("x", torch.from_numpy(s * beta)), "child")
    return body.with_rotations(Rn, [6, 9, 12])


def chest_rise_mm(base, out):
    """Peak-to-peak (mm) of the sternum's height relative to the pelvis joint, out against base."""
    from . import hands as H
    ids = _sternum()
    d = []
    for b in (base, out):
        P = H.body_points(b, ids).mean(1)
        d.append(P[:, 1] - b.joints().numpy()[:, 0, 1])
    x = d[1] - d[0]
    return float(1000 * np.ptp(x))


@functools.lru_cache(maxsize=2)
def seated_background(seed=BACKGROUND_SEED, duration_s=TITUB_S, rate=100.0):
    """(Body, info): the physiological background of the titubation clips on 'sit': seated sway at SIT_LADDER[0]'s
    line level (seated_sway, amplitude as solved for SARA 3 grade 0) and quiet breathing at BREATH_HZ (rate jitter
    BREATH_JITTER), the chest rising BREATH_CHEST_MM peak to peak (solved)."""
    rng = as_rng(seed)
    base = static_body("sit", duration_s, rate)
    T = base.T
    ap = colored_noise(T, rate, 1.0, SIT_BAND_HZ, rng)
    ml = colored_noise(T, rate, 1.0, SIT_BAND_HZ, rng)
    k = math.radians(SIT_LADDER[0]["line_deg"])
    for it in range(SIT_SOLVE + 1):
        sway = seated_sway(base, k * ap, k * ml, posture="sit")
        m = measure_trunk_sway(base, sway)
        if it < SIT_SOLVE:
            k *= SIT_LADDER[0]["line_deg"] / math.hypot(m["ap_rms_deg"], m["ml_rms_deg"]) * math.sqrt(2.0)
    f = BREATH_HZ * (1.0 + BREATH_JITTER * _unit_noise(T, rate, (0.02, 0.1), rng))
    phase = 2 * np.pi * np.cumsum(f) / rate
    wave = 0.5 - 0.5 * np.cos(phase)
    a = math.radians(1.0)
    for it in range(2):
        out = _breathing_apply(sway, a * wave)
        a *= BREATH_CHEST_MM / chest_rise_mm(sway, out)
    out = _breathing_apply(sway, a * wave)
    info = {"sway_line_deg": SIT_LADDER[0]["line_deg"], "sway_amplitude_deg": math.degrees(k),
            "sway_remeasured": measure_trunk_sway(base, out), "breath_hz": BREATH_HZ, "breath_jitter": BREATH_JITTER,
            "breath_extension_deg_pp": math.degrees(a), "chest_rise_mm": chest_rise_mm(sway, out), "seed": seed}
    return out, info


def _titub_apply(base, theta_trunk, theta_head):
    R = base.local_R()
    Rn = R.clone()
    for j, s in TITUB_CHAIN:
        th = theta_head if j in TITUB_HEAD else theta_trunk
        Rn[:, j] = compose(Rn[:, j], axis_rotation("x", torch.from_numpy(s * th)), "child")
    return base.with_rotations(Rn, [j for j, _ in TITUB_CHAIN])


def titubation(grade, seed=SEED, rate=100.0, hz=TITUBATION_HZ):
    """(Body, truth) of head and trunk titubation (module docstring): on the shared seated background
    (seated_background, the level 0), the chain oscillating about x at hz with a frequency jitter of TITUB_FM_HZ and
    an amplitude modulation of TITUB_AM (both slow zero-mean noise), the neck and head lagging the trunk by
    TITUB_LAG_DEG, 0.5 s raised-cosine ramps, the amplitude solved (linear gain, then corrected by the measure) so
    that the nose's largest cycle moves titubation_max_cm(grade) peak to peak against the level 0
    (measure_titubation)."""
    base, bg = seated_background(rate=rate)
    g = int(grade)
    cm = titubation_max_cm(g)
    t = base.times().numpy()
    T = base.T
    rng = as_rng(seed)
    env = np.clip(np.minimum(t / 0.5, (TITUB_S - t) / 0.5), 0.0, 1.0)
    env = 0.5 - 0.5 * np.cos(np.pi * env)
    am = 1.0 + TITUB_AM * _unit_noise(T, rate, TITUB_AM_BAND_HZ, rng)
    fm = _unit_noise(T, rate, TITUB_FM_BAND_HZ, rng)
    f = hz + TITUB_FM_HZ * (fm - fm.mean()) / max(np.abs(fm - fm.mean()).max(), 1e-12)
    phase = 2 * np.pi * np.cumsum(f) / rate
    lag = math.radians(TITUB_LAG_DEG)
    if g == 0:
        out = base.copy()
        th_t = th_h = np.zeros(T)
        A = 0.0
    else:
        A = math.radians(1.0)
        for _ in range(3):
            th_t, th_h = A * env * am * np.sin(phase), A * env * am * np.sin(phase - lag)
            out = _titub_apply(base, th_t, th_h)
            A *= cm / measure_titubation(base, out)["max_cm"]
        th_t, th_h = A * env * am * np.sin(phase), A * env * am * np.sin(phase - lag)
        out = _titub_apply(base, th_t, th_h)
    val = validity(static_body("sit", TITUB_S, rate), out)
    p = {"grade": g, "hz": hz, "max_cm": cm, "max_share_of_band_top": TITUB_MAX_SHARE if g else None,
         "reading": ("the nose's largest cycle peak to peak at max_cm; the head centre's median cycle above the band's "
                     "lower edge (the visible head, from its typical cycle to its largest, inside the band)"),
         "amplitude_deg": math.degrees(A),
         "chain": [list(x) for x in TITUB_CHAIN], "head_lag_deg": TITUB_LAG_DEG, "am_depth": TITUB_AM,
         "fm_peak_hz": TITUB_FM_HZ, "band_cm": list(titubation_band_cm(g)) if g else None,
         "ftm_band_cm": list(FTM_BANDS_CM[g - 1]) if g else None, "background": bg,
         "remeasured": measure_titubation(base, out)}
    return out, truth_record("titubation", seed, g, "FTM-banded titubation of the head and trunk (read at the head)",
                             p, ("dankova2020", "hpo_titubation", "ftm", "elble2006", "module"), val,
                             arrays={"theta_trunk_rad": th_t, "theta_head_rad": th_h})


def head_points(body):
    """{name: (T, 3) m} of TITUB_POINTS relative to the pelvis joint: the head joint and the head landmarks of the
    skeleton page (skeleton.landmarks_from on the body's forward kinematics)."""
    from .skeleton import landmarks_from
    P, G = body.fk()
    P, G = P.numpy(), G.numpy()
    L = landmarks_from(P, G, ("head_centre", "nose"))
    pts = {"head_joint": P[:, 15], "head_centre": L[:, 0], "nose": L[:, 1]}
    return {n: pts[n] - P[:, 0] for n in TITUB_POINTS}


def _cycle_pp(s):
    """(C,) the peak-to-peak extent of each cycle of a zero-mean oscillation s (T,), a cycle running from one upward
    zero crossing to the next."""
    up = np.flatnonzero((s[:-1] < 0) & (s[1:] >= 0))
    return np.array([np.ptp(s[a:b + 1]) for a, b in zip(up[:-1], up[1:])])


def window_chords(d, rate, window_s=1.0):
    """(W,) the largest distance between two positions of a path d (T, 3) inside each of its consecutive windows of
    window_s (the 3D chord of each window)."""
    n = int(round(window_s * rate))
    return np.array([np.linalg.norm(d[a:a + n, None] - d[None, a:a + n], axis=-1).max()
                     for a in range(0, len(d) - n + 1, n)])


def titubation_readings(db, do, rate, band=(2.0, 6.0), steady_s=1.0):
    """{name: readings} of the head points' paths of a level 0 (db) and a clip (do) ({name: (T, 3) m}, head_points
    or the skeleton's points at any rate), over the steady part (steady_s from each end): the displacement of the
    clip against the level 0, its first principal axis, the dominant frequency in band, the peak-to-peak of a
    sinusoid of its band power (measure.band_amplitude_pp, cm), the median, largest and smallest cycle peak to peak
    (cm) and the median and largest 1 s chord (window_chords, cm)."""
    from .measure import band_amplitude_pp, dominant_frequency
    k = int(round(steady_s * rate))
    out = {}
    for name in db:
        d = (do[name] - db[name])[k:len(do[name]) - k]
        d = d - d.mean(0)
        if not np.abs(d).max() > 0:
            out[name] = {"hz": math.nan, "band_pp_cm": 0.0, "cycle_median_cm": 0.0, "cycle_max_cm": 0.0,
                         "cycle_min_cm": 0.0, "n_cycles": 0, "window_median_cm": 0.0, "window_max_cm": 0.0}
            continue
        _, _, vt = np.linalg.svd(d, full_matrices=False)
        s = d @ vt[0]
        f = dominant_frequency(s, rate, band)
        c = _cycle_pp(s)
        w = window_chords(d, rate)
        out[name] = {"hz": f, "band_pp_cm": 100.0 * band_amplitude_pp(s, rate, f, 0.5),
                     "cycle_median_cm": 100.0 * float(np.median(c)), "cycle_max_cm": 100.0 * float(c.max()),
                     "cycle_min_cm": 100.0 * float(c.min()), "n_cycles": int(len(c)),
                     "window_median_cm": 100.0 * float(np.median(w)), "window_max_cm": 100.0 * float(w.max()),
                     "axis": [float(x) for x in vt[0]]}
    return out


def measure_titubation(base, out, band=(2.0, 6.0)):
    """The titubation of out against its level 0 base at the head points (head_points, titubation_readings): the
    nose's dominant frequency in band ('hz'), its largest cycle peak to peak ('max_cm', the levelled reading), the
    head centre's median cycle ('typical_cm'), the head joint's band reading ('head_pp_cm', the reading before review
    M13), the shoulders' midpoint band reading ('shoulders_pp_cm', the trunk's share) and every point's readings
    ('points')."""
    pb, po = head_points(base), head_points(out)
    for p, b in ((pb, base), (po, out)):
        J = b.joints().numpy()
        p["shoulders"] = 0.5 * (J[:, 16] + J[:, 17]) - J[:, 0]
    r = titubation_readings(pb, po, out.rate, band)
    return {"hz": r["nose"]["hz"], "max_cm": r["nose"]["cycle_max_cm"],
            "typical_cm": r["head_centre"]["cycle_median_cm"], "head_pp_cm": r["head_joint"]["band_pp_cm"],
            "shoulders_pp_cm": r["shoulders"]["band_pp_cm"], "points": r}


# ---------------------------------------------------------------- SARA 7 fast alternating hand movements
DDK_S = 15.0
DDK_N = 10
DDK_PERIOD0_S = 0.7            # SOURCES['sara'] demonstration: about 10 cycles in 7 s
DDK_G4_CYCLES = 5              # [module]
DDK_GRADES = {                 # module values (SOURCES['ddk_unsourced'])
    0: {"period_s": 0.70, "cv_period": 0.03, "cv_amp": 0.03, "trough": 0.0, "pauses": 0, "pause_s": 0.0,
        "overshoot": 0.0},
    1: {"period_s": 0.75, "cv_period": 0.12, "cv_amp": 0.12, "trough": 0.05, "pauses": 0, "pause_s": 0.0,
        "overshoot": 0.0},
    2: {"period_s": 0.80, "cv_period": 0.25, "cv_amp": 0.25, "trough": 0.25, "pauses": 2, "pause_s": 0.4,
        "overshoot": 0.3},
    3: {"period_s": 1.05, "cv_period": 0.35, "cv_amp": 0.30, "trough": 0.30, "pauses": 3, "pause_s": 0.5,
        "overshoot": 0.3},
    4: {"period_s": 1.10, "cv_period": 0.40, "cv_amp": 0.35, "trough": 0.30, "pauses": 1, "pause_s": 0.5,
        "overshoot": 0.0},
}
DDK_LEVEL0_SHARE = 0.8         # bradykinesia.LEVEL0_SHARE: the level-0 amplitude, share of the room
DDK_PEAK_MAX_DEG = 170.0       # the task angle is read as an axis-angle component: kept under 180 deg
DDK_HOLD_TROUGH = 0.15         # [module] an interruption holds a trough under this share of A0 (near the rest)
DDK_HOLD_DRIFT_DEG = (2.0, 4.0)    # [module] the slow drift toward supination during a hold
DDK_OVERSHOOT_DEG = (5.0, 15.0)    # [module] a supination overshoot of this size on 'overshoot' of the cycles
DDK_CORRECT_S = (0.1, 0.15)        # [module] corrected back to the peak over this
DDK_HOLD_V_DEG_S = 15.0        # measure_alternation: a hold is a run of the task angle under this speed


def _scores(rng, m):
    """(m,) standard normal scores at the quantiles (k + 0.5) / m, standardized to mean 0 and SD 1 exactly, in an
    order drawn from rng: a draw whose realized coefficient of variation is the asked one."""
    from scipy.stats import norm
    if m < 2:
        return np.zeros(m)
    z = norm.ppf((np.arange(m) + 0.5) / m)
    z = (z - z.mean()) / z.std()
    return rng.permutation(z)


def alternation(grade, seed=SEED, side="left", start_s=0.5, rate=100.0):
    """(Body, truth) of SARA 7 (module docstring) on 'sit'. The extremes: rest 0, then per cycle a supination peak
    A0 (1 + cv_amp z) (z standard normal scores in a seeded order, _scores, so that the realized CV is cv_amp; at
    least 0.3 A0, at most 0.95 of the room and DDK_PEAK_MAX_DEG) and a return to a trough of A0 |N(0, trough)| (at
    most half the smaller of its two neighbouring peaks, so that every rise stays distinct; 0 for the last cycle);
    the peak-to-peak intervals are period_s (1 + cv_period z) (the
    realized CV cv_period), each split at its midpoint by the trough, the first rise and the last return half a
    period_s (so that the cycles last n x period_s). An interruption holds a trough brought under DDK_HOLD_TROUGH A0
    and drifts DDK_HOLD_DRIFT_DEG toward supination during it; on 'overshoot' of the cycles the peak overshoots by
    DDK_OVERSHOOT_DEG (drawn among the peaks with room for at least the smaller, capped by it) and is corrected
    back over DDK_CORRECT_S."""
    from .bradykinesia import TASKS, task_axis, task_rooms, task_sign
    task = "pronation_supination"
    g = int(grade)
    cfg = DDK_GRADES[g]
    rng = as_rng(seed)
    base = static_body("sit", DDK_S, rate)
    k0 = int(round(start_s * rate))
    rooms = task_rooms(base, task, side, k0)
    room = float(sum(rooms))
    split = [x / room for x in rooms]
    A0 = DDK_LEVEL0_SHARE * room
    n = DDK_G4_CYCLES if g == 4 else DDK_N
    P0 = cfg["period_s"]
    iv = P0 * np.maximum(1.0 + cfg["cv_period"] * _scores(rng, n - 1), 0.3)      # peak-to-peak intervals
    halves = np.r_[P0 / 2, np.repeat(iv / 2, 2), P0 / 2]
    cap = min(0.95 * room, DDK_PEAK_MAX_DEG)
    peaks = np.minimum(A0 * np.maximum(1.0 + cfg["cv_amp"] * _scores(rng, n), 0.3), cap)
    troughs = np.abs(rng.normal(0.0, cfg["trough"], size=n)) * A0
    troughs[-1] = 0.0
    troughs = np.minimum(troughs, 0.5 * np.minimum(peaks, np.r_[peaks[1:], peaks[-1]]))   # every rise distinct
    after = sorted(rng.choice(np.arange(1, n), size=cfg["pauses"], replace=False).tolist()) if cfg["pauses"] else []
    drift = {}
    for c in after:                                  # a hold near the thigh, drifting slowly
        troughs[c - 1] = min(troughs[c - 1], DDK_HOLD_TROUGH * A0 * rng.uniform(0.3, 1.0))
        drift[c] = float(rng.uniform(*DDK_HOLD_DRIFT_DEG))
    room_up = np.flatnonzero(cap - peaks >= DDK_OVERSHOOT_DEG[0])     # the peaks with room for an overshoot
    n_over = min(int(round(cfg["overshoot"] * n)), len(room_up))
    over = sorted(rng.choice(room_up, size=n_over, replace=False).tolist()) if n_over else []
    over_deg = {c: float(min(rng.uniform(*DDK_OVERSHOOT_DEG), cap - peaks[c])) for c in over}
    over_s = {c: float(rng.uniform(*DDK_CORRECT_S)) for c in over}
    ext, dur, hold = [0.0], [], {}
    for c in range(n):
        if c in over_deg:
            ext += [peaks[c] + over_deg[c], peaks[c]]
            dur += [halves[2 * c], over_s[c]]
        else:
            ext += [peaks[c]]
            dur += [halves[2 * c]]
        ext += [troughs[c]]
        dur += [halves[2 * c + 1]]
        if (c + 1) in drift:
            hold[len(ext) - 1] = drift[c + 1]        # hold after this trough, drifting
            ext += [troughs[c] + drift[c + 1]]
            dur += [cfg["pause_s"]]
    o = np.zeros(base.T)
    tc = start_s
    t = base.times().numpy()
    marks = []
    for i in range(1, len(ext)):
        a, b_, h = ext[i - 1], ext[i], dur[i - 1]
        ins = (t >= tc) & (t <= tc + h)
        u = (t[ins] - tc) / h
        o[ins] = a + (b_ - a) * 0.5 * (1.0 - np.cos(np.pi * u))
        marks.append({"extreme": i, "t_s": float(tc + h), "value_deg": float(b_)})
        tc += h
    if tc >= DDK_S:
        raise ValueError("the alternation does not end inside the clip")
    o[t > tc] = ext[-1]
    tk = TASKS[task]
    chain = tk["chain"][side]
    sgn = task_sign(task, side)
    ax = task_axis(task, side, base)
    R = base.local_R()
    Rall = R.clone()
    for j, f, sh in zip(chain, tk["frames"], split):
        Rall[:, j] = compose(R[:, j], axis_rotation(ax, torch.from_numpy(np.radians(sgn * o * sh))), f)
    out = base.with_rotations(Rall, list(chain))
    val = validity(base, out)
    total = tc - start_s
    real_peaks = np.array([peaks[c] + over_deg.get(c, 0.0) for c in range(n)])
    starts = np.r_[0.0, troughs[:-1]]                # a drift rises from the trough: the rise counts from it
    pk_t = []
    for i, m_ in enumerate(marks):
        if ext[i + 1] > ext[i] and (i + 2 >= len(ext) or ext[i + 2] < ext[i + 1]):
            pk_t.append(m_["t_s"])
    p = {"task": task, "side": side, "grade": g,
         "posture": "sit (the forearm on the chair arm; SARA: on the thigh)",
         "n_cycles": n, "period_s": cfg["period_s"], "cv_period": cfg["cv_period"], "cv_amp": cfg["cv_amp"],
         "trough_share": cfg["trough"], "pauses_after_cycles": after, "pause_s": cfg["pause_s"],
         "hold_drift_deg": [drift[c] for c in after], "overshoot_cycles": over,
         "overshoot_deg": [over_deg[c] for c in over], "correction_s": [over_s[c] for c in over],
         "amplitude0_deg": A0, "room_deg": room, "peaks_deg": [float(x) for x in real_peaks],
         "troughs_deg": [float(x) for x in troughs], "half_cycles_s": [float(x) for x in halves],
         "total_s": float(total), "under_10_s": bool(total < 10.0), "completes_10": n == DDK_N,
         "realized_cv_period": float(np.std(iv) / np.mean(iv)) if n > 2 else None,
         "realized_cv_amp": float(np.std(real_peaks - starts) / np.mean(real_peaks - starts)),
         "peak_times_s": pk_t,
         "realized_cv_peak_interval": float(np.std(np.diff(pk_t)) / np.mean(np.diff(pk_t))) if n > 2 else None,
         "start_s": start_s, "end_s": float(tc)}
    p["remeasured"] = measure_alternation(o, rate)
    from .bradykinesia import task_angle
    p["remeasured_body"] = measure_alternation(task_angle(out, task, side, 0), rate)
    return out, truth_record("dysdiadochokinesia", seed, g, "SARA 7 fast alternating hand movements", p,
                             ("sara", "table", "ddk_unsourced", "module"), val,
                             arrays={"offset_deg": o})


def measure_alternation(angle_deg, rate, rel=0.1, pause_s=0.2, hold_v=DDK_HOLD_V_DEG_S):
    """The alternation read back from the task angle (deg, bradykinesia.task_angle): the extremes of the signal;
    a cycle is a maximum whose rise from the preceding minimum exceeds rel of the largest rise; the time from the
    first knot that leaves rest to the last knot that moves; the CV of the intervals between successive cycle
    peaks and of the rises; the interruptions, runs of at least pause_s slower than hold_v deg/s inside the
    sequence."""
    x = np.asarray(angle_deg, dtype=np.float64)
    from .parkinsonism import _extremes
    mx, mn = _extremes(x)
    rises, pk = [], []
    for k in mx:
        prev = mn[mn < k]
        lo = x[prev[-1]] if len(prev) else x[:k].min()
        rises.append(x[k] - lo)
        pk.append(k)
    rises, pk = np.asarray(rises), np.asarray(pk)
    keep = rises > rel * rises.max()
    rises, pk = rises[keep], pk[keep]
    v = np.abs(np.gradient(x, 1.0 / rate))
    moving = np.flatnonzero(v > 1e-3)
    t0, t1 = moving[0] / rate, moving[-1] / rate
    still = v[moving[0]:moving[-1] + 1] <= hold_v
    m = np.diff(np.r_[False, still, False].astype(np.int8))
    runs = [(a, e) for a, e in zip(np.flatnonzero(m == 1), np.flatnonzero(m == -1)) if (e - a) / rate >= pause_s]
    iv = np.diff(pk) / rate
    return {"n_cycles": int(len(pk)), "time_s": float(t1 - t0),
            "cv_interval": float(iv.std() / iv.mean()) if len(iv) > 1 else math.nan,
            "cv_rise": float(rises.std() / rises.mean()) if len(rises) > 1 else math.nan,
            "interruptions": len(runs), "mean_rise_deg": float(rises.mean())}


# ---------------------------------------------------------------- SARA 8 heel-shin slide, supine on the couch
COUCH_TOP_M = 0.70             # [module] the couch top above the room floor (an examination couch; no height read)
COUCH_WIDTH_M = 0.65           # [module] its width, at least the body's width at rest plus 2 COUCH_SIDE_M
COUCH_SIDE_M = 0.03            # [module]
COUCH_PAD_M = 0.07             # [module] the padded top's thickness
COUCH_INSET_M = 0.10           # [module] the base set in from the top's edges
COUCH_HEAD_M = 0.08            # [module] the top beyond the head
COUCH_FOOT_M = 0.12            # [module] the top beyond the heels
COUCH_YIELD_M = 0.012          # [module] the padded top yields this much under the resting body: the SMPL+H pose blend
#                                shapes of the moving hip move resting vertices (the other buttock, a finger) down by
#                                about 1 cm (no soft-tissue contact model); the couch check allows this depth
SUPINE_TILT_DEG = 1.0          # [module] the pelvis's forward axis tipped toward the feet (the head end 1 deg up),
#                                so that the root's heading (its forward axis on the floor) is defined: +z, the feet
SUPINE_PARTS = {"buttocks": (0, 1, 2), "upper back": (6, 9, 13, 14), "head": (15,),
                "arms": tuple(range(16, 52)), "left leg": (4, 7, 10), "right leg": (5, 8, 11)}   # arms: the hands too
SUPINE_GAPS = {"buttocks": (0, 1, 2), "upper back": (6, 9, 13, 14), "head": (15,), "arms": tuple(range(16, 52)),
               "thighs": (1, 2), "calves": (4, 5), "heels": (7, 8, 10, 11)}
HS_LEGS = {"left": (1, 4, 7, 10), "right": (2, 5, 8, 11)}
HS_CYCLES = 3                  # SOURCES['sara8']: performed 3 times
HS_KNEE_BELOW_M = 0.04         # [module] the heel's target on the other knee: its front, this far below the joint line
HS_SLIDE_END = 0.80            # [module] the slide ends this share of the way from the knee joint to the ankle joint:
#                                on the lower shin, short of the ankle that SARA names by the truth's
#                                shin['short_of_ankle_m'] (NOT_RENDERED: 0.85 and 0.9 failed the legs check at the
#                                front of the ankle, where the shin meets the upright foot; tried again after the review
#                                of 2026-11-03: 0.85 and 0.875 pass the level 0 and grade 4 but fail the legs and heel
#                                checks at grades 1 to 3, whose wobble and excursions reach the ankle's front; 0.9 fails
#                                the level 0 as well)
HS_SHIN_BAND_M = (0.02, 0.015)     # the shin line: the most anterior vertex of bands every 2 cm, 1.5 cm half width
PELVIS_TURN_DEG = 10.0         # [module] the pelvis turns this much in the couch plane, about the other hip, while the
#                                heel is on the other leg (the moving hip drawn toward the feet); the lumbar and
#                                thoracic joints turn back by SPINE_BACK_TURN, the other hip keeps its leg still
SPINE_BACK_TURN = {3: 1.45, 6: -0.4, 9: -0.05}  # [module] shares of the pelvis turn by which joints 3, 6 and 9 turn
#                                back in the couch plane (summing to 1: the thorax keeps its orientation): a side bend
#                                at the lumbar joint and a smaller one back at the upper thoracic joint, so that the
#                                thorax keeps its place on the couch as well (the split that keeps the upper body's
#                                resting vertices nearest their place with joints 3 and 9 at least 1 deg inside the SP1
#                                table, searched at PELVIS_TURN_DEG; equal thirds slid the upper body on the couch;
#                                open-body port, wave 6: searched again on the anatomical table, where the lumbar share
#                                of 1.7 passed jarm2022's lateral bend of 16.7 deg at the lumbar joint;
#                                scripts/openbody_wave6_checks.py, ataxia)
HS_REST_SLIDE_M = 0.03         # [module] heel_shin_validity: the upper body's resting vertices (head, upper back, arms
#                                and hands on the couch top) slide at most this far on it: the turn about the other hip
#                                carries the trunk toward the feet, which the side bends cannot take back, a little more
#                                on the right side (the mesh and the rest are not mirror images); the slide is in the
#                                truth, validity['resting_slide']
HS_REST_ON_M = 0.005           # a vertex rests on the couch when it lies within this of its top
HS_THIGH_PRESS_M = 0.02        # [module] heel_shin_validity: the moving thigh and the other thigh may overlap this deep
#                                where they press together (no soft-tissue contact model: the skinned thighs overlap)
HS_GROIN_M = 0.15              # [module] away from the groin: farther than this from the root joint
HS_TURN_LEAD = 0.6             # [module] the pelvis has turned fully by this share of the approach
HS_LEAD_S = 1.0                # [module] lying still before the first cycle
HS_LAND_HOLD_S = 0.15          # [module] a short stop where a dysmetric heel lands, before the correction
HS_HOLD_KNEE_S = 0.3           # [module] on the knee before the slide
HS_HOLD_END_S = 0.2            # [module] at the end of the slide
HS_RETURN_S = 1.0              # [module] the heel lifted and laid back on the couch
HS_REST_S = 0.8                # [module] between the cycles
HS_TAIL_S = 1.0                # [module] after the last cycle of the slowest grade
HS_LIFT_M = 0.08               # [module] the heel's arc above its straight path, to the knee (a single movement)
HS_HOVER_M = 0.05              # [module] a decomposed approach stops this far above the landing point, then lowers
HS_RETURN_LIFT_M = 0.10        # [module] the arc of the return
HS_CORRECT_LIFT_M = 0.01       # [module] the heel lifts this much in its correction toward the knee
HS_C0_M = 0.025                # first guess of the heel centre's height above the shin line in contact (then solved)
HS_CONTACT_PASSES = 3          # corrections of that height by the skinned clearance
HS_CLEAR_M = 0.0005            # off the shin, the foot is raised where it would come closer than this to the leg
HS_SWIVEL_GRID_DEG = tuple(np.arange(-20.0, 40.0 + 1e-9, 2.5))   # the knee's turn out of the vertical plane
HS_SWIVEL_BAND_DEG = 0.5       # the swivel: the mean of the grid angles within this of the best joint-limit margin
HS_SWIVEL_SMOOTH_S = 0.3       # then a centred moving mean
HS_ERROR_DIR_DEG = 30.0        # [module] a dysmetric landing lies this far from the shin's axis (plus or minus 10 deg):
#                                past the knee (proximal and lateral) or short of it (distal and medial)
HS_ERROR_JITTER = 0.2          # [module] each landing's error drawn within this share of the grade's
HS_PAST_MAX_M = 0.05           # a landing past the knee only when the largest drawn error stays under this: past it,
#                                the moving hip reaches the SP1 table's edge (placement_margin_deg), so grades 3 and 4
#                                land short (NOT_RENDERED)
HS_EVENT_S = 0.35              # [module] an off-shin excursion's duration (a raised cosine)
HS_EVENT_LAT_M = 0.04          # [module] its lateral peak (the heel centre off the shin line)
HS_EVENT_LIFT_M = 0.012        # [module] and its lift
HS_UNABLE_LAT_M = 0.045        # [module] grade 4: the heel slides down beside the shin, this far medial of it
HS_UNABLE_LIFT_M = 0.015       # [module] and this far above the shin line, never on it
OFF_SHIN_M = 0.02              # [module] measure_heel_shin: off the shin when the heel centre is farther than this from
#                                the shin line (laterally, or lifted above the shin's surface)
OFF_MERGE_S = 0.2              # [module] two off-shin runs closer than this are one excursion
OFF_MIN_S = 0.05               # [module] shorter runs are not counted
HS_NEAR_M = 0.12               # [module] the heel is at the shin within this distance (the slide windows)
HS_SLIDE_TOL_M = 0.03          # [module] a slide runs from within this of the knee target to within this of its end
HS_LAND_V = 0.05               # [module] m/s: the landing is the heel's first stop at the shin (speed under this,
HS_LAND_ABOVE_M = 0.05         # its centre less than this above the shin line: not a stop in the air above it)
HS_STOP_V = 0.005              # [module] m/s: a slide's time runs from the heel's last stop on the knee to its first
#                                stop at the slide's end, a stop being its speed along the shin under this
HS_TOL_M = 0.002               # the leg and couch tolerance (the stance tests' leg tolerance)
HS_NEAR_LEG_M = 0.03           # a point is tested against the other leg when its nearest vertex is this close
HS_SLIDE_S_SARA = 1.0          # SOURCES['sara8']: each slide within about this
# grades (module values, SOURCES['hs_unsourced']): the approach's time (s), the slide's time (s; the level 0 at
# HS_SLIDE_S_SARA), the landing error past or short of the knee (m), the approach's phases (decomposition,
# reach.PHASE_PAUSE_S between them), the heel's lateral oscillation at tremors.HZ['intention'] (peak m, on the slide
# and on the approach), the off-shin excursions over the slides (None: every slide beside the shin); the ataxic
# approach times long enough that no movement of the approach (its oscillation aside: the primary movement, each
# phase of a decomposed one) is faster than the level 0's single arc (tested)
HS_GRADES = {
    0: {"approach_s": 1.0, "slide_s": HS_SLIDE_S_SARA, "error_m": 0.0, "phases": 1, "osc_m": 0.0, "osc_approach_m": 0.0,
        "events": 0},
    1: {"approach_s": 1.8, "slide_s": 1.0, "error_m": 0.015, "phases": 1, "osc_m": 0.005, "osc_approach_m": 0.006,
        "events": 0},
    2: {"approach_s": 2.2, "slide_s": 1.2, "error_m": 0.03, "phases": 2, "osc_m": 0.008, "osc_approach_m": 0.012,
        "events": 2},
    3: {"approach_s": 2.4, "slide_s": 1.4, "error_m": 0.05, "phases": 3, "osc_m": 0.010, "osc_approach_m": 0.018,
        "events": 5},
    4: {"approach_s": 2.6, "slide_s": 1.6, "error_m": 0.08, "phases": 3, "osc_m": 0.015, "osc_approach_m": 0.025,
        "events": None},
}
HS_TEXT = {0: "normal: on the knee, slid along the shin, 3 cycles",
           1: "slightly abnormal: the heel wobbles, contact to the shin kept",
           2: "clearly abnormal: off the shin up to 3 times in 3 cycles",
           3: "severely abnormal: off the shin 4 or more times in 3 cycles",
           4: "unable: no slide keeps the heel on the shin"}
SOURCES.update({
    "sara8": "SARA item 8, heel-shin slide (Schmitz-Huebsch et al. 2006, the scale form as commonly reproduced, "
             "paraphrased): supine on an examination bed without sight of the legs, one heel lifted, pointed to the "
             "opposite knee, slid down the shin to the ankle and laid back, 3 times, each slide within about 1 s; "
             "0 normal, 1 slightly abnormal with contact to the shin kept, 2 clearly abnormal and off the shin up to "
             "3 times in 3 cycles, 3 severely abnormal and off the shin 4 or more times, 4 unable to perform (a "
             "slide without contact to the shin in all three trials is rated 4); each side rated, the item score "
             "the mean of both sides",
    "buijink2012": TR.SOURCES["buijink2012"],     # the intention frequency of tremors.HZ
    "hs_unsourced": "no published heel path, landing error, slide time or off-shin excursion size of the heel-shin "
                    "test was found (2026-11-02): module values; the off-shin counts are the anchors (sara8)",
})


def _held(R, T, rate=100.0, y=0.0):
    """A body of T knots holding local rotations R (22, 3, 3): the pelvis at x = z = 0, the translation's height y."""
    rest = rest_joints()
    tr = torch.zeros(T, 3, dtype=torch.float64)
    tr[:, [0, 2]] = -rest[0, [0, 2]]
    tr[:, 1] = float(y)
    return Body(matrix_to_rot6d(R)[None].expand(T, 22, 6).clone(), tr, torch.zeros(T, 2, dtype=torch.bool), rate=rate)


def _part_low(R, joints, V=None):
    """The lowest height (m) of the vertices whose dominant joint is in joints, R held (translation height 0)."""
    from .viewer import dominant_joint, knot_vertices
    V = knot_vertices(_held(R, 2), [0])[0] if V is None else V
    return float(V[np.isin(dominant_joint("neutral"), joints), 1].min())


def _bisect_to(fn, target, lo, hi, iters=40):
    """x in [lo, hi] with fn(x) = target for a monotonic fn (ValueError when [lo, hi] does not bracket it)."""
    fa, fb = fn(lo) - target, fn(hi) - target
    if fa * fb > 0:
        raise ValueError(f"[{lo}, {hi}] does not bracket the target")
    a, b = lo, hi
    for _ in range(iters):
        m = 0.5 * (a + b)
        fm = fn(m) - target
        if (fm > 0) == (fa > 0):
            a, fa = m, fm
        else:
            b = m
    return 0.5 * (a + b)


def _offset(R, specs):
    """R (22, 3, 3) with each (joint, axis, deg) of specs composed in the parent frame."""
    Rn = R.clone()
    for j, ax, deg in specs:
        Rn[j] = compose(Rn[j], axis_rotation(ax, torch.tensor(math.radians(deg), dtype=torch.float64)), "parent")
    return Rn


@functools.lru_cache(maxsize=1)
def supine_posture():
    """(R (22, 3, 3), info) of the supine rest on the couch (module docstring, item 6): the 'stand' posture (arms
    hanging 10 deg from the trunk) with the root turned onto the back (SUPINE_TILT_DEG), then four solves on the
    skinned mesh, each bringing a part's lowest vertex to the buttocks' level (the couch top): the spine (3, 6, 9)
    extended by each joint's share of the table's extension (supine_extension_shares; open-body port, wave 6: the
    thoracic joints alone, extended equally, passed jarm2022's 7.5 deg each) until the upper back lies on it, the neck
    and head (12, 15) flexed or extended equally until
    the head does, both shoulders extended (16, 17) until the arms do, each hip extended until its leg does (the heel
    or the calf). info: the angles (deg) and each region's lowest vertex above the couch top (m)."""
    from .viewer import knot_vertices
    R = posture_rotations("stand")
    R[0] = axis_rotation("x", torch.tensor(math.radians(-90.0 + SUPINE_TILT_DEG), dtype=torch.float64))
    sh = supine_extension_shares()

    def spine(e):
        return [(j, "x", -e * w) for j, w in sh.items()]

    def to_level(offsets, part):               # the part's lowest vertex less the buttocks', both on the offset R:
        def fn(e):                             # the buttocks' own vertices move a little with the spine and the hips
            Rn = _offset(R, offsets(e))        # on the open body (their skinning weights), wave 6
            V = knot_vertices(_held(Rn, 2), [0])[0]
            return _part_low(Rn, SUPINE_PARTS[part], V) - _part_low(Rn, SUPINE_PARTS["buttocks"], V)
        return fn
    ext = _bisect_to(to_level(spine, "upper back"), 0.0, 0.0, 30.0)
    R = _offset(R, spine(ext))
    neck = _bisect_to(to_level(lambda e: [(12, "x", e), (15, "x", e)], "head"), 0.0, -20.0, 20.0)
    R = _offset(R, [(12, "x", neck), (15, "x", neck)])
    arms = _bisect_to(to_level(lambda e: [(16, "x", e), (17, "x", e)], "arms"), 0.0, 0.0, 30.0)
    R = _offset(R, [(16, "x", arms), (17, "x", arms)])
    hips = {}
    for side, hip, part in (("left", 1, "left leg"), ("right", 2, "right leg")):
        hips[side] = _bisect_to(to_level(lambda e, h=hip: [(h, "x", e)], part), 0.0, 0.0, 15.0)
        R = _offset(R, [(hip, "x", hips[side])])
    V = knot_vertices(_held(R, 2), [0])[0]
    low = float(V[:, 1].min())
    info = {"root_deg": -90.0 + SUPINE_TILT_DEG, "spine_extension_deg": ext,
            "spine_extension_shares": {str(j): w for j, w in sh.items()}, "neck_head_flexion_deg_each": neck,
            "shoulder_extension_deg": arms, "hip_extension_deg": hips,
            "lowest_above_couch_m": {k: _part_low(R, j, V) - low for k, j in SUPINE_GAPS.items()},
            "solved_on": "the skinned neutral mesh, each part's lowest vertex to the buttocks' level"}
    return R, info


def supine_extension_shares():
    """{joint: share} of the supine rest's spinal extension over spine1 to spine3: each joint's extension bound in the
    table over their sum (the 'stand' spine at 0; jarm2022's thoracolumbar extension by design D14's shares)."""
    from .body import joint_limits
    lim = joint_limits()
    room = {j: -float(lim.lo[j][0]) for j in (3, 6, 9)}
    tot = sum(room.values())
    return {j: r / tot for j, r in room.items()}


def couch_top_y():
    """The couch top's height (m): COUCH_TOP_M above the room floor, which lies where a standing neutral body's soles
    do (viewer.mesh_floor_y(0)), so that the bed clips share the other clips' floor."""
    from .viewer import mesh_floor_y
    return mesh_floor_y(0.0) + COUCH_TOP_M


@functools.lru_cache(maxsize=1)
def _supine_height():
    from .viewer import knot_vertices
    R, _ = supine_posture()
    return couch_top_y() - float(knot_vertices(_held(R, 2), [0])[0][:, 1].min())


def supine_body(duration_s, rate=100.0):
    """A body holding supine_posture for duration_s (round(duration_s rate) + 1 knots), its lowest vertex on the couch
    top (couch_top_y), the pelvis at x = z = 0, the head toward -z and the feet toward +z."""
    R, _ = supine_posture()
    b = _held(R, int(round(duration_s * rate)) + 1, rate, y=_supine_height())
    return Body(b.rot6d, b.transl, contact_labels(b), b.betas, b.gender, b.rate)


def bed_floor_y(body):
    """The room floor (m) under a supine clip: the body's lowest vertex at its first knot (it rests on the couch top)
    minus COUCH_TOP_M. The viewer's scene_floor reads the lowest vertex of every frame, which here is the couch top,
    so a bed clip is drawn on this floor instead (exam.scene_mesh 'exam_bed')."""
    from .viewer import knot_vertices
    return float(knot_vertices(body, [0])[0][:, 1].min()) - COUCH_TOP_M


def bed_view(body):
    """{'floor_y', 'heading_deg'}: how the viewer draws a supine clip: the room floor bed_floor_y(body) (not the
    lowest vertex of the frames) and the couch's heading 0 (the feet toward +z), not the root's heading at the middle
    knot (the pelvis turns while a heel is on the other leg, and a lying root's forward axis points at the ceiling)."""
    return {"floor_y": bed_floor_y(body), "heading_deg": 0.0}


def couch_boxes(body):
    """[(lo (3,), hi (3,))] of the couch under a supine body (module docstring): the padded top (COUCH_PAD_M thick,
    its top at the body's lowest vertex at knot 0, COUCH_WIDTH_M wide or the body's width plus 2 COUCH_SIDE_M, from
    COUCH_HEAD_M beyond the head to COUCH_FOOT_M beyond the heels) on a base set in by COUCH_INSET_M, down to
    bed_floor_y."""
    from .viewer import knot_vertices
    V = knot_vertices(body, [0])[0]
    top = float(V[:, 1].min())
    floor = top - COUCH_TOP_M
    xc = float(body.joints()[0, 0, 0])
    half = max(0.5 * COUCH_WIDTH_M, float(np.abs(V[:, 0] - xc).max()) + COUCH_SIDE_M)
    z0, z1 = float(V[:, 2].min()) - COUCH_HEAD_M, float(V[:, 2].max()) + COUCH_FOOT_M
    pad = (np.array([xc - half, top - COUCH_PAD_M, z0]), np.array([xc + half, top, z1]))
    base = (np.array([xc - half + COUCH_INSET_M, floor, z0 + COUCH_INSET_M]),
            np.array([xc + half - COUCH_INSET_M, top - COUCH_PAD_M, z1 - COUCH_INSET_M]))
    return [pad, base]


def scene_mesh(kind, body, floor_y):
    """(vertices, faces) of the 'exam_bed' scene: the couch of couch_boxes (boxes of viewer.box) under a supine body,
    standing on bed_floor_y(body) (floor_y, the viewer's floor plane, is not used: the viewer must draw this clip's
    floor at bed_floor_y)."""
    from .viewer import box
    if kind != "exam_bed":
        raise ValueError(f"scene {kind!r}")
    vs, fs, off = [], [], 0
    for lo, hi in couch_boxes(body):
        v, f = box(lo, hi)
        vs.append(v)
        fs.append(f + off)
        off += len(v)
    return np.concatenate(vs), np.concatenate(fs)


def _other(side):
    return "right" if side == "left" else "left"


def shin_frame(body, side):
    """The shin line of the leg the heel of `side` slides on, from knot 0 of body (that leg never moves): K the other
    knee joint, a the unit axis from it to the other ankle joint (L m apart), n the shank's anterior axis made normal
    to a, l the horizontal axis normal to both, toward the moving leg (the medial side); the most anterior vertex of
    the other leg in bands along a (HS_SHIN_BAND_M) from the knee joint to HS_SLIDE_END of L (a band that holds no
    vertex left out), fitted: lateral offset cu (linear in s) and height ch (quadratic in s). s_knee = HS_KNEE_BELOW_M, s_end = HS_SLIDE_END L."""
    from .viewer import dominant_joint, knot_vertices
    oh, ok, oa, _ = HS_LEGS[_other(side)]
    V = knot_vertices(body, [0])[0]
    J, G = body.fk()
    J, G = J[0].numpy(), G[0].numpy()
    K, A = J[ok], J[oa]
    L = float(np.linalg.norm(A - K))
    a = (A - K) / L
    n = G[ok][:, 2] - (G[ok][:, 2] @ a) * a
    n /= np.linalg.norm(n)
    l = np.cross(a, n)
    l /= np.linalg.norm(l)
    if (J[HS_LEGS[side][0]] - J[oh]) @ l < 0:
        l = -l
    ids = np.flatnonzero(np.isin(dominant_joint("neutral"), (oh, ok, oa)))
    P = V[ids]
    s = (P - K) @ a
    s_end = HS_SLIDE_END * L
    ss, us, hs = [], [], []
    for s0 in np.arange(0.0, s_end + 1e-9, HS_SHIN_BAND_M[0]):
        m = np.abs(s - s0) < HS_SHIN_BAND_M[1]
        if not m.any():              # the open body's leg loops lie up to 4 cm apart below the knee: no vertex in the
            continue                 # band (open-body port, wave 6); the fit reads the bands that hold one
        h = (P[m] - K) @ n
        k = int(np.argmax(h))
        ss.append(float(s0))
        us.append(float((P[m][k] - K) @ l))
        hs.append(float(h[k]))
    return {"K": K, "a": a, "n": n, "l": l, "L": L, "cu": np.polyfit(ss, us, 1), "ch": np.polyfit(ss, hs, 2),
            "s_knee": HS_KNEE_BELOW_M, "s_end": s_end, "bands": {"s": ss, "u": us, "h": hs}}


def shin_point(fr, s, u=0.0):
    """(..., 3) the shin line's point at s (m along a from the knee joint), u m off it along l (toward the medial
    side)."""
    s = np.asarray(s, dtype=np.float64)
    u = np.broadcast_to(np.asarray(u, dtype=np.float64), s.shape)
    return (fr["K"] + s[..., None] * fr["a"] + (np.polyval(fr["cu"], s) + u)[..., None] * fr["l"] +
            np.polyval(fr["ch"], s)[..., None] * fr["n"])


def leg_surface(body, side, knot=0):
    """The surface of the leg of `side` (vertices whose dominant joint is its hip, knee, ankle or foot) at a knot:
    {'ids', 'V', 'N' (unit vertex normals of the whole mesh), 'tree' (scipy cKDTree of V)}."""
    from scipy.spatial import cKDTree
    from . import raster as RS
    from .viewer import dominant_joint, faces as body_faces, knot_vertices
    faces = body_faces()
    V = knot_vertices(body, [knot])[0]
    N = RS.vertex_normals(V, faces)
    ids = np.flatnonzero(np.isin(dominant_joint("neutral"), HS_LEGS[side]))
    return {"ids": ids, "V": V[ids], "N": N[ids], "tree": cKDTree(V[ids])}


def leg_distance(P, surf, near=HS_NEAR_LEG_M):
    """(...,) m: points P (..., 3) against a leg_surface: the signed distance along the normal of the nearest vertex
    (negative inside) when that vertex lies within near, the distance to it otherwise."""
    P = np.asarray(P, dtype=np.float64)
    d, j = surf["tree"].query(P.reshape(-1, 3))
    s = ((P.reshape(-1, 3) - surf["V"][j]) * surf["N"][j]).sum(-1)
    return np.where(d < near, s, d).reshape(P.shape[:-1])


@functools.lru_cache(maxsize=2)
def heel_ids(side):
    """Vertex ids of the heel of `side`: the heel set of its sole cloud (contact.heel_toe)."""
    from .contact import heel_toe, sole_vertices
    return sole_vertices(2)[side][heel_toe(2)[side][0]]


@functools.lru_cache(maxsize=2)
def foot_ids(side):
    """Vertex ids of the foot of `side` (dominant joint its ankle or foot joint)."""
    from .viewer import dominant_joint
    return np.flatnonzero(np.isin(dominant_joint("neutral"), HS_LEGS[side][2:]))


@functools.lru_cache(maxsize=2)
def shank_foot_ids(side):
    from .viewer import dominant_joint
    return np.flatnonzero(np.isin(dominant_joint("neutral"), HS_LEGS[side][1:]))


def _rot_about(v, axis, ang):
    """(T, 3) vectors v rotated about unit axes (T, 3) by ang (T,) rad (Rodrigues)."""
    c, s = torch.cos(ang)[:, None], torch.sin(ang)[:, None]
    return v * c + torch.cross(axis, v, dim=-1) * s + axis * (axis * v).sum(-1, keepdim=True) * (1 - c)


def _leg_solve(H, X, Gp, alpha, side, q, hinge0):
    """(R_hip (T, 3, 3), R_knee (T, 3, 3)) local rotations putting the heel centre (q (3,) in the shank frame) at X
    (T, 3) from the hip at H (T, 3) under the pelvis Gp (T, 3, 3) (ik.hinge_leg): the knee hinge axis is the rest
    thigh's x axis hinge0 (T, 3) turned about the hip-to-heel line by the swivel alpha (T,) rad, positive turning the
    knee outward (away from the midline)."""
    from .ik import hinge_leg
    hip, knee = HS_LEGS[side][:2]
    rest = rest_joints()
    Ht, Xt = torch.as_tensor(H, dtype=torch.float64), torch.as_tensor(X, dtype=torch.float64)
    u = Xt - Ht
    u = u / u.norm(dim=-1, keepdim=True)
    sgn = 1.0 if side == "left" else -1.0
    hinge = _rot_about(torch.as_tensor(hinge0, dtype=torch.float64), u,
                       -sgn * torch.as_tensor(alpha, dtype=torch.float64))
    G_th, k = hinge_leg(Ht, Xt, Ht, rest[knee] - rest[hip], q, Gp, hinge=hinge)
    return Gp.transpose(-1, -2) @ G_th, axis_angle_to_matrix(torch.stack([k, 0 * k, 0 * k], -1))


def _leg_margin(R_hip, R_knee, side):
    """(T,) deg: the smallest distance of the moving hip's and knee's axis-angle components to the SP1 table."""
    from ..body.rotations import matrix_to_axis_angle
    from .body import joint_limits
    lim = joint_limits()
    out = None
    for j, Rj in zip(HS_LEGS[side][:2], (R_hip, R_knee)):
        c = torch.rad2deg(matrix_to_axis_angle(Rj)).numpy()
        m = np.minimum((c - lim.lo[j]).min(-1), (lim.hi[j] - c).min(-1))
        out = m if out is None else np.minimum(out, m)
    return out


def placement_margin_deg(s_m, u_m, side="left", lift_m=0.0):
    """The SP1 table margin (deg, negative outside) of the moving hip and knee with the heel centre resting on the
    other shin at s_m along it and u_m medial of the shin line (HS_C0_M above it, lift_m more), the pelvis turned
    by PELVIS_TURN_DEG, at the best swivel of HS_SWIVEL_GRID_DEG."""
    base = supine_body(0.01)
    fr = shin_frame(base, side)
    X = shin_point(fr, np.array([s_m]), np.array([u_m])) + (HS_C0_M + lift_m) * np.array([0.0, 1.0, 0.0])
    one = np.ones(1)
    R, tr, Hm, h0, q = _pose_frame(base, side, PELVIS_TURN_DEG * one)
    best = -math.inf
    for a in HS_SWIVEL_GRID_DEG:
        best = max(best, float(_leg_margin(*_leg_solve(Hm, X, R[:, 0], np.radians(a) * one, side, q, h0), side)[0]))
    return best


def _pose_frame(base, side, turn_deg):
    """The pelvis turned in the couch plane by turn_deg (T,) about the other hip joint (module docstring): (R (T, 22,
    3, 3) the base's rotations with the root, lumbar and thoracic joints and the other hip changed, transl (T, 3), the
    moving hip joint (T, 3), the rest thigh's x axis of the moving leg (T, 3), the heel centre in the moving shank
    frame q (3,))."""
    from . import hands as H
    from .base import yaw_matrix
    hip, knee = HS_LEGS[side][:2]
    oh = HS_LEGS[_other(side)][0]
    T = len(turn_deg)
    Jb, Gb = base.fk()
    Jb, Gb = Jb[0], Gb[0]
    sgn = 1.0 if side == "left" else -1.0
    phi = torch.as_tensor(-sgn * np.radians(np.asarray(turn_deg, dtype=np.float64)))
    Ry = yaw_matrix(phi)
    R = base.local_R()[:1].expand(T, 22, 3, 3).clone()
    R[:, 0] = Ry @ R[:, 0]
    for j, par in ((3, 0), (6, 3), (9, 6)):
        Gp = Gb[par]
        R[:, j] = (Gp.T @ yaw_matrix(-phi * SPINE_BACK_TURN[j]) @ Gp) @ R[:, j]
    R[:, oh] = (Gb[0].T @ yaw_matrix(-phi) @ Gb[0]) @ R[:, oh]
    pivot = Jb[oh]
    rest = rest_joints()
    transl = pivot + torch.einsum("tij,j->ti", Ry, Jb[0] - pivot) - rest[0]
    Hm = pivot + torch.einsum("tij,j->ti", Ry, Jb[hip] - pivot)
    hinge0 = torch.einsum("tij,j->ti", Ry, Gb[hip][:, 0])
    c = H.body_points(base, heel_ids(side), knots=[0])[0].mean(0)
    q = Gb[knee].T @ (torch.from_numpy(c) - Jb[knee])
    return R, transl, Hm, hinge0, q


def _hs_plan(g, rng):
    """The cycles of a heel-shin grade (module docstring): per cycle its times (s) and the landing error, the
    oscillation phases, the off-shin excursions (time, side, size)."""
    from .reach import PHASE_PAUSE_S, PRIMARY_SHARE
    cfg = HS_GRADES[g]
    err = cfg["error_m"]
    kinds = ["short"] * HS_CYCLES
    if err and (1.0 + HS_ERROR_JITTER) * err <= HS_PAST_MAX_M:   # past and short landings, at least one of each
        kinds = [str(x) for x in rng.permutation(["past", "short", "past" if rng.uniform() < 0.5 else "short"])]
    t = HS_LEAD_S
    cycles = []
    for c in range(HS_CYCLES):
        e = err * rng.uniform(1.0 - HS_ERROR_JITTER, 1.0 + HS_ERROR_JITTER) if err else 0.0
        primary = PRIMARY_SHARE * cfg["approach_s"] if err else cfg["approach_s"]
        cyc = {"cycle": c + 1, "lift_s": t, "phases": cfg["phases"], "pause_s": PHASE_PAUSE_S,
               "error_m": e, "error_dir_deg": HS_ERROR_DIR_DEG + (rng.uniform(-10.0, 10.0) if err else 0.0),
               "landing": kinds[c] if err else "on the knee"}
        t += primary + (cfg["phases"] - 1) * PHASE_PAUSE_S
        cyc["land_s"] = t
        if err:
            t += HS_LAND_HOLD_S
            cyc["correct_s"] = t
            t += cfg["approach_s"] - primary
        cyc["knee_s"] = t
        t += HS_HOLD_KNEE_S
        cyc["slide_s"] = t
        t += cfg["slide_s"]
        cyc["slide_end_s"] = t
        t += HS_HOLD_END_S
        cyc["return_s"] = t
        t += HS_RETURN_S
        cyc["rest_s"] = t
        t += HS_REST_S
        cyc["osc_phase"] = [float(x) for x in rng.uniform(0.0, 2 * np.pi, 2)]
        cyc["events"] = []
        cycles.append(cyc)
    n = cfg["events"]
    if n:                                          # off-shin excursions spread over the slides, at most 2 per slide
        per = np.zeros(HS_CYCLES, int)
        for k in rng.permutation(np.repeat(np.arange(HS_CYCLES), 2))[:n]:
            per[k] += 1
        for cyc, m in zip(cycles, per):
            d = cfg["slide_s"]
            slots = [0.3, 0.7] if m == 2 else [rng.uniform(0.3, 0.7)]
            for u in slots[:m]:
                side = "lateral" if rng.uniform() < 0.5 else "medial"
                cyc["events"].append({"t_s": cyc["slide_s"] + u * d, "side": side, "lat_m": HS_EVENT_LAT_M,
                                      "lift_m": HS_EVENT_LIFT_M, "dur_s": HS_EVENT_S})
    return cycles


@functools.lru_cache(maxsize=1)
def heel_shin_duration():
    """Seconds of every heel-shin clip: the slowest grade's last cycle plus HS_TAIL_S (the shared time base)."""
    return float(math.ceil(max(_hs_plan(g, as_rng(0))[-1]["rest_s"] for g in HS_GRADES) + HS_TAIL_S))


def _arc(u):
    """The lift profile of an arc over u in [0, 1]: sin(pi u) squared, peak 1 at the middle, zero velocity at both
    ends (the heel leaves and lands without a velocity jump)."""
    return np.sin(np.pi * np.clip(u, 0.0, 1.0)) ** 2


def _hs_path(fr, c_rest, cycles, g, delta, t, f_hz, rate=100.0, oscillation=True):
    """(X (T, 3) the heel centre's target, contact (T,) bool, envelope (T,) the pelvis turn's share) of the plan
    (module docstring); delta (T,) m the solved contact height above HS_C0_M; oscillation False leaves out the
    heel's lateral oscillation (the voluntary path alone, for the tests)."""
    from .parkinsonism import _mj
    from .reach import PHASE_TIME_EXP
    cfg = dict(HS_GRADES[g])
    if not oscillation:
        cfg.update(osc_m=0.0, osc_approach_m=0.0)
    up = np.array([0.0, 1.0, 0.0])
    T = len(t)
    X = np.repeat(np.asarray(c_rest, dtype=np.float64)[None], T, 0)
    contact = np.zeros(T, bool)
    env = np.zeros(T)
    unable = cfg["events"] is None
    l = fr["l"]

    def on(s, u, k, lift=0.0):
        return shin_point(fr, s, u) + np.asarray(HS_C0_M + delta[k] + lift)[..., None] * up

    def idx(a, b):
        return np.flatnonzero((t >= a - 1e-9) & (t < b - 1e-9))
    for cyc in cycles:
        e, th = cyc["error_m"], math.radians(cyc["error_dir_deg"])
        sg = -1.0 if cyc["landing"] == "past" else 1.0          # past: proximal and lateral; short: distal and medial
        s_land, u_land = fr["s_knee"] + sg * e * math.cos(th), sg * e * math.sin(th)
        k_land = min(int(round(cyc["land_s"] * rate)), T - 1)
        if unable:                                         # beside the knee, never on it
            s_knee, u_knee, lift_k = fr["s_knee"], HS_UNABLE_LAT_M, HS_UNABLE_LIFT_M
        else:
            s_knee, u_knee, lift_k = fr["s_knee"], 0.0, 0.0
        X_land = on(np.array(s_land), np.array(u_land), k_land, HS_UNABLE_LIFT_M if unable else 0.0)
        # the approach: one arc, or phases (lift, carry, lower) with pauses
        a0, a1 = cyc["lift_s"], cyc["land_s"]
        ph = cyc["phases"]
        if ph == 1:
            pts, lifts = [c_rest, X_land], [HS_LIFT_M]
        elif ph == 2:
            pts, lifts = [c_rest, X_land + HS_HOVER_M * up, X_land], [0.5 * HS_LIFT_M, 0.0]
        else:
            pts = [c_rest, c_rest + (X_land[1] - c_rest[1] + HS_HOVER_M) * up, X_land + HS_HOVER_M * up, X_land]
            lifts = [0.0, 0.0, 0.0]
        lens = np.array([np.linalg.norm(pts[i + 1] - pts[i]) for i in range(len(pts) - 1)])
        w = np.maximum(lens, 1e-3) ** PHASE_TIME_EXP
        moving = (a1 - a0) - (ph - 1) * cyc["pause_s"]
        durs = moving * w / w.sum()
        tc = a0
        for i in range(ph):
            ks = idx(tc, tc + durs[i])
            u = (t[ks] - tc) / durs[i]
            P0, P1 = pts[i], pts[i + 1]
            d = P1 - P0
            if math.hypot(d[0], d[2]) > 1e-3 and abs(d[1]) > 1e-3:
                wv, wh = _mj(u / 0.6), _mj((u - 0.15) / 0.85)    # the heel rises first, then travels
            else:                                                # a purely vertical or horizontal phase
                wv = wh = _mj(u)
            X[ks] = P0 + np.outer(wh, d * np.array([1.0, 0.0, 1.0])) + np.outer(wv, d * up) + \
                np.outer(lifts[i] * _arc(u), up)
            tc += durs[i]
            if i < ph - 1:
                kp = idx(tc, tc + cyc["pause_s"])
                X[kp] = P1
                tc += cyc["pause_s"]
        ka = idx(a0, a1)
        ua = (t[ka] - a0) / (a1 - a0)
        prof = ua ** 3 * (1.0 - ua) ** 2 / 0.03456            # rises toward the target (peak 1 at 0.6 of the
        #                                                       approach), 0 with zero slope at the lift and landing
        X[ka] += np.outer(cfg["osc_approach_m"] * prof * np.sin(2 * np.pi * f_hz * (t[ka] - a0) + cyc["osc_phase"][0]),
                          l)
        env[ka] = _mj(ua / HS_TURN_LEAD)
        k_knee = min(int(round(cyc["knee_s"] * rate)), T - 1)
        X_knee = on(np.array(s_knee), np.array(u_knee), k_knee, lift_k)
        if "correct_s" in cyc:                                 # a stop where it landed, then the correction
            kh = idx(a1, cyc["correct_s"])
            X[kh] = X_land
            contact[kh] = not unable
            env[kh] = 1.0
            kc = idx(cyc["correct_s"], cyc["knee_s"])
            u = (t[kc] - cyc["correct_s"]) / (cyc["knee_s"] - cyc["correct_s"])
            X[kc] = X_land + np.outer(_mj(u), X_knee - X_land) + np.outer(HS_CORRECT_LIFT_M * _arc(u), up)
            env[kc] = 1.0
        kk = idx(cyc["knee_s"], cyc["slide_s"])
        X[kk] = on(np.full(len(kk), s_knee), np.full(len(kk), u_knee), kk, lift_k)
        contact[kk] = not unable
        env[kk] = 1.0
        # the slide, its oscillation and its excursions
        ks = idx(cyc["slide_s"], cyc["return_s"])
        d = cfg["slide_s"]
        u = np.clip((t[ks] - cyc["slide_s"]) / d, 0.0, 1.0)
        s = s_knee + _mj(u) * (fr["s_end"] - s_knee)
        taper = np.clip(np.minimum((t[ks] - cyc["slide_s"]) / 0.15, (cyc["slide_end_s"] - t[ks]) / 0.15), 0.0, 1.0)
        osc = cfg["osc_m"] * (0.5 - 0.5 * np.cos(np.pi * taper)) * np.sin(
            2 * np.pi * f_hz * (t[ks] - cyc["slide_s"]) + cyc["osc_phase"][1])
        lat, lift = np.full(len(ks), u_knee) + osc, np.full(len(ks), lift_k)
        bump = np.zeros(len(ks))
        for ev in cyc["events"]:
            x = np.clip((t[ks] - ev["t_s"] + 0.5 * ev["dur_s"]) / ev["dur_s"], 0.0, 1.0)
            b_ = 0.5 - 0.5 * np.cos(2 * np.pi * x)
            sg = 1.0 if ev["side"] == "medial" else -1.0
            lat += sg * ev["lat_m"] * b_
            lift += ev["lift_m"] * b_
            bump = np.maximum(bump, b_)
        X[ks] = on(s, lat, ks, lift)
        contact[ks] = (bump < 1e-3) & (not unable)
        env[ks] = 1.0
        # the return: lifted, carried out to the side first, then laid back
        kr = idx(cyc["return_s"], cyc["rest_s"])
        u = (t[kr] - cyc["return_s"]) / HS_RETURN_S
        P0 = X[ks[-1]] if len(ks) else X_knee
        dlt = c_rest - P0
        side_part = (dlt @ l) * l
        X[kr] = P0 + np.outer(_mj(u / 0.6), side_part) + np.outer(_mj((u - 0.4) / 0.6), dlt - side_part) + \
            np.outer(HS_RETURN_LIFT_M * _arc(u), up)
        env[kr] = 1.0 - _mj(u)
    return X, contact, env


def _hs_assemble(base, side, X, env):
    """The body of a heel-shin clip from the heel centre's path X (T, 3) and the pelvis turn's share env (T,):
    _pose_frame, then the swivel of each knot (the mean of the HS_SWIVEL_GRID_DEG angles within HS_SWIVEL_BAND_DEG of
    the best joint-limit margin, smoothed over HS_SWIVEL_SMOOTH_S, times env: 0 at rest) and _leg_solve. Returns
    (Body, swivel (T,) deg)."""
    hip, knee = HS_LEGS[side][:2]
    R, transl, Hm, h0, q = _pose_frame(base, side, PELVIS_TURN_DEG * env)
    Gp = R[:, 0]
    M = np.stack([_leg_margin(*_leg_solve(Hm, X, Gp, np.full(len(X), math.radians(a)), side, q, h0), side)
                  for a in HS_SWIVEL_GRID_DEG])
    good = M >= M.max(0)[None] - HS_SWIVEL_BAND_DEG
    grid = np.asarray(HS_SWIVEL_GRID_DEG)[:, None]
    a_star = (good * grid).sum(0) / good.sum(0)
    n = 2 * int(round(0.5 * HS_SWIVEL_SMOOTH_S * base.rate)) + 1
    a_s = np.convolve(np.pad(a_star, (n // 2, n // 2), mode="edge"), np.ones(n) / n, mode="valid") * env
    R_hip, R_knee = _leg_solve(Hm, X, Gp, np.radians(a_s), side, q, h0)
    R[:, hip], R[:, knee] = R_hip, R_knee
    b = Body(matrix_to_rot6d(R), transl, torch.zeros(len(X), 2, dtype=torch.bool), rate=base.rate)
    return Body(b.rot6d, b.transl, contact_labels(b), b.betas, b.gender, b.rate), a_s


def _mesh_take(body, ids, knots=None, chunk=256):
    """(len(knots), M, 3) the vertices ids of a Body (hands at rest) at the knots (default every knot): the selected
    body model's skinning (hands.body_points; open-body port, wave 6: this module blended the SMPL+H arrays itself)."""
    from . import hands as H
    ids = np.asarray(ids, dtype=np.int64)
    ks = np.arange(body.T) if knots is None else np.asarray(knots, dtype=np.int64)
    return H.body_points(body, ids, knots=ks, chunk=chunk)


def _foot_clearance(body, side, surf, knots=None, far=HS_NEAR_LEG_M):
    """(T,) m: the lowest signed distance of the moving foot's vertices to the other leg's surface (leg_distance) at
    the knots (default every knot); far at the other knots."""
    out = np.full(body.T, far)
    ks = np.arange(body.T) if knots is None else np.asarray(knots, dtype=np.int64)
    if len(ks):
        out[ks] = leg_distance(_mesh_take(body, foot_ids(side), ks), surf).min(-1)
    return out


def heel_shin(grade, side="left", seed=SEED + 80, rate=100.0):
    """(Body, truth) of SARA 8 (module docstring, item 6): supine on the couch (supine_body), the heel of `side` lifted
    to the other knee, slid down the other shin and laid back, HS_CYCLES times, on the plan of HS_GRADES[grade]
    (_hs_plan, _hs_path); the pelvis turns while the heel is on the other leg (_pose_frame); the leg solved by the
    heel centre (_hs_assemble); the contact height solved on the skinned mesh so that the foot touches the shin
    where the plan says contact (HS_CONTACT_PASSES); then the read-back (measure_heel_shin) and the supine validity
    (heel_shin_validity)."""
    from .tremors import HZ
    g = int(grade)
    rng = as_rng(seed)
    dur = heel_shin_duration()
    base = supine_body(dur, rate)
    T = base.T
    t = base.times().numpy()
    fr = shin_frame(base, side)
    surf = leg_surface(base, _other(side))
    from . import hands as H
    c_rest = H.body_points(base, heel_ids(side), knots=[0])[0].mean(0)
    cycles = _hs_plan(g, rng)
    f_hz = HZ["intention"]
    delta = np.zeros(T)
    passes = []
    for it in range(HS_CONTACT_PASSES + 1):
        X, cmask, env = _hs_path(fr, c_rest, cycles, g, delta, t, f_hz, rate)
        out, swivel = _hs_assemble(base, side, X, env)
        gap = _foot_clearance(out, side, surf, np.flatnonzero((env > 0) | cmask))     # off the couch
        passes.append({"pass": it,
                       "contact_gap_mm_max": float(1000 * np.abs(gap[cmask]).max()) if cmask.any() else 0.0})
        if it < HS_CONTACT_PASSES:
            ks = np.flatnonzero(cmask)
            new = np.interp(np.arange(T), ks, delta[ks] - gap[ks]) if len(ks) else delta.copy()
            low = ~cmask & (gap < HS_CLEAR_M)               # off the shin: never into it
            new[low] = np.maximum(new[low], delta[low] + HS_CLEAR_M - gap[low])
            delta = new
    m = measure_heel_shin(out, side)
    val = heel_shin_validity(base, out, side)
    cfg = HS_GRADES[g]
    p = {"grade": g, "side": side, "on": f"the {side} heel on the {_other(side)} shin", "text": HS_TEXT[g],
         "cycles_planned": HS_CYCLES, "duration_s": dur, "plan": cycles,
         "approach_s": cfg["approach_s"], "slide_s": cfg["slide_s"], "slide_s_sara": HS_SLIDE_S_SARA,
         "landing_error_m": cfg["error_m"], "landings": [c["landing"] for c in cycles],
         "decomposition_phases": cfg["phases"], "intention_hz": f_hz, "oscillation_m": cfg["osc_m"],
         "oscillation_approach_m": cfg["osc_approach_m"],
         "off_shin_planned": sum(len(c["events"]) for c in cycles) if cfg["events"] is not None else None,
         "unable": cfg["events"] is None, "off_shin_threshold_m": OFF_SHIN_M,
         "knee_target_below_joint_m": HS_KNEE_BELOW_M, "slide_end_share": HS_SLIDE_END,
         "shin": {"knee_to_ankle_m": fr["L"], "s_knee_m": fr["s_knee"], "s_end_m": fr["s_end"],
                  "short_of_ankle_m": fr["L"] - fr["s_end"]},
         "pelvis_turn_deg": PELVIS_TURN_DEG, "swivel_deg": {"min": float(swivel.min()), "max": float(swivel.max())},
         "limit_margin_deg_min": float(_leg_margin(*(out.local_R()[:, j] for j in HS_LEGS[side][:2]), side).min()),
         "contact_passes": passes, "posture": supine_posture()[1],
         "couch": {"top_m": COUCH_TOP_M, "boxes": [[list(map(float, lo)), list(map(float, hi))]
                                                     for lo, hi in couch_boxes(base)]},
         "remeasured": m}
    srcs = ["sara8", "sara", "hs_unsourced", "module"] + (["buijink2012"] if (cfg["osc_m"] or cfg["osc_approach_m"])
                                                           else [])
    return out, truth_record("heel_shin", seed, g, "SARA 8 heel-shin slide", p, srcs, val,
                             arrays={"heel_target_m": X, "contact_planned": cmask, "pelvis_turn_share": env,
                                     "distance_m": m.pop("distance_m"), "s_m": m.pop("s_m")})


def measure_heel_shin(body, side="left", off_m=OFF_SHIN_M):
    """The heel-shin slide read back from the body (module docstring): the shin line of the other leg at knot 0
    (shin_frame); per knot the heel centre (heel_ids) projected on it, s along it and u its offset from the line
    (along l), the foot's clearance g above the other leg (_foot_clearance), the heel-to-shin-line distance D =
    hypot(u, max(g, 0)); the slides (runs of D under HS_NEAR_M along the shin, from the last knot within
    HS_SLIDE_TOL_M of the knee target to the first within it of the slide's end: its core_duration_s), cycles done =
    their count; each slide's time (duration_s) from the heel's last stop before that core to its first stop after
    it (the speed along the shin under HS_STOP_V: the whole slide, its slow start and end included); off-shin
    excursions = runs of D above off_m inside a slide (merged across gaps under OFF_MERGE_S, at least OFF_MIN_S
    long); the share of each slide on the shin; the landing of each cycle (the heel's first stop before its slide:
    speed under HS_LAND_V, its centre less than HS_LAND_ABOVE_M above the shin line) and its placement error,
    hypot(s - s_knee, u) there, past or short of the knee; the lateral RMS of u on each slide; the grade the counts
    give (4: no slide, or no slide with a knot on the shin; 3: 4 or more excursions; 2: 1 to 3; None: 0 or 1, the
    anchors' contact kept, told apart by the wobble and the landing)."""
    fr = shin_frame(body, side)
    surf = leg_surface(body, _other(side))
    hid, fid = heel_ids(side), foot_ids(side)
    P = _mesh_take(body, np.r_[hid, fid])
    c = P[:, :len(hid)].mean(1)
    s = (c - fr["K"]) @ fr["a"]
    u = ((c - shin_point(fr, s)) @ fr["l"])
    h = ((c - shin_point(fr, s)) @ fr["n"])
    gap = leg_distance(P[:, len(hid):], surf).min(-1)
    D = np.hypot(u, np.maximum(gap, 0.0))
    r = body.rate
    near = (D < HS_NEAR_M) & (s > fr["s_knee"] - HS_NEAR_M) & (s < fr["s_end"] + HS_NEAR_M)
    runs = []
    m = np.diff(np.r_[0, near.astype(np.int8), 0])
    for a, e in zip(np.flatnonzero(m == 1), np.flatnonzero(m == -1)):
        runs.append((int(a), int(e)))
    v = np.linalg.norm(np.gradient(c, 1.0 / r, axis=0), axis=1)
    vs = np.abs(np.gradient(s, 1.0 / r))
    slides, landings = [], []
    for a, e in runs:
        ss = s[a:e]
        done = np.flatnonzero(ss >= fr["s_end"] - HS_SLIDE_TOL_M)
        if not len(done):
            continue
        kb = a + int(done[0])
        start = np.flatnonzero(s[a:kb] <= fr["s_knee"] + HS_SLIDE_TOL_M)
        if not len(start):
            continue
        ka = a + int(start[-1])
        stop = np.flatnonzero((v[a:ka + 1] < HS_LAND_V) & (h[a:ka + 1] < HS_LAND_ABOVE_M))
        kl = a + int(stop[0]) if len(stop) else ka
        landings.append({"t_s": kl / r, "s_m": float(s[kl]), "u_m": float(u[kl]),
                         "error_m": float(math.hypot(s[kl] - fr["s_knee"], u[kl])),
                         "side_of_knee": "past" if s[kl] < fr["s_knee"] else "short"})
        off = D[ka:kb + 1] > off_m
        mm = np.diff(np.r_[0, off.astype(np.int8), 0])
        ev = [[int(x), int(y)] for x, y in zip(np.flatnonzero(mm == 1), np.flatnonzero(mm == -1))]
        merged = []
        for x, y in ev:
            if merged and (x - merged[-1][1]) / r < OFF_MERGE_S:
                merged[-1][1] = y
            else:
                merged.append([x, y])
        merged = [x for x in merged if (x[1] - x[0]) / r >= OFF_MIN_S]
        still0 = np.flatnonzero(vs[a:ka + 1] < HS_STOP_V)
        still1 = np.flatnonzero(vs[kb:e] < HS_STOP_V)
        k0 = a + int(still0[-1]) if len(still0) else ka
        k1 = kb + int(still1[0]) if len(still1) else kb
        slides.append({"from_s": ka / r, "to_s": kb / r, "core_duration_s": (kb - ka) / r,
                       "duration_s": (k1 - k0) / r, "start_s": k0 / r, "end_s": k1 / r, "off_shin": len(merged),
                       "off_shin_s": [[(ka + x) / r, (ka + y) / r] for x, y in merged],
                       "on_shin_share": float(1.0 - off.mean()), "lateral_rms_m": float(np.std(u[ka:kb + 1])),
                       "distance_max_m": float(D[ka:kb + 1].max())})
    n_off = sum(x["off_shin"] for x in slides)
    no_contact = bool(slides) and all(x["on_shin_share"] == 0.0 for x in slides)
    if not slides or no_contact:
        g = 4
    elif n_off >= 4:
        g = 3
    elif n_off >= 1:
        g = 2
    else:
        g = None                                         # 0 or 1: the anchors' contact is kept; see the wobble
    return {"cycles": len(slides), "slides": slides, "landings": landings, "off_shin": n_off,
            "no_contact_in_all_slides": no_contact, "grade_from_counts": g,
            "placement_error_m_max": max((x["error_m"] for x in landings), default=None),
            "lateral_rms_m_mean": float(np.mean([x["lateral_rms_m"] for x in slides])) if slides else None,
            "threshold_m": off_m, "distance_m": D, "s_m": s}


def heel_shin_validity(base, out, side):
    """The checks that fit a lying body (module docstring): joint limits against the supine base (body.limit_
    violations), continuity (_continuity), the couch (no vertex inside it: every vertex over the couch top's
    footprint at or above the top less the padded top's yield COUCH_YIELD_M, at every knot), the resting upper body
    (its vertices on the couch top at rest, within HS_REST_ON_M of it, slide at most HS_REST_SLIDE_M horizontally
    while they still rest on it; the pelvis's and the still leg's slides reported: the pelvis turns on purpose), the
    heel on the shin (the moving foot never deeper than HS_TOL_M below the other leg's surface), the legs apart (no
    vertex of the moving shank and foot deeper than HS_TOL_M inside the other leg, nor of the other leg inside them,
    by the nearest vertex's normal within HS_NEAR_LEG_M), the thighs (the moving thigh and the other thigh pressed
    together at most HS_THIGH_PRESS_M deep, both ways, away from the groin: farther than HS_GROIN_M from the root
    joint; the groin's depth and the rest's reported); 'contact' joins the couch, the resting upper body, the heel,
    the legs and the thighs for the renderer's checks. The standing checks (foot skate in stance, the soles' floor
    contact, the lowest foot joint) do not apply: 'not_applicable'."""
    from scipy.spatial import cKDTree
    from . import raster as RS
    from .body import limit_violations
    from .viewer import dominant_joint, faces as body_faces, knot_vertices
    lim = limit_violations(out.rot6d, base.rot6d if base.T == out.T else None)
    surf = leg_surface(base, _other(side))
    (lo, hi), _ = couch_boxes(base)
    top = float(hi[1])
    faces = body_faces()
    dom = dominant_joint("neutral")
    mv = shank_foot_ids(side)
    sub = np.isin(faces, mv).all(1)
    remap = -np.ones(len(dom), np.int64)
    remap[mv] = np.arange(len(mv))
    fsub = remap[faces[sub]]
    V0 = knot_vertices(base, [0])[0]
    on0 = V0[:, 1] - top < HS_REST_ON_M
    still = {"upper body": on0 & ~np.isin(dom, (0, 1, 2, 3) + HS_LEGS["left"][1:] + HS_LEGS["right"][1:]),
             "pelvis": on0 & np.isin(dom, (0, 1, 2, 3)), "still leg": on0 & np.isin(dom, HS_LEGS[_other(side)][1:])}
    still = {k: np.flatnonzero(m) for k, m in still.items()}
    slide = {k: 0.0 for k in still}
    th_m = np.flatnonzero(dom == HS_LEGS[side][0])
    th_o = np.flatnonzero(dom == HS_LEGS[_other(side)][0])
    root = out.joints().numpy()[:, 0]
    th_min, th_far, th_rest = math.inf, math.inf, math.inf
    couch_min, heel_min, legs_min, n_in = math.inf, math.inf, math.inf, 0
    fids = np.isin(mv, foot_ids(side))
    for a in range(0, out.T, 48):
        ks = list(range(a, min(a + 48, out.T)))
        V = knot_vertices(out, ks)
        inside = (V[..., 0] >= lo[0]) & (V[..., 0] <= hi[0]) & (V[..., 2] >= lo[2]) & (V[..., 2] <= hi[2])
        couch_min = min(couch_min, float(np.where(inside, V[..., 1] - top, np.inf).min()))
        for k, ids in still.items():                           # the resting parts' slide while they rest
            if len(ids):
                P = V[:, ids]
                dxz = np.linalg.norm(P[..., [0, 2]] - V0[ids][None][..., [0, 2]], axis=-1)
                slide[k] = max(slide[k], float(np.where(P[..., 1] - top < HS_REST_ON_M, dxz, 0.0).max()))
        P = V[:, mv]
        d = leg_distance(P, surf)
        heel_min = min(heel_min, float(d[:, fids].min()))
        legs_min = min(legs_min, float(d.min()))
        n_in += int((d < -HS_TOL_M).sum())
        for i in range(len(ks)):
            Nf = RS.vertex_normals(V[i], faces)                # the thighs, both ways
            for pa, pb in ((th_m, th_o), (th_o, th_m)):
                dd, jj = cKDTree(V[i][pb]).query(V[i][pa])
                near = dd < HS_NEAR_LEG_M
                if near.any():
                    sd = ((V[i][pa][near] - V[i][pb][jj[near]]) * Nf[pb][jj[near]]).sum(-1)
                    far = np.linalg.norm(V[i][pa][near] - root[ks[i]], axis=-1) > HS_GROIN_M
                    th_min = min(th_min, float(sd.min()))
                    if far.any():
                        th_far = min(th_far, float(sd[far].min()))
                    if ks[i] == 0:
                        th_rest = min(th_rest, float(sd.min()))
            dd, _ = surf["tree"].query(P[i], distance_upper_bound=HS_NEAR_LEG_M)
            if not np.isfinite(dd).any():                      # the other leg inside the moving shank and foot
                continue
            N = RS.vertex_normals(P[i], fsub)
            tr = cKDTree(P[i])
            d2, j2 = tr.query(surf["V"])
            near = d2 < HS_NEAR_LEG_M
            if near.any():
                sd = ((surf["V"][near] - P[i][j2[near]]) * N[j2[near]]).sum(-1)
                legs_min = min(legs_min, float(sd.min()))
                n_in += int((sd < -HS_TOL_M).sum())

    def mm(x):
        return 1000 * x if math.isfinite(x) else None
    couch = {"ok": bool(couch_min >= -COUCH_YIELD_M), "min_mm": 1000 * couch_min, "yield_mm": 1000 * COUCH_YIELD_M,
             "top_m": top, "rule": "every vertex over the couch top's footprint, every knot, at most the padded top's "
                                   "yield below it"}
    rest = {"ok": bool(slide["upper body"] <= HS_REST_SLIDE_M), "upper_body_mm": 1000 * slide["upper body"],
            "bound_mm": 1000 * HS_REST_SLIDE_M, "pelvis_mm": 1000 * slide["pelvis"],
            "still_leg_mm": 1000 * slide["still leg"], "n_vertices": {k: int(len(v)) for k, v in still.items()},
            "rule": "the vertices resting on the couch top at rest (within the rest distance of it): their largest "
                    "horizontal slide while they still rest on it, every knot; bounded for the upper body (head, upper "
                    "back, arms and hands), reported for the pelvis (it turns about the other hip on purpose) and the "
                    "still leg"}
    heel = {"ok": bool(heel_min >= -HS_TOL_M), "min_mm": 1000 * heel_min, "tol_mm": 1000 * HS_TOL_M,
            "rule": "the moving foot against the other leg's surface (nearest vertex normal), every knot"}
    legs = {"ok": bool(n_in == 0), "n_inside": n_in, "min_mm": 1000 * legs_min, "tol_mm": 1000 * HS_TOL_M,
            "rule": "the moving shank and foot against the other leg and back, every knot"}
    thighs = {"ok": bool(th_far >= -HS_THIGH_PRESS_M), "min_mm": mm(th_min), "beyond_groin_min_mm": mm(th_far),
              "rest_min_mm": mm(th_rest), "press_mm": 1000 * HS_THIGH_PRESS_M, "groin_m": HS_GROIN_M,
              "rule": "the moving thigh and the other thigh, both ways (nearest vertex normal within the near "
                      "distance), every knot: no deeper than the press allowance farther than the groin distance from "
                      "the root joint; the skinned mesh has no soft-tissue contact, so thighs pressed together overlap"}
    return {"limits_outside": lim["outside"], "limits_new": lim["new"], "limits_worst_deg": lim["worst_deg"],
            "continuity": _continuity(out), "couch": couch, "resting_slide": rest, "heel_on_shin": heel,
            "legs_apart": legs, "thighs": thighs,
            "contact": {"ok": couch["ok"] and rest["ok"] and heel["ok"] and legs["ok"] and thighs["ok"],
                        "rule": "supine: the couch, the resting upper body, the heel on the shin, the legs apart and "
                                "the thighs"},
            "not_applicable": {"foot skate in stance": "no foot bears weight: the body lies on the couch and the "
                                                       "moving heel slides on the shin on purpose (the resting parts' "
                                                       "slide on the couch is checked instead)",
                               "sole contact with the floor": "the soles face the foot of the couch",
                               "lowest foot joint": "the floor is the couch top here (the couch check)"}}


# ---------------------------------------------------------------- clip specifications
def _stance_specs():
    out = []
    titles = {"natural": "natural stance", "together": "feet together", "tandem": "tandem stance"}
    for (kind, g) in STANCE_LADDER:
        l0 = f"atax_SARA2_{kind}_g0"

        def build(kind=kind, g=g):
            return stance_grade(kind, g, seed=SEED + 10 * g + len(kind))
        out.append(ClipSpec(f"atax_SARA2_{kind}_g{g}", "ataxia", "SARA2", "SARA 2",
                            f"{titles[kind]}, eyes open", g,
                            f"SARA stance grade {g}" if g else "level 0 (SARA stance grade 0)",
                            stance_text(kind, g), ("front", "left"), None if g == 0 else l0, None, build,
                            since=SINCE, segment_title=f"stance, {titles[kind]}"))
    return out


def _romberg_specs():
    out = []
    l0 = "atax_ROMB_level0"
    post = "Romberg, posturographic ratio"
    eyes = f"eyes open 0 to {ROMBERG_CLOSE_S:g} s, eyes closed {ROMBERG_CLOSE_S:g} to {ROMBERG_S:g} s"
    healthy = ROMBERG["healthy"]["ratio"]

    def lvl0():
        return romberg("healthy", seed=SEED + 99)
    out.append(ClipSpec(l0, "ataxia", "ROMB", post,
                        "feet together, eyes open then closed (healthy, posturographic sway)", 0,
                        f"level 0 (healthy): posturographic Romberg ratio {healthy:g}",
                        f"{eyes}; normal sway, {healthy:g} times as large with the eyes closed",
                        ("front", "left"), None, None, lvl0, since=SINCE, segment_title="Romberg, feet together"))
    # review of 2026-11-03: the posturographic clips show no bedside Romberg sign; their texts say so (the
    # cerebellar clip's large sway and 3 Hz tremor are visible at the bedside: only the Romberg sign is absent)
    after = {"cerebellar": "as in healthy people: no Romberg sign (large sway with the eyes open too)",
             "sensory": "a platform finding, below the threshold of a bedside Romberg sign (see the bedside clip)"}
    absent = {"cerebellar": "no Romberg sign", "sensory": "no bedside Romberg sign"}
    for kind in ("cerebellar", "sensory"):
        def build(kind=kind):
            return romberg(kind, seed=SEED + (1 if kind == "cerebellar" else 2))
        rm = ROMBERG[kind]
        times = {2: "twice"}.get(rm["ratio"], f"{rm['ratio']:g} times")
        out.append(ClipSpec(f"atax_ROMB_{kind}", "ataxia", "ROMB", post,
                            f"{kind} ataxia, feet together, eyes open then closed (posturographic sway, "
                            f"{absent[kind]})", "level",
                            f"{kind} ataxia: posturographic Romberg ratio {rm['ratio']:g} (published means)",
                            f"{eyes}; the sway {times} as large with the eyes closed, {after[kind]}" +
                            (f"; {TREMOR3_TEXT}" if rm["tremor3"] else ""),
                            ("front", "left"), l0, None, build, since=SINCE, segment_title="Romberg, feet together"))
    rm = ROMBERG["sensory_bedside"]

    def build_bed():
        return romberg("sensory_bedside", seed=SEED + 3)
    out.append(ClipSpec("atax_ROMB_sensory_bedside", "ataxia", "ROMB", "Romberg sign (bedside)",
                        "sensory ataxia, feet together, eyes open then closed", "level",
                        f"sensory ataxia: Romberg sign, the sway {rm['ratio']:g} times as large with the eyes closed, "
                        f"then a step",
                        f"{eyes}; the sway grows over {ROMBERG_GROW_S:g} s after the eyes close, then a catching step",
                        ("front", "left"), l0, None,
                        build_bed, since=SINCE, segment_title="Romberg, feet together"))
    return out


def _sitting_specs():
    out = []
    for g in SIT_LADDER:
        def build(g=g):
            return sitting_grade(g, seed=SEED + 30 + g)
        lad = SIT_LADDER[g]
        txt = {0: f"no difficulty: trunk line {lad['line_deg']:g} deg RMS",
               1: f"intermittent sway: two bouts of {SIT_BOUT_S:g} s, trunk line {lad['line_deg']:g} deg RMS in them",
               2: f"constant sway: trunk line {lad['line_deg']:g} deg RMS"}[g]
        out.append(ClipSpec(f"atax_SARA3_sitting_g{g}", "ataxia", "SARA3", "SARA 3",
                            "sitting, arms outstretched, back free (feet on the floor; SARA: on a bed, feet free)", g,
                            f"SARA sitting grade {g}" if g else "level 0 (SARA sitting grade 0)", txt,
                            ("front", "left"), None if g == 0 else "atax_SARA3_sitting_g0", "chair", build,
                            since=SINCE, segment_title="sitting, arms outstretched"))
    return out


def _titubation_views():
    """('left', 'head_left') once viewer.VIEWS has the side close-up of the head (the oscillation is forward-back,
    along the line of sight of the frontal 'head' close-up); ('left', 'head') until then."""
    from .viewer import VIEWS
    return ("left", "head_left") if "head_left" in VIEWS else ("left", "head")


def _titubation_specs():
    out = []
    for g in range(5):
        def build(g=g):
            return titubation(g, seed=SEED + 40 + g)
        cm = titubation_max_cm(g)
        grade_text, params_text = "level 0", "no titubation: seated sway and breathing"
        if g:
            lo, hi = FTM_BANDS_CM[g - 1]
            band = (f"under {hi:g} cm" if lo == 0 else f"over {lo:g} cm" if math.isinf(hi) else
                    f"{lo:g} to {hi:g} cm")
            grade_text = f"titubation grade {g} (head amplitude {band}, Fahn-Tolosa-Marin bands)"
            # review of 2026-11-05: levelled on the nose's largest cycle along its axis; the page's 1 s 3D chord
            # reads 2 to 12 percent above it, so the caption says "about"
            params_text = (f"{TITUBATION_HZ:g} Hz forward and back, the head up to about {cm:.1f} cm peak to peak "
                           f"at the nose, irregular")
        out.append(ClipSpec(f"atax_TITUB_g{g}", "ataxia", "TITUB", "titubation", "head and trunk titubation, seated",
                            g, grade_text, params_text, _titubation_views(),
                            None if g == 0 else "atax_TITUB_g0", "chair", build, since=SINCE,
                            segment_title="seated rest, titubation"))
    return out


def _alternation_specs():
    out = []
    for g in DDK_GRADES:
        def build(g=g):
            return alternation(g, seed=SEED + 50 + g)
        cfg = DDK_GRADES[g]
        txt = (f"{DDK_G4_CYCLES} irregular cycles, then stops" if g == 4 else
               f"10 cycles, period {cfg['period_s']:g} s, rhythm CV {cfg['cv_period']:g}, amplitude "
               f"CV {cfg['cv_amp']:g}, {cfg['pauses']} interruptions" +
               (f", overshoots on {100 * cfg['overshoot']:.0f} percent of cycles" if cfg["overshoot"] else ""))
        out.append(ClipSpec(f"atax_SARA7_ddk_g{g}", "ataxia", "SARA7", "SARA 7",
                            "fast alternating hand movements, left (on the chair arm; SARA: on the thigh)", g,
                            f"SARA fast alternating movements grade {g}" if g else "level 0 (SARA 7 grade 0)", txt,
                            ("front", "hands_top", "hands"), None if g == 0 else "atax_SARA7_ddk_g0", "chair",
                            build, since=SINCE, segment_title="fast alternating hand movements"))
    return out


HS_VIEWS = ("left", "top", "right")           # the right side: the slid-on shin with the heel on it
HS_VIEWS_RIGHT = ("right", "top", "left")


def _heel_shin_text(g):
    """The caption's parameter text of a heel-shin grade, from HS_GRADES and the intention frequency."""
    from .tremors import HZ
    cfg = HS_GRADES[g]
    if g == 0:
        # review of 2026-11-03: the slide's end, short of the ankle (NOT_RENDERED), is stated without a number:
        # HS_SLIDE_END places the heel contact, while the drawn heel point of the skeleton page reaches less far
        # down the knee-to-ankle line than the heel contact
        return (f"{HS_CYCLES} cycles: heel on the knee, slid down the shin in {cfg['slide_s']:g} s, stopping short "
                f"of the ankle, laid back; pelvis turn {PELVIS_TURN_DEG:g} deg")
    where = "past or short of" if (1.0 + HS_ERROR_JITTER) * cfg["error_m"] <= HS_PAST_MAX_M else "short of"
    then = "then beside it" if cfg["events"] is None else "corrected"
    # review of 2026-11-05: every grade stops at HS_SLIDE_END, so every caption says so, as the level 0's does
    parts = [f"landing {100 * cfg['error_m']:.1f} cm {where} the knee, {then}",
             "each slide stopping short of the ankle"]
    if cfg["phases"] > 1:
        parts.append(f"approach in {cfg['phases']} phases")
    parts.append(f"heel wobble {100 * cfg['osc_m']:.1f} cm at {HZ['intention']:g} Hz")
    if cfg["events"] is None:
        parts.insert(0, f"every slide beside the shin, {100 * HS_UNABLE_LAT_M:.1f} cm off it")
    elif cfg["events"]:
        parts.insert(0, f"off the shin {cfg['events']} times in {HS_CYCLES} cycles")
    else:
        parts.append("contact to the shin kept")
    return "; ".join(parts)


def _heel_shin_specs():
    out = []
    l0 = "atax_SARA8_heelshin_g0"
    for g in HS_GRADES:
        def build(g=g):
            return heel_shin(g, seed=SEED + 80 + g)
        out.append(ClipSpec(f"atax_SARA8_heelshin_g{g}", "ataxia", "SARA8", "SARA 8",
                            "heel-shin slide, left heel on the right shin, supine on the couch", g,
                            f"SARA heel-shin grade {g}" if g else "level 0 (SARA heel-shin grade 0)",
                            _heel_shin_text(g), HS_VIEWS, None if g == 0 else l0, "exam_bed", build, since=SINCE,
                            segment_title="heel-shin slide, supine"))

    def build_right():
        return heel_shin(0, side="right", seed=SEED + 80)
    out.append(ClipSpec("atax_SARA8_heelshin_right_g0", "ataxia", "SARA8", "SARA 8",
                        "heel-shin slide, right heel on the left shin, supine on the couch", 0,
                        "level 0 (SARA heel-shin grade 0), right side", _heel_shin_text(0), HS_VIEWS_RIGHT, None,
                        "exam_bed", build_right, since=SINCE, segment_title="heel-shin slide, supine"))
    return out


NOT_RENDERED = {
    "SARA 2 grades 4 to 6": "need support (an arm, an examiner): no support in the scene",
    "SARA 3 grades 3 and 4": "need support: not shown",
    "SARA 3 examination bed": ("the feet rest on the floor of the chair scene: the 'exam_bed' couch exists (SARA 8) "
                               "but no seated posture on its edge with the feet free"),
    "SARA 7 on the thigh": ("the alternation runs with the forearm on the chair arm ('sit'), not on the thigh: from "
                            "the upright trunk the palm rests on the thigh only with the forearm sloping about 30 deg "
                            "and the wrist extended, which leaves at most about 80 deg of supination inside the "
                            "joint-limit table"),
    "SARA 8 large hypermetric landings": ("a heel landing past the knee (on the thigh and beyond the knee's far side) "
                                          "by the larger errors of grades 3 and 4 takes the moving hip out of the SP1 "
                                          "table (placement_margin_deg negative): those grades land short of the "
                                          "knee, grades 1 and 2 past and short"),
    "SARA 8 right side": ("the ladder is built for the left heel on the right shin; the right side has its level 0 "
                          "only (SARA rates both sides, the item score the mean of the two)"),
    "SARA 8 without sight of the legs": "the head lies on the couch looking up; no eyes on the body to show it",
    "SARA 8 slide to the ankle": ("the slide ends at HS_SLIDE_END of the way from the knee joint to the ankle joint, "
                                  "on the lower shin, short of the ankle joint by params['shin']['short_of_ankle_m']: "
                                  "later ends put the moving foot into the other leg (or the other leg into it) beyond "
                                  "the leg tolerance at the front of the ankle, where the shin meets the upright foot "
                                  "(heel_shin_validity, nearest-vertex normals; the exact distance of SH8's leg "
                                  "contact may allow them); tried again after the review of 2026-11-03 (the heel "
                                  "landmark of the skeleton page stops about three quarters of the way down): 0.85 and "
                                  "0.875 pass the level 0 and grade 4 only, the ataxic wobble and excursions of grades "
                                  "1 to 3 take the foot into the other leg at the ankle's front; a shared end for the "
                                  "whole ladder is kept"),
    "SARA 8 relaxed feet": ("the supine rest keeps the standing ankle angle: both feet point straight up; a "
                            "relaxed supine patient's feet fall into plantar flexion and outward rotation (not built: "
                            "the moving heel's contact and the slid-on shin are solved on this rest)"),
    "SARA 8 thighs pressed together": ("near the groin the moving thigh and the other thigh overlap on the skinned "
                                       "mesh (no soft-tissue contact model): bounded away from the groin by "
                                       "HS_THIGH_PRESS_M, the depths in validity['thighs']"),
    "stance conditions in one clip": "each clip shows one condition; the protocol's three are separate clips",
}


def specs():
    """[ClipSpec] of this module (module docstring); level-0 partners included."""
    s = (_stance_specs() + _romberg_specs() + _sitting_specs() + _titubation_specs() + _alternation_specs() +
         _heel_shin_specs())
    ids = [x.clip_id for x in s]
    if len(set(ids)) != len(ids):
        raise RuntimeError("duplicate clip ids")
    for x in s:
        if x.level0 is not None and x.level0 not in ids:
            raise RuntimeError(f"{x.clip_id}: level 0 {x.level0} missing")
    return s
