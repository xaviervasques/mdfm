"""The flowing involuntary-movement family of the synthetic-patient symptom catalogue (task CAT-INVOL): the
hyperkinesias that are not tremor, jerks or fixed postures, as motion on the program's body (22 joints and the 30
hand joints, the 100 Hz clock, the shared neutral shape; docs/proposals/SYNTH_PATIENT_DRAFT.md; the open body Anny
since the open-body port's wave 3, docs/OPEN_BODY_PORT_DESIGN.md section 14): chorea
(generalized and by region, piano-playing fingers, motor impersistence of the outstretched arms), hemiballism,
athetosis and choreoathetosis, pseudoathetosis, levodopa-induced dyskinesia (peak-dose choreic, seated and
standing; diphasic kicking and stepping), tardive dyskinesia (truncal rocking, pelvic thrusting, limb
choreoathetoid movements), adult stereotypies (body rocking, hand flapping, head nodding), akathisia (seated and
standing), restless legs while seated, mirror movements during unilateral finger tapping, and moving toes.
Not shown, and said so in the captions: the face, lips, jaw and tongue (the UHDRS face and bucco-oral-lingual
regions, AIMS items 1 to 4, the orobuccolingual part of tardive dyskinesia), the eyes (pseudoathetosis is shown with
the eyes closed stated, not drawn), individual toes (moving toes moves the toe block of the program's foot joint), pain.
The catalogue gap wave (lane hyper_gaps, the block at the end of the module) adds peak-dose dystonic dyskinesia, the
dystonia of Huntington's disease on its chorea, and tardive stereotypy (marching in place, rubbing the thighs).

The engine (shared by every operator).
- Channels (CHANNELS): a joint chain, an axis and an anatomical sign; the offset of a channel turns each joint of the
  chain about the axis fixed in its parent's frame (R = exp(w) R_base, w the axis-angle offset), split over a chain
  in proportion to each joint's range of motion on that axis. Forearm rotation turns the elbow about the forearm's
  long axis (the rest bone from elbow to wrist, carried into the parent frame by the base rotation). Range of motion
  (ROM) of a channel: the sum over its joints of hi - lo of the SP1 joint-limit table on the axis component
  (decision D3: the table as the denominator, as dystonia.py).
- Signals, three kinds:
  * events: a random sequence of discrete movement fragments on one to three channels of a region picked at random
    (the moving part switches from fragment to fragment), the next fragment starting before or just after the
    previous one ends, inside bouts of common.schedule, so that a region moves for its share of the clip ("an
    ongoing random-appearing sequence of discrete involuntary movements", Sanger et al. 2010). Two shapes. 'flow'
    (chorea, peak-dose and tardive dyskinesia): each fragment carries its channels from where they are to a new
    target posture (or back to rest) by a minimum-jerk transition, no posture held longer than dwell_max, every
    channel back to rest at the end of its bout (the visual check of 2026-11-02: fragments that each returned to
    rest drew a star of out-and-back strokes around the rest posture, not chorea's flow), and a slow drift
    (DRIFT_SHARE, 0.3 to 1 Hz) inside the bouts so that no posture between two movements is perfectly still; each
    transition of a joint is at least as long as its excursion asks at that joint's speed cap (SPEED_CAP_DEG_S: the
    review of 2026-11-03 read grades 3 and 4 as ballistic), inside its bout, and the composed body played slower
    where a segment still turns faster than a limb's bound (govern_rates, RATE_BOUND_DEG_S: the collar under the
    shoulder, the seated carriage and the guard add to the channels' own speed). 'pulse'
    (the seated foot excursions, moving toes): out by a minimum-jerk rise, an optional hold, back by a minimum-jerk
    return. Each fragment is weighed by the room the table leaves on its side (room_scale), so an excursion fills
    each direction in proportion to its room; a channel may be held to one side of its base (signs: a supported arm
    can only leave its support, the seated trunk bends forward only, a hanging arm moves away from the trunk). The
    flings of hemiballism and the lapses of motor impersistence are drawn by their own operators (fling_signals,
    impersistence);
  * band-limited noise (athetosis, pseudoathetosis drift): Gaussian noise filtered to a band by a raised-cosine
    window in the frequency domain, continuous inside its bouts (raised-cosine ramps at their edges);
  * patterns (stereotypies, rocking, akathisia, diphasic kicking, moving toes): a sinusoid of a stated rate with a
    small cycle-to-cycle jitter of rate and amplitude (STEREO_JITTER, a module value), the same movement repeated.
- Amplitude (DIS reading, below): for each region, one gain on its raw signals is solved by bisection so that the
  region's realized amplitude, the largest over its channels of the peak-to-peak excursion over the channel's ROM,
  equals the grade's level. Each channel is first bounded by a soft limit inside the room the table leaves from the
  base posture in each direction (body.room_deg, chain room the smallest over its joints of room over split; y =
  L tanh(x / L) on each side, L = SOFT_MARGIN times the room), then every joint is checked against the table at
  every knot and, where several channels on one joint leave it together, its offset is scaled down by a scale that
  never rises above the per-knot largest admissible one (a moving minimum then a moving mean, as base.walk), so that
  no new limit violation is introduced (body.validity). A level the table cannot reach is reported (reachable
  False), never forced.
- Seated base (base_body 'sit'): the AIMS position (Guy 1976), the palms lying on the thighs, the forearms sloping
  down to them (HANDS_ON_THIGHS, solve_hands_on_thighs; palm_rest_report reads the palm on the mesh), and
  the body SEAT_FORWARD_M forward of the static sit, clear of the backrest; the chair drawn by the viewer is the
  static sit's seat and backrest, without armrests (scene 'chair_armless', SH5), and the contact guard reads the
  same chair.
- Feet. Seated, the pelvis stays on the seat; a leg moves by inverse kinematics (ik.hinge_leg) with the hip fixed:
  the knee swings in and out about the hip-to-ankle line with the foot planted (knee swivel, outward weighted
  SWIVEL_IN_OUT; seated, it is mostly a rotation of the hip about the vertical, the hip_rot channel), or the foot is
  lifted vertically first, moved, and set down vertically (foot excursions): the horizontal travel has its own
  minimum-jerk excursion of at least TRAVEL_MIN_S out and back inside the time the foot is above AIR_M
  (airborne_travel), at most TRAVEL_RHO times the lift, so the foot neither slides on the floor nor jumps (the
  review of 2026-11-02 found a travel switched on by the foot's height within 0.05 s), the foot kept flat or pitched
  by a channel. Standing, both feet stay planted (ankle joints fixed, foot rotations fixed) while the
  pelvis sways, turns, tilts or dips; the pelvis height is the level 0's minus the dip, low-passed (PELVIS_SIGMA_S),
  kept under the highest height from which both legs reach their ankles at REACH of the leg length (smoothed below it
  over PELVIS_SMOOTH_S) by a smooth clamp (soft_clamp), the level-0 pelvis STAND_DIP_M lower (stand_soft): a pelvis
  riding at the reachable height snapped the knees and bounced on every step; a step in place shifts the weight over
  the stance foot first, raises the heel, lifts the foot and sets it down on its spot (step_track). Standing, a
  bowing trunk moves the pelvis back (balance_standing, BALANCE_M). A seated trunk rocking tilts the pelvis about the
  line through the two hip joints with the thighs' global rotations kept, so the legs do not move. Contacts are
  relabelled by the body-store rule when a foot moves; body.validity checks the skate over the contact knots of both
  feet.
- The seated knees kept apart (handoff 13: the tardive limb clips' knees came close, then passed through each
  other). The knees of the seated base lie a few centimetres apart, so an inward swivel of both closes the gap within
  a few degrees. A seated clip whose legs move by build's legs (knee swivels, foot excursions; a toe channel of the
  plan leaves the knees in place) goes through the knee guard (knee_guard, after its legs): the knots
  where the knees, ankles or feet came closer together than in the base less KNEE_GUARD_PREFILTER_M are measured on
  the skinned mesh (legcontact.gap_from_pose), and where a hard gap (a knee, shank or foot against the other leg)
  lies below KNEE_CLEARANCE_M plus KNEE_GUARD_MARGIN_M both knees swivel outward about their hip-to-ankle lines (the
  hips, ankles and feet where they were), by the swivel a one-degree finite difference asks, split by how far each
  knee came inward, held around the knot and smoothed by a Gaussian (KNEE_GUARD_SIGMA_S), measured again until the
  gap holds; a clip whose gaps hold keeps its body bit for bit, and a seated leg region's amplitude is read again
  after the guard. The validity entry 'knee_clearance' (knee_clearance) measures the final body the same way, 'ok'
  when its hard gap is at least KNEE_CLEARANCE_M. The tardive limb movements (td_limbs) swivel the knees both ways at
  the step's size, so their closing (the sum of the two inward swivels) is first kept under the ceiling of the seated
  posture (closing_ceiling: the largest closing whose hard gap stays KNEE_GUARD_MARGIN_M above the clearance on a
  probe of the posture, however it is split between the knees): the inward targets of the swivel fragments that
  would carry the closing past the ceiling are weighted down (td_inward_targets; a weighted target keeps its
  minimum-jerk transitions, so a knee comes inward less far and more slowly instead of stopping against a limit, and
  a knee going inward while the other is out keeps its fragment), clamp_closing keeps any residue under the ceiling
  plus CLOSING_BLEND_DEG and the knee guard any knot the movement brings closer, the outward swivels as asked; a step
  whose asked closing stays under the ceiling keeps its swivels as asked. The guard's swivel at a knot is at most
  KNEE_GUARD_MAX_DEG over its rounds and adds no joint-limit violation; a contact that an outward knee swivel does
  not open (the shanks or feet crossing with the knees apart) is reported in the guard's record and the validity
  entry (ok False), not forced.
- Arms. Seated, the hands rest on the thighs: while the trunk or a leg moves, each arm is re-solved so that the
  wrist stays where it lies on its thigh, carried by the thigh (support_arms: the elbow by its hinge in closed form,
  the upper arm aligned and turned toward where the trunk carries the elbow). Standing, the upper arms hang: they do
  not follow the trunk's turns rigidly (hang_arms, HANG_KEEP), except where the hand would come toward the midline
  (the arm on the convex side of a bend rests against the trunk). The arm channels move the arms from that
  carriage; seated, a moving hand lifts off its thigh (lift_events, SUPPORTED_LIFT; a writhing one, lift_noise,
  NOISE_LIFT). Contact guard (guard_arms): at GUARD_HZ knots, the arm and hand vertices are tested against the scene's
  boxes (the chair), against the body's trunk and thigh surface, and against the head and neck and the other forearm
  and hand (each kept HEAD_CLEAR_M, PAIR_CLEAR_M outside their skin); where they pass more than GUARD_TOL_M inside,
  the arm's own movement is scaled back toward its carriage, and where even the carriage touches, the shoulder is
  turned outward (the arm pushed aside by the body; seated, only an arm whose palm is off its thigh), all smoothed so
  that no new contact or jump appears; seated, where the guard leaves an arm inside, the hand is lifted off
  (lift_off), and the contact left on the final body is a validity entry that refuses the clip above GUARD_TOL_M
  (arm_contact).
- Fingers: per finger, flexion of the MCP, PIP and DIP joints (thumb CMC, MCP, IP; FINGER_SPLIT, the PIP leading)
  and abduction of the knuckles about the anatomical axes of hands.hand_frame, inside the hand limits of hands.py
  (hands.hand_limit_report). Finger events move neighbouring fingers together (FINGER_COUPLING), keep each long
  finger within FINGER_GAP_DEG of its neighbours, rise over 0.15 to 0.35 s, dwell at most 0.3 s (FINGER_TIMING) and
  reach at most FINGER_LEVEL_CAP of the flexion range: a single finger flexed 90 degrees beside straight ones is
  anatomically impossible (review of 2026-11-02). A hand resting on the thigh mostly lifts its fingers.

Grades (each clip's params_text prints the levels and their sources).
- The DIS reading (Monbaliu et al. 2012, as reproduced in the IAACD toolkit 2018; the choreoathetosis subscale):
  amplitude as a share of the ROM (bands under 10, 10 to under 50, 50 to under 90, 90 percent and more) and duration
  as a share of the observation time (the same bands). DIS_LEVELS: each band at its centre (5, 30, 70, 95 percent).
- UHDRS maximal chorea (Huntington Study Group 1996), per region 0 to 4: slight/intermittent, mild/common or
  moderate/intermittent, moderate/common, marked/prolonged. Declared map (QUAL, a module reading for the
  neurologist): amplitude terms slight 5, mild 20, moderate 40, marked 70 percent of the ROM (slight in DIS amplitude
  band 1, mild and moderate the two halves of band 2, marked in band 3); duration terms intermittent 30, common 70,
  prolonged 95 percent of the clip (DIS duration bands 2, 3, 4 at their centres). The face and bucco-oral-lingual
  regions are not shown; the neck moves with the trunk at the trunk's grade (the UHDRS has no neck region). The trunk
  and the neck reach at most TRUNK_NECK_CAP of their ROM, soft-limited at TRUNK_NECK_MARGIN of the room each way
  (a module reading: end-range bows and head throws read as a caricature).
- UDysRS part 3 (Goetz et al. 2008) for levodopa-induced dyskinesia, AIMS items 5 to 7 (Guy 1976) for tardive
  dyskinesia: the four qualitative scores (UDysRS slight, mild, moderate, severe; AIMS minimal, mild, moderate,
  severe) on the same declared ladder STEPS (amplitude 5, 20, 40, 70 percent of the ROM;
  share 30, 70, 70, 95 percent). The UDysRS rates during four tasks (communication, drinking, dressing, walking): the
  clips show seated rest and standing, the tasks are not drawn.
- Barnes Akathisia Rating Scale objective item (Barnes 1989), seated and standing: 1 the characteristic movements
  under half the observation time (30 percent), 2 at least half (70 percent), 3 constant (100 percent, standing:
  walking on the spot). The scale observes 2 minutes each; the clips last AKATHISIA_S.
- Woods and Teuber 1978 for mirror movements: 1 barely discernible, 2 slight (or stronger but briefer), 3 strong and
  sustained, 4 equal to the intended hand; declared amplitude ratios of the mirror to the voluntary pinch opening
  (MIRROR_RATIO) inside the published range of Hermsdoerfer et al. 1995 (1.4 to 78.4 percent).
- Declared ladders otherwise (stereotypies, hemiballism, pseudoathetosis, restless legs, moving toes, motor
  impersistence): stated per clip, with what is published and what is a module value.

Re-measurement (measure_regions, measure_fingers, remeasure): from the body against its level-0 base only, per
region: the share of time with movement (angular speed above V_MOVE_DEG_S, gaps under GAP_S closed, runs under
MIN_RUN_S dropped), the amplitude (the largest peak-to-peak channel excursion over its ROM, and its degrees), the
end-point excursion in cm (95th percentile of the end joint's displacement in the pelvis heading frame), the
dominant frequency of the region's offsets (measure.dominant_frequency, 0.1 to 8 Hz), the mean speed while moving,
and the count of distinct body parts involved (regions whose amplitude exceeds PART_SHARE of the ROM).
"""
import functools
import math
from dataclasses import replace

import numpy as np
import torch

from ..body.rotations import axis_angle_to_matrix, matrix_to_axis_angle, matrix_to_rot6d
from . import hands as H
from .base import REACH, static_body
from .body import (F64, SUPINATION_SIGN, Body, bone_axis, contact_labels, joint_limits, rest_joints,
                   room_deg, validity)
from .common import as_rng, schedule, truth_record
from .ik import hinge_leg

SINCE = "catalogue_v1"
SEED = 20261102
OPERATOR = "involuntary"
SOFT_MARGIN = 0.97          # [module] the soft limit's share of the room the table leaves in a direction
V_MOVE_DEG_S = 3.0          # [module] a region moves when one of its joints turns faster than this
GAP_S = 0.4                 # [module] stillness shorter than this inside movement is closed
MIN_RUN_S = 0.1             # [module] movement runs shorter than this are dropped
PART_SHARE = 0.02           # [module] a region counts as involved above this share of its ROM
BOUT_RAMP_S = 1.0           # [module] common.schedule ramp: bouts and gaps at least 1 s long
STEREO_JITTER = 0.05        # [module] cycle-to-cycle jitter (share) of a pattern's rate and amplitude
CHOREA_S = 20.0             # [module] chorea, dyskinesia, athetosis clips
AKATHISIA_S = 30.0          # [module] akathisia clips (the BARS observes 2 minutes seated and 2 standing)
RLS_S = 60.0                # [module] restless legs clip

SOURCES = {
    "uhdrs": "Huntington Study Group 1996, Unified Huntington's Disease Rating Scale: reliability and consistency, "
             "Mov Disord 11: 136-142, doi 10.1002/mds.870110204 (anchor wording through secondary sources: 2 mild/"
             "common or moderate/intermittent, 3 moderate/common, 4 marked/prolonged; 1 slight/intermittent not "
             "confirmed in a retrieved text); maximal chorea per region: face, bucco-oral-lingual, trunk, each arm, "
             "each leg",
    "dis": "Dyskinesia Impairment Scale (Monbaliu et al. 2012, doi 10.1111/j.1469-8749.2011.04209.x) as reproduced in "
           "the IAACD toolkit 2018: choreoathetosis amplitude as a share of the range of motion and duration as a "
           "share of time, bands under 10, 10 to under 50, 50 to under 90, 90 percent and more",
    "sanger2010": "Sanger TD et al. 2010, Definition and classification of hyperkinetic movements in childhood, Mov "
                  "Disord 25: 1538-1549, doi 10.1002/mds.23088 (definitions of chorea, athetosis, ballism, "
                  "stereotypies)",
    "hallett1981": "Hallett M, Kaufman C 1981, JNNP 44: 829, doi 10.1136/jnnp.44.9.829 (abstract): choreic EMG bursts "
                   "lasting more than 100 ms, asynchronous in antagonists (one case of Sydenham chorea)",
    "dinesh2020": "Adams/Dinesh et al. 2020, J Huntingtons Dis, doi 10.3233/JHD-190375 (full text): truncal chorea "
                  "index on a 1 to 5 Hz band (a design choice of the index, not a measured peak)",
    "manson2000": "Manson AJ et al. 2000, JNNP 68: 196-201, doi 10.1136/jnnp.68.2.196 (full text): levodopa-induced "
                  "dyskinesia, shoulder accelerometry, the 1 to 3 Hz band tracks the AIMS and Goetz scores",
    "hoff2001": "Hoff JI et al. 2001, Mov Disord 16: 58-61 (abstract): dyskinesia analysed in the 1 to 4 Hz band",
    "udysrs": "Goetz CG, Nutt JG, Stebbins GT 2008, The Unified Dyskinesia Rating Scale, Mov Disord 23: 2398-2403, "
              "doi 10.1002/mds.22341: part 3 impairment per region 0 to 4; the anchors 0 normal, 1 slight, 2 mild, 3 "
              "moderate, 4 severe as Pahwa R et al. 2022 report them (Clinically important change on the Unified "
              "Dyskinesia Rating Scale among patients with Parkinson's disease experiencing dyskinesia, Front Neurol, "
              "full text PMC9632663), as the catalogue's row L01",
    "luquin1992": "Luquin MR et al. 1992, Mov Disord 7: 117-124, doi 10.1002/mds.870070204 (abstract): repetitive "
                  "stereotyped lower-limb movements correspond to diphasic dyskinesia; no rate given",
    "aims": "Abnormal Involuntary Movement Scale, Guy W 1976, ECDEU Assessment Manual, DHEW, Rockville (scale form): "
            "items 5 upper extremities, 6 lower extremities (lateral knee movement, foot tapping, heel dropping, foot "
            "squirming), 7 neck, shoulders, hips (rocking, twisting, squirming, pelvic gyrations); 0 none, 1 minimal, "
            "2 mild, 3 moderate, 4 severe; do not include tremor",
    "wirshing1991": "Wirshing WC et al. 1991, J Neuropsychiatry Clin Neurosci 3: 10-17, doi 10.1176/jnp.3.1.10 "
                    "(abstract): tardive dyskinesia hand and foot movements with more power at 1 to 2 Hz",
    "janno2008": "Janno S et al. 2008, BMC Neurol 8: 10, doi 10.1186/1471-2377-8-10 (full text): seated ankle "
                 "actometry, dominant frequency median 0.58 Hz in akathisia (IQR 0.40 to 1.01), 0.55 Hz in tardive "
                 "dyskinesia",
    "neilson1974": "Neilson PD 1974, JNNP 37: 171-177, doi 10.1136/jnnp.37.2.171 (abstract): athetoid cerebral palsy, "
                   "a rhythmic component at 0.3 to 0.6 Hz",
    "tai2016": "Tai et al. 2016, Mov Disord Clin Pract, doi 10.1002/mdc3.12356 (abstract): hemiballism EMG bursts of "
               "200 to 1500 ms",
    "yahata1997": "Yahata et al. 1997, Jpn J Rehabil Med 34: 283-286 (abstract): ballistic movement at 1 to 1.5 Hz on "
                  "video, one case after a subthalamic infarct",
    "postuma2003": "Postuma RB, Lang AE 2003, Hemiballism: revisiting a classic disorder, Lancet Neurol 2: 661-668, "
                   "doi 10.1016/S1474-4422(03)00554-4 (abstract): unilateral flinging of the limbs",
    "sharp1994": "Sharp FR et al. 1994, Arch Neurol 51: 1103-1109, doi 10.1001/archneur.1994.00540230041010 "
                 "(abstract): pseudoathetoid movements restricted to the parts with proprioceptive loss",
    "goodwin_rock": "Goodwin and Albinali accelerometer datasets as reanalysed in bioRxiv 2025, doi "
                    "10.1101/2025.09.03.674008 (full text): mean bout durations rocking 11.78 s, hand flapping "
                    "5.26 s (6 participants with autism aged 12 to 20, not adults)",
    "newell2005": "Newell KM, Hong SL, Bodfish JW 2005, ISB abstract (full text): adults with profound intellectual "
                  "disability rocking on a force platform, modal frequency shown on a 0.2 to 0.9 Hz axis",
    "barnes1989": "Barnes TRE 1989, A rating scale for drug-induced akathisia, Br J Psychiatry 154: 672-676, doi "
                  "10.1192/bjp.154.5.672 (scale form): objective item 0 to 3, seated and standing 2 minutes each",
    "braude1984": "Braude WM, Charles IP, Barnes TRE 1984, Psychopharmacology 82: 95-101, doi 10.1007/BF00426389 "
                  "(abstract): akathisia, large-amplitude rhythmic foot movements below 4 Hz",
    "zucconi2006": "Zucconi M et al. 2006, Sleep Med 7: 175-183 (secondary source): periodic limb movements while "
                   "awake, movement 0.5 to 10 s, intervals 4 to 90 s",
    "montplaisir1998": "Montplaisir J et al. 1998, Mov Disord 13: 324-329, doi 10.1002/mds.870130220 (abstract): "
                       "suggested immobilization test, an index above 40 movements per hour separates RLS from "
                       "controls",
    "woods1978": "Woods BT, Teuber HL 1978, Neurology 28: 1152-1157 (scale wording via Holzapfel et al. 2024, doi "
                 "10.1186/s12883-024-03828-4): 0 none, 1 barely discernible, 2 slight or stronger but briefer, 3 "
                 "strong and sustained, 4 equal to the intended hand",
    "hermsdoerfer1995": "Hermsdoerfer J et al. 1995, Exp Brain Res (secondary source): mirror over voluntary force "
                        "1.4 to 19.1 and 3.4 to 78.4 percent in two relatives",
    "bermejo2008": "Bermejo PE, Zabala JA 2008, Eur Spine J 17 Suppl 2: 294, doi 10.1007/s00586-008-0590-4 (full "
                   "text, one case): toe flexion-extension at low frequency, one Hz",
    "dressler1994": "Dressler D et al. 1994, Mov Disord 9: 13-21, doi 10.1002/mds.870090104 (secondary source): "
                    "painful legs and moving toes, EMG bursts of 0.5 to 1 s",
    "alvarez2008": "Alvarez MV et al. 2008, Mov Disord 23: 2062-2066, doi 10.1002/mds.22272 (abstract): toe "
                   "flexion-extension, abduction-adduction, fanning or clawing",
    "table": "SP1 joint-limit table (body.joint_limits): the anatomical table of reports/openbody/joint_limits.json "
             "since the open-body port's wave 1 (decision D3)",
    "hand_limits": "hands.py hand joint limits (B.K. et al. 2024; Boike et al. 2025; module bounds)",
    "module": "a reading or default of this module, not taken from a source (stated where used)",
}

# ---------------------------------------------------------------- channels
SIDE_SIGN = {"left": 1.0, "right": -1.0}


def _ch(joints, axis, sign, region, rom_axis=None):
    return {"joints": tuple(joints), "axis": axis, "sign": float(sign), "region": region,
            "rom_axis": rom_axis if rom_axis is not None else ({"x": 0, "y": 1, "z": 2}.get(axis, 0))}


def _channels():
    c = {"trunk_flex": _ch((3, 6, 9), "x", 1.0, "trunk"), "trunk_lat": _ch((3, 6, 9), "z", 1.0, "trunk"),
         "trunk_rot": _ch((3, 6, 9), "y", 1.0, "trunk"), "neck_flex": _ch((12, 15), "x", 1.0, "neck"),
         "neck_lat": _ch((12, 15), "z", 1.0, "neck"), "neck_rot": _ch((12, 15), "y", 1.0, "neck")}
    for s, k in (("left", 0), ("right", 1)):
        sg = SIDE_SIGN[s]
        arm, leg = f"arm_{s}", f"leg_{s}"
        c.update({
            f"collar_{s}": _ch((13 + k,), "z", sg, arm),                       # elevation
            f"sh_abd_{s}": _ch((16 + k,), "z", sg, arm),                       # abduction
            f"sh_flex_{s}": _ch((16 + k,), "x", -1.0, arm),                    # forward flexion of the hanging arm
            f"sh_rot_{s}": _ch((16 + k,), "y", sg, arm),                       # axial rotation of the humerus
            f"elbow_{s}": _ch((18 + k,), "y", -sg, arm),                       # flexion (body.ELBOW_FLEXION_SIGN)
            f"forearm_{s}": _ch((18 + k,), ("bone", 18 + k, 20 + k), SUPINATION_SIGN[s], arm, 0),  # supination
            f"wrist_flex_{s}": _ch((20 + k,), "z", -sg, arm),                  # flexion (dystonia.py)
            f"wrist_dev_{s}": _ch((20 + k,), "y", sg, arm),
            f"hip_flex_{s}": _ch((1 + k,), "x", -1.0, leg), f"hip_abd_{s}": _ch((1 + k,), "z", sg, leg),
            f"hip_rot_{s}": _ch((1 + k,), "y", sg, leg),                       # seated: the knee outward
            f"knee_{s}": _ch((4 + k,), "x", 1.0, leg), f"ankle_{s}": _ch((7 + k,), "x", 1.0, leg),   # plantarflexion
            f"toes_{s}": _ch((10 + k,), "x", -1.0, leg),                       # extension (toes up)
        })
    return c


CHANNELS = _channels()
REGIONS = {"trunk": (3, 6, 9), "neck": (12, 15), "arm_left": (13, 16, 18, 20), "arm_right": (14, 17, 19, 21),
           "leg_left": (1, 4, 7, 10), "leg_right": (2, 5, 8, 11)}
END_JOINT = {"trunk": 12, "neck": 15, "arm_left": 20, "arm_right": 21, "leg_left": 7, "leg_right": 8}
LEG = {"left": (1, 4, 7, 10), "right": (2, 5, 8, 11)}

# declared ladders (module docstring)
DIS_LEVELS = {1: 0.05, 2: 0.30, 3: 0.70, 4: 0.95}
DIS_BANDS = ((0.0, 0.10), (0.10, 0.50), (0.50, 0.90), (0.90, 1.0001))
AMP = {"slight": 0.05, "mild": 0.20, "moderate": 0.40, "marked": 0.70}
DUR = {"intermittent": 0.30, "common": 0.70, "prolonged": 0.95}
UHDRS = {1: ("slight", "intermittent"), 2: ("mild", "common"), "2b": ("moderate", "intermittent"),
         3: ("moderate", "common"), 4: ("marked", "prolonged")}
STEPS = {1: (0.05, 0.30), 2: (0.20, 0.70), 3: (0.40, 0.70), 4: (0.70, 0.95)}
UDYSRS_TERMS = {1: "slight", 2: "mild", 3: "moderate", 4: "severe"}       # SOURCES['udysrs']
AIMS_TERMS = {1: "minimal", 2: "mild", 3: "moderate", 4: "severe"}
BARS = {1: 0.30, 2: 0.70, 3: 1.0}
MIRROR_RATIO = {1: 0.05, 2: 0.20, "2b": 0.50, 3: 0.50, 4: 1.0}

# timing of the movement fragments: rise, hold and return (s, log-uniform), the next fragment's start as a share of
# the current fragment's length (uniform), channels per fragment
TIMING = {
    # chorea: bursts of more than 100 ms (hallett1981), the acceleration spectrum inside the 1 to 5 Hz band of
    # dinesh2020's index; the transition durations (0.2 to 0.5 s) and dwell are module values (no posture held past
    # 0.45 s: longer plateaus read as dystonic, clinical review of 2026-11-02)
    "chorea": {"rise": (0.2, 0.5), "hold": (0.0, 0.15), "ret": (0.2, 0.6), "next": (0.45, 1.0), "n_ch": (1, 3),
               "amp": (0.3, 1.0), "mode": "flow", "rest_p": 0.25, "dwell_max": 0.45},
    # peak-dose dyskinesia: the 1 to 3 Hz band of manson2000 (fragments of one third of a second to one second);
    # more overlapping than chorea's (the next fragment after 0.3 to 0.7 of the last), on two or three channels, seldom
    # back to rest (the clinical review of 2026-11-03: with chorea's seed and nearly its timing the peak-dose clips
    # shared the chorea clips' bouts; module values; fragments of 0.3 to 0.7 s at the speed caps brought the
    # acceleration's dominant frequency under 1 Hz, out of the band)
    "lid": {"rise": (0.2, 0.55), "hold": (0.0, 0.1), "ret": (0.2, 0.55), "next": (0.3, 0.7), "n_ch": (2, 3),
            "amp": (0.3, 1.0), "mode": "flow", "rest_p": 0.1, "dwell_max": 0.5},
    # tardive limb movements: more power at 1 to 2 Hz (wirshing1991)
    "tardive": {"rise": (0.25, 0.5), "hold": (0.0, 0.15), "ret": (0.25, 0.5), "next": (0.5, 1.0), "n_ch": (1, 2),
                "amp": (0.3, 1.0), "mode": "flow", "rest_p": 0.3, "dwell_max": 0.5},
    # ballism of the leg: bursts of 200 to 1500 ms (tai2016), movement at 1 to 1.5 Hz (yahata1997); the arm's
    # flings are drawn by fling_signals (BALLISM_FLING)
    "ballism": {"rise": (0.2, 0.35), "hold": (0.0, 0.1), "ret": (0.3, 0.5), "next": (0.9, 1.6), "n_ch": (2, 2),
                "amp": (0.75, 1.0)},
}
DRIFT_SHARE = 0.06          # [module] a flowing region's slow drift (0.3 to 1 Hz) inside its bouts, share of its raw
#                             amplitude: no posture is held perfectly still between two choreic movements
BANDS_HZ = {"athetosis": (0.3, 0.6),          # neilson1974
            "pseudoathetosis": (0.1, 0.4)}    # module: slower than athetosis, no published speed


# ---------------------------------------------------------------- small helpers
def minimum_jerk(x):
    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)
    return x ** 3 * (10.0 - 15.0 * x + 6.0 * x ** 2)


def _loguniform(rng, lo, hi):
    return float(math.exp(rng.uniform(math.log(lo), math.log(hi)))) if hi > lo else float(lo)


def _moving(x, n, fn):
    if n <= 1:
        return x.copy()
    pad = np.pad(x, (n // 2, n - 1 - n // 2), mode="edge")
    w = np.lib.stride_tricks.sliding_window_view(pad, n)
    return fn(w, axis=1)


def smooth_below(s, n):
    """A smooth version of s (T,) that never rises above it: a centred moving minimum then a centred moving mean of
    width n (base.walk's pelvis rule)."""
    return _moving(_moving(np.asarray(s, np.float64), n, np.min), n, np.mean)


def _runs(mask):
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.diff(m.astype(np.int8))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def close_mask(mask, rate, gap_s=GAP_S, min_run_s=MIN_RUN_S):
    """mask (T,) bool with stillness gaps shorter than gap_s inside movement closed and runs shorter than min_run_s
    dropped."""
    m = np.asarray(mask, bool).copy()
    runs = _runs(m)
    for (a0, b0), (a1, b1) in zip(runs, runs[1:]):
        if (a1 - b0) / rate < gap_s:
            m[b0:a1] = True
    for a, b in _runs(m):
        if (b - a) / rate < min_run_s:
            m[a:b] = False
    return m


def _rot(aa_deg):
    return axis_angle_to_matrix(torch.deg2rad(torch.as_tensor(aa_deg, dtype=F64)))


def _log_deg(R):
    return torch.rad2deg(matrix_to_axis_angle(R)).numpy()


def _bounds(lim, comp_base):
    return np.minimum(lim.lo[None], comp_base) - 1e-6, np.maximum(lim.hi[None], comp_base) + 1e-6


# ---------------------------------------------------------------- signals
def bouts(duration_s, share, rng, n_bouts=None, bout_s=3.0, start_s=0.5):
    """On intervals (s) of a region: share of [start_s, duration_s - 0.25] in n_bouts bouts (default one per bout_s of
    movement), placed by common.schedule (BOUT_RAMP_S long gaps and bouts at least); share 1 the whole span."""
    end = duration_s - 0.25
    if share <= 0:
        return []
    if share >= 0.999:
        return [(float(start_s), float(end))]
    on = share * (end - start_s)
    n = n_bouts if n_bouts is not None else max(1, int(round(on / bout_s)))
    while n > 1:
        try:
            return schedule(end, share, n, BOUT_RAMP_S, rng, 100.0, start_s=start_s)
        except ValueError:
            n -= 1
    return schedule(end, share, 1, BOUT_RAMP_S, rng, 100.0, start_s=start_s)


def fragment_events(intervals, channels, rng, timing, weights=None, signs=None):
    """Movement fragments (module docstring, events) inside on intervals [(a, b)] (s) on the channels (names):
    [{'t0', 'rise', 'hold', 'ret', 'amps': {channel: signed share}, 'delay': {channel: s}}]. A fragment that would
    pass the end of its interval is shortened to fit (its three phases scaled); none starts closer than 0.3 s to
    the end. signs {channel: +1 or -1} keeps a channel on one side of its base (a supported limb that can only
    leave its support, the seated trunk that cannot pass the backrest); the other channels take random signs."""
    tm = timing
    w = np.ones(len(channels)) if weights is None else np.asarray([weights.get(c, 1.0) for c in channels], float)
    w = w / w.sum()
    sg = dict(signs or {})

    def sign(c):
        if tm.get("positive"):
            return 1.0
        return float(sg[c]) if sg.get(c) else float(rng.choice([-1.0, 1.0]))
    out = []
    for a, b in intervals:
        t = float(a)
        last = {}                                        # flow: a channel's last move and its target
        while b - t > 0.3:
            flow = tm.get("mode") == "flow"
            r = _loguniform(rng, *tm["rise"])
            h = float(rng.uniform(*tm["hold"]))
            f = 0.0 if flow else _loguniform(rng, *tm["ret"])
            L = r + h + f
            if t + L > b:
                k = (b - t) / L
                r, h, f, L = r * k, h * k, f * k, b - t
            n = int(rng.integers(tm["n_ch"][0], min(tm["n_ch"][1], len(channels)) + 1))
            pick = rng.choice(len(channels), size=n, replace=False, p=w)
            amps = {channels[i]: float(sign(channels[i]) * rng.uniform(*tm["amp"])) for i in pick}
            delay = {channels[i]: (0.0 if j == 0 else float(rng.uniform(0.0, 0.25 * r))) for j, i in enumerate(pick)}
            if flow:                                     # a target posture; some fragments go back to rest
                amps = {c: (0.0 if rng.uniform() < tm.get("rest_p", 0.0) else a) for c, a in amps.items()}
                for c, (t_last, tgt) in sorted(last.items()):   # no posture held past dwell_max: it moves on
                    if c not in amps and tgt != 0.0 and t - t_last > tm.get("dwell_max", 1e9):
                        amps[c] = 0.0 if rng.uniform() < 0.6 else float(sign(c) * rng.uniform(*tm["amp"]))
                        delay[c] = float(rng.uniform(0.0, 0.25 * r))
                for c in list(amps):                    # a fragment always moves: no rest-to-rest target
                    if amps[c] == 0.0 and last.get(c, (0.0, 0.0))[1] == 0.0:
                        amps[c] = float(sign(c) * rng.uniform(*tm["amp"]))
                last.update({c: (t, v) for c, v in amps.items()})
            out.append({"t0": t, "rise": r, "hold": h, "ret": f, "amps": amps, "delay": delay,
                        "flow": flow, "bout_end": float(b)})
            t = t + (L if not flow else r + h) * float(rng.uniform(*tm["next"])) + (0.0 if not flow else h)
    return out


FLOW_RETURN_S = 0.35        # [module] at a bout's end a flowing region returns to rest over this (at least: a far
#                             posture comes back at the speed caps, SPEED_CAP_DEG_S)

# Speed caps of the flowing movements (the clinical review of 2026-11-03, M9: at grades 3 and 4 the chorea and the
# peak-dose dyskinesia reached wrist speeds of 2.2 to 6.2 m/s and read as ballism, because a transition's duration did
# not depend on its size and a delayed channel's rise was shortened by its delay). Each flowing transition of a channel
# group takes at least SPEED_PEAK times its excursion (the norm over the group's channels, in degrees at the level's
# gain) over the group's cap, so that its peak angular speed stays at the cap; a delayed channel keeps the fragment's
# rise and starts later; a group's next transition starts when its previous one ends (no two transitions of one
# joint add their speeds; the groups of an arm still overlap, so that the movement flows); the return to rest at a
# bout's end takes at least FLOW_RETURN_S and at least its distance at the cap. A group's transition that would not
# end, with that return after it, by its bout's end goes a shorter way toward its target (FLOW_FIT), or is left out
# (the group holds where it is): the movement stays inside its bouts, at the share of time of the duration grade
# (without this rule the delayed transitions spilled past the bouts: on 8 s seated builds at UHDRS 'common', 0.63 of
# the time nominal, the trunk moved 0.72 to 0.93 of it).
SPEED_PEAK = 1.875          # the peak over the mean speed of a minimum-jerk transition (15/8, minimum_jerk)
SPEED_CAP_DEG_S = {"shoulder": 160.0, "elbow": 220.0, "wrist": 260.0, "collar": 100.0, "neck": 200.0,
                   "trunk": 90.0}
# [module] peak angular speed of a flowing transition per channel group (the elbow with the forearm's rotation; the
# trunk the whole chain, lumbar to thoracic), far below the flings of hemiballism (BALLISM_FLING, 500 to 900 deg/s at
# the shoulder). Set on 8 s builds of the chorea at the page's 30 fps: at the review's values (shoulder 250, elbow and
# forearm 350, neck 200; wrist 400, collar 150, trunk 120) the standing grade 4 still peaked at 3.9 m/s at the wrist
# and 5.5 m/s at the index tip; at these 2.6 and 3.3 m/s on those 8 s builds, 3.1 and 4.2 m/s on the catalogue's 20 s
# build (its peak-dose step 4: 3.0 and 3.7 m/s), about 2.6 times the catalogue's grade 2 (1.2 and 1.6 m/s; seated grade
# 4: 1.5 and 2.4 m/s), the amplitudes unchanged (the arm at 0.68 to 0.70 of its range of motion at grade 4, 0.20 at
# grade 2, on the 8 s builds). The neck keeps the review's 200 (at 140 the acceleration of the seated grade-3 neck
# peaked at 0.8 Hz, under the 1 to 5 Hz band of the chorea index); the trunk is lower than the review's 120 (at 120
# the standing grade-4 wrist, carried by the trunk, rose to 2.9 m/s on the 8 s builds)
SPEED_ROUNDS = 3            # [module] rounds of solving the gain and rendering again at its degrees (capped_fragments)
FLOW_FIT = (1.0, 0.6, 0.3)  # [module] the shares of the way to its target a capped transition near a bout's end tries


def speed_group(channel):
    """The SPEED_CAP_DEG_S group of a channel name (its joint), or None (the legs, the toes, a pelvis channel)."""
    key = channel.rsplit("_", 1)[0] if channel.endswith(("_left", "_right")) else channel
    side = channel.rsplit("_", 1)[1] if channel.endswith(("_left", "_right")) else ""
    g = {"sh_flex": "shoulder", "sh_abd": "shoulder", "sh_rot": "shoulder", "elbow": "elbow", "forearm": "elbow",
         "wrist_flex": "wrist", "wrist_dev": "wrist", "collar": "collar"}.get(key)
    if g:
        return f"{g}_{side}"
    return "trunk" if channel.startswith("trunk_") else "neck" if channel.startswith("neck_") else None


def _group_rise(rise, deltas, speed):
    """{group: s}: the duration of a group's transitions, at least rise and SPEED_PEAK times the norm over the group
    of the channels' excursions in degrees over the group's cap; deltas {channel: excursion in raw units}; speed
    (caps {channel: deg/s}, degrees per unit {channel: deg}) or None (rise for every channel)."""
    if speed is None:
        return {}
    caps, dpu = speed
    sq = {}
    for c, d in deltas.items():
        if c in caps:
            g = speed_group(c)
            sq[g] = sq.get(g, 0.0) + (d * dpu[c]) ** 2
    cap_of = {speed_group(c): caps[c] for c in deltas if c in caps}
    return {g: max(rise, SPEED_PEAK * math.sqrt(v) / cap_of[g]) for g, v in sq.items()}


def render_fragments(events, channels, t, scale=None, speed=None):
    """(T, C) raw signals of fragments on channels at times t (s). A pulse fragment goes out with a minimum-jerk
    rise, holds, and comes back with a minimum-jerk return (a delayed channel keeps the fragment's end). A flowing
    fragment (TIMING mode 'flow') moves each of its channels from where it is to a new target posture by a
    minimum-jerk transition, so that the body flows from one posture to the next instead of returning to rest after
    every movement; at the end of its bout every channel returns to rest over FLOW_RETURN_S. scale {channel:
    (positive, negative)} weighs each fragment by the room on its side (an excursion fills each direction in
    proportion to the room the table leaves there; the sum of overlapping smooth fragments stays smooth). speed
    (caps {channel: deg/s}, degrees per raw unit {channel: deg}), for the flowing fragments: the speed caps (comment
    above SPEED_PEAK); without it a delayed flowing channel's rise is shortened by its delay (the former rendering,
    kept for the callers that pass no caps)."""
    X = np.zeros((len(t), len(channels)))
    idx = {c: i for i, c in enumerate(channels)}
    cur = {c: 0.0 for c in channels}
    busy = {}                                            # with speed caps: the end of each group's last transition
    bout = None
    for e in sorted(events, key=lambda e: e["t0"]):
        if e.get("flow") and bout is not None and e["bout_end"] != bout:
            _flow_rest(X, idx, cur, t, bout, speed, busy)
        bout = e.get("bout_end") if e.get("flow") else None
        targets = {}
        for c, a in e["amps"].items():
            sp, sm = (1.0, 1.0) if scale is None else scale[c]
            targets[c] = a * (sp if a > 0 else sm)
        grise = _group_rise(e["rise"], {c: v - cur[c] for c, v in targets.items()}, speed) if e.get("flow") else {}
        end = {}
        skip = set()
        if e.get("flow") and speed is not None:          # each group's transition and its return inside the bout
            for g in sorted({speed_group(c) or c for c in e["amps"]}):
                cs = [c for c in e["amps"] if (speed_group(c) or c) == g]
                start = max(max(e["t0"] + e["delay"][c], busy.get(g, -math.inf)) for c in cs)
                for f in FLOW_FIT:
                    tg = {c: cur[c] + f * (targets[c] - cur[c]) for c in cs}
                    r = _group_rise(e["rise"], {c: tg[c] - cur[c] for c in cs}, speed).get(g, e["rise"])
                    state = {c: tg.get(c, cur[c]) for c in channels if (speed_group(c) or c) == g}
                    back = _group_rise(FLOW_RETURN_S, {c: -v for c, v in state.items() if v != 0.0}, speed)
                    if start + r + back.get(g, FLOW_RETURN_S) <= e["bout_end"] + 1e-9:
                        targets.update(tg)
                        grise[g] = r
                        break
                else:
                    skip.update(cs)
        for c, a in e["amps"].items():
            sp, sm = (1.0, 1.0) if scale is None else scale[c]
            d = e["delay"][c]
            if e.get("flow"):
                if c in skip:
                    continue
                if speed is None:
                    r, t0 = max(e["rise"] - d, 0.05), e["t0"] + d
                else:
                    g = speed_group(c) or c
                    r = grise.get(g, e["rise"])
                    t0 = max(e["t0"] + d, busy.get(g, -math.inf))
                    end[g] = max(end.get(g, -math.inf), t0 + r)
                X[:, idx[c]] += (targets[c] - cur[c]) * minimum_jerk((t - t0) / r)
                cur[c] = targets[c]
            else:
                r = max(e["rise"] - d, 0.05)
                t0 = e["t0"] + d
                up = minimum_jerk((t - t0) / r)
                dn = minimum_jerk((t - (t0 + r + e["hold"])) / e["ret"])
                X[:, idx[c]] += a * (sp if a > 0 else sm) * (up - dn)
        busy.update(end)
    if bout is not None:
        _flow_rest(X, idx, cur, t, bout, speed, busy)
    return X


def _flow_rest(X, idx, cur, t, b, speed=None, busy=None):
    """Every flowing channel back to rest ending at the bout's end b, over FLOW_RETURN_S; with speed caps (render
    fragments) over at least its group's distance at the cap (_group_rise), each group starting no earlier than the
    end of its last transition (busy {group: s})."""
    r = _group_rise(FLOW_RETURN_S, {c: -v for c, v in cur.items() if v != 0.0}, speed)
    end = {}
    for c, i in idx.items():
        if cur[c] != 0.0:
            g = speed_group(c) or c
            rr = r.get(g, FLOW_RETURN_S) if speed is not None else FLOW_RETURN_S
            t0 = b - rr if speed is None else max(b - rr, busy.get(g, -math.inf))
            X[:, i] -= cur[c] * minimum_jerk((t - t0) / rr)
            cur[c] = 0.0
            end[g] = max(end.get(g, -math.inf), t0 + rr)
    if busy is not None:
        busy.update(end)


def capped_fragments(base, events, chans, t, scale, level, gains=None, margin=SOFT_MARGIN, extra=None,
                     rounds=SPEED_ROUNDS):
    """((T, C) raw signals, info): the flowing fragments rendered at the speed caps (comment above SPEED_PEAK): the
    degrees per raw unit of each channel are those of the gain region_signals solves for level on the signals (plus
    extra (T, C), the slow drift), rendered again at those degrees, rounds times (the gain moves a little as the
    transitions slow). info: the caps, the last gain, and the realized peak speed of each group in deg/s at that
    gain (before the soft limits, which only slow a signal)."""
    caps = {c: SPEED_CAP_DEG_S[speed_group(c).rsplit("_", 1)[0]] for c in chans if speed_group(c)}
    X = render_fragments(events, chans, t, scale)
    if not caps or level is None or level <= 0:
        return X, None
    w = {c: (gains or {}).get(c, 1.0) for c in chans}
    g = None
    for _ in range(rounds):
        raw = {c: X[:, i] + (0.0 if extra is None else extra[:, i]) for i, c in enumerate(chans)}
        _, inf = region_signals(base, list(chans), raw, level, gains, margin)
        g = inf["gain"]
        dpu = {c: g * w[c] * channel_rom(c) for c in chans}
        X = render_fragments(events, chans, t, scale, speed=(caps, dpu))
    dt = float(t[1] - t[0])
    peak = {}
    for i, c in enumerate(chans):
        if c in caps:
            grp = speed_group(c)
            v = np.gradient(X[:, i] * g * w[c] * channel_rom(c), dt)
            peak[grp] = np.sqrt(peak.get(grp, 0.0) ** 2 + v ** 2) if grp in peak else np.abs(v)
    return X, {"caps_deg_s": {grp: SPEED_CAP_DEG_S[grp.rsplit("_", 1)[0]] for grp in peak},
               "gain_estimate": float(g), "peak_deg_s": {grp: float(np.max(v)) for grp, v in peak.items()},
               "rule": "each flowing transition at most its group's cap at its peak (SPEED_PEAK times the excursion "
                       "over the duration); a channel's transitions one after the other (module)"}


# The rate governor (wave SH1d fix B). The speed caps above bound each channel group's own transitions; the body's
# segments add what the other stages bring: the collar under the shoulder, the carriage of the seated arms (the hands
# kept on the thighs while the trunk moves, support_arms: near the arm's full reach the elbow's angle changes many
# times faster than the shoulder-to-wrist distance), the contact guard's blends, the standing pelvis under the trunk.
# On fe89df3 the skeleton page read the grade-4 chorea and the step-4 peak-dose and dystonic dyskinesia over the plan's
# bounds of a limb (the right upper arm of invol_chorea_E1_g4 at 258 deg/s, its shoulder joint turning at 244 deg/s
# where its own channels stayed at 160; the right elbow of invol_chorea_E1_trunk_g4 at 368 deg/s, every bit of it from
# the carriage; the trunk of invol_chorea_E7_g4 at 142 deg/s). govern_rates reads the composed body as the page reads
# it and, where a segment turns faster than RATE_MARGIN of its bound, plays the movement slower there: the clip's time
# is warped inside a window around each fast run (the poses kept, only their timing), the quiet knots of the window
# played up to GOVERN_SPEEDUP_MAX faster so that the window keeps its length and every knot outside it stays bit for
# bit. Each choreic event keeps its excursion and the clip its amplitudes; its peak velocity is bounded.
RATE_BOUND_DEG_S = {"trunk": 120.0, "neck": 200.0, "upper_arm": 250.0, "elbow": 350.0}
# [plan] the angular speeds the clinical review of 2026-11-03 gives as a limb's bounds (PLAN M9: the shoulder about 250
# deg/s, the elbow 350, the neck 200; M10: the trunk about 120), read as the skeleton page reads them
# (render_skeleton_catalogue.RATE_CAPS_DEG_S, rate_peaks): the pelvis-to-neck line in the room, the neck-to-head line
# and each upper arm (shoulder to elbow) in the trunk's frame, each elbow's angle
RATE_MARGIN = 0.9           # [module] the governor keeps each reading at the body's 100 Hz clock under this share of
#   its bound: the page reads central differences at 30 fps on the export's 1 mm grid, whose rounding adds up to about
#   25 deg/s to an upper arm read in the trunk's frame (its own sqrt(3) mm per end and the trunk frame's)
GOVERN_PAD_S = 0.5          # [module] the window around a fast run, each way, widened by this until its warp fits
GOVERN_MAX_PAD_S = 3.0      # [module] at most
GOVERN_SPEEDUP_MAX = 1.3    # [module] the quiet knots of a window play at most this much faster
GOVERN_EASE_S = 0.3         # [module] the speed-up eased in from each edge of the window over this (no step of speed)
GOVERN_ENVELOPE_S = 0.05    # [module] the fast runs read on an envelope of the readings over this (never under them)
GOVERN_ROUNDS = 3           # [module] read again after the warp, and warped again where the knots still exceed it
GOVERN_TOL = 0.01           # [module] what the resampling between knots may leave over the margin (ok within it: the
#                             grade-4 chorea's arms 1.0004 times it after three rounds)


def segment_rates(body):
    """{reading: (T,) deg/s} of a body at its own clock (central differences, 0 at both ends): the readings of
    RATE_BOUND_DEG_S ('trunk', 'neck', 'left_upper_arm', 'right_upper_arm', 'left_elbow', 'right_elbow'), as the
    skeleton page reads them on its positions (the trunk's frame: y the pelvis to neck line, x the right to left
    shoulder line made square to it)."""
    J = body.joints().numpy()
    T = len(J)

    def unit(x):
        return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)

    def turn(u):
        w = np.zeros(T)
        if T > 2:
            w[1:-1] = np.degrees(np.arccos(np.clip((u[2:] * u[:-2]).sum(-1), -1.0, 1.0))) * body.rate / 2.0
        return w
    y = unit(J[:, 12] - J[:, 0])
    x = J[:, 16] - J[:, 17]
    x = unit(x - (x * y).sum(-1, keepdims=True) * y)
    F = np.stack([x, y, np.cross(x, y)], 1)                       # rows: the trunk's axes in the room
    out = {"trunk": turn(y), "neck": turn(np.einsum("tij,tj->ti", F, unit(J[:, 15] - J[:, 12])))}
    for s, (sh, el, wr) in (("left", (16, 18, 20)), ("right", (17, 19, 21))):
        ua, fa = unit(J[:, el] - J[:, sh]), unit(J[:, wr] - J[:, el])
        out[f"{s}_upper_arm"] = turn(np.einsum("tij,tj->ti", F, ua))
        ang = np.degrees(np.arccos(np.clip((ua * fa).sum(-1), -1.0, 1.0)))
        w = np.zeros(T)
        if T > 2:
            w[1:-1] = np.abs(ang[2:] - ang[:-2]) * body.rate / 2.0
        out[f"{s}_elbow"] = w
    return out


def _rate_ratio(body, margin=RATE_MARGIN):
    """(T,) the largest reading of segment_rates over its bound times margin, and the reading that sets it."""
    rates = segment_rates(body)
    keys = list(rates)
    W = np.stack([rates[k] / (margin * RATE_BOUND_DEG_S[k.split("_", 1)[1] if k.startswith(("left_", "right_"))
                                                        else k]) for k in keys], 1)
    return W.max(1), [keys[i] for i in W.argmax(1)]


def _warp_window(q_of, a, b, T, rate):
    """(knot times (n,) in the source of the output knots a..b, speed-up k) of a window [a, b] (knots): q_of(k) the
    source's time per output time at each source knot of the window (at least 1 where the body is too fast, 1 / k
    elsewhere, eased from 1 at the edges), k solved by bisection so that the window keeps its length; None when no k
    up to GOVERN_SPEEDUP_MAX fits."""
    L = float(b - a)

    def length(k):
        q = q_of(k)
        return float(0.5 * (q[1:] + q[:-1]).sum())
    if length(GOVERN_SPEEDUP_MAX) > L:
        return None
    lo, hi = 1.0, GOVERN_SPEEDUP_MAX
    for _ in range(50):
        m = 0.5 * (lo + hi)
        if length(m) > L:
            lo = m
        else:
            hi = m
    q = q_of(hi)
    t_new = np.r_[0.0, np.cumsum(0.5 * (q[1:] + q[:-1]))]
    t_new *= L / t_new[-1]                                  # the rounding of the bisection
    src = np.arange(a, b + 1, dtype=np.float64)
    return np.interp(np.arange(b - a + 1, dtype=np.float64), t_new, src), hi


def _lerp_rotations(R, a, b, w):
    """(n, J, 3, 3) the rotations R (T, J, 3, 3) between the knots a and b (n,) at the shares w (n,): each joint's
    axis-angle components interpolated linearly (a box of the joint-limit table holds the whole way between two knots
    inside it, which the Hermite curve of the viewer did not: it left the table on the grade-4 chorea), geodesically
    for row 0 (the root's global orientation) and wherever the two knots' axis-angle vectors lie more than 30 deg
    apart (the wrap near a half turn)."""
    Ra, Rb = R[a], R[b]
    A, B = matrix_to_axis_angle(Ra), matrix_to_axis_angle(Rb)
    ww = w[:, None, None]
    lin = axis_angle_to_matrix(A + ww * (B - A))
    geo = Ra @ axis_angle_to_matrix(ww * matrix_to_axis_angle(Ra.transpose(-1, -2) @ Rb))
    far = (B - A).norm(dim=-1) > math.radians(30.0)
    far[:, 0] = True
    return torch.where(far[..., None, None], geo, lin)


def _hand_outside(R_hand):
    """(n, 30) bool: the hand joints of R_hand (n, 30, 3, 3) outside the hand table (hands.hand_limit_report's rule:
    the flexion range and the abduction limit of each joint)."""
    comp = H.hand_components(R_hand)
    fr = H.hand_frame()
    out = np.zeros(tuple(R_hand.shape[:2]), dtype=bool)
    for side in H.SIDES:
        for j, ax in fr[side]["joints"].items():
            f, lv = ax["finger"], ax["level"]
            fl, ab, _ = (np.asarray(x) for x in comp[j])
            name = (H.THUMB_LEVELS if f == "thumb" else H.LEVELS)[lv]
            if f == "thumb" and name == "cmc":
                lo, hi, alim = -H.CMC_LIMIT_DEG, H.CMC_LIMIT_DEG, H.CMC_LIMIT_DEG
            else:
                (lo, hi), alim = H.FLEX_LIMITS_DEG[f][name], H.ABDUCTION_LIMIT_DEG
            out[:, H.hand_row(j)] = (fl < lo) | (fl > hi) | (np.abs(ab) > alim)
    return out


def _replant(new, old, sel, tau, still_m=1e-7):
    """new (old resampled at the source knots tau (T,) on its knots sel) with each leg re-solved (ik.hinge_leg, the
    knee's hinge along the resampled thigh's x axis) at the knots of sel whose two source knots around tau hold its
    foot still on the floor's plane (the ankle and toe joints within still_m horizontally), its knee a hinge (R_x),
    where the resampling moved the ankle or the toes along the floor:
    the ankle on the straight line between the source's two, its global rotation on the geodesic between theirs, the
    toes' local rotation the nearer source knot's. Joint by joint the resampling moved a planted foot along the floor
    (0.02 mm/s on the standing grade-4 chorea, 0.06 mm/s on a seated foot rising straight up at the start of a leg
    excursion, against the validity's 0.001 mm/s)."""
    Jo, Go = old.fk()
    Jn, Gn = new.fk()
    rest = new.rest()
    R = new.local_R().clone()
    Ro = old.local_R()
    t = np.asarray(tau[sel], dtype=np.float64)
    i0 = torch.as_tensor(np.clip(np.floor(t).astype(np.int64), 0, old.T - 2))
    w = torch.as_tensor(t - i0.numpy(), dtype=F64)
    kk = torch.as_tensor(np.asarray(sel, dtype=np.int64))
    r6 = new.rot6d.clone()
    changed = False
    for side in ("left", "right"):
        h, kn, a, f = LEG[side]
        kx = matrix_to_axis_angle(Ro[i0, kn])                           # a hinge knee (R_x), as the legs' solves
        moved = (((Jn[kk, a] - Jo[i0, a])[:, [0, 2]].norm(dim=-1) > 1e-12)
                 | ((Jn[kk, f] - Jo[i0, f])[:, [0, 2]].norm(dim=-1) > 1e-12))
        still = (((Jo[i0 + 1, a] - Jo[i0, a])[:, [0, 2]].norm(dim=-1) < still_m)
                 & ((Jo[i0 + 1, f] - Jo[i0, f])[:, [0, 2]].norm(dim=-1) < still_m)
                 & (kx[:, 1:].abs().amax(-1) < 1e-6) & moved)
        if not bool(still.any()):
            continue
        k, i, ww = kk[still], i0[still], w[still]
        A = Jo[i, a] + ww[:, None] * (Jo[i + 1, a] - Jo[i, a])
        Ga = Go[i, a] @ axis_angle_to_matrix(ww[:, None]
                                             * matrix_to_axis_angle(Go[i, a].transpose(-1, -2) @ Go[i + 1, a]))
        G_ref, Hh = Gn[k, 0], Jn[k, h]
        G_th, kap = hinge_leg(Hh, A, Hh + G_ref[:, :, 2], rest[kn] - rest[h], rest[a] - rest[kn], G_ref,
                              hinge=Gn[k, h][:, :, 0])
        Rk = axis_angle_to_matrix(torch.stack([kap, 0 * kap, 0 * kap], -1))
        R[k, h] = G_ref.transpose(-1, -2) @ G_th
        R[k, kn] = Rk
        R[k, a] = (G_th @ Rk).transpose(-1, -2) @ Ga
        R[k, f] = Ro[torch.where(ww < 0.5, i, i + 1), f]
        for j in (h, kn, a, f):                     # these knots only: every other knot keeps its rot6d bit for bit
            r6[k, j] = matrix_to_rot6d(R[k, j])
        changed = True
    return replace(new, rot6d=r6) if changed else new


def govern_rates(body, margin=RATE_MARGIN):
    """(Body, info): body with its movement played slower where a reading of segment_rates exceeds margin times its
    bound (RATE_BOUND_DEG_S; comment above): the fast runs of the readings' envelope (a moving maximum then a moving
    mean over GOVERN_ENVELOPE_S, never under the readings), each in a window widened by GOVERN_PAD_S each way (up to
    GOVERN_MAX_PAD_S) until its warp keeps the window's length with the quiet knots at most GOVERN_SPEEDUP_MAX faster;
    the body sampled again at the warped times (_lerp_rotations on the joints and the hands, transl linear; a hand
    joint the interpolation puts outside its limits takes the nearer source knot; a foot in contact where both source
    knots around it are, and re-planted where the source holds it still, _replant), the knots
    outside the windows bit for bit; GOVERN_ROUNDS rounds.
    The body itself (unchanged) where no reading exceeds its bound."""
    n_env = max(1, int(round(GOVERN_ENVELOPE_S * body.rate)))
    ease = max(1, int(round(GOVERN_EASE_S * body.rate)))
    r0, which0 = _rate_ratio(body, margin)
    info = {"margin": margin, "bounds_deg_s": dict(RATE_BOUND_DEG_S), "ratio_max_before": float(r0.max()),
            "reading_before": which0[int(r0.argmax())], "windows": [], "rounds": 0}
    out, r = body, r0
    touched = np.zeros(body.T, dtype=bool)
    for rnd in range(GOVERN_ROUNDS):
        if r.max() <= 1.0:
            break
        env = _moving(_moving(r, n_env, np.max), n_env, np.mean)
        env = np.maximum(env, r)
        T = out.T
        tau = np.arange(T, dtype=np.float64)
        wins = []
        for a0, b0 in _runs(env > 1.0):
            pad = GOVERN_PAD_S
            while True:
                a = max(0, a0 - int(round(pad * out.rate)))
                b = min(T - 1, b0 - 1 + int(round(pad * out.rate)))
                if wins and a <= wins[-1][1]:                # merged with the window before
                    a = wins[-1][0]
                k = np.arange(a, b + 1)
                edge = np.minimum(k - a, b - k) / ease
                ramp = np.where(edge >= 1.0, 1.0, edge * edge * (3.0 - 2.0 * edge))

                def q_of(s, ramp=ramp, k=k):
                    return np.maximum(1.0 - (1.0 - 1.0 / s) * ramp, env[k])
                res = _warp_window(q_of, a, b, T, out.rate)
                if res is not None or pad >= GOVERN_MAX_PAD_S or (a == 0 and b == T - 1):
                    break
                pad = min(pad + GOVERN_PAD_S, GOVERN_MAX_PAD_S)
            if res is None:
                info["windows"].append({"s": [round(a / out.rate, 3), round(b / out.rate, 3)], "fits": False,
                                        "round": rnd})
                continue
            if wins and a == wins[-1][0]:
                wins.pop()
            wins.append((a, b, res[1]))
            tau[a:b + 1] = res[0]
        info["windows"] += [{"s": [round(a / out.rate, 3), round(b / out.rate, 3)], "speedup": round(float(k), 4),
                             "fits": True, "round": rnd} for a, b, k in wins]
        if not wins:
            break
        sel = np.flatnonzero(tau != np.arange(T))
        touched[sel] = True
        i0 = np.clip(np.floor(tau[sel]).astype(np.int64), 0, T - 2)
        w = torch.as_tensor(tau[sel] - i0, dtype=F64)
        a_, b_ = torch.as_tensor(i0), torch.as_tensor(i0 + 1)
        r6, trn, ct = out.rot6d.clone(), out.transl.clone(), out.contact.clone()
        with torch.no_grad():
            r6[sel] = matrix_to_rot6d(_lerp_rotations(out.local_R(), a_, b_, w))
            trn[sel] = out.transl[a_] + w[:, None] * (out.transl[b_] - out.transl[a_])
            ct[sel] = out.contact[a_] & out.contact[b_]          # in contact where both source knots are
            new = replace(out.copy(), rot6d=r6, transl=trn, contact=ct)
            if H.has_hands(out):
                Rh = out.hand_R()
                Rn = _lerp_rotations(Rh, a_, b_, w)
                bad = _hand_outside(Rn)                  # (n, 30): a hand joint the interpolation put outside
                if bad.any():                            # its limits takes the nearer source knot
                    near = torch.where(w < 0.5, a_, b_)
                    Rn = torch.where(torch.as_tensor(bad)[..., None, None], Rh[near], Rn)
                h6 = out.hand6d.clone()
                h6[sel] = matrix_to_rot6d(Rn)
                new = replace(new, hand6d=h6)
        out = _replant(new, out, sel, tau)
        info["rounds"] = rnd + 1
        r, _ = _rate_ratio(out, margin)
    r1, which1 = _rate_ratio(out, margin) if out is not body else (r0, which0)
    info.update(ratio_max_after=float(r1.max()), reading_after=which1[int(r1.argmax())],
                warped_s=float(touched.sum() / body.rate), ok=bool(r1.max() <= 1.0 + GOVERN_TOL),
                rule="each reading of segment_rates at most margin times its bound (RATE_BOUND_DEG_S): the clip's time "
                     "warped inside windows around the fast runs, the poses kept (module)")
    return out, info


def band_noise(T, rate, band, rng, n=1):
    """(T, n) Gaussian noise filtered to band (Hz) by a raised-cosine window over the band in the frequency domain,
    each column of unit standard deviation."""
    f = np.fft.rfftfreq(T, 1.0 / rate)
    lo, hi = band
    c, w = 0.5 * (lo + hi), 0.5 * (hi - lo)
    g = np.where(np.abs(f - c) <= w * 1.5, 0.5 + 0.5 * np.cos(np.pi * np.clip((f - c) / (1.5 * w), -1, 1)), 0.0)
    out = np.empty((T, n))
    for i in range(n):
        X = np.fft.rfft(rng.standard_normal(T)) * g
        x = np.fft.irfft(X, n=T)
        out[:, i] = x / (x.std() + 1e-12)
    return out


def ramp_envelope(t, intervals, ramp_s=0.5):
    """(T,) 1 inside each interval, raised-cosine ramps of ramp_s inside its edges, 0 outside."""
    e = np.zeros_like(t)
    for a, b in intervals:
        up = 0.5 - 0.5 * np.cos(np.pi * np.clip((t - a) / ramp_s, 0, 1))
        dn = 0.5 - 0.5 * np.cos(np.pi * np.clip((b - t) / ramp_s, 0, 1))
        e = np.maximum(e, np.minimum(up, dn))
    return e


def pattern(t, rate_hz, rng, intervals=None, jitter=STEREO_JITTER, phase=None, ramp_s=0.5, wave="sine"):
    """(T,) a repeated pattern (module docstring): sin of a phase whose rate wanders by jitter (a smooth random walk
    of the rate, one value per cycle interpolated), amplitude jittered by cycle; inside intervals (ramped) or always.
    wave 'sine' (centred) or 'pulse' (0 to 1, (1 - cos) / 2: a lift from rest and back)."""
    n_cyc = int(math.ceil(t[-1] * rate_hz)) + 2
    rk = rate_hz * (1.0 + jitter * rng.standard_normal(n_cyc).clip(-2, 2))
    ak = 1.0 + jitter * rng.standard_normal(n_cyc).clip(-2, 2)
    tc = np.arange(n_cyc) / rate_hz
    r = np.interp(t, tc, rk)
    a = np.interp(t, tc, ak)
    ph = (rng.uniform(0, 2 * math.pi) if phase is None else phase) + 2 * math.pi * np.concatenate(
        [[0.0], np.cumsum(0.5 * (r[1:] + r[:-1]) * np.diff(t))])
    x = a * (np.sin(ph) if wave == "sine" else 0.5 - 0.5 * np.cos(ph))
    if intervals is not None:
        x = x * ramp_envelope(t, intervals, ramp_s)
    return x


# ---------------------------------------------------------------- channel geometry
def channel_axes(body, name, k=0):
    """[(joint, axis (3,) unit in the parent frame, split)] of a channel at knot k of the body; split by ROM."""
    ch = CHANNELS[name]
    lim = joint_limits()
    if isinstance(ch["axis"], tuple):                 # axial rotation about a bone, carried into the parent frame
        _, j, child = ch["axis"]
        u = bone_axis(j, child, body.betas, body.gender)
        a = (body.local_R()[k, j] @ u).numpy()
        hinge = np.zeros(3)
        hinge[1] = 1.0                                   # orthogonal to the elbow hinge (y): no cross-talk
        a = a - (a @ hinge) * hinge
        return [(j, ch["sign"] * a / np.linalg.norm(a), 1.0)]
    v = np.zeros(3)
    v[{"x": 0, "y": 1, "z": 2}[ch["axis"]]] = ch["sign"]
    roms = np.array([lim.rom[j, ch["rom_axis"]] for j in ch["joints"]])
    return [(j, v.copy(), float(r / roms.sum())) for j, r in zip(ch["joints"], roms)]


def channel_rom(name):
    """deg: the channel's ROM (module docstring), the sum of hi - lo over its joints on the axis component."""
    ch = CHANNELS[name]
    lim = joint_limits()
    return float(sum(lim.rom[j, ch["rom_axis"]] for j in ch["joints"]))


def channel_room(body, name, k=0):
    """(room_plus, room_minus) deg of a channel from knot k of the body: the smallest over its joints of the joint's
    room (body.room_deg in the parent frame) over its split."""
    plus = minus = 1e9
    R = body.local_R()[k]
    for j, a, s in channel_axes(body, name, k):
        plus = min(plus, room_deg(R[j], j, a, 1.0, "parent", max_deg=180.0) / s)
        minus = min(minus, room_deg(R[j], j, a, -1.0, "parent", max_deg=180.0) / s)
    return plus, minus


SOFT_KNEE = 0.7             # [module] the soft limit leaves a signal unchanged below this share of its bound


def room_scale(body, chans, k=0):
    """{channel: (positive, negative)}: the room on each side of the base posture over their mean (render_fragments)."""
    out = {}
    for c in chans:
        p, m = channel_room(body, c, k)
        mean = 0.5 * (p + m)
        out[c] = (p / mean, m / mean) if mean > 0 else (1.0, 1.0)
    return out


def soft_limit(x, plus, minus, margin=SOFT_MARGIN, knee=SOFT_KNEE):
    """x (deg) bounded smoothly inside (-margin minus, margin plus): unchanged up to knee times the bound L, then
    k L + (1 - k) L tanh((|x| - k L) / ((1 - k) L)) (continuous with its slope), on each side."""
    x = np.asarray(x, dtype=np.float64)
    Lp, Lm = max(margin * plus, 1e-6), max(margin * minus, 1e-6)

    def side(v, L):
        a = knee * L
        return np.where(v <= a, v, a + (1 - knee) * L * np.tanh((v - a) / ((1 - knee) * L)))
    return np.where(x >= 0, side(np.abs(x), Lp), -side(np.abs(x), Lm))


def offsets_of(body, signals, k=0):
    """{joint: (T, 3) deg axis-angle offset in the parent frame} of channel signals {name: (T,) deg}."""
    out = {}
    for name, x in signals.items():
        for j, a, s in channel_axes(body, name, k):
            out.setdefault(j, np.zeros((len(x), 3)))
            out[j] += s * np.asarray(x)[:, None] * a[None]
    return out


OFFSET_GRID = 32            # [module] apply_offsets reads each joint's path from scale 0 to 1 at this many steps


def apply_offsets(body, offs, smooth_s=0.2, bounds=None):
    """(Body, info): the offsets {joint: (T, 3) deg} composed on the body (R = exp(w) R_base), each joint's offset
    scaled down at the knots where it would leave the table (or the base's own excursion) by a scale that never rises
    above the per-knot admissible one (smooth_below over smooth_s): no new limit violation (module docstring). bounds:
    (lo, hi) (T, 22, 3) deg of the table the clip is judged on (limits_ext.knot_bounds), the healthy table by
    default."""
    if not offs:
        return body.copy(), {"scaled_joints": {}}
    lim = joint_limits()
    R = body.local_R()
    Rn = R.clone()
    info = {}
    n = max(1, int(round(smooth_s * body.rate)))
    for j, w in offs.items():
        cb = _log_deg(R[:, j])
        b_lo, b_hi = (lim.lo[j], lim.hi[j]) if bounds is None else (bounds[0][:, j], bounds[1][:, j])
        lo, hi = np.minimum(b_lo, cb) - 1e-6, np.maximum(b_hi, cb) + 1e-6
        W = torch.as_tensor(w, dtype=F64)

        def inside(s):
            c = _log_deg(_rot(W * torch.as_tensor(s, dtype=F64)[:, None]) @ R[:, j])
            return ((c >= lo) & (c <= hi)).all(-1)
        # the first scale on the way from 0 to 1 where the joint leaves the box (open-body port, wave 3: an offset
        # past 180 deg wraps and can end inside again, as the positive control's 300 deg wrist; read on a grid)
        first = np.full(body.T, 2.0)
        for g in np.linspace(1.0, 0.0, OFFSET_GRID + 1)[:-1]:
            first = np.where(inside(np.full(body.T, g)), first, g)
        ok = first > 1.0
        s = np.ones(body.T)
        if not ok.all():
            bad = np.flatnonzero(~ok)
            a_, b_ = np.zeros(len(bad)), first[bad]
            for _ in range(30):
                m = 0.5 * (a_ + b_)
                sm = np.ones(body.T)
                sm[bad] = m
                good = inside(sm)[bad]
                a_, b_ = np.where(good, m, a_), np.where(good, b_, m)
            s[bad] = a_
            s = smooth_below(s, n)
            for _ in range(20):                          # the smoothed scale is inside where s was; check
                ok2 = inside(s)
                if ok2.all():
                    break
                s = np.where(ok2, s, s * 0.95)
            info[int(j)] = {"scaled_knots": int((s < 1).sum()), "scale_min": float(s.min())}
        Rn[:, j] = _rot(W * torch.as_tensor(s, dtype=F64)[:, None]) @ R[:, j]
    return body.with_rotations(Rn, list(offs)), {"scaled_joints": info}


def channel_values(body, base, name, k=0):
    """(T,) deg: a channel's value read from a body against its base: the projection of each joint's offset
    log(R R_base^T) on the channel's axis, summed over the chain (the inverse of offsets_of for orthogonal
    channels)."""
    R, Rb = body.local_R(), base.local_R()
    x = np.zeros(body.T)
    for j, a, s in channel_axes(base, name, k):
        w = _log_deg(R[:, j] @ Rb[:, j].transpose(-1, -2))
        x += w @ a
    return x


def extent_share(x, rom):
    x = np.asarray(x)
    return float((x.max() - x.min()) / rom) if len(x) else 0.0


# ---------------------------------------------------------------- legs and pelvis
def _fk_np(body):
    J, G = body.fk()
    return J.numpy(), G.numpy()


def seated_legs(base, side, lift=None, fwd=None, lat=None, swivel_deg=None, pitch_deg=None):
    """(R (T, 22, 3, 3) torch, info): the leg of side re-solved with the hip fixed (module docstring, Feet): the ankle
    at its base position plus (lat along the body's left, lift up, fwd along the heading) (T,) m, the knee turned by
    swivel_deg (T,) about the hip-to-ankle line from the base knee, the foot's global rotation the base's pitched by
    pitch_deg (T,) about the body's lateral axis (positive toes up). Every argument None leaves the leg as it is."""
    h, kn, a, ft = LEG[side]
    T = base.T
    z = np.zeros(T)
    lift, fwd, lat = (z if v is None else np.asarray(v, float) for v in (lift, fwd, lat))
    sw = np.radians(z if swivel_deg is None else np.asarray(swivel_deg, float))
    pitch = np.radians(z if pitch_deg is None else np.asarray(pitch_deg, float))
    J, G = base.fk()
    R = base.local_R().clone()
    rest = base.rest()
    p, q = rest[kn] - rest[h], rest[a] - rest[kn]
    Hh = J[:, h]
    A0 = J[:, a]
    fwd_dir = G[:, 0][:, :, 2].clone()
    fwd_dir[:, 1] = 0
    fwd_dir = fwd_dir / fwd_dir.norm(dim=-1, keepdim=True)
    left_dir = torch.cross(torch.tensor([0.0, 1.0, 0.0], dtype=F64).expand(T, 3), fwd_dir, dim=-1)
    A = A0 + torch.from_numpy(lat)[:, None] * left_dir + torch.from_numpy(lift)[:, None] * \
        torch.tensor([0.0, 1.0, 0.0], dtype=F64) + torch.from_numpy(fwd)[:, None] * fwd_dir
    L = float(p.norm() + q.norm())
    d = (A - Hh).norm(dim=-1)
    if bool((d > 0.985 * L).any()):
        k = (0.985 * L / d).clamp(max=1.0)[:, None]
        A = Hh + (A - Hh) * k
    u = (A - Hh) / (A - Hh).norm(dim=-1, keepdim=True)
    K0 = J[:, kn] - Hh
    pole = Hh + torch.einsum("tij,tj->ti", axis_angle_to_matrix(u * torch.from_numpy(sw)[:, None]), K0)
    G_th, kap = hinge_leg(Hh, A, pole, p, q, G[:, 0])
    Rk = axis_angle_to_matrix(torch.stack([kap, 0 * kap, 0 * kap], -1))
    G_sh = G_th @ Rk
    Gf = axis_angle_to_matrix(left_dir * (-torch.from_numpy(pitch))[:, None]) @ G[:, a]
    R[:, h] = G[:, 0].transpose(-1, -2) @ G_th
    R[:, kn] = Rk
    R[:, a] = G_sh.transpose(-1, -2) @ Gf
    return R, {"reach_clipped": bool((d > 0.985 * L).any())}


PELVIS_SMOOTH_S = 0.5       # [module] standing: the reachable pelvis height smoothed below over this window
PELVIS_SIGMA_S = 0.16       # [module] standing: the pelvis height target low-passed (Gaussian sigma, about 1 Hz)
REACH_SIGMA_S = 0.08        # [module] standing: the reachable height smoothed again by a Gaussian of this sigma
REACH_SAFETY_M = 0.002      # [module] and lowered by this, so that it stays under the reachable height
SOFT_KNEES_M = 0.006        # [module] standing: the width of the smooth clamp that keeps a pelvis target under the
#                             reachable height (until 2026-11-03 also the level-0 pelvis's dip, now STAND_DIP_M)
STAND_DIP_M = 0.001         # [module] standing: the level-0 pelvis lowered by this in stand_soft's second pass (the
#                             knees 16.5 deg from straight between the thigh and shank axes; at SOFT_KNEES_M, 20.3 deg,
#                             the clinical review of 2026-11-03 read the stance as crouched against 13.2 deg, 12.1 to
#                             17.1, on the motion-capture stance ref_stand_E7)


def gauss_smooth(x, sigma_n):
    """(T,) x smoothed by a Gaussian of sigma_n knots (edges padded with the end values)."""
    x = np.asarray(x, np.float64)
    if sigma_n < 0.5:
        return x.copy()
    h = int(math.ceil(4 * sigma_n))
    k = np.exp(-0.5 * (np.arange(-h, h + 1) / sigma_n) ** 2)
    return np.convolve(np.pad(x, h, mode="edge"), k / k.sum(), "valid")


def soft_clamp(target, ceiling, w):
    """target kept under ceiling smoothly: ceiling - g(ceiling - target) with g(d) = d for d >= w, 0 for d <= -w and
    (d + w)^2 / (4 w) between (continuous with its slope, never above target nor ceiling; exactly target wherever
    it lies w or more below the ceiling)."""
    target, ceiling = np.asarray(target, np.float64), np.asarray(ceiling, np.float64)
    d = ceiling - target
    g = np.where(d >= w, d, np.where(d <= -w, 0.0, (d + w) ** 2 / (4 * w)))
    return ceiling - g


def standing_legs(base, dx=None, dz=None, roll_deg=None, yaw_deg=None, tilt_deg=None, dip_m=None, lift=None,
                  smooth_s=PELVIS_SMOOTH_S, keep_height=False, heading=None, heel_deg=None):
    """(Body, info): a standing base with both feet planted (module docstring, Feet): the pelvis moved by dx (to the
    body's left) and dz (forward) (T,) m, turned by roll_deg (about the heading axis, positive to the right), yaw_deg
    (about the vertical, to the left) and tilt_deg (about the lateral axis, forward); its height the base's minus
    dip_m low-passed (PELVIS_SIGMA_S), held under the highest height from which both legs reach their ankles at
    base.REACH of the leg length (smoothed below it over smooth_s, then by REACH_SIGMA_S, less REACH_SAFETY_M) by
    soft_clamp (SOFT_KNEES_M wide): the pelvis
    no longer rides at the reachable height, whose quick changes snapped the knees (clinical review of 2026-11-02),
    and does not rise when a foot is lifted; lift (T, 2) m lifts the left or right ankle vertically (the foot flat,
    marching in place). keep_height: the base's pelvis height minus dip_m, only lowered where the legs do not reach
    (a second pass on a body already solved); heading: a body whose pelvis heading sets the forward direction (the
    level 0's, so that a second pass keeps the knees' hinge axes). The ankles otherwise stay at the base's, the feet
    keep their global rotations; the legs are solved by ik.hinge_leg, knees bending forward. heel_deg (T, 2), if
    given: each foot pitched heel up about its ball (the foot joint) by that angle about the body's lateral axis
    (heel_off), the toes keeping their global rotation, before the lift."""
    T = base.T
    z = np.zeros(T)
    dx, dz, roll, yaw, tilt, dip = (z if v is None else np.asarray(v, float) for v in
                                    (dx, dz, roll_deg, yaw_deg, tilt_deg, dip_m))
    lift = np.zeros((T, 2)) if lift is None else np.asarray(lift, float)
    J, G = base.fk()
    feet = None
    if heel_deg is not None and np.abs(heel_deg).max() > 0.0:
        feet = heel_off(J, G, base if heading is None else heading, np.asarray(heel_deg, float))
        J, G = J.clone(), G.clone()
        for i, s in enumerate(("left", "right")):
            a, f = LEG[s][2], LEG[s][3]
            J[:, a], G[:, a] = feet[i][0], feet[i][1]
    rest = base.rest()
    R = base.local_R().clone()
    G0 = G[:, 0]
    fwd = (G0 if heading is None else heading.fk()[1][:, 0])[:, :, 2].clone()
    fwd[:, 1] = 0
    fwd = fwd / fwd.norm(dim=-1, keepdim=True)
    up = torch.tensor([0.0, 1.0, 0.0], dtype=F64).expand(T, 3)
    left = torch.cross(up, fwd, dim=-1)
    E = (axis_angle_to_matrix(fwd * torch.from_numpy(np.radians(roll))[:, None])
         @ axis_angle_to_matrix(up * torch.from_numpy(np.radians(yaw))[:, None])
         @ axis_angle_to_matrix(left * torch.from_numpy(np.radians(tilt))[:, None]))
    G0n = E @ G0
    P = J[:, 0] + torch.from_numpy(dx)[:, None] * left + torch.from_numpy(dz)[:, None] * fwd
    hips = [torch.einsum("tij,j->ti", G0n, rest[LEG[s][0]] - rest[0]) for s in ("left", "right")]
    A = [J[:, LEG[s][2]] + torch.from_numpy(lift[:, i])[:, None] * up for i, s in enumerate(("left", "right"))]
    Ls = [float((rest[LEG[s][1]] - rest[LEG[s][0]]).norm() + (rest[LEG[s][2]] - rest[LEG[s][1]]).norm())
          for s in ("left", "right")]
    hs = []
    for i in range(2):
        d2 = ((P[:, [0, 2]] + hips[i][:, [0, 2]] - A[i][:, [0, 2]]) ** 2).sum(-1)
        hs.append(A[i][:, 1] + torch.sqrt(((REACH * Ls[i]) ** 2 - d2).clamp_min(0.0)) - hips[i][:, 1])
    raw = torch.minimum(hs[0], hs[1]).numpy()
    reach = (gauss_smooth(smooth_below(raw, max(1, int(round(smooth_s * base.rate)))), REACH_SIGMA_S * base.rate)
             - REACH_SAFETY_M)
    if keep_height:                                      # the base's own height, under the reachable one
        yv = soft_clamp(J[:, 0, 1].numpy() - dip, reach, SOFT_KNEES_M)
    else:
        yv = soft_clamp(gauss_smooth(J[:, 0, 1].numpy() - dip, PELVIS_SIGMA_S * base.rate), reach, SOFT_KNEES_M)
    yv = np.minimum(yv, raw - 1e-6)                      # never above the reachable height (rarely binding)
    P[:, 1] = torch.from_numpy(yv)
    R[:, 0] = G0n
    for i, s in enumerate(("left", "right")):
        h, kn, a, _ = LEG[s]
        Hh = P + hips[i]
        G_th, kap = hinge_leg(Hh, A[i], Hh + fwd, rest[kn] - rest[h], rest[a] - rest[kn], G0n,
                              hinge=torch.cross(fwd, A[i] - Hh, dim=-1))
        Rk = axis_angle_to_matrix(torch.stack([kap, 0 * kap, 0 * kap], -1))
        R[:, h] = G0n.transpose(-1, -2) @ G_th
        R[:, kn] = Rk
        R[:, a] = (G_th @ Rk).transpose(-1, -2) @ G[:, a]
    rows = [0, 1, 2, 4, 5, 7, 8]
    if feet is not None:                                 # the toes keep their global rotation (flat on the floor)
        for i, s in enumerate(("left", "right")):
            a, f = LEG[s][2], LEG[s][3]
            on = torch.from_numpy(np.asarray(heel_deg, float)[:, i] != 0.0)
            R[:, f] = torch.where(on[:, None, None], G[:, a].transpose(-1, -2) @ G[:, f], R[:, f])
        rows = rows + [10, 11]
    out = base.with_rotations(R, rows)
    out = replace(out, transl=P - rest[0])
    out = replace(out, contact=contact_labels(out))
    return out, {"pelvis_y_range_m": [float(yv.min()), float(yv.max())]}


def heel_off(J, G, heading, heel_deg):
    """[(ankle (T, 3), ankle global rotation (T, 3, 3))] per foot (left, right): the foot of J, G (forward kinematics)
    pitched heel up by heel_deg (T, 2) about the body's lateral axis (heading's pelvis frame) through its ball (the
    foot joint), then moved horizontally so that its foot point (the mean of the ankle and foot joints,
    body.foot_points, the point the skate check reads) stays where it was: the foot rolls over its ball."""
    fwd = heading.fk()[1][:, 0][:, :, 2].clone()
    fwd[:, 1] = 0
    fwd = fwd / fwd.norm(dim=-1, keepdim=True)
    up = torch.tensor([0.0, 1.0, 0.0], dtype=F64).expand(J.shape[0], 3)
    left = torch.cross(up, fwd, dim=-1)
    out = []
    for i, s in enumerate(("left", "right")):
        a, f = LEG[s][2], LEG[s][3]
        Rp = axis_angle_to_matrix(left * torch.from_numpy(np.radians(heel_deg[:, i]))[:, None])
        rel = J[:, a] - J[:, f]
        new = torch.einsum("tij,tj->ti", Rp, rel)
        c = 0.5 * (rel - new)
        c[:, 1] = 0.0
        out.append((J[:, f] + c + new, Rp @ G[:, a]))
    return out


def tilt_pelvis_seated(base, tilt_deg):
    """(Body): a seated base whose pelvis turns forward by tilt_deg (T,) about the line through its two hip joints,
    the thighs' global rotations kept (the legs and feet do not move; module docstring)."""
    J, G = base.fk()
    rest = base.rest()
    ax = J[:, 1] - J[:, 2]
    ax = ax / ax.norm(dim=-1, keepdim=True)                # toward the body's left
    C = 0.5 * (J[:, 1] + J[:, 2])
    Rt = axis_angle_to_matrix(ax * torch.from_numpy(np.radians(np.asarray(tilt_deg, float)))[:, None])
    G0n = Rt @ G[:, 0]
    P = C + torch.einsum("tij,tj->ti", Rt, J[:, 0] - C)
    R = base.local_R().clone()
    R[:, 0] = G0n
    for h in (1, 2):
        R[:, h] = G0n.transpose(-1, -2) @ G[:, h]
    out = base.with_rotations(R, [0, 1, 2])
    return replace(out, transl=P - rest[0])


def stand_soft(duration_s, rate=100.0):
    """The standing level 0 of this module: base.static_body('stand') with the pelvis STAND_DIP_M below the reach
    height of standing_legs (knees soft at base.REACH), the base every standing operator here moves."""
    b, _ = standing_legs(static_body("stand", duration_s, rate))
    b, _ = standing_legs(b, dip_m=np.full(b.T, STAND_DIP_M))
    return b


# ---------------------------------------------------------------- standing: a step in place over a shifted weight
# The clinical review of 2026-11-03 (M12): the standing foot lifts of the diphasic stepping, the tardive marching and
# the akathisia march and shuffle left the centre of mass 73 to 84 mm (median) lateral of the stance foot (the pelvis
# swayed 3 cm, a sine unrelated to the steps) and lifted heel and toe together. An alternating step in place now runs
# in each half cycle: the weight goes over the stance foot (the first STEP_SHIFT_SHARE of the half cycle, both feet
# down), the other foot's heel rising with its ball on the floor over the end of the shift (HEEL_SHARE of the half
# cycle, from HEEL_LEAD before the shift ends, up to HEEL_DEG), then the foot lifts and comes down onto its ball at the
# half cycle's end, and its heel lowers over the first HEEL_SHARE of the next half cycle while the weight comes back
# onto it; the weight stays on the stance foot until the foot is down.
# "The weight over the stance foot" is read as the centre of pressure, the point the body rests on, which must lie
# under the stance foot while the other foot is up (hof2005: the base of support is the area to which the centre of
# pressure is confined). The centre of mass need not: in the inverted pendulum of hof2005 (eq. 2) the centre of
# pressure u and the centre of mass x (their projections on the floor) obey u = x - x''/omega0^2, omega0 =
# sqrt(g / l), and a body stepping at about 1 Hz rocks its centre of mass by a fraction of the feet's spacing while the
# centre of pressure goes from foot to foot (hof2005 fig. 5, walking: the lateral centre of mass stays between the
# feet). So step_track lays down the centre of pressure (from the midline to the stance foot's centre during the
# shift, under it while the other foot is up) and the centre of mass is the pendulum's bounded answer to it (lip_com),
# carried by the pelvis (step_geometry 'com_gain'). A first answer of 2026-11-04 put the centre of mass itself over
# the stance foot (the review plan's reading): the pelvis then crossed 18 cm in 0.28 s at 0.8 steps per second and
# carried the hands at 1.2 to 2.2 m/s, a lurch no pendulum of the body's length can make (the centre of pressure its
# acceleration asks for lies far outside the feet).
SOURCES["hof2005"] = ("Hof AL, Gazendam MGJ, Sinke WE 2005, The condition for dynamic stability, J Biomech 38: 1-8, "
                      "doi 10.1016/j.jbiomech.2004.03.025 (full text): eq. 2, u = x - x''/omega0^2 with omega0 = "
                      "sqrt(g/l); section 2.2, l about 1.34 times the trochanteric height for movements in the frontal "
                      "plane (Massen and Kodde 1979); fig. 5, lateral centre of pressure and centre of mass in walking")
STEP_SHIFT_SHARE = 0.3      # [module] the share of each half cycle in which the weight shifts, both feet down
HEEL_SHARE = 0.2            # [module] the share of a half cycle over which the stepping heel rises before the ball
#                             leaves the floor (and over which it lowers after the ball lands)
HEEL_LEAD = 0.1             # [module] the heel starts to rise this share of the half cycle before the shift ends: the
#                             ball is up for 1 - STEP_SHIFT_SHARE - HEEL_SHARE + HEEL_LEAD = 0.6 of the half cycle (at
#                             0.65 the pelvis of the akathisia's march had come 13.6 mm toward the stance foot when the
#                             ball left, against 15 mm or more at 0.6; the
#                             review of wave SH1d: with the shift, the heel's rise and its lowering in turn, 0.25 each,
#                             the ball was up 0.375 of the half cycle, 0.13 to 0.20 s above 10 mm, and the ankle
#                             accelerated at 38 to 51 m/s^2, against 3.6 to 9.0 before the steps had a shift)
STEP_BALL_ACCEL_M_S2 = 20.0  # [module] a step's lift is cut where its sine-squared bump (height A over D seconds) would
#                             accelerate the ball by more than this (its peak, 2 pi^2 A / D^2): a fast cycle of the
#                             akathisia's jittered march (rate and size each up to 1.4 times) read 29 m/s^2 at the ankle
MJ_PEAK_ACC = 10.0 / math.sqrt(3.0)   # a minimum-jerk move of D over T peaks at this times D / T^2 (minimum_jerk)
STEP_ACCEL_CAP_M_S2 = 25.0  # [module] the stepping ankle's acceleration bound the tests read (the page's steps of
#                             before this wave reached 3 to 17 m/s^2 at 30 fps; the heel's pitch and the pelvis add
#                             to the ball's bump)
HEEL_DEG = 15.0             # [module] the foot's pitch about its ball at heel-off, for a lift of HEEL_FULL_LIFT_M or
#                             more; a smaller lift pitches the foot in proportion (a small step barely peels the heel)
HEEL_FULL_LIFT_M = 0.06     # [module]
SHIFT_FULL_ENV = 0.3        # [module] the shift reaches its full size where the bout's envelope reaches this share: a
#                             foot lifted on the bout's ramp already has the weight over the other foot
STEP_UP_M = 0.01            # [module] stance_report: single support where a foot's ball is this far above its level 0
G_M_S2 = 9.81               # standard gravity (m/s^2)
PENDULUM_TROCHANTER = 1.34  # hof2005 section 2.2: the frontal-plane pendulum length over the trochanteric height (here
#                             the hip joints' height over the soles of the level 0)
LIP_SPAN = 8.0              # [module] lip_com's kernel reaches this many time constants (1/omega0) each way
COP_STANCE_TOL_M = 0.025    # [module] stance_report: the realized centre of pressure within this of the stance foot's
#                             centre while the other foot is up (the sole reaches 38 and 40 mm medial of the left and
#                             right foot points on the rest mesh: inside the sole with room for the pendulum's error)


def body_com(body):
    """(T, 3) m: the centre of mass of the de Leva 1996 segments on the body joints (kernels.force.com, the body's sex
    code; the neutral shape takes the mean of the female and male fractions)."""
    from ..kernels.force import com
    return com(body.joints().numpy(), int(body.gender))


def _lateral_frame(base):
    """(T, 3) unit: the body's left axis in the horizontal plane (the pelvis heading of base)."""
    fwd = base.fk()[1][:, 0][:, :, 2].clone()
    fwd[:, 1] = 0
    fwd = fwd / fwd.norm(dim=-1, keepdim=True)
    return torch.cross(torch.tensor([0.0, 1.0, 0.0], dtype=F64).expand(base.T, 3), fwd, dim=-1).numpy()


def step_geometry(base):
    """{'half_width_m', 'com_gain', 'pendulum_m', 'omega0', 'ankle_ball_m'} of a standing base, read at its first
    knot (the ankle-to-ball distance, the mean of the two feet, for step_track's heel): the mean
    lateral distance (the body's left axis) of the two foot points (body.foot_points) from the centre of mass
    (body_com); the centre of mass's lateral move per metre of pelvis shift (standing_legs dx, both feet planted, the
    mean of a 5 cm shift each way); the pendulum length PENDULUM_TROCHANTER times the hip joints' height over the
    lowest skin vertex (the soles), and omega0 = sqrt(G_M_S2 / length) (hof2005)."""
    from . import viewer as V
    from .body import foot_points
    b2 = Body(rot6d=base.rot6d[:2], transl=base.transl[:2], contact=base.contact[:2], betas=base.betas,
              gender=base.gender, rate=base.rate)
    left = _lateral_frame(b2)[0]
    F = foot_points(b2.joints())[0].numpy()
    c0 = body_com(b2)[0] @ left
    k = []
    for sg in (1.0, -1.0):
        b1, _ = standing_legs(b2, dx=np.full(2, sg * 0.05), keep_height=True)
        k.append((body_com(b1)[0] @ left - c0) / (sg * 0.05))
    J = b2.joints().numpy()[0]
    sole = float(V.knot_vertices(b2, [0])[0][:, 1].min())
    length = PENDULUM_TROCHANTER * (0.5 * (J[1, 1] + J[2, 1]) - sole)
    return {"half_width_m": float(np.mean([abs(F[i] @ left - c0) for i in (0, 1)])), "com_gain": float(np.mean(k)),
            "pendulum_m": float(length), "omega0": float(math.sqrt(G_M_S2 / length)),
            "ankle_ball_m": float(np.mean([np.linalg.norm(J[a] - J[f]) for a, f in ((7, 10), (8, 11))]))}


def lip_com(p, omega0, rate):
    """(T,) the centre of mass of the inverted pendulum (hof2005 eq. 2, x - x''/omega0^2 = p) that keeps the centre
    of pressure on p (T,), its bounded answer: p smoothed by the two-sided kernel (omega0 / 2) exp(-omega0 |t|)
    (LIP_SPAN time constants each way, normalised), the ends of p held."""
    p = np.asarray(p, float)
    n = int(math.ceil(LIP_SPAN / omega0 * rate))
    tau = np.arange(-n, n + 1) / rate
    w = np.exp(-omega0 * np.abs(tau))
    w /= w.sum()
    return np.convolve(np.r_[np.full(n, p[0]), p, np.full(n, p[-1])], w, mode="valid")


def step_track(ph, am, env, lift_m, geo, rate, heel_deg=HEEL_DEG):
    """(lift (T, 2) m, heel (T, 2) deg, dx (T,) m, cop (T,) m): an alternating step in place (comment above
    STEP_SHIFT_SHARE) on a phase track ph (rad, phase_track: the left foot steps in the first half of each cycle, the
    right in the second) at rate (knots per s), the lift of the stepping foot's ball lift_m times the cycle's
    amplitude am and the envelope env (a sine-squared bump from the heel's full pitch to the half cycle's end), its
    heel pitched up to heel_deg (times lift_m over HEEL_FULL_LIFT_M, at most 1) over HEEL_SHARE from HEEL_LEAD before
    the shift ends, and lowered over the first HEEL_SHARE of the next half cycle; cop the centre of pressure laid down
    (to the body's left of the level 0's centre of mass): from one foot's centre to the other's (geo 'half_width_m',
    a minimum-jerk move over STEP_SHIFT_SHARE) while both feet are down, under the stance foot while the other is up,
    scaled by the envelope's share (full where env reaches SHIFT_FULL_ENV); dx the pelvis shift that carries the
    pendulum's centre of mass for it (lip_com over geo 'com_gain'). The ball lifts only over a full shift (its
    envelope rising from 0 where env reaches SHIFT_FULL_ENV to 1 where env does: on the bout's ramps the feet lifted
    with the shift partial and the heel down, 2026-11-04)."""
    u = np.mod(np.asarray(ph, float) / (2 * math.pi), 1.0)
    first = u < 0.5
    v = np.where(first, 2 * u, 2 * u - 1.0)
    ts, hs = STEP_SHIFT_SHARE, HEEL_SHARE
    h0 = ts - HEEL_LEAD                                    # the heel starts to rise
    b0 = h0 + hs                                           # the ball leaves the floor
    sgn = np.where(first, -1.0, 1.0)                      # the left foot steps: the weight on the right foot
    shift = -sgn + 2 * sgn * minimum_jerk(v / ts)
    heel_deg = heel_deg * min(1.0, lift_m / HEEL_FULL_LIFT_M)
    full = minimum_jerk(np.asarray(env, float) / SHIFT_FULL_ENV)
    step_env = minimum_jerk((np.asarray(env, float) - SHIFT_FULL_ENV) / (1.0 - SHIFT_FULL_ENV))
    lift = np.zeros((len(u), 2))
    heel = np.zeros((len(u), 2))
    for i in (0, 1):
        w = 2.0 * np.mod(u - 0.5 * i, 1.0)                 # [0, 1): this foot's stepping half, [1, 2): its stance half
        stepping = w < 1.0
        y = np.clip((w - b0) / (1.0 - b0), 0.0, 1.0)
        up = np.sin(math.pi * y) ** 2 * stepping
        pitch = np.where(stepping, minimum_jerk((w - h0) / hs), 1.0 - minimum_jerk((w - 1.0) / hs))
        lf = lift_m * np.asarray(am, float) * step_env * up
        for a, b in _runs(up > 0.0):                     # one size per step: the envelope's least over the step (on
            lf[a:b] = lift_m * np.asarray(am, float)[a:b] * step_env[a:b].min() * up[a:b]   # a bout's ramp,
            top = lf[a:b].max()                          # cutting the bump within the step raised the ankle's
            if top > 0.0:                                # acceleration from 12 to 20 m/s^2), at most STEP_BALL_ACCEL
                D = (b - a) / float(rate)
                lf[a:b] *= min(1.0, STEP_BALL_ACCEL_M_S2 * D * D / (2.0 * math.pi ** 2 * top))
        lift[:, i] = lf
        fh = full.copy()
        for a, b in _runs(pitch > 0.0):                  # and one heel pitch per step, the largest the envelope
            fh[a:b] = full[a:b].max()                    # gives it over the step (a falling envelope lowered the heel
            if "ankle_ball_m" in geo and heel_deg > 0.0:   # at twice its pace), its rise over HEEL_SHARE of the step
                th = hs * (b - a) / float(rate)          # (the run spans one half cycle) at most STEP_BALL_ACCEL at
                top = STEP_BALL_ACCEL_M_S2 * th * th / (MJ_PEAK_ACC * geo["ankle_ball_m"] * math.radians(heel_deg))
                fh[a:b] = min(fh[a:b].max(), top)        # the ankle (a fast cycle of small steps read 25.6 m/s^2)
        heel[:, i] = heel_deg * fh * pitch
    cop = geo["half_width_m"] * full * shift
    return lift, heel, lip_com(cop, geo["omega0"], rate) / geo["com_gain"], cop


def stance_report(body, base, up_m=STEP_UP_M, omega0=None):
    """{side: {...}}: per stepping foot, the knots where its ball (the foot joint) is up_m or more above its place in
    base (single support on the other foot), and there the lateral distance (the body's left axis of base) from the
    stance foot's centre (body.foot_points) of the centre of pressure the body's centre of mass asks for (hof2005 eq.
    2, x - x''/omega0^2, omega0 from step_geometry(base) unless given; central differences) and of the centre of mass
    itself (body_com): median and largest, mm; and the lifted foot's heel rise over its ball's rise at the knot where
    the ball leaves (heel before toe), JSON-ready."""
    from .body import foot_points
    J, Jb = body.joints().numpy(), base.joints().numpy()
    left = _lateral_frame(base)
    w0 = step_geometry(base)["omega0"] if omega0 is None else omega0
    C = body_com(body)
    xc = (C * left).sum(-1)
    dt = 1.0 / float(body.rate)
    cop = xc - np.gradient(np.gradient(xc, dt), dt) / w0 ** 2
    F = foot_points(torch.from_numpy(J)).numpy()
    out = {}
    for i, s, j in ((0, "left", 10), (1, "right", 11)):
        upk = (J[:, j, 1] - Jb[:, j, 1]) >= up_m
        rec = {"single_support_share": float(upk.mean())}
        if upk.any():
            fs = (F[:, 1 - i] * left).sum(-1)
            d = np.abs(xc - fs)[upk]
            e = np.abs(cop - fs)[upk]
            rec.update(com_to_stance_mm_median=float(1000 * np.median(d)), com_to_stance_mm_max=float(1000 * d.max()),
                       cop_to_stance_mm_median=float(1000 * np.median(e)), cop_to_stance_mm_max=float(1000 * e.max()))
            a = LEG[s][2]
            k = [a_ for a_, _ in _runs((J[:, j, 1] - Jb[:, j, 1]) > 1e-4)]
            rise = [(J[q, a, 1] - Jb[q, a, 1]) for q in k]
            rec["ankle_rise_at_ball_off_mm_min"] = float(1000 * min(rise)) if rise else None
        out[s] = rec
    out["omega0"] = float(w0)
    out["rule"] = (f"while a ball is {1000 * up_m:g} mm or more up, the centre of pressure of the pendulum (hof2005) "
                   f"within {1000 * COP_STANCE_TOL_M:g} mm of the stance foot's centre (module)")
    out["ok"] = bool(all(out[s].get("cop_to_stance_mm_max", 0.0) <= 1000 * COP_STANCE_TOL_M for s in ("left", "right")))
    return out


# ---------------------------------------------------------------- fingers
FINGER_JOINTS = {f: (("mcp", 0), ("pip", 1), ("dip", 2)) if f != "thumb" else (("cmc", 0), ("mcp", 1), ("ip", 2))
                 for f in H.FINGERS}
FINGER_SPLIT = (0.35, 0.45, 0.2)        # [module] a finger's flexion over its three joints, the PIP leading
FINGER_COUPLING = (("index", "middle", 0.3), ("middle", "ring", 0.4), ("ring", "pinky", 0.6))
# [module] neighbouring fingers move together (the share of a neighbour's raw movement added to a finger's): no long
# finger flexes alone while the others stay straight (clinical review of 2026-11-02)
FINGER_GAP_DEG = 50.0       # [module] a long finger's flexion stays within this of its neighbours'
FINGER_LEVEL_CAP = 0.4      # [module] finger events at most this share of a finger's flexion range
FINGER_TIMING = {"rise": (0.15, 0.35), "hold": (0.0, 0.1), "ret": (0.15, 0.35), "next": (0.4, 0.9), "n_ch": (1, 2),
                 "amp": (0.3, 1.0), "mode": "flow", "rest_p": 0.3, "dwell_max": 0.3}   # [module] finger fragments


def finger_limits(finger):
    """(flex lo, flex hi) deg of the finger's whole flexion (the sum over its joints) at FINGER_SPLIT, the smallest
    joint bound over its split (hands.FLEX_LIMITS_DEG; the thumb CMC by hands.CMC_LIMIT_DEG)."""
    lo, hi = -1e9, 1e9
    for (name, lv), s in zip(FINGER_JOINTS[finger], FINGER_SPLIT):
        a, b = (-H.CMC_LIMIT_DEG, H.CMC_LIMIT_DEG) if (finger == "thumb" and name == "cmc") else \
            H.FLEX_LIMITS_DEG[finger][name]
        lo, hi = max(lo, a / s), min(hi, b / s)
    return lo, hi


def apply_fingers(body, side, flex, abd=None, margin=SOFT_MARGIN, limited=False):
    """(HandBody, info): flexion (T,) deg per finger (dict finger -> whole flexion over FINGER_SPLIT, positive
    toward the palm) and knuckle abduction (dict finger -> deg) on top of the body's hand pose, soft-limited inside
    the hand limits (module docstring); limited: the flexion was soft-limited by the caller (not limited again)."""
    hb = H.hands_of(body) if not H.has_hands(body) else body.copy()
    fr = H.hand_frame()[side]["joints"]
    Rh = hb.hand_R().clone()
    info = {}
    for f, x in flex.items():
        lo, hi = finger_limits(f)
        xl = np.asarray(x, float) if limited else soft_limit(np.asarray(x, float), hi, -lo, margin)
        info[f] = {"flex_range_deg": [float(xl.min()), float(xl.max())], "limits_deg": [lo, hi]}
        ab = None if abd is None or f not in abd else soft_limit(np.asarray(abd[f], float), H.ABDUCTION_LIMIT_DEG,
                                                                 H.ABDUCTION_LIMIT_DEG, margin)
        for (name, lv), s in zip(FINGER_JOINTS[f], FINGER_SPLIT):
            j = H.hand_joint(side, f, lv)
            M = axis_angle_to_matrix(torch.as_tensor(fr[j]["flex"])[None]
                                     * torch.deg2rad(torch.as_tensor(s * xl))[:, None])
            if ab is not None and fr[j]["abd"] is not None and lv == 0:
                M = M @ axis_angle_to_matrix(torch.as_tensor(fr[j]["abd"])[None]
                                             * torch.deg2rad(torch.as_tensor(ab))[:, None])
            Rh[:, H.hand_row(j)] = M @ Rh[:, H.hand_row(j)]
    rows = list(range(0, 15)) if side == "left" else list(range(15, 30))
    return hb.with_hands(Rh, rows), info


def finger_values(body, base, side):
    """{finger: (T,) deg} the whole flexion of each finger read back (the sum of its joints' flexion components
    against the base hand pose, hands.hand_components)."""
    Rb = base.hand_R() if H.has_hands(base) else torch.eye(3, dtype=F64).repeat(body.T, H.N_HAND, 1, 1)
    D = body.hand_R() @ Rb.transpose(-1, -2)
    comp = H.hand_components(D)
    return {f: sum(comp[H.hand_joint(side, f, lv)][0] for _, lv in FINGER_JOINTS[f]) for f in H.FINGERS}


# ---------------------------------------------------------------- re-measurement
def _region_offsets(body, base, joints):
    R, Rb = body.local_R(), base.local_R()
    return np.concatenate([_log_deg(R[:, j] @ Rb[:, j].transpose(-1, -2)) for j in joints], 1)


def _speed_deg_s(W, rate):
    """(T,) the largest over joints of the norm of d(offset)/dt (deg/s), W (T, 3 n)."""
    v = np.gradient(W, 1.0 / rate, axis=0).reshape(len(W), -1, 3)
    return np.linalg.norm(v, axis=-1).max(1)


REGION_PARENT = {"trunk": 0, "neck": 9, "arm_left": 9, "arm_right": 9, "leg_left": None, "leg_right": None}


def _end_disp_cm(Jo, Go, Jb, Gb, r):
    """(T,) cm: the region's end joint displaced from the base's, in the frame of the region's parent segment (the
    region's own movement; the legs in the world frame, their hips fixed or the feet planted)."""
    e, p = END_JOINT[r], REGION_PARENT[r]
    if p is None:
        return 100 * np.linalg.norm(Jo[:, e] - Jb[:, e], axis=-1)
    ro = np.einsum("tji,tj->ti", Go[:, p], Jo[:, e] - Jo[:, p])
    rb = np.einsum("tji,tj->ti", Gb[:, p], Jb[:, e] - Jb[:, p])
    return 100 * np.linalg.norm(ro - rb, axis=-1)


def _dominant(W, rate, derivative=0, band=(0.1, 8.0)):
    from .measure import dominant_frequency
    X = np.asarray(W, float)
    for _ in range(derivative):
        X = np.gradient(X, 1.0 / rate, axis=0)
    return float(dominant_frequency(X, rate, band))


def measure_regions(body, base, regions=None, channels=None):
    """{region: {share, amplitude_share, amplitude_channel, amplitude_deg, channel_shares, end_cm_p95, dominant_hz
    (of the offsets), dominant_acc_hz (of their second derivative, the quantity an accelerometer reads), speed_deg_s
    (mean while moving), involved}} read from the body against its level-0 base (module docstring); 'pelvis' (cm and
    deg of the root against the base) and 'n_parts' (involved regions)."""
    regions = list(REGIONS) if regions is None else list(regions)
    Jo, Go = (x.numpy() for x in body.fk())
    Jb, Gb = (x.numpy() for x in base.fk())
    out = {}
    for r in regions:
        W = _region_offsets(body, base, REGIONS[r])
        sp = _speed_deg_s(W, body.rate)
        mv = close_mask(sp > V_MOVE_DEG_S, body.rate)
        chs = [c for c in (channels or CHANNELS) if CHANNELS[c]["region"] == r]
        shares = {c: extent_share(channel_values(body, base, c), channel_rom(c)) for c in chs}
        top = max(shares, key=shares.get) if shares else None
        amp = shares[top] if top else 0.0
        rec = {"share": float(mv.mean()), "amplitude_share": float(amp), "amplitude_channel": top,
               "amplitude_deg": float(amp * channel_rom(top)) if top else 0.0, "channel_shares": shares,
               "end_cm_p95": float(np.percentile(_end_disp_cm(Jo, Go, Jb, Gb, r), 95)),
               "speed_deg_s": float(sp[mv].mean()) if mv.any() else 0.0, "involved": bool(amp > PART_SHARE)}
        rec["dominant_hz"] = _dominant(W, body.rate) if rec["involved"] else None
        rec["dominant_acc_hz"] = _dominant(W, body.rate, 2) if rec["involved"] else None
        out[r] = rec
    dp = Jo[:, 0] - Jb[:, 0]
    D = np.einsum("tji,tjk->tik", Gb[:, 0], Go[:, 0])
    ang = np.degrees(np.arccos(np.clip((np.trace(D, axis1=1, axis2=2) - 1) / 2, -1, 1)))
    out["pelvis"] = {"displacement_cm_max": float(100 * np.linalg.norm(dp, axis=-1).max()),
                     "displacement_cm_pp_lateral": float(100 * (dp[:, 0].max() - dp[:, 0].min())),
                     "displacement_cm_pp_forward": float(100 * (dp[:, 2].max() - dp[:, 2].min())),
                     "rotation_deg_max": float(ang.max())}
    out["n_parts"] = int(sum(1 for r in regions if out[r]["involved"]))
    return out


def measure_fingers(body, base, side):
    """{share, flex_extent_deg (per finger), amplitude_share (largest extent over the finger's flexion range),
    dominant_hz, n_fingers} of one hand against the base hand pose."""
    v = finger_values(body, base, side)
    X = np.stack([v[f] for f in H.FINGERS], 1)
    sp = np.abs(np.gradient(X, 1.0 / body.rate, axis=0)).max(1)
    mv = close_mask(sp > V_MOVE_DEG_S, body.rate)
    ext = {f: float(v[f].max() - v[f].min()) for f in H.FINGERS}
    shares = {f: ext[f] / (finger_limits(f)[1] - finger_limits(f)[0]) for f in H.FINGERS}
    n = sum(1 for f in H.FINGERS if shares[f] > PART_SHARE)
    return {"share": float(mv.mean()), "flex_extent_deg": ext, "amplitude_share": float(max(shares.values())),
            "dominant_hz": _dominant(X, body.rate) if n else None,
            "dominant_acc_hz": _dominant(X, body.rate, 2) if n else None, "n_fingers": int(n)}


def remeasure(body, base, truth):
    """The re-measurement of a clip of this module (module docstring) for the regions and hands its truth names."""
    p = truth["params"]
    out = {"regions": measure_regions(body, base, p.get("regions_measured"))}
    for side in p.get("hands", ()):
        out[f"fingers_{side}"] = measure_fingers(body, base, side)
    return out


# ---------------------------------------------------------------- the engine
def _solve_gain(build, level, lo=1e-3, hi=30.0, iters=34):
    """The gain g whose realized amplitude build(g) equals level (bisection on log g; build monotone in g). Returns
    (g, amplitude, reachable)."""
    a_hi = build(hi)
    if a_hi < level:
        return hi, a_hi, False
    a, b = math.log(lo), math.log(hi)
    for _ in range(iters):
        m = 0.5 * (a + b)
        if build(math.exp(m)) < level:
            a = m
        else:
            b = m
    g = math.exp(b)
    return g, build(g), True


def region_signals(base, chans, raw, level, gains=None, margin=SOFT_MARGIN):
    """({channel: (T,) deg}, info): raw signals {channel: (T,)} (unit amplitude ~1) scaled by one gain so that the
    region's amplitude (the largest peak-to-peak excursion over ROM over its channels, after the soft limits) equals
    level (a share of the ROM); gains {channel: w} weight each channel's reference amplitude (w times its ROM);
    margin: the soft limit's share of the room in each direction (TRUNK_NECK_MARGIN for the axial regions)."""
    rooms = {c: channel_room(base, c) for c in chans}
    roms = {c: channel_rom(c) for c in chans}
    w = {c: (gains or {}).get(c, 1.0) for c in chans}

    def make(g):
        return {c: soft_limit(g * w[c] * roms[c] * raw[c], *rooms[c], margin=margin) for c in chans}

    def amp(g):
        s = make(g)
        return max(extent_share(s[c], roms[c]) for c in chans)
    if level <= 0:
        return {c: np.zeros_like(raw[c]) for c in chans}, {"gain": 0.0, "reachable": True, "level": 0.0}
    g, a, ok = _solve_gain(amp, level)
    return make(g), {"gain": g, "level": float(level), "realized_before_table": float(a), "reachable": bool(ok),
                     "rooms_deg": {c: [float(x) for x in rooms[c]] for c in chans}, "soft_margin": float(margin),
                     "roms_deg": {c: roms[c] for c in chans}}


def room_split(body, name, sign=1.0, k=0):
    """([(joint, axis (3,) unit in the parent frame, share)], room deg): a channel's chain split over its joints by
    each joint's room in the direction sign at knot k (body.room_deg), the shares summing to 1, and the chain's room
    (their sum): the chain bends as a whole, each joint reaching its bound together (open-body port, wave 3
    follow-up, D40: the seated trunk's lumbar joint stands 2.5 deg from its flexion bound, so channel_axes' split by
    ROM left the chain 5 deg of the 25 the spine has forward)."""
    axes = channel_axes(body, name, k)
    R = body.local_R()[k]
    rooms = np.array([room_deg(R[j], j, a, sign, "parent", max_deg=180.0) for j, a, _ in axes])
    tot = float(rooms.sum())
    return [(j, a, float(r / tot) if tot > 0 else 0.0) for (j, a, _), r in zip(axes, rooms)], tot


def given_room_offsets(base, name, raw, deg):
    """({joint: (T, 3) deg offset}, info): a one-signed raw signal (T,) (0 to 1) times deg on one channel, soft-limited
    against the chain's room in its direction and split over the chain by room (room_split), not by ROM."""
    sign = 1.0 if deg >= 0 else -1.0
    split, room = room_split(base, name, sign)
    sig = sign * soft_limit(abs(deg) * np.asarray(raw, np.float64), room, room)
    offs = {j: s * sig[:, None] * a[None] for j, a, s in split}
    loss = float(np.abs(np.abs(sig) - abs(deg) * np.asarray(raw)).max())
    return offs, sig, {"deg_per_unit": {name: float(deg)}, "soft_limit_loss_deg": loss, "reachable": bool(loss < 0.05),
                       "level": None, "split": "room", "room_deg": room,
                       "split_shares": {int(j): round(s, 4) for j, _, s in split}}


def given_signals(base, chans, raw, deg):
    """({channel: (T,) deg}, info): raw signals times stated degrees {channel: deg per unit}, soft-limited."""
    rooms = {c: channel_room(base, c) for c in chans}
    sig = {c: soft_limit(deg[c] * raw[c], *rooms[c]) for c in chans}
    loss = max(float(np.abs(sig[c] - deg[c] * raw[c]).max()) for c in chans)
    return sig, {"deg_per_unit": {c: float(deg[c]) for c in chans}, "soft_limit_loss_deg": loss,
                 "reachable": bool(loss < 0.05), "level": None,
                 "rooms_deg": {c: [float(x) for x in rooms[c]] for c in chans}}


def _seed(seed):
    return seed if isinstance(seed, (int, np.integer)) else None


ARM = {"left": (13, 16, 18, 20), "right": (14, 17, 19, 21)}     # collar, shoulder, elbow, wrist
THIGH = {"left": 1, "right": 2}
HANG_KEEP = 0.6             # [module] share of a trunk or pelvis turn that a hanging upper arm does not follow
HANG_MEDIAL_M = 0.03        # [module] a hanging hand may come this much closer to the midline than in the base
BALANCE_M = 0.25            # [module] standing: the pelvis moves back by this (m) times the sine of the trunk's bow


def _toward(Ra, Rb, s):
    """exp(s log(Rb Ra^T)) Ra: rotations Ra (T, 3, 3) turned toward Rb by the share s ((T,) or a number)."""
    W = matrix_to_axis_angle(Rb @ Ra.transpose(-1, -2))
    s = torch.as_tensor(s, dtype=F64)
    return axis_angle_to_matrix(W * (s[:, None] if s.dim() == 1 else s)) @ Ra


def support_arms(body, base, sides=("left", "right"), bounds=None):
    """(Body, info): _support_solve, then where a palm still presses more than RUB_PRESS_M into the skin (its thigh
    rising under a hand the table or the arm's reach held back, a palm 12 mm in on the seated grade-4 leg chorea,
    2026-11-05), the wrist's target raised by that depth (rub_press_lift, read on the solved body) and solved again,
    once."""
    out, info = _support_solve(body, base, sides, bounds=bounds)
    up = torch.tensor([0.0, 1.0, 0.0], dtype=F64)
    raise_ = {s: rub_press_lift(out, s) for s in sides}
    info["press_raise_mm_max"] = {s: float(1000 * r.max()) for s, r in raise_.items()}
    extra = None
    if any(r.max() > 0.0 for r in raise_.values()):
        extra = {s: torch.from_numpy(r)[:, None] * up for s, r in raise_.items()}
        out, info2 = _support_solve(body, base, sides, extra, bounds=bounds)
        info.update(sides=info2["sides"], table=info2["table"])
    out, info["elbow_swivel"] = _support_swivel(out, body, base, sides, extra, bounds)
    return out, info


SUPPORT_SWIVEL_DEG = (5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0)   # [module] seated: the elbow's outward swing about the
#   shoulder-to-wrist line tried where the trunk or the thigh comes onto the supported arm (the hand stays on the
#   thigh; a seated grade-4 trunk chorea brought the trunk 2.1 cm into a forearm the contact guard, which only scales
#   the arm's own movement, could not move, 2026-11-05)
SUPPORT_RAISE_M = (0.01, 0.02, 0.03, 0.045, 0.06, 0.08)   # [module] then the wrist's target raised by this (the hand
#   lifted off the thigh), with the swing of SUPPORT_SWIVEL_DEG that left the least depth, where no swing clears the
#   run (the review of wave SH1d: on the seated grade-4 trunk chorea the thumb, not the forearm, went 3.4 cm into the
#   thigh, and a 40 deg swing ended the run still inside)


def _support_swivel(out, body, base, sides, extra, bounds=None):
    """(Body, info): where an arm of the supported carriage out passes more than GUARD_TOL_M into the trunk or the
    thighs (_penetration, GUARD_HZ knots), each such run (widened by GUARD_RAMP_S) takes the smallest outward elbow
    swing of SUPPORT_SWIVEL_DEG that clears it, reached by minimum-jerk ramps of GUARD_RAMP_S or longer (the
    shoulder's speed cap, SPEED_CAP_DEG_S); where none clears, the swing that left the least depth with the smallest
    raise of the wrist's target of SUPPORT_RAISE_M that clears it (ramped as the swing); the arms are then solved
    again (_support_solve with the swing and the raise). Each run records its swing, its raise, whether it cleared
    ('clear') and the depth it left ('left_cm')."""
    from . import viewer as V
    boxes = _boxes(SCENE["sit"], out)
    step = max(1, int(round(out.rate / GUARD_HZ)))
    ks = np.unique(np.r_[np.arange(0, out.T, step), out.T - 1])
    Vk = V.knot_vertices(out, ks)
    ctx = _target_ctx(Vk)
    wid = int(math.ceil(GUARD_RAMP_S * GUARD_HZ))
    prof = {s: np.zeros(out.T) for s in sides}
    lift = {s: np.zeros(out.T) for s in sides}
    up = torch.tensor([0.0, 1.0, 0.0], dtype=F64)
    rec = {}

    def with_raise(s, r):
        e = {k: (torch.zeros(out.T, 3, dtype=F64) if extra is None else extra[k]) for k in ("left", "right")}
        e[s] = e[s] + torch.from_numpy(r)[:, None] * up
        return e
    for s in sides:
        pen = _penetration(Vk, s, boxes, ctx)
        runs = [(max(a - wid, 0), min(b - 1 + wid, len(ks) - 1)) for a, b in _runs(pen > GUARD_TOL_M)]
        done = []
        for a, b in runs:
            k0, k1 = int(ks[a]), int(ks[b])
            best, ok = None, False
            ladder = [(deg, 0.0) for deg in SUPPORT_SWIVEL_DEG]
            i = 0
            while i < len(ladder):
                deg, rz = ladder[i]
                rr = max(int(round(GUARD_RAMP_S * out.rate)),
                         int(math.ceil(SPEED_PEAK * deg / SPEED_CAP_DEG_S["shoulder"] * out.rate)))
                p = np.maximum(prof[s], _run_profile(out.T, [(k0, k1)], [deg], rr, 0.0))
                lz = np.maximum(lift[s], _run_profile(out.T, [(k0, k1)], [rz], rr, 0.0))
                cand, _ = _support_solve(body, base, sides, with_raise(s, lz), {s: p}, bounds)
                kk = np.arange(max(k0 - rr, 0), min(k1 + rr, out.T - 1) + 1, step)
                Vc = V.knot_vertices(cand, kk)
                depth = float(_penetration(Vc, s, boxes, _target_ctx(Vc)).max())
                if best is None or depth < best[0] - 1e-9 or (rz > 0 and depth <= best[0] + 1e-9):
                    best = (depth, deg, rz, p, lz)
                if depth <= GUARD_TOL_M:
                    ok = True
                    break
                if i == len(SUPPORT_SWIVEL_DEG) - 1:             # no swing clears: raise with the best swing
                    ladder += [(best[1], r) for r in SUPPORT_RAISE_M]
                i += 1
            depth, deg, rz, prof[s], lift[s] = (depth, deg, rz, p, lz) if ok else best
            done.append({"knots": [k0, k1], "deg": deg, "raise_m": rz, "clear": ok, "left_cm": 100 * depth})
        rec[s] = {"runs": done, "deg_max": float(prof[s].max()), "raise_m_max": float(lift[s].max()),
                  "clear": all(r["clear"] for r in done)}
    if any(prof[s].max() > 0.0 or lift[s].max() > 0.0 for s in sides):
        e = extra
        for s in sides:
            if lift[s].max() > 0.0:
                e = with_raise(s, lift[s]) if e is None else dict(e, **{s: with_raise(s, lift[s])[s]})
        out, _ = _support_solve(body, base, sides, e, prof, bounds)
    return out, rec


def _support_solve(body, base, sides=("left", "right"), extra=None, swivel=None, bounds=None):
    """(Body, info): the seated arms of body (module docstring, Arms) re-solved so that each wrist stays where the
    base puts it on its thigh, carried by the thigh (the hand rests on it while the trunk or the leg moves), the hand
    keeping its orientation on the thigh: the elbow flexion about the elbow's hinge (the y axis of the upper arm's
    frame) in closed form, the upper arm aligned on the shoulder-to-wrist line and turned so that the elbow lies
    toward where the trunk carries it. Where the knees swing in, the two wrists' targets slide apart along the body's
    lateral axis (hand_room) so that the hands come no closer than HAND_APPROACH_M to each other than in the base.
    Each joint's change is then scaled down where it would leave the table (apply_offsets, bounds: the table the
    clip is judged on). With neither trunk nor leg moving, the solve returns the base arms. extra {side: (T, 3) m}: added to the wrist's target; swivel {side:
    (T,) deg}: the elbow swung that much outward about the shoulder-to-wrist line (_support_swivel)."""
    from .ik import rmin
    lateral = torch.from_numpy(_lateral_frame(body))
    J, G = body.fk()
    Jb, Gb = base.fk()
    R = body.local_R()
    rest = body.rest()
    yax = torch.tensor([0.0, 1.0, 0.0], dtype=F64)
    offs, info = {}, {}
    targets, thumbs = {}, {}
    tips = torch.from_numpy(H.body_points(base, [H.tip_vertices()[s]["thumb"] for s in ("left", "right")]))
    for i, s in enumerate(("left", "right")):
        th, wr = THIGH[s], ARM[s][3]
        Gt = G[:, th] @ Gb[:, th].transpose(-1, -2)
        targets[s] = (J[:, th] + torch.einsum("tij,tj->ti", Gt, Jb[:, wr] - Jb[:, th]), Gt @ Gb[:, wr])
        thumbs[s] = targets[s][0] + torch.einsum("tij,tj->ti", Gt, tips[:, i] - Jb[:, wr])   # the hand carried rigidly
    slide = hand_room(body, base, targets["left"][0], targets["right"][0], thumbs=(thumbs["left"], thumbs["right"]),
                      thumb_room=THUMB_ROOM_M)
    for s in sides:
        c, sh, el, wr = ARM[s]
        W, Gw = targets[s]
        W = W + SIDE_SIGN[s] * slide + (0.0 if extra is None else extra[s])
        S, P = J[:, sh], J[:, el]
        p = rest[el] - rest[sh]
        q = torch.einsum("tij,j->ti", R[:, el], rest[wr] - rest[el])
        A = (p * p).sum() + (q * q).sum(-1) + 2 * p[1] * q[:, 1]
        a = p[0] * q[:, 0] + p[2] * q[:, 2]
        b = p[0] * q[:, 2] - p[2] * q[:, 0]
        rho = torch.sqrt(a * a + b * b)
        delta = torch.atan2(b, a)
        dmax = 0.985 * torch.sqrt(A + 2 * rho)
        d = (W - S).norm(dim=-1)
        clipped = d > dmax
        W = torch.where(clipped[:, None], S + (W - S) * (dmax / d)[:, None], W)
        d = torch.minimum(d, dmax)
        ac = torch.acos(((d * d - A) / (2 * rho)).clamp(-1.0, 1.0))
        ph = torch.stack([delta + ac, delta - ac], -1)
        ph = torch.atan2(torch.sin(ph), torch.cos(ph))
        phi = torch.where(ph[:, 0].abs() <= ph[:, 1].abs(), ph[:, 0], ph[:, 1])
        Ry = axis_angle_to_matrix(yax[None] * phi[:, None])
        v = p[None] + torch.einsum("tij,tj->ti", Ry, q)
        u = (W - S) / d[:, None]
        G1 = rmin(torch.einsum("tij,tj->ti", G[:, sh], v) / v.norm(dim=-1, keepdim=True), u) @ G[:, sh]
        kp = torch.einsum("tij,j->ti", G1, p)
        pp = P - S
        kp = kp - (kp * u).sum(-1, keepdim=True) * u
        pp = pp - (pp * u).sum(-1, keepdim=True) * u
        kp = kp / kp.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        pp = pp / pp.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        gam = torch.atan2((torch.cross(kp, pp, dim=-1) * u).sum(-1), (kp * pp).sum(-1))
        G16 = axis_angle_to_matrix(u * gam[:, None]) @ G1
        if swivel is not None and s in swivel:              # outward: the elbow moves along the body's lateral axis
            e = torch.einsum("tij,j->ti", G16, p)
            sg = torch.sign((torch.cross(u, e, dim=-1) * lateral).sum(-1) * SIDE_SIGN[s])
            G16 = axis_angle_to_matrix(u * (sg * torch.from_numpy(np.radians(swivel[s])))[:, None]) @ G16
        R16 = G[:, c].transpose(-1, -2) @ G16
        R18 = Ry @ R[:, el]
        R20 = (G16 @ R18).transpose(-1, -2) @ Gw
        R18, R20, handed = _twist_to_elbow(R18, R20, rest[wr] - rest[el], wr, base, bounds)
        for j, Rj in ((sh, R16), (el, R18), (wr, R20)):
            offs[j] = _log_deg(Rj @ R[:, j].transpose(-1, -2))
        info[s] = {"wrist_reach_clipped_share": float(clipped.double().mean()),
                   "elbow_change_deg_max": math.degrees(float(phi.abs().max())),
                   "hand_slide_mm_max": float(1000 * slide.norm(dim=-1).max()),
                   "twist_to_elbow_deg_max": float(np.abs(handed).max())}
    out, ap = apply_offsets(body, offs, bounds=bounds)
    return out, {"sides": info, "table": ap["scaled_joints"]}


TWIST_SOFT_DEG = 3.0        # [module] the hand-over of the wrist's twist to the elbow (_twist_to_elbow) blends in over
#                             twice this past the band (a C1 hinge, zero inside it)


def _hinge(d, soft):
    """C1 hinge: 0 for d <= -soft, (d + soft)^2 / (4 soft) between, d for d >= soft."""
    d = np.asarray(d, np.float64)
    return np.where(d >= soft, d, np.where(d <= -soft, 0.0, (d + soft) ** 2 / (4.0 * soft)))


def _twist_to_elbow(R18, R20, bone, wr, base, bounds=None, soft=TWIST_SOFT_DEG, rounds=2):
    """(R18, R20, (T,) deg): the carried hand's turn about the forearm's long axis handed from the wrist to the elbow
    where the wrist's twist (its x component: the canonical arm lies along x) would pass the band the seated rest
    leaves it, [min(lo + REST_ROOM_DEG, rest), max(hi - REST_ROOM_DEG, rest)] (rest: the base's own component, the
    table's bounds, or the clip's bounds): the elbow turns about the forearm bone (bone, elbow to wrist, in the
    elbow's frame) and the wrist back by as much, so that the wrist joint and the hand's global rotation stay where
    they were. Open-body port, wave 3 follow-up (D41): the static sit holds the wrist 5.3 deg from its 45 deg share of
    pronation (D14's halves), and carrying a resting hand on its thigh while the trunk turned asked the wrist for more:
    the table's scaling then rolled the palm up to 17 mm off its thigh (the catalogue's seated grade-3 trunk chorea),
    while the elbow's half, where the twist split carries the turn on the skin (D16), had 16 deg left. A C1 hinge over
    2 soft past the band (zero inside it: a hand that stays inside comes back bit for bit; the wrist then passes the
    band by at most soft)."""
    lim = joint_limits()
    T = R20.shape[0]
    if bounds is None:
        lo, hi = np.full(T, lim.lo[wr, 0]), np.full(T, lim.hi[wr, 0])
    else:
        lo, hi = np.asarray(bounds[0])[:, wr, 0], np.asarray(bounds[1])[:, wr, 0]
    c0 = float(_log_deg(base.local_R()[:1, wr])[0, 0])
    up, dn = np.maximum(hi - REST_ROOM_DEG, c0), np.minimum(lo + REST_ROOM_DEG, c0)
    u = torch.as_tensor(np.asarray(bone, np.float64) / np.linalg.norm(np.asarray(bone, np.float64)), dtype=F64)
    sgn = 1.0 if float(u[0]) >= 0 else -1.0
    total = np.zeros(T)
    for _ in range(rounds):
        x = _log_deg(R20)[:, 0]
        e = _hinge(x - up - soft, soft) - _hinge(dn - soft - x, soft)      # 0 inside the band, at most soft past it
        if not np.any(np.abs(e) > 0):
            break
        th = torch.as_tensor(np.radians(sgn * e), dtype=F64)[:, None] * u[None]
        R18 = R18 @ axis_angle_to_matrix(th)
        R20 = axis_angle_to_matrix(-th) @ R20
        total += e
    return R18, R20, total


HAND_APPROACH_M = 0.02      # [module] seated, the hands resting on the thighs come at most this much closer to each
#                             other than in the level 0 (whose hand joints and fingertips lie 47 mm apart, wave SH1d
#                             reading of the page's points): where both knees swung in, the thighs carried the thumbs
#                             within 5 mm of each other (the page's check asks 15 mm)
HAND_ROOM_SOFT_M = 0.005    # [module] the slide starts smoothly over this (a quadratic blend, zero at rest)
THUMB_ROOM_M = 0.0175       # [module] and the two thumb tips, carried rigidly with their hands, keep at least this
#                             lateral separation less half of HAND_ROOM_SOFT_M (the soft blend): PAIR_CLEAR_M (15 mm,
#                             the clearance the contact guard and arm_contact keep between the two hands' skins) plus
#                             that half, so that the resting tips keep PAIR_CLEAR_M. On the open body the level 0's
#                             thumb tips lie 30 mm apart and the hands turned with the thighs bring the thumbs in
#                             faster than the wrists: under the wrists' rule alone the knees swinging in left the thumb
#                             tips of the seated leg chorea grades 2 and 4 4.0 and 4.2 mm apart (the page's self-contact
#                             check: two skin points within 5 mm). At 20 mm the seated chorea grade 3 also passes its
#                             arm_contact (0.81 cm against 0.8 with this rule as with the wrists' alone), but the grade
#                             4's layered thumb then reads 8.6 mm into the thigh on the page (8 allowed; 6.0 raw)
#                             (scripts/openbody_phaseE_checks.py 'seated_thumbs')


def _hinge_up(x, soft):
    """The C1 hinge of hand_room: 0 for x <= 0, x^2 / (2 soft) up to soft, x - soft / 2 beyond."""
    return torch.where(x <= 0, torch.zeros_like(x), torch.where(x < soft, x * x / (2 * soft), x - 0.5 * soft))


def hand_room(body, base, W_left, W_right, approach=HAND_APPROACH_M, soft=HAND_ROOM_SOFT_M, thumbs=None,
              thumb_room=THUMB_ROOM_M):
    """(T, 3) m: the slide of the left wrist's target along the body's left axis (the right's goes the other way)
    that keeps the two targets' lateral separation at least the base's less approach: half the shortfall, blended in
    by a C1 hinge over soft (zero where the separation is the base's or more). thumbs (left, right) (T, 3) m: the
    thumb tips carried with the targets; where they still lie closer laterally than thumb_room after that slide, the
    slide grows by the hinge of the rest of the shortfall (zero where they do not: the wrists' rule unchanged)."""
    left = torch.from_numpy(_lateral_frame(body))
    Jb = base.joints()
    sep0 = float(((Jb[0, ARM["left"][3]] - Jb[0, ARM["right"][3]]) * torch.from_numpy(_lateral_frame(base))[0]).sum())
    x = (sep0 - approach) - ((W_left - W_right) * left).sum(-1)
    d = _hinge_up(x, soft)
    if thumbs is not None:
        d = d + _hinge_up(thumb_room - ((thumbs[0] - thumbs[1]) * left).sum(-1) - d, soft)
    return 0.5 * d[:, None] * left


def hang_arms(body, base, keep=HANG_KEEP, sides=("left", "right")):
    """(Body, info): standing, each upper arm turned back toward its global orientation in the base by keep of the
    way (a hanging arm does not follow the trunk's turns rigidly), less where that would bring the hand closer to
    the body's midline than in the base less HANG_MEDIAL_M (the arm on the convex side of a bend rests against the
    trunk and is carried by it): per knot the share is cut so that the wrist's lateral distance from the pelvis, in
    the pelvis frame, stays at least that, smoothed below over 0.3 s; inside the table (apply_offsets)."""
    J, G = body.fk()
    Jb, Gb = base.fk()
    R = body.local_R()
    n = max(1, int(round(0.3 * body.rate)))
    offs, info = {}, {}
    for s in sides:
        c, sh, _, wr = ARM[s]
        sg = SIDE_SIGN[s]
        Gk = _toward(G[:, sh], Gb[:, sh], keep)
        arm = torch.einsum("tji,tj->ti", G[:, sh], J[:, wr] - J[:, sh])          # the wrist in the upper arm's frame
        Wk = J[:, sh] + torch.einsum("tij,tj->ti", Gk, arm)

        def lateral(Jx, Gx, W):
            return sg * torch.einsum("ti,ti->t", Gx[:, 0][:, :, 0], W - Jx[:, 0]).numpy()
        lat_b = lateral(Jb, Gb, Jb[:, wr])
        lat_r = lateral(J, G, J[:, wr])
        lat_k = lateral(J, G, Wk)
        floor = lat_b - HANG_MEDIAL_M
        beta = np.where(lat_k >= floor, 1.0, np.clip((lat_r - floor) / np.maximum(lat_r - lat_k, 1e-9), 0.0, 1.0))
        beta = smooth_below(beta, n)
        R16 = G[:, c].transpose(-1, -2) @ _toward(G[:, sh], Gb[:, sh], keep * torch.from_numpy(beta))
        offs[sh] = _log_deg(R16 @ R[:, sh].transpose(-1, -2))
        info[s] = {"keep_min": float(keep * beta.min()), "keep_cut_share": float((beta < 1 - 1e-9).mean())}
    out, ap = apply_offsets(body, offs)
    return out, {"keep": keep, "sides": info, "table": ap["scaled_joints"]}


def balance_standing(body, base):
    """(Body, info): standing, the pelvis moved back by BALANCE_M times the sine of the upper trunk's forward lean
    against the base (forward when it leans back), both feet planted (standing_legs)."""
    J, G = body.fk()
    Jb, Gb = base.fk()
    fwd = Gb[:, 0][:, :, 2].clone()
    fwd[:, 1] = 0
    fwd = fwd / fwd.norm(dim=-1, keepdim=True)

    def lean(Jx):
        v = Jx[:, 12] - Jx[:, 0]
        return torch.atan2((v * fwd).sum(-1), v[:, 1])
    th = (lean(J) - lean(Jb)).numpy()
    if np.abs(th).max() < 1e-6:
        return body, {"pelvis_back_m_max": 0.0}
    dz = -BALANCE_M * np.sin(th)
    rows = [1, 2, 4, 5, 7, 8]
    for k in (1.0, 0.8, 0.6, 0.4, 0.2):                  # the largest share of the shift inside the table at every
        out, _ = standing_legs(body, dz=k * dz, keep_height=True, heading=base)   # knot (one share: a per-knot
        if _knots_inside(out, base, rows).all():           # share kinked the knees)
            break
    else:
        out, k = body, 0.0
    return out, {"pelvis_back_m_max": float(np.max(-k * dz)), "lean_deg_max": float(np.degrees(th.max())),
                 "shift_share": float(k)}


def _knots_inside(out, base, rows):
    """(T,) bool: the knots where the rows of out add no joint-limit violation against base."""
    from .body import LIMIT_TOL_DEG, components_deg
    lim = joint_limits()
    a, b = components_deg(out.rot6d)[:, rows], components_deg(base.rot6d)[:, rows]
    lo, hi = np.minimum(lim.lo[rows][None], b), np.maximum(lim.hi[rows][None], b)
    return ((a >= lo - LIMIT_TOL_DEG) & (a <= hi + LIMIT_TOL_DEG)).all((1, 2))


# ---------------------------------------------------------------- the contact guard
GUARD_TOL_M = 0.008         # [module] skin pressed into a surface (the chair, the trunk or a thigh) before the guard
#                             turns an arm back; the level-0 trunk rests 0.7 to 1.5 cm into the backrest
GUARD_HZ = 20.0             # [module] knots checked per second
GUARD_ROUNDS = 5            # [module] rounds of applying the runs and reading the result (at 3 the severe tardive
#                             limb movements kept 5 knots of a hand within the other's clearance, the review of wave
#                             SH1d)
GUARD_RAMP_S = 0.2           # [module] a guarded run is widened by this on both sides and ramped in and out over it
GUARD_GRID = (0.8, 0.6, 0.45, 0.3, 0.15, 0.0)
GUARD_ABD_DEG = (4.0, 8.0, 12.0, 16.0, 20.0, 25.0, 30.0)   # [module] outward shoulder turns tried where ref touches
#   (not on the seated rest: an outward turn of an arm whose palm lies on the thigh drags the hand into the thigh, 2 cm
#   at 15 percent of the abduction's range, and the guard's turn there pressed a thumb 3.9 cm in on the seated grade-3
#   chorea, 2026-11-04; the carriage itself, support_arms, keeps the hand on the thigh)
GUARD_WIDEN_SEATED = (0, 2, 4, 6, 10, 16)   # [module] seated, the knots a run's core is widened by each way when the
#                                             arm brought back to its carriage still touches in the ramps
HEAD_CLEAR_M = 0.010        # [module] the guard keeps a forearm and hand at least this from the skin of the head and
#                             the neck (the clinical review of 2026-11-03, M7: a choreic index tip came 62 mm from the
#                             head centre, inside the jaw, the guard testing only the chair, the trunk and the thighs)
PAIR_CLEAR_M = 0.015        # [module] and from the skin of the other forearm and hand (M7: the hands passed through
#                             each other, down to 4 to 9 mm between hand points on the skeleton): the page's check
#                             flags two hands whose joints or fingertips come within 15 mm (the review plan's bound),
#                             and those points lie on or inside the skin, so a skin gap of 15 mm keeps them apart
PAIR_REACH_M = 0.06         # [module] a probe vertex is tested against the nearest target vertex within this (as the
#                             trunk and thighs)


def _model_name():
    from .body import model_name
    return model_name()


def _guard_geometry():
    """Vertex sets of the guard (the selected body model's rest mesh, cached per model name): per side the probes
    against the chair (upper arm, forearm, hand) and against the body (forearm and hand), the body's trunk and thigh
    vertices (the self target) with the sparse face incidence that gives their normals."""
    return _guard_geometry_of(_model_name())


@functools.lru_cache(maxsize=2)
def _guard_geometry_of(name):
    from scipy.sparse import csr_matrix
    from . import viewer as V
    dom = V.dominant_joint("neutral")
    faces = V.faces()                                    # the selected body model's triangles
    hand = {"left": tuple(range(22, 37)), "right": tuple(range(37, 52))}
    chair = {s: np.flatnonzero(np.isin(dom, (ARM[s][1], ARM[s][2], ARM[s][3]) + hand[s])) for s in ARM}
    self_ = {s: np.flatnonzero(np.isin(dom, (ARM[s][2], ARM[s][3]) + hand[s])) for s in ARM}
    tgt = np.flatnonzero(np.isin(dom, (0, 1, 2, 3, 6, 9)))
    pos = {v: i for i, v in enumerate(tgt)}
    rows, cols = [], []
    for f, tri in enumerate(faces):
        for v in tri:
            if v in pos:
                rows.append(pos[v])
                cols.append(f)
    inc = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(tgt), len(faces)))
    return chair, self_, tgt, inc, faces


def _boxes(scene, body):
    """[(lo (3,), hi (3,))] of the clip's scene boxes (exam.scene_mesh at the viewer's floor), [] without scene."""
    if scene is None:
        return []
    from . import exam as EX
    from . import viewer as V
    floor_y, _ = V.scene_floor(body, 10.0)
    sc = EX.scene_mesh(scene, body, floor_y)
    if sc is None:
        return []
    v = sc[0].reshape(-1, 8, 3)
    return [(b.min(0), b.max(0)) for b in v]


def _penetration(Vk, side, boxes, ctx):
    """(n,) m: per knot the deepest probe vertex of the arm of side inside a scene box or behind the body's trunk
    and thigh surface (the nearest target vertex within 6 cm, against its normal); ctx: per knot (tree, points,
    normals) of the target (_target_ctx). A ctx made with heads=True (the guard's) adds the head and the neck and the
    other forearm and hand, each read as a depth behind a surface HEAD_CLEAR_M (PAIR_CLEAR_M) outside the skin plus
    GUARD_TOL_M: at the guard's tolerance a probe vertex passes when it lies at least the clearance outside that
    skin."""
    chair, self_, _, _, _ = _guard_geometry()
    out = np.zeros(len(Vk))
    for i, v in enumerate(Vk):
        d = 0.0
        p = v[chair[side]]
        for lo, hi in boxes:
            e = np.minimum(p - lo, hi - p)
            ins = (e > 0).all(1)
            if ins.any():
                d = max(d, float(e[ins].min(1).max()))
        tree, pts, nrm = ctx[i][:3]
        q = v[self_[side]]
        # the searches stop at the reach that counts (open-body port, wave 3: the same nearest vertices within it,
        # the probes farther away read 0 as before; Anny's denser mesh made the full searches the guard's main cost)
        dist, j = tree.query(q, distance_upper_bound=0.06)
        near = dist < 0.06
        if near.any():
            inner = -np.einsum("ij,ij->i", q[near] - pts[j[near]], nrm[j[near]])
            d = max(d, float(inner.max()))
        if len(ctx[i]) > 3:
            head, arms = ctx[i][3], ctx[i][4]
            other = "right" if side == "left" else "left"
            for (tr_, pt_, n_), clear in ((head, HEAD_CLEAR_M), (arms[other], PAIR_CLEAR_M)):
                dist, j = tr_.query(q, distance_upper_bound=PAIR_REACH_M)
                near = dist < PAIR_REACH_M
                if near.any():
                    inner = -np.einsum("ij,ij->i", q[near] - pt_[j[near]], n_[j[near]]) + clear + GUARD_TOL_M
                    d = max(d, float(inner.max()))
        out[i] = d
    return out


def _head_geometry():
    """(ids, sparse vertex-face incidence) of the head and the neck (the vertices skinned mostly by joints 12, 15), on
    the selected body model's rest mesh (cached per model name)."""
    return _head_geometry_of(_model_name())


@functools.lru_cache(maxsize=2)
def _head_geometry_of(name):
    from scipy.sparse import csr_matrix
    from . import viewer as V
    dom = V.dominant_joint("neutral")
    faces = V.faces()                                    # the selected body model's triangles
    ids = np.flatnonzero(np.isin(dom, (12, 15)))
    pos = {v: i for i, v in enumerate(ids)}
    rows, cols = [], []
    for f, tri in enumerate(faces):
        for v in tri:
            if v in pos:
                rows.append(pos[v])
                cols.append(f)
    return ids, csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(ids), len(faces)))


def _tree(pts):
    """A cKDTree built for a few queries (unbalanced, not compacted: faster to build, the same nearest vertices)."""
    from scipy.spatial import cKDTree
    return cKDTree(pts, balanced_tree=False, compact_nodes=False)


def _target_ctx(Vk, heads=False):
    """Per knot (tree, points, normals) of the trunk and thighs (the contact guard's self target); heads=True adds
    (tree, points, normals) of the head and the neck (_head_geometry) and {side: (tree, points, normals)} of each
    forearm and hand (_pair_geometry, PAIR_JOINTS), read by _penetration."""
    _, _, tgt, inc, faces = _guard_geometry()
    if heads:
        hid, hinc = _head_geometry()
        psets, pinc, _ = _pair_geometry("arms")
    ctx = []

    def unit(n):
        return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)
    for v in Vk:
        fn = np.cross(v[faces[:, 1]] - v[faces[:, 0]], v[faces[:, 2]] - v[faces[:, 0]])
        pts = v[tgt]
        c = (_tree(pts), pts, unit(inc @ fn))
        if heads:
            hp = v[hid]
            c = c + ((_tree(hp), hp, unit(hinc @ fn)),
                     {s: (_tree(v[psets[s]]), v[psets[s]], unit(pinc[s] @ fn)) for s in ("left", "right")})
        ctx.append(c)
    return ctx


def _arm_blend(out, ref, scales):
    """A copy of out whose arm joints and hand rows of each side in scales {side: (T,) share} are turned from ref
    toward out by that share (1 keeps out, 0 gives ref)."""
    R, Rr = out.local_R(), ref.local_R()
    Rn = R.clone()
    rows = []
    for s, sc in scales.items():
        for j in ARM[s]:
            Rn[:, j] = _toward(Rr[:, j], R[:, j], sc)
            rows.append(j)
    b = out.with_rotations(Rn, rows)
    if H.has_hands(out):
        Rh = out.hand_R()
        Rhr = ref.hand_R() if H.has_hands(ref) else torch.eye(3, dtype=F64).repeat(out.T, H.N_HAND, 1, 1)
        Rhn = Rh.clone()
        hrows = []
        for s, sc in scales.items():
            r = list(range(0, 15)) if s == "left" else list(range(15, 30))
            Rhn[:, r] = _toward(Rhr[:, r].reshape(-1, 3, 3), Rh[:, r].reshape(-1, 3, 3),
                                torch.as_tensor(sc, dtype=F64).repeat_interleave(15)).reshape(out.T, 15, 3, 3)
            hrows += r
        b = b.with_hands(Rhn, hrows)
    return b


def _abducted(body, side, deg):
    """body with the shoulder of side turned outward by deg (T,) about the abduction channel's axis, scaled down where
    it would leave the table (apply_offsets)."""
    out, _ = apply_offsets(body, offsets_of(body, {f"sh_abd_{side}": np.asarray(deg, float)}))
    return out


def _inside_from(ref, body, joints):
    """body with each of joints turned back toward ref where it would leave the table (apply_offsets on ref of the
    offsets from ref to body): a blend of two postures inside the table may leave it."""
    R, Rr = body.local_R(), ref.local_R()
    offs = {j: _log_deg(R[:, j] @ Rr[:, j].transpose(-1, -2)) for j in joints}
    out, _ = apply_offsets(ref, offs)
    if H.has_hands(body):
        out = H.hands_of(out, body.hand6d.clone())
    return out


def _run_profile(T, runs, values, ramp_n, base_value):
    """(T,) a profile at base_value outside the runs [(k0, k1)] (knot indices), at each run's value inside it, with
    minimum-jerk ramps of ramp_n knots (one number, or one per run) just outside each run; overlapping runs take the
    value farthest from base_value."""
    prof = np.full(T, float(base_value))
    k = np.arange(T)
    ramps = list(ramp_n) if isinstance(ramp_n, (list, tuple)) else [ramp_n] * len(runs)
    for (k0, k1), v, ramp_n in zip(runs, values, ramps):
        up = minimum_jerk((k - (k0 - ramp_n)) / max(ramp_n, 1))
        dn = 1.0 - minimum_jerk((k - k1) / max(ramp_n, 1))
        w = np.minimum(up, dn)
        cand = base_value + (v - base_value) * w
        prof = np.minimum(prof, cand) if v < base_value else np.maximum(prof, cand)
    return prof


def guard_arms(out, ref, scene, sides=("left", "right"), tol=GUARD_TOL_M, speed_ramps=True):
    """(Body, info): the contact guard (module docstring, Arms). At GUARD_HZ knots the probe vertices of each arm
    (upper arm, forearm, hand) are tested against the scene's boxes and the body's trunk and thigh surface, the head
    and the neck, and the other forearm and hand (_penetration with _target_ctx heads=True); each run
    of knots where they pass more than tol inside (widened by GUARD_RAMP_S on both sides) takes one share of the
    arm's own movement (its collar, shoulder, elbow and wrist, and its hand, turned from ref toward out): the largest
    of GUARD_GRID that is clear over the whole widened run, reached by minimum-jerk ramps of GUARD_RAMP_S inside the
    widening (where out itself is clear); where even ref touches (a thigh moving under a hanging hand), the run takes
    the smallest outward shoulder turn of GUARD_ABD_DEG that clears it (the arm pushed aside by the body; seated,
    scene SCENE['sit'], only where the palm is off its thigh over the run, palm_off; a resting hand takes no turn:
    the run's core is widened, GUARD_WIDEN_SEATED, instead). With
    speed_ramps, a run's ramps last longer where its blend or turn would move a joint of the arm faster than its speed
    cap (SPEED_PEAK times the joint's turn over its cap, SPEED_CAP_DEG_S: the clinical review of 2026-11-03 found the
    guard's 0.2 s ramps pulling a seated grade-4 choreic hand back at 3.7 m/s against 1.7 m/s without it); a
    ballistic plan (hemiballism's flings) keeps GUARD_RAMP_S (ramps of 0.77 s spread over the next flings). The result
    is checked at and between the checked knots, and a run that still touches takes the next share down or the next
    turn up (up to GUARD_ROUNDS rounds in all). ref must be free of contact or close to it (the supported or hanging
    arms)."""
    from . import viewer as V
    boxes = _boxes(scene, out)
    Ro, Rr = out.local_R(), ref.local_R()
    turn = {s: np.stack([np.linalg.norm(_log_deg(Ro[:, j] @ Rr[:, j].transpose(-1, -2)), axis=-1) for j in ARM[s]], 1)
            for s in sides}                              # (T, 4) deg: each arm joint's turn from ref to out
    caps = np.array([SPEED_CAP_DEG_S[g] for g in ("collar", "shoulder", "elbow", "wrist")])
    step = max(1, int(round(out.rate / GUARD_HZ)))
    ks = np.unique(np.r_[np.arange(0, out.T, step), out.T - 1])
    Vk = V.knot_vertices(out, ks)
    ctx = _target_ctx(Vk, heads=True)
    before = {s: _penetration(Vk, s, boxes, ctx) for s in sides}
    ramp = max(1, int(round(GUARD_RAMP_S * out.rate)))
    wid = int(math.ceil(GUARD_RAMP_S * GUARD_HZ))
    plan = {}
    seated = scene == SCENE["sit"]

    def turns_of(s, i0, i1):
        """the outward turns a run may take: off the seated rest, all; on it, only where the palm is off its thigh
        at every core knot (palm_off: a hand lifted into a posture or a fling), so that no turn drags a resting hand
        into its thigh (the review of wave SH1d, blocker 1)."""
        if not seated:
            return GUARD_ABD_DEG
        return GUARD_ABD_DEG if palm_off(Vk[i0:i1 + 1], s).all() else ()
    for s in sides:
        bad = before[s] > tol
        runs = []
        for i0, i1 in _runs(bad):
            j0, j1 = max(i0 - wid, 0), min(i1 - 1 + wid, len(ks) - 1)
            if runs and j0 <= runs[-1][3] + 1:
                runs[-1] = (runs[-1][0], i1 - 1, runs[-1][2], j1)
            else:
                runs.append((i0, i1 - 1, j0, j1))
        plan[s] = [{"run": r, "g_idx": -1, "a_idx": -1, "turns": turns_of(s, r[0], r[1])} for r in runs]

    def ramp_of(s, k0, k1, g, a):
        """knots: the run's ramp, GUARD_RAMP_S or longer, so that the blend's (1 - g) share of each joint's turn
        (the largest near the run) and the outward turn a move no joint faster than its cap."""
        if not speed_ramps:
            return ramp
        th = turn[s][max(k0 - ramp, 0):min(k1 + ramp, out.T - 1) + 1].max(0)
        need = max(float((SPEED_PEAK * (1.0 - g) * th / caps).max()), SPEED_PEAK * a / SPEED_CAP_DEG_S["shoulder"])
        return max(ramp, int(math.ceil(need * out.rate)))

    def clear(s, core, g, a):
        """(clear, ramp): the run's own profile (value g or turn a on core, ramps around it) clear at the widened
        run's knots and between them."""
        k0, k1 = ks[core[0]], ks[core[1]]
        rr = ramp_of(s, k0, k1, g, a)
        sc = _run_profile(out.T, [(k0, k1)], [g], rr, 1.0)
        bg = _arm_blend(out, ref, {s: sc})
        if a > 0:
            bg = _abducted(bg, s, _run_profile(out.T, [(k0, k1)], [a], rr, 0.0))
        kk = np.arange(max(k0 - rr - step, 0), min(k1 + rr + step, out.T - 1) + 1, max(step // 2, 1))
        Vk_ = V.knot_vertices(bg, kk)
        cx = _target_ctx(Vk_, heads=True)
        return bool((_penetration(Vk_, s, boxes, cx) <= tol).all()), rr

    def settle(s, p):
        """the next share down (then turn up) whose profile is clear, the run's core widened when its ramps touch."""
        i0, i1 = p["run"][:2]
        turns = p["turns"]
        while p["g_idx"] + 1 < len(GUARD_GRID) or p["a_idx"] + 1 < len(turns):
            if p["g_idx"] + 1 < len(GUARD_GRID):
                p["g_idx"] += 1
                g, a = GUARD_GRID[p["g_idx"]], 0.0
            else:
                p["a_idx"] += 1
                g, a = 0.0, turns[p["a_idx"]]
            for w in ((0, 2, 4, 6) if a > 0 or g > 0 or turns else GUARD_WIDEN_SEATED):
                core = (max(i0 - w, 0), min(i1 + w, len(ks) - 1))
                ok, rr = clear(s, core, g, a)
                if ok:
                    p["run"] = (core[0], core[1], max(core[0] - wid, 0), min(core[1] + wid, len(ks) - 1))
                    p["ramp"] = rr
                    return
            p["ramp"] = rr
    for s in sides:
        for p in plan[s]:
            settle(s, p)
    ks2 = np.unique(np.clip(np.r_[ks + step // 2, ks], 0, out.T - 1))
    for rnd in range(GUARD_ROUNDS):
        scales, abds = {}, {}
        for s in sides:
            runs = [(ks[p["run"][0]], ks[p["run"][1]]) for p in plan[s]]
            g = [GUARD_GRID[p["g_idx"]] if p["g_idx"] >= 0 else 1.0 for p in plan[s]]
            a = [GUARD_ABD_DEG[p["a_idx"]] if p["a_idx"] >= 0 else 0.0 for p in plan[s]]
            rr = [p.get("ramp", ramp) for p in plan[s]]
            scales[s] = _run_profile(out.T, runs, g, rr, 1.0)
            abds[s] = _run_profile(out.T, runs, a, rr, 0.0)
        res = out
        if any(v.min() < 1.0 - 1e-12 for v in scales.values()):
            res = _inside_from(ref, _arm_blend(out, ref, scales), [j for s in sides for j in ARM[s]])
        for s in sides:
            if abds[s].max() > 0:
                res = _abducted(res, s, abds[s])
        V2 = V.knot_vertices(res, ks2)
        ctx2 = _target_ctx(V2, heads=True)
        after = {s: _penetration(V2, s, boxes, ctx2) for s in sides}
        if rnd == GUARD_ROUNDS - 1 or all((after[s] <= tol).all() for s in sides):
            break
        for s in sides:                                  # a run that still touches takes the next step
            left = ks2[after[s] > tol]
            hit = [p for p in plan[s] if ((left >= ks[p["run"][2]] - step) & (left <= ks[p["run"][3]] + step)).any()]
            for p in hit:
                settle(s, p)
            new = [k for k in left if not any(ks[p["run"][2]] - step <= k <= ks[p["run"][3]] + step for p in plan[s])]
            for k in new:                                # a touch outside every run: a new run there
                i = int(np.clip(np.searchsorted(ks, k), 0, len(ks) - 1))
                p = {"run": (i, i, max(i - wid, 0), min(i + wid, len(ks) - 1)), "g_idx": -1, "a_idx": -1,
                     "turns": turns_of(s, i, i)}
                settle(s, p)
                plan[s].append(p)
    info = {s: {"before_cm_max": float(100 * before[s].max()), "before_share": float((before[s] > tol).mean()),
                "after_cm_max": float(100 * after[s].max()), "after_share": float((after[s] > tol).mean()),
                "runs": len(plan[s]), "scaled_share": float((scales[s] < 1.0 - 1e-9).mean()),
                "scale_min": float(scales[s].min()), "abduction_deg_max": float(abds[s].max()) + 0.0,
                "abducted_share": float((abds[s] > 1e-9).mean()),
                "ramp_s_max": float(max([p.get("ramp", ramp) for p in plan[s]], default=ramp) / out.rate)}
            for s in sides}
    return res, {"tol_m": tol, "scene": scene, "sides": info}


def compose_clip(base, region_plan, rng, t=None, legs=None, fingers=None, check_feet=(0, 1), arms=None, scene=None,
                 balance=False, guard=None):
    """(Body, record): the free-channel regions of region_plan [{region, channels, kind, level, share|intervals,
    timing|band|rate, gains, weights, signs, margin}] built on base (after legs: a callable base -> (Body, info)
    moving legs or pelvis first): the axial and leg channels first, then (balance) the standing pelvis under a
    bowing trunk, the arms' carriage (arms 'support': seated hands resting on the thighs, support_arms; 'hang':
    standing, hang_arms), the arm channels on that carriage, the fingers [{side, kind, level, intervals, ...}], and
    (guard, default with arms) the contact guard against the scene and the body (module docstring)."""
    t = base.times().numpy() if t is None else t
    body1, leg_info = (base.copy(), None) if legs is None else legs(base)
    signals, recs = {}, {}
    room_offs = {}                                   # entries split over their chain by room (rp 'split' 'room')
    for rp in region_plan:
        chans = list(rp["channels"])
        T = base.T
        iv = rp.get("intervals")
        if iv is None:
            iv = bouts(base.duration, rp["share"], rng, n_bouts=rp.get("n_bouts"), bout_s=rp.get("bout_s", 3.0))
        kind = rp["kind"]
        if kind == "events":
            tm = TIMING[rp["timing"]] if isinstance(rp["timing"], str) else rp["timing"]
            ev = fragment_events(iv, chans, rng, tm, rp.get("weights"), rp.get("signs"))
            lift_ch = f"elbow_{rp['region'][4:]}" if rp["region"].startswith("arm_") else None
            if arms == "support" and tm.get("mode") == "flow" and lift_ch in chans:
                ev = lift_events(ev, lift_ch)            # the resting hand lifts off its thigh as it moves
            scale = room_scale(body1, chans)
            X = render_fragments(ev, chans, t, scale)
            D = None
            if tm.get("mode") == "flow" and rp.get("drift", True) and iv:
                N = band_noise(T, base.rate, (0.3, 1.0), rng, len(chans)) * ramp_envelope(t, iv, 0.5)[:, None]
                D = np.zeros_like(X)
                for i, c in enumerate(chans):
                    sg = (rp.get("signs") or {}).get(c)
                    D[:, i] = DRIFT_SHARE * (np.abs(N[:, i]) * sg if sg else N[:, i])
            extra = {"n_fragments": len(ev)}
            if tm.get("mode") == "flow" and rp.get("speed_cap", True) and rp.get("deg") is None:
                X, extra["speed"] = capped_fragments(body1, ev, chans, t, scale, rp["level"], rp.get("gains"),
                                                     rp.get("margin", SOFT_MARGIN), D)
            if D is not None:
                X = X + D
            raw = {c: X[:, i] for i, c in enumerate(chans)}
        elif kind == "noise":
            N = band_noise(T, base.rate, rp["band"], rng, len(chans))
            env = ramp_envelope(t, iv, 0.5)
            os_ = rp.get("one_sided")
            for i, c in enumerate(chans):              # a smooth one-signed drift: softplus, 0 outside the bouts
                sg = (os_.get(c) if isinstance(os_, dict) else (1.0 if os_ else None))
                if sg:
                    N[:, i] = sg * np.log1p(np.exp(3.0 * N[:, i])) / 3.0
            raw = {c: N[:, i] * env for i, c in enumerate(chans)}
            extra = {"band_hz": list(rp["band"])}
            lift_ch = f"elbow_{rp['region'][4:]}" if rp["region"].startswith("arm_") else None
            if arms == "support" and lift_ch in chans and len(chans) > 1:
                raw[lift_ch] = lift_noise(raw, lift_ch)  # the resting hand lifts off its thigh as it writhes
                extra["lift"] = {"channel": lift_ch, "share": NOISE_LIFT}
        elif kind == "given":
            raw = {c: np.asarray(rp["raw"][c], dtype=np.float64) for c in chans}
            extra = {}
        elif kind == "pattern":
            ph = rng.uniform(0, 2 * math.pi)
            raw = {}
            for c in chans:
                lag = rp.get("phase", {}).get(c, 0.0)
                raw[c] = pattern(t, rp["rate"], rng, iv, phase=ph + lag, wave=rp.get("wave", "sine"))
            extra = {"rate_hz": float(rp["rate"])}
        else:
            raise ValueError(f"signal kind {kind!r}")
        if rp.get("split") == "room":                # one channel, one-signed: the chain bends by its joints' rooms
            (c,) = chans
            offs, _, info = given_room_offsets(body1, c, raw[c], rp["deg"][c])
            for j, w in offs.items():
                room_offs[j] = room_offs.get(j, 0.0) + w
            recs[rp["region"]] = dict(info, kind=kind, intervals_s=[[float(a), float(b)] for a, b in iv],
                                      nominal_share=float(sum(b - a for a, b in iv) / base.duration), **extra)
            continue
        if rp.get("deg") is not None:
            sig, info = given_signals(body1, chans, raw, rp["deg"])
        else:
            sig, info = region_signals(body1, chans, raw, rp["level"], rp.get("gains"),
                                       rp.get("margin", SOFT_MARGIN))
        for c in chans:
            signals[c] = signals.get(c, 0.0) + sig[c]
        recs[rp["region"]] = dict(info, kind=kind, intervals_s=[[float(a), float(b)] for a, b in iv],
                                  nominal_share=float(sum(b - a for a, b in iv) / base.duration), **extra)
    arm_sig = {c: x for c, x in signals.items() if CHANNELS[c]["region"].startswith("arm_")}
    axial = {c: x for c, x in signals.items() if c not in arm_sig}
    axial_offs = offsets_of(body1, axial)
    for j, w in room_offs.items():
        axial_offs[j] = axial_offs.get(j, 0.0) + w
    out1, ap = apply_offsets(body1, axial_offs)
    rec = {"regions": recs, "table": ap, "legs": leg_info}
    if balance:
        out1, rec["balance"] = balance_standing(out1, base)
    if arms == "support":
        out1, rec["arms"] = support_arms(out1, base)
    elif arms == "hang":
        out1, rec["arms"] = hang_arms(out1, base)
    out, ap2 = apply_offsets(out1, offsets_of(out1, arm_sig))
    rec["table"] = {"scaled_joints": {**ap["scaled_joints"], **ap2["scaled_joints"]}}
    finfo = {}
    for fp in fingers or ():
        out, fi = fingers_stream(out, fp, rng, t)
        finfo[fp["side"]] = fi
    rec["fingers"] = finfo
    if guard is None:
        guard = arms is not None and (bool(arm_sig) or bool(fingers) or bool(room_offs) or
                                      any(CHANNELS[c]["region"] in ("trunk", "neck") for c in axial))
    if guard:
        out, rec["guard"] = guard_arms(out, out1, scene,
                                       speed_ramps=not any(rp.get("ballistic") for rp in region_plan))
        if arms == "support" and any(v["after_cm_max"] > 100 * GUARD_TOL_M + 1e-9
                                     for v in rec["guard"]["sides"].values()):
            ks = np.unique(np.r_[np.arange(0, out.T, max(1, int(round(out.rate / (2 * GUARD_HZ))))), out.T - 1])
            kept = out
            out, rec["lift"] = lift_off(out, base, scene, ks, seated=True)      # a hand coming back onto its thigh
            rec["contact_after"] = arm_contact_reading(out, scene, ks)
            if (max(v["after_cm_max"] for v in rec["contact_after"].values())
                    > max(v["after_cm_max"] for v in rec["guard"]["sides"].values())):
                out, rec["lift_dropped"] = kept, True                         # a deeper contact: the lift dropped
                rec["contact_after"] = arm_contact_reading(out, scene, ks)
    if arms is not None:
        how = "support_arms, the hands staying on the thighs" if arms == "support" else "hang_arms"
        rec["arms_carriage"] = (f"{arms}: the arm regions measured against the level 0 include the arms' passive "
                                f"carriage ({how}); arms_own measures the arms' own movement against that carriage")
    if arms is not None and arm_sig:
        rec["arms_own"] = {r: {k: v[k] for k in ("amplitude_share", "amplitude_channel", "share", "end_cm_p95")}
                           for r, v in measure_regions(out, out1, ("arm_left", "arm_right"),
                                                       list(arm_sig)).items() if r.startswith("arm_")}
    if legs is not None or balance:
        out = replace(out, contact=contact_labels(out))
    return out, rec


def fingers_stream(body, fp, rng, t):
    """(HandBody, info): finger movements on one hand: kind 'events' (piano-playing: per finger fragments, flexion
    mostly toward the palm) or 'noise' (a slow drift of each finger), scaled so that the largest finger's flexion
    excursion is level times its flexion range."""
    side = fp["side"]
    iv = fp["intervals"]
    T = len(t)
    raw = {}
    level = float(fp["level"])
    if fp["kind"] == "events":
        level = min(level, FINGER_LEVEL_CAP)
        ev = fragment_events(iv, list(H.FINGERS), rng, FINGER_TIMING)
        X = render_fragments(ev, list(H.FINGERS), t)
        X = X + DRIFT_SHARE * band_noise(T, 100.0, (0.3, 1.0), rng, len(H.FINGERS)) * ramp_envelope(t, iv, 0.5)[:, None]
        Xd = {f: X[:, i] for i, f in enumerate(H.FINGERS)}
        Xc = {f: Xd[f].copy() for f in H.FINGERS}
        for a_, b_, c_ in FINGER_COUPLING:              # neighbouring fingers move together
            Xc[a_] += c_ * Xd[b_]
            Xc[b_] += c_ * Xd[a_]
        flex_share = 0.2 if fp.get("supported") else 1.0     # a hand resting on the thigh mostly lifts its fingers
        ext_share = 1.0 if fp.get("supported") else 0.35      # free: flexion larger than extension
        for f in H.FINGERS:
            raw[f] = np.where(Xc[f] > 0, flex_share * Xc[f], ext_share * Xc[f])
        n_ev = len(ev)
    elif fp["kind"] == "relax":                         # given (T,) >= 0: every finger flexes a little with it
        for f in H.FINGERS:
            raw[f] = np.asarray(fp["raw"], float) * (0.5 if f == "thumb" else 1.0)
        n_ev = None
    else:
        N = band_noise(T, 100.0, fp["band"], rng, len(H.FINGERS))
        env = ramp_envelope(t, iv, 0.5)
        for i, f in enumerate(H.FINGERS):
            raw[f] = N[:, i] * env
        n_ev = None
    abd_raw = None
    if fp.get("abduction"):
        Nb = band_noise(T, 100.0, fp.get("abd_band", (0.3, 1.5)), rng, 4)
        env = ramp_envelope(t, iv, 0.5)
        abd_raw = {f: Nb[:, i] * env for i, f in enumerate(("index", "middle", "pinky", "ring"))}
    rngs = {f: finger_limits(f) for f in H.FINGERS}

    def make(g):
        fl = {f: soft_limit(g * (rngs[f][1] - rngs[f][0]) * raw[f], rngs[f][1], -rngs[f][0]) for f in H.FINGERS}
        return neighbour_clamp(fl)

    def amp(g):
        fl = make(g)
        return max((fl[f].max() - fl[f].min()) / (rngs[f][1] - rngs[f][0]) for f in H.FINGERS)
    g, a, ok = _solve_gain(amp, level)
    flex = make(g)
    abd = None if abd_raw is None else {f: fp.get("abd_deg", 6.0) * abd_raw[f] for f in abd_raw}
    out, info = apply_fingers(body, side, flex, abd, limited=True)
    return out, {"kind": fp["kind"], "level": float(fp["level"]), "level_applied": level, "gain": g, "reachable": ok,
                 "fragments": n_ev, "intervals_s": [[float(x), float(y)] for x, y in iv], "joints": info}


def neighbour_clamp(fl, gap=None):
    """{finger: (T,) deg} with each pair of neighbouring long fingers brought to within gap (FINGER_GAP_DEG) of
    each other by moving both toward their mean (a few sweeps)."""
    gap = FINGER_GAP_DEG if gap is None else gap
    out = {f: np.asarray(x, float).copy() for f, x in fl.items()}
    order = [f for f in ("index", "middle", "ring", "pinky") if f in out]
    for _ in range(6):
        for a_, b_ in zip(order, order[1:]):
            d = out[a_] - out[b_]
            ex = np.sign(d) * np.clip(np.abs(d) - gap, 0.0, None)
            out[a_] -= 0.5 * ex
            out[b_] += 0.5 * ex
    return out


def arm_contact_reading(body, scene, ks):
    """{side: {'after_cm_max'}}: the deepest probe vertex of each arm inside the scene, the trunk and thighs, or within
    the clearances of the head and the other forearm and hand (_penetration with the guard's target, heads=True) at
    the knots ks."""
    from . import viewer as V
    Vk = V.knot_vertices(body, ks)
    ctx = _target_ctx(Vk, heads=True)
    boxes = _boxes(scene, body)
    return {s: {"after_cm_max": float(100 * _penetration(Vk, s, boxes, ctx).max())} for s in ARM}


def arm_contact(rec, tol=GUARD_TOL_M):
    """The validity entry of the arms' contact (None without a contact guard): the deepest reading left on the final
    body, per side (build_guarded's 'body_after' and 'pair_after', else compose_clip's 'contact_after' after a lift,
    else the guard's own 'after'), 'ok' when every reading is at most tol (the review of wave SH1d: the guard's own
    reading of 3.4 to 5.9 cm left inside was recorded and refused nothing); the seated carriage's runs no elbow swing
    or raise cleared are counted (they enter the readings through the guard)."""
    if "body_after" in rec:
        sides = {s: v["after_cm_max"] for s, v in rec["body_after"].items()}
        sides["pair"] = rec["pair_after"]["after_cm_max"]
        how = "build_guarded"
    elif "contact_after" in rec:
        sides = {s: v["after_cm_max"] for s, v in rec["contact_after"].items()}
        how = "lift_off"
    elif "guard" in rec:
        sides = {s: v["after_cm_max"] for s, v in rec["guard"]["sides"].items()}
        how = "guard_arms"
    else:
        return None
    sw = ((rec.get("arms") or {}).get("elbow_swivel") or {})
    open_runs = sum(1 for v in sw.values() for r in v.get("runs", ()) if not r.get("clear", True))
    return {"after_cm_max": sides, "tol_cm": 100 * tol, "read_after": how, "carriage_runs_not_cleared": open_runs,
            "ok": bool(all(v <= 100 * tol + 1e-9 for v in sides.values())),
            "rule": (f"no arm vertex more than {100 * tol:g} cm inside the chair, the trunk or a thigh, or within the "
                     f"head's {1000 * HEAD_CLEAR_M:g} mm and the other hand's {1000 * PAIR_CLEAR_M:g} mm by more "
                     f"than that (module)")}


def finish(base, out, rec, operator_name, grade, scale, params, sources, seed, check_feet=(0, 1), hands=()):
    """The truth of a clip: params with the construction record and the re-measurement, validity against the base
    (and the hand limits where hands move; the seated knees' clearance, knee_clearance, where the construction went
    through the knee guard, also when the truth is made again, build_guarded; the arms' contact, arm_contact, where
    the construction went through the contact guard)."""
    val = validity(base, out, feet=check_feet)
    val["finite"] = bool(torch.isfinite(out.rot6d).all() and torch.isfinite(out.transl).all())
    if H.has_hands(out):
        val["hand_limits"] = H.hand_limit_report(out.hand_R())
    if "knee_guard" in rec:
        val["knee_clearance"] = knee_clearance(out, base)
    ac = arm_contact(rec)
    if ac is not None:
        val["arm_contact"] = ac
    params = dict(params, construction=rec, hands=list(hands))
    truth = truth_record(operator_name, _seed(seed), grade, scale, params, sources, val)
    truth["params"]["measured"] = remeasure(out, base, truth)
    return truth


# ---------------------------------------------------------------- bases
def _limits_ok(out, base, joints):
    from .body import limit_violations
    return limit_violations(out.rot6d, base.rot6d, joints=joints)["new"] == 0


FLAP_POSTURE = {"elbow_left": 100.0, "elbow_right": 100.0, "sh_flex_left": 15.0, "sh_flex_right": 15.0,
                "sh_abd_left": 20.0, "sh_abd_right": 20.0, "forearm_left": -60.0, "forearm_right": -60.0}
# [module] hands up in front of the chest, the forearms pronated so that the palms face down and a flap moves the
# hand up and down across the line of sight of the front view; the shoulders abducted 20 deg (the clinical review of
# 2026-11-03, M7: at 10 deg the hands' skin lay 13 mm apart with the wrists straight and touched at 25 deg of wrist
# flexion; at 20 deg 65 mm or more over the flap's 25 deg either way, the wrists 329 mm apart)
SEAT_FORWARD_M = 0.08       # [module] the seated base sits this far forward of base.static_body('sit'), clear of
#                             the backrest (the chair is drawn under the static sit)

# The seated rest of the hands (the clinical review of 2026-11-03, M1: the palms of the former rest hovered 39 to 44 mm
# above the thigh skin on the mesh, the forearms held nearly level): the palm lies on the thigh, the forearm slopes
# down to it. The AIMS position (Guy 1976) asks for the hands on the knees; with the trunk upright the SMPL+H neutral
# body's arm (shoulder to wrist 0.51 m on the rest skeleton) reached the middle of the thigh, so the palm rested there.
# The open body's arm reaches nearer the knee (its seated shoulder stands 0.33 m over the thigh's top), and the rest
# moves toward the knee (open-body port, wave 3 follow-up, D43).
HANDS_KEYS = ("sh_flex", "sh_abd", "sh_rot", "elbow", "forearm", "wrist_flex", "wrist_dev")
THIGH_SHARE = 0.65          # [module] the palm centre over the top of the thigh at this share of the hip-to-knee
#                             length. 0.45 on SMPL+H and in wave 3; on the open body at 0.45 the hand's skin lay 20.9
#                             mm from the abdomen and the elbow's 3.5 mm from the flank, and under the seated grade-3
#                             chorea's carriage (before the elbow's swing) the hand went 4.2 cm into the abdomen and the
#                             elbow 3.3 cm into the flank (wave 3 follow-up, D43); the smallest share of the scan (0.45,
#                             0.55, 0.6, 0.65, 0.75) whose rest clears both excursions (the rest's gap plus the depth,
#                             63 and 37 mm): 0.65, the hand 99.5 mm and the elbow 42.8 mm from the trunk (0.6: 69.6 and
#                             26.7) (scripts/openbody_wave3_followup.py, rest and carriage)
PALM_LATERAL_M = 0.02       # [module] and this far lateral of the thigh's highest point (negative: medial). At
#                             THIGH_SHARE 0.65 on the open body (wave 3 follow-up, D43), where the thighs' tops lie
#                             nearer each other: the smallest lateral place of the scan (-0.5, 0, 1, 1.5, 2, 2.5 cm)
#                             whose hands' skins lie no closer than at wave 3's rest (27.6 mm): 2 cm, 28.3 mm, the
#                             contact guard reading the rest 3.5 mm (at 1.5 cm the skins 17.9 mm apart and the reading
#                             8.4 mm, past its tolerance; at 0 the hands touch) (scripts/openbody_wave3_followup.py,
#                             rest). Wave 3, at 0.45: -5 mm, the rest the next lines read. On SMPL+H
#                             +8 mm (over its top, the two hands' skins lay 27 mm apart, and a knee swinging in brought
#                             them within 9 mm in the seated akathisia; at 15 mm the palm's edge pressed 2.4 mm into the
#                             thigh's lateral slope and the contact guard read the rest itself as touching). On the open
#                             body (wave 3), with the thumb along the index (SEATED_THUMB_CMC), the palm's distal pads
#                             reach the thigh first: at +8 mm the centre patch lay 10.2 mm off the skin at rest and 11.3
#                             under the trunk's carriage, past PALM_CONTACT_M; on the top, 8.0 and 9.6; 5 mm medial,
#                             6.6 and 9.0
#                             (scripts/openbody_wave3_checks.py, seated rest)
PALM_PITCH_DEG = 20.0       # [module] the palm faces the thigh tilted toward the knee by this (the fingers lower than
#                             the wrist, following the thigh's top as it thins toward the knee; with the palm level the
#                             solve extends the wrist 55 to 60 deg from the static sit's, against 36 to 40 at 15 deg on
#                             SMPL+H, where it was 15 deg). On the open body in wave 3, at THIGH_SHARE 0.45, 5 deg (the
#                             centre patch 6.6 mm off the skin); at the follow-up's place, where the thigh's top slopes
#                             down more, 20 deg: the patch 7.6 mm off, against 10.2 at 15 and 10.0 at 25
#                             (scripts/openbody_wave3_followup.py, rest)
ELBOW_OUT_M = 0.08          # [module] the elbow this far lateral of its shoulder joint, clear of the flank (with the
#                             elbow free to hang, the forearm's proximal skin went 2 cm into the trunk)
PALM_CENTRE_SHARE = 0.5     # [module] the palm centre halfway from the wrist joint to the middle knuckle (as
#                             postures.PALM_SHARE)
PALM_PATCH_M = 0.03         # [module] the palm's centre patch: its palmar skin within this of the palm centre
PALM_GAP_M = 0.005          # [module] the rest puts the patch's median this far from the thigh's skin (a light rest:
#                             the closest palm vertex then lies 2 mm off the skin)
PALM_MIN_GAP_M = 0.001      # [module] and every palm vertex at least this outside the skin (the background layer's
#                             depth check allows 1 mm; a rest with a palm vertex 1.95 mm inside left 1.2 to 2.1 mm added
#                             by it on the seated clips, the review of wave SH1d)
PALM_CONTACT_M = 0.010      # [module] the palm rests on the thigh when its patch's median lies within this of the
#                             skin (the review's bound, M1) and no palm vertex deeper than PALM_PRESS_M inside it
PALM_PRESS_M = 0.003        # [module] the soft tissue a resting palm may press in
REST_ROOM_DEG = 12.0        # [module] the solve keeps the shoulder, elbow and wrist this far inside the table's bounds
#                             (at 1 deg the shoulder and the wrist rested 5 to 6 deg from a bound, and carrying the
#                             hands on the thighs while the trunk or a leg moved left the table: the table's scaling
#                             lifted a palm up to 76 mm off its thigh and pressed a thumb into it in the seated
#                             akathisia); where the base itself lies nearer a bound (the static sit's wrist pronation
#                             share, 40 of 45 deg on the anatomical table), the base's own component is the margin
HANDS_ON_THIGHS = {
    "left": {"sh_flex": -1.33, "sh_abd": 9.48, "sh_rot": -31.59, "elbow": -10.62, "forearm": 18.66,
             "wrist_flex": 0.87, "wrist_dev": 0.0},
    "right": {"sh_flex": -1.33, "sh_abd": 9.48, "sh_rot": -31.59, "elbow": -10.62, "forearm": 18.66,
              "wrist_flex": 0.87, "wrist_dev": 0.0}}
# [module] channel offsets (deg) from the arms of base.POSTURES['sit'], per side, as solve_hands_on_thighs() returns
# them rounded to 0.01 deg on the open body (open-body port, wave 3 and its follow-up, D43; the test solves them again;
# the open body is symmetric, so the two sides are mirror images; the SMPL+H rest was (6.84, 7.77, -38.2, -36.91,
# 13.41, -23.92, -9.48) on the left, wave 3's at THIGH_SHARE 0.45 (-21.14, 3.87, -31.47, 5.88, 1.25, -17.91, 0)). Read
# on the open body (scripts/openbody_wave3_followup.py, rest): the palm centre at 0.65 of the hip-to-knee length and
# 2 cm lateral of the thigh's top, its normal tilted 20 deg (PALM_PITCH_DEG), the nearest palm vertex 1.0 mm off the
# skin (PALM_MIN_GAP_M) and the median of the palm's centre patch 7.6 mm, the two hands' skins 28.3 mm apart, the
# contact guard's reading of the rest 3.5 mm (its tolerance 8), the hand's skin 107.9 mm from the abdomen and the
# elbow's 44.6 mm from the flank, the forearm sloping 14.2 deg down, the thumbs along the index (SEATED_THUMB_CMC), the
# shoulders and elbows at least 12 deg inside the table (REST_ROOM_DEG) and the wrists' pronation share under the static
# sit's own (the base's own margin); the forearms leave the armrests of the static sit, so the chair is drawn without
# them (SCENE['sit'])


SEATED_THUMB_CMC = (0.0, -30.0)
# [module] the seated rest's thumbs (hands.hand_rotations: CMC flexion and palmar abduction, deg), open-body port, wave
# 3: Anny's rest thumb lies turned toward the palm, its tip 33.5 mm past the palm's most palmar vertex, so a palm laid on
# the thigh put the thumb into it, and the trunk's forward bend met it; brought back 30 deg toward the palm's plane,
# along the index, it lies 0.4 mm past it (scripts/openbody_wave3_checks.py, thumb), inside the CMC's module bound
# (hands.CMC_LIMIT_DEG). The old rest hand's thumb lay flat already.


def seated_hands(body):
    """The body as a HandBody whose thumbs take SEATED_THUMB_CMC (the other hand joints at rest)."""
    from .hands import hand_rotations
    Rh = hand_rotations({(s, "thumb", 0): SEATED_THUMB_CMC for s in ("left", "right")})
    return H.hands_of(body, matrix_to_rot6d(Rh)[None].expand(body.T, -1, 6).clone())


def _sit_forward(seconds):
    """base.static_body('sit') moved SEAT_FORWARD_M forward (the arms of base.POSTURES['sit']), the thumbs at the
    seated rest's (seated_hands)."""
    b = static_body("sit", seconds)
    return seated_hands(replace(b, transl=b.transl + torch.tensor([0.0, 0.0, SEAT_FORWARD_M], dtype=F64)))


@functools.lru_cache(maxsize=1)
def palm_vertices():
    """{side: (ids, patch)}: the palmar skin of each hand on the neutral rest mesh (the vertices skinned mostly by the
    wrist joint on the palm side of the plane through the wrist and the index and pinky knuckles, hands.hand_frame
    'palmar') and the mask of its centre patch (within PALM_PATCH_M of the palm centre, PALM_CENTRE_SHARE)."""
    J52, v, dom = H._rest()
    hf = H.hand_frame()
    out = {}
    for s in ("left", "right"):
        w = H.WRIST[s]
        ids = np.flatnonzero(dom == w)
        ids = ids[(v[ids] - J52[w]) @ hf[s]["palmar"] > 0.0]
        c = J52[w] + PALM_CENTRE_SHARE * (J52[H.hand_joint(s, "middle", 0)] - J52[w])
        out[s] = (ids, np.linalg.norm(v[ids] - c, axis=1) < PALM_PATCH_M)
    return out


def palm_gap(body, knots=None, sides=("left", "right")):
    """{side: {'patch_median_m', 'min_m'} (n,) m}: per knot (all, or the given ones) the signed distance of each hand's
    palmar skin (palm_vertices) to the skin of the body without its arms (physio.signed_distance, negative inside):
    the median over the centre patch and the minimum over the palm."""
    from . import viewer as V
    from .physio import signed_distance
    ks = list(range(body.T)) if knots is None else [int(k) for k in knots]
    Vk = V.knot_vertices(body, ks)
    pv = palm_vertices()
    out = {}
    for s in sides:
        ids, patch = pv[s]
        sd = np.stack([signed_distance(v, v[ids]) for v in Vk])
        out[s] = {"patch_median_m": np.median(sd[:, patch], axis=1), "min_m": sd.min(1)}
    return out


def palm_rest_report(body, knots=None, sides=("left", "right")):
    """{side: {...}, 'ok'}: the palms resting on the thighs (PALM_CONTACT_M, PALM_PRESS_M) at the knots, JSON-ready."""
    g = palm_gap(body, knots, sides)
    rep = {s: {"patch_median_mm_max": float(1000 * g[s]["patch_median_m"].max()),
               "patch_median_mm_min": float(1000 * g[s]["patch_median_m"].min()),
               "palm_min_mm": float(1000 * g[s]["min_m"].min())} for s in sides}
    rep["ok"] = bool(all(g[s]["patch_median_m"].max() <= PALM_CONTACT_M and g[s]["min_m"].min() >= -PALM_PRESS_M
                         for s in sides))
    rep["rule"] = (f"the median of the palm's centre patch within {1000 * PALM_CONTACT_M:g} mm of the skin of the "
                   f"body, no palm vertex more than {1000 * PALM_PRESS_M:g} mm inside it (module)")
    return rep


def palm_off(Vk, side, off=PALM_CONTACT_M):
    """(n,) bool: per knot of the vertices Vk (n, V, 3) the palm of side off the skin of the body (the median of
    its centre patch, palm_vertices, farther than off from it, physio.signed_distance)."""
    from .physio import signed_distance
    ids, patch = palm_vertices()[side]
    return np.array([float(np.median(signed_distance(v, v[ids])[patch])) > off for v in Vk], bool)


def _raw_arms(b, sig):
    """b with the channel signals {name: (T,) deg} composed raw (no table scaling): the solver's smooth pose."""
    offs = offsets_of(b, sig)
    R = b.local_R().clone()
    for j, w in offs.items():
        R[:, j] = _rot(torch.as_tensor(w, dtype=F64)) @ R[:, j]
    return b.with_rotations(R, list(offs))


def solve_hands_on_thighs(iters=13):
    """({side: {key: deg}}, record): the seated rest of the hands (module comment above HANDS_KEYS), per side: the arm
    channels HANDS_KEYS from the arms of base.POSTURES['sit'] (_sit_forward) that put the palm centre over the top of
    its thigh (PALM_LATERAL_M lateral of the highest thigh vertex of the rest mesh posed seated within 12 mm of
    THIGH_SHARE of the hip-to-knee length; the two sides' points made mirror images of each other about the pelvis,
    their mean), the palm's normal (hands.hand_frame) facing down tilted PALM_PITCH_DEG toward the knee, the elbow
    ELBOW_OUT_M lateral of its shoulder and the wrist without deviation, with a penalty REST_ROOM_DEG inside every
    bound of the table, by least squares on the joints (from zero offsets); the palm centre's height over the thigh's
    top found by bisection on the mesh (iters steps between -1 and 8 cm): the lowest at which the patch's median
    distance to the skin is PALM_GAP_M or more and every palm vertex PALM_MIN_GAP_M or more outside the skin (palm_gap:
    the review of wave SH1d found the left palm's deepest vertex 1.95 mm inside at a median of 5 mm); the sides solved
    in turn twice, each with the other arm at its latest solution (moving one arm moves the other palm's gap by about
    1 mm through the pose blend shapes)."""
    from scipy.optimize import least_squares
    from . import viewer as V
    from .body import components_deg
    b0 = _sit_forward(0.01)
    J52, _, dom = H._rest()
    hf = H.hand_frame()
    lim = joint_limits()
    c0 = components_deg(b0.rot6d)[0]
    Jb = b0.joints().numpy()[0]
    Vb = V.knot_vertices(b0, [0])[0]
    p = math.radians(PALM_PITCH_DEG)
    want = np.array([0.0, -math.cos(p), -math.sin(p)])
    out, rec = {s: {k: 0.0 for k in HANDS_KEYS} for s in ("left", "right")}, {}
    lat = _lateral_frame(b0)[0]
    tops = {}
    for s in ("left", "right"):
        hip, kn = (1, 4) if s == "left" else (2, 5)
        u = (Jb[kn] - Jb[hip]) / np.linalg.norm(Jb[kn] - Jb[hip])
        L = float(np.linalg.norm(Jb[kn] - Jb[hip]))
        P = Vb[np.flatnonzero(np.isin(dom, (hip, kn)))]
        sl = P[np.abs((P - Jb[hip]) @ u - THIGH_SHARE * L) < 0.012]
        tops[s] = sl[np.argmax(sl[:, 1])] - Jb[0]
    side_m = 0.5 * (tops["left"] @ lat - tops["right"] @ lat)       # the mirror images' mean about the pelvis
    rest_m = 0.5 * ((tops["left"] - (tops["left"] @ lat) * lat) + (tops["right"] - (tops["right"] @ lat) * lat))
    for s in ("left", "right", "left", "right"):        # twice: the mesh's pose blend shapes couple the two sides
        other = {f"{k}_{o}": np.full(b0.T, v) for o in ("left", "right") if o != s for k, v in out[o].items()}
        c, sh, el, wr = ARM[s]
        top = Jb[0] + rest_m + SIDE_SIGN[s] * side_m * lat
        off = PALM_CENTRE_SHARE * (J52[H.hand_joint(s, "middle", 0)] - J52[wr])
        rows = [sh, el, wr]
        lo_t, hi_t = np.minimum(lim.lo[rows], c0[rows]), np.maximum(lim.hi[rows], c0[rows])
        # the margin REST_ROOM_DEG inside the table, or the base's own component where the base already lies nearer
        # its bound (open-body port, wave 3: the static sit's wrist holds 39 deg of its 45 deg pronation share)
        lo_r = np.minimum(lo_t + REST_ROOM_DEG, np.maximum(c0[rows], lo_t))
        hi_r = np.maximum(hi_t - REST_ROOM_DEG, np.minimum(c0[rows], hi_t))

        def pose(x):
            return _raw_arms(b0, dict(other, **{f"{k}_{s}": np.full(b0.T, v) for k, v in zip(HANDS_KEYS, x)}))

        def solve(h, x0):
            tgt = top + np.array([SIDE_SIGN[s] * PALM_LATERAL_M, h, 0.0])

            def res(x):
                b = pose(x)
                J, G = (a.numpy()[0] for a in b.fk())
                pc = J[wr] + G[wr] @ off
                cc = components_deg(b.rot6d)[0, rows]
                ex = np.concatenate([np.maximum(lo_r - cc, 0.0), np.maximum(cc - hi_r, 0.0)]).ravel()
                return np.concatenate([1000.0 * (pc - tgt), 300.0 * (G[wr] @ hf[s]["palmar"] - want),
                                       [50.0 * (SIDE_SIGN[s] * (J[el, 0] - J[sh, 0]) - ELBOW_OUT_M)], 30.0 * ex,
                                       [0.3 * x[HANDS_KEYS.index("wrist_dev")]]])
            sol = least_squares(res, x0, x_scale=10.0, max_nfev=300)
            return sol.x, float(np.abs(sol.fun[:3]).max())
        x = np.array([out[s][k] for k in HANDS_KEYS])
        lo, hi = -0.01, 0.08
        best = None
        for _ in range(iters):
            h = 0.5 * (lo + hi)
            x, err = solve(h, x)
            pg = palm_gap(pose(x), [0], (s,))[s]
            g, m = float(pg["patch_median_m"][0]), float(pg["min_m"][0])
            if g < PALM_GAP_M or m < PALM_MIN_GAP_M:
                lo = h
            else:
                hi, best = h, (h, x.copy(), err, g, m)
        h, x, err, g, m = best if best is not None else (h, x, err, g, m)
        out[s] = {k: float(v) for k, v in zip(HANDS_KEYS, x)}
        rec[s] = {"height_over_top_m": float(h), "target_error_mm": err, "patch_median_m": g, "palm_min_m": m,
                  "thigh_top_m": [float(v) for v in top]}
    return out, rec


def base_body(kind, seconds):
    """The level-0 base of a clip: 'sit' (base.static_body('sit') moved SEAT_FORWARD_M forward with the palms resting
    on the thighs, HANDS_ON_THIGHS), 'stand' (stand_soft), 'arms_forward' (the seated 3.15 posture,
    base.POSTURES['sit_arms_forward']), 'flap' (stand_soft with FLAP_POSTURE, the hands before the chest)."""
    if kind == "sit":
        b = _sit_forward(seconds)
        sig = {f"{k}_{s}": np.full(b.T, v) for s in ("left", "right") for k, v in HANDS_ON_THIGHS[s].items()}
        out, _ = apply_offsets(b, offsets_of(b, sig))
        return out
    if kind == "stand":
        return stand_soft(seconds)
    if kind == "arms_forward":
        return static_body("sit_arms_forward", seconds)
    if kind == "flap":
        b = stand_soft(seconds)
        sig = {c: np.full(b.T, v) for c, v in FLAP_POSTURE.items()}
        out, _ = apply_offsets(b, offsets_of(b, sig))
        return out
    raise ValueError(f"base {kind!r}")


SCENE = {"sit": "chair_armless", "arms_forward": "chair", "stand": None, "flap": None}
CARRIAGE = {"sit": ("support", SCENE["sit"], False), "stand": ("hang", None, True),
            "arms_forward": (None, SCENE["arms_forward"], False)}
# posture -> (arms' carriage, scene of the contact guard, standing balance): seated, the hands rest on the thighs
# (support_arms) and the guard checks the drawn chair (its seat and backrest) and the body; standing, the arms hang
# (hang_arms), the guard checks the body, and the pelvis moves back under a bowing trunk (balance_standing); the arms
# outstretched keep their posture (no carriage) and the guard, where a clip asks for it (the proximal arm chorea of
# the piano-playing fingers), checks the drawn chair with its armrests, the body, the head and the other arm

# ---------------------------------------------------------------- channel sets
ARM_KEYS = ("collar", "sh_abd", "sh_flex", "sh_rot", "elbow", "forearm", "wrist_flex", "wrist_dev")
ARM_GAINS = {"collar": 0.5, "sh_rot": 0.6, "wrist_dev": 0.6}       # [module] lesser components of an arm fragment
TRUNK_CH = ("trunk_lat", "trunk_rot", "trunk_flex")
TRUNK_GAINS = {"trunk_flex": 0.4}                                   # [module]
NECK_CH = ("neck_lat", "neck_rot", "neck_flex")
NECK_GAINS = {"neck_flex": 0.5, "neck_rot": 0.6}                    # [module]
DISTAL_KEYS = ("forearm", "wrist_flex", "wrist_dev", "elbow")
DISTAL_GAINS = {"elbow": 0.3}                                       # [module] athetosis: distal, the elbow a little
PROXIMAL_KEYS = ("sh_abd", "sh_flex", "sh_rot", "elbow", "collar")
PROXIMAL_GAINS = {"elbow": 0.6, "collar": 0.5}                      # [module] ballism: proximal
LEG_CH = {s: tuple(f"{k}_{s}" for k in ("hip_flex", "hip_abd", "hip_rot", "knee", "ankle")) for s in ("left", "right")}
LEG_REF = {"sw_deg": 15.0, "lift_m": 0.08, "pitch_deg": 10.0}       # [module] reference sizes of a seated leg
SWIVEL_IN_OUT = (0.4, 1.6)  # [module] weights of an inward and an outward knee swivel (the inward swivel the
#                             smaller; knee_guard keeps the seated knees apart)
PELVIS_REF = {"dx": 0.05, "dz": 0.03, "roll": 5.0, "yaw": 10.0, "tilt": 5.0, "dip": 0.03}   # [module] standing


TRUNK_NECK_CAP = 0.45      # [module] trunk and neck amplitude at most this share of the ROM: end-range bows, head
#                            throws and ear-to-shoulder tilts read as a caricature (clinical review of 2026-11-02)
TRUNK_NECK_MARGIN = 0.55   # [module] the trunk and neck soft limit at this share of the room in each direction
SUPPORTED_SIGNS = {"elbow": 1.0, "sh_flex": 1.0, "sh_abd": 1.0, "collar": 1.0, "wrist_flex": -1.0}
# [module] seated with the hands on the thighs, an arm fragment can only leave the thigh: elbow flexion, shoulder
# flexion and abduction, shoulder elevation, wrist extension (the other channels turn both ways; made one-sided as
# well, 2026-11-04, the seated grade-3 chorea still pressed a hand 3.5 cm into its thigh after the guard)
HANGING_SIGNS = {"sh_abd": 1.0}     # [module] standing, the hanging arm moves away from the trunk, never into it
SUPPORTED_LIFT = 0.5        # [module] seated, an arm's flowing fragments flex the elbow at least this share of the
#                             largest target of the arm's other channels (lift_events): a hand resting on its thigh
#                             lifts off it as it moves (rolled or swung on the thigh, the palm on the skin since the
#                             review's M1, a thumb went 5.9 cm into the thigh on the seated grade-3 chorea and the
#                             contact guard held that arm at its rest 36 percent of the time, 2026-11-05)


def lift_events(events, lift_ch, k=SUPPORTED_LIFT):
    """events (fragment_events, flowing) with the target of lift_ch (the elbow, one-sided: flexion) at least k times
    the largest absolute target the arm's other channels hold after each fragment (their targets carried over the
    bout, every channel back to rest at its end), the channel's own target kept where larger; changed in place."""
    cur, own, bout = {}, 0.0, None
    for e in sorted(events, key=lambda e: e["t0"]):
        if not e.get("flow"):
            continue
        if e["bout_end"] != bout:
            cur, own, bout = {}, 0.0, e["bout_end"]
        for c, a in e["amps"].items():
            if c == lift_ch:
                own = abs(a)
            else:
                cur[c] = a
        e["amps"][lift_ch] = max(own, k * max((abs(v) for v in cur.values()), default=0.0))
        e["delay"].setdefault(lift_ch, 0.0)
    return events


NOISE_LIFT = 0.5            # [module] seated, a writhing arm (kind 'noise') flexes the elbow at least this share of
#                             the size of its other channels' raw signals (lift_noise; the elbow weighed by its gain,
#                             DISTAL_GAINS 0.3): with the palms on the thighs since the review's M1, the forearm's roll
#                             and the wrist's deviation of the athetosis pressed the hand's edge 3.2 cm into its thigh
#                             and the contact guard held the left arm back 35 percent of the time (its own amplitude
#                             0.24 of the range of motion at the DIS level 0.30); on that 8 s build at 0.5: 1.5 and 2.0
#                             cm before the guard, held back 12 percent of the time, own amplitude 0.29 to 0.30 (at
#                             1.0 and 2.0 the more flexed elbow brought the hand into the body more, 2026-11-05)
NOISE_LIFT_SOFT = 0.05      # [module] softness of the smooth maximum (raw units) between the elbow's own signal and the
#                             lift
GUARDED_FINGER_SHARE = 0.9  # [module] the finger amplitude a seated writhing hand keeps against its level at least: the
#                             contact guard takes a hand meeting its thigh back toward its rest, the fingers with it
#                             (0.28 against the athetosis's 0.30 on an 8 s build, inside DIS band 2, 0.10 to 0.50)


def lift_noise(raw, lift_ch, k=NOISE_LIFT, soft=NOISE_LIFT_SOFT):
    """(T,): the raw signal of lift_ch (the elbow, one-sided: flexion) at least k times the norm of the other
    channels' raw signals at each knot (a smooth norm and a smooth maximum, so that the lift adds no kink), the
    channel's own signal kept where larger."""
    others = [np.asarray(v, dtype=np.float64) for c, v in raw.items() if c != lift_ch]
    a = np.asarray(raw[lift_ch], dtype=np.float64)
    b = k * (np.sqrt(sum(v * v for v in others) + soft * soft) - soft)
    return 0.5 * (a + b + np.sqrt((a - b) ** 2 + soft * soft)) - 0.5 * soft


SUPPORTED_REST_P = 0.45     # [module] seated, an arm fragment's chance of going back to the thigh (free: TIMING)


def arm_timing(name, posture):
    """TIMING[name], with the hands resting on the thighs more often seated (SUPPORTED_REST_P): a one-sided arm
    that seldom returned to its support held the arms up in front of the body."""
    return dict(TIMING[name], rest_p=SUPPORTED_REST_P) if posture == "sit" else name


def arm_ch(side, keys=ARM_KEYS):
    return tuple(f"{k}_{side}" for k in keys)


def arm_signs(side, posture, keys=ARM_KEYS):
    """{channel: sign} of the one-sided arm channels of side in posture (SUPPORTED_SIGNS, HANGING_SIGNS)."""
    table = SUPPORTED_SIGNS if posture == "sit" else HANGING_SIGNS if posture == "stand" else {}
    return {f"{k}_{side}": v for k, v in table.items() if k in keys}


def axial_entries(posture, level, rng, base, share, timing="chorea"):
    """The trunk and neck entries of a generalized hyperkinesia: level capped at TRUNK_NECK_CAP, the soft limit at
    TRUNK_NECK_MARGIN of the room, seated the trunk bending forward only from upright (the backrest)."""
    lv = min(level, TRUNK_NECK_CAP)
    sg = {"trunk_flex": 1.0} if posture == "sit" else None
    return [_entry("trunk", TRUNK_CH, lv, _iv(base, share, rng), timing=timing, gains=TRUNK_GAINS,
                   margin=TRUNK_NECK_MARGIN, signs=sg),
            _entry("neck", NECK_CH, lv, _iv(base, share, rng), timing=timing, gains=NECK_GAINS,
                   margin=TRUNK_NECK_MARGIN)]


def arm_gains(side, gains=ARM_GAINS):
    return {f"{k}_{side}": v for k, v in gains.items()}


def _entry(region, channels, level, intervals, kind="events", timing="chorea", gains=None, **kw):
    return dict(region=region, channels=channels, level=level, intervals=intervals, kind=kind, timing=timing,
                gains=gains, **kw)


# ---------------------------------------------------------------- legs: seated fragments and standing sway
def _pulse(t, t0, r, h, f):
    return minimum_jerk((t - t0) / r) - minimum_jerk((t - (t0 + r + h)) / f)


TRAVEL_RHO = (0.8, 1.5)      # [module] a seated foot excursion's horizontal travel per unit of its lift
TRAVEL_MIN_S = 0.25          # [module] the lifted foot's horizontal travel out (and back) takes at least this long
AIR_M = 0.065                # [module] the foot travels and pitches only while lifted above this (H0 = 5 cm + 1.5)
CHOREA_KICK_P = {3: 0.3, 4: 0.5}   # [module] seated chorea of the legs: the share of the foot excursions thrown
#                                    forward as kicks at grades 3 and 4 (the clinical review of 2026-11-03: grade 4
#                                    showed no kick and read as moderate)
KICK_RHO = (1.6, 2.2)        # [module] a choreic kick's forward travel per unit of its lift (the knee extending)
KICK_PHI = 0.3               # [module] rad: a kick's direction at most this from straight forward
KICK_MIN_S = 0.45            # [module] a kick rises and returns over at least this (at chorea's 0.2 s rise a 16 cm
#                              kick threw the ankle at 3.6 m/s, a ballistic fling)
KICK_HOLD_S = 0.4            # [module] and stays up at least this, so that its travel, airborne only, has the time of
#                              a kick (airborne_travel; travelling from the lift's start, as the cube of the lift
#                              pulse, the foot slid on the floor while still a contact: 0.13 m/s at grade 3, 2026-11-05)
KICK_LIFT = 1.4              # [module] a kick lifts the ankle this much more than a plain excursion of its size, so
#                              that the foot clears the floor (AIR_M) before it is thrown forward (a floor in metres,
#                              summed over overlapping kicks, lifted a foot out of the table at every gain)
KICK_TRAVEL_S = 0.35         # [module] a kick's forward throw (and its return) takes at least this (airborne_travel)


def swivel_scale(side):
    """(positive, negative) weights of a seated knee swivel of side (SWIVEL_IN_OUT by SWIVEL_OUT)."""
    i, o = SWIVEL_IN_OUT
    return (o, i) if SWIVEL_OUT[side] > 0 else (i, o)


def seated_leg_fragments(t, intervals, rng, timing, swivel_scale=None):
    """Raw seated-leg signals (module docstring, Feet): 'sw' the knee swivel (signed, weighted by swivel_scale
    (outward, inward) when given), 'lift' the foot lift (>= 0), 'fwd' and 'lat' the lift-weighted direction of each
    fragment's horizontal travel (TRAVEL_RHO travel per unit lift, an inward travel at half), in units of LEG_REF;
    seated_leg_region turns the travel into airborne_travel (a timed minimum-jerk excursion while the foot is up).
    timing 'kick_p', if given: that share of the excursions are kicks ('kicks': per kick its lift pulse, rise and
    return at least KICK_MIN_S, held at least KICK_HOLD_S, its size, KICK_PHI of forward and KICK_RHO travel per unit
    of its size); a kick lifts the foot KICK_LIFT times its size, and seated_leg_region throws the foot only while
    the kick's own lift holds it above AIR_M, over at least KICK_TRAVEL_S (through the common airborne gate, at the
    common lift, the grade-4 kicks brought the ankle only 66 mm forward, 2026-11-04)."""
    ev = fragment_events(intervals, ["sw", "ex"], rng, dict(timing, mode="pulse"))   # the foot lands where it rose
    kick_p = float(timing.get("kick_p", 0.0))
    z = np.zeros(len(t))
    sw, lift, fwd, lat = z.copy(), z.copy(), z.copy(), z.copy()
    kicks = []
    for e in ev:
        for c, a in e["amps"].items():
            d = e["delay"][c]
            r = max(e["rise"] - d, 0.05)
            p = _pulse(t, e["t0"] + d, r, e["hold"], e["ret"])
            if c == "sw":
                sp, sm = (1.0, 1.0) if swivel_scale is None else swivel_scale
                sw += a * (sp if a > 0 else sm) * p
            else:
                kick = kick_p > 0.0 and rng.uniform() < kick_p
                if kick:
                    p = _pulse(t, e["t0"] + d, max(r, KICK_MIN_S), max(e["hold"], KICK_HOLD_S),
                               max(e["ret"], KICK_MIN_S))
                lv = abs(a) * (KICK_LIFT if kick else 1.0) * p
                lift += lv
                if kick:
                    phi, rho = rng.uniform(-KICK_PHI, KICK_PHI), rng.uniform(*KICK_RHO)
                    kicks.append({"pulse": p, "size": abs(a), "fwd": rho * math.cos(phi), "lat": rho * math.sin(phi)})
                    continue
                phi = rng.uniform(-math.pi / 2 - 0.5, math.pi / 2 + 0.5)
                rho = rng.uniform(*TRAVEL_RHO)
                fwd += rho * math.cos(phi) * lv
                lat += rho * math.sin(phi) * lv
    return {"sw": sw, "lift": lift, "fwd": fwd, "lat": lat, "kicks": kicks}, len(ev)


def airborne_travel(t, lift_m, air=AIR_M, min_s=TRAVEL_MIN_S):
    """((T,) profile, runs): per run of lift_m above air (s), a minimum-jerk rise from 0 to 1 that takes
    max(min_s, a third of the run) and a minimum-jerk return to 0 ending with the run (both at most half the run); a
    run shorter than 2 min_s is scaled down by its length over 2 min_s, so that no travel is faster than a 2 min_s
    excursion. The profile is 0 wherever the foot is below air: a planted or landing foot never slides."""
    up = np.asarray(lift_m) > air
    prof = np.zeros(len(t))
    runs = []
    for a, b in _runs(up):
        ta, tb = float(t[a]), float(t[b - 1])
        span = tb - ta
        if span <= 0.02:
            continue
        d = min(max(min_s, span / 3.0), span / 2.0)
        k = min(1.0, span / (2.0 * min_s))
        prof[a:b] = k * (minimum_jerk((t[a:b] - ta) / d) - minimum_jerk((t[a:b] - (tb - d)) / d))
        runs.append((int(a), int(b)))
    return prof, runs


def run_travel(sig, runs, n):
    """(fwd, lat, lift) (T,) raw values of each airborne run held over the run: the travel and the lift at the
    run's highest lift (seated_leg_fragments)."""
    fwd, lat, lift = np.zeros(n), np.zeros(n), np.zeros(n)
    for a, b in runs:
        k = a + int(np.argmax(sig["lift"][a:b]))
        fwd[a:b], lat[a:b], lift[a:b] = sig["fwd"][k], sig["lat"][k], sig["lift"][k]
    return fwd, lat, lift


def _gain_with_limits(make, amp_of, ok_of, level, iters=24, hi=12.0):
    """(g, body, amplitude, reachable): the gain whose body reaches level, or the largest gain inside the table when
    that comes first (bisection on log g; amplitude rising with g)."""
    def at(g):
        try:
            b = make(g)
        except ValueError:                               # an ankle beyond the leg's reach: outside, as the table
            return None, math.inf, False
        return b, amp_of(b), ok_of(b)
    b_hi, a_hi, ok_hi = at(hi)
    if a_hi < level and ok_hi:
        return hi, b_hi, a_hi, False
    lo_, hi_ = math.log(1e-3), math.log(hi)
    for _ in range(iters):
        m = 0.5 * (lo_ + hi_)
        _, a, ok = at(math.exp(m))
        if ok and a < level:
            lo_ = m
        else:
            hi_ = m
    g = math.exp(hi_)
    b, a, ok = at(g)
    if not ok:                                           # the table binds before the level: the last gain inside
        g = math.exp(lo_)
        b, a, ok = at(g)
    return g, b, a, bool(ok and abs(a - level) <= 0.01 * level + 1e-9)


def seated_leg_region(body, side, sig, level, ref=LEG_REF):
    """(Body, info): the seated leg of side moved by raw signals (seated_leg_fragments) at the gain whose leg
    amplitude (the largest share over LEG_CH) is level, or at the largest gain inside the table."""
    rows = list(LEG[side][:3])
    t = body.times().numpy()

    def make(g):
        lift = g * ref["lift_m"] * sig["lift"]
        kf, kl = np.zeros(body.T), np.zeros(body.T)
        for k in sig.get("kicks", ()):                   # a kick: thrown only while its own lift holds it airborne
            pk, _ = airborne_travel(t, g * ref["lift_m"] * KICK_LIFT * k["size"] * k["pulse"], min_s=KICK_TRAVEL_S)
            reach = g * ref["lift_m"] * k["size"]
            kf, kl = kf + k["fwd"] * reach * pk, kl + k["lat"] * reach * pk
        prof, runs = airborne_travel(t, lift)
        fr, lr, lp = run_travel(sig, runs, body.T)
        R, _ = seated_legs(body, side, lift=lift,
                           fwd=g * ref["lift_m"] * fr * prof + kf,
                           lat=g * ref["lift_m"] * lr * prof + kl,
                           swivel_deg=g * ref["sw_deg"] * sig["sw"],
                           pitch_deg=g * ref["pitch_deg"] * lp * prof)
        return body.with_rotations(R, rows)

    def amp(b):
        return max(extent_share(channel_values(b, body, c), channel_rom(c)) for c in LEG_CH[side])
    g, out, a, ok = _gain_with_limits(make, amp, lambda b: _limits_ok(b, body, rows), level)
    return out, {"gain": g, "level": float(level), "realized": float(a), "reachable": ok,
                 "lift_max_m": float(g * ref["lift_m"] * sig["lift"].max()),
                 "swivel_max_deg": float(g * ref["sw_deg"] * np.abs(sig["sw"]).max())}


# ---------------------------------------------------------------- legs: the seated knees kept apart
KNEE_CLEARANCE_M = 0.010       # [module] seated: the smallest hard leg-to-leg gap kept (legcontact 'hard_m': a knee,
#                                shank or foot against the other leg); the rigid mesh cannot show two knees pressed
#                                together (handoff 13: the tardive limb clips' knees came close, then passed through)
KNEE_GUARD_MARGIN_M = 0.003    # [module] the guard and the closing ceiling keep this above the clearance: the stages
#                                after the legs (the arms on the thighs, the trunk, the fingers) change the leg surface
#                                a little through the pose blend shapes, and the final body is measured again
#                                (knee_clearance)
KNEE_GUARD_AIM_M = 0.001       # [module] a guard round aims this far beyond the margin, so that its rounds end above it
KNEE_GUARD_PREFILTER_M = 0.02  # [module] a knot whose knees, ankles and feet lie no closer together than in the body
#                                before its legs moved, less this, is not measured: on the swivel probe the hard gap
#                                follows the knee distance about one to one (the tests measure every knot)
KNEE_GUARD_RANGE_M = 0.02      # [module] the gap is measured exactly up to the clearance plus this (legcontact cull)
KNEE_GUARD_SIGMA_S = 0.08      # [module] the guard's outward swivel: held 2.5 sigma on each side of a knot that needs
#                                it (a moving maximum), then smoothed by a Gaussian of this sigma
KNEE_GUARD_ITERS = 8           # [module] at most this many rounds of measure and swivel
KNEE_GUARD_SPLIT_M = 0.005     # [module] softness (softplus) of the split of a swivel between the knees, by how far
#                                each knee has come inward from the body before its legs moved
KNEE_GUARD_RATE_MIN = 0.0005   # [module] m of gap per degree: a knot whose one-degree probe opens its gap by less is
#                                left to the report (an outward knee swivel does not open that contact)
KNEE_GUARD_MAX_DEG = 10.0      # [module] the guard's outward swivel at a knot, summed over its rounds and both knees,
#                                at most this: a contact that a larger swivel would be needed to open (the shanks or
#                                feet crossing with the knees apart) is reported ('unresolved_knots', ok False), not
#                                forced; the seated clips need a few degrees (their truths, knee_guard)
KNEE_GUARD_SHARES = (1.0, 0.75, 0.5, 0.25, 0.0)   # [module] the shares of the guard's swivel tried at a knot where
#                                it would add a joint-limit violation (_guard_within_limits)
CLOSING_BLEND_DEG = 2.0        # [module] the closing clamp's safety net: above the ceiling, the closing swivel blends
#                                over this (tanh) toward the ceiling plus this
CLOSING_SPLIT_DEG = 1.0        # [module] softness (softplus) of the split of a cut between the knees, by their swivel
CLOSING_SPLITS = ((0.5, 0.0), (1.0, 0.0), (0.0, 0.0), (1.0, 10.0), (0.0, -10.0), (1.0, 20.0), (0.0, -20.0))
# [module] the probe of closing_ceiling: a closing S is split (share * S + offset, the rest) between the left and the
# right knee, an offset carrying one knee past the closing and the other outward by as much


def _softplus(x):
    return np.logaddexp(0.0, x)


def leg_gaps(body, knots, cull_m=KNEE_CLEARANCE_M + KNEE_GUARD_RANGE_M):
    """legcontact.gap_from_pose at body's knots (indices): {'hard_m', 'soft_m', 'where_hard'} on those knots."""
    from .legcontact import gap_from_pose
    k = torch.as_tensor(np.asarray(knots, dtype=np.int64))
    return gap_from_pose(body.local_R()[k].numpy(), body.transl[k].numpy(), body.betas.numpy(), int(body.gender),
                         cull_m=cull_m)


def _pair_distances(J):
    """(T, 3) m: the distances between the two knees, the two ankles and the two foot joints."""
    return np.stack([np.linalg.norm(J[:, a] - J[:, b], axis=-1) for a, b in ((4, 5), (7, 8), (10, 11))], -1)


def knee_candidates(body, before, prefilter=KNEE_GUARD_PREFILTER_M):
    """Knots where the knees, the ankles or the feet of body lie closer together than in before (the body before its
    legs moved) less prefilter: the knots the knee guard measures."""
    d, d0 = _pair_distances(body.joints().numpy()), _pair_distances(before.joints().numpy())
    return np.flatnonzero((d < d0 - prefilter).any(-1))


def _swivel_out(body, deg):
    """body with each side's knee turned outward by deg[side] (T,) degrees about its hip-to-ankle line (seated_legs:
    the hip, the ankle and the foot's global rotation kept), its rows changed only where deg is not 0."""
    out = body
    for side, d in deg.items():
        sel = np.flatnonzero(np.abs(d) > 0.0)
        if not len(sel):
            continue
        R, _ = seated_legs(out, side, swivel_deg=SWIVEL_OUT[side] * np.asarray(d, np.float64))
        rows = list(LEG[side][:3])
        r6 = out.rot6d.clone()
        k = torch.as_tensor(sel)
        r6[k[:, None], torch.as_tensor(rows)[None, :]] = matrix_to_rot6d(R[k][:, rows])
        out = replace(out.copy(), rot6d=r6)
    return out


def _moving_max(x, m):
    if m < 1:
        return np.asarray(x, np.float64).copy()
    p = np.pad(np.asarray(x, np.float64), m, mode="edge")
    return np.lib.stride_tricks.sliding_window_view(p, 2 * m + 1).max(-1)


def _inward_weights(body, before, soft_m=KNEE_GUARD_SPLIT_M):
    """(T,) the left knee's share of a swivel split between the knees: softplus of how far each knee has come inward
    (along the body's left axis, the pelvis's x axis of before) over soft_m, normalized."""
    J, (J0, G0) = body.joints().numpy(), before.fk()
    J0, left = J0.numpy(), G0[:, 0, :, 0].numpy()
    left = left / np.linalg.norm(left, axis=-1, keepdims=True)
    il = -((J[:, 4] - J0[:, 4]) * left).sum(-1)
    ir = ((J[:, 5] - J0[:, 5]) * left).sum(-1)
    pl, pr = _softplus(il / soft_m), _softplus(ir / soft_m)
    return pl / (pl + pr)


def knee_guard(body, before, clearance=KNEE_CLEARANCE_M, iters=KNEE_GUARD_ITERS):
    """(Body, info): a seated body whose legs moved (body; before, the body before its legs moved) kept with every
    hard leg-to-leg gap of the skinned mesh at least clearance (module docstring, Feet: the knee guard). The knots
    where the legs came together (knee_candidates) are measured (leg_gaps); where a hard gap lies below clearance
    plus KNEE_GUARD_MARGIN_M, both knees swivel outward about their hip-to-ankle lines (_swivel_out: the hips,
    ankles and feet stay where they are) by the swivel a finite difference of one degree says lifts the gap to
    KNEE_GUARD_MARGIN_M plus KNEE_GUARD_AIM_M above clearance, split between the knees by how far each has come
    inward (_inward_weights), held around the knot and smoothed (KNEE_GUARD_SIGMA_S); then measured again, up to
    iters rounds, until every hard gap lies KNEE_GUARD_MARGIN_M above clearance ('ok': at least clearance). A knot
    whose probe opens its gap by less than KNEE_GUARD_RATE_MIN per degree is left out of the later rounds, and the
    swivel at a knot, summed over the rounds, never passes KNEE_GUARD_MAX_DEG, nor adds a joint-limit violation
    (_guard_within_limits): a contact that an outward knee swivel does not open is reported ('unresolved_knots', the
    knots whose hard gap stays below clearance; ok False), not forced. A body whose gaps hold comes back bit for
    bit."""
    cand = knee_candidates(body, before)
    info = {"clearance_m": float(clearance), "margin_m": KNEE_GUARD_MARGIN_M, "measured_knots": int(len(cand)),
            "rounds": 0, "guarded_knots": 0, "swivel_out_deg_max": {"left": 0.0, "right": 0.0},
            "swivel_out_deg_max_total": 0.0, "max_deg": KNEE_GUARD_MAX_DEG, "dropped_knots": 0,
            "unresolved_knots": [], "limit_scaled_knots": 0, "prefilter_m": KNEE_GUARD_PREFILTER_M,
            "rule": "outward knee swivel about the hip-to-ankle line where a hard leg-to-leg gap (legcontact) lies "
                    "below the clearance plus the margin, at most the module's total swivel; a contact the swivel "
                    "does not open reported, not forced (module)"}
    if not len(cand):
        info.update(hard_min_m=None, soft_min_m=None, hard_min_before_m=None, ok=True)
        return body, info
    g = leg_gaps(body, cand)
    hard = np.asarray(g["hard_m"])
    info["hard_min_before_m"] = float(hard.min())
    out = body
    T = body.T
    total = {"left": np.zeros(T), "right": np.zeros(T)}
    dropped = np.zeros(len(cand), bool)            # knots an outward swivel does not open, or at the total swivel
    keep, target = clearance + KNEE_GUARD_MARGIN_M, clearance + KNEE_GUARD_MARGIN_M + KNEE_GUARD_AIM_M
    sig = KNEE_GUARD_SIGMA_S * float(body.rate)
    for _ in range(iters):
        if not ((hard < keep) & ~dropped).any():
            break
        need = np.flatnonzero((hard < target) & ~dropped)
        kn = cand[need]
        wl = _inward_weights(out, before)
        one = np.zeros(T)
        one[kn] = 1.0
        probe = _swivel_out(out, {"left": one * wl, "right": one * (1.0 - wl)})
        h1 = np.asarray(leg_gaps(probe, kn)["hard_m"])
        gain = h1 - hard[need]
        room = KNEE_GUARD_MAX_DEG - (total["left"] + total["right"])[kn]
        good = (gain >= KNEE_GUARD_RATE_MIN) & (room > 1e-9)
        dropped[need[~good]] = True
        if not good.any():
            break
        info["rounds"] += 1
        kn, need = kn[good], need[good]
        c = np.zeros(T)
        c[kn] = np.minimum((target - hard[need]) / gain[good], room[good])
        E = gauss_smooth(_moving_max(c, int(math.ceil(2.5 * sig))), sig)
        E[np.abs(E) < 1e-12] = 0.0
        E = np.minimum(E, np.maximum(KNEE_GUARD_MAX_DEG - (total["left"] + total["right"]), 0.0))
        out, d, share, n_lim = _guard_within_limits(out, E, wl, 2 * int(math.ceil(2.5 * sig)) + 1)
        info["limit_scaled_knots"] += n_lim
        dropped[need[share[kn] <= 0.0]] = True     # the table leaves that knot no outward swivel
        for s in total:
            total[s] += d[s]
        g = leg_gaps(out, cand)
        hard = np.asarray(g["hard_m"])
    rows = [r for s in ("left", "right") for r in LEG[s][:3]]
    info.update(hard_min_m=float(hard.min()), soft_min_m=float(np.asarray(g["soft_m"]).min()),
                guarded_knots=int(((total["left"] + total["right"]) > 0).sum()),
                swivel_out_deg_max={s: float(total[s].max()) for s in total},
                swivel_out_deg_max_total=float((total["left"] + total["right"]).max()),
                dropped_knots=int(dropped.sum()), unresolved_knots=[int(k) for k in cand[hard < clearance]],
                limits_ok=bool(_limits_ok(out, before, rows)), ok=bool(hard.min() >= clearance))
    return out, info


def _guard_within_limits(body, E, wl, n):
    """(Body, {side: (T,) deg}, share (T,), knots): body with the guard's outward swivel E (T,) deg, split by wl,
    applied where it adds no joint-limit violation to body (bounds [min(lo, body), max(hi, body)] on the swivelled
    rows, body.LIMIT_TOL_DEG): at a knot where it would add one, the largest of KNEE_GUARD_SHARES of it that does not,
    the share smoothed below (smooth_below, width n odd) so that it never rises above a knot's own, lowered again at
    any knot still outside; a knot outside after the last share keeps its rows (share 0: the swivel acts knot by
    knot). Where nothing is outside, E as it is (share 1, the body bit for bit as without this step). knots: the
    knots where the whole of E would add a violation."""
    from .body import LIMIT_TOL_DEG, components_deg
    rows = [r for s in ("left", "right") for r in LEG[s][:3]]
    lim = joint_limits()
    c0 = components_deg(body.rot6d)[:, rows]
    lo, hi = np.minimum(lim.lo[rows][None], c0), np.maximum(lim.hi[rows][None], c0)
    shares = np.asarray(KNEE_GUARD_SHARES, np.float64)
    level = np.zeros(body.T, np.int64)               # the index of each knot's admissible share
    share = np.ones(body.T)
    n_bad = None
    for it in range(len(shares) + 1):
        Es = E * share
        d = {"left": Es * wl, "right": Es * (1.0 - wl)}
        out = _swivel_out(body, d)
        a = components_deg(out.rot6d)[:, rows]
        bad = ((a < lo - LIMIT_TOL_DEG) | (a > hi + LIMIT_TOL_DEG)).any((1, 2)) & (Es > 0.0)
        n_bad = int(bad.sum()) if n_bad is None else n_bad
        if not bad.any():
            return out, d, share, n_bad
        if it == len(shares):
            break
        level[bad] = np.minimum(level[bad] + 1, len(shares) - 1)
        share = smooth_below(shares[level], n)
    share = np.where(bad, 0.0, share)               # the other knots as in the last (clean) pass
    Es = E * share
    d = {"left": Es * wl, "right": Es * (1.0 - wl)}
    return _swivel_out(body, d), d, share, n_bad


def closing_ceiling(body, knot=0, clearance=KNEE_CLEARANCE_M, splits=CLOSING_SPLITS, hi=40.0, iters=14):
    """deg: the largest closing swivel S (the sum of the two knees' inward swivels about their hip-to-ankle lines,
    seated_legs) at which every hard leg-to-leg gap of body's posture at knot stays at least clearance plus
    KNEE_GUARD_MARGIN_M, however S is split between the knees (splits), by bisection on a probe of that knot."""
    n = len(splits)
    probe = Body(rot6d=body.rot6d[knot:knot + 1].repeat(n, 1, 1), transl=body.transl[knot:knot + 1].repeat(n, 1),
                 contact=body.contact[knot:knot + 1].repeat(n, 1), betas=body.betas, gender=body.gender, rate=body.rate)
    target = clearance + KNEE_GUARD_MARGIN_M

    def ok(S):
        il = np.array([a * S + o for a, o in splits])
        ir = S - il
        b = probe
        for side, m in (("left", il), ("right", ir)):
            R, _ = seated_legs(b, side, swivel_deg=-SWIVEL_OUT[side] * m)
            b = b.with_rotations(R, list(LEG[side][:3]))
        return float(np.min(leg_gaps(b, np.arange(n))["hard_m"])) >= target
    if not ok(0.0):
        return 0.0
    lo = 0.0
    if ok(hi):
        return hi
    for _ in range(iters):
        m = 0.5 * (lo + hi)
        if ok(m):
            lo = m
        else:
            hi = m
    return lo


def clamp_closing(in_left, in_right, ceiling, blend=CLOSING_BLEND_DEG, soft=CLOSING_SPLIT_DEG):
    """(left, right, cut) (T,) deg: the two knees' inward swivels with their sum S kept under ceiling: S where it lies
    blend or more below it, ceiling - blend + blend tanh((S - ceiling + blend) / blend) above (continuous to the second
    derivative, never at the ceiling); the cut S - S' taken from the knees in proportion to softplus of their own
    inward swivel over soft (the knee that came inward gives it back, the other stays nearly where it is)."""
    il, ir = np.asarray(in_left, np.float64), np.asarray(in_right, np.float64)
    S = il + ir
    a = ceiling - blend
    S2 = np.where(S > a, a + blend * np.tanh((S - a) / blend), S)
    cut = S - S2
    pl, pr = _softplus(il / soft), _softplus(ir / soft)
    wl = pl / (pl + pr)
    return il - cut * wl, ir - cut * (1.0 - wl), cut


def td_inward_targets(events, gain, t, limit, iters=60):
    """({side: (n,) weight per fragment}, {side: (T,) inward swivel deg}): the knee swivel fragments (events {side:
    fragment_events}, flowing to targets, times gain {side: deg per unit}) with the inward targets weighted only
    where the closing (the sum of the two inward swivels) would pass limit (deg): at the peak of each run above it,
    the fragments of each knee coming inward there (the last two started) take the share that brings the peak to the
    limit, in proportion to how far each knee is in; repeated (iters at most) until no run is left. A weighted target
    keeps its minimum-jerk transitions, so the knee comes inward less far and more slowly instead of stopping against
    a limit; a knee going inward while the other is out keeps its fragment as asked, and so does every outward
    target. A knee none of whose fragments is weighted renders bit for bit as asked."""
    w = {s: np.ones(len(ev)) for s, ev in events.items()}
    start = {s: np.array([float(e["t0"]) for e in ev]) for s, ev in events.items()}

    def render():
        out = {}
        for s, ev in events.items():
            evw = ev if (w[s] == 1.0).all() else [dict(e, amps={c: a * w[s][i] for c, a in e["amps"].items()})
                                                  for i, e in enumerate(ev)]
            out[s] = -SWIVEL_OUT[s] * gain[s] * render_fragments(evw, ["sw"], t)[:, 0]
        return out
    x = render()
    for _ in range(iters):
        S = x["left"] + x["right"]
        runs = _runs(S > limit)
        if not runs:
            break
        for a, b in runs:
            k = a + int(np.argmax(S[a:b]))
            pos = {s: max(float(x[s][k]), 0.0) for s in x}
            tot = sum(pos.values())
            if tot <= 0.0:
                continue
            f = max(0.0, 1.0 - (float(S[k]) - limit) / tot)
            for s, ev in events.items():
                if pos[s] <= 0.0:
                    continue
                for i in np.flatnonzero(start[s] <= t[k])[-2:]:
                    if -SWIVEL_OUT[s] * ev[i]["amps"].get("sw", 0.0) > 0.0:
                        w[s][i] *= f
        x = render()
    return w, x


def pelvis_fragments(t, intervals, rng, timing, channels=("dx", "dz", "roll", "yaw", "tilt", "dip")):
    """Raw standing-pelvis signals: fragments on the pelvis channels of PELVIS_REF (the dip one-signed)."""
    ev = fragment_events(intervals, list(channels), rng, timing)
    X = render_fragments(ev, list(channels), t)
    out = {c: X[:, i] for i, c in enumerate(channels)}
    if "dip" in out:
        out["dip"] = np.abs(out["dip"])
    return out, len(ev)


def standing_region(body, sig, level, ref=PELVIS_REF, lift=None):
    """(Body, info): the standing body's pelvis moved by raw signals {channel: (T,)} in units of PELVIS_REF, both
    feet planted (standing_legs), at the gain whose leg amplitude (the largest share over both legs' LEG_CH) is
    level, or at the largest gain inside the table. lift (T, 2) m, if given, is added unscaled (marching)."""
    rows = [1, 2, 4, 5, 7, 8]
    z = np.zeros(body.T)

    def make(g):
        b, _ = standing_legs(body, dx=g * ref["dx"] * sig.get("dx", z), dz=g * ref["dz"] * sig.get("dz", z),
                             roll_deg=g * ref["roll"] * sig.get("roll", z), yaw_deg=g * ref["yaw"] * sig.get("yaw", z),
                             tilt_deg=g * ref["tilt"] * sig.get("tilt", z), dip_m=g * ref["dip"] * sig.get("dip", z),
                             lift=lift)
        return b

    def amp(b):
        return max(extent_share(channel_values(b, body, c), channel_rom(c)) for s in ("left", "right")
                   for c in LEG_CH[s])
    g, out, a, ok = _gain_with_limits(make, amp, lambda b: _limits_ok(b, body, rows), level)
    J = out.joints().numpy()
    Jb = body.joints().numpy()
    return out, {"gain": g, "level": float(level), "realized": float(a), "reachable": ok,
                 "pelvis_cm_pp": [float(100 * (J[:, 0, i].max() - J[:, 0, i].min())) for i in range(3)],
                 "ankle_max_move_m": float(np.abs(J[:, [7, 8]] - Jb[:, [7, 8]]).max())}


# ---------------------------------------------------------------- one clip
def build(base, plan=(), seed=0, legs=None, fingers=None, operator="involuntary", grade=None, scale="",
          params=None, sources=(), feet=(0, 1), regions_measured=None, posture=None, guard=None, balance=None,
          govern=True):
    """(Body, truth): base, region plan (compose_clip), legs: [('seated', side, intervals, timing, level) |
    ('standing', intervals, timing, level, channels) | callable (body, rng) -> (Body, info)], fingers, and the truth
    (finish); posture ('sit', 'stand' or None) sets the arms' carriage, the scene of the contact guard and the
    standing balance (CARRIAGE); govern: the composed body through govern_rates (not a plan with a ballistic region).
    Deterministic under the seed (an int, or a numpy Generator already used by the caller)."""
    rng = as_rng(seed)
    t = base.times().numpy()
    arms, scene, bal = CARRIAGE.get(posture, (None, None, False))
    balance = bal if balance is None else balance
    legs_fn = None
    guarded = {}
    if legs:
        def legs_fn(b):
            info, before = [], b
            for item in legs:
                if callable(item):
                    b, inf = item(b, rng)
                    info.extend(inf)
                    continue
                if item[0] == "seated":
                    _, side, iv, timing, level = item
                    sig, n = seated_leg_fragments(t, iv, rng, TIMING[timing] if isinstance(timing, str) else timing,
                                                  swivel_scale=swivel_scale(side))
                    b, inf = seated_leg_region(b, side, sig, level)
                    inf.update(side=side)
                else:
                    _, iv, timing, level, chans = item
                    sig, n = pelvis_fragments(t, iv, rng, TIMING[timing] if isinstance(timing, str) else timing,
                                              chans)
                    b, inf = standing_region(b, sig, level)
                inf.update(fragments=n, intervals_s=[[float(x), float(y)] for x, y in iv],
                           nominal_share=float(sum(y - x for x, y in iv) / base.duration))
                info.append(inf)
            if posture == "sit":                         # the seated knees kept apart (knee_guard)
                b, guarded["knee_guard"] = knee_guard(b, before)
                if guarded["knee_guard"]["rounds"]:      # a seated leg region's amplitude, read again after it
                    for inf in info:
                        if {"side", "realized", "level"} <= set(inf):
                            a = max(extent_share(channel_values(b, before, c), channel_rom(c))
                                    for c in LEG_CH[inf["side"]])
                            inf.update(realized_before_guard=inf["realized"], realized=float(a),
                                       reachable=bool(inf["reachable"] and abs(a - inf["level"])
                                                      <= 0.01 * inf["level"] + 1e-9))
            return b, info
    out, rec = compose_clip(base, list(plan), rng, t, legs_fn, fingers, arms=arms, scene=scene,
                            balance=balance and any(CHANNELS[c]["region"] == "trunk" for rp in plan
                                                    for c in rp["channels"]), guard=guard)
    if guarded:
        rec["knee_guard"] = guarded["knee_guard"]
    if govern and not any(rp.get("ballistic") for rp in plan):     # a fling is ballistic: not governed
        out, rec["rate_governor"] = govern_rates(out)
    hands = tuple(fp["side"] for fp in fingers or ())
    p = dict(params or {})
    p["regions_measured"] = list(regions_measured or REGIONS)
    truth = finish(base, out, rec, operator, grade, scale, p, sources, seed if not isinstance(seed, np.random.Generator)
                   else None, feet, hands)
    return out, truth


def knee_clearance(body, base, clearance=KNEE_CLEARANCE_M):
    """The validity entry of a seated clip's knees (KNEE_CLEARANCE_M): the smallest hard and soft leg-to-leg gaps of
    the skinned mesh (legcontact) at the knots where the legs came together (knee_candidates against base), 'ok' when
    the hard one is at least clearance (none measured: ok, the legs no closer than in base less the prefilter)."""
    cand = knee_candidates(body, base)
    rec = {"clearance_m": float(clearance), "measured_knots": int(len(cand)), "prefilter_m": KNEE_GUARD_PREFILTER_M,
           "hard_min_m": None, "soft_min_m": None, "ok": True}
    if len(cand):
        g = leg_gaps(body, cand)
        rec.update(hard_min_m=float(np.min(g["hard_m"])), soft_min_m=float(np.min(g["soft_m"])),
                   ok=bool(np.min(g["hard_m"]) >= clearance - 1e-9))
    return rec


# ---------------------------------------------------------------- the symptoms
def _iv(base, share, rng, bout_s=3.0, n_bouts=None):
    return bouts(base.duration, share, rng, n_bouts=n_bouts, bout_s=bout_s)


def chorea(posture="sit", grade=1, regions=("trunk", "arms", "legs"), seconds=CHOREA_S, seed=SEED, fingers=True):
    """Huntington-type chorea (module docstring): every region of regions ('trunk' with the neck, 'arms' with the
    fingers, 'legs') at the UHDRS grade's declared amplitude and duration (UHDRS, AMP, DUR), fragments of TIMING
    'chorea', each region in its own random bouts."""
    base = base_body(posture, seconds)
    rng = as_rng(seed)
    a_term, d_term = UHDRS[grade]
    level, share = AMP[a_term], DUR[d_term]
    plan, legs, fing = [], [], []
    if "trunk" in regions:
        plan += axial_entries(posture, level, rng, base, share)
    if "arms" in regions:
        for s in ("left", "right"):
            iv = _iv(base, share, rng)
            plan.append(_entry(f"arm_{s}", arm_ch(s), level, iv, timing=arm_timing("chorea", posture),
                               gains=arm_gains(s), signs=arm_signs(s, posture)))
            if fingers:
                fing.append({"side": s, "kind": "events", "level": level, "intervals": iv,
                             "supported": posture == "sit"})
    if "legs" in regions:
        if posture == "sit":
            tm = dict(TIMING["chorea"], kick_p=CHOREA_KICK_P[grade]) if grade in CHOREA_KICK_P else "chorea"
            for s in ("left", "right"):
                legs.append(("seated", s, _iv(base, share, rng), tm, level))
        else:
            legs.append(("standing", _iv(base, share, rng), "chorea", level, ("dx", "dz", "roll", "yaw", "tilt",
                                                                               "dip")))
    params = {"symptom": "chorea", "posture": posture, "uhdrs_grade": grade, "amplitude_term": a_term,
              "duration_term": d_term, "level_share_of_rom": level, "share_of_time": share,
              "regions": list(regions), "timing": TIMING["chorea"]}
    return build(base, plan, rng, legs, fing, "chorea", grade, "UHDRS maximal chorea", params,
                 ("uhdrs", "dis", "sanger2010", "hallett1981", "dinesh2020", "table", "hand_limits", "module"),
                 posture=posture)


PIANO_ARM = {3: AMP["moderate"], 4: AMP["marked"]}  # [module] proximal arm chorea added to the fingers at grades 3, 4,
#                                                     at the grade's own amplitude term (UHDRS: 3 moderate, 4 marked;
#                                                     the clinical review of 2026-11-03: at mild and moderate the
#                                                     grade-4 clip said marked and showed moderate)


def piano_fingers(grade=1, seconds=CHOREA_S, seed=SEED):
    """Piano-playing finger chorea of both hands, the arms outstretched (seated 3.15 posture): finger fragments at
    the UHDRS arm grade's declared amplitude (a share of each finger's flexion range, at most FINGER_LEVEL_CAP) and
    duration, the wrists with small fragments at a quarter of that level; at grades 3 and 4 the arms carry
    proximal chorea too (PIANO_ARM: moderate at 3, marked at 4), since fingers alone read as UHDRS arm grade 1 to 2
    (clinical review of 2026-11-02)."""
    base = base_body("arms_forward", seconds)
    rng = as_rng(seed)
    a_term, d_term = UHDRS[grade]
    level, share = AMP[a_term], DUR[d_term]
    plan, fing = [], []
    for s in ("left", "right"):
        iv = _iv(base, share, rng, bout_s=2.0)
        fing.append({"side": s, "kind": "events", "level": level, "intervals": iv, "abduction": True,
                     "abd_deg": 4.0 + 8.0 * min(level, FINGER_LEVEL_CAP)})
        plan.append(_entry(f"arm_{s}", arm_ch(s, ("wrist_flex", "wrist_dev", "forearm")), 0.25 * level, iv))
        if grade in PIANO_ARM:
            plan.append(_entry(f"arm_{s}", arm_ch(s, ("collar", "sh_abd", "sh_flex", "sh_rot", "elbow")),
                               PIANO_ARM[grade], _iv(base, share, rng), gains=arm_gains(s)))
    params = {"symptom": "chorea, piano-playing fingers", "uhdrs_grade": grade, "amplitude_term": a_term,
              "duration_term": d_term, "level_share_of_finger_range": min(level, FINGER_LEVEL_CAP),
              "share_of_time": share, "wrist_level_share_of_rom": 0.25 * level,
              "proximal_arm_level_share_of_rom": PIANO_ARM.get(grade, 0.0)}
    return build(base, plan, rng, None, fing, "chorea", grade, "UHDRS maximal chorea (arms)", params,
                 ("uhdrs", "dis", "hallett1981", "hand_limits", "table", "module"),
                 regions_measured=("arm_left", "arm_right"), posture="arms_forward", guard=grade in PIANO_ARM)


IMPERSISTENCE_LAPSES = {1: 2, 2: 5, 3: 10}          # [module] lapses of the held posture per 20 s
IMPERSISTENCE = {"sink_deg": (10.0, 20.0), "drop_deg": (15.0, 25.0), "rise": (0.4, 0.9), "hold": (0.0, 0.3),
                 "ret": (0.2, 0.35), "lag_s": (0.05, 0.15), "finger_level": 0.15}
# [module] a lapse: the outstretched arm sinks by 10 to 20 deg at the shoulder and the hand drops by 15 to 25 deg at
# the wrist over 0.4 to 0.9 s (the wrist 50 to 150 ms behind), held up to 0.3 s, then a quicker correction back over
# 0.2 to 0.35 s; the fingers relax (flex slightly) during the lapse


def impersistence(grade=1, seconds=CHOREA_S, seed=SEED):
    """Motor impersistence of the outstretched arms: lapses of the held posture (IMPERSISTENCE: the arm sinks, the
    hand drops, the fingers relax, then a quick correction), IMPERSISTENCE_LAPSES per 20 s spread over both arms (a
    declared ladder: the UHDRS rates impersistence on the tongue only, not shown)."""
    base = base_body("arms_forward", seconds)
    rng = as_rng(seed)
    t = base.times().numpy()
    P = IMPERSISTENCE
    n = int(round(IMPERSISTENCE_LAPSES[grade] * seconds / 20.0))
    slots = np.sort(rng.choice(np.arange(1, int(seconds) - 1), size=n, replace=False)).astype(float)
    sides = rng.choice(["left", "right"], size=n)
    plan, fing, lapses = [], [], []
    for s in ("left", "right"):
        sig = {f"sh_flex_{s}": np.zeros(base.T), f"wrist_flex_{s}": np.zeros(base.T)}
        flex = np.zeros(base.T)
        iv = []
        for a, sd in zip(slots, sides):
            if sd != s:
                continue
            a = float(a) + float(rng.uniform(0.0, 0.4))
            r, h, f = _loguniform(rng, *P["rise"]), float(rng.uniform(*P["hold"])), float(rng.uniform(*P["ret"]))
            lag = float(rng.uniform(*P["lag_s"]))
            sink, drop = float(rng.uniform(*P["sink_deg"])), float(rng.uniform(*P["drop_deg"]))
            sig[f"sh_flex_{s}"] -= sink * _pulse(t, a, r, h, f)
            sig[f"wrist_flex_{s}"] += drop * _pulse(t, a + lag, r, h, f)
            flex += _pulse(t, a + lag, r, h, f)
            iv.append((a, a + r + h + f + lag))
            lapses.append({"t0_s": a, "side": s, "sink_deg": sink, "drop_deg": drop, "drift_s": r, "correction_s": f})
        if not iv:
            continue
        plan.append(_entry(f"arm_{s}", tuple(sig), None, iv, kind="given", raw=sig, deg={c: 1.0 for c in sig}))
        fing.append({"side": s, "kind": "relax", "level": P["finger_level"], "intervals": iv, "raw": flex})
    params = {"symptom": "chorea, motor impersistence of the outstretched arms", "grade": grade, "lapses": lapses,
              "lapse_model": IMPERSISTENCE}
    return build(base, plan, rng, None, fing, "motor_impersistence", grade, "declared ladder (lapses per 20 s)",
                 params, ("uhdrs", "module", "table", "hand_limits"), regions_measured=("arm_left", "arm_right"))


BALLISM_FLING = {"rise": (0.15, 0.3), "hold": (0.0, 0.08), "ret": (0.4, 0.6), "next": (0.75, 1.1),
                 "abd": (0.7, 0.95), "flex": (0.1, 0.7), "rot": (0.3, 0.8), "collar": (0.2, 0.6),
                 "elbow": (0.4, 0.8), "lag_s": (0.06, 0.1), "wrist": 0.25, "wrist_lag_s": (0.06, 0.1)}
# [module] one fling of hemiballism: out by a minimum-jerk rise of 0.15 to 0.3 s (500 to 900 deg/s at the shoulder
# at its peak), back over 0.4 to 0.6 s, the next fling starting at 0.75 to 1.1 of the fling's length (most often
# before the return ends): about 1 to 1.5 flings per second (yahata1997), bursts of 200 to 1500 ms (tai2016); shares
# of the room the table leaves outward: shoulder abduction 0.7 to 0.95 (median about 0.8), forward flexion 0.1 to
# 0.7, axial rotation 0.3 to 0.8 either way, elevation 0.2 to 0.6; the elbow extends by 0.4 to 0.8 of its room 60 to
# 100 ms after the shoulder (the whip) and the wrist follows passively (0.25 of its room either way, 60 to 100 ms
# later still); on the way back both are home the elbow's lag before the shoulder (the review of wave SH1d: returning
# behind the shoulder, the straightened arm and the turned wrist brought the thumb back into the thigh, 5.6 cm on the
# guard's reading); the arm is flung outward only, never into the trunk
BALLISM_KICK = {"rise": (0.2, 0.3), "hold": (0.0, 0.1), "ret": (0.3, 0.45), "gap_s": (0.3, 2.5), "rho": (1.8, 2.5),
                "lat": (0.3, 0.7), "sw": (0.5, 1.0)}
# [module] a kick of the same side's leg: the foot lifted (rise 0.2 to 0.3 s, back over 0.3 to 0.45 s) and thrown
# forward (1.8 to 2.5 times its lift, 15 to 25 cm) and out (0.3 to 0.7 times), the knee swivelled outward (hip
# abduction and external rotation), at irregular gaps of 0.3 to 2.5 s
BALLISM_KICK_SPEED = 3.3    # [module] a kick's lift share per second of its rise at most this (amp over rise; the
#                             rise drawn from BALLISM_KICK lengthened where it is quicker): the hand at rest on the
#                             kicking thigh rides it, and the grade 1 kick of amp 0.992 over 0.229 s (4.33 per second)
#                             turned the resting upper arm at 264.9 deg/s outside the flings, against the page's 250
#                             (PLAN M9); at 3.3 the page reads 217.0 deg/s there, under RATE_MARGIN of that bound (the
#                             governor's share, RATE_BOUND_DEG_S, which a ballistic plan does not go through), at 3.7
#                             246.1 and at 3.0 202.7 (scripts/openbody_phaseE_checks.py 'hemiballism_kick')
BALLISM_LEG_LEVEL = 0.45                            # [module] the leg's level (the seated hip and knee bound it)
SWIVEL_OUT = {"left": -1.0, "right": 1.0}           # sign of an outward knee swivel (seated_legs)


def fling_signals(base, side, intervals, rng, t):
    """({channel: (T,) deg}, n): the flings of hemiballism on the arm of side inside intervals (BALLISM_FLING),
    each channel in degrees of the room the base leaves on its side."""
    F = BALLISM_FLING
    ch = {k: f"{k}_{side}" for k in ("sh_abd", "sh_flex", "sh_rot", "collar", "elbow", "wrist_flex")}
    rooms = {k: channel_room(base, c) for k, c in ch.items()}
    sig = {c: np.zeros(len(t)) for c in ch.values()}
    u = lambda lo_hi: float(rng.uniform(*lo_hi))  # noqa: E731
    n = 0
    for a, b in intervals:
        t0 = float(a)
        while b - t0 > 0.4:
            r, h, f = u(F["rise"]), u(F["hold"]), u(F["ret"])
            L = r + h + f
            if t0 + L > b:
                k = (b - t0) / L
                r, h, f, L = r * k, h * k, f * k, b - t0
            p = _pulse(t, t0, r, h, f)
            sig[ch["sh_abd"]] += u(F["abd"]) * rooms["sh_abd"][0] * p
            sig[ch["sh_flex"]] += u(F["flex"]) * rooms["sh_flex"][0] * p
            sr = float(rng.choice([-1.0, 1.0]))
            sig[ch["sh_rot"]] += sr * u(F["rot"]) * rooms["sh_rot"][0 if sr > 0 else 1] * p
            sig[ch["collar"]] += u(F["collar"]) * rooms["collar"][0] * p
            lag = u(F["lag_s"])                          # the elbow and the wrist behind the shoulder on the way
            sig[ch["elbow"]] -= (u(F["elbow"]) * rooms["elbow"][1]    # out, back before it on the way in
                                 * _pulse(t, t0 + lag, r, h, f - 2 * lag))
            sw = float(rng.choice([-1.0, 1.0]))
            lag2 = lag + u(F["wrist_lag_s"])
            sig[ch["wrist_flex"]] += (sw * F["wrist"] * rooms["wrist_flex"][0 if sw > 0 else 1]
                                      * _pulse(t, t0 + lag2, r, h, f - lag - lag2))
            n += 1
            t0 += L * u(F["next"])
    return sig, n


def kick_signals(t, intervals, rng, side):
    """(raw seated-leg signals as seated_leg_fragments, n): the kicks of hemiballism (BALLISM_KICK)."""
    K = BALLISM_KICK
    z = np.zeros(len(t))
    sig = {"sw": z.copy(), "lift": z.copy(), "fwd": z.copy(), "lat": z.copy()}
    u = lambda lo_hi: float(rng.uniform(*lo_hi))  # noqa: E731
    n = 0
    for a, b in intervals:
        t0 = float(a) + u((0.0, 0.5))
        while b - t0 > 0.6:
            r, h, f = u(K["rise"]), u(K["hold"]), u(K["ret"])
            amp = u((0.7, 1.0))
            r = max(r, amp / BALLISM_KICK_SPEED)          # the resting hand carried by the thigh (BALLISM_KICK_SPEED)
            p = _pulse(t, t0, r, h, f)
            sig["lift"] += amp * p
            sig["fwd"] += u(K["rho"]) * amp * p
            sig["lat"] += SIDE_SIGN[side] * u(K["lat"]) * amp * p
            sig["sw"] += SWIVEL_OUT[side] * u(K["sw"]) * p
            n += 1
            t0 += r + h + f + u(K["gap_s"])
    return sig, n


def hemiballism(grade=1, side="left", seconds=CHOREA_S, seed=SEED):
    """Hemiballism (seated, one side): flings of the arm (fling_signals, BALLISM_FLING: proximal, outward, fast, the
    elbow whipping open behind the shoulder, about 1 to 1.5 per second, tai2016, yahata1997) and kicks of the same
    side's leg (kick_signals, BALLISM_KICK, at irregular gaps; seated_leg_region at BALLISM_LEG_LEVEL); declared
    ladder of the share of time 30, 70, 95 percent (grades 1 to 3)."""
    base = base_body("sit", seconds)
    rng = as_rng(seed)
    t = base.times().numpy()
    share = {1: DUR["intermittent"], 2: DUR["common"], 3: DUR["prolonged"]}[grade]
    iv = _iv(base, share, rng)
    sig, n_fl = fling_signals(base, side, iv, rng, t)
    plan = [_entry(f"arm_{side}", tuple(sig), None, iv, kind="given", raw=sig, deg={c: 1.0 for c in sig},
                   ballistic=True)]
    ivl = _iv(base, share, rng)

    def legs(b, rng_):
        ks, n_k = kick_signals(t, ivl, rng_, side)
        out, inf = seated_leg_region(b, side, ks, BALLISM_LEG_LEVEL)
        return out, [dict(inf, side=side, kicks=n_k, intervals_s=[[float(x), float(y)] for x, y in ivl])]
    params = {"symptom": "hemiballism", "side": side, "grade": grade, "share_of_time": share, "flings": n_fl,
              "ballistic_intervals_s": [[float(x), float(y)] for x, y in iv],
              "fling": BALLISM_FLING, "kick": BALLISM_KICK, "kick_speed_per_s": BALLISM_KICK_SPEED,
              "leg_level_share_of_rom": BALLISM_LEG_LEVEL}
    return build(base, plan, rng, [legs], None, "hemiballism", grade, "declared ladder (share of time)", params,
                 ("postuma2003", "tai2016", "yahata1997", "sanger2010", "dis", "table", "module"), posture="sit")


def athetosis(amplitude_grade=2, duration_grade=4, seconds=CHOREA_S, seed=SEED, chorea_mix=False):
    """Athetosis (adult dyskinetic cerebral palsy or post-stroke), seated: slow writhing of the forearms, wrists and
    fingers of both hands and the toes of both feet, band-limited noise in the 0.3 to 0.6 Hz band (neilson1974), DIS
    choreoathetosis amplitude and duration at the band centres (DIS_LEVELS), the hand lifting off its thigh as it
    writhes (lift_noise). chorea_mix adds chorea fragments to the arms and trunk at half the level
    (choreoathetosis)."""
    base = base_body("sit", seconds)
    rng = as_rng(seed)
    level, share = DIS_LEVELS[amplitude_grade], DIS_LEVELS[duration_grade]
    plan, fing = [], []
    band = BANDS_HZ["athetosis"]
    for s in ("left", "right"):
        iv = _iv(base, share, rng, bout_s=6.0)
        plan.append(_entry(f"arm_{s}", arm_ch(s, DISTAL_KEYS), level, iv, kind="noise", band=band,
                           gains=arm_gains(s, DISTAL_GAINS), one_sided=arm_signs(s, "sit", DISTAL_KEYS)))
        fing.append({"side": s, "kind": "noise", "band": band, "level": level, "intervals": iv, "abduction": True,
                     "abd_band": band, "abd_deg": 12.0})
        plan.append(_entry(f"leg_{s}", (f"toes_{s}",), min(level, 0.45), iv, kind="noise", band=band,
                           one_sided=True))
    plan.append(_entry("neck", NECK_CH, 0.5 * level, _iv(base, share, rng, bout_s=6.0), kind="noise", band=band,
                       gains=NECK_GAINS))
    if chorea_mix:
        for s in ("left", "right"):
            plan.append(_entry(f"arm_{s}", arm_ch(s, ("sh_abd", "sh_flex", "elbow")), 0.5 * level,
                               _iv(base, share, rng), gains=arm_gains(s),
                               signs=arm_signs(s, "sit", ("sh_abd", "sh_flex", "elbow"))))
        plan.append(_entry("trunk", TRUNK_CH, 0.5 * level, _iv(base, share, rng), gains=TRUNK_GAINS,
                           margin=TRUNK_NECK_MARGIN, signs={"trunk_flex": 1.0}))
    params = {"symptom": "choreoathetosis" if chorea_mix else "athetosis", "dis_amplitude_grade": amplitude_grade,
              "dis_duration_grade": duration_grade, "level_share_of_rom": level, "share_of_time": share,
              "band_hz": list(band), "toes_level": min(level, 0.45),
              "toes_note": "the toes move as one block (the program's foot joint), toward extension only from the planted "
                           "foot (the toes cannot flex into the floor)"}
    return build(base, plan, rng, None, fing, "athetosis", f"DIS {amplitude_grade}/{duration_grade}",
                 "DIS choreoathetosis (amplitude, duration)", params,
                 ("dis", "sanger2010", "neilson1974", "table", "hand_limits", "module"), posture="sit")


PSEUDO_PARTS = {1: ("fingers",), 2: ("fingers", "wrist"), 3: ("fingers", "wrist", "arm")}   # declared from sharp1994
PSEUDO_LEVEL = {"fingers": 0.30, "wrist": 0.20, "arm": 0.10}                           # [module] shares of ROM
PSEUDO_DRIFT_M = {"down": 0.045, "out": 0.03}
# [module] grade 3: under the wandering, the unseen arm sinks and drifts outward slowly over the clip by these at the
# wrist (shoulder flexion and rotation turned by their arc over the shoulder-to-wrist length of the rest skeleton: with
# the arm held forward the shoulder's flexion channel moves the wrist up and down and its rotation moves it out,
# reached by a minimum-jerk ramp from 0.5 s to the clip's end; the clinical review of 2026-11-03 asked for 3 to 6 cm
# down and 2 to 4 cm outward over 20 s: the wandering alone came back to the posture, its 5 s means within 4 mm)


def pseudoathetosis(grade=1, side="left", seconds=CHOREA_S, seed=SEED):
    """Pseudoathetosis, arms outstretched with the eyes closed (stated, not drawn): slow drift of the parts with
    proprioceptive loss (sharp1994), a declared ladder by extent (PSEUDO_PARTS: fingers, then the wrist, then the
    arm), band-limited noise in 0.1 to 0.4 Hz (module: no published speed), on one side, continuous; the wrist drifts
    down only and the arm down and outward only (one-sided drifts: the unseen arm sinks and wanders off rather than
    oscillating about the held posture), and at grade 3 the arm sinks and drifts outward over the clip
    (PSEUDO_DRIFT_M)."""
    base = base_body("arms_forward", seconds)
    rng = as_rng(seed)
    band = BANDS_HZ["pseudoathetosis"]
    iv = [(0.5, base.duration - 0.25)]
    parts = PSEUDO_PARTS[grade]
    plan, fing = [], [{"side": side, "kind": "noise", "band": band, "level": PSEUDO_LEVEL["fingers"],
                       "intervals": iv, "abduction": True, "abd_band": band, "abd_deg": 10.0}]
    if "wrist" in parts:
        plan.append(_entry(f"arm_{side}", arm_ch(side, ("wrist_flex", "wrist_dev", "forearm")), PSEUDO_LEVEL["wrist"],
                           iv, kind="noise", band=band, one_sided={f"wrist_flex_{side}": 1.0}))
    drift = None
    if "arm" in parts:
        plan.append(_entry(f"arm_{side}", arm_ch(side, ("sh_flex", "sh_abd", "elbow")), PSEUDO_LEVEL["arm"], iv,
                           kind="noise", band=band, one_sided={f"sh_flex_{side}": -1.0, f"sh_abd_{side}": 1.0}))
        rest = base.rest().numpy()
        c, sh, el, wr = ARM[side]
        L = float(np.linalg.norm(rest[el] - rest[sh]) + np.linalg.norm(rest[wr] - rest[el]))
        t = base.times().numpy()
        ramp = minimum_jerk((t - iv[0][0]) / (iv[0][1] - iv[0][0]))
        drift = {k: math.degrees(v / L) for k, v in PSEUDO_DRIFT_M.items()}
        plan.append(_entry(f"arm_{side}", (f"sh_flex_{side}", f"sh_rot_{side}"), None, iv, kind="given",
                           raw={f"sh_flex_{side}": -ramp, f"sh_rot_{side}": ramp},
                           deg={f"sh_flex_{side}": drift["down"], f"sh_rot_{side}": drift["out"]}))
    params = {"symptom": "pseudoathetosis", "side": side, "grade": grade, "parts": list(parts),
              "levels": {k: PSEUDO_LEVEL[k] for k in parts}, "band_hz": list(band), "eyes": "closed (stated)",
              "drift_m": PSEUDO_DRIFT_M if drift else None, "drift_deg": drift}
    return build(base, plan, rng, None, fing, "pseudoathetosis", grade, "declared ladder (extent)", params,
                 ("sharp1994", "table", "hand_limits", "module"), regions_measured=("arm_left", "arm_right"))


LID_SEED = SEED + 51        # [module] the peak-dose dyskinesia's own seed (its bouts not those of chorea)


def lid_peak_dose(posture="sit", step=1, seconds=CHOREA_S, seed=LID_SEED):
    """Peak-dose choreic levodopa-induced dyskinesia: trunk, neck, both arms and both legs (seated) or the pelvis
    sway with the feet planted (standing), fragments of TIMING 'lid' (the 1 to 3 Hz band of manson2000), at the
    UDysRS part-3 step's declared amplitude and share (STEPS), from its own seed (LID_SEED)."""
    base = base_body(posture, seconds)
    rng = as_rng(seed)
    level, share = STEPS[step]
    plan = axial_entries(posture, level, rng, base, share, timing="lid")
    fing = []
    for s in ("left", "right"):
        iv = _iv(base, share, rng)
        plan.append(_entry(f"arm_{s}", arm_ch(s), level, iv, timing=arm_timing("lid", posture), gains=arm_gains(s),
                           signs=arm_signs(s, posture)))
        fing.append({"side": s, "kind": "events", "level": 0.5 * level, "intervals": iv, "supported": posture == "sit"})
    if posture == "sit":
        legs = [("seated", s, _iv(base, share, rng), "lid", level) for s in ("left", "right")]
    else:
        legs = [("standing", _iv(base, share, rng), "lid", level, ("dx", "dz", "roll", "yaw", "tilt", "dip"))]
    params = {"symptom": "levodopa-induced dyskinesia, peak-dose choreic", "posture": posture, "udysrs_step": step,
              "udysrs_term": UDYSRS_TERMS[step], "level_share_of_rom": level, "share_of_time": share,
              "timing": TIMING["lid"], "tasks": "UDysRS part 3 rates during communication, drinking, dressing and "
                                                "walking; shown at rest"}
    return build(base, plan, rng, legs, fing, "dyskinesia", step, "UDysRS part 3 (declared ladder)", params,
                 ("udysrs", "manson2000", "hoff2001", "dis", "table", "hand_limits", "module"), posture=posture)


DIPHASIC_HZ = 0.8          # [module] alternating leg cycles per second (no published rate, luquin1992)
STEP_LIFT_M = {1: 0.03, 2: 0.06, 3: 0.10}           # [module] standing stepping: the ball's lift per step


def diphasic(posture="sit", step=1, seconds=CHOREA_S, seed=SEED):
    """Diphasic dyskinesia: repetitive stereotyped alternating leg movements (luquin1992): seated, kicking (the foot
    lifted and the knee extended, the legs in antiphase) at DIPHASIC_HZ with the UDysRS step's declared leg amplitude
    and share; standing, stepping in place (STEP_LIFT_M) for the step's share, the weight shifted over the stance
    foot before each step and the heel rising before the ball (step_track, step_geometry)."""
    base = base_body(posture, seconds)
    rng = as_rng(seed)
    level, share = STEPS[step]
    t = base.times().numpy()
    iv = _iv(base, share, rng, bout_s=5.0)
    env = ramp_envelope(t, iv, 0.5)
    if posture == "sit":
        def legs(b, rng_):
            info = []
            for i, s in enumerate(("left", "right")):
                p = pattern(t, DIPHASIC_HZ, rng_, None, phase=math.pi * i, wave="pulse") * env
                sig = {"sw": 0.15 * pattern(t, DIPHASIC_HZ, rng_, None, phase=math.pi * i) * env,
                       "lift": p, "fwd": 2.5 * p, "lat": np.zeros_like(p)}
                b, inf = seated_leg_region(b, s, sig, level)
                info.append(dict(inf, side=s))
            return b, info
    else:
        def legs(b, rng_):
            ph, am = phase_track(t, DIPHASIC_HZ, rng_)
            geo = step_geometry(b)
            lift, heel, dx, cop = step_track(ph, am, env, STEP_LIFT_M[step], geo, b.rate)
            b2, inf = standing_legs(b, dx=dx, lift=lift, heel_deg=heel)
            return b2, [dict(inf, step_geometry=geo, pelvis_shift_m_max=float(np.abs(dx).max()),
                             lift_m=STEP_LIFT_M[step], heel_deg=HEEL_DEG,
                             stance=stance_report(b2, b, omega0=geo["omega0"]))]
    params = {"symptom": "diphasic dyskinesia, " + ("kicking" if posture == "sit" else "stepping in place"),
              "posture": posture, "udysrs_step": step, "udysrs_term": UDYSRS_TERMS[step], "share_of_time": share,
              "rate_hz": DIPHASIC_HZ, "intervals_s": [[float(a), float(b)] for a, b in iv]}
    if posture == "sit":
        params["leg_level_share_of_rom"] = level
    else:
        params["lift_m"] = STEP_LIFT_M[step]
    return build(base, (), rng, [legs], None, "dyskinesia", step, "UDysRS part 3 (declared ladder)", params,
                 ("udysrs", "luquin1992", "table", "module") + (("hof2005",) if posture != "sit" else ()),
                 regions_measured=("leg_left", "leg_right"), posture=posture)


TD_ROCK_HZ = 0.5                                     # [module] truncal rocking rate (no published rate)
TD_ROCK_DEG = {1: 6.0, 2: 12.0, 3: 20.0, 4: 30.0}    # [module] trunk pitch, deg peak to peak, AIMS 7 steps
TD_THRUST_HZ = 0.7                                   # [module] pelvic thrusting rate (no published rate)
TD_THRUST_M = {1: 0.015, 2: 0.03, 3: 0.05, 4: 0.08}  # [module] pelvis forward travel, m peak to peak
THRUST_TILT_DEG_M = 200.0                           # [module] backward pelvic tilt per m of forward thrust
LUMBAR_COUNTER = 0.7                                # [module] lumbar counter-flexion, share of the pelvic tilt
THRUST_JITTER = 0.15                                # [module] cycle-to-cycle jitter of the thrusting
PELVIS_TILT_SHARE = 0.5                              # [module] share of a seated rocking taken by the pelvis


def _rock_seated(base, rng, rate_hz, deg_pp, iv, pelvis_share=PELVIS_TILT_SHARE):
    """(legs callable, trunk plan entry): a seated trunk rocking of deg_pp peak to peak at rate_hz inside iv, forward
    from upright and back to upright (the trunk never leans back toward the backrest), the pelvis tilting about the
    hip line (tilt_pelvis_seated) for pelvis_share of it and the lumbar and thoracic spine (trunk_flex) for the
    rest, in phase, the spine's part split over its joints by their forward room (split 'room', room_split: the
    open body's seated lumbar joint stands 2.5 deg from its flexion bound, and the ROM split of the trunk_flex
    channel flattened the rocking at 5 deg of the spine's 25, the open-body port's wave 3 follow-up, D40)."""
    t = base.times().numpy()
    x = pattern(t, rate_hz, rng, iv, wave="pulse")

    def legs(b, rng_):
        return tilt_pelvis_seated(b, pelvis_share * deg_pp * x), [{"pelvis_tilt_deg_pp": pelvis_share * deg_pp}]
    entry = _entry("trunk", ("trunk_flex",), None, iv, kind="given", raw={"trunk_flex": x},
                   deg={"trunk_flex": (1 - pelvis_share) * deg_pp}, split="room")
    return legs, entry


def td_rocking(step=2, seconds=CHOREA_S, seed=SEED):
    """Tardive dyskinesia, truncal rocking seated (AIMS item 7: rocking): TD_ROCK_DEG[step] of trunk pitch at
    TD_ROCK_HZ for the AIMS step's declared share (STEPS); orobuccolingual movements not shown."""
    base = base_body("sit", seconds)
    rng = as_rng(seed)
    share = STEPS[step][1]
    iv = _iv(base, share, rng, bout_s=6.0)
    legs, entry = _rock_seated(base, rng, TD_ROCK_HZ, TD_ROCK_DEG[step], iv)
    params = {"symptom": "tardive dyskinesia, truncal rocking", "aims_item": 7, "aims_step": step,
              "aims_term": AIMS_TERMS[step], "deg_pp": TD_ROCK_DEG[step], "rate_hz": TD_ROCK_HZ, "share_of_time": share,
              "not_shown": "orobuccolingual movements (AIMS items 1 to 4)"}
    return build(base, [entry], rng, [legs], None, "tardive_dyskinesia", step, "AIMS item 7 (declared ladder)", params,
                 ("aims", "module", "table"), regions_measured=("trunk", "leg_left", "leg_right"), posture="sit")


def td_pelvic_thrust(step=2, seconds=CHOREA_S, seed=SEED):
    """Tardive dyskinesia, pelvic thrusting standing (AIMS item 7: pelvic gyrations): the pelvis forward and back by
    TD_THRUST_M[step] at TD_THRUST_HZ (cycle-to-cycle jitter THRUST_JITTER) with a backward tilt as it thrusts
    (THRUST_TILT_DEG_M, 2 deg per cm), the lumbar spine flexing back by LUMBAR_COUNTER of that tilt so that the upper
    trunk stays nearer upright (not one rigid plank), both feet planted, for the AIMS step's declared share."""
    base = base_body("stand", seconds)
    rng = as_rng(seed)
    share = STEPS[step][1]
    iv = _iv(base, share, rng, bout_s=6.0)
    t = base.times().numpy()
    x = pattern(t, TD_THRUST_HZ, rng, iv, jitter=THRUST_JITTER)
    dz = 0.5 * TD_THRUST_M[step] * x

    def legs(b, rng_):
        out, inf = standing_legs(b, dz=dz, tilt_deg=-THRUST_TILT_DEG_M * dz)
        return out, [dict(inf, dz_m_pp=TD_THRUST_M[step])]
    plan = [_entry("trunk", ("trunk_flex",), None, iv, kind="given", raw={"trunk_flex": THRUST_TILT_DEG_M * dz},
                   deg={"trunk_flex": LUMBAR_COUNTER})]
    params = {"symptom": "tardive dyskinesia, pelvic thrusting", "aims_item": 7, "aims_step": step,
              "aims_term": AIMS_TERMS[step], "pelvis_m_pp": TD_THRUST_M[step], "rate_hz": TD_THRUST_HZ,
              "share_of_time": share, "not_shown": "orobuccolingual movements (AIMS items 1 to 4)"}
    return build(base, plan, rng, [legs], None, "tardive_dyskinesia", step, "AIMS item 7 (declared ladder)", params,
                 ("aims", "module", "table"), regions_measured=("trunk", "leg_left", "leg_right"), posture="stand",
                 balance=False)


def td_limbs(step=2, seconds=CHOREA_S, seed=SEED):
    """Tardive dyskinesia of the limbs seated (AIMS items 5 and 6): choreoathetoid fragments of the forearms,
    wrists and fingers (TIMING 'tardive', more power at 1 to 2 Hz, wirshing1991), lateral knee movement (knee
    swivel with the foot planted) and toe raising of both feet, at the AIMS step's declared amplitude and share; the
    knees' closing kept under the posture's ceiling (closing_ceiling) by weighting the inward targets of the swivel
    fragments that would pass it (td_inward_targets), clamp_closing (over the ceiling) and the knee guard as safety
    nets (the knees KNEE_CLEARANCE_M apart), the outward swivels as asked (params['knee_closing'], each leg's asked
    and kept swivels); a step whose asked closing stays under the ceiling keeps its swivels as asked."""
    base = base_body("sit", seconds)
    rng = as_rng(seed)
    level, share = STEPS[step]
    plan, fing = [], []
    for s in ("left", "right"):
        iv = _iv(base, share, rng)
        plan.append(_entry(f"arm_{s}", arm_ch(s, DISTAL_KEYS), level, iv, timing="tardive",
                           gains=arm_gains(s, DISTAL_GAINS), signs=arm_signs(s, "sit", DISTAL_KEYS)))
        fing.append({"side": s, "kind": "events", "level": level, "intervals": iv, "supported": True})
        ivl = _iv(base, share, rng)
        plan.append(_entry(f"leg_{s}", (f"toes_{s}",), min(level, 0.45), ivl, timing=dict(TIMING["tardive"],
                                                                                          positive=True)))
    t = base.times().numpy()
    closing = {}

    def legs(b, rng_):
        info, asked, events, gain = [], {}, {}, {}
        for s in ("left", "right"):
            iv = _iv(base, share, rng_)
            ev = fragment_events(iv, ["sw"], rng_, TIMING["tardive"])
            sw = render_fragments(ev, ["sw"], t)[:, 0]
            k = used = 1.0
            for _ in range(40):                          # scaled down until the leg stays inside the table
                used = k
                R, _ = seated_legs(b, s, swivel_deg=used * TD_KNEE_DEG * level / 0.4 * sw)
                if _limits_ok(b.with_rotations(R, list(LEG[s][:3])), b, list(LEG[s][:3])):
                    break
                k *= 0.9
            asked[s], events[s], gain[s] = used * TD_KNEE_DEG * level / 0.4 * sw, ev, used * TD_KNEE_DEG * level / 0.4
            info.append({"side": s, "table_scale": k, "intervals_s": [[float(x), float(y)] for x, y in iv]})
        # the knees kept apart (KNEE_CLEARANCE_M): the inward targets of the swivel fragments weighted where the
        # closing (both knees' inward swivels) would pass the ceiling of the seated posture (closing_ceiling;
        # td_inward_targets), the outward targets as asked; clamp_closing then keeps any residue under the ceiling
        # plus CLOSING_BLEND_DEG (a safety net, untouched at or under the ceiling), and the knee guard (build) lifts
        # any knot the posture's movement brings under the clearance plus the margin
        ceiling = closing_ceiling(b, 0)
        w_in, inward_w = td_inward_targets(events, gain, t, ceiling)
        inward = {s: -SWIVEL_OUT[s] * asked[s] for s in asked}
        il, ir, cut = clamp_closing(inward_w["left"], inward_w["right"], ceiling + CLOSING_BLEND_DEG)
        kept = {"left": -SWIVEL_OUT["left"] * il, "right": -SWIVEL_OUT["right"] * ir}
        for x in info:
            s = x["side"]
            R, _ = seated_legs(b, s, swivel_deg=kept[s])
            b = b.with_rotations(R, list(LEG[s][:3]))
            own = {"left": il, "right": ir}[s]
            x.update(knee_swivel_deg_max=float(np.abs(kept[s]).max()),
                     knee_swivel_deg_max_asked=float(np.abs(asked[s]).max()),
                     inward_deg_max=float(own.max()), inward_deg_max_asked=float(inward[s].max()),
                     outward_deg_max=float((-own).max()), outward_deg_max_asked=float((-inward[s]).max()))
        closing.update(ceiling_deg=float(ceiling), blend_deg=CLOSING_BLEND_DEG,
                       clamp_ceiling_deg=float(ceiling + CLOSING_BLEND_DEG), clearance_m=KNEE_CLEARANCE_M,
                       margin_m=KNEE_GUARD_MARGIN_M,
                       fragments_weighted={x: int((w_in[x] < 1.0).sum()) for x in w_in},
                       fragments={x: int(len(w_in[x])) for x in w_in},
                       weight_min=float(min(w_in[x].min() if len(w_in[x]) else 1.0 for x in w_in)),
                       closing_deg_max_asked=float((inward["left"] + inward["right"]).max()),
                       closing_deg_max=float((il + ir).max()), cut_deg_max=float(cut.max()),
                       cut_share=float((cut > 1e-9).mean()),
                       rule="the inward targets of the knee swivel fragments weighted where the sum of the two "
                            "knees' inward swivels would pass the largest closing at which the hard leg-to-leg gap of "
                            "the seated posture stays the margin above the clearance (closing_ceiling; "
                            "td_inward_targets), clamp_closing keeping any residue under it plus the blend, the knee "
                            "guard lifting any knot under the clearance plus the margin; the outward swivels as asked")
        return b, info
    params = {"symptom": "tardive dyskinesia, limb choreoathetoid movements", "aims_items": [5, 6], "aims_step": step,
              "aims_term": AIMS_TERMS[step], "level_share_of_rom": level, "share_of_time": share,
              "timing": TIMING["tardive"], "knee_swivel_deg_at_moderate": TD_KNEE_DEG, "knee_closing": closing,
              "not_shown": "orobuccolingual movements (AIMS items 1 to 4), foot inversion and eversion"}
    return build(base, plan, rng, [legs], fing, "tardive_dyskinesia", step, "AIMS items 5 and 6 (declared ladder)",
                 params, ("aims", "wirshing1991", "dis", "module", "table", "hand_limits"), posture="sit")


TD_KNEE_DEG = 15.0          # [module] lateral knee movement (knee swivel) at the moderate step, scaled with the level

STEREO = {"rocking": {"rate_hz": 0.6, "deg_pp": 20.0, "bout_s": 11.78, "posture": "sit"},
          "flapping": {"rate_hz": 3.0, "deg_pp": 50.0, "bout_s": 5.26, "posture": "flap"},
          "nodding": {"rate_hz": 1.0, "deg_pp": 16.0, "bout_s": 6.0, "posture": "sit"}}
# rocking: rate inside the 0.2 to 0.9 Hz axis of newell2005 (module value), bouts goodwin_rock; flapping: rate and
# size module values (no published rate), bouts goodwin_rock; nodding: rate, size and bouts module values
STEREO_SHARE = {1: DUR["intermittent"], 2: DUR["common"], 3: DUR["prolonged"]}
FLAP_JITTER = 0.12          # [module] hand flapping: cycle-to-cycle jitter of rate and amplitude
FLAP_ELBOW_DEG = 4.0        # [module] hand flapping: the elbow's small flexion with each flap (50 ms behind)


def stereotypy_bouts(kind, grade, seconds=CHOREA_S, seed=SEED):
    """[(a, b)] s: the bouts stereotypy(kind, grade, seconds, seed) draws first (bouts at STEREO's mean bout length),
    for the clip's caption (the clinical review of 2026-11-03: the captions printed the source's mean bout length,
    11.78 s, where the clips held bouts of 5.5 to 17.9 s)."""
    return bouts(float(seconds), STEREO_SHARE[grade], as_rng(seed), bout_s=STEREO[kind]["bout_s"])


def _bouts_text(iv):
    """'one bout of 5.5 s' or 'bouts of 4.4, 1.9 and 6.8 s' (one decimal)."""
    d = [f"{b - a:.1f}" for a, b in iv]
    if len(d) == 1:
        return f"one bout of {d[0]} s"
    return f"bouts of {', '.join(d[:-1])} and {d[-1]} s"


def stereotypy(kind="rocking", grade=1, seconds=CHOREA_S, seed=SEED):
    """Adult stereotypies (sanger2010: repetitive, patterned, rhythmic): body rocking seated (trunk and pelvis),
    hand flapping standing (both wrists in phase, the hands before the chest), head nodding seated (the neck), each
    a pattern of STEREO's rate and size in bouts of its mean length, a declared ladder of the share of time
    (STEREO_SHARE)."""
    p = STEREO[kind]
    base = base_body(p["posture"], seconds)
    rng = as_rng(seed)
    share = STEREO_SHARE[grade]
    iv = _iv(base, share, rng, bout_s=p["bout_s"])
    t = base.times().numpy()
    legs, plan, regions = None, [], None
    if kind == "rocking":
        lg, entry = _rock_seated(base, rng, p["rate_hz"], p["deg_pp"], iv)
        legs, plan, regions = [lg], [entry], ("trunk", "leg_left", "leg_right")
    elif kind == "flapping":
        x = pattern(t, p["rate_hz"], rng, iv, ramp_s=0.2, jitter=FLAP_JITTER)
        for s in ("left", "right"):
            plan.append(_entry(f"arm_{s}", (f"wrist_flex_{s}", f"elbow_{s}"), None, iv, kind="given",
                               raw={f"wrist_flex_{s}": x, f"elbow_{s}": np.roll(x, int(0.05 * base.rate))},
                               deg={f"wrist_flex_{s}": 0.5 * p["deg_pp"], f"elbow_{s}": FLAP_ELBOW_DEG}))
        regions = ("arm_left", "arm_right")
    else:
        x = pattern(t, p["rate_hz"], rng, iv, ramp_s=0.3)
        plan.append(_entry("neck", ("neck_flex",), None, iv, kind="given", raw={"neck_flex": x},
                           deg={"neck_flex": 0.5 * p["deg_pp"]}))
        regions = ("neck",)
    what = {"rocking": "body rocking", "flapping": "hand flapping"}.get(kind, "head nodding")
    params = {"symptom": f"stereotypy, {what}",
              "grade": grade, "share_of_time": share, **p}
    srcs = ("sanger2010", "goodwin_rock", "newell2005", "module", "table") if kind != "nodding" else \
        ("sanger2010", "module", "table")
    return build(base, plan, rng, legs, None, "stereotypy", grade, "declared ladder (share of time)", params, srcs,
                 regions_measured=regions, posture=p["posture"] if p["posture"] == "sit" else None)


AKATHISIA_SEATED = {"tramp": {"rate_hz": 1.2, "lift_m": 0.035},          # [module] alternating foot lifts in place
                    "swing": {"rate_hz": 0.58, "lift_m": 0.075, "fwd_m": 0.14, "back_m": 0.04},
                    # janno2008 rate; size module: the foot swung 4 cm back to 14 cm forward of its lifted place on an
                    # arc about the knee (the clinical review of 2026-11-03: the heel held 75 mm while it travelled 0
                    # to 110 mm, a foot on a rail)
                    "shift": {"rate_hz": 0.3, "trunk_lat_deg": 20.0, "tilt_deg": 6.0, "swivel_deg": 12.0,
                              "swivel_hz": 0.6}}
# [module] shifting on the seat: the trunk 20 deg peak to peak side to side with a pelvic tilt, always with the knees
# swinging in and out (alternately)
AKATHISIA_STANDING = {"rock": {"rate_hz": 0.5, "dx_m": 0.05},                       # rocking from foot to foot
                      "march": {"rate_hz": 0.9, "lift_m": 0.07},                    # walking on the spot
                      "small_steps": {"rate_hz": 1.3, "lift_m": 0.025},             # small steps on the spot
                      "sway": {"rate_hz": 0.45, "dz_m": 0.07}}                       # rocking forward and back
# [module] the standing movements and their sizes; a step on the spot shifts the weight over the stance foot first
# (step_track); the BARS names rocking from foot to foot and walking on the spot standing (the clinical review of
# 2026-11-03: shuffling is its seated item)
AKATHISIA_SEGMENT_S = (3.0, 6.0)     # [module] a bout is split into movements of 3 to 6 s, each different from the last
AKATHISIA_PAUSE_S = (0.5, 2.0)       # [module] pauses between movements (grades 1 and 2; at grade 3 0.5 to 0.8 s)
AKATHISIA_JITTER = 0.2               # [module] cycle-to-cycle jitter of an akathisia movement's rate and size
AKATHISIA_FIDGET = 0.15              # [module] finger fidgets with every movement (share of the flexion range)


def phase_track(t, rate_hz, rng, jitter=STEREO_JITTER):
    """((T,) phase rad, (T,) amplitude): the phase and amplitude of pattern() (a rate and an amplitude jittered by
    cycle), for signals that must share one phase."""
    n_cyc = int(math.ceil(t[-1] * rate_hz)) + 2
    rk = rate_hz * (1.0 + jitter * rng.standard_normal(n_cyc).clip(-2, 2))
    ak = 1.0 + jitter * rng.standard_normal(n_cyc).clip(-2, 2)
    tc = np.arange(n_cyc) / rate_hz
    r, a = np.interp(t, tc, rk), np.interp(t, tc, ak)
    ph = rng.uniform(0, 2 * math.pi) + 2 * math.pi * np.concatenate(
        [[0.0], np.cumsum(0.5 * (r[1:] + r[:-1]) * np.diff(t))])
    return ph, a


def akathisia_segments(intervals, rng, menu, grade, first=None):
    """[(a, b, movement)]: each interval split into movements of AKATHISIA_SEGMENT_S (the last at least 2 s), each
    movement drawn from menu unlike the one before (first, if given, forced on every other movement), with a pause
    of AKATHISIA_PAUSE_S (0.5 to 0.8 s at grade 3) after half of them."""
    out, prev = [], None
    pause = (0.5, 0.8) if grade == 3 else AKATHISIA_PAUSE_S
    for a, b in intervals:
        t0 = float(a)
        while b - t0 >= 2.0:
            d = float(rng.uniform(*AKATHISIA_SEGMENT_S))
            e = b if b - (t0 + d) < 2.0 else t0 + d
            if first is not None and len(out) % 2 == 0:
                k = first
            else:
                k = str(rng.choice([m for m in menu if m != prev] or menu))
            out.append((t0, float(e), k))
            prev = k
            t0 = float(e)
            if b - t0 >= 2.0 + pause[1] and rng.uniform() < 0.5:
                t0 += float(rng.uniform(*pause))
    return out


def akathisia(posture="sit", grade=1, seconds=AKATHISIA_S, seed=SEED):
    """Akathisia (BARS objective item, barnes1989): the characteristic restless movements for the grade's share of
    the observation (BARS), each bout split into movements of 3 to 6 s that change from one to the next, with short
    pauses (akathisia_segments), their rates and sizes jittered by cycle (AKATHISIA_JITTER), the fingers fidgeting
    throughout: seated, tramping (the feet lifted in turn in place), swinging one leg (the foot lifted, the knee
    swinging at the 0.58 Hz of janno2008) or shifting on the seat (the trunk side to side with a pelvic tilt and the
    knees swinging in and out); standing, rocking from foot to foot or forward and back (both feet planted), small
    steps or walking on the spot (grades 2 and 3; grade 3 marches every other movement; the weight shifted over the
    stance foot before each step, step_track). Crossing and uncrossing the legs is not built."""
    base = base_body(posture, seconds)
    rng = as_rng(seed)
    share = BARS[grade]
    iv = _iv(base, share, rng, bout_s=4.0)
    t = base.times().numpy()
    T = base.T
    J = AKATHISIA_JITTER
    if posture == "sit":
        segs = akathisia_segments(iv, rng, list(AKATHISIA_SEATED), grade)
        rest = base.rest().numpy()
        shank = {s: float(np.linalg.norm(rest[LEG[s][2]] - rest[LEG[s][1]])) for s in ("left", "right")}
        lift = {s: np.zeros(T) for s in ("left", "right")}
        fwd = {s: np.zeros(T) for s in ("left", "right")}
        swv = {s: np.zeros(T) for s in ("left", "right")}
        lat_t, tilt = np.zeros(T), np.zeros(T)
        for a, b, k in segs:
            env = ramp_envelope(t, [(a, b)], 0.4)
            P = AKATHISIA_SEATED[k]
            if k == "tramp":
                ph, am = phase_track(t, P["rate_hz"] * rng.uniform(0.85, 1.15), rng, J)
                for i, s in enumerate(("left", "right")):
                    lift[s] += P["lift_m"] * am * (0.5 - 0.5 * np.cos(ph + math.pi * i)) * env
            elif k == "swing":
                s = str(rng.choice(["left", "right"]))
                sw = pattern(t, P["rate_hz"] * rng.uniform(0.85, 1.15), rng, None, jitter=J)
                inner = ramp_envelope(t, [(a + 0.6, b - 0.6)], 0.4) if b - a > 2.0 else np.zeros(T)
                f = ((P["fwd_m"] + P["back_m"]) * (0.5 + 0.5 * sw) - P["back_m"]) * inner
                lift[s] += P["lift_m"] * env + shank[s] - np.sqrt(shank[s] ** 2 - f ** 2)   # the arc about the knee
                fwd[s] += f
            else:
                x = pattern(t, P["rate_hz"] * rng.uniform(0.85, 1.15), rng, None, jitter=J)
                lat_t += P["trunk_lat_deg"] * 0.5 * x * env
                tilt += P["tilt_deg"] * 0.5 * np.abs(x) * env
                ph, am = phase_track(t, P["swivel_hz"], rng, J)
                for i, s in enumerate(("left", "right")):
                    v = np.sin(ph + math.pi * i)
                    swv[s] += P["swivel_deg"] * am * np.where(SWIVEL_OUT[s] * v > 0, 1.0, 0.5) * v * env

        def legs(b, rng_):
            for s in ("left", "right"):
                R, _ = seated_legs(b, s, lift=lift[s], fwd=fwd[s], swivel_deg=swv[s])
                b = b.with_rotations(R, list(LEG[s][:3]))
            return tilt_pelvis_seated(b, tilt), [{"lift_max_m": {s: float(lift[s].max()) for s in lift}}]
        plan = [_entry("trunk", ("trunk_lat",), None, iv, kind="given", raw={"trunk_lat": lat_t},
                       deg={"trunk_lat": 1.0})]
        movements = AKATHISIA_SEATED
    else:
        menu = ["rock", "sway"] if grade == 1 else ["rock", "sway", "small_steps", "march"]
        segs = akathisia_segments(iv, rng, menu, grade, first="march" if grade == 3 else None)
        dx, dz = np.zeros(T), np.zeros(T)
        lift, heel = np.zeros((T, 2)), np.zeros((T, 2))
        geo = step_geometry(base)
        for a, b, k in segs:
            env = ramp_envelope(t, [(a, b)], 0.4)
            P = AKATHISIA_STANDING[k]
            ph, am = phase_track(t, P["rate_hz"] * rng.uniform(0.85, 1.15), rng, J)
            if k == "sway":
                dz += P["dz_m"] * 0.5 * am * np.sin(ph) * env
            elif k == "rock":
                dx += P["dx_m"] * 0.5 * am * np.sin(ph) * env
            else:                                        # a step on the spot over a shifted weight
                lf, hl, sx, _ = step_track(ph, am, env, P["lift_m"], geo, base.rate)
                lift, heel, dx = lift + lf, heel + hl, dx + sx

        def legs(b, rng_):
            out, inf = standing_legs(b, dx=dx, dz=dz, lift=lift, heel_deg=heel)
            return out, [dict(inf, step_geometry=geo, pelvis_shift_m_max=float(np.abs(dx).max()),
                              stance=stance_report(out, b, omega0=geo["omega0"]))]
        plan = []
        movements = AKATHISIA_STANDING
    fing = [{"side": s, "kind": "events", "level": AKATHISIA_FIDGET, "intervals": [(a, b) for a, b, _ in segs],
             "supported": posture == "sit"} for s in ("left", "right")] if segs else None
    params = {"symptom": "akathisia", "posture": posture, "bars_objective": grade, "share_of_time": share,
              "bouts": [{"interval_s": [float(a), float(b)], "movement": k} for a, b, k in segs],
              "movements": movements, "segment_s": AKATHISIA_SEGMENT_S, "jitter": J, "finger_fidget": AKATHISIA_FIDGET,
              "observation": f"{seconds:g} s (the BARS observes 2 minutes seated and 2 minutes standing)"}
    return build(base, plan, rng, [legs], fing, "akathisia", grade, "BARS objective item", params,
                 ("barnes1989", "janno2008", "braude1984", "module", "table")
                 + (("hof2005",) if posture != "sit" else ()),
                 regions_measured=("trunk", "leg_left", "leg_right"), posture=posture, balance=False)


RLS_INDEX = {1: 60, 2: 120, 3: 240, 4: 480}          # [module] movements per hour (montplaisir1998: above 40)
RLS_MOVE = {"lift_m": 0.10, "fwd_m": (0.08, 0.15), "lat_m": 0.03, "up_s": (0.9, 1.0), "pitch_deg": (10.0, 20.0),
            "pitch_hz": (0.5, 1.0), "relocate_s": (1.5, 3.0), "toes_deg": (15.0, 25.0), "toes_hz": (0.5, 1.0),
            "single_fwd_m": (0.05, 0.08), "rise_s": 0.25}
# [module] a restless-legs movement: the foot lifted 10 cm (over 0.25 s) for 0.9 to 1 s (never longer), the ankle
# stretching and flexing by 10 to 20 deg at 0.5 to 1 Hz while it is up, the leg stretched 8 to 15 cm forward and the
# foot set down there (the knee extended); while it rests there the toes lift and relax (15 to 25 deg, 0.5 to 1 Hz)
# and every 1.5 to 3 s the foot is lifted and set down again a little further or nearer; at the end it is lifted back
# to its place (the knee relaxed); the foot travels only while it is up (above AIR_M), by a minimum-jerk excursion
# from 0.03 s before the lift is complete to 0.03 s after the descent begins. A movement under 2.3 s is one stretch
# out and back (5 to 8 cm) in the air, lasting 1 s.


def rls_track(t, a, d, rng):
    """(lift, fwd, lat, pitch deg, toes deg) (T,) of one restless-legs movement of d s from a s (RLS_MOVE), and its
    record."""
    M = RLS_MOVE
    T = len(t)
    lift, fwd, lat, pitch, toes = (np.zeros(T) for _ in range(5))
    u = lambda lo_hi: float(rng.uniform(*lo_hi))  # noqa: E731

    r = M["rise_s"]

    def air(t0, up, x0, x1, y0, y1):
        lift[:] += M["lift_m"] * _pulse(t, t0, r, up - 2 * r, r)
        s = minimum_jerk((t - (t0 + r - 0.03)) / (up - 2 * r + 0.06))
        on = (t >= t0) & (t <= t0 + up)
        fwd[on] = (x0 + (x1 - x0) * s)[on]
        lat[on] = (y0 + (y1 - y0) * s)[on]
        f, A = u(M["pitch_hz"]), u(M["pitch_deg"])
        pitch[:] += A * np.sin(2 * math.pi * f * (t - t0)) * _pulse(t, t0 + r, 0.12, max(up - 2 * r - 0.24, 0.0), 0.12)

    if d < 2.3:
        up = 1.0
        F = u(M["single_fwd_m"])
        lift[:] += M["lift_m"] * _pulse(t, a, r, up - 2 * r, r)
        dd = 0.5 * (up - 2 * r + 0.06)
        fwd[:] = F * (minimum_jerk((t - (a + r - 0.03)) / dd) - minimum_jerk((t - (a + r - 0.03 + dd)) / dd))
        f, A = u(M["pitch_hz"]), u(M["pitch_deg"])
        pitch[:] += A * np.sin(2 * math.pi * f * (t - a)) * _pulse(t, a + r, 0.12, max(up - 2 * r - 0.24, 0.0), 0.12)
        return lift, fwd, lat, pitch, toes, {"onset_s": a, "duration_s": up, "set_down_at_m": []}
    up_out, up_back = u(M["up_s"]), u(M["up_s"])
    x, y = u(M["fwd_m"]), float(rng.uniform(-M["lat_m"], M["lat_m"]))
    air(a, up_out, 0.0, x, 0.0, y)
    spots = [(x, y)]
    t0 = a + up_out
    back = a + d - up_back
    while True:
        nxt = t0 + u(M["relocate_s"])
        up = u(M["up_s"])
        if nxt + up > back - 0.5:
            break
        x2 = float(np.clip(x + rng.uniform(-0.05, 0.05), 0.08, 0.2))
        y2 = float(np.clip(y + rng.uniform(-0.02, 0.02), -M["lat_m"], M["lat_m"]))
        fwd[(t > t0) & (t < nxt)], lat[(t > t0) & (t < nxt)] = x, y
        air(nxt, up, x, x2, y, y2)
        x, y, t0 = x2, y2, nxt + up
        spots.append((x, y))
    fwd[(t > t0) & (t < back)], lat[(t > t0) & (t < back)] = x, y
    air(back, up_back, x, 0.0, y, 0.0)
    planted = [(a + up_out + 0.1, back - 0.1)]
    f, A = u(M["toes_hz"]), u(M["toes_deg"])
    toes[:] = A * (0.5 - 0.5 * np.cos(2 * math.pi * f * (t - planted[0][0]))) * ramp_envelope(t, planted, 0.3)
    return lift, fwd, lat, pitch, toes, {"onset_s": a, "duration_s": d,
                                         "set_down_at_m": [[round(p, 3), round(q, 3)] for p, q in spots]}


def restless_legs(grade=1, seconds=RLS_S, seed=SEED):
    """Restless legs while seated: leg movements of 0.5 to 10 s (log-uniform) at intervals of at least 4 s
    (zucconi2006), one leg at a time drawn at random (rls_track, RLS_MOVE: the foot lifted at most 1 s at a time, the
    ankle stretching and flexing while it is up, the leg stretched forward and the foot set down there with the toes
    moving, then brought back), RLS_INDEX[grade] movements per hour (a declared ladder above the 40 per hour of
    montplaisir1998; the suggested immobilization test is done in bed, the clip shows the chair). Rubbing one leg
    with the other and crossing them are not built."""
    base = base_body("sit", seconds)
    rng = as_rng(seed)
    n = max(1, int(round(RLS_INDEX[grade] * seconds / 3600.0)))
    t = base.times().numpy()
    moves, t0 = [], 1.0
    slack = seconds - 1.0 - 0.5
    durs = [_loguniform(rng, 0.5, 10.0) for _ in range(n)]
    while sum(durs) + 4.0 * (n - 1) > slack:
        durs = [d * 0.9 for d in durs]
    free = slack - sum(durs) - 4.0 * (n - 1)
    gaps = rng.dirichlet(np.ones(n + 1)) * free
    t0 = 1.0 + gaps[0]
    for i in range(n):
        moves.append((float(t0), float(durs[i]), str(rng.choice(["left", "right"]))))
        t0 += durs[i] + 4.0 + gaps[i + 1]
    tracks = {s: [np.zeros(base.T) for _ in range(5)] for s in ("left", "right")}
    recs = []
    for a, d, s in moves:
        *arr, rec = rls_track(t, a, d, rng)
        for k in range(5):
            tracks[s][k] += arr[k]
        recs.append(dict(rec, side=s))
    plan = [_entry(f"leg_{s}", (f"toes_{s}",), None, [(0.0, base.duration)], kind="given",
                   raw={f"toes_{s}": tracks[s][4]}, deg={f"toes_{s}": 1.0}) for s in ("left", "right")
            if tracks[s][4].any()]

    def legs(b, rng_):
        for s in ("left", "right"):
            lm, fw, la, pi, _ = tracks[s]
            R, _ = seated_legs(b, s, lift=lm, fwd=fw, lat=SIDE_SIGN[s] * la, pitch_deg=pi)
            b = b.with_rotations(R, list(LEG[s][:3]))
        return b, [{"movements": len(moves)}]
    params = {"symptom": "restless legs while seated", "grade": grade, "index_per_hour": RLS_INDEX[grade],
              "movements": recs, "move": RLS_MOVE}
    return build(base, plan, rng, [legs], None, "restless_legs", grade, "declared ladder (movements per hour)", params,
                 ("zucconi2006", "montplaisir1998", "module", "table"), regions_measured=("leg_left", "leg_right"),
                 posture="sit")


TOES_TIMING = {"rise": (0.25, 0.5), "hold": (0.0, 0.4), "ret": (0.25, 0.5), "next": (0.6, 1.6), "n_ch": (1, 1),
               "amp": (0.35, 1.0), "positive": True}
# bursts of 0.5 to 1 s (dressler1994), about one a second (bermejo2008), at irregular intervals and sizes (the next
# burst after 0.6 to 1.6 of the last one's length, its size 0.35 to 1 of the level; the clinical review of 2026-11-03
# read the former 0.8 to 1.2 and 0.6 to 1 as a regular 1.35 Hz with flat tops; module values); the toe block of a
# planted foot only rises (it cannot flex into the floor)


@functools.lru_cache(maxsize=4)
def toes_level(amplitude_grade, side="left"):
    """Share of the toe block's ROM a moving-toes clip of the DIS amplitude grade reaches: the band centre
    (DIS_LEVELS), or, where the planted foot's toes cannot reach it, the largest share the soft limit leaves toward
    extension (SOFT_MARGIN times the extension room of the seated base over the ROM). Open-body port, wave 3: on the
    anatomical table the toe block's ROM is jarm2022's 40 deg of extension and 35 of flexion, and a planted foot's toes
    rise only (they cannot flex into the floor), so the largest share is 0.97 x 40 / 75 = 0.517, inside DIS band 3
    (50 to under 90 percent) but short of its centre (70); on the AMASS table the block flexed 2 to 11 deg only and
    the centre was reachable."""
    b = base_body("sit", 0.02)
    room, _ = channel_room(b, f"toes_{side}")
    return float(min(DIS_LEVELS[amplitude_grade], SOFT_MARGIN * room / channel_rom(f"toes_{side}")))


def moving_toes(amplitude_grade=2, side="left", seconds=CHOREA_S, seed=SEED):
    """Moving toes, seated (painful legs and moving toes; the pain is not shown): the toes of one foot raised and
    lowered as one block (the program's foot joint carries no single toe: flexion-extension of the block is shown,
    abduction, fanning and clawing are not), irregular bursts (TOES_TIMING), DIS amplitude at the band centre of the
    toe block's range or the largest share a planted foot's toes reach inside the band (toes_level), continuous."""
    base = base_body("sit", seconds)
    rng = as_rng(seed)
    level = toes_level(amplitude_grade, side)
    plan = [_entry(f"leg_{side}", (f"toes_{side}",), level, [(0.5, base.duration - 0.25)], timing=TOES_TIMING)]
    params = {"symptom": "moving toes", "side": side, "dis_amplitude_grade": amplitude_grade,
              "level_share_of_rom": level, "toe_rom_deg": channel_rom(f"toes_{side}"), "timing": TOES_TIMING,
              "not_shown": "single toes (fanning, clawing, abduction), pain"}
    return build(base, plan, rng, None, None, "moving_toes", amplitude_grade, "DIS amplitude (declared)", params,
                 ("alvarez2008", "dressler1994", "bermejo2008", "dis", "table", "module"),
                 regions_measured=("leg_left", "leg_right"))


# ---------------------------------------------------------------- mirror movements
def tapping_both_body(seconds):
    """A seated HandBody with both arms in the finger-tapping posture of motions.POSTURES3 and both hands pinched
    (the left arm and hand from posture3(.., 'left'), the right from posture3(.., 'right'))."""
    from .motions import ARM_JOINTS, posture3
    Rl, Hl = posture3("sit_finger_tapping", "left")
    Rr, Hr = posture3("sit_finger_tapping", "right")
    R = Rl.clone()
    R[list(ARM_JOINTS["right"])] = Rr[list(ARM_JOINTS["right"])]
    Rh = Hl.clone()
    Rh[15:30] = Hr[15:30]
    T = int(round(seconds * 100.0)) + 1
    rest = rest_joints()
    tr = torch.zeros(T, 3, dtype=F64)
    tr[:, [0, 2]] = -rest[0, [0, 2]]
    from .body import floor_to_zero
    b = H.HandBody(matrix_to_rot6d(R)[None].expand(T, -1, 6).clone(), tr, torch.zeros(T, 2, dtype=torch.bool),
                   hand6d=matrix_to_rot6d(Rh)[None].expand(T, -1, 6).clone())
    b = floor_to_zero(b)
    return H.HandBody(b.rot6d, b.transl, contact_labels(b), b.betas, b.gender, b.rate, b.hand6d)


def mirror_seconds():
    from . import grades as G
    return round(0.5 + G.N_CYCLES / G.normal_rate_hz("finger_tapping")[0] + 1.5, 2)


def mirror_movements(grade=None, seed=SEED):
    """Mirror movements: the right hand taps (handtasks.hand_task at its cited level 0), the left hand, held in the
    same posture, opens and closes in phase by MIRROR_RATIO[grade] of the right hand's pinch opening (Woods and
    Teuber; grade '2b' the stronger but briefer form: the ratio of grade 3 for the first three cycles only). grade
    None: the level 0 (no mirror)."""
    from .handtasks import hand_at, hand_task, share_of_amplitude, task_rows
    base = tapping_both_body(mirror_seconds())
    vol, tv = hand_task(base, "finger_tapping", "right", decrement_onset=None, seed=seed)
    if grade is None:
        return vol, truth_record("none", seed, 0, "level 0", {"base": "both arms in the finger-tapping posture, the "
                                                              "right hand tapping (handtasks level 0)",
                                                              "voluntary": {k: tv["params"][k] for k in
                                                                            ("rate_hz", "amplitude0", "n_cycles")}},
                                 ("woods1978",), tv["validity"])
    ratio = MIRROR_RATIO[grade]
    cyc = tv["params"]["cycles"]
    k0 = int(round(tv["params"]["start_s"] * 100.0))
    amps = np.array([c["amplitude"] * ratio * (1.0 if (grade != "2b" or c["cycle"] <= 3) else 0.0) for c in cyc])
    on = amps > 0
    shares = np.zeros(len(cyc))
    if on.any():
        shares[on] = share_of_amplitude(base, "finger_tapping", "left", amps[on], k0)
    t = base.times().numpy()
    s = np.zeros(base.T)
    for c, sh in zip(cyc, shares):
        a, P = c["start_s"], c["period_s"]
        inside = (t >= a) & (t <= a + P)
        s[inside] = sh * 0.5 * (1.0 - np.cos(2.0 * math.pi * (t[inside] - a) / P))
    Rh = vol.hand_R().clone()
    Rh[:, task_rows("left")] = hand_at("finger_tapping", "left", s)[:, task_rows("left")]
    out = vol.with_hands(Rh, task_rows("left"))
    val = validity(base, out)
    val["hand_limits"] = H.hand_limit_report(out.hand_R())
    val["finite"] = bool(torch.isfinite(out.hand6d).all())
    m = measure_mirror(out, vol)
    params = {"symptom": "mirror movements", "woods_teuber": grade, "ratio": ratio,
              "brief": grade == "2b", "voluntary_side": "right", "mirror_side": "left",
              "voluntary": {k: tv["params"][k] for k in ("rate_hz", "amplitude0", "n_cycles", "start_s")},
              "mirror_amplitudes_mm": amps.tolist(), "measured": m}
    return out, truth_record("mirror_movements", seed, grade, "Woods and Teuber 1978", params,
                             ("woods1978", "hermsdoerfer1995", "module", "hand_limits"), val,
                             arrays={"mirror_share": s})


def measure_mirror(body, level0):
    """{ratio_per_cycle, ratio_median, lag_s, cycles_with_mirror}: the left (mirror) fingertip opening over the right
    (voluntary) one per voluntary cycle (handtasks.tip_distance_m minus its closed value), and the lag of the mirror's
    cross-correlation peak."""
    from .handtasks import read_cycles, tip_distance_m
    dl = tip_distance_m(body, "left")
    dr = tip_distance_m(body, "right")
    dl0 = tip_distance_m(level0, "left")
    ol, orr = dl - dl0, dr - dr.min()
    cyc = read_cycles(orr, body.rate)
    ratios = []
    for i, (a, k, v) in enumerate(cyc):
        b = cyc[i + 1][0] if i + 1 < len(cyc) else len(orr)
        ratios.append(float(ol[a:b].max() / v) if v > 0 else 0.0)
    x, y = ol - ol.mean(), orr - orr.mean()
    lag = 0.0
    if np.abs(x).max() > 1e-6:
        cc = np.correlate(x, y, "full")
        lag = float((np.argmax(cc) - (len(y) - 1)) / body.rate)
    return {"ratio_per_cycle": ratios, "ratio_median": float(np.median(ratios)) if ratios else 0.0,
            "cycles_with_mirror": int(sum(r > 0.02 for r in ratios)), "lag_s": lag,
            "mirror_open_max_mm": float(1000 * ol.max())}


# ---------------------------------------------------------------- the clip specifications
NOT_SHOWN = {
    "face": "the face, lips, jaw and tongue: the UHDRS face and bucco-oral-lingual chorea, AIMS items 1 to 4 "
            "(orobuccolingual tardive dyskinesia), the UHDRS tongue impersistence; the body has no face",
    "eyes": "pseudoathetosis is shown with the eyes closed stated in the caption: the body has no eyes",
    "toes": "single toes: moving toes moves the toe block of the program's foot joint (flexion-extension; fanning, "
            "clawing and abduction not shown); one block lifting by at most 2.5 cm is hard to see at body scale; the "
            "pain is not shown",
    "seat": "seated as in the AIMS examination (hands resting on the thighs, sitting forward of the backrest); the "
            "drawn chair has no armrests (scene 'chair_armless')",
    "leg crossing": "akathisia seated: crossing and uncrossing the legs is not built (tramping, swinging one leg and "
                    "shifting on the seat are); restless legs: rubbing and crossing the legs are not built",
    "standing leg chorea": "standing, the legs move through the pelvis with both feet planted (sway, turns, tilts, dips); "
                           "no foot is lifted by chorea while standing",
    "foot inversion": "AIMS item 6 foot squirming, inversion and eversion: not built (a planted foot cannot roll in this "
                      "module)",
    "UDysRS tasks": "the UDysRS part 3 rates during communication, drinking, dressing and walking; the clips show seated "
                    "rest and standing",
    "observation length": "BARS observes 2 minutes seated and 2 standing, the SIT one hour in bed; the clips last 30 s and "
                          "60 s in a chair",
}

L0 = {"E1": ("invol_E1_level0", "sit", CHOREA_S, ("front", "left"), "E1", ""),
      "E7": ("invol_E7_level0", "stand", CHOREA_S, ("front", "left"), "E7", ""),
      "E2": ("invol_E2_level0", "arms_forward", CHOREA_S, ("front", "hands"), "E2", ""),
      "flap": ("invol_E7_flap_level0", "flap", CHOREA_S, ("front", "left"), "E7", "standing, hands before the chest"),
      "E1_30": ("invol_E1_level0_30s", "sit", AKATHISIA_S, ("front", "left"), "E1", ""),
      "E7_30": ("invol_E7_level0_30s", "stand", AKATHISIA_S, ("front", "left"), "E7", ""),
      "E1_60": ("invol_E1_level0_60s", "sit", RLS_S, ("front", "left"), "E1", "")}
MIRROR_L0 = "invol_E4a_mirror_level0"
MIRROR_TITLE = "finger tapping, right hand; the left hand held in the same posture"


def _spec(**kw):
    from .exam import ClipSpec
    kw.setdefault("since", SINCE)
    return ClipSpec(**kw)


def _level0_spec(key):
    cid, kind, seconds, views, seg, title = L0[key]

    def build_fn():
        b = base_body(kind, seconds)
        return b, truth_record("none", None, 0, "level 0", {"base": kind, "clip_s": seconds,
                                                            "note": "standing bases: the procedural stand with the knees "
                                                                    "soft (stand_soft); seated: the AIMS position, "
                                                                    "hands on the thighs (base_body)"}, (), None)
    return _spec(clip_id=cid, family="reference", segment=seg, item="level 0",
                 operator=f"none ({kind.replace('_', ' ')})", grade=0,
                 grade_text="level 0", params_text=f"the shared neutral body, {kind.replace('_', ' ')}, {seconds:g} s"
                 + (", hands resting on the thighs" if kind == "sit" else ""),
                 views=views, level0=None, scene=SCENE[kind], build=build_fn, segment_title=title,
                 note=NOT_SHOWN["seat"] if kind == "sit" else "")


def _pct(x):
    return f"{100 * x:.0f} percent"


def specs():
    """[ClipSpec] of the flowing involuntary-movement family (module docstring), since 'catalogue_v1'."""
    out = [_level0_spec(k) for k in L0]
    sv = ("front", "left")

    def add(cid, family, seg, item, op, grade, gtext, ptext, views, l0, scene, fn, title="", note=""):
        out.append(_spec(clip_id=cid, family=family, segment=seg, item=item, operator=op, grade=grade, grade_text=gtext,
                         params_text=ptext, views=views, level0=l0, scene=scene, build=fn, segment_title=title,
                         note=note))
    face = "face not shown"
    # chorea, generalized
    for posture, seg, key in (("sit", "E1", "E1"), ("stand", "E7", "E7")):
        for g in (1, 2, "2b", 3, 4):
            a, d = UHDRS[g]
            add(f"invol_chorea_{seg}_g{g}", "chorea", seg, "UHDRS maximal chorea", "chorea, generalized", g,
                f"UHDRS maximal chorea grade {str(g)[0]} ({a}/{d}), every region; {face}",
                f"amplitude {_pct(AMP[a])} of the range of motion (trunk and neck at most "
                f"{_pct(min(AMP[a], TRUNK_NECK_CAP))}), {_pct(DUR[d])} of the time (declared map); movements of 0.2 "
                f"to 0.5 s, the larger ones slower, flowing from posture to posture"
                + ("; the feet at times kicked forward" if posture == "sit" and g in CHOREA_KICK_P else ""),
                sv, L0[key][0], SCENE[posture], lambda posture=posture, g=g: chorea(posture, g), note=NOT_SHOWN["face"]
                + ("; " + NOT_SHOWN["standing leg chorea"] if posture == "stand" else "; " + NOT_SHOWN["seat"]))
    # chorea by region, seated
    for reg in ("trunk", "arms", "legs"):
        for g in (1, 2, 3, 4):
            a, d = UHDRS[g]
            add(f"invol_chorea_E1_{reg}_g{g}", "chorea", "E1", f"UHDRS maximal chorea, {reg}", f"chorea, {reg}", g,
                f"UHDRS maximal chorea {reg} grade {g} ({a}/{d}); other regions 0",
                f"amplitude {_pct(min(AMP[a], TRUNK_NECK_CAP) if reg == 'trunk' else AMP[a])} of the range of "
                f"motion, {_pct(DUR[d])} of the time (declared map)"
                + ("; the arms follow passively, the hands resting on the thighs" if reg == "trunk" else "")
                + (", a hand lifted off its thigh when the trunk sways onto the arm" if reg == "trunk" and g >= 3
                   else "")
                + ("; the feet at times kicked forward" if reg == "legs" and g in CHOREA_KICK_P else ""),
                sv, L0["E1"][0], SCENE["sit"], lambda reg=reg, g=g: chorea("sit", g, regions=(reg,)),
                note=NOT_SHOWN["seat"])
    # piano-playing fingers and impersistence, arms outstretched
    for g in (1, 2, 3, 4):
        a, d = UHDRS[g]
        add(f"invol_chorea_E2_fingers_g{g}", "chorea", "E2", "UHDRS maximal chorea, arms", "chorea, piano-playing "
            "fingers, both hands", g, f"UHDRS maximal chorea arms grade {g} ({a}/{d}), fingers"
            + (f" and proximal arm chorea ({'moderate' if g == 3 else 'marked'})" if g in PIANO_ARM else ""),
            f"finger flexion {_pct(min(AMP[a], FINGER_LEVEL_CAP))} of its range, neighbouring fingers moving "
            f"together, {_pct(DUR[d])} of the time; wrist movements at a quarter of the finger level"
            + (f"; shoulders and elbows at {_pct(PIANO_ARM[g])} of the range of motion" if g in PIANO_ARM else ""),
            ("hands", "hand_left_medial") if g > 2 else ("hands", "hand_left"), L0["E2"][0], "chair",
            lambda g=g: piano_fingers(g))
    for g in (1, 2, 3):
        add(f"invol_impersistence_E2_g{g}", "chorea", "E2", "motor impersistence", "motor impersistence of the "
            "outstretched arms", g, f"declared ladder grade {g}",
            f"{IMPERSISTENCE_LAPSES[g]} lapses of the posture per 20 s: the arm sinks 10 to 20 deg and the hand drops "
            f"15 to 25 deg over 0.4 to 0.9 s, then a quick correction",
            ("left", "hands"), L0["E2"][0], "chair", lambda g=g: impersistence(g),
            note="the UHDRS rates impersistence on the tongue only (not shown): a declared ladder")
    # hemiballism
    for g in (1, 2, 3):
        add(f"invol_hemiballism_E1_left_g{g}", "ballism", "E1", "hemiballism", "hemiballism, left arm and leg", g,
            f"declared ladder grade {g} (share of time)",
            f"flings of the shoulder outward to 70 to 95 percent of its room (out in 0.15 to 0.3 s, the elbow "
            f"whipping open behind it), about 1 to 1.5 per second; kicks of the leg forward and out at irregular "
            f"intervals; {_pct(DUR[('intermittent', 'common', 'prolonged')[g - 1]])} of the time", sv, L0["E1"][0],
            SCENE["sit"], lambda g=g: hemiballism(g), note=NOT_SHOWN["seat"])
    # athetosis and choreoathetosis
    for ag, dg in ((1, 4), (2, 4), (3, 4), (2, 2), (2, 3)):
        add(f"invol_athetosis_E1_a{ag}_d{dg}", "athetosis", "E1", "DIS choreoathetosis", "athetosis, hands, "
            "forearms, toes, neck", f"{ag}/{dg}", f"DIS amplitude {ag}, duration {dg}",
            f"writhing at 0.3 to 0.6 Hz, {_pct(DIS_LEVELS[ag])} of the range of motion, {_pct(DIS_LEVELS[dg])} of the time",
            ("front", "hands", "left"), L0["E1"][0], SCENE["sit"], lambda ag=ag, dg=dg: athetosis(ag, dg),
            note=NOT_SHOWN["toes"])
    for ag, dg in ((2, 3), (3, 3)):
        add(f"invol_choreoathetosis_E1_a{ag}_d{dg}", "athetosis", "E1", "DIS choreoathetosis", "choreoathetosis",
            f"{ag}/{dg}", f"DIS amplitude {ag}, duration {dg}",
            f"writhing at 0.3 to 0.6 Hz and chorea fragments, {_pct(DIS_LEVELS[ag])} of the range of motion, "
            f"{_pct(DIS_LEVELS[dg])} of the time", ("front", "hands", "left"), L0["E1"][0], SCENE["sit"],
            lambda ag=ag, dg=dg: athetosis(ag, dg, chorea_mix=True))
    # pseudoathetosis
    for g in (1, 2, 3):
        add(f"invol_pseudoathetosis_E2_left_g{g}", "athetosis", "E2", "pseudoathetosis", "pseudoathetosis, left, eyes "
            "closed", g, f"declared ladder grade {g} ({', '.join(PSEUDO_PARTS[g])})",
            f"slow drift at 0.1 to 0.4 Hz of the {', '.join(PSEUDO_PARTS[g])}"
            + (", the wrist sinking" if g >= 2 else "") + (", the arm sinking and drifting out" if g == 3 else "")
            + "; eyes closed (stated, not drawn)",
            ("hands", "left"), L0["E2"][0], "chair", lambda g=g: pseudoathetosis(g),
            title="arms outstretched, eyes closed", note=NOT_SHOWN["eyes"])
    # levodopa-induced dyskinesia
    for posture, seg, key in (("sit", "E1", "E1"), ("stand", "E7", "E7")):
        for st in (1, 2, 3, 4):
            lv, sh = STEPS[st]
            add(f"invol_lid_peak_{seg}_s{st}", "dyskinesia", seg, "UDysRS part 3", "levodopa-induced dyskinesia, "
                "peak-dose choreic", st, f"UDysRS part 3 score {st} ({UDYSRS_TERMS[st]}), declared map; {face}",
                f"amplitude {_pct(lv)} of the range of motion (trunk and neck at most "
                f"{_pct(min(lv, TRUNK_NECK_CAP))}), "
                f"{_pct(sh)} of the time, fragments of 1 to 3 Hz",
                sv, L0[key][0], SCENE[posture], lambda posture=posture, st=st: lid_peak_dose(posture, st),
                note=NOT_SHOWN["UDysRS tasks"] + ("; " + NOT_SHOWN["seat"] if posture == "sit" else ""))
    for st in (1, 2, 3):
        add(f"invol_lid_diphasic_E1_kicking_s{st}", "dyskinesia", "E1", "UDysRS part 3", "diphasic dyskinesia, "
            "alternating kicking", st, f"UDysRS part 3 score {st} ({UDYSRS_TERMS[st]}), declared map",
            f"legs in antiphase at {DIPHASIC_HZ:g} Hz, {_pct(STEPS[st][0])} of the leg's range, {_pct(STEPS[st][1])} of "
            f"the time", sv, L0["E1"][0], SCENE["sit"], lambda st=st: diphasic("sit", st))
        add(f"invol_lid_diphasic_E7_stepping_s{st}", "dyskinesia", "E7", "UDysRS part 3", "diphasic dyskinesia, "
            "stepping in place", st, f"UDysRS part 3 score {st} ({UDYSRS_TERMS[st]}), declared map",
            f"alternating steps at {DIPHASIC_HZ:g} Hz, lift {100 * STEP_LIFT_M[st]:.0f} cm, the weight shifted onto "
            f"the standing foot before each step and the heel rising first, {_pct(STEPS[st][1])} of the time", sv,
            L0["E7"][0], None, lambda st=st: diphasic("stand", st))
    # tardive dyskinesia
    for st in (1, 2, 3, 4):
        add(f"invol_td_E1_rocking_s{st}", "dyskinesia", "E1", "AIMS item 7", "tardive dyskinesia, truncal rocking", st,
            f"AIMS item 7 score {st} ({AIMS_TERMS[st]}), declared map; {face}",
            f"trunk pitch {TD_ROCK_DEG[st]:g} deg peak to peak at {TD_ROCK_HZ:g} Hz, forward from upright and back, "
            f"{_pct(STEPS[st][1])} of the time",
            ("left", "front"), L0["E1"][0], SCENE["sit"], lambda st=st: td_rocking(st), note=NOT_SHOWN["face"])
        add(f"invol_td_E1_limbs_s{st}", "dyskinesia", "E1", "AIMS items 5 and 6", "tardive dyskinesia, limbs", st,
            f"AIMS items 5 and 6 score {st} ({AIMS_TERMS[st]}), declared map; {face}",
            f"choreoathetoid fragments at {_pct(STEPS[st][0])} of the range of motion, {_pct(STEPS[st][1])} of the "
            f"time; lateral knee movement, toe raising", ("front", "hands", "left"), L0["E1"][0], SCENE["sit"],
            lambda st=st: td_limbs(st), note=NOT_SHOWN["face"] + "; " + NOT_SHOWN["foot inversion"])
    for st in (2, 3, 4):
        add(f"invol_td_E7_thrusting_s{st}", "dyskinesia", "E7", "AIMS item 7", "tardive dyskinesia, pelvic "
            "thrusting", st, f"AIMS item 7 score {st} ({AIMS_TERMS[st]}), declared map; {face}",
            f"pelvis about {100 * TD_THRUST_M[st]:.1f} cm forward and back at about {TD_THRUST_HZ:g} Hz, tilting back as it "
            f"thrusts with the lumbar spine flexing in counter-motion, {_pct(STEPS[st][1])} of the time, feet planted",
            ("left", "front"), L0["E7"][0], None, lambda st=st: td_pelvic_thrust(st),
            note=NOT_SHOWN["face"])
    # stereotypies
    for kind, l0key, views in (("rocking", "E1", ("left", "front")), ("flapping", "flap", ("front", "left")),
                               ("nodding", "E1", ("left", "front"))):
        p = STEREO[kind]
        for g in (1, 2, 3):
            add(f"invol_stereotypy_{kind}_g{g}", "stereotypy", L0[l0key][4], "stereotypy", f"stereotypy, {kind}", g,
                f"declared ladder grade {g} (share of time)",
                f"about {p['rate_hz']:g} Hz, about {p['deg_pp']:g} deg peak to peak (jittered by cycle)"
                + (", forward from upright and back" if kind == "rocking" else "")
                + (", palms down" if kind == "flapping" else "")
                + f", {_bouts_text(stereotypy_bouts(kind, g))}, {_pct(STEREO_SHARE[g])} of the time", views,
                L0[l0key][0],
                SCENE[p["posture"]],
                lambda kind=kind, g=g: stereotypy(kind, g), title=L0[l0key][5])
    # akathisia
    for posture, seg, key in (("sit", "E1", "E1_30"), ("stand", "E7", "E7_30")):
        for g in (1, 2, 3):
            mv = "tramping, swinging one leg, shifting on the seat" if posture == "sit" else \
                ("rocking from foot to foot or forward and back"
                 + (", small steps or walking on the spot" if g > 1 else ""))
            add(f"invol_akathisia_{seg}_g{g}", "akathisia", seg, "BARS objective", f"akathisia, {mv}", g,
                f"BARS objective grade {g}", f"{_pct(BARS[g])} of {AKATHISIA_S:g} s; {mv}, changing every 3 to 6 s, "
                f"the fingers fidgeting", sv, L0[key][0],
                SCENE[posture], lambda posture=posture, g=g: akathisia(posture, g),
                note=NOT_SHOWN["observation length"] + ("; " + NOT_SHOWN["leg crossing"] if posture == "sit" else ""))
    # restless legs
    for g in (1, 2, 3, 4):
        add(f"invol_rls_E1_g{g}", "restless_legs", "E1", "restless legs, seated", "restless legs while seated", g,
            f"declared ladder grade {g}", f"{RLS_INDEX[g]} leg movements per hour (0.5 to 10 s each), "
            f"{max(1, int(round(RLS_INDEX[g] * RLS_S / 3600)))} in {RLS_S:g} s: the foot lifted at most 1 s at a time, "
            f"the ankle stretching and flexing, the leg stretched forward and set down, the toes moving",
            sv, L0["E1_60"][0], SCENE["sit"], lambda g=g: restless_legs(g),
            note=NOT_SHOWN["observation length"] + "; " + NOT_SHOWN["leg crossing"])
    # moving toes
    for ag in (2, 3):
        add(f"invol_moving_toes_E1_left_a{ag}", "moving_toes", "E1", "moving toes", "moving toes, left foot", ag,
            f"DIS amplitude {ag} (declared)", f"the toes lifting from the floor by {_pct(toes_level(ag))} of their "
            f"range and back, bursts of 0.5 to 1 s at irregular intervals, about one a second", ("foot_left", "front"),
            L0["E1"][0], SCENE["sit"], lambda ag=ag: moving_toes(ag),
            note=NOT_SHOWN["toes"])
    # mirror movements
    out.append(_spec(clip_id=MIRROR_L0, family="reference", segment="E4a", item="level 0",
                     operator="right finger tapping, no mirror movement", grade=0, grade_text="level 0",
                     params_text="the right hand taps at the cited normal rate, the left hand still",
                     views=("hand_left", "hands"),
                     level0=None, scene="chair", build=lambda: mirror_movements(None), segment_title=MIRROR_TITLE))
    for g in (1, 2, "2b", 3, 4):
        add(f"invol_mirror_E4a_g{g}", "mirror", "E4a", "Woods and Teuber", "mirror movements, left hand", g,
            f"Woods and Teuber grade {str(g)[0]}" + (" (stronger but briefer)" if g == "2b" else ""),
            f"left pinch opening {_pct(MIRROR_RATIO[g])} of the right" + (", first 3 cycles" if g == "2b" else ""),
            ("hand_left", "hands"), MIRROR_L0, "chair", lambda g=g: mirror_movements(g), title=MIRROR_TITLE)
    out += hyper_gap_specs()
    ids = [s.clip_id for s in out]
    if len(set(ids)) != len(ids):
        raise RuntimeError("duplicate clip ids in the involuntary family")
    known = set(ids)
    for s in out:
        if s.level0 is not None and s.level0 not in known:
            raise RuntimeError(f"{s.clip_id}: level 0 {s.level0} is not in the family")
    return out


# ================================================================ catalogue gap wave, lane hyper_gaps (2026-11-02)
# Catalogue rows L02 (peak-dose dystonic dyskinesia), C07 (dystonia in Huntington's disease) and S04 (tardive
# stereotypy): new builders and clip specifications appended to the family (hyper_gap_specs); no existing clip, scene
# or function changes.
#
# Dystonic postures on a choreic background (L02, C07), one engine (twist_spells, dystonic_mix):
# - a posture is a set of channels of one region turned together (TWIST_AXIAL: the neck or the trunk twisting,
#   rotation with lateral bending; TWIST_ARM, HD_ARM: an arm turned and flexed), reached by a slow minimum-jerk rise,
#   held, then released (TWIST: spells of 2 to 4.8 s, about 0.2 to 0.5 Hz, below the 1 to 3 Hz of the choreic
#   fragments: choreic above dystonic, burkhard1999); the distal channels of an arm (forearm, wrist) lag behind on the
#   way out and lead on the way back, so that the hand leaves the thigh before it twists; during the hold the posture
#   writhes slowly (a band-limited modulation of TWIST 'writhe' of its size at 0.2 to 0.5 Hz): no limb is a statue;
#   spells follow one another inside the region's bouts (common.schedule), each arm posture different from the last
#   (the neck and the trunk have one posture each, their sides drawn per spell);
# - the choreic background: the module's flowing fragments (TIMING 'lid' or 'chorea', drift included, the events path
#   of compose_clip) on the same regions, solved at their own level, gated off while a posture is held (multiplied by
#   one minus the posture's envelope): the postures appear and fade on a choreic background and the other regions
#   keep moving. The background is drawn from its own generator before the postures, so that two clips with the
#   same seed share it outside the postures (dystonia grade 0 is the background alone);
# - levels: the posture component solved on its deviation from rest (posture_signals: the largest channel |value|
#   over the channel's ROM, before the writhing) at the grade's level, so that a posture deviates from rest by its
#   amplitude (TWIST 'amp', 0.75 to 1) times the level whichever sides its spells take (the peak-to-peak DIS reading
#   of the choreic background made a posture half the level when spells took both sides); trunk and neck at most
#   POSTURE_AXIAL_CAP (the end a choreic movement at TRUNK_NECK_CAP reaches) at TRUNK_NECK_MARGIN of the room; the
#   background at its own level, peak to peak (region_signals); both summed per channel and bounded by the table
#   (apply_offsets) and the contact guard (seated, the hands on the thighs); each spell's deviation at the middle of
#   its hold read back on the final body (realize_spells: the guard trims an arm that meets the chair or the body).
# Re-measurement (measure_holds): per region, the held runs: knots where the region turns slower than HOLD_SPEED_SHARE
# of its own 95th-percentile speed while moving (a threshold read from the clip: the trunk's choreic drift is slow in
# degrees per second, an arm's fragments fast) and its largest channel deviation from the level 0 exceeds HOLD_SHARE
# of the level, kept when they last HOLD_MIN_S or more (a choreic flow dwells at most 0.45 to 0.5 s, TIMING); their
# count and durations, each run's median deviation from rest, the posture's peak-to-peak over the held knots and the
# rest posture as a share of the ROM, and the share of time held. The inner thighs' contact is read back (standing,
# the level-0 thighs touch), not guarded.
#
# Tardive stereotypy (S04): the same movement repeated in long bouts with a steady rate (STEREO_JITTER), unlike the
# changing movements of akathisia: marching in place standing (the pelvis over the stance foot, the other foot lifted
# vertically and set down on its spot, as the akathisia march) and rubbing the thighs seated (both palms sliding along
# the thighs, the wrists placed by a closed-form arm solve on a target that follows the thigh's surface, the hand
# pitched to lie on it: rub_lift, rub_track, place_wrist). Crossing the legs is not built (the legs would pass through
# each other without a leg-to-leg contact model).
SOURCES.update({
    "burkhard1999": "Burkhard PR, Shale H, Langston JW, Tetrud JW 1999, Quantification of dyskinesia in Parkinson's "
                    "disease, Mov Disord 14: 754-763 (abstract, as the catalogue reads it): dominant frequency 0.25 to "
                    "3.25 Hz, choreic dyskinesia above dystonic",
    "luquin1992_types": "Luquin MR et al. 1992, Mov Disord 7: 117-124, doi 10.1002/mds.870070204 (counts through "
                        "Pandey2017, as the catalogue reads them): of 168 dyskinetic patients, chorea in 113, limb "
                        "dystonic posturing in 63, craniocervical dystonia in 15",
    "uhdrs_dystonia": "UHDRS maximal dystonia (Huntington Study Group 1996, doi 10.1002/mds.870110204), per region "
                      "trunk, each arm, each leg, 0 to 4 with the anchors of the maximal chorea (catalogue row C07): 1 "
                      "slight/intermittent, 2 mild/common or moderate/intermittent, 3 moderate/common, 4 "
                      "marked/prolonged",
    "stacy1993": "Stacy M, Cardoso F, Jankovic J 1993, Neurology 43: 937-941, doi 10.1212/wnl.43.5.937 (abstract, "
                 "as the catalogue reads it): tardive stereotypy in 78 of 100 tardive dyskinesia patients; marching "
                 "in place, leg crossing and repetitive rubbing as the catalogue lists them",
})

TWIST = {"rise": (0.6, 1.2), "hold": (0.8, 2.4), "ret": (0.6, 1.2), "gap": (0.0, 0.3), "amp": (0.75, 1.0),
         "lag_s": (0.2, 0.35), "writhe": 0.08, "writhe_band": (0.2, 0.5), "min_s": 1.8, "bout_s": 6.0}
# [module] a dystonic spell: rise 0.6 to 1.2 s (log-uniform), hold 0.8 to 2.4 s, release 0.6 to 1.2 s, 0 to 0.3 s to
# the next spell (the postures follow one another through the rest posture), each at 0.75 to 1 of the region's level;
# the distal arm channels 0.2 to 0.35 s behind; the writhing 8 percent of the posture at 0.2 to 0.5 Hz; no spell
# shorter than 1.8 s, the last spell of a bout held to the bout's end when less than that would remain (a hold of up to
# about 4.5 s), so that the postures are present for the bouts' share of the time; bouts of about 6 s
TWIST_DISTAL = ("forearm", "wrist_flex", "wrist_dev")
TWIST_AXIAL = {"neck": {"neck_rot": 1.0, "neck_lat": 0.6, "neck_flex": 0.3},
               "trunk": {"trunk_rot": 1.0, "trunk_lat": 0.7, "trunk_flex": 0.35}}
TWIST_AXIAL_SIGNS = {"neck": ("neck_rot", "neck_lat", "neck_flex"), "trunk": ("trunk_rot", "trunk_lat")}
# [module] the twisting of the neck and the trunk: rotation with lateral bending (and some flexion or extension of the
# neck), the sides drawn per spell; seated, the trunk bends forward only (the backrest)
TWIST_ARM = ({"elbow": 0.5, "sh_rot": -0.2, "forearm": -0.8, "wrist_flex": 0.8, "sh_abd": 0.5, "sh_flex": 0.9},
             {"sh_abd": 0.8, "sh_rot": 0.5, "forearm": 0.7, "wrist_flex": -0.6, "elbow": 0.25},
             {"collar": 0.6, "sh_flex": 0.8, "elbow": 0.6, "wrist_flex": 0.7, "forearm": -0.5},
             {"sh_abd": 0.9, "elbow": 0.4, "sh_rot": 0.1, "forearm": -0.8, "wrist_flex": 0.7, "sh_flex": 0.3})
# [module] limb dystonic posturing of peak-dose dyskinesia, seated: (1) the hand lifted forward and turned in (the
# shoulder flexed and a little abducted and internally rotated, the elbow bent, the forearm pronated, the wrist
# flexed); (2) the arm out and turned up (shoulder abducted and externally rotated, forearm supinated, wrist
# extended); (3) the arm raised forward with the shoulder elevated, the wrist flexed and the forearm pronated; (4) the
# arm out to the side with the elbow bent and the hand turned in (shoulder abducted, forearm pronated, wrist flexed).
# The hand turns in by the forearm and the wrist, the shoulder's internal rotation small (the review of wave SH1d,
# major 3: with the palms on the thighs, the internal rotation of 0.8 and 0.4 of the first and the fourth swung the
# forearm into the belly, 3 to 6 cm on the guard's reading through the holds, and the guard held the arm back, the
# left arm's postures reading a median 0.20 of the range at mid-hold at the step-4 level 0.70; at these weights the
# arms' planned postures at step 4 enter the trunk 1.7 percent of the time, at most 1.7 cm, and the other arm never);
# the elbow lifts the hand off the thigh before it turns. Weights relative to each channel's ROM; positive the
# channel's sign
HD_ARM = ({"sh_rot": -1.0, "elbow": 0.5, "forearm": -0.6, "wrist_flex": 0.5},
          {"sh_rot": -0.8, "sh_flex": 0.4, "elbow": 0.6, "wrist_flex": 0.6, "forearm": -0.4},
          {"sh_abd": 0.6, "sh_rot": -0.6, "elbow": 0.4, "forearm": -0.7, "wrist_flex": 0.6},
          {"sh_flex": -0.5, "sh_rot": -0.8, "sh_abd": 0.3, "elbow": 0.15, "forearm": -0.8, "wrist_flex": 0.7})
HD_TRUNK = {"trunk_lat": 1.0, "trunk_rot": 0.6, "trunk_flex": 0.3}
# [module] dystonia in Huntington's disease, standing: the hanging arm held in internal rotation of the shoulder with
# the fist clenched (HD_FIST), the elbow and wrist flexed, the forearm pronated, in front of the body, raised forward,
# out to the side (the third) or held back behind the hip (the fourth: the shoulder extended, the elbow nearly
# straight); the trunk bent sideways and turned (signs drawn per spell). Standing, not seated: with the hands starting
# on the thighs the seated fist went into the thigh where the contact guard could not take it out (4 to 6 cm,
# measured 2026-11-02). A module repertoire: no source read gives the distribution of HD dystonic postures
MIDLINE = {"lid": (0, 2), "hd": (0, 1)}
# [module] the arm postures that bring the hand in front of the body (TWIST_ARM 1 and 3, HD_ARM 1 and 2): the second
# arm takes one of the others (two lateral postures each) while the first arm holds one of these (twist_spells
# 'avoid'; with none allowed, or only the posture it has just held, it waits), so that the forearms never meet in
# front of the body (arm_pair_penetration reads it back); which arm goes first is drawn per clip
POSTURE_AXIAL_CAP = 0.20   # [module] a held neck or trunk posture deviates from rest by at most this share of the ROM:
#                            inside the soft limit TRUNK_NECK_MARGIN leaves on either side of the trunk's rotation (0.22
#                            of its ROM), near the end a choreic movement at TRUNK_NECK_CAP reaches (0.225)
TWIST_BG = 0.4              # [module] L02: the choreic background's level, share of the posture's level
LID_DYST_FINGERS = 0.30     # [module] L02: the fingers curl with an arm posture, share of their flexion range
HD_FIST = 0.60              # [module] C07: the fist clenched with an arm posture, share of the flexion range, at most
#                             FIST_PER_LEVEL times the grade's level (a mild dystonia does not clench a full fist)
FIST_PER_LEVEL = 1.5        # [module] the fingers' share of their range per unit of the posture's level (L02, C07)
HD_CHOREA_BG = 2            # [module] C07: the choreic background at UHDRS maximal chorea grade 2 (mild, common)
HOLD_SPEED_SHARE = 0.2      # [module] measure_holds: a held posture turns slower than this share of the region's
#                             95th-percentile speed while moving (above V_MOVE_DEG_S)
HOLD_SHARE = 0.25           # [module] measure_holds: and deviates by more than this share of the level
HOLD_MIN_S = 0.8            # [module] measure_holds: for at least this long


def twist_spells(t, intervals, rng, postures, sign_keys=(), timing=TWIST, midline=(), avoid=()):
    """(raw {key: (T,)}, envelope (T,) in [0, 1], [spell record], distal envelope (T,), clean {key: (T,)}): dystonic
    spells inside intervals (module comment above): each a posture of `postures` (dicts key -> weight) unlike the one
    before (a single posture repeats with its sides drawn again), at an amplitude drawn from timing 'amp', reached by a
    minimum-jerk rise, held, released; the keys of sign_keys take a side drawn per spell; the distal keys
    (TWIST_DISTAL) lag by timing 'lag_s' on the way out and lead on the way back; the whole raw signal of each key
    modulated by 1 + writhe x band-limited noise (writhe_band); clean: the raw signal before the writhing (the
    posture's nominal size, posture_signals). The envelope is the largest proximal pulse at each knot (1 while a
    posture is held); the distal envelope the largest lagged pulse (the hand twists after the arm has left the thigh
    and untwists before it lands). midline, avoid: a spell overlapping an interval of avoid (s) takes none of the
    postures of index in midline (the other arm holds one); with no other posture left it waits for that interval's
    end."""
    T = len(t)
    raw, env, env_d, out, prev = {}, np.zeros(T), np.zeros(T), [], None
    for a, b in intervals:
        t0 = float(a)
        while b - t0 >= timing["min_s"]:
            r, h, f = (_loguniform(rng, *timing["rise"]), float(rng.uniform(*timing["hold"])),
                       _loguniform(rng, *timing["ret"]))
            L = r + h + f
            if t0 + L > b:
                k = (b - t0) / L
                r, h, f, L = r * k, h * k, f * k, b - t0
            gap = float(rng.uniform(*timing["gap"]))
            if 0.0 < b - (t0 + L + gap) < timing["min_s"]:            # too little left for a spell: hold to the end
                h += b - (t0 + L)
                L = b - t0
            choice = [i for i in range(len(postures)) if i != prev] or [0]
            over = [b_ for a_, b_ in avoid if t0 < b_ and a_ < t0 + L]
            if over:
                choice = [i for i in choice if i not in midline]
                if not choice:                         # nothing allowed but the last posture: wait for the other arm
                    t0 = max(over)
                    continue
            i = int(rng.choice(choice))
            amp = float(rng.uniform(*timing["amp"]))
            lag = min(float(rng.uniform(*timing["lag_s"])), 0.5 * h)
            sg = {k_: float(rng.choice([-1.0, 1.0])) for k_ in sign_keys}
            for key, w in postures[i].items():
                d = lag if any(key == k_ or key.startswith(k_ + "_") for k_ in TWIST_DISTAL) else 0.0
                raw[key] = raw.get(key, np.zeros(T)) + amp * sg.get(key, 1.0) * w * _pulse(t, t0 + d, r, h - 2 * d, f)
            env = np.maximum(env, _pulse(t, t0, r, h, f))
            env_d = np.maximum(env_d, _pulse(t, t0 + lag, r, h - 2 * lag, f))
            out.append({"t0_s": t0, "rise_s": r, "hold_s": h, "return_s": f, "posture": i, "amp": amp, "lag_s": lag,
                        "midline": i in midline, "signs": sg, "held_s": [t0 + r, t0 + r + h]})
            prev = i
            t0 += L + gap
    keys = sorted(raw)
    clean = {k_: v.copy() for k_, v in raw.items()}
    if keys:
        N = band_noise(T, 100.0, timing["writhe_band"], rng, len(keys)).clip(-2.5, 2.5)
        for j, key in enumerate(keys):
            raw[key] = raw[key] * (1.0 + timing["writhe"] * N[:, j])
    return raw, env, out, env_d, clean


def posture_signals(base, chans, raw, clean, level, margin):
    """({channel: (T,) deg}, info): the posture component of a region (region_signals' one-sided variant): raw and
    clean {channel: (T,)} (twist_spells) scaled by one gain so that the largest deviation from rest of the clean
    postures (the largest over channels of max |x| over the channel's ROM, after the soft limits) equals level: each
    posture deviates from rest by its amplitude (TWIST 'amp') times level whatever the sides drawn (a peak-to-peak
    reading made a posture half the level when spells took both sides); the writhing rides on it."""
    rooms = {c: channel_room(base, c) for c in chans}
    roms = {c: channel_rom(c) for c in chans}

    def make(g, src):
        return {c: soft_limit(g * roms[c] * src[c], *rooms[c], margin=margin) for c in chans}

    def amp(g):
        s = make(g, clean)
        return max(float(np.abs(s[c]).max()) / roms[c] for c in chans)
    if level <= 0:
        return {c: np.zeros_like(raw[c]) for c in chans}, {"gain": 0.0, "reachable": True, "level": 0.0}
    g, a, ok = _solve_gain(amp, level)
    return make(g, raw), {"gain": g, "level": float(level), "realized_before_table": float(a), "reachable": bool(ok),
                          "reading": "one-sided: the largest deviation from rest over the ROM, before the writhing",
                          "rooms_deg": {c: [float(x) for x in rooms[c]] for c in chans}, "soft_margin": float(margin),
                          "roms_deg": {c: roms[c] for c in chans}}


def flow_raw(base, chans, intervals, rng, timing, signs=None, level=None, gains=None, margin=SOFT_MARGIN):
    """({channel: (T,)}, n fragments): the raw flowing fragments of compose_clip's events path on chans inside
    intervals (fragment_events, render_fragments weighed by the room, the slow drift), for a caller that gates or
    mixes them before the level is solved; with level (and the gains and margin region_signals will take), at the
    speed caps (capped_fragments)."""
    t = base.times().numpy()
    chans = list(chans)
    tm = TIMING[timing] if isinstance(timing, str) else timing
    ev = fragment_events(intervals, chans, rng, tm, None, signs)
    scale = room_scale(base, chans)
    X = render_fragments(ev, chans, t, scale)
    D = np.zeros_like(X)
    if intervals:
        N = band_noise(base.T, base.rate, (0.3, 1.0), rng, len(chans)) * ramp_envelope(t, intervals, 0.5)[:, None]
        for i, c in enumerate(chans):
            sg = (signs or {}).get(c)
            D[:, i] = DRIFT_SHARE * (np.abs(N[:, i]) * sg if sg else N[:, i])
    if level is not None and tm.get("mode") == "flow":
        X, _ = capped_fragments(base, ev, chans, t, scale, level, gains, margin, D)
    X = X + D
    return {c: X[:, i] for i, c in enumerate(chans)}, len(ev)


def _twist_regions(kind, posture="sit"):
    """[(region, channels, postures {channel: weight}, sign channels, background gains, background signs, margin,
    cap)] of a dystonic clip: kind 'lid' (neck, trunk and both arms, TWIST_AXIAL and TWIST_ARM) or 'hd' (trunk and
    both arms, HD_TRUNK and HD_ARM; the neck carries the background only: the UHDRS rates no neck dystonia)."""
    seated = posture == "sit"
    trunk_signs = {"trunk_flex": 1.0} if seated else None
    out = []
    if kind == "lid":
        out.append(("neck", NECK_CH, [TWIST_AXIAL["neck"]], TWIST_AXIAL_SIGNS["neck"], NECK_GAINS, None,
                    TRUNK_NECK_MARGIN, TRUNK_NECK_CAP))
        out.append(("trunk", TRUNK_CH, [TWIST_AXIAL["trunk"]], TWIST_AXIAL_SIGNS["trunk"], TRUNK_GAINS, trunk_signs,
                    TRUNK_NECK_MARGIN, TRUNK_NECK_CAP))
        arm = TWIST_ARM
    else:
        out.append(("neck", NECK_CH, [], (), NECK_GAINS, None, TRUNK_NECK_MARGIN, TRUNK_NECK_CAP))
        out.append(("trunk", TRUNK_CH, [HD_TRUNK], ("trunk_lat", "trunk_rot"), TRUNK_GAINS, trunk_signs,
                    TRUNK_NECK_MARGIN, TRUNK_NECK_CAP))
        arm = HD_ARM
    for s in ("left", "right"):
        out.append((f"arm_{s}", arm_ch(s), [{f"{k}_{s}": w for k, w in p.items()} for p in arm], (), arm_gains(s),
                    arm_signs(s, posture), SOFT_MARGIN, 1.0))
    return out


def dystonic_mix(base, kind, level, share, bg_level, bg_share, bg_timing, seed, posture="sit", finger_level=0.3):
    """(plan, fingers, record): the region plan of a dystonic clip (module comment above): per region of
    _twist_regions(kind) the choreic background (flow_raw at bg_level, its bouts of bg_share, from a generator of
    seed) and the dystonic spells (twist_spells, bouts of share, from a generator of seed + 1; the posture solved on
    its deviation from rest, posture_signals, at level, the neck and the trunk at most POSTURE_AXIAL_CAP; which arm
    goes first, the other keeping out of its way, drawn from that generator), the background gated off while a
    posture is held, summed per channel into one 'given' entry; per arm the background finger fragments and the
    fingers curling with the arm's postures ('relax' at finger_level)."""
    rng_bg, rng_dy = as_rng(seed), as_rng(int(seed) + 1)
    t = base.times().numpy()
    regs = _twist_regions(kind, posture)
    bg, rec = {}, {}
    for region, chans, _, _, gains, signs, margin, cap in regs:          # every background first (its own stream)
        iv = _iv(base, bg_share, rng_bg)
        tm = arm_timing(bg_timing, posture) if region.startswith("arm_") else bg_timing
        raw, n = flow_raw(base, chans, iv, rng_bg, tm, signs, min(bg_level, cap), gains, margin)
        sig, info = region_signals(base, list(chans), raw, min(bg_level, cap), gains, margin)
        bg[region] = (sig, iv)
        rec[region] = {"background": dict(info, intervals_s=[[float(a), float(b)] for a, b in iv], fragments=n,
                                          timing=bg_timing)}
    order = [r for r in regs if not r[0].startswith("arm_")]
    arms = [r for r in regs if r[0].startswith("arm_")]
    if level > 0 and rng_dy.uniform() < 0.5:                           # which arm goes first
        arms = arms[::-1]
    order += arms
    entries, fing_of, avoid = {}, {}, []
    for region, chans, postures, sign_keys, _, _, margin, cap in order:
        sig_bg, iv_bg = bg[region]
        env = env_d = np.zeros(base.T)
        sig = {c: np.asarray(sig_bg[c], float).copy() for c in chans}
        arm = region.startswith("arm_")
        if postures and level > 0:
            iv = _iv(base, share, rng_dy, bout_s=TWIST["bout_s"])
            raw, env, spells, env_d, clean = twist_spells(t, iv, rng_dy, postures, sign_keys,
                                                          midline=MIDLINE[kind] if arm else (),
                                                          avoid=avoid if arm else ())
            if arm:                                                    # the next arm keeps out of the way
                avoid = avoid + [(x["t0_s"] - 0.3, x["t0_s"] + x["rise_s"] + x["hold_s"] + x["return_s"] + 0.3)
                                 for x in spells if x["midline"]]
            lv = min(level, 1.0 if arm else POSTURE_AXIAL_CAP)
            dy, info = posture_signals(base, sorted(raw), raw, clean, lv, margin)
            for c in chans:
                sig[c] = sig_bg[c] * (1.0 - env) + dy.get(c, 0.0)
            present = sum(s["rise_s"] + s["hold_s"] + s["return_s"] for s in spells)
            held = sum(s["hold_s"] + 0.5 * (s["rise_s"] + s["return_s"]) for s in spells)
            rec[region]["postures"] = dict(info, intervals_s=[[float(a), float(b)] for a, b in iv], spells=spells,
                                           posture_share=float(present / base.duration),
                                           held_half_share=float(held / base.duration),
                                           order=[r[0] for r in order].index(region),
                                           share_rule="posture_share: from each spell's rise to the end of its "
                                                      "release; held_half_share: above half its size")
        entries[region] = _entry(region, tuple(chans), None, iv_bg, kind="given", raw=sig, deg={c: 1.0 for c in chans})
        if arm:
            side = region.split("_")[1]
            fing_of[region] = [{"side": side, "kind": "events", "level": 0.5 * min(bg_level, cap), "intervals": iv_bg,
                                "supported": posture == "sit"}]
            if env_d.any():                                            # the fingers curl with the hand's twist
                on = [(float(t[a]), float(t[b - 1])) for a, b in _runs(env_d > 0)]
                fing_of[region].append({"side": side, "kind": "relax", "level": finger_level, "intervals": on,
                                        "raw": env_d})
    plan = [entries[r[0]] for r in regs]                                # the plan in the fixed region order
    fing = [f for r in regs if r[0] in fing_of for f in fing_of[r[0]]]
    return plan, fing, rec


PAIR_JOINTS = {"left": (18, 20) + tuple(range(22, 37)), "right": (19, 21) + tuple(range(37, 52))}
# the forearm and the hand of each side (the dominant joints of their vertices): the vertex sets of
# arm_pair_penetration
THIGH_PAIR_JOINTS = {"left": (1,), "right": (2,)}   # the thighs: thigh_pair_penetration (the inner thighs pressed)
PAIR_NEAR_M = 0.02          # [module] arm_pair_penetration: a vertex counts against the other arm's surface within this


def _pair_geometry(which="arms"):
    """(vertex ids per side, sparse vertex-face incidence per side, faces) of the forearms and hands (PAIR_JOINTS)
    or of the thighs ('thighs', THIGH_PAIR_JOINTS), on the selected body model's rest mesh (cached per model name)."""
    return _pair_geometry_of(which, _model_name())


@functools.lru_cache(maxsize=4)
def _pair_geometry_of(which, name):
    from scipy.sparse import csr_matrix
    from . import viewer as V
    dom = V.dominant_joint("neutral")
    faces = V.faces()                                    # the selected body model's triangles
    table = PAIR_JOINTS if which == "arms" else THIGH_PAIR_JOINTS
    sets = {s: np.flatnonzero(np.isin(dom, j)) for s, j in table.items()}
    inc = {}
    for s, ids in sets.items():
        pos = {v: i for i, v in enumerate(ids)}
        rows, cols = [], []
        for f, tri in enumerate(faces):
            for v in tri:
                if v in pos:
                    rows.append(pos[v])
                    cols.append(f)
        inc[s] = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(ids), len(faces)))
    return sets, inc, faces


def thigh_pair_penetration(Vk):
    """(n,) m: arm_pair_penetration between the two thighs (the inner thighs pressed into each other; the level-0
    standing thighs touch by a few millimetres)."""
    return arm_pair_penetration(Vk, "thighs")


def arm_pair_penetration(Vk, which="arms"):
    """(n,) m: per knot (vertices Vk (n, V, 3)) the deepest vertex of one forearm and hand behind the surface of
    the other's (its nearest vertex within PAIR_NEAR_M, against that vertex's normal), both ways (a test between the
    two arms, which guard_arms does not make); which='thighs': between the thighs."""
    from scipy.spatial import cKDTree
    sets, inc, faces = _pair_geometry(which)
    out = np.zeros(len(Vk))
    for i, v in enumerate(Vk):
        fn = np.cross(v[faces[:, 1]] - v[faces[:, 0]], v[faces[:, 2]] - v[faces[:, 0]])
        d = 0.0
        for a, b in (("left", "right"), ("right", "left")):
            n = inc[b] @ fn
            n = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)
            pts = v[sets[b]]
            dist, j = cKDTree(pts).query(v[sets[a]])
            inner = np.where(dist < PAIR_NEAR_M, -np.einsum("ij,ij->i", v[sets[a]] - pts[j], n[j]), 0.0)
            d = max(d, float(inner.max()))
        out[i] = d
    return out


SETTLE_RAMP_S = 0.1        # [module] settle_limits: the turn back is ramped in and out over this around the bad knots


def settle_limits(out, inside, base, joints):
    """(Body, knots): out with the rows of joints turned back toward inside (a body without new joint-limit
    violation against base, the base itself for its own rows) at the knots where out adds one beyond
    body.LIMIT_TOL_DEG (the rounding of the guards' chained table bounds reaches a few 1e-6 deg): the first of the
    shares 0.999, 0.99, 0.9, 0.5 and 0 of out's own turn that is clean, applied at the bad knots and ramped back to
    out over SETTLE_RAMP_S on both sides (no one-knot snap); raises when even the share 0 is not clean."""
    from .body import LIMIT_TOL_DEG, components_deg
    lim = joint_limits()
    joints = list(joints)
    b = components_deg(base.rot6d)[:, joints]
    lo, hi = np.minimum(lim.lo[joints][None], b), np.maximum(lim.hi[joints][None], b)

    def bad_knots(body):
        a = components_deg(body.rot6d)[:, joints]
        return ((a < lo - LIMIT_TOL_DEG) | (a > hi + LIMIT_TOL_DEG)).any((1, 2))
    bad = bad_knots(out)
    if not bad.any():
        return out, 0
    R, Ri = out.local_R(), inside.local_R()
    w = _run_profile(out.T, [(a, b_ - 1) for a, b_ in _runs(bad)], [1.0] * len(_runs(bad)),
                     max(1, int(round(SETTLE_RAMP_S * out.rate))), 0.0)
    for share in (0.999, 0.99, 0.9, 0.5, 0.0):
        sc = torch.from_numpy(1.0 - (1.0 - share) * w)
        Rn = R.clone()
        for j in joints:
            Rn[:, j] = _toward(Ri[:, j], R[:, j], sc)
        res = out.with_rotations(Rn, joints)
        if not bad_knots(res).any():
            return res, int(bad.sum())
    raise RuntimeError(f"settle_limits: {int(bad.sum())} knots stay outside the table even at the share 0")


GUARD2_TOL_SHARE = 0.5     # [module] build_guarded: the second guard pass clears to this share of GUARD_TOL_M


def build_guarded(base, plan, rng, legs, fingers, operator, grade, scale, params, sources, posture="sit"):
    """build() of a dystonic clip, then the contact read on the final body at twice GUARD_HZ: against the scene and
    the body ('body_after', guard_arms' measure) and between the two forearms and hands ('pair_after',
    arm_pair_penetration; the postures keep the arms apart, MIDLINE). Where an arm lies more than GUARD_TOL_M inside
    the chair or the body (the guard's three rounds checked at its own knots can end on a touching knot), a second
    pass of guard_arms at GUARD2_TOL_SHARE of its tolerance (a margin for the knots between its checks) against the
    carriage of the result (seated support_arms, the hands at rest on the thighs; standing hang_arms of the level-0
    arms, both from the level 0's arm rows; 'guard2'), and the contact read again; where an arm still touches, the
    hand lifted off (lift_off, 'lift'); the rounding of the guards' chained table bounds settled toward the level 0's
    arms (settle_limits); the inner thighs' contact read back; each spell's deviation at its mid-hold read back
    (realize_spells); then the truth again (finish)."""
    from . import viewer as V
    out, truth = build(base, plan, rng, legs, fingers, operator, grade, scale, params, sources, posture=posture)
    rec = truth["params"]["construction"]
    seated = posture == "sit"

    def deepest(r):
        return max([v["after_cm_max"] for v in r["body_after"].values()] + [r["pair_after"]["after_cm_max"]])
    arms = [j for s_ in ARM for j in ARM[s_]]
    ks = np.unique(np.r_[np.arange(0, base.T, max(1, int(round(base.rate / (2 * GUARD_HZ))))), base.T - 1])

    def contact(body):
        Vk = V.knot_vertices(body, ks)
        ctx = _target_ctx(Vk, heads=True)
        boxes = _boxes(SCENE["sit"] if seated else None, body)
        return ({s: {"after_cm_max": float(100 * _penetration(Vk, s, boxes, ctx).max())} for s in ARM},
                {"after_cm_max": float(100 * arm_pair_penetration(Vk).max()), "tol_cm": 100 * GUARD_TOL_M})
    out, n_settled = settle_limits(out, base, base, arms)
    rec["body_after"], rec["pair_after"] = contact(out)
    rec["stages_cm"] = {"guard": deepest(rec)}
    if any(v["after_cm_max"] > 100 * GUARD_TOL_M + 1e-9 for v in rec["body_after"].values()):
        kept = (out, rec["body_after"], rec["pair_after"])
        if seated:              # from the level 0's arm rows: support_arms keeps the elbow's own twist, a posture's
            ref, _ = support_arms(out.with_rotations(base.local_R(), arms), base)   # pronation that touched
        else:
            ref, _ = hang_arms(out.with_rotations(base.local_R(), arms), base)
        if H.has_hands(ref):
            ref = H.HandBody(ref.rot6d, ref.transl, ref.contact, ref.betas, int(ref.gender), float(ref.rate), None)
        out, rec["guard2"] = guard_arms(out, ref, SCENE["sit"] if seated else None,
                                        tol=GUARD2_TOL_SHARE * GUARD_TOL_M)
        out, n2 = settle_limits(out, base, base, arms)
        n_settled += n2
        rec["body_after"], rec["pair_after"] = contact(out)
        rec["stages_cm"]["guard2"] = deepest(rec)
        if rec["stages_cm"]["guard2"] > rec["stages_cm"]["guard"]:       # a pass that leaves a deeper contact is
            out, rec["body_after"], rec["pair_after"] = kept                 # dropped (the review of wave SH1d)
            rec["guard2_dropped"] = True
    if any(v["after_cm_max"] > 100 * GUARD_TOL_M + 1e-9 for v in rec["body_after"].values()):
        kept, before = (out, rec["body_after"], rec["pair_after"]), deepest(rec)
        out, rec["lift"] = lift_off(out, base, SCENE["sit"] if seated else None, ks, seated)
        out, n3 = settle_limits(out, base, base, arms)
        n_settled += n3
        rec["body_after"], rec["pair_after"] = contact(out)
        rec["stages_cm"]["lift"] = deepest(rec)
        if rec["stages_cm"]["lift"] > before:
            out, rec["body_after"], rec["pair_after"] = kept
            rec["lift_dropped"] = True
    rec["limit_settled_knots"] = n_settled
    th, th0 = (thigh_pair_penetration(V.knot_vertices(b_, ks)) for b_ in (out, base))
    rec["thigh_pair_after"] = {"after_cm_max": float(100 * th.max()), "base_cm_max": float(100 * th0.max()),
                               "over_base_1cm_share": float(((th - th0) > 0.01).mean()),
                               "rule": "read back, not guarded: the inner thighs pressed beyond the level 0's contact"}
    if not any(rp.get("ballistic") for rp in plan):      # the second guard and the lift move the arms again
        out, rec["rate_governor_final"] = govern_rates(out)
    p = {k: v for k, v in truth["params"].items() if k not in ("construction", "hands", "measured")}
    if "mix" in p:
        realize_spells(out, base, p["mix"])
    hands = tuple(fp["side"] for fp in fingers or ())
    truth = finish(base, out, rec, operator, grade, scale, p, sources, None, (0, 1), hands)
    truth["validity"]["thigh_pair_cm"] = {k: rec["thigh_pair_after"][k] for k in ("after_cm_max", "base_cm_max",
                                                                                  "over_base_1cm_share")}
    return out, truth


LIFT_DEG = (2.0, 4.0, 7.0, 10.0, 14.0, 18.0, 24.0, 30.0)   # [module] lift_off: the elbow flexion (seated) or shoulder
#   abduction (standing) tried, in order, at a touch the guards left (24 and 30 since the review of wave SH1d: the
#   severe tardive limb movements rolled a resting hand 3.7 cm into its thigh where 18 deg did not clear it)
LIFT_RAMP_S = 0.15         # [module] lift_off: reached and released by minimum-jerk ramps of this length, or longer
#                            where the lift would turn the joint faster than its speed cap (SPEED_PEAK times the lift
#                            over SPEED_CAP_DEG_S: 0.26 s for 30 deg at the elbow)
LIFT_WIDEN = (0, 2, 4, 8)  # [module] lift_off: the run widened by this many knot steps each way where no lift of
#                            LIFT_DEG clears it (2026-11-04: with the palms resting on the thighs, a grade-3 peak-dose
#                            dystonic hand still pressed 10 mm into its thigh in the ramp after an 18 deg lift)


def _swivel_arm(body, side, deg):
    """body with the elbow of side swung outward by deg (T,) about its shoulder-to-wrist line, the wrist's place and the
    hand's global rotation kept (the shoulder turned about that line, the wrist turned back; outward: the elbow moves
    along the body's lateral axis, as _support_solve's swivel), scaled down where it would leave the table
    (apply_offsets)."""
    c, sh, el, wr = ARM[side]
    J, G = body.fk()
    R = body.local_R()
    lateral = torch.from_numpy(_lateral_frame(body))
    u = J[:, wr] - J[:, sh]
    u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    e = J[:, el] - J[:, sh]
    sg = torch.sign((torch.cross(u, e, dim=-1) * lateral).sum(-1) * SIDE_SIGN[side])
    Q = axis_angle_to_matrix(u * (sg * torch.from_numpy(np.radians(np.asarray(deg, float))))[:, None])
    G_sh = Q @ G[:, sh]
    R_sh = G[:, c].transpose(-1, -2) @ G_sh
    R_wr = (G_sh @ R[:, el]).transpose(-1, -2) @ G[:, wr]
    offs = {sh: _log_deg(R_sh @ R[:, sh].transpose(-1, -2)), wr: _log_deg(R_wr @ R[:, wr].transpose(-1, -2))}
    out, _ = apply_offsets(body, offs)
    return out


def lift_off(out, base, scene, ks, seated=None):
    """(Body, info): the last resort where an arm still lies more than GUARD_TOL_M inside the chair or the body after
    the guard (build_guarded after both its passes; compose_clip on the seated carriage: a hand landing on its thigh
    while the trunk is still twisted, or a flung arm coming back onto it: the guard's outward turn does not lift a
    hand off the top of a thigh): at the knots ks that touch (widened by one step), the smallest of LIFT_DEG of elbow
    flexion seated (where none clears, of shoulder abduction, then of the elbow's outward swing about the
    shoulder-to-wrist line with the hand kept on its thigh, _swivel_arm: the trunk coming onto a forearm whose hand
    rests, the seated choreoathetosis of 2026-11-10; standing, shoulder abduction), ramped over LIFT_RAMP_S or longer
    (the joint's speed cap), that clears them, checked at ks and between them; each run and side on its own. A run
    none clears keeps the candidate that left the least depth where it is shallower than the run's own depth on its
    touching knots, else nothing (until 2026-11-10 the last candidate tried, which could go deeper and made
    compose_clip drop every run's lift). seated: by default, a scene with a seat (SCENE['sit'] or the chair)."""
    from . import viewer as V
    boxes = _boxes(scene, out)
    seated = scene in (SCENE["sit"], "chair") if seated is None else seated
    keys = ("elbow", "sh_abd", "swivel") if seated else ("sh_abd",)
    step = int(ks[1] - ks[0]) if len(ks) > 1 else 1
    info = {}
    for s in ARM:
        Vk = V.knot_vertices(out, ks)
        pen = _penetration(Vk, s, boxes, _target_ctx(Vk, heads=True))
        runs = [(max(int(ks[a]) - step, 0), min(int(ks[b - 1]) + step, out.T - 1)) for a, b in _runs(pen > GUARD_TOL_M)]
        done = []
        for r0, r1 in runs:
            ok = False
            own = float(pen[(ks >= r0) & (ks <= r1)].max())       # the run's own depth on its touching knots
            best = None
            for ch in keys:                              # seated, the elbow's lift first, then an outward turn
                cap = SPEED_CAP_DEG_S["elbow" if ch == "elbow" else "shoulder"]
                for w in LIFT_WIDEN:                     # the run widened where its ramps still touch
                    k0, k1 = max(r0 - w * step, 0), min(r1 + w * step, out.T - 1)
                    for deg in LIFT_DEG:
                        ramp = max(int(round(LIFT_RAMP_S * out.rate)),
                                   int(math.ceil(SPEED_PEAK * deg / cap * out.rate)))
                        kk = np.arange(max(k0 - ramp, 0), min(k1 + ramp, out.T - 1) + 1, max(step // 2, 1))
                        prof = _run_profile(out.T, [(k0, k1)], [deg], ramp, 0.0)
                        if ch == "swivel":
                            cand = _swivel_arm(out, s, prof)
                        else:
                            cand, _ = apply_offsets(out, offsets_of(out, {f"{ch}_{s}": prof}))
                        Vc = V.knot_vertices(cand, kk)
                        depth = float(_penetration(Vc, s, boxes, _target_ctx(Vc, heads=True)).max())
                        if best is None or depth < best[0]:
                            best = (depth, cand, ch, deg, k0, k1)
                        if depth <= GUARD_TOL_M:
                            ok = True
                            break
                    if ok:
                        break
                if ok:
                    break
            depth, cand, ch, deg, k0, k1 = best
            kept = ok or depth < own
            if kept:
                out = cand
            done.append({"knots": [int(k0), int(k1)], "channel": ch, "deg": deg, "clear": ok, "kept": kept,
                         "left_cm": 100 * depth, "own_cm": 100 * own})
        info[s] = {"channels": list(keys), "runs": done}
    return out, info


def realize_spells(out, base, mix):
    """In place on the mix record of dystonic_mix: per region with postures, each spell's 'realized_share' (its
    deviation from rest at the middle of its hold on the final body, after the table and the contact guards: the
    largest over the posture's channels of |channel_values| over the ROM) and the region's 'realized_median' and
    'realized_min' (the guard trims an arm that meets the chair, a thigh or the trunk)."""
    for rec in mix.values():
        po = rec.get("postures")
        if not po or not po.get("spells"):
            continue
        chans = sorted(po["roms_deg"])
        dev = np.max([np.abs(channel_values(out, base, c)) / channel_rom(c) for c in chans], axis=0)
        r = []
        for s in po["spells"]:
            k = min(int(round(0.5 * sum(s["held_s"]) * out.rate)), out.T - 1)
            s["realized_share"] = float(dev[k])
            r.append(dev[k])
        po["realized_median"], po["realized_min"] = float(np.median(r)), float(np.min(r))
        po["realized_rule"] = "the deviation from rest at the middle of each hold, on the final body"


def guard_after_cm(truth):
    """{side: cm}: the deepest arm penetration into the chair or the body left after the guards (read at twice
    GUARD_HZ on the final body), and 'pair': between the two arms."""
    rec = truth["params"]["construction"]
    out = {s: v["after_cm_max"] for s, v in rec["body_after"].items()}
    out["pair"] = rec["pair_after"]["after_cm_max"]
    return out


def lid_dystonic(step=2, seconds=CHOREA_S, seed=SEED + 11):
    """Peak-dose dystonic levodopa-induced dyskinesia, seated (catalogue row L02): sustained twisting postures of the
    neck, the trunk and both arms (TWIST_AXIAL, TWIST_ARM; the fingers curl with an arm posture) at the UDysRS part-3
    step's declared amplitude for the step's share of the time (STEPS), mixed with choreic fragments of TIMING 'lid'
    at TWIST_BG of that level (luquin1992_types: limb dystonic posturing and chorea in the same patients; dystonic
    below choreic in frequency, burkhard1999); the legs carry the choreic part only (leg and foot dystonia: inversion
    not built, a planted foot cannot roll in this module)."""
    base = base_body("sit", seconds)
    level, share = STEPS[step]
    bg = TWIST_BG * level
    plan, fing, rec = dystonic_mix(base, "lid", level, share, bg, share, "lid", seed,
                                   finger_level=min(LID_DYST_FINGERS, FIST_PER_LEVEL * level))
    rng = as_rng(int(seed) + 2)
    legs = [("seated", s, _iv(base, share, rng), "lid", bg) for s in ("left", "right")]
    params = {"symptom": "levodopa-induced dyskinesia, peak-dose dystonic", "posture": "sit", "udysrs_step": step,
              "udysrs_term": UDYSRS_TERMS[step], "type": "dystonic, with choreic movements",
              "level_share_of_rom": level, "share_of_time": share, "background_level_share_of_rom": bg,
              "twist": TWIST, "seeds": [int(seed) + i for i in range(3)],
              "mix": rec, "finger_curl_share": min(LID_DYST_FINGERS, FIST_PER_LEVEL * level),
              "tasks": "UDysRS part 3 rates during communication, drinking, dressing and walking; shown at rest",
              "not_shown": "leg and foot dystonia (inversion not built); the face"}
    return build_guarded(base, plan, rng, legs, fing, "dyskinesia", step,
                         "UDysRS part 3 (declared ladder), type dystonic", params,
                         ("udysrs", "luquin1992_types", "burkhard1999", "manson2000", "dis", "table",
                          "hand_limits", "module"))


def hd_dystonia(grade=2, seconds=CHOREA_S, seed=SEED + 21):
    """Dystonia in Huntington's disease, standing (catalogue row C07): the trunk bent sideways and turned and the
    hanging arms held in internal rotation with the fist clenched (HD_TRUNK, HD_ARM, HD_FIST), at the UHDRS maximal
    dystonia
    grade's declared amplitude and share (the chorea map: UHDRS, AMP, DUR), appearing and fading on a choreic
    background of UHDRS maximal chorea grade HD_CHOREA_BG on every region (the same background for every grade of
    the same seed: grade 0 is the background alone), the pelvis under a bowing trunk (balance_standing); the legs carry
    the choreic background only (the pelvis sway with both feet planted; leg dystonia, foot inversion and knee flexion
    during gait, not built)."""
    base = base_body("stand", seconds)
    a_term, d_term = UHDRS[grade] if grade else ("none", "none")
    level, share = (AMP[a_term], DUR[d_term]) if grade else (0.0, 0.0)
    ba, bd = UHDRS[HD_CHOREA_BG]
    plan, fing, rec = dystonic_mix(base, "hd", level, share, AMP[ba], DUR[bd], "chorea", seed, posture="stand",
                                   finger_level=min(HD_FIST, FIST_PER_LEVEL * level))
    rng = as_rng(int(seed) + 2)
    legs = [("standing", _iv(base, DUR[bd], rng), "chorea", AMP[ba], ("dx", "dz", "roll", "yaw", "tilt", "dip"))]
    params = {"symptom": "dystonia in Huntington's disease", "posture": "stand", "uhdrs_dystonia_grade": grade,
              "amplitude_term": a_term, "duration_term": d_term, "level_share_of_rom": level, "share_of_time": share,
              "regions_dystonic": ["trunk", "arm_left", "arm_right"],
              "background": {"uhdrs_chorea_grade": HD_CHOREA_BG, "amplitude_term": ba, "duration_term": bd,
                             "level_share_of_rom": AMP[ba], "share_of_time": DUR[bd], "timing": TIMING["chorea"]},
              "twist": TWIST, "fist_share": min(HD_FIST, FIST_PER_LEVEL * level),
              "seeds": [int(seed) + i for i in range(3)], "mix": rec,
              "not_shown": "leg dystonia (foot inversion, knee flexion in gait); the face; the UHDRS rates no neck "
                           "dystonia (the neck carries the background chorea)"}
    return build_guarded(base, plan, rng, legs, fing, "dystonia_hd", grade, "UHDRS maximal dystonia (declared map)",
                         params, ("uhdrs_dystonia", "uhdrs", "dis", "hallett1981", "table", "hand_limits", "module"),
                         posture="stand")


def measure_holds(body, base, region, level, speed_share=HOLD_SPEED_SHARE, hold_share=HOLD_SHARE, min_s=HOLD_MIN_S):
    """{runs, durations_s, held_share, held_pp_share, held_channel, held_dev_share, run_dev_shares,
    speed_threshold_deg_s}: the held postures of a region read from the body against its level 0 (module comment
    above): knots where the region's angular speed (the largest over its joints, _speed_deg_s) is under speed_share of
    its 95th percentile over the knots where it moves (above V_MOVE_DEG_S) while the largest channel deviation over its
    ROM exceeds hold_share x level, in runs of min_s or more; held_pp_share: the largest over the region's channels of
    the peak-to-peak of the channel's values over the held knots and 0 (the rest posture), over the channel's ROM;
    held_dev_share: the largest deviation from rest over the held knots (the largest channel |value| over its ROM),
    run_dev_shares: per run its median deviation (the posture's size, the reading of posture_signals)."""
    chans = [c for c in CHANNELS if CHANNELS[c]["region"] == region]
    X = {c: channel_values(body, base, c) for c in chans}
    dev = np.max([np.abs(X[c]) / channel_rom(c) for c in chans], axis=0)
    sp = _speed_deg_s(_region_offsets(body, base, REGIONS[region]), body.rate)
    mv = sp[sp > V_MOVE_DEG_S]
    thr = speed_share * float(np.percentile(mv, 95)) if len(mv) else 0.0
    mask = (sp < thr) & (dev > hold_share * level)
    runs = [(a, b) for a, b in _runs(mask) if (b - a) / body.rate >= min_s]
    held = np.zeros(body.T, bool)
    for a, b in runs:
        held[a:b] = True
    pp = {c: (float(max(X[c][held].max(), 0.0) - min(X[c][held].min(), 0.0)) / channel_rom(c)) if held.any() else 0.0
          for c in chans}
    top = max(pp, key=pp.get)
    return {"runs": len(runs), "durations_s": [float((b - a) / body.rate) for a, b in runs],
            "held_share": float(held.mean()), "held_pp_share": float(pp[top]), "held_channel": top,
            "held_dev_share": float(dev[held].max()) if held.any() else 0.0,
            "run_dev_shares": [float(np.median(dev[a:b])) for a, b in runs], "speed_threshold_deg_s": thr}


TD_STEREO = {"marching": {"rate_hz": 0.9, "bout_s": 8.0},
             "rubbing": {"rate_hz": 1.2, "bout_s": 6.0, "lag_cycles": 0.08}}
# [module] tardive stereotypy: marching in place at 0.9 cycles per second (each foot once per cycle), the weight over
# the stance foot (step_track); rubbing the thighs at 1.2 strokes per second, the right hand 0.08 cycle behind the
# left; long bouts (8 and 6 s); no published rate or size (stacy1993 gives the prevalence)
TD_MARCH_LIFT_M = {2: 0.05, 3: 0.08, 4: 0.11}      # [module] AIMS item 6 declared ladder: the foot's lift
TD_RUB_PP_M = {2: 0.04, 3: 0.07, 4: 0.095}          # [module] AIMS item 5 declared ladder: the stroke along the thigh,
#                                                   spaced up to the reach (at 6, 8.5 and 9.5 cm the steps 3 and 4
#                                                   read alike, the review of wave SH1d);
#                                                   at most 9.5 cm: from the palm resting at mid-thigh (the rest of
#                                                   2026-11-04) the upright seated arm reaches no farther toward the
#                                                   knee: at 11 cm the right shoulder met its table limit, the wrist
#                                                   fell short 19 percent of the time and the palm's centre rose 13 mm
#                                                   off the thigh (at 10 cm 7 percent and 8 mm, at 9.5 cm 0.2 percent
#                                                   and 6 mm); the former rest, at 14 cm, fell up to 2 cm short
RUB_BACK_M = 0.02           # [module] a rubbing stroke goes at most this far back toward the hip (further, the thumb
#                             passes into the groin, and from 3 cm the pelvis under the palm lifted the right palm 1.8
#                             cm off the thigh): a longer stroke is centred toward the knee


def td_marching(step=2, seconds=CHOREA_S, seed=SEED + 31):
    """Tardive stereotypy, marching in place standing (catalogue row S04, AIMS item 6 declared ladder): the same
    alternating step repeated at TD_STEREO's steady rate (STEREO_JITTER), the weight over the stance foot, the other
    foot's heel rising first, the foot lifted by TD_MARCH_LIFT_M[step] and set down on its spot (step_track,
    standing_legs), in long bouts for the AIMS step's declared share (STEPS); the arms hang."""
    P = TD_STEREO["marching"]
    base = base_body("stand", seconds)
    rng = as_rng(seed)
    share = STEPS[step][1]
    iv = _iv(base, share, rng, bout_s=P["bout_s"])
    t = base.times().numpy()
    env = ramp_envelope(t, iv, 0.6)
    ph, am = phase_track(t, P["rate_hz"], rng, STEREO_JITTER)
    lm = TD_MARCH_LIFT_M[step]
    geo = step_geometry(base)
    lift, heel, dx, _ = step_track(ph, am, env, lm, geo, base.rate)

    def legs(b, rng_):
        out, inf = standing_legs(b, dx=dx, lift=lift, heel_deg=heel)
        return out, [dict(inf, lift_m=lm, step_geometry=geo, pelvis_shift_m_max=float(np.abs(dx).max()),
                          heel_deg=HEEL_DEG, stance=stance_report(out, b, omega0=geo["omega0"]))]
    params = {"symptom": "tardive stereotypy, marching in place", "aims_item": 6, "aims_step": step,
              "aims_term": AIMS_TERMS[step], "lift_m": lm, "rate_hz": P["rate_hz"], "jitter": STEREO_JITTER,
              "pelvis_shift_m_max": float(np.abs(dx).max()), "share_of_time": share,
              "intervals_s": [[float(a), float(b)] for a, b in iv],
              "not_shown": "orobuccolingual movements (AIMS items 1 to 4)"}
    return build(base, (), rng, [legs], None, "tardive_stereotypy", step, "AIMS item 6 (declared ladder)", params,
                 ("stacy1993", "aims", "sanger2010", "module", "table", "hof2005"),
                 regions_measured=("trunk", "leg_left", "leg_right"), posture="stand", balance=False)


RUB_PITCH_DEG = np.arange(-20.0, 20.1, 2.5)   # [module] the hand's pitch about the thigh's lateral axis tried per
#                                                 place of the stroke (rub_lift)
RUB_PRESS_M = 0.0015        # [module] rub_lift raises the hand where a palm vertex would lie deeper than this inside
#                             the skin (half of PALM_PRESS_M: the arm's solve and its table move the hand a little
#                             off the target)


@functools.lru_cache(maxsize=4)
def rub_lift(side, grid=(-0.06, 0.15, 43)):
    """(s (n,) m, dy (n,) m, pitch (n,) deg): along the thigh axis of the seated base (hip to knee), the hand's change
    that keeps the palm lying on the thigh as at rest when it moves by s along the axis (module comment above): turned
    by pitch about the thigh's lateral axis through the palm's centre (RUB_PITCH_DEG, the one under which the palm lies
    flattest on the skin: the smallest spread between the centre patch's median signed distance and the palm's
    closest vertex), then raised by dy so that the median signed distance of the palm's centre patch to the skin of
    the body (palm_gap's reading, physio.signed_distance) keeps its base value (PALM_GAP_M), and further where a palm
    vertex would lie more than RUB_PRESS_M inside the skin, by bisection (the strokes
    stay inside -RUB_BACK_M, where the pelvis is not yet under the palm, and the grid's end toward the knee). A
    vertical clearance under the palm (the highest thigh vertex within 2 cm) left the palm 0.5 to 1 cm higher than at
    rest where the thigh curves (2026-11-02); the smallest vertex-to-vertex distance from the hand to the thigh, kept
    until 2026-11-04, let the palm of the seated rest of the clinical review (M1) pass 8 mm into the thigh at
    mid-stroke, and the hand kept level pressed its heel 6 to 8 mm in where the thigh slopes toward the knee."""
    from . import viewer as V
    from .physio import signed_distance
    b = base_body("sit", 0.02)
    v = V.knot_vertices(b, [0])[0]
    ids, patch = palm_vertices()[side]
    palm = v[ids]
    c, u, lat = _rub_frame(b, side)
    c, u, lat = c[0], u[0], lat[0]
    up = np.array([0.0, 1.0, 0.0])

    def turned(th):
        R = axis_angle_to_matrix(torch.from_numpy(lat * math.radians(th))[None])[0].numpy()
        return c + (palm - c) @ R.T

    def gap(P, x, dy):
        sd = signed_distance(v, P + x * u + dy * up)
        return float(np.median(sd[patch])), float(sd.min())
    c0 = gap(palm, 0.0, 0.0)[0]
    s = np.linspace(*grid)
    dy, pitch = np.zeros(len(s)), np.zeros(len(s))
    for i, x in enumerate(s):
        best = None
        for th in RUB_PITCH_DEG:
            P = turned(th)
            lo, hi = -0.05, 0.05
            if gap(P, x, lo)[0] > c0 or gap(P, x, hi)[0] < c0:   # no crossing in the bracket
                continue
            for _ in range(20):
                m = 0.5 * (lo + hi)
                lo, hi = (m, hi) if gap(P, x, m)[0] < c0 else (lo, m)
            d = 0.5 * (lo + hi)
            med, mn = gap(P, x, d)
            if mn < -RUB_PRESS_M:                        # no palm vertex deeper than RUB_PRESS_M: the hand raised
                lo, hi = d, d + 0.03
                for _ in range(20):
                    m = 0.5 * (lo + hi)
                    lo, hi = (m, hi) if gap(P, x, m)[1] < -RUB_PRESS_M else (lo, m)
                d = hi
                med, mn = gap(P, x, d)
            if (best is None or med - mn < best[0] - 1e-9
                    or (abs(med - mn - best[0]) <= 1e-9 and abs(th) < abs(best[2]))):
                best = (med - mn, d, th)
        if best is not None:
            dy[i], pitch[i] = best[1], best[2]
    return s, dy, pitch


RUB_CHECK_S = 0.05          # [module] rub_press_lift reads the palm every this many seconds
RUB_PRESS_SIGMA_S = 0.08    # [module] and spreads the raise over a Gaussian of this sigma (after a running maximum
#                             over twice the sigma each way, so that the raise still covers each pressing knot)


def rub_press_lift(body, side):
    """(T,) m: the raise of the rubbing wrist that keeps the palm's skin outside the thigh's by RUB_PRESS_M, read on
    the built body (palm_gap every RUB_CHECK_S): the arm's solve, its table and the pose-dependent skin move the palm
    off rub_lift's rigid estimate (on the catalogue's step 4 the left palm pressed 3.9 mm in at the stroke's back
    end)."""
    step = max(1, int(round(RUB_CHECK_S * body.rate)))
    ks = np.unique(np.r_[np.arange(0, body.T, step), body.T - 1])
    need = np.maximum(0.0, -palm_gap(body, ks, (side,))[side]["min_m"] - RUB_PRESS_M)
    if need.max() <= 0.0:
        return np.zeros(body.T)
    x = np.interp(np.arange(body.T), ks, need)
    w = int(math.ceil(2 * RUB_PRESS_SIGMA_S * body.rate))
    x = np.array([x[max(i - w, 0):i + w + 1].max() for i in range(body.T)])
    return gauss_smooth(x, RUB_PRESS_SIGMA_S * body.rate)


def _rub_frame(base, side):
    """(palm centre (T, 3), thigh axis u (T, 3), lateral axis (T, 3)) of the seated base: the palm's centre (the wrist
    joint plus PALM_CENTRE_SHARE of the rest offset to the middle knuckle, turned by the wrist's global rotation), the
    unit hip-to-knee axis and the horizontal unit axis square to it (up cross u), the axis the rubbing hand pitches
    about."""
    J, G = (x.numpy() for x in base.fk())
    J52 = H._rest()[0]
    hip, knee = {"left": (1, 4), "right": (2, 5)}[side]
    wr = ARM[side][3]
    off = PALM_CENTRE_SHARE * (J52[H.hand_joint(side, "middle", 0)] - J52[wr])
    c = J[:, wr] + np.einsum("tij,j->ti", G[:, wr], off)
    u = J[:, knee] - J[:, hip]
    u = u / np.linalg.norm(u, axis=-1, keepdims=True)
    lat = np.cross(np.array([0.0, 1.0, 0.0]), u)
    lat = lat / np.linalg.norm(lat, axis=-1, keepdims=True)
    return c, u, lat


def place_wrist(body, side, W, Gw):
    """(Body, info): the arm of side re-solved so that its wrist lies at W (T, 3) with the hand's global rotation Gw
    (T, 3, 3), the shoulder fixed: the elbow flexion about its hinge in closed form, the upper arm aligned on the
    shoulder-to-wrist line and turned so that the elbow stays toward its place in body (support_arms' solve on a given
    target); each joint's change scaled down where it would leave the table (apply_offsets)."""
    from .ik import rmin
    J, G = body.fk()
    R = body.local_R()
    rest = body.rest()
    yax = torch.tensor([0.0, 1.0, 0.0], dtype=F64)
    c, sh, el, wr = ARM[side]
    W = torch.as_tensor(W, dtype=F64)
    S, P = J[:, sh], J[:, el]
    p = rest[el] - rest[sh]
    q = torch.einsum("tij,j->ti", R[:, el], rest[wr] - rest[el])
    A = (p * p).sum() + (q * q).sum(-1) + 2 * p[1] * q[:, 1]
    a = p[0] * q[:, 0] + p[2] * q[:, 2]
    b = p[0] * q[:, 2] - p[2] * q[:, 0]
    rho = torch.sqrt(a * a + b * b)
    delta = torch.atan2(b, a)
    dmax = 0.985 * torch.sqrt(A + 2 * rho)
    d = (W - S).norm(dim=-1)
    clipped = d > dmax
    W = torch.where(clipped[:, None], S + (W - S) * (dmax / d)[:, None], W)
    d = torch.minimum(d, dmax)
    ac = torch.acos(((d * d - A) / (2 * rho)).clamp(-1.0, 1.0))
    ph = torch.stack([delta + ac, delta - ac], -1)
    ph = torch.atan2(torch.sin(ph), torch.cos(ph))
    phi = torch.where(ph[:, 0].abs() <= ph[:, 1].abs(), ph[:, 0], ph[:, 1])
    Ry = axis_angle_to_matrix(yax[None] * phi[:, None])
    v = p[None] + torch.einsum("tij,tj->ti", Ry, q)
    u = (W - S) / d[:, None]
    G1 = rmin(torch.einsum("tij,tj->ti", G[:, sh], v) / v.norm(dim=-1, keepdim=True), u) @ G[:, sh]
    kp = torch.einsum("tij,j->ti", G1, p)
    pp = P - S
    kp = kp - (kp * u).sum(-1, keepdim=True) * u
    pp = pp - (pp * u).sum(-1, keepdim=True) * u
    kp = kp / kp.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    pp = pp / pp.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    gam = torch.atan2((torch.cross(kp, pp, dim=-1) * u).sum(-1), (kp * pp).sum(-1))
    G16 = axis_angle_to_matrix(u * gam[:, None]) @ G1
    R16 = G[:, c].transpose(-1, -2) @ G16
    R18 = Ry @ R[:, el]
    R20 = (G16 @ R18).transpose(-1, -2) @ torch.as_tensor(Gw, dtype=F64)
    offs = {j: _log_deg(Rj @ R[:, j].transpose(-1, -2)) for j, Rj in ((sh, R16), (el, R18), (wr, R20))}
    out, ap = apply_offsets(body, offs)
    return out, {"wrist_reach_clipped_share": float(clipped.double().mean()), "table": ap["scaled_joints"]}


def rub_track(base, side, stroke_m):
    """(W (T, 3), Gw (T, 3, 3)): the wrist target of a palm rubbing its thigh: the base's hand turned by rub_lift's
    pitch about the thigh's lateral axis through the palm's centre, moved by stroke_m (T,) m along the thigh axis (hip
    to knee, positive toward the knee) and raised by rub_lift's correction (_rub_frame)."""
    J, G = base.fk()
    wr = ARM[side][3]
    c, u, lat = _rub_frame(base, side)
    s_grid, dy, pitch = rub_lift(side)
    st = np.asarray(stroke_m, float)
    lift = np.interp(st, s_grid, dy)
    th = np.radians(np.interp(st, s_grid, pitch))
    R = axis_angle_to_matrix(torch.from_numpy(lat * th[:, None]))
    Ct = torch.from_numpy(c)
    W = (Ct + torch.einsum("tij,tj->ti", R, J[:, wr] - Ct) + torch.from_numpy(st[:, None] * u)
         + torch.from_numpy(lift)[:, None] * torch.tensor([0.0, 1.0, 0.0], dtype=F64))
    return W, R @ G[:, wr]


def td_rubbing(step=2, seconds=CHOREA_S, seed=SEED + 41):
    """Tardive stereotypy, rubbing the thighs seated (catalogue row S04, AIMS item 5 declared ladder): both palms
    slide back and forth along the thighs by TD_RUB_PP_M[step] peak to peak at TD_STEREO's steady rate (the right
    hand a little behind), at most RUB_BACK_M back toward the hip, following the thigh's surface (rub_track,
    place_wrist), in long bouts for the AIMS step's
    declared share (STEPS); the hand's clearance from the thigh, the trunk and the chair is read back (the contact
    guard's measure)."""
    from . import viewer as V
    P = TD_STEREO["rubbing"]
    base = base_body("sit", seconds)
    rng = as_rng(seed)
    share = STEPS[step][1]
    iv = _iv(base, share, rng, bout_s=P["bout_s"])
    t = base.times().numpy()
    env = ramp_envelope(t, iv, 0.4)
    ph, am = phase_track(t, P["rate_hz"], rng, STEREO_JITTER)
    pp = TD_RUB_PP_M[step]
    centre = max(0.5 * pp * (1.0 + 2.0 * STEREO_JITTER) - RUB_BACK_M, 0.0)      # the jittered cycles included
    strokes = {s: (centre + 0.5 * pp * am * np.sin(ph - 2 * math.pi * lag)) * env
               for s, lag in (("left", 0.0), ("right", P["lag_cycles"]))}

    def legs(b, rng_):
        info = []
        for s in ("left", "right"):
            W, Gw = rub_track(base, s, strokes[s])
            b1, inf = place_wrist(b, s, W, Gw)
            dy = rub_press_lift(b1, s)                   # read on the body: the hand raised where the palm presses in
            if dy.max() > 0.0:
                up = torch.tensor([0.0, 1.0, 0.0], dtype=F64)
                b1, inf = place_wrist(b, s, W + torch.from_numpy(dy)[:, None] * up, Gw)
            b = b1
            info.append(dict(inf, side=s, press_lift_mm_max=float(1000 * dy.max())))
        return b, info
    params = {"symptom": "tardive stereotypy, rubbing the thighs", "aims_item": 5, "aims_step": step,
              "aims_term": AIMS_TERMS[step], "stroke_m_pp": pp, "rate_hz": P["rate_hz"], "jitter": STEREO_JITTER,
              "lag_cycles": P["lag_cycles"], "share_of_time": share, "stroke_centre_m": centre,
              "intervals_s": [[float(a), float(b)] for a, b in iv],
              "not_shown": "orobuccolingual movements (AIMS items 1 to 4); crossing the legs (no leg-to-leg "
                           "contact model)"}
    out, truth = build(base, (), rng, [legs], None, "tardive_stereotypy", step, "AIMS item 5 (declared ladder)",
                       params, ("stacy1993", "aims", "sanger2010", "module", "table"),
                       regions_measured=("arm_left", "arm_right"), posture=None, guard=False)
    from .physio import signed_distance
    boxes = _boxes(SCENE["sit"], out)
    ks = np.arange(0, out.T, max(1, int(round(out.rate / GUARD_HZ))))
    Vk = V.knot_vertices(out, ks)
    _, self_, _, _, _ = _guard_geometry()
    no_tree = (_EmptyTree(), np.zeros((1, 3)), np.zeros((1, 3)))
    truth["validity"]["hand_clearance"] = {
        s: {"penetration_cm_max": float(100 * max(_penetration(Vk, s, boxes, [no_tree] * len(Vk)).max(),
                                                  max(0.0, -min(float(signed_distance(v, v[self_[s]]).min())
                                                                for v in Vk)))),
            "tol_cm": 100 * GUARD_TOL_M,
            "rule": "the deepest forearm or hand vertex inside a chair box or inside the body's skin (the signed "
                    "distance to the body without its arms, physio.signed_distance: the guard's nearest-vertex "
                    "reading over-read a palm sliding on the thigh by 3 to 6 mm)"} for s in ("left", "right")}
    truth["params"]["stroke_measured_m_pp"] = {
        s: float(np.ptp(((out.joints() - base.joints())[:, ARM[s][3]].numpy() * _thigh_axis(base, s)).sum(-1)))
        for s in ("left", "right")}
    return out, truth


class _EmptyTree:
    """A target with no vertex near any probe (the chair boxes alone through _penetration)."""

    def query(self, q, distance_upper_bound=np.inf):
        return np.full(len(q), np.inf), np.zeros(len(q), dtype=int)


def _thigh_axis(base, side):
    J = base.joints().numpy()
    hip, knee = {"left": (1, 4), "right": (2, 5)}[side]
    d = J[:, knee] - J[:, hip]
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def hyper_gap_specs():
    """[ClipSpec] of the lane hyper_gaps (catalogue rows L02, C07, S04), since 'catalogue_v1'."""
    out = []
    sv = ("front", "left")
    face = "face not shown"
    for st in (2, 3, 4):
        lv, sh = STEPS[st]
        out.append(_spec(
            clip_id=f"invol_lid_dystonic_E1_s{st}", family="dyskinesia", segment="E1", item="UDysRS part 3",
            operator="levodopa-induced dyskinesia, peak-dose dystonic, with choreic movements", grade=st,
            grade_text=f"UDysRS part 3 score {st} ({UDYSRS_TERMS[st]}), type dystonic, declared map; {face}",
            params_text=f"twisting postures of the neck, trunk and arms held about 1 to 4 s, turned from rest by up to "
                        f"{_pct(lv)} of the range of motion (trunk and neck up to {_pct(min(lv, POSTURE_AXIAL_CAP))}), "
                        f"{_pct(sh)} of the time, on choreic movements of {_pct(TWIST_BG * lv)} peak to peak",
            views=sv, level0=L0["E1"][0], scene=SCENE["sit"], build=lambda st=st: lid_dystonic(st),
            note=NOT_SHOWN["UDysRS tasks"] + "; leg and foot dystonia not built (a planted foot cannot roll); "
            + NOT_SHOWN["seat"]))
    for g in (0, 2, 3, 4):
        a, d = UHDRS[g] if g else ("none", "none")
        ba, bd = UHDRS[HD_CHOREA_BG]
        out.append(_spec(
            clip_id=f"invol_hd_dystonia_E7_g{g}", family="chorea", segment="E7", item="UHDRS maximal dystonia",
            operator="dystonia in Huntington's disease, trunk and arms, on chorea, standing", grade=g,
            grade_text=(f"UHDRS maximal dystonia grade {g} ({a}/{d}), trunk and arms" if g else
                        "UHDRS maximal dystonia 0 (the choreic background alone)")
            + f"; chorea grade {HD_CHOREA_BG} throughout; {face}",
            params_text=(f"the arm turned in with the fist clenched, the trunk bent sideways and turned, from rest "
                         f"by up to {_pct(AMP[a])} of the range of motion (trunk up to "
                         f"{_pct(min(AMP[a], POSTURE_AXIAL_CAP))}), {_pct(DUR[d])} of the time; " if g else "")
            + f"chorea {_pct(AMP[ba])} of the range of motion peak to peak, {_pct(DUR[bd])} of the time, feet planted",
            views=sv, level0=L0["E7"][0], scene=None, build=lambda g=g: hd_dystonia(g),
            note="leg dystonia (foot inversion, knee flexion in gait) not built; " + NOT_SHOWN["standing leg chorea"]))
    for st in (2, 3, 4):
        out.append(_spec(
            clip_id=f"invol_td_stereotypy_E7_marching_s{st}", family="stereotypy", segment="E7", item="AIMS item 6",
            operator="tardive stereotypy, marching in place", grade=st,
            grade_text=f"AIMS item 6 score {st} ({AIMS_TERMS[st]}), declared map; {face}",
            params_text=f"the same step repeated at {TD_STEREO['marching']['rate_hz']:g} Hz, the foot lifted "
                        f"{100 * TD_MARCH_LIFT_M[st]:.0f} cm and set down on its spot, the weight shifted onto the "
                        f"standing foot before each step and the heel rising first, "
                        f"{_pct(STEPS[st][1])} of the time",
            views=("front", "left"), level0=L0["E7"][0], scene=None, build=lambda st=st: td_marching(st),
            note=NOT_SHOWN["face"]))
        out.append(_spec(
            clip_id=f"invol_td_stereotypy_E1_rubbing_s{st}", family="stereotypy", segment="E1", item="AIMS item 5",
            operator="tardive stereotypy, rubbing the thighs", grade=st,
            grade_text=f"AIMS item 5 score {st} ({AIMS_TERMS[st]}), declared map; {face}",
            params_text=f"both palms sliding along the thighs, {100 * TD_RUB_PP_M[st]:g} cm strokes at "
                        f"{TD_STEREO['rubbing']['rate_hz']:g} Hz, {_pct(STEPS[st][1])} of the time",
            views=("front", "left", "hands"), level0=L0["E1"][0], scene=SCENE["sit"], build=lambda st=st: td_rubbing(st),
            note=NOT_SHOWN["face"] + "; crossing the legs not built (no leg-to-leg contact model); "
            + NOT_SHOWN["seat"]))
    return out
