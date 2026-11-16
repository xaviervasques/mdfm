"""A parametric gait generator of the synthetic patient (task CAT-GAIT of the symptom catalogue): a healthy adult
walk calibrated to published normative values, every parameter of which can be changed, per side where gait has
sides, and over time where a symptom evolves (festination, start hesitation), with exact foot contacts.

The procedural walker of base.py is a test base (flat feet, no heel-toe rocker, no pelvic rotation, stiff legs);
this module is new code that imports from base.py, body.py, contact.py and ik.py and changes none of them.

Construction (generate), on the 100 Hz clock, floor at y = 0 (the lowest foot joint of a flat foot), y up, the
heading on +z at the start (x the subject's left):
- Phase: one gait phase phi(t) (strides), the left heel strike at its integers and the right one at hs_offset past
  them; its rate is the cadence (a scalar or a function of time), ramped in at gait initiation, eased out at the
  stop, jittered step by step (step_time_cv). A gait amplitude a(t) in [0, 1] (0 when standing) scales the rocker,
  the pelvic and trunk oscillations and the arm swing, so that the body stands with flat feet before the first step
  and after the last.
- Progression: the walking line c(t) advances by two step lengths per stride (step_m, scalar or function of time),
  along the heading psi(t) (turns: raised-cosine yaw-rate pulses, the progression slowed during a turn by its
  speed share, so a share near 0 turns in place); lateral wander (wander_m) adds a smooth random offset, eased in
  after gait initiation and held after the stop.
- Footprints: each stance's flat foot is placed so that its ankle lies under the walking line at place_share of the
  stride after its heel strike, half the step width to its side, turned out by the foot progression angle, with a
  per-step jitter of length and width (step_length_cv, step_width_sd_m); a turning stance places the inside foot
  wider (turn_widen_m_rad per rad of heading change over the stance); the standing footprints (before the first
  step, and the last two at the stop) lie side by side at stand_width_m.
- Stance foot (heel-toe rocker): the flat foot pitched by phi_s(u) and rolled without slipping on the convex hull of
  its sole cloud in its sagittal plane (contact.roll: the contact point is still): from the contact pitch (heel
  first, toes up; or flat, or toes first) down to the mid pitch (0, flat; above 0 on the toes) at flat_share, held
  until heel_off_share, then rolled onto the toes to the lift-off pitch at toe-off; a short or turning step (its
  progression slowed to the turn's speed share) rolls less. During a turn the stance foot may pivot about the
  vertical through a touching sole point (pivot_share of the heading change since its placement): the heel while the
  heel rocker rolls, then from the forefoot's landing to toe-off the sole point the push-off rolls on most
  (_push_off_pivot), one point still through the forefoot contact. A stance foot may
  stand inverted on its lateral border (stance_inversion_deg).
- Pelvis: on the walking line plus a lateral sway toward the stance foot; its yaw is the heading plus the pelvic
  rotation (the swing side forward); its obliquity drops the swing side (per stance side, negative for a hip hike),
  and the longer leg's side hikes while that leg swings (leg_length_comp: the neutral right leg reaches about 3 mm
  lower); its tilt oscillates twice per stride, about a forward tilt (lordosis_deg) that the lumbar spine takes back.
  Its height follows a target knee flexion of the stance leg that landed last (loading flexion, then the midstance
  value; knee_buckles add sudden dips, anchored to a stance of their leg), capped so that every stance leg reaches
  its ankle, smoothed by a moving minimum then a moving mean so that it never rises above the cap.
- Legs: each stance knot by two-bone inverse kinematics (ik.hinge_leg: a pure knee hinge, the knee turned toward
  the foot's own direction), the ankle the rotation from the shank to the rolled foot; each swing in joint space,
  from the toe-off pose to the next heel-strike pose with matched velocities (cubic Hermite through via-points): the
  hip flexion overshoots its heel-strike value by hip_overshoot_deg, the knee peaks at knee_swing_deg at
  knee_peak_share of the swing, the ankle passes through swing_ankle_deg (dorsiflexion positive, plantarflexion
  negative: a foot drop), an abduction bump circumducts the leg (negative: an adduction bump, scissoring), over
  the whole swing or from swing_late_share of it on (scissoring: once the swing knee is level with the stance
  knee), a hip rotation bump turns the swing knee's front inward (swing_rotation_deg), an inversion bump inverts the
  foot; a lift (hip, knee and ankle flexion in the ratio lift_synergy) shaped as a bump that vanishes at both ends
  is solved per swing (bisection) so that the lowest sole point over the middle of the swing lies clearance_m
  above the floor (the minimum foot clearance, asked where the bump is at CLEAR_CORE of its peak or more, fading
  with its shape, so that a foot landing flat or toes first does not drive the lift to its cap), unless clearance_m
  is None; where the sole still dips below the floor, a lift localized at the dip (_dip_bump) raises it, and the
  mid-swing lift is solved again with it in place (the two do not stack). Every stance knot is an exact contact:
  a touching sole point does not move (stance_slip measures it: a rolling or pivoting foot keeps one still). The
  stances of both legs are solved before the swings, so that a swing sees the other leg.
- Leg-to-leg guard (leg_gap_m, LEG_GAP_M by default, None turns it off): each swing is measured against the other
  leg on the skinned mesh
  (legcontact.gap_from_pose, solved on every LEG_GUARD_SUB-th knot, then checked and solved again on every knot);
  where a knee, shank or foot would come closer than leg_gap_m to the other leg, or the inner thighs press deeper
  than legcontact.SOFT_TOL_M less LEG_SOFT_MARGIN_M, the swing hip abducts by a bump centred where the contact is
  deepest (_dip_bump of width LEG_GUARD_WIDTH), the smallest that keeps the gap (regula falsi, up to
  LEG_GUARD_BUMPS bumps, LEG_GUARD_MAX_DEG at most together), the lift solved again with it; info['leg_guard']
  records each swing (the shares of the swing where its bumps are centred under 'bump_at_share'). The guard is a
  swing operator: two legs in double support are not moved (a turn widens its base by turn_base_m instead). Every
  gait has it since the leg-to-leg calibration of 2026-11-03 (STEP_WIDTH_M, TURN_BASE_M, LEG_GAP_M); the scissoring
  clips ask a gap of their own (spastic_diplegic, SCISSOR_GAP_M). With the guard on, a double-support guard follows
  the construction (_ds_contacts): where the two planted legs pass into each other beyond the legcontact check (a
  short or narrow step, the knee of the trailing leg against the thigh of the leg that landed), the foot that landed
  is set down wider (a footprint offset, DS_PUSH_GAIN times the gap's lack plus DS_MARGIN_M) and the walk is
  generated again with it, up to DS_PUSH_ROUNDS times; info['ds_push'] records each push. With leg overlays (the
  guard measures a swing before them), a swing whose final legs pass into each other beyond the check gets an
  abduction bump of its own the same way (_overlay_swing_contacts, swing_extra_abd, info['overlay_guard']).
- Stance guard (open-body port, wave 5; STANCE_GUARD_*): the swing legs, the trunk, head and arms are clamped into
  the anatomical joint-limit table, a stance leg cannot be without moving its foot; where a stance hip's extension or
  a stance ankle's dorsiflexion passes the table, the stance foot's heel rises by a lift added to its rocker pitch
  (where a stance ankle's off-axis components pass it, its pitch falls), rolled on the sole's hull as the rocker is, and
  the walk is generated again with it (info['stance_lift']); a body inside the table is generated once, unchanged.
  A swing whose knee would turn faster than KNEE_SWING_RATE_DEG_S takes wider dip and landing bumps (_Swing.solve,
  DIP_WIDEN); a planted foot at a held pitch turns about the push-off's pivot only where that point touches the floor
  (PIVOT_TOE_NEAR_M); the read-back's stance reads the generator's touch (STANCE_READ_TOL_M) (wave 5 second follow-up).
- Trunk, head and arms: the trunk leans (trunk_lean_deg) and counter-rotates against the pelvis
  (trunk_counter_rotation), keeps its line against the pelvic obliquity, sways laterally toward the stance side
  (trunk_lateral_deg, the Duchenne lean when large); the head is held toward level and toward the heading; the arms
  hang (arm_abduction_deg from vertical) and swing in counterphase with the legs (arm_swing_deg about the mean
  arm_flexion_deg), the elbows slightly flexed (elbow_deg) and flexing as the arm swings forward (elbow_swing_deg);
  the arms abduct with the knee buckles (buckle_arm_deg) and take back a share of the trunk's lateral lean
  (arm_lean_follow); an arm posture (arm_posture, a side's held shoulder and elbow and wrist rotations, e.g. the
  Wernicke-Mann arm) replaces the swing of that arm.
- Overlays: after the construction, each overlay(t, info) returns {joint: (rotations (T, 3, 3), frame)} composed
  with that joint's local rotation (body.compose: 'child' or 'parent'): the hook that layers other operators (the
  choreic swing-leg fragments of this module, and operators built elsewhere). A leg overlay that is not zero in
  stance moves the stance foot; info['swing_weight'] (T, 2), zero in stance, lets a leg overlay act in swing only.
  The catalogue's course clips may finish the body with a post step (the Wernicke-Mann hand, the upper-body chorea
  of the involuntary family's engine).
- Gap wave (2026-11-02; every default leaves the former clips bit for bit unchanged): the rocker pitches, the swing
  hip overshoot and ankle and the arm abduction may also be functions of time (read per stance at its heel strike,
  per swing at its toe-off, or per knot), so that a gait can change within a clip (functional fluctuations, the
  shuffling of a freezing episode); the clearance bump's centre and width are parameters (clear_at, clear_width); a
  trunk flexion locked to the gait cycle (pump_deg, the dromedary gait) is shared by the pelvis (before the stance
  IK) and the spine, the head held level and the arms hanging against it; a dragged side (drag_sides) solves its
  swing lift per knot so that the lowest sole point slides on the floor (a deliberate skate, recorded per swing);
  course() takes generate's path hook (veer_path: a heading drift per metre walked), course_tv places the turn and
  the stop for parameters that change with time. gait_balance checks the mediolateral margin of stability of the
  extrapolated centre of mass and the standing support; path_drift, trunk_cycle, window_read, leg_frequency,
  turn_rate and drag_read read the new signs back from the body.

Re-measurement (measure_gait): from the body alone (the joints and the rigid sole cloud of contact.py), stance runs
as the runs where a foot's lowest sole point lies within FLOOR_TOL_M of the floor (runs shorter than 80 ms, a toe
scuffing the floor in early swing, ignored); heel strike and toe-off at their ends, the stance share and the double
support read on STANCE_READ_TOL_M (the foot touching);
footprints, step length and width, cadence, speed, stance share, knee flexion in loading and in swing, minimum foot
clearance, foot pitch at contact, arm swing, elbow flexion, trunk lean, pelvic excursions and step-to-step
variability, over the steady steps (no turn, not the first or last two). signed_step_widths reads each footprint's
signed distance from the other foot's line of progression (negative: the feet crossing), midline_offsets half of it;
scissor_record reads them, the knees' distance in mid swing and the leg-to-leg gap back.

Defaults (NORMAL) are a healthy adult at a comfortable speed, the sources in SOURCES; healthy_ranges() is the
range table the healthy walk is checked against (tests/test_synth_gait.py).
"""
import math
from dataclasses import asdict, dataclass, field, replace

import numpy as np
import torch

from ..body.rotations import axis_angle_to_matrix, matrix_to_rot6d
from ..body.joints import N_BODY
from . import contact as CT
from . import legcontact as LCN
from .jerks import STIMULUS_MARK_S
from .body import F64, NEUTRAL, RATE, Body, contact_labels, rest_joints

SIDES = ("left", "right")
SIGN = {"left": 1.0, "right": -1.0}          # the subject's left is +x
LEG = {"left": (1, 4, 7, 10), "right": (2, 5, 8, 11)}
ARM = {"left": (13, 16, 18, 20), "right": (14, 17, 19, 21)}
SPINE = (3, 6, 9)
SPINE_SHARE = (0.40, 0.35, 0.25)
BUCKLE_ONSET = 0.0                           # a sudden knee buckle starts at this share of its stance: at heel
#                                              strike, as the weight comes onto the leg [module; later onsets
#                                              dropped the pelvis under the other leg's swing knee peak, which
#                                              then flexed to about 90 deg to clear]
LORDOSIS_SHARE = (0.6, 0.4, 0.0)             # the lordosis's extension over the spine joints (lumbar) [module]
ARM_LEAN_LAG_S = 0.10                        # the arms' lag behind the trunk's lateral lean [module]
TRUNK_SWAY_SOFT_DEG = 12.0                   # [module] the random lateral sway (trunk_sway_deg) soft-limited to about
#                                              this (tanh): the bound the clinical review of 2026-11-03 asked, against
#                                              27 deg of spine bend in the astasia-abasia walk
TRUNK_SWAY_PELVIS_SHARE = 0.18               # [module] share of that sway carried by the whole body over the ankles:
#                                              the pelvis displaced sideways by this share of the trunk's (pelvis to
#                                              neck) lateral travel, the spine bending by the rest (0.4 and 0.25 rolled
#                                              the astasia-abasia walk's wide-based stance ankle 1.5 and 0.4 deg beyond
#                                              the joint-limit table)
HEAD_RIGHTING = 0.8                          # [module] share of the trunk's lateral lean the neck and head take back
#                                              (the head kept near level on roll)
FLOOR_TOL_M = 0.003                          # a foot stands when its lowest sole point is within this of the floor
#                                              (the continuity check, the drag read, the footprints of the step widths)
STANCE_READ_TOL_M = 0.001                    # the read-back's stance share and double support (measure_gait; its
#                                              step events keep FLOOR_TOL_M): the lowest sole point within the
#                                              generator's own touch tolerance (PIVOT_NEAR_M). Wave 5 second follow-up:
#                                              FLOOR_TOL_M was set on SMPL+H's sole; after a toe-off Anny's toe rises by
#                                              0.68 to 3.76 mm over the first seven knots, so at 3 mm the healthy walk's
#                                              read stance ran to 0.686 against its generated 0.605 to 0.622, at 1 mm
#                                              it reads 0.61 (scripts/openbody_wave5_checks.py second, stance_read).
#                                              Anny's toe rest clearance (its toe set's lowest point 0.15 mm over the
#                                              sole's lowest, 0 to 0.08 mm on SMPL+H) is not the cause
EARLY_TOL_M = 0.001                          # the sole may not dip more than this below the floor in a swing
GAIN_MAX_DEG = 15.0                          # largest clearance lift (hip d, knee 2 d, ankle d)
ROCKER_REF_M = 0.30                          # steps shorter than this scale the heel strike and push-off down
CLEAR_AT = 0.58                              # centre of the clearance bump in the swing (the minimum foot
#                                              clearance lies in mid swing) [module]
CLEAR_CORE = 0.5                             # share of the clearance bump's peak where the clearance is asked
#                                              [module]
HIP_EARLY = 0.6                              # share of the swing hip flexion reached at the knee's peak [module,
#                                              the normative hip curve: about 20 of 30 deg by 73 percent]
LEG_GUARD_MAX_DEG = 20.0                     # [module] largest swing hip abduction the leg-to-leg guard adds (its
#                                              bumps of a swing together)
LEG_GUARD_STEPS = 4                          # regula falsi steps of each guard bump
LEG_GUARD_SUB = 3                            # the guard solves on every third knot of a swing (30 ms at 100 Hz)
LEG_GUARD_BUMPS = 3                          # guard bumps of a swing at most
LEG_GUARD_WIDTH = 0.35                       # [module] the guard bump's width (share of the swing, _dip_bump): a
#                                              width of 0.15 (about 70 ms) threw the swing leg outward in a kick
LEG_GUARD_REACH = 0.5                        # a guard bump is solved on the knots where it reaches this share
LEG_GUARD_EDGE = 0.25                        # [module] a guard bump rises from toe-off and falls to heel strike over at
#                                              least this share of the swing (_dip_bump's edge): over 0.08, a 9 deg
#                                              bump right after toe-off threw the dystonic foot 6 cm sideways in one
#                                              frame (2026-11-03 review)
LEG_GUARD_RAMP_S = 0.15                      # [module] and over at least this time: a 20 deg bump falling to heel
#                                              strike over a quarter of a 0.45 s swing kicked the landing foot at 166
#                                              m/s2 in the Trendelenburg turn (the review's bound 100 m/s2); the bumps
#                                              of a swing together are scaled down to LEG_GUARD_MAX_DEG (no longer
#                                              clipped, which flattened their top and steepened their fall)
LEG_GUARD_END_SHARE = 0.1                    # [module] a contact this close to toe-off or heel strike (share of the
#                                              swing) gets no guard bump: the stance it lies by is set down wider and
#                                              the walk generated again (land_push in generate; the last round of the
#                                              double-support guard bumps it as before). Bumped, a contact at 0.95 of
#                                              the swing in the level 0's turn, where a bump with LEG_GUARD_EDGE fades,
#                                              took 12.6 deg of swing hip abduction (wave SH1d)
LEG_SOFT_MARGIN_M = 0.005                    # [module] the guard keeps the inner thighs this far inside
#                                              legcontact.SOFT_TOL_M
# The healthy walk's base on the program's body (2026-11-03, the leg-to-leg calibration): the neutral mesh's thighs
# touch at rest below the crotch and its knees are wide, so that on the former 0.09 m base (FORMER_BASE_M) the healthy
# walk passed its knees into each other in straight walking and its shanks in the turn, beyond the legcontact check
# (tests/test_synth_legcontact.py and tests/test_synth_gait.py keep it as a positive control). Thicker thighs widen
# the step (westlake2013: the physical constraint of the thigh segments; no value).
# STEP_WIDTH_M [module: the narrowest base on which the neutral mesh's level-0 course holds the leg check, the swing
# guard adding less than 2 deg (measured on 2026-11-03: the narrower bases tried, 0.09 and 0.11 m, left contacts
# beyond the check in the turn or needed larger guard bumps; tests/test_synth_gait.py,
# test_the_healthy_walks_keep_their_legs_apart). It lies inside collins2013's spread at the forefoot markers (0.168,
# SD 0.044 m; a forefoot distance, which the toe-out widens over the ankles'), and above pirker2017's 8 to 10 cm of
# the elderly, who walk about 40 percent wider, which implies a narrower young base: no source gives this value]
STEP_WIDTH_M = 0.12
FORMER_BASE_M = 0.09                         # the healthy base before 2026-11-03 (pirker2017 8 to 10 cm), against
#                                              which the module widths of the gaits below were set


def _wider(w):
    """A module step width set against FORMER_BASE_M, kept at its distance from the healthy base: w plus
    STEP_WIDTH_M less FORMER_BASE_M (a wide-based gait stays as much wider than the level 0 as it was)."""
    return round(w + STEP_WIDTH_M - FORMER_BASE_M, 3)


TURN_BASE_M = 0.10                           # [module] the base widened by this at a turn's middle (turn_base_m): on
#                                              the walking base the healthy 180 deg turn crossed its shanks in double
#                                              support, which the swing guard cannot move (the same calibration)
DS_PUSH_ROUNDS = 3                           # the double-support guard's rounds at most (each one a new generate)
DS_PUSH_GAIN = 2.0                           # [module] the landing ankle set down wider by this times the knee's
#                                              lack of gap (the knee lies about halfway between hip and ankle)
DS_MARGIN_M = 0.005                          # [module] added to each push
LEG_GAP_M = 0.0                              # [module] the guard's hard gap by default: no swing passes a knee,
#                                              shank or foot into the other leg (the scissoring clips ask
#                                              SCISSOR_GAP_M)

SOURCES = {
    "perry2010": "Perry J, Burnfield JM. Gait Analysis: Normal and Pathological Function, 2nd ed. 2010 (textbook "
                 "values as commonly reported, not re-read here): free walking 82 m/min (1.37 m/s), cadence 113 "
                 "steps/min, stride 1.41 m; stance about 60 percent of the cycle; heel, ankle and forefoot rockers",
    "bohannon2011": "Bohannon RW, Williams Andrews A. Physiotherapy 2011; 97: 182-189, doi 10.1016/j.physio.2010."
                    "12.004 (abstract; docs/proposals/SYNTH_PATIENT_DRAFT.md section 7): comfortable speed 0.943 to "
                    "1.434 m/s across age and sex strata",
    "pirker2017": "Pirker W, Katzenschlager R. Gait disorders in adults and the elderly: a clinical guide. Wien Klin "
                  "Wochenschr 2017; 129: 81-95, doi 10.1007/s00508-016-1096-4 (PMC5318488, full text read "
                  "2026-11-02): preferred speed about 1.4 m/s to age 59; cadence 115 to 120 steps/min in young "
                  "adults; step width about 8 cm (elderly women) and 10 cm (elderly men), the elderly about 40 "
                  "percent wider; stance about 60 percent; and the clinical description of every gait type of "
                  "this module (parkinsonian, cerebellar, hemiparetic, spastic paraparetic, steppage, waddling, "
                  "antalgic, frontal and NPH, cautious, dystonic, choreic, functional)",
    "knee_norm": "Normative table of 20 healthy adults aged 27.9 (PMC4016931, Table 3, as returned by a web search "
                 "on 2026-11-02, the article not read): knee peak flexion in swing 64.9 (SD 5.6) deg, in loading "
                 "about 16 (SD 5.9) deg",
    "po_norm": "Pelvic obliquity in healthy adults, 0 to 5.6 deg, median 2.0 (J Exp Orthop 2023, doi 10.1186/"
               "s40634-023-00613-z, as returned by a web search on 2026-11-02, not read); pelvic rotation about 10 "
               "to 13 deg range in a gait table of the same search",
    "elbow2023": "The elbow is the load-bearing joint during arm swing (Korean study of 15 adults, 2023, "
                 "koreascience JAKO202326857029300, abstract as returned by a web search on 2026-11-02): dominant "
                 "elbow flexion-extension range 29.7 (SD 10.2) deg during gait",
    "zanardi2021": "Zanardi APJ et al. Gait parameters of Parkinson's disease compared with healthy controls: a "
                   "systematic review and meta-analysis. Sci Rep 2021; 11: 752, doi 10.1038/s41598-020-80768-2 "
                   "(full text read 2026-11-02; 72 studies, people with PD on medication): speed 0.17 m/s slower, "
                   "stride 0.16 m shorter, cadence higher by about 1.9 steps/min (the text; the sign of its SMD is "
                   "inconsistent), double support 1.8 percent longer, swing 1.8 percent shorter, step width not "
                   "different, sagittal hip range 5.3 deg smaller",
    "lewek2010": "Lewek MD et al. Arm swing magnitude and asymmetry during gait in the early stages of Parkinson's "
                 "disease. Gait Posture 2010; 31: 256-260 (abstract): arm swing asymmetry 13.9 (SD 7.9) percent in "
                 "early PD against 5.1 (SD 4.0) in controls; magnitude not different in early PD",
    "mds_updrs": "Goetz CG et al. 2008, doi 10.1002/mds.22340; item 3.10 gait (0 normal; 1 independent with minor "
                 "impairment; 2 independent with substantial impairment; 3 needs an assistance device; 4 cannot "
                 "walk or only with another person's assistance) and 3.12 postural stability (recovery steps "
                 "after a pull: 3 to 5 steps grade 1, more than 5 grade 2), paraphrased, never copied",
    "sara": "Schmitz-Huebsch T et al. 2006, Neurology 66: 1717-1720 (SARA); gait item 0 to 8 as reproduced by the "
            "SRALab and PhenX protocol pages (web search 2026-11-02): 0 normal, tandem with at most one misstep; "
            "1 slight, visible only in 10 consecutive tandem steps; 2 clearly abnormal, tandem beyond 10 steps "
            "impossible; 3 considerable staggering, difficulty in the half-turn, no support; 4 marked staggering, "
            "intermittent wall support; 5 to 8 need a stick or a person, or cannot walk",
    "ataxia_var": "Step and stride length variability rise with SARA in degenerative cerebellar ataxia (Ilg, "
                  "Schniepp and others, as returned by a web search on 2026-11-02; Serrao et al. Cerebellum 2012, "
                  "doi 10.1007/s12311-011-0296-8, abstract: markedly higher intra-subject variability of almost "
                  "every gait parameter); no value per SARA grade read",
    "lempert1991": "Lempert T, Brandt T, Dieterich M, Huppert D. How to identify psychogenic disorders of stance "
                   "and gait: a video study in 37 patients. J Neurol 1991; 238: 140-146 (as recalled, not re-read): "
                   "excessive slowness, hesitation, the 'walking on ice' pattern, sudden knee buckling usually "
                   "without falls, uneconomic postures",
    "nonnekes2020": "Nonnekes J et al. Functional gait disorders: a sign-based approach. Neurology 2020; 94: "
                    "1093-1099 (PMC7455329, read 2026-11-02): knee buckling with normal strength, exaggerated "
                    "truncal sway without falls, signs inconsistent and incongruent",
    "kahn2019": "Kahn MB et al. J Neuroeng Rehabil 2019; 16: 160, Table 2, healthy controls (as cited by the REFWALK "
                "report reports/synth/reference_walk/summary.md, not read here): elbow mean 11.9 (SD 6.0), elbow peak "
                "27.4 (SD 7.9), shoulder range 24.5 (SD 8.5) deg",
    "chung2010": "Chung CY et al. J Neuroeng Rehabil 2010; 7: 9 (as cited by the REFWALK report, not read here): "
                 "cadence men 107.6 (SD 4.3), women 114.5 (SD 7.6) steps/min; trunk tilt -0.2 (SD 3.6) deg",
    "park2023": "Park J, Joo SY, Choi BO, et al. Gait pattern in Charcot-Marie-Tooth disease type 1A according to "
                "disease severity. J Pers Med 2023; 13: 1473, doi 10.3390/jpm13101473 (as cited by the catalogue row "
                "G07, not re-read here): moderate CMT1A against young controls, swing hip flexion 44.9 against 37.8 "
                "deg, peak swing knee flexion 70.5 against 65.6",
    "uhdrs": "Huntington Study Group 1996, Unified Huntington's Disease Rating Scale, Mov Disord 11: 136-142, doi "
             "10.1002/mds.870110204, through the declared amplitude and duration map of involuntary.py (UHDRS, AMP, "
             "DUR; its SOURCES entry 'uhdrs' states what was read): grade 2 mild amplitude and common duration",
    "collins2013": "Collins SH, Kuo AD. Two independent contributions to step variability during over-ground human "
                   "walking. PLoS ONE 2013; 8: e73597, doi 10.1371/journal.pone.0073597 (full text read 2026-11-03 "
                   "through Europe PMC): 14 healthy adults aged 21 to 37 at their self-selected speed, step width "
                   "0.168 (SD 0.044) m from markers on each forefoot",
    "westlake2013": "Westlake CG, Milner CE, Zhang S, Fitzhugh EC. Do thigh circumference and mass changes alter knee "
                    "biomechanics during walking? Gait Posture 2013; 37: 359-362, doi 10.1016/j.gaitpost.2012.07.031 "
                    "(abstract read 2026-11-03): in 20 healthy-weight young adults a thigh circumference added to the "
                    "size of a 10-point higher body mass index widened the step, read as the physical constraint of "
                    "wider thighs (no step width value in the abstract)",
    "module": "a default or reading of this module, not taken from a source (stated where used)",
}


# ---------------------------------------------------------------- parameters
@dataclass
class GaitParams:
    """Every parameter of the generator (module docstring). A pair is (left, right); a scalar applies to both
    sides. cadence_spm, step_m, step_width_m, clearance_m, knee_swing_deg, arm_swing_deg, trunk_lean_deg and
    elbow_deg may also be functions of time t (array, s) returning an array. Angles in degrees, lengths in m."""
    cadence_spm: object = 114.0          # perry2010 113, pirker2017 115 to 120
    step_m: object = 0.68                # perry2010 stride 1.41 m, scaled to the neutral body's shorter legs
    step_asymmetry: float = 0.0          # left steps step_m (1 + r), right steps step_m (1 - r)
    step_width_m: object = STEP_WIDTH_M  # ankle to ankle (STEP_WIDTH_M, a module value)
    stand_width_m: float = 0.17          # ankle to ankle when standing (the rest pose's)
    toe_out_deg: object = (6.0, 6.0)     # foot progression angle [module]
    stance_share: object = (0.61, 0.61)  # perry2010, pirker2017: about 60 percent
    hs_offset: float = 0.5               # the right heel strike in the left cycle
    place_share: float = 0.20            # the hip over the ankle at this share of the stride after heel strike [module]
    # stance rocker (contact.roll convention: positive pitch lowers the toes, the heel up)
    contact_pitch_deg: object = (-18.0, -18.0)   # heel strike with the toes up [module, perry2010 heel rocker]
    mid_pitch_deg: object = (0.0, 0.0)           # flat foot; above 0 the stance stays on the toes
    liftoff_pitch_deg: object = (66.0, 66.0)     # rolled onto the toes at toe-off (a rigid foot) [module]
    flat_share: object = (0.09, 0.09)            # foot flat at this share of the stride after heel strike
    heel_off_share: object = (0.28, 0.28)
    pivot_share: float = 0.9             # share of a turn's heading change a planted foot pivots through
    turn_rocker_drop: object = None      # a turning stance's rocker drop per side (None: TURN_ROCKER_DROP)
    # knee (target of the stance leg, then the swing peak)
    knee_contact_deg: object = (3.0, 3.0)
    knee_loading_deg: object = (16.0, 16.0)      # knee_norm 16
    knee_loading_share: float = 0.13
    knee_mid_deg: object = (4.0, 4.0)
    knee_swing_deg: object = (62.0, 62.0)        # knee_norm 64.9 (SD 5.6)
    knee_peak_share: object = 0.30               # of the swing (a scalar or a pair)
    knee_stand_deg: float = 2.0
    stance_knee_share: object = 1.0      # the stance knee target's excursion above knee_stand_deg (contact, loading,
    #                                      mid stance) scaled by this: a scalar, a pair or functions of time (steps in
    #                                      place barely load the knee)
    knee_buckles: tuple = ()             # stance knee dips (functional): (side, t_s, depth_deg, width_s) a Gaussian
    #                                      dip, or (side, t_s, depth_deg, drop_s, rise_s) a sudden drop reaching its
    #                                      depth at t_s and a slower recovery
    buckle_arm_deg: float = 0.0          # both arms abduct by this at each buckle's depth (a catch reaction) [module]
    # hip and ankle in swing
    hip_overshoot_deg: object = (4.0, 4.0)
    hip_peak_share: float = 0.82
    swing_abduction_deg: object = (0.0, 0.0)     # circumduction bump (negative: an adduction bump, scissoring)
    swing_rotation_deg: object = (0.0, 0.0)      # internal rotation bump of the swing hip (scissoring: with the
    #                                              knee flexed it carries the foot out while the knee goes in)
    swing_late_share: float = 0.0                # the abduction bump starts at this share of the swing
    #                                              (0: over the whole swing; scissoring: once the swing knee is level
    #                                              with the stance knee, so that it is not pushed into it from behind)
    swing_extra_abd: tuple = ()                  # ((side, toe-off knot, deg, centre share), ...): abduction bumps
    #                                              added to given swings (the post-overlay guard's, generate)
    leg_gap_m: object = LEG_GAP_M               # the leg-to-leg guard: the smallest hard gap (legcontact) a swing
    #                                              keeps from the other leg, its hip abducting where needed; None:
    #                                              no guard
    swing_ankle_deg: object = (3.0, 3.0)         # ankle at mid swing, dorsiflexion positive (foot drop negative)
    swing_ankle_share: object = 0.45             # of the swing (a scalar or a pair)
    swing_inversion_deg: object = (0.0, 0.0)
    stance_inversion_deg: object = (0.0, 0.0)    # walking stances on the lateral border (the sole toward the
    #                                              midline), the standing ones flat
    clearance_m: object = (0.016, 0.016)         # minimum foot clearance [module, from the healthy walk]
    clearance_window: tuple = (0.25, 0.85)
    lift_synergy: tuple = (1.0, 2.0, 1.0)        # (hip, knee, ankle) flexion per unit of the clearance lift, or a
    #                                              pair of them; a foot drop lifts by the hip and knee alone, the
    #                                              shank kept at its angle: (2, 2, 0) [module]
    # pelvis
    pelvis_rotation_deg: float = 5.0             # amplitude (half the range), po_norm 10 to 13 range
    pelvis_obliquity_deg: object = (3.5, 3.5)    # swing side drop during that side's stance [po_norm]
    leg_length_comp: float = 1.0                 # share of the rest skeleton's leg-length difference (about 3 mm,
    #                                              the right leg longer; its swing foot passes about 6 mm lower)
    #                                              taken up by a hike of the longer leg's pelvic side while that leg
    #                                              swings (up to about 2.4 deg), so that both swings clear alike
    pelvis_tilt_deg: float = 1.5
    lordosis_deg: float = 0.0                    # lumbar hyperlordosis: the pelvis tilted forward by this, the
    #                                              lumbar spine extended by as much (LORDOSIS_SHARE), the trunk's
    #                                              line kept (myopathic waddling gait)
    lateral_sway_m: float = 0.022
    # trunk and head
    trunk_lean_deg: object = 5.0                 # [module] a slight forward lean (the visual check)
    trunk_lateral_deg: object = (0.0, 0.0)       # lean toward the stance side during that side's stance
    trunk_counter_rotation: float = 0.6          # thorax yaw against the pelvis, share of its amplitude
    trunk_sway_deg: float = 0.0                  # slow random lateral sway amplitude (ataxia, functional)
    trunk_sway_hz: float = 0.6
    trunk_twist_deg: float = 0.0                 # held axial twist of the trunk against the pelvis (dystonia)
    trunk_twist_osc_deg: float = 0.0
    turn_lead_s: float = 0.30                    # head and thorax anticipate a turn's heading by this [module:
    #                                              about 35 deg of head-pelvis separation in the level-0 turn, the
    #                                              Spildooren2013 control peak 35.4 deg of the catalogue row P14]
    turn_lead_trunk: float = 0.4                 # share of that lead in the thorax (the rest in the neck and head)
    head_level: float = 0.7
    head_flexion_deg: float = 0.0
    # arms
    arm_swing_deg: object = (16.0, 16.0)         # shoulder amplitude [module; elbow2023 for the elbow]
    arm_flexion_deg: object = (2.0, 2.0)
    arm_abduction_deg: object = (11.0, 11.0)     # from vertical
    arm_lag: float = 0.03
    arm_lean_follow: float = 0.0                 # share of the trunk's lateral lean the arms take back (abducting on
    #                                              the side of the lean, adducting on the other): 0 locked to the
    #                                              thorax, 1 hanging as before the lean [module]
    elbow_deg: object = (12.0, 12.0)
    elbow_swing_deg: object = (9.0, 9.0)         # elbow2023: range 29.7 (SD 10.2)
    arm_posture: dict = field(default_factory=dict)   # {side: {joint: [(axis, deg), ...]}} held, no swing
    # variability
    step_time_cv: float = 0.015                  # [module] healthy stride-time CV about 1 to 2 percent
    step_length_cv: float = 0.02
    step_width_sd_m: float = 0.012
    clamp_upper: bool = True                      # trunk, head and arm rotations clamped into the table
    footprint_offsets: tuple = ()                 # ((side, stance index, lateral m outward, forward m), ...)
    wander_m: float = 0.0
    wander_hz: float = 0.25
    # initiation, stop, turns
    start_s: float = 0.7
    turn_speed_share: float = 0.45
    turn_widen_m_rad: float = 0.05               # the inside foot of a turning stance placed this much wider per rad
    #                                              of heading change over the stance [module]
    turn_base_m: float = TURN_BASE_M             # the step width widened by this in a turn (at its slowest)
    stop_steps: int = 2
    # the gap wave of the catalogue (2026-11-02); every default leaves the former clips bit for bit unchanged
    clear_at: float = CLEAR_AT                   # centre of the clearance bump in the swing (share of the swing)
    clear_width: float = 0.2                     # its Gaussian width (share of the swing) [module]
    pump_deg: float = 0.0                        # a trunk flexion locked to the gait cycle (the dromedary gait): 0 to
    #                                              pump_deg once per stride, a squared raised cosine peaking at
    #                                              pump_peak of the left cycle, pump_hip_share of it taken at the hips
    #                                              (the pelvis tilted forward, both hips flexing), the rest by the
    #                                              spine; the arms hang (PUMP_ARM_SHARE taken back at the
    #                                              shoulders) [module]
    pump_peak: float = 0.30
    pump_hip_share: float = 0.4
    drag_sides: tuple = ()                       # sides whose swing slides the foot on the floor (a dragged leg): the
    #                                              lift solved per knot so that the lowest sole point lies
    #                                              drag_height_m above the floor (a deliberate skate, info['drag'])
    drag_height_m: float = 0.0
    # wave SH1d (2026-11-03): every default leaves the former clips as they are
    swing_min_jerk_m: object = None              # the swing's ankle on a minimum-jerk path from its toe-off to its
    #                                              heel-strike position, lifted by this (m) in mid swing, the foot's
    #                                              rotation on the same profile, the leg by inverse kinematics
    #                                              (_min_jerk_swing: the pull test's backward recovery steps); None:
    #                                              the joint-space swing
    dip_synergy: object = None                   # (hip, knee, ankle) flexion per unit of the dip lift (_dip_bump, a
    #                                              sole dipping below the floor in the swing), or a function of the
    #                                              swing's toe-off time (s) returning such a triple or None; None:
    #                                              lift_synergy
    arm_swing_step_ref_m: object = None          # the arm swing scaled by the step length over this (m, at most 1), so
    #                                              that it follows the step plan (steps in place swing the arms little);
    #                                              None: the gait amplitude alone scales it


NORMAL = GaitParams()


def pair(v):
    """(left, right) of a scalar or a pair."""
    if isinstance(v, (tuple, list)):
        return float(v[0]), float(v[1])
    return float(v), float(v)


def _at(v, t):
    """Values of a scalar or a function of time at times t (array)."""
    t = np.asarray(t, dtype=np.float64)
    return np.asarray(v(t), dtype=np.float64) * np.ones_like(t) if callable(v) else np.full_like(t, float(v))


def _side_at(v, i, t):
    """Values of side i (0 left) of a scalar, pair, function, or pair of functions, at times t."""
    if isinstance(v, (tuple, list)):
        return _at(v[i], t)
    return _at(v, t)


def _pick(v, i, t_s):
    """Side i's value of a scalar, pair, function of time or pair of them at the time t_s (s): the parameters read
    once per stance (at its heel strike) or per swing (at its toe-off). A scalar or a pair gives pair(v)[i]."""
    if callable(v) or (isinstance(v, (tuple, list)) and any(callable(x) for x in v)):
        return float(_side_at(v, i, np.array([float(t_s)]))[0])
    return pair(v)[i]


# ---------------------------------------------------------------- small numerics
def _raised(t, a, d):
    if d <= 0:
        return (np.asarray(t) >= a).astype(np.float64)
    u = np.clip((np.asarray(t) - a) / d, 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * u)


def hermite_knots(s, knots):
    """Piecewise cubic Hermite at s (n,) through knots [(s_i, v_i, slope_i)] (slopes per unit s)."""
    s = np.asarray(s, dtype=np.float64)
    out = np.full(s.shape, float(knots[-1][1]))
    out[s <= knots[0][0]] = knots[0][1]
    for (s0, v0, m0), (s1, v1, m1) in zip(knots, knots[1:]):
        if s1 <= s0:
            continue
        sel = (s >= s0) & (s <= s1)
        h = s1 - s0
        x = (s[sel] - s0) / h
        out[sel] = ((2 * x ** 3 - 3 * x ** 2 + 1) * v0 + (x ** 3 - 2 * x ** 2 + x) * h * m0
                    + (-2 * x ** 3 + 3 * x ** 2) * v1 + (x ** 3 - x ** 2) * h * m1)
    return out


def _movmin_mean(x, n):
    """Moving minimum then moving mean over n samples (n made odd, so that both windows are centred; edges held):
    never above x."""
    n = int(n) | 1
    if n <= 1:
        return np.asarray(x, dtype=np.float64).copy()
    pad = (n // 2, n - 1 - n // 2)
    lo = np.lib.stride_tricks.sliding_window_view(np.pad(x, pad, mode="edge"), n).min(1)
    return np.convolve(np.pad(lo, pad, mode="edge"), np.ones(n) / n, mode="valid")


def _smooth(x, n):
    if n <= 1:
        return np.asarray(x, dtype=np.float64).copy()
    w = np.hanning(n + 2)[1:-1]
    w /= w.sum()
    pad = (n // 2, n - 1 - n // 2)
    return np.convolve(np.pad(x, pad, mode="edge"), w, mode="valid")


def smooth_noise(rng, t, hz, sd):
    """A smooth random signal of standard deviation about sd: white noise low-passed below hz (a Hann kernel of
    one period), rescaled to sd over the clip."""
    if sd == 0:
        return np.zeros_like(t)
    rate = 1.0 / (t[1] - t[0])
    n = max(3, int(round(rate / hz)))
    x = _smooth(rng.normal(size=len(t) + 2 * n), n)[n:-n]
    return sd * (x - x.mean()) / max(x.std(), 1e-12)


def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


DIP_WIDTH = 0.12                             # the dip bump's Gaussian width and edge (share of the swing)
DIP_EDGE = 0.08
DIP_WIDEN = 1.5                              # [module] a swing whose knee turns too fast takes dip and landing bumps
DIP_WIDEN_STEPS = 4                          # this much wider, up to this many times (_Swing.solve)
KNEE_SWING_RATE_DEG_S = 700.0                # the swing knee's rate the widening keeps under: the continuity check's
#                                              KNEE_RATE_MAX_DEG_S, read here on every knot over its frame interval
#                                              (its frames are a subset). At 0.95 of it the widening also reached the
#                                              freezing by shuffling, whose knee stays under the bound, and its wider
#                                              bump took the ankle's peak acceleration past PEAK_ACC_MAX_M_S2


def _knee_rate(kq, dt, fps=30.0):
    """The largest knee flexion speed (deg/s) of a swing's knee path kq (deg per knot, spacing dt) over the
    continuity check's frame interval (1/fps), read on every knot."""
    n = max(1, int(round(1.0 / (fps * dt))))
    if len(kq) <= n:
        return 0.0
    return float(np.abs(kq[n:] - kq[:-n]).max() / (n * dt))


def _dip_bump(s, sc, width=DIP_WIDTH, edge=DIP_EDGE):
    """The anti-penetration lift shape of a swing (s in [0, 1]) for a sole dip at sc: a Gaussian about sc whose
    edges reach 0 with a zero slope at toe-off and heel strike over the shorter of sc and 0.25 of the swing, and
    over edge at least. (The former shape, the Gaussian times sin(pi s) squared, nearly vanished at a dip just after
    toe-off, so that lifting a toe scuff of a millimetre took gains of 25 to 75 deg whose tail flexed the knee 20
    to 50 deg too far at a third of the swing.) The leg-to-leg guard's bumps take edge LEG_GUARD_EDGE."""
    a0 = max(float(np.clip(sc, edge, 0.25)), edge)
    a1 = max(float(np.clip(1.0 - sc, edge, 0.25)), edge)
    return np.exp(-0.5 * ((s - sc) / width) ** 2) * _smoothstep(s / a0) * _smoothstep((1.0 - s) / a1)


DRAG_TOL_M = 0.002                           # a dragged foot's lowest sole point within this of its height slides
DRAG_LIFT_RANGE = (-12.0, 30.0)              # the per-knot lift of a dragged swing, deg of the synergy [module]
PUMP_ARM_SHARE = 1.0                         # share of the dromedary pump the shoulders take back (the arms
#                                              hang instead of turning with the trunk) [module]


def _drag_lift(low_fn, n, target, iters=32):
    """(n,) the per-knot lift d (deg of the lift synergy) whose lowest sole point low_fn(d) (n,) lies at target
    (m above the floor): a bisection on every knot at once (the sole rises with the lift); a knot whose sole lies
    above target even at the lowest lift keeps that lift (it floats), one below it at the highest keeps that."""
    lo, hi = DRAG_LIFT_RANGE
    a, b = np.full(n, lo), np.full(n, hi)
    fa, fb = low_fn(a) - target, low_fn(b) - target
    for _ in range(iters):
        m = 0.5 * (a + b)
        up = low_fn(m) - target < 0.0
        a, b = np.where(up, m, a), np.where(up, b, m)
    d = 0.5 * (a + b)
    return np.where(fa > 0.0, lo, np.where(fb < 0.0, hi, d))


TURN_ROCKER_DROP = 0.25                      # [module] a turning stance's rocker drops by this share per 90 deg of
#                                              heading change over the stance (at most by half)


PIVOT_NEAR_M = 0.001                         # [module] a sole cloud point of a rolled stance foot this close to the
#                                              floor touches it (the pivot's candidates: the neutral rest sole under
#                                              the ball lies 0 to 0.8 mm above its lowest point)
PIVOT_HOLD_M = 0.005                         # [module] the flat foot's pivot is kept unless another touching point
#                                              lies this much nearer the ball (no flicker between near points)


def _pivot_offsets(Ry, O, j0):
    """(N, 3) the translations b_n of a planted foot that turns by the yaw Ry (N, 3, 3) about its contact point O
    (N, 3, the rolled foot's, before the turn) knot to knot: the foot's points map to Ry_n x + b_n, b_j0 = 0 (the
    placement knot), and the contact point of the later knot of each pair stays where it was at the earlier one
    (Ry_n O_n + b_n = Ry_{n-1} O_n + b_{n-1}), so that the foot never jumps when its contact passes from the heel to
    the forefoot."""
    N = len(O)
    b = np.zeros((N, 3))
    for n in range(j0 + 1, N):
        b[n] = b[n - 1] + (Ry[n - 1] - Ry[n]) @ O[n]
    for n in range(j0 - 1, -1, -1):
        b[n] = b[n + 1] + (Ry[n + 1] - Ry[n]) @ O[n + 1]
    return b


PIVOT_TOE_NEAR_M = 0.001                     # [module] the push-off's pivot point turns the foot at a held pitch
#                                              when it lies this close to the floor, PIVOT_NEAR_M, the touch of a
#                                              stance contact. 2 mm on SMPL+H, whose flat foot's toe pad lay 1 mm
#                                              above its lowest point; on Anny it lies 1.0 to 1.5 mm above it, so at
#                                              2 mm the turning flat foot turned about its pad while its lowest point,
#                                              on the floor, slid (the gait tests' turn: 5.564 mm/s; wave 5 second
#                                              follow-up, scripts/openbody_wave5_checks.py second, slip)


def _push_off_pivot(cloud, A0, G0, K, piv, rolling):
    """(cloud index, (N,) bool) of a stance's push-off pivot, or None without a push-off: the sole cloud point
    (cloud (M, 3), the footprint's: ankle A0, rotation G0) that the roll onto the toes (contact.roll, its hull of K
    vertices) pivots on over the most knots of the push-off, and the knots of the push-off (rolling into the knot,
    contact.roll's pivot ahead of the ankle along the foot) until the roll goes on to a vertex ahead of it (the toe
    tip at the end of a long roll, which then turns the foot about itself, the push-off's pivot lifting off the floor).
    generate turns a turning stance's foot about it on those knots and on the knots at a held pitch on which it
    touches the floor, so that one sole point stands still from the forefoot's landing to toe-off. (The ball of the
    foot, the touching point nearest the toe joint, turned the flat foot, and the toes the roll leaves still turned it
    in the push-off, 7 cm ahead: the ball, 1 to 4 mm above the toes and touching, circled them, and the toes had
    circled the ball before, so that no sole point stood still through a turning contact, 5.4 to 18 mm in the turns
    of ten walks on the page's skate check, commit fe89df3.)"""
    f, _ = CT.foot_axes(G0[None])
    S = cloud @ f[0]
    h = CT._hull(S, cloud[:, 1])
    hidx = np.r_[h, np.repeat(h[:1], K - len(h))]                # hull_points' order (padded by its first vertex)
    Sv = (cloud[hidx] - A0) @ f[0]                               # the vertices along the foot from the ankle
    fwd = np.asarray(rolling, bool) & (Sv[piv] > 0.0)
    if not fwd.any():
        return None
    ks, counts = np.unique(hidx[piv[fwd]], return_counts=True)
    k = int(ks[np.argmax(counts)])
    past = Sv[piv] > float((cloud[k] - A0) @ f[0]) + 1e-9          # the roll gone on to a vertex ahead of it
    return k, fwd & ~past


DRAG_SMOOTH_S = 0.08                         # [module] the dragged swing's lift eased over this (_drag_smooth)


def _drag_smooth(dd, floats, n):
    """The per-knot lift dd (n,) of a dragged swing (_drag_lift) made smooth: the floating knots (their sole above
    the target even at the lowest lift, which the bisection returns there) take the lift of their nearest solved
    knots, the series eased to 0 at both ends of the swing (toe-off and heel strike carry no lift) by a Hann window
    of n knots, and never below the solved lift where the sole reaches the target (the sole kept on or above it).
    (Unsmoothed, a floating knot's lowest lift beside a solved one turned the hip, knee and ankle by up to 13 deg of
    the synergy in 10 ms: the dragged foot jumped 18 cm back with the heel 9 cm up in one frame, 2026-11-03 review.)"""
    x = np.asarray(dd, dtype=np.float64).copy()
    ok = ~np.asarray(floats, bool)
    if ok.any() and not ok.all():
        idx = np.arange(len(x))
        x[~ok] = np.interp(idx[~ok], idx[ok], x[ok])
    if n > 1:
        x = _smooth(np.r_[np.zeros(n), x, np.zeros(n)], n)[n:-n]
    return np.where(ok, np.maximum(x, dd), x)


def _buckle_profile(t, t_s, *shape):
    """(T,) in [0, 1]: a knee buckle's profile, 1 at its depth t_s: shape (width_s,) a Gaussian of that full width
    at half maximum; (drop_s, rise_s) a sudden drop over drop_s ending at t_s, then a recovery over rise_s
    (smoothstep both)."""
    if len(shape) == 1:
        return np.exp(-0.5 * ((t - t_s) / (shape[0] / 2.355)) ** 2)
    drop, rise = shape
    return np.where(t <= t_s, _smoothstep((t - (t_s - drop)) / drop), 1.0 - _smoothstep((t - t_s) / rise))


def _rot_axis(axis, deg):
    """(N, 3, 3) numpy rotations about 'x', 'y' or 'z' by deg (N,)."""
    a = np.radians(np.asarray(deg, dtype=np.float64))
    c, s = np.cos(a), np.sin(a)
    o, z = np.ones_like(a), np.zeros_like(a)
    if axis == "x":
        m = [[o, z, z], [z, c, -s], [z, s, c]]
    elif axis == "y":
        m = [[c, z, s], [z, o, z], [-s, z, c]]
    else:
        m = [[c, -s, z], [s, c, z], [z, z, o]]
    return np.moveaxis(np.array(m), (0, 1), (-2, -1))


def _xzy(a, d, g):
    """Rx(a) Rz(d) Ry(g), degrees (N,) each."""
    return _rot_axis("x", a) @ _rot_axis("z", d) @ _rot_axis("y", g)


def _xzy_angles(M):
    """(a, d, g) degrees of M = Rx(a) Rz(d) Ry(g)."""
    d = np.degrees(np.arcsin(np.clip(-M[..., 0, 1], -1.0, 1.0)))
    g = np.degrees(np.arctan2(M[..., 0, 2], M[..., 0, 0]))
    a = np.degrees(np.arctan2(M[..., 2, 1], M[..., 1, 1]))
    return a, d, g


def _rot_unit(axis, deg):
    """(3, 3) numpy rotation about the unit vector axis (3,) by deg (Rodrigues)."""
    u = np.asarray(axis, dtype=np.float64)
    u = u / np.linalg.norm(u)
    a = math.radians(deg)
    K = np.array([[0.0, -u[2], u[1]], [u[2], 0.0, -u[0]], [-u[1], u[0], 0.0]])
    return np.eye(3) + math.sin(a) * K + (1.0 - math.cos(a)) * (K @ K)


def _spec_rot(spec):
    M = np.eye(3)
    for ax, deg in spec:
        M = _rot_axis(ax, np.array([deg]))[0] @ M
    return M


# ---------------------------------------------------------------- geometry of the neutral body
class _Geometry:
    def __init__(self, betas=None, gender=NEUTRAL):
        from .base import static_body
        self.rest = rest_joints(betas, gender).numpy()
        r = self.rest
        stand = static_body("stand", 0.02, betas=betas, gender=gender)
        self.cloud = {}
        self.rest_fpa = {}
        for side in SIDES:
            h, k, a, f = LEG[side]
            joints, off = CT.sole_rest(stand, side)
            self.cloud[side] = off + (r[joints] - r[a])          # rigid foot (the foot joint's rotation is identity)
            v = r[f] - r[a]
            self.rest_fpa[side] = math.degrees(math.atan2(SIGN[side] * v[0], v[2]))   # toe-out of the rest foot
        # the left foot joint on the floor y = 0 when flat; each foot's sole at the same floor (the right leg of the
        # neutral rest skeleton is 7 mm longer and its foot joint sits higher above its sole)
        self.cloud_floor = float(r[7, 1] - r[10, 1]) + float(self.cloud["left"][:, 1].min())
        self.ankle_flat = {s: self.cloud_floor - float(self.cloud[s][:, 1].min()) for s in SIDES}
        self.p = {s: r[LEG[s][1]] - r[LEG[s][0]] for s in SIDES}
        self.q = {s: r[LEG[s][2]] - r[LEG[s][1]] for s in SIDES}
        self.hip_off = {s: r[LEG[s][0]] - r[0] for s in SIDES}
        self.kgrid = np.linspace(0.0, 150.0, 1501)
        self.dgrid = {}
        for s in SIDES:
            p, q = self.p[s], self.q[s]
            R = _rot_axis("x", self.kgrid)
            self.dgrid[s] = np.linalg.norm(p[None] + R @ q, axis=-1)     # hip-ankle distance against knee angle
        # leg-length difference: the sole of a straight leg below the root, per side (the right one reaches about
        # 3 mm lower); the pelvic obliquity that levels the two (negative: the right hip higher). Uncompensated,
        # the right swing foot passes about 6 mm lower than the left one (3 mm from the shorter left stance leg,
        # 3 mm from its own longer leg): generate hikes the right side of the pelvis by twice this obliquity
        # while the right leg swings (leg_length_comp)
        reach = {s: float(self.hip_off[s][1]) - float(self.dgrid[s][0]) - self.ankle_flat[s] for s in SIDES}
        self.leg_length_diff_m = reach["left"] - reach["right"]          # positive: the right sole reaches lower
        self.obliquity_comp_deg = math.degrees(math.atan2(reach["right"] - reach["left"],
                                                          float(self.hip_off["left"][0] - self.hip_off["right"][0])))

    def dist(self, side, knee_deg):
        return np.interp(knee_deg, self.kgrid, self.dgrid[side])


_GEOM = {}


def geometry(betas=None, gender=NEUTRAL):
    key = (None if betas is None else tuple(float(x) for x in np.asarray(betas)), int(gender))
    if key not in _GEOM:
        _GEOM[key] = _Geometry(betas, gender)
    return _GEOM[key]


def _yaw(psi_rad):
    return _rot_axis("y", np.degrees(psi_rad))


# ---------------------------------------------------------------- phase, progression, footprints
def _phase(t, prm, rng, start_at_s, stop_at_s, rate):
    """Gait phase (strides; the left heel strike at integers), the gait amplitude a(t), the times of the stop."""
    T = len(t)
    s_l = pair(prm.stance_share)[0]
    f = _at(prm.cadence_spm, t) / 120.0
    ramp = _raised(t, start_at_s, prm.start_s)
    n_half = int(np.ceil(t[-1] * f.max() * 2)) + 8
    eps = rng.normal(0.0, prm.step_time_cv, size=n_half) if prm.step_time_cv > 0 else np.zeros(n_half)
    phi = np.empty(T)
    phi[0] = s_l - 0.002                                  # the left foot just before toe-off when gait starts
    for k in range(1, T):
        m = 2.0 * (phi[k - 1] - s_l)
        i = int(np.clip(np.floor(m), 0, n_half - 2))
        e = eps[i] + (eps[i + 1] - eps[i]) * (m - np.floor(m))
        phi[k] = phi[k - 1] + f[k - 1] * ramp[k - 1] / (1.0 + e) / rate
    stop = None
    if stop_at_s is not None:
        k0 = int(np.searchsorted(t, stop_at_s))
        if k0 < T - 1:
            p0 = phi[k0]
            # the stop: the next stop_steps heel strikes, then the phase rests in double support just after the
            # last one (both feet down); an ease-out whose initial slope is the current rate
            hs = np.sort(np.r_[np.arange(np.floor(p0) - 1, np.floor(p0) + 4),
                               np.arange(np.floor(p0) - 1, np.floor(p0) + 4) + prm.hs_offset])
            hs = hs[hs > p0 + 0.05]
            target = hs[prm.stop_steps - 1] + 0.04
            rate0 = (phi[k0] - phi[k0 - 1]) * rate
            D = 2.0 * (target - p0) / max(rate0, 1e-9)
            x = np.clip((t[k0:] - t[k0]) / D, 0.0, 1.0)
            phi[k0:] = p0 + (target - p0) * (1.0 - (1.0 - x) ** 2)
            stop = {"k0": k0, "t0": float(t[k0]), "t_end": float(t[k0] + D), "phi_end": float(target),
                    "last_hs": [float(h) for h in hs[:prm.stop_steps]]}
    amp = ramp.copy()
    if stop is not None:
        amp = amp * (1.0 - _raised(t, stop["t0"] + 0.25 * (stop["t_end"] - stop["t0"]),
                                   0.75 * (stop["t_end"] - stop["t0"])))
    return phi, amp, stop


def _heading(t, turns, rate, default_share):
    """(heading psi (T,) rad, progression factor (T,)): turns as raised-cosine yaw-rate pulses; the progression
    slowed to the turn's speed share over a window spanning the turn and its edges."""
    yaw_rate = np.zeros(len(t))
    sf = np.ones(len(t))
    for tr in turns:
        ts, ang, dur = tr[:3]
        share = tr[3] if len(tr) > 3 else default_share
        u = (t - ts) / dur
        inside = (u >= 0) & (u <= 1)
        yaw_rate[inside] += math.radians(ang) / dur * (1 - np.cos(2 * np.pi * u[inside]))
        lead = (t - (ts - 0.4 * dur)) / (1.8 * dur)
        inn = (lead >= 0) & (lead <= 1)
        w2 = np.zeros(len(t))
        w2[inn] = 0.5 - 0.5 * np.cos(2 * np.pi * lead[inn])
        sf = np.minimum(sf, 1.0 - w2 * (1.0 - share))
    psi = np.concatenate([[0.0], np.cumsum(0.5 * (yaw_rate[1:] + yaw_rate[:-1]) / rate)])
    return psi, sf


def _stances(phi, s_lr, hs_offset):
    """Per side: [(k_start, k_end_inclusive, n)] runs of stance, n the stride index of the heel strike."""
    out = {}
    for i, side in enumerate(SIDES):
        lp = phi - (0.0 if i == 0 else hs_offset)
        n = np.floor(lp)
        u = lp - n
        st = u < s_lr[i]
        runs = []
        m = np.concatenate([[False], st, [False]])
        d = np.diff(m.astype(np.int8))
        for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
            runs.append((int(a), int(b - 1), int(n[a])))
        out[side] = (runs, u, st)
    return out


# ---------------------------------------------------------------- the swing lift
LAND_CLEAR_M = 0.003                         # [module] a closing step's sole kept this high above the floor (_Swing)
LAND_GLIDE_M = 0.001                         # [module] until its ankle lies this close to the landing place
LAND_BUMP_WIDTH = 0.15                       # [module] the narrowest landing bump (share of the swing)
#   (the page's skate check reads a skin point within 2 mm of the foot's lowest standing point, standing within 5 mm
#   of its rest height and slower than 0.25 m/s, as a contact: the walking on ice and the fluctuating gait's closing
#   steps of commit fe89df3 glided 3 cm in their last 0.2 s within 2 mm of the floor, 6.5 and 8.4 mm in one contact)


class _Swing:
    """One swing of generate: its joint paths (hip flexion ha, adduction-abduction hd0, rotation hg; knee kv; ankle
    aa, ad, ag; degrees over the swing's knots s), the hip positions Hs and root rotations Gr_s, and solve(d_abd): the
    clearance lift and the dip lifts (module docstring) of the swing whose hip abduction path is hd0 + d_abd."""

    def __init__(self, s, ha, hd0, hg, kv, aa, ad, ag, Hs, Gr_s, geo, side, syn, cw, core, win, want, a_amp):
        self.s, self.ha, self.hd0, self.hg, self.kv, self.aa, self.ad, self.ag = s, ha, hd0, hg, kv, aa, ad, ag
        self.Hs, self.Gr_s, self.geo, self.side, self.syn = Hs, Gr_s, geo, side, syn
        self.cw, self.core, self.win, self.want, self.a_amp = cw, core, win, want, a_amp
        self.inner = (s >= 0.03) & (s <= 0.97)
        self.dip_syn = None                      # the dip lift's synergy (generate sets it from dip_synergy)
        self.land_xz = None                      # the landing ankle (x, z) of a closing step (generate sets it)
        self.dt = None                           # the knot spacing, s (generate sets it): the knee-rate widening

    def low(self, hd, extra, gain, g2=0.0, ce=None, dd=None):
        s, syn, geo, side = self.s, self.syn, self.geo, self.side
        eb = sum((g * c for g, c in extra), np.zeros_like(s))
        if ce is not None:
            eb = eb + g2 * ce
        # the lift synergy (lift_synergy, default): hip flexion d, knee flexion 2 d, ankle dorsiflexion d shortens the
        # leg and keeps the thigh-shank symmetry and the foot's pitch (a knee flexion alone tips the toes down);
        # dd: a lift given per knot (the dragged leg), the bumps left out; the dip bumps take dip_synergy where set
        d = gain * self.cw + eb if dd is None else dd
        if self.dip_syn is None or dd is not None:
            dh = dk = da = d
            sh_, sk_, sa_ = syn
        else:
            dc = gain * self.cw
            sh_, sk_, sa_ = 1.0, 1.0, 1.0
            dh = syn[0] * dc + self.dip_syn[0] * eb
            dk = syn[1] * dc + self.dip_syn[1] * eb
            da = syn[2] * dc + self.dip_syn[2] * eb
        Rh = _xzy(self.ha - sh_ * dh, hd, self.hg)
        Gt = self.Gr_s @ Rh
        kq = np.maximum(self.kv + sk_ * dk, 0.5)                 # never hyperextended in swing
        Gs = Gt @ _rot_axis("x", kq)
        Ga = Gs @ _xzy(self.aa - sa_ * da, self.ad, self.ag)
        Aa = self.Hs + np.einsum("tij,j->ti", Gt, geo.p[side]) + np.einsum("tij,j->ti", Gs, geo.q[side])
        pts = Aa[:, None] + np.einsum("tij,kj->tki", Ga, geo.cloud[side])
        return pts[..., 1].min(1) - geo.cloud_floor, Rh, kq, Gs, Ga

    @staticmethod
    def _root(fun, lo_g, hi_g):
        """The root of a decreasing or increasing fun on [lo_g, hi_g], or the end nearer to it."""
        flo, fhi = fun(lo_g), fun(hi_g)
        if flo * fhi > 0:
            return lo_g if abs(flo) < abs(fhi) else hi_g
        for _ in range(30):
            mid = 0.5 * (lo_g + hi_g)
            fm = fun(mid)
            if (fm > 0) == (fhi > 0):
                hi_g, fhi = mid, fm
            else:
                lo_g, flo = mid, fm
        return 0.5 * (lo_g + hi_g)

    def solve(self, d_abd):
        """(lowest sole height above the floor (n,), Rh, kq, Gs, Ga, clearance gain, dip gain) of the swing. Where
        its knee turns faster than KNEE_SWING_RATE_DEG_S (read over the continuity check's frame interval), the dip
        and landing bumps are widened by DIP_WIDEN, up to DIP_WIDEN_STEPS times (open-body port, wave 5 second
        follow-up: on Anny's longer forefoot a narrow anti-penetration bump flexed and extended the swing knee
        faster than the continuity check admits)."""
        out = self._solve(d_abd, 1.0)
        if self.dt is None:
            return out
        scale = 1.0
        for _ in range(DIP_WIDEN_STEPS):
            if _knee_rate(out[2], self.dt) <= KNEE_SWING_RATE_DEG_S:
                break
            scale *= DIP_WIDEN
            out = self._solve(d_abd, scale)
        return out

    def _solve(self, d_abd, scale):
        s, core, win, inner = self.s, self.core, self.win, self.inner
        hd = self.hd0 + d_abd
        extra = []                                   # anti-penetration bumps: (gain, shape)

        def low(gain, g2=0.0, ce=None):
            return self.low(hd, extra, gain, g2, ce)
        solve = self._root
        gain = g2 = 0.0
        req = None

        def clear_gain():
            f0 = (low(0.0)[0] - req)[core].min()
            return solve(lambda g: (low(g)[0] - req)[core].min(),
                         *((-40.0, 0.0) if f0 > 0 else (0.0, GAIN_MAX_DEG)))
        if self.want is not None and win.any() and self.a_amp > 0.05:
            # the clearance asked in mid swing, fading with the lift bump toward the window's edges: a foot that
            # lands flat (or toes first) approaches the floor before the window ends, which no mid-swing lift can or
            # should raise (it would saturate the lift and flex the knee far beyond knee_swing_deg)
            req = self.want * self.cw / max(float(self.cw.max()), 1e-12)
            gain = clear_gain()
        # no part of the sole below the floor anywhere inside the swing: a knee and hip flexion bump centred where
        # the sole dips lifts it (up to three bumps); the clearance lift is solved again with the bumps in place, so
        # that the two do not stack (stacked, a toe dragging in early swing raised the swing knee 25 to 50 deg above
        # knee_swing_deg and the mid-swing clearance to about 45 mm)
        for _ in range(3):
            lv0 = low(gain)[0]
            if not inner.any() or lv0[inner].min() >= -EARLY_TOL_M:
                break
            sc = s[inner][int(np.argmin(lv0[inner]))]
            ce = _dip_bump(s, sc, DIP_WIDTH * scale, DIP_EDGE * scale)
            near = inner & (np.abs(s - sc) <= 0.25)
            g_new = solve(lambda g: low(gain, g, ce)[0][near].min() - EARLY_TOL_M, 0.0, 25.0)
            extra.append((g_new, ce))
            g2 += g_new
            if req is not None:
                gain = clear_gain()
        if inner.any() and low(gain)[0][inner].min() < -EARLY_TOL_M:
            # the re-solved lift let the sole dip again: the smallest lift that keeps it above the tolerance
            gain = solve(lambda g: low(g)[0][inner].min() + 0.5 * EARLY_TOL_M, gain, GAIN_MAX_DEG)
        if self.land_xz is not None:
            # the closing step: where its sole comes within LAND_CLEAR_M of the floor while its ankle is still more
            # than LAND_GLIDE_M from the landing place (a flat foot gliding in at floor height under the pelvis
            # easing to rest), a dip bump over those knots (_dip_bump, its width their span) lifts it to LAND_CLEAR_M,
            # so that the foot comes down where it lands; up to two bumps
            for _ in range(2):
                lv0, Rh0, _, Gs0, _ = low(gain)
                Aa = (self.Hs + np.einsum("tij,j->ti", self.Gr_s @ Rh0, self.geo.p[self.side])
                      + np.einsum("tij,j->ti", Gs0, self.geo.q[self.side]))
                rem = np.hypot(*(Aa[:, [0, 2]] - self.land_xz[None]).T)
                bad = (s >= 0.5) & inner & (lv0 < LAND_CLEAR_M) & (rem > LAND_GLIDE_M)
                if not bad.any():
                    break
                sa_, sb_ = float(s[bad].min()), float(s[bad].max())
                ce = _dip_bump(s, 0.5 * (sa_ + sb_), max(0.5 * (sb_ - sa_), LAND_BUMP_WIDTH) * scale)
                g_new = solve(lambda g: low(gain, g, ce)[0][bad].min() - LAND_CLEAR_M, 0.0, 25.0)
                extra.append((g_new, ce))
                g2 += g_new
        lo_v, Rh, kq, Gs, Ga = low(gain)
        return lo_v, Rh, kq, Gs, Ga, gain, g2


def _min_jerk_swing(H, Gr, A0, A1, F0, F1, s, lift_m, geo, side):
    """A swing as a foot path (GaitParams.swing_min_jerk_m): the ankle from A0 (toe-off) to A1 (heel strike) on the
    minimum-jerk profile m(s) = 10 s^3 - 15 s^4 + 6 s^5 (its speed peaks at 1.875 times the mean, mid swing), lifted
    by lift_m 16 s^2 (1 - s)^2 (lift_m at mid swing, no vertical speed at either end), the foot's global rotation
    from F0 to F1 on the same profile (about the axis of F0^T F1), the leg solved by ik.hinge_leg from the hips H
    (n, 3) and the pelvis rotations Gr (n, 3, 3), the knee turned toward the foot's own direction as in stance.
    Returns (hip local rotations, knee deg, shank global, foot global, lowest sole point above the floor), (n, ...)."""
    from ..body.rotations import matrix_to_axis_angle
    from .ik import hinge_leg
    m = s ** 3 * (10.0 - 15.0 * s + 6.0 * s * s)
    A = A0[None] + (A1 - A0)[None] * m[:, None]
    A[:, 1] += lift_m * 16.0 * s ** 2 * (1.0 - s) ** 2
    rel = matrix_to_axis_angle(torch.from_numpy(F0.T @ F1)[None])[0]
    Gf = F0[None] @ axis_angle_to_matrix(torch.from_numpy(m)[:, None] * rel[None]).numpy()
    fw = Gf[:, :, 2].copy()
    fw[:, 1] = 0.0
    fw /= np.linalg.norm(fw, axis=1, keepdims=True)
    Ht, At, Ft = (torch.from_numpy(np.ascontiguousarray(x)) for x in (H, A, fw))
    G_th, kap = hinge_leg(Ht, At, Ht + 0.3 * Ft, torch.from_numpy(geo.p[side]), torch.from_numpy(geo.q[side]),
                          torch.from_numpy(np.ascontiguousarray(Gr)), hinge=torch.cross(Ft, At - Ht, dim=-1))
    G_th = G_th.numpy()
    kq = np.degrees(kap.numpy())
    G_sh = G_th @ _rot_axis("x", kq)
    pts = A[:, None] + np.einsum("tij,kj->tki", Gf, geo.cloud[side])
    return np.einsum("tji,tjk->tik", Gr, G_th), kq, G_sh, Gf, pts[..., 1].min(1) - geo.cloud_floor


# ---------------------------------------------------------------- the generator
def generate(prm=NORMAL, duration_s=12.0, start_at_s=1.0, stop_at_s=None, turns=(), seed=0, rate=RATE,
             overlays=(), betas=None, gender=NEUTRAL, path_fn=None, _ds_round=0, _ds_pushes=(), _ov_bumps=(),
             _lift=None, _lift_round=0):
    """(Body, info) of a walk with parameters prm (module docstring): standing until start_at_s, gait
    initiation, walking, turns ((t_s, angle_deg, dur_s[, speed_share]), positive to the left), a stop from
    stop_at_s (None: no stop) ending with the feet side by side. info: arrays (phase, amplitude, stance and swing
    masks, swing weights, pitch, footprints, events) and the parameters."""
    rng = np.random.default_rng(seed)
    geo = geometry(betas, gender)
    rest = geo.rest
    T = int(round(duration_s * rate)) + 1
    t = np.arange(T) / rate
    s_lr = pair(prm.stance_share)
    phi, amp, stop = _phase(t, prm, rng, start_at_s, stop_at_s, rate)
    psi, slow_f = _heading(t, turns, rate, prm.turn_speed_share)
    # progression: two step lengths per stride, ramped in over the first half stride
    L = _at(prm.step_m, t)
    dphi = np.diff(phi, prepend=phi[0])
    gate = np.clip(0.2 + 1.6 * (phi - s_lr[0]), 0.0, 1.0)
    ds = 2.0 * L * dphi * gate * slow_f
    if stop is not None:
        # the stop: the first stop heel strike lands where the walk would have placed it, the last one beside it;
        # the progression eases out (its initial slope the walking speed) to rest above that footprint
        k0 = stop["k0"]
        s_cum = np.cumsum(ds)
        s0 = s_cum[k0]
        v0 = max(ds[k0] * rate, 1e-6)
        f0 = (phi[k0] - phi[k0 - 1]) * rate
        s_A = s0 + 2.0 * L[k0] * slow_f[k0] * max(stop["last_hs"][0] + prm.place_share - phi[k0], 0.05) * \
            (v0 / max(2.0 * L[k0] * slow_f[k0] * f0, 1e-9))
        D_s = 2.0 * (s_A - s0) / v0
        x = np.clip((t[k0:] - t[k0]) / D_s, 0.0, 1.0)
        s_cum[k0:] = s0 + (s_A - s0) * (1.0 - (1.0 - x) ** 2)
        ds = np.diff(s_cum, prepend=0.0)
        stop["progression_end_s"] = float(t[k0] + D_s)
    fwd = np.stack([np.sin(psi), np.cos(psi)], -1)                 # (T, 2) x, z
    left = np.stack([np.cos(psi), -np.sin(psi)], -1)
    c = np.cumsum(ds[:, None] * fwd, 0)
    # lateral wander, held at its value at gait initiation before it and at its value at the end of the stop's
    # progression after it: the standing body does not drift sideways over its planted feet; it is eased in over
    # twice the gait initiation (a wander that set off at full speed with the first step drifted the pelvis 20 cm
    # over the standing foot, the first step crossing over it)
    wn = smooth_noise(rng, t, prm.wander_hz, prm.wander_m)
    k_a = int(np.clip(np.searchsorted(t, start_at_s), 0, T - 1))
    wn = wn[k_a] + _raised(t, start_at_s, 2.0 * prm.start_s) * (wn - wn[k_a])
    if stop is not None:
        k_b = int(np.clip(np.searchsorted(t, stop["progression_end_s"]), k_a, T - 1))
        wn[k_b:] = wn[k_b]
    c = c + wn[:, None] * left
    if path_fn is not None:
        c, psi = path_fn(t, c, psi)
        fwd = np.stack([np.sin(psi), np.cos(psi)], -1)
        left = np.stack([np.cos(psi), -np.sin(psi)], -1)
    ST = _stances(phi, s_lr, prm.hs_offset)
    # ---- footprints and stance feet
    W = _at(prm.step_width_m, t)
    if prm.turn_base_m and turns:
        # the base widened in a turn by turn_base_m at its middle, over the turn's own window (a raised cosine over
        # [t_s, t_s + dur], the shape of its yaw rate): the steps of the approach and of the exit keep the walk's
        # base (shaped by the slowed progression, whose window spans 1.8 turns, the widening reached them)
        w_turn = np.zeros(T)
        for tr in turns:
            u = (t - tr[0]) / tr[2]
            inn = (u >= 0) & (u <= 1)
            w_turn[inn] = np.maximum(w_turn[inn], 0.5 - 0.5 * np.cos(2 * np.pi * u[inn]))
        W = W + prm.turn_base_m * w_turn
    footprints = {s: [] for s in SIDES}
    foot_A = np.full((T, 2, 3), np.nan)          # stance ankle positions
    foot_flat = np.full((T, 2, 3), np.nan)       # the flat footprint's ankle of the stance under way
    foot_G = np.full((T, 2, 3, 3), np.nan)       # stance foot global rotations
    pitch = np.zeros((T, 2))
    piv_out = np.full((T, 2), -1, np.int64)
    k_end_stop = None if stop is None else int(np.searchsorted(t, stop["t_end"]))
    for i, side in enumerate(SIDES):
        sg = SIGN[side]
        runs, u, st = ST[side]
        fpa = pair(prm.toe_out_deg)[i]
        u_ff, u_ho = pair(prm.flat_share)[i], pair(prm.heel_off_share)[i]
        s_i = s_lr[i]
        asym = 1.0 + (prm.step_asymmetry if i == 0 else -prm.step_asymmetry)
        for j, (ka, kb, n) in enumerate(runs):
            first = ka == 0
            # the rocker pitches of this stance (a function of time read at its heel strike)
            c_pitch, m_pitch, l_pitch = (_pick(v, i, t[ka]) for v in (prm.contact_pitch_deg, prm.mid_pitch_deg,
                                                                     prm.liftoff_pitch_deg))
            # placement time: the knot where this stance reaches place_share (or its last knot)
            hit = np.flatnonzero(u[ka:kb + 1] >= prm.place_share)
            kp = ka + (int(hit[0]) if len(hit) else kb - ka)
            standing_end = stop is not None and t[ka] >= stop["t0"] - 1e-9 and \
                (n + (0.0 if i == 0 else prm.hs_offset)) >= stop["last_hs"][0] - 1e-6
            if first:
                kp = 0
                w_half = 0.5 * prm.stand_width_m
            elif standing_end:
                kp = T - 1
                w_half = 0.5 * prm.stand_width_m
            else:
                w_half = 0.5 * W[kp]
            jit_l = jit_w = 0.0
            if not first and not standing_end:
                jit_l = rng.normal(0.0, prm.step_length_cv * L[kp] / math.sqrt(2.0)) if prm.step_length_cv else 0.0
                jit_w = rng.normal(0.0, prm.step_width_sd_m / math.sqrt(2.0)) if prm.step_width_sd_m else 0.0
                jit_l += (asym - 1.0) * L[kp] / 2.0 * (1.0 if i == 0 else 1.0)
            if not first and not standing_end and prm.turn_widen_m_rad:
                # a turning stance places the inside foot wider, in proportion to the heading change over the
                # stance: the body's path curves over the inside foot, whose hip otherwise adducted beyond the
                # table in tight turns of wide-based gaits
                d_turn = float(psi[kb] - psi[ka])
                if d_turn * sg > 0:
                    jit_w += prm.turn_widen_m_rad * abs(d_turn)
            for (fs, fj, dlat, dlong) in prm.footprint_offsets:       # a given misstep (tandem, staggering)
                if fs == side and fj == j:
                    jit_w += dlat                 # outward for either foot (pos multiplies jit_w by the side sign)
                    jit_l += dlong
            pos = c[kp] + (sg * w_half + sg * jit_w) * left[kp] + jit_l * fwd[kp]
            yaw_f = psi[kp] + math.radians(sg * (fpa - geo.rest_fpa[side])) * (1.0 if not first else 1.0)
            A0 = np.array([pos[0], geo.ankle_flat[side], pos[1]])
            G0 = _yaw(np.array([yaw_f]))[0]
            inv_deg = 0.0 if (first or standing_end) else pair(prm.stance_inversion_deg)[i]
            if inv_deg:
                # stance inversion: the flat foot turned about its own long axis (the sole toward the midline) and
                # set down on its lateral border; the rocker then rolls it in its sagittal plane as usual
                low0 = float((G0 @ geo.cloud[side].T)[1].min())
                G0 = _rot_unit(G0[:, 2], -sg * inv_deg) @ G0
                A0 = A0 + np.array([0.0, low0 - float((G0 @ geo.cloud[side].T)[1].min()), 0.0])
            footprints[side].append({"k_hs": ka, "k_to": kb, "n": int(n), "ankle": A0.tolist(),
                                     "yaw_deg": math.degrees(yaw_f), "kp": int(kp), "first": bool(first),
                                     "final": bool(standing_end)})
            ks = np.arange(ka, kb + 1)
            uu = u[ks]
            prof = np.where(uu < u_ff, m_pitch + (c_pitch - m_pitch) * (1.0 - np.clip(uu / max(u_ff, 1e-9), 0, 1)) ** 2,
                            np.where(uu < u_ho, m_pitch,
                                     m_pitch + (l_pitch - m_pitch) * np.clip((uu - u_ho) / max(s_i - u_ho, 1e-9),
                                                                             0, 1) ** 2))
            # a short step lands flatter and pushes off less (in-place shuffling lifts a nearly flat foot); so does a
            # turning step, whose progression is slowed to the turn's speed share (rolled onto its toes at the full
            # lift-off pitch, a foot left ahead of the turning pelvis plantarflexed its ankle beyond the table)
            rs = float(np.clip(_at(prm.step_m, np.array([t[ka]]))[0] * float(slow_f[ka:kb + 1].mean()) /
                               ROCKER_REF_M, 0.2, 1.0))
            if not (first or standing_end) and prm.pivot_share:
                # a stance that pivots through a turn pushes off less (TURN_ROCKER_DROP per quarter turn of its
                # heading): pivoting on the ball of the foot through 60 deg and more, the full roll onto the toes
                # plantarflexed the ankle beyond the table at toe-off
                drop = TURN_ROCKER_DROP if prm.turn_rocker_drop is None else pair(prm.turn_rocker_drop)[i]
                rs *= float(np.clip(1.0 - drop * abs(float(psi[kb] - psi[ka])) / (0.5 * np.pi), 0.5, 1.0))
            prof = m_pitch + rs * (prof - m_pitch)
            if first:
                # the standing foot stays flat until heel-off, then rises from flat: the mid pitch (m_pitch, on the
                # toes above 0) eased in over the push-off (unchanged where m_pitch is 0); stepped in at heel-off, the
                # toe walk's first heel rose 4 cm in 10 ms, the knee flexing 17 deg (2026-11-03 review)
                x_ho = np.clip((uu - u_ho) / max(s_i - u_ho, 1e-9), 0.0, 1.0)
                prof = np.where(uu < u_ho, 0.0, prof - m_pitch * (1.0 - x_ho))
            ph = np.radians(prof * amp[ks] + (0.0 if _lift is None else _lift[ks, i]))   # the stance guard's lift
            cloud = A0[None] + (G0 @ geo.cloud[side].T).T
            P = CT.hull_points(cloud[None], G0[None])
            N = len(ks)
            Pn = np.repeat(P, N, 0)
            ball = A0 + G0 @ (geo.rest[LEG[side][3]] - geo.rest[LEG[side][2]])      # the toe (foot) joint
            Xn = np.repeat(np.concatenate([np.stack([A0, ball]), cloud])[None], N, 0)   # the cloud from column 2
            Gn = np.repeat(G0[None], N, 0)
            floor = np.full(N, float(cloud[:, 1].min()))
            Pr, Xr, Gr, piv = CT.roll(Pn, Xn, Gn, ph, floor)
            # pivot about the vertical through the contact point while the heading turns, knot to knot: each knot
            # turns the foot about its pivot point of that knot (the later knot's of each pair: the heel before the
            # forefoot lands, then the ball a turning foot pivots on), the placement knot kp unmoved
            # (_pivot_offsets). (The former rotation of the whole heading change since kp about the current contact
            # point moved the planted foot by that angle times the heel-to-forefoot distance when the contact passed
            # from the heel to the forefoot: 5 cm in one frame in the level 0's turn, 2026-11-03 review.) While the
            # foot rolls on its heel (the heel rocker) the pivot is the roll's own contact point (contact.roll), the
            # one point the roll leaves still; from the forefoot's landing to toe-off it is the push-off's pivot
            # (_push_off_pivot: the sole point the roll onto the toes pivots on most), on the knots that roll onto
            # the toes and on those at a held pitch on which it touches; on a held knot it does not touch (the
            # stance has no push-off, or stands on its heel or its lateral border) it is the touching sole point
            # (within PIVOT_NEAR_M of the floor) nearest the vertical through the toe joint, held unless another
            # lies PIVOT_HOLD_M nearer. (The forward-most touching point the pivot was on the flat foot before wave
            # SH1d left the ball and the lateral sole circling it, 25 to 32 mm in each turning stance of the walks,
            # read on the named sole points; the ball that replaced it, with the toes turning the push-off, left no
            # sole point still through a turning contact on the page's sole cloud, commit fe89df3)
            dpsi = prm.pivot_share * (psi[ks] - psi[kp]) if not (first or standing_end) else np.zeros(N)
            if np.any(np.abs(dpsi) > 1e-9):
                Ry = _yaw(dpsi)
                fw = Gr[:, :, 2].copy()
                fw[:, 1] = 0.0
                Cr = Xr[:, 2:]
                near = np.where(Cr[..., 1] - floor[:, None] <= PIVOT_NEAR_M,
                                np.hypot(*(Cr - Xr[:, 1:2])[..., [0, 2]].transpose(2, 0, 1)), np.inf)
                pick = np.argmin(near, 1)
                for n in range(1, N):                    # the ball's point held unless another lies PIVOT_HOLD_M nearer
                    if near[n, pick[n - 1]] <= near[n, pick[n]] + PIVOT_HOLD_M:
                        pick[n] = pick[n - 1]
                rolling = np.r_[True, np.abs(np.diff(ph)) > 1e-12]          # the pitch changes into knot n
                O = np.where(rolling[:, None], Pr[np.arange(N), piv], Cr[np.arange(N), pick])
                toe = _push_off_pivot(cloud, A0, G0, P.shape[1], piv, rolling)
                if toe is not None:
                    # the push-off's pivot (_push_off_pivot) is the pivot of the whole forefoot contact: of the
                    # knots that roll onto the toes, and of the knots at a held pitch on which it touches (within
                    # PIVOT_TOE_NEAR_M of the floor); the heel rocker keeps the roll's own contact point
                    k_toe, push = toe
                    on_toe = push | (~rolling & (Cr[:, k_toe, 1] - floor <= PIVOT_TOE_NEAR_M))
                    O = np.where(on_toe[:, None], Cr[:, k_toe], O)
                off = _pivot_offsets(Ry, O, int(np.clip(kp - ka, 0, N - 1)))
                Xr = np.einsum("nij,nkj->nki", Ry, Xr) + off[:, None]
                Gr = Ry @ Gr
            foot_A[ks, i] = Xr[:, 0]
            foot_flat[ks, i] = A0
            foot_G[ks, i] = Gr
            pitch[ks, i] = np.degrees(ph)
            piv_out[ks, i] = piv
    stance = np.stack([ST[s][2] for s in SIDES], -1)
    # ---- the slow random lateral sway (trunk_sway_deg), drawn here, at the place it always held in the random
    # stream (nothing draws between the footprints and the trunk): soft-limited to TRUNK_SWAY_SOFT_DEG, a share of it
    # carried by the whole body above the ankles (the pelvis displaced sideways, TRUNK_SWAY_PELVIS_SHARE), the rest
    # by the spine (sway_lean_deg)
    sway_n = smooth_noise(rng, t, prm.trunk_sway_hz, prm.trunk_sway_deg)
    sway_shift = np.zeros(T)
    if prm.trunk_sway_deg:
        sway_n = TRUNK_SWAY_SOFT_DEG * np.tanh(sway_n / TRUNK_SWAY_SOFT_DEG)
        h_trunk = float(np.linalg.norm(rest[12] - rest[0]))
        sway_shift = TRUNK_SWAY_PELVIS_SHARE * h_trunk * np.sin(np.radians(sway_n))
        sway_n = (1.0 - TRUNK_SWAY_PELVIS_SHARE) * sway_n
    # ---- pelvis orientation and horizontal position
    phl = phi
    phr = phi - prm.hs_offset
    rot = prm.pelvis_rotation_deg * -np.cos(2 * np.pi * phl) * amp
    o_l, o_r = pair(prm.pelvis_obliquity_deg)
    lobe = np.sin(2 * np.pi * (phl + 0.05))
    obl = amp * np.where(lobe > 0, o_l * lobe, o_r * lobe)
    # the longer leg's side of the pelvis hikes while that leg swings (up to twice the levelling obliquity at mid
    # swing), so that both swing feet pass the floor alike; the longer leg's stances keep their obliquity (a static
    # tilt cost that hip about 1.2 deg of its adduction room in every stance, and a hike begun before its toe-off
    # adducted it beyond the table in a wide-based gait)
    oc = prm.leg_length_comp * 2.0 * geo.obliquity_comp_deg
    if oc != 0.0:
        runs_long = ST["right" if oc < 0 else "left"][0]
        hike = np.zeros(T)                       # sin^2 over each swing of the longer leg: 0 at toe-off and contact
        for (ka0, kb0, n0), (ka1, kb1, n1) in zip(runs_long, runs_long[1:]):
            if ka1 - kb0 >= 2:
                kk = np.arange(kb0 + 1, ka1)
                hike[kk] = np.sin(np.pi * (kk - kb0) / (ka1 - kb0)) ** 2
        obl = obl + amp * oc * hike
    tilt = prm.pelvis_tilt_deg * np.cos(4 * np.pi * (phl - 0.05)) * amp
    sway = prm.lateral_sway_m * np.sin(2 * np.pi * (phl - 0.05)) * amp
    yaw_root = psi + np.radians(rot)
    # the dromedary pump (pump_deg): a forward trunk flexion once per stride, pump_hip_share of it at the hips
    pump = None
    if prm.pump_deg:
        pump = prm.pump_deg * amp * (0.5 + 0.5 * np.cos(2 * np.pi * (phl - prm.pump_peak))) ** 2
    tilt_root = tilt + prm.lordosis_deg
    if pump is not None:
        tilt_root = tilt_root + prm.pump_hip_share * pump
    G_root = _yaw(yaw_root) @ _rot_axis("x", tilt_root) @ _rot_axis("z", obl)
    Pxz = c + (sway + sway_shift)[:, None] * left
    hipoff = {s: np.einsum("tij,j->ti", G_root, geo.hip_off[s]) for s in SIDES}
    # ---- pelvis height from the stance-leg knee target
    kc, kl, km = (np.array(pair(v)) for v in (prm.knee_contact_deg, prm.knee_loading_deg, prm.knee_mid_deg))
    ctrl_target = np.full(T, np.inf)
    cap = np.full(T, np.inf)
    last_hs = np.full((T, 2), -1)
    for i, side in enumerate(SIDES):
        for (ka, kb, n) in ST[side][0]:
            last_hs[ka:kb + 1, i] = ka
    ctrl = np.where(last_hs[:, 0] >= last_hs[:, 1], 0, 1)
    buck = np.zeros((T, 2))
    buck_env = np.zeros(T)                       # the buckles' profile, 1 at each one's depth (the arms' catch)
    buckles = []
    for bk in prm.knee_buckles:
        bi = SIDES.index(bk[0])
        t_d, shape = float(bk[1]), tuple(bk[3:])
        if len(shape) == 2:
            # a sudden buckle happens in that side's first stance from t_s (a buckle timed in a swing would not be
            # seen: only a stance knee carries the pelvis): it starts at BUCKLE_ONSET of that stance and reaches its
            # depth drop_s later
            nxt = [(ka, kb) for (ka, kb, n) in ST[bk[0]][0] if ka > 0 and t[ka] >= t_d - 1e-9]
            if nxt:
                ka, kb = nxt[0]
                t_d = float(t[ka] + BUCKLE_ONSET * (t[kb] - t[ka]) + shape[0])
        prof = _buckle_profile(t, t_d, *shape)
        buck[:, bi] += bk[2] * prof
        buck_env = np.maximum(buck_env, prof)
        buckles.append({"side": bk[0], "t_depth_s": t_d, "depth_deg": float(bk[2]), "shape_s": list(shape)})
    for i, side in enumerate(SIDES):
        u = ST[side][1]
        st = stance[:, i]
        ku = np.where(u < prm.knee_loading_share,
                      hermite_knots(u, [(0.0, kc[i], 1.0 * (kl[i] - kc[i]) / prm.knee_loading_share),
                                        (prm.knee_loading_share, kl[i], 0.0)]),
                      hermite_knots(u, [(prm.knee_loading_share, kl[i], 0.0), (0.42, km[i], 0.0),
                                        (1.0, km[i], 0.0)]))
        ku = prm.knee_stand_deg + amp * _side_at(prm.stance_knee_share, i, t) * (ku - prm.knee_stand_deg) + buck[:, i]
        A = foot_A[:, i]
        H_xz = Pxz + hipoff[side][:, [0, 2]]
        horiz2 = ((A[:, [0, 2]] - H_xz) ** 2).sum(-1)
        dmax = geo.dist(side, 0.8) * 0.9995
        with np.errstate(invalid="ignore"):
            hc = A[:, 1] + np.sqrt(np.maximum(dmax ** 2 - horiz2, 0.0)) - hipoff[side][:, 1]
            ht = A[:, 1] + np.sqrt(np.maximum(geo.dist(side, ku) ** 2 - horiz2, 0.0)) - hipoff[side][:, 1]
        if np.any(st & (horiz2 >= dmax ** 2)):
            raise ValueError(f"{side} stance ankle beyond the leg's reach: shorten the steps or widen the stance")
        cap = np.where(st, np.minimum(cap, hc), cap)
        ctrl_target = np.where(st & (ctrl == i), ht, ctrl_target)
    if not np.all(np.isfinite(cap)):
        raise ValueError("a knot with no foot in stance (flight phase): the stance shares are too short")
    raw = np.minimum(np.where(np.isfinite(ctrl_target), ctrl_target, cap), cap)
    y = _smooth(raw, int(round(0.10 * rate)))
    y = np.minimum(y, cap)
    y = _movmin_mean(y, int(round(0.06 * rate)))
    P = np.stack([Pxz[:, 0], y, Pxz[:, 1]], -1)
    # ---- legs: the stances of both sides first, then the swings, so that each swing sees the other leg's stance
    # (the leg-to-leg guard, leg_gap_m)
    R = np.repeat(np.eye(3)[None, None], T, 0).repeat(N_BODY, 1)
    R[:, 0] = G_root
    from .ik import hinge_leg
    sw_w = np.zeros((T, 2))
    swing_info = {s: [] for s in SIDES}
    clr = pair(prm.clearance_m) if prm.clearance_m is not None and not callable(prm.clearance_m) and \
        not (isinstance(prm.clearance_m, (tuple, list)) and any(callable(x) or x is None for x in prm.clearance_m)) \
        else None
    knees = {}
    for i, side in enumerate(SIDES):
        h, kn, a, ft = LEG[side]
        st = stance[:, i]
        ks = np.flatnonzero(st)
        H = P[ks] + hipoff[side][ks]
        A = foot_A[ks, i]
        Gf = foot_G[ks, i]
        ffwd = Gf[:, :, 2].copy()
        ffwd[:, 1] = 0.0
        # the knee turned toward the foot's own direction (the rest foot's toe-out removed)
        yaw_k = np.arctan2(ffwd[:, 0], ffwd[:, 2])
        kdir = np.stack([np.sin(yaw_k), np.zeros_like(yaw_k), np.cos(yaw_k)], -1)
        Ht, At, Gr_t = (torch.from_numpy(x) for x in (H, A, G_root[ks]))
        hinge = torch.cross(torch.from_numpy(kdir), At - Ht, dim=-1)
        pole = Ht + 0.3 * torch.from_numpy(kdir)
        G_th, kap = hinge_leg(Ht, At, pole, torch.from_numpy(geo.p[side]), torch.from_numpy(geo.q[side]), Gr_t,
                              hinge=hinge)
        G_th = G_th.numpy()
        kap = np.degrees(kap.numpy())
        G_sh = G_th @ _rot_axis("x", kap)
        R[ks, h] = np.einsum("tji,tjk->tik", G_root[ks], G_th)
        R[ks, kn] = _rot_axis("x", kap)
        R[ks, a] = np.einsum("tji,tjk->tik", G_sh, Gf)
        knees[side] = np.zeros(T)
        knees[side][ks] = kap
    guard = []
    land_push = []                                   # the swing guard's footprint pushes (a contact it cannot clear)
    for i, side in enumerate(SIDES):
        h, kn, a, ft = LEG[side]
        # swings: between consecutive stance runs
        runs = ST[side][0]
        hip_a, hip_d, hip_g = _xzy_angles(R[:, h])
        ank_a, ank_d, ank_g = _xzy_angles(R[:, a])
        knee = knees[side]
        for j_sw, ((ka0, kb0, n0), (ka1, kb1, n1)) in enumerate(zip(runs, runs[1:])):
            k0, k1 = kb0, ka1
            if k1 - k0 < 2:
                continue
            kk = np.arange(k0 + 1, k1)
            dur = (k1 - k0) / rate
            s = (kk - k0) / (k1 - k0)

            def slope0(x):
                return (x[k0] - x[k0 - 1]) * rate * dur if k0 - 1 >= ka0 else 0.0

            def slope1(x):
                return (x[k1 + 1] - x[k1]) * rate * dur if k1 + 1 <= kb1 else 0.0
            tm = t[k0]
            a_amp = float(np.clip(amp[k0:k1 + 1].mean(), 0.0, 1.0))
            if prm.swing_min_jerk_m is not None:
                # the swing as a foot path (swing_min_jerk_m): no lift solve, no guard (the pull test's backward
                # steps keep their legs apart, tests/test_synth_gait.py); the leg-to-leg check reads the result
                Rh, kq, Gs, Ga, lo_v = _min_jerk_swing(P[kk] + hipoff[side][kk], G_root[kk], foot_A[k0, i],
                                                       foot_A[k1, i], foot_G[k0, i], foot_G[k1, i], s,
                                                       float(prm.swing_min_jerk_m) * a_amp, geo, side)
                R[kk, h] = Rh
                R[kk, kn] = _rot_axis("x", kq)
                R[kk, a] = np.einsum("tji,tjk->tik", Gs, Ga)
                sw_w[kk, i] = np.sin(np.pi * s) ** 2
                win = (s >= prm.clearance_window[0]) & (s <= prm.clearance_window[1])
                inner = (s >= 0.03) & (s <= 0.97)
                swing_info[side].append({"k_to": int(k0), "k_hs": int(k1), "gain_deg": 0.0, "early_gain_deg": 0.0,
                                         "clearance_m": float(lo_v[win].min()) if win.any() else None,
                                         "lowest_m": float(lo_v[inner].min()) if inner.any() else None,
                                         "knee_peak_deg": float(np.max(kq)), "abduction_added_deg": 0.0,
                                         "leg_gap_m": None, "min_jerk": True})
                continue
            kpk = float(_side_at(prm.knee_swing_deg, i, np.array([tm]))[0])
            # a swing shorter than a stride of the clip's step length (the first step, the closing step, a turn
            # step) flexes the knee less: the peak scaled by the square root of the travel's share
            travel = float(np.hypot(*(foot_flat[k1, i, [0, 2]] - foot_flat[k0, i, [0, 2]])))
            Lm = float(_at(prm.step_m, np.array([tm]))[0])
            share_tr = min(1.0, travel / max(2.0 * Lm, 1e-6))
            kpk = prm.knee_stand_deg + (kpk - prm.knee_stand_deg) * math.sqrt(share_tr)
            ov = _pick(prm.hip_overshoot_deg, i, tm) * a_amp
            abd = pair(prm.swing_abduction_deg)[i] * a_amp
            rot_in = pair(prm.swing_rotation_deg)[i] * a_amp
            ank_sw = _pick(prm.swing_ankle_deg, i, tm)
            inv = pair(prm.swing_inversion_deg)[i] * a_amp
            # the hip flexes fast in initial swing: HIP_EARLY of the way to its peak by the knee's peak
            pk = hip_a[k1] - ov
            se = pair(prm.knee_peak_share)[i]
            sa = pair(prm.swing_ankle_share)[i]
            fin = bool(footprints[side][j_sw + 1]["final"]) if j_sw + 1 < len(footprints[side]) else False
            hk_a = [(0.0, hip_a[k0], slope0(hip_a)),
                    (se, hip_a[k0] + HIP_EARLY * (pk - hip_a[k0]), 1.1 * (pk - hip_a[k0]) / prm.hip_peak_share),
                    (prm.hip_peak_share, pk, 0.0), (1.0, hip_a[k1], 0.0)]
            if ov <= 0:
                hk_a = [hk_a[0], hk_a[1], (1.0, hip_a[k1], 0.0)]
            ha = hermite_knots(s, hk_a)
            hd0 = hermite_knots(s, [(0.0, hip_d[k0], slope0(hip_d)), (1.0, hip_d[k1], slope1(hip_d))])
            hg = hermite_knots(s, [(0.0, hip_g[k0], slope0(hip_g)), (1.0, hip_g[k1], slope1(hip_g))])
            bump = np.sin(np.pi * s) ** 2
            # the abduction (adduction) bump: over the whole swing, or from swing_late_share of it on
            late = bump if not prm.swing_late_share else \
                np.sin(np.pi * np.clip((s - prm.swing_late_share) / (1.0 - prm.swing_late_share), 0.0, 1.0)) ** 2
            # internal rotation over the whole swing: the knee's front toward the midline, the flexed knee's shank and
            # foot carried outward, around the stance foot
            hg = hg - SIGN[side] * rot_in * bump
            kmid = max(kpk, knee[k0], knee[k1], 0.0)      # the knee keeps flexing after toe-off
            # the knee and the ankle arrive at heel strike at rest (the impact starts their stance motion)
            kv = hermite_knots(s, [(0.0, knee[k0], slope0(knee)), (se, kmid, 0.0),
                                   (1.0, knee[k1], 0.0)])
            ank_mid = ank_a[k0] + (ank_a[k1] - ank_a[k0]) * sa
            ank_mid = ank_mid + a_amp * (-ank_sw - ank_mid) if a_amp > 0 else ank_mid
            aa = hermite_knots(s, [(0.0, ank_a[k0], slope0(ank_a)), (sa, ank_mid, 0.0),
                                   (1.0, ank_a[k1], 0.0)])
            ad = hermite_knots(s, [(0.0, ank_d[k0], slope0(ank_d)), (1.0, ank_d[k1], slope1(ank_d))])
            ag = hermite_knots(s, [(0.0, ank_g[k0], slope0(ank_g)), (1.0, ank_g[k1], slope1(ank_g))])
            ad = ad - SIGN[side] * inv * bump            # inversion: the sole turned toward the midline
            Hs = P[kk] + hipoff[side][kk]
            Gr_s = G_root[kk]
            if inv:
                # the inversion seen in the world: with the hip flexed and rotated, the knee's hinge axis rolls in mid
                # swing (the thigh's lateral axis rises), and the foot rolled with it, which took back most of the
                # asked inversion (2.8 of 16 deg read on the dystonic foot, 2026-11-03 review); the foot's own roll
                # takes that excursion of the thigh axis back (against the line between its toe-off and heel-strike
                # values), so that the sole turns toward the midline by the asked inversion
                e_th = np.degrees(np.arcsin(np.clip((Gr_s @ _xzy(ha, hd0, hg))[:, 1, 0], -1.0, 1.0)))
                ad = ad - (e_th - (e_th[0] + (e_th[-1] - e_th[0]) * (s - s[0]) / max(float(s[-1] - s[0]), 1e-9)))
            win = (s >= prm.clearance_window[0]) & (s <= prm.clearance_window[1])
            cw = np.sin(np.pi * s) ** 2 * np.exp(-0.5 * ((s - prm.clear_at) / prm.clear_width) ** 2)   # the bump
            # the clearance is asked where the lift acts (its bump at CLEAR_CORE of its peak or more): toward the
            # window's edges a foot that lands flat is already near the floor, which a mid-swing lift cannot raise
            # (asked there, the lift saturated at GAIN_MAX_DEG and flexed the knee about 20 deg beyond
            # knee_swing_deg, with the mid-swing foot 6 to 8 cm high in the walking-on-ice gait)
            core = win & (cw >= CLEAR_CORE * float(cw.max()))
            core = core if core.any() else win

            syn = prm.lift_synergy[i] if isinstance(prm.lift_synergy[0], (tuple, list)) else prm.lift_synergy
            want = None
            if clr is not None:
                want = clr[i]
            elif prm.clearance_m is not None:
                cm = prm.clearance_m[i] if isinstance(prm.clearance_m, (tuple, list)) else prm.clearance_m
                want = None if cm is None else float(_at(cm, np.array([tm]))[0])
            sw = _Swing(s, ha, hd0, hg, kv, aa, ad, ag, Hs, Gr_s, geo, side, syn, cw, core, win, want, a_amp)
            sw.dt = 1.0 / rate
            if fin and a_amp > 0.05:                     # the closing step lands without gliding (_Swing.solve)
                sw.land_xz = foot_A[k1, i][[0, 2]].copy()
            ds_ = prm.dip_synergy(tm) if callable(prm.dip_synergy) else prm.dip_synergy
            if ds_ is not None:
                sw.dip_syn = tuple(float(x) for x in ds_)
            # the asked abduction bump plus the post-overlay guard's bumps of this swing (swing_extra_abd)
            g_edge = max(LEG_GUARD_EDGE, LEG_GUARD_RAMP_S / max(dur, 1e-9))      # the guard bumps' edge (share)
            late_abd = abd * late + sum((d_ * _dip_bump(s, c_, LEG_GUARD_WIDTH, g_edge) for sd_, k_, d_, c_ in
                                         prm.swing_extra_abd if sd_ == side and k_ == k0), np.zeros_like(s))
            out = sw.solve(SIGN[side] * late_abd)
            g_add, gap = 0.0, None
            if prm.leg_gap_m is not None and a_amp > 0.05:
                # the leg-to-leg guard: where the swing leg would pass into the other leg (a hard contact closer than
                # leg_gap_m, or the inner thighs deeper than legcontact.SOFT_TOL_M less LEG_SOFT_MARGIN_M), the swing
                # hip abducts by a bump centred where the contact is deepest (the dip lift's shape, _dip_bump), the
                # smallest that keeps the gap (regula falsi on [0, LEG_GUARD_MAX_DEG]), up to LEG_GUARD_BUMPS bumps;
                # the asked abduction (an adduction for scissoring) is kept wherever the gap holds
                # measured on every LEG_GUARD_SUB-th knot of the swing while solving, on every knot at the end
                sub = np.unique(np.r_[np.arange(0, len(kk), LEG_GUARD_SUB), len(kk) - 1])

                def gap_of(add, knots=sub):
                    o = sw.solve(SIGN[side] * (late_abd + add))
                    Rk = R[kk[knots]].copy()
                    Rk[:, h], Rk[:, kn] = o[1][knots], _rot_axis("x", o[2][knots])
                    Rk[:, a] = np.einsum("tji,tjk->tik", o[3][knots], o[4][knots])
                    g = LCN.gap_from_pose(Rk, P[kk[knots]] - rest[0], betas, gender)
                    f = np.full(len(kk), np.inf)
                    f[knots] = np.minimum(g["hard_m"] - prm.leg_gap_m, g["soft_m"] + LCN.SOFT_TOL_M - LEG_SOFT_MARGIN_M)
                    return o, (float(g["hard_m"].min()), float(g["soft_m"].min())), f

                def cap(x):                                     # the bumps of a swing together, at most this
                    m = float(np.max(x)) if len(x) else 0.0
                    return x * (LEG_GUARD_MAX_DEG / m) if m > LEG_GUARD_MAX_DEG else x
                add = np.zeros_like(s)
                centres = []
                for knots in (sub, np.arange(len(kk))):        # solved on the subsample, checked on every knot
                    out, gap, fk = gap_of(add, knots)
                    for _ in range(LEG_GUARD_BUMPS):
                        if fk.min() >= 0:
                            break
                        # a bump where the contact is deepest, solved on the knots it reaches (LEG_GUARD_REACH of its
                        # peak or more, and the deepest knot: a bump near toe-off or heel strike peaks inside the
                        # swing, above its value at the contact); a contact elsewhere is the next bump's
                        k_deep = int(np.argmin(fk))
                        if _ds_round < DS_PUSH_ROUNDS and min(float(s[k_deep]), 1.0 - float(s[k_deep])) < \
                                LEG_GUARD_END_SHARE:
                            break                       # a contact at toe-off or heel strike: its stance is set
                            #                             down wider (below), no bump (LEG_GUARD_END_SHARE)
                        shape = _dip_bump(s, float(s[k_deep]), LEG_GUARD_WIDTH, g_edge)
                        reach = shape >= LEG_GUARD_REACH * float(shape.max())
                        reach[k_deep] = True
                        centres.append(float(s[k_deep]))
                        o_hi, g_hi, f_hi = gap_of(cap(add + LEG_GUARD_MAX_DEG * shape), knots)
                        if f_hi[reach].min() < 0:
                            add, out, gap, fk = cap(add + LEG_GUARD_MAX_DEG * shape), o_hi, g_hi, f_hi
                            continue
                        lo_x, hi_x = 0.0, LEG_GUARD_MAX_DEG
                        v_lo, v_hi = float(fk[reach].min()), float(f_hi[reach].min())
                        kept = 0
                        for _ in range(LEG_GUARD_STEPS):
                            if v_hi - v_lo <= 1e-12:
                                break
                            x = hi_x - v_hi * (hi_x - lo_x) / (v_hi - v_lo)
                            x = min(max(x, lo_x + 0.02 * (hi_x - lo_x)), hi_x - 0.02 * (hi_x - lo_x))
                            o_m, g_m, f_m = gap_of(cap(add + x * shape), knots)
                            if f_m[reach].min() >= 0:
                                hi_x, o_hi, g_hi, f_hi, v_hi = x, o_m, g_m, f_m, float(f_m[reach].min())
                                if kept == 1:
                                    v_lo *= 0.5                          # Illinois: the stale end's value halved
                                kept = 1
                            else:
                                lo_x, v_lo = x, float(f_m[reach].min())
                                if kept == -1:
                                    v_hi *= 0.5
                                kept = -1
                        add, out, gap, fk = cap(add + hi_x * shape), o_hi, g_hi, f_hi
                g_add = float(add.max())
                holds = bool(fk.min() >= 0)
                if not holds:
                    # a contact the bumps cannot clear (near toe-off or heel strike, where they fade): the stance
                    # it lies nearer to is set down wider (the toe-off stance in the swing's first half, the landing
                    # one in its second), by DS_PUSH_GAIN times the lack plus DS_MARGIN_M, and the walk generated
                    # again (as the double-support guard; the standing first stance is not moved)
                    k_bad = int(np.argmin(fk))
                    j_to = next(j for j, r in enumerate(runs) if r[1] == k0)
                    j_st = j_to + 1 if (s[k_bad] >= 0.5 or j_to == 0) else j_to
                    land_push.append({"round": int(_ds_round), "side": side, "stance": int(j_st),
                                      "t_s": float(t[kk[k_bad]]), "hard_m": gap[0], "soft_m": gap[1],
                                      "outward_m": float(DS_PUSH_GAIN * -float(fk.min()) + DS_MARGIN_M),
                                      "from": "swing"})
                guard.append({"side": side, "k_to": int(k0), "added_deg": g_add, "hard_gap_m": gap[0],
                              "thigh_gap_m": gap[1], "holds": holds, "bump_at_share": centres})
            lo_v, Rh, kq, Gs, Ga, gain, g2 = out
            drag = side in prm.drag_sides and a_amp > 0.05
            if drag:
                # the dragged leg: the lift solved per knot so that the lowest sole point lies drag_height_m above
                # the floor over the whole swing (the foot slides; where even a straight knee leaves it above that
                # height it floats, where the largest lift leaves it below it is lifted no further)
                hd_fin = sw.hd0 + SIGN[side] * (late_abd + (add if g_add else 0.0))

                def low_d(x):
                    return sw.low(hd_fin, [], 0.0, dd=x)[0]
                dd = _drag_lift(low_d, len(s), prm.drag_height_m)
                floats = low_d(np.full(len(s), DRAG_LIFT_RANGE[0])) > prm.drag_height_m
                dd = _drag_smooth(dd, floats, int(round(DRAG_SMOOTH_S * rate)))
                lo_v, Rh, kq, Gs, Ga = sw.low(hd_fin, [], 0.0, dd=dd)
                gain = g2 = 0.0
            R[kk, h] = Rh
            R[kk, kn] = _rot_axis("x", kq)
            R[kk, a] = np.einsum("tji,tjk->tik", Gs, Ga)
            sw_w[kk, i] = bump
            swing_info[side].append({"k_to": int(k0), "k_hs": int(k1), "gain_deg": float(gain),
                                     "early_gain_deg": float(g2),
                                     "clearance_m": float(lo_v[win].min()) if win.any() else None,
                                     "lowest_m": float(lo_v[sw.inner].min()) if sw.inner.any() else None,
                                     "knee_peak_deg": float(np.max(kq)), "abduction_added_deg": float(g_add),
                                     "leg_gap_m": None if gap is None else gap[0]})
            if drag:
                swing_info[side][-1].update(
                    drag=True, drag_lift_deg=[float(dd.min()), float(dd.max())],
                    drag_contact_share=float(np.mean(np.abs(lo_v[sw.inner] - prm.drag_height_m) <= DRAG_TOL_M))
                    if sw.inner.any() else 0.0)
    # ---- trunk and head
    lean = _at(prm.trunk_lean_deg, t)
    tl_l, tl_r = pair(prm.trunk_lateral_deg)
    tlat = amp * np.where(lobe > 0, tl_l * lobe, tl_r * lobe)     # lean to the left positive, toward the stance side
    tlat = tlat + sway_n                                           # the spine's share of the random sway
    # the lean the arms take back lags the trunk by about ARM_LEAN_LAG_S (the arms hang from it) [module]
    n_lag = int(round(ARM_LEAN_LAG_S * rate))
    tlat_arm = np.concatenate([np.full(n_lag, tlat[0]), tlat[:T - n_lag]]) if 0 < n_lag < T else tlat.copy()
    thor_rot = -prm.trunk_counter_rotation * rot                  # thorax yaw in the heading frame
    twist = prm.trunk_twist_deg + prm.trunk_twist_osc_deg * np.sin(2 * np.pi * phl)
    # turns: the head, then the thorax, lead the pelvis into the new heading (the heading turn_lead_s ahead; the
    # thorax takes turn_lead_trunk of that lead, the neck and head the rest); turn_lead_s near 0 turns en bloc
    ahead = np.degrees(np.interp(t + prm.turn_lead_s, t, psi) - psi) if prm.turn_lead_s > 0 else np.zeros(T)
    rel_yaw = thor_rot - rot + twist + prm.turn_lead_trunk * ahead
    rel_lat = -(tlat + obl)                     # the spine's bend to the right that gives the lean over the pelvis
    rel_flex = lean - tilt
    lean_h = lean                               # the trunk line the head is held level against
    if pump is not None:                        # the spine takes the pump's share the hips do not
        rel_flex = rel_flex + (1.0 - prm.pump_hip_share) * pump
        lean_h = lean + pump
    for j, w, wl in zip(SPINE, SPINE_SHARE, LORDOSIS_SHARE):
        R[:, j] = _rot_axis("y", w * rel_yaw) @ _rot_axis("z", w * rel_lat) @ \
            _rot_axis("x", w * rel_flex - wl * prm.lordosis_deg)
    head_lead = 0.5 * (1.0 - prm.turn_lead_trunk) * ahead
    # head righting on roll: the neck and the head take back HEAD_RIGHTING of the trunk's lateral lean, half each
    # (the head took back 0.5 head_level of it alone, so that the ears tilted with the trunk, 2026-11-03 review)
    right_z = 0.5 * HEAD_RIGHTING * tlat
    R[:, 12] = _rot_axis("y", -0.5 * prm.head_level * (thor_rot + twist) + head_lead) @ \
        _rot_axis("z", right_z) @ _rot_axis("x", -0.5 * prm.head_level * lean_h + 0.5 * prm.head_flexion_deg)
    R[:, 15] = _rot_axis("y", -0.5 * prm.head_level * (thor_rot + twist) + head_lead) @ \
        _rot_axis("z", right_z) @ _rot_axis("x", -0.5 * prm.head_level * lean_h + 0.5 * prm.head_flexion_deg)
    # ---- arms
    for i, side in enumerate(SIDES):
        col, sh, el, wr = ARM[side]
        sg = SIGN[side]
        ph_i = phl - (0.0 if i == 0 else prm.hs_offset)
        if side in prm.arm_posture:
            post = prm.arm_posture[side]
            for j in (col, sh, el, wr):
                if j in post:
                    R[:, j] = _spec_rot(post[j])[None]
            continue
        # the swing follows the gait amplitude, and the step plan where arm_swing_step_ref_m is set
        a_arm = amp if prm.arm_swing_step_ref_m is None else \
            amp * np.clip(L / float(prm.arm_swing_step_ref_m), 0.0, 1.0)
        A_sw = _side_at(prm.arm_swing_deg, i, t) * a_arm
        flex = _side_at(prm.arm_flexion_deg, i, t) - A_sw * np.cos(2 * np.pi * (ph_i - prm.arm_lag))
        if pump is not None:                     # the arms hang while the trunk pumps (an arm locked to a thorax that
            flex = flex + PUMP_ARM_SHARE * pump  # pitches forward points back: the shoulder flexes to take it back)
        # the abduction: held (a function of time allowed), plus the catch of the buckles, plus a share of the
        # trunk's lateral lean that the arm takes back (an arm locked to a leaning thorax swings out on one side and
        # in on the other)
        abd = _side_at(prm.arm_abduction_deg, i, t) + prm.buckle_arm_deg * buck_env + \
            sg * prm.arm_lean_follow * tlat_arm
        R[:, sh] = _rot_axis("x", -flex) @ _rot_axis("z", -sg * (90.0 - abd))
        e = _side_at(prm.elbow_deg, i, t) - _side_at(prm.elbow_swing_deg, i, t) * a_arm * \
            np.cos(2 * np.pi * (ph_i - prm.arm_lag - 0.04))
        R[:, el] = _rot_axis("y", -sg * e)
    # ---- overlays
    info = {"t": t, "phase": phi, "amplitude": amp, "heading_rad": psi, "stance": stance, "swing_weight": sw_w,
            "pitch_deg": pitch, "pivot": piv_out, "walk_line": c, "pelvis_rotation_deg": rot,
            "pelvis_obliquity_deg": obl, "pelvis_tilt_deg": tilt + prm.lordosis_deg, "footprints": footprints,
            "swings": swing_info,
            "stop": stop, "start_at_s": float(start_at_s), "turns": [tuple(float(x) for x in tr) for tr in turns],
            "buckles": buckles, "leg_guard": guard,
            "params": params_record(prm), "seed": seed, "rate": rate}
    if pump is not None:
        info["pump_deg"] = pump
    Rt = torch.from_numpy(R)
    for ov in overlays:
        for j, (Roff, frame) in ov(t, info).items():
            Ro = torch.as_tensor(np.asarray(Roff), dtype=F64)
            Rt[:, j] = Rt[:, j] @ Ro if frame == "child" else Ro @ Rt[:, j]
    if prm.clamp_upper:
        Rt = _clamp_to_table(Rt, UPPER)
    for i, side in enumerate(SIDES):                 # swing legs (no contact) kept inside the table too
        ks = torch.from_numpy(np.flatnonzero(sw_w[:, i] > 0))
        if len(ks):
            Rt[ks] = _clamp_to_table(Rt[ks], LEG[side][:3])
    b = Body(matrix_to_rot6d(Rt), torch.from_numpy(P - rest[0]), torch.zeros(T, 2, dtype=torch.bool),
             torch.zeros(16, dtype=F64) if betas is None else betas, gender, rate)
    b = replace(b, contact=contact_labels(b))
    info["ds_push"] = list(_ds_pushes)
    info["overlay_guard"] = list(_ov_bumps)
    info["stance_lift"] = {"rounds": int(_lift_round),
                           "max_deg": 0.0 if _lift is None else float(np.abs(_lift).max())}
    if _lift_round < STANCE_GUARD_ROUNDS:
        ex = _stance_excess(Rt, stance)
        if np.abs(ex).max() > 0.0:
            gain = np.where(ex > 0.0, STANCE_GUARD_GAIN, STANCE_GUARD_OFFAXIS_GAIN)
            lift = (np.zeros((T, 2)) if _lift is None else _lift) + gain * ex
            n = max(1, int(round(STANCE_GUARD_SMOOTH_S * rate)))
            lift = np.stack([_movmax_mean(np.maximum(lift[:, i], 0.0), n) -
                             _movmax_mean(np.maximum(-lift[:, i], 0.0), n) for i in range(2)], -1) * stance
            return generate(prm, duration_s, start_at_s, stop_at_s, turns, seed, rate, overlays, betas, gender,
                            path_fn, _ds_round=_ds_round, _ds_pushes=_ds_pushes, _ov_bumps=_ov_bumps, _lift=lift,
                            _lift_round=_lift_round + 1)
    if prm.leg_gap_m is not None and _ds_round < DS_PUSH_ROUNDS:
        push = _ds_contacts(b, info, prm, _ds_round)
        push = push + [p for p in land_push if not any(q["side"] == p["side"] and q["stance"] == p["stance"]
                                                       for q in push)]
        bumps = _overlay_swing_contacts(b, info, prm, _ds_round, geo) if overlays else []
        if push or bumps:
            q = replace(prm, footprint_offsets=tuple(prm.footprint_offsets) +
                        tuple((p["side"], p["stance"], p["outward_m"], 0.0) for p in push),
                        swing_extra_abd=tuple(prm.swing_extra_abd) +
                        tuple((x["side"], x["k_to"], x["deg"], x["at_share"]) for x in bumps))
            return generate(q, duration_s, start_at_s, stop_at_s, turns, seed, rate, overlays, betas, gender,
                            path_fn, _ds_round=_ds_round + 1, _ds_pushes=info["ds_push"] + push,
                            _ov_bumps=info["overlay_guard"] + bumps, _lift=_lift, _lift_round=_lift_round)
    return b, info


def _overlay_swing_contacts(body, info, prm, rnd, geo):
    """The swing guard after the overlays (module docstring): the guard measures each swing before the overlays are
    composed (a choreic leg jerk, chorea_legs); [{'round', 'side', 'k_to' (the swing's toe-off knot), 't_s',
    'hard_m', 'soft_m', 'deg', 'at_share'}] for each run of single-support knots of the final body where the legs pass
    into each other beyond the legcontact check: an abduction bump of that swing (swing_extra_abd, the guard's shape
    centred where the contact is deepest) of DS_PUSH_GAIN times what the gap lacks plus DS_MARGIN_M, read as an
    angle over the thigh's rest length."""
    st = np.asarray(info["stance"], bool)
    ks = np.flatnonzero(st.sum(1) == 1)
    if not len(ks):
        return []
    g = LCN.gap_from_pose(body.local_R().numpy()[ks], body.transl.numpy()[ks], body.betas.numpy(),
                          int(body.gender))
    hard, soft = g["hard_m"], g["soft_m"]
    bad = (hard < -LCN.TOL_M) | (soft < -LCN.SOFT_TOL_M)
    out = []
    for idx in np.split(np.arange(len(ks)), np.flatnonzero(np.diff(ks) > 1) + 1):
        if not bad[idx].any():
            continue
        lack = np.maximum(prm.leg_gap_m - hard[idx], -(LCN.SOFT_TOL_M - LEG_SOFT_MARGIN_M) - soft[idx])
        m = idx[int(np.argmax(lack))]
        k = int(ks[m])
        side = SIDES[0] if not st[k, 0] else SIDES[1]
        for w in info["swings"][side]:
            if w["k_to"] < k < w["k_hs"]:
                thigh = float(np.linalg.norm(geo.p[side]))
                out.append({"round": int(rnd), "side": side, "k_to": int(w["k_to"]), "t_s": k / body.rate,
                            "hard_m": float(hard[m]), "soft_m": float(soft[m]),
                            "deg": math.degrees(DS_PUSH_GAIN * (float(lack.max()) + DS_MARGIN_M) / thigh),
                            "at_share": (k - w["k_to"]) / (w["k_hs"] - w["k_to"])})
                break
    return out


def _ds_contacts(body, info, prm, rnd):
    """The double-support guard (module docstring): [{'round', 'side', 'stance' (index among that side's stances, as
    footprint_offsets counts them), 't_s', 'hard_m', 'soft_m', 'outward_m'}] for each run of double-support knots
    where the legs pass into each other beyond the check (a hard gap below -legcontact.TOL_M, or the inner thighs
    deeper than legcontact.SOFT_TOL_M): the foot that landed last is set down wider by DS_PUSH_GAIN times what the
    gap lacks (hard: leg_gap_m; soft: SOFT_TOL_M less LEG_SOFT_MARGIN_M) plus DS_MARGIN_M."""
    st = np.asarray(info["stance"], bool)
    ds = np.flatnonzero(st.all(1))
    if not len(ds):
        return []
    g = LCN.gap_from_pose(body.local_R().numpy()[ds], body.transl.numpy()[ds], body.betas.numpy(),
                          int(body.gender))
    hard, soft = g["hard_m"], g["soft_m"]
    bad = (hard < -LCN.TOL_M) | (soft < -LCN.SOFT_TOL_M)
    out = []
    cut = np.flatnonzero(np.diff(ds) > 1) + 1
    for idx in np.split(np.arange(len(ds)), cut):
        if not bad[idx].any():
            continue
        lack = np.maximum(prm.leg_gap_m - hard[idx], -(LCN.SOFT_TOL_M - LEG_SOFT_MARGIN_M) - soft[idx])
        m = idx[int(np.argmax(lack))]
        k = int(ds[m])
        land = None
        for side in SIDES:
            for j, f in enumerate(info["footprints"][side]):
                if f["k_hs"] <= k <= f["k_to"] and (land is None or f["k_hs"] > land[2]):
                    land = (side, j, f["k_hs"])
        if land is None:
            continue
        out.append({"round": int(rnd), "side": land[0], "stance": int(land[1]), "t_s": k / body.rate,
                    "hard_m": float(hard[m]), "soft_m": float(soft[m]),
                    "outward_m": float(DS_PUSH_GAIN * float(lack.max()) + DS_MARGIN_M)})
    return out


UPPER = (3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21)    # joints without a foot contact below them
CLAMP_MARGIN_DEG = 0.25
# The stance guard (open-body port, wave 5): a stance leg cannot be clamped into the table without moving its foot,
# so where a stance hip's extension or a stance ankle's dorsiflexion passes the anatomical table less
# CLAMP_MARGIN_DEG, the stance foot's pitch rises by STANCE_GUARD_GAIN times the larger excess (deg) at those knots,
# and where instead a stance ankle's off-axis components (y, z) pass it, the pitch falls by as much (the push-off's
# plantarflexion turns the foot's toe-out and inversion into them); the lift is smoothed (a moving maximum then a
# moving mean over STANCE_GUARD_SMOOTH_S of each sign, inside the stance), and the walk is generated again with it, up
# to STANCE_GUARD_ROUNDS times (info['stance_lift']): the heel is rolled by contact.roll on the sole's hull, so the
# contact stays exact; raising the heel brings the ankle up and forward over the toes, which takes the shank's
# forward lean off the ankle and the thigh's backward lean off the hip
STANCE_GUARD_ROUNDS = 6
STANCE_GUARD_GAIN = 1.5      # [module] deg of lift per deg of excess: the excess falls by less than the lift (the knee
#                              takes part of it), so a gain above 1 converges in fewer rounds
STANCE_GUARD_OFFAXIS_GAIN = 4.0   # [module] deg of pitch fall per deg of an off-axis excess: the ankle's y and z
#                              follow the pitch weakly (the choreic walk's toe-off, wave 5), so their gain is larger
STANCE_GUARD_SMOOTH_S = 0.08  # [module]
STANCE_GUARD_JOINTS = {"hip": ((1, 2), 0, "hi"), "ankle": ((7, 8), 0, "lo")}   # hip extension +x, dorsiflexion -x
STANCE_GUARD_OFFAXIS = ((7, 8), (1, 2))      # the ankles' y and z components: the pitch falls where they pass


def _movmax_mean(x, n):
    """A moving maximum then a moving mean of width n (never below x where x is locally flat)."""
    pad = (n // 2, n - 1 - n // 2)
    hi = np.lib.stride_tricks.sliding_window_view(np.pad(x, pad, mode="edge"), n).max(1)
    return np.convolve(np.pad(hi, pad, mode="edge"), np.ones(n) / n, mode="valid")


def _stance_excess(Rt, stance):
    """(T, 2) deg, signed: per knot and leg, the larger excess of its stance hip's extension and its stance ankle's
    dorsiflexion past the anatomical table less CLAMP_MARGIN_DEG (positive: the heel to rise); where neither passes,
    minus the larger excess of the stance ankle's y and z components (the pitch to fall); 0 inside and in swing."""
    from ..body.rotations import matrix_to_axis_angle
    from .body import joint_limits
    lim = joint_limits()
    up = np.zeros(stance.shape)
    for joints, ax, side in STANCE_GUARD_JOINTS.values():
        c = torch.rad2deg(matrix_to_axis_angle(Rt[:, list(joints)]))[..., ax].numpy()      # (T, 2)
        for i, j in enumerate(joints):
            if side == "hi":
                e = c[:, i] - (lim.hi[j, ax] - CLAMP_MARGIN_DEG)
            else:
                e = (lim.lo[j, ax] + CLAMP_MARGIN_DEG) - c[:, i]
            up[:, i] = np.maximum(up[:, i], np.where(stance[:, i], np.maximum(e, 0.0), 0.0))
    down = np.zeros(stance.shape)
    joints, axes = STANCE_GUARD_OFFAXIS
    c = torch.rad2deg(matrix_to_axis_angle(Rt[:, list(joints)])).numpy()                # (T, 2, 3)
    for i, j in enumerate(joints):
        for ax in axes:
            e = np.maximum(c[:, i, ax] - (lim.hi[j, ax] - CLAMP_MARGIN_DEG),
                           (lim.lo[j, ax] + CLAMP_MARGIN_DEG) - c[:, i, ax])
            down[:, i] = np.maximum(down[:, i], np.where(stance[:, i], np.maximum(e, 0.0), 0.0))
    return np.where(up > 0.0, up, -down)


def _clamp_to_table(Rt, joints):
    """Local rotations Rt (T, 22, 3, 3) whose rows `joints` have every axis-angle component brought inside the
    SP1 joint-limit table (a margin of CLAMP_MARGIN_DEG): the overlays and held postures of the trunk, head and arms
    never leave the table; no foot depends on these joints."""
    from ..body.rotations import matrix_to_axis_angle
    from .body import joint_limits
    lim = joint_limits()
    j = list(joints)
    aa = torch.rad2deg(matrix_to_axis_angle(Rt[:, j]))
    lo = torch.as_tensor(lim.lo[j] + CLAMP_MARGIN_DEG, dtype=F64)
    hi = torch.as_tensor(lim.hi[j] - CLAMP_MARGIN_DEG, dtype=F64)
    out = Rt.clone()
    out[:, j] = axis_angle_to_matrix(torch.deg2rad(torch.maximum(torch.minimum(aa, hi), lo)))
    return out


def params_record(prm):
    """JSON-ready parameters (functions of time named, not evaluated)."""
    out = {}
    for k, v in asdict(prm).items():
        if callable(v):
            out[k] = getattr(v, "__doc__", None) or "function of time"
        elif isinstance(v, tuple) and any(callable(x) for x in v):
            out[k] = [getattr(x, "__doc__", None) or "function of time" if callable(x) else x for x in v]
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------- the exam course
def course(prm=NORMAL, distance_m=10.0, turn_deg=180.0, turn_s=2.4, turn_share=None, back=True, lead_s=1.0,
           tail_s=1.2, seed=0, overlays=(), turn_pause_s=0.0, path_fn=None):
    """(Body, info): the walk of MDS-UPDRS 3.10 on the program's body: standing lead_s, walk distance_m away,
    turn turn_deg (positive to the left) over turn_s, walk distance_m back (back False: no turn and no return),
    stop, stand tail_s. Durations follow the parameters' nominal speed (steps times cadence). path_fn: generate's
    path hook (a veering walk)."""
    v = float(np.mean(_at(prm.step_m, np.linspace(0, 60, 61)))) * float(np.mean(_at(prm.cadence_spm,
                                                                                np.linspace(0, 60, 61)))) / 60.0
    t1 = lead_s + 0.5 * prm.start_s + distance_m / v
    turns = []
    if back:
        share = prm.turn_speed_share if turn_share is None else turn_share
        turns = [(t1 - 0.2 * turn_s, turn_deg, turn_s, share)]
        t2 = t1 + turn_s * (1.0 + 0.0) + turn_pause_s + distance_m / v
    else:
        t2 = t1
    steps_t = prm.stop_steps * 60.0 / float(np.mean(_at(prm.cadence_spm, np.linspace(0, 60, 61))))
    dur = t2 + 2.5 * steps_t + tail_s
    b, info = generate(prm, duration_s=round(dur, 2), start_at_s=lead_s, stop_at_s=t2, turns=turns, seed=seed,
                       overlays=overlays, path_fn=path_fn)
    info["course"] = {"distance_m": distance_m, "turn_deg": turn_deg if back else 0.0, "turn_s": turn_s,
                      "nominal_speed_m_s": v}
    return b, info


# ---------------------------------------------------------------- re-measurement
def _runs(mask):
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.diff(m.astype(np.int8))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1))


def sole_cloud_points(body, side, fk=None):
    """(T, M, 3) the rigid sole cloud of a foot (contact.body_sole_points)."""
    return CT.body_sole_points(body, side, fk)


SLIP_TOUCH_M = PIVOT_NEAR_M   # a sole cloud point this close to the knot's lowest one touches the floor (stance_slip)


def stance_slip(body, info, fk=None):
    """{'p95_m_s', 'max_m_s', 'n'}: the horizontal speed of each stance foot's contact point (the least-moving of
    the sole cloud points within SLIP_TOUCH_M of the lowest one, each followed as the same cloud point from knot to
    knot) over the knots the generator holds in stance (info['stance']); a foot rolling or pivoting without slip
    keeps one touching point still and reads 0, a foot gliding moves them all. (The lowest point alone was read
    before wave SH1d: on a flat foot pivoting on its ball in a turn, it is a toe point circling the ball.)"""
    fk = body.fk() if fk is None else fk
    st = np.asarray(info["stance"], bool)
    sp = []
    for i, side in enumerate(SIDES):
        X = sole_cloud_points(body, side, fk)
        k = np.flatnonzero(st[1:-1, i] & st[:-2, i] & st[2:, i]) + 1
        if not len(k):
            continue
        touch = X[k, :, 1] <= X[k, :, 1].min(1, keepdims=True) + SLIP_TOUCH_M
        v = (X[k + 1] - X[k - 1]) * body.rate / 2.0
        sp.append(np.where(touch, np.hypot(v[..., 0], v[..., 2]), np.inf).min(1))
    s = np.concatenate(sp) if sp else np.zeros(0)
    return {"p95_m_s": float(np.percentile(s, 95)) if len(s) else 0.0, "max_m_s": float(s.max()) if len(s) else 0.0,
            "n": int(len(s))}


def _angle(u, v):
    c = (u * v).sum(-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1))
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


def _sagittal(v, psi):
    """Angle (deg) of vectors v (T, 3) from the downward vertical in the sagittal plane of the heading psi (T,),
    positive forward."""
    f = v[:, 0] * np.sin(psi) + v[:, 2] * np.cos(psi)
    return np.degrees(np.arctan2(f, -v[:, 1]))


def knee_flexion_deg(J, psi, side, rest):
    """(T,) deg: the sagittal angle between the thigh (hip to knee) and the shank (knee to ankle) lines in the
    heading frame, flexion positive, minus the same angle in the rest pose (the rest leg reads 0)."""
    h, kn, a, _ = LEG[side]
    k = _sagittal(J[:, kn] - J[:, h], psi) - _sagittal(J[:, a] - J[:, kn], psi)
    r0 = _sagittal((rest[kn] - rest[h])[None], np.zeros(1)) - _sagittal((rest[a] - rest[kn])[None], np.zeros(1))
    return k - float(r0[0])


def measure_gait(body, floor_tol_m=FLOOR_TOL_M, turn_deg=15.0, skip=2, turn_rate_deg_s=10.0, turn_margin_s=1.0,
                 touch_tol_m=STANCE_READ_TOL_M):
    """Gait measures read back from a body (module docstring), JSON-ready; arrays of per-step values under
    'steps'. Steady steps: neither among the first or last `skip` heel strikes of the clip nor turning more than
    turn_deg over the step."""
    J, G = (x.numpy() for x in body.fk())
    rate = body.rate
    T = body.T
    t = np.arange(T) / rate
    clouds = {s: sole_cloud_points(body, s, (torch.from_numpy(J), torch.from_numpy(G))) for s in SIDES}
    lows = np.stack([clouds[s][..., 1].min(-1) for s in SIDES], -1)
    floor = float(np.median(lows.min(1)))         # at almost every knot a foot stands: the floor
    st = lows <= floor + floor_tol_m
    touch = lows <= floor + touch_tol_m           # the stance share and the double support: the generator's touch
    from ..body.rotations import yaw_angle
    psi = np.unwrap(yaw_angle(body.local_R()[:, 0]).numpy())
    events = []
    for i, side in enumerate(SIDES):
        for a, b in _runs(st[:, i]):
            if b - a < max(3, int(round(0.08 * rate))):    # a toe scuffing the floor in early swing, not a stance
                continue
            seg = slice(a, b + 1)
            fz = G[seg, LEG[side][2]][:, :, 2]
            pitch = np.degrees(np.arctan2(-fz[:, 1], np.hypot(fz[:, 0], fz[:, 2])))     # heel up positive
            kf = a + int(np.argmin(np.abs(pitch)))
            events.append({"side": side, "i": i, "k_hs": int(a), "k_to": int(b), "t_hs": a / rate, "t_to": b / rate,
                           "foot": J[kf, LEG[side][2]][[0, 2]].copy(), "pitch_hs_deg": float(pitch[0]),
                           "pitch_to_deg": float(pitch[-1]), "flat": bool(np.abs(pitch).min() < 3.0),
                           "start_clipped": a == 0, "end_clipped": b == T - 1})
    events.sort(key=lambda e: e["k_hs"])
    hs = [e for e in events if not e["start_clipped"]]
    out = {"floor_m": floor, "n_stance_runs": len(events)}
    # steps: consecutive heel strikes of alternating feet
    steps = []
    for e0, e1 in zip(hs, hs[1:]):
        if e0["side"] == e1["side"]:
            continue
        dpsi = abs(math.degrees(psi[e1["k_hs"]] - psi[e0["k_hs"]]))
        steps.append({"side": e1["side"], "t": e1["t_hs"], "time_s": e1["t_hs"] - e0["t_hs"], "turn_deg": dpsi,
                      "k0": e0["k_hs"], "k1": e1["k_hs"], "e0": e0, "e1": e1})
    pel = J[:, 0, [0, 2]]
    for k, s in enumerate(steps):
        # progression direction: the pelvis's path over the step (its mean heading)
        h = psi[s["k0"]:s["k1"] + 1].mean()
        f = np.array([math.sin(h), math.cos(h)])
        lft = np.array([math.cos(h), -math.sin(h)])
        d = s["e1"]["foot"] - s["e0"]["foot"]
        s["length_m"] = float(d @ f)
        s["width_m"] = float(abs(d @ lft))
    # steady: not among the first or last `skip` steps, no heel strike within turn_margin_s of a turn (the heading
    # smoothed over 1 s turning faster than turn_rate_deg_s), and less than turn_deg of heading change over the step
    n2 = int(round(2.0 * rate))                       # two strides: the pelvic rotation averages out
    psi_sm = np.convolve(np.pad(psi, (n2 // 2, n2 - 1 - n2 // 2), mode="edge"), np.ones(n2) / n2, mode="valid")
    turning = np.abs(np.degrees(np.gradient(psi_sm, 1.0 / rate))) > turn_rate_deg_s
    m_k = int(round(turn_margin_s * rate))
    near_turn = np.convolve(turning.astype(float), np.ones(2 * m_k + 1), mode="same") > 0
    steady = [s for k, s in enumerate(steps) if skip <= k < len(steps) - skip and s["turn_deg"] <= turn_deg
              and not near_turn[s["k0"]] and not near_turn[s["k1"]]]
    out["n_steps"] = len(steps)
    out["n_steady"] = len(steady)

    def stat(x):
        x = np.asarray(x, dtype=np.float64)
        if not len(x):
            return {"median": None, "mean": None, "sd": None, "cv": None, "n": 0}
        m = float(x.mean())
        return {"median": float(np.median(x)), "mean": m, "sd": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
                "cv": float(x.std(ddof=1) / abs(m)) if len(x) > 1 and m else None, "n": int(len(x))}
    out["step_length_m"] = stat([s["length_m"] for s in steady])
    out["step_width_m"] = stat([s["width_m"] for s in steady])
    out["step_time_s"] = stat([s["time_s"] for s in steady])
    for side in SIDES:
        out[f"step_length_{side}_m"] = stat([s["length_m"] for s in steady if s["side"] == side])
        out[f"step_time_{side}_s"] = stat([s["time_s"] for s in steady if s["side"] == side])
    out["cadence_spm"] = 60.0 / out["step_time_s"]["mean"] if steady else None
    if len(steady) >= 2:
        k0, k1 = steady[0]["k0"], steady[-1]["k1"]
        path = np.linalg.norm(np.diff(pel[k0:k1 + 1], axis=0), axis=1).sum()
        out["speed_m_s"] = float(path / ((k1 - k0) / rate))
        straight = np.linalg.norm(pel[k1] - pel[k0])
        out["speed_straight_m_s"] = float(straight / ((k1 - k0) / rate))
        win = slice(k0, k1 + 1)
    else:
        out["speed_m_s"] = None
        win = slice(0, T)
    # stance share per side over steady cycles
    for i, side in enumerate(SIDES):
        ev = [e for e in hs if e["side"] == side and not e["end_clipped"]]
        sh = []
        for e0, e1 in zip(ev, ev[1:]):
            if win.start <= e0["k_hs"] and e1["k_hs"] <= win.stop:
                tr = touch[e0["k_hs"]:e0["k_to"] + 1, i]
                on = np.flatnonzero(tr)
                k_to = e0["k_to"]
                if len(on):                                    # the touch's first lift-off after it begins
                    off = np.flatnonzero(~tr[on[0]:])
                    k_to = e0["k_hs"] + int(on[0]) + (int(off[0]) - 1 if len(off) else len(tr) - 1 - int(on[0]))
                sh.append((k_to - e0["k_hs"]) / (e1["k_hs"] - e0["k_hs"]))
        out[f"stance_share_{side}"] = stat(sh)
    # knee flexion (angle between thigh and shank vectors), loading and swing peaks
    rest = body.rest().numpy()
    for i, side in enumerate(SIDES):
        h, kn, a, ft = LEG[side]
        kang = knee_flexion_deg(J, psi, side, rest)
        ev = [e for e in hs if e["side"] == side and not e["end_clipped"] and win.start <= e["k_hs"] < win.stop]
        load, swing, mtc, ph = [], [], [], []
        for e0 in ev:
            nxt = [e for e in events if e["side"] == side and e["k_hs"] > e0["k_hs"]]
            if not nxt:
                continue
            e1 = nxt[0]
            cyc = e1["k_hs"] - e0["k_hs"]
            load.append(kang[e0["k_hs"]:e0["k_hs"] + int(0.25 * cyc)].max())
            swing.append(kang[e0["k_to"]:e1["k_hs"] + 1].max())
            sw = np.arange(e0["k_to"], e1["k_hs"] + 1)
            s = (sw - sw[0]) / max(len(sw) - 1, 1)
            mid = sw[(s >= 0.25) & (s <= 0.85)]
            if len(mid):
                mtc.append(lows[mid, i].min() - floor)
            ph.append(-e0["pitch_hs_deg"])
        out[f"knee_loading_{side}_deg"] = stat(load)
        out[f"knee_swing_{side}_deg"] = stat(swing)
        out[f"clearance_{side}_m"] = stat(mtc)
        out[f"contact_pitch_{side}_deg"] = stat(ph)
    # arms: shoulder flexion in the heading frame (upper arm against the trunk's vertical), elbow flexion,
    # wrist excursion against the pelvis
    psi_s = _smooth(psi, int(round(1.0 * rate)))          # the walking direction (the pelvic rotation removed)
    c, s_ = np.cos(psi_s), np.sin(psi_s)
    for i, side in enumerate(SIDES):
        col, sh, el, wr = ARM[side]
        ua = J[:, el] - J[:, sh]
        fwd_c = ua[:, 0] * s_ + ua[:, 2] * c
        ang = np.degrees(np.arctan2(fwd_c, -ua[:, 1]))
        ea = _angle(J[:, el] - J[:, sh], J[:, wr] - J[:, el]) - float(_angle((rest[el] - rest[sh])[None],
                                                                            (rest[wr] - rest[el])[None])[0])
        wrel = (J[:, wr] - J[:, 0])
        wf = wrel[:, 0] * s_ + wrel[:, 2] * c
        out[f"shoulder_range_{side}_deg"] = float(np.percentile(ang[win], 98) - np.percentile(ang[win], 2))
        out[f"shoulder_mean_{side}_deg"] = float(ang[win].mean())
        out[f"elbow_mean_{side}_deg"] = float(ea[win].mean())
        out[f"elbow_range_{side}_deg"] = float(np.percentile(ea[win], 98) - np.percentile(ea[win], 2))
        out[f"wrist_excursion_{side}_m"] = float(np.percentile(wf[win], 98) - np.percentile(wf[win], 2))
    L_, R_ = out["shoulder_range_left_deg"], out["shoulder_range_right_deg"]
    out["arm_swing_asymmetry_pct"] = float((45.0 - math.degrees(math.atan2(min(L_, R_), max(L_, R_)))) / 90.0 * 100.0) \
        if max(L_, R_) > 0 else 0.0
    # trunk: the pelvis-to-neck line against vertical, sagittal (heading frame) and lateral
    tr = J[:, 12] - J[:, 0]
    tf = tr[:, 0] * s_ + tr[:, 2] * c
    tlat = tr[:, 0] * c - tr[:, 2] * s_
    r_tr = rest[12] - rest[0]
    out["trunk_lean_deg"] = float(np.degrees(np.arctan2(tf[win], tr[win, 1])).mean()
                                  - math.degrees(math.atan2(r_tr[2], r_tr[1])))
    lat = np.degrees(np.arctan2(tlat[win], tr[win, 1]))
    out["trunk_lateral_range_deg"] = float(np.percentile(lat, 98) - np.percentile(lat, 2))
    py = J[win, 0, 1]
    out["pelvis_vertical_range_m"] = float(np.percentile(py, 98) - np.percentile(py, 2))
    # pelvis lateral excursion about the walk line (heading frame, detrended)
    dev = pel - np.stack([_smooth(pel[:, 0], int(round(1.2 * rate))), _smooth(pel[:, 1], int(round(1.2 * rate)))], -1)
    plat = (dev[:, 0] * c - dev[:, 1] * s_)[win]
    out["pelvis_lateral_range_m"] = float(np.percentile(plat, 98) - np.percentile(plat, 2))
    out["double_support_share"] = float((touch[win, 0] & touch[win, 1]).mean())
    out["steps"] = [{k: (v if not isinstance(v, np.ndarray) else v.tolist()) for k, v in s.items()
                     if k not in ("e0", "e1")} for s in steps]
    return out


def healthy_ranges():
    """{summary key: (lo, hi, source)}: the ranges the healthy walk is checked against (tests/test_synth_gait.py).
    Sourced bounds are the cited value's band (mean plus or minus two SD where an SD was read); 'module' bounds are
    this module's plausibility bands, not taken from a source."""
    R = {"speed_m_s": (1.15, 1.50, "bohannon2011 (strata 20 to 69: about 1.24 to 1.43), pirker2017 1.4"),
         "cadence_spm": (107.6 - 1.96 * 4.3, 114.5 + 1.96 * 7.6, "chung2010 men 107.6 (SD 4.3) to women 114.5 (SD "
                                                                  "7.6), REFWALK; perry2010 113"),
         "step_length_m": (0.60, 0.80, "perry2010 stride 1.41 m (step 0.705)"),
         "step_width_m": (0.05, 0.13, "module band (pirker2017 8 to 10 cm in the elderly; the neutral mesh walks "
                                      "on STEP_WIDTH_M, a module value)"),
         "trunk_lean_deg": (-0.2 - 1.96 * 3.6, -0.2 + 1.96 * 3.6, "chung2010 -0.2 (SD 3.6), REFWALK"),
         "arm_swing_asymmetry_pct": (0.0, 5.1 + 1.96 * 4.0, "lewek2010 controls 5.1 (SD 4.0), REFWALK"),
         "step_length_cv": (0.0, 0.05, "module"),
         "step_time_cv": (0.0, 0.04, "module"),
         "pelvis_vertical_range_m": (0.02, 0.06, "module")}
    for side in SIDES:
        R[f"stance_share_{side}"] = (0.57, 0.66, "perry2010, pirker2017: about 60 percent")
        R[f"knee_swing_{side}_deg"] = (64.9 - 2 * 5.6, 64.9 + 2 * 5.6, "knee_norm 64.9 (SD 5.6)")
        R[f"knee_loading_{side}_deg"] = (16.0 - 2 * 5.9, 16.0 + 2 * 5.9, "knee_norm 16 (SD 5.9)")
        R[f"clearance_{side}_m"] = (0.005, 0.03, "module")
        R[f"shoulder_range_{side}_deg"] = (24.5 - 1.96 * 8.5, 24.5 + 1.96 * 8.5, "kahn2019 24.5 (SD 8.5), REFWALK")
        R[f"elbow_range_{side}_deg"] = (29.7 - 2 * 10.2, 29.7 + 2 * 10.2, "elbow2023 29.7 (SD 10.2)")
        R[f"elbow_mean_{side}_deg"] = (11.9 - 1.96 * 6.0, 11.9 + 1.96 * 6.0, "kahn2019 11.9 (SD 6.0), the REFWALK "
                                                                          "range (reports/synth/reference_walk)")
    return R


def summary(m):
    """The headline numbers of measure_gait, flat (JSON-ready)."""
    g = lambda k: m[k]["median"] if isinstance(m.get(k), dict) else m.get(k)  # noqa: E731
    out = {"speed_m_s": m["speed_m_s"], "cadence_spm": m["cadence_spm"], "step_length_m": g("step_length_m"),
           "step_width_m": g("step_width_m"), "step_length_cv": m["step_length_m"]["cv"],
           "step_time_cv": m["step_time_s"]["cv"], "step_width_sd_m": m["step_width_m"]["sd"],
           "double_support_share": m["double_support_share"], "trunk_lean_deg": m["trunk_lean_deg"],
           "pelvis_vertical_range_m": m["pelvis_vertical_range_m"],
           "pelvis_lateral_range_m": m["pelvis_lateral_range_m"],
           "arm_swing_asymmetry_pct": m["arm_swing_asymmetry_pct"]}
    for side in SIDES:
        for k in ("knee_loading_{}_deg", "knee_swing_{}_deg", "clearance_{}_m", "contact_pitch_{}_deg",
                  "stance_share_{}"):
            out[k.format(side)] = g(k.format(side))
        for k in ("shoulder_range_{}_deg", "elbow_mean_{}_deg", "elbow_range_{}_deg", "wrist_excursion_{}_m"):
            out[k.format(side)] = m[k.format(side)]
        out[f"step_length_{side}_m"] = g(f"step_length_{side}_m")
    return out


# ---------------------------------------------------------------- overlays
CHOREA_GRADE = 2              # the walking choreic patient's upper-body chorea on the UHDRS map of involuntary.py
CHOREA_LEG_DEG = 15.0         # [module] largest hip flexion of a choreic swing-leg fragment (the knee 1.2 times it)


def chorea_upper(seed, grade=CHOREA_GRADE, fingers=True):
    """post(body) -> (body, record) of a walking clip: the chorea engine of involuntary.py (discrete movement
    fragments flowing from one posture to the next, TIMING 'chorea', in random bouts; involuntary.compose_clip) on
    the trunk, neck, both arms and the fingers, at the UHDRS grade's declared amplitude and share of time
    (involuntary.UHDRS, AMP, DUR), each region in its own bouts; the offsets are kept inside the joint-limit table.
    The legs are left to the gait (exact contacts) and to chorea_legs."""
    def post(body):
        from . import involuntary as IV
        rng = np.random.default_rng(seed)
        a_term, d_term = IV.UHDRS[grade]
        level, share = IV.AMP[a_term], IV.DUR[d_term]

        def iv():
            return IV.bouts(body.duration, share, rng)
        plan = [dict(region="trunk", channels=IV.TRUNK_CH, level=level, intervals=iv(), kind="events",
                     timing="chorea", gains=IV.TRUNK_GAINS),
                dict(region="neck", channels=IV.NECK_CH, level=level, intervals=iv(), kind="events",
                     timing="chorea", gains=IV.NECK_GAINS)]
        fing = []
        for side in SIDES:
            iv_s = iv()
            plan.append(dict(region=f"arm_{side}", channels=IV.arm_ch(side), level=level, intervals=iv_s,
                             kind="events", timing="chorea", gains=IV.arm_gains(side)))
            if fingers:
                fing.append({"side": side, "kind": "events", "level": level, "intervals": iv_s})
        # the contact guard of involuntary.py on (guard=True): a choreic hand that would pass into the thigh or the
        # trunk takes back its own movement there (without it the right index tip came within 10 mm of the femur's
        # axis in the walking clip, 2026-11-03 review)
        out, rec = IV.compose_clip(body, plan, rng, None, None, fing, guard=True)
        return out, {"chorea": {"engine": "involuntary.compose_clip", "uhdrs_grade": grade, "amplitude_term": a_term,
                                "duration_term": d_term, "level_share_of_rom": level, "share_of_time": share,
                                "regions": {k: {kk: v.get(kk) for kk in ("level", "gain", "reachable",
                                                                         "intervals_s", "n_fragments")}
                                            for k, v in rec["regions"].items()},
                                "contact_guard": rec.get("guard")}}
    post.__doc__ = f"chorea engine of involuntary.py on the upper body, UHDRS grade {grade}, seed {seed}"
    return post


def chorea_legs(seed, grade=CHOREA_GRADE, leg_deg=CHOREA_LEG_DEG):
    """An overlay (generate's hook): choreic hip and knee flexion fragments of the swing legs (the fragment
    engine of involuntary.py, TIMING 'chorea', in random bouts at the grade's share of time), each fragment's size
    up to leg_deg of hip flexion and 1.2 leg_deg of knee flexion, weighted by info['swing_weight'] so that the stance
    feet stay planted [module amplitudes]."""
    def ov(t, info):
        from . import involuntary as IV
        rng = np.random.default_rng(seed)
        share = IV.DUR[IV.UHDRS[grade][1]]
        sw = info["swing_weight"]
        out = {}
        for i, (h, kn) in enumerate(((1, 4), (2, 5))):
            ev = IV.fragment_events(IV.bouts(float(t[-1]), share, rng), ["flex"], rng, IV.TIMING["chorea"])
            x = IV.render_fragments(ev, ["flex"], t)[:, 0]
            jerk = leg_deg * np.abs(x) * sw[:, i]
            out[h] = (_rot_axis("x", -jerk), "parent")
            out[kn] = (_rot_axis("x", 1.2 * jerk), "child")
        return out
    ov.__doc__ = f"choreic swing-leg fragments, seed {seed}, up to {leg_deg} deg"
    return ov


def chorea_overlay(seed, amp_deg=12.0, hz=1.2, leg_deg=15.0, share=0.6):
    """An overlay (generate's hook) of smooth random movements, the former placeholder of the choreic gait (the
    catalogue clip now uses chorea_upper and chorea_legs, the chorea engine of involuntary.py): smooth random
    rotations (white noise low-passed below hz) of the trunk, neck, shoulders, elbows and wrists, switched on and
    off by a smooth random gate (on about `share` of the time), and random hip and knee flexion jerks during swing
    only (info['swing_weight'], so that the stance feet stay planted). Amplitudes in degrees (standard deviation
    of each component) [module: no published amplitude was read]."""
    def ov(t, info):
        rng = np.random.default_rng(seed)
        gate = np.clip(smooth_noise(rng, t, 0.4, 1.0) + (share - 0.5) * 2.0, 0.0, 1.0)
        out = {}
        plan = {3: ("y", "z"), 6: ("z", "x"), 12: ("y", "z"), 16: ("x", "z"), 17: ("x", "z"), 18: ("y",),
                19: ("y",), 20: ("x", "z"), 21: ("x", "z")}
        for j, axes in plan.items():
            M = np.repeat(np.eye(3)[None], len(t), 0)
            for ax in axes:
                a = 0.5 if j in (3, 6) else 1.0
                M = M @ _rot_axis(ax, gate * smooth_noise(rng, t, hz, amp_deg * a))
            out[j] = (M, "child")
        sw = info["swing_weight"]
        for i, (h, kn) in enumerate(((1, 4), (2, 5))):
            jerk = np.abs(smooth_noise(rng, t, hz * 1.5, leg_deg)) * sw[:, i] * gate
            out[h] = (_rot_axis("x", -jerk), "parent")
            out[kn] = (_rot_axis("x", 1.2 * jerk), "child")
        return out
    ov.__doc__ = f"chorea overlay, seed {seed}, {amp_deg} deg at {hz} Hz, legs {leg_deg} deg in swing"
    return ov


def stepped(values, times):
    """A function of time that eases (raised cosine) from values[i] to values[i + 1] between times[i] and
    times[i + 1], held before the first and after the last."""
    def f(t):
        t = np.asarray(t, dtype=np.float64)
        out = np.full(t.shape, float(values[0]))
        for (a, b), (va, vb) in zip(zip(times, times[1:]), zip(values, values[1:])):
            out = np.where(t >= a, va + (vb - va) * _raised(t, a, b - a), out)
        return out
    f.__doc__ = f"eases through {list(values)} at {list(times)} s"
    return f


# ---------------------------------------------------------------- the catalogue (task CAT-GAIT)
SINCE = "catalogue_v1"
SEED = 20261102
VIEWS = ("track_left", "examiner")
LEVEL0 = "gait_E8_level0"
# the Wernicke-Mann arm (left): the upper arm adducted against the trunk and internally rotated, the elbow flexed
# about 80 deg, the forearm across the front of the lower chest and pronated (the palm toward the trunk), the wrist
# flexed 45 deg; the fingers flexed into the palm by WM_HAND_SHARE of the fist (wm_hand). Solved once on the standing
# neutral body (postures.arm_ik_in_table: the wrist 0.17 m medial, 0.27 m below and 0.20 m in front of the shoulder
# joint, the palm toward the trunk and down; then the wrist flexed about its palmar axis), every row inside the
# joint-limit table, and written here as x, y, z rotations [module; pirker2017 describes the posture]. The former
# values (shoulder 28 deg about x, elbow 95 deg) left the forearm pointing forward, the wrist and fingers straight.
WM_ARM = {16: [("x", 55.5), ("y", -16.0), ("z", -87.6)], 18: [("y", -79.2)],
          20: [("x", 13.2), ("y", 7.7), ("z", -44.6)]}
WM_HAND_SHARE = 0.85                         # [module] the spastic fingers flexed short of a full fist


def wm_hand(side="left", share=WM_HAND_SHARE):
    """post(body) of a course clip: the side's fingers flexed toward the fist of handtasks.FIST (thumb included) by
    share, held through the clip (a HandBody)."""
    def post(body):
        from . import hands as H
        from .handtasks import FIST
        pose = {}
        for f, vals in FIST.items():
            for lv, v in enumerate(vals):
                pose[(side, f, lv)] = tuple(share * x for x in v) if isinstance(v, tuple) else share * v
        R = H.hand_rotations(pose)
        return H.hands_of(body).with_hands(R[None].expand(body.T, -1, -1, -1))
    post.__doc__ = f"{side} fingers flexed to {share} of the fist"
    return post
NOT_RENDERED = {
    "MDS-UPDRS 3.10 grades 3 and 4": "grade 3 needs an assistance device and grade 4 another person's help or cannot "
                                     "walk: the body has no device and no second person",
    "MDS-UPDRS 3.12 grades 3 and 4": "grade 3 is a fall caught by the examiner, grade 4 a spontaneous loss of "
                                     "balance: no examiner is drawn and no fall is simulated",
    "SARA gait 4 to 8": "intermittent wall support, a stick, a person's arm or no walking: no wall, device or person",
    "sensory ataxia: gaze and eye closure": "the sensory ataxic clips (gap wave) look at the feet by a head flexion "
                                            "only (the body has no eyes) and state the eye closure in the caption "
                                            "and params['frame_text'] (the viewer does not burn it into the frames)",
    "vestibular gait: eye closure": "stated in the caption and params['frame_text'], not visible",
    "vestibular gait: the veer on screen": "in the body and read back in the truth (path_drift), but no camera of the "
                                           "shared viewer shows it from the start heading: the examiner and track "
                                           "cameras align with the walk's start-to-end chord, the fixed views (front, "
                                           "back, top) with the heading at mid-clip, and in each the pelvis first "
                                           "moves toward the body's right in the image (eyes closed: 35 to 93 px of "
                                           "960) before coming back, a veer to the left read as one to the right; the "
                                           "clips keep the side track view only (slower, wider, the arms out) until "
                                           "a camera along the start heading exists (a shared change)",
    "PSP gait: backward instability": "the PSP clip walks without a perturbation; retropulsion is the pull-test "
                                      "clips' (gait_E7p_retropulsion_g*), and no fall is simulated",
    "functional gait: incongruence across tasks": "the fluctuating clip changes within one walk; walking backward, "
                                                  "running, sideways or on a swivel chair (nonnekes2020) are not "
                                                  "built",
    "dromedary gait: grades": "one described level; the BFMDRS trunk anchors were not read",
    "foot slap sound": "no audio",
    "scissoring with the legs crossed past each other": "the scissoring clips bring the knees together, touching at "
                                                        "most (legcontact), and land the feet on the midline or "
                                                        "near it; feet crossed further, or the thighs crossing, "
                                                        "would pass the legs of the rigid body model through each "
                                                        "other (the swinging foot through the stance foot, the "
                                                        "knees through each other at toe-off)",
}


def _prm(**kw):
    return replace(NORMAL, **kw)


def _two(side, a, b):
    """(left, right) with a on side and b on the other."""
    return (a, b) if side == "left" else (b, a)


LOW_FLAT_ANKLE_DEG = 20.0      # [module] the low flat-footed swings (3.10 grade 2, the walking on ice): the ankle
#                                dorsiflexed this far early in swing (swing_ankle_share 0.3), jarm2022's dorsiflexion,
#                                so that Anny's longer forefoot clears the floor by the ankle rather than by a lift that
#                                flexes the knee twice as much (open-body port, wave 8 second part, D162: at 16 and 14
#                                deg the grade 2 swing knees read 53.0 and 48.8 deg, 4.23 apart, and the walking on
#                                ice's right knee 47.4 deg; scripts/openbody_wave8b_checks.py 'low_flat')
LOW_FLAT_DIP_SYNERGY = (1.0, 1.0, 2.0)   # [module] the 3.10 grade 2 walk's dip lift (dip_synergy): a toe dipping in
#                                early swing lifted by the ankle first (the default lift_synergy flexes the knee twice)


def parkinsonian(grade, side="left"):
    """(GaitParams, course kwargs) of MDS-UPDRS 3.10 grade 0, 1 or 2, side the more affected (specs docstring)."""
    if grade == 0:
        return NORMAL, {}
    if grade == 1:
        return _prm(step_m=0.60, cadence_spm=116.0, arm_swing_deg=_two(side, 8.0, 14.0),
                    elbow_deg=_two(side, 24.0, 21.0), trunk_lean_deg=9.0, head_flexion_deg=4.0,
                    clearance_m=(0.012, 0.012), contact_pitch_deg=(-11.0, -11.0), liftoff_pitch_deg=(55.0, 55.0),
                    knee_swing_deg=(58.0, 58.0), hip_overshoot_deg=(2.0, 2.0), pelvis_rotation_deg=3.5,
                    trunk_counter_rotation=0.3, step_time_cv=0.025, step_length_cv=0.035, turn_lead_s=0.15), \
            {"turn_s": 3.2, "turn_share": 0.30}
    if grade == 2:
        # shuffling: little push-off (the heel barely rises), the ankle dorsiflexed early in swing so that the foot
        # travels nearly level and low, a small knee flexion (the clearance lift then stays small: a toes-down foot
        # after a full push-off scraped the floor and the lift raised the swing knee to about 70 deg)
        return _prm(step_m=0.40, cadence_spm=120.0, arm_swing_deg=_two(side, 2.5, 7.0),
                    elbow_deg=_two(side, 42.0, 34.0), elbow_swing_deg=_two(side, 2.0, 5.0),
                    arm_flexion_deg=_two(side, 8.0, 6.0), arm_abduction_deg=_two(side, 9.0, 10.0),
                    trunk_lean_deg=18.0, head_flexion_deg=10.0, clearance_m=(0.011, 0.011),
                    contact_pitch_deg=(-3.0, -3.0), liftoff_pitch_deg=(18.0, 18.0), knee_swing_deg=(42.0, 42.0),
                    swing_ankle_deg=(LOW_FLAT_ANKLE_DEG, LOW_FLAT_ANKLE_DEG), swing_ankle_share=0.3,
                    dip_synergy=LOW_FLAT_DIP_SYNERGY,
                    knee_mid_deg=(10.0, 10.0), knee_loading_deg=(18.0, 18.0), hip_overshoot_deg=(0.0, 0.0),
                    pelvis_rotation_deg=2.0, pelvis_obliquity_deg=(2.0, 2.0), lateral_sway_m=0.018,
                    trunk_counter_rotation=0.0, step_time_cv=0.035, step_length_cv=0.05, turn_lead_s=0.0), \
            {"turn_s": 4.8, "turn_share": 0.06}
    raise ValueError(f"MDS-UPDRS 3.10 grade {grade} is not rendered (NOT_RENDERED)")


FESTINATION_SWING_ANKLE_DEG = 16.0   # [module] the festination's swing ankle (D171)
FESTINATION_WIDER_M = 0.02     # [module] the festinating base this much wider than the grade-2 walk's: on the healthy
#                                base the thighs met at toe-off in the quickest steps (190 steps/min), and the
#                                leg-to-leg guard turned the swing hip out by 13 deg within 60 ms of a 0.25 s swing
#                                (2026-11-03 wave)


def festination():
    """Grade-2 parkinsonian gait whose steps shorten while the cadence rises and the trunk leans further, on a base
    FESTINATION_WIDER_M wider."""
    p, _ = parkinsonian(2)
    times = (3.0, 9.0)
    # the festinating swing keeps the grade 2 walk's ankle and dip lift of before D162 (FESTINATION_SWING_ANKLE_DEG,
    # the default dip synergy): its quickest swings (0.25 s at 190 steps/min) with the ankle at LOW_FLAT_ANKLE_DEG
    # failed gait.validity's continuity check (open-body port, wave 8 second part, D171)
    p = replace(p, swing_ankle_deg=(FESTINATION_SWING_ANKLE_DEG, FESTINATION_SWING_ANKLE_DEG), dip_synergy=None)
    return replace(p, step_m=stepped((0.42, 0.16), times), cadence_spm=stepped((118.0, 190.0), times),
                   trunk_lean_deg=stepped((18.0, 28.0), times), knee_swing_deg=stepped((42.0, 30.0), times),
                   step_width_m=round(p.step_width_m + FESTINATION_WIDER_M, 3))


HESITATION_STEP_M = 0.03       # [module] the steps in place of the start hesitation
HESITATION_KNEE_DEG = 10.0     # [module] their swing knee (18 deg lifted the heel 8 to 9 cm behind)
HESITATION_CLEARANCE_M = 0.004   # [module] their foot clearance
HESITATION_LIFTOFF_DEG = 15.0  # [module] their toe-off roll (the foot rolled this far onto the toes)
HESITATION_LIFTOFF_RAMP_S = (3.0, 4.0)   # [module] the roll eased back to the walk's between these times (s)
HESITATION_ANKLE_DEG = 10.0    # [module] their swing ankle, dorsiflexed (the foot held level); after the hesitation
#                                every parameter is the 3.10 walk's (the clip's level 0), the start ramp (start_s) aside
HESITATION_STANCE_KNEE = 0.3   # [module] their stance knee's loading (stance_knee_share): with the walk's loading the
#                                trailing knee stood at 30 to 34 deg at each toe-off in place, and each swing knee
#                                peaked 18 to 29 deg above standing (wave SH1d review of lane gait)


def start_hesitation(base_grade=1):
    """3.10 grade base_grade walk that starts with about 2.5 s of small shuffling steps in place."""
    p, kw = parkinsonian(base_grade)
    L = p.step_m
    plan = (1.0, 3.4, 4.6)
    # in place the feet barely leave the floor: the knee flexes little and the foot is held level (a full swing
    # knee on a 3 cm step kicked the heel up behind, a jog in place; with 18 deg asked the heel still rose 8 to 9 cm,
    # 2026-11-03 review); the arms swing with the step length (arm_swing_step_ref_m: the walking arm swing in place
    # read as marching, 2026-11-03 review)
    # the toe-off roll in place is small (HESITATION_LIFTOFF_DEG): the swing knee's start follows the stance knee's
    # flexion at push-off, so that with the walk's full roll onto the toes the in-place knee still peaked at 37 to
    # 50 deg (wave SH1d); the roll comes back to the walk's over HESITATION_LIFTOFF_RAMP_S, before the first full step
    # (eased with the step length, the first step's flat toe-off moved the knee at 124 m/s2)
    K = pair(p.knee_swing_deg)[0]
    lo = stepped((HESITATION_LIFTOFF_DEG, HESITATION_LIFTOFF_DEG, pair(p.liftoff_pitch_deg)[0]),
                 (plan[0],) + HESITATION_LIFTOFF_RAMP_S)
    return replace(p, step_m=stepped((HESITATION_STEP_M, HESITATION_STEP_M, L), plan),
                   liftoff_pitch_deg=(lo, lo),
                   stance_knee_share=stepped((HESITATION_STANCE_KNEE, HESITATION_STANCE_KNEE, 1.0), plan),
                   knee_swing_deg=stepped((HESITATION_KNEE_DEG, HESITATION_KNEE_DEG, K), plan),
                   clearance_m=stepped((HESITATION_CLEARANCE_M, HESITATION_CLEARANCE_M, pair(p.clearance_m)[0]),
                                       plan),
                   swing_ankle_deg=stepped((HESITATION_ANKLE_DEG, HESITATION_ANKLE_DEG, pair(p.swing_ankle_deg)[0]),
                                           plan), start_s=0.35, arm_swing_step_ref_m=L), kw


SARA1_FREE_WIDER_M = 0.01      # [module] the SARA 1 free walk's base this much wider than the healthy one, asked: the
#                                walk realizes about 0.2 cm of it (12.36 against 12.15 cm read back, wave SH1d review
#                                of lane gait), so that its caption says the base barely wider
ATAXIC_SARA1 = dict(step_width_m=round(STEP_WIDTH_M + SARA1_FREE_WIDER_M, 3), step_length_cv=0.03, step_time_cv=0.02,
                    step_width_sd_m=0.012, trunk_sway_deg=1.5, wander_m=0.015, step_m=0.66, cadence_spm=112.0,
                    liftoff_pitch_deg=(63.0, 63.0), arm_abduction_deg=(12.0, 12.0))     # [module]


def ataxic(grade):
    """(GaitParams, course kwargs) of SARA gait 0 to 3 (specs docstring)."""
    if grade == 0:
        return NORMAL, {}
    # SARA 1 (sara: slight difficulties, visible only in 10 consecutive tandem steps): a near-normal free walk, the
    # base SARA1_FREE_WIDER_M wider than the healthy one and the steps a little less regular (ATAXIC_SARA1); the
    # former values (the base 6.7 cm wider, a width SD of 2.5 cm) read as SARA 2 (2026-11-03 review)
    table = {1: dict(ATAXIC_SARA1),
             2: dict(liftoff_pitch_deg=(60.0, 60.0), step_width_m=_wider(0.17), step_length_cv=0.09, step_time_cv=0.06,
                     step_width_sd_m=0.035,
                     trunk_sway_deg=5.0, wander_m=0.06, step_m=0.55, cadence_spm=104.0,
                     arm_abduction_deg=(18.0, 18.0), trunk_lean_deg=7.0),
             # SARA 3, considerable staggering: four lurches (a step thrown 9 to 11 cm outward, the next one
             # corrective) on top of the noise; the width SD above grade 2's (it was below it); the arms take back
             # 0.7 of the trunk's lateral lean (locked to the thorax they tilted with it like a statue's) [module]
             3: dict(step_width_m=_wider(0.19), step_length_cv=0.13, step_time_cv=0.09, step_width_sd_m=0.04,
                     trunk_sway_deg=10.0, wander_m=0.08, step_m=0.47, cadence_spm=98.0, liftoff_pitch_deg=(58.0, 58.0),
                     arm_abduction_deg=(24.0, 24.0), trunk_lean_deg=9.0, lateral_sway_m=0.035, arm_lean_follow=0.7,
                     footprint_offsets=(("right", 6, 0.10, -0.04), ("left", 11, 0.09, 0.03), ("right", 17, 0.11, -0.05),
                                        ("left", 24, 0.10, 0.0)))}
    turn = {1: {"turn_s": 2.8}, 2: {"turn_s": 3.4, "turn_share": 0.3}, 3: {"turn_s": 4.2, "turn_share": 0.2}}
    return _prm(**table[grade]), turn[grade]


TANDEM_HALF_M = 0.015          # [module] each tandem footprint's ankle this far off the line (2026-11-03): with the
#                                ankles on the line and the feet straight, the legs of the rigid mesh passed about
#                                6 cm into each other in double support (the rear knee into the front one; a living
#                                tandem presses them together), out of reach of the swing guard
TANDEM_TOE_OUT_DEG = 20.0      # [module] the tandem feet turned out by this, so that each knee flexes beside the other
#                                leg (tests/test_synth_gait.py, test_the_healthy_walks_keep_their_legs_apart)
TANDEM_NOTE = (f"the feet turned out about {TANDEM_TOE_OUT_DEG:g} deg and each ankle {100 * TANDEM_HALF_M:g} cm off "
               "the line, the same on SARA 0 to 2")    # in clinical English (the reason, the rigid mesh's legs, is
#                                                        stated at TANDEM_HALF_M)


TANDEM_START_S, TANDEM_STOP_S, TANDEM_DURATION_S = 1.0, 15.0, 20.5   # [module] the tandem walk: at 60 steps/min,
#                                about 13 tandem steps (9 in the former 10 s walk, too few for the anchor of SARA 1);
#                                the stop's steps end by 19.2 s (SARA 2's, the widest), then the body stands (an 18 s
#                                clip cut SARA 2's last step in its swing: gait.validity's stop_in_clip)
TANDEM_MIN_STEPS = 12          # [module] the tandem steps every tandem clip takes at least (sara: grade 1 is seen only
#                                over 10 consecutive tandem steps; the 2026-11-03 review asked 12 or more)
TANDEM_MISSTEPS = {1: (("left", 3, 0.09, 0.0), ("right", 5, 0.08, 0.0))}   # [module] SARA 1: two steps put down
#                                beside the line (side, stance of that side, outward m, forward m), the 5th and the 10th
#                                of the 13 tandem steps tandem_read counts, more than the one misstep SARA 0 allows
#                                (sara)
NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four"}
TANDEM_OFF_LINE_M = 0.05       # a tandem footprint whose ankle lies this far or more from the line is a misstep (the
#                                line-keeping footprints lie TANDEM_HALF_M off it) [module]


def tandem(grade):
    """Tandem gait (heel to toe on a line) for SARA gait 0 to 2 (sara): 0 on the line; 1 two steps off the line
    (TANDEM_MISSTEPS) over at least TANDEM_MIN_STEPS tandem steps; 2 the line kept for 3 steps per foot, then
    every step out to the side (module proposal)."""
    # the feet heel to toe, each ankle TANDEM_HALF_M off the line, turned out by TANDEM_TOE_OUT_DEG
    p = _prm(step_m=0.27, cadence_spm=60.0, step_width_m=2.0 * TANDEM_HALF_M,
             toe_out_deg=(TANDEM_TOE_OUT_DEG, TANDEM_TOE_OUT_DEG),
             swing_abduction_deg=(5.0, 5.0), step_length_cv=0.0, step_time_cv=0.03, step_width_sd_m=0.0,
             arm_swing_deg=(4.0, 4.0), arm_abduction_deg=(18.0, 18.0), contact_pitch_deg=(-8.0, -8.0),
             liftoff_pitch_deg=(30.0, 30.0), pelvis_rotation_deg=2.0, lateral_sway_m=0.012, place_share=0.25,
             knee_swing_deg=(50.0, 50.0))
    if grade == 1:
        p = replace(p, footprint_offsets=TANDEM_MISSTEPS[1], trunk_sway_deg=3.0)
    elif grade == 2:
        p = replace(p, footprint_offsets=tuple((s, j, 0.045 + 0.015 * (j % 2), 0.0) for s in SIDES
                                               for j in range(3, 12)),
                    trunk_sway_deg=4.0, arm_abduction_deg=(28.0, 28.0))
    return p


def _mirror_arm(post):
    """The right-side copy of a left arm posture: joint indices to the right, rotations about y and z negated."""
    mp = {13: 14, 16: 17, 18: 19, 20: 21}
    return {mp[j]: [(ax, deg if ax == "x" else -deg) for ax, deg in spec] for j, spec in post.items()}


HEMI_WIDER_M = 0.03           # [module] the hemiparetic base this much wider than the healthy one (post-stroke walkers
#                               take wider steps than controls, kao2014 [Kao, Dingwell, Higginson, Binder-Macleod,
#                               Dynamic instability during post-stroke hemiparetic walking, Gait Posture
#                               2014;40:457-463]:
#                               the direction, the size is this module's). On the healthy 12 cm base the circumducting
#                               paretic leg met the stance leg and the leg-to-leg guard added up to 9.8 deg of swing hip
#                               abduction (the asked 12 read as 23, a wide outward arc, 2026-11-03 review); at 15 cm it
#                               adds none (wave SH1d)
HEMI_SWING_ABD_DEG = 6.0      # [module] the paretic swing hip's asked abduction (swing_abduction_deg). The hip hike's
#                               pelvic obliquity adds to it in the world: with 12 asked the paretic thigh reached 23 deg
#                               from the vertical seen from the front (the 2026-11-03 review read 23 against 12 asked),
#                               with 6 it reaches 19 and the hip 11 in the pelvis's frame (wave SH1d, the clip's own
#                               build): the outward arc mostly from the hike, as Kerrigan2000 (row G01) describes


def hemiparetic(side="left"):
    """Hemiparetic gait of side (pirker2017): circumduction, stiff knee, plantarflexed and inverted foot in swing,
    hip hike, shorter paretic stance, the Wernicke-Mann arm held (WM_ARM), a base HEMI_WIDER_M wider than the
    healthy one [levels: module]."""
    # the paretic push-off is weak (liftoff 10 deg): with 25 deg the plantarflexed toe scraped in early swing and
    # the anti-penetration lift flexed the 'stiff' knee to about 60 deg; now it peaks near 30 deg (stiff knee below
    # 44 deg, Lee2024 in the catalogue row G01). The swing leg clears mostly by the hip hike (9 deg of pelvic
    # obliquity) with a moderate circumduction (HEMI_SWING_ABD_DEG): swing hip abduction is not increased after
    # stroke (Kerrigan2000, row G01), and 18 deg swung the foot in a wide outward arc
    return _prm(step_m=0.45, cadence_spm=86.0, swing_abduction_deg=_two(side, HEMI_SWING_ABD_DEG, 0.0),
                knee_swing_deg=_two(side, 28.0, 58.0), hip_overshoot_deg=_two(side, 0.0, 4.0),
                swing_ankle_deg=_two(side, -4.0, 3.0), swing_inversion_deg=_two(side, 10.0, 0.0),
                stance_inversion_deg=_two(side, 5.0, 0.0),
                contact_pitch_deg=_two(side, 4.0, -15.0), liftoff_pitch_deg=_two(side, 10.0, 60.0),
                knee_mid_deg=_two(side, 1.0, 4.0), stance_share=_two(side, 0.58, 0.70),
                pelvis_obliquity_deg=_two(side, 3.0, -9.0), trunk_lateral_deg=_two(side, 0.0, 5.0),
                arm_swing_deg=_two(side, 0.0, 12.0),
                arm_posture={side: WM_ARM if side == "left" else _mirror_arm(WM_ARM)},
                clearance_m=_two(side, None, 0.016), trunk_lean_deg=6.0,
                step_width_m=round(STEP_WIDTH_M + HEMI_WIDER_M, 3))


SCISSOR = {  # level: (step width m, swing hip adduction deg, swing hip internal rotation deg, toe-out deg) [module]
    2: (0.0, 10.0, 15.0, 18.0),
    1: (0.04, 5.0, 8.0, 16.0)}
SCISSOR_GAP_M = 0.002          # [module] the guard's hard gap: the knees may touch, never pass into each other
SCISSOR_HIP_OVERSHOOT_DEG = 10.0   # [module] the swing hip flexes this far past its heel-strike value, so that the
#                                    swing knee passes ahead of the crouched stance knee
SCISSOR_LATE_SHARE = 0.25          # [module] the swing adduction starts at this share of the swing
#                                    (swing_late_share), once the swing knee is about level with the stance knee
SCISSOR_TURN_BASE_M = 0.12         # [module] the base widened at the turn's middle (turn_base_m, over the turn's own
#                                    window): on the narrow base the turning feet passed into each other in double
#                                    support, where the guard cannot act
SCISSOR_TURN_S = 5.0               # [module] the 180 deg turn's duration (3.4 s turned the legs into each other)
SCISSOR_TURN_SHARE = 0.45          # [module] the progression's speed share in the turn (the generator's default
#                                    turn_speed_share): slower turns took shorter steps whose feet overlapped
# the captions in clinical English (the 2026-11-03 review found engineering prose in them); the rigid body model's
# constraints (the turned-out feet, the guard that keeps the knees from passing into each other) are stated in the
# docstrings of spastic_diplegic and the module
SCISSOR_TEXT = {lv: (f"crouch, on the toes; scissoring: the hips adduct in swing (about {add:g} deg, from "
                     f"{100 * SCISSOR_LATE_SHARE:g} percent of the swing on) and rotate inward (about {rot:g} deg: "
                     f"the knees in, the swinging foot carried around the stance foot), the knees coming together "
                     f"and touching; "
                     + ("each foot set down on the midline; " if w == 0.0 else
                        f"a narrow base (about {100 * w:g} cm, ankle to ankle); ")
                     + f"the feet turned out about {toe:g} deg; the base widens by up to "
                       f"{100 * SCISSOR_TURN_BASE_M:g} cm in the turn")
                for lv, (w, add, rot, toe) in SCISSOR.items()}


def spastic_diplegic(level=2):
    """Spastic diplegic gait with scissoring (pirker2017: crouch, on the toes, narrow base, the knees close together,
    scissoring) [levels: module]. Level 2: each foot asked on the midline (SCISSOR), the swing hip adducted from
    SCISSOR_LATE_SHARE of the swing on, once the swing knee is about level with the stance knee (adducted earlier,
    the swing knee was pushed into the stance knee from behind), and internally rotated over the whole swing, so
    that the knees come together while the flexed knee carries the swinging foot out around the stance foot; the
    swing hip flexes past its heel-strike value (SCISSOR_HIP_OVERSHOOT_DEG) so that the swing knee passes ahead of
    the stance knee. Level 1: a milder form on a narrow base. The leg-to-leg guard (leg_gap_m SCISSOR_GAP_M) takes
    the asked adduction back wherever a knee would pass into the other leg, so that the knees touch at most. The
    feet are turned out (SCISSOR), a compromise of the rigid body model: the knee's plane follows the foot in this
    generator, and with the feet straighter the crouched knees of the neutral body pass into each other at toe-off,
    where the guard (a swing operator) can only throw the swinging leg outward. For the same reason the base widens
    inside the turn (SCISSOR_TURN_BASE_M at its middle, the turn's own window), which takes SCISSOR_TURN_S at
    SCISSOR_TURN_SHARE of the walking speed."""
    w, add, rot, toe = SCISSOR[level]
    return _prm(step_m=0.38, cadence_spm=100.0, step_width_m=w, toe_out_deg=(toe, toe),
                contact_pitch_deg=(18.0, 18.0), mid_pitch_deg=(14.0, 14.0), liftoff_pitch_deg=(40.0, 40.0),
                flat_share=(0.12, 0.12), knee_contact_deg=(22.0, 22.0), knee_loading_deg=(30.0, 30.0),
                knee_mid_deg=(26.0, 26.0), knee_swing_deg=(40.0, 40.0),
                hip_overshoot_deg=(SCISSOR_HIP_OVERSHOOT_DEG, SCISSOR_HIP_OVERSHOOT_DEG),
                swing_ankle_deg=(-10.0, -10.0), clearance_m=(0.01, 0.01), trunk_lean_deg=12.0,
                arm_abduction_deg=(20.0, 20.0), elbow_deg=(35.0, 35.0), arm_swing_deg=(8.0, 8.0),
                pelvis_rotation_deg=6.0, lateral_sway_m=0.03, trunk_lateral_deg=(4.0, 4.0),
                swing_abduction_deg=(-add, -add), swing_rotation_deg=(rot, rot), swing_late_share=SCISSOR_LATE_SHARE,
                leg_gap_m=SCISSOR_GAP_M, turn_base_m=SCISSOR_TURN_BASE_M)


STEPPAGE_HIP_PEAK_SHARE = 0.72   # [module] the high swing hip peaks earlier (hip_peak_share): at 0.82 its return to the
#                                  heel-strike value in the last 80 ms of the swing jerked the landing foot at 107 m/s2
STEPPAGE_TURN_S = 3.2            # [module] the steppage course's 180 deg turn (the 3.10 grade 1 walk's): over the
#                                  default 2.4 s, and with the turning stance's rocker drop (TURN_ROCKER_DROP) on the
#                                  hanging foot, the bilateral steppage's turn left the right foot nearly flat at
#                                  toe-off (the knee at 35 deg against 60 to 65 on the straight), and the dip lift
#                                  flexed the knee from 35 to 100 deg in 0.12 s (902 deg/s at 100 Hz, wave SH1d review
#                                  of lane gait); the hanging foot's turning stance pushes off as a straight one
#                                  (turn_rocker_drop 0, its liftoff pitch already under the healthy walk's)


STEPPAGE_KNEE_DEG = 65.0       # [module] the dropped side's asked swing knee (60 before the open body, D161)
STEPPAGE_HIP_DEG = 16.0        # [module] the dropped side's hip overshoot (12 before the open body, D161)


def steppage(sides=("left",)):
    """Steppage gait (pirker2017): the foot hangs plantarflexed in swing, the knee and hip lift high to clear it,
    the forefoot or the flat foot lands first [levels: module]. The swing knee lands near the 70.5 deg of moderate
    CMT1A (Park2023, catalogue row G07; 64 deg in this module's healthy walk) and peaks later in the swing (0.4),
    the hip about 11 deg above the healthy walk's (Park2023: 7 deg); the foot hangs about 15 deg below the healthy
    swing's (module). The former values (knee 82 asked, foot 28 deg down, clearance 3 cm) realized a 89 deg knee
    whose heel flicked back in early swing. On the open body (wave 8 second part, D161) the hanging foot's clearance
    lift sets the realized knee, and a lower asked knee raises it (more lift, twice its share at the knee): asked at
    60 deg with 12 of hip overshoot the knee read 78.14 deg, at 55 deg 85.46; the dropped side asks STEPPAGE_KNEE_DEG
    and STEPPAGE_HIP_DEG, which read 74.67 deg on the 8 s steppage walk of the tests (scripts/openbody_wave8b_checks.py
    'steppage')."""
    def two(a, b):
        return tuple(a if s in sides else b for s in SIDES)
    return _prm(swing_ankle_deg=two(-18.0, 3.0), knee_swing_deg=two(STEPPAGE_KNEE_DEG, 62.0),
                hip_overshoot_deg=two(STEPPAGE_HIP_DEG, 4.0),
                knee_peak_share=two(0.4, 0.3), clearance_m=two(0.02, 0.016), contact_pitch_deg=two(6.0, -18.0),
                flat_share=two(0.03, 0.09), liftoff_pitch_deg=two(50.0, 66.0), step_m=0.60, cadence_spm=104.0,
                hip_peak_share=STEPPAGE_HIP_PEAK_SHARE, turn_rocker_drop=two(0.0, TURN_ROCKER_DROP))


DUCHENNE_ARM_ABD_DEG = 8.0     # [module] the weak side's arm abducted this much more in the compensated form: with the
#                                healthy walk's, the trunk leaning over the weak leg brought the hanging hand's thumb
#                                19 mm into the thigh in the turn (the skeleton page's contact check, wave SH1d)


def trendelenburg(side="right", duchenne=False):
    """Hip abductor weakness of side (pirker2017): the pelvis drops on the swing side while standing on it;
    duchenne: the trunk leans over the weak stance leg instead (the compensated form) [levels: module]."""
    obl = list(NORMAL.pelvis_obliquity_deg)
    lat = [0.0, 0.0]
    abd = list(NORMAL.arm_abduction_deg)
    i = SIDES.index(side)
    if duchenne:
        obl[i], lat[i] = 4.0, 10.0
        abd[i] += DUCHENNE_ARM_ABD_DEG
    else:
        obl[i] = 7.5
    return _prm(pelvis_obliquity_deg=tuple(obl), trunk_lateral_deg=tuple(lat), step_m=0.62, cadence_spm=108.0,
                step_width_m=_wider(0.11), lateral_sway_m=0.015, arm_abduction_deg=tuple(abd))


def waddling():
    """Waddling gait (bilateral hip girdle weakness, pirker2017) with the lumbar hyperlordosis of the myopathic
    patient (catalogue row G09): the pelvis tilted 15 deg forward and the lumbar spine extended by as much, the
    trunk's line slightly behind the vertical [levels: module, unsourced]."""
    return _prm(pelvis_obliquity_deg=(7.0, 7.0), trunk_lateral_deg=(9.0, 9.0), step_width_m=_wider(0.17),
                lateral_sway_m=0.02, step_m=0.55, cadence_spm=104.0, trunk_lean_deg=-2.0, lordosis_deg=15.0,
                arm_abduction_deg=(15.0, 15.0))


ANTALGIC_ANKLE_DEG = 20.0      # [module] the painful leg's ankle held up early in swing (12 before the open body): at
#                                12 deg Anny's longer forefoot needed the clearance lift and the guarded knee read 66.7
#                                deg against the 40 asked (open-body port, wave 8 second part, D172;
#                                scripts/openbody_wave8b_checks.py 'antalgic')


def antalgic(side="left"):
    """Antalgic gait (pirker2017): short stance on the painful side, a quick short step of the other leg, the
    painful ankle held [levels: module]. The painful leg pushes off little and swings guarded: its ankle held up
    early in swing (12 deg of dorsiflexion by 0.3 of the swing), so that the toe clears without a high knee (with
    the healthy ankle path the toe scraped after toe-off and the lift raised that knee to about 68 deg, 45 mm above
    the floor: a high step, not a guarded one)."""
    return _prm(stance_share=_two(side, 0.52, 0.68), hs_offset=0.44 if side == "left" else 0.56,
                knee_swing_deg=_two(side, 40.0, 60.0), liftoff_pitch_deg=_two(side, 20.0, 66.0),
                swing_ankle_deg=_two(side, ANTALGIC_ANKLE_DEG, 3.0), swing_ankle_share=_two(side, 0.3, 0.45),
                contact_pitch_deg=_two(side, -4.0, -18.0), trunk_lateral_deg=_two(side, 6.0, 0.0), step_m=0.55,
                cadence_spm=108.0, clearance_m=_two(side, 0.012, 0.016))


NPH_PIVOT_SHARE = 0.0          # [module] the magnetic gait's planted foot turns through none of the heading change
#                                (pivot_share; GaitParams' 0.9 before): its flat stance foot (contact and lift-off pitch 0
#                                and 12 deg) pivoting through 28 deg of the slow 6.5 s turn read 14.22 mm of travel at
#                                its least-moving sole point on the page's skate check, against that contact's 5.17 mm
#                                (plan phase E; scripts/openbody_phaseE_checks.py 'nph_turn')


def nph_magnetic():
    """Magnetic gait of normal-pressure hydrocephalus (pirker2017: broad base, short steps, start difficulty,
    'magnetic feet', turn difficulty), arm swing kept [levels: module]."""
    L = 0.30
    plan = (1.0, 4.0, 5.0)
    # the feet barely lifted and held level (the ankle dorsiflexed early in swing, little push-off): with a full
    # push-off the toes scraped and the clearance lift flexed the knee to about 60 deg; a 4 m course keeps the clip
    # near 30 s. The planted foot does not pivot with the slow turn (NPH_PIVOT_SHARE, as the vestibular walk's):
    # each footprint lands on the new heading
    return _prm(step_m=stepped((0.02, 0.02, L), plan), cadence_spm=90.0, step_width_m=_wider(0.19),
                toe_out_deg=(15.0, 15.0), clearance_m=(0.007, 0.007), contact_pitch_deg=(0.0, 0.0),
                liftoff_pitch_deg=(12.0, 12.0), knee_swing_deg=stepped((18.0, 18.0, 36.0), plan),
                swing_ankle_deg=(14.0, 14.0), swing_ankle_share=0.3, knee_mid_deg=(8.0, 8.0),
                hip_overshoot_deg=(0.0, 0.0), trunk_lean_deg=8.0, arm_swing_deg=(12.0, 12.0),
                pelvis_rotation_deg=2.0, start_s=0.35, turn_lead_s=0.05, pivot_share=NPH_PIVOT_SHARE), \
        {"turn_s": 6.5, "turn_share": 0.04, "distance_m": 4.0}


CAUTIOUS_BASE = 1.4            # pirker2017: the elderly base about 40 percent wider (the cautious gait's 0.13 m on the
#                                former 0.09 m healthy base, kept as a ratio of STEP_WIDTH_M)


def cautious():
    """Cautious gait (pirker2017: slow, wider base, reduced arm swing, slight stoop) [levels: module; the base
    about 40 percent wider than the healthy walk's, pirker2017]."""
    return _prm(step_m=0.55, cadence_spm=100.0, step_width_m=round(CAUTIOUS_BASE * STEP_WIDTH_M, 3),
                stance_share=(0.65, 0.65),
                arm_abduction_deg=(22.0, 22.0), arm_swing_deg=(7.0, 7.0), elbow_deg=(25.0, 25.0), trunk_lean_deg=9.0,
                knee_swing_deg=(55.0, 55.0), contact_pitch_deg=(-10.0, -10.0), liftoff_pitch_deg=(50.0, 50.0),
                pelvis_rotation_deg=3.0, turn_lead_s=0.15), {"turn_s": 3.5, "turn_share": 0.25}


DYSTONIC_CLEARANCE_M = 0.008   # [module] the dystonic foot's asked clearance: with 0.02 and the foot 18 deg more
#                                plantarflexed, the lift saturated and the swing knee reached 85 to 91 deg (a steppage)


def dystonic(side="left", level=2):
    """Dystonic gait (pirker2017: foot inversion and plantarflexion while walking), with a trunk twist; level 1
    mild, 2 moderate [levels: module]. The foot inverts in swing against the roll of the knee's axis (generate), so
    that the sole is seen turned toward the midline (swing_read); it hangs plantarflexed less than the steppage foot
    and clears by a moderate knee (DYSTONIC_CLEARANCE_M), the foot progression angle the healthy one. (Toed in, the
    hip's internal rotation rolled the knee's axis outward in mid swing, which the ankle's inversion room of the
    joint-limit table could not take back, and the inverted stance foot pressed the inner thighs together at
    toe-off, where the leg-to-leg guard threw the swing leg out by 13 deg within 0.1 s: a sideways snap of the foot,
    2026-11-03 review.)"""
    g = {1: 0.5, 2: 1.0}[level]
    return _prm(swing_inversion_deg=_two(side, 16.0 * g, 0.0), swing_ankle_deg=_two(side, -6.0 * g, 3.0),
                stance_inversion_deg=_two(side, 14.0 * g, 0.0),
                contact_pitch_deg=_two(side, 6.0 * g - 18.0 * (1 - g), -18.0), liftoff_pitch_deg=(58.0, 58.0),
                trunk_twist_deg=10.0 * g, trunk_twist_osc_deg=5.0 * g,
                step_m=0.58, cadence_spm=104.0, knee_swing_deg=(62.0, 62.0),
                clearance_m=_two(side, DYSTONIC_CLEARANCE_M, 0.016))


def _choreic_text():
    """The choreic walk's caption, its amplitude and share of time from involuntary.py's UHDRS map (CHOREA_GRADE)."""
    from . import involuntary as IV
    a_term, d_term = IV.UHDRS[CHOREA_GRADE]
    return (f"irregular steps with lateral deviations; chorea of the trunk, neck, arms and fingers ({a_term} "
            f"amplitude, about {100 * IV.AMP[a_term]:.0f} percent of each region's range, present about "
            f"{100 * IV.DUR[d_term]:.0f} percent of the time, as UHDRS grade {CHOREA_GRADE}) and brief choreic hip and "
            "knee movements in swing")


def choreic():
    """Choreic gait (pirker2017: irregular, dance-like, swaying, variable step length and direction, wider and
    slower); the involuntary movements come from the chorea engine of involuntary.py (chorea_upper, chorea_legs)
    [levels: module]."""
    return _prm(step_m=0.55, cadence_spm=102.0, step_width_m=_wider(0.14), step_length_cv=0.12, step_time_cv=0.08,
                step_width_sd_m=0.04, wander_m=0.10, trunk_sway_deg=4.0, liftoff_pitch_deg=(58.0, 58.0))


def functional(kind):
    """Functional gait signs (lempert1991, nonnekes2020): knee buckling without falls, astasia-abasia, slow motion,
    walking on ice [levels: module]."""
    if kind == "knee_buckling":
        # three sudden buckles, each in the first stance of its side from the time given: the knee gives way over
        # 0.13 s as the weight comes onto it, then straightens over 0.4 s, both arms flung out by 18 deg at the
        # dip [module]. (The former symmetric 0.45 s Gaussian dips read as deliberate squats, and two of the three
        # fell in a swing of their leg, where they were not seen.)
        bt = (("left", 4.0, 24.0, 0.13, 0.4), ("right", 7.6, 26.0, 0.13, 0.4), ("left", 12.8, 22.0, 0.13, 0.4))
        return _prm(step_m=0.60, cadence_spm=104.0, knee_buckles=bt, buckle_arm_deg=18.0,
                    arm_abduction_deg=(14.0, 14.0))
    if kind == "astasia_abasia":
        # the arms take back 0.7 of the trunk's lateral sway (locked to the thorax they tilted with it) [module]
        # standing on the same wide base as it walks (from a 0.17 m stance the first steps swayed the pelvis over
        # the standing foot) [module]
        return _prm(step_m=0.35, cadence_spm=82.0, step_width_m=_wider(0.22), stand_width_m=_wider(0.22),
                    liftoff_pitch_deg=(50.0, 50.0), step_length_cv=0.15, step_time_cv=0.10,
                    step_width_sd_m=0.04, wander_m=0.12, trunk_sway_deg=14.0, trunk_sway_hz=0.5,
                    arm_abduction_deg=(35.0, 35.0), arm_swing_deg=(3.0, 3.0), elbow_deg=(30.0, 30.0),
                    knee_buckles=(("left", 5.0, 25.0, 0.5), ("right", 9.5, 28.0, 0.5)), knee_mid_deg=(10.0, 10.0),
                    lateral_sway_m=0.05, arm_lean_follow=0.7)
    if kind == "slow_motion":
        return _prm(step_m=0.48, cadence_spm=44.0, arm_swing_deg=(10.0, 10.0), start_s=1.2,
                    liftoff_pitch_deg=(58.0, 58.0))
    if kind == "walking_on_ice":
        return _prm(step_m=0.30, cadence_spm=96.0, step_width_m=_wider(0.20), contact_pitch_deg=(0.0, 0.0),
                    liftoff_pitch_deg=(10.0, 10.0), knee_swing_deg=(34.0, 34.0), knee_mid_deg=(12.0, 12.0),
                    swing_ankle_deg=(LOW_FLAT_ANKLE_DEG, LOW_FLAT_ANKLE_DEG), swing_ankle_share=0.3,
                    knee_loading_deg=(14.0, 14.0), hip_overshoot_deg=(0.0, 0.0), arm_abduction_deg=(32.0, 32.0),
                    arm_swing_deg=(1.5, 1.5), arm_flexion_deg=(12.0, 12.0), elbow_deg=(30.0, 30.0),
                    elbow_swing_deg=(1.0, 1.0), trunk_lean_deg=6.0, pelvis_rotation_deg=1.0,
                    clearance_m=(0.009, 0.009), trunk_counter_rotation=0.0)
    raise ValueError(kind)


FUNCTIONAL_COURSE = {"slow_motion": {"turn_s": 6.2, "distance_m": 3.0},
                     "astasia_abasia": {"distance_m": 5.5, "turn_s": 4.0},
                     "walking_on_ice": {"distance_m": 5.5, "turn_s": 3.2}}
#                                     the slow functional gaits walk a shorter course: each clip near 30 s; the
#                                     lurching astasia-abasia turns over 4 s (in 2.4 s, the healthy turn, its inside
#                                     hip adducted beyond the table on half of the seeds tried); the walking on ice
#                                     over 3.2 s, the fluctuating gait's 3.4 s about (in 2.4 s one stance turned
#                                     through 87 deg, and with the turning foot pivoting on the toes the push-off
#                                     rolls on, the next swing's knee flexed at 707 deg/s, above the continuity
#                                     check's KNEE_RATE_MAX_DEG_S) [module]


RETRO_STEPS = {0: 2, 1: 4, 2: 7}      # recovery steps per 3.12 grade: 0 at most 2, 1 three to five, 2 more than 5
RETRO_PULL_S = 1.0                    # the pull's time (stated, no examiner drawn)
RETRO_LIFT_M = 0.03                   # [module] the backward step's ankle lift at mid swing (swing_min_jerk_m): a low,
#                                       quick step back with a few cm of clearance (the joint-space swing of the
#                                       forward walk, its knee at 35 deg, flicked the heel up 19 to 22 cm, 2026-11-03).
#                                       Its knee still peaks at 34 to 38 deg, above the about 20 deg the clinical
#                                       review's plan asked: the knee follows the stance leg's loading and the step's
#                                       reach more than the lift (wave SH1d: with 2 cm it read 30 to 34 deg, with 1.2 cm
#                                       the grade 2's first step lay beyond the leg's reach)
PULL_MARK_S = STIMULUS_MARK_S         # how long a viewer shows the pull's mark (the stimulus marks' of jerks.py)


def retropulsion(grade):
    """(Body, info): MDS-UPDRS 3.12 grade 0 to 2 as recovery steps backward after a backward pull stated at
    RETRO_PULL_S (RETRO_STEPS; mds_updrs, paraphrased): the walk line runs backward while the body faces forward, toes
    first at contact, the trunk thrown back and the arms raised forward at the pull, the steps shortening; each step a
    low quick step back on a minimum-jerk foot path (RETRO_LIFT_M)."""
    n = RETRO_STEPS[grade]
    cad = 150.0
    Ts = 60.0 / cad
    p = _prm(step_m=stepped((0.24, 0.10), (1.0, 1.0 + n * Ts)), cadence_spm=cad, contact_pitch_deg=(10.0, 10.0),
             liftoff_pitch_deg=(-8.0, -8.0), flat_share=(0.08, 0.08), heel_off_share=(0.3, 0.3), place_share=0.22,
             # the trunk is carried back by the pull from 1.0 s, the arms come forward after a reaction latency of
             # 0.1 s [module] (they used to start at 0.9 s, before the pull)
             swing_min_jerk_m=RETRO_LIFT_M,
             knee_swing_deg=(35.0, 35.0), trunk_lean_deg=stepped((5.0, -9.0, 0.0), (1.0, 1.25, 1.2 + n * Ts + 0.6)),
             arm_flexion_deg=stepped((2.0, 28.0, 4.0), (1.1, 1.35, 1.2 + n * Ts + 0.6)), arm_swing_deg=(2.0, 2.0),
             arm_abduction_deg=(18.0, 18.0), elbow_deg=(30.0, 30.0), start_s=0.2, step_time_cv=0.0,
             step_length_cv=0.0, step_width_sd_m=0.0, clearance_m=(0.01, 0.01), hip_overshoot_deg=(0.0, 0.0),
             pelvis_rotation_deg=1.0)

    def back(t, c, psi):
        return -c, psi
    stop_at = 1.0 + max(n - 2, 0) * Ts + 0.25
    return generate(p, duration_s=round(stop_at + 3.0, 2), start_at_s=1.0, stop_at_s=stop_at, seed=SEED + grade,
                    path_fn=back)


def retro_steps_read(body, info):
    """The swings of a walk read back from the body, JSON-ready (the pull test's backward steps): per swing of the
    generator (info['swings']), its ankle's horizontal travel, the ankle's highest rise above its height at toe-off
    and the sole's highest point above the floor (its lowest point, at mid swing), the knee's peak flexion
    (knee_flexion_deg), whether the ankle's travel along its step is monotone (no step back of more than 1 mm) and its
    peak horizontal speed over the mean; and the largest of these over the swings."""
    J = body.joints().numpy()
    rest = body.rest().numpy()
    psi = np.asarray(info["heading_rad"])
    lows = {s: sole_cloud_points(body, s)[..., 1].min(-1) for s in SIDES}
    floor = float(np.median(np.minimum(lows["left"], lows["right"])))
    rows = []
    for i, side in enumerate(SIDES):
        an = LEG[side][2]
        kf = knee_flexion_deg(J, psi, side, rest)
        for w in info["swings"][side]:
            k0, k1 = w["k_to"], w["k_hs"]
            A = J[k0:k1 + 1, an]
            d = A[-1, [0, 2]] - A[0, [0, 2]]
            L = float(np.hypot(*d))
            if L < 1e-3:
                continue
            along = (A[:, [0, 2]] - A[0, [0, 2]]) @ (d / L)
            v = np.hypot(*np.diff(A[:, [0, 2]], axis=0).T) * body.rate
            rows.append({"side": side, "t_s": k0 / body.rate, "travel_m": L,
                         "ankle_rise_m": float(A[:, 1].max() - A[0, 1]),
                         "sole_high_m": float(lows[side][k0:k1 + 1].max() - floor),
                         "knee_peak_deg": float(kf[k0:k1 + 1].max()),
                         "monotone": bool(np.diff(along).min() > -1e-3),
                         "peak_over_mean_speed": float(v.max() / max(v.mean(), 1e-9))})
    keys = ("ankle_rise_m", "sole_high_m", "knee_peak_deg", "peak_over_mean_speed")
    return {"n": len(rows), "swings": rows, "max": {k: max((r[k] for r in rows), default=None) for k in keys},
            "all_monotone": all(r["monotone"] for r in rows)}


def count_steps(body, after_s=0.0):
    """Stance runs (a foot's lowest sole point on the floor) that begin after after_s: the steps taken."""
    lows = np.stack([sole_cloud_points(body, s)[..., 1].min(-1) for s in SIDES], -1)
    floor = float(np.median(lows.min(1)))
    st = lows <= floor + FLOOR_TOL_M
    gap = int(round(0.1 * body.rate))                  # runs closer than 0.1 s are one contact (a rolling foot)
    n = 0
    for i in range(2):
        runs = _runs(st[:, i])
        merged = []
        for a, b in runs:
            if merged and a - merged[-1][1] <= gap:
                merged[-1] = (merged[-1][0], b)
            else:
                merged.append((a, b))
        n += sum(1 for a, b in merged if a / body.rate > after_s and a > 0)
    return int(n)


def tandem_read(body, info, off_m=TANDEM_OFF_LINE_M):
    """The tandem steps read back from the body, JSON-ready: each stance run after the first step (a foot's lowest
    sole point on the floor, as count_steps) that begins before the stop (info['stop']: its steps are set down side
    by side), its ankle's mean position over the run and its lateral distance from the walking line
    (info['walk_line'] at the run's middle, the line's direction from the walk line's start to its end); the steps
    whose distance is off_m or more are missteps."""
    J = body.joints().numpy()
    lows = {s: sole_cloud_points(body, s)[..., 1].min(-1) for s in SIDES}
    floor = float(np.median(np.minimum(lows["left"], lows["right"])))
    c = np.asarray(info["walk_line"])
    d = c[-1] - c[0]
    d = d / max(float(np.linalg.norm(d)), 1e-9)
    nrm = np.array([d[1], -d[0]])                     # (x, z) across the line (its sign is not used)
    rows = []
    for i, side in enumerate(SIDES):
        an = LEG[side][2]
        for a, b_ in _runs(lows[side] <= floor + FLOOR_TOL_M):       # b_ the run's last knot
            if a == 0 or b_ + 1 - a < int(0.08 * body.rate):
                continue
            m = (a + b_) // 2
            p = J[a:b_ + 1, an][:, [0, 2]].mean(0)
            rows.append({"side": side, "t_s": a / body.rate, "lateral_m": abs(float((p - c[m]) @ nrm))})
    t_stop = float((info.get("stop") or {}).get("t0", np.inf))
    rows = sorted((r for r in rows if r["t_s"] < t_stop), key=lambda r: r["t_s"])    # the stop's steps, set down
    #                                                 side by side, are not tandem steps
    return {"steps": len(rows), "missteps": int(sum(r["lateral_m"] >= off_m for r in rows)), "off_line_m": off_m,
            "lateral_m": [round(r["lateral_m"], 4) for r in rows], "t_s": [round(r["t_s"], 2) for r in rows]}


def trunk_head_roll(body):
    """The trunk's lateral lean and the head's roll in the world, read from the body, JSON-ready (degrees): per knot
    the lean of the pelvis-to-neck line (joints 0 to 12) across the hip line (toward the subject's left positive) and
    the roll of the head (the elevation of joint 15's lateral axis, its left side up positive, the ear line's tilt);
    the slope of the head's roll on the trunk's lean (least squares; near 0 when the head rights itself, near -1 when
    it tilts with the trunk), the largest lean and its 95th percentile, and the largest tilt of the collar line
    (joints 16 to 17) against the hip line (1 to 2)."""
    J, G = body.fk()
    J, G = J.numpy(), G.numpy()
    hip = J[:, 1] - J[:, 2]
    hip[:, 1] = 0.0
    hip /= np.linalg.norm(hip, axis=1, keepdims=True)
    tr = J[:, 12] - J[:, 0]
    lean = np.degrees(np.arctan2((tr * hip).sum(1), tr[:, 1]))
    roll = np.degrees(np.arcsin(np.clip(G[:, 15, 1, 0], -1.0, 1.0)))

    def tilt(a, b_):
        v = J[:, a] - J[:, b_]
        return np.degrees(np.arctan2(v[:, 1], np.hypot(v[:, 0], v[:, 2])))
    k = int(np.argmax(np.abs(lean)))
    return {"slope": float(np.polyfit(lean, roll, 1)[0]) if np.ptp(lean) > 1e-9 else 0.0,
            "lean_max_deg": float(lean[k]), "lean_p95_deg": float(np.percentile(np.abs(lean), 95)),
            "head_roll_at_max_deg": float(roll[k]),
            "collar_vs_hip_max_deg": float(np.max(np.abs(tilt(16, 17) - tilt(1, 2))))}


TRUNK_PITCH_WALK_M_S = 0.3    # [module] the knots read as walking by trunk_pitch: the pelvis faster than this


def trunk_pitch(body):
    """The stoop read back from the body, JSON-ready (degrees, forward positive): the forward lean of the
    pelvis-to-neck line (joints 0 to 12) and of the neck-to-head line (12 to 15) in the vertical plane across the hip
    line, medians over the walking knots (the pelvis's horizontal speed above TRUNK_PITCH_WALK_M_S; None without
    any) and over every knot. (A trunk_lean_deg is spread over the spine and partly taken back by the head
    (head_level): the 2026-11-03 review read 8 deg more than the healthy walk on the pelvis-to-neck line where 13 more
    were asked; this records what the body shows.)"""
    J = body.joints().numpy()
    hip = J[:, 1] - J[:, 2]
    hip[:, 1] = 0.0
    hip /= np.linalg.norm(hip, axis=1, keepdims=True)
    fwd = np.stack([-hip[:, 2], np.zeros(len(hip)), hip[:, 0]], 1)
    sp = np.r_[0.0, np.linalg.norm(np.diff(J[:, 0][:, [0, 2]], axis=0), axis=1)] * body.rate
    walk = sp > TRUNK_PITCH_WALK_M_S
    out = {}
    for name, (a, c) in (("pelvis_neck", (0, 12)), ("neck_head", (12, 15))):
        v = J[:, c] - J[:, a]
        p = np.degrees(np.arctan2((v * fwd).sum(1), v[:, 1]))
        out[f"{name}_walk_deg"] = float(np.median(p[walk])) if walk.any() else None
        out[f"{name}_all_deg"] = float(np.median(p))
    return out


def foot_roll_read(body, info, side):
    """A foot's roll in the world, read from the body, JSON-ready: the elevation of the foot's lateral axis (the
    ankle joint's global x axis; for the left foot its lateral side, for the right foot its medial side, so that
    inversion lowers the left value and raises the right one), its median at mid swing (the middle fifth of each
    swing of info['swings']) and in mid stance (the middle fifth of each stance of the walk, the standing ones left
    out), and the swing knee's peak flexion (knee_flexion_deg), the median over the swings."""
    J, G = body.fk()
    G = G.numpy()
    a = LEG[side][2]
    el = np.degrees(np.arcsin(np.clip(G[:, a, 1, 0], -1.0, 1.0)))
    kf = knee_flexion_deg(J.numpy(), np.asarray(info["heading_rad"]), side, body.rest().numpy())
    sw, pk = [], []
    for w in info["swings"][side]:
        k0, k1 = w["k_to"], w["k_hs"]
        n = k1 - k0
        sw.append(el[k0 + int(0.4 * n):k0 + int(0.6 * n) + 1])
        pk.append(float(kf[k0:k1 + 1].max()))
    st = []
    for f in info["footprints"][side]:
        if not (f["first"] or f["final"]):
            n = f["k_to"] - f["k_hs"]
            st.append(el[f["k_hs"] + int(0.4 * n):f["k_hs"] + int(0.6 * n) + 1])
    return {"mid_swing_deg": float(np.median(np.concatenate(sw))) if sw else None,
            "mid_stance_deg": float(np.median(np.concatenate(st))) if st else None,
            "swing_knee_peak_deg": float(np.median(pk)) if pk else None}


CONTINUITY_FPS = 30.0         # the frame rate the continuity check reads the body at (the skeleton page's)
CONTACT_JUMP_MAX_M = 0.03     # [module] the largest horizontal move of a foot's touching sole point between two frames
#                               (the 2026-11-03 review's bound: 3 cm in one frame at 30 fps)
PEAK_ACC_MAX_M_S2 = 100.0     # [module] the largest joint acceleration over three frames (the review's bound: the
#                               generated heel strikes read 50 to 60 m/s2, the reference walk 31, the snaps 72 to 182)
SKIM_HEIGHT_M = 0.005         # [module] the floor skim's contact height above a sole point's standing height (the
#                               skeleton page's CONTACT_TOL_MM)
SKIM_FORE_SHARE = 0.8         # [module] the floor skim reads the sole points behind this share of the line from the
#                               ankle to the toe joint (the ball and the toes ahead of it left out)
SKIM_PLANTED_M_S = 0.25       # [module] a sole point at contact height moving slower than this into a frame is planted
#                               there (the skeleton page's CONTACT_SPEED_M_S)
KNEE_RATE_MAX_DEG_S = 700.0   # [module] the largest knee flexion speed between two frames: about 1.3 times the
#                               healthy level 0's peak (531 deg/s, the wave SH1d review of lane gait), under the
#                               one-frame knee flick of the bilateral steppage's turn that the review read (844 deg/s,
#                               78.8 m/s2, under PEAK_ACC_MAX_M_S2)


def _at_frames(x, rate, fps=CONTINUITY_FPS):
    """(F, ...) the knot series x (T, ...) at the frame times f / fps (linear between knots)."""
    T = len(x)
    t = np.arange(int(np.floor((T - 1) / rate * fps)) + 1) / fps
    u = t * rate
    k0 = np.clip(np.floor(u).astype(np.int64), 0, T - 2)
    w = (u - k0).reshape((-1,) + (1,) * (np.ndim(x) - 1))
    return (1.0 - w) * x[k0] + w * x[k0 + 1], k0


def continuity(body, info=None, fps=CONTINUITY_FPS):
    """The frame-to-frame continuity of a walking body read at fps (the skeleton page's frame rate; positions
    linear between knots), JSON-ready: per side, the largest horizontal move between two consecutive frames of the
    foot's touching sole point (the lowest point of its rigid sole cloud at the first frame, within FLOOR_TOL_M of
    the floor there, followed as the same cloud point) while the generator holds the foot in stance at both frames
    (info['stance']; a swing toe skimming the floor after toe-off is not a contact; no info: every knot), the swings
    of a dragged side (info['swings'], drag: a deliberate skate) left out; the floor skim, the skeleton page's foot
    jump read on the sole cloud whatever the generator's stance flag (a swing foot launched or gliding at floor
    height, which the stance reading does not see: the releases of the shuffling episodes, wave SH1d review of lane
    gait): the largest horizontal move between two consecutive frames of a sole point (rest height within
    SKIM_HEIGHT_M of the floor, behind SKIM_FORE_SHARE of the ankle to toe joint line) standing within SKIM_HEIGHT_M
    of its rest height at both frames and planted on the frame before (moving slower than SKIM_PLANTED_M_S into
    it), the swings of a dragged side left out; the forefoot ahead of it is the foot's lift-off point, the last to
    leave the floor and leaving it forward (the healthy walk's toe pads move 5 to 11 cm a frame within 2 mm of the
    floor just after toe-off, the shuffle's ball 27 to 31 mm a frame after its flat toe-off); the largest knee
    flexion speed between two frames (the 3D angle between the thigh and the shank); and the largest acceleration
    of the 22 body joints over three consecutive frames (the second difference). ok: the contact moves at most
    CONTACT_JUMP_MAX_M, the knee speed at most KNEE_RATE_MAX_DEG_S and the acceleration at most PEAK_ACC_MAX_M_S2."""
    J, G = body.fk()
    clouds = {s: sole_cloud_points(body, s, (J, G)) for s in SIDES}
    lows = np.stack([clouds[s][..., 1].min(-1) for s in SIDES], -1)
    floor = float(np.median(lows.min(1)))
    st = np.ones((body.T, 2), bool) if info is None else np.asarray(info["stance"], bool).copy()
    drag = np.zeros((body.T, 2), bool)
    for i, side in enumerate(SIDES):
        for w in ((info or {}).get("swings") or {}).get(side, []):
            if w.get("drag"):
                st[w["k_to"]:w["k_hs"] + 1, i] = False
                drag[w["k_to"]:w["k_hs"] + 1, i] = True
    out = {"fps": fps, "contact_jump_max_m": CONTACT_JUMP_MAX_M, "peak_acc_max_m_s2": PEAK_ACC_MAX_M_S2,
           "knee_rate_max_deg_s": KNEE_RATE_MAX_DEG_S}
    worst = skim = 0.0
    for i, side in enumerate(SIDES):
        X, k0 = _at_frames(clouds[side], body.rate, fps)
        k1 = np.minimum(k0 + 1, body.T - 1)
        sf = st[k0, i] & st[k1, i]
        j = np.argmin(X[:-1, :, 1], 1)
        r = np.arange(len(X) - 1)
        a, b = X[r, j], X[r + 1, j]
        touch = (a[:, 1] <= floor + FLOOR_TOL_M) & sf[:-1] & sf[1:]
        d = np.where(touch, np.hypot(*(b - a)[:, [0, 2]].T), 0.0)
        m = int(np.argmax(d))
        out[f"contact_jump_{side}_m"] = float(d[m])
        out[f"contact_jump_{side}_t_s"] = float((m + 1) / fps)
        worst = max(worst, float(d[m]))
        _, _, ak, ft = LEG[side]
        fa = J[0, ft].numpy() - J[0, ak].numpy()
        fa[1] = 0.0
        fwd_m = ((clouds[side][0] - J[0, ak].numpy()) @ fa) / float(fa @ fa)   # 0 at the ankle, 1 at the toe joint
        h0 = clouds[side][0, :, 1] - floor                                       # the standing first knot's heights
        sole = (h0 <= SKIM_HEIGHT_M) & (fwd_m <= SKIM_FORE_SHARE)
        Xs = X[:, sole]
        at = Xs[..., 1] - floor <= h0[sole] + SKIM_HEIGHT_M                     # (F, M') at contact height
        free = ~(drag[k0, i] | drag[k1, i])
        step = np.hypot(*(Xs[1:] - Xs[:-1])[..., [0, 2]].transpose(2, 0, 1))    # (F - 1, M'): frame f to f + 1
        planted = np.r_[np.zeros((1, Xs.shape[1]), bool), at[:-2] & (step[:-1] < SKIM_PLANTED_M_S / fps)]
        both = at[:-1] & at[1:] & planted & (free[:-1] & free[1:])[:, None]
        mv = np.where(both, step, 0.0).max(1)
        m = int(np.argmax(mv))
        out[f"floor_skim_{side}_m"] = float(mv[m])
        out[f"floor_skim_{side}_t_s"] = float((m + 1) / fps)
        skim = max(skim, float(mv[m]))
    P, _ = _at_frames(J.numpy()[:, :N_BODY], body.rate, fps)
    acc = np.linalg.norm(P[2:] - 2 * P[1:-1] + P[:-2], axis=-1) * fps * fps
    k, jj = np.unravel_index(int(np.argmax(acc)), acc.shape)
    rate_k = {}
    for side in SIDES:
        h, kn, a_, _ = LEG[side]
        u, v = P[:, h] - P[:, kn], P[:, a_] - P[:, kn]
        c = (u * v).sum(-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1))
        flex = 180.0 - np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))
        sp = np.abs(np.diff(flex)) * fps
        m = int(np.argmax(sp))
        rate_k[side] = (float(sp[m]), float((m + 1) / fps))
        out[f"knee_rate_{side}_deg_s"] = rate_k[side][0]
        out[f"knee_rate_{side}_t_s"] = rate_k[side][1]
    knee_rate = max(r_[0] for r_ in rate_k.values())
    out.update(peak_acc_m_s2=float(acc[k, jj]), peak_acc_joint=int(jj), peak_acc_t_s=float((k + 1) / fps),
               contact_jump_worst_m=worst, floor_skim_worst_m=skim, knee_rate_worst_deg_s=knee_rate)
    out["ok"] = bool(worst <= CONTACT_JUMP_MAX_M and skim <= CONTACT_JUMP_MAX_M and
                     knee_rate <= KNEE_RATE_MAX_DEG_S and out["peak_acc_m_s2"] <= PEAK_ACC_MAX_M_S2)
    return out


SLIP_TOL_M_S = 0.005          # the stance contact point's 95th percentile horizontal speed
PENETRATION_TOL_M = 0.004     # a swing sole may dip at most this below the floor
LEGS_REFUSE = True            # the leg-to-leg check inside gait.ok (validity), since the calibration of 2026-11-03


def validity(body, info, legs=None):
    """The checks of a generated body, JSON-ready: joint-limit table components outside the table (every one is
    'new', a generated body has no base), the stance slip of the contact point, the deepest sole point below the
    floor in swing, finite values, and (LEGS_REFUSE) the leg-to-leg contact check; gait.ok when all hold. 'legs':
    the leg-to-leg contact check (legcontact.validity, its own 'ok'; legs a gap_from_pose result already computed),
    with the distal band's reading beside it (legcontact 'distal_*': the thigh just above the knee counted hard, in
    the record only; the rule before 2026-11-03 read most of that band as hard).
    Until 2026-11-03 'legs' stood outside gait.ok (34 of the 41 gait clips failed it, the healthy walk included); since
    the calibration (STEP_WIDTH_M, TURN_BASE_M, the guards) and legcontact's soft zone read at the closest point
    (the same day: most of the medial thigh's band from legcontact.KNEE_ZONE_M to KNEE_ZONE_M + DISTAL_BAND_M above
    the knee joint read hard before, soft now, its reading kept in 'legs' as 'distal_*') every healthy walk and
    every gait clip of the catalogue holds it, so that a failure is a defect of the clip and refuses it
    (gait.legs_in_ok)."""
    from .body import limit_violations
    lv = limit_violations(body.rot6d)
    sl = stance_slip(body, info)
    lows = [w["lowest_m"] for s in SIDES for w in info["swings"][s] if w["lowest_m"] is not None]
    deepest = float(min(lows)) if lows else 0.0
    finite = bool(torch.isfinite(body.rot6d).all() and torch.isfinite(body.transl).all())
    lg = LCN.validity(body, legs)
    cont = continuity(body, info)
    stop = info.get("stop") or {}
    in_clip = bool(stop.get("t_end") is None or stop["t_end"] <= (body.T - 1) / body.rate)   # the stop's last step
    #                                             lands inside the clip (a swing cut by the clip's end is left unset)
    ok = (lv["outside"] == 0 and sl["p95_m_s"] < SLIP_TOL_M_S and deepest > -PENETRATION_TOL_M and finite
          and (lg["ok"] or not LEGS_REFUSE) and cont["ok"] and in_clip)
    return {"limits_outside": lv["outside"], "limits_new": lv["outside"], "limits_worst_deg": lv["worst_deg"],
            "stance_slip_p95_m_s": sl["p95_m_s"], "stance_slip_max_m_s": sl["max_m_s"], "swing_lowest_m": deepest,
            "finite": finite, "gait": {"ok": bool(ok), "slip_tol_m_s": SLIP_TOL_M_S,
                                       "penetration_tol_m": PENETRATION_TOL_M, "legs_in_ok": LEGS_REFUSE,
                                       "continuity_ok": cont["ok"], "stop_in_clip": in_clip},
            "legs": lg, "continuity": cont}


def signed_step_widths(body, floor_tol_m=FLOOR_TOL_M, skip=2, turn_rate_deg_s=10.0, turn_margin_s=1.0):
    """From the body alone: per side, the signed step widths (m) of its steady footprints. A footprint is the ankle
    joint at the flattest knot of a stance run (as in measure_gait: a run of a foot's lowest sole point within
    floor_tol_m of the floor for 80 ms or more); its step width is its distance from the line through the previous
    and the next footprint of the other foot (that foot's line of progression, the heel-to-heel base of support of
    instrumented walkways), positive on the foot's own side, negative past that line (the feet crossing). Steady:
    neither among the first or last `skip` runs of the side nor, at the footprint or at either footprint of the
    other foot, within turn_margin_s of a turn (the heading, a 2 s moving mean of the root's yaw as in measure_gait,
    turning faster than turn_rate_deg_s)."""
    from ..body.rotations import yaw_angle
    J, G = (x.numpy() for x in body.fk())
    rate = body.rate
    lows = np.stack([sole_cloud_points(body, s, (torch.from_numpy(J), torch.from_numpy(G)))[..., 1].min(-1)
                     for s in SIDES], -1)
    floor = float(np.median(lows.min(1)))
    st = lows <= floor + floor_tol_m
    n2 = int(round(2.0 * rate))                       # two strides: the pelvic rotation averages out (measure_gait)
    psi = np.unwrap(yaw_angle(body.local_R()[:, 0]).numpy())
    psi = np.convolve(np.pad(psi, (n2 // 2, n2 - 1 - n2 // 2), mode="edge"), np.ones(n2) / n2, mode="valid")
    turning = np.abs(np.degrees(np.gradient(psi, 1.0 / rate))) > turn_rate_deg_s
    m_k = int(round(turn_margin_s * rate))
    near_turn = np.convolve(turning.astype(float), np.ones(2 * m_k + 1), mode="same") > 0
    prints = {}                                       # side: [(knot, (x, z))], in time order
    for side in SIDES:
        pr = []
        for a, b in _runs(st[:, SIDES.index(side)]):
            if b - a < max(3, int(round(0.08 * rate))):
                continue
            fz = G[a:b + 1, LEG[side][2]][:, :, 2]
            kf = a + int(np.argmin(np.abs(np.arctan2(-fz[:, 1], np.hypot(fz[:, 0], fz[:, 2])))))
            pr.append((kf, J[kf, LEG[side][2], [0, 2]]))
        prints[side] = pr
    out = {}
    for side, other in (("left", "right"), ("right", "left")):
        ws = []
        ko = np.array([k for k, _ in prints[other]])
        for k, A in prints[side][skip:len(prints[side]) - skip]:
            j = int(np.searchsorted(ko, k))
            if j == 0 or j >= len(ko) or near_turn[k] or near_turn[ko[j - 1]] or near_turn[ko[j]]:
                continue
            P0, P1 = prints[other][j - 1][1], prints[other][j][1]
            d = P1 - P0
            nd = float(np.hypot(*d))
            if nd < 1e-6:
                continue
            lft = np.array([d[1], -d[0]]) / nd           # the subject's left of the other foot's line (x left, z on)
            ws.append(SIGN[side] * float((A - P0) @ lft))
        out[side] = ws
    return out


def midline_offsets(body, **kw):
    """Per side, the signed lateral offsets (m) of its steady footprints from the walking line: half their signed
    step widths (signed_step_widths, its keywords), the walking line running midway between the lines of progression
    of the two feet; positive on the foot's own side, negative past the midline. (The former walking line, the
    pelvis path smoothed by a Hann window of 1.2 s, kept about half of the pelvis's sideways sway at the stride
    frequency and drew the line toward each stance foot.)"""
    w = signed_step_widths(body, **kw)
    return {s: [0.5 * x for x in w[s]] for s in SIDES}


def scissor_record(body, info, legs=None):
    """JSON-ready read-back of the scissoring of a walk: the signed step widths (signed_step_widths: median and
    smallest, negative where the feet cross) and the footprints' offsets from the midline (half of them: median per
    side, smallest, the share past the midline, that is of the steps crossing), the knee joints' lateral distance
    (centre to centre, heading frame) in mid swing (swing weight above 0.9: median and smallest), the leg-to-leg gap
    (legcontact: the smallest hard gap, the share of knots touching) and what the guard took back of the asked swing
    adduction (the added abduction per swing, degrees)."""
    wid = signed_step_widths(body)
    off = {s: [0.5 * x for x in wid[s]] for s in SIDES}
    allw = np.array(wid["left"] + wid["right"])
    allo = 0.5 * allw
    J = body.joints().numpy()
    psi = info["heading_rad"]
    lft = np.stack([np.cos(psi), -np.sin(psi)], -1)
    kd = ((J[:, 4, [0, 2]] - J[:, 5, [0, 2]]) * lft).sum(-1)
    mid = (np.asarray(info["swing_weight"]) > 0.9).any(1)
    g = LCN.leg_gap(body) if legs is None else legs
    rep_ = LCN.report(g, body.rate)
    add = [x["added_deg"] for x in info.get("leg_guard", [])]
    return {"step_width_signed_median_m": float(np.median(allw)) if len(allw) else None,
            "step_width_signed_min_m": float(allw.min()) if len(allw) else None,
            "footprint_offset_median_m": {s: float(np.median(off[s])) if off[s] else None for s in SIDES},
            "footprint_offset_min_m": float(allo.min()) if len(allo) else None,
            "share_past_midline": float((allo < 0).mean()) if len(allo) else None, "n_footprints": int(len(allo)),
            "knee_distance_mid_swing_median_m": float(np.median(kd[mid])) if mid.any() else None,
            "knee_distance_mid_swing_min_m": float(kd[mid].min()) if mid.any() else None,
            "hard_gap_min_m": rep_["hard_min_m"], "hard_gap_where": rep_["hard_where"],
            "touch_share": rep_["touch_share"], "swing_adduction_asked_deg": list(pair(info["params"]
                                                                                       ["swing_abduction_deg"])),
            "guard_added_abduction_deg": {"median": float(np.median(add)) if add else None,
                                          "max": float(np.max(add)) if add else None, "n_swings": len(add)}}


def guard_record(info):
    """JSON-ready summary of the leg-to-leg guard of a walk (info['leg_guard']): the guard's gap (leg_gap_m, None
    when off), the swings it measured, the hip abduction it added per swing (median and largest, degrees), the
    swings it added any to and the swings whose gap it could not keep; the double-support guard's pushes (a landing
    foot set down wider: side, stance index, time, outward m) and the post-overlay swing bumps (side, toe-off knot,
    time, degrees)."""
    g = info.get("leg_guard", [])
    add = np.array([x["added_deg"] for x in g])
    return {"leg_gap_m": info["params"].get("leg_gap_m"), "n_swings": len(g),
            "added_deg_median": float(np.median(add)) if len(add) else None,
            "added_deg_max": float(add.max()) if len(add) else None,
            "n_added": int((add > 0).sum()), "n_not_held": int(sum(not x["holds"] for x in g)),
            "ds_push": [{k: p[k] for k in ("side", "stance", "t_s", "outward_m")} for p in info.get("ds_push", [])],
            "overlay_guard": [{k: p[k] for k in ("side", "k_to", "t_s", "deg")} for p in info.get("overlay_guard", [])]}


def _truth(body, info, operator, grade, scale, sources, extra=None, measure=True):
    from .common import truth_record
    params = {"gait_params": info["params"], "course": info.get("course"), "turns": info["turns"],
              "stop": info["stop"], "seed": info["seed"]}
    if info.get("buckles"):
        params["buckles_realized"] = info["buckles"]
    if measure:
        try:
            params["measured"] = summary(measure_gait(body))
        except Exception as e:                                # a clip too short or irregular to measure
            params["measured"] = {"error": str(e)}
    params.update(extra or {})
    params["trunk_head_roll"] = trunk_head_roll(body)    # the lateral lean and the head's righting, read back
    params["trunk_pitch"] = trunk_pitch(body)            # the stoop, read back
    legs = LCN.leg_gap(body)
    params["leg_guard"] = guard_record(info)
    if info.get("stance_lift", {}).get("rounds"):
        params["stance_lift"] = info["stance_lift"]      # the stance guard's rounds and largest lift (wave 5)
    if operator == "gait_spastic":                       # the scissoring clips (every gait has the guard now)
        params["scissoring"] = scissor_record(body, info, legs)
    return truth_record(operator, info["seed"], grade, scale, params, sources, validity(body, info, legs),
                        arrays={"stance": info["stance"], "phase": info["phase"]})


def _course_build(prm_fn, operator, grade, scale, sources, seed, distance_m=10.0, overlays=(), extra=None, post=None,
                  read=None, **kw):
    """build() of a course clip; the course kwargs of prm_fn may set distance_m (a slow gait walks a shorter course,
    so that the clip stays near 30 s); post(body) -> body or (body, record) finishes the body (a hand posture, the
    upper-body chorea); read(body, info) -> dict joins the truth's params (a read-back of the sign)."""
    def build():
        p, ckw = prm_fn()
        ckw = dict(ckw, **kw)
        d = ckw.pop("distance_m", distance_m)
        b, info = course(p, distance_m=d, seed=seed, overlays=overlays, **ckw)
        ex = extra
        if post is not None:
            b = post(b)
            if isinstance(b, tuple):                     # (body, record): the record joins the truth's params
                b, rec = b
                ex = dict(extra or {}, **rec)
        if read is not None:
            ex = dict(ex or {}, **read(b, info))
        return b, _truth(b, info, operator, grade, scale, sources, ex)
    return build


def _spec(cid, family, item, operator, grade, grade_text, params_text, build, segment="E8", segment_title="",
          views=VIEWS, level0=LEVEL0, note=""):
    from .exam import ClipSpec
    return ClipSpec(cid, family, segment, item, operator, grade, grade_text, params_text, tuple(views), level0, None,
                    build, note=note, since=SINCE, segment_title=segment_title)


PD_WALK_LEVEL0 = "gait_E8_parkinsonian_g1"   # the level 0 of the clips built on the 3.10 grade 1 walk (the start
#                                             hesitation, freezing by shuffling): their comparison shows the freezing
#                                             sign alone (paired with the healthy walk, it mixed in the hypokinetic
#                                             gait, 2026-11-03 review)


def specs():
    """[ClipSpec] of the gait part of the symptom catalogue (since 'catalogue_v1'): the healthy level 0 of this
    generator, then each gait and grade. Grade maps are proposals of this module, anchored as SOURCES says:
    MDS-UPDRS 3.10 grade 1 at the pooled PD-to-control differences of zanardi2021 (stride 0.16 m shorter, speed
    about 0.15 m/s slower, cadence a little higher) with an arm swing asymmetry (lewek2010), grade 2 at a
    substantial impairment read from pirker2017 (short shuffling steps, stooped trunk, arms bent and nearly still,
    en bloc turning with many steps); SARA gait 1 to 3 by rising step variability, base width and sway (sara,
    ataxia_var: no value per grade was read, the steps are this module's); the other gaits at one described level
    (pirker2017, nonnekes2020, lempert1991). NOT_RENDERED lists what is left out and why."""
    S = SEED
    out = []
    nb = _course_build(lambda: (NORMAL, {}), "gait_generator", 0, "level 0 (healthy adult walk)",
                       ("perry2010", "bohannon2011", "pirker2017", "knee_norm", "po_norm", "elbow2023", "module"), S)
    out.append(_spec(LEVEL0, "gait", "3.10 level 0", "healthy walk (parametric gait generator)", 0,
                     "level 0 (healthy adult, comfortable speed)", "10 m, turn 180 deg, 10 m back", nb, level0=None))
    pd_src = ("mds_updrs", "zanardi2021", "lewek2010", "pirker2017", "module")
    for g in (1, 2):
        out.append(_spec(f"gait_E8_parkinsonian_g{g}", "gait_parkinsonian", "3.10",
                         "parkinsonian gait, left more affected", g, f"MDS-UPDRS 3.10 grade {g}",
                         {1: "shorter steps, less left arm swing, slight stoop, slower turn",
                          2: "short shuffling steps, stooped, arms bent and nearly still, en bloc turn in place"}[g],
                         _course_build(lambda g=g: parkinsonian(g), "parkinsonian_gait", g, "MDS-UPDRS 3.10", pd_src,
                                       S + g)))
    out.append(_spec("gait_E8_parkinsonian_festination", "gait_parkinsonian", "3.10",
                     "parkinsonian gait, festination", 2, "MDS-UPDRS 3.10 grade 2 with festination",
                     "steps shorten from 0.42 to 0.16 m while the cadence rises from 118 to 190 steps/min",
                     _course_build(lambda: (festination(), {}), "festination", 2, "MDS-UPDRS 3.10", pd_src, S + 5,
                                   distance_m=6.0, back=False)))
    out.append(_spec("gait_E8_parkinsonian_start_hesitation", "gait_parkinsonian", "3.11",
                     "parkinsonian gait, start hesitation (small steps in place)", 1,
                     "MDS-UPDRS 3.11 grade 1 (one hesitation, at the start) on 3.10 grade 1",
                     "about 2.5 s of shuffling in place before the first effective step",
                     _course_build(lambda: start_hesitation(1), "start_hesitation", 1, "MDS-UPDRS 3.11", pd_src,
                                   S + 6), level0=PD_WALK_LEVEL0))
    sara_src = ("sara", "ataxia_var", "pirker2017", "module")
    for g in (1, 2, 3):
        out.append(_spec(f"gait_E8_ataxic_sara{g}", "gait_ataxic", "SARA 1 (gait)", "cerebellar ataxic gait", g,
                         f"SARA gait {g}", {1: "near-normal free walk, the base barely wider and the steps a little "
                                               "less regular (at SARA 1 the difficulty shows only in tandem walking: "
                                               "see the tandem clip of SARA 1)",
                                            2: "wide base, irregular steps and timing, lateral sway",
                                            3: "staggering, very wide and irregular, difficult half-turn"}[g],
                         _course_build(lambda g=g: ataxic(g), "ataxic_gait", g, "SARA gait", sara_src, S + 10 + g)))
    for g in (0, 1, 2):
        def bt(g=g):
            p = tandem(g)
            b, info = generate(p, duration_s=TANDEM_DURATION_S, start_at_s=TANDEM_START_S, stop_at_s=TANDEM_STOP_S,
                               seed=S + 20 + g)
            info["course"] = {"tandem": True}
            return b, _truth(b, info, "tandem_gait", g, "SARA gait (tandem part)", sara_src,
                             {"offsets": [list(x) for x in p.footprint_offsets], "tandem_read": tandem_read(b, info)},
                             measure=False)
        out.append(_spec(f"gait_E8t_tandem_sara{g}", "gait_ataxic", "SARA 1 (tandem)", "tandem gait, heel to toe", g,
                         f"SARA gait {g} (tandem)", {0: "heel to toe on the line",
                                                     1: f"{NUMBER_WORDS[len(TANDEM_MISSTEPS[1])]} steps off the "
                                                        f"line over at least {TANDEM_MIN_STEPS} tandem steps",
                                                     2: "steps off the line after 3 tandem steps per foot"}[g], bt,
                         segment="E8t", segment_title="tandem gait",
                         level0=None if g == 0 else "gait_E8t_tandem_sara0", note=TANDEM_NOTE))
    desc = ("pirker2017", "module")
    plan = [
        ("gait_E8_hemiparetic_left", "gait_hemiparetic", "hemiparetic gait, left (Wernicke-Mann)",
         "moderate (described level)", "left leg circumducted, stiff knee, foot plantarflexed and inverted in swing, "
         f"hip hiked; a base about {100 * HEMI_WIDER_M:.0f} cm wider than in the healthy walk; left arm held flexed, "
         "fist clenched; turns toward the sound side",
         lambda: (hemiparetic("left"), {"turn_s": 4.6, "turn_share": 0.25, "turn_deg": -180.0})),
        ("gait_E8_spastic_diplegic", "gait_spastic", "spastic diplegic gait with scissoring",
         "moderate (described level)", SCISSOR_TEXT[2],
         lambda: (spastic_diplegic(2), {"turn_s": SCISSOR_TURN_S, "turn_share": SCISSOR_TURN_SHARE})),
        ("gait_E8_steppage_left", "gait_steppage", "steppage gait, left foot drop", "unilateral",
         "left foot hangs in swing, the hip and knee lifted higher than in the healthy walk, flat or toe-first "
         "landing (foot slap not heard)",
         lambda: (steppage(("left",)), {"turn_s": STEPPAGE_TURN_S})),
        ("gait_E8_steppage_bilateral", "gait_steppage", "steppage gait, bilateral foot drop", "bilateral",
         "both feet hang in swing, the hips and knees lifted higher than in the healthy walk, flat or toe-first "
         "landing (foot slap not heard)", lambda: (steppage(SIDES), {"turn_s": STEPPAGE_TURN_S})),
        ("gait_E8_trendelenburg_right", "gait_waddling", "Trendelenburg gait, right hip abductors", "uncompensated",
         "the pelvis drops on the left (swing) side while standing on the right leg",
         lambda: (trendelenburg("right"), {})),
        ("gait_E8_duchenne_right", "gait_waddling", "Trendelenburg gait, right, compensated (Duchenne)",
         "compensated", "the trunk leans over the right leg while standing on it",
         lambda: (trendelenburg("right", duchenne=True), {})),
        ("gait_E8_waddling", "gait_waddling", "waddling gait (bilateral hip girdle weakness)", "bilateral",
         "pelvis drops on each swing side, trunk sways over each stance leg, wider base, lumbar hyperlordosis "
         "(pelvis tilted 15 deg forward, the trunk upright)", lambda: (waddling(), {})),
        ("gait_E8_antalgic_left", "gait_antalgic", "antalgic gait, left leg", "moderate (described level)",
         "short stance on the left, quick short step of the right, left leg guarded (little push-off, the knee "
         "flexed about 40 deg in swing, the ankle held up)", lambda: (antalgic("left"), {})),
        ("gait_E8_nph_magnetic", "gait_frontal", "magnetic gait (normal-pressure hydrocephalus)",
         "moderate (described level)", "start hesitation, short wide steps with the feet barely lifted, turned out, "
         "turn with many steps, arm swing kept", nph_magnetic),
        ("gait_E8_cautious", "gait_cautious", "cautious gait", "described level",
         "slow, shorter and wider steps, arms slightly abducted with little swing, slight stoop", cautious),
        ("gait_E8_dystonic_left_l1", "gait_dystonic", "dystonic gait, left foot and trunk", "mild",
         "left foot plantarflexed and inverted in swing, on its lateral border in stance, mild trunk twist",
         lambda: (dystonic("left", 1), {"turn_s": 3.4, "turn_deg": -180.0})),
        ("gait_E8_dystonic_left_l2", "gait_dystonic", "dystonic gait, left foot and trunk", "moderate",
         "left foot plantarflexed and inverted in swing, on its lateral border in stance, trunk twisted",
         lambda: (dystonic("left", 2), {"turn_s": 3.4, "turn_deg": -180.0})),
    ]
    # the hemiparetic turn goes toward the sound (right) side, pivoting on the sound leg: turning toward the paretic
    # side swept the pelvis over the planted paretic foot and adducted that hip beyond the joint-limit table
    posts = {"gait_E8_hemiparetic_left": wm_hand("left")}

    def foot_rolls(b, info):                     # the inverted foot read back (dystonic, hemiparetic)
        return {"foot_roll": {sd: foot_roll_read(b, info, sd) for sd in SIDES}}
    reads = {c: foot_rolls for c in ("gait_E8_dystonic_left_l1", "gait_E8_dystonic_left_l2",
                                     "gait_E8_hemiparetic_left")}
    src = {"gait_E8_hemiparetic_left": ("pirker2017", "kao2014", "module"),
           "gait_E8_steppage_left": ("pirker2017", "park2023", "module"),
           "gait_E8_steppage_bilateral": ("pirker2017", "park2023", "module")}
    for k, (cid, fam, op, gtext, ptext, fn) in enumerate(plan):
        out.append(_spec(cid, fam, "described level", op, gtext, gtext, ptext,
                         _course_build(fn, fam, gtext, "described level", src.get(cid, desc), S + 30 + k,
                                       distance_m=8.0, post=posts.get(cid), read=reads.get(cid))))
    # the milder scissoring (its own seed, after the plan, so that the plan's seeds stay those of catalogue_v1)
    out.append(_spec("gait_E8_spastic_diplegic_mild", "gait_spastic", "described level",
                     "spastic diplegic gait with mild scissoring", "mild (described level)", "mild (described level)",
                     SCISSOR_TEXT[1],
                     _course_build(lambda: (spastic_diplegic(1), {"turn_s": SCISSOR_TURN_S,
                                                                  "turn_share": SCISSOR_TURN_SHARE}),
                                   "gait_spastic",
                                   "mild (described level)", "described level", desc, S + 45, distance_m=8.0),
                     level0=LEVEL0))
    out.append(_spec("gait_E8_choreic", "gait_choreic", "described level",
                     "choreic (dancing) gait", "described level", "described level",
                     _choreic_text(),
                     _course_build(lambda: (choreic(), {"turn_s": 3.0}), "choreic_gait", "described level",
                                   "described level", desc + ("uhdrs",), S + 50, distance_m=8.0,
                                   overlays=(chorea_legs(S + 52),), post=chorea_upper(S + 51))))
    fsrc = ("lempert1991", "nonnekes2020", "pirker2017", "module")
    for k, (kind, ptext) in enumerate((
            ("knee_buckling", "sudden knee buckling on three steps (the knee gives way over 0.13 s at weight "
                              "acceptance, the arms fling out), recovered without a fall"),
            ("astasia_abasia", "very wide, lurching, swaying steps with near-falls and no fall"),
            ("slow_motion", "a normal step pattern in slow motion (44 steps/min)"),
            ("walking_on_ice", "short, wide, flat-footed steps with stiff knees and abducted arms"))):
        out.append(_spec(f"gait_E8_functional_{kind}", "gait_functional", "described level",
                         f"functional gait, {kind.replace('_', ' ')}", "described level",
                         "described level (functional sign)", ptext,
                         _course_build(lambda kind=kind: (functional(kind), dict(FUNCTIONAL_COURSE.get(kind, {}))),
                                       "functional_gait", "described level",
                                       "described level", fsrc, S + 60 + k, distance_m=8.0)))
    for g in (0, 1, 2):
        def br(g=g):
            b, info = retropulsion(g)
            info["course"] = {"pull_s": RETRO_PULL_S, "direction": "backward"}
            return b, _truth(b, info, "retropulsion", g, "MDS-UPDRS 3.12", ("mds_updrs", "module"),
                             {"recovery_steps": count_steps(b, after_s=RETRO_PULL_S), "planned_steps": RETRO_STEPS[g],
                              "stimulus_marks": [{"t_s": RETRO_PULL_S, "dur_s": PULL_MARK_S, "text": "PULL"}],
                              "backward_steps": retro_steps_read(b, info)},
                             measure=False)
        out.append(_spec(f"gait_E7p_retropulsion_g{g}", "gait_postural", "3.12",
                         "recovery steps after a backward pull (the pull is stated, no examiner shown)", g,
                         f"MDS-UPDRS 3.12 grade {g}", {0: "two recovery steps backward",
                                                      1: "four recovery steps backward",
                                                      2: "seven recovery steps backward, recovers unaided"}[g], br,
                         segment="E7p", segment_title="pull test (backward perturbation stated, examiner not shown)",
                         views=("left", "back"), level0=None if g == 0 else "gait_E7p_retropulsion_g0"))
    out += gap_specs()
    ids = [s.clip_id for s in out]
    if len(set(ids)) != len(ids):
        raise RuntimeError("duplicate clip ids in the gait catalogue")
    return out


# ---------------------------------------------------------------- the gap wave of the catalogue (2026-11-02)
# Seven catalogue rows built on the generator (gap_specs): AP02 PSP gait, D19 dromedary gait, G14 vestibular gait
# (eyes open and closed), F08 functional dragging monoplegic leg, F09 functional gait with momentary fluctuations,
# A13 sensory ataxic gait (eyes open and closed), P17 freezing of gait by shuffling forward. Each clip's truth carries
# its read-back (params['read_back'], from the body alone), its module values (params['unsourced']), the balance check
# (validity['balance'], gait_balance) and, for the eye-state clips, params['frame_text'] (the body has no face; the
# viewer does not burn the text into the frames, the caption states it).
SOURCES.update({
    "kao2014": "Kao PC, Dingwell JB, Higginson JS, Binder-Macleod S. Dynamic instability during post-stroke "
               "hemiparetic walking. Gait Posture 2014; 40(3): 457-463, doi 10.1016/j.gaitpost.2014.05.014 "
               "(PMC4251664, read 2026-11-05): post-stroke individuals walked with a greater average step width than "
               "controls (p = 0.003; the values only in a figure); preferred speed 1.0 against 1.4 m/s",
    "dale2023": "Dale ML et al. Balance and gait in progressive supranuclear palsy: a narrative review of objective "
                "metrics and exercise interventions. Front Neurol 2023; 14: 1212185, doi 10.3389/fneur.2023.1212185 "
                "(as cited by the catalogue row AP02, secondary for the Hatanaka and Takamatsu values, not re-read "
                "here): PSP speed 0.83 against 1.10 m/s in controls, cadence 100.5 against 115.9 steps/min "
                "(Hatanaka); step width 10.6 against 7.7, units not stated (Takamatsu)",
    "schaafsma2003": "as grades.SOURCES['schaafsma2003'] (abstract), with the catalogue row P17: shuffling forward "
                     "with small steps and trembling in place the common forms of freezing, total akinesia rare",
    "wang2020": "as grades.SOURCES['wang2020']: episodes about 1 s to 2 min, median 3 s",
    "fog_grades": "grades.FOG_GRADES, the program's map of MDS-UPDRS 3.11: grade 3 one freeze in straight walking "
                  "with the two trigger episodes of grade 2 (a module reading of the draft, section 4)",
    "nielsen2017": "as jerks.SOURCES['nielsen2017']: the S-FMDRS per region severity 0 absent, 1 mild, 2 moderate, 3 "
                   "severe, and duration once or a few times, frequent but intermittent, continuous (the gait region "
                   "for the functional gait clips F08 and F09, their duration a module reading)",
    "anagnostou2024": "Anagnostou E et al. Romberg's test revisited. Neurophysiol Clin 2024; 54: 102999, doi "
                      "10.1016/j.neucli.2024.102999 (abstract, as cited by the catalogue rows A04 and A13): a Romberg "
                      "ratio of about 2 in pure sensory neuropathy (stance sway)",
})

PSP_GAIT = {"speed_m_s": 0.83, "cadence_spm": 100.5, "control_speed_m_s": 1.10, "control_cadence_spm": 115.9,
            "width_psp": 10.6, "width_control": 7.7}          # dale2023 (Hatanaka; Takamatsu, units not stated)
DROMEDARY = {"pump_deg": 30.0, "pump_peak": 0.30, "pump_hip_share": 0.4}           # [module, unsourced]
DROMEDARY_SEEN_DEG = {"trunk": 23, "pelvis": 12}   # the caption's values: trunk_cycle's read-back of the clip (the
#                     pelvis-to-neck line and the pelvis's own pitch, ranges per stride, rounded; the 30 deg asked of
#                     the pelvis and spine shows less on the hip-to-neck line); a test keeps them in step
VESTIBULAR = {                                  # [module, unsourced: the catalogue row G14 read no source]
    "open": dict(veer_deg_per_m=0.8, step_m=0.62, cadence_spm=108.0, step_width_m=_wider(0.12), step_width_sd_m=0.018,
                 step_length_cv=0.035, step_time_cv=0.025, trunk_sway_deg=2.0, arm_abduction_deg=13.0),
    "closed": dict(veer_deg_per_m=3.0, step_m=0.52, cadence_spm=100.0, step_width_m=_wider(0.15), step_width_sd_m=0.03,
                   step_length_cv=0.06, step_time_cv=0.04, trunk_sway_deg=4.0, arm_abduction_deg=18.0)}
SENSORY_EC_GAIN = 2.0           # anagnostou2024: the stance Romberg ratio of about 2, carried over to the gait's step
#                                 variability, sway and wander with the eyes closed (a module reading)
SENSORY_VAR = dict(step_length_cv=0.06, step_time_cv=0.04, step_width_sd_m=0.022, trunk_sway_deg=3.0,
                   wander_m=0.03)   # eyes open [module]; times SENSORY_EC_GAIN with the eyes closed
SENSORY = {                         # [module, unsourced: the catalogue row A13 gives kinematics as module defaults]
    "open": dict(step_m=0.50, cadence_spm=98.0, step_width_m=_wider(0.19), head_flexion_deg=30.0, trunk_lean_deg=10.0,
                 arm_abduction_deg=18.0, gain=1.0),
    "closed": dict(step_m=0.44, cadence_spm=92.0, step_width_m=_wider(0.23), head_flexion_deg=0.0, trunk_lean_deg=8.0,
                   arm_abduction_deg=26.0, gain=SENSORY_EC_GAIN)}   # no looking down with the eyes closed (at 8 deg
#                                     of head flexion on the 8 deg trunk lean the head read 10 deg down, not up)
FOG_SHUFFLE = dict(step_m=0.04, cadence_spm=270.0, knee_swing_deg=14.0, clearance_m=0.013, contact_pitch_deg=0.0,
                   liftoff_pitch_deg=6.0, swing_ankle_deg=8.0, hip_overshoot_deg=0.0, arm_swing_deg=2.0,
                   trunk_lean_deg=14.0)  # inside a shuffling episode [module, unsourced: the row P17's values; the
#                                          cadence 300 steps/min before wave SH1d: the flat toe-offs of its 0.17 s
#                                          swings launched the sole 27 to 31 mm a frame within 5 mm of the floor,
#                                          at the bound of the continuity check's floor skim; the clearance 5 mm
#                                          before 2026-11-11: in the turn's stall, whose steps move a foot 6 mm, the
#                                          heel (8 mm above the floor at rest) stayed within 5 mm of its rest height
#                                          through each swing, so that the page's skate check read two stances and
#                                          the swing between as one contact travelling 7.7 mm]
FOG_RAMP_S = 1.4                # [module] the shuffle sets in and ends over this (a raised cosine centred on each edge
#                                 of an episode: the steps shorten and quicken over about three steps before the freeze
#                                 and lengthen again after it); at 0.3 s the pelvis stopped at 0.77 g and relaunched at
#                                 0.85 g, as against a wall, against 0.25 g at the level 0's gait initiation
FOG_CADENCE_POWER = 2.0         # [module] the cadence eased on the squared envelope of an episode (episodes_fn): it
#                                 rises once the steps have shortened and falls before they lengthen again (eased
#                                 together, a release step half lengthened at nearly the shuffle's cadence swung its
#                                 knee from 19 to 49 deg in 40 ms and the pelvis outran the walk, in an intermediate
#                                 build of wave SH1d; the 2026-11-03 review read the knee from 8 to 52 deg in two
#                                 frames, 128 m/s2)
FOG_LIFT_POWER = 3.0            # [module] the swing's lift (clearance, knee) eased on the cubed envelope of an episode
FOG_STEP_POWER = 0.5            # [module] the step length on the mirrored square of the envelope (episodes_fn): at an
#                                 offset the steps stay short until the lift has come back, and shorten first at an
#                                 onset (on the envelope's fourth root the relaunch hit 4.1 m/s2; eased as the
#                                 cadence, then with the lift on the squared envelope and the steps on the envelope
#                                 itself, the releases still launched 10 to 30 cm swings within 2 to 6 mm of the
#                                 floor, the start episode's sliding the right foot flat 35 cm in 0.23 s, wave SH1d
#                                 review of lane gait)
FOG_POWER = {"cadence_spm": FOG_CADENCE_POWER, "clearance_m": FOG_LIFT_POWER, "knee_swing_deg": FOG_LIFT_POWER,
             "step_m": FOG_STEP_POWER}
FOG_DIP_SYNERGY = (0.0, 0.5, 2.0)   # [module] the shuffle's dip lift (dip_synergy): a toe scraping right after a
#                                 flat toe-off is lifted by the ankle (dorsiflexion 2 d) and a little knee (0.5 d), no
#                                 hip; with the walk's synergy (hip d, knee 2 d) the first toe-off of an episode's onset
#                                 threw the knee from 34 to 73 deg in 30 ms (172 m/s2, an intermediate build of
#                                 wave SH1d)
FOG_PUSH_OFF_ONSET_POWER = FOG_STEP_POWER   # [module] the push-off (liftoff_pitch_deg) eased at an episode's onset
#                                 as the step length (onset_fn: the mirrored power, it falls to the shuffle's sooner),
#                                 at its offset on the envelope itself as before: a short step pushes off less. Eased on
#                                 the envelope at the straight episode's onset, the left foot rolled to 36 deg of pitch
#                                 before a 0.23 s swing, which threw it 14 cm back at 2.95 m/s and read 102.6 m/s2 at the
#                                 heel on the page (bound 100); at the onsets only, 80.3 there (the clip's largest on
#                                 the page 89.4, the right heel's sole point at 22.2 s, after the straight episode's
#                                 offset, which this rule leaves as it was); eased as the step at both edges, 80.3 there
#                                 too, but the ankle 0.24 deg past its dorsiflexion bound after the start episode's
#                                 offset (3.81 to 3.86 s: the long steps back before the push-off), 0.04 deg at the power
#                                 0.7 (scripts/openbody_phaseE_checks.py 'fog_shuffling' and 'fog_shuffling_tries')


def onset_fn(outside, inside, intervals, ramp_s, power):
    """A function of time as episodes_fn, the onsets eased on the mirrored power (power below 1: 1 - (1 - rise) to
    the power 1 / power, the value leaving `outside` sooner) and the offsets on the raised cosine itself (episodes_fn's
    power 1)."""
    iv = [tuple(float(x) for x in ab) for ab in intervals]
    h = ramp_s / 2.0

    def f(t):
        t = np.asarray(t, dtype=np.float64)
        w = np.zeros_like(t)
        for a, b in iv:
            up = np.ones_like(t) if a <= 0 or ramp_s <= 0 else np.clip((t - (a - h)) / ramp_s, 0.0, 1.0)
            dn = np.clip(((b + h) - t) / ramp_s, 0.0, 1.0) if ramp_s > 0 else np.ones_like(t)
            rise = 0.5 - 0.5 * np.cos(np.pi * up)
            fall = 0.5 - 0.5 * np.cos(np.pi * dn)
            w = np.maximum(w, np.minimum(1.0 - (1.0 - rise) ** (1.0 / power), fall))
        return outside + (inside - outside) * w
    f.__doc__ = (f"{outside} outside, {inside} inside {iv} s (ramps {ramp_s} s, the onsets on the mirrored power "
                 f"{power:g})")
    return f


FOG_START_S = 2.5               # the start episode, from the gait initiation to its offset [module]
FOG_STRAIGHT_S = 3.0            # the straight-walking episode, the median of wang2020
FOG_STRAIGHT_AFTER_M = 4.0      # it starts after this much nominal progression back from the turn [module]
FOG_TURN_IN = (60.0, 1.2)       # deg, s: the turn's first part, before it stalls [module]
FOG_TURN_STALL_S = 3.0          # the turn stalls this long inside the turn episode (the median of wang2020), the
#                                 shuffling feet turning the body by FOG_TURN_CREEP_DEG only
FOG_TURN_CREEP_DEG = 15.0       # [module]
FOG_TURN_OUT_S = 1.8            # s: the rest of the turn, after the release [module]
FLUCT_RAMP_S = 0.5              # [module] a fluctuation switches over this (momentary, lempert1991)
FLUCT_BOUTS = {"first_s": (5.0, 8.0), "after_turn_s": 2.5, "second_len_s": 3.0}   # the near-normal bouts of the
#                                 fluctuating clip: one on the way out, one on the way back after the turn [module]


def episodes_fn(outside, inside, intervals, ramp_s, power=1.0):
    """A function of time: `outside` away from the intervals [(a, b)] s, `inside` within them, eased by raised
    cosines over ramp_s centred on each edge (common.envelope), the envelope e raised to power (above 1: the value
    leaves `outside` later at an onset and comes back to it sooner at an offset; below 1 the mirror 1 - (1 - e) to
    the power 1 / power: it leaves `outside` sooner and comes back later, with a finite slope at both ends of the
    ramp, where e itself to such a power rose with an unbounded slope)."""
    from .common import envelope
    iv = [tuple(float(x) for x in ab) for ab in intervals]

    def f(t):
        e = envelope(np.asarray(t, dtype=np.float64), iv, ramp_s, 1e9)
        w = e if power == 1.0 else (e ** power if power > 1.0 else 1.0 - (1.0 - e) ** (1.0 / power))
        return outside + (inside - outside) * w
    f.__doc__ = f"{outside} outside, {inside} inside {iv} s (ramps {ramp_s} s" + \
        ("" if power == 1.0 else f", envelope to the power {power:g}") + ")"
    return f


def veer_path(deg_per_m):
    """generate's path hook of a veering walk: the walking line turned by deg_per_m per metre walked (positive to
    the left), every increment of the line rotated by the heading drift accumulated up to it."""
    def fn(t, c, psi):
        d = np.diff(c, axis=0, prepend=c[:1])
        a = math.radians(deg_per_m) * np.cumsum(np.hypot(d[:, 0], d[:, 1]))
        ca, sa = np.cos(a), np.sin(a)
        r = np.stack([d[:, 0] * ca + d[:, 1] * sa, d[:, 1] * ca - d[:, 0] * sa], -1)
        return c[:1] + np.cumsum(r, 0), psi + a
    fn.__doc__ = f"heading drift of {deg_per_m:g} deg per m walked (positive to the left)"
    return fn


def _reach(prm, t_from, dist_m, dt=0.01, t_max=400.0):
    """The time at which the nominal progression (step_m(t) cadence(t) / 60 m/s) integrated from t_from covers
    dist_m."""
    tt = np.arange(float(t_from), t_max, dt)
    v = _at(prm.step_m, tt) * _at(prm.cadence_spm, tt) / 60.0
    k = int(np.searchsorted(np.cumsum(v) * dt, dist_m))
    if k >= len(tt):
        raise ValueError("the walk does not cover the distance")
    return float(tt[k])


def course_tv(prm, distance_m=8.0, turn_deg=180.0, turn_s=2.4, turn_share=None, back=True, lead_s=1.0, tail_s=1.2,
              seed=0, path_fn=None):
    """course() for parameters that change with time: the turn and the stop placed where the nominal progression,
    integrated from the gait initiation (lead_s plus half the initiation), has covered distance_m."""
    t1 = _reach(prm, lead_s + 0.5 * prm.start_s, distance_m)
    turns, t2 = [], t1
    if back:
        share = prm.turn_speed_share if turn_share is None else turn_share
        turns = [(t1 - 0.2 * turn_s, turn_deg, turn_s, share)]
        t2 = _reach(prm, t1 + turn_s, distance_m)
    cad = float(_at(prm.cadence_spm, np.array([t2]))[0])
    dur = t2 + 2.5 * prm.stop_steps * 60.0 / cad + tail_s
    b, info = generate(prm, duration_s=round(dur, 2), start_at_s=lead_s, stop_at_s=t2, turns=turns, seed=seed,
                       path_fn=path_fn)
    info["course"] = {"distance_m": distance_m, "turn_deg": turn_deg if back else 0.0, "turn_s": turn_s,
                      "turn_at_s": t1 - 0.2 * turn_s if back else None, "stop_at_s": t2,
                      "timing": "nominal progression integrated over time"}
    return b, info


PSP_SWING_ANKLE_DEG = 15.0     # [module] the PSP swing's ankle held up early in swing (10 before the open body): at 10
#                                deg the clearance lift bent the asked 38 deg knee to 58.45 on Anny's longer forefoot,
#                                at 15 it reads 52.0 against the level 0's 65.99 (wave 8 second part, D163;
#                                scripts/openbody_wave8b_checks.py 'gap_clips')


def psp_gait():
    """(GaitParams, course kwargs) of the PSP gait (catalogue row AP02): speed and cadence at the PSP group's values
    of Hatanaka (dale2023: 0.83 m/s, 100.5 steps/min, the step length their ratio), the step width at the healthy
    walk's times the PSP over control ratio of Takamatsu (dale2023: 10.6 over 7.7, units not stated, so only the
    ratio is used); stiff legs (the swing knee asked at 38 deg and the loading knee at 8, the ankle held up early in
    swing and little push-off so that the clearance lift does not bend the knee back to the healthy walk's: about
    48 deg realized against 63), an upright trunk with the neck extended (5 deg: at 12 the head looked at the
    ceiling), little arm swing with straight elbows, a slow gait initiation (1.6 s against 0.7) and the turn in the
    level 0's 2.4 s, the head and trunk turning with the pelvis (axial rigidity) [module values, unsourced; the
    classical contrast of a PSP pivot with the many-step en bloc turn of PD was not read, so the caption claims
    neither: the former 3.0 s turn read as the PD form]. Backward instability is not walked (no perturbation in the
    course): the pull-test clips of 3.12 show retropulsion (gait_E7p_retropulsion_g*)."""
    P = PSP_GAIT
    return _prm(step_m=P["speed_m_s"] * 60.0 / P["cadence_spm"], cadence_spm=P["cadence_spm"],
                step_width_m=NORMAL.step_width_m * P["width_psp"] / P["width_control"],
                knee_swing_deg=(38.0, 38.0), knee_loading_deg=(8.0, 8.0), knee_mid_deg=(2.0, 2.0),
                knee_contact_deg=(1.0, 1.0), hip_overshoot_deg=(1.0, 1.0), contact_pitch_deg=(-13.0, -13.0),
                liftoff_pitch_deg=(40.0, 40.0), swing_ankle_deg=(PSP_SWING_ANKLE_DEG, PSP_SWING_ANKLE_DEG),
                swing_ankle_share=0.35, clearance_m=(0.014, 0.014), trunk_lean_deg=-2.0, head_flexion_deg=-5.0,
                arm_abduction_deg=(17.0, 17.0), arm_swing_deg=(9.0, 9.0), elbow_deg=(8.0, 8.0),
                elbow_swing_deg=(4.0, 4.0), pelvis_rotation_deg=2.5, trunk_counter_rotation=0.2, turn_lead_s=0.05,
                start_s=1.6, step_time_cv=0.035, step_length_cv=0.05, step_width_sd_m=0.02, trunk_sway_deg=2.0), \
        {"turn_s": 2.4, "turn_share": 0.3}


def dromedary():
    """(GaitParams, course kwargs) of the dromedary gait (catalogue row D19, DYT1-like): a forward trunk flexion of
    DROMEDARY['pump_deg'] once per stride, locked to the left stance (its peak at DROMEDARY['pump_peak'] of the
    left cycle), 0.4 of it at the hips, on a slightly slow walk [module values, unsourced: the row read no
    source]."""
    D = DROMEDARY
    return _prm(step_m=0.58, cadence_spm=104.0, pump_deg=D["pump_deg"], pump_peak=D["pump_peak"],
                pump_hip_share=D["pump_hip_share"], trunk_lean_deg=4.0, arm_swing_deg=(12.0, 12.0)), \
        {"turn_s": 3.0}


def vestibular(eyes, side="left"):
    """(GaitParams, course kwargs) of the vestibular gait of a side's lesion (catalogue row G14): one passage of 8 m
    veering toward the lesion (VESTIBULAR: 0.8 deg per m with the eyes open, 3 with them closed), the eyes-closed
    walk slower, wider and more variable, the arms more abducted, the head held still on the trunk; the planted foot
    does not pivot with the slow drift (each footprint lands on the new heading; pivoting a flat foot about its
    rolling contact point slid it by about 1 cm/s) [module values, unsourced]."""
    v = VESTIBULAR[eyes]
    return _prm(step_m=v["step_m"], cadence_spm=v["cadence_spm"], step_width_m=v["step_width_m"],
                step_width_sd_m=v["step_width_sd_m"], step_length_cv=v["step_length_cv"],
                step_time_cv=v["step_time_cv"], trunk_sway_deg=v["trunk_sway_deg"],
                arm_abduction_deg=(v["arm_abduction_deg"],) * 2, head_level=0.9, trunk_counter_rotation=0.4,
                pelvis_rotation_deg=4.0, pivot_share=0.0), \
        {"back": False, "distance_m": 8.0, "path_fn": veer_path(SIGN[side] * v["veer_deg_per_m"])}


SENSORY_HIP_PEAK_SHARE = 0.72   # [module] the thrown-forward swing hip peaks earlier (hip_peak_share, as the
#                                 steppage's):
#                                 at 0.82 its 14 deg return to the landing in the last 15 percent of the swing moved the
#                                 landing foot at 102 m/s2 (the bound of gait.continuity 100)


def sensory_ataxic(eyes):
    """(GaitParams, course kwargs) of the sensory ataxic gait (catalogue row A13): high steps (7 cm of clearance
    asked, the knee about 86 deg, the sole up to about 12 cm, the leg thrown forward past the landing by 14 deg of
    hip flexion, the lift peaking before mid swing so that the foot comes down steeply; at 5 cm and 75 deg the
    close-up read as a healthy step), the heel striking with the toes 24 deg up and the forefoot slapped down within
    4 percent of the stride (the stamping), a wide base; eyes open: looking at the feet (the head flexed 30 deg on a
    trunk leaning 10 deg); eyes closed: the head up (not flexed: no use in looking at the feet), shorter, slower and
    wider steps, the step variability, sway and wander times SENSORY_EC_GAIN (the stance Romberg ratio of
    anagnostou2024), one passage of 6 m [module values, unsourced but for that ratio]."""
    s = SENSORY[eyes]
    kw = {"turn_s": 3.4, "turn_share": 0.3, "distance_m": 8.0} if eyes == "open" else \
        {"back": False, "distance_m": 6.0}
    return _prm(step_m=s["step_m"], cadence_spm=s["cadence_spm"], step_width_m=s["step_width_m"],
                stand_width_m=_wider(0.22), clearance_m=(0.07, 0.07), knee_swing_deg=(85.0, 85.0),
                hip_overshoot_deg=(14.0, 14.0), clear_at=0.52, clear_width=0.16, contact_pitch_deg=(-24.0, -24.0),
                flat_share=(0.035, 0.035), knee_contact_deg=(0.0, 0.0), liftoff_pitch_deg=(58.0, 58.0),
                head_flexion_deg=s["head_flexion_deg"], trunk_lean_deg=s["trunk_lean_deg"],
                arm_abduction_deg=(s["arm_abduction_deg"],) * 2, arm_swing_deg=(8.0, 8.0),
                hip_peak_share=SENSORY_HIP_PEAK_SHARE, **{k: v * s["gain"] for k, v in SENSORY_VAR.items()}), kw


def functional_drag(side="left"):
    """(GaitParams, course kwargs) of the functional dragging monoplegic leg (catalogue row F08; nonnekes2020 sign
    groups): the leg of side dragged behind, its swing a slide of the forefoot on the floor (drag_sides: a
    deliberate skate), the ankle slightly plantarflexed, the knee flexing only as much as the slide needs (about 30
    deg at most, as the leg passes under the hip), the leg turned out 22 deg, no circumduction, no hip hike and no
    drop of that side of the pelvis in its swing (the hemiparetic signs absent), its steps short (step-to: the
    dragged foot brought up beside the sound one, its ankle about 7 cm ahead, step_asymmetry), the sound leg stepping
    ahead; slow (0.33 m/s nominal) on a 4 m course [module values, unsourced]. Brought level with the sound foot or
    behind it (step_asymmetry -0.85 to -1.0), the leg no longer reached the floor late in its swing and the forefoot
    floated up to 1.5 cm over 5 to 9 percent of the swing: the slide would no longer hold."""
    return _prm(step_m=0.28, cadence_spm=70.0, step_asymmetry=-0.7 * SIGN[side],
                stance_share=_two(side, 0.55, 0.72), drag_sides=(side,), clearance_m=_two(side, None, 0.016),
                knee_swing_deg=_two(side, 8.0, 58.0), hip_overshoot_deg=_two(side, 0.0, 4.0),
                swing_ankle_deg=_two(side, -4.0, 3.0), swing_ankle_share=_two(side, 0.5, 0.45),
                contact_pitch_deg=_two(side, 6.0, -15.0), liftoff_pitch_deg=_two(side, 8.0, 60.0),
                flat_share=_two(side, 0.05, 0.09), toe_out_deg=_two(side, 22.0, 6.0),
                pelvis_obliquity_deg=_two(side, 2.5, 0.0), pelvis_rotation_deg=3.0, arm_swing_deg=(10.0, 10.0),
                arm_abduction_deg=(14.0, 14.0)), {"turn_s": 4.0, "turn_share": 0.25, "distance_m": 4.0}


FLUCT_ABNORMAL = {k: pair(getattr(functional("walking_on_ice"), k))[0] for k in
                  ("step_m", "cadence_spm", "step_width_m", "knee_swing_deg", "arm_swing_deg", "trunk_lean_deg",
                   "elbow_deg", "arm_abduction_deg", "arm_flexion_deg", "elbow_swing_deg", "contact_pitch_deg",
                   "liftoff_pitch_deg", "swing_ankle_deg", "hip_overshoot_deg", "clearance_m")}
#                 the walking on ice of functional() [module]: the values that change in a fluctuation
FLUCT_NORMAL = dict(step_m=0.60, cadence_spm=108.0, step_width_m=_wider(0.10), knee_swing_deg=58.0, arm_swing_deg=13.0,
                    trunk_lean_deg=5.0, elbow_deg=14.0, arm_abduction_deg=12.0, arm_flexion_deg=2.0,
                    elbow_swing_deg=8.0, contact_pitch_deg=-15.0, liftoff_pitch_deg=58.0, swing_ankle_deg=4.0,
                    hip_overshoot_deg=3.0, clearance_m=0.015)       # the near-normal bouts [module]


def functional_fluctuating(lead_s=1.0, distance_m=8.0, turn_s=3.4, turn_share=0.3):
    """(GaitParams, course kwargs) of a functional gait with momentary fluctuations (catalogue row F09,
    lempert1991): the walking-on-ice pattern of functional() (short, wide, flat-footed, stiff-kneed steps, arms
    abducted) with two bouts of near-normal walking (FLUCT_NORMAL) inside one clip, one on the way out
    (FLUCT_BOUTS['first_s']) and one on the way back (FLUCT_BOUTS['after_turn_s'] after the turn, for
    FLUCT_BOUTS['second_len_s']), each switch over FLUCT_RAMP_S; the stance knee, the pelvic rotation and the trunk
    counter-rotation held between the two patterns [module values; the bouts' timing unsourced]. The course kwargs
    carry the plan (popped by the build). The incongruence across tasks (backward, running, a swivel chair) is not
    shown."""
    def with_bouts(bouts):
        kw = {k: episodes_fn(FLUCT_ABNORMAL[k], FLUCT_NORMAL[k], bouts, FLUCT_RAMP_S) for k in FLUCT_ABNORMAL}
        for k in ("knee_swing_deg", "arm_swing_deg", "elbow_deg", "arm_abduction_deg", "arm_flexion_deg",
                  "elbow_swing_deg", "contact_pitch_deg", "liftoff_pitch_deg", "swing_ankle_deg",
                  "hip_overshoot_deg"):
            kw[k] = (kw[k], kw[k])
        # the flat-footed steps' toe scrape lifted by the ankle (FOG_DIP_SYNERGY, as the shuffle's): with the walk's
        # synergy the turn's flat toe-off flexed the right knee from 14 to 34 deg in 30 ms (703 deg/s at 30 fps, above
        # KNEE_RATE_MAX_DEG_S, wave SH1d)
        return _prm(knee_mid_deg=(8.0, 8.0), knee_loading_deg=(14.0, 14.0), pelvis_rotation_deg=3.0,
                    trunk_counter_rotation=0.3, swing_ankle_share=0.35, stand_width_m=0.17,
                    dip_synergy=FOG_DIP_SYNERGY, **kw)
    b1 = tuple(FLUCT_BOUTS["first_s"])
    p = with_bouts([b1])
    t1 = _reach(p, lead_s + 0.5 * p.start_s, distance_m)
    turn = (t1 - 0.2 * turn_s, t1 + 0.8 * turn_s)
    b2 = (turn[1] + FLUCT_BOUTS["after_turn_s"], turn[1] + FLUCT_BOUTS["after_turn_s"] + FLUCT_BOUTS["second_len_s"])
    p = with_bouts([b1, b2])
    return p, {"turn_s": turn_s, "turn_share": turn_share, "distance_m": distance_m, "lead_s": lead_s,
               "plan": {"near_normal_bouts_s": [list(b1), [round(x, 4) for x in b2]],
                        "turn_s": [round(x, 4) for x in turn], "switch_s": FLUCT_RAMP_S}}


def fog_turn_pulses(t_turn, share):
    """The turn of the freezing clip as generate's yaw-rate pulses [(t_s, deg, dur_s, share)]: FOG_TURN_IN into the
    turn, a stall of FOG_TURN_STALL_S in which the shuffling feet turn the body by FOG_TURN_CREEP_DEG only, the rest
    of the 180 deg over FOG_TURN_OUT_S after the release."""
    a_in, d_in = FOG_TURN_IN
    t_out = t_turn + d_in + FOG_TURN_STALL_S
    return [(t_turn, a_in, d_in, share), (t_turn + d_in, FOG_TURN_CREEP_DEG, FOG_TURN_STALL_S, share),
            (t_out, 180.0 - a_in - FOG_TURN_CREEP_DEG, FOG_TURN_OUT_S, share)]


def fog_shuffling(lead_s=1.0, distance_m=8.0, turn_share=0.05):
    """(GaitParams, episodes, course timing) of a 3.10 grade-1 walk (parkinsonian(1)) with three freezing episodes
    of the shuffling-forward form (catalogue row P17; schaafsma2003): small fast steps (FOG_SHUFFLE: 4 cm at 270
    steps/min, 4.5 steps a second, each leg at 2.25 Hz, below the 3 to 8 Hz freeze band of grades.SOURCES['moore2008'],
    which is the trembling form's; module values) at the start (FOG_START_S), in the turn and once in straight
    walking (FOG_STRAIGHT_S, the median of wang2020), FOG_STRAIGHT_AFTER_M after the turn: MDS-UPDRS 3.11 grade 3 on
    the program's map (fog_grades: two trigger episodes and one in straight walking). The turn episode covers the
    turn's stall (fog_turn_pulses: 60 deg in, then FOG_TURN_STALL_S of shuffling that turns the body by 15 deg, the
    rest after the release), so that the turn itself stops progressing, not only the steps (in 3.2 s at about 97
    deg/s inside the episode, the former turn read as a quick pivot on tiny steps). Each episode sets in and ends
    over FOG_RAMP_S, centred on its onset and offset (the push-off eased at the onsets as the step length,
    FOG_PUSH_OFF_ONSET_POWER)."""
    p1, _ = parkinsonian(1)
    base = {k: (pair(getattr(p1, k))[0] if not callable(getattr(p1, k)) else None) for k in FOG_SHUFFLE}

    def with_windows(win):
        kw = {k: episodes_fn(base[k], FOG_SHUFFLE[k], win, FOG_RAMP_S, FOG_POWER.get(k, 1.0)) for k in FOG_SHUFFLE}
        kw["liftoff_pitch_deg"] = onset_fn(base["liftoff_pitch_deg"], FOG_SHUFFLE["liftoff_pitch_deg"], win,
                                           FOG_RAMP_S, FOG_PUSH_OFF_ONSET_POWER)
        for k in ("knee_swing_deg", "contact_pitch_deg", "liftoff_pitch_deg", "swing_ankle_deg",
                  "hip_overshoot_deg", "arm_swing_deg"):
            if k == "arm_swing_deg":           # the grade-1 asymmetry kept outside the episodes
                kw[k] = tuple(episodes_fn(a, FOG_SHUFFLE[k], win, FOG_RAMP_S) for a in pair(p1.arm_swing_deg))
            else:
                kw[k] = (kw[k], kw[k])
        kw["clearance_m"] = (kw["clearance_m"], kw["clearance_m"])
        iv = tuple(tuple(float(x) for x in ab) for ab in win)

        def dip_fn(t_to):                      # the shuffle's dip synergy on the swings of the episodes and their
            #                                    ramps only: outside them the walk is the grade 1 walk's
            return FOG_DIP_SYNERGY if any(a - FOG_RAMP_S <= t_to <= b + FOG_RAMP_S for a, b in iv) else None
        dip_fn.__doc__ = f"FOG_DIP_SYNERGY {FOG_DIP_SYNERGY} for a toe-off within {FOG_RAMP_S} s of {iv} s, else None"
        return replace(p1, swing_ankle_share=0.35, dip_synergy=dip_fn, **kw)
    w_start = (0.0, lead_s + FOG_START_S)
    p = with_windows([w_start])
    t1 = _reach(p, lead_s + 0.5 * p.start_s, distance_m)
    t_turn = t1 - 0.2 * FOG_TURN_IN[1]
    turns = fog_turn_pulses(t_turn, turn_share)
    stall = (turns[1][0], turns[2][0])
    t_end = turns[2][0] + turns[2][2]
    w_turn = (t_turn + 0.5 * FOG_TURN_IN[1], stall[1] + 0.4)
    p = with_windows([w_start, w_turn])
    t_s = _reach(p, t_end, FOG_STRAIGHT_AFTER_M)
    w_str = (t_s, t_s + FOG_STRAIGHT_S)
    p = with_windows([w_start, w_turn, w_str])
    t2 = _reach(p, t_end + 0.4, distance_m)
    episodes = [{"context": "start", "onset_s": lead_s, "offset_s": w_start[1], "manifestation": "shuffling"},
                {"context": "turn", "onset_s": w_turn[0], "offset_s": w_turn[1], "manifestation": "shuffling",
                 "turn_stall_s": [round(x, 4) for x in stall]},
                {"context": "straight", "onset_s": w_str[0], "offset_s": w_str[1], "manifestation": "shuffling"}]
    for e in episodes:
        e["duration_s"] = round(e["offset_s"] - e["onset_s"], 4)
    timing = {"lead_s": lead_s, "turns": turns, "turn_at_s": t_turn, "turn_end_s": t_end, "stall_s": stall,
              "stop_at_s": t2, "duration_s": round(t2 + 2.5 * p.stop_steps * 60.0 / pair(p1.cadence_spm)[0] + 1.2, 2),
              "distance_m": distance_m}
    return p, episodes, timing


# ---------------------------------------------------------------- read-backs and the balance check
def _xcom(body, floor):
    """(T, 2) the extrapolated centre of mass in (x, z) (hof2005 as in ataxia.xcom_margin: x + v / omega0, omega0 =
    sqrt(g / l), l the centre of mass's height over the floor; de Leva table, kernels.force.com)."""
    from ..kernels.force import com
    c = com(body.joints().numpy(), int(body.gender))
    v = np.gradient(c, 1.0 / body.rate, axis=0)
    w0 = np.sqrt(9.81 / np.maximum(c[:, 1] - floor, 0.3))
    return c[:, [0, 2]] + v[:, [0, 2]] / w0[:, None]


BALANCE_MOS_P2_M = 0.0          # [module] the 2nd percentile of the mediolateral margin of stability is not negative


def gait_balance(body, info):
    """The balance check of a walking body, JSON-ready: the mediolateral margin of stability (Hof 2005: the lateral
    border of the single-support foot's sole cloud minus the extrapolated centre of mass, in the walking heading's
    frame; positive inside) over the generator's single-support knots, and the share of the standing knots (both feet
    planted, the gait amplitude 0) whose extrapolated centre of mass lies inside the convex hull of both soles. ok:
    the margin's 2nd percentile at least BALANCE_MOS_P2_M and every standing knot inside."""
    from scipy.spatial import Delaunay
    clouds = {s: sole_cloud_points(body, s) for s in SIDES}
    lows = np.stack([clouds[s][..., 1].min(-1) for s in SIDES], -1)
    floor = float(np.median(lows.min(1)))
    X = _xcom(body, floor)
    psi = np.asarray(info["heading_rad"])
    lft = np.stack([np.cos(psi), -np.sin(psi)], -1)
    st = np.asarray(info["stance"], bool)
    single = st[:, 0] ^ st[:, 1]
    mos = []
    for i, side in enumerate(SIDES):
        k = np.flatnonzero(single & st[:, i])
        if not len(k):
            continue
        lat = (clouds[side][k][:, :, [0, 2]] * lft[k][:, None]).sum(-1)
        edge = lat.max(1) if side == "left" else lat.min(1)
        xl = (X[k] * lft[k]).sum(-1)
        mos.append(SIGN[side] * (edge - xl))
    mos = np.concatenate(mos) if mos else np.zeros(0)
    stand = st[:, 0] & st[:, 1] & (np.asarray(info["amplitude"]) <= 0.0)
    inside = []
    for a, b in _runs(stand):
        P = np.concatenate([clouds[s][a][:, [0, 2]] for s in SIDES])
        tri = Delaunay(P)
        inside.append(tri.find_simplex(X[a:b + 1]) >= 0)
    inside = np.concatenate(inside) if inside else np.ones(0, bool)
    p2 = float(np.percentile(mos, 2)) if len(mos) else None
    ok = (p2 is None or p2 >= BALANCE_MOS_P2_M) and bool(inside.all())
    return {"ml_mos_min_m": float(mos.min()) if len(mos) else None, "ml_mos_p2_m": p2,
            "ml_mos_median_m": float(np.median(mos)) if len(mos) else None, "n_single_support": int(len(mos)),
            "standing_knots": int(len(inside)), "standing_inside_share": float(inside.mean()) if len(inside) else None,
            "ok": bool(ok), "mos_p2_floor_m": BALANCE_MOS_P2_M}


def _moving_mean(x, n):
    n = max(int(n), 1)
    pad = (n // 2, n - 1 - n // 2)
    return np.stack([np.convolve(np.pad(x[:, j], pad, mode="edge"), np.ones(n) / n, mode="valid")
                     for j in range(x.shape[1])], -1)


def path_drift(body, min_speed=0.3, edge_m=1.0):
    """The walking path's heading drift from the body alone, JSON-ready: the pelvis path smoothed over 1 s (the
    pelvic sway removed); over the knots where it moves faster than min_speed, its direction against the distance
    walked (a line fit away from the first and last edge_m: deg per m, positive to the left), the change between its
    mean directions over the first and the last edge_m walked, and the lateral deviation of the end from the line of
    the first edge_m (positive to the left)."""
    p = _moving_mean(body.joints().numpy()[:, 0, [0, 2]], int(round(body.rate)))
    v = np.gradient(p, 1.0 / body.rate, axis=0)
    sp = np.linalg.norm(v, axis=1)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
    mov = np.flatnonzero(sp > min_speed)
    a, b = int(mov[0]), int(mov[-1])
    th = np.unwrap(np.arctan2(v[:, 0], v[:, 1]))
    sel = (sp > min_speed) & (s > s[a] + edge_m) & (s < s[b] - edge_m)
    slope = float(np.polyfit(s[sel], np.degrees(th[sel]), 1)[0])
    ka = int(np.searchsorted(s, s[a] + edge_m))
    kb = int(np.searchsorted(s, s[b] - edge_m))
    u0 = (p[ka] - p[a]) / max(np.linalg.norm(p[ka] - p[a]), 1e-9)
    u1 = (p[b] - p[kb]) / max(np.linalg.norm(p[b] - p[kb]), 1e-9)
    change = math.degrees(math.atan2(u0[1] * u1[0] - u0[0] * u1[1], float(u0 @ u1)))
    left0 = np.array([u0[1], -u0[0]])
    return {"deg_per_m": slope, "heading_change_deg": change, "lateral_m": float((p[b] - p[a]) @ left0),
            "path_m": float(s[b] - s[a])}


def trunk_cycle(body, m=None):
    """The trunk's flexion over each left stride of the body, JSON-ready: the sagittal angle of the pelvis-to-neck
    line in the walking direction (as measure_gait), per stride between consecutive left heel strikes of
    measure_gait's steps (no turn over the stride), its range and its peak's share of the stride (circular mean),
    the pelvis's own forward pitch range likewise, and the strides' frequency."""
    m = measure_gait(body) if m is None else m
    J = body.joints().numpy()
    psi = _smooth(np.unwrap(_root_yaw(body)), int(round(body.rate)))
    c, s_ = np.cos(psi), np.sin(psi)
    tr = J[:, 12] - J[:, 0]
    ang = np.degrees(np.arctan2(tr[:, 0] * s_ + tr[:, 2] * c, tr[:, 1]))
    up = body.fk()[1][:, 0, :, 1].numpy()                   # the pelvis's own up axis in the world
    pel = np.degrees(np.arctan2(up[:, 0] * s_ + up[:, 2] * c, up[:, 1]))
    hs = [st["k1"] for st in m["steps"] if st["side"] == "left"]
    rng_, ph, prng, T_ = [], [], [], []
    for k0, k1 in zip(hs, hs[1:]):
        if abs(math.degrees(psi[k1] - psi[k0])) > 15.0 or k1 - k0 < 20:
            continue
        seg = ang[k0:k1]
        rng_.append(float(seg.max() - seg.min()))
        ph.append(2 * math.pi * int(np.argmax(seg)) / (k1 - k0))
        T_.append((k1 - k0) / body.rate)
        prng.append(float(pel[k0:k1].max() - pel[k0:k1].min()))
    if not rng_:
        return {"n": 0}
    peak = (math.atan2(np.mean(np.sin(ph)), np.mean(np.cos(ph))) / (2 * math.pi)) % 1.0
    return {"n": len(rng_), "range_deg": float(np.median(rng_)), "peak_share": float(peak),
            "pelvis_pitch_range_deg": float(np.median(prng)),
            "stride_hz": float(1.0 / np.median(T_)), "mean_flexion_deg": float(ang.mean())}


def _root_yaw(body):
    from ..body.rotations import yaw_angle
    return yaw_angle(body.local_R()[:, 0]).numpy()


def leg_frequency(body, a_s, b_s, side):
    """Hz: the stepping frequency of one leg over [a_s, b_s] s, from the body alone: the peak between 0.3 and 8 Hz of
    the Hann-windowed periodogram (zero-padded to 0.01 Hz, the linear trend removed) of the horizontal speed of
    side's ankle joint, which peaks once per stride (in its swing)."""
    _, _, an, _ = LEG[side]
    J = body.joints().numpy()
    v = np.gradient(J[:, an][:, [0, 2]], 1.0 / body.rate, axis=0)
    k0, k1 = int(round(a_s * body.rate)), int(round(b_s * body.rate))
    x = np.hypot(v[k0:k1, 0], v[k0:k1, 1])
    u = np.arange(len(x), dtype=np.float64)
    x = (x - np.polyval(np.polyfit(u, x, 1), u)) * np.hanning(len(x))
    n = int(round(100.0 * body.rate))
    X = np.abs(np.fft.rfft(x, n)) ** 2
    f = np.fft.rfftfreq(n, 1.0 / body.rate)
    sel = (f >= 0.3) & (f <= 8.0)
    return float(f[sel][np.argmax(X[sel])])


def turn_rate(body, a_s, b_s):
    """deg/s: the mean rate of the pelvis heading over [a_s, b_s] s, its net change over the span (the heading
    smoothed over 0.4 s, the pelvic rotation of the steps removed), positive to the left."""
    psi = np.degrees(_smooth(np.unwrap(_root_yaw(body)), int(round(0.4 * body.rate))))
    k0, k1 = int(round(a_s * body.rate)), int(round(b_s * body.rate))
    return float((psi[k1] - psi[k0]) / max(b_s - a_s, 1e-9))


def steps_in(m, a, b):
    """measure_gait's steps whose heel strike lies inside [a, b] s."""
    return [s for s in m["steps"] if a <= s["t"] <= b]


def window_read(body, m, a, b):
    """Median step length, width and time, the step rate and the pelvis's net progression speed over [a, b] s."""
    st = steps_in(m, a, b)
    J = body.joints().numpy()
    k0, k1 = int(round(a * body.rate)), int(round(b * body.rate))
    out = {"from_s": a, "to_s": b, "n_steps": len(st),
           "pelvis_speed_m_s": float(np.linalg.norm(J[k1, 0, [0, 2]] - J[k0, 0, [0, 2]]) / max(b - a, 1e-9))}
    if st:
        out.update(step_length_m=float(np.median([s["length_m"] for s in st])),
                   step_width_m=float(np.median([s["width_m"] for s in st])),
                   step_time_s=float(np.median([s["time_s"] for s in st])))
        out["step_rate_hz"] = 1.0 / out["step_time_s"]
    return out


def drag_read(body, info, side, tol_m=FLOOR_TOL_M):
    """The dragged foot read back from the body, JSON-ready: over the generator's swings of side (the deliberate
    skate), the share of the swing knots whose lowest sole point lies within tol_m of the floor, the horizontal speed
    of that point while it does (the same sole point followed from knot to knot, as stance_slip), the distance it
    slides per swing and where the swing ends (the dragged ankle ahead of the sound one along the heading, over the
    swings turning less than 15 deg); the generator's stance knots keep their exact contact (stance_slip). The slide
    per swing is the net horizontal travel of the touching sole while it touches (the followed point's velocities
    summed as vectors); the path those velocities trace (their lengths summed: the toe pad sweeping about the ankle
    as the dragged foot pitches, longer on Anny's longer forefoot) is reported beside it (wave 8 second part, D165:
    0.648 m of path against 0.550 m of the ankle's net travel on a 0.56 m stride)."""
    i = SIDES.index(side)
    X = sole_cloud_points(body, side)
    lows = np.stack([sole_cloud_points(body, s)[..., 1].min(-1) for s in SIDES], -1)
    floor = float(np.median(lows.min(1)))
    j = np.argmin(X[..., 1], 1)
    low = X[np.arange(body.T), j, 1] - floor
    sw = np.asarray(info["swing_weight"])[:, i] > 0
    k = np.flatnonzero(sw[1:-1]) + 1
    on = low[k] <= tol_m
    v = (X[k + 1, j[k]] - X[k - 1, j[k]]) * body.rate / 2.0
    hv = np.hypot(v[:, 0], v[:, 2])
    slides, paths, ends = [], [], []
    Jt = body.joints().numpy()
    an, an_o = LEG[side][2], LEG[SIDES[1 - i]][2]
    yaw0 = np.unwrap(_root_yaw(body))
    for w in info["swings"][side]:
        kk = np.arange(w["k_to"] + 1, w["k_hs"])
        sel = np.isin(k, kk) & on
        slides.append(float(np.hypot(v[sel, 0].sum(), v[sel, 2].sum()) / body.rate))
        paths.append(float(hv[sel].sum() / body.rate))
        if abs(math.degrees(yaw0[w["k_hs"]] - yaw0[w["k_to"]])) < 15.0:     # where the swing ends: its ankle
            h = yaw0[w["k_hs"]]                                              # ahead of the sound one's
            ends.append(float((Jt[w["k_hs"], an, [0, 2]] - Jt[w["k_hs"], an_o, [0, 2]]) @
                              np.array([math.sin(h), math.cos(h)])))
    # the walk itself, read from the sound foot (the dragged one never leaves the floor, so measure_gait finds no
    # alternating steps): its stance runs, the strides between their starts, the pelvis's progression over the
    # strides that turn less than 15 deg
    lo = lows[:, 1 - i]
    runs = [(a, b) for a, b in _runs(lo <= floor + tol_m) if a > 0 and b - a >= int(round(0.08 * body.rate))]
    hs = np.array([a for a, _ in runs])
    yaw = np.unwrap(_root_yaw(body))
    P = body.joints().numpy()[:, 0, [0, 2]]
    strides = [(a, b) for a, b in zip(hs, hs[1:]) if abs(math.degrees(yaw[b] - yaw[a])) < 15.0]
    st_s = np.array([(b - a) / body.rate for a, b in strides])
    spd = np.array([np.linalg.norm(P[b] - P[a]) / ((b - a) / body.rate) for a, b in strides])
    return {"side": side, "phase": "swing", "deliberate": True, "contact_share": float(on.mean()) if len(on) else 0.0,
            "slide_speed_median_m_s": float(np.median(hv[on])) if on.any() else 0.0,
            "slide_m_per_swing_median": float(np.median(slides)) if slides else 0.0, "n_swings": len(slides),
            "slide_path_m_per_swing_median": float(np.median(paths)) if paths else 0.0,
            "end_ahead_of_sound_m_median": float(np.median(ends)) if ends else None,
            "lowest_m": float(low[k].min()) if len(k) else None,
            "sound_strides": int(len(strides)),
            "cadence_spm": float(120.0 / np.median(st_s)) if len(st_s) else None,
            "speed_m_s": float(np.median(spd)) if len(spd) else None}


def _gap_truth(body, info, operator, grade, scale, sources, extra, measure=True):
    """_truth plus the balance check (gait_balance) under validity['balance'] and validity['gait']['balance_ok']."""
    tr = _truth(body, info, operator, grade, scale, sources, extra, measure)
    bal = gait_balance(body, info)
    tr["validity"]["balance"] = bal
    tr["validity"]["gait"]["balance_ok"] = bal["ok"]
    return tr


def eyes_text(eyes, duration_s):
    """params['frame_text'] of an eye-state clip (as ataxia.py's Romberg): one span over the whole clip."""
    return [{"from_s": 0.0, "to_s": float(duration_s), "text": "EYES OPEN" if eyes == "open" else "EYES CLOSED"}]


def gap_read(kind, body, info, m=None):
    """The read-back of a gap-wave clip (params['read_back']), from the body alone but for the episode and bout
    windows of the plan."""
    m = measure_gait(body) if m is None else m
    s = summary(m)
    st = m["steps"]
    out = {"speed_steps_m_s": float(s["step_length_m"] * s["cadence_spm"] / 60.0) if s["cadence_spm"] else None,
           "first_step_s": float(st[0]["t"]) if st else None}
    keys = ("speed_m_s", "cadence_spm", "step_length_m", "step_width_m", "step_length_cv", "step_time_cv",
            "step_width_sd_m", "trunk_lean_deg", "knee_swing_left_deg", "knee_swing_right_deg",
            "knee_loading_left_deg", "clearance_left_m", "clearance_right_m", "contact_pitch_left_deg",
            "shoulder_range_left_deg", "elbow_mean_left_deg")
    out.update({k: s.get(k) for k in keys})
    if kind == "dromedary":
        out["trunk_cycle"] = trunk_cycle(body, m)
    if kind == "vestibular":
        out["path"] = path_drift(body)
    out["head_pitch_down_deg"] = head_pitch(body)
    if kind == "drag":                       # no alternating steps to read: the walk is read in drag_read
        out = {k: v for k, v in out.items() if k in ("trunk_lean_deg", "head_pitch_down_deg",
                                                      "shoulder_range_left_deg", "elbow_mean_left_deg",
                                                      "knee_swing_right_deg", "clearance_right_m")}
    return out


def head_pitch(body, a_s=None, b_s=None):
    """deg: the mean downward pitch of the head's forward axis (the head joint's global rotation applied to the rest
    forward +z) over [a_s, b_s] (default the whole body), minus the rest pose's; positive looking down."""
    G = body.fk()[1].numpy()[:, 15]
    f = G[:, :, 2]
    pitch = np.degrees(np.arcsin(np.clip(-f[:, 1], -1.0, 1.0)))
    k0 = 0 if a_s is None else int(round(a_s * body.rate))
    k1 = body.T if b_s is None else int(round(b_s * body.rate))
    return float(pitch[k0:k1].mean())


GAP_SEED = SEED + 100


def _gap_course_build(kind, prm_fn, operator, grade, scale, sources, seed, unsourced, eyes=None, tv=False,
                      extra_fn=None):
    """build() of a gap-wave course clip: course (or course_tv for parameters that change with time), the read-back
    (gap_read), the module values (unsourced), the eye state (frame_text) and the balance check (_gap_truth)."""
    def build():
        p, ckw = prm_fn()
        ckw = dict(ckw)
        d = ckw.pop("distance_m", 8.0)
        plan = ckw.pop("plan", None)
        b, info = (course_tv if tv else course)(p, distance_m=d, seed=seed, **ckw)
        if plan is not None:
            info["plan"] = plan
        m = measure_gait(b)
        extra = {"unsourced": list(unsourced), "read_back": gap_read(kind, b, info, m)}
        if eyes is not None:
            extra["eyes"] = eyes
            extra["frame_text"] = eyes_text(eyes, b.duration)
        if extra_fn is not None:
            extra.update(extra_fn(b, info, m))
        return b, _gap_truth(b, info, operator, grade, scale, sources, extra)
    return build


def _drag_extra(side):
    def fn(b, info, m):
        return {"deliberate_skate": drag_read(b, info, side)}
    return fn


def _fluct_extra(b, info, m):
    """The fluctuating clip's bouts read back: each near-normal bout and each abnormal stretch between them (away
    from the switches, the start, the turn and the stop)."""
    plan = info["plan"]
    t_turn = info["course"]["turn_at_s"]
    turn = (t_turn - 0.5, t_turn + info["course"]["turn_s"] + 0.8)
    (n1a, n1b), (n2a, n2b) = plan["near_normal_bouts_s"]
    stretches = [("abnormal", info["start_at_s"] + 1.5, n1a), ("near_normal", n1a, n1b),
                 ("abnormal", n1b, turn[0]), ("abnormal", turn[1], n2a), ("near_normal", n2a, n2b),
                 ("abnormal", n2b, info["course"]["stop_at_s"] - 0.5)]
    rows = []
    for kind, a, b_ in stretches:
        a2, b2 = a + FLUCT_RAMP_S + 0.3, b_ - FLUCT_RAMP_S
        if b2 - a2 < 1.2:
            continue
        r = window_read(b, m, a2, b2)
        r["kind"] = kind
        rows.append(r)
    return {"bouts": rows, "plan": plan}


def _fog_read(b, m, a, b_):
    """window_read plus each leg's stepping frequency (leg_frequency) and the heading's mean rate (turn_rate)."""
    r = window_read(b, m, a, b_)
    r["leg_hz"] = {s: leg_frequency(b, a, b_, s) for s in SIDES}
    r["turn_rate_deg_s"] = turn_rate(b, a, b_)
    return r


def _build_fog(seed):
    def build():
        p, eps, tm = fog_shuffling()
        b, info = generate(p, duration_s=tm["duration_s"], start_at_s=tm["lead_s"], stop_at_s=tm["stop_at_s"],
                           turns=tm["turns"], seed=seed)
        info["course"] = {"distance_m": tm["distance_m"], "turn_deg": 180.0, "turn_at_s": tm["turn_at_s"],
                          "turn_s": round(tm["turn_end_s"] - tm["turn_at_s"], 4),
                          "turn_pulses": [list(x) for x in tm["turns"]],
                          "turn_stall_s": list(tm["stall_s"]), "stop_at_s": tm["stop_at_s"]}
        m = measure_gait(b)
        h = 0.5 * FOG_RAMP_S
        rows = []
        for e in eps:          # inside each episode, away from its eased edges (the start: after the initiation)
            a = e["onset_s"] + (0.6 if e["context"] == "start" else h + 0.1)
            rows.append(dict(e, read=_fog_read(b, m, a, e["offset_s"] - h)))
        walk = [(eps[0]["offset_s"] + h + 0.3, tm["turn_at_s"] - 0.5),
                (tm["turn_end_s"] + 0.8, eps[2]["onset_s"] - h - 0.3)]
        wr = [_fog_read(b, m, a, b_) for a, b_ in walk]
        (a1, _, d1, _), _, (a3, _, d3, _) = tm["turns"]
        turn = {"in": {"from_s": a1, "to_s": a1 + d1, "turn_rate_deg_s": turn_rate(b, a1, a1 + d1)},
                "out": {"from_s": a3, "to_s": a3 + d3, "turn_rate_deg_s": turn_rate(b, a3, a3 + d3)}}
        extra = {"unsourced": ["FOG_SHUFFLE (step length, cadence, knee, clearance, arm swing, trunk lean in the "
                               "shuffle)", "FOG_START_S", "FOG_STRAIGHT_AFTER_M", "FOG_RAMP_S",
                               "FOG_TURN_IN, FOG_TURN_CREEP_DEG, FOG_TURN_OUT_S (the turn around its stall)"],
                 "episodes": rows, "walking": wr, "turn_parts": turn, "fog_grade_map": {"trigger": 2, "straight": 1},
                 "base": "parkinsonian(1), MDS-UPDRS 3.10 grade 1", "read_back": gap_read("fog", b, info, m)}
        return b, _gap_truth(b, info, "freezing_shuffling", 3, "MDS-UPDRS 3.11",
                             ("mds_updrs", "schaafsma2003", "wang2020", "fog_grades", "zanardi2021", "lewek2010",
                              "module"), extra)
    return build


VEST_CLOSED_SEED = GAP_SEED + 20     # the eyes-closed vestibular walk: the first seed from GAP_SEED + 20 whose steady
#                                      steps (about 12 in one 8 m passage) realize the planned step-width SD within
#                                      20 percent (GAP_SEED + 4 realized 0.013 m of the 0.030 asked, below the eyes-open
#                                      walk's 0.021: sampling over so few steps)
S_FMDRS_GAIT = {"gait_E8_functional_dragging_left": "continuous",
                "gait_E8_functional_fluctuating": "frequent but intermittent"}   # nielsen2017's duration anchor of
#                                      the gait region, as each clip shows it [module reading]; the severity is not
#                                      mapped


def gap_specs():
    """[ClipSpec] of the gap wave (module section above); since 'catalogue_v1', level 0 the healthy walk."""
    S = GAP_SEED
    P, D = PSP_GAIT, DROMEDARY_SEEN_DEG
    out = [
        _spec("gait_E8_psp", "gait_atypical_parkinsonian", "described level", "PSP gait (progressive supranuclear "
              "palsy)", "described level", "described level (PSPRS gait not read)",
              f"{P['speed_m_s']:g} m/s, {P['cadence_spm']:.0f} steps/min (Hatanaka), base "
              f"{P['width_psp'] / P['width_control']:.2f} times wider (Takamatsu), stiff knees, upright, neck "
              "extended; no backward perturbation",     # 'slow start' dropped: the longer start ramp (start_s 1.6)
              #                                           set in 0.34 s later than the healthy walk (2026-11-03
              #                                           review), not a readable slow start
              _gap_course_build("psp", psp_gait, "psp_gait", "described level", "described level (PSPRS not read)",
                                ("dale2023", "module"), S + 1,
                                ["knee, hip and ankle values (stiff legs)", "trunk, head and arms", "start_s 1.6",
                                 "turn", "variability"])),
        _spec("gait_E8_dromedary", "gait_dystonic", "described level", "dromedary gait (DYT1-like trunk dystonia)",
              "described level", "described level (BFMDRS trunk not read)",
              f"the trunk bends forward about {D['trunk']} deg once per stride with the left stance ({D['pelvis']} at "
              "the pelvis) and comes back upright, the arms hanging",
              _gap_course_build("dromedary", dromedary, "dromedary_gait", "described level",
                                "described level (BFMDRS trunk not read)", ("module",), S + 2,
                                ["DROMEDARY (amplitude, phase, hip share)", "walk"]))]
    for eyes, seg, seed in (("open", "E8", S + 3), ("closed", "E19", VEST_CLOSED_SEED)):
        v = VESTIBULAR[eyes]
        out.append(_spec(f"gait_{seg}_vestibular_left_eyes_{eyes}", "gait_vestibular", "described level",
                         f"vestibular gait, left lesion, eyes {eyes}", "described level",
                         f"described level, eyes {eyes} (eye state stated, not visible)",
                         f"8 m walked away, veering to the left by {v['veer_deg_per_m']:g} deg per m"
                         + (", slower, wider, more irregular" if eyes == "closed" else ", slightly wide base"),
                         _gap_course_build("vestibular", lambda eyes=eyes: vestibular(eyes, "left"), "vestibular_gait",
                                           "described level", "described level", ("module",), seed,
                                           ["VESTIBULAR (veering, speed, width, variability, arms)"], eyes=eyes),
                         segment=seg, segment_title="" if seg == "E8" else "walking variants, eyes closed",
                         views=("track_left",)))        # NOT_RENDERED['vestibular gait: the veer on screen']
    cid = "gait_E8_functional_dragging_left"
    out.append(_spec(cid, "gait_functional", "described level", "functional gait, dragging monoplegic leg (left)",
                     "described level", f"described level (S-FMDRS gait duration: {S_FMDRS_GAIT[cid]})",
                     "left leg dragged behind, forefoot sliding on the floor up beside the sound "
                     "foot, turned out; no circumduction or hip hike",
                     _gap_course_build("drag", lambda: functional_drag("left"), "functional_gait", "described level",
                                       "described level", ("nonnekes2020", "lempert1991", "nielsen2017", "module"),
                                       S + 5, ["drag kinematics (knee, ankle, turn-out, step asymmetry, speed)",
                                               "S-FMDRS duration reading"],
                                       extra_fn=_drag_extra("left"))))
    (n1a, n1b), (n2a, n2b) = functional_fluctuating()[1]["plan"]["near_normal_bouts_s"]
    cid = "gait_E8_functional_fluctuating"
    out.append(_spec(cid, "gait_functional", "described level", "functional gait, momentary fluctuations",
                     "described level", f"described level (S-FMDRS gait duration: {S_FMDRS_GAIT[cid]})",
                     "short, wide, flat-footed steps, arms abducted (walking on ice), with two bouts of near-normal "
                     f"walking ({n1a:.1f} to {n1b:.1f} s out, {n2a:.1f} to {n2b:.1f} s back)",
                     _gap_course_build("fluct", functional_fluctuating, "functional_gait", "described level",
                                       "described level", ("lempert1991", "nonnekes2020", "nielsen2017", "module"),
                                       S + 6, ["FLUCT_ABNORMAL, FLUCT_NORMAL", "FLUCT_BOUTS", "FLUCT_RAMP_S",
                                               "S-FMDRS duration reading"],
                                       tv=True, extra_fn=_fluct_extra)))
    for eyes, seg, k in (("open", "E8", 8), ("closed", "E19", 9)):
        out.append(_spec(f"gait_{seg}_sensory_ataxic_eyes_{eyes}", "gait_ataxic", "SARA 1 (descriptive)",
                         f"sensory ataxic gait, eyes {eyes}", "described level",
                         f"described level, eyes {eyes} (eye state stated, not visible)",
                         ("high stamping steps on a wide base, looking at the feet (head flexed)" if eyes == "open"
                          else "high stamping steps, wider, shorter, twice as irregular and swaying, head up; "
                               "6 m, no turn"),
                         _gap_course_build("sensory", lambda eyes=eyes: sensory_ataxic(eyes), "sensory_ataxic_gait",
                                           "described level", "no rating scale (SARA 1 descriptively)",
                                           ("anagnostou2024", "module"), S + k,
                                           ["SENSORY (steps, base, head, trunk, arms)", "SENSORY_VAR",
                                            "stamping and high-step kinematics"], eyes=eyes),
                         segment=seg, segment_title="" if seg == "E8" else "walking variants, eyes closed"))
    out.append(_spec("gait_E8_fog_shuffling_g3", "gait_parkinsonian", "3.11", "freezing of gait, shuffling forward",
                     3, "MDS-UPDRS 3.11 grade 3 (two triggered episodes and one in straight walking)",
                     f"small fast steps ({100 * FOG_SHUFFLE['step_m']:.0f} cm, {FOG_SHUFFLE['cadence_spm'] / 60:g} "
                     f"a second) at the start, in the turn (stalled {FOG_TURN_STALL_S:g} s) "
                     f"and for {FOG_STRAIGHT_S:g} s walking back, on a 3.10 grade 1 walk", _build_fog(S + 7),
                     level0=PD_WALK_LEVEL0))
    return out
