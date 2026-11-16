"""The first exam set of the synthetic patient viewer (SP4 version (a)): which bodies are rendered, from which
views, with which caption, for the examination of docs/proposals/SYNTH_PATIENT_DRAFT.md (section 2) and the grade
maps of its sections 3 to 7 (mdfm/synth/grades.py).

Healthy references (draft section 7), made by the program's own generators on the catalogue's one body since the
open-body port's wave 7 (exam_v10; owner decision C8: no AMASS motion reaches the synthetic patient, retargeted or
not; docs/OPEN_BODY_PORT_DESIGN.md section 18):
- the walk: REF_WALK, the healthy walk of MDS-UPDRS 3.10 by the gait generator (gait.course on gait.NORMAL with the
  catalogue's seed: standing, gait initiation, 10 m away, a 180 degree turn, 10 m back, the stop, standing), the
  body of the catalogue's level 0 gait_E8_level0, its truth that clip's (gait.validity in it);
- the stand: REF_STAND, the procedural stand (base.POSTURES 'stand') for the segment's 15 s, the arms down; in the
  symptom catalogue the background layer of physio.py gives it its breathing, head drift and quiet sway.
Until wave 7 both were AMASS takes of the train split (CMU/15/15_03 from 35 s, KIT/441/conversation01 from knot 3950),
retargeted to the neutral SMPL+H shape (retarget_neutral) and grounded by their mesh (ground_by_mesh); amass_take,
select_stand and retarget_neutral are no longer read by the exam set and live in smplh_reference.py since wave 8,
behind the reference flag (scripts/render_mesh_video.py --select-stand keeps its record with them), ground_by_mesh
stays a generic helper.

Exam segments (draft section 2) and what this set renders of them:
- E1 seated rest (45 s): rest tremor 3.17, grades 0, 1 below and above the estimator floor, 2, 3, 4, at 5 Hz on
  the left forearm and hand, present throughout; 3.18 constancy, grades 1 to 4 at the 3.17 grade-2 level. Since
  wave SH1d (exam_v7) the tremor is the 'rest' driver of tremor.py: forearm pronation-supination with the hand
  flexing and extending at the wrist, the elbow joining as a small share from grade 3 (review M8: version 2 drove
  the elbow alone, a see-saw of the forearm under a still hand), levelled at the hand on the skinned mesh (review
  B1, and the session's decision on D4 under the owner's delegation, 2026-11-03: every hand tremor is levelled at
  the hand, as 3.15 is, because raters score the largest amplitude they see; D4's reading at the wrist is retired
  for the hand): at the fingertip that moves most (tremor.FINGERTIPS), since with the forearm's turn the thumb and
  index tips move more than the middle fingertip the decision names (tremor.py, the 'rest' driver). The forearm
  lies just above the chair's armrest, the hand past its end, and the tremor keeps clear of the chair
  (tremor.ChairRoom: the elbow would flex the forearm off the armrest just enough where a phase of the turn would
  enter a box; no exam grade needs it).
- E2 arms outstretched (30 s): postural tremor 3.15, the same ladder at 6 Hz, driven at the wrist with the shoulder
  joining from 3 cm and read at the middle fingertip (tremor.py driver 'postural', the drive of the tremor family's
  outstretched hand: the hand moves up and down; shared wave SH4, 2026-11-02: the elbow hinge, vertical in this
  posture, swept the hands sideways); the wrist joint does not move below 3 cm. One re-emergent tremor at grade 2
  after the published mean latency (grades.REEMERGENCE_MEAN_S), on the same drive.
  Both ladders are levelled at the hand since exam_v7, 3.15 at the middle fingertip (the hand flexing at the
  wrist moves the middle fingertip most), 3.17 at the fingertip that moves most (until then 3.17 was levelled at
  the wrist, decision D4 as first proposed, and its hand read a band higher at grades 1 and 2). Every E1 and E2
  tremor truth records the three readings against the clip's base (params wrist_pp_cm, middle_tip_pp_cm and
  fingertips_pp_cm, the largest windowed chord, and readings: tremor.point_readings).
  On both ladders the top grades follow the tremor family's rules (tremors.py): grades 3 and 4 slower
  (tremors.limb_hz, TOP_HZ, module values) and grade 4, the open top band, placed just above its lower edge
  (tremors.TOP_LEVEL_CM, module value) instead of the log placement of grades.tremor_level_cm; the truth names
  these rules (params frequency_rule and level_rule) and the caption marks them.
- E3 reach to nose (3.16): not in this set; the symptom catalogue renders it (tremors.py on reach.py, the reach to
  the examiner's finger, scene 'chair_target').
- E4 pronation-supination (3.6), E5 toe tapping (3.7), E6 leg agility (3.8), left side, 10 cycles: level 0, the
  decrement ladder (grades 1 to 3), the interruption ladder (grades 1 to 3) and the arrest (grade 3). E4 is seen
  from the front, from above (hands_top) and in the hands close-up, the views draft section 5 asks for.
- E7 standing (15 s, the arising not rendered: no sit-to-stand motion): the procedural stand (REF_STAND, exam_v10;
  until then an AMASS stand); the dystonic postures on
  the procedural stand (its level 0): TWSTRS neck rotation 1 to 4, laterocollis 1 to 3, anterocollis 1 to 3,
  retrocollis 1 to 3, shoulder elevation 1 to 3, one intermittent rotation (TWSTRS duration score 3); DIS trunk
  lateral flexion, proximal arm abduction and elbow flexion 1 to 4, trunk lateral flexion grade 3 crossed with
  DIS duration 2 and 3; hip abduction 1 and 2 and knee flexion 1 and 2 with the foot planted (contact.py; held on
  the toes where the heel rises, flagged and kept out of DIS grading); the starting postures of the video draft
  (head tilt 20 degrees, trunk lateral flexion 15 degrees, elbow flexion 100 degrees with shoulder abduction 30
  degrees, knee flexion 30 degrees). Ankle plantarflexion (DIS 1 to 4) is shown seated (E1, its level 0 the
  procedural sit, 15 s): standing on a straight leg with the pelvis fixed, a planted plantarflexion needs a finite
  knee bend before any ankle angle, so its ramp would jump (contact.py). The leg grades the planted foot bounds
  below their level are listed with their bound (bounded_ladders), the other omissions in NOT_RENDERED.
- E8 gait and turn: the healthy walk (REF_WALK, the gait generator's 3.10 course; the AMASS walk until exam_v10)
  and freezing of gait 3.11 grades 1 to 4 (trembling, the knees alternating since exam_v7), and grade 2 as akinesia,
  the onsets on the skinned soles (freezing support 'mesh', both soles level and, since exam_v7, both heels down: the
  review found holds on the toes of one foot); in every hold the arms settle from mid-swing to the swing centre
  (freezing arms 'settle'). Since exam_v10 the freezing clips stand on their own walk, FOG_WALK (base.walk's
  plantigrade footstep walk, short flat steps), with its own level 0 FOG_LEVEL0: on the healthy walk a natural double
  support has the trailing heel up, and the least miss of support 'mesh' left a heel 33 mm up and the soles 19 mm
  apart; a freezer's short flat steps are the plantigrade walk's (design D75), and a clip beside its own walk shows
  the freezing sign alone (as gait.py's freezing by shuffling beside its 3.10 grade 1 walk). The clips and the level
  0 take the background layer on the walk's still runs and on the holds in their own build
  (physio.with_hold_background, still).
- E9 turning in place: not rendered (no turning-in-place base body).
Exam_v3 (task SP5; the specs whose since is 'exam_v3', rendered into their own set folder, the index page linking
the exam_v2 clips): finger tapping (3.4, segment E4a) and hand movements (3.5, E4b) with the SMPL+H hand joints
(handtasks.py: level 0 at the cited normal rate and amplitude, decrement and interruption ladders, the arrest, the
slowing ladder anchored to the cited normal rate), seen from the front, in a close-up of the left hand from the
front and in one from the front and the midline (where the pinch opens); E2 on the raised posture (motions.
outstretched: level 0, postural tremor grade 2 from the posture onset, re-emergent tremor counted from it); the
arising of E7 (motions.arise, healthy level 0, a chair with arms under the seated body).
Exam_v4 (review of task SP5, 2026-10-05; the specs whose since is 'exam_v4'): the clips of exam_v3 whose body
changed, rendered again into their own set folder: the arising (balanced: the extrapolated centre of mass inside the
feet from lift-off, motions.py), the hand movements (the fist clear of the palm, handtasks.FIST) and finger tapping
(level 0 at the rate and amplitude of one cohort, grades.BRADY_NORMAL). That review left the three raise clips of
exam_v3 unchanged; the index page of exam_v4 links them and the exam_v2 clips.
Exam_v5 (shared wave of the clinical review, 2026-11-02; the specs whose since is 'exam_v5', BODY_CHANGED_V5): the
exam clips whose body changed, rendered again into their own set folder: tremor_E1_rest_g4 (grade 4 at the
family's top level and frequency, its centre moved clear of the chair's armrest), the E2 ladder
tremor_E2_postural_g1b to g4, the re-emergent tremor and the two raise tremors (the 'postural' drive, read at the
middle fingertip). The other tremor clips keep their body and their set: on the chair of SH5 (viewer.chair_mesh,
the armrests moved outside the hips) the E1 grades 1 to 3 and the constancy clips need no shift, so their bodies
are the version-2 bodies; they gained the D4 reading in their truth and caption, the E2 level-0 clips the
operator text 'left hand'.
Exam_v6 (wave SH1b, 2026-11-03; the specs whose since is 'exam_v6', BODY_CHANGED_V6): the arising ref_arise_E7,
its arms refolded with the forearms apart and the hands on the opposite arms (motions.ARM_POSTURES['fold'], shared
change SH9; the forearms of the earlier fold passed through each other) and its standing hips lowered by the drop
motions.arise solves for the cited plantarflexion; the three shoulder elevations dystonia_E7_shoulder_elevation_
elevation_g1 to g3, whose arm has hung since SH2 (dystonia.py, arm 'hang', common.VERSION v7) and which carried no mark
until wave SH1c; the other exam bodies keep their sets (each exam clip that builds without the AMASS index was rebuilt
against the branch base of the wave). Wave SH1c changed one exam body against the base of that wave, the arising
again (the fold's solve in the tree, motions.solve_fold): it keeps this mark.
Exam_v7 (wave SH1d, 2026-11-03, the clinical review of the English skeleton page; the specs whose since is 'exam_v7',
BODY_CHANGED_V7): the E1 rest and constancy ladders on the 'rest' driver, levelled at the hand (B1, M8);
the freezing clips with the knees trembling alternately and the arms settling in the holds (M3b: in phase, the
knees flexed together and the arms stayed frozen mid-swing); every exam hesitation at least HESITATION_MIN_S long
(bradykinesia.hesitation_length_s, M14: the finger-tapping and hand-movement hesitations were half a period, 0.14
and 0.15 s, too short to count) and every arrest at least bradykinesia.ARREST_MIN_S (the finger-tapping and
hand-movement arrests, two periods, were then under twice a hesitation); the two tremor raise clips with
REST_BEFORE_RAISE_S of rest before the raise, on a new level 0 (tremor_E2_raise_long_rest_g0), the re-emergent one
with its rest tremor stopping with the raise and a build-up of REEMERGENT_RAMP_S after the latency (the review: no
rest tremor before the raise); the trunk lateral flexion clips of dystonia.py, the arms hanging, the head righted and
the pelvis shifted in the same wave. tremor_E2_raise_g0 keeps its body and its set exam_v3: it is
motions.outstretched at its default rest, the level 0 and base of the outstretched clips of jerks.py and postures.py.
Exam_v8 (the settling of waves SH1c and SH1d in main, 2026-11-10; the specs whose since is 'exam_v8',
BODY_CHANGED_V8): the three raise clips of exam_v7 (tremor_E2_raise_long_rest_g0 and its postural and re-emergent
clips) rest REST_BEFORE_RAISE_S, back to the value the review asked once the background layer no longer flips the
anchored right arm in the raise (physio.solve, 'anchor_damped'); the other exam bodies keep their sets.
Exam_v9 (wave SH1d part C, 2026-11-11; BODY_CHANGED_V9): the freezing clips take the standing background of
physio.py on their holds (physio.with_hold_background: breathing, the head drift and the quiet sway, crossfaded over
physio.STILL_FADE_S inside each hold, the walk outside the holds the clip's own bit for bit; the review's M3b: a hold
held still for seconds read as a statue); the other exam bodies keep their sets.
Exam_v10 (open-body port, wave 7, 2026-11-13; BODY_CHANGED_V10): the two healthy references and the freezing clips
built on the walk leave AMASS (owner decision C8): the stand and the walk by the program's generators (module
docstring, healthy references), the freezing clips on the new walk; every other exam body changed with the open body
itself (waves 1 to 6: the rest skeleton, the anatomical joint-limit table), which no mark here records. Plan phase E
(2026-11-14) changes exam bodies inside exam_v10, of which no set was rendered (the rule of the port's D173): the leg
agility ladder (E6, its eight clips) seated with the shoulders turned outward (bradykinesia.arms_out: the raised thigh
ran into the left hand of the 'sit') and the seated plantarflexed ankles of DIS grades 3 and 4 with the left hand
lifted off the rising thigh (dystonia.arm_lift; grades 1 and 2 keep their bodies).
Scenes (scene_mesh): 'chair', 'chair_arise', 'chair_table' and 'chair_table_both' (postures.py), 'chair_target'
(the chair and the examiner's finger at the reach target, reach.scene_mesh), 'chair_armless'
(viewer.chair_mesh(floor_y, armrests=False)) and 'exam_bed' (ataxia.scene_mesh).

Views per family follow draft sections 3 to 6 (viewer.VIEWS): tremor front and the hands close-up (E2 adds the
left side, which the draft names and where the hand's up and down movement is seen, and the view from above);
bradykinesia front and the
hands close-up and the close-up from above (3.6), front and the left-foot close-up (3.7), front and left side
(3.8); freezing the side view following the walk and the examiner's view; dystonic postures front, and above
(rotation), the head close-up (neck), left side (antero- and retrocollis, elbow, knee, ankle).
"""
import math
from dataclasses import dataclass, field, replace

import numpy as np
import torch

from . import grades as G
from .base import static_body
from .body import contact_labels, skate

# the healthy references (module docstring; exam_v10, open-body port wave 7: the program's generators, no AMASS take)
REF_WALK = {"generator": "gait.course", "params": "gait.NORMAL", "seed": "gait.SEED", "distance_m": 10.0,
            "same_body_as": "gait_E8_level0"}
REF_STAND = {"generator": "base.static_body", "posture": "stand", "duration_s": 15.0}
# the freezing clips' walk and its level 0 (module docstring, E8): base.walk's plantigrade footstep walk (design D75),
# 30 s as the AMASS walk was, gait initiation at 1 s, a 180 deg turn, the stop; the turn's length by the leg check
# (FOG_TURN_RULE)
FOG_LEVEL0 = "freezing_E8_level0"
FOG_WALK = {"generator": "base.walk", "kind": "plantigrade", "duration_s": 30.0, "turns": ((12.0, 180.0, 5.0),),
            "stop_at_s": 26.0}
FOG_TURN_RULE = ("the shortest of 1 s steps from 4 s at which the walk's legs keep legcontact's hard tolerance "
                 "(TOL_M) through the turn [module; scripts/openbody_wave7_checks.py fog_walk]")
DUR = {"E1": 45.0, "E2": 30.0, "E7": 15.0}
TREMOR_HZ = {"rest": 5.0, "postural": 6.0}
# the drive of each condition (module docstring, E1 and E2; tremor.py): the rest tremor of the forearm and hand
# (pronation-supination and the wrist, the elbow joining from grade 3), clear of the chair; the postural tremor of the
# outstretched hand at the wrist and shoulder; both read at the hand, the rest tremor at the fingertip that moves
# most, the postural at the middle fingertip (exam_v7; until then the rest tremor was 'elbow_flexion', read at the
# wrist)
TREMOR_DRIVER = {"rest": "rest", "postural": "postural"}
REST_BEFORE_RAISE_S = 4.0  # [module] the raise clips rest this long before the raise (the review: 3 to 5 s of rest
#                            tremor before the re-emergent tremor's raise; 1 s until exam_v7, as tremor_E2_raise_g0
#                            keeps for the clips of jerks.py and postures.py built on it; 4.5 s in exam_v7, where the
#                            background layer's anchored right arm flipped for one knot in the raise at 3 and 4 s:
#                            physio-v6 solves that anchoring with the damped elbow step, physio.solve 'anchor_damped')
REEMERGENT_RAMP_S = 1.5    # [module] the re-emergent tremor builds up over this ramp after its latency (the review:
#                            over 1 to 2 s; tremor.RAMP_S until exam_v7)
SEED = 20261005
GROUND_WINDOW_S = 0.2      # [module] Hann window of the mesh grounding of the AMASS references (ground_by_mesh)
BOUNDED = (("distal_leg", "knee_flexion", (3, 4)), ("proximal_leg", "abduction", (3, 4)))


# ---------------------------------------------------------------- the healthy references (exam_v10)
def reference_walk():
    """(Body, truth) of REF_WALK: the gait generator's healthy course, the catalogue's level 0 (gait.specs()
    gait_E8_level0, built by its own spec so that the two stay one body), its truth with the reference named."""
    from . import gait as GA
    spec = {s.clip_id: s for s in GA.specs()}[GA.LEVEL0]
    b, t = spec.build()
    src = dict(REF_WALK, kind="procedural", seed=int(GA.SEED), duration_s=float(b.duration),
               licence="the program's own generator (no recording)")
    return b, dict(t, params=dict(t["params"], source=src))


def freezing_walk():
    """(Body, info) of FOG_WALK: base.walk's plantigrade walk, whose double support stands on both heels (the holds of
    support 'mesh'; on the gait generator's healthy walk the least miss leaves a heel 33 mm up and the soles 19 mm
    apart, design section 18)."""
    from .base import walk
    w = FOG_WALK
    b, info = walk(w["duration_s"], turns=[tuple(t) for t in w["turns"]], stop_at_s=w["stop_at_s"], kind=w["kind"])
    src = dict(w, turns=[list(t) for t in w["turns"]], kind_of_source="procedural", turn_rule=FOG_TURN_RULE,
               licence="the program's own generator (no recording)")
    return b, src


def reference_stand():
    """(Body, truth) of REF_STAND: the procedural stand for its duration, a level 0 truth naming it."""
    from .common import truth_record
    b = static_body(REF_STAND["posture"], REF_STAND["duration_s"])
    src = dict(REF_STAND, kind="procedural", licence="the program's own generator (no recording)")
    return b, truth_record("none", None, 0, "healthy reference", {"source": src}, (), None)


# ---------------------------------------------------------------- the former AMASS references (until exam_v10)
# amass_take, select_stand and retarget_neutral, which built the AMASS references of exam_v2 to exam_v9, live in
# smplh_reference.py since the open-body port's wave 8, behind the reference flag; that module never ships.


def hann_smooth(x, n):
    """x (T,) smoothed by a centred Hann window of n samples (edges padded with the end values); n <= 1: a copy."""
    x = np.asarray(x, dtype=np.float64)
    if n <= 1:
        return x.copy()
    w = np.hanning(n + 2)[1:-1]
    w = w / w.sum()
    return np.convolve(np.pad(x, (n // 2, n - 1 - n // 2), mode="edge"), w, mode="valid")


def ground_by_mesh(body, window_s=GROUND_WINDOW_S):
    """(Body, info): a reference re-floored by its own mesh (module docstring): the lowest skinned sole point of
    the two feet at each knot (contact.mesh_sole_low), smoothed by a centred Hann window of window_s, brought to
    the height of a flat neutral sole on the joint floor (viewer.mesh_floor_y(0)) by a vertical shift of the
    translation; rotations and the horizontal path kept, contacts relabelled. info: the window, the shift's range,
    the residual of the knots' lowest sole point against that height before and after (min, 2nd percentile,
    median, 98th percentile, max, share within contact.TOL_M), the 99th percentile of the pelvis's vertical speed
    and the 95th percentile foot skate in stance, before and after."""
    from . import contact as CT
    from .viewer import mesh_floor_y
    target = mesh_floor_y(0.0)
    low = CT.mesh_sole_low(body).min(1)
    shift = hann_smooth(low, int(round(window_s * body.rate))) - target
    tr = body.transl.clone()
    tr[:, 1] -= torch.from_numpy(shift)
    out = replace(body, transl=tr)
    out = replace(out, contact=contact_labels(out))
    low2 = CT.mesh_sole_low(out).min(1)

    def resid(x):
        r = x - target
        return {"min_m": float(r.min()), "p2_m": float(np.percentile(r, 2)), "median_m": float(np.median(r)),
                "p98_m": float(np.percentile(r, 98)), "max_m": float(r.max()),
                "within_tol_share": float((np.abs(r) <= CT.TOL_M).mean())}

    def vy99(b):
        p = b.joints()[:, 0, 1].numpy()
        return float(np.percentile(np.abs(np.gradient(p) * b.rate), 99))
    info = {"rule": "lowest skinned sole point per knot, Hann-smoothed, to a flat neutral sole on the joint floor",
            "window_s": float(window_s), "target_y_m": float(target), "shift_min_m": float(shift.min()),
            "shift_max_m": float(shift.max()), "residual_before": resid(low), "residual_after": resid(low2),
            "tol_m": CT.TOL_M, "pelvis_vy_p99_before_m_s": vy99(body), "pelvis_vy_p99_after_m_s": vy99(out),
            "skate_p95_before_m_s": skate(body.joints(), body.contact, body.rate)["p95"],
            "skate_p95_after_m_s": skate(out.joints(), out.contact, out.rate)["p95"]}
    return out, info


# ---------------------------------------------------------------- the clip specification
@dataclass
class ClipSpec:
    """One clip of the exam set: id, family, segment (E1 to E9), item (scale item), operator (caption), grade and
    its caption text, the parameters' caption text, views, the clip id of its level 0 (None for a level 0), the
    scene (a kind of scene_mesh, or None), and build() -> (Body, truth dict)."""
    clip_id: str
    family: str
    segment: str
    item: str
    operator: str
    grade: object
    grade_text: str
    params_text: str
    views: tuple
    level0: object
    scene: object
    build: object = field(repr=False)
    note: str = ""
    since: str = "exam_v2"
    segment_title: str = ""      # a segment outside SEGMENTS names itself here (the symptom catalogue's tasks)

    def caption_spec(self):
        return {"operator": self.operator, "grade_text": self.grade_text, "params_text": self.params_text,
                "segment_text": f"{self.segment} {segment_title(self)}", "clip_id": self.clip_id}


def segment_title(spec):
    """The title of a clip's exam segment: its own segment_title when set, else SEGMENTS[segment]."""
    return spec.segment_title or SEGMENTS[spec.segment]


SEGMENTS = {"E1": "seated rest", "E2": "arms outstretched", "E3": "reach to nose", "E4": "pronation-supination",
            "E5": "toe tapping", "E6": "leg agility", "E7": "standing", "E8": "gait and turn",
            "E9": "turning in place", "E4a": "finger tapping", "E4b": "hand movements"}
NOT_RENDERED = {
    "E3": ("not in this set: the symptom catalogue renders the reach to the examiner's finger (tremors.py on "
           "reach.py, 3.16 and the cerebellar signs, scene 'chair_target')"),
    "E9": "no turning-in-place body exists in mdfm/synth",
    "3.9 grades": "the arising is rendered as the healthy level 0 only: no operator acts on it yet",
    "3.4 and 3.5 grade 4": "the task can barely be performed: no ladder level (draft section 5)",
    "slowing of 3.6 to 3.8": ("the slowing ladder is anchored to a cited normal rate for 3.4 and 3.5 only "
                              "(grades.BRADY_NORMAL); 3.6 to 3.8 keep their version-2 module rates, so that the SP3 "
                              "probe stimuli do not change"),
    "3.15 ladder on the raise": ("the 3.15 grade ladder stays on the static posture (E2 clips of exam_v2); the raised "
                                 "posture carries its level 0, grade 2 and the re-emergent tremor"),
    "DIS duration 1": (f"its band centre ({100 * G.dis_share(1):g} percent of the {DUR['E7']:g} s segment) is shorter "
                       f"than the posture's ramp, so common.schedule refuses the bout"),
    "DIS duration crossing": ("one crossing is rendered (trunk lateral flexion grade 3 at DIS duration 2, 3 and 4); "
                              "every other posture is held for the whole segment (DIS duration 4)"),
    "DIS in action": ("every posture is held at rest, standing or seated; no task runs under it (task dependence is "
                      "an operator schedule this set does not use)"),
    "DIS other directions": ("one direction per region: trunk lateral flexion (not flexion, extension or rotation), "
                             "proximal arm abduction (not adduction), distal arm elbow flexion (not extension, not the "
                             "wrist), proximal leg abduction (not flexion, extension or adduction), distal leg knee "
                             "flexion and ankle plantarflexion (not dorsiflexion)"),
    "DIS neck": "the neck is graded on TWSTRS in this set",
    "TWSTRS lateral and sagittal shift": "not modelled: the posture operator applies rotations only",
    "standing ankle postures": ("shown seated: on a standing leg with the pelvis fixed, a planted plantarflexion that "
                                "ramps jumps at its first planted knot (its continuity check fails)"),
    "knee and hip postures standing on a flat foot": ("with the pelvis fixed the planted foot compensates by ankle "
                                                      "flexion and the heel rises; such clips are rendered, flagged "
                                                      "'toe stance' and kept out of DIS grading until the neurologist "
                                                      "decides how they are graded")}


def bounded_ladders():
    """{label: reason} of the standing leg grades not rendered because the planted foot bounds them below their
    level (BOUNDED): each built on the procedural stand, its asked and held levels read from its truth."""
    from .dystonia import dystonic_posture
    out = {}
    for region, direction, grades in BOUNDED:
        for g in grades:
            _, t = dystonic_posture(_static("stand", 1.0), region, direction, "left", amplitude_grade=g,
                                    seed=SEED + g)
            p = t["params"]
            out[f"DIS {region.replace('_', ' ')} {direction.replace('_', ' ')} grade {g}"] = (
                f"standing with the pelvis fixed and the foot planted it holds {p['realized_peak_deg']:.1f} deg of "
                f"the {p['level_deg']:.1f} deg asked (planted-foot scale {p['plant_scale_min']:.3f}), so the clip "
                f"would show a lower grade's posture")
    return out


def _static(posture, seconds):
    return static_body(posture, seconds)


def _truth_level0(base, kind, extra=None):
    from .common import truth_record
    return truth_record("none", None, 0, "level 0", dict({"base": kind}, **(extra or {})), (), None)


# ---------------------------------------------------------------- families
def tremor_level(grade, grade1="above_floor"):
    """The level (cm peak to peak) of an exam tremor grade: the tremor family's MDS-UPDRS level (tremors.level_cm:
    grades.tremor_level_cm, the open top band at tremors.TOP_LEVEL_CM)."""
    from .tremors import level_cm
    return level_cm("MDS-UPDRS", grade, grade1)


def tremor_hz(cond, grade):
    """Hz of an exam tremor: TREMOR_HZ[cond], slower at the top grades by the tremor family's rule (tremors.limb_hz,
    module values)."""
    from .tremors import limb_hz
    return limb_hz("MDS-UPDRS", grade, TREMOR_HZ[cond])


def tremor_rules(grade):
    """{level_rule, frequency_rule} of an exam tremor grade whose level or frequency is a module value of the tremor
    family (grade 4: tremors.TOP_LEVEL_CM; grades 3 and 4: tremors.limb_hz, TOP_HZ), for the truth and the caption;
    {} below."""
    out = {}
    if grade is not None and grade >= 3:
        out["frequency_rule"] = "tremors.limb_hz, TOP_HZ (module: slower at the top grades)"
    if grade == 4:
        out["level_rule"] = "tremors.TOP_LEVEL_CM (module: the open top band, just above its lower edge)"
    return out


TIP_WORDS = {"thumb": "the thumb tip", "index": "the index fingertip", "middle": "the middle fingertip",
             "ring": "the ring fingertip", "pinky": "the little fingertip"}   # caption words of tremor.FINGERTIPS


def _tremor_caption(p):
    """The caption's parameter text of an exam tremor from its truth params (or the same keys before the build): the
    frequency and the level at the hand, where every exam hand tremor is levelled since exam_v7 (the fingertip that
    moves most for the rest driver, named once built, the middle fingertip for the postural; the wrist for the elbow
    driver, which no exam clip uses since then, with the fingertip's reading once built), the
    module marks of the top grades (the rules themselves are in the truth: frequency_rule, level_rule), the drive in
    clinical words and the centre shift (the rest driver's elbow lifting the forearm off the armrest; the elbow
    driver's shift). Short enough for the caption band of viewer.Annotator (one line, no ellipsis)."""
    from .tremor import FINGERTIPS, postural_text, rest_text
    fr = " (module)" if p.get("frequency_rule") else ""
    lv = " (module level)" if p.get("level_rule") else ""
    rd = p.get("readings") or {}
    if p.get("point") == "middle_tip":
        t = f"{p['frequency_hz']:g} Hz{fr}, {p['cm_pp']:.2f} cm peak to peak at the middle fingertip{lv}"
    elif p.get("point") == FINGERTIPS:
        where = TIP_WORDS.get(p.get("level_tip"), "the fingertip that moves most")
        t = f"{p['frequency_hz']:g} Hz{fr}, {p['cm_pp']:.2f} cm peak to peak at {where}{lv}"
    else:
        t = f"{p['frequency_hz']:g} Hz{fr}, {p['cm_pp']:.2f} cm peak to peak at the wrist{lv}"
        if "middle_tip" in rd:
            t += f", middle fingertip {rd['middle_tip']['pp_cm_max']:.2f} cm"
    if p.get("driver") == "postural":
        t += f", {postural_text(p['cm_pp'])}"
    if p.get("driver") == "rest":
        t += f", {rest_text(p['cm_pp'])}"
        if p.get("elbow_lift_deg"):
            t += f", the elbow flexed {p['elbow_lift_deg']:.1f} deg off the armrest"
    elif p.get("centre_shift_deg") and p.get("shift_toward"):
        sc = p.get("scene_clearance") or {}
        why = "clear of the chair" if sc.get("added_shift_deg") else "joint-limit table"
        t += f", centred {abs(p['centre_shift_deg']):.1f} deg toward {p['shift_toward']} ({why})"
    return t


def _tremor_text(cond, cm, hz, g=None):
    """The caption's parameter text of an exam tremor before its build (_tremor_caption on the spec's values)."""
    from .tremor import FINGERTIPS
    p = dict({"frequency_hz": hz, "cm_pp": cm, "driver": TREMOR_DRIVER[cond]}, **tremor_rules(g))
    if TREMOR_DRIVER[cond] in ("postural", "rest"):
        p["point"] = "middle_tip" if TREMOR_DRIVER[cond] == "postural" else FINGERTIPS
    return _tremor_caption(p)


def _with_readings(out, truth, base, grade=None):
    """(out, truth) with both reading points of the clip in its params (tremor.point_readings: the wrist and the
    skinned middle fingertip and the fingertip that moves most, against base, the clip without its tremor) and the
    module rules of its grade
    (tremor_rules); a level 0 is returned as it is."""
    from .tremor import point_readings
    p = dict(truth["params"])
    if not p.get("cm_pp"):
        return out, truth
    p.update(point_readings(out, base, p["side"], truth["arrays"]["envelope"]), **tremor_rules(grade))
    return out, dict(truth, params=p)


def _tremor_specs():
    from .tremor import chair_boxes, tremor
    out = []
    ladder = [(0, None, "0"), (1, "below_floor", "1b"), (1, "above_floor", "1a"), (2, None, "2"), (3, None, "3"),
              (4, None, "4")]
    for seg, posture, cond, item, views in (("E1", "sit", "rest", "3.17", ("front", "hands")),
                                            ("E2", "sit_arms_forward", "postural", "3.15",
                                             ("front", "left", "top", "hands"))):
        ttype = "parkinsonian_rest" if cond == "rest" else "essential"
        l0 = f"tremor_{seg}_{cond}_g0"
        where = "left forearm and hand" if cond == "rest" else "left hand"
        for g, g1, tag in ladder:
            cm = tremor_level(g, g1 or "above_floor")
            hz = tremor_hz(cond, g)

            def build(g=g, g1=g1, posture=posture, cond=cond, hz=hz, ttype=ttype, seg=seg, cm=cm):
                base = _static(posture, DUR[seg])
                o, t = tremor(base, "left", grade=g, cm_pp=cm, frequency_hz=hz, condition=cond, tremor_type=ttype,
                              driver=TREMOR_DRIVER[cond], seed=SEED + g,
                              clear=chair_boxes() if cond == "rest" else None)
                o, t = _with_readings(o, t, base, g)
                if g1 is not None:             # the review of the lane: the floor read on the body's own clock
                    from .viewer import FPS as VIDEO_FPS          # the page's frame rate
                    low = 100 * (1 - math.cos(math.pi * hz / VIDEO_FPS))
                    t = dict(t, params=dict(t["params"], floor_rule=(
                        f"the level lies {g1.split('_')[0]} the estimator floor as read on the body's "
                        f"{base.rate:g} Hz clock; sampled at {VIDEO_FPS:g} frames per second, a {hz:g} Hz chord can "
                        f"read up to {low:.1f} percent low (the extremes missed by half a frame), so a check at that "
                        f"rate reads the floor with this tolerance")))
                return o, t
            lab = {"1b": "grade 1 (below the estimator floor)", "1a": "grade 1 (above the estimator floor)"}
            cid = f"tremor_{seg}_{cond}_g{tag}"
            out.append(ClipSpec(cid, "tremor", seg, item, f"tremor, {cond}, {where}", g,
                                f"MDS-UPDRS {item} {lab.get(tag, f'grade {g}')}",
                                _tremor_text(cond, cm, hz, g) if g else "no tremor (level 0)",
                                views, None if g == 0 else l0, "chair", build, since=_since(cid)))
    for cg in (1, 2, 3, 4):
        def build(cg=cg):
            base = _static("sit", DUR["E1"])
            o, t = tremor(base, "left", grade=2, frequency_hz=tremor_hz("rest", 2), condition="rest",
                          constancy_grade=cg, seed=SEED + 10 + cg, driver=TREMOR_DRIVER["rest"], clear=chair_boxes())
            return _with_readings(o, t, base, 2)
        cid = f"tremor_E1_constancy_g{cg}"
        out.append(ClipSpec(cid, "tremor", "E1", "3.18", "tremor, rest, left forearm and hand", cg,
                            f"MDS-UPDRS 3.18 grade {cg} (3.17 grade 2)",
                            f"{100 * G.constancy_share(cg):.1f} percent of the clip, {tremor_hz('rest', 2):g} Hz, "
                            f"{G.tremor_level_cm(2):.2f} cm peak to peak at the fingertip that moves most", ("hands",),
                            "tremor_E1_rest_g0", "chair", build, since=_since(cid)))

    def build_re():
        base = _static("sit_arms_forward", DUR["E2"])
        o, t = tremor(base, "left", grade=2, frequency_hz=tremor_hz("rest", 2), condition="postural",
                      tremor_type="parkinsonian_rest", latency_s=G.REEMERGENCE_MEAN_S,
                      driver=TREMOR_DRIVER["postural"], seed=SEED + 20)
        return _with_readings(o, t, base, 2)
    out.append(ClipSpec("tremor_E2_reemergent_g2", "tremor", "E2", "3.15", "tremor, re-emergent, left hand", 2,
                        "MDS-UPDRS 3.15 grade 2 (re-emergent)",
                        f"{_tremor_text('postural', tremor_level(2), tremor_hz('rest', 2))}, after "
                        f"{G.REEMERGENCE_MEAN_S:g} s", ("front", "hands"), "tremor_E2_postural_g0", "chair",
                        build_re, since=_since("tremor_E2_reemergent_g2")))
    return out


def _brady_specs():
    from .bradykinesia import ARREST_MIN_S, HESITATION_MIN_S, LEG_AGILITY_ARMS_OUT_DEG, TASKS, arms_out, bradykinesia
    out = []
    views = {"pronation_supination": ("front", "hands_top", "hands"), "toe_tapping": ("front", "foot_left"),
             "leg_agility": ("front", "left")}
    seg = {"pronation_supination": "E4", "toe_tapping": "E5", "leg_agility": "E6"}
    short = {"pronation_supination": "ps", "toe_tapping": "tt", "leg_agility": "la"}
    plan = [("level0", None, 0), ("decrement", 1, 1), ("decrement", 2, 2), ("decrement", 3, 3),
            ("interruptions", 1, 1), ("interruptions", 2, 2), ("interruptions", 3, 3), ("arrest", 3, 3)]
    for task, tk in TASKS.items():
        l0 = f"brady_{seg[task]}_{short[task]}_level0"
        for crit, g, gn in plan:
            def build(task=task, crit=crit, g=g, tk=tk):
                base = _static(tk["posture"], _brady_duration(task, crit, g))
                if task == "leg_agility":          # exam_v10 (plan phase E): the hands beside the raised thigh
                    base = arms_out(base)          # (bradykinesia.py, Arms of the leg agility)
                if crit == "level0":
                    o, t = bradykinesia(base, task, "left", decrement_onset=None, seed=SEED)
                else:
                    o, t = bradykinesia(base, task, "left", grade=g, criterion=crit, seed=SEED + g,
                                        hesitation_floor_s=HESITATION_MIN_S, arrest_floor_s=ARREST_MIN_S)
                if task == "leg_agility":
                    t = dict(t, params=dict(t["params"], arms_out_deg=LEG_AGILITY_ARMS_OUT_DEG))
                return o, t
            if crit == "level0":
                gt, pt = "level 0", f"10 cycles at {tk['level0_hz']:g} Hz, no criterion"
            elif crit == "decrement":
                gt = f"MDS-UPDRS {tk['item']} grade {g} (decrement)"
                pt = f"decrement from cycle {G.BRADY_DECREMENT_ONSET[g]}, {100 * G.BRADY_DECREMENT_SLOPE:g} percent per cycle"
            elif crit == "interruptions":
                gt = f"MDS-UPDRS {tk['item']} grade {g} (interruptions)"
                pt = f"{G.BRADY_INTERRUPTIONS[g]} hesitations"
            else:
                gt, pt = f"MDS-UPDRS {tk['item']} grade 3 (arrest)", "one arrest"
            cid = f"brady_{seg[task]}_{short[task]}_{'level0' if crit == 'level0' else f'{crit}_g{g}'}"
            out.append(ClipSpec(cid, "bradykinesia", seg[task], tk["item"],
                                f"bradykinesia, {task.replace('_', '-')}, left", gn, gt, pt, views[task],
                                None if crit == "level0" else l0, "chair", build, since=_since(cid)))
    return out


def _brady_duration(task, crit, g):
    """Seconds of the base body: 0.5 s lead, the 10 cycles, the pauses (each hesitation at least
    bradykinesia.HESITATION_MIN_S and each arrest at least bradykinesia.ARREST_MIN_S since exam_v7), 1.5 s after."""
    from .bradykinesia import HESITATION_MIN_S, TASKS, arrest_length_s, hesitation_length_s
    P = 2 * int(round(100.0 / TASKS[task]["level0_hz"] / 2)) / 100.0
    pauses = 0.0
    if crit == "interruptions":
        pauses = G.BRADY_INTERRUPTIONS[g] * hesitation_length_s(P, HESITATION_MIN_S)
    elif crit == "arrest":
        pauses = arrest_length_s(P)
    return round(0.5 + G.N_CYCLES * P + pauses + 1.5, 2)


_DYST_WORDS = {"rotation_left": "neck turned to the left", "tilt_left": "neck tilted to the left",
               "anterocollis": "anterocollis (the neck flexed forward)",
               "retrocollis": "retrocollis (the neck extended backward)",
               "elevation": "left shoulder elevation", "lateral_left": "trunk flexed laterally to the left",
               "abduction": "abduction", "elbow_flexion": "left elbow flexion", "knee_flexion": "left knee flexion",
               "plantarflexion_ankle": "left ankle plantarflexion"}


def dystonia_words(region, direction):
    """The clinical words of an exam dystonic posture's region and direction (the left side): 'neck turned to the
    left', 'left shoulder elevation', 'left arm abduction at the shoulder', 'left hip abduction', and so on (the
    review of the English page: the region and direction words of version 2 repeated or lacked the side)."""
    if direction == "abduction":
        return {"proximal_arm": "left arm abduction at the shoulder", "proximal_leg": "left hip abduction"}[region]
    return _DYST_WORDS[direction]


def dystonia_operator(region, direction):
    """The operator text of an exam dystonic posture's caption: 'dystonic posture, ' and dystonia_words."""
    return f"dystonic posture, {dystonia_words(region, direction)}"


def _dystonia_part(params, fallback):
    """The words of one part of a composed posture, from its truth params (dystonia_words), else fallback."""
    try:
        return dystonia_words(params.get("region"), params.get("direction"))
    except KeyError:
        return fallback


def _dystonia_specs():
    from .dystonia import dystonic_posture
    out = []
    l0 = "dystonia_E7_level0"

    def lvl0():
        b = _static("stand", DUR["E7"])
        return b, _truth_level0(b, "procedural stand")
    out.append(ClipSpec(l0, "dystonia", "E7", "level 0", "none (procedural stand)", 0, "level 0",
                        "the shared neutral body standing", ("front", "top", "left", "head"), None, None, lvl0))
    plan = [("neck", "rotation_left", g, ("front", "top", "head"), "TWSTRS rotation") for g in (1, 2, 3, 4)]
    plan += [("neck", "tilt_left", g, ("front", "head"), "TWSTRS laterocollis") for g in (1, 2, 3)]
    plan += [("neck", "anterocollis", g, ("left", "head"), "TWSTRS anterocollis") for g in (1, 2, 3)]
    plan += [("neck", "retrocollis", g, ("left", "head"), "TWSTRS retrocollis") for g in (1, 2, 3)]
    plan += [("shoulder_elevation", "elevation", g, ("front",), "TWSTRS shoulder elevation") for g in (1, 2, 3)]
    plan += [("trunk", "lateral_left", g, ("front", "back"), "DIS trunk") for g in (1, 2, 3, 4)]
    plan += [("proximal_arm", "abduction", g, ("front",), "DIS proximal arm") for g in (1, 2, 3, 4)]
    plan += [("distal_arm", "elbow_flexion", g, ("front", "left"), "DIS distal arm") for g in (1, 2, 3, 4)]
    plan += [("proximal_leg", "abduction", g, ("front", "left"), "DIS proximal leg") for g in (1, 2)]
    plan += [("distal_leg", "knee_flexion", g, ("left", "front"), "DIS distal leg") for g in (1, 2)]
    for region, direction, g, views, scale in plan:
        def build(region=region, direction=direction, g=g):
            return dystonic_posture(_static("stand", DUR["E7"]), region, direction, "left", amplitude_grade=g,
                                    seed=SEED + g)
        cid = f"dystonia_E7_{region}_{direction}_g{g}"
        out.append(ClipSpec(cid, "dystonia", "E7", scale, dystonia_operator(region, direction), g,
                            f"{scale} grade {g}", "", views, l0, None, build, since=_since(cid)))
    for dg in (2, 3):                                     # one DIS duration crossing (NOT_RENDERED)
        def build_dur(dg=dg):
            return dystonic_posture(_static("stand", DUR["E7"]), "trunk", "lateral_left", "left", amplitude_grade=3,
                                    duration_grade=dg, seed=SEED + 40 + dg)
        cid = f"dystonia_E7_trunk_lateral_left_g3_duration{dg}"
        out.append(ClipSpec(cid, "dystonia", "E7", "DIS trunk", dystonia_operator("trunk", "lateral_left"), 3,
                            f"DIS trunk grade 3, DIS duration {dg}", "", ("front", "back"), l0, None, build_dur,
                            since=_since(cid)))

    # ankle postures seated (E1): standing on a straight leg with the pelvis fixed, a planted plantarflexion needs
    # a finite knee bend before any ankle angle (contact.py), so its ramp would jump; seated, it starts from zero
    l0s = "dystonia_E1_level0"

    def lvl0_sit():
        b = _static("sit", DUR["E7"])
        return b, _truth_level0(b, "procedural sit")
    out.append(ClipSpec(l0s, "dystonia", "E1", "level 0", "none (procedural sit)", 0, "level 0",
                        "the shared neutral body seated", ("left", "foot_left"), None, "chair", lvl0_sit))
    for g in (1, 2, 3, 4):
        def build_pf(g=g):
            return dystonic_posture(_static("sit", DUR["E7"]), "distal_leg", "plantarflexion_ankle", "left",
                                    amplitude_grade=g, seed=SEED + g)
        cid = f"dystonia_E1_distal_leg_plantarflexion_ankle_g{g}"
        out.append(ClipSpec(cid, "dystonia", "E1", "DIS distal leg",
                            dystonia_operator("distal_leg", "plantarflexion_ankle"), g, f"DIS distal leg grade {g}",
                            "", ("left", "foot_left"), l0s, "chair", build_pf, since=_since(cid)))

    def build_int():
        return dystonic_posture(_static("stand", DUR["E7"]), "neck", "rotation_left", "left", amplitude_grade=2,
                                duration_grade=3, seed=SEED + 30)
    out.append(ClipSpec("dystonia_E7_neck_rotation_left_g2_duration3", "dystonia", "E7", "TWSTRS rotation",
                        dystonia_operator("neck", "rotation_left"), 2, "TWSTRS rotation grade 2, duration score 3",
                        "", ("front", "top"), l0, None, build_int,
                        since=_since("dystonia_E7_neck_rotation_left_g2_duration3")))
    # the starting postures of the video draft (levels in degrees, not grades; the caption says a held posture)
    starts = [("neck", "tilt_left", 20.0, ("front",)), ("trunk", "lateral_left", 15.0, ("front",)),
              ("distal_leg", "knee_flexion", 30.0, ("left", "front"))]
    for region, direction, deg, views in starts:
        def build(region=region, direction=direction, deg=deg):
            return dystonic_posture(_static("stand", DUR["E7"]), region, direction, "left", level=deg, seed=SEED)
        cid = f"dystonia_E7_start_{region}_{direction}_{int(deg)}deg"
        out.append(ClipSpec(cid, "dystonia", "E7", "level", dystonia_operator(region, direction), "level",
                            f"a held posture of {deg:g} degrees", "", views, l0, None, build, since=_since(cid)))

    def build_arm():
        b, t1 = dystonic_posture(_static("stand", DUR["E7"]), "proximal_arm", "abduction", "left", level=30.0,
                                 seed=SEED)
        b2, t2 = dystonic_posture(b, "distal_arm", "elbow_flexion", "left", level=100.0, seed=SEED)
        t2 = dict(t2, composed_with=t1)
        return b2, t2
    out.append(ClipSpec("dystonia_E7_start_arm_abduction30_elbow100", "dystonia", "E7", "level",
                        "dystonic posture, left arm abduction at the shoulder with the elbow flexed", "level",
                        "a held posture: the shoulder abducted 30 degrees, the elbow flexed 100 degrees", "",
                        ("front", "left"), l0, None, build_arm,
                        since=_since("dystonia_E7_start_arm_abduction30_elbow100")))
    return out


def _freezing_specs(walk_loader, fog_walk_loader):
    """ref_walk_E8 (the healthy walk, REF_WALK), then the freezing clips on their own walk (FOG_WALK) with its level
    0 FOG_LEVEL0 (module docstring, E8; exam_v10)."""
    from .freezing import freezing_grade
    from .physio import series_seed, with_hold_background
    out = []

    def ref():
        b, t = walk_loader()
        return b, dict(t)
    out.append(ClipSpec("ref_walk_E8", "reference", "E8", "3.10 reference", "healthy reference walk (gait generator)",
                        0, "level 0 (healthy reference)", "10 m, turn 180 deg, 10 m back", ("track_left", "examiner"),
                        None, None, ref, since=_since("ref_walk_E8")))
    seed = series_seed(FOG_LEVEL0)

    def lvl0():
        b, info = fog_walk_loader()
        from .common import truth_record
        t = truth_record("none", None, 0, "level 0", {"base": "plantigrade walk", "source": info}, (), None)
        return with_hold_background(b, t, seed=seed, still=True)
    out.append(ClipSpec(FOG_LEVEL0, "freezing", "E8", "3.11", "none (the freezing clips' walk)", 0, "level 0",
                        "short flat steps, a 180 deg turn, the stop", ("track_left", "examiner"), None, None, lvl0,
                        since=_since(FOG_LEVEL0)))
    for g in (1, 2, 3, 4):                 # exam_v7: the knees alternating, the arms settling (module docstring);
        def build(g=g):                    # exam_v9: the standing background on the holds; exam_v10: and on the
            b, _ = fog_walk_loader()       # walk's still runs, the walk procedural
            return with_hold_background(*freezing_grade(b, g, seed=SEED + g, manifestation="trembling",
                                                        pattern="alternating", support="mesh", arms="settle"),
                                        seed=seed, still=True)
        cid = f"freezing_E8_g{g}_trembling"
        out.append(ClipSpec(cid, "freezing", "E8", "3.11", "freezing of gait, trembling", g,
                            f"MDS-UPDRS 3.11 grade {g}",
                            f"the knees trembling alternately at {G.TREMBLE_DEFAULT_HZ:g} Hz, the arms settling at "
                            f"the sides", ("track_left", "examiner"), FOG_LEVEL0, None, build, since=_since(cid)))

    def build_ak():
        b, _ = fog_walk_loader()
        return with_hold_background(*freezing_grade(b, 2, seed=SEED + 2, manifestation="akinesia", support="mesh",
                                                    arms="settle"), seed=seed, still=True)
    out.append(ClipSpec("freezing_E8_g2_akinesia", "freezing", "E8", "3.11", "freezing of gait, akinesia", 2,
                        "MDS-UPDRS 3.11 grade 2",
                        "akinesia (no stepping during the episodes, the arms settling at the sides)",
                        ("track_left", "examiner"), FOG_LEVEL0, None, build_ak,
                        since=_since("freezing_E8_g2_akinesia")))
    return out


def _reference_specs(stand_loader):
    def ref():
        b, t = stand_loader()
        return b, dict(t)
    return [ClipSpec("ref_stand_E7", "reference", "E7", "reference", "healthy reference stand (procedural)", 0,
                     "level 0 (healthy reference)", "quiet upright stance, arms down", ("front", "left"), None, None,
                     ref, since=_since("ref_stand_E7"))]


# ---------------------------------------------------------------- the exam motions of task SP5 (exam_v3)
V3 = "exam_v3"
V4 = "exam_v4"            # review of SP5: the exam_v3 clips whose body changed (module docstring)
V5 = "exam_v5"            # shared wave of the clinical review (2026-11-02): the exam clips whose body changed
# (module docstring; tests/test_synth_tremors.py rebuilds the version-2 body of every E1 tremor clip and checks that
# this list holds exactly the ones that differ: an E1 clip changes only where the chair's armrest needs a shift)
BODY_CHANGED_V5 = ("tremor_E1_rest_g4", "tremor_E2_postural_g1b", "tremor_E2_postural_g1a", "tremor_E2_postural_g2",
                   "tremor_E2_postural_g3", "tremor_E2_postural_g4", "tremor_E2_reemergent_g2",
                   "tremor_E2_raise_postural_g2", "tremor_E2_raise_reemergent_g2")
V6 = "exam_v6"            # wave SH1b (2026-11-03): the exam clips whose body changed (module docstring)
BODY_CHANGED_V6 = ("ref_arise_E7", "dystonia_E7_shoulder_elevation_elevation_g1",
                   "dystonia_E7_shoulder_elevation_elevation_g2", "dystonia_E7_shoulder_elevation_elevation_g3")
V7 = "exam_v7"            # wave SH1d (2026-11-03): the exam clips whose body changed (module docstring); the E1 rest
# and constancy bodies, the tremor raise clips and their new level 0, the freezing clips and the interruption ladders
# of this module's lane, and the trunk lateral flexion clips of dystonia.py (the same wave: the arms hanging, the head
# righted, the pelvis shifted). Every exam dystonia, bradykinesia and E2 clip was rebuilt on the wave and on its branch
# base (b231e28):
# among them these are exactly the ones whose body differs (the shoulder elevation clips, which hung their arms
# before the wave, do not)
BODY_CHANGED_V7 = tuple([f"tremor_E1_rest_g{t}" for t in ("1b", "1a", "2", "3", "4")]
                        + [f"tremor_E1_constancy_g{g}" for g in (1, 2, 3, 4)]
                        + ["tremor_E2_raise_long_rest_g0", "tremor_E2_raise_postural_g2",
                           "tremor_E2_raise_reemergent_g2"]
                        + [f"freezing_E8_g{g}_trembling" for g in (1, 2, 3, 4)] + ["freezing_E8_g2_akinesia"]
                        + [f"brady_{seg}_interruptions_g{g}" for seg in ("E4_ps", "E5_tt", "E4a_ft", "E4b_hm")
                           for g in (1, 2, 3)]
                        + ["brady_E4a_ft_arrest_g3", "brady_E4b_hm_arrest_g3"]
                        + [f"dystonia_E7_trunk_lateral_left_g{g}" for g in (1, 2, 3, 4)]
                        + [f"dystonia_E7_trunk_lateral_left_g3_duration{d}" for d in (2, 3)]
                        + ["dystonia_E7_start_trunk_lateral_left_15deg"])


V8 = "exam_v8"            # the settling of waves SH1c and SH1d (2026-11-10): the exam clips whose body changed
# (module docstring): the raise clips at REST_BEFORE_RAISE_S; every other exam body is unchanged (the background
# layer of physio.py does not wrap the exam set, and no exam clip is built by involuntary.py)
BODY_CHANGED_V8 = ("tremor_E2_raise_long_rest_g0", "tremor_E2_raise_postural_g2", "tremor_E2_raise_reemergent_g2")
V9 = "exam_v9"            # wave SH1d part C (2026-11-11): the exam clips whose body changed (module docstring): the
# freezing clips, whose holds take the standing background of physio.py (physio.with_hold_background); every other
# exam body is unchanged (the exam set is not wrapped by the layer)
BODY_CHANGED_V9 = tuple(f"freezing_E8_g{g}_trembling" for g in (1, 2, 3, 4)) + ("freezing_E8_g2_akinesia",)
V10 = "exam_v10"          # open-body port, wave 7 (2026-11-13): the references and the freezing clips leave AMASS
# (module docstring); the freezing clips' level 0 is new (their own walk, FOG_LEVEL0); plan phase E (2026-11-14), no
# set of exam_v10 rendered yet: the leg agility ladder seated on bradykinesia.arms_out and the plantarflexed ankles of
# DIS grades 3 and 4 with their hand lifted (dystonia.arm_lift), the seated hands kept off the open body's thighs
BODY_CHANGED_V10 = (("ref_stand_E7", "ref_walk_E8", FOG_LEVEL0) + BODY_CHANGED_V9
                    + ("brady_E6_la_level0",) + tuple(f"brady_E6_la_{c}_g{g}" for c in ("decrement", "interruptions")
                                                      for g in (1, 2, 3))
                    + ("brady_E6_la_arrest_g3",)
                    + tuple(f"dystonia_E1_distal_leg_plantarflexion_ankle_g{g}" for g in (3, 4)))


def _since(clip_id, default="exam_v2"):
    """The exam set of a clip: the latest set whose list of changed bodies holds it (BODY_CHANGED_V10, V9, V8, V7,
    V6, V5), else default."""
    for mark, ids in ((V10, BODY_CHANGED_V10), (V9, BODY_CHANGED_V9), (V8, BODY_CHANGED_V8), (V7, BODY_CHANGED_V7),
                      (V6, BODY_CHANGED_V6), (V5, BODY_CHANGED_V5)):
        if clip_id in ids:
            return mark
    return default


HAND_SHORT = {"finger_tapping": ("E4a", "ft"), "hand_movements": ("E4b", "hm")}
HAND_VIEWS = {"finger_tapping": ("front", "hand_left_medial", "hand_left"),
              "hand_movements": ("front", "hand_left", "hand_left_medial")}


def _hand_duration(task, crit, g):
    """Seconds of the hand task's base body: 0.5 s lead, the 10 cycles at the clip's rate, the pauses, 1.5 s after
    (rates and pause lengths as handtasks.hand_task sets them)."""
    from .bradykinesia import HESITATION_MIN_S, arrest_length_s, hesitation_length_s
    f0 = G.normal_rate_hz(task)[0]
    f = G.slowing_rate_hz(task, g) if crit == "slowing" else f0
    pauses = 0.0
    if crit == "interruptions":
        pauses = G.BRADY_INTERRUPTIONS[g] * hesitation_length_s(1.0 / f0, HESITATION_MIN_S)
    elif crit == "arrest":
        pauses = arrest_length_s(1.0 / f0)
    return round(0.5 + G.N_CYCLES / f + pauses + 1.5, 2)


def _hand_specs():
    """Finger tapping (3.4) and hand movements (3.5), left hand: level 0 at the cited normal rate and amplitude, the
    decrement and interruption ladders, the arrest and the slowing ladder (handtasks.py)."""
    from .handtasks import TASKS, hand_task
    from .motions import posture_body
    out = []
    plan = [("level0", None, 0), ("decrement", 1, 1), ("decrement", 2, 2), ("decrement", 3, 3),
            ("interruptions", 1, 1), ("interruptions", 2, 2), ("interruptions", 3, 3), ("arrest", 3, 3),
            ("slowing", 1, 1), ("slowing", 2, 2), ("slowing", 3, 3)]
    for task, tk in TASKS.items():
        seg, short = HAND_SHORT[task]
        l0 = f"brady_{seg}_{short}_level0"
        name = task.replace("_", " ")
        for crit, g, gn in plan:
            def build(task=task, crit=crit, g=g, tk=tk):
                base = posture_body(tk["posture"], _hand_duration(task, crit, g))
                if crit == "level0":
                    return hand_task(base, task, "left", decrement_onset=None, seed=SEED)
                return hand_task(base, task, "left", grade=g, criterion=crit, seed=SEED + g)
            f0 = G.normal_rate_hz(task)[0]
            if crit == "level0":
                gt, pt = "level 0", f"10 cycles at the cited normal rate, {f0:.2f} Hz, no criterion"
            elif crit == "decrement":
                gt = f"MDS-UPDRS {tk['item']} grade {g} (decrement)"
                pt = (f"decrement from cycle {G.BRADY_DECREMENT_ONSET[g]}, {100 * G.BRADY_DECREMENT_SLOPE:g} percent "
                      f"per cycle")
            elif crit == "interruptions":
                gt = f"MDS-UPDRS {tk['item']} grade {g} (interruptions)"
                pt = f"{G.BRADY_INTERRUPTIONS[g]} hesitations"
            elif crit == "arrest":
                gt, pt = f"MDS-UPDRS {tk['item']} grade 3 (arrest)", "one arrest"
            else:
                gt = f"MDS-UPDRS {tk['item']} grade {g} (slowing, proposed ladder)"
                pt = f"{G.slowing_rate_hz(task, g):.2f} Hz, {G.slowing_ratio(task) ** g:.3f} of the cited normal rate"
            cid = f"brady_{seg}_{short}_{'level0' if crit == 'level0' else f'{crit}_g{g}'}"
            out.append(ClipSpec(cid, "bradykinesia", seg, tk["item"], f"bradykinesia, {name}, left", gn, gt, pt,
                                HAND_VIEWS[task], None if crit == "level0" else l0, "chair", build,
                                since=_since(cid, V4)))
    return out


def _raise_specs():
    """E2 on the raised posture (motions.outstretched: REST_BEFORE_RAISE_S at rest in 'sit', the arms raised from the
    chair to the 3.15 posture, fingers apart): level 0, postural tremor grade 2 from the posture onset, re-emergent
    tremor grade 2: a rest tremor of the forearm and hand (the 'rest' driver at the 3.17 grade-2 level, clear of the
    chair) from the start that stops with the raise (its on interval ends at the raise's start, the ramp centred
    there), then the tremor of the outstretched hand after the published mean latency counted from the posture
    onset, building up over REEMERGENT_RAMP_S from that time (the ramp starts at the latency: tremor.py centres its
    ramp on latency_s, which is the latency plus half the ramp here; the truth's visible_from_s is the latency, the
    first knot the tremor moves the hand, which the caption gives)."""
    from .common import truth_record
    from .motions import outstretched
    from .tremor import chair_boxes, tremor
    out = []
    l0_shared = "tremor_E2_raise_g0"       # the outstretched base at its default rest (module docstring, Exam_v7)
    l0 = "tremor_E2_raise_long_rest_g0"    # the level 0 of the two tremor raise clips (wave SH1d)
    views = ("front", "left", "top", "hands")

    def base():
        return outstretched(DUR["E2"], rest_s=REST_BEFORE_RAISE_S)

    def lvl0_shared():
        b, info = outstretched(DUR["E2"])
        return b, truth_record("none", None, 0, "level 0", {"base": "outstretched", "base_motion": info}, (), None)
    out.append(ClipSpec(l0_shared, "tremor", "E2", "3.15", "tremor, postural, left hand (arms raised to the posture)",
                        0, "MDS-UPDRS 3.15 grade 0", "no tremor (level 0): the arms raised from the chair, then held",
                        views, None, "chair", lvl0_shared, since=V3))

    def lvl0():
        b, info = base()
        return b, truth_record("none", None, 0, "level 0", {"base": "outstretched", "base_motion": info}, (), None)
    out.append(ClipSpec(l0, "tremor", "E2", "3.15", "tremor, postural, left hand (arms raised to the posture)", 0,
                        "MDS-UPDRS 3.15 grade 0",
                        f"no tremor (level 0): {REST_BEFORE_RAISE_S:g} s at rest, the arms raised from the chair, "
                        f"then held", views, None, "chair", lvl0, since=_since(l0)))

    def build_post():
        b, info = base()
        o, t = tremor(b, "left", grade=2, frequency_hz=tremor_hz("postural", 2), condition="postural",
                      tremor_type="essential", latency_s=info["onset_s"], driver=TREMOR_DRIVER["postural"],
                      seed=SEED + 52)
        o, t = _with_readings(o, t, b, 2)
        return o, dict(t, base_motion=info)
    out.append(ClipSpec("tremor_E2_raise_postural_g2", "tremor", "E2", "3.15",
                        "tremor, postural, left hand (arms raised to the posture)", 2, "MDS-UPDRS 3.15 grade 2",
                        "", views, l0, "chair", build_post, since=_since("tremor_E2_raise_postural_g2")))

    def build_re():
        b, info = base()
        r, tr_rest = tremor(b, "left", grade=2, cm_pp=tremor_level(2), frequency_hz=tremor_hz("rest", 2),
                            condition="rest", tremor_type="parkinsonian_rest", driver=TREMOR_DRIVER["rest"],
                            intervals=[(0.0, info["rest_s"])], seed=SEED + 54, clear=chair_boxes())
        o, t = tremor(r, "left", grade=2, frequency_hz=tremor_hz("rest", 2), condition="postural",
                      tremor_type="parkinsonian_rest",
                      latency_s=info["onset_s"] + G.REEMERGENCE_MEAN_S + REEMERGENT_RAMP_S / 2,
                      ramp_s=REEMERGENT_RAMP_S, driver=TREMOR_DRIVER["postural"], seed=SEED + 53)
        o, t = _with_readings(o, t, b, 2)
        rp = tr_rest["params"]
        rest_phase = {"intervals_s": rp["intervals_s"], "cm_pp": rp["cm_pp"], "frequency_hz": rp["frequency_hz"],
                      "driver": rp["driver"], "point": rp["point"], "angle_shares": rp["angle_shares"],
                      "deg_pp": rp["deg_pp"], "elbow_lift_deg": rp.get("elbow_lift_deg", 0.0),
                      "level_tip": rp.get("level_tip"), "stops_at_s": info["rest_s"], "ramp_s": rp["ramp_s"],
                      "validity": tr_rest["validity"], "mesh_calibration": rp["mesh_calibration"]}
        return o, dict(t, base_motion=info, rest_phase=rest_phase,
                       visible_from_s=info["onset_s"] + G.REEMERGENCE_MEAN_S,
                       arrays=dict(t["arrays"], rest_envelope=tr_rest["arrays"]["envelope"]))
    out.append(ClipSpec("tremor_E2_raise_reemergent_g2", "tremor", "E2", "3.15",
                        "tremor, re-emergent, left hand (arms raised to the posture)", 2,
                        "MDS-UPDRS 3.15 grade 2 (re-emergent)", "", ("front", "hands"), l0, "chair", build_re,
                        since=_since("tremor_E2_raise_reemergent_g2")))
    return out


def _arise_specs():
    """E7 arising (3.9): the healthy procedural arising with the arms folded (motions.arise), level 0."""
    from .common import truth_record
    from .motions import arise, arise_validity

    def build():
        b, info = arise(standing_s=DUR["E7"])
        return b, truth_record("arise", None, 0, "MDS-UPDRS 3.9 level 0", info,
                               ("li2021", "inkster2004", "hof2005", "deleva1996", "module"), arise_validity(b, info))
    return [ClipSpec("ref_arise_E7", "reference", "E7", "3.9", "healthy reference arising from a chair (procedural, "
                     "arms folded)", 0, "level 0 (cited phases, trunk solved for balance)", "", ("front", "left"), None,
                     "chair_arise", build, since=_since("ref_arise_E7", V4))]     # exam_v4 until SH9 (V6)


def scene_mesh(kind, body, floor_y):
    """(vertices, faces) of a clip's scene: 'chair' (viewer.chair_mesh under the procedural sit), 'chair_armless'
    (the same chair without its arms, viewer.chair_mesh(floor_y, armrests=False), shared wave SH5), 'chair_arise'
    (motions.arise_chair under the arising body's first knot), 'chair_table' and 'chair_table_both' (the chair and
    the writing table of the catalogue's writing clips, postures.scene_mesh), 'chair_target' (the chair and the
    examiner's finger at the reach target, reach.scene_mesh, shared wave SH6), 'exam_bed' (the examination couch
    under a supine body, ataxia.scene_mesh, shared wave SH7), None. An unknown kind raises ValueError."""
    if kind is None:
        return None
    from . import viewer as V
    if kind == "chair":
        return V.chair_mesh(floor_y)
    if kind == "chair_armless":
        return V.chair_mesh(floor_y, armrests=False)
    if kind == "chair_arise":
        from .motions import arise_chair
        return arise_chair(body, floor_y)
    if kind in ("chair_table", "chair_table_both"):
        from .postures import scene_mesh as posture_scene
        return posture_scene(kind, body, floor_y)
    if kind == "chair_target":
        from .reach import scene_mesh as reach_scene
        return reach_scene(kind, body, floor_y)
    if kind == "exam_bed":
        from .ataxia import scene_mesh as bed_scene
        return bed_scene(kind, body, floor_y)
    raise ValueError(f"scene {kind!r}")


def describe(spec, truth):
    """The caption's parameter text of a built clip, from its truth record (the levels actually applied)."""
    p = truth.get("params", {}) if truth else {}
    op = truth.get("operator") if truth else None
    if op == "tremor" and p.get("cm_pp"):
        t = _tremor_caption(p)
        rp = truth.get("rest_phase")
        if rp:                                       # the re-emergent tremor of the raise (_raise_specs)
            vis = truth.get("visible_from_s", p["latency_s"])
            return f"rest tremor until the raise ({rp['stops_at_s']:g} s), re-emerging at {vis:.2f} s: {t}"
        if p.get("share", 1.0) < 0.999:
            t += f", present {100 * p['share']:.1f} percent of the clip"
        if p.get("latency_s"):
            t += f", from {p['latency_s']:g} s"
        return t
    if op == "freezing" and p.get("episodes"):
        n = len(p["episodes"])
        eps = "; ".join(f"{e['context']} {e['onset_s']:.1f} to {e['offset_s']:.1f} s" for e in p["episodes"])
        e0 = p["episodes"][0]
        man = e0["manifestation"] + (", the knees alternating" if e0.get("pattern") == "alternating" else "")
        return f"{n} episode{'' if n == 1 else 's'} ({man}): {eps}"
    if op == "bradykinesia" and p.get("cycles") and p.get("unit"):          # the hand tasks (handtasks.py)
        return (f"{spec.params_text}; level-0 amplitude {p['amplitude0']:.1f} {p['unit']} at {p['level0_hz']:.2f} "
                f"Hz (cited normal)")
    if op == "bradykinesia" and p.get("cycles"):
        return f"{spec.params_text}; level-0 amplitude {p['amplitude0_deg']:.0f} deg at {p['level0_hz']:g} Hz"
    if op == "arise":
        t = (f"seat {100 * p['seat_height_m']:.0f} cm, rise {p['rise_s']:g} s (cited), lift-off at "
             f"{100 * p['liftoff_share']:.1f} percent, arms folded, then standing")
        tr, bl = p.get("trunk"), p.get("balance")
        if tr and bl:
            t += (f"; trunk lean {tr['realized_lean_peak_deg']:.0f} deg, solved for balance (module): XcoM inside "
                  f"the feet from lift-off")
        return t
    if op == "dystonic_posture" and "realized_peak_deg" in p:
        t = f"{p['realized_peak_deg']:.1f} deg held"
        first = truth.get("composed_with")
        if first and "realized_peak_deg" in first.get("params", {}):   # a composed posture names both parts
            t = (f"{_dystonia_part(first['params'], 'first posture')} "
                 f"{first['params']['realized_peak_deg']:.1f} deg and "
                 f"{_dystonia_part(p, 'second posture')} {p['realized_peak_deg']:.1f} deg held")
        if not p.get("reachable", True):
            why = "the planted foot" if p.get("plant_scale_min", 1.0) < 1.0 else "the joint-limit table"
            t += f" (of {p['level_deg']:.1f} asked, bounded by {why})"
        if p.get("share", 1.0) < 0.999:
            t += f", {100 * p['share']:.1f} percent of the clip"
        fp = p.get("foot_pose")
        if p.get("foot") == "plant" and fp:
            mode = (p.get("plant") or {}).get("mode")
            if p.get("toe_stance"):
                t += (f"; on the toes: heel {100 * fp['heel_lift_max_m']:.1f} cm up, ankle "
                      f"{fp['ankle_change_deg']:.0f} deg; not for DIS grading")
            elif mode == "ankle":
                t += (f"; foot planted, heel {100 * max(fp['heel_lift_max_m'], 0.0):.1f} cm up, knee "
                      f"{fp['knee_change_deg']:.0f} and hip {fp['hip_change_deg']:.0f} deg follow")
            else:
                t += "; foot planted flat"
        return t
    return spec.params_text


def exam_set(store=None):
    """[ClipSpec] of the first exam set (module docstring). store: kept for the callers' signature (catalogue_set,
    video.exam_driver); since exam_v10 no reference reads a body store, so it is not used. The walks and the stand
    are built once per call and shared by the clips built on them."""
    cache = {}

    def walk():
        if "walk" not in cache:
            cache["walk"] = reference_walk()
        return cache["walk"]

    def fog_walk():
        if "fog_walk" not in cache:
            cache["fog_walk"] = freezing_walk()
        return cache["fog_walk"]

    def stand():
        if "stand" not in cache:
            cache["stand"] = reference_stand()
        return cache["stand"]

    specs = (_reference_specs(stand) + _arise_specs() + _freezing_specs(walk, fog_walk) + _tremor_specs()
             + _raise_specs() + _brady_specs() + _hand_specs() + _dystonia_specs())
    ids = [s.clip_id for s in specs]
    if len(set(ids)) != len(ids):
        raise RuntimeError("duplicate clip ids in the exam set")
    known = set(ids)
    for s in specs:
        if s.level0 is not None and s.level0 not in known:
            raise RuntimeError(f"{s.clip_id}: level 0 {s.level0} is not in the set")
    return specs
