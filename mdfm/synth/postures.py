"""Abnormal postures of the symptom catalogue (task CAT-POST, since 'catalogue_v1'): parkinsonian axial postures, the
cervical dystonia subtypes and their time course, limb dystonia, the distribution of dystonia, and the paroxysmal
dyskinesias as episodic events, on the program's body (docs/proposals/SYNTH_PATIENT_DRAFT.md, section 6), each tied
to a scale item with its grades and the parameters of its sources (SOURCES here and grades.SOURCES).

The exam set already holds the static dystonic postures of dystonia.py (exam._dystonia_specs: TWSTRS neck rotation 1
to 4, laterocollis 1 to 3, antero- and retrocollis 1 to 3, shoulder elevation 1 to 3 held constant, one intermittent
rotation; DIS trunk lateral flexion, proximal arm abduction, elbow flexion, hip abduction, knee flexion; seated ankle
plantarflexion); this module does not rebuild them and points its standing clips' level 0 to the exam's.

Engine (module values unless a source is named):
- terms: a rotation offset of a joint or a dystonia.py region chain about one axis, in degrees at the full level,
  times an envelope (T,) in [0, 1] (apply_terms; dystonia._apply composes a chain, its last joint taking the
  remainder so that the end segment deviates by the offset). Terms that would leave the SP1 joint-limit table are
  scaled down per knot (fit_terms; limit_scale in the truth), so that no new violation is introduced.
- standing (plant_stand): a posture of the trunk, pelvis and upper body on a standing body keeps both feet where the
  base has them: the ankles stay at the base's positions, each foot keeps the base's global rotation (flat, same
  heading), the legs are solved by two-bone inverse kinematics (ik.hinge_leg, the knee bending forward) with the
  knee of the more extended leg at the posture's knee flexion; the pelvis moves horizontally so that the centre of
  mass (de Leva table, kernels/force.py) keeps the base's place over the ankles (fore-aft and lateral), and the
  ankle heights are corrected by the skinned soles (contact.mesh_sole_low) so that the mesh's lowest sole point
  stays where the base's is. A stooped patient flexes the knees and moves the pelvis back to keep the centre of
  mass over the feet; that the centre of mass keeps the healthy stand's place is this module's rule.
- foot touching (touch_floor): an ankle and toe posture (inversion, plantarflexion, adduction, toes) turns the foot
  about the ankle; the leg above is then raised just enough that the foot's lowest skinned point rests on the
  floor, by hip and knee flexion (the shank kept parallel: seated the thigh rises off the seat and the heel lifts,
  the foot pivoting on its forefoot), standing capped at LIFT_CAP_DEG and continued by a hike of the pelvis on that
  side, solved per knot by bisection: the foot touches on its lateral border or toes and bears little weight, the
  other foot stands.
- hanging arms (hang_arms): after a trunk posture the shoulders are re-aimed so that each humerus keeps the level-0
  stand's world direction (the arms hang instead of riding the trunk), inside the table (an internal rotation of the
  humerus, then the collar, where the shoulder's own room ends). Under a raised collar (a TWSTRS D shoulder
  elevation, SH2) the shoulder turns back with the collar (dystonia.hang_from_collar, fit_terms hang): the shoulder
  rises and the arm hangs under it.
- joint-limit tables (SH2, limits_ext.py): the dystonia operators of this module (cervical dystonia, head shifts,
  the geste antagoniste, limb dystonia and its distributions, writer's cramp and its overflow and mirror forms, the
  generalized torsion dystonia, opisthotonus, the paroxysmal dyskinesias but the choreic PKD attack) read the
  dystonia table (the healthy SP1
  table widened at the ankle's inversion to 30 deg, the toes' flexion to 35 deg and the shoulder's adduction by
  the collar's rise over the base) in their solvers and validity, and record it in their truths ('limits_table'); the
  parkinsonian and task-force axial postures, the choreic PKD attack and every level 0 keep the healthy table. A fit
  that cannot keep a knot inside the table takes the base at that knot (fit_terms), and the leg solves of a standing
  limb posture are clamped where they graze the table (project_rows, at most PROJECT_MAX_DEG). The paroxysmal
  attacks fit their arm flexions, arm turns and trunk and neck apart (attack_groups, fit_groups), each scale smoothed
  over FIT_SMOOTH_S (fit_terms smooth), so that a joint at its bound neither shrinks the rest of the attack nor snaps
  it back in one knot.
- scene checks: the geste's hand-to-head distance (head_clearance: reach.clearance on the skinned mesh, nothing
  inside the head or neck, the pads' fingers GESTE_GAP_M from it while the hand rests on the cheek; measure_geste: the
  palm and the wrist off the front of the neck, the wrist lateral to the face and below the jaw line; wrist_bend_deg:
  the wrist's flexion and deviation inside WRIST_RANGE_DEG; the hand and the forearm off the trunk), the paroxysmal
  dyskinesias' armrest check (armrest_report: no forearm or hand vertex inside a chair's armrest; the seated
  excerpt's left collar raised with the arm hanging under it where that forearm would enter it, clear_armrests, and
  the right arm held back, guard_rows), backrest
  check (backrest_report: no vertex inside the chair's backrest; the trunk's turn and the arm held back where an arm
  would enter it, guard_rows) and hand-to-thigh and hand-to-trunk checks (thigh_report, trunk_report, on the skin by
  arm_part_gap, from lift-off; the humerus's inward turn, and in the seated excerpt the elbow's flexion, held back
  where the forearm would enter the trunk, and the choreic arm where the hand would enter a thigh, guard_rows); the
  opening of the folded arms read on the final body (motions.unfold_check, the validity's 'fold'); the writing scene's
  check (table_report: the writing hand resting on the paper by the lowest skinned point of its palm and its ring and
  little fingers, the curled little finger in the tripod grip, at every knot, seat_hand; the pen's tip on the paper
  within words, seat_pen; the resting hand on the table; nothing through the table or the paper; the two hands apart).
- arm reach (arm_ik): the wrist or the palm centre put on a world point by a two-bone solve (the elbow a hinge
  about its local y axis, the shoulder turned so that the elbow points toward a pole direction), the forearm then
  twisted about its axis (half at the elbow, half at the wrist) so that the palm faces a given direction.
- time course: raised-cosine bouts (common.schedule), slow fluctuation (smooth_noise, FLUCT), mobile spasms and
  dystonic jerks (mobile_envelope), minimum-jerk ramps (motions.minimum_jerk).

Measures (re-measurement, measure_*): read back from the body, never from the parameters:
- axial_angles: the photograph angles of the MDS task force (Tinazzi et al. 2022, SOURCES['tinazzi2022']) on skin
  landmarks of the skinned mesh (LANDMARKS: vertices chosen once on the neutral rest mesh by the rules of
  landmark_ids): the total camptocormia angle (malleolus method: the lateral malleolus to L5 against L5 to C7), the
  upper camptocormia angle (upper method: L5 to the thoracic fulcrum against the fulcrum to C7), the neck angle
  (perpendicular method: C7 to the tragus against the vertical), the lateral trunk angle (perpendicular method: L5
  to C7 against the vertical, seen from behind), all in degrees in the body's heading frame; the level-0 stand's
  own values are reported beside (the neutral body is not at 0 on these methods);
- joint_deviation: the angle (deg) of a joint's or segment's rotation against the level-0 body, per knot; the share
  of time held (deviation above half its peak); the reduction under the geste antagoniste (mean deviation while the
  hand is on the face against the hold before it);
- the knee flexion, the pelvis shift, the centre of mass against the ankles, the foot's lowest skinned point.

Grades and their sources are in each operator's docstring and printed by params_text; a module value is marked
[module] and printed as such. Every body is procedural, in the shared neutral shape, adult, on the 100 Hz clock.
"""
import functools
import math
from dataclasses import replace

import numpy as np
import torch

from ..body.rotations import axis_angle_to_matrix, matrix_to_axis_angle, matrix_to_rot6d
from ..body.joints import fk
from . import contact as CT
from . import grades as G
from . import hands as H
from . import limits_ext as LX
from .base import STAND_REACH, static_body
from .body import (LIMIT_TOL_DEG, F64, axis_rotation, contact_labels, deviation, joint_limits, rotation_angle_deg,
                   validity)
from .common import as_rng, envelope, realized_share, schedule, truth_record
from .dystonia import _apply, _resolve, girdle_drop, hang_from_collar
from .dystonia import rooms as region_rooms
from .ik import hinge_leg, rmin
from .motions import minimum_jerk

CAT = "catalogue_v1"
SEED = 20261102
DUR_S = 15.0                 # [module] the standing and seated segments of the catalogue, as exam DUR['E7']
SOURCES = {
    "tinazzi2022": "Tinazzi M, Geroin C, Bhidayasiri R et al. Task force consensus on nosology and cut-off values for "
                   "axial postural abnormalities in parkinsonism. Mov Disord Clin Pract 2022; 9(5): 594-603, doi "
                   "10.1002/mdc3.13460 (PMC9274349, Table 2): normal lateral trunk < 5 deg, lateral trunk flexion 5 to "
                   "10, Pisa syndrome > 10 (perpendicular method); thoracic normal < 25, anterior trunk flexion 25 to "
                   "45, camptocormia with thoracic fulcrum > 45 (upper method); lumbar normal <= 15, anterior trunk "
                   "flexion > 15 to 30, camptocormia with lumbar fulcrum > 30 (malleolus method); neck normal <= 35, "
                   "anterior neck flexion > 35 to 45, antecollis > 45 (perpendicular method)",
    "margraf2018": "Margraf NG et al. Consensus for the measurement of the camptocormia angle in the standing patient. "
                   "Parkinsonism Relat Disord 2018 (PubMed 29907329): total angle from the lateral malleolus through "
                   "L5 to C7, upper angle at the thoracic fulcrum between L5 and C7 (abstract)",
    "mds_3_13": "MDS-UPDRS 3.13 posture (Goetz et al. 2008, doi 10.1002/mds.22340), paraphrased: 1 not quite erect, "
                "could be normal for an older person; 2 definite flexion, scoliosis or leaning, corrected to normal "
                "when asked; 3 stooped, scoliosis or leaning not corrected volitionally; 4 flexion, scoliosis or "
                "leaning causing an extreme abnormality",
    "twstrs_full": "TWSTRS severity as reproduced by Dystonia Canada (grades.SOURCES['twstrs']), read in full: A4 "
                   "lateral shift and A5 sagittal shift present or absent; B duration 0 to 5 (occasional < 25, "
                   "intermittent 25 to 50, frequent 50 to 75, constant > 75 percent of the time, crossed with mostly "
                   "submaximal or often maximal); C sensory tricks 0 complete relief, 1 partial or limited relief, 2 "
                   "little or no benefit; D shoulder elevation 1 mild (< 1/3 of the range) intermittent, 2 moderate "
                   "(1/3 to 2/3) and constant or severe (> 2/3) and intermittent, 3 severe and constant",
    "albanese2013": "Albanese A et al. Phenomenology and classification of dystonia: a consensus update. Mov Disord "
                    "2013; 28: 863-873, doi 10.1002/mds.25475: sustained or intermittent contractions, patterned and "
                    "twisting movements that may be tremulous, initiated or worsened by voluntary action, overflow; "
                    "distribution focal, segmental, multifocal, generalized, hemidystonia",
    "bfm": "Burke-Fahn-Marsden movement scale as reproduced in the IAACD toolkit 2018 (grades.SOURCES['dis']): per "
           "region a severity factor 0 to 4 times a provoking factor 0 to 4 (1 on a particular action, 2 on many "
           "actions, 3 on action of a distant part or intermittently at rest, 4 present at rest)",
    "wcrs": "Writer's Cramp Rating Scale, Wissel J et al. J Neurol Neurosurg Psychiatry 1996; 61: 172-175 (abstract "
            "read; items as listed in ClinicalTrials.gov NCT01738581: flexion or extension at the wrist, fingers and "
            "elbow, tremor, latency of dystonic symptoms, writing speed); the 0 to 2 anchors were not read",
    "latorre_bhatia": "Latorre A, Bhatia KP. Treatment of paroxysmal dyskinesia. Author manuscript, UCL Discovery "
                      "10096591 (full text, Table 2): PKD attacks of seconds to minutes, usually a few seconds up to "
                      "1 min, triggered by sudden movement, 1 per month to 100 per day (Bruno et al. 2004, doi "
                      "10.1212/01.wnl.0000147298.05983.50); PNKD attacks of minutes to hours, most often 15 to 40 "
                      "min, rarely under 5 min (Erro et al. 2014, doi 10.1002/mds.25933), triggered by coffee, tea, "
                      "alcohol, stress, fatigue, 1 per year to 23 per day; PED dystonia after prolonged exertion, "
                      "attacks of 5 to 30 min in Lance's 1977 family, subsiding with rest",
    "shaikh2015": "Shaikh AG, Zee DS, Jinnah HA. Oscillatory head movements in cervical dystonia: dystonia, tremor, or "
                  "both? Mov Disord 2015; 30(6): 834-842, doi 10.1002/mds.26231 (via the symptom catalogue, row D12): "
                  "jerky dystonic head movements as irregular sawtooth pulls at 0.1 to 0.5 Hz",
    "bruno2007": "Bruno MK, Lee HY, Auburger GW et al. Genotype-phenotype correlation of paroxysmal nonkinesigenic "
                 "dyskinesia. Neurology 2007; 68(21): 1782-1789, doi 10.1212/01.wnl.0000262029.91552.e0 (abstract, via "
                 "the symptom catalogue, row X02): attacks of 10 min to 1 h; caffeine, alcohol and stress as triggers",
    "limits_ext": "the SH2 dystonia extension of the joint-limit table (limits_ext.py; its SOURCES: jarm2022 for the "
                  "ankle inversion and the toe flexion, corroborated by menadue2006 and ocran2023; jarm2022 and "
                  "kenhub_sc for the collar, whose bound is kept)",
    "patel2014": "Patel N, Hanfelt J, Marsh L, Jankovic J, members of the Dystonia Coalition. Alleviating manoeuvres "
                 "(sensory tricks) in cervical dystonia. J Neurol Neurosurg Psychiatry 2014; 85(8): 882-884, doi "
                 "10.1136/jnnp-2013-307316 (abstract, via Europe PMC): of 154 patients, 138 (89.6 percent) used "
                 "alleviating manoeuvres; light touch, usually to the lower face or neck, in more than 90 percent",
    "jarm2022_wrist": "as limits_ext.SOURCES['jarm2022'], the same measurement table (jarm.or.jp document "
                      "kadou/03.pdf): wrist palmar flexion 0-90 deg, dorsiflexion 0-70 deg, radial deviation 0-25 "
                      "deg, ulnar deviation 0-55 deg (WRIST_RANGE_DEG, the geste's wrist)",
    "module": "a value of this module, not taken from a source (stated where used)",
}
HEALTHY, DYST = LX.HEALTHY, LX.DYSTONIA   # the joint-limit tables (limits_ext): level 0s and parkinsonian, dystonia


# ---------------------------------------------------------------- terms
class Term:
    """A rotation offset (module docstring): chain (joints), axis, sign, frame, deg (full level, before sign), env
    (T,) share of the level, split (one share per joint of the chain), label."""

    def __init__(self, chain, axis, sign, frame, deg, env, split=None, label=""):
        self.chain, self.axis, self.sign, self.frame = tuple(chain), axis, float(sign), frame
        self.deg, self.env = float(deg), np.asarray(env, dtype=np.float64)
        self.split = tuple(split) if split is not None else tuple([1.0 / len(self.chain)] * len(self.chain))
        self.label = label

    def record(self):
        return {"label": self.label, "chain": list(self.chain), "axis": self.axis, "sign": self.sign,
                "frame": self.frame, "deg": self.deg, "split": [float(s) for s in self.split],
                "env_max": float(self.env.max()) if self.env.size else 0.0}


def region_term(base, region, direction, side, deg, env, label=None, k=0, table=HEALTHY):
    """A Term on a dystonia.py region (REGIONS): its chain, axis, sign and frame, split by each joint's room at knot k
    of the base in table (as dystonic_posture splits)."""
    chain, axis, sign, _, frame, _ = _resolve(region, direction, side)
    rm = region_rooms(base, chain, axis, sign, frame, k, table=table)
    split = [r / sum(rm) for r in rm] if sum(rm) > 0 else None
    return Term(chain, axis, sign, frame, deg, env, split, label or f"{region} {direction} {side}")


def region_room(base, region, direction, side, k=0, table=HEALTHY):
    """Degrees of offset table leaves from knot k of the base in a region's direction (summed over its chain)."""
    chain, axis, sign, _, frame, _ = _resolve(region, direction, side)
    return float(sum(region_rooms(base, chain, axis, sign, frame, k, table=table)))


def joint_term(joint, axis, deg, env, frame="parent", label=""):
    """A Term on one joint; deg carries its sign."""
    return Term((joint,), axis, 1.0, frame, deg, env, (1.0,), label or f"joint {joint} {axis}")


def apply_terms(R, terms, scale=None):
    """(T, 22, 3, 3): R with every term composed in order, each knot's offsets times scale (T,) (default 1)."""
    R = R.clone()
    for t in terms:
        if t.deg == 0:
            continue
        th = np.radians(t.deg) * t.sign * t.env * (1.0 if scale is None else scale)
        offs = [torch.from_numpy(th * s) for s in t.split]
        R = _apply(R, t.chain, t.axis, offs, t.frame)
    return R


def _outside(R, base_comp, rows, table=HEALTHY):
    """(T,) bool: knots where a component of the rows leaves [min(lo, base), max(hi, base)] of table at that knot
    (rows 1 to 21; limits_ext.excess)."""
    rows = [r for r in rows if r > 0]
    if not rows:
        return np.zeros(R.shape[0], bool)
    comp = np.degrees(matrix_to_axis_angle(R).numpy())
    return (LX.excess(table, comp, base_comp, rows) > 1e-6).any(-1).any(-1)


FIT_SMOOTH_S = 0.15           # [module] the paroxysmal attacks' fit scale smoothed over +-0.15 s (fit_terms smooth)
FIT_SMOOTH_PASSES = 3


def smooth_scale(s, k):
    """(T,): a running minimum of the scale s over +-k knots, then a running mean of the same width. Never above s at
    a knot: each mean's window holds only minima whose own window holds that knot (edge-padded at both ends)."""
    s = np.asarray(s, dtype=np.float64)
    T = len(s)
    if k <= 0 or (s == s[0]).all():
        return s.copy()
    p = np.pad(s, k, mode="edge")
    m = np.min([p[i:i + T] for i in range(2 * k + 1)], axis=0)
    return np.convolve(np.pad(m, k, mode="edge"), np.ones(2 * k + 1) / (2 * k + 1), mode="valid")


def fit_terms(base_R, terms, iters=80, table=HEALTHY, hang=(), smooth=0):
    """(R, scale (T,)): the terms applied to base_R, each knot's offsets scaled down by 3 percent steps where a joint
    would leave table (module docstring); a knot still outside after iters steps takes scale 0 (the base: since SH2
    the fit never leaves a new violation, before it stopped at 0.97^iters); hang: collars (13, 14) whose shoulders
    turn back with them so that the arm keeps hanging as in base_R (dystonia.hang_from_collar), checked with the
    terms. smooth (knots, 0: off): the scale then replaced by smooth_scale (never above the per-knot scale, so that a
    joint reaching its bound in one knot does not snap the posture back in one knot), checked against the table again
    and scaled down further where it still leaves it (FIT_SMOOTH_PASSES passes; the last result never leaves it)."""
    rows = sorted({j for t in terms for j in t.chain} | {c + 3 for c in hang})
    base_comp = np.degrees(matrix_to_axis_angle(base_R).numpy())

    def build(s):
        R = apply_terms(base_R, terms, s)
        return hang_from_collar(base_R, R, hang) if hang else R

    def shrink(s):
        for _ in range(iters):
            R = build(s)
            bad = _outside(R, base_comp, rows, table)
            if not bad.any():
                return R, s
            s = np.where(bad, s * 0.97, s)
        R = build(s)
        bad = _outside(R, base_comp, rows, table)
        if bad.any():
            s = np.where(bad, 0.0, s)
            R = build(s)
        return R, s
    R, s = shrink(np.ones(base_R.shape[0]))
    for _ in range(FIT_SMOOTH_PASSES if smooth else 0):
        s2 = smooth_scale(s, int(smooth))
        if np.array_equal(s2, s):
            break
        R2 = build(s2)
        if not _outside(R2, base_comp, rows, table).any():
            return R2, s2
        R, s = shrink(s2)
    return R, s


def fit_groups(base_R, groups, table=HEALTHY, smooth=0):
    """(R, scale (T,)): fit_terms on each group of terms in turn, each on the result of the groups before it, so that a
    joint at its bound scales its own group only; scale: the knot-wise minimum over the groups."""
    R, s = base_R, np.ones(base_R.shape[0])
    for g in groups:
        if g:
            R, sg = fit_terms(R, list(g), table=table, smooth=smooth)
            s = np.minimum(s, sg)
    return R, s


# ---------------------------------------------------------------- envelopes
def ramp_env(T, rate, on_s=0.0, ramp_s=1.0, off_s=None, off_ramp_s=1.0):
    """(T,) a minimum-jerk rise from on_s over ramp_s (ramp_s 0: a step; on_s 0 and ramp_s 0: held from the first
    knot) and, with off_s, a minimum-jerk fall from off_s over off_ramp_s."""
    t = np.arange(T) / rate
    up = (t >= on_s).astype(np.float64) if ramp_s <= 0 else minimum_jerk((t - on_s) / ramp_s)
    if off_s is None:
        return up
    dn = (t >= off_s).astype(np.float64) if off_ramp_s <= 0 else minimum_jerk((t - off_s) / off_ramp_s)
    return up * (1.0 - dn)


def smooth_noise(T, rate, rng, band=(0.1, 0.4), n=5):
    """(T,) a smooth random signal in [-1, 1]: n sinusoids with frequencies drawn in band (Hz), random phases and
    weights, normalized by its largest magnitude."""
    t = np.arange(T) / rate
    f = rng.uniform(band[0], band[1], n)
    ph = rng.uniform(0.0, 2 * np.pi, n)
    a = rng.uniform(0.5, 1.0, n)
    x = (a[:, None] * np.sin(2 * np.pi * f[:, None] * t[None] + ph[:, None])).sum(0)
    return x / max(np.abs(x).max(), 1e-12)


# ---------------------------------------------------------------- landmarks and axial angles
def _posterior_midline(v, dom, y, band=0.012, joints=(0, 3, 6, 9, 12)):
    """The vertex of the back's midline (|x| < 2 cm, dominant joint in joints) nearest height y, most posterior."""
    sel = np.flatnonzero(np.isin(dom, joints) & (np.abs(v[:, 0]) < 0.02) & (np.abs(v[:, 1] - y) < band))
    return int(sel[np.argmin(v[sel, 2])])


@functools.lru_cache(maxsize=1)
def landmark_ids():
    """{name: vertex id} of the skin landmarks of axial_angles on the neutral rest mesh (rules, never typed ids):
    'c7' the most posterior midline vertex at the neck joint's height (the vertebra prominens); 'l5' the most
    posterior midline vertex 3 cm above the pelvis joint (the lumbosacral level, at the iliac crests); 'fulcrum' the
    most posterior midline vertex halfway between the spine2 and spine3 joints (the thoracic apex of the bend this
    module applies); 'malleolus' the most lateral vertex of the left ankle segment within 1.5 cm of the ankle
    joint's height; 'tragus' the most lateral vertex of the left side of the head within 2 cm of 3 cm above the
    head joint and behind the head joint's forward position plus 1 cm."""
    J, v, dom = H._rest()
    out = {"c7": _posterior_midline(v, dom, J[12, 1]), "l5": _posterior_midline(v, dom, J[0, 1] + 0.03),
           "fulcrum": _posterior_midline(v, dom, 0.5 * (J[6, 1] + J[9, 1]))}
    ank = np.flatnonzero((dom == 7) & (np.abs(v[:, 1] - J[7, 1]) < 0.015))
    out["malleolus"] = int(ank[np.argmax(v[ank, 0])])
    hd = np.flatnonzero((dom == 15) & (np.abs(v[:, 1] - (J[15, 1] + 0.03)) < 0.02) & (v[:, 2] < J[15, 2] + 0.01))
    out["tragus"] = int(hd[np.argmax(v[hd, 0])])
    return out


LANDMARKS = ("c7", "l5", "fulcrum", "malleolus", "tragus")


def _heading_frame(P, R_root):
    """(T, M, 3) points P in each knot's heading frame (the root's yaw removed, origin kept)."""
    from ..body.rotations import rot_y, yaw_angle
    Hy = rot_y(-yaw_angle(torch.as_tensor(R_root, dtype=F64))).numpy()
    return np.einsum("tij,tkj->tki", Hy, P)


def _sag(v):
    """Forward inclination (deg) of vectors (..., 3) from the vertical in the sagittal plane (y up, z forward)."""
    return np.degrees(np.arctan2(v[..., 2], v[..., 1]))


def _cor(v):
    """Lateral inclination (deg) from the vertical in the coronal plane, toward the body's left (+x) positive."""
    return np.degrees(np.arctan2(v[..., 0], v[..., 1]))


def axial_angles(body, knots=None):
    """{name: (N,) deg} of the photograph angles (module docstring) at the knots (default every knot): 'tcc'
    (malleolus method, forward flexion positive), 'ucc' (upper method), 'neck' (perpendicular, forward positive),
    'lateral' (perpendicular, toward the body's left positive)."""
    ids = landmark_ids()
    order = [ids[k] for k in LANDMARKS]
    P = H.body_points(body, order, knots=knots)
    Rr = body.local_R()[:, 0] if knots is None else body.local_R()[list(knots), 0]
    P = _heading_frame(P, Rr)
    c7, l5, fu, ma, tr = (P[:, i] for i in range(5))
    return {"tcc": _sag(c7 - l5) - _sag(l5 - ma), "ucc": _sag(c7 - fu) - _sag(fu - l5),
            "neck": _sag(tr - c7), "lateral": _cor(c7 - l5)}


def band_of(kind, deg):
    """The task force category of a measured angle (tinazzi2022): kind 'lumbar' (tcc), 'thoracic' (ucc), 'neck',
    'lateral' (absolute value)."""
    a = abs(deg) if kind == "lateral" else deg
    if kind == "lateral":
        return "normal" if a < 5 else "lateral trunk flexion" if a <= 10 else "Pisa syndrome"
    if kind == "lumbar":
        return ("normal" if a <= 15 else "anterior trunk flexion (lumbar)" if a <= 30 else
                "camptocormia (lumbar fulcrum)")
    if kind == "thoracic":
        return "normal" if a < 25 else "anterior trunk flexion (thoracic)" if a <= 45 else \
            "camptocormia (thoracic fulcrum)"
    if kind == "neck":
        return "normal" if a <= 35 else "anterior neck flexion" if a <= 45 else "antecollis"
    raise ValueError(kind)


TF_BANDS = {"lumbar": {"normal": (None, 15.0), "mild": (15.0, 30.0), "cut": 30.0},
            "thoracic": {"normal": (None, 25.0), "mild": (25.0, 45.0), "cut": 45.0},
            "neck": {"normal": (None, 35.0), "mild": (35.0, 45.0), "cut": 45.0},
            "lateral": {"normal": (None, 5.0), "mild": (5.0, 10.0), "cut": 10.0}}
OPEN_BAND_FACTOR = 1.5       # [module] a level in an open band (camptocormia, Pisa, antecollis) at 1.5 x its cut-off
NECK_OPEN_DEG = 52.0         # [module] the antecollis of the catalogue (open-body port, wave 4, D53): 1.5 x the 45 deg
#                              cut-off lies past what jarm2022's 60 deg of cervical flexion reaches from the level-0
#                              stand on the open body (58.5 deg, the level 0 reading 27.4); the cut-off plus half of the
#                              13.5 deg left past it, rounded (scripts/openbody_wave4_checks.py, axial)


# ---------------------------------------------------------------- standing: legs re-solved, centre of mass kept
LEGS = ((1, 4, 7), (2, 5, 8))
HIP_MARGIN_M = 1e-10         # [module] plant_stand sets the hip this far inside the more extended leg's reach, so that
#                              ik.hinge_leg never reads the ankle past it; open-body port, wave 4 (D55): 1e-7 before,
#                              which on the open body's straight legs (design D30) bent the knee by the square root of
#                              it, 0.18 mm of knee travel at the first knot of a posture that differs from the base
SOLE_PASSES = 2
COM_ITERS = 8


def _rx(k):
    z = torch.zeros_like(k)
    return axis_angle_to_matrix(torch.stack([k, z, z], -1))


def leg_distance(p, q, kappa_rad):
    """|p + R_x(kappa) q| (m) of rest bones p (hip to knee) and q (knee to ankle) at knee flexions kappa (T,)."""
    k = torch.as_tensor(kappa_rad, dtype=F64)
    qk = torch.einsum("tij,j->ti", _rx(k), q)
    return (p[None] + qk).norm(dim=-1)


def _com(J, gender):
    from ..kernels.force import com
    return com(np.asarray(J), int(gender))


def plant_stand(base, R, knee_deg, transl0=None, com_keep=True):
    """(Body, info): the standing solve of the module docstring. base: the standing level-0 body; R (T, 22, 3, 3) the
    posture's local rotations (root, trunk, neck and arms; the leg rows are replaced); knee_deg (T,) the knee flexion
    of the more extended leg. info: the pelvis shift (m), the knee flexions per leg, the centre of mass against the
    ankles, the sole correction."""
    rest = base.rest()
    T = base.T
    Jb, Gb = base.fk()
    A0 = Jb[:, [7, 8]].clone()
    Gfoot = Gb[:, [7, 8]]
    G0 = R[:, 0]
    kap = torch.as_tensor(np.radians(np.broadcast_to(np.asarray(knee_deg, dtype=np.float64), (T,))).copy())
    pq = [(rest[kn] - rest[h], rest[a] - rest[kn]) for h, kn, a in LEGS]
    # never straighter than 'stand' (base.STAND_REACH, design D125): a knee asked straighter (an asked flexion under
    # the stand's, a posture corrected back to normal) holds its hip STAND_REACH of the leg's length from its ankle,
    # off the straight leg's singularity, where the layer's sway re-solve of the legs kinked (wave 8 second part, D157)
    d = [torch.minimum(leg_distance(p, q, kap), torch.tensor(STAND_REACH * float(p.norm() + q.norm()), dtype=F64))
         for p, q in pq]
    hip_off = [torch.einsum("tij,j->ti", G0, rest[h] - rest[0]) for h, _, _ in LEGS]
    cb = _com(Jb.numpy(), base.gender)
    amid_b = 0.5 * (Jb[:, 7] + Jb[:, 8]).numpy()
    target = (cb - amid_b)[:, [0, 2]]
    P = Jb[:, 0].clone()
    fwd = G0[:, :, 2].clone()
    fwd[:, 1] = 0.0
    fwd = fwd / fwd.norm(dim=-1, keepdim=True)
    dy = torch.zeros(T, 2, dtype=F64)
    lb = CT.mesh_sole_low(base)

    def solve(P, A):
        ys = []
        for i in range(2):
            Hh = P + hip_off[i]
            h2 = ((Hh[:, [0, 2]] - A[:, i, [0, 2]]) ** 2).sum(-1)
            ys.append(A[:, i, 1] - hip_off[i][:, 1] + torch.sqrt((d[i] ** 2 - h2).clamp_min(0.0)))
        P = P.clone()
        P[:, 1] = torch.minimum(ys[0], ys[1]) - HIP_MARGIN_M
        Rn = R.clone()
        kn_out = []
        for i, (h, kn, a) in enumerate(LEGS):
            Hh = P + hip_off[i]
            G_th, k = hinge_leg(Hh, A[:, i], Hh + fwd, pq[i][0], pq[i][1], G0,
                                hinge=torch.cross(fwd, A[:, i] - Hh, dim=-1))
            Rk = _rx(k)
            Rn[:, h] = G0.transpose(-1, -2) @ G_th
            Rn[:, kn] = Rk
            Rn[:, a] = (G_th @ Rk).transpose(-1, -2) @ Gfoot[:, i]
            kn_out.append(k)
        return P, Rn, kn_out

    for _ in range(SOLE_PASSES + 1):
        A = A0.clone()
        A[:, :, 1] += dy
        for _ in range(COM_ITERS if com_keep else 1):
            P, Rn, ks = solve(P, A)
            with torch.no_grad():
                J, _ = fk(Rn, P - rest[0], rest)
            if not com_keep:
                break
            c = _com(J.numpy(), base.gender)
            amid = 0.5 * (J[:, 7] + J[:, 8]).numpy()
            err = (c - amid)[:, [0, 2]] - target
            P[:, 0] -= torch.from_numpy(err[:, 0])
            P[:, 2] -= torch.from_numpy(err[:, 1])
        P, Rn, ks = solve(P, A)
        out = replace(base.copy(), rot6d=matrix_to_rot6d(Rn), transl=P - rest[0])
        lo = CT.mesh_sole_low(out)
        dy = dy + torch.from_numpy(lb - lo)
    out = replace(out, contact=contact_labels(out))
    J = out.joints().numpy()
    c = _com(J, out.gender)
    amid = 0.5 * (J[:, 7] + J[:, 8])
    info = {"pelvis_shift_m": {"x_max": float(np.abs(J[:, 0, 0] - Jb[:, 0, 0].numpy()).max()),
                               "z_min": float((J[:, 0, 2] - Jb[:, 0, 2].numpy()).min()),
                               "y_min": float((J[:, 0, 1] - Jb[:, 0, 1].numpy()).min())},
            "knee_flexion_deg": {"left_max": float(np.degrees(ks[0].numpy()).max()),
                                 "right_max": float(np.degrees(ks[1].numpy()).max())},
            "com_minus_ankles_m": {"base": [float(x) for x in target[0]],
                                   "max_error": float(np.abs((c - amid)[:, [0, 2]] - target).max())},
            "ankle_correction_m": [float(dy.min()), float(dy.max())],
            "rule": "ankles and foot rotations of the base kept; pelvis moved so that the centre of mass keeps the "
                    "base's place over the ankles [module]"}
    return out, info


def stand_validity(base, out, table=HEALTHY):
    """validity() of a standing output (its joint limits read against table, limits_ext.check) plus the skinned
    soles of both feet against the base (contact.sole_report on each side: within contact.TOL_M) and the leg
    continuity (contact.step_report)."""
    v = LX.check(table, base, out)
    rl, rr = CT.sole_report(base, out, "left", True), CT.sole_report(base, out, "right", True)
    v["contact"] = {"ok": bool(rl["ok"] and rr["ok"]), "left": rl, "right": rr}
    sl, sr = CT.step_report(base, out, "left"), CT.step_report(base, out, "right")
    v["continuity"] = {"ok": bool(sl["ok"] and sr["ok"]), "left": sl, "right": sr}
    return v


# ---------------------------------------------------------------- parkinsonian axial postures
# Shapes of a trunk bend (deg of each joint per degree of amplitude; axis x, flexion positive) [module]: 'stoop' the
# parkinsonian flexion spread over the pelvis (hip flexion) and the three spine joints; 'lumbar' the bend of the
# lumbar fulcrum (pelvis and spine1: L1 to L5 and the hips, tinazzi2022); 'thoracic' the bend of the thoracic fulcrum
# (spine2 and spine3: C7 to T12).
ARM_COLLAR = {"left": 13, "right": 14}
TRUNK_SHAPES = {"stoop": ((0, "parent", 0.35), (3, "child", 0.25), (6, "child", 0.25), (9, "child", 0.15)),
                "lumbar": ((0, "parent", 0.5), (3, "child", 0.5)),
                "thoracic": ((6, "child", 0.68), (9, "child", 0.32))}
MEASURE_OF = {"stoop": "tcc", "lumbar": "tcc", "thoracic": "ucc", "lateral": "lateral", "neck": "neck"}
# MDS-UPDRS 3.13 ladder (module readings of the paraphrased anchors, SOURCES['mds_3_13'], against the task force bands
# of SOURCES['tinazzi2022'] on the malleolus method): tcc the total camptocormia angle held; knee, elbow the added
# knee and elbow flexion; correct the share of the posture undone when the patient is asked to stand straight
# (grades 1 and 2: corrected to normal; 3: not corrected, a small attempt; 4: extreme, a smaller one).
PD_313 = {1: {"tcc": 12.0, "knee": 5.0, "elbow": 10.0, "internal": 5.0, "correct": 1.0},
          2: {"tcc": 20.0, "knee": 10.0, "elbow": 20.0, "internal": 10.0, "correct": 1.0},
          3: {"tcc": 27.5, "knee": 15.0, "elbow": 30.0, "internal": 15.0, "correct": 0.25},
          4: {"tcc": 45.0, "knee": 20.0, "elbow": 40.0, "internal": 20.0, "correct": 0.10}}
#   internal: the hanging humerus turned inward (deg, about its own axis) so that the flexed forearms bring the hands
#   in front of the thighs [module]
ASK_S = 8.0                  # [module] the examiner asks the patient to stand straight at this time
CORRECT_S = 1.5              # [module] the volitional correction's duration
HEAD_COMP = {"stoop": (0.15, -0.35), "lumbar": (0.0, -0.25), "thoracic": (0.0, -0.25)}   # [module] neck, head x
PISA_HEAD_COMP = 0.3         # [module] the neck tilts back toward the vertical by this share of the trunk's bend
BISECT = 26


def _shape_terms(base, shape, amp, env, side="left"):
    """Terms of a trunk bend of amplitude amp (deg), with the head compensation of HEAD_COMP (the gaze kept up)."""
    if shape == "lateral":
        terms = [region_term(base, "trunk", f"lateral_{side}", side, amp, env, "trunk lateral flexion")]
        other = "right" if side == "left" else "left"
        terms.append(region_term(base, "neck", f"tilt_{other}", side, PISA_HEAD_COMP * amp, env,
                                 "neck tilt back toward the vertical"))
        # the arms keep hanging: the arm on the bent side abducts by the bend; the other arm's girdle drops under it
        # in axial_body (far_arm_drop), by dystonia's rule for the same bend (girdle_drop: until its forearm rests
        # against the trunk, at most the girdle's depression range), and past that the arm rides the bend off the
        # vertical (open-body port, wave 4, D52: the collar lowered by the whole bend capped the bend at jarm2022's
        # 10 deg of depression, the Pisa syndrome then read 5.5 deg, and put that forearm 14 mm inside the trunk and
        # 23 mm inside the thigh: scripts/openbody_wave4_checks.py, axial)
        terms.append(region_term(base, "proximal_arm", "abduction", side, amp, env, f"arm hangs ({side})"))
        return terms
    if shape == "neck":
        return [region_term(base, "neck", "anterocollis", side, amp, env, "neck flexion")]
    terms = [joint_term(j, "x", share * amp, env, fr, f"flexion {j}") for j, fr, share in TRUNK_SHAPES[shape]]
    nk, hd = HEAD_COMP[shape]
    if nk:
        terms.append(joint_term(12, "x", nk * amp, env, "child", "neck forward"))
    if hd:
        terms.append(joint_term(15, "x", hd * amp, env, "child", "head extended to keep the gaze"))
    return terms


def far_arm_drop(base, R, env, side="left"):
    """(R, info): the rotations R of a lateral bend toward side with the other arm's collar lowered under it at every
    knot by the drop dystonia.girdle_drop reads at the knot of the largest envelope (the forearm and the hand resting
    against the trunk, the girdle's depression range at most) times the knot's share of that envelope."""
    from .dystonia import _lowered
    other = "right" if side == "left" else "left"
    e = np.asarray(env, dtype=np.float64)
    k = int(np.argmax(e))
    if e[k] <= 0.0:
        return R, {"girdle_drop_deg": 0.0}
    deg, info = girdle_drop(base, R, other, k)
    c = ARM_COLLAR[other]
    R = R.clone()
    R[:, c] = _lowered(R[:, c], other, torch.as_tensor(deg * e / e[k], dtype=F64))
    return R, dict(info, side=other, knot=k)


def _shape_max(base, shape, side="left"):
    """The largest amplitude (deg) whose shape terms stay inside the table at knot 0 (each joint's room over its
    share; the region rooms for 'lateral' and 'neck')."""
    from .body import room_deg
    R = base.local_R()[0]
    if shape == "lateral":                       # the other arm rides past its girdle's and shoulder's rooms
        return min(region_room(base, "trunk", f"lateral_{side}", side),
                   region_room(base, "proximal_arm", "abduction", side))
    if shape == "neck":
        return region_room(base, "neck", "anterocollis", side)
    caps = [room_deg(R[j], j, "x", 1.0, fr) / share for j, fr, share in TRUNK_SHAPES[shape] if j > 0]
    nk, hd = HEAD_COMP[shape]
    if nk:
        caps.append(room_deg(R[12], 12, "x", 1.0, "child") / nk)
    if hd:
        caps.append(room_deg(R[15], 15, "x", -1.0, "child") / -hd)
    return float(min(caps))


HANG_SHAPES = ("stoop", "lumbar", "thoracic")   # sagittal bends: the arms hang (hang_arms), never ride the trunk
HANG_ELBOW_DEG = 10.0        # [module] the relaxed elbow flexion added to the hanging arms of a sagittal bend


def axial_body(base, shape, amp, env, knee_deg=0.0, extra=(), side="left", stand=True, post=(), far_arm=False):
    """(Body, terms, scale, info) of a trunk or neck posture of amplitude amp on a standing base (plant_stand unless
    stand is False or the shape is the neck's: a neck posture leaves the legs alone); in a sagittal bend
    (HANG_SHAPES) the arms hang (hang_arms) before the legs are solved; post: terms composed after the hang (an arm's
    own posture on the hanging arm), scaled into the table as the others. far_arm: in a lateral bend the girdle of
    the arm away from the bend drops under it (far_arm_drop; solve_amplitude leaves it out, since it moves neither
    landmark of the lateral measure)."""
    terms = _shape_terms(base, shape, amp, env, side) + list(extra)
    R, s = fit_terms(base.local_R(), terms)
    hang = None
    if shape == "lateral" and far_arm:
        R, hang = far_arm_drop(base, R, env, side)
        hang = {"far_arm": hang}
    if shape in HANG_SHAPES:
        R, hang = hang_arms(base, R)
    if post:
        R, s2 = fit_terms(R, list(post))
        s = np.minimum(s, s2)
        terms = terms + list(post)
    if stand and shape != "neck":
        out, info = plant_stand(base, R, np.asarray(knee_deg, dtype=np.float64) * np.ones(base.T))
    else:
        rows = sorted({j for t in terms for j in t.chain} | ({13, 14, 16, 17} if hang else set()))
        out, info = base.with_rotations(R, rows), {}
        out = replace(out, contact=contact_labels(out))
    if hang:
        info["arms_hang"] = hang
    return out, terms, s, info


def solve_amplitude(base, shape, target_deg, knee_deg=0.0, extra_fn=None, side="left", stand=True, post_fn=None):
    """(amp deg, reachable, measured deg): the amplitude whose held posture reads target_deg on its measure
    (MEASURE_OF) at a two-knot copy of the base's first knot, by bisection up to _shape_max; extra_fn and post_fn
    (b, env) -> Terms: the terms composed with the shape's and those composed after the arms hang (axial_body)."""
    from .body import crop
    b2 = crop(base, 0, 2)
    kind = MEASURE_OF[shape]
    hi = _shape_max(base, shape, side)
    one = np.ones(2)

    def read(a):
        extra = extra_fn(b2, one) if extra_fn else ()
        o, _, sc, _ = axial_body(b2, shape, a, one, knee_deg, extra, side, stand, post_fn(b2, one) if post_fn else ())
        if sc.min() < 1.0:                       # a term left the table: this amplitude is not held as asked
            return -math.inf
        v = float(axial_angles(o, knots=[0])[kind][0])
        return -v if (kind == "lateral" and side == "right") else v
    if read(hi) == -math.inf:                    # the extra terms bound the amplitude below the shape's room
        lo_, hi_ = 0.0, hi
        for _ in range(BISECT):
            m = 0.5 * (lo_ + hi_)
            lo_, hi_ = (m, hi_) if read(m) > -math.inf else (lo_, m)
        hi = lo_
    v_hi = read(hi)
    if v_hi < target_deg:
        return hi, False, v_hi
    lo = 0.0
    for _ in range(BISECT):
        m = 0.5 * (lo + hi)
        lo, hi = (m, hi) if read(m) < target_deg else (lo, m)
    a = 0.5 * (lo + hi)
    return a, True, read(a)


def _axial_truth(name, base, out, terms, s, info, params, sources, val, kind, grade, scale_name, extra_arrays=None):
    ang = axial_angles(out)
    a0 = axial_angles(base, knots=[0])
    params = dict(params, terms=[t.record() for t in terms], limit_scale_min=float(s.min()),
                  base_angles_deg={k: float(v[0]) for k, v in a0.items()}, stand=info)
    arrays = {"axial_" + k: v for k, v in ang.items()}
    arrays.update(extra_arrays or {})
    return truth_record(name, None, grade, scale_name, params, sources, val, arrays=arrays)


def parkinsonian_posture(base, grade, ask_s=ASK_S, side="left"):
    """(Body, truth) of MDS-UPDRS 3.13 grade 0 to 4 on a standing base (module docstring): the stoop (PD_313: the
    total camptocormia angle held, knees and elbows flexed, the head extended to keep the gaze) from the first knot,
    and at ask_s the request to stand straight, undone by PD_313[grade]['correct'] over CORRECT_S. Grade 0: a copy."""
    g = int(grade)
    if g == 0:
        return base.copy(), truth_record("parkinsonian_posture", None, 0, "MDS-UPDRS 3.13", {"level0": True},
                                         ("mds_3_13",), None)
    p = PD_313[g]
    T, rate = base.T, base.rate
    t = np.arange(T) / rate
    env = 1.0 - p["correct"] * minimum_jerk((t - ask_s) / CORRECT_S)

    def extra(b, e):
        out = []
        for sd in ("left", "right"):        # the hanging arm has no adduction room in the table: elbows only
            out.append(region_term(b, "distal_arm", "elbow_flexion", sd, p["elbow"], e, f"elbow flexion {sd}"))
        return out

    def post(b, e):                        # on the hanging arms (hang_arms): the humerus turned inward
        return [arm_internal_term(sd, p["internal"], e) for sd in ("left", "right")]
    amp, reach, got = solve_amplitude(base, "stoop", p["tcc"], p["knee"], extra, post_fn=post)
    out, terms, s, info = axial_body(base, "stoop", amp, env, p["knee"] * env, extra(base, env), post=post(base, env))
    val = stand_validity(base, out)
    params = {"grade": g, "target_tcc_deg": p["tcc"], "amplitude_deg": amp, "reachable": reach,
              "held_tcc_deg": got, "knee_deg": p["knee"], "elbow_deg": p["elbow"], "internal_deg": p["internal"],
              "ask_s": ask_s,
              "correction_share": p["correct"], "correction_s": CORRECT_S,
              "band_held": band_of("lumbar", got), "levels_source": "module reading of mds_3_13 on tinazzi2022"}
    return out, _axial_truth("parkinsonian_posture", base, out, terms, s, info, params,
                             ("mds_3_13", "tinazzi2022", "margraf2018", "module"), val, "tcc", g, "MDS-UPDRS 3.13",
                             {"envelope": env})


def _hang_elbows(b, e):
    """Terms of the relaxed elbow flexion (HANG_ELBOW_DEG) of both hanging arms on the envelope e."""
    return [region_term(b, "distal_arm", "elbow_flexion", sd, HANG_ELBOW_DEG, e, f"elbow flexion {sd} (relaxed)")
            for sd in ("left", "right")]


def axial_posture(base, shape, target_deg, knee_deg=0.0, side="left", ramp_s=0.0, grade=None, label=None):
    """(Body, truth) of a task-force category posture (tinazzi2022) on a standing base: shape 'lumbar' or 'thoracic'
    (anterior trunk flexion, camptocormia), 'lateral' (lateral trunk flexion, Pisa syndrome, toward side), 'neck'
    (anterior neck flexion, antecollis), held at target_deg on its photograph measure; ramp_s 0 (the catalogue): a
    persistent posture held from the first knot (review M10: a rise from upright reads as voluntary), otherwise a
    rise from 0.5 s over ramp_s. In a sagittal bend the arms hang with the elbows relaxed (HANG_ELBOW_DEG)."""
    T, rate = base.T, base.rate
    env = np.ones(T) if ramp_s <= 0 else ramp_env(T, rate, 0.5, ramp_s)
    extra_fn = _hang_elbows if shape in HANG_SHAPES else None
    amp, reach, got = solve_amplitude(base, shape, target_deg, knee_deg, extra_fn, side)
    out, terms, s, info = axial_body(base, shape, amp, env, knee_deg * env, extra_fn(base, env) if extra_fn else (),
                                     side, far_arm=True)
    val = stand_validity(base, out) if shape != "neck" else validity(base, out)
    kind = {"lumbar": "lumbar", "thoracic": "thoracic", "lateral": "lateral", "neck": "neck"}[shape]
    params = {"shape": shape, "side": side, "target_deg": float(target_deg), "amplitude_deg": amp,
              "reachable": reach, "held_deg": got, "knee_deg": float(knee_deg), "ramp_s": ramp_s,
              "measure": {"lumbar": "malleolus method", "thoracic": "upper method", "lateral": "perpendicular method",
                          "neck": "perpendicular method"}[shape], "band_held": band_of(kind, got)}
    return out, _axial_truth(label or f"axial_{shape}", base, out, terms, s, info, params,
                             ("tinazzi2022", "margraf2018", "module"), val, MEASURE_OF[shape], grade, "task force",
                             {"envelope": env})


# ---------------------------------------------------------------- cervical dystonia, time course
HOLD = {"maximal": 0.75, "submaximal": G.SUBMAXIMAL}   # [module] the held share of the maximal excursion
FLUCT = 0.08                 # [module] slow fluctuation of the held posture, share of the level (0.1 to 0.4 Hz)
SPASM_RATE_HZ = {"maximal": 0.25, "submaximal": 0.05}   # [module] mobile spasms per second while held
SPASM_S = (1.5, 2.5)         # [module] duration range of a mobile spasm
JERK_RATE_HZ = 0.35          # [module] mean rate of the jerky pulls, inside the 0.1 to 0.5 Hz of the catalogue's D12
#                              (irregular sawtooth pulls, introduction of Shaikh et al. 2015, doi 10.1002/mds.26231)
JERK_GAP_S = (2.0, 10.0)     # [module] bounds of the gamma-distributed intervals between pulls (shape JERK_GAMMA_K)
JERK_GAMMA_K = 3.0
JERK_PULL_S = (0.10, 0.20)   # [module] the fast pull toward the dystonic direction
JERK_RETURN_S = (1.0, 3.0)   # [module] the slow exponential return (to 5 percent of the pull)
JERK_ROOM = (0.6, 1.0)       # [module] a pull's size, share of the room left between the held level and the maximum
JERK_HOLD = 0.75             # [module] the held share between pulls (pulls of 0.15 to 0.25 of the level): HOLD's
#                              maximal share, so that the jerky course is often maximal as its TWSTRS B score states
#                              (wave SH1d part B, review M17: at 0.65 the realized course read one B score lower)
PRESENT_FRAC = 0.25          # [module] the posture is present while it exceeds this share of the level (held_share)
MAXIMAL_FRAC = 0.7           # [module] a present posture is often maximal when its median over the present knots
#                              reaches this share of the maximal excursion (HOLD['maximal'] less the fluctuation)
DRIFT_S = (2.0, 3.0)         # [module] intermittent bouts (share under 1/2): a slow drift into the posture ...
CORRECT_BACK_S = (0.3, 0.5)  # [module] ... and a quick voluntary correction back (review M8)
BOUT_MIN_S = 0.8             # [module] the shortest intermittent bout (common.schedule's ramp argument)


def _runs(mask):
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.diff(m.astype(np.int8))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _bumps(T, rate, rng, rate_hz, dur_range, mask, min_gap_s, at_least_one=False):
    """(T,) raised-cosine bumps of height 1 inside each run of mask: the first 0.3 to 1.0 s into the run, the next
    after intervals of 1 / rate_hz times a factor drawn in [0.6, 1.4] (at least min_gap_s), each wholly inside the
    run; at_least_one: a run long enough for one bump gets one even when the interval would skip it. The onsets
    (s)."""
    t = np.arange(T) / rate
    out = np.zeros(T)
    on = []
    if rate_hz <= 0:
        return out, on
    for a, b in _runs(mask):
        x = a / rate + rng.uniform(0.3, 1.0)
        if not at_least_one:
            x += rng.uniform(0.0, 1.0 / rate_hz)
        while True:
            d = float(rng.uniform(*dur_range)) if isinstance(dur_range, tuple) else float(dur_range)
            if x + d > (b - 1) / rate:
                break
            u = np.clip((t - x) / d, 0.0, 1.0)
            out = np.maximum(out, np.where((t >= x) & (t <= x + d), 0.5 - 0.5 * np.cos(2 * np.pi * u), 0.0))
            on.append(float(x))
            x += max(min_gap_s, d, rng.uniform(0.6, 1.4) / rate_hz)
    return out, on


def jerk_events(T, rate, rng, mask):
    """((T,) in [0, 1], events): irregular sawtooth pulls inside each run of mask (catalogue D12): onsets 0.3 to 1.5 s
    into the run, then after gamma-distributed intervals (mean 1 / JERK_RATE_HZ, shape JERK_GAMMA_K, inside
    JERK_GAP_S); each a fast minimum-jerk pull over JERK_PULL_S to its size (JERK_ROOM, the size jitter) and a slow
    exponential return over JERK_RETURN_S (to 5 percent), continued past the run's end; overlapping returns are
    joined by their maximum. events:
    [onset s, pull s, return s, size]."""
    t = np.arange(T) / rate
    out = np.zeros(T)
    ev = []
    mean = 1.0 / JERK_RATE_HZ
    for a, b in _runs(mask):
        x = a / rate + rng.uniform(0.3, 1.5)
        while x < (b - 1) / rate - 0.3:
            p = float(rng.uniform(*JERK_PULL_S))
            r = float(rng.uniform(*JERK_RETURN_S))
            g = float(rng.uniform(*JERK_ROOM))
            up = minimum_jerk((t - x) / p)
            down = np.where(t > x + p, np.exp(-3.0 * (t - x - p) / r), 1.0)
            out = np.maximum(out, g * up * down)
            ev.append([float(x), p, r, g])
            x += float(np.clip(rng.gamma(JERK_GAMMA_K, mean / JERK_GAMMA_K), *JERK_GAP_S))
    return out, ev                    # a return outlasting its run fades with the bout (mobile_envelope), never cut


def _mj_inverse(q):
    """u in [0, 1] with motions.minimum_jerk(u) = q (a fine grid)."""
    u = np.linspace(0.0, 1.0, 2001)
    return float(np.interp(q, minimum_jerk(u), u))


def drift_bouts(t, intervals, rng, t_end, threshold):
    """((T,) in [0, 1], [drift s, correction s] per bout): asymmetric bouts on the intervals of common.schedule: a slow
    minimum-jerk drift in (DRIFT_S, shortened so that the bout reaches its full level inside the interval) and a quick
    minimum-jerk correction back (CORRECT_BACK_S), placed so that the bout exceeds threshold (the share at which the
    posture counts as present) over the interval itself; a bout touching the clip's start or end has no ramp there."""
    e = np.zeros_like(t)
    shapes = []
    uq, vq = _mj_inverse(threshold), _mj_inverse(1.0 - threshold)
    for a, b in intervals:
        c = float(rng.uniform(*CORRECT_BACK_S))
        d = float(rng.uniform(*DRIFT_S))
        d = min(d, max((b - a - vq * c) / (1.0 - uq), 0.2))
        up = np.ones_like(t) if a <= 0 else minimum_jerk((t - (a - uq * d)) / d)
        dn = np.ones_like(t) if b >= t_end else 1.0 - minimum_jerk((t - (b - vq * c)) / c)
        e = np.maximum(e, np.minimum(up, dn))
        shapes.append([d, c])
    return e, shapes


def mobile_envelope(T, rate, rng, share=1.0, excursion="maximal", n_bouts=None, ramp_s=1.0, fluct=FLUCT,
                    spasms=True, jerks=False):
    """(env (T,), info): the time course of a cervical dystonic posture as a share of its maximal excursion
    (module docstring): bouts holding share of the time (common.schedule; under half the time, drift_bouts: a slow
    drift in and a quick correction back, 1 to 3 bouts; otherwise raised-cosine ramps of ramp_s), held at
    HOLD[excursion] (JERK_HOLD with jerks) with a slow fluctuation of +-fluct; mobile spasms (SPASM_RATE_HZ, SPASM_S)
    take the posture from the held level to the maximal excursion, x + (1 - x) s, and, with jerks, the sawtooth pulls
    of jerk_events take it a share of the room left, x + (1 - x) j: the course never exceeds the maximal excursion
    and never stalls flat on it (review, minor 1)."""
    dur = (T - 1) / rate
    t = np.arange(T) / rate
    hold = JERK_HOLD if jerks else HOLD[excursion]
    asym = 0.0 < share < 0.5
    if asym:
        if n_bouts is None:
            n_bouts = int(max(1, min(3, (share * dur) // 0.9)))
        iv = schedule(dur, share, n_bouts, BOUT_MIN_S, rng, rate)
        bout, shapes = drift_bouts(t, iv, rng, dur, min(1.0, PRESENT_FRAC / hold))
    else:
        if n_bouts is None:
            n_bouts = 1 if (share >= 0.9 or share * dur < 4 * ramp_s) else 2
        iv = schedule(dur, share, n_bouts, ramp_s, rng, rate)
        bout, shapes = envelope(t, iv, ramp_s, dur), None
    x = np.clip(hold + fluct * smooth_noise(T, rate, rng), 0.0, 1.0)
    held = bout >= 1.0 - 1e-9
    sp, sp_on = _bumps(T, rate, rng, SPASM_RATE_HZ[excursion] if spasms else 0.0, SPASM_S, held, 1.0,
                       at_least_one=excursion == "maximal")
    x = x + (1.0 - x) * sp
    jk, jk_ev = jerk_events(T, rate, rng, held) if jerks else (np.zeros(T), [])
    x = x + (1.0 - x) * jk
    env = np.clip(bout * x, 0.0, 1.0)
    if excursion == "maximal" and env.max() > 0 and not sp_on:   # no spasm: scaled to reach the maximal excursion
        env = env / env.max()
    return env, {"intervals_s": [list(v) for v in iv], "share": realized_share(iv, dur), "hold": hold,
                 "fluct": fluct, "spasm_onsets_s": sp_on, "jerk_events": jk_ev, "n_bouts": int(n_bouts),
                 "ramp_s": ramp_s, "excursion": excursion, "drift_correction_s": shapes}


def joint_deviation(base, out, joint, segment=True):
    """(T,) deg: the deviation of a segment's global rotation (segment True) or of a joint's local rotation from the
    level-0 body (body.deviation)."""
    if segment:
        _, Gb = base.fk()
        _, Go = out.fk()
        return deviation(Gb[:, joint], Go[:, joint])[0].numpy()
    Lb, Lo = base.local_R()[:, joint], out.local_R()[:, joint]
    return rotation_angle_deg(Lb.transpose(-1, -2) @ Lo).numpy()


def held_share(dev, level, frac=PRESENT_FRAC):
    """Share of knots whose deviation exceeds frac of the level: the time with the posture present."""
    return float((np.asarray(dev) > frac * level).mean())


def twstrs_duration_of(dev, env, frac=PRESENT_FRAC, maximal=MAXIMAL_FRAC):
    """{'score', 'present_share', 'median_share', 'often_maximal', 'maximal_deg'}: the TWSTRS duration score B read
    on a realized course (wave SH1d part B, review M17): the maximal excursion the deviation dev (T,) deg reaches at
    full envelope (dev.max() / env.max(): the course's own scale), the share of knots present (above frac of it, the
    TWSTRS_DURATION_PCT band 1 to 4) and 'often maximal' when the median over the present knots reaches maximal of
    it: the score is the band, plus 1 when often maximal (grades.twstrs_duration's reading)."""
    dev, env = np.asarray(dev, dtype=np.float64), np.asarray(env, dtype=np.float64)
    if env.max() <= 0 or dev.max() <= 0:
        return {"score": 0, "present_share": 0.0, "median_share": 0.0, "often_maximal": False, "maximal_deg": 0.0}
    mx = float(dev.max() / env.max())
    present = dev > frac * mx
    share = float(present.mean())
    band = 1 + sum(share >= hi / 100.0 for _, hi in G.TWSTRS_DURATION_PCT[:-1])
    med = float(np.median(dev[present]) / mx) if present.any() else 0.0
    often = med >= maximal
    return {"score": int(band + (1 if often else 0)) if present.any() else 0, "present_share": share,
            "median_share": med, "often_maximal": bool(often), "maximal_deg": mx,
            "rule": "band of the present share (above PRESENT_FRAC of the maximal excursion) plus 1 when the median "
                    "over the present knots reaches MAXIMAL_FRAC of it (module reading of the TWSTRS duration factor)"}


def cervical_course(base, parts, duration_grade=None, share=None, excursion="maximal", jerks=False, seed=0,
                    spasms=True, label="cervical dystonia"):
    """(Body, truth) of a time-varying cervical dystonia (module docstring): parts [(region, direction, side, level
    deg)] of dystonia.py's neck and shoulder regions, all on one time course (mobile_envelope) whose share and
    excursion come from the TWSTRS duration score (grades.twstrs_duration) or from share."""
    rng = as_rng(seed)
    if duration_grade is not None:
        share, excursion = G.twstrs_duration(duration_grade, excursion)
    share = 1.0 if share is None else float(share)
    env, cinfo = mobile_envelope(base.T, base.rate, rng, share, excursion, jerks=jerks, spasms=spasms)
    terms = [region_term(base, r, d, sd, lv, env, f"{r} {d}", table=DYST) for r, d, sd, lv in parts]
    collars = tuple(sorted({ARM[sd][0] for r, _, sd, _ in parts if r == "shoulder_elevation"}))
    R, s = fit_terms(base.local_R(), terms, table=DYST, hang=collars)   # the arm hangs under a raised collar (SH2)
    rows = sorted({j for t in terms for j in t.chain} | {c + 3 for c in collars})
    out = base.with_rotations(R, rows)
    out = replace(out, contact=contact_labels(out))
    neck = [lv for r, _, _, lv in parts if r == "neck"]
    seg = 15 if neck else terms[0].chain[-1]              # the head, or the collar of a shoulder-only posture
    dev = joint_deviation(base, out, seg)
    lvl = sum(neck) or max(lv for _, _, _, lv in parts)
    params = {"parts": [{"region": r, "direction": d, "side": sd, "level_deg": float(lv)} for r, d, sd, lv in parts],
              "duration_grade": duration_grade, "course": cinfo, "limit_scale_min": float(s.min()),
              "measured_segment": int(seg), "realized_max_deg": float(dev.max()),
              "realized_share": held_share(dev, lvl), "arm_hangs": arm_hang_report(base, out, collars),
              "twstrs_duration_realized": twstrs_duration_of(dev, env),
              "terms": [t.record() for t in terms], "limits_table": LX.record(DYST)}
    return out, truth_record("cervical_course", seed if isinstance(seed, int) else None, duration_grade, "TWSTRS",
                             params, ("twstrs", "twstrs_full", "albanese2013") + (("shaikh2015",) if jerks else ())
                             + ("limits_ext", "module"), LX.check(DYST, base, out),
                             arrays={"envelope": env, "deviation_deg": dev, "limit_scale": s})


def arm_hang_report(base, out, collars):
    """{side: {'humerus_turn_deg_max', 'shoulder_rise_cm_max'}} of the arms under the raised collars (13, 14): the
    largest angle between the humerus (shoulder to elbow joint) of out and of base, and the shoulder joint's largest
    rise (cm) against base."""
    if not collars:
        return None
    Jb, Jo = base.joints().numpy(), out.joints().numpy()
    rep = {}
    for c in collars:
        sd = "left" if c == 13 else "right"
        s, e = c + 3, c + 5
        ub, uo = Jb[:, e] - Jb[:, s], Jo[:, e] - Jo[:, s]
        cos = (ub * uo).sum(-1) / np.linalg.norm(ub, axis=-1) / np.linalg.norm(uo, axis=-1)
        rep[sd] = {"humerus_turn_deg_max": float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))).max()),
                   "shoulder_rise_cm_max": float(100 * (Jo[:, s, 1] - Jb[:, s, 1]).max())}
    return rep


SHIFT_DEG = {"lateral": (6.0, 19.0), "anterior": (8.0, 25.0), "posterior": (6.0, 24.0)}
#   [module] the turns of spine3 (the upper thoracic segment) and of the neck joint toward the shift; the head joint
#   turns back by their sum. The neck-to-head lever is 8 cm only: with spine3 the lever is about 22 cm and the head
#   moves 4.4 to 6 cm (review M9: the neck alone read as no shift)


def head_shift(base, kind, side="left", ramp_s=1.5, seed=0):
    """(Body, truth) of TWSTRS A4 lateral shift (toward side) or A5 sagittal shift ('anterior' protraction,
    'posterior' retraction), present (score 1): spine3 and the neck joint turned one way (SHIFT_DEG) and the head
    joint the other by their sum, so that the head moves without tilting; held from a ramp at 0.5 s with the
    fluctuation of mobile_envelope; the arms keep hanging (hang_arms). The realized head translation (cm) and tilt
    (deg) are read back."""
    rng = as_rng(seed)
    a9, a12 = SHIFT_DEG[kind]
    env0 = ramp_env(base.T, base.rate, 0.5, ramp_s)
    env = np.clip(env0 * (1.0 + FLUCT * smooth_noise(base.T, base.rate, rng)) / (1.0 + FLUCT), 0.0, 1.0)
    if kind == "lateral":
        ax, sg = "z", (-1.0 if side == "left" else 1.0)
    else:
        ax, sg = "x", (1.0 if kind == "anterior" else -1.0)
    terms = [joint_term(9, ax, sg * a9, env, "child", "upper thoracic turn"),
             joint_term(12, ax, sg * a12, env, "child", "neck turn"),
             joint_term(15, ax, -sg * (a9 + a12), env, "child", "head counter-turn")]
    R, s = fit_terms(base.local_R(), terms, table=DYST)
    R, hang = hang_arms(base, R, table=DYST)
    out = base.with_rotations(R, [9, 12, 13, 14, 15, 16, 17])
    m = measure_head_shift(base, out)
    params = {"kind": kind, "side": side if kind == "lateral" else None, "turn_deg": {"spine3": a9, "neck": a12,
              "head": -(a9 + a12)}, "ramp_s": ramp_s, "fluct": FLUCT, "limit_scale_min": float(s.min()),
              "arms_hang": hang, **m, "terms": [t.record() for t in terms], "limits_table": LX.record(DYST)}
    item = "TWSTRS A4 lateral shift" if kind == "lateral" else "TWSTRS A5 sagittal shift"
    val = LX.check(DYST, base, out)
    val["arm_speed"] = arm_speed_report(out)
    return out, truth_record("head_shift", seed if isinstance(seed, int) else None, 1, item, params,
                             ("twstrs_full", "limits_ext", "module"), val,
                             arrays={"envelope": env})


ARM_SPEED_MAX_DEG_S = 120.0  # [module] the hanging arms' collar and shoulder rows turn at most this fast in a held
#                              posture (wave SH1d part B, review M4: a twist switch flicked the forearm in two knots)


def arm_speed_report(out, rows=(13, 14, 16, 17), bound=ARM_SPEED_MAX_DEG_S):
    """{'deg_s_max', 'row', 't_s', 'bound_deg_s', 'ok'}: the fastest turn of the collar and shoulder rows from one
    knot to the next (deg/s)."""
    R = out.local_R()
    best = (0.0, None, None)
    for j in rows:
        d = rotation_angle_deg(R[:-1, j].transpose(-1, -2) @ R[1:, j]).numpy() * out.rate
        k = int(np.argmax(d))
        if d[k] > best[0]:
            best = (float(d[k]), j, k / out.rate)
    return {"deg_s_max": best[0], "row": best[1], "t_s": best[2], "bound_deg_s": bound, "ok": bool(best[0] <= bound)}


def measure_head_shift(base, out):
    """{'shift_cm' (the largest head-joint translation from the base in the heading frame, lateral and forward
    components at that knot), 'head_tilt_deg' (the head segment's largest deviation)}."""
    Jb, Jo = base.joints().numpy(), out.joints().numpy()
    d = _heading_frame((Jo[:, 15] - Jb[:, 15])[:, None], out.local_R()[:, 0])[:, 0]
    k = int(np.argmax(np.linalg.norm(d, axis=-1)))
    return {"shift_cm": float(100 * np.linalg.norm(d[k])), "shift_lateral_cm": float(100 * d[k, 0]),
            "shift_forward_cm": float(100 * d[k, 2]), "head_tilt_deg": float(joint_deviation(base, out, 15).max())}


# ---------------------------------------------------------------- arm reach
ARM = {"left": (13, 16, 18, 20), "right": (14, 17, 19, 21)}
ELBOW_SIGN = {"left": -1.0, "right": 1.0}           # elbow flexion about the elbow's local y (base.py)
PALM_SHARE = 0.5             # [module] the palm centre: halfway from the wrist joint to the middle knuckle


def palm_offset(side, share=PALM_SHARE):
    """(3,) the rest offset from the wrist joint of the point share of the way to the middle knuckle (hands._rest
    geometry; PALM_SHARE the palm centre)."""
    J52, _, _ = H._rest()
    w, m = H.WRIST[side], H.hand_joint(side, "middle", 0)
    return torch.as_tensor(share * (J52[m] - J52[w]), dtype=F64)


def _rot_about(axis, ang):
    """(T, 3, 3) rotations about unit axes (T, 3) or (3,) by angles (T,) rad."""
    a = torch.as_tensor(ang, dtype=F64)
    u = torch.as_tensor(axis, dtype=F64)
    u = u.expand(a.shape[0], 3) if u.dim() == 1 else u
    return axis_angle_to_matrix(u * a[:, None])


def end_offset(side, end):
    """(3,) the rest offset from the wrist joint of an arm solve's end point, or None for the wrist: end 'wrist',
    'palm' (PALM_SHARE), a share of the way to the middle knuckle (float), or a (3,) offset in the wrist's rest frame
    (a point of the posed hand, geste_antagoniste's finger pads)."""
    if isinstance(end, str):
        if end not in ("wrist", "palm"):
            raise ValueError(f"arm end {end!r}")
        return None if end == "wrist" else palm_offset(side, PALM_SHARE)
    if np.ndim(end) == 0:
        return palm_offset(side, float(end))
    return torch.as_tensor(np.asarray(end, dtype=np.float64), dtype=F64).reshape(3)


def _arm_once(R, transl, rest, side, target, pole, end):
    c, s, e, w = ARM[side]
    with torch.no_grad():
        J, Gl = fk(R, transl, rest)
    S = J[:, s]
    p = rest[e] - rest[s]
    q = (rest[w] - rest[e])[None].expand(R.shape[0], 3)
    off = end_offset(side, end)
    if off is not None:
        q = q + torch.einsum("tij,j->ti", R[:, w], off)
    L = target - S
    d = L.norm(dim=-1)
    pn, qn = p.norm(), q.norm(dim=-1)
    d = torch.minimum(torch.maximum(d, (pn - qn).abs() + 1e-6), pn + qn - 1e-6)
    c1 = p[0] * q[:, 0] + p[2] * q[:, 2]
    c2 = p[0] * q[:, 2] - p[2] * q[:, 0]
    K = (d ** 2 - pn ** 2 - qn ** 2 - 2 * p[1] * q[:, 1]) / 2
    rho = torch.sqrt(c1 ** 2 + c2 ** 2)
    delta = torch.atan2(c2, c1)
    a = torch.acos((K / rho).clamp(-1.0, 1.0))
    cand = torch.stack([delta + a, delta - a], -1)
    cand = torch.atan2(torch.sin(cand), torch.cos(cand))
    sg = ELBOW_SIGN[side]
    score = torch.where(cand * sg >= -1e-9, cand.abs(), cand.abs() + 10.0)
    el = cand.gather(-1, score.argmin(-1, keepdim=True))[:, 0]
    z = torch.zeros_like(el)
    Ry = axis_angle_to_matrix(torch.stack([z, el, z], -1))
    v = p[None] + torch.einsum("tij,tj->ti", Ry, q)
    u = L / L.norm(dim=-1, keepdim=True)
    Gs = Gl[:, s]
    G1 = rmin(torch.einsum("tij,tj->ti", Gs, v / v.norm(dim=-1, keepdim=True)), u) @ Gs
    kp = torch.einsum("tij,j->ti", G1, p)
    pp = torch.as_tensor(pole, dtype=F64).expand_as(kp)
    kp = kp - (kp * u).sum(-1, keepdim=True) * u
    pp = pp - (pp * u).sum(-1, keepdim=True) * u
    kp = kp / kp.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    pp = pp / pp.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    gam = torch.atan2((torch.cross(kp, pp, dim=-1) * u).sum(-1), (kp * pp).sum(-1))
    Gn = _rot_about(u, gam) @ G1
    Rn = R.clone()
    Rn[:, s] = Gl[:, c].transpose(-1, -2) @ Gn
    Rn[:, e] = Ry
    return Rn


HANG_TWIST_MAX_DEG = 60.0    # [module] the largest internal rotation of the humerus the hanging solve may add
HANG_BISECT = 14
HANG_TWIST_SMOOTH_S = 0.3    # [module] the hanging twist's moving maximum then moving mean over this window (wave
#                              SH1d part B, review M4: the twist chosen knot by knot switched for 2 to 3 knots and
#                              flicked the forearm 45 mm in post_E7_cd_shift_lateral); a knot where the smoothed twist
#                              leaves the table keeps its own; an arm that no twist holds at some knot takes none


def _unit_t(v):
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _slerp(a, b, s):
    """(T, 3) unit vectors s (T,) of the way from a to b (T, 3) along the great circle."""
    c = (a * b).sum(-1, keepdim=True).clamp(-1.0, 1.0)
    om = torch.acos(c)
    so = torch.sin(om)
    s = torch.as_tensor(s, dtype=F64)[:, None]
    small = so < 1e-9
    w0 = torch.where(small, 1.0 - s, torch.sin((1.0 - s) * om) / so.clamp_min(1e-12))
    w1 = torch.where(small, s, torch.sin(s * om) / so.clamp_min(1e-12))
    return _unit_t(w0 * a + w1 * b)


def hang_arms(base, R, sides=("left", "right"), target=None, collar=True, table=HEALTHY):
    """(R, info): R with the shoulder of each side re-aimed so that the humerus keeps the base's world direction at each
    knot (or target {side: (T, 3)}): the arm hangs as in the level-0 stand whatever the trunk does. In the collar's
    frame the humerus is turned by the minimal rotation onto that direction, then about itself by the smallest internal
    rotation (0 to HANG_TWIST_MAX_DEG, 1 deg steps) that keeps the shoulder inside the joint-limit table (table:
    'healthy' or 'dystonia', limits_ext: there the shoulder's adduction bound follows the collar's rise over the base at
    each knot); where no twist does, it is turned without twist the largest share of the way that the table allows
    (bisection) and, with collar, the collar then turns the whole arm the rest of the way (the minimal rotation about
    the collar joint, the largest share the collar's table allows: a lateral bend toward the other side drops that
    shoulder); collar False where the collar carries the sign (a shoulder elevation). info: per side the smallest share
    reached and the largest twist (deg). The elbows, wrists and hands keep their local rotations."""
    rest = base.rest()
    T = R.shape[0]
    _, Gb = base.fk()
    Rn = R.clone()
    info = {}
    for sd in sides:
        c, s, e, _ = ARM[sd]
        with torch.no_grad():
            _, G = fk(Rn, base.transl, rest)
        hx = _unit_t(rest[e] - rest[s])
        h0 = torch.einsum("tij,j->ti", Rn[:, s], hx)                       # the humerus in the collar's frame
        tw = torch.einsum("tij,j->ti", Gb[:, s], hx) if target is None else torch.as_tensor(target[sd], dtype=F64)
        tg = torch.einsum("tji,tj->ti", G[:, c], _unit_t(tw))
        cball = np.degrees(matrix_to_axis_angle(base.local_R()).numpy())
        cb = cball[:, s]
        klo, khi = LX.knot_bounds(table, np.degrees(matrix_to_axis_angle(Rn).numpy()), cball)   # R's collar rise
        lo, hi = np.minimum(klo[:, s], cb) - 1e-6, np.maximum(khi[:, s], cb) + 1e-6
        taus = np.radians(np.arange(0.0, HANG_TWIST_MAX_DEG + 1e-9, 1.0)) * (1.0 if sd == "left" else -1.0)

        def cand(sv, taus=taus, R_s=Rn[:, s], h0=h0, tg=tg, lo=lo, hi=hi):
            d = _slerp(h0, tg, sv)
            Rm = rmin(h0, d) @ R_s
            ok = []
            for ta in taus:
                Rt = _rot_about(d, torch.full((T,), float(ta), dtype=F64)) @ Rm
                cc = np.degrees(matrix_to_axis_angle(Rt).numpy())
                ok.append(((cc >= lo) & (cc <= hi)).all(-1))
            return np.stack(ok, 1), d, Rm
        ok, d, Rm = cand(np.ones(T))
        sv = np.ones(T)
        feas = ok.any(1)
        if not feas.all():                       # a twist cannot hold every knot: none at all, so that the arm never
            feas = ok[:, 0]                      # switches between the twisted branch and the share path below
        idx = np.where(feas, np.argmax(ok, 1), 0)
        n = max(1, int(round(HANG_TWIST_SMOOTH_S * base.rate)))
        if n > 1 and idx.any():
            pad = np.pad(idx.astype(np.float64), (n // 2, n - 1 - n // 2), mode="edge")
            mx = np.lib.stride_tricks.sliding_window_view(pad, n).max(1)
            pad2 = np.pad(mx, (n // 2, n - 1 - n // 2), mode="edge")
            sm = np.ceil(np.convolve(pad2, np.ones(n) / n, mode="valid") - 1e-9).astype(int).clip(0, len(taus) - 1)
            use = feas & ok[np.arange(T), sm]
            idx = np.where(use, sm, idx)
        tau = np.where(feas, taus[idx], 0.0)
        if not feas.all():                       # unreachable: no twist, the largest share of the way (bisection)
            a, b_ = np.zeros(T), np.ones(T)
            for _ in range(HANG_BISECT):
                m = 0.5 * (a + b_)
                f = cand(m, taus[:1])[0][:, 0]
                a, b_ = np.where(f, m, a), np.where(f, b_, m)
            sv = np.where(feas, 1.0, a)
            _, d, Rm = cand(sv, taus[:1])
        Rn[:, s] = _rot_about(d, torch.as_tensor(tau, dtype=F64)) @ Rm
        info[sd] = {"share_min": float(sv.min()), "twist_max_deg": float(np.degrees(np.abs(tau)).max())}
        if collar and not feas.all():            # the rest by the collar (the shoulder at its table edge drops)
            with torch.no_grad():
                J, G = fk(Rn, base.transl, rest)
            hw = _unit_t(J[:, e] - J[:, s])
            tw_u = _unit_t(tw)
            Q = matrix_to_axis_angle(rmin(hw, tw_u))
            cbc = np.degrees(matrix_to_axis_angle(base.local_R()[:, c]).numpy())
            loc, hic = np.minimum(klo[:, c], cbc) - 1e-6, np.maximum(khi[:, c], cbc) + 1e-6
            Gp = G[:, 9]

            def coll(q):
                Qq = axis_angle_to_matrix(Q * torch.as_tensor(q, dtype=F64)[:, None])
                return Gp.transpose(-1, -2) @ Qq @ Gp @ Rn[:, c]

            def inside(q):
                cc = np.degrees(matrix_to_axis_angle(coll(q)).numpy())
                return ((cc >= loc) & (cc <= hic)).all(-1)
            a, b_ = np.zeros(T), np.ones(T)
            full = inside(np.ones(T))
            for _ in range(HANG_BISECT):
                m = 0.5 * (a + b_)
                f = inside(m)
                a, b_ = np.where(f, m, a), np.where(f, b_, m)
            q = np.where(full, 1.0, a) * (~feas)
            Rn[:, c] = coll(q)
            info[sd]["collar_share_max"] = float(q.max())
            with torch.no_grad():
                J, _ = fk(Rn, base.transl, rest)
            hw = _unit_t(J[:, e] - J[:, s])
            info[sd]["residual_deg_max"] = float(torch.rad2deg(torch.acos((hw * tw_u).sum(-1).clamp(-1, 1))).max())
    return Rn, info


def arm_internal_term(side, deg, env, joint=None):
    """A Term turning the humerus inward about its own axis (internal rotation, deg >= 0) at the shoulder (joint 16 or
    17, child frame; the bone axis of body.bone_axis), positive inward on both sides; joint 18 or 19 with the
    forearm's axis gives pronation the same way."""
    from .body import bone_axis
    j = joint if joint is not None else ARM[side][1]
    ax = bone_axis(j, j + 2)
    return Term((j,), ax, 1.0 if side == "left" else -1.0, "child", deg, env, (1.0,),
                f"{'internal rotation' if j in (16, 17) else 'pronation'} {side}")


def _palm_twist(R, transl, rest, side, palm_dir):
    """R with the forearm twisted about its own axis (half at the elbow, half at the wrist) so that the palmar normal
    points as close to palm_dir (T, 3) or (3,) as the twist allows."""
    _, s, e, w = ARM[side]
    with torch.no_grad():
        J, Gl = fk(R, transl, rest)
    n = torch.einsum("tij,j->ti", Gl[:, w], torch.as_tensor(H.hand_frame()[side]["palmar"], dtype=F64))
    ax = J[:, w] - J[:, e]
    ax = ax / ax.norm(dim=-1, keepdim=True)
    dd = torch.as_tensor(palm_dir, dtype=F64).expand_as(n)
    n = n - (n * ax).sum(-1, keepdim=True) * ax
    dd = dd - (dd * ax).sum(-1, keepdim=True) * ax
    phi = torch.atan2((torch.cross(n, dd, dim=-1) * ax).sum(-1), (n * dd).sum(-1))
    b = (rest[w] - rest[e]) / (rest[w] - rest[e]).norm()
    Rn = R.clone()
    Rn[:, e] = R[:, e] @ _rot_about(b, 0.5 * phi)
    Rn[:, w] = _rot_about(b, 0.5 * phi) @ R[:, w]
    return Rn


def arm_ik(body, R, side, target, pole, palm_dir=None, end="palm", passes=3):
    """(T, 22, 3, 3): R with the shoulder and elbow of side solved so that the palm centre (end 'palm') or the wrist
    joint lies on target (T, 3), the elbow toward pole (3,) or (T, 3), and with palm_dir the forearm twisted so that
    the palm faces it (module docstring); a few passes, since the twist moves the palm centre slightly."""
    rest = body.rest()
    tr = body.transl
    tg = torch.as_tensor(target, dtype=F64)
    for _ in range(passes):
        R = _arm_once(R, tr, rest, side, tg, pole, end)
        if palm_dir is not None:
            R = _palm_twist(R, tr, rest, side, palm_dir)
    return _arm_once(R, tr, rest, side, tg, pole, end)


POLES = ((0.6, -1.0, -0.2), (0.3, -1.0, 0.3), (1.0, -0.6, 0.0), (0.0, -1.0, 0.5), (0.6, -1.0, -0.6),
         (0.2, -1.0, -0.4), (1.0, -1.0, 0.4))      # [module] elbow directions tried in turn (x toward the side)


COLLARS = ((0.0, 0.0), (15.0, 0.0), (15.0, 10.0), (25.0, 10.0), (25.0, 20.0), (10.0, 20.0), (20.0, 15.0),
           (30.0, 15.0), (20.0, 25.0))
#                            [module] collar (forward deg about y, elevation deg about z) tried with each elbow pole


def arm_ik_in_table(body, R, side, target, palm_dir=None, knots=None, poles=POLES, collars=COLLARS, end="palm",
                    table=HEALTHY, blend=None):
    """(R, pole, collar, worst excess deg): arm_ik after a collar offset (forward and elevation, signs mirrored for
    the right) and with an elbow direction (x mirrored for the right arm), the first pair of collars x poles whose
    result keeps the arm rows inside [min(lo, base), max(hi, base)] of table (limits_ext) at the knots (default every
    knot), else the pair with the smallest excess; blend (T,): the check reads the arm rows blended from R toward the
    result by these shares (blend_rows: the path in and out of the posture, not the posture alone)."""
    rows = list(ARM[side])
    cb = np.degrees(matrix_to_axis_angle(R).numpy())
    sel = slice(None) if knots is None else np.asarray(knots)
    best = None
    sg = 1.0 if side == "left" else -1.0
    one = np.ones(R.shape[0])
    for fw, el in collars:
        Rc = apply_terms(R, [joint_term(ARM[side][0], "y", -sg * fw, one, "parent"),
                             joint_term(ARM[side][0], "z", sg * el, one, "parent")])
        for pl in poles:
            pole = torch.tensor([sg * pl[0], pl[1], pl[2]], dtype=F64)
            Rk = arm_ik(body, Rc, side, target, pole, palm_dir, end)
            c = np.degrees(matrix_to_axis_angle(Rk if blend is None else blend_rows(R, Rk, blend, rows)).numpy())
            ex = float(LX.excess(table, c, cb, rows)[sel].max())
            if best is None or ex < best[3]:
                best = (Rk, pole, (fw, el), ex)
            if ex <= 1e-6:
                return best
    return best


PROJECT_MAX_DEG = 0.5        # [module] the largest excess project_rows removes (larger ones are left and reported)


def project_rows(R, rows, base_R, table=HEALTHY, tol=0.0):
    """(R, largest component moved deg): the axis-angle components of the rows clamped into [min(lo, base), max(hi,
    base)] of table (limits_ext, at R's collar rises over base_R) where they leave it by more than tol and at most
    PROJECT_MAX_DEG (a blend or a solve that grazes the table's edge)."""
    klo, khi = LX.knot_bounds(table, np.degrees(matrix_to_axis_angle(R).numpy()),
                              np.degrees(matrix_to_axis_angle(base_R).numpy()))
    rows = [r for r in rows if r > 0]
    Rn = R.clone()
    moved = 0.0
    for j in rows:
        aa = np.degrees(matrix_to_axis_angle(R[:, j]).numpy())
        cb = np.degrees(matrix_to_axis_angle(base_R[:, j]).numpy())
        lo, hi = np.minimum(klo[:, j], cb), np.maximum(khi[:, j], cb)
        ex = np.maximum(lo - aa, aa - hi).max(-1)
        fix = (ex > tol) & (ex <= PROJECT_MAX_DEG)
        if fix.any():
            cl = np.clip(aa, lo, hi)
            Rn[fix, j] = axis_angle_to_matrix(torch.from_numpy(np.radians(cl[fix])))
            moved = max(moved, float(ex[fix].max()))
    return Rn, moved


def blend_rows(R0, R1, w, rows):
    """R0 with the rows turned toward R1 along the geodesic by shares w (T,)."""
    Rn = R0.clone()
    W = torch.as_tensor(np.asarray(w, dtype=np.float64))
    for j in rows:
        D = matrix_to_axis_angle(R0[:, j].transpose(-1, -2) @ R1[:, j])
        Rn[:, j] = R0[:, j] @ axis_angle_to_matrix(W[:, None] * D)
    return Rn


def palm_points(body, side, knots=None, share=PALM_SHARE):
    """(T, 3) the hand point share of the way from the wrist to the middle knuckle (default the palm centre): the
    wrist joint plus the rest offset turned by the wrist's global rotation."""
    J, Gl = body.fk()
    p = J[:, H.WRIST[side]] + torch.einsum("tij,j->ti", Gl[:, H.WRIST[side]], palm_offset(side, share))
    return p.numpy() if knots is None else p.numpy()[list(knots)]


# ---------------------------------------------------------------- the geste antagoniste
TRICK = {0: 1.0, 1: 0.5, 2: 0.1}   # TWSTRS C score: share of the posture undone while the hand is on the face;
#                                   complete relief (0) a source anchor, partial 0.5 and little 0.1 [module]
TRICK_TIMES = {"reach_on_s": 5.0, "reach_s": 1.2, "release_s": 11.0, "back_s": 1.2, "latency_s": 0.3,
               "relief_s": 1.2, "return_s": 1.5}            # [module]
#   reach_s and back_s 1.2 s (1.0 before the open body): on Anny's arm the reach to the cheek and the return turned the
#   upper arm at 250.8 to 264.6 deg/s, over the page's 250 deg/s bound of the shoulder (PLAN M9; the skeleton page's
#   continuity check, geste C0 to C2 of the wave 7 page); the turn's peak speed scales with the reach's inverse
#   duration (open-body port, wave 8 second part, D166)
GESTE_PADS = ("index", "middle")   # [module] the fingers whose pads rest on the cheek
GESTE_GAP_M = 0.003          # [module] the least distance of the pads' fingers (their distal phalanges) to the skin of
#                              the head and neck while the hand rests on the cheek, a light touch, solved by secant
#                              steps (review of the SH2 wave: at 1 cm for the whole hand it hovered before the throat)
GESTE_GAP_TOL_M = 0.001      # [module] the secant's tolerance on that distance (the pads 2 to 4 mm from the skin)
GESTE_GAP_ITERS = 6
GESTE_CONTACT_M = (0.03, 0.015)
#                            [module] the contact point on the cheek of the hand's side (cheek_contact_vertex): 3 cm in
#                            front of the head joint and 1.5 cm above the chin landmark (jaw_vertex), on the rest mesh:
#                            the lower cheek over the body of the mandible, between the chin and the angle of the jaw
GESTE_FINGERS_DEG = (0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0)
#                            [module] the hand frames searched (geste_frames): the hand's long axis (wrist to middle
#                            knuckle) in the skin's tangent plane at the contact, along the horizontal tangent pointing
#                            back (toward the ear) raised by this angle (90: up along the cheek)
GESTE_TILTS_DEG = (30.0, 45.0, 60.0, 75.0)
#                            [module] then tilted into the face about the hand's transverse axis by this angle: the
#                            finger pads touch, the palm and the wrist stand off the cheek (at 0 the hand lies flat on
#                            it, the palm on the skin)
GESTE_ROLLS_DEG = (-60.0, -30.0, 0.0, 30.0, 60.0, 90.0)
#                            [module] and turned about its long axis by this angle, the palm toward the floor for a
#                            positive one (review of 2026-11-03: the hand frame of the first SH1b build, raised 0,
#                            tilted 45 and turned 30 deg, bent the wrist back and toward the thumb past
#                            WRIST_RANGE_DEG; no arm of that frame keeps the wrist inside it)
WRIST_RANGE_DEG = {"palmar": 90.0, "dorsal": 70.0, "radial": 25.0, "ulnar": 55.0}
#                            SOURCES jarm2022_wrist: the wrist's flexion and deviation (wrist_bend_deg); the joint-limit
#                            table bounds each axis-angle component of the wrist apart, and a wrist inside it on every
#                            component can still be bent beyond these (review of 2026-11-03)
GESTE_WRIST_MARGIN_DEG = 5.0  # [module] the geste's wrist kept this far inside WRIST_RANGE_DEG while the hand rests
GESTE_LOAD_STEP = 0.05       # [module] arms whose wrist loads (wrist_load) fall in one step of this size are ranked by
#                              their elbow height
GESTE_SCREEN_GAP_M = 0.015   # [module] the pads' offset along the skin's normal used by the frame screen
#                              (geste_frames); the clip's own offset is solved afterwards (params 'pad_gap_m')
GESTE_SCREEN_STEP = 20       # [module] knots between two samples of the frame screen (0.2 s)
GESTE_ELBOW_DEG = tuple(float(x) for x in range(0, 360, 5))
#                            [module] the elbow's places sampled around its circle (the shoulder-to-wrist line) by the
#                            frame screen
GESTE_NECK_CLEAR_M = 0.01    # [module] the wrist beside the neck: its lateral offset in the neck joint's frame at least
#                              the neck's half width (throat_ids) plus this
GESTE_TRUNK_GAP_M = 0.01     # [module] the hand's and the forearm's least distance to the trunk's skin over the reach
GESTE_TRUNK_POINTS = 9       # [module] points along the forearm's axis read by geste_arms' trunk stand-in
GESTE_SCREEN_TRUNK_KNOTS = 6  # [module] knots of the frame screen at which the trunk stand-in is read
GESTE_SCREEN_TRUNK_STRIDE = 4  # [module] the frame screen reads every this-th vertex of the trunk
GESTE_FRAME_TRIES = 6        # [module] the ranked hand frames tried in turn until one has an arm inside the table with
#                              the wrist inside its range
GESTE_SHOULDER_SHARE = 0.4   # [module] the collar and shoulder rows move over the last this share of the reach and
#                              the first this share of the release, the elbow and wrist rows over the whole of each: the
#                              elbow folds ahead of the shoulder on the way in and unfolds after it on the way out
#                              (review of 2026-11-03: on one blend the arm rose nearly straight, the wrist 0.37 m in
#                              front of the shoulder joint, before folding back to the face; the elbow's own course is
#                              not shortened: over 0.6 s it turned 3.8 deg in one knot, beyond the 3 deg of the tests'
#                              snap check; at 0.4 the shoulder turns 2.2 deg in one knot at most)
GESTE_PALM_GAP_M = 0.01      # [module] the palm's and the forearm's least distance to the skin (measure_geste); the
#                              palm and the wrist kept as far from the front of the neck (throat_ids)
GESTE_BENDS = tuple((b, d) for b in (-60.0, -40.0, -20.0, 0.0, 20.0, 40.0, 60.0) for d in (-20.0, 0.0, 20.0))
#                            [module] geste_arms' wrist bends (beta: negative toward the back of the hand, positive
#                            toward the palm; delta: deviation, deg): the elbow is put where the forearm comes closest
#                            to the hand's long axis so bent; the wrist bend each arm reaches is read back
#                            (wrist_angles) and kept inside WRIST_RANGE_DEG
GESTE_COLLARS = ((0.0, 0.0), (15.0, 0.0), (15.0, 10.0), (25.0, 10.0), (-10.0, 0.0), (0.0, -10.0), (-15.0, -10.0))
#                            [module] collar offsets (forward deg about y, elevation deg about z, mirrored for the
#                            right) tried with each bend; negative: the shoulder girdle back or down
GESTE_TWIST_DEG = tuple(float(x) for x in range(-120, 121, 5))
#                            [module] the forearm's twist taken at the elbow (pronation or supination), the wrist
#                            turning back by as much, scanned for the arm farthest inside the table
GESTE_MARGIN_DEG = 1.0       # [module] the arms at least this far inside the table on the sampled knots ranked before
#                              those nearer its bounds (geste_arm_tier); every built arm is checked at every knot
GESTE_SEARCH_STEP = 10       # [module] knots between two samples of the arm search (0.1 s)
GESTE_ARM_TRIES = 8          # [module] the ranked arms built at every knot, in turn, until one adds no violation
GESTE_REACH_TOL_M = 1e-4     # [module] an arm whose wrist misses its target by more is not used
GESTE_CHECK_STEP = 10        # [module] knots between two reads of the hand-to-head distance (0.1 s)


@functools.lru_cache(maxsize=4)
def pad_vertices(side):
    """{finger: vertex}: the pad of each finger of GESTE_PADS on the neutral rest mesh: among the vertices of its
    distal phalanx (dominant joint: the finger's distal joint) in the distal half of the segment, the one farthest
    along the hand's palmar normal (hands.hand_frame)."""
    J, v, dom = H._rest()
    n = np.asarray(H.hand_frame()[side]["palmar"], dtype=np.float64)
    tips = H.tip_vertices()[side]
    out = {}
    for f in GESTE_PADS:
        j3 = H.hand_joint(side, f, 2)
        ids = np.flatnonzero(dom == j3)
        b = v[tips[f]] - J[j3]
        u = (v[ids] - J[j3]) @ b / float(b @ b)
        sel = ids[u >= 0.5]
        out[f] = int(sel[np.argmax(v[sel] @ n)])
    return out


@functools.lru_cache(maxsize=4)
def _pad_offset(side, pose_key):
    pose = {f: tuple(tuple(x) if isinstance(x, (tuple, list)) else x for x in vals) for f, vals in pose_key}
    b = H.hands_of(static_body("stand", 0.01))
    Rh = b.hand_R().clone()
    rows = list(range(15)) if side == "left" else list(range(15, 30))
    Rh[:, rows] = hand_pose_rows(pose, side)[rows][None]
    b = b.with_hands(Rh)
    J, Gl = b.fk()
    w = H.WRIST[side]
    pts = H.body_points(b, list(pad_vertices(side).values()), knots=[0])[0].mean(0)
    return (Gl[0, w].T @ (torch.as_tensor(pts, dtype=F64) - J[0, w])).numpy()


@functools.lru_cache(maxsize=4)
def _pads_in_wrist(side, pose_key):
    pose = {f: tuple(tuple(x) if isinstance(x, (tuple, list)) else x for x in vals) for f, vals in pose_key}
    b = H.hands_of(static_body("stand", 0.01))
    Rh = b.hand_R().clone()
    rows = list(range(15)) if side == "left" else list(range(15, 30))
    Rh[:, rows] = hand_pose_rows(pose, side)[rows][None]
    b = b.with_hands(Rh)
    J, Gl = b.fk()
    w = H.WRIST[side]
    pts = H.body_points(b, _geste_vertex_sets(side)["pads"], knots=[0])[0]
    return (pts - J[0, w].numpy()) @ Gl[0, w].numpy()


def pads_in_wrist(side, pose):
    """(N, 3) the vertices of the pads' fingers (_geste_vertex_sets 'pads') as offsets from the wrist joint in the
    wrist's rest frame, with the hand in pose ({finger: angles}): their places follow from the wrist's alone."""
    key = tuple((f, tuple(tuple(x) if isinstance(x, (tuple, list)) else x for x in vals)) for f, vals in pose.items())
    return _pads_in_wrist(side, key)


def pad_offset(side, pose):
    """(3,) the rest-frame offset from the wrist joint of the mean of the pads (pad_vertices) with the hand in pose
    ({finger: angles}): the end point of the geste's arm solve."""
    key = tuple((f, tuple(tuple(x) if isinstance(x, (tuple, list)) else x for x in vals)) for f, vals in pose.items())
    return _pad_offset(side, key)


@functools.lru_cache(maxsize=2)
def jaw_vertex(side="left"):
    """The chin landmark of the geste: the lowest vertex of the head segment 1.5 to 4 cm to the side of the midline
    and at least 6 cm in front of the head joint (the chin on that side), on the neutral rest mesh."""
    J, v, dom = H._rest()
    sg = 1.0 if side == "left" else -1.0
    sel = np.flatnonzero((dom == 15) & (sg * v[:, 0] > 0.015) & (sg * v[:, 0] < 0.04) & (v[:, 2] > J[15, 2] + 0.06))
    return int(sel[np.argmin(v[sel, 1])])


@functools.lru_cache(maxsize=2)
def cheek_contact_vertex(side="left"):
    """The geste's contact point on the cheek of side: among the vertices of the head segment more than 2 cm to that
    side of the midline, those within 6 mm (in height and depth) of the point GESTE_CONTACT_M[0] in front of the head
    joint and GESTE_CONTACT_M[1] above the chin landmark (jaw_vertex), the most lateral one, on the neutral rest
    mesh."""
    J, v, dom = H._rest()
    sg = 1.0 if side == "left" else -1.0
    dz, dy = GESTE_CONTACT_M
    ids = np.flatnonzero((dom == 15) & (sg * v[:, 0] > 0.02))
    d = np.hypot(v[ids, 1] - v[jaw_vertex(side), 1] - dy, v[ids, 2] - J[15, 2] - dz)
    sel = ids[d < 0.006]
    return int(sel[np.argmax(sg * v[sel, 0])]) if len(sel) else int(ids[np.argmin(d)])


def _skin_ids():
    """The vertices of the selected body model's skin: every vertex with a dominant joint (the open body's three inner
    parts, the mouth cavity and the eyes, take -1 and are left out, design section 12.3; every SMPL+H vertex)."""
    return np.flatnonzero(H._rest()[2] >= 0)


@functools.lru_cache(maxsize=1)
def _rest_normals():
    """(V, 3) the unit outward vertex normals of the selected body model's rest mesh (the area-weighted normals of
    the faces around each vertex; a vertex of no face, none on either body, would read 0)."""
    from .viewer import faces
    _, v, _ = H._rest()
    f = faces()
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    vn = np.zeros_like(v)
    for i in range(3):
        np.add.at(vn, f[:, i], fn)
    return vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-300)


@functools.lru_cache(maxsize=2)
def _hand_axes(side):
    """(3, 3) columns in the rest frame: the hand's long axis (wrist to middle knuckle), the palmar normal
    (hands.hand_frame) made orthogonal to it, and their cross product."""
    J, _, _ = H._rest()
    b = J[H.hand_joint(side, "middle", 0)] - J[H.WRIST[side]]
    b = b / np.linalg.norm(b)
    n = np.asarray(H.hand_frame()[side]["palmar"], dtype=np.float64)
    n = n - (n @ b) * b
    n = n / np.linalg.norm(n)
    return np.stack([b, n, np.cross(b, n)], 1)


def _geste_frame_rest(side, cv, finger_deg, tilt_deg, roll_deg):
    """(3, 3) the hand's rotation (rest to the head's rest frame) on the cheek at the contact vertex cv: in the skin's
    tangent plane at cv (the rest mesh), the hand's long axis along the horizontal tangent pointing back, raised by
    finger_deg, then turned into the face by tilt_deg about the hand's transverse axis; the palm toward the face, then
    turned about the long axis by roll_deg (positive: toward the floor)."""
    nh = _rest_normals()[cv]
    back = np.cross(nh, np.array([0.0, 1.0, 0.0]))
    back = back / np.linalg.norm(back)
    back = -back if back[2] > 0 else back
    up = np.cross(back, nh)
    up = up / np.linalg.norm(up)
    up = -up if up[1] < 0 else up
    th, ta = math.radians(finger_deg), math.radians(tilt_deg)
    u = math.cos(th) * back + math.sin(th) * up
    f = math.cos(ta) * u - math.sin(ta) * nh
    n = -math.sin(ta) * u - math.cos(ta) * nh
    down = np.cross(f, n)
    down = -down if down[1] > 0 else down
    n = math.cos(math.radians(roll_deg)) * n + math.sin(math.radians(roll_deg)) * down
    return np.stack([f, n, np.cross(f, n)], 1) @ _hand_axes(side).T


def geste_hand_frame(body, side, cv, finger_deg, tilt_deg, roll_deg):
    """(Rh (T, 3, 3), nrm (T, 3), P (T, 3)): the hand's global rotation (rest to world) on the cheek
    (_geste_frame_rest turned with the head), the skin's outward normal at the contact vertex cv and cv on the skinned
    body."""
    _, Gl = body.fk()
    G15 = Gl[:, 15].numpy()
    Rh = G15 @ _geste_frame_rest(side, cv, finger_deg, tilt_deg, roll_deg)[None]
    return Rh, np.einsum("tij,j->ti", G15, _rest_normals()[cv]), H.body_points(body, [cv])[:, 0]


def wrist_angles(fa, Rh, side):
    """(flexion, deviation) deg, arrays of fa's leading shape: the hand's long axis ha (_hand_axes, turned by the
    hand's global rotation Rh (..., 3, 3)) against the forearm's unit axis fa (..., 3) (elbow to wrist joint): flexion
    atan2(ha.nf, ha.fa), with nf the palmar normal made orthogonal to the forearm (positive toward the palm, negative
    toward the back of the hand), deviation asin(ha.(fa x nf)) signed positive toward the thumb (radial) side (fa x nf
    points there for the right hand, to the little finger for the left)."""
    ax = _hand_axes(side)
    ha = np.einsum("...ij,j->...i", Rh, ax[:, 0])
    pn = np.einsum("...ij,j->...i", Rh, ax[:, 1])
    nf = pn - (pn * fa).sum(-1, keepdims=True) * fa
    nf = nf / np.linalg.norm(nf, axis=-1, keepdims=True).clip(1e-12)
    sd = np.cross(fa, nf) * (1.0 if side == "right" else -1.0)
    flex = np.degrees(np.arctan2((ha * nf).sum(-1), (ha * fa).sum(-1)))
    return flex, np.degrees(np.arcsin(np.clip((ha * sd).sum(-1), -1.0, 1.0)))


def wrist_bend_deg(body, side, knots=None):
    """(flexion (K,), deviation (K,)) deg of the wrist of side at the knots (default all), read from the posed body
    (wrist_angles: + palmar flexion, - dorsiflexion; + radial, - ulnar deviation)."""
    J, Gl = body.fk()
    k = np.arange(body.T) if knots is None else np.asarray(knots, dtype=np.int64)
    _, _, e, w = ARM[side]
    fa = J[k, w].numpy() - J[k, e].numpy()
    fa = fa / np.linalg.norm(fa, axis=-1, keepdims=True)
    return wrist_angles(fa, Gl[k, w].numpy(), side)


def wrist_load(flex, dev, margin=0.0):
    """The largest share of WRIST_RANGE_DEG (each bound less margin deg) that the wrist's flexion and deviation (deg,
    wrist_angles' signs) take: <= 1 inside the range."""
    r = WRIST_RANGE_DEG
    fl = np.where(flex >= 0.0, flex / (r["palmar"] - margin), -flex / (r["dorsal"] - margin))
    dv = np.where(dev >= 0.0, dev / (r["radial"] - margin), -dev / (r["ulnar"] - margin))
    return np.maximum(fl, dv)


def _collar_rotations(R, side, collar):
    c = ARM[side][0]
    sg = 1.0 if side == "left" else -1.0
    one = np.ones(R.shape[0])
    return apply_terms(R, [joint_term(c, "y", -sg * collar[0], one, "parent"),
                           joint_term(c, "z", sg * collar[1], one, "parent")])


def geste_frames(bodies, side, cv, end, table=DYST):
    """[(load, (finger, tilt, roll) deg, collar)] ranked: the hand frames of GESTE_FINGERS_DEG x GESTE_TILTS_DEG x
    GESTE_ROLLS_DEG on the cheek at the contact vertex cv (_geste_frame_rest, the pads' mean point end (pad_offset)
    GESTE_SCREEN_GAP_M off the skin along its normal) read on bodies ([(body, knots)]: the clip's posture while the
    hand rests, and the level 0's head, so that the frame serves the head anywhere between its held posture and
    neutral). For each frame and collar offset of GESTE_COLLARS the elbow is put at each place of GESTE_ELBOW_DEG
    around the shoulder-to-wrist line; a place counts when the shoulder-to-wrist distance stays at or above the one the
    arm keeps at the elbow's flexion bound of table (no forearm twist) and the wrist bend (wrist_angles) stays
    GESTE_WRIST_MARGIN_DEG inside WRIST_RANGE_DEG at every knot; a frame counts when its wrist lies beside the neck
    (GESTE_NECK_CLEAR_M) at every knot. load: the least over its places and collars of the largest wrist_load over the
    knots, among the places whose forearm keeps GESTE_TRUNK_GAP_M off the trunk (geste_arms' stand-in, on
    GESTE_SCREEN_TRUNK_KNOTS knots and every GESTE_SCREEN_TRUNK_STRIDE-th trunk vertex) when there are some. Ranked:
    the frames with such a place first, each group by load in steps of GESTE_LOAD_STEP, then the lower finger
    elevation."""
    c, s, e, w = ARM[side]
    sg = 1.0 if side == "left" else -1.0
    rest = bodies[0][0].rest()
    p, q = (rest[e] - rest[s]).numpy(), (rest[w] - rest[e]).numpy()
    Lu, Lf = float(np.linalg.norm(p)), float(np.linalg.norm(q))
    b0, k0 = bodies[0]
    klo, khi = LX.knot_bounds(table, np.degrees(matrix_to_axis_angle(b0.local_R()[k0[:1]]).numpy()))
    bound = math.radians(float(khi[0, e, 1]) if side == "right" else -float(klo[0, e, 1]))
    cb, sb = math.cos(bound), math.sin(bound)
    qy = np.array([cb * q[0] + sb * q[2], q[1], -sb * q[0] + cb * q[2]])   # the forearm turned about the elbow's y
    dmin = float(np.linalg.norm(p + qy))
    _, half = throat_ids()
    G15, G12, J12, S = [], [], [], {cl: [] for cl in GESTE_COLLARS}
    for b, k in bodies:
        J, Gl = b.fk()
        G15.append(Gl[k, 15].numpy())
        G12.append(Gl[k, 12].numpy())
        J12.append(J[k, 12].numpy())
        for cl in GESTE_COLLARS:
            with torch.no_grad():
                Jc, _ = fk(_collar_rotations(b.local_R()[k], side, cl), b.transl[k], rest)
            S[cl].append(Jc[:, s].numpy())
    G15, G12, J12 = (np.concatenate(x) for x in (G15, G12, J12))
    S = {cl: np.concatenate(x) for cl, x in S.items()}
    Pc = np.concatenate([H.body_points(b, [cv], knots=k)[:, 0] for b, k in bodies])
    nrm = np.einsum("tij,j->ti", G15, _rest_normals()[cv])
    th = np.radians(np.asarray(GESTE_ELBOW_DEG))
    from .reach import _surface_sets
    tk = np.unique(np.linspace(0, len(G15) - 1, GESTE_SCREEN_TRUNK_KNOTS).round().astype(int))
    tv = np.concatenate([H.body_points(b, _surface_sets(side)["trunk"][0][::GESTE_SCREEN_TRUNK_STRIDE], knots=k)
                         for b, k in bodies])[tk]
    rad = _forearm_radius(side)
    sh = np.linspace(0.0, 1.0, GESTE_TRUNK_POINTS)
    out = []
    for fd in GESTE_FINGERS_DEG:
        for td in GESTE_TILTS_DEG:
            for rd in GESTE_ROLLS_DEG:
                Rh = G15 @ _geste_frame_rest(side, cv, fd, td, rd)[None]
                W = Pc - np.einsum("tij,j->ti", Rh, end) + GESTE_SCREEN_GAP_M * nrm
                lat = sg * np.einsum("tji,tj->ti", G12, W - J12)[:, 0]
                if lat.min() < half + GESTE_NECK_CLEAR_M:
                    continue
                best = None
                for cl in GESTE_COLLARS:
                    L = W - S[cl]
                    d = np.linalg.norm(L, axis=-1)
                    if d.min() < dmin or d.max() > Lu + Lf:
                        continue
                    u = L / d[:, None]
                    x = (Lu ** 2 - Lf ** 2 + d ** 2) / (2.0 * d)
                    r = np.sqrt(np.clip(Lu ** 2 - x ** 2, 0.0, None))
                    e1 = np.cross(u, np.array([0.0, 1.0, 0.0]))
                    e1 = e1 / np.linalg.norm(e1, axis=-1, keepdims=True)
                    e2 = np.cross(u, e1)
                    E = (S[cl] + u * x[:, None])[:, None] + r[:, None, None] * (
                        np.cos(th)[None, :, None] * e1[:, None] + np.sin(th)[None, :, None] * e2[:, None])
                    fa = W[:, None] - E
                    fa = fa / np.linalg.norm(fa, axis=-1, keepdims=True)
                    fl, dv = wrist_angles(fa, np.repeat(Rh[:, None], len(th), 1), side)
                    ld = wrist_load(fl, dv, GESTE_WRIST_MARGIN_DEG).max(0)
                    ok = np.flatnonzero(ld <= 1.0)
                    if not len(ok):
                        continue
                    ax = E[tk][:, ok, None] + sh[None, None, :, None] * (W[tk, None, None] - E[tk][:, ok, None])
                    tr = np.sqrt(((ax[:, :, :, None] - tv[:, None, None]) ** 2).sum(-1)).min((0, 2, 3)) - rad
                    clear = tr >= GESTE_TRUNK_GAP_M
                    i = ok[clear][int(np.argmin(ld[ok][clear]))] if clear.any() else ok[int(np.argmin(ld[ok]))]
                    cand = (not bool(clear.any()), float(ld[i]), cl)
                    if best is None or cand[:2] < best[:2]:
                        best = cand
                if best is not None:
                    out.append((best[1], (fd, td, rd), best[2], not best[0]))
    return [x[:3] for x in sorted(out, key=lambda x: (not x[3], int(x[0] / GESTE_LOAD_STEP), x[1][0], x[0]))]


def _geste_arm_once(rest, tr, R, side, W, Rh, beta, delta, collar, phis):
    """(K * len(phis), 22, 3, 3) rotations (rest joints rest, translations tr (K, 3)): R with the collar offset,
    the shoulder and elbow solved so that the wrist joint lies on W (K, 3) with the elbow where the forearm comes
    closest to the hand's long axis bent by beta (deg, negative toward the back of the hand) and delta (deviation,
    deg), the forearm twisted by each of phis (deg) at the elbow and the wrist turned so that the hand's global
    rotation is Rh (K, 3, 3); and (K, 3) the elbow joint, (K,) m the wrist joint's distance from W (above 0 where W
    is out of the arm's reach)."""
    c, s, e, w = ARM[side]
    K = R.shape[0]
    Rc = _collar_rotations(R, side, collar)
    with torch.no_grad():
        Jc, _ = fk(Rc, tr, rest)
    S = Jc[:, s].numpy()
    ax = _hand_axes(side)
    bh = np.einsum("tij,j->ti", Rh, ax[:, 0])
    nh = np.einsum("tij,j->ti", Rh, ax[:, 1])
    be = math.radians(beta)
    a = math.cos(be) * bh - math.sin(be) * nh
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    dl = math.radians(delta)
    a = a * math.cos(dl) + np.cross(nh, a) * math.sin(dl) + nh * (nh * a).sum(-1, keepdims=True) * (1 - math.cos(dl))
    Lf = float((rest[w] - rest[e]).norm())
    u = (W - S) / np.linalg.norm(W - S, axis=-1, keepdims=True)
    pole = (W - Lf * a) - S
    pole = pole - (pole * u).sum(-1, keepdims=True) * u
    pole = pole / np.linalg.norm(pole, axis=-1, keepdims=True).clip(1e-12)
    Rk = _arm_once(Rc, tr, rest, side, torch.as_tensor(W, dtype=F64), torch.as_tensor(pole, dtype=F64), "wrist")
    with torch.no_grad():
        Jk, Gk = fk(Rk, tr, rest)
    Rw = Gk[:, e].transpose(-1, -2) @ torch.as_tensor(Rh, dtype=F64)
    bax = (rest[w] - rest[e]) / (rest[w] - rest[e]).norm()
    NP = len(phis)
    ph = np.repeat(np.radians(np.asarray(phis, dtype=np.float64)), K)
    Rt = Rk[None].expand(NP, K, 22, 3, 3).reshape(NP * K, 22, 3, 3).clone()
    Rt[:, e] = Rk[:, e].repeat(NP, 1, 1) @ _rot_about(bax, ph)
    Rt[:, w] = _rot_about(bax, -ph) @ Rw.repeat(NP, 1, 1)
    return Rt, Jk[:, e].numpy(), np.linalg.norm(Jk[:, w].numpy() - W, axis=-1)


def geste_blend(R0, R1, side, prox, dist):
    """R0 with the arm rows of side turned toward R1 (blend_rows): the collar and the shoulder by the shares prox
    (T,), the elbow and the wrist by dist (T,) (GESTE_SHOULDER_SHARE)."""
    c, s, e, w = ARM[side]
    return blend_rows(blend_rows(R0, R1, prox, [c, s]), R1, dist, [e, w])


@functools.lru_cache(maxsize=2)
def _forearm_radius(side):
    """m: the largest distance of the forearm's vertices (_geste_vertex_sets 'fore') from the line of the elbow and
    wrist joints, on the neutral rest mesh."""
    J, v, _ = H._rest()
    _, _, e, w = ARM[side]
    a = (J[w] - J[e]) / np.linalg.norm(J[w] - J[e])
    x = v[_geste_vertex_sets(side)["fore"]] - J[e]
    return float(np.linalg.norm(x - (x @ a)[:, None] * a, axis=1).max())


def geste_arms(body, R, side, W, Rh, knots, blend, base_R, table=DYST):
    """[(score deg, elbow height m, (beta, delta), collar, twist deg, wrist load, wrist ok, trunk m)] ranked: the arms
    of side
    (collar, shoulder, elbow, wrist) that put the wrist joint on W (T, 3) and turn the hand to Rh (T, 3, 3): for each
    wrist bend of GESTE_BENDS and collar of GESTE_COLLARS (_geste_arm_once) the twist of GESTE_TWIST_DEG whose arm
    rows, blended from R by blend ((prox, dist), (T,) each: the path in and out of the posture, geste_blend), lie
    farthest inside table (limits_ext, against base_R) on the knots sampled every GESTE_SEARCH_STEP, the score being
    the largest excess (deg) over the allowance max(-GESTE_MARGIN_DEG, the excess of R's own rows there): <= 0 when
    every component stays GESTE_MARGIN_DEG inside its bound, or no closer to it than R already is. An arm whose wrist
    misses W by more than GESTE_REACH_TOL_M at a sampled knot is left out. The wrist bend of each arm (wrist_angles,
    the forearm from its elbow to W, the hand at Rh) is read where both blends are 1: wrist ok when it stays
    GESTE_WRIST_MARGIN_DEG inside WRIST_RANGE_DEG (wrist load <= 1, wrist_load). trunk: there, the least distance
    from the forearm's axis (elbow to wrist joint, GESTE_TRUNK_POINTS points) to the trunk's vertices
    (reach.PARTS['trunk']) less the forearm's radius (_forearm_radius), a cheap stand-in for the exact check of the
    build (head_clearance on the trunk). The table is read only for the arms with the wrist ok and trunk >=
    GESTE_TRUNK_GAP_M (score inf for the others). The arms with a score <= 0, the wrist ok and trunk >=
    GESTE_TRUNK_GAP_M come first (geste_arm_tier 0), then those inside the table by less than GESTE_MARGIN_DEG (tier
    1), each tier by wrist load in steps of GESTE_LOAD_STEP and then the lowest elbow (mean height there); then the
    others by score."""
    prox, dist = (np.asarray(x, dtype=np.float64) for x in blend)
    rows = list(ARM[side])
    sk = np.asarray(knots)[::GESTE_SEARCH_STEP]
    held = (prox[sk] >= 1.0 - 1e-9) & (dist[sk] >= 1.0 - 1e-9)
    held = held if held.any() else np.ones(len(sk), bool)
    rest = body.rest()
    NP = len(GESTE_TWIST_DEG)
    Rs, Ws, Rhs = R[sk], W[sk], Rh[sk]
    cb = np.degrees(matrix_to_axis_angle(base_R[sk]).numpy())
    allow = np.maximum(-GESTE_MARGIN_DEG, LX.excess(table, np.degrees(matrix_to_axis_angle(Rs).numpy()), cb, rows))
    Rrep = Rs.repeat(NP, 1, 1, 1)
    from .reach import _surface_sets
    tv = H.body_points(body, _surface_sets(side)["trunk"][0], knots=sk[held])
    rad = _forearm_radius(side)
    sh = np.linspace(0.0, 1.0, GESTE_TRUNK_POINTS)
    cands = []
    for (beta, delta), collar in ((bd, cl) for bd in GESTE_BENDS for cl in GESTE_COLLARS):
        Rt, Je, miss = _geste_arm_once(rest, body.transl[sk], Rs, side, Ws, Rhs, beta, delta, collar,
                                       GESTE_TWIST_DEG)
        if miss.max() > GESTE_REACH_TOL_M:
            continue
        fa = Ws[held] - Je[held]
        fa = fa / np.linalg.norm(fa, axis=-1, keepdims=True)
        load = float(wrist_load(*wrist_angles(fa, Rhs[held], side), GESTE_WRIST_MARGIN_DEG).max())
        ax = Je[held, None] + sh[None, :, None] * (Ws[held] - Je[held])[:, None]
        trunk = min(float(np.sqrt(((ax[j][:, None] - tv[j][None]) ** 2).sum(-1)).min()) for j in range(len(ax))) - rad
        if load > 1.0 or trunk < GESTE_TRUNK_GAP_M:     # the table read only for the arms that can serve
            cands.append((math.inf, float(Je[held, 1].mean()), (beta, delta), collar, 0.0, load, load <= 1.0, trunk))
            continue
        comp = np.degrees(matrix_to_axis_angle(geste_blend(Rrep, Rt, side, np.tile(prox[sk], NP),
                                                           np.tile(dist[sk], NP))).numpy())
        ex = LX.excess(table, comp, np.tile(cb, (NP, 1, 1)), rows).reshape(NP, len(sk), len(rows), 3)
        score = (ex - allow[None]).max((1, 2, 3))
        i = int(np.argmin(score))
        cands.append((float(score[i]), float(Je[held, 1].mean()), (beta, delta), collar, GESTE_TWIST_DEG[i], load,
                      True, trunk))

    def key(c):
        tier = geste_arm_tier(c)
        return (tier, int(c[5] / GESTE_LOAD_STEP) if tier < 2 else 0, c[1] if tier < 2 else c[0])
    return sorted(cands, key=key)


def geste_arm_tier(c):
    """0: an arm of geste_arms with the wrist ok, trunk >= GESTE_TRUNK_GAP_M and a score <= 0; 1: the same inside the
    table by less than GESTE_MARGIN_DEG (0 < score < GESTE_MARGIN_DEG); 2: the others."""
    if not (c[6] and c[7] >= GESTE_TRUNK_GAP_M):
        return 2
    return 0 if c[0] <= 0.0 else 1 if c[0] < GESTE_MARGIN_DEG else 2


@functools.lru_cache(maxsize=2)
def _geste_vertex_sets(side):
    """{'pads': the vertices of the distal phalanges of GESTE_PADS, 'palm': the wrist's, 'fore': the forearm's}
    (dominant joints of the neutral rest mesh)."""
    from .reach import ELBOW
    _, _, dom = H._rest()
    pads = np.flatnonzero(np.isin(dom, [H.hand_joint(side, f, 2) for f in GESTE_PADS]))
    return {"pads": pads, "palm": np.flatnonzero(dom == H.WRIST[side]), "fore": np.flatnonzero(dom == ELBOW[side])}


@functools.lru_cache(maxsize=1)
def throat_ids():
    """(vertex ids, half width m): the front of the neck on the neutral rest mesh: the vertices of the neck and head
    segments (dominant joints 12 and 15) below the lower of the two chin landmarks (jaw_vertex), in front of the neck
    joint and within the neck segment's half width of the midline; that half width (the neck segment's largest
    lateral extent)."""
    J, v, dom = H._rest()
    y0 = min(v[jaw_vertex("left"), 1], v[jaw_vertex("right"), 1])
    half = float(np.abs(v[dom == 12, 0]).max())
    sel = np.flatnonzero(np.isin(dom, (12, 15)) & (v[:, 1] < y0) & (v[:, 2] > J[12, 2]) & (np.abs(v[:, 0]) < half))
    return sel, half


def skin_distances(body, side, knots):
    """{'pads', 'palm', 'fore'}: (K,) m, the least signed distance (negative inside) of each vertex set of
    _geste_vertex_sets to the skin of the head and neck (reach.PARTS['head'], exact on the skinned mesh) at the
    knots."""
    from .reach import _signed_to_mesh, _surface_sets
    sets = _geste_vertex_sets(side)
    hid, hf = _surface_sets(side)["head"]
    names = ("pads", "palm", "fore")
    ids = np.concatenate([sets[n] for n in names] + [hid])
    off = np.cumsum([0] + [len(sets[n]) for n in names])
    X = H.body_points(body, ids, knots=knots)
    out = {n: np.empty(len(X)) for n in names}
    for i in range(len(X)):
        V = X[i, off[-1]:]
        for j, n in enumerate(names):
            out[n][i] = _signed_to_mesh(X[i, off[j]:off[j + 1]], V, hf).min()
    return out


def geste_antagoniste(base, direction="rotation_left", grade=2, trick_score=1, side=None, seed=0,
                      times=TRICK_TIMES):
    """(Body, truth) of a cervical dystonia (dystonia.py neck region, TWSTRS level of grade, held throughout on the
    course of mobile_envelope) with the geste antagoniste (TWSTRS C), a light touch of the lower face (SOURCES
    patel2014): the hand of side (default the side away from the chin's turn or tilt: on the side of the turn the cheek
    lies so near the same-side shoulder that no hand frame of geste_frames keeps the wrist inside its range,
    NOT_RENDERED) rises from the side to the cheek of its own side, rests there and leaves; while it is there the
    posture is undone by TRICK[trick_score] (after a latency, over relief_s; back over return_s after the release). The
    hand follows the head: the pads of the index and middle fingers (pad_vertices, the loosely curled hand GESTE_HAND)
    are the end point, on the outward normal of the skin at the contact point on the lower cheek (cheek_contact_vertex);
    the hand turned by geste_hand_frame, on the first frame of geste_frames' ranking (the least wrist bend among the
    frames whose wrist lies beside the neck and whose forearm can stay off the trunk) for which geste_arms gives an arm
    inside the table, with the wrist GESTE_WRIST_MARGIN_DEG inside WRIST_RANGE_DEG (SOURCES jarm2022_wrist) and the
    forearm off the trunk; the first of those arms, built at every knot, that adds no violation of the table, turns no
    arm joint more than GESTE_SNAP_DEG in one knot beyond the level 0, keeps the hand and the forearm GESTE_TRUNK_GAP_M
    off the trunk's skin over the reach, takes the hand in and out without a sweep (reach_path within
    GESTE_PATH_RATIO_MAX and GESTE_FORWARD_EXTRA_M) and puts no hand or forearm vertex inside the head or neck
    (head_clearance; wave SH1c); where no arm of the tried frames keeps the wrist within GESTE_FORWARD_EXTRA_M, the
    passing arm of least excess within GESTE_FORWARD_EXTRA_MAX_M (params arm 'forward_relaxed'). The elbow and the wrist
    lead the collar and the shoulder on the way in and trail them on the way out (GESTE_SHOULDER_SHARE, geste_blend).
    The pads' distance from the cheek is solved by secant steps, on the hand's place alone (pads_in_wrist) then on the
    posed body at every check while the hand rests (the offset along the skin's normal interpolated between them), so
    that the least distance of those two fingers' distal phalanges to the skin of the head and neck there is GESTE_GAP_M
    (a light touch). Reviews: of 2026-11-02 (with the palm on the chin the fingers entered the face), of the SH2 wave
    (with the pads under the chin and the whole hand 1 cm off the skin it hovered before the throat) and of 2026-11-03
    (with the pads on the chin from below, the hand stood upright with the palm and the wrist in front of the neck;
    then, with the hand's long axis back toward the ear, the wrist bent back and toward the thumb past WRIST_RANGE_DEG,
    and the arm rose nearly straight before folding to the face). The hand-to-head check (head_clearance:
    reach.clearance on the skinned mesh, every GESTE_CHECK_STEP knots of the reach) asks that no hand or forearm vertex
    enters the head or neck, the trunk check that they stay GESTE_TRUNK_GAP_M off the trunk; measure_geste reads the
    hand's place back, wrist_bend_deg the wrist, reach_path the way in. base: a HandBody or a Body (hands at rest then).
    """
    rng = as_rng(seed)
    b = H.hands_of(base)
    T, rate = b.T, b.rate
    t = np.arange(T) / rate
    turn = "left" if direction.endswith("left") else "right"
    if side is None:
        side = "right" if turn == "left" else "left"
    chain, axis, sign, kind, frame, _ = _resolve("neck", direction, "left")
    lvl = G.twstrs_level_deg(kind, grade, region_room(b, "neck", direction, "left", table=DYST))
    env0 = np.clip(1.0 - FLUCT + FLUCT * smooth_noise(T, rate, rng), 0.0, 1.0)   # constant, near maximal
    cinfo = {"share": 1.0, "hold": 1.0 - FLUCT, "fluct": FLUCT, "twstrs_duration": 5}
    tm = dict(times)
    on, rs, rel, bk = tm["reach_on_s"], tm["reach_s"], tm["release_s"], tm["back_s"]
    touch = on + rs
    relief = TRICK[trick_score] * (minimum_jerk((t - touch - tm["latency_s"]) / tm["relief_s"])
                                   - minimum_jerk((t - rel) / tm["return_s"]))
    env = env0 * (1.0 - relief)
    term = region_term(b, "neck", direction, "left", lvl, env, f"neck {direction}", table=DYST)
    R, s = fit_terms(b.local_R(), [term], table=DYST)
    post = b.with_rotations(R, list(chain))
    cv = cheek_contact_vertex(side)
    reach = minimum_jerk((t - on) / rs) - minimum_jerk((t - rel) / bk)
    prox_in, prox_out = GESTE_SHOULDER_SHARE * rs, GESTE_SHOULDER_SHARE * bk   # the collar and shoulder rows
    prox = minimum_jerk((t - on - (rs - prox_in)) / prox_in) - minimum_jerk((t - rel) / prox_out)
    k = reach > 0
    held = np.flatnonzero(reach >= 1.0 - 1e-9)
    checks = np.unique(np.r_[np.flatnonzero(k)[::GESTE_CHECK_STEP], np.flatnonzero(k)[-1:]])
    held_checks = np.intersect1d(checks, held)
    end = pad_offset(side, GESTE_HAND)
    grip = H.hand_rotations({(side, f, lv): v for f, vals in GESTE_HAND.items() for lv, v in enumerate(vals)})
    Rh0 = b.hand_R()
    Rh1 = Rh0.clone()
    rows = list(range(15)) if side == "left" else list(range(15, 30))
    Rh1[:, rows] = grip[rows][None]
    Rh = Rh0.clone()
    for r in rows:
        D = matrix_to_axis_angle(Rh0[:, r].transpose(-1, -2) @ Rh1[:, r])
        Rh[:, r] = Rh0[:, r] @ axis_angle_to_matrix(torch.as_tensor(reach)[:, None] * D)
    kr = np.flatnonzero(k)
    from .reach import _signed_to_mesh, _surface_sets
    pads = pads_in_wrist(side, GESTE_HAND)
    hid, hface = _surface_sets(side)["head"]
    skin = H.body_points(post, hid, knots=held_checks)

    def pad_skin(W, Rhw):                                 # the pads' least distance to the skin of the head and neck
        X = W[held_checks, None] + np.einsum("tij,nj->tni", Rhw[held_checks], pads)
        return min(float(_signed_to_mesh(X[i], skin[i], hface).min()) for i in range(len(held_checks)))
    frames = geste_frames([(post, held[::GESTE_SCREEN_STEP]), (b, held[::GESTE_SCREEN_STEP])], side, cv, end)
    if not frames:
        raise RuntimeError(f"geste_antagoniste: no hand frame on the {side} cheek keeps the wrist inside its range")

    def build(W, nrm, Rhw, gap, arm):                    # gap: the pads' offset along the skin's normal, (T,)
        Rk, _, _ = _geste_arm_once(post.rest(), post.transl, R, side, W + gap[:, None] * nrm, Rhw, *arm[2], arm[3],
                                   (arm[4],))
        Rf, moved = project_rows(geste_blend(R, Rk, side, prox, reach), ARM[side], b.local_R(), table=DYST)
        return b.with_rotations(Rf, sorted(set(chain) | set(ARM[side]))).with_hands(Rh), moved
    found, fallback = None, None
    base_steps = _steps_deg(b, ARM[side])
    for ftried, frame in enumerate(frames[:GESTE_FRAME_TRIES], 1):   # the first frame and arm inside the table, the
        Rhw, nrm, P = geste_hand_frame(post, side, cv, *frame[1])      # wrist inside its range, the hand and forearm
        W0 = P - np.einsum("tij,j->ti", Rhw, end)                      # off the trunk
        gaps, reads = [GESTE_GAP_M], []
        for _ in range(GESTE_GAP_ITERS):                  # at GESTE_GAP_M, from the hand's place alone (the arm that
            d = pad_skin(W0 + gaps[-1] * nrm, Rhw)        # reaches it does not move the hand)
            reads.append(d)
            if abs(d - GESTE_GAP_M) < GESTE_GAP_TOL_M:
                break
            if len(reads) == 1:
                gaps.append(gaps[-1] + (GESTE_GAP_M - d))
            else:
                slope = (reads[-1] - reads[-2]) / (gaps[-1] - gaps[-2]) if gaps[-1] != gaps[-2] else 1.0
                gaps.append(gaps[-1] + (GESTE_GAP_M - d) / (slope if abs(slope) > 0.2 else 1.0))
        gaps = gaps[:len(reads)]
        cands = geste_arms(post, R, side, W0 + gaps[-1] * nrm, Rhw, kr, (prox, reach), b.local_R())
        good = [c for c in cands if geste_arm_tier(c) < 2]
        for tried, arm in enumerate(good[:GESTE_ARM_TRIES], 1):
            g, refine = np.full(len(held_checks), gaps[-1]), []
            for _ in range(GESTE_GAP_ITERS):              # the gap read again on the posed body at every check while
                gt = np.interp(np.arange(T), held_checks, g)   # the hand rests (the neck's turn and the pose blend
                o, projected = build(W0, nrm, Rhw, gt, arm)    # shapes move the skin under the pads by a millimetre or
                d1 = skin_distances(o, side, held_checks)["pads"]   # so), the offset interpolated between the checks
                refine.append((gt, d1))
                if np.abs(d1 - GESTE_GAP_M).max() < GESTE_GAP_TOL_M / 2:
                    break
                g = g + (GESTE_GAP_M - d1)
            if LX.violations(DYST, o.rot6d, b.rot6d)["new"] > 0:
                continue
            if (_steps_deg(o, ARM[side]) - base_steps).max() > GESTE_SNAP_DEG:
                continue
            trunk = head_clearance(o, side, checks, part="trunk")
            if trunk["min_m"] < GESTE_TRUNK_GAP_M:
                continue
            rp = reach_path(b, o, side, tm)               # wave SH1c: the way in and out and the head checked on the
            fwd = rp["wrist_forward_m"] - rp["wrist_forward_held_m"]   # built arm, so that the curled hand's first
            if rp["ratio"] > GESTE_PATH_RATIO_MAX or fwd > GESTE_FORWARD_EXTRA_MAX_M:   # arm that sweeps out or enters
                continue                                  # the chin gives way to the next one
            if fwd > GESTE_FORWARD_EXTRA_M and fallback is not None and fwd >= fallback["forward_extra_m"]:
                continue
            if not head_clearance(o, side, checks)["ok"]:
                continue
            state = {"found": (o, projected, arm, tried, trunk), "refine": list(refine), "gaps": list(gaps),
                     "reads": list(reads), "frame": frame, "ftried": ftried, "P": P, "nrm": nrm, "good": len(good),
                     "cands": len(cands), "forward_extra_m": fwd}
            if fwd > GESTE_FORWARD_EXTRA_M:               # kept in case no arm keeps the preferred bound
                fallback = state
                continue
            found = state
            break
        if found is not None:
            break
    relaxed = found is None and fallback is not None
    found = fallback if found is None else found
    if found is None:
        raise RuntimeError(f"geste_antagoniste: no arm of the first {GESTE_FRAME_TRIES} hand frames on the {side} "
                           "cheek stays inside the table, the wrist inside its range, the arm off the trunk and the "
                           "head, without a sweep")
    o, projected, arm, tried, trunk = found["found"]
    refine, frame, ftried, P, nrm = found["refine"], found["frame"], found["ftried"], found["P"], found["nrm"]
    gt = refine[-1][0]
    gaps = found["gaps"] + [float(x[0][held_checks].mean()) for x in refine]
    reads = found["reads"] + [float(x[1].min()) for x in refine]
    arm_info = {"bend_deg": list(arm[2]), "collar_deg": list(arm[3]), "twist_deg": arm[4], "score_deg": arm[0],
                "elbow_height_m": arm[1], "wrist_load": arm[5], "trunk_stand_in_m": arm[7],
                "arms_inside": found["good"], "arms_tried": found["cands"], "built": tried,
                "forward_extra_m": found["forward_extra_m"],
                "forward_relaxed": relaxed}
    out = replace(o, contact=contact_labels(o))
    m = measure_geste(b, out, side, times, lvl, pad_target=P + gt[:, None] * nrm, contact_vertex=cv)
    fl, dv = wrist_bend_deg(out, side, kr)
    hk = np.isin(kr, held)
    m["wrist_deg"] = {"held_flexion": [float(fl[hk].min()), float(fl[hk].max())],
                      "held_deviation": [float(dv[hk].min()), float(dv[hk].max())],
                      "reach_flexion": [float(fl.min()), float(fl.max())],
                      "reach_deviation": [float(dv.min()), float(dv.max())],
                      "held_load": float(wrist_load(fl[hk], dv[hk]).max()),
                      "reach_load": float(wrist_load(fl, dv).max()),
                      "range_deg": dict(WRIST_RANGE_DEG)}
    m["reach_path"] = reach_path(b, out, side, times)
    val = LX.check(DYST, b, out)
    val["hand_limits"] = {"ok": bool(H.hand_limit_report(out.hand_R())["ok"])}
    layer_checks = scene_checks_record("geste", side=side, knots=[int(x) for x in checks])
    val.update(scene_checks_of(out, layer_checks))
    val["wrist_range"] = {"ok": bool(m["wrist_deg"]["reach_load"] <= 1.0), "load": m["wrist_deg"]["reach_load"]}
    params = {"direction": direction, "grade": grade, "level_deg": float(lvl), "trick_score": trick_score,
              "hand": side, "touch": f"the {side} cheek", "contact_vertex": cv, "contact_m": list(GESTE_CONTACT_M),
              "finger_deg": frame[1][0], "tilt_deg": frame[1][1], "roll_deg": frame[1][2],
              "frame": {"screen_load": frame[0], "screen_collar_deg": list(frame[2]), "frames_inside": len(frames),
                        "tried": ftried, "shoulder_share": GESTE_SHOULDER_SHARE},
              "arm": arm_info, "projected_deg": projected,
              "relief_share": TRICK[trick_score], "times_s": tm, "pad_gap_m": float(gaps[-1]),
              "pad_gap_range_m": [float(gt[held].min()), float(gt[held].max())],
              "gap_solve": {"gaps_m": gaps, "least_m": reads}, "pads": GESTE_PADS, "course": cinfo,
              "limit_scale_min": float(s.min()), "jaw_vertex": jaw_vertex(side), "limits_table": LX.record(DYST),
              "layer_checks": layer_checks, **m}
    sources = ("twstrs", "twstrs_full", "patel2014", "limits_ext", "jarm2022_wrist", "module")
    return out, truth_record("geste_antagoniste", seed if isinstance(seed, int) else None, trick_score,
                             "TWSTRS C sensory trick", params, sources, val,
                             arrays={"envelope": env, "reach": reach, "shoulder_reach": prox, "relief": relief})


def _steps_deg(body, joints):
    """(T - 1, len(joints)) deg: the angle each joint's local rotation turns from one knot to the next."""
    L = body.local_R()[:, list(joints)]
    return rotation_angle_deg(L[1:].transpose(-1, -2) @ L[:-1]).numpy()


def reach_path(base, out, side, times):
    """{'tip_path_m', 'tip_chord_m', 'ratio', 'wrist_forward_m', 'wrist_forward_held_m'}: during the reach (from
    reach_on_s to the touch) the path length of the middle fingertip vertex (tip_vertices) against the straight line
    from where it starts to where it rests, their ratio, and the wrist joint's largest forward offset from the
    shoulder joint (in the pelvis's frame) over the reach and the release against its largest while the hand rests."""
    r = out.rate
    on, rs, rel, bk = (times[x] for x in ("reach_on_s", "reach_s", "release_s", "back_s"))
    ka, kb = int(round(on * r)), int(round((on + rs) * r))
    kc, kd = int(round(rel * r)), min(out.T - 1, int(round((rel + bk) * r)))
    tip = H.body_points(out, [H.tip_vertices()[side]["middle"]], knots=np.arange(ka, kb + 1))[:, 0]
    path = float(np.linalg.norm(np.diff(tip, axis=0), axis=-1).sum())
    chord = float(np.linalg.norm(tip[-1] - tip[0]))
    J, Gl = out.fk()
    _, s, _, w = ARM[side]
    fwd = np.einsum("tji,tj->ti", Gl[:, 0].numpy(), J[:, w].numpy() - J[:, s].numpy())[:, 2]
    moving = np.r_[np.arange(ka, kb + 1), np.arange(kc, kd + 1)]
    return {"tip_path_m": path, "tip_chord_m": chord, "ratio": path / chord if chord > 0 else 0.0,
            "wrist_forward_m": float(fwd[moving].max()), "wrist_forward_held_m": float(fwd[kb:kc + 1].max())}


def head_clearance(body, side, knots, part="head"):
    """{'min_m', 'min_hand_m', 'min_fore_m', 'at_s', 'n_knots', 'ok'}: the least signed distance (m; negative inside)
    between the hand or the forearm of side and the skin of part (reach.PARTS: the head and neck by default, or the
    trunk; reach.clearance, exact on the skinned mesh) at the knots; ok when nothing enters (the hand-to-head check of
    the reach module)."""
    from .reach import clearance
    c = clearance(body, side, knots, parts=(part,))[part]
    both = np.minimum(c["hand"], c["fore"])
    i = int(np.argmin(both))
    return {"min_m": float(both[i]), "min_hand_m": float(c["hand"].min()), "min_fore_m": float(c["fore"].min()),
            "at_s": float(np.asarray(knots)[i] / body.rate), "n_knots": int(len(knots)), "ok": bool(both[i] >= 0.0)}


GESTE_HAND = {"index": ((45.0, -6.0), 50.0, 25.0), "middle": (45.0, 55.0, 25.0), "ring": ((55.0, -5.0), 65.0, 30.0),
              "pinky": ((60.0, -10.0), 70.0, 30.0), "thumb": ((20.0, 10.0), 20.0, 10.0)}
#   [module] a loosely curled hand, the fingers together (the knuckle abductions of this module's RELAXED_HAND below,
#   -6, -5 and -10 deg; parkinsonism.RELAXED_HAND has its own), the ring and little fingers curled more so that the
#   index and middle fingertips lead the touch (review of 2026-11-03: the open hand with spread fingers read as a hand
#   raised before the face; wave SH1c: with the fingers barely flexed the front view still showed an open hand raised
#   beside the jaw, the back of the hand toward the viewer, so the fingers now curl and the fingertips rest on the
#   cheek with the palm turned toward the face)
GESTE_PATH_RATIO_MAX = 1.4   # [module] an arm whose middle fingertip travels more than this times its straight line
#                              over the reach is not used (reach_path; the review of 2026-11-03 measured 1.51 to 1.55)
GESTE_FORWARD_EXTRA_M = 0.05  # [module] nor one whose wrist comes further forward of the shoulder over the reach or the
#                              release than while the hand rests by more than this (reach_path: no sweep out in front)
GESTE_SNAP_DEG = 3.0         # [module] nor one whose collar, shoulder, elbow or wrist turns more than this in one knot
#                              beyond the level 0's own step (300 deg/s at 100 Hz: the snap check of the tests)
GESTE_FORWARD_EXTRA_MAX_M = 0.065  # [module] where no arm of the tried frames keeps GESTE_FORWARD_EXTRA_M, the arm that
#                              passes every other check with the least such excess, if within this (wave SH1c: with the
#                              curled hand no arm tried for the C2 clip kept GESTE_FORWARD_EXTRA_M; an earlier shoulder,
#                              GESTE_SHOULDER_SHARE 0.3, kept it there but turned the shoulder of C0 beyond
#                              GESTE_SNAP_DEG in one knot; the arm taken and its excess: the geste truth's params arm
#                              'forward_extra_m' and 'forward_relaxed'). Open-body port, wave 4 (D59): 0.06 was the
#                              bound that admitted the SMPL+H body's least excess; on the open body the tests' C2 clip
#                              (trick score 2: the head stays turned while the hand rests) has no arm under 6.07 cm of
#                              excess, so the bound is that excess rounded up to the half centimetre
#                              (scripts/openbody_wave4_checks.py, geste)


def measure_geste(base, out, side, times, level, chin_side=None, pad_target=None, contact_vertex=None):
    """{'held_deg' (mean head deviation over 1 s up to the reach), 'touch_deg' (mean over the last second before the
    release), 'reduction' (1 - touch / held), 'contact_gap_m' (largest distance of the pads' mean point from its
    target during the contact; with pad_target None, from the chin landmark), 'pad_contact_m' (the pads' mean point's
    least and largest distance from the contact vertex, default cheek_contact_vertex, during the contact), 'skin_m'
    (the least and largest distance to the skin of the head and neck of the pads' distal phalanges, and the least of
    the palm and of the forearm, during the contact every GESTE_CHECK_STEP knots: skin_distances), 'hand_place'
    (during the contact, in the head's frame with x toward the hand's side, y up, z forward: the wrist joint's least
    lateral offset from the head joint and the face's half width at the contact ('wrist_lateral_m', 'face_half_m'),
    the wrist's least depth below the chin landmark ('wrist_below_jaw_m', >= 0 at or below the jaw line), its
    forward offset ('wrist_forward_m'), the hand's long axis (wrist to middle knuckle) as its largest elevation above
    the head's horizontal and its least backward component ('axis_elevation_deg', 'axis_back'), the least cosine of
    the palmar normal with the direction from the palm centre to the contact vertex ('palm_to_face_cos'); in the neck
    joint's frame the wrist's least lateral offset ('wrist_lateral_neck_m') against the neck's half width
    (throat_ids, 'neck_half_m'); the least distance of the palm's vertices and of the wrist joint to the front of the
    neck ('throat_palm_m', 'throat_wrist_m')),
    'index_tip_head_centre_m' (the index fingertip vertex's least distance from the skeleton page's head centre,
    skeleton.head_landmarks(), the selected body model's, whose circle has a radius of 0.095 m)}, read from the
    bodies."""
    from .skeleton import head_landmarks
    dev = joint_deviation(base, out, 15)
    r = out.rate
    k0, k1 = int(round((times["reach_on_s"] - 1.0) * r)), int(round(times["reach_on_s"] * r))
    k2, k3 = int(round((times["release_s"] - 1.0) * r)), int(round(times["release_s"] * r))
    held, touch = float(dev[k0:k1].mean()), float(dev[k2:k3].mean())
    ka, kb = int(round((times["reach_on_s"] + times["reach_s"]) * r)), k3
    kk = list(range(ka, kb + 1))
    cv = cheek_contact_vertex(side) if contact_vertex is None else contact_vertex
    jaw = jaw_vertex(chin_side or side)
    tid, half = throat_ids()
    wid = _geste_vertex_sets(side)["palm"]
    X = H.body_points(out, [cv, jaw] + list(pad_vertices(side).values()), knots=kk)
    C, Jw, pads = X[:, 0], X[:, 1], X[:, 2:].mean(1)
    tgt = Jw if pad_target is None else np.asarray(pad_target)[kk]
    J, Gl = out.fk()
    G15, G12 = Gl[kk, 15].numpy(), Gl[kk, 12].numpy()
    h = J[kk, 15].numpy()
    w = H.WRIST[side]
    Wj = J[kk, w].numpy()
    sg = np.array([1.0 if side == "left" else -1.0, 1.0, 1.0])

    def hf(x, G=G15, o=h):
        return np.einsum("tji,tj->ti", G, x - o) * sg
    Jr, v, _ = H._rest()
    ax = _hand_axes(side)
    Gw = Gl[kk, w].numpy()
    axis = np.einsum("tji,tj->ti", G15, np.einsum("tij,j->ti", Gw, ax[:, 0])) * sg
    pn = np.einsum("tij,j->ti", Gw, ax[:, 1])
    palm = Wj + np.einsum("tij,j->ti", Gw, palm_offset(side).numpy())
    to_c = (C - palm) / np.linalg.norm(C - palm, axis=-1, keepdims=True)
    wh, jh, ch = hf(Wj), hf(Jw), hf(C)
    wn = hf(Wj, G12, J[kk, 12].numpy())
    sub = kk[::GESTE_CHECK_STEP]
    Y = H.body_points(out, np.r_[tid, wid], knots=sub)
    Wsub = J[sub, w].numpy()
    th_palm = min(float(np.linalg.norm(Y[i, len(tid):, None] - Y[i, None, :len(tid)], axis=-1).min())
                  for i in range(len(sub)))
    th_wrist = min(float(np.linalg.norm(Y[i, :len(tid)] - Wsub[i], axis=-1).min()) for i in range(len(sub)))
    hc = torch.tensor(head_landmarks()["head_centre"], dtype=F64)
    HC = (J[kk, 15] + torch.einsum("tij,j->ti", Gl[kk, 15], hc)).numpy()
    tip = H.body_points(out, [H.tip_vertices()[side]["index"]], knots=kk)[:, 0]
    pc = np.linalg.norm(pads - C, axis=-1)
    sk = skin_distances(out, side, sub)
    hand = {"wrist_lateral_m": float(wh[:, 0].min()), "face_half_m": float(ch[:, 0].max()),
            "wrist_below_jaw_m": float((jh[:, 1] - wh[:, 1]).min()),
            "wrist_forward_m": [float(wh[:, 2].min()), float(wh[:, 2].max())],
            "axis_elevation_deg": float(np.degrees(np.arcsin(np.clip(axis[:, 1], -1.0, 1.0))).max()),
            "axis_back": float((-axis[:, 2]).min()), "palm_to_face_cos": float((pn * to_c).sum(-1).min()),
            "wrist_lateral_neck_m": float(wn[:, 0].min()), "neck_half_m": half, "throat_palm_m": th_palm,
            "throat_wrist_m": th_wrist}
    return {"held_deg": held, "touch_deg": touch, "reduction": 1.0 - touch / held if held > 0 else 0.0,
            "contact_gap_m": float(np.linalg.norm(pads - tgt, axis=-1).max()),
            "pad_contact_m": [float(pc.min()), float(pc.max())],
            "skin_m": {"pads": [float(sk["pads"].min()), float(sk["pads"].max())], "palm_min": float(sk["palm"].min()),
                       "fore_min": float(sk["fore"].min())}, "hand_place": hand,
            "index_tip_head_centre_m": float(np.linalg.norm(tip - HC, axis=-1).min()),
            "level_deg_asked": float(level)}


# ---------------------------------------------------------------- hands and feet
FIST_POSE = None             # handtasks.FIST, read lazily (the fist of 3.5: clear of the palm)
THUMB_ABD_SHIFT_DEG = -15.0
#   [module] open-body port, wave 4 (design D50): Anny's rest thumb lies turned toward the palm, its tip 33.5 mm past
#   the palm's plane (design 14.2), where the old rest hand's lay flat; the three hand poses of the writing scene below,
#   authored on the old thumb, take this much less palmar abduction at the thumb's CMC, the least of 5 deg steps at
#   which the hand resting on the table rests on its fingertips and not on its thumb's tip (before: the left hand
#   stood on the thumb, the palm 49 mm and the fingertips 18 to 19 mm over the table; the writing thumb's tip dipped
#   0.9 mm into the paper); scripts/openbody_wave4_checks.py, table_hands
WRITING_HAND = {"index": (40.0, 40.0, 10.0), "middle": (45.0, 50.0, 20.0), "ring": (55.0, 65.0, 30.0),
                "pinky": (60.0, 65.0, 30.0), "thumb": ((35.0, 35.0 + THUMB_ABD_SHIFT_DEG), 20.0, 15.0)}
#   [module] a tripod pen grip: the thumb opposed (CMC flexion and palmar abduction) to the index and middle fingers,
#   the ring and little fingers curled under (review B5)
RELAXED_HAND = {"index": ((15.0, -6.0), 20.0, 5.0), "middle": (15.0, 20.0, 5.0), "ring": ((18.0, -5.0), 22.0, 5.0),
                "pinky": ((20.0, -10.0), 22.0, 5.0), "thumb": ((10.0, 10.0 + THUMB_ABD_SHIFT_DEG), 5.0, 5.0)}
#   [module] a relaxed hand: the fingers slightly flexed and together (review B5: the flat splayed hand)
CRAMP_HAND = {"index": (70.0, 70.0, 45.0), "middle": (65.0, 75.0, 40.0), "ring": (80.0, 85.0, 55.0),
              "pinky": (80.0, 85.0, 55.0), "thumb": ((42.0, 40.0 + THUMB_ABD_SHIFT_DEG), 45.0, 50.0)}
#   [module] the pen-grip dystonia at the fingers' full item score: the thumb and index squeezing the pen (added flexion
#   at their MCP and IP joints), the ring and little fingers curled (review B4: the move toward the fist laid the
#   knuckles on the table); the wrist flexion, ulnar deviation and pronation are WCRS_ANGLES'


def hand_pose_rows(pose, side):
    """(30, 3, 3) the hand rotations of a {finger: angles} pose (handtasks' format) on one side, the other at rest."""
    return H.hand_rotations({(side, f, lv): v for f, vals in pose.items() for lv, v in enumerate(vals)})


def fist_pose():
    from .handtasks import FIST
    return FIST


def hand_toward(R_hand, pose_from, pose_to, side, share):
    """(T, 30, 3, 3): the side's hand rows turned from pose_from (dict or None: the rows of R_hand) toward pose_to
    by shares (T,) along each joint's geodesic."""
    Rh = R_hand.clone()
    A = R_hand[0] if pose_from is None else hand_pose_rows(pose_from, side)
    B = hand_pose_rows(pose_to, side)
    rows = list(range(15)) if side == "left" else list(range(15, 30))
    W = torch.as_tensor(np.asarray(share, dtype=np.float64))
    for r in rows:
        D = matrix_to_axis_angle(A[r].T @ B[r])
        Rh[:, r] = A[r][None] @ axis_angle_to_matrix(W[:, None] * D[None])
    return Rh


# Foot dystonia (equinovarus): ankle plantarflexion (+x), inversion (-z left, +z right: the sole turned toward the
# midline) and adduction (-y left, +y right: the toes toward the midline) in the ankle's parent frame, and toe
# flexion at the foot joint (+x). The rooms below are the table's: the anatomical table (reports/openbody/
# joint_limits.json, jarm2022) holds the ankle's inversion of 30 deg and the toes' flexion of 35 deg that SH2 had
# widened the AMASS table to, and the dystonia table differs from it by the shoulder coupling only (design 12.1).
FOOT_DIRS = {"plantarflexion": ("x", {"left": 1.0, "right": 1.0}, 7),
             "inversion": ("z", {"left": -1.0, "right": 1.0}, 7),
             "adduction": ("y", {"left": -1.0, "right": 1.0}, 7), "toe_flexion": ("x", {"left": 1.0, "right": 1.0}, 10),
             "toe_extension": ("x", {"left": -1.0, "right": -1.0}, 10)}


def foot_terms(base, side, shares, env, k=0, table=HEALTHY):
    """Terms of a foot posture: {component of FOOT_DIRS: share of its room at knot k in table} (DIS amplitude as a
    share of the possible range, decision D3); the levels in degrees are in each term."""
    from .body import room_deg
    R = base.local_R()[k]
    box = LX.box_at(table)                   # read from the base itself: no collar has risen over it
    out = []
    for comp, sh in shares.items():
        axis, sg, j = FOOT_DIRS[comp]
        jj = j if side == "left" else j + 1
        room = room_deg(R[jj], jj, axis, sg[side], "parent", limits=box)
        out.append(Term((jj,), axis, sg[side], "parent", sh * room, env, (1.0,), f"foot {comp} ({sh:g} of {room:.1f})"))
    return out


TOUCH_BISECT = 28
LIFT_CAP_DEG = 10.0          # [module] standing: the hip and knee lift the foot by at most this much each (review M1)
HIKE_MAX_DEG = 12.0          # [module] standing: then the pelvis hikes on the foot's side by at most this much
#                              (the stance hip kept in place, the trunk kept upright); beyond, the lift resumes


def _lift_parts(u, seated, hike_max=HIKE_MAX_DEG):
    """(lam deg, hike deg) of the lift parameter u (deg): seated lam = u; standing lam = u up to LIFT_CAP_DEG, then
    the hike up to hike_max (HIKE_MAX_DEG, or less per knot where the table bounds it), then lam again."""
    u = np.asarray(u, dtype=np.float64)
    if seated:
        return u, np.zeros_like(u)
    hike = np.clip(u - LIFT_CAP_DEG, 0.0, hike_max)
    return u - hike, hike


def _aa_np(R):
    return matrix_to_axis_angle(torch.from_numpy(np.ascontiguousarray(R))).numpy()


def _lift_leg(Rx, tr, u, hmax, seated, side, rest):
    """(R, transl) of touch_floor's lift u (deg per knot, _lift_parts) applied to local rotations Rx (N, 22, 3, 3)
    and translations tr (N, 3): the hike first (a turn of the pelvis about its forward axis about the stance hip, the
    trunk and both thighs keeping their global rotations), then the hip and the knee of side flexed by lam each."""
    l, h = _lift_parts(u, seated, hmax)
    return _lift_apply(Rx, tr, l, h, side, rest)


def _lift_apply(Rx, tr, l, h, side, rest):
    """(R, transl): the pelvis hike h and the hip and knee lift l (deg per knot) of _lift_leg."""
    kn, hp = (4, 1) if side == "left" else (5, 2)
    hs = 2 if side == "left" else 1                   # the stance hip, kept in place by the hike
    sg = 1.0 if side == "left" else -1.0              # the hike turns the side's hip up about the forward axis
    l, h = np.asarray(l, dtype=np.float64), np.asarray(h, dtype=np.float64)
    Rx, tr = Rx.copy(), np.array(tr, dtype=np.float64)
    if (h > 0).any():
        G0 = Rx[:, 0].copy()
        Hk = CT._rot(G0[:, :, 2], sg * np.radians(h))
        G0n = Hk @ G0
        for j in (1, 2, 3):
            Rx[:, j] = np.swapaxes(G0n, -1, -2) @ (G0 @ Rx[:, j])
        Rx[:, 0] = G0n
        tr = tr + np.einsum("nij,j->ni", G0 - G0n, rest[hs] - rest[0])
    ex = np.repeat([[1.0, 0.0, 0.0]], len(Rx), 0)
    Rx[:, hp] = CT._rot(ex, -np.radians(l)) @ Rx[:, hp]
    Rx[:, kn] = CT._rot(ex, np.radians(l)) @ Rx[:, kn]
    return Rx, tr


def touch_floor(base, R, side, active, floor=None, env=None, grid=41, record=None):
    """(R, lam deg (T,), transl (T, 3), hike deg (T,)): the leg of side lifted by the smallest amount that brings the
    lowest skinned sole point of that foot back to the floor (module docstring: the foot touches, unloaded) at the
    active knots; where the posture leaves the foot above the floor nothing moves. The lift flexes the hip and the
    knee by lam each, so that the shank keeps its direction (a knee flexion alone first dips the toes): seated the
    thigh rises off the seat and the foot pivots on its forefoot, the knee angle kept (review B6: a knee extension
    slid the foot forward and hid the equinus); standing lam is capped at LIFT_CAP_DEG and the pelvis then hikes on
    that side (_lift_parts: a rotation about the pelvis's forward axis about the stance hip, the trunk and both thighs
    keeping their global rotations, so that the stance leg does not move; review M1: a larger lift read as a
    relaxed bent leg), the hike bounded per knot so that the hips and spine1 stay inside the table. floor (T,): the
    floor heights (default the base's own lowest sole point of that foot); base gives the translation. env (T,): when
    the posture is env times a fixed posture on a static leg, the lift is solved on a grid of env values (grid) and
    interpolated, so that it is a smooth function of the envelope. record (a dict): receives the lift parameter u
    (deg, (T,)) and the hike cap (deg, (T,)) solved, for a later solve on the same lift (pin_roll_standing)."""
    kn = 4 if side == "left" else 5
    i = 0 if side == "left" else 1
    lb = CT.mesh_sole_low(base)[:, i] if floor is None else np.asarray(floor, dtype=np.float64)
    seated = float(np.degrees(matrix_to_axis_angle(base.local_R()[0, kn]).numpy())[0]) > 45.0
    rest = base.rest().numpy()
    R = R.clone()
    T = base.T
    lam, hike = np.zeros(T), np.zeros(T)
    tr0 = base.transl.numpy().copy()

    def lifted(Rx, tr, u, hmax=HIKE_MAX_DEG):
        return _lift_leg(Rx, tr, u, hmax, seated, side, rest)
    kk = np.flatnonzero(active)
    if not len(kk):
        return R, lam, base.transl.clone(), hike
    Rn = R.numpy()
    betas = base.betas.numpy()

    lim = joint_limits()

    def hike_cap(Rs, tr):                         # the largest hike keeping the pelvis's children inside the table
        rows = [1, 2, 3]
        c0 = np.degrees(_aa_np(Rs[:, rows]))
        lo, hi = np.minimum(lim.lo[rows][None], c0) - 1e-6, np.maximum(lim.hi[rows][None], c0) + 1e-6

        def inside(h):
            Rx, _ = lifted(Rs, tr, LIFT_CAP_DEG + h, h)
            c = np.degrees(_aa_np(Rx[:, rows]))
            return ((c >= lo) & (c <= hi)).all(-1).all(-1)
        if seated:
            return np.zeros(len(Rs))
        a, b = np.zeros(len(Rs)), np.full(len(Rs), HIKE_MAX_DEG)
        full = inside(b)
        for _ in range(16):
            m = 0.5 * (a + b)
            f = inside(m)
            a, b = np.where(f, m, a), np.where(f, b, m)
        return np.where(full, HIKE_MAX_DEG, a)

    def solve(Rs, tr, target):
        hm = hike_cap(Rs, tr)

        def low(u):
            Rx, t2 = lifted(Rs, tr, u, hm)
            return CT.skin_points(Rx, t2, betas, base.gender, side)[..., 1].min(1)
        need = low(np.zeros(len(Rs))) < target - 1e-9
        a, b = np.zeros(len(Rs)), np.full(len(Rs), 80.0)
        for _ in range(TOUCH_BISECT):
            m = 0.5 * (a + b)
            under = low(m) < target
            a, b = np.where(under, m, a), np.where(under, b, m)
        return np.where(need, b, 0.0), hm
    u, hcap = np.zeros(T), np.zeros(T)
    legs = Rn[:, [0, kn, 1 if side == "left" else 2]].reshape(T, -1)
    static = (np.abs(legs - legs[:1]).max() < 1e-12 and float((base.transl - base.transl[:1]).abs().max()) < 1e-12
              and np.ptp(lb) < 1e-9)
    if env is not None and static:
        e = np.asarray(env, dtype=np.float64)
        ks = [int(np.argmin(np.abs(e - g))) for g in np.linspace(e[kk].min(), e[kk].max(), grid)]
        ks = sorted(set(ks))
        ev = e[ks]
        ug, hg = solve(Rn[ks], tr0[ks], lb[ks])
        order = np.argsort(ev)
        u[kk] = np.interp(e[kk], ev[order], ug[order])
        hcap[kk] = np.interp(e[kk], ev[order], hg[order])
    else:
        first, inv = CT._unique_rows(Rn[kk][:, [0, 1, 2, 3, kn, 7, 8, 10, 11]], tr0[kk])
        sel = kk[first]
        us, hs_ = solve(Rn[sel], tr0[sel], lb[sel])
        u[kk], hcap[kk] = us[inv], hs_[inv]
    lam, hike = _lift_parts(u, seated, hcap)
    Rx, tr = lifted(R.numpy(), tr0, u, hcap)
    if record is not None:
        record.update(u=u.copy(), hcap=hcap.copy(), seated=seated)
    return torch.from_numpy(Rx), lam, torch.from_numpy(tr), hike


PIN_ITERS = 6                # [module] Newton steps of the contact pin (pin_contact)
PIN_STEP_DEG = 5.0           # [module] the largest turn of one Newton step
PIN_JUMP_DEG = 1.0           # [module] a knot's solution farther than this from its neighbour's is solved again from it
PIN_DOWN_M = 0.001           # [module] standing: the pinned point touches down at the first active knot where it lies
#                              within this of the lowest sole point; from there it keeps its horizontal place
PIN_ROUNDS = 3               # [module] standing: the pin and touch_floor's height alternated, ending on the pin
ROLL_FLOOR_TOL_MM = 0.1      # [module] standing: a roll whose lowest sole point misses the floor by more than this at
#                              some knot is refused for the touchdown pin (the hike at its cap: pin_roll_standing)
ROLL_ONSET_SHARE = 0.25      # [module] standing: a foot posture whose envelope at its first active knot lies under this
#                              share of its largest rises from rest (an onset: the foot turns from flat onto its
#                              border) and is rolled (pin_roll_standing); one held from its first knot is pinned from
#                              its touchdown (pin_contact_standing)
ROLL_ITERS = 16              # [module] damped Newton steps of the standing roll about the lateral border
#                              (pin_roll_standing)
ROLL_DAMP0 = 1e-6            # [module] their first damping, a share of the mean diagonal of the normal matrix
ROLL_BORDER_M = 0.015        # [module] the lateral border: the sole cloud's vertices within this of its most lateral
#                              one at rest, across the foot
ROLL_ROUNDS = 3              # [module] the roll solved again on the border vertex its foot rests on, at most this often
PIN_EPS_DEG = 0.05           # [module] the finite-difference step of its Jacobian
PIN_HELD_SHARE = 0.9         # [module] the pinned sole point: the lowest one most often over the knots whose
#                              envelope reaches this share of its largest value


def pin_contact(base, R, tr, side, active, floor, env):
    """(R, record): the touching foot pinned on the floor (wave SH1d part B, review M11: touch_floor restored only
    the height of the lowest sole point, so the foot glided 15 to 125 mm while touching): the sole cloud's vertex that
    is the lowest most often over the held knots (the envelope above PIN_HELD_SHARE of its largest) keeps, at every
    active knot, the horizontal place it has on the base (the foot rolls and pivots about it), while the lowest sole
    point stays at the floor height floor (T,): three equations per knot solved by Newton steps (PIN_ITERS, a
    finite-difference Jacobian) on the hip's turns about its parent's x and z axes and the knee's flexion, from the
    rows of R; a knot whose solution jumps from its neighbour's by more than PIN_JUMP_DEG is solved again from the
    neighbour's (the solve kept on one branch along the clip). The record holds the vertex and its travel before and
    after."""
    kn = 4 if side == "left" else 5
    hp = 1 if side == "left" else 2
    kk = np.flatnonzero(active)
    rec = {"rule": "pin_contact (module)", "side": side}
    if not len(kk):
        return R, dict(rec, vertex_in_cloud=None)
    betas, g = base.betas.numpy(), base.gender
    Rn = R.numpy().copy()
    trn = np.asarray(tr, dtype=np.float64)
    e = np.asarray(env, dtype=np.float64)
    held = kk[e[kk] >= PIN_HELD_SHARE * e[kk].max()]
    cloud = CT.skin_points(Rn[held], trn[held], betas, g, side)
    m = int(np.bincount(cloud[..., 1].argmin(1)).argmax())
    P0 = CT.skin_points(base.local_R().numpy()[kk], base.transl.numpy()[kk], betas, g, side)[:, m]
    goal_all = np.stack([P0[:, 0], P0[:, 2], np.asarray(floor, dtype=np.float64)[kk]], 1)
    h = np.radians(PIN_EPS_DEG)

    def pose(q, idx):
        Rx = Rn[kk[idx]].copy()
        ex = np.repeat([[1.0, 0.0, 0.0]], len(idx), 0)
        ez = np.repeat([[0.0, 0.0, 1.0]], len(idx), 0)
        Rx[:, hp] = CT._rot(ex, q[:, 0]) @ CT._rot(ez, q[:, 1]) @ Rx[:, hp]
        Rx[:, kn] = CT._rot(ex, q[:, 2]) @ Rx[:, kn]
        return Rx

    def f(q, idx):
        C = CT.skin_points(pose(q, idx), trn[kk[idx]], betas, g, side)
        return np.stack([C[:, m, 0], C[:, m, 2], C[..., 1].min(1)], 1) - goal_all[idx]

    def newton(q, idx, iters=PIN_ITERS):
        for _ in range(iters):
            r = f(q, idx)
            Jm = np.stack([(f(q + h * np.eye(3)[c], idx) - r) / h for c in range(3)], -1)
            step = np.linalg.solve(Jm + 1e-9 * np.eye(3), r[..., None])[..., 0]
            q = q - np.clip(step, -np.radians(PIN_STEP_DEG), np.radians(PIN_STEP_DEG))
        return q
    allk = np.arange(len(kk))
    before = f(np.zeros((len(kk), 3)), allk)
    q = newton(np.zeros((len(kk), 3)), allk)
    redone = 0
    lim = np.radians(PIN_JUMP_DEG)
    for i in range(1, len(kk)):                     # one branch along the clip: a jump is solved from the neighbour
        if kk[i] == kk[i - 1] + 1 and np.abs(q[i] - q[i - 1]).max() > lim:
            q[i:i + 1] = newton(q[i - 1:i].copy(), np.array([i]), 2 * PIN_ITERS)
            redone += 1
    after = f(q, allk)
    Rn[kk] = pose(q, allk)
    trav = lambda x: float(np.hypot(x[:, 0], x[:, 1]).max())  # noqa: E731
    rec.update(vertex_in_cloud=m, travel_before_mm=1000.0 * trav(before), travel_after_mm=1000.0 * trav(after),
               floor_error_after_mm=1000.0 * float(np.abs(after[:, 2]).max()), knots_resolved=redone,
               turn_deg_max=[float(np.degrees(np.abs(q[:, i])).max()) for i in range(3)])
    return torch.from_numpy(Rn), rec


def pin_contact_standing(base, R, tr, side, active, env, m=None, goal_xz=None):
    """(R, record): the standing variant of pin_contact (wave SH1d part B): pinned on the level 0 the standing leg
    passed through a straight knee into hyperextension (3 deg) and the layer's leg solve flipped there (284 m/s2 at
    the knee of post_E7_hemidystonia). Here the vertex m (default: the lowest most often while held) keeps, from the
    knot it touches down (PIN_DOWN_M) on, the horizontal place goal_xz it has there (a small correction: the foot
    rolls about it), by the hip's two turns only; the height stays touch_floor's (limb_posture alternates the two,
    ending on the pin, PIN_ROUNDS)."""
    hp = 1 if side == "left" else 2
    kk = np.flatnonzero(active)
    rec = {"rule": "pin_contact_standing (module)", "side": side}
    if not len(kk):
        return R, dict(rec, vertex_in_cloud=None, touchdown_s=None)
    betas, g = base.betas.numpy(), base.gender
    Rn = R.numpy().copy()
    trn = np.asarray(tr, dtype=np.float64)
    e = np.asarray(env, dtype=np.float64)
    C = CT.skin_points(Rn[kk], trn[kk], betas, g, side)
    if m is None:
        held = e[kk] >= PIN_HELD_SHARE * e[kk].max()
        m = int(np.bincount(C[held][..., 1].argmin(1)).argmax())
    down = np.flatnonzero(C[:, m, 1] - C[..., 1].min(1) < PIN_DOWN_M)
    if not len(down):
        return R, dict(rec, vertex_in_cloud=m, touchdown_s=None)
    k_td = int(kk[down[0]])
    if goal_xz is None:
        goal_xz = [float(C[down[0], m, 0]), float(C[down[0], m, 2])]
    rec.update(vertex_in_cloud=m, touchdown_s=k_td / base.rate, goal_xz_m=list(goal_xz))
    kk = kk[kk >= k_td]
    goal = np.asarray(goal_xz, dtype=np.float64)[None]
    ex, ez = np.repeat([[1.0, 0.0, 0.0]], len(kk), 0), np.repeat([[0.0, 0.0, 1.0]], len(kk), 0)
    R0 = Rn[kk]

    def pose(q):
        Rx = R0.copy()
        Rx[:, hp] = CT._rot(ex, q[:, 0]) @ CT._rot(ez, q[:, 1]) @ R0[:, hp]
        return Rx

    def f(q):
        return CT.skin_points(pose(q), trn[kk], betas, g, side)[:, m][:, [0, 2]] - goal

    h = np.radians(PIN_EPS_DEG)
    q = np.zeros((len(kk), 2))
    before = f(q)
    for _ in range(2 * PIN_ITERS):
        r = f(q)
        Jm = np.stack([(f(q + h * np.eye(2)[c]) - r) / h for c in range(2)], -1)
        step = np.linalg.solve(Jm + 1e-9 * np.eye(2), r[..., None])[..., 0]
        q = q - np.clip(step, -np.radians(PIN_STEP_DEG), np.radians(PIN_STEP_DEG))
    after = f(q)
    Rn[kk] = pose(q)
    trav = lambda x: float(np.hypot(x[:, 0], x[:, 1]).max())  # noqa: E731
    rec.update(travel_before_mm=1000.0 * trav(before), travel_after_mm=1000.0 * trav(after),
               turn_deg_max=[float(np.degrees(np.abs(q[:, i])).max()) for i in range(2)])
    return torch.from_numpy(Rn), rec


def _has_onset(env):
    """True when a foot posture's envelope env (T,) rises from rest: at its first active knot it lies under
    ROLL_ONSET_SHARE of its largest value (the off-period foot, held_env from 0.5 s); False for one held from its first
    knot (hemidystonia, multifocal, PED, the torsion dystonias)."""
    e = np.asarray(env, dtype=np.float64)
    kk = np.flatnonzero(e > 0)
    return bool(len(kk)) and bool(e[kk[0]] < ROLL_ONSET_SHARE * e[kk].max())


def _lateral_border(base, side, band=None):
    """(M,) bool: the sole cloud's vertices on the lateral border of the foot of side, those whose rest position (the
    base's first knot) lies within band (ROLL_BORDER_M) of the cloud's most lateral one, read across the body (+x is
    the body's left, so the left foot's lateral side is +x)."""
    band = ROLL_BORDER_M if band is None else band
    C = CT.skin_points(base.local_R().numpy()[:1], base.transl.numpy()[:1], base.betas.numpy(), base.gender, side)[0]
    lat = C[:, 0] if side == "left" else -C[:, 0]
    return lat >= lat.max() - band


def pin_roll_standing(base, R_in, side, active, floor, env, lift):
    """(R, lam deg (T,), transl (T, 3), hike deg (T,), record): the standing dystonic foot rolled about its lateral
    border (Phase D of docs/PLAN_2026-11-12.md). A foot that inverts and plantarflexes while it touches the floor
    rests on its lateral border, which stays on the floor while the foot turns about it: the contact vertex m is the
    sole cloud's vertex of the lateral border (_lateral_border) that is the lowest most often over the held knots
    (the envelope above PIN_HELD_SHARE of its largest) under touch_floor's lift, chosen again on the rolled foot
    (ROLL_ROUNDS), and it keeps, at every active knot from the first, the horizontal place it has at the first one,
    while the lowest sole point stays at the floor height floor (T,). Three equations per knot, solved together by
    damped Newton steps (ROLL_ITERS, Levenberg-Marquardt from ROLL_DAMP0, a finite-difference Jacobian) on the hip's
    turns about its parent's x and z axes and the leg's height: touch_floor's pelvis hike (record lift) changed first,
    inside 0 and its cap, the rest taken by touch_floor's hip and knee lift, never below the posture's (the knee never
    extended past it), from R_in (the posture before the lift) and the base's translation (_lift_apply); a knot whose
    solution jumps from its neighbour's by more than PIN_JUMP_DEG is solved again from the neighbour's.
    limb_posture rolls a foot whose posture rises from rest (_has_onset) and keeps the roll where its lowest sole point
    holds the floor within ROLL_FLOOR_TOL_MM; a foot held from its first knot, or whose roll misses the floor, keeps
    pin_contact_standing (pinned from its touchdown). The touchdown pin, on the off-period foot until
    2026-11-12, never acted there: its vertex never came within PIN_DOWN_M of the lowest sole point (2.5 to 3 mm
    above it), so the foot slid as it inverted and adducted (the page's skate check 31.67 mm), and the variants that
    pinned from the first knot sank the foot (12.6 mm) or stepped the hip and the knee. On the feet held from their
    first knot the roll does worse than the touchdown pin: there the hike stands at its cap and the hip and knee lift
    raises the foot mostly by moving it forward (a 3 s FOOT_STAND foot sank 9.3 mm, hemidystonia's lift stepped 8.2
    deg). The record holds the vertex, its rest place, the first knot it lies within PIN_DOWN_M of the lowest sole
    point (touchdown_s), its travel before and after, the floor error and the largest turns."""
    hp = 1 if side == "left" else 2
    kk = np.flatnonzero(active)
    seated = bool(lift.get("seated", False))
    rec = {"rule": "pin_roll_standing (module)", "side": side}
    rest = base.rest().numpy()
    tr0 = base.transl.numpy().astype(np.float64)
    Rn0 = R_in.numpy().copy()
    u0 = np.asarray(lift["u"], dtype=np.float64)
    hcap = np.asarray(lift["hcap"], dtype=np.float64)
    lam0, hike0 = _lift_parts(u0, seated, hcap)
    R_all, tr_all = _lift_apply(Rn0, tr0, lam0, hike0, side, rest)
    if not len(kk):
        return torch.from_numpy(R_all), lam0, torch.from_numpy(tr_all), hike0, dict(rec, vertex_in_cloud=None)
    betas, g = base.betas.numpy(), base.gender
    e = np.asarray(env, dtype=np.float64)
    flo = np.asarray(floor, dtype=np.float64)
    border = _lateral_border(base, side)
    C = CT.skin_points(R_all[kk], tr_all[kk], betas, g, side)
    held = e[kk] >= PIN_HELD_SHARE * e[kk].max()
    ex = np.repeat([[1.0, 0.0, 0.0]], len(kk), 0)
    ez = np.repeat([[0.0, 0.0, 1.0]], len(kk), 0)

    def contact_vertex(P):                           # the border vertex lowest most often while held
        yb = np.where(border[None], P[held][..., 1], np.inf)
        return int(np.bincount(yb.argmin(1), minlength=P.shape[1]).argmax())

    def parts(e, idx):                               # the vertical: the hike first, inside [0, its cap], the rest
        k = kk[idx]                                  # on the hip and knee lift (never below the posture's)
        v = hike0[k] + np.degrees(e)
        hk = np.clip(v, 0.0, hcap[k])
        return np.maximum(lam0[k] + v - hk, 0.0), hk

    def pose(q, idx):
        Rx = Rn0[kk[idx]].copy()
        Rx[:, hp] = CT._rot(ex[idx], q[:, 0]) @ CT._rot(ez[idx], q[:, 1]) @ Rx[:, hp]
        return _lift_apply(Rx, tr0[kk[idx]], *parts(q[:, 2], idx), side, rest)

    def f(q, idx):
        Rx, t2 = pose(q, idx)
        P = CT.skin_points(Rx, t2, betas, g, side)
        return np.stack([P[:, m, 0] - goal[0], P[:, m, 2] - goal[1], P[..., 1].min(1) - flo[kk[idx]]], 1)

    h = np.radians(PIN_EPS_DEG)
    step_max = np.radians(PIN_STEP_DEG)

    def newton(q, idx, iters=ROLL_ITERS):            # damped (Levenberg-Marquardt): a step is kept where it lowers
        r = f(q, idx)                                # the residual, the damping raised where it does not (at the hike's
        mu = np.full(len(idx), ROLL_DAMP0)           # cap the lift raises the foot mostly by moving it forward)
        for _ in range(iters):
            Jm = np.stack([(f(q + h * np.eye(3)[c], idx) - r) / h for c in range(3)], -1)
            A = np.swapaxes(Jm, -1, -2) @ Jm
            g = (np.swapaxes(Jm, -1, -2) @ r[..., None])[..., 0]
            lam_ = mu[:, None, None] * (np.trace(A, axis1=-2, axis2=-1)[:, None, None] / 3.0 + 1e-12) * np.eye(3)
            step = np.clip(np.linalg.solve(A + lam_, g[..., None])[..., 0], -step_max, step_max)
            q_new = q - step
            r_new = f(q_new, idx)
            better = np.linalg.norm(r_new, axis=1) < np.linalg.norm(r, axis=1)
            q, r = np.where(better[:, None], q_new, q), np.where(better[:, None], r_new, r)
            mu = np.where(better, mu / 3.0, mu * 4.0)
        return q
    allk = np.arange(len(kk))
    q0 = np.zeros((len(kk), 3))
    lim = np.radians(PIN_JUMP_DEG)
    m, tried = contact_vertex(C), []
    for r_ in range(ROLL_ROUNDS):                    # the vertex the rolled foot rests on, pinned in its turn
        tried.append(m)
        goal = np.array([C[0, m, 0], C[0, m, 2]])
        before = f(q0, allk)
        q = newton(q0.copy(), allk)
        redone = 0
        for i in range(1, len(kk)):                 # one branch along the clip: a jump is solved from the neighbour
            if kk[i] == kk[i - 1] + 1 and np.abs(q[i] - q[i - 1]).max() > lim:
                q[i:i + 1] = newton(q[i - 1:i].copy(), np.array([i]), 2 * ROLL_ITERS)
                redone += 1
        m_next = contact_vertex(CT.skin_points(*pose(q, allk), betas, g, side))
        if m_next == m or m_next in tried or r_ == ROLL_ROUNDS - 1:
            break
        m = m_next
    after = f(q, allk)
    Rx, tx = pose(q, allk)
    lam, hike = lam0.copy(), hike0.copy()
    lam[kk], hike[kk] = parts(q[:, 2], allk)
    R_all[kk], tr_all[kk] = Rx, tx
    P = CT.skin_points(Rx, tx, betas, g, side)
    down = np.flatnonzero(P[:, m, 1] - P[..., 1].min(1) < PIN_DOWN_M)
    trav = lambda x: float(np.hypot(x[:, 0], x[:, 1]).max())  # noqa: E731
    C0 = CT.skin_points(base.local_R().numpy()[:1], tr0[:1], betas, g, side)[0, m]
    rec.update(vertex_in_cloud=m, vertex_rest_m=[float(x) for x in C0], pinned_from_s=float(kk[0] / base.rate),
               touchdown_s=float(kk[down[0]] / base.rate) if len(down) else None, goal_xz_m=[float(x) for x in goal],
               travel_before_mm=1000.0 * trav(before[:, :2]), travel_after_mm=1000.0 * trav(after[:, :2]),
               floor_error_after_mm=1000.0 * float(np.abs(after[:, 2]).max()), knots_resolved=redone,
               turn_deg_max=[float(np.degrees(np.abs(q[:, 0])).max()), float(np.degrees(np.abs(q[:, 1])).max())],
               hike_change_deg_max=float(np.abs(hike[kk] - hike0[kk]).max()),
               lift_change_deg_max=float(np.abs(lam[kk] - lam0[kk]).max()),
               hike_at_cap_knots=int((hike[kk] >= hcap[kk] - 1e-9).sum()), vertices_tried=tried)
    return torch.from_numpy(R_all), lam, torch.from_numpy(tr_all), hike, rec


def heel_rise_cm(base, out, side):
    """The largest rise (cm) of the heel (the most posterior skinned vertex of the ankle and foot segments, chosen on
    the rest mesh) against the base, over the knots."""
    J, v, dom = H._rest()
    a, f = (7, 10) if side == "left" else (8, 11)
    sel = np.flatnonzero(np.isin(dom, (a, f)))
    heel = int(sel[np.argmin(v[sel, 2])])
    yb = H.body_points(base, [heel])[:, 0, 1]
    yo = H.body_points(out, [heel])[:, 0, 1]
    return float(100 * (yo - yb).max())


def foot_validity(base, out, side, table=HEALTHY):
    """validity() with the other foot's stance checked (its joint limits read against table, limits_ext.check), the
    moved foot's lowest skinned point against the base's (contact.sole_report, planted: touching within
    contact.TOL_M, never below), and the leg's continuity."""
    v = LX.check(table, base, out, feet=(1,) if side == "left" else (0,))
    v["contact"] = CT.sole_report(base, out, side, planted=True)
    v["continuity"] = CT.step_report(base, out, side)
    return v


# ---------------------------------------------------------------- limb dystonia and the distribution of dystonia
DIS_SHARE = {1: G.dis_share(1), 2: G.dis_share(2), 3: G.dis_share(3), 4: G.dis_share(4)}


def limb_posture(base, parts, env, foot=None, hand=None, label="limb dystonia", grade=None, scale="DIS",
                 sources=("dis", "albanese2013", "module"), stand=False, extra_terms=(), knee_deg=0.0, pre_terms=(),
                 hang=(), foot_env=None, hang_collar=True, table=DYST):
    """(Body, truth) of a composed sustained posture on one time course env (T,): parts [(region, direction, side,
    share of the region's room at knot 0)] of dystonia.py's regions (DIS amplitude, decision D3), foot (side,
    {component: share}) an equinovarus foot that touches the floor (touch_floor; on foot_env, default env), hand
    (side, pose_to, share) the hand turned toward a pose, extra_terms ready Terms; pre_terms: Terms composed first
    (a trunk twist, a shoulder elevation), after which the arms of hang keep hanging (hang_arms), the parts and the
    extra terms then acting on the hanging arms; stand: the trunk and pelvis terms solved standing (plant_stand)
    before the foot is placed. table: the joint-limit table (limits_ext; the dystonia table by default, SH2). A
    HandBody base keeps its hands."""
    b = H.hands_of(base)
    R, s = fit_terms(b.local_R(), list(pre_terms), table=table) if pre_terms else (b.local_R(), np.ones(b.T))
    hang_info = None
    if hang:
        R, hang_info = hang_arms(b, R, tuple(hang), collar=hang_collar, table=table)
    terms = [region_term(b, r, d, sd, sh * region_room(b, r, d, sd, table=table), env, f"{r} {d} {sd}", table=table)
             for r, d, sd, sh in parts]
    terms += list(extra_terms)
    R, s1 = fit_terms(R, terms, table=table)
    s = np.minimum(s, s1)
    terms = list(pre_terms) + terms
    info = {}
    if stand:                                            # trunk and pelvis first, both feet planted
        o, info = plant_stand(b, R, knee_deg * np.ones(b.T))
        R = o.local_R()
        b2 = H.hands_of(replace(b, transl=o.transl))
    else:
        b2 = b
    lam, hike = np.zeros(b.T), np.zeros(b.T)
    fenv = env if foot_env is None else foot_env
    if foot is not None:                                 # then the foot, touching the base's floor
        ft = foot_terms(b, foot[0], foot[1], fenv, table=table)
        R, s2 = fit_terms(R, ft, table=table)
        s = np.minimum(s, s2)
        terms += ft
        flo = CT.mesh_sole_low(b)[:, 0 if foot[0] == "left" else 1]
        R_in, lift = R, {}
        R, lam, tr, hike = touch_floor(b2, R, foot[0], fenv > 0, flo, None if stand else fenv, record=lift)
        kn_ = 4 if foot[0] == "left" else 5
        seated_leg = float(np.degrees(matrix_to_axis_angle(b.local_R()[0, kn_]).numpy())[0]) > 45.0
        if seated_leg:                   # seated: pinned on the level 0's place (pin_contact)
            R, pin = pin_contact(b, R, tr.numpy(), foot[0], fenv > 0, flo, fenv)
        else:
            roll, refused = None, None
            if _has_onset(fenv):         # standing, rising from rest: rolled about the lateral border (Phase D),
                roll = pin_roll_standing(b2, R_in, foot[0], fenv > 0, flo, fenv, lift)   # kept where it holds the
                if roll[4]["floor_error_after_mm"] > ROLL_FLOOR_TOL_MM:                    # floor
                    roll, refused = None, roll[4]["floor_error_after_mm"]
            if roll is not None:
                R, lam, tr, hike, pin = roll
            else:                        # standing, held from the first knot (or the roll refused): from the
                pin = {}                 # touchdown, alternated with touch_floor's height
            for r_ in range(PIN_ROUNDS if roll is None else 0):
                if r_:
                    R, lam2, tr, hike2 = touch_floor(replace(b2, transl=tr), R, foot[0], fenv > 0, flo, None)
                    lam, hike = lam + lam2, hike + hike2
                R, pin = pin_contact_standing(b, R, tr.numpy(), foot[0], fenv > 0, fenv, pin.get("vertex_in_cloud"),
                                              pin.get("goal_xz_m"))
            if refused is not None:
                pin["roll_refused_floor_error_mm"] = float(refused)
        b2 = replace(b2, transl=tr)
    rows = sorted({j for t in terms for j in t.chain} | ({0, 1, 2, 3, 4} if foot and foot[0] == "left" else
                                                        {0, 1, 2, 3, 5} if foot else set())
                  | ({0, 1, 2, 4, 5, 7, 8} if stand else set()) | ({j for sd in hang for j in ARM[sd][:2]}))
    out = b2.with_rotations(R, rows)
    projected = 0.0
    if stand or foot is not None:              # the leg solves may graze the table's edge (SH2 wave: 0.005 deg at the
        legs = [1, 2, 4, 5, 7, 8]              # left hip of the severity-3 torsion dystonia): clamped, never further
        Rp, projected = project_rows(out.local_R(), legs, b.local_R(), table, tol=LIMIT_TOL_DEG)
        if projected > 0.0:
            out = out.with_rotations(Rp, legs)
    if hand is not None:
        out = out.with_hands(hand_toward(out.hand_R(), None, hand[1], hand[0], hand[2] * env))
    out = replace(out, contact=contact_labels(out))
    if foot is not None:
        val = foot_validity(b, out, foot[0], table)
    elif stand:
        val = stand_validity(b, out, table)
    else:
        val = LX.check(table, b, out)
    val["hand_limits"] = {"ok": bool(H.hand_limit_report(out.hand_R())["ok"])}
    devs = {t.label: float(joint_deviation(b, out, t.chain[-1], segment=len(t.chain) > 1).max()) for t in terms}
    collars = tuple(c for c in (13, 14) if any(c in t.chain for t in pre_terms))
    params = {"terms": [t.record() for t in terms], "limit_scale_min": float(s.min()),
              "realized_deviation_deg": devs, "held_share": float((env >= 0.5).mean()),
              "foot": None if foot is None else {"side": foot[0], "shares": foot[1],
                                                 "knee_lift_deg_max": float(lam.max()),
                                                 "pelvis_hike_deg_max": float(hike.max()),
                                                 "heel_rise_cm": heel_rise_cm(b, out, foot[0]), "pin": pin},
              "hand": None if hand is None else {"side": hand[0], "share": hand[2]}, "stand": info,
              "arms_hang": hang_info, "arm_under_collar": arm_hang_report(b, out, collars) if hang else None,
              "legs_projected_deg": projected,
              "limits_table": LX.record(table)}
    return out, truth_record(label, None, grade, scale, params,
                             tuple(sources) + (("limits_ext",) if table == DYST else ()), val,
                             arrays={"envelope": env, "knee_lift": lam, "pelvis_hike": hike})


def held_env(T, rate, rng, on_s=0.5, ramp_s=1.5, fluct=FLUCT, fixed=False):
    """(T,) a posture rising from on_s over ramp_s and held with a slow fluctuation of +-fluct (fixed: 1 from the
    first knot, no fluctuation: the fixed dystonia that does not vary)."""
    if fixed:
        return np.ones(T)
    e = ramp_env(T, rate, on_s, ramp_s)
    return np.clip(e * (1.0 - fluct + fluct * smooth_noise(T, rate, rng)), 0.0, 1.0)


def task_specific_arm(grade, side="left", latency_s=2.0, ramp_s=1.5, duration_s=30.0, seed=0):
    """(Body, truth): task-specific arm dystonia on the raise to the 3.15 posture (motions.outstretched): absent at
    rest, a dystonic posture of the wrist (flexion), the forearm (pronation) and the fingers after the arms reach the
    posture (onset + latency_s, a module value), at the DIS amplitude share of grade (DIS in action; BFM provoking
    factor 1, on a particular action)."""
    from .body import bone_axis
    from .motions import outstretched
    b, info = outstretched(duration_s)
    rng = as_rng(seed)
    env = held_env(b.T, b.rate, rng, info["onset_s"] + latency_s, ramp_s)
    sh = DIS_SHARE[grade]
    e, w = (18, 20) if side == "left" else (19, 21)
    ax = bone_axis(e, w)
    sg = 1.0 if side == "left" else -1.0                 # pronation: the opposite of body.SUPINATION_SIGN
    Rk = b.local_R()[0, e]
    room_p = 0.0
    for deg in np.arange(0.5, 90.0, 0.5):                # the room of the twist about the forearm bone (child frame)
        Rt = Rk @ axis_rotation(ax, torch.as_tensor(np.radians(sg * deg)))
        c = np.degrees(matrix_to_axis_angle(Rt).numpy())
        lim = joint_limits()
        if ((c < lim.lo[e]) | (c > lim.hi[e])).any():
            break
        room_p = deg
    extra = [Term((e,), ax, sg, "child", sh * room_p, env, (1.0,), f"forearm pronation ({sh:g} of {room_p:.1f})")]
    out, tr = limb_posture(b, [("wrist", "flexion", side, sh), ("distal_arm", "elbow_flexion", side, 0.5 * sh)], env,
                           hand=(side, fist_pose(), 0.6 * sh / DIS_SHARE[3]), extra_terms=extra,
                           label="task_specific_arm_dystonia", grade=grade, scale="DIS (action)",
                           sources=("dis", "bfm", "albanese2013", "module"))
    tr["params"].update(onset_s=info["onset_s"] + latency_s, latency_s=latency_s, base_motion="outstretched",
                        rest_deviation_deg=float(joint_deviation(b, out, w)[: int(info["onset_s"] * b.rate)].max()))
    return out, tr


# ---------------------------------------------------------------- writing (writer's cramp, overflow, mirror)
WRITE = {"wrist_from_shoulder_m": (0.10, -0.22, 0.33), "pole": (-0.5, -0.8, -0.3), "palm": (0.75, -0.66, 0.0),
         "rest_from_shoulder_m": (-0.04, -0.24, 0.32), "rest_palm": (0.0, -1.0, 0.0), "rest_clear_m": 0.03,
         "start_s": 1.0}
#   [module] the writing arm (wrist target from the shoulder joint, x toward the midline, y up, z forward; elbow pole;
#   palm normal toward the midline and down: the forearm semi-pronated, about 50 deg short of full pronation, the hand
#   on its ulnar border; review B5: a fully pronated hand read as a claw), the other arm resting on the table beside
#   the paper (palm down, the relaxed hand, the wrist 4 cm lateral to its shoulder, and rest_clear_m further out on a
#   left writer's right arm: the line runs to the writer's right, so the left hand travels toward that resting hand;
#   review of wave SH1c: the two hands passed through each other; table_report reads the hands' skin gap); writing
#   from start_s
STROKE = {"hz": 4.5, "hz_jitter": 0.6, "finger_deg": 6.0, "wrist_deg": 3.0, "lift_deg": 4.0, "edge_s": 0.06,
          "letter_band_hz": (0.8, 2.0)}
#   [module] the strokes of handwriting: flexion of the tripod's fingers (the pen up and down the line) and wrist
#   deviation (the pen along it) a quarter cycle apart, at about 4.5 Hz with jitter, the size varying letter by letter
#   (letter_band_hz), the pen lifted by a small wrist extension between words (review B5: a pure 4 Hz wrist
#   oscillation with a still hand read as a writing tremor)
LINE = {"speed_m_s": 0.015, "word_s": (1.0, 2.0), "gap_s": (0.25, 0.45), "gap_m": 0.006, "line_m": 0.07,
        "line_step_m": 0.012, "return_s": 0.7}
#   [module] the hand's progression along the line (1.5 cm/s while writing, a word every 1 to 2 s, the gaps between
#   words), the line return (back to the line's start and one line nearer the writer)
PAPER_M = (0.21, 0.297, 0.001)  # an A4 sheet (width, depth, thickness), scene geometry
WRITE_EDGE_GAP_M = 0.001     # [module] the writing hand rests on the paper: the lowest skinned point of its palm and
#                              of its ring and little fingers (EDGE_FINGERS; the thumb, index and middle hold the pen)
#                              this far above the paper at every knot (seat_hand; wave SH1c: before it the hand hovered
#                              over the table, the table placed under the resting hand). In the tripod grip of
#                              WRITING_HAND that point lies on the curled little finger and the forearm stays free above
#                              the table (table_report's forearm_gap_mm; review of wave SH1c: with the forearm lowered
#                              onto the table, the curled little finger stayed under the paper over the wrist
#                              extensions tried)
WRITE_REST_GAP_M = 0.001     # [module] the resting hand's lowest skinned point this far above the table top
EDGE_FINGERS = ("ring", "pinky")
PEN = {"touch_m": 0.0005, "lift_m": 0.003,
       "joints": (("index", 0, 1.0), ("index", 1, 1.0), ("thumb", 1, 1.0), ("thumb", 0, 0.5))}
#   [module] the pen (not drawn, SH13; a measure of the hand, pen_points): held at the grip point (midway between the
#   thumb and index fingertip vertices), along the line from the index knuckle (where it rests in the first web space)
#   through the grip, its tip as far beyond the grip as reaches the paper at the first knot (pen_length); the tip
#   touch_m above the paper while a word is written and before the first word, lifted by lift_m more between words and
#   on the line return (the course of the wrist's pen lift, STROKE lift_deg). The tripod's strokes alone moved the tip
#   both into the paper and millimetres above it within a word (wave SH1c), so the index's knuckle and middle joint
#   and the thumb's two proximal joints (joints: finger, level, weight; the thumb's base joint at half the angle) take a
#   common flexion offset at each knot that holds the tip at its height (seat_pen): the index's distal joint and the
#   thumb's interphalangeal joint keep their strokes alone, and the pen's tip slides along the paper
SEAT_AIM_M = 1e-6            # [module] the seat solve (seat_hand) iterates until within this of its goal
SEAT_TOL_M = 1e-5            # [module] a knot of a clip built on a seated body is solved again only if off by more
SEAT_ITERS = 10              # [module] at most this many steps of a seat solve
WCRS_ITEMS = {1: {"elbow": 0, "wrist": 1, "fingers": 2, "latency_factor": 1, "tremor": 0},
              2: {"elbow": 1, "wrist": 3, "fingers": 4, "latency_factor": 2, "tremor": 0}}
#   [module] the writer's cramp levels as Writer's Cramp Rating Scale item scores (structure of the catalogue's D14:
#   elbow 0 to 2, wrist 0 to 4, fingers 0 to 6, a latency factor 1 or 2, writing tremor 0 to 2 counted twice; the
#   per-item anchors were not read, so the scores are a module reading of 'mild' and 'marked', review M7)
WCRS_ANGLES = {"wrist_per_point": 0.15, "ulnar_deg_per_point": 3.0, "pronation_deg_per_point": 5.0,
               "fingers_per_point": 1.0 / 6.0, "elbow_lift_deg_per_point": 15.0}
#   [module] the angles of the item scores: per wrist point 0.15 of the wrist's flexion room (g2: about 32 deg, inside
#   the -55 to -10 deg of most patients in Delrobaei 2016 via the catalogue's D14), ulnar deviation and
#   pronation; per finger point a sixth of the way from the pen grip (or the relaxed hand) to CRAMP_HAND; per elbow
#   point 15 deg of the elbow turned up and out about the shoulder-wrist line (the elbow abducted and lifted)
WCRS_LATENCY_S = {1: 4.0, 2: 1.0}   # [module] latency factor 1: the posture appears later, 2: as writing starts
WCRS_RAMP_S = 2.0
WCRS_STROKE_DROP = 0.5       # [module] the strokes shrink by this share of the posture's envelope (slowed, effortful)


def wcrs_total(items):
    """The WCRS writing movement score of item scores: (elbow + wrist + fingers) x latency factor + 2 x tremor."""
    return (items["elbow"] + items["wrist"] + items["fingers"]) * items["latency_factor"] + 2 * items["tremor"]


def _mirror_vec(v, side):
    v = np.asarray(v, dtype=np.float64)
    return v if side == "right" else v * np.array([-1.0, 1.0, 1.0])


def writing_line(T, rate, rng, start_s=WRITE["start_s"]):
    """{along (T,) m, across (T,) m, gate (T,) in [0, 1], segments}: the hand's progression on the paper (LINE) from
    start_s: words at speed_m_s, gaps (the hand carried gap_m in a minimum-jerk move, the pen lifted), the line return
    after line_m (minimum-jerk, back to 0 along and one line_step_m across); gate 1 while a word is written (raised
    cosine edges of STROKE['edge_s'])."""
    t = np.arange(T) / rate
    along, across, gate = np.zeros(T), np.zeros(T), np.zeros(T)
    segs = []
    x, line, tt = 0.0, 0, float(start_s)
    end = t[-1]
    while tt < end:
        w = float(rng.uniform(*LINE["word_s"]))
        segs.append(("word", tt, tt + w, x, x + LINE["speed_m_s"] * w, line, line))
        x += LINE["speed_m_s"] * w
        tt += w
        if x >= LINE["line_m"]:
            r = LINE["return_s"]
            segs.append(("return", tt, tt + r, x, 0.0, line, line + 1))
            x, line = 0.0, line + 1
            tt += r
        else:
            g = float(rng.uniform(*LINE["gap_s"]))
            segs.append(("gap", tt, tt + g, x, x + LINE["gap_m"], line, line))
            x += LINE["gap_m"]
            tt += g
    e = STROKE["edge_s"]
    for kind, t0, t1, x0, x1, l0, l1 in segs:
        m = (t >= t0) & (t < t1)
        u = (t[m] - t0) / (t1 - t0)
        if kind == "word":
            along[m] = x0 + (x1 - x0) * u
            up = np.clip((t - t0) / e, 0.0, 1.0)
            dn = np.clip((t1 - t) / e, 0.0, 1.0)
            gate = np.maximum(gate, np.minimum(0.5 - 0.5 * np.cos(np.pi * up), 0.5 - 0.5 * np.cos(np.pi * dn)))
        else:
            along[m] = x0 + (x1 - x0) * minimum_jerk(u)
        across[m] = (l0 + (l1 - l0) * (minimum_jerk(u) if kind == "return" else u * 0)) * LINE["line_step_m"]
    last = t >= (segs[-1][2] if segs else end + 1)
    if segs:
        along[last], across[last] = segs[-1][4], segs[-1][6] * LINE["line_step_m"]
    return {"along": along, "across": across, "gate": gate,
            "segments": [[k, round(t0, 3), round(t1, 3)] for k, t0, t1, *_ in segs]}


def stroke_signals(T, rate, rng, gate, gain=None):
    """(vertical (T,), horizontal (T,)) in [-1, 1]: the pen's strokes (STROKE): a phase at STROKE['hz'] with
    smooth jitter, an amplitude varying letter by letter, gated by the words and times gain (T,) (default 1)."""
    f = STROKE["hz"] + STROKE["hz_jitter"] * smooth_noise(T, rate, rng, (0.3, 1.0))
    ph = 2 * np.pi * np.cumsum(f) / rate
    amp = gate * (0.55 + 0.45 * np.abs(smooth_noise(T, rate, rng, STROKE["letter_band_hz"], 7)))
    if gain is not None:
        amp = amp * np.asarray(gain, dtype=np.float64)
    return amp * np.sin(ph), amp * np.cos(ph)


def _arm_low(body, sides):
    """(T,) the lowest skinned point (y) of the forearms and hands of sides."""
    _, _, dom = H._rest()
    ids = np.flatnonzero(np.isin(dom, [j for sd in sides for j in (ARM[sd][2], ARM[sd][3])] +
                                 [j for sd in sides for j in range(H.FIRST[sd], H.FIRST[sd] + 15)]))
    return H.body_points(body, ids)[..., 1].min(-1)


@functools.lru_cache(maxsize=8)
def hand_vertex_ids(side, fingers=None):
    """The vertices of the hand of side whose dominant joint (neutral rest mesh) is the wrist or a joint of one of
    fingers (a tuple of hands.FINGERS names; default all five)."""
    _, _, dom = H._rest()
    fs = H.FINGERS if fingers is None else tuple(fingers)
    return np.flatnonzero(np.isin(dom, [H.WRIST[side]] + [H.hand_joint(side, f, lv) for f in fs for lv in range(3)]))


def seat_hand(body, side, goal_y, ids, pole, base_R, table=HEALTHY, tol=SEAT_AIM_M, mask=None):
    """(Body, info): the arm of side re-solved (arm_ik, end 'wrist', the elbow toward pole (3,) or (T, 3); project_rows
    against base_R in table) with its wrist moved straight up or down at each knot so that the lowest of the vertices
    ids (or ids(body, knots), a function giving (K,) heights, resting_low) lies at goal_y (m, scalar or (T,)): Newton
    steps on the wrist's height (the hand follows its wrist nearly one for one), at most SEAT_ITERS, to SEAT_AIM_M, on
    the knots off by more than tol at the start (and inside mask (T,) bool when given); every other knot keeps its rows
    bit for bit. info: the largest error before and after (mm), the largest wrist shift (mm), the knots solved and the
    steps."""
    T = body.T
    goal = np.broadcast_to(np.asarray(goal_y, dtype=np.float64), (T,)).copy()
    w = ARM[side][3]
    tgt = body.joints().numpy()[:, w].copy()
    start = tgt[:, 1].copy()
    err0 = goal - _low(body, ids)
    sel = np.abs(err0) > tol
    if mask is not None:
        sel &= np.asarray(mask, bool)
    out, err, steps = body, err0.copy(), 0
    pole_t = torch.as_tensor(np.asarray(pole, dtype=np.float64))
    for _ in range(SEAT_ITERS):
        bad = sel & (np.abs(err) > SEAT_AIM_M)
        if not bad.any():
            break
        tgt[bad, 1] += err[bad]
        R = arm_ik(out, out.local_R(), side, torch.as_tensor(tgt), pole_t, None, end="wrist")
        R, _ = project_rows(R, ARM[side], base_R, table=table)
        Rn = out.local_R().clone()
        kb = torch.as_tensor(np.flatnonzero(bad))
        Rn[kb] = R[kb]
        out = out.with_rotations(Rn, list(ARM[side]))
        k = np.flatnonzero(bad)
        err[k] = goal[k] - _low(out, ids, k)
        steps += 1
    return out, {"error_before_mm": float(1000 * np.abs(err0).max()), "error_after_mm": float(1000 * np.abs(err).max()),
                 "error_after_solved_mm": float(1000 * np.abs(err[sel]).max()) if sel.any() else 0.0,
                 "wrist_shift_mm": float(1000 * np.abs(tgt[:, 1] - start).max()), "knots_solved": int(sel.sum()),
                 "steps": steps}


def _low(body, ids, knots=None):
    """(K,) m: the lowest height of the vertices ids of the body at the knots (default all), or ids(body, knots)."""
    k = np.arange(body.T) if knots is None else np.asarray(knots, dtype=np.int64)
    return np.asarray(ids(body, k)) if callable(ids) else H.body_points(body, ids, knots=k)[..., 1].min(-1)


def resting_low(side, pen_m):
    """The low function of seat_hand for a writing hand under a posture (writers_cramp): at each knot the lower of the
    hand's lowest skinned point less WRITE_EDGE_GAP_M and the pen's tip (pen_points, the pen of pen_m) less
    PEN['touch_m']: seated at the paper's height, the hand rests on the paper by whatever of it or of the pen lies
    lowest (the cramped fingers pressing the pen lift the hand's edge; a hand turned off its edge comes down), and
    nothing of either goes through it."""
    ids = hand_vertex_ids(side)

    def low(body, k):
        return np.minimum(H.body_points(body, ids, knots=k)[..., 1].min(-1) - WRITE_EDGE_GAP_M,
                          pen_points(body, side, pen_m, k)[0][:, 1] - PEN["touch_m"])
    return low


def _grip_knuckle(body, side, knots):
    """(grip (K, 3), knuckle (K, 3)): the grip point (midway between the thumb and index fingertip vertices,
    hands.tip_vertices) and the index knuckle (the wrist joint plus its rest offset turned by the wrist's global
    rotation) of side at the knots."""
    k = np.asarray(knots, dtype=np.int64)
    tips = H.tip_vertices()[side]
    grip = H.body_points(body, [tips["thumb"], tips["index"]], knots=k).mean(1)
    J, Gl = body.fk()
    J52, _, _ = H._rest()
    w = H.WRIST[side]
    knuckle = J[k, w].numpy() + np.einsum("tij,j->ti", Gl[k, w].numpy(), J52[H.hand_joint(side, "index", 0)] - J52[w])
    return grip, knuckle


def pen_length(body, side, tip_y):
    """m: the pen's length from the grip to the tip (PEN): at the body's first knot, the length along the line from
    the index knuckle through the grip down to the height tip_y (the tip touching the paper there)."""
    g, kn = _grip_knuckle(body, side, [0])
    a = (g - kn)[0] / np.linalg.norm((g - kn)[0])
    if a[1] >= 0.0:
        raise ValueError(f"the {side} knuckle-to-grip line does not point down to the paper")
    return float((g[0, 1] - tip_y) / -a[1])


def pen_points(body, side, length, knots=None):
    """(tip (K, 3), grip (K, 3)): the pen of PEN in the hand of side at the knots (default all): from the grip along
    the line from the index knuckle through it, its tip length m beyond the grip."""
    k = np.arange(body.T) if knots is None else np.asarray(knots, dtype=np.int64)
    g, kn = _grip_knuckle(body, side, k)
    a = (g - kn) / np.linalg.norm(g - kn, axis=-1, keepdims=True)
    return g + length * a, g


def _pen_hand(Rh, side, u, signs):
    """(T, 30, 3, 3): Rh with each joint of PEN['joints'] turned by sign x weight x u (T,) deg about its flexion axis
    (hands.hand_frame, fixed in the parent frame: the joint's flexion changes by that much)."""
    fr = H.hand_frame()[side]["joints"]
    out = Rh.clone()
    for (f, lv, wt), sg in zip(PEN["joints"], signs):
        j = H.hand_joint(side, f, lv)
        a = torch.as_tensor(fr[j]["flex"], dtype=F64)
        ang = np.radians(sg * wt * np.asarray(u, dtype=np.float64))
        out[:, H.hand_row(j)] = _rot_about(a, ang) @ Rh[:, H.hand_row(j)]
    return out


def _flex_range(f, lv):
    """(lo, hi) deg of a hand joint's flexion: hands.FLEX_LIMITS_DEG, the thumb's base joint hands.CMC_LIMIT_DEG."""
    if f == "thumb" and lv == 0:
        return -H.CMC_LIMIT_DEG, H.CMC_LIMIT_DEG
    return H.FLEX_LIMITS_DEG[f][(H.THUMB_LEVELS if f == "thumb" else H.LEVELS)[lv]]


def seat_pen(body, side, goal_y, length, tol=SEAT_AIM_M, mask=None):
    """(HandBody, info): the joints of PEN['joints'] flexed together by u (T,) deg times their weights (each the way
    that lowers the pen's tip, read on the first knot by a 1 deg step) so that the tip (pen_points, the pen of length)
    lies at goal_y (m, (T,)): secant steps, at most SEAT_ITERS, to SEAT_AIM_M, on the knots off by more than tol at the
    start (and inside mask when given), every other knot keeping its hand rows bit for bit; each joint kept inside its
    flexion range (_flex_range: u clamped, the residual reported). info: the largest error before and after (mm), the
    range of u (deg), the knots solved and clamped."""
    T = body.T
    goal = np.broadcast_to(np.asarray(goal_y, dtype=np.float64), (T,)).copy()
    comp = H.hand_components(body.hand_R())
    lo, hi = np.full(T, -np.inf), np.full(T, np.inf)
    signs = []
    b0 = crop_knots(body, [0])
    tip0 = pen_points(b0, side, length, [0])[0][0, 1]
    for i, (f, lv, wt) in enumerate(PEN["joints"]):
        one = [1.0 if m == i else 0.0 for m in range(len(PEN["joints"]))]
        t1 = pen_points(b0.with_hands(_pen_hand(b0.hand_R(), side, np.ones(2), one)), side, length, [0])[0][0, 1]
        sg = 1.0 if t1 < tip0 else -1.0                   # the sign that lowers the tip for a positive u
        signs.append(sg)
        lim = _flex_range(f, lv)
        c = comp[H.hand_joint(side, f, lv)][0]            # u such that c + sg wt u stays inside the range
        lo = np.maximum(lo, np.where(sg > 0, lim[0] - c, c - lim[1]) / wt)
        hi = np.minimum(hi, np.where(sg > 0, lim[1] - c, c - lim[0]) / wt)
    u = np.zeros(T)
    err0 = goal - pen_points(body, side, length)[0][:, 1]
    sel = np.abs(err0) > tol
    if mask is not None:
        sel &= np.asarray(mask, bool)
    err, clamped, out = err0.copy(), np.zeros(T, bool), body
    for _ in range(SEAT_ITERS):
        bad = sel & (np.abs(err) > SEAT_AIM_M) & ~clamped
        if not bad.any():
            break
        k = np.flatnonzero(bad)
        b1 = out.with_hands(_pen_hand(out.hand_R(), side, bad.astype(np.float64), signs))
        slope = pen_points(b1, side, length, k)[0][:, 1] - (goal[k] - err[k])   # m per deg (negative)
        ok = np.abs(slope) > 1e-6
        un = np.clip(u[k] + np.where(ok, err[k] / np.where(ok, slope, 1.0), 0.0), lo[k], hi[k])
        clamped[k] = (un <= lo[k] + 1e-9) | (un >= hi[k] - 1e-9)
        du = np.zeros(T)
        du[k] = un - u[k]
        u[k] = un
        out = out.with_hands(_pen_hand(out.hand_R(), side, du, signs))
        err[k] = goal[k] - pen_points(out, side, length, k)[0][:, 1]
    return out, {"error_before_mm": float(1000 * np.abs(err0).max()), "error_after_mm": float(1000 * np.abs(err).max()),
                 "u_deg": [float(u.min()), float(u.max())], "signs": signs, "knots_solved": int(sel.sum()),
                 "knots_clamped": int((clamped & sel).sum())}


def pen_lift_share(T, rate, seed=SEED):
    """(T,) in [0, 1]: the pen's lift of writing_base (the line of writing_line with the rng of seed, the first draw of
    writing_base): 1 between words and on the line returns once writing has started (the wrist's pen lift course),
    0 while a word is written and before the first."""
    line = writing_line(T, rate, as_rng(seed))
    return ramp_env(T, rate, WRITE["start_s"], 0.3) * (1.0 - line["gate"])


def _turn_hand_rows(Rh, side, pose, share):
    """(T, 30, 3, 3): Rh with the side's rows turned toward a {finger: angles} pose by shares (T,) (each row along
    its geodesic from its own rotation at each knot)."""
    B = hand_pose_rows(pose, side)
    rows = list(range(15)) if side == "left" else list(range(15, 30))
    W = torch.as_tensor(np.asarray(share, dtype=np.float64))
    out = Rh.clone()
    for r in rows:
        D = matrix_to_axis_angle(Rh[:, r].transpose(-1, -2) @ B[r][None])
        out[:, r] = Rh[:, r] @ axis_angle_to_matrix(W[:, None] * D)
    return out


def writing_base(duration_s=DUR_S, sides=("left", "right"), seed=SEED, write_side="right", stroke_gain=None):
    """(HandBody, info): seated ('sit' posture body) at the table: the arm of write_side writing (arm_ik_in_table: the
    wrist on WRITE's target moving along the line (writing_line), the forearm semi-pronated, the tripod grip
    WRITING_HAND, the strokes of stroke_signals in the fingers and the wrist, the pen lifted between words), every other
    arm of sides resting on the table palm down with RELAXED_HAND (WRITE's rest_clear_m further out when the left hand
    writes). The writing hand rests on the paper by its curled
    little finger, the forearm free above the table: the paper's upper face lies WRITE_EDGE_GAP_M under the edge (the
    lowest skinned point of the palm and of EDGE_FINGERS) at the first knot, the table top PAPER_M[2] under it, and
    the arm is re-solved at every knot so that the edge stays WRITE_EDGE_GAP_M above the paper (seat_hand); the pen
    (PEN, pen_points) is as long as reaches the paper from the grip along the knuckle-to-grip line at the first knot
    (pen_length), its tip held on the paper while a word is written and lifted by PEN['lift_m'] between words
    (pen_lift_share) by the knuckles' common offset (seat_pen); each resting hand's lowest skinned point lies
    WRITE_REST_GAP_M above the table top (seat_hand). write_side None: no writing movement (the arm keeps the line's
    start, no pen). seed: the writing's own (the level 0 and the cramp clips write the same line with the module
    SEED). info['surface']: the paper's and the table top's heights and the pen's length (writing_table and
    table_report read them through writing_surface)."""
    from .motions import posture_body
    b = posture_body("sit", duration_s)
    T, rate = b.T, b.rate
    rng = as_rng(seed)
    R = b.local_R()
    J = b.joints().numpy()
    one = np.ones(T)
    ws = write_side or "right"
    line = writing_line(T, rate, rng) if write_side is not None else {"along": np.zeros(T), "across": np.zeros(T),
                                                                      "gate": np.zeros(T), "segments": []}
    move = np.stack([-line["along"], np.zeros(T), -line["across"]], -1)  # to the writer's right, then nearer
    rest_off = np.array(WRITE["rest_from_shoulder_m"])
    if write_side == "left":                             # the line runs toward the resting right hand
        rest_off[0] -= WRITE["rest_clear_m"]
    targets, poles = {}, {}
    order = [ws] + [sd for sd in sides if sd != ws]
    for sd in order:
        s_ = ARM[sd][1]
        writing = sd == ws
        off = WRITE["wrist_from_shoulder_m"] if writing else rest_off
        tgt = J[0, s_][None] + _mirror_vec(off, sd)[None] + (move if writing else 0.0) * one[:, None]
        pole = _mirror_vec(WRITE["pole"], sd)
        palm = _mirror_vec(WRITE["palm"] if writing else WRITE["rest_palm"], sd)
        R2, pl, collar, ex = arm_ik_in_table(b, R, sd, torch.as_tensor(tgt), torch.as_tensor(palm), knots=[0],
                                             poles=(tuple(pole * np.array([1.0 if sd == "left" else -1.0, 1, 1])),)
                                             + POLES, end="wrist")
        R, _ = project_rows(R2, ARM[sd], b.local_R())
        targets[sd] = tgt
        poles[sd] = {"pole": [float(x) for x in pl], "collar_deg": list(collar), "excess_deg": ex}
    out = b.with_rotations(R, sorted({j for sd in sides for j in ARM[sd]} | set(ARM[ws])))
    Rh = out.hand_R().clone()
    for sd in set(sides) | {ws}:
        rows = list(range(15)) if sd == "left" else list(range(15, 30))
        Rh[:, rows] = hand_pose_rows(WRITING_HAND if (sd == ws and write_side is not None) else RELAXED_HAND,
                                     sd)[rows][None]
    out = out.with_hands(Rh)
    info = {"arms": poles, "write_side": write_side, "rest_sides": [sd for sd in sides if sd != ws],
            "write": WRITE, "stroke": STROKE, "line": dict(LINE, segments=line["segments"]),
            "hand": "WRITING_HAND and RELAXED_HAND [module]"}
    if write_side is not None:
        v, h = stroke_signals(T, rate, rng, line["gate"], stroke_gain)
        go = ramp_env(T, rate, WRITE["start_s"], 0.3)
        w = H.WRIST[write_side]
        sg = 1.0 if write_side == "right" else -1.0
        lift = STROKE["lift_deg"] * go * (1.0 - line["gate"])
        Rw = apply_terms(out.local_R(), [joint_term(w, "y", sg * STROKE["wrist_deg"], h, "parent", "wrist strokes"),
                                         region_term(out, "wrist", "extension", write_side, 1.0, lift, "pen lift")])
        out = out.with_rotations(Rw, [w])
        f = {"index": (STROKE["finger_deg"],) * 3, "middle": (STROKE["finger_deg"], 0.5 * STROKE["finger_deg"], 0.0),
             "thumb": ((0.5 * STROKE["finger_deg"], 0.0), STROKE["finger_deg"], STROKE["finger_deg"])}
        base_h = out.hand_R()
        delta = hand_pose_rows(f, write_side)
        rows = list(range(15)) if write_side == "left" else list(range(15, 30))
        Rh2 = base_h.clone()
        for r in rows:
            D = matrix_to_axis_angle(delta[r])
            Rh2[:, r] = axis_angle_to_matrix(torch.as_tensor(v)[:, None] * D[None]) @ base_h[:, r]
        out = out.with_hands(Rh2)
        info["strokes"] = {"vertical_abs_max": float(np.abs(v).max()), "gate_share": float((line["gate"] > 0.5).mean())}
    edge = hand_vertex_ids(ws, EDGE_FINGERS)
    paper = float(H.body_points(crop_knots(out, [0]), edge, knots=[0])[0, :, 1].min()) - WRITE_EDGE_GAP_M
    top = paper - PAPER_M[2]
    info["seat"] = {}
    for _ in range(2):                                   # each arm's pose blend shapes move the other hand's skin by a
        for sd in info["rest_sides"]:                    # fraction of a millimetre: the resting arms, the writing arm,
            out, info["seat"][sd] = seat_hand(out, sd, top + WRITE_REST_GAP_M, hand_vertex_ids(sd),   # then again
                                              poles[sd]["pole"], b.local_R())
        out, info["seat"][ws] = seat_hand(out, ws, paper + WRITE_EDGE_GAP_M, edge, poles[ws]["pole"], b.local_R())
    pen_m = None
    if write_side is not None:                           # the pen's tip on the paper, then the edge again (the
        pen_m = pen_length(out, ws, paper + PEN["touch_m"])   # knuckles' pose blend shapes move the palm's skin)
        share = ramp_env(T, rate, WRITE["start_s"], 0.3) * (1.0 - line["gate"])
        out, info["pen"] = seat_pen(out, ws, paper + PEN["touch_m"] + PEN["lift_m"] * share, pen_m)
        out, info["seat"][ws + "_after_pen"] = seat_hand(out, ws, paper + WRITE_EDGE_GAP_M, edge, poles[ws]["pole"],
                                                         b.local_R())
    out = replace(out, contact=contact_labels(out))
    info["targets_m"] = {sd: [float(x) for x in targets[sd][0]] for sd in targets}
    info["surface"] = {"paper_top_y": paper, "table_top_y": top, "pen_m": pen_m}
    info["table_top_y"] = top
    info["paper_top_y"] = paper
    return out, info


def crop_knots(body, knots):
    """The body at the given knots only (a HandBody keeps its hands; a single knot is doubled: a body has two)."""
    k = list(knots) * (2 if len(knots) == 1 else 1)
    extra = {"hand6d": body.hand6d[k]} if H.has_hands(body) else {}
    return replace(body, rot6d=body.rot6d[k], transl=body.transl[k], contact=body.contact[k], **extra)


@functools.lru_cache(maxsize=2)
def writing_surface(write_side="right"):
    """{'paper_top_y', 'table_top_y', 'pen_m'} m: the writing scene's surfaces with write_side writing and the pen's
    length (writing_base: the paper under the writing hand's edge at the first knot, the same for every duration and
    seed, since nothing moves before WRITE['start_s']), read on a 0.02 s build so that every clip and its layered body
    stand on one table and hold one pen."""
    return dict(writing_base(0.02, ("left", "right"), SEED, write_side)[1]["surface"])


def line_reach_m():
    """m: the hand's longest travel along a line of writing_line: a line returns once a word has carried the hand past
    LINE['line_m'], so the last word starts short of line_m plus one gap (LINE['gap_m']) and runs at most
    LINE['speed_m_s'] x the longest word (LINE['word_s']) beyond."""
    return LINE["line_m"] + LINE["gap_m"] + LINE["speed_m_s"] * LINE["word_s"][1]


def writing_boxes(body, write_side="right", knot=0):
    """{'top': (lo, hi), 'paper': (lo, hi), 'legs': [(lo, hi)] x 4} m of the writing scene (module sizes): a 3 cm top
    whose upper face lies at writing_surface's table top, from 3 cm in front of the abdomen at knot to 60 cm beyond,
    1.1 m wide, four legs (their feet set by writing_table), and an A4 sheet (PAPER_M) under the writing hand, its near
    edge 4 cm nearer the writer than the wrist at knot, its middle across the line midway between the hand's edge and
    the pen's tip at knot (pen_points, the edge's lowest point) carried half the hand's longest travel along a line
    (line_reach_m) to the writer's right, the way the hand moves (wave SH1c: centred on the wrist, the sheet left the
    pen's tip beside it; carried half of LINE['line_m'] alone, the edge of the catalogue's mild writer's cramp left the
    sheet at the end of a line, where the last word runs past line_m)."""
    dom = H._rest()[2]
    V = H.body_points(body, np.arange(len(dom)), knots=[knot])[0]      # every vertex: the ids below index it
    surf = writing_surface(write_side)
    top = surf["table_top_y"]
    z0 = float(V[np.isin(dom, (0, 3, 6, 9)), 2].max()) + 0.03
    J = body.joints().numpy()[knot]
    wz = float(J[H.WRIST[write_side], 2])
    ids = hand_vertex_ids(write_side, EDGE_FINGERS)
    ex = float(V[ids][np.argmin(V[ids, 1]), 0])
    tx = float(pen_points(crop_knots(body, [knot]), write_side, surf["pen_m"], [0])[0][0, 0]) if surf["pen_m"] else ex
    cx = 0.5 * (ex + tx) - 0.5 * line_reach_m()
    pz0 = max(z0 + 0.005, wz - 0.04)
    legs = [(np.array([x, -np.inf, z]), np.array([x + 0.04, top - 0.03, z + 0.04]))
            for x in (-0.55, 0.51) for z in (z0, z0 + 0.56)]
    return {"top": (np.array([-0.55, top - 0.03, z0]), np.array([0.55, top, z0 + 0.6])),
            "paper": (np.array([cx - 0.5 * PAPER_M[0], top, pz0]),
                      np.array([cx + 0.5 * PAPER_M[0], top + PAPER_M[2], pz0 + PAPER_M[1]])), "legs": legs}


def writing_table(body, floor_y, knot=0, sides=("left", "right"), write_side="right"):
    """(vertices, faces, height): the writing table of writing_boxes standing on the floor plane floor_y (its legs from
    the floor to the top) with the A4 sheet on it; height: the top's height above the floor. sides: kept for the
    callers (the table no longer depends on the resting arms: wave SH1c)."""
    from .viewer import box
    bx = writing_boxes(body, write_side, knot)
    parts = [box(*bx["top"])]
    for lo, hi in bx["legs"]:
        parts.append(box((lo[0], float(floor_y), lo[2]), tuple(hi)))
    parts.append(box(*bx["paper"]))
    vs, fs, off = [], [], 0
    for v, f in parts:
        vs.append(v)
        fs.append(f + off)
        off += len(v)
    return np.concatenate(vs), np.concatenate(fs), float(bx["top"][1][1]) - float(floor_y)


def table_report(body, write_side, rest_sides, knots=None, lift=None, writing=True, strict=True):
    """{'edge_gap_mm', 'pen_m', 'pen_gap_mm', 'pen_gap_word_mm', 'pen_incline_deg', 'rest_gap_mm', 'depth_mm',
    'over_paper', 'ok'}: on the writing scene (writing_boxes, writing_surface) every SCENE_CHECK_STEP knots (or at
    knots): the writing hand's edge (the lowest skinned point of its palm and EDGE_FINGERS) above the paper ([least,
    largest] mm), the pen's tip (pen_points, the pen of writing_surface's length) above the paper, over all the knots
    and over those where it is not lifted (lift (T,) in [0, 1], pen_lift_share, at most 0.01), the pen's incline from
    the horizontal, each resting hand's lowest point above the table top, the deepest entry of a forearm or hand vertex
    of either side into the table top or the paper (mm), whether the edge's lowest point and the tip lie over the
    paper, the writing forearm's lowest skinned point over the table top above it ('forearm_gap_mm', [least, largest]
    mm, None where it never lies over the top; reported, not judged), and the two hands' least skin gap
    ('hands_gap_mm', motions.fold_gaps, negative inside); ok when the edge lies 0 to WRITE_EDGE_GAP_M above the paper,
    the tip 0 to PEN touch_m plus lift_m above it (to touch_m where not lifted), every resting hand 0 to
    WRITE_REST_GAP_M above the table, all within SCENE_TOL_M, nothing enters the table or the paper by more than
    SCENE_TOL_M, both points lie over the paper and the hands stay apart.
    writing False: no pen (the hand of write_side is then judged as resting on the paper by its edge alone). strict
    False (the writing hand under a posture, writers_cramp): instead of the edge and the tip's ranges, the writing
    hand's lowest point ('hand_gap_mm') and the tip on or above the paper, and the lower of the hand's lowest point
    less WRITE_EDGE_GAP_M and the tip less touch_m ('rest_low_mm', resting_low) on the paper, all within
    SCENE_TOL_M."""
    from .motions import fold_gaps
    k = np.arange(0, body.T, SCENE_CHECK_STEP) if knots is None else np.asarray(knots, dtype=np.int64)
    surf = writing_surface(write_side)
    paper, top = surf["paper_top_y"], surf["table_top_y"]
    bx = writing_boxes(body, write_side)
    tol = SCENE_TOL_M
    E = H.body_points(body, hand_vertex_ids(write_side, EDGE_FINGERS), knots=k)
    i = E[..., 1].argmin(-1)
    el = E[np.arange(len(k)), i]
    edge = el[:, 1] - paper
    plo, phi = bx["paper"]
    over = bool(((el[:, 0] >= plo[0]) & (el[:, 0] <= phi[0]) & (el[:, 2] >= plo[2]) & (el[:, 2] <= phi[2])).all())
    hand = H.body_points(body, hand_vertex_ids(write_side), knots=k)[..., 1].min(-1) - paper
    rep = {"edge_gap_mm": [float(1000 * edge.min()), float(1000 * edge.max())],
           "hand_gap_mm": [float(1000 * hand.min()), float(1000 * hand.max())]}
    ok = bool(hand.min() >= -tol)
    if strict:
        ok = ok and bool(edge.min() >= -tol and edge.max() <= WRITE_EDGE_GAP_M + tol)
    if writing:
        ls = np.zeros(body.T) if lift is None else np.asarray(lift, dtype=np.float64)
        tip, grip = pen_points(body, write_side, surf["pen_m"], k)
        g = tip[:, 1] - paper
        low = ls[k] <= 0.01
        rep["pen_m"] = surf["pen_m"]
        rep["pen_gap_mm"] = [float(1000 * g.min()), float(1000 * g.max())]
        rep["pen_gap_word_mm"] = [float(1000 * g[low].min()), float(1000 * g[low].max())] if low.any() else None
        inc = np.degrees(np.arcsin(np.clip((grip[:, 1] - tip[:, 1]) / surf["pen_m"], -1.0, 1.0)))
        rep["pen_incline_deg"] = [float(inc.min()), float(inc.max())]
        over = over and bool(((tip[:, 0] >= plo[0]) & (tip[:, 0] <= phi[0]) & (tip[:, 2] >= plo[2]) &
                              (tip[:, 2] <= phi[2])).all())
        ok = ok and bool(g.min() >= -tol)
        if strict:
            ok = ok and bool(g.max() <= PEN["touch_m"] + PEN["lift_m"] + tol)
            ok = ok and (not low.any() or bool(g[low].max() <= PEN["touch_m"] + tol))
        else:
            rl = np.minimum(hand - WRITE_EDGE_GAP_M, g - PEN["touch_m"])
            rep["rest_low_mm"] = [float(1000 * rl.min()), float(1000 * rl.max())]
            ok = ok and bool(np.abs(rl).max() <= tol)
    _, _, dom = H._rest()
    X = H.body_points(body, np.flatnonzero(dom == ARM[write_side][2]), knots=k)
    tlo, thi = bx["top"]
    on_top = (X[..., 2] >= tlo[2]) & (X[..., 0] >= tlo[0]) & (X[..., 0] <= thi[0])
    fg = np.where(on_top, X[..., 1] - top, np.inf).min(-1)
    fg = fg[np.isfinite(fg)]
    rep["forearm_gap_mm"] = [float(1000 * fg.min()), float(1000 * fg.max())] if len(fg) else None
    rep["rest_gap_mm"] = {}
    for sd in rest_sides:
        r = H.body_points(body, hand_vertex_ids(sd), knots=k)[..., 1].min(-1) - top
        rep["rest_gap_mm"][sd] = [float(1000 * r.min()), float(1000 * r.max())]
        ok = ok and bool(r.min() >= -tol and r.max() <= WRITE_REST_GAP_M + tol)
    depth = 0.0
    for sd in ("left", "right"):
        P_ = H.body_points(body, arm_vertex_ids(sd), knots=k)
        depth = max(depth, float(_box_depth(P_, *bx["top"]).max()), float(_box_depth(P_, *bx["paper"]).max()))
    hands = float(fold_gaps(body, k, None, ("hands",))[0]["hands"].min())
    rep.update(depth_mm=1000 * depth, over_paper=over, n_knots=int(len(k)), hands_gap_mm=1000 * hands,
               ok=bool(ok and depth <= tol and over and hands >= 0.0))
    return rep


def table_check_record(write_side, rest_sides, strict=True):
    """{'write_side', 'rest_sides', 'lift', 'writing', 'strict'}: the arguments of a clip's table_report, kept in its
    truth (params['table_check']) so that the check can be run again on another body of the clip, such as its layered
    body (table_report_of; review of wave SH1c: the layered writing clips copied the raw table check)."""
    return {"write_side": write_side, "rest_sides": list(rest_sides), "lift": "pen_lift_share (module SEED)",
            "writing": True, "strict": bool(strict)}


def table_report_of(body, rec):
    """table_report of body with the arguments of a table_check_record rec (the pen's lift of pen_lift_share on the
    body's own clock)."""
    return table_report(body, rec["write_side"], rec["rest_sides"], lift=pen_lift_share(body.T, body.rate),
                        writing=rec["writing"], strict=rec["strict"])


def writing_level0(write_side="right", duration_s=DUR_S):
    """(HandBody, info, validity) of a writing level 0 (writing_base with the module SEED, both arms on the table):
    body.validity against the seated base and the writing scene's table_report (its arguments in
    info['table_check'], table_check_record)."""
    b, info = writing_base(duration_s, ("left", "right"), SEED, write_side)
    val = validity(H.hands_of(static_body("sit", duration_s)), b)
    info["table_check"] = table_check_record(write_side, info["rest_sides"])
    val["table"] = table_report_of(b, info["table_check"])
    return b, info, val


def writers_cramp(grade, duration_s=DUR_S, seed=0, overflow=False, mirror=False):
    """(Body, truth) of writer's cramp of the right hand (WCRS_ITEMS item scores, WCRS_ANGLES; grade 0: the writing
    level 0): seated at the table writing (writing_base, the level 0's line: seed drives the posture's course only), the
    dystonic posture appearing after the latency of its latency factor (WCRS_LATENCY_S): the pen grip tightening (thumb
    and index squeezing the pen, the ring and little fingers curling: CRAMP_HAND), the wrist flexing with ulnar
    deviation and pronation, the elbow abducted and lifted (the elbow turned up and out about the shoulder-wrist
    line, the wrist kept on its path), the strokes shrinking (WCRS_STROKE_DROP; writing_base holds the pen's tip on the
    paper under the shrunk strokes as in the level 0); where the posture acts the hand rests on the paper by whatever of
    it or of the pen lies lowest (seat_hand on resting_low: the cramped fingers keep their posture, the pen pressed on
    the paper may lift the hand's edge, and nothing goes through the paper). overflow: the posture spreads (Albanese
    2013) to the right shoulder (elevation, TWSTRS D level 2) and the neck (tilt to the right, laterocollis level 1) 1.5
    s later, the arm solved again so that the wrist stays on its path. mirror: the left hand writes and the right,
    resting palm down on the table, takes the posture (mirror dystonia), its lowest point kept on the table. The
    validity's 'table' entry: table_report."""
    rng = as_rng(seed)
    ws = "left" if mirror else "right"
    T = int(round(duration_s * 100.0)) + 1
    if grade == 0:
        b, winfo, val = writing_level0(ws, duration_s)
        return b, truth_record("writers_cramp", seed, 0, "WCRS (module item scores)",
                               {"level0": True, "writing": winfo, "table_check": winfo["table_check"]},
                               ("wcrs", "module"), val)
    it = WCRS_ITEMS[grade]
    A = WCRS_ANGLES
    lat = WCRS_LATENCY_S[it["latency_factor"]]
    env = held_env(T, 100.0, rng, WRITE["start_s"] + lat, WCRS_RAMP_S)
    gain = None if mirror else 1.0 - WCRS_STROKE_DROP * env
    b, winfo = writing_base(duration_s, ("left", "right"), SEED, ws, stroke_gain=gain)
    rate = b.rate
    ax = bone_axis_of(19, 21)
    terms = [region_term(b, "wrist", "flexion", "right",
                         A["wrist_per_point"] * it["wrist"] * region_room(b, "wrist", "flexion", "right", table=DYST),
                         env, "wrist flexion right", table=DYST),
             joint_term(21, "y", -A["ulnar_deg_per_point"] * it["wrist"], env, "parent", "wrist ulnar deviation right"),
             Term((21,), ax, -1.0, "parent", A["pronation_deg_per_point"] * it["wrist"], env, (1.0,),
                  "pronation right (at the wrist row)")]
    R, s = fit_terms(b.local_R(), terms[:2], table=DYST)
    R, s_pro = fit_terms(R, terms[2:], table=DYST)     # the pronation on its own: a resting hand palm down on the
    s = np.minimum(s, s_pro)                            # table may already lie at the table's pronation bound
    J = b.joints().numpy()
    tmp = b.with_rotations(R, [21])
    Jw = tmp.joints().numpy()[:, 21]                     # the wrist keeps its path on the paper
    over = []
    if overflow:
        env2 = held_env(T, rate, rng, WRITE["start_s"] + lat + 1.5, WCRS_RAMP_S)
        lv_sh = G.twstrs_level_deg("shoulder_elevation", 2,
                                   region_room(b, "shoulder_elevation", "elevation", "right", table=DYST))
        over = [region_term(b, "shoulder_elevation", "elevation", "right", lv_sh, env2, "shoulder elevation right",
                            table=DYST),
                region_term(b, "neck", "tilt_right", "left", G.twstrs_level_deg("laterocollis", 1, None), env2,
                            "neck tilt right", table=DYST)]
        R2, s2 = fit_terms(R, over, table=DYST)
        R, s = R2, np.minimum(s, s2)
        tmp = b.with_rotations(R, sorted({j for t in terms + over for j in t.chain}))
    Jt = tmp.joints().numpy()
    S = Jt[:, 17]
    u = (Jw - S) / np.linalg.norm(Jw - S, axis=-1, keepdims=True)
    pole0 = J[:, 19] - J[:, 17]
    lift = -np.radians(A["elbow_lift_deg_per_point"] * it["elbow"]) * env   # the elbow turned up and out
    pole = np.einsum("tij,tj->ti", CT._rot(u, lift), pole0)
    R = arm_ik(tmp, R, "right", torch.as_tensor(Jw), torch.as_tensor(pole), None, end="wrist")
    R, projected = project_rows(R, ARM["right"], b.local_R(), table=DYST)
    rows = sorted({j for t in terms + over for j in t.chain} | set(ARM["right"]))
    out = b.with_rotations(R, rows)
    out = out.with_hands(_turn_hand_rows(out.hand_R(), "right", CRAMP_HAND,
                                         np.clip(A["fingers_per_point"] * it["fingers"] * env, 0.0, 1.0)))
    act = env > 0.0                                      # before the posture the clip is its level 0's
    surf = winfo["surface"]
    if mirror:                                           # the resting right hand keeps its lowest point on the table
        out, seat = seat_hand(out, "right", surf["table_top_y"] + WRITE_REST_GAP_M, hand_vertex_ids("right"), pole,
                              b.local_R(), table=DYST, tol=SEAT_TOL_M, mask=act)
    else:                                                # the writing hand resting on the paper, by its edge or
        out, seat = seat_hand(out, "right", surf["paper_top_y"], resting_low("right", surf["pen_m"]), pole,   # the pen
                              b.local_R(), table=DYST, tol=SEAT_TOL_M, mask=act)
    out = replace(out, contact=contact_labels(out))
    val = LX.check(DYST, b, out)
    val["hand_limits"] = {"ok": bool(H.hand_limit_report(out.hand_R())["ok"])}
    check = table_check_record(ws, ["right"] if mirror else ["left"], strict=mirror)
    val["table"] = table_report_of(out, check)
    wrist_dev = joint_deviation(b, out, 21, segment=False)
    Jo = out.joints().numpy()
    params = {"grade": grade, "wcrs_items": it, "wcrs_writing_movement": wcrs_total(it), "angles": A,
              "latency_s": lat, "ramp_s": WCRS_RAMP_S, "writing": winfo, "overflow": overflow, "mirror": mirror,
              "limit_scale_min": float(s.min()), "pronation_scale_held_max": float(s_pro[env >= 0.5].max()),
              "projected_deg": projected, "realized_wrist_deg": float(wrist_dev.max()),
              "realized_elbow_rise_cm": float(100 * (Jo[:, 19, 1] - J[:, 19, 1]).max()),
              "wrist_shift_mm": float(1000 * np.abs(Jo[:, 21] - Jw).max()), "seat": seat,
              "terms": [t.record() for t in terms + over], "limits_table": LX.record(DYST), "table_check": check}
    if overflow:
        params["realized_head_deg"] = float(joint_deviation(b, out, 15).max())
    lab = "mirror_dystonia" if mirror else ("overflow_dystonia" if overflow else "writers_cramp")
    return out, truth_record(lab, seed, grade, "WCRS (module item scores)", params,
                             ("wcrs", "albanese2013", "twstrs", "dis", "limits_ext", "module"), val,
                             arrays={"envelope": env})


def bone_axis_of(joint, child):
    from .body import bone_axis
    return bone_axis(joint, child)


def scene_mesh(kind, body, floor_y):
    """(vertices, faces) of a clip's scene: 'chair_table' (viewer.chair_mesh and writing_table, the right hand
    writing) and 'chair_table_both' (the left hand writing), other kinds through exam.scene_mesh (which hands the
    two writing scenes to this function)."""
    from . import exam as EX
    from . import viewer as V
    if kind in ("chair_table", "chair_table_both"):
        cv, cf = V.chair_mesh(floor_y)
        tv, tf, _ = writing_table(body, floor_y, write_side="left" if kind == "chair_table_both" else "right")
        return np.concatenate([cv, tv]), np.concatenate([cf, tf + len(cv)])
    return EX.scene_mesh(kind, body, floor_y)


def measure_foot(base, out, side):
    """{component: deg} of the ankle's and the foot joint's rotation against the base (axis-angle of R_out R_base^T,
    the parent frame of the foot terms) at the knot of the largest deviation: plantarflexion (+x), inversion and
    adduction (signed so that positive is the dystonic direction), toe (+x flexion)."""
    a, f = (7, 10) if side == "left" else (8, 11)
    sg = 1.0 if side == "left" else -1.0

    def comp(j):
        D = out.local_R()[:, j] @ base.local_R()[:, j].transpose(-1, -2)
        return np.degrees(matrix_to_axis_angle(D).numpy())
    ca, cf = comp(a), comp(f)
    k = int(np.argmax(np.linalg.norm(ca, axis=-1)))
    return {"plantarflexion_deg": float(ca[k, 0]), "inversion_deg": float(-sg * ca[k, 2]),
            "adduction_deg": float(-sg * ca[k, 1]), "toe_flexion_deg": float(cf[k, 0]), "knot": k}


# ---------------------------------------------------------------- generalized torsion dystonia
TORSION = {"trunk": (("trunk", "rotation_left"), ("trunk", "lateral_right")), "pelvis": (("y", 1.0), ("z", 1.0))}
TORSION_BAND_HZ = (0.1, 0.3)    # [module] the slow twisting movements (patterned, twisting: albanese2013)
TORSION_HOLD = 0.8              # [module] each course holds at 0.8 of its level and varies by +-0.2 (sustained holds)
PELVIS_TORSION_DEG = 12.0       # [module] the pelvis's largest twist and tilt at severity 4 (scaled by the share)
TORSION_KNEE_DEG = 10.0         # [module] the knees' flexion while the pelvis twists (feet planted)
TORSION_NECK = {2: (2, 1), 3: (3, 2)}   # [module] severity: TWSTRS rotation and laterocollis grades of the neck
TORSION_ARMS = {"abduction_deg": {"left": 0.0, "right": 15.0}, "internal_deg": 30.0, "pronation_deg": 45.0,
                "right_elbow": DIS_SHARE[2]}
#   [module] the dystonic arms (review M3: alternating large abductions read as flapping): little abduction, the
#   forearms hyperpronated (the twist of task_specific_arm), the wrists flexed at the severity's share; the left arm
#   extended at the elbow (its full extension room) and turned inward, the right elbow flexed (the hand held in front
#   of the thigh, not on the hip). Open-body port, wave 4 (D57): the abduction was 0.15 of its room on both sides,
#   which grew with jarm2022's room from the hanging arm to 90 deg above the T pose (25.5 deg from the stand), and the
#   trunk's lateral bend to the right takes 0.7 of jarm2022's 50 deg (35 at severity 3), past what the
#   left (convex) arm's shoulder and girdle can take back (10 deg each): that arm rides the bend 31 deg from the
#   vertical with no abduction of its own, 50 with the share; the right arm keeps 15 deg (21.7 from the vertical)
#   (scripts/openbody_wave4_checks.py, torsion)
TORSION_FOOT = {"inversion": DIS_SHARE[3], "plantarflexion": 0.2, "adduction": DIS_SHARE[2]}
#   [module] the left foot of the generalized dystonia, inverted and touching (the legs take part)


def torsion_dystonia(base, severity, seed=0):
    """(Body, truth) of a generalized torsion dystonia (DYT1-like; BFM severity factor 2 or 3 in every region,
    provoking factor 4, present at rest; Albanese 2013 generalized), standing, from the first knot: the trunk
    (rotation and lateral flexion), the pelvis (twist and tilt) and the neck (TWSTRS_NECK levels) on slow independent
    courses (TORSION_HOLD + 0.2 x smooth_noise in TORSION_BAND_HZ: sustained holds that drift, a patterned
    movement), the arms hanging from the twisted trunk (hang_arms) and dystonic (TORSION_ARMS), the left foot inverted
    and touching (TORSION_FOOT), the trunk and arms at the DIS amplitude share of the severity grade (a module map of
    the BFM severity, which has no angle); both feet planted (plant_stand), the left then lifted to its touch."""
    rng = as_rng(seed)
    T, rate = base.T, base.rate
    sv = int(severity)
    sh = DIS_SHARE[sv]

    def course():
        return np.clip(TORSION_HOLD + (1.0 - TORSION_HOLD) * smooth_noise(T, rate, rng, TORSION_BAND_HZ), 0.0, 1.0)
    pre = [region_term(base, r, d, "left", sh * region_room(base, r, d, "left"), course(), f"{r} {d}")
           for r, d in TORSION["trunk"]]
    gr, gl = TORSION_NECK[sv]
    pre += [region_term(base, "neck", "rotation_right", "left", G.twstrs_level_deg("rotation", gr, None), course(),
                        "neck rotation_right"),
            region_term(base, "neck", "tilt_left", "left", G.twstrs_level_deg("laterocollis", gl, None), course(),
                        "neck tilt_left")]
    pre += [joint_term(0, ax, sg * PELVIS_TORSION_DEG * sh / DIS_SHARE[4], course(), "parent", f"pelvis {ax}")
            for ax, sg in TORSION["pelvis"]]
    A = TORSION_ARMS
    arms = []
    for sd in ("left", "right"):
        e = ARM[sd][2]
        arms += [region_term(base, "proximal_arm", "abduction", sd,
                             min(A["abduction_deg"][sd], region_room(base, "proximal_arm", "abduction", sd)),
                             course(),
                             f"proximal_arm abduction {sd}"),
                 arm_internal_term(sd, (A["internal_deg"] if sd == "left" else 0.0) * sh / DIS_SHARE[3], course()),
                 arm_internal_term(sd, A["pronation_deg"] * sh / DIS_SHARE[3], course(), joint=e),
                 region_term(base, "wrist", "flexion", sd, sh * region_room(base, "wrist", "flexion", sd), course(),
                             f"wrist flexion {sd}")]
    arms.append(region_term(base, "distal_arm", "elbow_extension", "left",
                            region_room(base, "distal_arm", "elbow_extension", "left"), course(),
                            "elbow extension left"))
    arms.append(region_term(base, "distal_arm", "elbow_flexion", "right",
                            A["right_elbow"] * region_room(base, "distal_arm", "elbow_flexion", "right"), course(),
                            "elbow flexion right"))
    fenv = course()
    out, tr = limb_posture(base, [], np.ones(T), foot=("left", TORSION_FOOT), label="torsion_dystonia", grade=sv,
                           scale="BFM severity x provoking 4",
                           sources=("bfm", "albanese2013", "dis", "twstrs", "module"),
                           stand=True, extra_terms=arms, knee_deg=TORSION_KNEE_DEG, pre_terms=pre,
                           hang=("left", "right"), foot_env=fenv)
    tr["params"].update(severity=sv, dis_share=sh, provoking_factor=4, band_hz=TORSION_BAND_HZ, hold=TORSION_HOLD,
                        neck_twstrs_grades={"rotation": gr, "laterocollis": gl}, arms=A,
                        foot_measured=measure_foot(base, out, "left"))
    tr["seed"] = seed if isinstance(seed, int) else None
    return out, tr


# ---------------------------------------------------------------- opisthotonus
OPISTHOTONUS = {"neck": 0.45, "trunk": 0.6, "pelvis_deg": -8.0, "knee_deg": 10.0}
#   [module] the arched extension: retrocollis and trunk extension at shares of their rooms, the pelvis tilted back
#   (the neck share lowered from 0.7 after the clinical review, B1: the head was thrown too far back)


def opisthotonus(base):
    """(Body, truth): a sustained extension posture standing (OPISTHOTONUS; no grade: one severe level), the neck in
    retrocollis, the trunk extended and the pelvis tilted back, held from the first knot (a persistent posture), the
    arms hanging (hang_arms) with the elbows relaxed, the feet planted and the centre of mass kept over them
    (plant_stand)."""
    o = OPISTHOTONUS
    env = np.ones(base.T)
    terms = [region_term(base, "neck", "retrocollis", "left", o["neck"] * region_room(base, "neck", "retrocollis",
                                                                                       "left"), env, "retrocollis"),
             region_term(base, "trunk", "extension", "left", o["trunk"] * region_room(base, "trunk", "extension",
                                                                                      "left"), env, "trunk extension"),
             joint_term(0, "x", o["pelvis_deg"], env, "parent", "pelvis tilted back")] + _hang_elbows(base, env)
    R, s = fit_terms(base.local_R(), terms, table=DYST)
    R, hang = hang_arms(base, R, table=DYST)
    out, info = plant_stand(base, R, o["knee_deg"] * env)
    info["arms_hang"] = hang
    ang = axial_angles(out)
    params = {"levels": o, "terms": [t.record() for t in terms], "limit_scale_min": float(s.min()), "stand": info,
              "held_tcc_deg": float(ang["tcc"][-1]), "held_neck_deg": float(ang["neck"][-1]),
              "base_tcc_deg": float(axial_angles(base, knots=[0])["tcc"][0]), "limits_table": LX.record(DYST)}
    return out, truth_record("opisthotonus", None, "severe", "none (descriptive)", params,
                             ("albanese2013", "limits_ext", "module"), stand_validity(base, out, DYST),
                             arrays={"envelope": env, "axial_tcc": ang["tcc"], "axial_neck": ang["neck"]})


# ---------------------------------------------------------------- paroxysmal dyskinesias
PKD = {"latency_s": 0.3, "duration_s": 10.0, "on_s": 0.4, "off_s": 1.5, "unfold_trunk_deg": 20.0,
       "unfold_dur_s": 0.8}
#   unfold_dur_s: open-body port, wave 6 follow-up, 0.6 before: the open body's fold opens on one clock (motions D85,
#   no lag), and over 0.6 s its shoulders turned 3.60 and 3.41 deg in one knot, past the 300 deg/s of the snap review
#   (M2 of the SH2 wave); the least of 0.1 s steps at which the opening keeps every arm joint under it (0.7: 3.08, 0.8:
#   2.70 deg; scripts/openbody_wave6_checks.py, pkd_opening)
#   [module] the attack begins latency_s after lift-off (the sudden movement) or as the arms open, whichever is later,
#   and lasts duration_s, within the 'few seconds up to 1 min' of latorre_bhatia (Bruno et al. 2004), rising over
#   on_s, fading over off_s; the arms, folded across the chest for the arising, unfold to the sides over unfold_dur_s
#   once the trunk (trunk_flexion_deg) is back within unfold_trunk_deg of the vertical after lift-off, and join the
#   attack as they unfold (review M6: an attack acting on folded arms reads as a shrug of the folded arms; review of
#   2026-11-02: arms opened during the forward lean of the lift-off put the hands by the thighs and on the armrests)
PKD_TRUNK_GUARD_ROWS = {sd: (3, 6, 9, ARM[sd][1]) for sd in ARM}   # [module] the rows the PKD attack's trunk guard
#   holds back: the spine (the attack's lateral flexion toward the arm) and the shoulder (open-body port, wave 6
#   follow-up: the shoulder alone, before, held at its base, left the open body's dystonic forearm within TRUNK_CLEAR_M
#   of the abdomen, the trunk bending onto it, and the humerus's inward turn under the 5 deg its test asks;
#   scripts/openbody_wave6_checks.py, pkd_opening)
PKD_ARMS_CHECKS = ("forearms", "hand_left_forearm_right", "hand_right_forearm_left", "hands")   # [module] the skin
#   pairs of the two arms the PKD attack keeps apart over the opening of the fold and UNFOLD_CHECK_PAD_S after it (the
#   attacking arms held back toward the opening, guard_rows; open-body port, wave 6 follow-up: with the trunk guard on
#   the spine the dystonic arm's inward turn carried the left hand into the right one as the arms opened)
ARMREST_CLEAR_M = 0.002         # [module] the hands and forearms kept this far outside a chair's armrests
ARMREST_PASSES = 4              # [module] passes of the armrest clearance (clear_armrests)
SCENE_TOL_M = 0.0005            # [module] the armrest and thigh checks' tolerance on an entry deeper than the base's
SCENE_CHECK_STEP = 5            # [module] knots between two reads of the armrest and thigh checks (0.05 s)
TRUNK_CLEAR_M = 0.003           # [module] the hands and forearms of an attack kept this far off the skin of the trunk
#                                 and of the thighs, or no nearer than the level 0's own (guard_rows; wave SH1c: the
#                                 dystonic PKD forearm lay inside the abdomen over the whole attack)
PNKD_THIGH_CLEAR_M = 0.01       # [module] the PNKD excerpt's hands and forearms kept this far off the thighs (the
#   guard rounds of paroxysmal_excerpt; plan phase E): with the trunk guard alone the left hand passed 0.36 mm over the
#   thigh at 1.95 s, and the background layer (physio.py, its own arms) carried it 3.80 mm inside, which the excerpt's
#   'hand_thigh' read on the layered body refused (wave8b_checks.json layer); reach.CLEAR_M, the seated hands' distance
#   from the thighs of the same phase (dystonia.arm_lift), held by the shoulder, elbow and wrist rows (the shoulder and
#   elbow alone found no share that ends the approach: the wrist's flexion carries the hand down), keeps it outside: the
#   left hand 10.21 mm off on the raw body, 10.67 on the layered one (scripts/openbody_phaseE_checks.py pnkd)
GUARD_STEP = 10                 # [module] knots between two reads of guard_rows (0.1 s; its share is smoothed over
#                                 FIT_SMOOTH_S, and the checks read every SCENE_CHECK_STEP knots)
GUARD_FAST_STEP = 2             # [module] knots between two reads of the guard on a chorea's fast arms (review of wave
#                                 SH1c: a hand swinging past the thigh entered it between two reads of GUARD_STEP and of
#                                 SCENE_CHECK_STEP); the checks of an attack on its way past the skin read every knot
GUARD_BISECT = 6                # [module] bisection steps of guard_rows' share (1/64 of the way back to the base)
GUARD_PASSES = 3                # [module] passes of guard_rows
GUARD_GRID = 8                  # [module] steps of guard_rows' grid where no share ends an entry (the share of least
#                                 need)
GUARD_ROUNDS = 4                # [module] rounds of the PNKD excerpt's armrest clearance and trunk guard in turn
#                                 (paroxysmal_excerpt: each can move the forearm the other judges; the guard reads
#                                 every GUARD_STEP knots in the first round, every knot after it: review of wave SH1c,
#                                 an entry between two reads)
CHOREA_BAND_HZ = (0.5, 2.5)     # [module] the irregular, flowing choreic movements
CHOREA_BLEND = 0.3              # [module] the width, in course units, of chorea_term's passage between its two slopes
#                                 (review of wave SH1c: a slope changing in one knot put a hitch in the velocity at
#                                 every zero crossing of the course)
CHOREA_SHARE = 0.7              # [module] the choreic excursion, share of each direction's room (review M5: 0.35 of
#                                 the smaller room read as a slight fidget)
CHOREA_ARM_DEG = {"abduction": 25.0, "rotation": 25.0, "elbow": 30.0}
#   [module] the choreic excursions of a free arm: shoulder abduction (one way: the hanging arm has under 1 deg of
#   adduction room in the table), the humerus turned both ways about itself, the elbow both ways (at most this much each
#   way, CHOREA_SHARE of its room: a hanging elbow has little extension room)
DYSTONIC_ARM = {"internal_deg": 30.0, "pronation_deg": 40.0}
#   [module] the dystonic arm of an attack: the humerus turned inward and the forearm pronated, with elbow and wrist
#   flexion at the DIS grade 3 share, no abduction (review M5 and M6: an abducted, scarecrow arm)
WRITHE = (0.125, (0.2, 0.5))    # [module] the dystonic posture's slow writhing: +-share of the level, band (Hz)


def chorea_term(base, region, d_pos, d_neg, side, share, course, label, k=0, cap_deg=None):
    """One Term moving a region both ways on a course (T,) in [-1, 1]: +1 at share of the room toward d_pos, -1 at
    share of the room toward d_neg (rooms read at knot k; each at most cap_deg), 0 at 0 (SH2 wave: the single line from
    -1 to +1 put an offset at a course of 0, so that the choreic PKD clip left its level 0 before the attack), its slope
    passing smoothly between the two sides' over CHOREA_BLEND."""
    rp, rn = region_room(base, region, d_pos, side, k), region_room(base, region, d_neg, side, k)
    ap, an = share * rp, share * rn
    if cap_deg is not None:
        ap, an = min(ap, cap_deg), min(an, cap_deg)
    if ap <= 1e-9:
        return region_term(base, region, d_neg, side, an, np.clip(-course, 0.0, 1.0), label, k)
    c = np.asarray(course, dtype=np.float64)
    w = 0.5 * (1.0 + np.tanh(c / CHOREA_BLEND))
    g = c * (w + (1.0 - w) * an / ap)
    return region_term(base, region, d_pos, side, ap, g, label, k)


def attack_terms(base, env, kind, rng, side="left", k=0, sides=None, arm_env=None):
    """Terms of a paroxysmal attack on the envelope env: 'dystonic' (the arm of side turned inward with elbow and
    wrist flexion and forearm pronation, DYSTONIC_ARM and the DIS grade 3 shares, writhing slowly by WRITHE; the
    trunk bent and the neck tilted toward it at DIS grade 2) or 'choreic' (irregular movements of both arms (sides,
    default both) at the shoulder, humerus, elbow and wrist, of the neck and the trunk, on independent smooth
    courses in CHOREA_BAND_HZ: CHOREA_ARM_DEG for the arm's large joints, CHOREA_SHARE of each direction's room for
    the wrist, neck and trunk); the rooms read at knot k. arm_env (T,): the arms' envelope (default env)."""
    sh = DIS_SHARE[3]
    T, rate = base.T, base.rate
    ea = env if arm_env is None else np.asarray(arm_env, dtype=np.float64)
    if kind == "dystonic":
        ew = ea * (1.0 - WRITHE[0] + WRITHE[0] * smooth_noise(T, rate, rng, WRITHE[1]))
        e = ARM[side][2]
        parts = [("distal_arm", "elbow_flexion", side, sh), ("wrist", "flexion", side, sh)]
        out = [region_term(base, r, d, sd, f * region_room(base, r, d, sd, k), ew, f"{r} {d} {sd}", k)
               for r, d, sd, f in parts]
        out += [arm_internal_term(side, DYSTONIC_ARM["internal_deg"], ew),
                arm_internal_term(side, DYSTONIC_ARM["pronation_deg"], ew, joint=e)]
        out += [region_term(base, r, d, sd, f * region_room(base, r, d, sd, k), env, f"{r} {d} {sd}", k)
                for r, d, sd, f in (("trunk", f"lateral_{side}", "left", DIS_SHARE[2]),
                                    ("neck", f"tilt_{side}", "left", DIS_SHARE[2]))]
        return out
    out = []

    def course(e=env):
        return e * smooth_noise(T, rate, rng, CHOREA_BAND_HZ, 6)
    for sd in (sides or ("left", "right")):
        ca = CHOREA_ARM_DEG
        out.append(region_term(base, "proximal_arm", "abduction", sd, ca["abduction"],
                               ea * 0.5 * (1.0 + smooth_noise(T, rate, rng, CHOREA_BAND_HZ, 6)),
                               f"chorea shoulder abduction {sd}", k))
        out.append(arm_internal_term(sd, ca["rotation"], course(ea)))
        out[-1].label = f"chorea humerus rotation {sd}"
        out.append(chorea_term(base, "distal_arm", "elbow_flexion", "elbow_extension", sd, CHOREA_SHARE, course(ea),
                               f"chorea elbow {sd}", k, cap_deg=ca["elbow"]))
        out.append(chorea_term(base, "wrist", "flexion", "extension", sd, CHOREA_SHARE, course(ea),
                               f"chorea wrist {sd}", k))
    for r, d, opp in (("neck", "rotation_left", "rotation_right"), ("trunk", "rotation_left", "rotation_right")):
        out.append(chorea_term(base, r, d, opp, "left", 0.5 * CHOREA_SHARE, course(), f"chorea {r}", k))
    return out


AXIAL_ROWS = (3, 6, 9, 12, 15)


def attack_groups(terms):
    """[arm flexions, arm turns, trunk and neck]: the terms of attack_terms grouped for fit_groups (review of the SH2
    wave: one fit for all of them let a hanging shoulder at its adduction bound shrink the trunk, the neck and the
    elbow's flexion with the humerus's inward turn, and the flexed forearm dropped in one knot); the arm flexions
    (elbow, wrist, shoulder abduction: about an axis of the joint's frame) come before the turns about a bone's own
    axis (the humerus's inward turn, the pronation) acting on the same joints, as in attack_terms' order."""
    axial = [t for t in terms if set(t.chain) <= set(AXIAL_ROWS)]
    arm = [t for t in terms if not set(t.chain) <= set(AXIAL_ROWS)]
    return [[t for t in arm if isinstance(t.axis, str)], [t for t in arm if not isinstance(t.axis, str)], axial]


def trunk_flexion_deg(body):
    """(T,) deg: the angle between the line from the pelvis joint to the neck joint (0 to 12) and the vertical."""
    J = body.joints().numpy()
    v = J[:, 12] - J[:, 0]
    return np.degrees(np.arccos(np.clip(v[:, 1] / np.linalg.norm(v, axis=-1), -1.0, 1.0)))


def armrest_boxes(kind, body):
    """{side: (lo (3,), hi (3,))} m of the armrests of a scene kind (exam.scene_mesh): 'chair' (viewer.chair_mesh,
    its boxes 2 and 3) or 'chair_arise' (motions.arise_chair under the body's first knot, its boxes 6 and 7); the
    left armrest on the body's left (x > 0). The armrests do not depend on the floor plane."""
    if kind == "chair":
        from .viewer import chair_mesh
        V = chair_mesh(0.0)[0].reshape(-1, 8, 3)[[2, 3]]
    elif kind == "chair_arise":
        from .motions import arise_chair
        V = arise_chair(body, 0.0)[0].reshape(-1, 8, 3)[[6, 7]]
    else:
        raise ValueError(f"armrests of scene {kind!r}")
    return {"left": (V[0].min(0), V[0].max(0)), "right": (V[1].min(0), V[1].max(0))}


@functools.lru_cache(maxsize=2)
def arm_vertex_ids(side):
    """The vertices of the forearm, the wrist and the hand of side (dominant joint on the neutral rest mesh)."""
    _, _, dom = H._rest()
    joints = (ARM[side][2], ARM[side][3]) + tuple(range(H.FIRST[side], H.FIRST[side] + 15))
    return np.flatnonzero(np.isin(dom, joints))


def _box_depth(P, lo, hi, grow=0.0):
    """(K,) m: the deepest penetration (>= 0) of points P (K, N, 3) into the box [lo - grow, hi + grow]."""
    d = np.minimum(P - (lo - grow), (hi + grow) - P).min(-1)
    return np.maximum(d, 0.0).max(-1)


def armrest_depth(body, side, boxes, knots, grow=0.0):
    """(K,) m: the deepest penetration of the side's forearm and hand vertices into either armrest of boxes
    (armrest_boxes), grown by grow, at the knots."""
    P = H.body_points(body, arm_vertex_ids(side), knots=knots)
    return np.maximum(_box_depth(P, *boxes["left"], grow), _box_depth(P, *boxes["right"], grow))


def armrest_report(body, kind, knots=None, base=None):
    """{side: {'depth_mm_max', 'at_s', 'base_depth_mm_max', 'new_mm_max'}, 'n_knots', 'ok'}: the deepest entry of each
    forearm and hand into the armrests of the scene kind (armrest_depth) every SCENE_CHECK_STEP knots (or at knots);
    with a base body on the same clock, the entry the operator adds to the base's own at each knot (new_mm_max);
    ok when nothing enters (without base) or nothing new enters (more than SCENE_TOL_M deeper than the base)."""
    boxes = armrest_boxes(kind, body)
    k = np.arange(0, body.T, SCENE_CHECK_STEP) if knots is None else np.asarray(knots)
    rep, ok = {}, True
    for sd in ("left", "right"):
        d = armrest_depth(body, sd, boxes, k)
        i = int(np.argmax(d))
        rep[sd] = {"depth_mm_max": float(1000 * d[i]), "at_s": float(k[i] / body.rate)}
        if base is not None:
            d0 = armrest_depth(base, sd, boxes, k)
            new = np.maximum(d - d0, 0.0)
            rep[sd].update(base_depth_mm_max=float(1000 * d0.max()), new_mm_max=float(1000 * new.max()))
            ok = ok and bool(new.max() <= SCENE_TOL_M)
        else:
            ok = ok and bool(d[i] <= 0.0)
    rep.update(n_knots=int(len(k)), ok=ok)
    return rep


SKIN_CHECKS = {"thighs": "thigh", "trunk": "trunk"}   # the motions.fold_gaps checks of a part, by side
SKIN_READ_M = 0.03              # [module] arm_part_gap reads fold_gaps only where reach.clearance is under this: they
#                                 differ only in the sign of a vertex inside the part, a part's skin then near


def arm_part_gap(body, side, knots, part, skin=True):
    """(K,) m: the least signed distance between the hand or forearm of side and the skin of part at the knots, negative
    inside: reach.clearance, whose sign is the closest triangle's, and for a part of SKIN_CHECKS the lower reading of
    motions.fold_gaps, whose sign is the winding number (the part meshes are open at their joints: review of wave SH1c,
    the choreic PKD's right hand lay inside the thigh between two reads). skin False: reach.clearance alone (a guard's
    many reads over a long clip; the check then reads both)."""
    from .motions import fold_gaps
    from .reach import clearance
    k = np.asarray(knots, dtype=np.int64)
    c = clearance(body, side, k, parts=(part,))[part]
    gap = np.minimum(c["hand"], c["fore"])
    near = np.flatnonzero(gap < SKIN_READ_M)
    if skin and part in SKIN_CHECKS and len(near):
        check = f"{SKIN_CHECKS[part]}_{side}"
        gap[near] = np.minimum(gap[near], fold_gaps(body, k[near], None, (check,))[0][check])
    return gap


def part_report(body, knots, part, sides=("left", "right"), base=None):
    """{side: {'min_mm', 'at_s', 'base_min_mm', 'new_mm_max'}, 'ok'}: the least signed distance between each hand or
    forearm and the skin of part (reach.PARTS: 'thighs', 'trunk' or 'head'; arm_part_gap, on the skinned mesh; negative
    inside) at the knots; with a base body on the same clock, how much deeper than the base's own entry the operator
    goes at a knot (new_mm_max); ok when nothing enters (without base) or nothing new enters (more than SCENE_TOL_M
    deeper)."""
    rep, ok = {}, True
    for sd in sides:
        both = arm_part_gap(body, sd, knots, part)
        i = int(np.argmin(both))
        rep[sd] = {"min_mm": float(1000 * both[i]), "at_s": float(np.asarray(knots)[i] / body.rate)}
        if base is not None:
            k = np.asarray(knots, dtype=np.int64)
            at = np.union1d(np.arange(0, len(k), SCENE_CHECK_STEP), np.flatnonzero(both < 0.0))
            b0 = arm_part_gap(base, sd, k[at], part)   # where the body enters, and every SCENE_CHECK_STEP-th read
            new = np.where(both[at] < 0.0, np.maximum(np.minimum(b0, 0.0) - both[at], 0.0), 0.0)
            rep[sd].update(base_min_mm=float(1000 * b0.min()), new_mm_max=float(1000 * new.max()))
            ok = ok and bool(new.max() <= SCENE_TOL_M)
        else:
            ok = ok and bool(both[i] >= 0.0)
    rep["ok"] = ok
    return rep


def thigh_report(body, knots, sides=("left", "right"), base=None):
    """part_report on the thighs: the hand-to-thigh check."""
    return part_report(body, knots, "thighs", sides, base)


def trunk_report(body, knots, sides=("left", "right"), base=None):
    """part_report on the trunk (pelvis and spine segments): the hand-to-trunk check (wave SH1c)."""
    return part_report(body, knots, "trunk", sides, base)


def backrest_box(kind, body=None):
    """(lo (3,), hi (3,)) m of the backrest of a scene kind: 'chair' or 'chair_armless' (viewer.chair_mesh, its box 1,
    the same in both)."""
    if kind not in ("chair", "chair_armless"):
        raise ValueError(f"backrest of scene {kind!r}")
    from .viewer import chair_mesh
    V = chair_mesh(0.0, armrests=False)[0].reshape(-1, 8, 3)[1]
    return V.min(0), V.max(0)


def backrest_report(body, kind, knots=None, base=None):
    """{'depth_mm_max', 'at_s', 'part', 'base_depth_mm_max', 'new_mm_max', 'n_knots', 'ok'}: the deepest entry of any
    vertex of the body into the backrest of the scene kind (backrest_box) every SCENE_CHECK_STEP knots (or at knots),
    the dominant joint of the deepest vertex ('part'); with a base body on the same clock, the entry the operator adds
    to the base's own at each knot; ok when nothing enters (without base) or nothing new enters by more than
    SCENE_TOL_M (wave SH1c: the right upper arm of the PNKD excerpt entered the backrest)."""
    lo, hi = backrest_box(kind, body)
    k = np.arange(0, body.T, SCENE_CHECK_STEP) if knots is None else np.asarray(knots)
    sid = _skin_ids()
    P_ = H.body_points(body, sid, knots=k)
    d = np.maximum(np.minimum(P_ - lo, hi - P_).min(-1), 0.0)
    dk = d.max(-1)
    i = int(np.argmax(dk))
    dom = H._rest()[2][sid]
    rep = {"depth_mm_max": float(1000 * dk[i]), "at_s": float(k[i] / body.rate),
           "part": int(dom[int(np.argmax(d[i]))]) if dk[i] > 0 else None, "n_knots": int(len(k))}
    if base is not None:
        P0 = H.body_points(base, sid, knots=k)
        d0 = np.maximum(np.minimum(P0 - lo, hi - P0).min(-1), 0.0).max(-1)
        new = np.maximum(dk - d0, 0.0)
        rep.update(base_depth_mm_max=float(1000 * d0.max()), new_mm_max=float(1000 * new.max()),
                   ok=bool(new.max() <= SCENE_TOL_M))
    else:
        rep["ok"] = bool(dk[i] <= 0.0)
    return rep


def scene_checks_record(kind, **args):
    """{'kind', ...}: the arguments of a clip's scene checks (scene_checks_of), kept in its truth
    (params['layer_checks']) so that they run again on another body of the clip, its layered body (physio.LayeredBuild,
    Phase D of docs/PLAN_2026-11-12.md: the layered truth copied 'fold', 'hand_thigh', 'hand_trunk' and 'backrest'
    from the raw body; only the writing clips' 'table' was read again). kind: 'pnkd' (duration_s), 'pkd' (standing_s,
    lift_knot, unfold_knot, unfold_knots) or 'geste' (side, knots)."""
    if kind not in ("pnkd", "pkd", "geste"):
        raise ValueError(f"scene checks of {kind!r}")
    return dict(kind=kind, **args)


def scene_checks_of(body, rec, base=None):
    """{validity entry: report} of the scene checks of a scene_checks_record rec on body: 'pnkd' the armrests, the
    backrest, the trunk and the thighs against the seated base (every SCENE_CHECK_STEP knots); 'pkd' the arising
    chair's armrests, the thighs and the trunk from lift-off against the arising level 0 (motions.arise, or base) and
    the opening of the fold (motions.unfold_check); 'geste' the head and the trunk at the geste's check knots
    (head_clearance)."""
    kind = rec["kind"]
    if kind == "pnkd":
        from .motions import posture_body
        b = posture_body("sit", rec["duration_s"]) if base is None else base
        k = np.arange(0, body.T, SCENE_CHECK_STEP)
        return {"armrest": armrest_report(body, "chair"), "backrest": backrest_report(body, "chair", base=b),
                "hand_trunk": trunk_report(body, k, base=b), "hand_thigh": thigh_report(body, k, base=b)}
    if kind == "pkd":
        from .motions import arise, unfold_check
        b0 = arise(standing_s=rec["standing_s"])[0] if base is None else base
        after = np.arange(rec["lift_knot"], body.T)   # every knot: a swinging hand passes a thigh between two reads
        return {"armrest": armrest_report(body, "chair_arise", base=b0),
                "hand_thigh": thigh_report(body, after, base=b0), "hand_trunk": trunk_report(body, after, base=b0),
                "fold": unfold_check(body, rec["unfold_knot"], rec["unfold_knots"])}
    if kind == "geste":
        k = np.asarray(rec["knots"], dtype=np.int64)
        return {"hand_head": head_clearance(body, rec["side"], k),
                "hand_trunk": head_clearance(body, rec["side"], k, part="trunk")}
    raise ValueError(f"scene checks of {kind!r}")


BACKREST_ROWS = (3, 6, 9)       # the spine rows that turn and bend the trunk (with an arm's collar and shoulder rows,
#                                 guard_rows' rows for the backrest)
BACKREST_TRUNK = (0, 3, 6, 9)   # the pelvis and spine segments (dominant joints) read against the backrest (D58)


def _clear_need(b, side, part, clear_m, skin=True):
    """need of guard_rows: (K,) m by which the hand or forearm of side comes nearer to the skin of part (arm_part_gap,
    with its skin) than clear_m, or than the base b's own distance at the knot where that is smaller."""
    cache = {}

    def base_at(kk):
        miss = sorted({int(x) for x in kk} - set(cache))
        if miss:
            sub = crop_knots(b, miss)
            g = arm_part_gap(sub, side, np.arange(sub.T), part, skin)
            for i, x in enumerate(miss):
                cache[x] = float(g[i])
        return np.array([cache[int(x)] for x in kk])

    def need(sub, kk):
        return np.minimum(clear_m, base_at(kk)) - arm_part_gap(sub, side, np.arange(sub.T), part, skin)
    return need


def _arms_need(k0, k1, checks=PKD_ARMS_CHECKS):
    """need of guard_rows: (K,) m by which the two arms' skins (motions.fold_gaps, checks) come nearer than
    motions.FOLD_OPEN_CLEAR_M at the knots in [k0, k1) (an opening of the fold and its pad); none elsewhere."""
    from .motions import FOLD_OPEN_CLEAR_M, fold_gaps

    def need(sub, kk):
        kk = np.asarray(kk)
        out = np.full(len(kk), -1.0)
        inw = np.flatnonzero((kk >= k0) & (kk < k1))
        if len(inw):
            g, _ = fold_gaps(sub, inw, None, checks)
            out[inw] = FOLD_OPEN_CLEAR_M - np.min(np.stack([g[c] for c in checks]), axis=0)
        return out
    return need


def _armrest_need(b, side, kind):
    """need of guard_rows: (K,) m: the hand and forearm of side inside the armrests of the scene kind, grown by
    ARMREST_CLEAR_M (armrest_depth)."""
    boxes = armrest_boxes(kind, b)

    def need(sub, kk):
        return armrest_depth(sub, side, boxes, np.arange(sub.T), ARMREST_CLEAR_M)
    return need


def _backrest_need(b, side, kind):
    """need of guard_rows: (K,) m by which the arm of side (its collar, shoulder, elbow, wrist and finger segments)
    comes nearer to the backrest of the scene kind (backrest_box) than viewer.CHAIR_BACK_CLEAR_M (the trunk's own
    clearance), beyond the base b's own entry into that margin at the knot. side 'trunk': the pelvis and spine
    segments' skin entering the backrest itself (no margin: the back may lean on it), beyond the base's own entry
    (open-body port, wave 4, D58: the PNKD excerpt's upper back entered the backrest by 13 mm)."""
    from .viewer import CHAIR_BACK_CLEAR_M
    lo, hi = backrest_box(kind, b)
    _, _, dom = H._rest()
    if side == "trunk":
        ids, margin = np.flatnonzero(np.isin(dom, BACKREST_TRUNK)), 0.0
    else:
        ids = np.flatnonzero(np.isin(dom, ARM[side] + tuple(range(H.FIRST[side], H.FIRST[side] + 15))))
        margin = CHAIR_BACK_CLEAR_M
    cache = {}

    def depth(body):
        return _box_depth(H.body_points(body, ids), lo, hi, margin)

    def need(sub, kk):
        miss = sorted({int(x) for x in kk} - set(cache))
        if miss:
            d0 = depth(crop_knots(b, miss))
            for i, x in enumerate(miss):
                cache[x] = float(d0[i])
        return depth(sub) - np.array([cache[int(x)] for x in kk])
    return need


def blend_components(R0, R1, w, rows):
    """R0 with the rows' axis-angle components moved toward R1's on straight lines by shares w (T,): a joint inside
    the joint-limit table's box at both ends stays inside it (and a shoulder under a collar blended by the same share
    inside its coupled bound, limits_ext), where a geodesic blend can leave it."""
    Rn = R0.clone()
    W = torch.as_tensor(np.asarray(w, dtype=np.float64))[:, None]
    for j in rows:
        a0, a1 = matrix_to_axis_angle(R0[:, j]), matrix_to_axis_angle(R1[:, j])
        Rn[:, j] = axis_angle_to_matrix(a0 + W * (a1 - a0))
    return Rn


def guard_rows(b, R, rows, need, step=GUARD_STEP, passes=GUARD_PASSES):
    """(R, w (T,), info): R (an operator's rotations on the base body b) with the rows blended back toward b's
    (blend_components) by a share w where the operator would put the body where it must not be: need(body, knots) gives
    (K,) m, positive where a body at the knots enters (a hand in the trunk, an arm in a backrest) beyond what is
    allowed. At each knot read (every step) where it does, the smallest share on a bisection of GUARD_BISECT steps that
    ends it; where no share ends it (the entry comes from rows the guard does not hold, as a trunk's lean), the share of
    least need on a grid of GUARD_GRID steps from the knot's share to 1 (the smallest of equal ones: review of wave
    SH1c, the share 1 taken there held the PNKD excerpt's whole left arm at its base and still entered); linear between
    the knots read, then a running maximum over +-FIT_SMOOTH_S and a running mean of the same width (never below the
    share found at a knot read), a few passes: the attack is held back where the body would meet itself or the chair,
    the rest of it untouched. info: the knots that needed it, the largest share, the knots still entering after the last
    pass."""
    T = b.T
    base_R = b.local_R()
    ks = np.unique(np.r_[np.arange(0, T, step), T - 1])
    k_s = max(1, int(round(FIT_SMOOTH_S * b.rate)))
    w = np.zeros(T)

    def at(kn, wk):
        kk = list(kn) * (2 if len(kn) == 1 else 1)
        wv = np.asarray(list(wk) * (2 if len(kn) == 1 else 1), dtype=np.float64)
        sub = crop_knots(b, kk)
        return sub.with_rotations(blend_components(R[kk], base_R[kk], wv, rows)), kk

    def needs(kn, wk):
        sub, kk = at(kn, wk)
        return np.asarray(need(sub, np.asarray(kk)))[:len(kn)]
    info = {"knots": 0, "passes": 0, "rows": list(rows)}
    for _ in range(passes):
        d = needs(ks, w[ks])
        bad = d > 0.0
        info["passes"] += 1
        if not bad.any():
            break
        info["knots"] = max(info["knots"], int(bad.sum()))
        kb = ks[bad]
        lo_, hi_ = w[kb].copy(), np.ones(len(kb))
        for _ in range(GUARD_BISECT):
            m = 0.5 * (lo_ + hi_)
            f = needs(kb, m) <= 0.0
            hi_, lo_ = np.where(f, m, hi_), np.where(f, lo_, m)
        never = hi_ >= 1.0
        if never.any():
            kn = kb[never]
            never[never] = needs(kn, np.ones(len(kn))) > 0.0
        if never.any():                                  # no share ends it: the share of least need
            kn, w0 = kb[never], w[kb[never]]
            grid = [w0 + (1.0 - w0) * q for q in np.linspace(0.0, 1.0, GUARD_GRID + 1)]
            nd = np.stack([needs(kn, g_) for g_ in grid])
            hi_[never] = np.stack(grid)[np.argmin(nd, axis=0), np.arange(len(kn))]
            info["least_need_knots"] = info.get("least_need_knots", 0) + int(never.sum())
        sk = w[ks].copy()
        sk[bad] = hi_
        lin = np.interp(np.arange(T), ks, sk)
        mx = np.max([np.pad(lin, k_s, mode="edge")[i:i + T] for i in range(2 * k_s + 1)], axis=0)
        sm = np.convolve(np.pad(mx, k_s, mode="edge"), np.ones(2 * k_s + 1) / (2 * k_s + 1), mode="valid")
        w = np.clip(np.maximum(w, sm), 0.0, 1.0)
    left = needs(ks, w[ks]) > 0.0
    info.update(share_max=float(w.max()), still_entering=int(left.sum()))
    return blend_components(R, base_R, w, rows), w, info


def _collar_lifted(R, side, deg):
    """(K, 22, 3, 3): R with the collar of side raised by deg (K,) about z in its parent's frame and the shoulder turned
    back by as much (the arm keeps its rotation in the collar's parent frame: dystonia.hang_from_collar)."""
    c = ARM[side][0]
    sg = 1.0 if side == "left" else -1.0                 # elevation: +z on the left, -z on the right (dystonia.REGIONS)
    Rn = R.clone()
    Rn[:, c] = _rot_about(torch.tensor([0.0, 0.0, 1.0], dtype=F64), np.radians(sg * np.asarray(deg))) @ R[:, c]
    return hang_from_collar(R, Rn, (c,))


def clear_armrests(b, terms, kind, table, passes=ARMREST_PASSES, groups=None, smooth=0, fitted=None,
                   sides=("left", "right")):
    """(R, scale (T,) of fit_terms, {side: (T,) deg}, info): the terms on b (fit_terms in table, or fit_groups on
    groups, smooth knots; or fitted, (R, scale) of such a fit already made and guarded); where a forearm or hand of one
    of sides would enter an armrest of the scene kind (grown by ARMREST_CLEAR_M), that side's collar is raised with the
    arm kept hanging under it (_collar_lifted; the SH2 table's coupled shoulder bound): the forearm rests on the armrest
    and holds the shoulder up. At each such knot the smallest raise on a 1 deg grid (up to the collar's room in table)
    that keeps it out, the raises smoothed (a running maximum over 0.15 s, then a running mean), a few passes, the sum
    of the passes kept inside the collar's room at each knot (review of wave SH1c: passes adding up past it left the
    table).
    An arm that swings freely over an armrest does not lean on it: leave it out of sides and hold the arm itself back
    (paroxysmal_excerpt). info: the knots that needed it, the largest raise and the knots still entering after the last
    pass, on both sides."""
    T = b.T
    boxes = armrest_boxes(kind, b)
    if fitted is not None:
        R1, s = fitted
    elif groups:
        R1, s = fit_groups(b.local_R(), groups, table=table, smooth=smooth)
    else:
        R1, s = fit_terms(b.local_R(), terms, table=table, smooth=smooth)
    rows = sorted({j for t in terms for j in t.chain} | {13, 14, 16, 17})
    lift = {"left": np.zeros(T), "right": np.zeros(T)}
    k = max(1, int(round(0.15 * b.rate)))
    comp1 = np.degrees(matrix_to_axis_angle(R1).numpy())
    bnd1 = LX.knot_bounds(table, comp1, np.degrees(matrix_to_axis_angle(b.local_R()).numpy()))
    info = {"passes": 0}

    def lifted(sel=slice(None)):
        R = R1[sel]
        for sd in ("left", "right"):
            if lift[sd].max() > 0:
                R = _collar_lifted(R, sd, lift[sd][sel])
        return R
    for _ in range(passes):
        out = b.with_rotations(lifted(), rows)
        moved = False
        for sd in sides:
            bad = np.flatnonzero(armrest_depth(out, sd, boxes, np.arange(T), ARMREST_CLEAR_M) > 0.0)
            if not len(bad):
                continue
            moved = True
            sel = list(bad) if len(bad) > 1 else [int(bad[0])] * 2
            sub = crop_knots(b, sel)
            c, sg = ARM[sd][0], (1.0 if sd == "left" else -1.0)
            room = bnd1[1 if sg > 0 else 0][:, c, 2] * sg - sg * comp1[:, c, 2]
            cap = float(np.min(room))
            need = np.full(len(sel), np.nan)
            for e in np.arange(1.0, max(cap, 1.0) + 1e-9, 1.0):
                Rl = _collar_lifted(lifted(sel), sd, np.full(len(sel), e))
                ok = armrest_depth(sub.with_rotations(Rl, rows), sd, boxes, np.arange(len(sel)), ARMREST_CLEAR_M) <= 0
                need = np.where(np.isnan(need) & ok, e, need)
                if not np.isnan(need).any():
                    break
            add = np.zeros(T)
            add[bad] = np.nan_to_num(need[:len(bad)], nan=max(cap, 0.0))
            add = np.max([np.pad(add, k, mode="edge")[i:i + T] for i in range(2 * k + 1)], axis=0)
            add = np.convolve(np.pad(add, k, mode="edge"), np.ones(2 * k + 1) / (2 * k + 1), mode="valid")
            lift[sd] = np.minimum(lift[sd] + add, np.maximum(room, 0.0))
            info.setdefault(sd, {})["knots"] = int(len(bad))
        info["passes"] += 1
        if not moved:
            break
    info.update({sd + "_lift_deg_max": float(lift[sd].max()) for sd in ("left", "right")})
    out = b.with_rotations(lifted(), rows)
    info.update({sd + "_still_entering": int((armrest_depth(out, sd, boxes, np.arange(T), ARMREST_CLEAR_M) > 0.0).sum())
                 for sd in ("left", "right")})
    return lifted(), s, lift, info


def paroxysmal_kinesigenic(kind="dystonic", seed=0, standing_s=DUR_S):
    """(Body, truth) of paroxysmal kinesigenic dyskinesia: the arising from a chair (motions.arise, the trigger: a
    sudden movement), the folded arms unfolding to the sides once the trunk is back within PKD['unfold_trunk_deg']
    of the vertical after lift-off (the stand's hanging arms and open hands), each arm on its own clock, one waiting by
    the smallest lag that keeps the skins apart (motions.unfold_to_stand: on one clock the right hand passed through
    the left forearm), an attack of kind ('dystonic' or 'choreic', attack_terms) beginning PKD['latency_s'] after
    lift-off or as the arms open, whichever is later, and lasting PKD['duration_s'] (latorre_bhatia: a few seconds up
    to 1 min; module values), the arms joining it as they unfold, the legs untouched (the feet stay planted). Where an
    attacking arm's forearm or hand would come nearer the trunk's skin than TRUNK_CLEAR_M (or than the unfolded base's
    own distance), the trunk's and the humerus's turns are held back toward the base (guard_rows on the spine and
    shoulder rows, PKD_TRUNK_GUARD_ROWS, _clear_need; wave SH1c: the dystonic forearm lay inside the abdomen over the
    whole attack): the forearm comes to rest against the abdomen; where it would come as near a thigh's skin, the
    shoulder, elbow and wrist are held back (review of wave SH1c: a choreic hand inside the thigh), and where it would
    come nearer the other arm than motions.FOLD_OPEN_CLEAR_M over the opening and UNFOLD_CHECK_PAD_S after it, too
    (_arms_need, PKD_ARMS_CHECKS; open-body port, wave 6 follow-up). The truth's validity adds the hand-to-thigh and hand-to-trunk checks
    (thigh_report, trunk_report, from lift-off) and the armrest check (armrest_report, the scene's arise chair), all
    against the level 0's own (the folded arms of the arising, motions.arise, entered the right thigh and armrest at the
    forward lean in the level 0 until SH9 refolded them clear), so that the operator adds nothing there, and the opening
    read on the final body at every knot (motions.unfold_check, the validity's 'fold')."""
    from .motions import arise, LIFTOFF_SHARE, UNFOLD_CHECK_PAD_S, unfold_to_stand
    b0, ainfo = arise(standing_s=standing_s)
    rng = as_rng(seed)
    T, rate = b0.T, b0.rate
    t = np.arange(T) / rate
    t_lift = ainfo["onset_s"] + LIFTOFF_SHARE * ainfo["rise_s"]
    flex = trunk_flexion_deg(b0)
    upright = np.flatnonzero((t >= t_lift) & (flex <= PKD["unfold_trunk_deg"]))
    if not len(upright):
        raise ValueError(f"the arising never brings the trunk within {PKD['unfold_trunk_deg']} deg of the vertical")
    t_unf = float(t[upright[0]])
    k_unf, n_unf = int(upright[0]), int(round(PKD["unfold_dur_s"] * rate))
    Ru, Rh, w_side, unfold_rec = unfold_to_stand(b0, k_unf, n_unf)
    wu = np.maximum(w_side["left"], w_side["right"])      # the attack joins the arms as they unfold
    arm_rows = list(range(13, 22))
    b = b0.with_rotations(Ru, arm_rows).with_hands(Rh)
    t0 = max(t_lift + PKD["latency_s"], t_unf)            # once up: no attack on the folded, forward-leaning arms
    env = ramp_env(T, rate, t0, PKD["on_s"], t0 + PKD["duration_s"], PKD["off_s"])
    k = int(round((t_unf + PKD["unfold_dur_s"]) * rate))           # the rooms of the unfolded arms
    terms = attack_terms(b, env, kind, rng, k=k, arm_env=env * wu)
    table = DYST if kind == "dystonic" else HEALTHY       # chorea is not a dystonia operator
    R, s = fit_groups(b.local_R(), attack_groups(terms), table=table, smooth=int(round(FIT_SMOOTH_S * rate)))
    guard, thigh_guard = {}, {}
    for sd in (("left",) if kind == "dystonic" else ("left", "right")):   # the arm held back where its forearm
        R, _, guard[sd] = guard_rows(b, R, PKD_TRUNK_GUARD_ROWS[sd], _clear_need(b, sd, "trunk", TRUNK_CLEAR_M))
        R, _, thigh_guard[sd] = guard_rows(b, R, ARM[sd][1:], _clear_need(b, sd, "thighs", TRUNK_CLEAR_M),   # would
                                           step=GUARD_FAST_STEP)                 # enter the trunk, or the arm a thigh
    k_end = k_unf + n_unf + int(round(UNFOLD_CHECK_PAD_S * rate))
    arms_guard = {}
    for sd in (("left",) if kind == "dystonic" else ("left", "right")):   # the attacking arm held back where it would
        R, _, arms_guard[sd] = guard_rows(b, R, ARM[sd][1:], _arms_need(k_unf, k_end),   # meet the other one as the
                                          step=1)       # arms open, read at every knot (wave 6 last pass: read at
        #                                                 every second, the hands met between two reads)
    rows = sorted({j for t_ in terms for j in t_.chain} | set(arm_rows))
    out = b.with_rotations(R, rows)
    out = replace(out, contact=contact_labels(out))
    v = LX.check(table, b0, out)
    layer_checks = scene_checks_record("pkd", standing_s=float(standing_s), lift_knot=int(round(t_lift * rate)),
                                       unfold_knot=int(k_unf), unfold_knots=int(n_unf))
    v.update(scene_checks_of(out, layer_checks, base=b0))
    dev = {t_.label: float(joint_deviation(b, out, t_.chain[-1], segment=len(t_.chain) > 1).max()) for t_ in terms}
    active = np.flatnonzero(env > 0.5)
    params = {"kind": kind, "trigger": "arising from the chair (lift-off)", "liftoff_s": t_lift, "attack_on_s": t0,
              "arms_unfold_s": [t_unf, t_unf + PKD["unfold_dur_s"]],
              "arms_unfold_lag": {k_: unfold_rec[k_] for k_ in ("lag", "lag_side", "tried")},
              "trunk_flexion_at_liftoff_deg": float(np.interp(t_lift, t, flex)),
              "trunk_flexion_at_unfold_deg": float(flex[upright[0]]),
              "attack_s": PKD["duration_s"], "pkd": PKD, "terms": [t_.record() for t_ in terms],
              "limit_scale_min": float(s.min()), "realized_deviation_deg": dev,
              "realized_attack_s": float((active[-1] - active[0] + 1) / rate) if len(active) else 0.0,
              "pre_attack_deviation_deg": float(joint_deviation(b0, out, 20)[:int(t_lift * rate)].max()),
              "fit": {"groups": [[t_.label for t_ in g] for g in attack_groups(terms)], "smooth_s": FIT_SMOOTH_S},
              "trunk_guard": guard, "thigh_guard": thigh_guard, "arms_guard": arms_guard, "trunk_clear_m": TRUNK_CLEAR_M,
              "limits_table": LX.record(table), "layer_checks": layer_checks}
    return out, truth_record("paroxysmal_kinesigenic", seed if isinstance(seed, int) else None, kind, "episode",
                             params, ("latorre_bhatia", "albanese2013") + (("limits_ext",) if table == DYST else ())
                             + ("module",), v,
                             arrays={"envelope": env, "arms_unfold": wu})


PNKD_ATTACK_TEXT = ("attacks last 10 min to 1 h (Bruno 2007), most often 15 to 40 min (Erro 2014, via Latorre and "
                    "Bhatia), after coffee, alcohol, stress or fatigue")
PED_ATTACK_TEXT = ("attacks of 5 to 30 min after prolonged exertion, subsiding with rest (Lance 1977, via Latorre and "
                   "Bhatia)")


def paroxysmal_excerpt(kind, seed=0, duration_s=DUR_S):
    """(Body, truth) of an excerpt of a longer attack, the attack under way from the first knot: 'pnkd' seated, a
    dystonic attack of the left arm, trunk and neck with choreic movements of both arms, the neck and the trunk
    (PNKD_ATTACK_TEXT), the arms kept out of the chair's backrest (guard_rows on the spine and each arm's collar and
    shoulder rows, _backrest_need: where an arm would come nearer the backrest than the trunk's own clearance, the
    trunk's turn and that arm's swing are held back toward the base; wave SH1c: the right upper arm entered it;
    backrest_report in the validity), the forearms out of its armrests (clear_armrests on the left: where the trunk's
    lean would sink that forearm into one, the left collar is raised with the arm hanging under it, the forearm resting
    on the armrest holding the shoulder up as the trunk leans, the arm itself held back past the collar's room; the
    right arm swings freely over its armrest and is held back alone, guard_rows and _armrest_need, review of wave SH1c:
    a raised right collar read as repeated phasic shrugs the clip does not declare; armrest_report in the validity) and
    out of the trunk (guard_rows on each
    arm's shoulder and elbow rows, _clear_need: the humerus's inward turn and the elbow's flexion held back where the
    forearm or hand would come nearer the trunk's skin than TRUNK_CLEAR_M, or than the seated base's own distance,
    and the thighs' than PNKD_THIGH_CLEAR_M, plan phase E; trunk_report and thigh_report in the validity), the two in
    turn for at most GUARD_ROUNDS rounds (review of wave
    SH1c: an armrest guard on the arm's rows held the whole left arm at its base, since the entry came from the trunk's
    lean, and the raised collar then brought the left forearm into the trunk); 'ped' standing, the left foot in
    equinovarus touching the floor with the leg unloaded after prolonged exertion (PED_ATTACK_TEXT).
    """
    from .motions import posture_body
    rng = as_rng(seed)
    if kind == "pnkd":
        b = posture_body("sit", duration_s)
        env = np.clip(1.0 - FLUCT + FLUCT * smooth_noise(b.T, b.rate, rng), 0.0, 1.0)
        terms = attack_terms(b, env, "dystonic", rng) + attack_terms(b, env, "choreic", rng)
        R1, s = fit_groups(b.local_R(), attack_groups(terms), table=DYST, smooth=int(round(FIT_SMOOTH_S * b.rate)))
        R_fit = R1
        guard, trunk_guard, clear = {}, [], []
        R1, _, guard["trunk"] = guard_rows(b, R1, BACKREST_ROWS, _backrest_need(b, "trunk", "chair"),   # the back
                                           step=1)       # itself, read at every knot (D58)
        for sd in ("right", "left"):                     # the trunk's turn and the arm held back where the arm would
            R1, _, guard[sd] = guard_rows(b, R1, BACKREST_ROWS + ARM[sd][:2],   # enter the backrest (wave SH1c)
                                          _backrest_need(b, sd, "chair"))
        R = R1
        lift = {"left": np.zeros(b.T), "right": np.zeros(b.T)}
        for r_ in range(GUARD_ROUNDS):                   # the forearms on the armrests (the collar raised), then the
            R, s, add, cinfo = clear_armrests(b, terms, "chair", DYST, fitted=(R, s),   # arm held back where its
                                              sides=("left",))
            clear.append(cinfo)                          # forearm would enter the trunk (the raised collar brought the
            for sd in lift:                              # left forearm into it: review of wave SH1c), in turn;
                lift[sd] = lift[sd] + add[sd]            # from the second round the guard reads every knot, and the
            rnd, step = {}, (GUARD_STEP if r_ == 0 else 1)   # rounds end once it moves nothing there (both clear)
            for sd in ("right", "left"):                 # past the collar's room, the arm itself held back
                if cinfo[sd + "_still_entering"]:
                    R, _, rnd[sd + "_armrest"] = guard_rows(b, R, ARM[sd][1:], _armrest_need(b, sd, "chair"), step=1)
            for sd in ("right", "left"):
                R, _, rnd[sd] = guard_rows(b, R, ARM[sd][1:3], _clear_need(b, sd, "trunk", TRUNK_CLEAR_M, skin=False),
                                           step=step)
            for sd in ("right", "left"):                 # plan phase E: and off the thighs (PNKD_THIGH_CLEAR_M)
                R, _, rnd[sd + "_thigh"] = guard_rows(b, R, ARM[sd][1:],
                                                      _clear_need(b, sd, "thighs", PNKD_THIGH_CLEAR_M, skin=False),
                                                      step=step)
            trunk_guard.append(dict(rnd, step=step))
            if step == 1 and not any(rnd[k]["knots"] for sd in ("right", "left") for k in (sd, sd + "_thigh")):
                break
        out = b.with_rotations(R, sorted({j for t in terms for j in t.chain} | {13, 14, 16, 17}))
        out = replace(out, contact=contact_labels(out))
        val = LX.check(DYST, b, out)
        layer_checks = scene_checks_record("pnkd", duration_s=float(duration_s))
        val.update(scene_checks_of(out, layer_checks))
        dev = {t.label: float(joint_deviation(b, out, t.chain[-1], segment=len(t.chain) > 1).max()) for t in terms}
        B0 = b.local_R()

        def row_median(Rx):                              # each arm row's median angle from the seated base, deg
            return {str(j): float(np.median(rotation_angle_deg(B0[:, j].transpose(-1, -2) @ Rx[:, j]).numpy()))
                    for sd in ("left", "right") for j in ARM[sd][1:]}
        params = {"kind": kind, "excerpt_of_attack": True, "terms": [t.record() for t in terms],
                  "arm_rows_median_deg": {"fit": row_median(R_fit), "clip": row_median(R)},
                  "limit_scale_min": float(s.min()), "armrest_clearance": clear, "backrest_guard": guard,
                  "trunk_guard": trunk_guard, "trunk_clear_m": TRUNK_CLEAR_M, "realized_deviation_deg": dev,
                  "fit": {"groups": [[t.label for t in g] for g in attack_groups(terms)], "smooth_s": FIT_SMOOTH_S},
                  "typical_attack_min": (10, 60), "most_often_min": (15, 40), "attack_text": PNKD_ATTACK_TEXT,
                  "limits_table": LX.record(DYST), "layer_checks": layer_checks}
        return out, truth_record("paroxysmal_nonkinesigenic", seed if isinstance(seed, int) else None, "excerpt",
                                 "episode", params, ("latorre_bhatia", "bruno2007", "limits_ext", "module"), val,
                                 arrays={"envelope": env, "armrest_lift_left_deg": lift["left"],
                                         "armrest_lift_right_deg": lift["right"]})
    if kind == "ped":
        b = posture_body("stand", duration_s)
        env = np.clip(1.0 - FLUCT + FLUCT * smooth_noise(b.T, b.rate, rng), 0.0, 1.0)
        out, tr = limb_posture(b, [], env, foot=("left", FOOT_STAND), label="paroxysmal_exercise_induced",
                               grade="excerpt", scale="episode", sources=("latorre_bhatia", "dis", "module"))
        tr["params"].update(kind=kind, excerpt_of_attack=True, typical_attack_min=(5, 30), foot_measured=
                            measure_foot(b, out, "left"), attack_text=PED_ATTACK_TEXT)
        return out, tr
    raise ValueError(kind)


FOOT_LEVELS = {2: {"plantarflexion": DIS_SHARE[2], "inversion": DIS_SHARE[2], "adduction": DIS_SHARE[2],
                   "toe_flexion": DIS_SHARE[2]},
               3: {"plantarflexion": DIS_SHARE[3], "inversion": DIS_SHARE[3], "adduction": DIS_SHARE[3],
                   "toe_flexion": DIS_SHARE[3]}}
#   DIS grades 2 and 3 of the foot (equinovarus) as shares of each component's room in the SH2 table (D3; limits_ext:
#   inversion 30 deg, toe flexion 35 deg), the toes curled at the grade's share like the other components (before
#   SH2 they curled to the healthy table's 2.3 deg at its whole room, invisibly)
FOOT_STAND = {"plantarflexion": 0.3, "inversion": DIS_SHARE[3], "adduction": DIS_SHARE[3]}
#   [module] the standing equinovarus (hemidystonia, multifocal, PED): the plantarflexion lowered to 0.3 of its room,
#   the inversion and adduction at DIS grade 3 shares, so that the foot rests on its lateral border with the heel
#   raised and the leg nearly straight (review M1); no toe flexion: with the SH2 table's inversion the knee lift is
#   12.7 deg, and curled toes, even at 0.25 of their 35 deg, take it to 16.1 deg (20.9 at the DIS grade 3 share),
#   past the review's cap of LIFT_CAP_DEG + 6 (foot_dystonia(3, standing=True), 3 s, seed 2, as in
#   tests/test_synth_postures.py; before SH2 the toes curled to the healthy table's 2.3 deg, invisibly)
FIXED_FOOT = {"inversion": 0.8, "plantarflexion": 0.5, "adduction": 0.5, "toe_flexion": 0.8}
#   [module] the fixed (functional) foot (F02 of the symptom catalogue: plantar-flexed and inverted with curled toes):
#   inversion and toe flexion at 0.8 of their rooms, plantarflexion and adduction at 0.5
OFF_FOOT = {"plantarflexion": DIS_SHARE[2], "inversion": DIS_SHARE[3], "adduction": DIS_SHARE[2],
            "toe_extension": DIS_SHARE[3]}
#   [module] the off-period foot: inverted, the toes extended (the striatal toe; the SMPL+H foot joint moves all toes)


def foot_dystonia(grade=None, standing=False, levels=None, fixed=False, seed=0, duration_s=DUR_S, side="left"):
    """(Body, truth) of foot dystonia: the equinovarus foot of FOOT_LEVELS[grade] (or levels) on a seated or standing
    base, touching the floor (touch_floor); fixed: present from the first knot and constant (the fixed, functional
    form); otherwise rising from 0.5 s and held with the fluctuation of held_env."""
    from .motions import posture_body
    b = posture_body("stand" if standing else "sit", duration_s)
    rng = as_rng(seed)
    env = held_env(b.T, b.rate, rng, fixed=fixed)
    lv = levels if levels is not None else FOOT_LEVELS[grade]
    out, tr = limb_posture(b, [], env, foot=(side, lv), label="foot_dystonia", grade=grade, scale="DIS distal leg",
                           sources=("dis", "albanese2013", "module"))
    tr["params"].update(foot_measured=measure_foot(b, out, side), fixed=fixed, standing=standing)
    return out, tr


# ---------------------------------------------------------------- distribution of dystonia
FIXED_FIST_SHARE = {"index": 0.4, "middle": 0.7, "ring": 1.0, "pinky": 1.0}
FIXED_FIST_THUMB = ((40.0, 0.0), 35.0, 20.0)
#   [module] the fixed (functional) clenched hand: the fingers at these shares of handtasks.FIST, the ring and little
#   fingers flexed most (Schrag et al. 2004 via the catalogue's F02; review M12: the uniform fist flexed the index as
#   much as the little finger), the thumb flexed and adducted across the palm (CMC flexion, no palmar abduction).
#   The fixed fist's lift (plan phase E): the clenched fist and the flexed wrist of the 'sit' hang their fingers into the open
#   body's thigh (the index tip 22.7 mm inside on the mesh, 23.5 on the page's rigid thigh); distribution lifts the left
#   arm by the least shoulder flexion on dystonia.ARM_LIFT_STEP_DEG's grid that keeps the hand and forearm reach.CLEAR_M
#   off the thighs on the mesh (dystonia.arm_lift, read on the first build, then a term of the build itself, so that the
#   table check reads it); params 'arm_lift'
SEGMENTAL_ARM = {"elbow_deg": 50.0, "wrist": DIS_SHARE[2], "pronation_deg": 30.0, "fist": 0.5}
#   [module] the arm of the segmental dystonia: elbow flexion (40 to 60 deg), wrist flexion at the DIS grade 2 share,
#   forearm pronation, the fingers half way to the fist, no added abduction (review M4: a straight arm abducted read
#   as a wing)


def fixed_fist_pose():
    """{finger: angles} of the fixed clenched hand (FIXED_FIST_SHARE of handtasks.FIST, FIXED_FIST_THUMB)."""
    F = fist_pose()
    out = {f: tuple(FIXED_FIST_SHARE[f] * x for x in F[f]) for f in FIXED_FIST_SHARE}
    out["thumb"] = FIXED_FIST_THUMB
    return out


def distribution(kind, seed=0, duration_s=DUR_S):
    """(Body, truth) of a distribution of dystonia (Albanese 2013), standing, held from the first knot with the
    fluctuation of held_env (a persistent posture: review M10): 'hemidystonia' the left arm (elbow and wrist flexion
    at DIS grade 3 shares, the fist closing to 0.8 of handtasks.FIST) and the left foot (FOOT_STAND, touching);
    'segmental' contiguous regions, the neck (TWSTRS rotation to the right, grade 2), the left shoulder (elevation at
    half its range, TWSTRS D 2) and the left arm (SEGMENTAL_ARM); 'multifocal' non-contiguous regions, the neck
    (laterocollis to the right, TWSTRS grade 2) and the left foot (FOOT_STAND); 'fixed_fist' seated, the left hand
    clenched (fixed_fist_pose) and the wrist flexed (0.6 of its room) from the first knot without any change (fixed,
    functional dystonia); 'fixed_foot' seated, the left foot inverted with the toes curled (FIXED_FOOT, the SH2
    table's rooms) from the first knot, constant."""
    from .motions import posture_body
    rng = as_rng(seed)
    seated = kind.startswith("fixed")
    b = posture_body("sit" if seated else "stand", duration_s)
    env = held_env(b.T, b.rate, rng, on_s=0.0, ramp_s=0.0, fixed=seated)

    def share_of(region, direction, side, deg):
        return deg / region_room(b, region, direction, side)
    if kind == "hemidystonia":
        args = dict(parts=[("distal_arm", "elbow_flexion", "left", DIS_SHARE[3]), ("wrist", "flexion", "left",
                                                                                     DIS_SHARE[3])],
                    foot=("left", FOOT_STAND), hand=("left", fist_pose(), 0.8))
    elif kind == "segmental":
        rot = G.twstrs_level_deg("rotation", 2, None)
        sh = G.twstrs_level_deg("shoulder_elevation", 2, region_room(b, "shoulder_elevation", "elevation", "left"))
        a = SEGMENTAL_ARM
        args = dict(parts=[("neck", "rotation_right", "left", share_of("neck", "rotation_right", "left", rot)),
                           ("wrist", "flexion", "left", a["wrist"])],
                    pre_terms=[region_term(b, "shoulder_elevation", "elevation", "left", sh, env,
                                           "shoulder_elevation elevation left")],
                    hang=("left",), hang_collar=False,
                    extra_terms=[region_term(b, "distal_arm", "elbow_flexion", "left", a["elbow_deg"], env,
                                             "distal_arm elbow_flexion left"),
                                 arm_internal_term("left", a["pronation_deg"], env, joint=18)],
                    hand=("left", fist_pose(), a["fist"]))
    elif kind == "multifocal":
        lat = G.twstrs_level_deg("laterocollis", 2, None)
        args = dict(parts=[("neck", "tilt_right", "left", share_of("neck", "tilt_right", "left", lat))],
                    foot=("left", FOOT_STAND))
    elif kind == "fixed_fist":
        args = dict(parts=[("wrist", "flexion", "left", 0.6)], hand=("left", fixed_fist_pose(), 1.0))
    elif kind == "fixed_foot":
        args = dict(parts=[], foot=("left", FIXED_FOOT))
    else:
        raise ValueError(kind)
    out, tr = limb_posture(b, env=env, label=kind if not seated else "fixed_dystonia", grade=kind,
                           scale="distribution", sources=("albanese2013", "dis", "twstrs", "module"), **args)
    lift = None
    if kind == "fixed_fist":             # plan phase E: the fixed fist's lift (the comment of FIXED_FIST_SHARE)
        from .dystonia import arm_lift
        _, lift = arm_lift(out, env, sides=("left",))
        deg = lift["sides"]["left"]["deg"]
        if deg:
            args["extra_terms"] = [joint_term(16, "x", -deg, env, label="shoulder flexion left (the fist off the "
                                                                          "thigh)")]
            out, tr = limb_posture(b, env=env, label="fixed_dystonia", grade=kind, scale="distribution",
                                   sources=("albanese2013", "dis", "twstrs", "module"), **args)
            tr["params"]["arm_lift"] = lift
    if args.get("foot"):
        tr["params"]["foot_measured"] = measure_foot(b, out, "left")
    tr["params"]["fixed"] = seated
    tr["params"]["env_range"] = [float(env.min()), float(env.max())]
    return out, tr


# ---------------------------------------------------------------- the clip set
def _l0(seg):
    return {"E7": "dystonia_E7_level0", "E1": "dystonia_E1_level0", "E2": "tremor_E2_raise_g0",
            "arise": "ref_arise_E7"}[seg]


WRITING_SEGMENT = ("W", "writing at a table (seated)")   # the table is scene geometry of this module (scene_mesh)
NOT_RENDERED = {
    "camptocormia, thoracic fulcrum": ("the upper method's angle reaches 44 deg at most inside the table (spine2 and "
                                       "spine3 rooms), below the task force's 45 deg cut-off"),
    "foot inversion beyond 30 deg": ("the SH2 table's ankle inversion is 30 deg (limits_ext, jarm2022, frontal "
                                     "plane): the equinovarus clips invert to DIS shares of that room"),
    "toe curling of each toe apart": "one SMPL+H toe joint moves all toes at once (35 deg, limits_ext, jarm2022)",
    "curled toes on the standing equinovarus foot": ("the curled toes lower the touching foot, and the leg's lift "
                                                     "would pass the review's cap (FOOT_STAND): shown seated only"),
    "complete PNKD and PED attacks": "minutes to hours: shown as 15 s excerpts, their durations in the caption",
    "chorea of the legs in the PNKD excerpt": "the seated feet stay planted (a moving foot on the floor would skate)",
    "BFM provoking factors 1 to 3 of the generalized dystonia": "the torsion clips are at rest (factor 4) only",
    "WCRS tremor and writing speed, the pen and the paper": ("no pen is drawn (the viewer's scene is static, a pen "
                                                            "must follow the hand); the A4 sheet is scene geometry in "
                                                            "the table's colour; no handwriting trace"),
    "TWSTRS duration score 3 on the mobile course": "the exam set's static intermittent rotation covers score 3",
    "TWSTRS D 3 (severe and constant shoulder elevation)": "the exam set's static shoulder elevation grade 3",
    "hemidystonia of the face": "no face in the body",
    "the geste antagoniste on the side of the turn": ("the cheek toward which the chin turns lies so near the "
                                                      "shoulder below it that no hand frame on it keeps the wrist "
                                                      "inside its range with the elbow inside the table "
                                                      "(geste_antagoniste, side 'left'; geste_frames): the hand of the "
                                                      "other side touches its own cheek"),
}


def _txt(x, nd=1):
    t = f"{x:.{nd}f}"
    return t.rstrip("0").rstrip(".") if "." in t else t


def specs():
    """[exam.ClipSpec] of the posture catalogue (module docstring), since 'catalogue_v1'; each build() returns
    (Body, truth)."""
    from .exam import ClipSpec
    out = []

    def stand():
        return static_body("stand", DUR_S)

    def add(cid, family, seg, item, operator, grade, grade_text, params_text, views, level0, build, scene=None,
            note="", title=""):
        out.append(ClipSpec(cid, family, seg, item, operator, grade, grade_text, params_text, tuple(views), level0,
                            scene, build, note=note, since=CAT, segment_title=title))
    # parkinsonian posture, MDS-UPDRS 3.13
    for g, p in PD_313.items():
        corr = "corrected to normal" if p["correct"] >= 1 else f"only {_txt(100 * p['correct'], 0)} percent undone"
        add(f"post_E7_pd_posture_g{g}", "parkinsonian_posture", "E7", "MDS-UPDRS 3.13", "parkinsonian posture, stooped",
            g, f"MDS-UPDRS 3.13 grade {g}",
            f"total camptocormia angle {_txt(p['tcc'])} deg (malleolus method), knees {_txt(p['knee'])} and elbows "
            f"{_txt(p['elbow'])} deg flexed, the arms hanging, turned {_txt(p['internal'])} deg inward [module]; asked "
            f"to stand straight at {_txt(ASK_S)} s: {corr}",
            ("left", "front"), _l0("E7"), lambda g=g: parkinsonian_posture(stand(), g))
    tf = [("lumbar", "mild", 0.5 * sum(TF_BANDS["lumbar"]["mild"]), 10.0, "anterior trunk flexion, lumbar fulcrum"),
          ("lumbar", "open", OPEN_BAND_FACTOR * TF_BANDS["lumbar"]["cut"], 15.0, "camptocormia, lumbar fulcrum"),
          ("thoracic", "mild", 0.5 * sum(TF_BANDS["thoracic"]["mild"]), 5.0,
           "anterior trunk flexion, thoracic fulcrum"),
          ("lateral", "mild", 0.5 * sum(TF_BANDS["lateral"]["mild"]), 5.0, "lateral trunk flexion, to the left"),
          ("lateral", "open", OPEN_BAND_FACTOR * TF_BANDS["lateral"]["cut"], 5.0, "Pisa syndrome, to the left"),
          ("neck", "mild", 0.5 * sum(TF_BANDS["neck"]["mild"]), 0.0, "anterior neck flexion"),
          ("neck", "open", NECK_OPEN_DEG, 0.0, "antecollis")]
    method = {"lumbar": "malleolus method", "thoracic": "upper method", "lateral": "perpendicular method, coronal",
              "neck": "perpendicular method, C7 to tragus"}
    for shape, band, deg, knee, name in tf:
        lo_hi = TF_BANDS[shape]["mild"]
        btxt = (f"centre of the {_txt(lo_hi[0])} to {_txt(lo_hi[1])} deg band" if band == "mild" else
                f"past the {_txt(TF_BANDS[shape]['cut'])} deg cut-off, inside the cervical flexion range [module]"
                if shape == "neck" else
                f"{_txt(OPEN_BAND_FACTOR)} x the {_txt(TF_BANDS[shape]['cut'])} deg cut-off [module]")
        views = {"lumbar": ("left", "front"), "thoracic": ("left", "front"), "lateral": ("back", "front"),
                 "neck": ("left", "head")}[shape]
        add(f"post_E7_{shape}_{band}", "parkinsonian_posture", "E7", "task force axial posture", name, band,
            f"{name} (MDS task force 2022)",
            f"{_txt(deg)} deg, {method[shape]} ({btxt}), held from the first frame; knees {_txt(knee)} deg [module]; "
            + ("the arms hanging; " if shape in ("lumbar", "thoracic") else "")
            + ("a subtle change from the level 0, as the band is narrow; " if (shape, band) == ("neck", "mild") else "")
            + "feet planted, centre of mass kept over the feet", views, _l0("E7"),
            lambda shape=shape, deg=deg, knee=knee: axial_posture(stand(), shape, deg, knee, grade=band))
    # cervical dystonia, time course
    rot2 = G.twstrs_level_deg("rotation", 2, None)
    for dg in (1, 2, 4, 5):
        share, exc = G.twstrs_duration(dg, "maximal")
        add(f"post_E7_cd_rotation_g2_duration{dg}", "cervical_dystonia", "E7", "TWSTRS A1 and B",
            "cervical dystonia, rotation to the left, mobile", f"duration {dg}",
            f"TWSTRS rotation grade 2 and duration score {dg}",
            f"maximal excursion {_txt(rot2)} deg; present {_txt(100 * share)} percent of the time, mostly {exc}; "
            + (f"bouts drifting in over up to {_txt(DRIFT_S[1])} s and corrected back in {_txt(CORRECT_BACK_S[0])} to "
               f"{_txt(CORRECT_BACK_S[1])} s; " if share < 0.5 else "")
            + f"held at {_txt(100 * HOLD[exc])} percent with slow fluctuation and mobile spasms [module]",
            ("top", "front", "head"), _l0("E7"),
            lambda dg=dg: cervical_course(stand(), [("neck", "rotation_left", "left", rot2)], duration_grade=dg,
                                          seed=SEED + dg))
    lat2 = G.twstrs_level_deg("laterocollis", 2, None)
    jerky = [("rotation_left", "rotation", rot2, 5, ("top", "head")), ("tilt_left", "laterocollis", lat2, 4,
                                                                        ("front", "head"))]
    for d, kind, lv, dg, views in jerky:
        add(f"post_E7_cd_{d}_jerky", "cervical_dystonia", "E7", "TWSTRS A and B",
            f"cervical dystonia, {kind}, with dystonic jerks", 2, f"TWSTRS {kind} grade 2, duration {dg}",
            f"{_txt(lv)} deg; held at {_txt(100 * JERK_HOLD, 0)} percent with irregular sawtooth pulls toward the "
            f"posture (0.1 to 0.5 Hz, Shaikh 2015): {_txt(JERK_RATE_HZ, 2)} per s on average, "
            f"{_txt(JERK_GAP_S[0])} to {_txt(JERK_GAP_S[1])} s apart, a {_txt(1000 * JERK_PULL_S[0], 0)} to "
            f"{_txt(1000 * JERK_PULL_S[1], 0)} ms pull and a {_txt(JERK_RETURN_S[0])} to {_txt(JERK_RETURN_S[1])} s "
            f"return [module timing]", views, _l0("E7"),
            lambda d=d, lv=lv, dg=dg: cervical_course(stand(), [("neck", d, "left", lv)], duration_grade=dg,
                                                      jerks=True, spasms=False, seed=SEED + 11))
    add("post_E7_cd_retrocollis_mobile", "cervical_dystonia", "E7", "TWSTRS A3b and B",
        "cervical dystonia, retrocollis, mobile", 2, "TWSTRS retrocollis grade 2, duration 4",
        "half the possible range (the scale's grade 2); slow fluctuation and mobile spasms [module]", ("left", "head"),
        _l0("E7"), lambda: (lambda b: cervical_course(b, [("neck", "retrocollis", "left", G.twstrs_level_deg(
            "retrocollis", 2, region_room(b, "neck", "retrocollis", "left")))], duration_grade=4, seed=SEED + 12))(
            stand()))
    for kind, views in (("lateral", ("front", "head")), ("anterior", ("left", "head")),
                        ("posterior", ("left", "head"))):
        item = "TWSTRS A4 lateral shift" if kind == "lateral" else "TWSTRS A5 sagittal shift"
        add(f"post_E7_cd_shift_{kind}", "cervical_dystonia", "E7", item,
            f"cervical dystonia, {kind} shift" + (" to the left" if kind == "lateral" else ""), 1, f"{item}: present",
            f"upper thoracic spine {_txt(SHIFT_DEG[kind][0])} and neck {_txt(SHIFT_DEG[kind][1])} deg "
            + {"lateral": "tilted to the left, the head tilted", "anterior": "flexed forward, the head extended",
               "posterior": "extended backward, the head flexed"}[kind]
            + f" {_txt(sum(SHIFT_DEG[kind]))} deg the other way: the head moved "
            + {"lateral": "to the left", "anterior": "forward", "posterior": "backward"}[kind]
            + " without tilting [module]", views, _l0("E7"),
            lambda kind=kind: head_shift(stand(), kind, seed=SEED + 13))
    for name, third, share, dgrade in (("D1_mild_intermittent", 1, 0.375, 1), ("D2_moderate_constant", 2, 1.0, 2),
                                       ("D2_severe_intermittent", 3, 0.375, 2)):
        add(f"post_E7_cd_shoulder_{name}", "cervical_dystonia", "E7", "TWSTRS D",
            "cervical dystonia, left shoulder elevation", dgrade, f"TWSTRS D grade {dgrade}",
            f"{('mild, the first', 'moderate, the middle', 'severe, the last')[third - 1]} third of the possible "
            f"range, the arm hanging under the raised shoulder; present {_txt(100 * share)} percent of the time"
            + ("; bouts drifting in, corrected back quickly" if share < 0.5 else ""), ("front", "head"), _l0("E7"),
            lambda third=third, share=share: (lambda b: cervical_course(
                b, [("shoulder_elevation", "elevation", "left", G.twstrs_level_deg("shoulder_elevation", third,
                                                                                   region_room(b, "shoulder_elevation",
                                                                                               "elevation", "left")))],
                share=share, seed=SEED + 14 + third))(stand()))
    lat1 = G.twstrs_level_deg("laterocollis", 1, None)
    add("post_E7_cd_torticollis_complex", "cervical_dystonia", "E7", "TWSTRS A and D",
        "cervical dystonia, rotation and tilt to the left with left shoulder elevation", "complex",
        "TWSTRS rotation 2, laterocollis 1, shoulder elevation 1, duration 4",
        f"rotation {_txt(rot2)} and tilt {_txt(lat1)} deg, the shoulder at half its range with the arm hanging under "
        f"it, rising and falling with the neck's course (moderate but not constant: TWSTRS D 1); mobile course "
        f"[module]",
        ("front", "top", "head"), _l0("E7"),
        lambda: (lambda b: cervical_course(b, [("neck", "rotation_left", "left", rot2),
                                               ("neck", "tilt_left", "left", lat1),
                                               ("shoulder_elevation", "elevation", "left", G.twstrs_level_deg(
                                                   "shoulder_elevation", 2, region_room(b, "shoulder_elevation",
                                                                                        "elevation", "left")))],
                                           duration_grade=4, seed=SEED + 18))(stand()))
    for sc in (0, 1, 2):        # the three TWSTRS C scores on the rotation of grade 2
        add(f"post_E7_cd_geste_C{sc}", "cervical_dystonia", "E7", "TWSTRS C", "cervical dystonia with the geste "
            "antagoniste (right fingertips on the right cheek)", sc, f"TWSTRS C score {sc}",
            f"rotation {_txt(rot2)} deg (grade 2); the right fingertips on the right cheek "
            f"{_txt(TRICK_TIMES['reach_on_s'] + TRICK_TIMES['reach_s'])} to {_txt(TRICK_TIMES['release_s'])} s "
            f"undo {_txt(100 * TRICK[sc], 0)} percent"
            + ("" if sc == 0 else " [module]"), ("front", "top", "head"), _l0("E7"),
            lambda sc=sc: geste_antagoniste(stand(), "rotation_left", 2, sc, seed=SEED + 20))
    # limb dystonia
    W, WT = WRITING_SEGMENT
    write_txt = (f"tripod pen grip, forearm semi-pronated, the hand resting on the paper by its curled little finger, "
                 f"the forearm free above the table; strokes at about {_txt(STROKE['hz'])} Hz, the hand moving "
                 f"along the line at {_txt(100 * LINE['speed_m_s'])} cm/s with pauses between words and line "
                 f"returns, from {_txt(WRITE['start_s'])} s, the pen on the paper within words and lifted "
                 f"{_txt(1000 * PEN['lift_m'])} mm between them; the other hand on the table [module]")
    add("post_W_writing_level0", "limb_dystonia", W, "level 0", "writing, right hand (no dystonia)", 0, "level 0",
        write_txt, ("front", "right", "hands"), None, lambda: writers_cramp(0), scene="chair_table", title=WT)
    for g, it in WCRS_ITEMS.items():
        add(f"post_W_writers_cramp_g{g}", "limb_dystonia", W, "WCRS (module item scores)", "writer's cramp, right hand",
            g, f"writer's cramp, {('mild', 'marked')[g - 1]} (WCRS writing movement {wcrs_total(it)} of 28)",
            f"WCRS items fingers {it['fingers']}, wrist {it['wrist']}, elbow {it['elbow']}, latency factor "
            f"{it['latency_factor']}, tremor {it['tremor']} [module reading]; after "
            f"{_txt(WCRS_LATENCY_S[it['latency_factor']])} s of writing: the pen grip tightens (thumb and index), the "
            f"ring and little fingers curl, the wrist flexes with ulnar deviation and pronation"
            + (", the elbow lifts and turns outward" if it["elbow"] else "") + "; the strokes shrink [module]",
            ("front", "right", "hands"), "post_W_writing_level0",
            lambda g=g: writers_cramp(g, seed=SEED + 30 + g), scene="chair_table", title=WT)
    for g in (2, 3):
        add(f"post_E1_foot_dystonia_g{g}", "limb_dystonia", "E1", "DIS distal leg", "foot dystonia (equinovarus), left",
            g, f"DIS distal leg amplitude {g}",
            f"plantarflexion, inversion, adduction and toe flexion at {_txt(100 * DIS_SHARE[g], 0)} percent of their "
            f"rooms (dystonia table: inversion to {_txt(LX.INVERSION_DEG)}, toes to {_txt(LX.TOE_FLEXION_DEG)} deg); "
            f"the foot pivoting on its forefoot, the heel "
            f"rising, the thigh lifting off the seat",
            ("left", "front", "foot_left"), _l0("E1"),
            lambda g=g: foot_dystonia(g, seed=SEED + 40 + g), scene="chair")
    add("post_E7_off_period_foot", "limb_dystonia", "E7", "off-period dystonia",
        "off-period foot dystonia, left (early morning, standing)", "present", "off-period dystonia (descriptive)",
        "inverted foot, toes extended (striatal toe); the leg unloaded, the foot touching, the pelvis hiked [module]",
        ("front", "left", "back", "foot_left"), _l0("E7"),
        lambda: foot_dystonia(levels=OFF_FOOT, standing=True, seed=SEED + 45))
    for g in (2, 3):
        add(f"post_E2_task_specific_arm_g{g}", "limb_dystonia", "E2", "DIS distal arm (action)",
            "posture-induced (action) dystonia, left arm (arms outstretched)", g,
            f"DIS distal arm in action, amplitude {g}",
            f"absent at rest; 2 s after the posture is reached: wrist flexion and forearm pronation at "
            f"{_txt(100 * DIS_SHARE[g], 0)} percent of their rooms [module latency]",
            ("front", "left", "hand_left_medial"),
            _l0("E2"), lambda g=g: task_specific_arm(g, seed=SEED + 50 + g), scene="chair")
    # distribution
    dist = [("hemidystonia", "E7", "hemidystonia, left arm and leg", ("front", "left", "back")),
            ("segmental", "E7", "segmental dystonia: neck, left shoulder and arm", ("front", "top", "head")),
            ("multifocal", "E7", "multifocal dystonia: neck and left foot", ("front", "left", "back")),
            ("fixed_fist", "E1", "fixed dystonia, clenched left fist", ("front", "hand_left_medial")),
            ("fixed_foot", "E1", "fixed dystonia, inverted left foot", ("front", "foot_left"))]
    DIST_TEXT = {
        "hemidystonia": "held from the first frame with slow fluctuation: left elbow and wrist flexed, the fist "
                        "closing; the left foot inverted on its lateral border, heel raised, the pelvis hiked "
                        "[module]",
        "segmental": "held from the first frame with slow fluctuation: neck rotated to the right (TWSTRS 2), left "
                     "shoulder raised (TWSTRS D 2) with the arm hanging under it, elbow flexed 50 deg, wrist flexed "
                     "and pronated, fingers curled [module]",
        "multifocal": "held from the first frame with slow fluctuation: neck tilted to the right (TWSTRS 2) and the "
                      "left foot inverted, heel raised [module]",
        "fixed_fist": "fixed from the first frame, no change: ring and little fingers flexed most, the index least, "
                      "the thumb across the palm, the wrist flexed [module]",
        "fixed_foot": "fixed from the first frame, no change: inverted, plantarflexed and adducted, the toes "
                      "curled, the heel off the floor [module]"}
    for kind, seg, name, views in dist:
        add(f"post_{seg}_{kind}", "dystonia_distribution", seg, "distribution (Albanese 2013)", name, kind, name,
            DIST_TEXT[kind], views, _l0(seg), lambda kind=kind: distribution(kind, SEED + 60),
            scene="chair" if seg == "E1" else None)
    for sv in (2, 3):
        add(f"post_E7_torsion_dystonia_sev{sv}", "dystonia_distribution", "E7", "BFM severity x provoking factor",
            "generalized torsion dystonia (DYT1-like), at rest", sv, f"BFM severity {sv}, provoking factor 4",
            f"trunk and pelvis twisting at {_txt(100 * DIS_SHARE[sv], 0)} percent of their rooms, the neck at TWSTRS "
            f"rotation {TORSION_NECK[sv][0]} and laterocollis {TORSION_NECK[sv][1]}; forearms hyperpronated, wrists "
            f"flexed, the left arm extended and turned inward, the right elbow flexed; the left foot inverted; "
            f"sustained holds drifting on {_txt(TORSION_BAND_HZ[0], 2)} to {_txt(TORSION_BAND_HZ[1], 2)} Hz courses "
            f"[module map]",
            ("front", "back", "top"), _l0("E7"), lambda sv=sv: torsion_dystonia(stand(), sv, SEED + 70 + sv))
    add("post_W_overflow", "dystonia_distribution", W, "overflow (Albanese 2013)",
        "writer's cramp with overflow to the shoulder and neck", 2, "overflow dystonia",
        "writer's cramp marked, then the right shoulder elevation and the neck tilt 1.5 s later [module]",
        ("front", "right", "hands"), "post_W_writing_level0", lambda: writers_cramp(2, seed=SEED + 80, overflow=True),
        scene="chair_table", title=WT)
    add("post_W_writing_both_level0", "dystonia_distribution", W, "level 0",
        "writing with the left hand, the right resting (no dystonia)", 0, "level 0",
        "the left hand writing (tripod grip, strokes, the hand moving along the line, resting on the paper by its "
        "curled little finger, the pen on the paper within words), the right resting palm down on the table "
        "[module]",
        ("front", "hands"), None,
        lambda: (lambda bi: (bi[0], truth_record("writing_base", None, 0, "level 0", bi[1], ("module",), bi[2])))(
            writing_level0("left")), scene="chair_table_both", title=WT)
    add("post_W_mirror", "dystonia_distribution", W, "mirror dystonia", "mirror dystonia: the right hand while the "
        "left writes", 2, "mirror dystonia", "the right hand, resting palm down on the table, takes the writer's cramp "
        "posture (marked: fingers curling, wrist flexing with ulnar deviation, the elbow lifting) while the left "
        "writes [module]", ("front", "hands"), "post_W_writing_both_level0",
        lambda: writers_cramp(2, seed=SEED + 81, mirror=True), scene="chair_table_both", title=WT)
    add("post_E7_opisthotonus", "dystonia_distribution", "E7", "descriptive", "opisthotonus (sustained extension), "
        "standing", "severe", "opisthotonus",
        f"held from the first frame: retrocollis at {_txt(100 * OPISTHOTONUS['neck'], 0)} and trunk extension at "
        f"{_txt(100 * OPISTHOTONUS['trunk'], 0)} percent of their rooms, pelvis tilted back "
        f"{_txt(-OPISTHOTONUS['pelvis_deg'])} deg, the arms hanging [module]", ("left", "front"), _l0("E7"),
        lambda: opisthotonus(stand()))
    # paroxysmal dyskinesias
    for kind in ("dystonic", "choreic"):
        add(f"post_E7_pkd_{kind}", "paroxysmal", "E7", "PKD (episode)", f"paroxysmal kinesigenic dyskinesia, {kind}",
            "episode", f"PKD attack, {kind}",
            f"triggered by arising: from {_txt(PKD['latency_s'])} s after lift-off or as the arms open, for "
            f"{_txt(PKD['duration_s'])} s "
            f"(attacks of a few seconds up to 1 min); the folded arms unfold to the sides once the trunk is within "
            f"{_txt(PKD['unfold_trunk_deg'])} deg of upright, then join the attack; "
            + ("the left arm turned inward, elbow and wrist flexed, forearm pronated, writhing slowly"
               if kind == "dystonic" else "irregular flowing movements of both arms, the neck and the trunk")
            + " [module timing]", ("front", "left"), _l0("arise"),
            lambda kind=kind: paroxysmal_kinesigenic(kind, SEED + 90), scene="chair_arise")
    add("post_E1_pnkd_excerpt", "paroxysmal", "E1", "PNKD (episode)", "paroxysmal non-kinesigenic dyskinesia "
        "(excerpt of an attack)", "excerpt", "PNKD attack under way",
        f"{PNKD_ATTACK_TEXT}; a 15 s excerpt: the left arm dystonic (turned inward, elbow and wrist flexed, pronated) "
        f"with flowing movements of both arms, the neck and the trunk, the left forearm resting on the armrest holding "
        f"the shoulder up as the trunk leans [module]", ("front", "left"),
        _l0("E1"), lambda: paroxysmal_excerpt("pnkd", SEED + 92), scene="chair")
    add("post_E7_ped_excerpt", "paroxysmal", "E7", "PED (episode)", "paroxysmal exercise-induced dyskinesia, left foot "
        "(excerpt)", "excerpt", "PED attack under way",
        f"{PED_ATTACK_TEXT}; a 15 s excerpt: the left foot inverted on its lateral border, heel raised, the pelvis "
        f"hiked [module]", ("front", "left", "back", "foot_left"),
        _l0("E7"), lambda: paroxysmal_excerpt("ped", SEED + 93))
    ids = [s.clip_id for s in out]
    if len(set(ids)) != len(ids):
        raise RuntimeError("duplicate clip ids in the posture catalogue")
    return out
